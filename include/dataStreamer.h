#pragma once
#ifndef _DATA_STREAMER_H_
#define _DATA_STREAMER_H_

#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <data_msgs/msg/gripper.hpp>
#include <data_msgs/msg/teleop_status.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <string>
#include <map>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <jsoncpp/json/json.h>
#include <netdb.h>
#include <fcntl.h>
#include <errno.h>

// =====================================================================
// ObservationBuffer — Ring buffer for aligned camera + arm state
// =====================================================================
// Stores the most recent N camera frames and arm poses, aligned by
// timestamp so that each observation step has synchronized data.
class ObservationBuffer {
public:
    ObservationBuffer(int n_cameras, int n_arms, int n_obs_steps, double control_freq, double camera_freq)
        : n_cameras_(n_cameras), n_arms_(n_arms), n_obs_steps_(n_obs_steps),
          control_freq_(control_freq), camera_freq_(camera_freq) {
        dt_ = 1.0 / control_freq;
        int k = static_cast<int>(std::ceil(n_obs_steps * (camera_freq / control_freq)));
        image_buffers_.resize(n_cameras);
        image_ts_buffers_.resize(n_cameras);
        image_heads_.resize(n_cameras, 0);
        image_counts_.resize(n_cameras, 0);
        for (int c = 0; c < n_cameras; ++c) {
            image_buffers_[c].resize(k);
            image_ts_buffers_[c].resize(k);
        }
        pose_buffers_.resize(n_arms);
        gripper_buffers_.resize(n_arms);
        state_ts_buffers_.resize(n_arms);
        state_heads_.assign(n_arms, 0);
        state_counts_.assign(n_arms, 0);
        for (int a = 0; a < n_arms; ++a) {
            pose_buffers_[a].resize(k);
            gripper_buffers_[a].resize(k);
            state_ts_buffers_[a].resize(k);
        }
    }

    void addImage(int cam_idx, const cv::Mat& img, double ts);
    void addArmState(int arm_idx, const std::vector<double>& pose, double gripper, double ts);

    struct AlignedObs {
        std::vector<cv::Mat> images;
        std::vector<std::vector<double>> poses;
        std::vector<double> grippers;
        std::vector<double> timestamps;
    };
    bool getAlignedObs(AlignedObs& out);

    int n_obs_steps() const { return n_obs_steps_; }
    int n_cameras() const { return n_cameras_; }
    int n_arms() const { return n_arms_; }
    double dt() const { return dt_; }

private:
    int n_cameras_;
    int n_arms_;
    int n_obs_steps_;
    double control_freq_;
    double camera_freq_;
    double dt_;

    std::mutex mutex_;
    std::vector<std::vector<cv::Mat>> image_buffers_;
    std::vector<std::vector<double>> image_ts_buffers_;
    std::vector<int> image_heads_;
    std::vector<int> image_counts_;
    std::vector<std::vector<std::vector<double>>> pose_buffers_;
    std::vector<std::vector<double>> gripper_buffers_;
    std::vector<std::vector<double>> state_ts_buffers_;
    std::vector<int> state_heads_;
    std::vector<int> state_counts_;
};


// =====================================================================
// SocketClient — TCP client for JSON request/response with inference server
// =====================================================================
// Uses newline-delimited JSON over a persistent TCP connection.
// Supports automatic reconnection with bounded retry attempts.
class SocketClient {
public:
    SocketClient(const std::string& host, int port)
        : host_(host), port_(port), sock_fd_(-1), connected_(false), reconnect_attempts_(0) {}

    ~SocketClient() { disconnect(); }

    bool connect(double timeout_sec = 5.0);
    bool reconnect(double timeout_sec = 5.0);
    int reconnectAttempts() const { return reconnect_attempts_; }
    static constexpr int maxReconnectAttempts() { return max_reconnect_attempts_; }
    void disconnect();
    bool sendJson(const Json::Value& obj);
    Json::Value recvJson(double timeout_sec = 5.0);
    bool isConnected() const { return connected_; }

private:
    std::string host_;
    int port_;
    int sock_fd_;
    std::atomic<bool> connected_;
    int reconnect_attempts_;
    static constexpr int max_reconnect_attempts_ = 50;
};


// =====================================================================
// DataStreamer — Observation streaming + inference action execution node
// =====================================================================
//
// Architecture (3 producer threads, 1 consumer thread):
//
//   sendLoop       — Producer: grabs aligned obs from buffer, sends JSON over
//                    TCP socket to inference server, then blocks waiting for
//                    the action pipeline to finish before the next cycle.
//   recvLoop       — Consumer: receives JSON responses (action/reset_ack) from
//                    the server and populates pending_actions_.
//   actionStepLoop — Executor: publishes individual action steps (at inference_freq)
//                    from pending_actions_, one step per tick. Also waits for
//                    FK pose convergence before notifying sendLoop to continue.
//
// Pipeline: sendLoop sends obs
//              -> recvLoop receives action
//              -> actionStepLoop publishes all steps
//              -> waits for robot execution (FK convergence)
//              -> sendLoop unblocks and sends next obs
//
// Key safety guard: checkActionSafety() clamps per-step delta (pos, angle)
//                   and enforces publish_rate_hz. Steps that fail safety
//                   are silently skipped; if all steps are rejected the
//                   entire sequence is aborted and sendLoop reconnects.
class DataStreamer : public rclcpp::Node {
public:
    DataStreamer(const rclcpp::NodeOptions& options);
    ~DataStreamer();

private:
    // ---- ROS2 callbacks ----
    void cameraColorCallback(const sensor_msgs::msg::Image::SharedPtr msg, int cam_buf_idx, const std::string& arm_name);
    void armEndPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg, int idx, const std::string& arm_name);
    void pikaPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg, const std::string& arm_name);
    void teleopStatusCallback(const data_msgs::msg::TeleopStatus::SharedPtr msg);
    void gripperEncoderCallback(const data_msgs::msg::Gripper::SharedPtr msg, int idx);

    // ---- Core loops ----
    void sendLoop();
    void recvLoop();
    void actionStepLoop();
    void waitForExecutionComplete();

    // ---- Helpers ----
    bool sendReset();
    std::string matToBase64Jpeg(const cv::Mat& img, int quality = 85);
    std::vector<double> poseStampedToVec7(const geometry_msgs::msg::PoseStamped& msg);
    bool checkActionSafety(const std::vector<double>& action, int arm_idx, const std::string& arm_name,
                          const std::vector<double>& current_pose, double current_gripper);
    bool publishAction(const std::vector<std::vector<double>>& actions);

    // ---- ROS2 subscriptions ----
    std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subCameraColors_;
    std::vector<rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr> subArmEndPoses_;
    std::vector<rclcpp::Subscription<data_msgs::msg::Gripper>::SharedPtr> subGripperEncoders_;
    std::map<std::string, rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr> subPikaPoses_;
    rclcpp::Subscription<data_msgs::msg::TeleopStatus>::SharedPtr subTeleopStatus_;

    // ---- ROS2 publishers ----
    std::vector<rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr> action_pose_pubs_;
    std::vector<rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr> action_gripper_pubs_;
    int action_arm_index_ = 0;

    // ---- Config (from YAML) ----
    std::vector<std::string> cameraColorTopics_;
    std::vector<std::string> cameraColorNames_;
    std::vector<std::string> armEndPoseTopics_;
    std::vector<std::string> armEndPoseNames_;
    std::vector<std::string> gripperEncoderTopics_;
    std::vector<std::string> gripperEncoderNames_;
    std::map<std::string, std::string> arm_pika_pose_topic_;
    std::string server_host_;
    int server_port_;
    int n_arms_;
    double inference_freq_;
    int n_obs_steps_;
    double camera_freq_;
    int jpeg_quality_;
    bool auto_reset_;

    // ---- State ----
    std::unique_ptr<ObservationBuffer> obs_buffer_;
    std::unique_ptr<SocketClient> socket_client_;

    std::thread send_thread_;
    std::thread recv_thread_;
    std::thread action_step_thread_;
    std::atomic<bool> running_{false};

    std::mutex gripper_mutex_;
    std::map<std::string, double> gripper_latest_;

    // arm_name -> ObservationBuffer arm_idx
    std::vector<std::string> arm_names_;
    std::map<std::string, int> arm_name_to_buffer_idx_;
    std::map<std::string, std::string> arm_gripper_name_;
    std::map<std::string, std::vector<int>> arm_to_global_cam_idx_;
    std::vector<std::string> ordered_per_arm_cam_names_;

    // Remote sensor (pika_pose) tracking for teleop-style offset mode
    std::mutex pika_pose_mutex_;
    std::map<std::string, std::vector<double>> pika_pose_init_;
    std::map<std::string, std::vector<double>> pika_pose_current_;
    std::map<std::string, bool> pika_pose_init_recorded_;

    // ---- Debug ----
    std::atomic<bool> debug_enabled_{false};
    std::string debug_dir_;
    std::atomic<int> debug_frame_idx_{0};

    // ---- Counters ----
    std::atomic<int64_t> observations_sent_{0};
    std::mutex action_mutex_;
    std::mutex exec_target_mutex_;
    std::vector<std::vector<double>> exec_target_poses_;

    // ---- Pipeline sync (observe -> action -> execute -> observe) ----
    std::mutex pipeline_mutex_;
    std::condition_variable pipeline_cv_;
    bool waiting_for_action_ = false;

    std::vector<std::vector<std::vector<double>>> pending_actions_;
    int action_sequence_index_ = 0;
    int action_sequence_steps_total_ = 0;
    bool action_sequence_in_progress_ = false;
    bool action_sequence_any_published_ = false;

    // ---- Safety filter ----
    bool safety_enabled_ = true;
    double safety_max_pos_delta_ = 0.05;
    double safety_max_angle_delta_ = 0.5;
    double safety_publish_rate_hz_ = 20.0;
    bool safety_log_rejections_ = true;
    std::chrono::steady_clock::time_point last_safe_publish_time_;

    // Cumulative absolute pose (per arm) — updated every callback, used for safety delta checks
    std::mutex cumulative_pose_mutex_;
    std::map<std::string, std::vector<double>> arm_cumulative_pose_;
    std::map<std::string, double> arm_cumulative_gripper_;
};

#endif
