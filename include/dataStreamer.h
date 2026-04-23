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

    void addImage(int cam_idx, const cv::Mat& img, double ts) {
        if (cam_idx < 0 || cam_idx >= n_cameras_) return;
        std::lock_guard<std::mutex> lk(mutex_);
        int k = static_cast<int>(image_buffers_[cam_idx].size());
        int idx = (image_heads_[cam_idx] + image_counts_[cam_idx]) % k;
        image_buffers_[cam_idx][idx] = img;
        image_ts_buffers_[cam_idx][idx] = ts;
        if (image_counts_[cam_idx] < k) {
            image_counts_[cam_idx]++;
        } else {
            image_heads_[cam_idx] = (image_heads_[cam_idx] + 1) % k;
        }
    }

    void addArmState(int arm_idx, const std::vector<double>& pose, double gripper, double ts) {
        if (arm_idx < 0 || arm_idx >= n_arms_) return;
        std::lock_guard<std::mutex> lk(mutex_);
        int k = static_cast<int>(pose_buffers_[arm_idx].size());
        int idx = (state_heads_[arm_idx] + state_counts_[arm_idx]) % k;
        pose_buffers_[arm_idx][idx] = pose;
        gripper_buffers_[arm_idx][idx] = gripper;
        state_ts_buffers_[arm_idx][idx] = ts;
        if (state_counts_[arm_idx] < k) {
            state_counts_[arm_idx]++;
        } else {
            state_heads_[arm_idx] = (state_heads_[arm_idx] + 1) % k;
        }
    }

    struct AlignedObs {
        std::vector<cv::Mat> images;
        std::vector<std::vector<double>> poses;
        std::vector<double> grippers;
        std::vector<double> timestamps;
    };

    bool getAlignedObs(AlignedObs& out) {
        std::lock_guard<std::mutex> lk(mutex_);
        int k = 0;
        if (!pose_buffers_.empty()) k = static_cast<int>(pose_buffers_[0].size());

        for (int c = 0; c < n_cameras_; ++c) {
            if (image_counts_[c] < k) return false;
        }
        for (int a = 0; a < n_arms_; ++a) {
            if (state_counts_[a] < k) return false;
        }

        double last_ts = 0.0;
        for (int a = 0; a < n_arms_; ++a) {
            double t = state_ts_buffers_[a][(state_heads_[a] + state_counts_[a] - 1) % k];
            if (a == 0 || t > last_ts) last_ts = t;
        }

        out.timestamps.resize(n_obs_steps_);
        for (int i = 0; i < n_obs_steps_; ++i) {
            out.timestamps[i] = last_ts - (n_obs_steps_ - 1 - i) * dt_;
        }

        out.images.resize(n_cameras_ * n_obs_steps_);
        for (int c = 0; c < n_cameras_; ++c) {
            int k_cam = static_cast<int>(image_buffers_[c].size());
            std::vector<double> cam_ts(image_counts_[c]);
            for (int i = 0; i < image_counts_[c]; ++i) {
                int buf_idx = (image_heads_[c] + i) % k_cam;
                cam_ts[i] = image_ts_buffers_[c][buf_idx];
            }
            double cam_oldest = image_counts_[c] > 0 ? cam_ts[0] : 0.0;
            double cam_newest = image_counts_[c] > 0 ? cam_ts[image_counts_[c] - 1] : 0.0;
            for (int i = 0; i < n_obs_steps_; ++i) {
                double t = out.timestamps[i];
                int best_idx = 0;
                if (image_counts_[c] <= 1) {
                    best_idx = 0;
                } else if (t <= cam_oldest) {
                    best_idx = 0;
                } else if (t >= cam_newest) {
                    best_idx = image_counts_[c] - 1;
                } else {
                    for (int j = 0; j < image_counts_[c] - 1; ++j) {
                        if (cam_ts[j] <= t && cam_ts[j + 1] > t) { best_idx = j; break; }
                    }
                }
                int buf_idx = (image_heads_[c] + best_idx) % k_cam;
                out.images[c * n_obs_steps_ + i] = image_buffers_[c][buf_idx];
            }
        }

        out.poses.resize(n_arms_ * n_obs_steps_);
        out.grippers.resize(n_arms_ * n_obs_steps_);
        for (int a = 0; a < n_arms_; ++a) {
            int k_s = static_cast<int>(pose_buffers_[a].size());
            std::vector<double> arm_ts(state_counts_[a]);
            for (int i = 0; i < state_counts_[a]; ++i) {
                int buf_idx = (state_heads_[a] + i) % k_s;
                arm_ts[i] = state_ts_buffers_[a][buf_idx];
            }
            double arm_oldest = state_counts_[a] > 0 ? arm_ts[0] : 0.0;
            double arm_newest = state_counts_[a] > 0 ? arm_ts[state_counts_[a] - 1] : 0.0;
            for (int i = 0; i < n_obs_steps_; ++i) {
                double t = out.timestamps[i];
                int best_idx = 0;
                // Clamp t to valid range to avoid stale/wrong data when arm is lagging.
                if (state_counts_[a] <= 1) {
                    best_idx = 0;
                } else if (t <= arm_oldest) {
                    best_idx = 0;
                } else if (t >= arm_newest) {
                    best_idx = state_counts_[a] - 1;
                } else {
                    for (int j = 0; j < state_counts_[a] - 1; ++j) {
                        if (arm_ts[j] <= t && arm_ts[j + 1] > t) { best_idx = j; break; }
                    }
                }
                int buf_idx = (state_heads_[a] + best_idx) % k_s;
                out.poses[a * n_obs_steps_ + i] = pose_buffers_[a][buf_idx];
                out.grippers[a * n_obs_steps_ + i] = gripper_buffers_[a][buf_idx];
            }
        }

        return true;
    }

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


class SocketClient {
public:
    SocketClient(const std::string& host, int port)
        : host_(host), port_(port), sock_fd_(-1), connected_(false), reconnect_attempts_(0) {}

    ~SocketClient() { disconnect(); }

    bool connect(double timeout_sec = 5.0) {
        disconnect();

        struct addrinfo hints{}, *res;
        memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        int ret = getaddrinfo(host_.c_str(), std::to_string(port_).c_str(), &hints, &res);
        if (ret != 0) {
            RCLCPP_ERROR(rclcpp::get_logger("SocketClient"), "DNS lookup failed: %s", gai_strerror(ret));
            return false;
        }

        sock_fd_ = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (sock_fd_ < 0) {
            RCLCPP_ERROR(rclcpp::get_logger("SocketClient"), "socket() failed: %s", strerror(errno));
            freeaddrinfo(res);
            return false;
        }

        struct timeval tv;
        tv.tv_sec = static_cast<long>(timeout_sec);
        tv.tv_usec = static_cast<long>((timeout_sec - (long)timeout_sec) * 1e6);
        setsockopt(sock_fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        setsockopt(sock_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        if (::connect(sock_fd_, res->ai_addr, res->ai_addrlen) < 0) {
            RCLCPP_ERROR(rclcpp::get_logger("SocketClient"), "connect() to %s:%d failed: %s",
                         host_.c_str(), port_, strerror(errno));
            freeaddrinfo(res);
            close(sock_fd_);
            sock_fd_ = -1;
            connected_ = false;
            return false;
        }
        freeaddrinfo(res);
        connected_ = true;
        reconnect_attempts_ = 0;
        return true;
    }

    bool reconnect(double timeout_sec = 5.0) {
        reconnect_attempts_++;
        RCLCPP_INFO(rclcpp::get_logger("SocketClient"),
                    "[SocketClient] Attempting reconnect #%d to %s:%d (attempt %d of %d)",
                    reconnect_attempts_, host_.c_str(), port_, reconnect_attempts_, max_reconnect_attempts_);
        return connect(timeout_sec);
    }

    int reconnectAttempts() const { return reconnect_attempts_; }
    static constexpr int maxReconnectAttempts() { return max_reconnect_attempts_; }

    void disconnect() {
        if (sock_fd_ >= 0) {
            ::close(sock_fd_);
            sock_fd_ = -1;
        }
        connected_ = false;
    }

    bool sendJson(const Json::Value& obj) {
        if (!connected_ || sock_fd_ < 0) return false;
        std::string msg = Json::FastWriter().write(obj) + "\n";
        ssize_t sent = ::send(sock_fd_, msg.c_str(), msg.size(), MSG_NOSIGNAL);
        return sent == static_cast<ssize_t>(msg.size());
    }

    Json::Value recvJson(double timeout_sec = 5.0) {
        Json::Value result;  // default-constructed = nullValue
        if (!connected_ || sock_fd_ < 0) return result;

        std::string line;
        line.reserve(4096);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(timeout_sec));

        while (true) {
            char buf[4096];
            ssize_t n = ::recv(sock_fd_, buf, sizeof(buf) - 1, 0);
            if (n <= 0) break;
            buf[n] = '\0';
            line += buf;

            size_t pos;
            while ((pos = line.find('\n')) != std::string::npos) {
                std::string msg = line.substr(0, pos);
                line.erase(0, pos + 1);
                if (!msg.empty()) {
                    Json::Value parsed;
                    Json::Reader reader;
                    if (reader.parse(msg, parsed)) {
                        // Only return object or null — reject arrays/strings/etc
                        if (parsed.isObject()) {
                            return parsed;
                        }
                        RCLCPP_WARN(rclcpp::get_logger("SocketClient"),
                            "recvJson: discarding non-object JSON (type=%d)", parsed.type());
                    }
                }
            }

            if (std::chrono::steady_clock::now() >= deadline) break;
        }
        return result;  // nullValue
    }

    bool isConnected() const { return connected_; }

private:
    std::string host_;
    int port_;
    int sock_fd_;
    std::atomic<bool> connected_;
    int reconnect_attempts_;
    static constexpr int max_reconnect_attempts_ = 50;
};


class DataStreamer : public rclcpp::Node {
public:
    DataStreamer(const rclcpp::NodeOptions& options);
    ~DataStreamer();

private:
    // Camera callbacks
    void cameraColorCallback(const sensor_msgs::msg::Image::SharedPtr msg, int cam_buf_idx, const std::string& arm_name);
    // Pose callbacks (arm end pose from FK/IK)
    void armEndPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg, int idx, const std::string& arm_name);
    // Remote sensor pose callback (teleop master device)
    void pikaPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg, const std::string& arm_name);
    void teleopStatusCallback(const data_msgs::msg::TeleopStatus::SharedPtr msg);
    // Gripper encoder callbacks
    void gripperEncoderCallback(const data_msgs::msg::Gripper::SharedPtr msg, int idx);

    // Core loops
    void collectLoop();
    void sendLoop();
    void recvLoop();
    void actionStepLoop();

    // Helpers
    bool sendReset();
    std::string matToBase64Jpeg(const cv::Mat& img, int quality = 85);
    std::vector<double> poseStampedToVec7(const geometry_msgs::msg::PoseStamped& msg);
    bool checkActionSafety(const std::vector<double>& action, int arm_idx);
    void publishAction(const std::vector<std::vector<double>>& actions);

    // ROS2 subscriptions
    std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subCameraColors_;
    std::vector<rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr> subArmEndPoses_;
    std::vector<rclcpp::Subscription<data_msgs::msg::Gripper>::SharedPtr> subGripperEncoders_;
    std::map<std::string, rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr> subPikaPoses_;
    rclcpp::Subscription<data_msgs::msg::TeleopStatus>::SharedPtr subTeleopStatus_;

    // ROS2 publishers for inference action output
    std::vector<rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr> action_pose_pubs_;
    std::vector<rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr> action_gripper_pubs_;
    int action_arm_index_ = 0;

    // Config
    std::vector<std::string> cameraColorTopics_;
    std::vector<std::string> cameraColorNames_;
    std::vector<std::string> armEndPoseTopics_;
    std::vector<std::string> armEndPoseNames_;
    std::vector<std::string> gripperEncoderTopics_;
    std::vector<std::string> gripperEncoderNames_;
    std::vector<bool> armEndPoseOrients_;
    std::vector<double> armEndPoseOffsets_;
    std::map<std::string, std::string> arm_pika_pose_topic_;  // arm_name -> topic name (e.g. arm_r -> /pika_pose_r)

    // Network config
    std::string server_host_;
    int server_port_;
    int n_arms_;
    double inference_freq_;
    int n_obs_steps_;
    double camera_freq_;
    int jpeg_quality_;
    bool auto_reset_;

    // Observation buffer
    std::unique_ptr<ObservationBuffer> obs_buffer_;

    // Socket
    std::unique_ptr<SocketClient> socket_client_;

    // Threads
    std::thread send_thread_;
    std::thread recv_thread_;
    std::thread action_step_thread_;
    std::atomic<bool> running_{false};

    // Mutex for gripper latest data
    std::mutex gripper_mutex_;
    std::map<std::string, double> gripper_latest_;

    // Arm config
    std::vector<std::string> arm_names_;  // active arm names from YAML, e.g. ['arm_l', 'arm_r'] or ['arm_l']
    std::map<std::string, int> arm_name_to_buffer_idx_;  // arm_name -> ObservationBuffer arm_idx
    std::map<std::string, std::string> arm_gripper_name_;  // arm_name -> gripper encoder name (e.g. arm_l -> gripper_l)

    // Per-arm camera: arm_name -> [global_cam_idx, ...]
    std::map<std::string, std::vector<int>> arm_to_global_cam_idx_;

    // Ordered per-arm camera names (positions match obs.images layout)
    std::vector<std::string> ordered_per_arm_cam_names_;

    // Init pose: first recorded pose per arm, protected by init_pose_mutex_
    std::mutex init_pose_mutex_;
    std::map<std::string, std::vector<double>> arm_init_pose_;  // arm_name -> 7D pose (xyz + quaternion)
    std::map<std::string, bool> arm_init_pose_recorded_;

    // Remote sensor (pika_pose) tracking for teleop-style offset mode
    std::mutex pika_pose_mutex_;
    std::map<std::string, std::vector<double>> pika_pose_init_;      // arm_name -> first recorded sensor pose (calibration)
    std::map<std::string, std::vector<double>> pika_pose_current_;   // arm_name -> latest sensor pose
    std::map<std::string, bool> pika_pose_init_recorded_;
    std::map<std::string, bool> pika_pose_side_is_left_;            // arm_name -> true if left arm

    // Debug
    std::atomic<bool> debug_enabled_{false};
    std::string debug_dir_;
    std::atomic<int> debug_frame_idx_{0};

    // Action save (for testing inference server output)
    std::atomic<bool> save_action_enabled_{false};
    std::string save_action_dir_;
    std::atomic<int64_t> save_action_count_{0};

    // Counters
    std::atomic<int64_t> observations_sent_{0};
    std::atomic<int64_t> actions_received_{0};
    std::mutex action_mutex_;

    std::vector<std::vector<std::vector<double>>> pending_actions_;  // per-arm multi-step sequences, index matches arm_names_
    int action_sequence_index_ = 0;  // current step index within the pending sequence
    int action_sequence_steps_total_ = 0;  // total steps in current sequence
    bool action_sequence_in_progress_ = false;  // guard: don't overwrite pending sequence with new action

    // Camera info
    int image_width_ = 1280;
    int image_height_ = 720;

    // Safety filter state
    bool safety_enabled_ = true;
    double safety_max_pos_delta_ = 0.05;
    double safety_max_gripper_delta_ = 0.3;
    double safety_max_angle_delta_ = 0.5;
    double safety_pos_x_min_ = 0.0, safety_pos_x_max_ = 0.8;
    double safety_pos_y_min_ = -0.6, safety_pos_y_max_ = 0.6;
    double safety_pos_z_min_ = -0.2, safety_pos_z_max_ = 0.7;
    double safety_publish_rate_hz_ = 20.0;
    bool safety_log_rejections_ = true;
    bool safety_quat_multiply_body_frame_ = true;
    std::vector<double> last_safe_action_;
    std::mutex safety_mutex_;
    std::chrono::steady_clock::time_point last_safe_publish_time_;

    // Cumulative absolute pose for delta-based safety (per arm)
    std::mutex cumulative_pose_mutex_;
    std::map<std::string, std::vector<double>> arm_cumulative_pose_;  // arm_name -> 7D pose (xyz + quaternion)
    std::map<std::string, double> arm_cumulative_gripper_;            // arm_name -> absolute gripper value [0,1]
};

#endif
