#include "dataStreamer.h"
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <jsoncpp/json/json.h>
#include <sys/stat.h>
#include <iomanip>
#include <fstream>
#include <sstream>

namespace {
const char* LOGGER_NAME = "DataStreamer";

static const char BASE64_TABLE[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

inline std::string base64Encode(const unsigned char* data, size_t len) {
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    for (size_t i = 0; i < len; i += 3) {
        int a = data[i];
        int b = (i + 1 < len) ? data[i + 1] : 0;
        int c = (i + 2 < len) ? data[i + 2] : 0;
        out.push_back(BASE64_TABLE[(a >> 2) & 0x3F]);
        out.push_back(BASE64_TABLE[((a << 4) | (b >> 4)) & 0x3F]);
        out.push_back((i + 1 < len) ? BASE64_TABLE[((b << 2) | (c >> 6)) & 0x3F] : '=');
        out.push_back((i + 2 < len) ? BASE64_TABLE[c & 0x3F] : '=');
    }
    return out;
}

bool mkdirRecursive(const std::string& path) {
    size_t pos = 0;
    std::string p = path;
    while ((pos = p.find('/', pos + 1)) != std::string::npos) {
        std::string dir = p.substr(0, pos);
        if (dir.empty() || dir == ".") continue;
        ::mkdir(dir.c_str(), 0755);
    }
    ::mkdir(p.c_str(), 0755);
    return true;
}

void saveDebugFrame(const std::string& dir, int frame_idx,
                    const std::vector<cv::Mat>& images,
                    const std::vector<std::string>& camera_names,
                    int n_obs_steps,
                    int n_arms,
                    const std::vector<std::string>& arm_names,
                    const std::vector<std::vector<double>>& poses,
                    const std::vector<double>& grippers,
                    const Json::Value& json_msg) {
    std::ostringstream idx_str;
    idx_str << std::setw(6) << std::setfill('0') << frame_idx;
    std::string base_prefix = dir + "/" + idx_str.str();

    mkdirRecursive(dir);

    std::ofstream f(base_prefix + "_obs.json");
    f << Json::FastWriter().write(json_msg);
    f.close();

    int n_cameras = static_cast<int>(camera_names.size());
    for (int c = 0; c < n_cameras; ++c) {
        for (int t = 0; t < n_obs_steps; ++t) {
            int flat_idx = c * n_obs_steps + t;
            if (flat_idx < (int)images.size()) {
                std::ostringstream img_name;
                img_name << base_prefix << "_" << camera_names[c] << "_t" << t << ".jpg";
                cv::imwrite(img_name.str(), images[flat_idx]);
            }
        }
    }

    for (int a = 0; a < n_arms && a < (int)arm_names.size(); ++a) {
        std::ostringstream pose_name;
        pose_name << base_prefix << "_arm" << arm_names[a] << ".txt";
        std::ofstream pf(pose_name.str());
        for (int t = 0; t < n_obs_steps; ++t) {
            int flat_idx = a * n_obs_steps + t;
            if (flat_idx < (int)poses.size()) {
                pf << "t" << t << ": ";
                for (double v : poses[flat_idx]) pf << v << " ";
                pf << " | gripper=" << (flat_idx < (int)grippers.size() ? grippers[flat_idx] : -1.0) << "\n";
            }
        }
        pf.close();
    }
}
}

DataStreamer::DataStreamer(const rclcpp::NodeOptions& options) : Node("data_streamer", options) {

    rcl_interfaces::msg::ParameterDescriptor desc;
    desc.name = "camera.color.topics";
    desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_STRING_ARRAY;
    this->declare_parameter("camera.color.topics", rclcpp::ParameterValue(std::vector<std::string>()), desc);
    desc.name = "camera.color.names";
    this->declare_parameter("camera.color.names", rclcpp::ParameterValue(std::vector<std::string>()), desc);
    desc.name = "arm.endPose.topics";
    this->declare_parameter("arm.endPose.topics", rclcpp::ParameterValue(std::vector<std::string>()), desc);
    desc.name = "arm.endPose.names";
    this->declare_parameter("arm.endPose.names", rclcpp::ParameterValue(std::vector<std::string>()), desc);
    desc.name = "arm.endPose.orient";
    this->declare_parameter("arm.endPose.orient", rclcpp::ParameterValue(std::vector<bool>()), desc);
    desc.name = "arm.endPose.offset";
    this->declare_parameter("arm.endPose.offset", rclcpp::ParameterValue(std::vector<double>()), desc);
    desc.name = "arm.remote_pose.topics";
    this->declare_parameter("arm.remote_pose.topics", rclcpp::ParameterValue(std::vector<std::string>()), desc);
    desc.name = "arm.remote_pose.names";
    this->declare_parameter("arm.remote_pose.names", rclcpp::ParameterValue(std::vector<std::string>()), desc);
    desc.name = "gripper.encoder.topics";
    this->declare_parameter("gripper.encoder.topics", rclcpp::ParameterValue(std::vector<std::string>()), desc);
    desc.name = "gripper.encoder.names";
    this->declare_parameter("gripper.encoder.names", rclcpp::ParameterValue(std::vector<std::string>()), desc);

    this->declare_parameter<std::string>("server.host", "127.0.0.1");
    this->declare_parameter<int>("server.port", 8007);
    this->declare_parameter<double>("inference.freq", 20.0);
    this->declare_parameter<int>("inference.n_obs_steps", 2);
    this->declare_parameter<double>("inference.camera_freq", 30.0);
    this->declare_parameter<int>("inference.jpeg_quality", 85);
    this->declare_parameter<bool>("inference.auto_reset", true);
    this->declare_parameter<int>("camera.color.width", 1280);
    this->declare_parameter<int>("camera.color.height", 720);
    this->declare_parameter<int>("action.arm_index", 0);
    this->declare_parameter<bool>("action.enabled", true);
    this->declare_parameter<bool>("debug.enabled", false);
    this->declare_parameter<std::string>("debug.dir", "/home/yxgn/data_streamer_debug");

    this->declare_parameter<std::string>("arm.active_arms", "arm_l,arm_r");
    this->declare_parameter<std::string>("arm.active_arm_gripper_names", "arm_l:gripper_l,arm_r:gripper_r");
    this->declare_parameter<std::string>("arm.per_arm_cameras", "arm_l:fisheye_l,arm_l:fisheye_r,arm_r:fisheye_l,arm_r:fisheye_r");

    // Safety filter parameters
    this->declare_parameter<bool>("safety.enabled", true);
    this->declare_parameter<double>("safety.max_position_delta", 0.05);
    this->declare_parameter<double>("safety.max_gripper_delta", 0.3);
    this->declare_parameter<double>("safety.max_angle_delta", 0.5);
    this->declare_parameter<double>("safety.position_x_min", 0.0);
    this->declare_parameter<double>("safety.position_x_max", 0.8);
    this->declare_parameter<double>("safety.position_y_min", -0.6);
    this->declare_parameter<double>("safety.position_y_max", 0.6);
    this->declare_parameter<double>("safety.position_z_min", -0.2);
    this->declare_parameter<double>("safety.position_z_max", 0.7);
    this->declare_parameter<double>("safety.publish_rate_hz", 20.0);
    this->declare_parameter<bool>("safety.log_rejections", true);
    this->declare_parameter<bool>("safety.quaternion_multiply_body_frame", true);

    this->get_parameter("camera.color.topics", cameraColorTopics_);
    this->get_parameter("camera.color.names", cameraColorNames_);
    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Loaded camera names (%zu): [%s], topics (%zu)",
                cameraColorNames_.size(),
                cameraColorNames_.empty() ? "EMPTY" : [&]() {
                    std::string s = cameraColorNames_[0];
                    for (size_t i = 1; i < cameraColorNames_.size(); ++i) s += ", " + cameraColorNames_[i];
                    return s;
                }().c_str(),
                cameraColorTopics_.size());

    this->get_parameter("arm.endPose.topics", armEndPoseTopics_);
    this->get_parameter("arm.endPose.names", armEndPoseNames_);
    this->get_parameter("arm.endPose.orient", armEndPoseOrients_);
    this->get_parameter("arm.endPose.offset", armEndPoseOffsets_);

    std::vector<std::string> remote_pose_topics_raw;
    std::vector<std::string> remote_pose_names_raw;
    this->get_parameter("arm.remote_pose.topics", remote_pose_topics_raw);
    this->get_parameter("arm.remote_pose.names", remote_pose_names_raw);

    // Build arm_name -> pika_pose topic map
    arm_pika_pose_topic_.clear();
    for (size_t i = 0; i < remote_pose_topics_raw.size() && i < remote_pose_names_raw.size(); ++i) {
        arm_pika_pose_topic_[remote_pose_names_raw[i]] = remote_pose_topics_raw[i];
        bool is_left = (remote_pose_names_raw[i].find("l") != std::string::npos);
        pika_pose_side_is_left_[remote_pose_names_raw[i]] = is_left;
    }

    std::string active_arms_str;
    this->get_parameter("arm.active_arms", active_arms_str);
    {
        std::string s = active_arms_str;
        size_t start = 0;
        for (size_t i = 0; i <= s.size(); ++i) {
            if (i == s.size() || s[i] == ',') {
                std::string part = s.substr(start, i - start);
                if (!part.empty()) arm_names_.push_back(part);
                start = i + 1;
            }
        }
    }

    n_arms_ = static_cast<int>(arm_names_.size());

    arm_name_to_buffer_idx_.clear();
    for (int buf_idx = 0; buf_idx < (int)arm_names_.size(); ++buf_idx) {
        const std::string& arm_name = arm_names_[buf_idx];
        for (int i = 0; i < (int)armEndPoseNames_.size(); ++i) {
            if (armEndPoseNames_[i] == arm_name) {
                arm_name_to_buffer_idx_[arm_name] = buf_idx;
                break;
            }
        }
    }

    std::string arm_gripper_str;
    this->get_parameter("arm.active_arm_gripper_names", arm_gripper_str);
    arm_gripper_name_.clear();
    {
        std::string s = arm_gripper_str;
        size_t start = 0;
        for (size_t i = 0; i <= s.size(); ++i) {
            if (i == s.size() || s[i] == ',') {
                std::string part = s.substr(start, i - start);
                if (!part.empty()) {
                    size_t colon = part.find(':');
                    if (colon != std::string::npos) {
                        std::string arm_k = part.substr(0, colon);
                        std::string gripper_k = part.substr(colon + 1);
                        arm_gripper_name_[arm_k] = gripper_k;
                    }
                }
                start = i + 1;
            }
        }
    }

    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Active arms: %d, names: [%s], gripper mapping: %s",
                 n_arms_, active_arms_str.c_str(), arm_gripper_str.c_str());

    // Build per-arm camera mapping and filter cameras for active arms
    std::string per_arm_cameras_str;
    this->get_parameter("arm.per_arm_cameras", per_arm_cameras_str);
    std::map<std::string, std::vector<std::string>> per_arm_cameras;  // arm_name -> [camera_name, ...]
    {
        std::string s = per_arm_cameras_str;
        size_t start = 0;
        for (size_t i = 0; i <= s.size(); ++i) {
            if (i == s.size() || s[i] == ',') {
                std::string part = s.substr(start, i - start);
                if (!part.empty()) {
                    size_t colon = part.find(':');
                    if (colon != std::string::npos) {
                        std::string arm_k = part.substr(0, colon);
                        std::string cam_k = part.substr(colon + 1);
                        per_arm_cameras[arm_k].push_back(cam_k);
                    }
                }
                start = i + 1;
            }
        }
    }

    std::map<std::string, int> camera_name_to_global_idx;
    for (int i = 0; i < (int)cameraColorNames_.size(); ++i) {
        camera_name_to_global_idx[cameraColorNames_[i]] = i;
    }

    // Build filtered camera list: for each arm, list of ordered buffer indices belonging to it
    // The ordered position in ordered_per_arm_cam_names IS the ObservationBuffer index.
    std::map<std::string, std::vector<int>> arm_to_global_cam_idx;
    std::vector<std::string> ordered_per_arm_cam_names;
    for (const auto& arm_pair : per_arm_cameras) {
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] mapping arm '%s' -> cameras [%s]",
                    arm_pair.first.c_str(),
                    [&]() {
                        std::ostringstream s;
                        for (auto& c : arm_pair.second) s << c << " ";
                        return s.str();
                    }().c_str());
        for (const std::string& cam_name : arm_pair.second) {
            auto it = camera_name_to_global_idx.find(cam_name);
            if (it != camera_name_to_global_idx.end()) {
                int ordered_pos = (int)ordered_per_arm_cam_names.size();
                arm_to_global_cam_idx[arm_pair.first].push_back(ordered_pos);
                ordered_per_arm_cam_names.push_back(cam_name);
            } else {
                RCLCPP_WARN(this->get_logger(), "[DataStreamer] camera '%s' not found in cameraColorNames_, skipping",
                            cam_name.c_str());
            }
        }
    }

    // Total cameras = sum of all per-arm cameras
    int n_total_cameras = (int)ordered_per_arm_cam_names.size();
    arm_to_global_cam_idx_ = arm_to_global_cam_idx;
    ordered_per_arm_cam_names_ = ordered_per_arm_cam_names;
    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Total cameras for streaming: %d", n_total_cameras);

    this->get_parameter("gripper.encoder.topics", gripperEncoderTopics_);
    this->get_parameter("gripper.encoder.names", gripperEncoderNames_);

    this->get_parameter("server.host", server_host_);
    this->get_parameter("server.port", server_port_);
    this->get_parameter("inference.freq", inference_freq_);
    this->get_parameter("inference.n_obs_steps", n_obs_steps_);
    this->get_parameter("inference.camera_freq", camera_freq_);
    this->get_parameter("inference.jpeg_quality", jpeg_quality_);
    this->get_parameter("inference.auto_reset", auto_reset_);
    this->get_parameter("camera.color.width", image_width_);
    this->get_parameter("camera.color.height", image_height_);
    this->get_parameter("action.arm_index", action_arm_index_);
    bool action_enabled;
    this->get_parameter("action.enabled", action_enabled);

    bool debug_enabled;
    this->get_parameter("debug.enabled", debug_enabled);
    this->get_parameter("debug.dir", debug_dir_);
    debug_enabled_.store(debug_enabled);

    if (debug_enabled_) {
        mkdirRecursive(debug_dir_);
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] DEBUG mode ON: saving to %s", debug_dir_.c_str());
    }

    this->get_parameter("safety.enabled", safety_enabled_);
    this->get_parameter("safety.max_position_delta", safety_max_pos_delta_);
    this->get_parameter("safety.max_gripper_delta", safety_max_gripper_delta_);
    this->get_parameter("safety.max_angle_delta", safety_max_angle_delta_);
    this->get_parameter("safety.position_x_min", safety_pos_x_min_);
    this->get_parameter("safety.position_x_max", safety_pos_x_max_);
    this->get_parameter("safety.position_y_min", safety_pos_y_min_);
    this->get_parameter("safety.position_y_max", safety_pos_y_max_);
    this->get_parameter("safety.position_z_min", safety_pos_z_min_);
    this->get_parameter("safety.position_z_max", safety_pos_z_max_);
    this->get_parameter("safety.publish_rate_hz", safety_publish_rate_hz_);
    this->get_parameter("safety.log_rejections", safety_log_rejections_);
    this->get_parameter("safety.quaternion_multiply_body_frame", safety_quat_multiply_body_frame_);
    last_safe_publish_time_ = std::chrono::steady_clock::now() - std::chrono::milliseconds(100);

    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Safety filter: enabled=%d, max_pos_delta=%.3fm, max_gripper_delta=%.3f, rate=%.1fHz",
                 safety_enabled_, safety_max_pos_delta_, safety_max_gripper_delta_, safety_publish_rate_hz_);

    if (armEndPoseOrients_.empty())
        armEndPoseOrients_.resize(armEndPoseTopics_.size(), true);
    if (armEndPoseOffsets_.empty())
        armEndPoseOffsets_.resize(armEndPoseTopics_.size(), 0.0);

    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Connecting to %s:%d", server_host_.c_str(), server_port_);

    socket_client_ = std::make_unique<SocketClient>(server_host_, server_port_);
    if (!socket_client_->connect(10.0)) {
        RCLCPP_ERROR(this->get_logger(), "[DataStreamer] Failed to connect to inference server, will retry in send loop!");
    } else {
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] Connected to inference server.");
    }

    obs_buffer_ = std::make_unique<ObservationBuffer>(
        n_total_cameras, n_arms_, n_obs_steps_, inference_freq_, camera_freq_);

    // Subscribe to each camera using the ordered buffer index (which matches ordered_per_arm_cam_names_)
    for (const auto& arm_pair : arm_to_global_cam_idx) {
        const std::string& arm_name = arm_pair.first;
        for (int ordered_idx : arm_pair.second) {
            const std::string& cam_name = ordered_per_arm_cam_names[ordered_idx];
            int global_idx = camera_name_to_global_idx.at(cam_name);
            auto sub = this->create_subscription<sensor_msgs::msg::Image>(
                cameraColorTopics_[global_idx], rclcpp::SensorDataQoS(),
                [this, ordered_idx, arm_name](const sensor_msgs::msg::Image::SharedPtr msg) {
                    this->cameraColorCallback(msg, ordered_idx, arm_name);
                });
            subCameraColors_.push_back(sub);
            RCLCPP_INFO(this->get_logger(), "[DataStreamer] Subscribed to camera (arm=%s): %s -> buf_idx=%d",
                        arm_name.c_str(), cameraColorTopics_[global_idx].c_str(), ordered_idx);
        }
    }

    for (const std::string& active_arm : arm_names_) {
        int topic_idx = -1;
        for (int i = 0; i < (int)armEndPoseNames_.size(); ++i) {
            if (armEndPoseNames_[i] == active_arm) {
                topic_idx = i;
                break;
            }
        }
        if (topic_idx < 0 || topic_idx >= (int)armEndPoseTopics_.size()) {
            RCLCPP_WARN(this->get_logger(), "[DataStreamer] No topic found for active arm '%s', skipping subscription", active_arm.c_str());
            continue;
        }
        auto sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            armEndPoseTopics_[topic_idx], rclcpp::SensorDataQoS(),
            [this, active_arm](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                int buf_idx = 0;
                auto it = arm_name_to_buffer_idx_.find(active_arm);
                if (it != arm_name_to_buffer_idx_.end()) buf_idx = it->second;
                this->armEndPoseCallback(msg, buf_idx, active_arm);
            });
        subArmEndPoses_.push_back(sub);
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] Subscribed to end pose for arm '%s' (buf_idx=%d, topic_idx=%d): %s",
                    active_arm.c_str(), arm_name_to_buffer_idx_[active_arm], topic_idx, armEndPoseTopics_[topic_idx].c_str());
    }

    // Subscribe to remote sensor poses (teleop master device) for offset-mode action
    for (const auto& pair : arm_pika_pose_topic_) {
        const std::string& arm_name = pair.first;
        const std::string& topic = pair.second;
        auto sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            topic, rclcpp::SensorDataQoS(),
            [this, arm_name](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                this->pikaPoseCallback(msg, arm_name);
            });
        subPikaPoses_[arm_name] = sub;
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] Subscribed to pika_pose for arm '%s': %s (left=%d)",
                    arm_name.c_str(), topic.c_str(), pika_pose_side_is_left_[arm_name]);
    }

    // Subscribe to teleop_status for init_pose reset coordination
    auto teleop_sub = this->create_subscription<data_msgs::msg::TeleopStatus>(
        "/teleop_status", rclcpp::SystemDefaultsQoS(),
        [this](const data_msgs::msg::TeleopStatus::SharedPtr msg) {
            this->teleopStatusCallback(msg);
        });
    subTeleopStatus_ = teleop_sub;
    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Subscribed to /teleop_status");

    for (size_t i = 0; i < gripperEncoderTopics_.size(); ++i) {
        int idx = static_cast<int>(i);
        auto sub = this->create_subscription<data_msgs::msg::Gripper>(
            gripperEncoderTopics_[i], rclcpp::SensorDataQoS(),
            [this, idx](const data_msgs::msg::Gripper::SharedPtr msg) {
                this->gripperEncoderCallback(msg, idx);
            });
        subGripperEncoders_.push_back(sub);
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] Subscribed to gripper: %s", gripperEncoderTopics_[i].c_str());
    }

    // Create action publishers if enabled (use first active arm)
    if (action_enabled && !arm_names_.empty()) {
        if (action_arm_index_ < 0 || action_arm_index_ >= (int)arm_names_.size()) {
            RCLCPP_WARN(this->get_logger(), "[DataStreamer] action.arm_index %d out of range, clamping to 0", action_arm_index_);
            action_arm_index_ = 0;
        }
        std::string arm_name = arm_names_[action_arm_index_];
        // nero_IK index_name convention: '' (single) -> /nero_inference/action, '_r' -> /nero_inference_r/action
        std::string configured_topic;
        this->get_parameter("action.topic", configured_topic);
        std::string pose_topic;
        std::string gripper_topic;
        if (!configured_topic.empty()) {
            // Use topic configured in yaml (e.g. "/nero_inference/action")
            pose_topic = configured_topic;
            gripper_topic = configured_topic + "_gripper";
        } else {
            // Derive suffix from arm_name: strip "arm" prefix (keep underscore) so "arm_r" -> "_r"
            std::string arm_suffix = arm_name;
            if (arm_suffix.rfind("arm_", 0) == 0) {
                arm_suffix = arm_suffix.substr(3);
            }
            pose_topic = "/nero_inference" + arm_suffix + "/action";
            gripper_topic = "/nero_inference" + arm_suffix + "/action_gripper";
        }

        auto pose_pub = this->create_publisher<std_msgs::msg::Float64MultiArray>(pose_topic, 1);
        action_pose_pubs_.push_back(pose_pub);
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] Created action publisher: %s", pose_topic.c_str());

        auto gripper_pub = this->create_publisher<std_msgs::msg::Float32>(gripper_topic, 1);
        action_gripper_pubs_.push_back(gripper_pub);
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] Created action gripper publisher: %s", gripper_topic.c_str());
    } else if (action_enabled) {
        RCLCPP_WARN(this->get_logger(), "[DataStreamer] action.enabled=true but arm.endPose.names is empty, action publishers not created");
    }

    running_ = true;

    if (auto_reset_) {
        if (!sendReset()) {
            RCLCPP_WARN(this->get_logger(), "[DataStreamer] Reset failed, continuing anyway...");
        }
    }

    send_thread_ = std::thread(&DataStreamer::sendLoop, this);
    recv_thread_ = std::thread(&DataStreamer::recvLoop, this);
    action_step_thread_ = std::thread(&DataStreamer::actionStepLoop, this);

    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Started. Observations sent: %ld, Actions received: %ld",
                 observations_sent_.load(), actions_received_.load());
}

void DataStreamer::cameraColorCallback(const sensor_msgs::msg::Image::SharedPtr msg, int cam_buf_idx, const std::string& arm_name) {
    (void)arm_name;
    try {
        cv::Mat img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
        double ts = rclcpp::Time(msg->header.stamp).seconds();
        obs_buffer_->addImage(cam_buf_idx, img, ts);
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_WARN_ONCE(this->get_logger(), "[DataStreamer] cv_bridge error: %s", e.what());
    }
}

void DataStreamer::armEndPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg, int arm_idx, const std::string& arm_name) {
    std::vector<double> pose7 = poseStampedToVec7(*msg);
    double ts = rclcpp::Time(msg->header.stamp).seconds();

    if (arm_idx < 0 || arm_idx >= n_arms_) return;

    std::string gripper_encoder_name;
    if (arm_gripper_name_.find(arm_name) != arm_gripper_name_.end()) {
        gripper_encoder_name = arm_gripper_name_[arm_name];
    }

    double gripper = 1.0;
    if (!gripper_encoder_name.empty()) {
        std::lock_guard<std::mutex> lk(gripper_mutex_);
        auto it = gripper_latest_.find(gripper_encoder_name);
        if (it != gripper_latest_.end()) {
            gripper = it->second;
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "[DataStreamer] armEndPoseCallback: no gripper value for '%s' (arm='%s'), defaulting to 1.0",
                gripper_encoder_name.c_str(), arm_name.c_str());
        }
    }

    // Record init pose on first arrival
    {
        std::lock_guard<std::mutex> lk(init_pose_mutex_);
        if (!arm_init_pose_recorded_[arm_name]) {
            arm_init_pose_[arm_name] = pose7;
            arm_init_pose_recorded_[arm_name] = true;
            RCLCPP_INFO(this->get_logger(), "[DataStreamer] Recorded init_pose for %s: [%.4f, %.4f, %.4f]",
                        arm_name.c_str(), pose7[0], pose7[1], pose7[2]);
        }
    }

    obs_buffer_->addArmState(arm_idx, pose7, gripper, ts);

    // Record cumulative absolute pose for safety delta checking
    {
        std::lock_guard<std::mutex> lk(cumulative_pose_mutex_);
        arm_cumulative_pose_[arm_name] = pose7;
        arm_cumulative_gripper_[arm_name] = gripper;
    }
}

void DataStreamer::pikaPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg, const std::string& arm_name) {
    std::vector<double> pose7 = poseStampedToVec7(*msg);

    std::lock_guard<std::mutex> lk(pika_pose_mutex_);
    // Record init pose on first arrival (calibration at teleop start)
    if (!pika_pose_init_recorded_[arm_name]) {
        pika_pose_init_[arm_name] = pose7;
        pika_pose_init_recorded_[arm_name] = true;
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] pika_pose init recorded for %s: pos=[%.4f, %.4f, %.4f] quat=[%.4f, %.4f, %.4f, %.4f]",
                    arm_name.c_str(),
                    pose7[0], pose7[1], pose7[2],
                    pose7[3], pose7[4], pose7[5], pose7[6]);
    }
    pika_pose_current_[arm_name] = pose7;
}

void DataStreamer::teleopStatusCallback(const data_msgs::msg::TeleopStatus::SharedPtr msg) {
    // When teleop quits (not fail), reset init pose records so the next pose arrival
    // re-records them for the next teleop/inference session.
    if (msg->quit && !msg->fail) {
        std::lock_guard<std::mutex> lk_init(init_pose_mutex_);
        for (auto& pair : arm_init_pose_recorded_) {
            pair.second = false;
        }
        std::lock_guard<std::mutex> lk_pika(pika_pose_mutex_);
        for (auto& pair : pika_pose_init_recorded_) {
            pair.second = false;
        }
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] Teleop quit detected, init_pose records reset");
    }
}

void DataStreamer::gripperEncoderCallback(const data_msgs::msg::Gripper::SharedPtr msg, int idx) {
    (void)idx;
    double gripper_value = 0.0;
    if (msg->distance > 0.001) {
        gripper_value = std::min(1.0, msg->distance / 0.10);
    } else if (msg->angle > 0.001) {
        gripper_value = std::min(1.0, msg->angle / 1.5708);
    }

    if (idx >= 0 && idx < (int)gripperEncoderNames_.size()) {
        std::string enc_name = gripperEncoderNames_[idx];
        std::lock_guard<std::mutex> lk(gripper_mutex_);
        gripper_latest_[enc_name] = gripper_value;
        if (!enc_name.empty() && enc_name[0] != 'g') {
            gripper_latest_["gripper_" + enc_name] = gripper_value;
        }
        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "[DataStreamer] gripper enc='%s' = %.3f (dist=%.4f, angle=%.4f)",
            enc_name.c_str(), gripper_value, msg->distance, msg->angle);
    }
}

std::vector<double> DataStreamer::poseStampedToVec7(const geometry_msgs::msg::PoseStamped& msg) {
    std::vector<double> pose7(7);
    pose7[0] = msg.pose.position.x;
    pose7[1] = msg.pose.position.y;
    pose7[2] = msg.pose.position.z;
    pose7[3] = msg.pose.orientation.x;
    pose7[4] = msg.pose.orientation.y;
    pose7[5] = msg.pose.orientation.z;
    pose7[6] = msg.pose.orientation.w;
    return pose7;
}

std::string DataStreamer::matToBase64Jpeg(const cv::Mat& img, int quality) {
    std::vector<uchar> buf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
    cv::Mat rgb;
    if (img.channels() == 3) {
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    } else {
        rgb = img;
    }
    if (cv::imencode(".jpg", rgb, buf, params)) {
        return base64Encode(buf.data(), buf.size());
    }
    RCLCPP_WARN_ONCE(this->get_logger(), "[DataStreamer] cv::imencode failed for image (channels=%d, size=%dx%d)",
                     img.channels(), img.cols, img.rows);
    return {};
}

bool DataStreamer::checkActionSafety(const std::vector<double>& action, int arm_idx) {
    if (!safety_enabled_) return true;
    if (action.empty()) return false;

    auto now = std::chrono::steady_clock::now();
    double min_interval = 1.0 / safety_publish_rate_hz_;
    if (std::chrono::duration<double>(now - last_safe_publish_time_).count() < min_interval) {
        if (safety_log_rejections_) {
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[Safety] Rate limit: skipping publish (rate=%.1fHz)", safety_publish_rate_hz_);
        }
        return false;
    }

    // Retrieve current cumulative pose for this arm
    std::vector<double> current_pose;
    double current_gripper = 1.0;
    std::string arm_name;
    if (arm_idx >= 0 && arm_idx < static_cast<int>(arm_names_.size())) {
        arm_name = arm_names_[arm_idx];
    }
    {
        std::lock_guard<std::mutex> lk(cumulative_pose_mutex_);
        auto it = arm_cumulative_pose_.find(arm_name);
        if (it != arm_cumulative_pose_.end()) {
            current_pose = it->second;
        }
        auto it_g = arm_cumulative_gripper_.find(arm_name);
        if (it_g != arm_cumulative_gripper_.end()) {
            current_gripper = it_g->second;
        }
    }

    if (current_pose.size() != 7) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "[Safety] No cumulative pose for arm '%s' yet, skipping publish", arm_name.c_str());
        return false;
    }

    // Compute position delta: action target - current pose
    double dx = action[0] - current_pose[0];
    double dy = action[1] - current_pose[1];
    double dz = action[2] - current_pose[2];
    double pos_delta = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (pos_delta > safety_max_pos_delta_) {
        if (safety_log_rejections_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[Safety] Rejected: pos delta %.3fm > %.3fm (target=[%.4f, %.4f, %.4f] vs current=[%.4f, %.4f, %.4f])",
                pos_delta, safety_max_pos_delta_,
                action[0], action[1], action[2],
                current_pose[0], current_pose[1], current_pose[2]);
        }
        return false;
    }

    if (action.size() >= 7) {
        // Action quaternion is a local delta; q_norm checks unit length, angle_delta checks rotation magnitude.
        double cx = current_pose[3], cy = current_pose[4], cz = current_pose[5], cw = current_pose[6];
        double ax = action[3], ay = action[4], az = action[5], aw = action[6];
        double q_norm = std::sqrt(ax*ax + ay*ay + az*az + aw*aw);
        if (std::abs(q_norm - 1.0) > 0.1) {
            if (safety_log_rejections_) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "[Safety] Rejected: quaternion normalization error %.4f (norm=%.4f)", std::abs(q_norm - 1.0), q_norm);
            }
            return false;
        }
        double dot = cx*ax + cy*ay + cz*az + cw*aw;
        double angle_delta = 2.0 * std::acos(std::max(-1.0, std::min(1.0, std::abs(dot))));
        if (angle_delta > safety_max_angle_delta_) {
            if (safety_log_rejections_) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "[Safety] Rejected: quaternion angle delta %.3f rad > %.3f rad", angle_delta, safety_max_angle_delta_);
            }
            return false;
        }
    }

    double gripper_val = (action.size() >= 8) ? action[7] : current_gripper;
    if (gripper_val < 0.0 || gripper_val > 1.0) {
        if (safety_log_rejections_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[Safety] Rejected: gripper value %.3f out of [0,1] range", gripper_val);
        }
        return false;
    }

    {
        std::lock_guard<std::mutex> lk(safety_mutex_);
        last_safe_action_ = action;
        last_safe_publish_time_ = now;
    }
    return true;
}

    void DataStreamer::publishAction(const std::vector<std::vector<double>>& actions) {
        if (actions.empty() || action_pose_pubs_.empty()) return;

        for (size_t arm_idx = 0; arm_idx < actions.size() && arm_idx < action_pose_pubs_.size(); ++arm_idx) {
            const std::vector<double>& action = actions[arm_idx];
            if (action.empty()) continue;

            if (!checkActionSafety(action, arm_idx)) continue;

            std::string arm_name;
            if (arm_idx < arm_names_.size()) {
                arm_name = arm_names_[arm_idx];
            }

            std::string arm_init_str = "[not recorded]";
            std::string sensor_delta_str = "[no data]";
            {
                std::lock_guard<std::mutex> lk(init_pose_mutex_);
                auto it = arm_init_pose_.find(arm_name);
                if (it != arm_init_pose_.end() && it->second.size() == 7) {
                    char buf[128];
                    snprintf(buf, sizeof(buf), "[%.4f, %.4f, %.4f]",
                             it->second[0], it->second[1], it->second[2]);
                    arm_init_str = buf;
                }
            }
            {
                std::lock_guard<std::mutex> lk(pika_pose_mutex_);
                auto it_init = pika_pose_init_.find(arm_name);
                auto it_curr = pika_pose_current_.find(arm_name);
                if (it_init != pika_pose_init_.end() && it_curr != pika_pose_current_.end() &&
                    it_init->second.size() == 7 && it_curr->second.size() == 7) {
                    char buf[128];
                    snprintf(buf, sizeof(buf), "[%.4f, %.4f, %.4f]",
                             it_curr->second[0] - it_init->second[0],
                             it_curr->second[1] - it_init->second[1],
                             it_curr->second[2] - it_init->second[2]);
                    sensor_delta_str = buf;
                }
            }

            std_msgs::msg::Float64MultiArray action_msg;
            action_msg.data = action;
            action_pose_pubs_[arm_idx]->publish(action_msg);

            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[DataStreamer] Action published arm[%s]: delta=[%.4f, %.4f, %.4f], "
                "arm_init=%s, sensor_delta=%s",
                arm_name.c_str(), action[0], action[1], action[2],
                arm_init_str.c_str(), sensor_delta_str.c_str());
        }
    }

void DataStreamer::sendLoop() {
    double dt = 1.0 / inference_freq_;
    double actual_interval = dt * n_obs_steps_;

    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Waiting for observation buffer to fill...");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Starting inference send loop at %.1f Hz, n_obs_steps=%d",
                 inference_freq_, n_obs_steps_);

    while (rclcpp::ok() && running_) {
        auto loop_start = std::chrono::steady_clock::now();

        if (!socket_client_->isConnected()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "[DataStreamer] Not connected to server, attempting reconnect...");
            if (socket_client_->reconnectAttempts() >= SocketClient::maxReconnectAttempts()) {
                RCLCPP_ERROR(this->get_logger(),
                    "[DataStreamer] Max reconnect attempts (%d) reached, giving up.",
                    SocketClient::maxReconnectAttempts());
                break;
            }
            if (!socket_client_->reconnect(5.0)) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "[DataStreamer] Reconnect failed, retrying in 2s...");
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue;
            }
            RCLCPP_INFO(this->get_logger(), "[DataStreamer] Reconnected to inference server.");
            if (auto_reset_ && !sendReset()) {
                RCLCPP_WARN(this->get_logger(), "[DataStreamer] Reset after reconnect failed, continuing anyway...");
            }
        }

        ObservationBuffer::AlignedObs obs;
        if (!obs_buffer_->getAlignedObs(obs)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        Json::Value msg;
        msg["type"] = "observation";
        msg["send_timestamp"] = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() / 1e9;

        // Per-arm structured output: images, poses, grippers, init_pose grouped by arm

        for (const std::string& arm_name : arm_names_) {
            auto it_arm_buf = arm_name_to_buffer_idx_.find(arm_name);
            if (it_arm_buf == arm_name_to_buffer_idx_.end()) {
                msg[arm_name] = Json::nullValue;
                continue;
            }
            int buf_idx = it_arm_buf->second;

            // Collect images for this arm
            // Images in obs.images are indexed by position in ordered_per_arm_cam_names_,
            // which is the same order cameras were added during construction.
            // Simply iterate ordered names and check if each belongs to this arm.
            Json::Value images_arr = Json::arrayValue;
            for (int ordered_cam_pos = 0; ordered_cam_pos < (int)ordered_per_arm_cam_names_.size(); ++ordered_cam_pos) {
                // Does this camera belong to the current arm? arm_to_global_cam_idx_ values are ordered positions.
                bool belongs = false;
                auto it_arm_cams = arm_to_global_cam_idx_.find(arm_name);
                if (it_arm_cams != arm_to_global_cam_idx_.end()) {
                    for (int ordered_pos : it_arm_cams->second) {
                        if (ordered_pos == ordered_cam_pos) { belongs = true; break; }
                    }
                }
                if (!belongs) continue;
                for (int s = 0; s < n_obs_steps_; ++s) {
                    int flat_idx = ordered_cam_pos * n_obs_steps_ + s;
                    if (flat_idx < (int)obs.images.size()) {
                        images_arr.append(matToBase64Jpeg(obs.images[flat_idx], jpeg_quality_));
                    }
                }
            }

            // Collect poses/grippers for this arm
            Json::Value poses_arr = Json::arrayValue;
            Json::Value grippers_arr = Json::arrayValue;
            for (int t = 0; t < n_obs_steps_; ++t) {
                int flat = buf_idx * n_obs_steps_ + t;
                if (flat < (int)obs.poses.size()) {
                    Json::Value pose_val;
                    for (double v : obs.poses[flat]) pose_val.append(v);
                    poses_arr.append(pose_val);
                }
                if (flat < (int)obs.grippers.size()) {
                    grippers_arr.append(obs.grippers[flat]);
                }
            }

            // Init pose (recorded on first pose callback). Send null if not yet available.
            Json::Value init_pose_val;
            {
                std::lock_guard<std::mutex> lk(init_pose_mutex_);
                auto it_ip = arm_init_pose_.find(arm_name);
                if (it_ip != arm_init_pose_.end()) {
                    for (double v : it_ip->second) init_pose_val.append(v);
                }
                // else leave as Json::nullValue
            }

            Json::Value arm_data;
            arm_data["images"] = std::move(images_arr);
            arm_data["poses"] = std::move(poses_arr);
            arm_data["grippers"] = std::move(grippers_arr);
            arm_data["init_pose"] = std::move(init_pose_val);
            arm_data["timestamps"] = Json::arrayValue;
            for (double ts : obs.timestamps) arm_data["timestamps"].append(ts);
            msg[arm_name] = std::move(arm_data);
        }

        if (socket_client_->sendJson(msg)) {
            int64_t count = observations_sent_.load();
            observations_sent_.store(count + 1);
            if ((count % 20) == 0 || count < 5) {
                RCLCPP_INFO(this->get_logger(), "[DataStreamer] Observation #%ld sent (n_obs=%d, active_arms=%d, imgs=%zu)",
                             count + 1, n_obs_steps_, (int)arm_names_.size(), obs.images.size());
            }

            if (debug_enabled_.load()) {
                int idx = debug_frame_idx_.fetch_add(1);
                std::vector<std::string> cam_names_copy = ordered_per_arm_cam_names_;
                std::vector<std::string> arm_names_copy = arm_names_;
                int n_obs_copy = obs_buffer_->n_obs_steps();
                int n_arms_copy = obs_buffer_->n_arms();
                Json::Value msg_copy = msg;
                std::thread([msg_copy, obs_copy = obs, cam_names_copy, arm_names_copy, n_obs_copy, n_arms_copy, idx, this]() {
                    saveDebugFrame(debug_dir_, idx, obs_copy.images, cam_names_copy, n_obs_copy, n_arms_copy, arm_names_copy, obs_copy.poses, obs_copy.grippers, msg_copy);
                }).detach();
            }
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "[DataStreamer] Failed to send observation, closing connection for reconnect...");
            socket_client_->disconnect();
        }

        auto cycle_end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(cycle_end - loop_start).count() / 1e6;
        double sleep_time = actual_interval - elapsed;
        if (sleep_time > 0) {
            std::this_thread::sleep_for(std::chrono::duration<double>(sleep_time));
        }
    }
}

void DataStreamer::actionStepLoop() {
    double dt = 1.0 / inference_freq_;
    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Action step loop started at %.1f Hz", inference_freq_);

    while (rclcpp::ok() && running_) {
        auto loop_start = std::chrono::steady_clock::now();

        {
            std::lock_guard<std::mutex> lk(action_mutex_);
            if (!pending_actions_.empty() && action_sequence_index_ < action_sequence_steps_total_) {
                std::vector<std::vector<double>> current_step(pending_actions_.size());
                bool has_any = false;
                for (size_t i = 0; i < pending_actions_.size(); ++i) {
                    if (action_sequence_index_ < (int)pending_actions_[i].size()) {
                        current_step[i] = pending_actions_[i][action_sequence_index_];
                        has_any = true;
                    }
                }
                if (has_any) {
                    // publishAction(current_step);
                    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                        "[DataStreamer] ActionStep: published step %d/%d",
                        action_sequence_index_ + 1, action_sequence_steps_total_);
                }
                action_sequence_index_++;
                if (action_sequence_index_ >= action_sequence_steps_total_) {
                    int completed = action_sequence_steps_total_;
                    pending_actions_.clear();
                    action_sequence_in_progress_ = false;
                    action_sequence_steps_total_ = 0;
                    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                        "[DataStreamer] ActionStep: sequence complete, all %d steps published",
                        completed);
                }
            }
        }

        auto elapsed = std::chrono::steady_clock::now() - loop_start;
        auto sleep = std::chrono::duration<double>(dt - std::chrono::duration<double>(elapsed).count());
        if (sleep.count() > 0) {
            std::this_thread::sleep_for(sleep);
        } else {
            std::this_thread::yield();
        }
    }
}

void DataStreamer::recvLoop() {
    while (rclcpp::ok() && running_) {  
        if (!socket_client_->isConnected()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        Json::Value resp = socket_client_->recvJson(2.0);
        if (resp.isNull()) continue;

        std::string type = resp.get("type", "").asString();
        if (type == "action") {
            std::vector<std::vector<std::vector<double>>> actions;
            actions.resize(arm_names_.size());

            bool has_any_action = false;
            for (size_t i = 0; i < arm_names_.size(); ++i) {
                // Strip "arm_" prefix so "arm_l" -> "action_l", "arm_r" -> "action_r"
                std::string arm_name = arm_names_[i];
                if (arm_name.rfind("arm_", 0) == 0) {
                    arm_name = arm_name.substr(4);
                }
                std::string key = "action_" + arm_name;
                Json::Value arm_steps = resp[key];
                if (arm_steps.isArray() && !arm_steps.empty()) {
                    std::vector<std::vector<double>> arm_seq;
                    for (Json::ArrayIndex s = 0; s < arm_steps.size(); ++s) {
                        Json::Value& step = arm_steps[s];
                        std::vector<double> single;
                        if (step.isArray()) {
                            single.reserve(step.size());
                            for (Json::ArrayIndex j = 0; j < step.size(); ++j) {
                                single.push_back(step[j].asDouble());
                            }
                        }
                        arm_seq.push_back(std::move(single));
                    }
                    actions[i] = std::move(arm_seq);
                    has_any_action = true;
                }
            }

            if (!has_any_action) {
                Json::Value legacy_arr = resp["action"];
                if (legacy_arr.isArray()) {
                    actions.clear();
                    actions.resize(arm_names_.size());
                    for (Json::ArrayIndex i = 0; i < legacy_arr.size(); ++i) {
                        Json::Value& arm_action = legacy_arr[i];
                        std::vector<double> single;
                        if (arm_action.isArray()) {
                            for (Json::ArrayIndex j = 0; j < arm_action.size(); ++j) {
                                single.push_back(arm_action[j].asDouble());
                            }
                        }
                        if (i < actions.size()) {
                            actions[i].push_back(std::move(single));
                        }
                    }
                    has_any_action = true;
                }
            }

            if (!has_any_action) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "[DataStreamer] No action parsed from JSON. arm_names_=%zu",
                    arm_names_.size());
            }

            if (has_any_action) {
                std::lock_guard<std::mutex> lk(action_mutex_);
                if (action_sequence_in_progress_) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                        "[DataStreamer] Dropping new action: previous sequence (%d steps, at step %d) still in progress",
                        action_sequence_steps_total_, action_sequence_index_);
                } else {
                    pending_actions_ = actions;
                    action_sequence_index_ = 0;
                    action_sequence_steps_total_ = 0;
                    for (const auto& seq : pending_actions_) {
                        action_sequence_steps_total_ = std::max(action_sequence_steps_total_, (int)seq.size());
                    }
                    action_sequence_in_progress_ = true;

                    std::vector<std::vector<double>> current_step(actions.size());
                    for (size_t i = 0; i < actions.size(); ++i) {
                        if (!actions[i].empty() && !actions[i][0].empty()) {
                            current_step[i] = actions[i][0];
                        }
                    }

                    // Log action data for inspection (before publishing)
                    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                        "[DataStreamer] === Action Inspect ===");
                    for (size_t i = 0; i < actions.size(); ++i) {
                        if (actions[i].empty()) continue;
                        const auto& step0 = actions[i][0];

                        std::string arm_name_cur = (i < arm_names_.size()) ? arm_names_[i] : std::to_string(i);
                        std::string curr_pose_str = "[no data]";
                        double curr_gripper = 1.0;
                        {
                            std::lock_guard<std::mutex> lk(cumulative_pose_mutex_);
                            auto it = arm_cumulative_pose_.find(arm_name_cur);
                            if (it != arm_cumulative_pose_.end() && it->second.size() == 7) {
                                char buf[128];
                                snprintf(buf, sizeof(buf), "pos=[%.4f,%.4f,%.4f] quat=[%.4f,%.4f,%.4f,%.4f]",
                                         it->second[0], it->second[1], it->second[2],
                                         it->second[3], it->second[4], it->second[5], it->second[6]);
                                curr_pose_str = buf;
                            }
                            auto it_g = arm_cumulative_gripper_.find(arm_name_cur);
                            if (it_g != arm_cumulative_gripper_.end()) {
                                curr_gripper = it_g->second;
                            }
                        }

                        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                            "[DataStreamer] arm[%zu/%s] current:  %s gripper=%.4f | action step0: pos=[%.4f,%.4f,%.4f] quat=[%.4f,%.4f,%.4f,%.4f] gripper=%.4f",
                            i, arm_name_cur.c_str(),
                            curr_pose_str.c_str(), curr_gripper,
                            step0.size() > 0 ? step0[0] : 0.0,
                            step0.size() > 1 ? step0[1] : 0.0,
                            step0.size() > 2 ? step0[2] : 0.0,
                            step0.size() > 3 ? step0[3] : 0.0,
                            step0.size() > 4 ? step0[4] : 0.0,
                            step0.size() > 5 ? step0[5] : 0.0,
                            step0.size() > 6 ? step0[6] : 0.0,
                            step0.size() > 7 ? step0[7] : 0.0);
                    }

                    // publishAction(current_step);
                    actions_received_.fetch_add(1);
                }  // end else (action_sequence_in_progress_ guard)
            }  // end if (has_any_action)
        } else if (type == "reset_ack") {
            // Already logged in sendReset
        } else if (!type.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                "[DataStreamer] Unknown message type: %s", type.c_str());
        }
    }
}

DataStreamer::~DataStreamer() {
    running_ = false;
    if (send_thread_.joinable()) send_thread_.join();
    if (recv_thread_.joinable()) recv_thread_.join();
    if (action_step_thread_.joinable()) action_step_thread_.join();
}

bool DataStreamer::sendReset() {
    Json::Value reset_msg;
    reset_msg["type"] = "reset";
    if (socket_client_->sendJson(reset_msg)) {
        Json::Value resp = socket_client_->recvJson(5.0);
        if (!resp.isNull() && resp.get("type", "").asString() == "reset_ack") {
            RCLCPP_INFO(this->get_logger(), "[DataStreamer] Server reset acknowledged.");
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(false);
    auto node = std::make_shared<DataStreamer>(options);

    std::thread spin_thread([&node]() {
        while (rclcpp::ok()) {
            rclcpp::spin_some(node);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    spin_thread.join();
    rclcpp::shutdown();
    return 0;
}
