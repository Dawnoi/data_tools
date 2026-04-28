#include "dataStreamer.h"
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <cv_bridge/cv_bridge.h>
#include <jsoncpp/json/json.h>
#include <sys/stat.h>
#include <cerrno>
#include <iomanip>
#include <fstream>
#include <sstream>

// =====================================================================
// Helper utilities (anonymous namespace = internal linkage)
// =====================================================================

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
        if (::mkdir(dir.c_str(), 0755) < 0 && errno != EEXIST) {
            return false;
        }
    }
    if (::mkdir(p.c_str(), 0755) < 0 && errno != EEXIST) {
        return false;
    }
    return true;
}

// Saves one observation frame to disk for offline debugging (only when debug.enabled=true).
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


// =====================================================================
// ObservationBuffer — implementation
// =====================================================================

void ObservationBuffer::addImage(int cam_idx, const cv::Mat& img, double ts) {
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

void ObservationBuffer::addArmState(int arm_idx, const std::vector<double>& pose, double gripper, double ts) {
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

bool ObservationBuffer::getAlignedObs(AlignedObs& out) {
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

    // Align images by timestamp
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

    // Align arm poses by timestamp
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


// =====================================================================
// SocketClient — implementation
// =====================================================================

bool SocketClient::connect(double timeout_sec) {
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

bool SocketClient::reconnect(double timeout_sec) {
    reconnect_attempts_++;
    RCLCPP_INFO(rclcpp::get_logger("SocketClient"),
                "[SocketClient] Reconnect #%d to %s:%d (attempt %d of %d)",
                reconnect_attempts_, host_.c_str(), port_, reconnect_attempts_, max_reconnect_attempts_);
    return connect(timeout_sec);
}

void SocketClient::disconnect() {
    if (sock_fd_ >= 0) {
        ::close(sock_fd_);
        sock_fd_ = -1;
    }
    connected_ = false;
}

bool SocketClient::sendJson(const Json::Value& obj) {
    if (!connected_ || sock_fd_ < 0) return false;
    std::string msg = Json::FastWriter().write(obj) + "\n";
    ssize_t sent = ::send(sock_fd_, msg.c_str(), msg.size(), MSG_NOSIGNAL);
    return sent == static_cast<ssize_t>(msg.size());
}

Json::Value SocketClient::recvJson(double timeout_sec) {
    Json::Value result;
    if (!connected_ || sock_fd_ < 0) return result;

    std::string line;
    line.reserve(4096);
    auto deadline = std::chrono::steady_clock::now()
        + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(timeout_sec));

    while (true) {
        char buf[4096];
        ssize_t n = ::recv(sock_fd_, buf, sizeof(buf) - 1, 0);
        if (n <= 0) {
            if (n < 0) {
                RCLCPP_WARN(rclcpp::get_logger("SocketClient"), "recvJson: recv error errno=%d", errno);
            }
            break;
        }
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
                    if (parsed.isObject()) {
                        return parsed;
                    }
                    RCLCPP_WARN(rclcpp::get_logger("SocketClient"),
                                "recvJson: discarding non-object JSON (type=%d)", parsed.type());
                } else {
                    RCLCPP_WARN(rclcpp::get_logger("SocketClient"),
                                "recvJson: JSON parse error, raw='%.100s'", msg.c_str());
                }
            }
        }

        if (std::chrono::steady_clock::now() >= deadline) break;
    }
    return result;
}


// =====================================================================
// DataStreamer — implementation
// =====================================================================

DataStreamer::DataStreamer(const rclcpp::NodeOptions& options) : Node("data_streamer", options) {

    // ----------------------------------------------------------------
    // 1. Declare parameters with defaults
    // ----------------------------------------------------------------
    this->declare_parameter<std::string>("server.host", "127.0.0.1");
    this->declare_parameter<int>("server.port", 8007);
    this->declare_parameter<double>("inference.freq", 20.0);
    this->declare_parameter<int>("inference.n_obs_steps", 2);
    this->declare_parameter<double>("inference.camera_freq", 30.0);
    this->declare_parameter<int>("inference.jpeg_quality", 85);
    this->declare_parameter<bool>("inference.auto_reset", true);
    this->declare_parameter<int>("action.arm_index", 0);
    this->declare_parameter<bool>("action.enabled", true);
    this->declare_parameter<bool>("debug.enabled", false);
    this->declare_parameter<std::string>("debug.dir", "/home/yxgn/data_streamer_debug");

    this->declare_parameter<std::string>("arm.active_arms", "arm_l,arm_r");
    this->declare_parameter<std::string>("arm.active_arm_gripper_names", "arm_l:gripper_l,arm_r:gripper_r");
    this->declare_parameter<std::string>("arm.per_arm_cameras", "arm_l:fisheye_l,arm_l:fisheye_r,arm_r:fisheye_l,arm_r:fisheye_r");

    this->declare_parameter<bool>("safety.enabled", true);
    this->declare_parameter<double>("safety.max_position_delta", 0.05);
    this->declare_parameter<double>("safety.max_angle_delta", 0.5);
    this->declare_parameter<double>("safety.publish_rate_hz", 20.0);
    this->declare_parameter<bool>("safety.log_rejections", true);

    // ----------------------------------------------------------------
    // 2. Load ROS2 subscription topic parameters
    // ----------------------------------------------------------------
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
    desc.name = "arm.remote_pose.topics";
    this->declare_parameter("arm.remote_pose.topics", rclcpp::ParameterValue(std::vector<std::string>()), desc);
    desc.name = "arm.remote_pose.names";
    this->declare_parameter("arm.remote_pose.names", rclcpp::ParameterValue(std::vector<std::string>()), desc);
    desc.name = "gripper.encoder.topics";
    this->declare_parameter("gripper.encoder.topics", rclcpp::ParameterValue(std::vector<std::string>()), desc);
    desc.name = "gripper.encoder.names";
    this->declare_parameter("gripper.encoder.names", rclcpp::ParameterValue(std::vector<std::string>()), desc);

    // ----------------------------------------------------------------
    // 3. Get all parameters
    // ----------------------------------------------------------------
    this->get_parameter("camera.color.topics", cameraColorTopics_);
    this->get_parameter("camera.color.names", cameraColorNames_);
    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Camera names (%zu): [%s], topics (%zu)",
                cameraColorNames_.size(),
                cameraColorNames_.empty() ? "EMPTY" : [&]() {
                    std::string s = cameraColorNames_[0];
                    for (size_t i = 1; i < cameraColorNames_.size(); ++i) s += ", " + cameraColorNames_[i];
                    return s;
                }().c_str(),
                cameraColorTopics_.size());

    this->get_parameter("arm.endPose.topics", armEndPoseTopics_);
    this->get_parameter("arm.endPose.names", armEndPoseNames_);

    std::vector<std::string> remote_pose_topics_raw;
    std::vector<std::string> remote_pose_names_raw;
    this->get_parameter("arm.remote_pose.topics", remote_pose_topics_raw);
    this->get_parameter("arm.remote_pose.names", remote_pose_names_raw);
    for (size_t i = 0; i < remote_pose_topics_raw.size() && i < remote_pose_names_raw.size(); ++i) {
        arm_pika_pose_topic_[remote_pose_names_raw[i]] = remote_pose_topics_raw[i];
    }

    // Parse active arms from comma-separated string
    std::string active_arms_str;
    this->get_parameter("arm.active_arms", active_arms_str);
    arm_names_.clear();
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
                        arm_gripper_name_[part.substr(0, colon)] = part.substr(colon + 1);
                    }
                }
                start = i + 1;
            }
        }
    }
    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Active arms: %d, names: [%s]",
                n_arms_, active_arms_str.c_str());

    // Build per-arm camera mapping
    std::string per_arm_cameras_str;
    this->get_parameter("arm.per_arm_cameras", per_arm_cameras_str);
    std::map<std::string, std::vector<std::string>> per_arm_cameras;
    {
        std::string s = per_arm_cameras_str;
        size_t start = 0;
        for (size_t i = 0; i <= s.size(); ++i) {
            if (i == s.size() || s[i] == ',') {
                std::string part = s.substr(start, i - start);
                if (!part.empty()) {
                    size_t colon = part.find(':');
                    if (colon != std::string::npos) {
                        per_arm_cameras[part.substr(0, colon)].push_back(part.substr(colon + 1));
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

    std::map<std::string, std::vector<int>> arm_to_global_cam_idx;
    std::vector<std::string> ordered_per_arm_cam_names;
    for (const auto& arm_pair : per_arm_cameras) {
        for (const std::string& cam_name : arm_pair.second) {
            auto it = camera_name_to_global_idx.find(cam_name);
            if (it != camera_name_to_global_idx.end()) {
                int ordered_pos = (int)ordered_per_arm_cam_names.size();
                arm_to_global_cam_idx[arm_pair.first].push_back(ordered_pos);
                ordered_per_arm_cam_names.push_back(cam_name);
            } else {
                RCLCPP_WARN(this->get_logger(), "[DataStreamer] Camera '%s' not found in cameraColorNames_, skipping",
                            cam_name.c_str());
            }
        }
    }

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
    this->get_parameter("action.arm_index", action_arm_index_);
    bool action_enabled;
    this->get_parameter("action.enabled", action_enabled);

    bool debug_enabled;
    this->get_parameter("debug.enabled", debug_enabled);
    this->get_parameter("debug.dir", debug_dir_);
    debug_enabled_.store(debug_enabled);
    if (debug_enabled_) {
        if (!mkdirRecursive(debug_dir_)) {
            RCLCPP_WARN(this->get_logger(), "[DataStreamer] Failed to create debug dir '%s'", debug_dir_.c_str());
        }
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] DEBUG mode ON: saving to %s", debug_dir_.c_str());
    }

    this->get_parameter("safety.enabled", safety_enabled_);
    this->get_parameter("safety.max_position_delta", safety_max_pos_delta_);
    this->get_parameter("safety.max_angle_delta", safety_max_angle_delta_);
    this->get_parameter("safety.publish_rate_hz", safety_publish_rate_hz_);
    this->get_parameter("safety.log_rejections", safety_log_rejections_);
    last_safe_publish_time_ = std::chrono::steady_clock::now() - std::chrono::milliseconds(100);

    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Safety: enabled=%d, max_pos_delta=%.3fm, angle_delta=%.3f rad, rate=%.1fHz",
                safety_enabled_, safety_max_pos_delta_, safety_max_angle_delta_, safety_publish_rate_hz_);

    // ----------------------------------------------------------------
    // 4. Connect to inference server
    // ----------------------------------------------------------------
    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Connecting to %s:%d", server_host_.c_str(), server_port_);
    socket_client_ = std::make_unique<SocketClient>(server_host_, server_port_);
    if (!socket_client_->connect(10.0)) {
        RCLCPP_ERROR(this->get_logger(), "[DataStreamer] Failed to connect to inference server, will retry in send loop!");
    } else {
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] Connected to inference server.");
    }

    obs_buffer_ = std::make_unique<ObservationBuffer>(
        n_total_cameras, n_arms_, n_obs_steps_, inference_freq_, camera_freq_);

    // ----------------------------------------------------------------
    // 5. Subscribe to ROS2 topics
    // ----------------------------------------------------------------
    // Cameras: subscribed per active arm, using ordered buffer index
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
            RCLCPP_INFO(this->get_logger(), "[DataStreamer] Camera (arm=%s): %s -> buf_idx=%d",
                        arm_name.c_str(), cameraColorTopics_[global_idx].c_str(), ordered_idx);
        }
    }

    // Arm end poses (FK)
    for (const std::string& active_arm : arm_names_) {
        int topic_idx = -1;
        for (int i = 0; i < (int)armEndPoseNames_.size(); ++i) {
            if (armEndPoseNames_[i] == active_arm) { topic_idx = i; break; }
        }
        if (topic_idx < 0 || topic_idx >= (int)armEndPoseTopics_.size()) {
            RCLCPP_WARN(this->get_logger(), "[DataStreamer] No topic found for active arm '%s', skipping", active_arm.c_str());
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
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] End pose arm '%s' (buf_idx=%d): %s",
                    active_arm.c_str(), arm_name_to_buffer_idx_[active_arm], armEndPoseTopics_[topic_idx].c_str());
    }

    // Remote sensor poses (teleop master device)
    for (const auto& pair : arm_pika_pose_topic_) {
        const std::string& arm_name = pair.first;
        const std::string& topic = pair.second;
        auto sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            topic, rclcpp::SensorDataQoS(),
            [this, arm_name](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                this->pikaPoseCallback(msg, arm_name);
            });
        subPikaPoses_[arm_name] = sub;
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] pika_pose arm '%s': %s", arm_name.c_str(), topic.c_str());
    }

    // Teleop status (reset init poses on teleop quit)
    auto teleop_sub = this->create_subscription<data_msgs::msg::TeleopStatus>(
        "/teleop_status", rclcpp::SystemDefaultsQoS(),
        [this](const data_msgs::msg::TeleopStatus::SharedPtr msg) {
            this->teleopStatusCallback(msg);
        });
    subTeleopStatus_ = teleop_sub;

    // Gripper encoders
    for (size_t i = 0; i < gripperEncoderTopics_.size(); ++i) {
        int idx = static_cast<int>(i);
        auto sub = this->create_subscription<data_msgs::msg::Gripper>(
            gripperEncoderTopics_[i], rclcpp::SensorDataQoS(),
            [this, idx](const data_msgs::msg::Gripper::SharedPtr msg) {
                this->gripperEncoderCallback(msg, idx);
            });
        subGripperEncoders_.push_back(sub);
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] Gripper: %s", gripperEncoderTopics_[i].c_str());
    }

    // ----------------------------------------------------------------
    // 6. Create action publishers
    // ----------------------------------------------------------------
    if (action_enabled && !arm_names_.empty()) {
        if (action_arm_index_ < 0 || action_arm_index_ >= (int)arm_names_.size()) {
            RCLCPP_WARN(this->get_logger(), "[DataStreamer] action.arm_index %d out of range, clamping to 0", action_arm_index_);
            action_arm_index_ = 0;
        }
        std::string arm_name = arm_names_[action_arm_index_];
        std::string configured_topic;
        this->get_parameter("action.topic", configured_topic);
        std::string pose_topic;
        std::string gripper_topic;
        if (!configured_topic.empty()) {
            pose_topic = configured_topic;
            gripper_topic = configured_topic + "_gripper";
        } else {
            // Derive topic suffix from arm_name: "arm_r" -> "_r" -> /nero_inference_r/action
            std::string arm_suffix = arm_name;
            if (arm_suffix.rfind("arm_", 0) == 0) {
                arm_suffix = arm_suffix.substr(3);
            }
            pose_topic = "/nero_inference" + arm_suffix + "/action";
            gripper_topic = "/nero_inference" + arm_suffix + "/action_gripper";
        }
        auto pose_pub = this->create_publisher<std_msgs::msg::Float64MultiArray>(pose_topic, 1);
        action_pose_pubs_.push_back(pose_pub);
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] Action publisher: %s", pose_topic.c_str());
        auto gripper_pub = this->create_publisher<std_msgs::msg::Float32>(gripper_topic, 1);
        action_gripper_pubs_.push_back(gripper_pub);
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] Action gripper publisher: %s", gripper_topic.c_str());
    } else if (action_enabled) {
        RCLCPP_WARN(this->get_logger(), "[DataStreamer] action.enabled=true but no active arms, publishers not created");
    }

    // ----------------------------------------------------------------
    // 7. Start worker threads
    // ----------------------------------------------------------------
    running_ = true;
    send_thread_ = std::thread(&DataStreamer::sendLoop, this);
    recv_thread_ = std::thread(&DataStreamer::recvLoop, this);
    action_step_thread_ = std::thread(&DataStreamer::actionStepLoop, this);

    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Started. Observations sent: %ld",
                observations_sent_.load());
}


// =====================================================================
// ROS2 callbacks
// =====================================================================

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

    // Lookup latest gripper value for this arm
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
        }
    }

    // Push into ring buffer for observation alignment
    obs_buffer_->addArmState(arm_idx, pose7, gripper, ts);

    // Update cumulative pose (used for safety delta checks in checkActionSafety)
    {
        std::lock_guard<std::mutex> lk(cumulative_pose_mutex_);
        arm_cumulative_pose_[arm_name] = pose7;
        arm_cumulative_gripper_[arm_name] = gripper;
    }
}

void DataStreamer::pikaPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg, const std::string& arm_name) {
    std::vector<double> pose7 = poseStampedToVec7(*msg);

    std::lock_guard<std::mutex> lk(pika_pose_mutex_);
    if (!pika_pose_init_recorded_[arm_name]) {
        pika_pose_init_[arm_name] = pose7;
        pika_pose_init_recorded_[arm_name] = true;
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] pika_pose init recorded for %s: [%.4f, %.4f, %.4f]",
                    arm_name.c_str(), pose7[0], pose7[1], pose7[2]);
    }
    pika_pose_current_[arm_name] = pose7;
}

void DataStreamer::teleopStatusCallback(const data_msgs::msg::TeleopStatus::SharedPtr msg) {
    // When teleop quits (not fail), reset init pose records so the next pose arrival
    // re-records them for the next teleop/inference session.
    if (msg->quit && !msg->fail) {
        std::lock_guard<std::mutex> lk(pika_pose_mutex_);
        for (auto& pair : pika_pose_init_recorded_) {
            pair.second = false;
        }
        RCLCPP_INFO(this->get_logger(), "[DataStreamer] Teleop quit detected, init_pose records reset");
    }
}

void DataStreamer::gripperEncoderCallback(const data_msgs::msg::Gripper::SharedPtr msg, int idx) {
    double gripper_value = 0.0;
    if (msg->distance > 0.001) {
        gripper_value = std::min(0.1, msg->distance);
    } else if (msg->angle > 0.001) {
        gripper_value = std::min(0.1, msg->angle);
    }

    if (idx >= 0 && idx < (int)gripperEncoderNames_.size()) {
        std::string enc_name = gripperEncoderNames_[idx];
        std::lock_guard<std::mutex> lk(gripper_mutex_);
        gripper_latest_[enc_name] = gripper_value;
    }
}


// =====================================================================
// Helper implementations
// =====================================================================

std::vector<double> DataStreamer::poseStampedToVec7(const geometry_msgs::msg::PoseStamped& msg) {
    return {msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w};
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
    RCLCPP_WARN_ONCE(this->get_logger(), "[DataStreamer] cv::imencode failed (channels=%d, size=%dx%d)",
                     img.channels(), img.cols, img.rows);
    return {};
}


// =====================================================================
// Safety filter
// =====================================================================

bool DataStreamer::checkActionSafety(const std::vector<double>& action, int arm_idx, const std::string& arm_name,
                                   const std::vector<double>& current_pose, double current_gripper) {
    if (!safety_enabled_) return true;
    if (action.empty()) return false;

    auto now = std::chrono::steady_clock::now();
    double min_interval = 1.0 / safety_publish_rate_hz_;
    if (std::chrono::duration<double>(now - last_safe_publish_time_).count() < min_interval) {
        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "[Safety] Rate limit: skipping (rate=%.1fHz)", safety_publish_rate_hz_);
        return false;
    }

    if (current_pose.size() != 7) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "[Safety] No pose for arm '%s' yet, skipping publish", arm_name.c_str());
        return false;
    }

    // Position delta check
    double dx = action[0] - current_pose[0];
    double dy = action[1] - current_pose[1];
    double dz = action[2] - current_pose[2];
    double pos_delta = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (pos_delta > safety_max_pos_delta_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "[Safety] Rejected: pos delta %.3fm > %.3fm", pos_delta, safety_max_pos_delta_);
        return false;
    }

    // Orientation delta check (quaternion)
    if (action.size() >= 7) {
        double cx = current_pose[3], cy = current_pose[4], cz = current_pose[5], cw = current_pose[6];
        double ax = action[3], ay = action[4], az = action[5], aw = action[6];
        double q_norm = std::sqrt(ax*ax + ay*ay + az*az + aw*aw);
        if (std::abs(q_norm - 1.0) > 0.1) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[Safety] Rejected: quaternion norm error %.4f", std::abs(q_norm - 1.0));
            return false;
        }
        double dot = cx*ax + cy*ay + cz*az + cw*aw;
        double angle_delta = 2.0 * std::acos(std::max(-1.0, std::min(1.0, std::abs(dot))));
        if (angle_delta > safety_max_angle_delta_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[Safety] Rejected: angle delta %.3f rad > %.3f rad", angle_delta, safety_max_angle_delta_);
            return false;
        }
    }

    // Gripper range check
    double gripper_val = (action.size() >= 8) ? action[7] : current_gripper;
    if (gripper_val < 0.0 || gripper_val > 0.1) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "[Safety] Rejected: gripper %.3f out of [0,0.1]", gripper_val);
        return false;
    }

    last_safe_publish_time_ = now;
    return true;
}


// =====================================================================
// Action publishing
// =====================================================================

bool DataStreamer::publishAction(const std::vector<std::vector<double>>& actions) {
    if (actions.empty() || action_pose_pubs_.empty()) return false;

    bool any_published = false;
    for (size_t arm_idx = 0; arm_idx < actions.size() && arm_idx < action_pose_pubs_.size(); ++arm_idx) {
        const std::vector<double>& action = actions[arm_idx];
        if (action.empty()) continue;

        // Retrieve FK pose + gripper (for logging) and check safety
        std::vector<double> current_pose;
        double current_gripper = 1.0;
        std::string arm_name;
        if (arm_idx < (int)arm_names_.size()) {
            arm_name = arm_names_[arm_idx];
        }
        {
            std::lock_guard<std::mutex> lk(cumulative_pose_mutex_);
            auto it_pose = arm_cumulative_pose_.find(arm_name);
            if (it_pose != arm_cumulative_pose_.end()) {
                current_pose = it_pose->second;
            }
            auto it_g = arm_cumulative_gripper_.find(arm_name);
            if (it_g != arm_cumulative_gripper_.end()) {
                current_gripper = it_g->second;
            }
        }

        if (!checkActionSafety(action, arm_idx, arm_name, current_pose, current_gripper)) continue;

        // Publish pose action
        std_msgs::msg::Float64MultiArray action_msg;
        action_msg.data = action;

        // DEBUG: log FK pose vs action vs published value
        RCLCPP_WARN(this->get_logger(),
            "[DataStreamer] >>> PUBLISH arm[%s] | "
            "FK=[%.4f, %.4f, %.4f] | "
            "action=[%.4f, %.4f, %.4f, w=%.4f, gripper=%.4f]",
            arm_name.c_str(),
            current_pose.size() >= 3 ? current_pose[0] : 0.0,
            current_pose.size() >= 3 ? current_pose[1] : 0.0,
            current_pose.size() >= 3 ? current_pose[2] : 0.0,
            action[0], action[1], action[2],
            action.size() >= 7 ? action[6] : 0.0,
            action.size() >= 8 ? action[7] : -1.0);

        action_pose_pubs_[arm_idx]->publish(action_msg);

        // Publish gripper action (Float32). Falls back to current position if action[7] absent.
        if (arm_idx < action_gripper_pubs_.size()) {
            double gripper_value = (action.size() >= 8) ? action[7] : current_gripper;
            std_msgs::msg::Float32 gripper_msg;
            gripper_msg.data = static_cast<float>(gripper_value);
            action_gripper_pubs_[arm_idx]->publish(gripper_msg);
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[DataStreamer] Gripper arm[%s]: %.4f", arm_name.c_str(), gripper_msg.data);
        }

        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "[DataStreamer] Action arm[%s]: [%.4f, %.4f, %.4f]",
            arm_name.c_str(), action[0], action[1], action[2]);
        any_published = true;
    }
    return any_published;
}


// =====================================================================
// sendLoop — observation producer
// =====================================================================
// Sends a JSON observation to the inference server every cycle,
// then blocks on the pipeline CV until the action sequence completes.
// Pipeline: send obs -> recv action -> execute steps -> wait convergence -> next cycle

void DataStreamer::sendLoop() {
    double dt = 1.0 / inference_freq_;
    double actual_interval = dt * n_obs_steps_;

    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Waiting for observation buffer to fill...");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Pipeline: reset -> (obs -> action -> execute -> obs -> ...)");

    bool first_cycle = true;
    while (rclcpp::ok() && running_) {
        auto loop_start = std::chrono::steady_clock::now();

        if (!socket_client_->isConnected()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "[DataStreamer] Not connected, attempting reconnect...");
            if (socket_client_->reconnectAttempts() >= SocketClient::maxReconnectAttempts()) {
                RCLCPP_ERROR(this->get_logger(), "[DataStreamer] Max reconnect attempts reached, giving up.");
                break;
            }
            if (!socket_client_->reconnect(5.0)) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "[DataStreamer] Reconnect failed, retrying in 2s...");
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue;
            }
            RCLCPP_INFO(this->get_logger(), "[DataStreamer] Reconnected to inference server.");
            first_cycle = true;
        }

        // Send initial reset on first cycle or after reconnect
        if (auto_reset_ && first_cycle) {
            RCLCPP_INFO(this->get_logger(), "[DataStreamer] Sending initial reset...");
            if (!sendReset()) {
                RCLCPP_WARN(this->get_logger(), "[DataStreamer] Initial reset failed, retrying in 1s...");
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            first_cycle = false;
        }

        // Wait until observation buffer has enough aligned samples
        ObservationBuffer::AlignedObs obs;
        if (!obs_buffer_->getAlignedObs(obs)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Build JSON observation message, grouped by arm
        Json::Value msg;
        msg["type"] = "observation";
        msg["send_timestamp"] = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() / 1e9;

        for (const std::string& arm_name : arm_names_) {
            auto it_arm_buf = arm_name_to_buffer_idx_.find(arm_name);
            if (it_arm_buf == arm_name_to_buffer_idx_.end()) {
                msg[arm_name] = Json::nullValue;
                continue;
            }
            int buf_idx = it_arm_buf->second;

            // Images for this arm's cameras
            Json::Value images_arr = Json::arrayValue;
            for (int ordered_cam_pos = 0; ordered_cam_pos < (int)ordered_per_arm_cam_names_.size(); ++ordered_cam_pos) {
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

            // Poses and grippers
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

            // Current FK pose (real-time, used by inference server as action base reference)
            Json::Value current_pose_val;
            {
                std::lock_guard<std::mutex> lk(cumulative_pose_mutex_);
                auto it_cp = arm_cumulative_pose_.find(arm_name);
                if (it_cp != arm_cumulative_pose_.end()) {
                    for (double v : it_cp->second) current_pose_val.append(v);
                }
            }

            Json::Value arm_data;
            arm_data["images"] = std::move(images_arr);
            arm_data["poses"] = std::move(poses_arr);
            arm_data["grippers"] = std::move(grippers_arr);
            arm_data["arm_current_pose"] = std::move(current_pose_val);
            arm_data["timestamps"] = Json::arrayValue;
            for (double ts : obs.timestamps) arm_data["timestamps"].append(ts);
            msg[arm_name] = std::move(arm_data);
        }

        // Send observation
        if (socket_client_->sendJson(msg)) {
            int64_t count = observations_sent_.load();
            observations_sent_.store(count + 1);
            RCLCPP_INFO(this->get_logger(), "[DataStreamer] Obs #%ld sent (n_obs=%d, arms=%d, imgs=%zu), waiting for action...",
                        count + 1, n_obs_steps_, (int)arm_names_.size(), obs.images.size());

            // Debug frame save (async, non-blocking)
            if (debug_enabled_.load()) {
                int idx = debug_frame_idx_.fetch_add(1);
                std::vector<std::string> cam_names_copy = ordered_per_arm_cam_names_;
                std::vector<std::string> arm_names_copy = arm_names_;
                int n_obs_copy = obs_buffer_->n_obs_steps();
                int n_arms_copy = obs_buffer_->n_arms();
                Json::Value msg_copy = msg;
                std::thread([msg_copy, obs_copy = obs, cam_names_copy, arm_names_copy, n_obs_copy, n_arms_copy, idx, this]() {
                    saveDebugFrame(debug_dir_, idx, obs_copy.images, cam_names_copy, n_obs_copy, n_arms_copy,
                                   arm_names_copy, obs_copy.poses, obs_copy.grippers, msg_copy);
                }).detach();
            }
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "[DataStreamer] Failed to send observation, closing connection for reconnect...");
            socket_client_->disconnect();
            first_cycle = true;
            continue;
        }

        // Signal recvLoop and block until action pipeline completes
        {
            std::lock_guard<std::mutex> lk(pipeline_mutex_);
            waiting_for_action_ = true;
        }
        pipeline_cv_.notify_one();

        {
            std::unique_lock<std::mutex> lk(pipeline_mutex_);
            pipeline_cv_.wait(lk, [this]() { return !running_ || !waiting_for_action_; });
            if (!running_) break;
            RCLCPP_INFO(this->get_logger(), "[DataStreamer] Action sequence completed, ready for next cycle.");
        }
        first_cycle = false;

        auto cycle_end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(cycle_end - loop_start).count() / 1e6;
        double sleep_time = actual_interval - elapsed;
        if (sleep_time > 0) {
            std::this_thread::sleep_for(std::chrono::duration<double>(sleep_time));
        }
    }
}


// =====================================================================
// waitForExecutionComplete — poll FK convergence
// =====================================================================

void DataStreamer::waitForExecutionComplete() {
    const double kPosThreshold = 0.01;   // 1cm
    const double kQuatThreshold = 0.02; // ~1.1deg
    const double kMaxWaitSec = 5.0;

    std::vector<std::vector<double>> target_poses;
    {
        std::lock_guard<std::mutex> lk(exec_target_mutex_);
        target_poses = exec_target_poses_;
    }

    if (target_poses.empty()) {
        RCLCPP_WARN(this->get_logger(), "[DataStreamer] waitForExecutionComplete: no target poses recorded, skipping");
        return;
    }

    auto deadline = std::chrono::steady_clock::now()
        + std::chrono::duration<double>(kMaxWaitSec);

    while (rclcpp::ok() && running_ && std::chrono::steady_clock::now() < deadline) {
        bool all_converged = true;
        {
            std::lock_guard<std::mutex> lk(cumulative_pose_mutex_);
            for (size_t i = 0; i < arm_names_.size() && i < target_poses.size(); ++i) {
                const std::string& arm_name = arm_names_[i];
                const std::vector<double>& target = target_poses[i];
                if (target.size() != 7) continue;

                auto it_curr = arm_cumulative_pose_.find(arm_name);
                if (it_curr == arm_cumulative_pose_.end() || it_curr->second.size() != 7) {
                    all_converged = false;
                    break;
                }
                const auto& curr = it_curr->second;

                double dx = curr[0] - target[0];
                double dy = curr[1] - target[1];
                double dz = curr[2] - target[2];
                double pos_err = std::sqrt(dx*dx + dy*dy + dz*dz);

                double dot = std::abs(curr[3]*target[3] + curr[4]*target[4]
                                    + curr[5]*target[5] + curr[6]*target[6]);
                double angle_err = 2.0 * std::acos(std::min(1.0, dot));

                if (pos_err > kPosThreshold || angle_err > kQuatThreshold) {
                    all_converged = false;
                    break;
                }
            }
        }

        if (all_converged) {
            RCLCPP_INFO(this->get_logger(), "[DataStreamer] Execution converged: pos_err<%.2fcm, angle_err<%.1fdeg",
                        kPosThreshold * 100, kQuatThreshold * 180.0 / M_PI);
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    RCLCPP_WARN(this->get_logger(), "[DataStreamer] Execution wait timeout (%.1fs), proceeding anyway", kMaxWaitSec);
}


// =====================================================================
// actionStepLoop — action executor
// =====================================================================
// Runs at inference_freq. Each tick publishes one step from the pending
// multi-step sequence. When all steps are published, waits for FK pose
// convergence before notifying sendLoop to continue.

void DataStreamer::actionStepLoop() {
    double dt = 1.0 / inference_freq_;
    RCLCPP_INFO(this->get_logger(), "[DataStreamer] Action step loop started at %.1f Hz", inference_freq_);

    while (rclcpp::ok() && running_) {
        auto loop_start = std::chrono::steady_clock::now();
        bool sequence_done = false;
        int completed_steps = 0;

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
                    bool published = publishAction(current_step);
                    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                        "[DataStreamer] ActionStep: step %d/%d published=%d",
                        action_sequence_index_ + 1, action_sequence_steps_total_, (int)published);
                    if (published) {
                        action_sequence_any_published_ = true;
                    }
                }
                action_sequence_index_++;

                if (action_sequence_index_ >= action_sequence_steps_total_) {
                    completed_steps = action_sequence_steps_total_;
                    // Record FK pose at the last step as convergence target
                    {
                        std::lock_guard<std::mutex> lk(exec_target_mutex_);
                        exec_target_poses_.clear();
                        std::lock_guard<std::mutex> lk2(cumulative_pose_mutex_);
                        for (const std::string& arm_name : arm_names_) {
                            auto it = arm_cumulative_pose_.find(arm_name);
                            if (it != arm_cumulative_pose_.end() && !it->second.empty()) {
                                exec_target_poses_.push_back(it->second);
                            }
                        }
                    }
                    pending_actions_.clear();
                    action_sequence_in_progress_ = false;
                    action_sequence_steps_total_ = 0;
                    sequence_done = true;

                    if (!action_sequence_any_published_) {
                        RCLCPP_WARN(this->get_logger(),
                                    "[DataStreamer] ActionStep: all %d steps safety-rejected, aborting sequence",
                                    completed_steps);
                        waiting_for_action_ = false;
                    }
                    action_sequence_any_published_ = false;
                }
            }
        }

        // Signal sendLoop when the entire sequence is done
        if (sequence_done) {
            if (action_sequence_any_published_) {
                RCLCPP_INFO(this->get_logger(), "[DataStreamer] ActionStep: all %d steps published, waiting for execution...",
                            completed_steps);
                waitForExecutionComplete();
            }
            {
                std::lock_guard<std::mutex> lk(pipeline_mutex_);
                waiting_for_action_ = false;
            }
            pipeline_cv_.notify_one();
            RCLCPP_DEBUG(this->get_logger(), "[DataStreamer] ActionStep: pipeline notified, waiting_for_action_=false");
        }

        auto elapsed = std::chrono::steady_clock::now() - loop_start;
        double sleep_secs = dt - std::chrono::duration<double>(elapsed).count();
        if (sleep_secs > 0.001) {
            std::this_thread::sleep_for(std::chrono::duration<double>(sleep_secs));
        } else {
            std::this_thread::yield();
        }
    }
}


// =====================================================================
// recvLoop — action consumer
// =====================================================================
// Receives JSON responses from the inference server over TCP.
// Parses "action" messages into per-arm multi-step sequences and
// hands them off to actionStepLoop via pending_actions_.

void DataStreamer::recvLoop() {
    while (rclcpp::ok() && running_) {
        if (!socket_client_->isConnected()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }

        // Block until sendLoop has sent an observation and is waiting for action
        constexpr double RECV_TIMEOUT_SEC = 2.0;
        {
            std::unique_lock<std::mutex> lk(pipeline_mutex_);
            pipeline_cv_.wait(lk, [this]() { return !running_ || waiting_for_action_; });
            if (!running_) break;
        }

        RCLCPP_INFO(this->get_logger(), "[DataStreamer] recvLoop: waiting for action...");

        while (rclcpp::ok() && running_) {
            {
                std::lock_guard<std::mutex> lk(pipeline_mutex_);
                if (!waiting_for_action_) break;
            }

            Json::Value resp = socket_client_->recvJson(RECV_TIMEOUT_SEC);
            if (resp.isNull()) {
                static int timeout_count = 0;
                if (++timeout_count % 5 == 1) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                        "[DataStreamer] recvLoop: recvJson timeout #%d", timeout_count);
                }
                continue;
            }

            std::string type = resp.get("type", "").asString();
            RCLCPP_DEBUG(this->get_logger(), "[DataStreamer] recvLoop: type='%s'", type.c_str());

            if (type == "action") {
                std::vector<std::vector<std::vector<double>>> actions;
                actions.resize(arm_names_.size());
                bool has_any_action = false;

                for (size_t i = 0; i < arm_names_.size(); ++i) {
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
                        RCLCPP_DEBUG(this->get_logger(), "[DataStreamer] arm[%zu] parsed %zu steps",
                                     i, actions[i].size());
                    }
                }

                if (!has_any_action) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                        "[DataStreamer] No action parsed from JSON");
                }

                if (has_any_action) {
                    std::lock_guard<std::mutex> lk(action_mutex_);
                    if (action_sequence_in_progress_) {
                        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                            "[DataStreamer] Dropping new action: previous sequence still in progress");
                    } else {
                        pending_actions_ = actions;
                        action_sequence_index_ = 0;
                        action_sequence_steps_total_ = 0;
                        for (const auto& seq : pending_actions_) {
                            action_sequence_steps_total_ = std::max(action_sequence_steps_total_, (int)seq.size());
                        }
                        action_sequence_in_progress_ = true;
                        RCLCPP_INFO(this->get_logger(),
                                    "[DataStreamer] Action received: %d arms, %d steps",
                                    (int)actions.size(), action_sequence_steps_total_);
                    }
                }
            } else if (type == "reset_ack") {
                RCLCPP_DEBUG(this->get_logger(), "[DataStreamer] reset_ack received");
            } else if (!type.empty()) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "[DataStreamer] Unknown message type: %s", type.c_str());
            }
        }
    }
}


// =====================================================================
// Destructor & sendReset
// =====================================================================

DataStreamer::~DataStreamer() {
    running_ = false;
    {
        std::lock_guard<std::mutex> lk(pipeline_mutex_);
        waiting_for_action_ = false;
    }
    pipeline_cv_.notify_all();
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


// =====================================================================
// main
// =====================================================================

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
