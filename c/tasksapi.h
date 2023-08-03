#pragma once
#include <string>
#include <optional>
#include <memory>
#include <variant>
#include <vector>
#include <opencv2/opencv.hpp>

#if _WIN32
#   ifdef MEDIAPIPE_EXPORT
#       define TASKS_API __declspec(dllexport)
#   else
#       define TASKS_API __declspec(dllimport)
#   endif
#else
#   define TASKS_API __attribute__((visibility("default")))
#endif

namespace TasksApi
{
    // copied from task/cc/core/baseoptions.h
    struct BaseOptions
    {
        std::unique_ptr<std::string> model_asset_buffer;
        std::string model_asset_path = "";
        enum Delegate 
        {
            CPU = 0,
            GPU = 1,
            EDGETPU_NNAPI = 2,
        };

        Delegate delegate = CPU;

        struct CpuOptions {};

        struct GpuOptions
        {
            std::string cached_kernel_path;
            std::string serialized_model_dir;
            std::string model_token;
        };

        struct FileDescriptorMeta
        {
            int fd = -1;
            int length = -1;
            int offset = -1;
        } model_asset_descriptor_meta;

        std::optional<std::variant<CpuOptions, GpuOptions>> delegate_options;
    };

    // copied from task/cc/core/running_mode.h
    enum RunningMode
    {
        IMAGE = 1,
        VIDEO = 2,
        LIVE_STREAM = 3,
    };

    // copied from task/cc/vision/pose_landmarker.h
    struct PoseLandmarkerOptions
    {
        BaseOptions base_options;

        RunningMode running_mode = RunningMode::IMAGE;
        int num_poses = 1;
        float min_pose_detection_confidence = 0.5;
        float min_pose_presence_confidence = 0.5;
        float min_tracking_confidence = 0.5;
        bool output_segmentation_masks = false;
    };

    // copied from task/cc/vision/hand_landmarker.h
    struct HandLandmarkerOptions
    {
        BaseOptions base_options;

        RunningMode running_mode = RunningMode::IMAGE;
        int num_hands = 1;
        float min_hand_detection_confidence = 0.5;
        float min_hand_presence_confidence = 0.5;
        float min_tracking_confidence = 0.5;
    };

    struct Category 
    {
        int index;
        float score;
        std::optional<std::string> category_name = std::nullopt;
        std::optional<std::string> display_name = std::nullopt;
    };
    struct Classifications
    {
        std::vector<Category> categories;
        int head_index;
        std::optional<std::string> head_name = std::nullopt;
    };
    struct Landmark {
        float x;
        float y;
        float z;
        std::optional<float> visibility = std::nullopt;
        std::optional<float> presence = std::nullopt;
        std::optional<std::string> name = std::nullopt;
    }; 
    struct NormalizedLandmark {
        float x;
        float y;
        float z;
        std::optional<float> visibility = std::nullopt;
        std::optional<float> presence = std::nullopt;
        std::optional<std::string> name = std::nullopt;
    };
    struct PoseLandmarkerResult 
    {
        //std::optional<std::vector<Image>> segmentation_masks;
        std::vector<std::vector<NormalizedLandmark>> pose_landmarks;
        std::vector<std::vector<Landmark>> pose_world_landmarks;
    };
    struct HandLandmarkerResult 
    {
        std::vector<Classifications> handedness;
        std::vector<std::vector<NormalizedLandmark>> hand_landmarks;
        std::vector<std::vector<Landmark>> hand_world_landmarks;
    };

    // copied from task/components/containers/rect.h
    struct RectF
    {
        float left;
        float top;
        float right;
        float bottom;
        // copied from task/cc/core/running_mode.h
    };

    struct ImageProcessingOptions
    {
        std::optional<RectF> region_of_interest = std::nullopt;
        int rotation_degrees = 0;
    };

    class PoseLandmarker
    {
    public:
        TASKS_API static std::unique_ptr<PoseLandmarker> Create(PoseLandmarkerOptions& options);

        TASKS_API PoseLandmarkerResult Detect(cv::Mat* image, std::optional<ImageProcessingOptions> image_processing_options = std::nullopt);

        TASKS_API PoseLandmarkerResult DetectForVideo(cv::Mat* image, int64 timestamp_ms, std::optional<ImageProcessingOptions> image_processing_options = std::nullopt);

        TASKS_API bool DetectAsync(cv::Mat* image, int64 timestamp_ms, std::optional<ImageProcessingOptions> image_processing_options = std::nullopt);

        TASKS_API bool Close();

    private:
        void* internal_pose_landmarker;
    };

    class HandLandmarker 
    {
    public:
        TASKS_API static std::unique_ptr<HandLandmarker> Create(HandLandmarkerOptions& options);

        TASKS_API HandLandmarkerResult Detect(cv::Mat* image, std::optional<ImageProcessingOptions> image_processing_options = std::nullopt);

        TASKS_API HandLandmarkerResult DetectForVideo(cv::Mat* image, int64 timestamp_ms, std::optional<ImageProcessingOptions> image_processing_options = std::nullopt);

        TASKS_API bool DetectAsync(cv::Mat* image, int64 timestamp_ms, std::optional<ImageProcessingOptions> image_processing_options = std::nullopt);

        TASKS_API bool Close();

    private:
        void* internal_hand_landmarker;
    };

}