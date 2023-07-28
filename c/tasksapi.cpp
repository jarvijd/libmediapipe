#include "tasksapi.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker.h"


using namespace TasksApi;
namespace vision = mediapipe::tasks::vision::pose_landmarker;
namespace core = mediapipe::tasks::vision::core; 

mediapipe::Image MakeImage(cv::Mat* image)
{
	auto input_frame = std::make_shared<mediapipe::ImageFrame>();
	auto mp_format = static_cast<mediapipe::ImageFormat::Format>(mediapipe::ImageFormat::Format::ImageFormat_Format_SRGB);
	uint32_t mp_alignment_boundary = mediapipe::ImageFrame::kGlDefaultAlignmentBoundary;
	input_frame->CopyPixelData(mp_format, image->cols, image->rows, image->data, mp_alignment_boundary);

	mediapipe::Image mpimage(input_frame);
	return mpimage;
}

PoseLandmarkerResult MakeResult(vision::PoseLandmarkerResult& result)
{
	TasksApi::PoseLandmarkerResult pr;
	for (const auto& list : result.pose_landmarks) 
	{
		std::vector<NormalizedLandmark> new_list;
		for (const auto& lm : list.landmarks)
			new_list.push_back(NormalizedLandmark{ lm.x, lm.y, lm.z, lm.visibility, lm.presence, lm.name });
		pr.pose_landmarks.push_back(new_list);
	}

	for (const auto list : result.pose_world_landmarks)
	{
		std::vector<Landmark> new_list;
		for (const auto lm : list.landmarks)
			new_list.push_back(Landmark{ lm.x, lm.y, lm.z, lm.visibility, lm.presence, lm.name });
		pr.pose_world_landmarks.push_back(new_list);
	}

	return pr;
}

std::unique_ptr<PoseLandmarker> PoseLandmarker::Create(PoseLandmarkerOptions& options)
{
	auto opt = std::make_unique<vision::PoseLandmarkerOptions>();

	opt->base_options.model_asset_path = options.base_options.model_asset_path;
	opt->running_mode = (core::RunningMode)options.running_mode;
	opt->num_poses = options.num_poses;
	opt->min_pose_detection_confidence = options.min_pose_detection_confidence;
	opt->min_pose_presence_confidence = options.min_pose_presence_confidence;
	opt->min_tracking_confidence = options.min_tracking_confidence;
	opt->output_segmentation_masks = options.output_segmentation_masks;

	auto res = vision::PoseLandmarker::Create(std::move(opt));

	if (!res.ok())
	{
		printf("could not create PoseLandmarker: [%s]\n", res.status().ToString().c_str());
		return nullptr;
	}

	auto pl = std::make_unique<PoseLandmarker>();
	pl->internal_pose_landmarker = (void*)res.value().release();

	return pl;
}

PoseLandmarkerResult TasksApi::PoseLandmarker::Detect(cv::Mat* image, std::optional<ImageProcessingOptions> image_processing_options)
{
	std::unique_ptr<vision::PoseLandmarker> pl(static_cast<vision::PoseLandmarker*>(internal_pose_landmarker));
	core::ImageProcessingOptions opt;

	auto res = pl->Detect(MakeImage(image));

	if (!res.ok())
		return PoseLandmarkerResult();
	else
		return MakeResult(res.value());
}

PoseLandmarkerResult PoseLandmarker::DetectForVideo(cv::Mat* image, int64 timestamp_ms, std::optional<ImageProcessingOptions> image_processing_options)
{
	std::unique_ptr<vision::PoseLandmarker> visionPl(static_cast<vision::PoseLandmarker*>(internal_pose_landmarker));
	auto res = visionPl->DetectForVideo(MakeImage(image), timestamp_ms);

	if (!res.ok())
	{
		printf("could not DetectForVideo: [%s]\n", res.status().ToString().c_str());
		
		visionPl.release();
		return PoseLandmarkerResult();
	}

	auto result = MakeResult(res.value());
	visionPl.release();

	return result;
}

bool PoseLandmarker::DetectAsync(cv::Mat* image, int64 timestamp_ms, std::optional<ImageProcessingOptions> image_processing_options)
{
	return false;
}

bool PoseLandmarker::Close()
{
	std::unique_ptr<vision::PoseLandmarker> visionPl(static_cast<vision::PoseLandmarker*>(internal_pose_landmarker));
	auto res = visionPl->Close();
	return res.ok();
}
