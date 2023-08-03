#include "tasksapi.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.h"

using namespace TasksApi;
namespace hand = mediapipe::tasks::vision::hand_landmarker;
namespace core = mediapipe::tasks::vision::core;
mediapipe::Image MakeImage(cv::Mat* image);

HandLandmarkerResult MakeHandResult(hand::HandLandmarkerResult& result)
{
	HandLandmarkerResult hr;

	for (const auto& cs : result.handedness)
	{
		Classifications classifications;
		classifications.head_index = cs.head_index;
		classifications.head_name = cs.head_name;
		for(const auto& c : cs.categories)
		{
			classifications.categories.push_back(Category{ c.index, c.score, c.category_name, c.display_name });
		}
		hr.handedness.push_back(classifications);
	}

	for (const auto& list : result.hand_landmarks)
	{
		std::vector<NormalizedLandmark> new_list;
		for (const auto& lm : list.landmarks)
			new_list.push_back(NormalizedLandmark{ lm.x, lm.y, lm.z, lm.visibility, lm.presence, lm.name });
		
		hr.hand_landmarks.push_back(new_list);
	}

	for (const auto list : result.hand_world_landmarks)
	{
		std::vector<Landmark> new_list;
		for (const auto lm : list.landmarks)
			new_list.push_back(Landmark{ lm.x, lm.y, lm.z, lm.visibility, lm.presence, lm.name });
		
		hr.hand_world_landmarks.push_back(new_list);
	}

	return hr;
}

std::unique_ptr<HandLandmarker> HandLandmarker::Create(HandLandmarkerOptions& options)
{
	auto opt = std::make_unique<hand::HandLandmarkerOptions>();

	opt->base_options.model_asset_path = options.base_options.model_asset_path;
	opt->running_mode = (core::RunningMode)options.running_mode;
	opt->num_hands = options.num_hands;
	opt->min_hand_detection_confidence = options.min_hand_detection_confidence;
	opt->min_hand_presence_confidence = options.min_hand_presence_confidence;
	opt->min_tracking_confidence = options.min_tracking_confidence;

	auto res = hand::HandLandmarker::Create(std::move(opt));

	if (!res.ok())
	{
		printf("could not create HandLandmarker: [%s]\n", res.status().ToString().c_str());
		return nullptr;
	}

	auto pl = std::make_unique<HandLandmarker>();
	pl->internal_hand_landmarker = (void*)res.value().release();

	return pl;
}

HandLandmarkerResult TasksApi::HandLandmarker::Detect(cv::Mat* image, std::optional<ImageProcessingOptions> image_processing_options)
{
	std::unique_ptr<hand::HandLandmarker> hl(static_cast<hand::HandLandmarker*>(internal_hand_landmarker));
	core::ImageProcessingOptions opt;

	auto res = hl->Detect(MakeImage(image));

	if (!res.ok())
		return HandLandmarkerResult();
	else
		return MakeHandResult(res.value());
}

HandLandmarkerResult HandLandmarker::DetectForVideo(cv::Mat* image, int64 timestamp_ms, std::optional<ImageProcessingOptions> image_processing_options)
{
	std::unique_ptr<hand::HandLandmarker> visionHl(static_cast<hand::HandLandmarker*>(internal_hand_landmarker));
	auto res = visionHl->DetectForVideo(MakeImage(image), timestamp_ms);
	if (!res.ok())
	{
		printf("could not DetectForVideo: [%s]\n", res.status().ToString().c_str());

		visionHl.release();
		return HandLandmarkerResult();
	}

	auto result = MakeHandResult(res.value());
	visionHl.release();

	return result;
}

bool HandLandmarker::DetectAsync(cv::Mat* image, int64 timestamp_ms, std::optional<ImageProcessingOptions> image_processing_options)
{
	return false;
}

bool HandLandmarker::Close()
{
	std::unique_ptr<hand::HandLandmarker> visionHl(static_cast<hand::HandLandmarker*>(internal_hand_landmarker));
	auto res = visionHl->Close();
	return res.ok();
}