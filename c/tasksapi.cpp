#include "tasksapi.h"
#include "mediapipe/framework/formats/image.h"

mediapipe::Image MakeImage(cv::Mat* image)
{
	auto input_frame = std::make_shared<mediapipe::ImageFrame>();
	auto mp_format = static_cast<mediapipe::ImageFormat::Format>(mediapipe::ImageFormat::Format::ImageFormat_Format_SRGB);
	uint32_t mp_alignment_boundary = mediapipe::ImageFrame::kGlDefaultAlignmentBoundary;
	input_frame->CopyPixelData(mp_format, image->cols, image->rows, image->data, mp_alignment_boundary);

	mediapipe::Image mpimage(input_frame);
	return mpimage;
}
