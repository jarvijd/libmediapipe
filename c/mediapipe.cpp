#include "mediapipe.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/tool/options_util.h"

#ifndef MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h" 
#include "mediapipe/gpu/gpu_buffer.h" 
#include "mediapipe/gpu/gpu_shared_data_internal.h" 
#endif

#include "mediapipe/calculators/util/thresholding_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"

#ifdef __ANDROID__
#   include "mediapipe/util/android/asset_manager_util.h"
#   include "mediapipe/framework/port/singleton.h"
#endif

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "google/protobuf/util/json_util.h"

#include <string>
#include <cstring>
#include <variant>
#include <cassert>
#include <fstream>
#include <iostream>

#ifndef __ANDROID__
ABSL_DECLARE_FLAG(std::string, resource_root_dir);
#endif

thread_local absl::Status last_error;

#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker.h""

using namespace mediapipe::tasks::vision::pose_landmarker;
int jj_test()
{
    auto options = std::make_unique<PoseLandmarkerOptions>();

    auto pose_landmarker_options = std::make_unique<PoseLandmarkerOptions>();
    auto pl = PoseLandmarker::Create(std::move(pose_landmarker_options));
	return 0;
}
struct mp_node_option {
    const char* node;
    const char* option;
    std::variant<float, double, int> value;
};

struct mp_instance_builder {
    const char* graph_filename;
    const char* input_stream;
    std::vector<mp_node_option> options;
    std::map<std::string, mediapipe::Packet> side_packets;
};

struct mp_instance {
    mediapipe::CalculatorGraph graph;
    std::string input_stream;
    size_t frame_timestamp;

#ifndef MEDIAPIPE_DISABLE_GPU
    bool isGpu = false;
    mediapipe::GlCalculatorHelper gpu_helper;
#endif
};

struct mp_poller {
    mediapipe::OutputStreamPoller poller;
};

struct mp_packet {
    mediapipe::Packet packet;
};


template<typename List, typename Landmark>
static mp_landmark_list* get_landmarks(mp_packet* packet) {
    const List& mp_list = packet->packet.template Get<List>();
    auto* list = new mp_landmark[mp_list.landmark_size()];

    for (int j = 0; j < mp_list.landmark_size(); j++) {
        const Landmark& mp_landmark = mp_list.landmark(j);
        list[j].x = mp_landmark.x();
        list[j].y = mp_landmark.y();
        list[j].z = mp_landmark.z();
        list[j].visbility = mp_landmark.has_visibility() ? mp_landmark.visibility() : 0.0f;
        list[j].presence = mp_landmark.has_presence() ? mp_landmark.presence() : 0.0f; 

        //list[j] = {
        //    mp_landmark.x(),
        //    mp_landmark.y(),
        //    mp_landmark.z(),
        //    mp_landmark.has_visibility() ? mp_landmark.visibility() : 0.0f,
        //    mp_landmark.has_presence() ? mp_landmark.presence() : 0.0f
        //};
    }

    return new mp_landmark_list{
        list,
        (int)mp_list.landmark_size()
    };
}

template<typename List, typename Landmark>
static mp_multi_face_landmark_list* get_multi_face_landmarks(mp_packet* packet) {
    const auto& mp_data = packet->packet.template Get<std::vector<List>>();

    auto* lists = new mp_landmark_list[mp_data.size()];

    for (int i = 0; i < mp_data.size(); i++) {
        const List& mp_list = mp_data[i];
        auto* list = new mp_landmark[mp_list.landmark_size()];

        for (int j = 0; j < mp_list.landmark_size(); j++) {
            const Landmark& mp_landmark = mp_list.landmark(j);
            list[j] = {
                mp_landmark.x(),
                mp_landmark.y(),
                mp_landmark.z()
            };
        }

        lists[i] = mp_landmark_list{
            list,
            (int)mp_list.landmark_size()
        };
    }

    return new mp_multi_face_landmark_list{
        lists,
        (int)mp_data.size()
    };
}

template<typename Rect>
static mp_rect_list* get_rects(mp_packet* packet) {
    const auto& mp_data = packet->packet.template Get<std::vector<Rect>>();
    auto* list = new mp_rect[mp_data.size()];

    for (int i = 0; i < mp_data.size(); i++) {
        const Rect& mp_rect = mp_data[i];
        list[i] = {
            (float)mp_rect.x_center(),
            (float)mp_rect.y_center(),
            (float)mp_rect.width(),
            (float)mp_rect.height(),
            mp_rect.rotation(),
            mp_rect.rect_id()
        };
    }

    return new mp_rect_list{
        list,
        (int)mp_data.size()
    };
}

template<typename List, typename Handedness>
static mp_multi_face_handedness_list* get_handedness(mp_packet* packet) {
    const auto& mp_data = packet->packet.template Get<std::vector<List>>();

    auto* lists = new mp_handedness_list[mp_data.size()];

    for (int i = 0; i < mp_data.size(); i++) {
        const List& mp_list = mp_data[i];
        auto* list = new mp_handedness[mp_list.classification_size()];

        for (int j = 0; j < mp_list.classification_size(); j++) {
            const Handedness& handedness = mp_list.classification(j);
            list[j] = {
                handedness.label(),
                handedness.score()
            };
        }

        lists[i] = mp_handedness_list{
            list,
            (int)mp_list.classification_size()
        };
    }

    return new mp_multi_face_handedness_list{
        lists,
        (int)mp_data.size()
    };

}

extern "C" {

    MEDIAPIPE_API mp_instance_builder* mp_create_instance_builder(const char* graph_filename, const char* input_stream) {
        return new mp_instance_builder{ graph_filename, input_stream, {} };
    }

    MEDIAPIPE_API void mp_add_option_float(mp_instance_builder* instance_builder, const char* node, const char* option, float value) {
        instance_builder->options.push_back({ node, option, value });
    }

    MEDIAPIPE_API void mp_add_option_double(mp_instance_builder* instance_builder, const char* node, const char* option, double value) {
        instance_builder->options.push_back({ node, option, value });
    }

    MEDIAPIPE_API void mp_add_option_int(mp_instance_builder* instance_builder, const char* node, const char* option, int value) {
        instance_builder->options.push_back({ node, option, value }); 
    }

    MEDIAPIPE_API void mp_add_side_packet(mp_instance_builder* instance_builder, const char* name, mp_packet* packet) {
        instance_builder->side_packets.insert({ name, packet->packet });
        mp_destroy_packet(packet);
    }

    MEDIAPIPE_API mp_instance* mp_create_instance(mp_instance_builder* builder, bool isGpu) {
        mediapipe::CalculatorGraphConfig config;

        std::ifstream stream(builder->graph_filename, std::ios::binary | std::ios::ate);
        if (!stream) {
            last_error = absl::Status(absl::StatusCode::kNotFound, "Failed to open graph file");
            printf("mp_create_instance: error 1\n");
            return nullptr;
        }

        size_t size = stream.tellg();
        stream.seekg(0, std::ios::beg);

        char* memory = new char[size];
        stream.read(memory, size);
        config.ParseFromArray(memory, size);
        delete[] memory;

        mediapipe::ValidatedGraphConfig validated_config;
        validated_config.Initialize(config);
        mediapipe::CalculatorGraphConfig canonical_config = validated_config.Config();

        for (const mp_node_option& option : builder->options) {
            for (auto& node : *canonical_config.mutable_node()) { 
                if (node.name() != option.node) {
                    continue;
                }

                google::protobuf::Message* ext;

                if (node.calculator() == "ThresholdingCalculator")
                    ext = node.mutable_options()->MutableExtension(mediapipe::ThresholdingCalculatorOptions::ext);
                else if (node.calculator() == "TensorsToDetectionsCalculator")
                    ext = node.mutable_options()->MutableExtension(mediapipe::TensorsToDetectionsCalculatorOptions::ext);
                else {
                    // assert(!"Unknown node calculator");
                    // printf("mp_create_instance: error 2 -- %s\n", node.calculator().c_str());
                    // last_error = absl::Status(absl::StatusCode::kNotFound, "Unknown node calculator--" + node.calculator());
                    continue;
                }

                auto* descriptor = ext->GetDescriptor();
                auto* reflection = ext->GetReflection();
                auto* field_descriptor = descriptor->FindFieldByName(option.option);

                switch (option.value.index()) {
                case 0: reflection->SetFloat(ext, field_descriptor, std::get<0>(option.value)); break;
                case 1: reflection->SetDouble(ext, field_descriptor, std::get<1>(option.value)); break;
                case 2: reflection->SetInt32(ext, field_descriptor, std::get<2>(option.value)); break;
                }
            }
        }

        /* For printing the graph
        google::protobuf::util::JsonPrintOptions json_options;
        json_options.add_whitespace = true;

        std::string str;
        google::protobuf::util::MessageToJsonString(canonical_config, &str, json_options);
        std::cout << str << std::endl;
        */


        auto* instance = new mp_instance;
        absl::Status result = instance->graph.Initialize(canonical_config, builder->side_packets);
        if (!result.ok()) {
            last_error = result;
            printf("mp_create_instance: error 3\n");
            return nullptr;
        }


#ifndef MEDIAPIPE_DISABLE_GPU
        instance->isGpu = isGpu;
        if (instance->isGpu)
        {
            auto gpu_resources = mediapipe::GpuResources::Create();
            if (!gpu_resources.ok())
            {
                last_error = gpu_resources.status();
                printf("mp_create_instance: error 4\n");
                return nullptr;
            }

            auto status = instance->graph.SetGpuResources(std::move(gpu_resources.value()));
            if (!status.ok())
            {
                last_error = status;
                printf("mp_create_instance: error 5\n");
                return nullptr;
            }
            instance->gpu_helper.InitializeForTest(instance->graph.GetGpuResources().get());
        }
#endif

        instance->input_stream = builder->input_stream;
        instance->frame_timestamp = 0;

        delete builder;
        printf("mp_create_instance: DONE\n");
        return instance;
    }

    MEDIAPIPE_API mp_poller* mp_create_poller(mp_instance* instance, const char* output_stream, bool observe_timestamp_bounds) {
        absl::StatusOr<mediapipe::OutputStreamPoller> result = instance->graph.AddOutputStreamPoller(output_stream, observe_timestamp_bounds);
        if (!result.ok()) {
            last_error = result.status();
            return nullptr;
        }

        return new mp_poller{
            std::move(*result)
        };
    }

    MEDIAPIPE_API bool mp_start(mp_instance* instance) {
        absl::Status result = instance->graph.StartRun({});

        if (!result.ok()) {
            last_error = result;
            return false;
        }

        return true;
    }

    MEDIAPIPE_API bool mp_process(mp_instance* instance, mp_packet* packet)
    {
        //mediapipe::Timestamp mp_timestamp(instance->frame_timestamp++);
        instance->frame_timestamp += 33333;
        mediapipe::Timestamp mp_timestamp(instance->frame_timestamp);
        mediapipe::Packet mp_packet = packet->packet.At(mp_timestamp);

        auto result = instance->graph.AddPacketToInputStream(instance->input_stream, mp_packet);

        mp_destroy_packet(packet);

        if (!result.ok()) {
            last_error = result;
            return false;
        }

        return true;
    }

    MEDIAPIPE_API bool mp_process_gpu(mp_instance* instance, mp_image image, size_t timestamp)
    {
#ifndef MEDIAPIPE_DISABLE_GPU
        auto input_frame = std::make_unique<mediapipe::ImageFrame>();
        auto mp_format = static_cast<mediapipe::ImageFormat::Format>(image.format);
        uint32_t mp_alignment_boundary = mediapipe::ImageFrame::kGlDefaultAlignmentBoundary;
        input_frame->CopyPixelData(mp_format, image.width, image.height, image.data, mp_alignment_boundary);

        mediapipe::Timestamp mp_timestamp(timestamp);

        auto graph = &instance->graph;
        auto gpu_helper = &instance->gpu_helper;
        auto stream = instance->input_stream;

        auto result = instance->gpu_helper.RunInGlContext([&input_frame, mp_timestamp, graph, gpu_helper, stream]() -> absl::Status
            {
                // Convert ImageFrame to GpuBuffer.
                mediapipe::GlTexture texture = gpu_helper->CreateSourceTexture(*input_frame.get());
                auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
                glFlush();
                texture.Release();

                // Send GPU image packet into the graph.
                auto packet = mediapipe::Adopt(gpu_frame.release()).At(mp_timestamp);
                auto result = graph->AddPacketToInputStream(stream, packet);
                return result;
            }
        );

        if (!result.ok())
        {
            last_error = result;
            return false;
        }

        return true;
#else
        return false;
#endif
    }

    MEDIAPIPE_API bool mp_wait_until_idle(mp_instance* instance) {
        absl::Status result = instance->graph.WaitUntilIdle();

        if (!result.ok()) {
            last_error = result;
            return false;
        }

        return true;
    }

    MEDIAPIPE_API int mp_get_queue_size(mp_poller* poller) {
        return poller->poller.QueueSize();
    }

    MEDIAPIPE_API void mp_destroy_poller(mp_poller* poller) {
        delete poller;
    }

    MEDIAPIPE_API bool mp_destroy_instance(mp_instance* instance) {
        absl::Status result = instance->graph.CloseInputStream(instance->input_stream);
        if (!result.ok()) {
            last_error = result;
            return false;
        }

        result = instance->graph.WaitUntilDone();
        if (!result.ok()) {
            last_error = result;
            return false;
        }

        delete instance;
        return true;
    }

    MEDIAPIPE_API void mp_set_resource_dir(const char* dir) {
#ifndef __ANDROID__
        absl::SetFlag(&FLAGS_resource_root_dir, dir);
#endif
    }

#ifdef __ANDROID__
    MEDIAPIPE_API void mp_init_asset_manager(JNIEnv* env, jobject android_context, jstring cache_dir_path) {
        mediapipe::AssetManager* asset_manager = Singleton<mediapipe::AssetManager>::get();
        const char* c_cache_dir_path = env->GetStringUTFChars(cache_dir_path, nullptr);
        asset_manager->InitializeFromActivity(env, android_context, c_cache_dir_path);
        env->ReleaseStringUTFChars(cache_dir_path, c_cache_dir_path);
    }
#endif

    MEDIAPIPE_API mp_packet* mp_create_packet_int(int value) {
        return new mp_packet{
            mediapipe::MakePacket<int>(value)
        };
    }

    MEDIAPIPE_API mp_packet* mp_create_packet_float(float value) {
        return new mp_packet{
            mediapipe::MakePacket<float>(value)
        };
    }

    MEDIAPIPE_API mp_packet* mp_create_packet_bool(bool value) {
        return new mp_packet{
            mediapipe::MakePacket<bool>(value)
        };
    }

    MEDIAPIPE_API mp_packet* mp_create_packet_image(mp_image image) {
        auto mp_frame = std::make_unique<mediapipe::ImageFrame>();

        auto mp_format = static_cast<mediapipe::ImageFormat::Format>(image.format);

        uint32_t mp_alignment_boundary = mediapipe::ImageFrame::kDefaultAlignmentBoundary;
        mp_frame->CopyPixelData(mp_format, image.width, image.height, image.data, mp_alignment_boundary);

        return new mp_packet{
            mediapipe::Adopt(mp_frame.release())
        };
    }

    MEDIAPIPE_API std::unique_ptr<mediapipe::ImageFrame> mp_create_image_frame(mp_image image) {
        auto mp_frame = std::make_unique<mediapipe::ImageFrame>();

        auto mp_format = static_cast<mediapipe::ImageFormat::Format>(image.format);

        uint32_t mp_alignment_boundary = mediapipe::ImageFrame::kGlDefaultAlignmentBoundary;
        mp_frame->CopyPixelData(mp_format, image.width, image.height, image.data, mp_alignment_boundary);

        return mp_frame;
    }

    MEDIAPIPE_API mp_packet* mp_poll_packet(mp_poller* poller) {
        auto* packet = new mp_packet;
        poller->poller.Next(&packet->packet);
        return packet;
    }

    MEDIAPIPE_API void mp_destroy_packet(mp_packet* packet) {
        delete packet;
    }

    MEDIAPIPE_API const char* mp_get_packet_type(mp_packet* packet) {
        mediapipe::TypeId type = packet->packet.GetTypeId();
        std::string string = type.name();
        char* buffer = new char[string.size() + 1];
        std::strcpy(buffer, string.c_str());
        return buffer;
    }

    MEDIAPIPE_API void mp_free_packet_type(const char* type) {
        delete[] type;
    }

    MEDIAPIPE_API void mp_copy_packet_image(mp_packet* packet, uint8_t* out_data) {
        const auto& mp_frame = packet->packet.Get<mediapipe::ImageFrame>();
        size_t data_size = mp_frame.PixelDataSizeStoredContiguously();
        mp_frame.CopyToBuffer(out_data, data_size);
    }

    MEDIAPIPE_API mp_landmark_list* mp_get_landmarks(mp_packet* packet) {
        return get_landmarks<mediapipe::NormalizedLandmarkList, mediapipe::NormalizedLandmark>(packet);
    }

    MEDIAPIPE_API void mp_destroy_landmarks(mp_landmark_list* landmarks) {
        if (landmarks == nullptr)
            return;

        delete landmarks;
    }


    MEDIAPIPE_API mp_multi_face_landmark_list* mp_get_multi_face_landmarks(mp_packet* packet) {
        return get_multi_face_landmarks<mediapipe::LandmarkList, mediapipe::Landmark>(packet);
    }

    MEDIAPIPE_API mp_multi_face_landmark_list* mp_get_norm_multi_face_landmarks(mp_packet* packet) {
        return get_multi_face_landmarks<mediapipe::NormalizedLandmarkList, mediapipe::NormalizedLandmark>(packet);
    }

    MEDIAPIPE_API void mp_destroy_multi_face_landmarks(mp_multi_face_landmark_list* multi_face_landmarks) {
        if (multi_face_landmarks == nullptr)
            return;

        for (int i = 0; i < multi_face_landmarks->length; i++) {
            delete[] multi_face_landmarks->elements[i].elements;
        }

        delete[] multi_face_landmarks->elements;
        delete multi_face_landmarks;
    }

    MEDIAPIPE_API mp_rect_list* mp_get_rects(mp_packet* packet) {
        return get_rects<mediapipe::Rect>(packet);
    }

    MEDIAPIPE_API mp_rect_list* mp_get_norm_rects(mp_packet* packet) {
        return get_rects<mediapipe::NormalizedRect>(packet);
    }

    MEDIAPIPE_API void mp_destroy_rects(mp_rect_list* list) {
        delete[] list->elements;
        delete list;
    }


    MEDIAPIPE_API mp_multi_face_handedness_list* mp_get_handedness(mp_packet* packet)
    {
        return get_handedness<mediapipe::ClassificationList, mediapipe::Classification>(packet);
    }

    MEDIAPIPE_API void mp_destroy_handedness(mp_multi_face_handedness_list* multi_face_handedness) {
        if (multi_face_handedness == nullptr)
            return;

        for (int i = 0; i < multi_face_handedness->length; i++) {
            delete[] multi_face_handedness->elements[i].elements;
        }

        delete[] multi_face_handedness->elements;
        delete multi_face_handedness;
    }

    MEDIAPIPE_API const char* mp_get_last_error() {
        std::string string = last_error.ToString();
        char* buffer = new char[string.size() + 1];
        std::strcpy(buffer, string.c_str());
        return buffer;
    }

    MEDIAPIPE_API void mp_free_error(const char* message) {
        delete[] message;
    }

}

