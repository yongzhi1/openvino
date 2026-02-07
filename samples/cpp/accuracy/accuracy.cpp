//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cassert>
#ifdef GNU_LESS_9_1
#include <experimental/filesystem>
#else
#include <filesystem>
#endif
#include <string>
#include <unordered_set>

#include "gflags/gflags.h"

#include "openvino/openvino.hpp"

#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "openvino/core/shape.hpp"

#define MSE_THRESHOLD 0.01

static const char help_message[] = "Print a usage message.";
static const char model_message[] = "Required. Path to an IR .xml file.";
static const char target_device_message[] = "target device to compare against reference device (CPU)";
static const char folder_message[] = "folder to dump model input/output";
static const char input_message[] = "Use random inputs by defaults";


DEFINE_bool(h, false, help_message);
DEFINE_string(fixed, "", input_message);
DEFINE_string(m, "", model_message);
DEFINE_string(d, "NPU", target_device_message);
DEFINE_string(f, "random1", folder_message);

static void showUsage() {
    std::cout << std::endl;
    std::cout << "query_model [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -m \"<path>\"           " << model_message << std::endl;
    std::cout << "    -d \"<device>\"         " << target_device_message << std::endl;
    std::cout << "    -f \"<folder>\"         " << folder_message << std::endl;
    std::cout << "    -fixed \"<input prefix>\"         " << input_message << std::endl;
}

void ParseAndCheckCommandLine(int argc, char* argv[]) {
    const bool empty_args = (argc == 1);

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);

    if (FLAGS_h || empty_args) {
        showUsage();
        exit(1);
    }

    if (FLAGS_d.empty()) {
        throw std::logic_error("Parameter -d is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_f.empty()) {
        FLAGS_f = "random1";
    }
}

void generate_random_int64_data(int64_t* data, size_t size, size_t byte_size, std::string output_name) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dis(1, 10);

    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    std::ofstream outFile(output_name, std::ios::binary);

    const auto out_data_ptr = (char*)(data);
    outFile.write(out_data_ptr, byte_size);
}

void generate_random_int32_data(int32_t* data, size_t size, size_t byte_size, std::string output_name) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, 1000);

    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    std::ofstream outFile(output_name, std::ios::binary);

    const auto out_data_ptr = (char*)(data);
    outFile.write(out_data_ptr, byte_size);
}

// 转置最里面的两个维度
template <typename T>
std::vector<T> transpose_inner_two_dims(const std::vector<T>& data, const std::vector<size_t>& shape) {
    if (shape.size() < 2) {
        throw std::runtime_error("Shape must have at least two dimensions.");
    }

    size_t outer_size = 1;
    for (size_t i = 0; i < shape.size() - 2; ++i) {
        outer_size *= shape[i];
    }

    size_t dim1 = shape[shape.size() - 2];
    size_t dim2 = shape[shape.size() - 1];
    std::vector<T> transposed_data(data.size());

    for (size_t i = 0; i < outer_size; ++i) {
        for (size_t j = 0; j < dim1; ++j) {
            for (size_t k = 0; k < dim2; ++k) {
                transposed_data[i * dim1 * dim2 + k * dim1 + j] = data[i * dim1 * dim2 + j * dim2 + k];
            }
        }
    }

    return transposed_data;
}

void generate_random_fp32_data(float* data, size_t size, size_t byte_size, std::string output_name, ov::Shape shape,
                               bool transpose) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

    std::vector<float> orig_data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
        orig_data[i] = data[i];
    }

    std::ofstream outFile(output_name, std::ios::binary);
    if (transpose) {
        std::vector<float> transposed_data = transpose_inner_two_dims(orig_data, shape);

        const auto out_data_ptr = (char*)(transposed_data.data());
        outFile.write(out_data_ptr, byte_size);
        printf("%s is transposed\n", output_name.c_str());
        return;
    }

    const auto out_data_ptr = (char*)(data);
    outFile.write(out_data_ptr, byte_size);
}

// 生成随机fp16数据
void generate_random_fp16_data(ov::float16* data, size_t size, size_t byte_size, std::string output_name, ov::Shape shape,
                               bool transpose) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

    std::vector<ov::float16> orig_data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = ov::float16(dis(gen));
        orig_data[i] = data[i];
    }
    std::ofstream outFile(output_name, std::ios::binary);

    if (transpose) {
        std::vector<ov::float16> transposed_data = transpose_inner_two_dims(orig_data, shape);

        const auto out_data_ptr = (char*)(transposed_data.data());
        outFile.write(out_data_ptr, byte_size);
        printf("%s is transposed\n", output_name.c_str());
        return;
    }
    const auto out_data_ptr = (char*)(data);
    outFile.write(out_data_ptr, byte_size);
}

void generate_random_int8_data(int8_t* data, size_t size, size_t byte_size, std::string output_name, ov::Shape shape,
                               bool transpose) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int8_t> dis(-1, 1);

    std::vector<int8_t> orig_data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
        orig_data[i] = data[i];
    }
    std::ofstream outFile(output_name, std::ios::binary);

    if (transpose) {
        std::vector<int8_t> transposed_data = transpose_inner_two_dims(orig_data, shape);

        const auto out_data_ptr = (char*)(transposed_data.data());
        outFile.write(out_data_ptr, byte_size);
        printf("%s is transposed\n", output_name.c_str());
        return;
    }

    const auto out_data_ptr = reinterpret_cast<char*>(data);
    outFile.write(out_data_ptr, byte_size);
}

// 生成随机int4数据
void generate_random_int4_data(int8_t* data, size_t size, int8_t zp, size_t byte_size, std::string output_name, ov::Shape shape,
                               bool transpose) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(-1, 1);  // int4范围是-8到7
    std::vector<int8_t> idata((size + 1) / 2);
    for (size_t i = 0; i < size; ++i) {
        int8_t ivalue = dis(gen);
        int8_t uvalue = ivalue + zp;
        if (i % 2 == 0) {
            data[i / 2] = (uvalue & 0x0F);   // 存储低4位
            idata[i / 2] = (ivalue & 0x0F);  // 存储低4位
        } else {
            data[i / 2] |= (uvalue << 4);   // 存储高4位
            idata[i / 2] |= (ivalue << 4);  // 存储高4位
        }
    }

    std::ofstream outFile(output_name, std::ios::binary);

    if (transpose && shape.size() == 2 && (shape[1] % 2 == 0) && (shape[0] % 2 == 0)) {
        std::vector<int8_t> decoded_data(size);
        for (size_t i = 0; i < size; ++i) {
            if (i % 2 == 0) {
                decoded_data[i] = idata[i / 2] & 0x0F;
            } else {
                decoded_data[i] = (idata[i / 2] >> 4) & 0x0F;
            }
        }

        std::vector<int8_t> transposed_data(size);
        size_t dim0 = shape[0];
        size_t dim1 = shape[1];

        for (size_t i = 0; i < dim0; ++i) {
            for (size_t j = 0; j < dim1; ++j) {
                size_t src_index = i * dim1 + j;
                size_t dst_index = j * dim0 + i;
                transposed_data[dst_index] = decoded_data[src_index];
            }
        }

        std::vector<int8_t> encoded_data((size + 1) / 2);
        for (size_t i = 0; i < size; ++i) {
            int8_t value = transposed_data[i];
            if (i % 2 == 0) {
                encoded_data[i / 2] = (value & 0x0F);
            } else {
                encoded_data[i / 2] |= (value << 4);
            }
        }

        const auto out_data_ptr = (char*)(encoded_data.data());
        outFile.write(out_data_ptr, byte_size);
        printf("%s is transposed\n", output_name.c_str());
        return;
    }

    const auto out_data_ptr = (char*)(idata.data());
    outFile.write(out_data_ptr, byte_size);
}

std::string sanitize_name(std::string name) {
    for (char& c : name) {
        if (c == '/' || c == ':' || c == '\\') {
            c = '_';
        }
    }
    return name;
}

void load_random_inputs(ov::InferRequest& request, ov::CompiledModel model, std::string dir) {
    auto inputs = model.inputs();
    auto num_of_inputs = inputs.size();
    for (auto i = 0; i < num_of_inputs; i++) {
        bool transpose = false;
        const auto& input = inputs[i];
        auto input_name = input.get_any_name();
        input_name = sanitize_name(input_name);
        std::vector<std::string> input_names_to_be_transposed = {};
        if (std::find(input_names_to_be_transposed.begin(), input_names_to_be_transposed.end(), input_name) !=
            input_names_to_be_transposed.end()) {
            std::cout << "Transpose input with name: " << input_name << std::endl;
            transpose = true;
        }

        auto input_tensor = request.get_input_tensor(i);
        auto element_type = input_tensor.get_element_type();
        auto byte_size = input_tensor.get_byte_size();
        auto shape = input_tensor.get_shape();
        size_t num_elements = 1;
        for (auto dim : shape) {
            num_elements *= dim;
        }

	std::string file_name = dir + "/" + "input-" + std::to_string(i) + "-" + input_name + ".bin";
	std::cout << "Generating random input for " << input_name << ", writing to " << file_name << std::endl;
        if (element_type == ov::element::f32) {
            std::vector<float> random_data(num_elements);
            generate_random_fp32_data(random_data.data(), num_elements, byte_size, file_name, shape, transpose);
            memcpy(input_tensor.data(), random_data.data(), byte_size);
            printf("Input-%d is FP32.\n", i);
        } else if (element_type == ov::element::f16) {
            std::vector<ov::float16> random_data(num_elements);
            generate_random_fp16_data(random_data.data(), num_elements, byte_size, file_name, shape, transpose);
            memcpy(input_tensor.data(), random_data.data(), byte_size);
            printf("Input-%d is FP16.\n", i);
        } else if (element_type == ov::element::u4) {
            std::vector<int8_t> random_data((num_elements + 1) / 2);  // 每个uint8_t存储两个uint4
            generate_random_int4_data(random_data.data(), num_elements, 8, byte_size, file_name, shape, transpose);
            memcpy(input_tensor.data(), random_data.data(), byte_size);
            printf("Input-%d is u4.\n", i);
        } else if (element_type == ov::element::i4) {
            std::vector<int8_t> random_data((num_elements + 1) / 2);  // 每个uint8_t存储两个uint4
            generate_random_int4_data(random_data.data(), num_elements, 0, byte_size, file_name, shape, transpose);
            memcpy(input_tensor.data(), random_data.data(), byte_size);
            printf("Input-%d is i4.\n", i);
        } else if (element_type == ov::element::i32) {
            std::vector<int32_t> random_data(num_elements);
            generate_random_int32_data(random_data.data(), num_elements, byte_size, file_name);
            memcpy(input_tensor.data(), random_data.data(), byte_size);
            printf("Input-%d is i32.\n", i);
        } else if (element_type == ov::element::i64) {
            std::vector<int64_t> random_data(num_elements);
            generate_random_int64_data(random_data.data(), num_elements, byte_size, file_name);
            memcpy(input_tensor.data(), random_data.data(), byte_size);
            printf("Input-%d is i64.\n", i);
        } else if (element_type == ov::element::i8) {
            std::vector<int8_t> random_data(num_elements);
            generate_random_int8_data(random_data.data(), num_elements, byte_size, file_name, shape, transpose);
            memcpy(input_tensor.data(), random_data.data(), byte_size);
            printf("Input-%d is i8.\n", i);
        } else {
            throw std::runtime_error("Unsupported input tensor element type");
        }
    }
}

void load_inputs(ov::InferRequest& request, ov::CompiledModel model, std::string dir) {
    auto inputs = model.inputs();
    auto num_of_inputs = inputs.size();
    for (auto i = 0; i < num_of_inputs; i++) {
	const auto& input = inputs[i];
        auto input_name = input.get_any_name();
        input_name = sanitize_name(input_name);
	std::string file_name = dir + "/" + "input-" + std::to_string(i) + "-" + input_name + ".bin";

        if (!FLAGS_fixed.empty()) {
           std::ostringstream oss;
           // for example Model16_prefill_00_input_
           oss << dir << "/" << FLAGS_fixed << std::setw(2) << std::setfill('0') << i << ".bin";
           file_name = oss.str();
        }

        std::ifstream file(file_name, std::ios::binary);
        if (!file) {
            throw std::ios_base::failure("File " + file_name + " not found");
        }

        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        auto input_tensor = request.get_input_tensor(i);
        if (input_tensor.get_element_type() == ov::element::i4) {
            printf("data type is i4\n");
        }
        if (fileSize != input_tensor.get_byte_size()) {
            throw std::runtime_error("fileSize != input_tensor_size, " + std::to_string(fileSize) +
                                     " != " + std::to_string(input_tensor.get_byte_size()));
        }

        std::vector<char> buffer(fileSize);
        file.read(buffer.data(), fileSize);

        memcpy(input_tensor.data(), buffer.data(), buffer.size());
    }
}

void log_outputs(ov::InferRequest& infer_request, ov::CompiledModel compiled_model, std::string dir)
{
    for (int i = 0; i < compiled_model.outputs().size(); i++) {
        auto name = compiled_model.output(i).get_any_name();
	//WA somehow ref model doesn't name output 0 logits...
	if (i==0) {
            name = "logits";
	}
        name = sanitize_name(name);
        ov::Tensor output_tensor = infer_request.get_output_tensor(i);
        const auto out_data_ptr = static_cast<char*>(output_tensor.data());
        auto ptr = static_cast<ov::float16*>(output_tensor.data());
        std::string output_name = dir + "/output-" + std::to_string(i) + "-" + name + ".bin";

        std::ofstream outFile(output_name, std::ios::binary);

        if (!outFile) {
            throw std::ios_base::failure("File " + output_name + " not found");
        }

        outFile.write(out_data_ptr, output_tensor.get_byte_size());
        std::cerr << "Write into " << output_name << std::endl;
    }
}

template <typename InT>
void to_f32(const ov::Tensor& in, ov::Tensor& out) {
    if (ov::element::Type_t::f32 == in.get_element_type()) {
        in.copy_to(out);
        return;
    }

    const InT* in_buffer = in.data<InT>();
    const auto out_buffer = out.data<float>();
    for(int64_t index = 0; index < in.get_size(); index++) {
	out_buffer[index] = static_cast<float>(in_buffer[index]);
    }
}

void npuw_util_to_f32(const ov::Tensor& in, ov::Tensor& out) {
    switch (in.get_element_type()) {
    case ov::element::Type_t::f32:
        to_f32<float>(in, out);
        break;
    case ov::element::Type_t::u64:
        to_f32<uint64_t>(in, out);
        break;
    case ov::element::Type_t::i64:
        to_f32<int64_t>(in, out);
        break;
    case ov::element::Type_t::u32:
        to_f32<uint32_t>(in, out);
        break;
    case ov::element::Type_t::i32:
        to_f32<int32_t>(in, out);
        break;
    case ov::element::Type_t::u16:
        to_f32<uint16_t>(in, out);
        break;
    case ov::element::Type_t::i16:
        to_f32<int16_t>(in, out);
        break;
    case ov::element::Type_t::u8:
        to_f32<uint8_t>(in, out);
        break;
    case ov::element::Type_t::i8:
        to_f32<int8_t>(in, out);
        break;
    case ov::element::Type_t::f16:
        to_f32<ov::float16>(in, out);
        break;
    case ov::element::Type_t::bf16:
        to_f32<ov::bfloat16>(in, out);
        break;
    default:
        std::cout << "unsupported " << in.get_element_type().get_type_name() << std::endl;
        break;
    }
}

//openvino/src/plugins/intel_npu/src/plugin/npuw/accuracy/comparator.cpp
bool compare(const ov::Tensor& actual, const ov::Tensor& reference)
{
    if (!actual.is_continuous() || !reference.is_continuous()) {
	std::cout << "actual.is_continuous: " << actual.is_continuous() << ", reference.is_continuous: " << reference.is_continuous() << std::endl;
        return false;
    }
    if (actual.get_shape() != reference.get_shape())
    {
        std::cout << "actual and reference don't match in shape " << actual.get_shape() << " vs " << reference.get_shape() << std::endl;
	return false;
    }
    if (actual.get_byte_size() != reference.get_byte_size())
    {
        std::cout << "actual and reference don't match in size, skip." << actual.get_byte_size() << " vs " << reference.get_byte_size() << std::endl;
        return false;
    }

    ov::Tensor actual_f32;
    ov::Tensor reference_f32;

    if (ov::element::Type_t::f32 == actual.get_element_type()) {
        actual_f32 = actual;
    } else {
        ov::Tensor dst(ov::element::Type_t::f32, actual.get_shape());
        npuw_util_to_f32(actual, dst);
        actual_f32 = std::move(dst);
    }

    if (ov::element::Type_t::f32 == reference.get_element_type()) {
        reference_f32 = reference;
    } else {
        ov::Tensor dst(ov::element::Type_t::f32, reference.get_shape());
        npuw_util_to_f32(reference, dst);
        reference_f32 = dst;
    }

    float* actual_data = actual_f32.data<float>();
    float* reference_data = reference_f32.data<float>();
    const std::size_t size = actual_f32.get_size();

    double squared_error{};
    for (size_t i = 0; i < size; ++i) {
        double diff = (actual_data[i] - reference_data[i]);
        squared_error += (diff * diff);
    }

    if (squared_error <= std::numeric_limits<double>::epsilon()) {
        std::cout << "NRMSE loss: 0.0, threshold: " << MSE_THRESHOLD << "." << " Success: 1" << std::endl;
        return true;
    }

    double rmse = sqrt(squared_error / size);

    auto actual_min_max = std::minmax_element(actual_data, actual_data + size);
    auto reference_min_max = std::minmax_element(reference_data, reference_data + size);
    double den = std::max({0.001f,
                           std::max(0.f, *reference_min_max.second) - std::min(0.f, *reference_min_max.first),
                           std::max(0.f, *actual_min_max.second) - std::min(0.f, *actual_min_max.first)});

    double nrmse = rmse / den;
    bool success = nrmse <= MSE_THRESHOLD;
    std::cout << "NRMSE loss: " << nrmse << ", threshold: " << MSE_THRESHOLD << "." << " Success: " << success << std::endl;

    return success;
}

void compare_outputs(ov::InferRequest& infer_request, ov::CompiledModel compiled_model, std::string ref_dir)
{
    for (int i = 0; i < compiled_model.outputs().size(); i++) {
        auto name = compiled_model.output(i).get_any_name();
        ov::Tensor output_tensor = infer_request.get_output_tensor(i);

        //WA somehow ref model doesn't name output 0 logits...
        if (i==0) {
            name = "logits";
        }
        name = sanitize_name(name);

        std::string ref_name = ref_dir + "/output-" + std::to_string(i) + "-" + name + ".bin";
        std::ifstream file(ref_name, std::ios::binary);
        if (!file) {
            throw std::ios_base::failure("File " + ref_name + " not found");
        }

        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        if (fileSize != output_tensor.get_byte_size()) {
            throw std::runtime_error("fileSize != input_tensor_size, " + std::to_string(fileSize) +
                                     " != " + std::to_string(output_tensor.get_byte_size()));
        }

        std::vector<char> buffer(fileSize);
        file.read(buffer.data(), fileSize);
        ov::Tensor ref_tensor(output_tensor.get_element_type(), output_tensor.get_shape());
        memcpy(ref_tensor.data(), buffer.data(), buffer.size());
        std::cout << "Comparing output " << name << " with ref " << ref_name << std::endl;
        compare(output_tensor, ref_tensor);

    }

}

//if b_target is false: reference, generate random input
//if b_target is true: target, compare output after inference
void run_model(std::string model_path, std::string device, std::string input_dir, std::string output_dir, std::string ref_dir, bool b_target)
{
    std::cout << "Inference on device " << device << " started" << std::endl;

    ov::AnyMap cpu_config = {};

    ov::AnyMap npuw_config = {
	{"NPU_USE_NPUW", "YES"},
        {"NPUW_DEVICES", "CPU"},
        {"NPUW_FOLD", "YES"},
        {"NPUW_FUNCALL_FOR_ALL", "YES"},
        {"NPUW_WEIGHTS_BANK", "shared"},
        {"NPUW_ONLINE_KEEP_BLOCKS", "26"},
        {"NPUW_SLICE_OUT", "YES"},
        {"NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES"}
    };

    ov::Core core;
    ov::CompiledModel compiled_model;
    if (device == "NPU") {
        compiled_model = core.compile_model(model_path, device, npuw_config);
    } else {
	compiled_model = core.compile_model(model_path, device, cpu_config);
    }
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    bool transpose = false;
    if (!b_target && FLAGS_fixed.empty()) {
        load_random_inputs(infer_request, compiled_model, input_dir);
    } else {
        load_inputs(infer_request, compiled_model, input_dir);
    }
    infer_request.infer();
    std::cout << "Inference on device " << device << " finished" << std::endl;
    std::cout << "logging output" << std::endl;
    log_outputs(infer_request, compiled_model, output_dir);
    if (b_target) {
	std::cout << "compare outputs" << std::endl;
        compare_outputs(infer_request, compiled_model, ref_dir);
    }
}

#define INPUT_SUBDIR "input"
#define REFOUT_SUBDIR "ref_output"
#define DEVOUT_SUBDIR "device_output"

#define MSE_threshold "0.01"

int main(int argc, char* argv[]) {
    ParseAndCheckCommandLine(argc, argv);
    const std::string model_path = FLAGS_m;
    const std::string device = "CPU";
    const std::string dir = FLAGS_f;
    std::string input_dir = dir + "/" + INPUT_SUBDIR;
    std::string ref_dir = dir + "/" + REFOUT_SUBDIR;
    std::string dev_dir = dir + "/" + DEVOUT_SUBDIR;

    if (system(("mkdir -p " + input_dir).c_str()) != 0 ||
	system(("mkdir -p " + ref_dir).c_str()) != 0 ||
	system(("mkdir -p " + dev_dir).c_str()) != 0) {
	std::cerr << "Cannot create output dir" << std::endl;
	return 1;
    }

    run_model(model_path, "CPU", input_dir, ref_dir, ref_dir, false);
    run_model(model_path, FLAGS_d, input_dir, dev_dir, ref_dir, true);
}
