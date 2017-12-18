/*!
 *  Copyright (c) 2017 by Contributors
 * \file file_util.cc
 */
#include <dmlc/json.h>
#include <dmlc/logging.h>
#include <fstream>

#include "./file_util.h"

namespace tvm {
namespace runtime {

void FunctionInfo::Save(dmlc::JSONWriter* writer) const {
  std::vector<std::string> sarg_types(arg_types.size());
  for (size_t i = 0; i < arg_types.size(); ++i) {
    sarg_types[i] = TVMType2String(arg_types[i]);
  }
  writer->BeginObject();
  writer->WriteObjectKeyValue("name", name);
  writer->WriteObjectKeyValue("arg_types", sarg_types);
  writer->WriteObjectKeyValue("thread_axis_tags", thread_axis_tags);
  writer->WriteObjectKeyValue("sophon_device_type", sophon_device_type);
  writer->WriteObjectKeyValue("sophon_kernel", sophon_kernel);
  writer->WriteObjectKeyValue("sophon_input_n", sophon_input_n);
  writer->WriteObjectKeyValue("sophon_input_c", sophon_input_c);
  writer->WriteObjectKeyValue("sophon_input_h", sophon_input_h);
  writer->WriteObjectKeyValue("sophon_input_w", sophon_input_w);
  writer->WriteObjectKeyValue("sophon_input_dsize", sophon_input_dsize);
  writer->WriteObjectKeyValue("sophon_output_dsize", sophon_output_dsize);
  writer->WriteObjectKeyValue("sophon_weight_bsize", sophon_weight_bsize);
  writer->WriteObjectKeyValue("sophon_neuron_bsize", sophon_neuron_bsize);
  writer->WriteObjectKeyValue("sophon_output_offset", sophon_output_offset);
  writer->EndObject();
}

void FunctionInfo::Load(dmlc::JSONReader* reader) {
  dmlc::JSONObjectReadHelper helper;
  std::vector<std::string> sarg_types;
  helper.DeclareField("name", &name);
  helper.DeclareField("arg_types", &sarg_types);
  helper.DeclareField("thread_axis_tags", &thread_axis_tags);
  helper.DeclareField("sophon_device_type", &sophon_device_type);
  helper.DeclareField("sophon_kernel", &sophon_kernel);
  helper.DeclareField("sophon_input_n", &sophon_input_n);
  helper.DeclareField("sophon_input_c", &sophon_input_c);
  helper.DeclareField("sophon_input_h", &sophon_input_h);
  helper.DeclareField("sophon_input_w", &sophon_input_w);
  helper.DeclareField("sophon_input_dsize", &sophon_input_dsize);
  helper.DeclareField("sophon_output_dsize", &sophon_output_dsize);
  helper.DeclareField("sophon_weight_bsize", &sophon_weight_bsize);
  helper.DeclareField("sophon_neuron_bsize", &sophon_neuron_bsize);
  helper.DeclareField("sophon_output_offset", &sophon_output_offset);
  helper.ReadAllFields(reader);
  arg_types.resize(sarg_types.size());
  for (size_t i = 0; i < arg_types.size(); ++i) {
    arg_types[i] = String2TVMType(sarg_types[i]);
  }
}

void FunctionInfo::Save(dmlc::Stream* writer) const {
  writer->Write(name);
  writer->Write(arg_types);
  writer->Write(thread_axis_tags);
  writer->Write(sophon_device_type);
  writer->Write(sophon_kernel);
  writer->Write(sophon_input_n);
  writer->Write(sophon_input_c);
  writer->Write(sophon_input_h);
  writer->Write(sophon_input_w);
  writer->Write(sophon_input_dsize);
  writer->Write(sophon_output_dsize);
  writer->Write(sophon_weight_bsize);
  writer->Write(sophon_neuron_bsize);
  writer->Write(sophon_output_offset);
}

bool FunctionInfo::Load(dmlc::Stream* reader) {
  if (!reader->Read(&name)) return false;
  if (!reader->Read(&arg_types)) return false;
  if (!reader->Read(&thread_axis_tags)) return false;
  if (!reader->Read(&sophon_device_type)) return false;
  if (!reader->Read(&sophon_kernel)) return false;
  if (!reader->Read(&sophon_input_n)) return false;
  if (!reader->Read(&sophon_input_c)) return false;
  if (!reader->Read(&sophon_input_h)) return false;
  if (!reader->Read(&sophon_input_w)) return false;
  if (!reader->Read(&sophon_input_dsize)) return false;
  if (!reader->Read(&sophon_output_dsize)) return false;
  if (!reader->Read(&sophon_weight_bsize)) return false;
  if (!reader->Read(&sophon_neuron_bsize)) return false;
  if (!reader->Read(&sophon_output_offset)) return false;
  return true;
}

std::string GetFileFormat(const std::string& file_name,
                          const std::string& format) {
  std::string fmt = format;
  if (fmt.length() == 0) {
    size_t pos = file_name.find_last_of(".");
    if (pos != std::string::npos) {
      return file_name.substr(pos + 1, file_name.length() - pos - 1);
    } else {
      return "";
    }
  } else {
    return format;
  }
}

std::string GetMetaFilePath(const std::string& file_name) {
  size_t pos  = file_name.find_last_of(".");
  if (pos != std::string::npos) {
    return file_name.substr(0, pos) + ".tvm_meta.json";
  } else {
    return file_name + ".tvm_meta.json";
  }
}

std::string GetKernelFilePath(const std::string& metafile_name, const std::string& file_name) {
  size_t pos  = metafile_name.find_last_of("/");
  if (pos != std::string::npos) {
    return metafile_name.substr(0, pos) + "/" + file_name;
  } else {
    return metafile_name + "/" + file_name;
  }
}

void LoadBinaryFromFile(const std::string& file_name,
                        std::string* data) {
  std::ifstream fs(file_name, std::ios::in | std::ios::binary);
  CHECK(!fs.fail()) << "Cannot open " << file_name;
  // get its size:
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data->resize(size);
  fs.read(&(*data)[0], size);
}

void SaveBinaryToFile(
    const std::string& file_name,
    const std::string& data) {
  std::ofstream fs(file_name, std::ios::out | std::ios::binary);
  CHECK(!fs.fail()) << "Cannot open " << file_name;
  fs.write(&data[0], data.length());
}

void SaveMetaDataToFile(
    const std::string& file_name,
    const std::unordered_map<std::string, FunctionInfo>& fmap) {
  std::string version = "0.1.0";
  std::ofstream fs(file_name.c_str());
  CHECK(!fs.fail()) << "Cannot open file " << file_name;
  dmlc::JSONWriter writer(&fs);
  writer.BeginObject();
  writer.WriteObjectKeyValue("tvm_version", version);
  writer.WriteObjectKeyValue("func_info", fmap);
  writer.EndObject();
  fs.close();
}

void LoadMetaDataFromFile(
    const std::string& file_name,
    std::unordered_map<std::string, FunctionInfo>* fmap) {
  std::ifstream fs(file_name.c_str());
  CHECK(!fs.fail()) << "Cannot open file " << file_name;
  std::string version;
  dmlc::JSONReader reader(&fs);
  dmlc::JSONObjectReadHelper helper;
  helper.DeclareField("tvm_version", &version);
  helper.DeclareField("func_info", fmap);
  helper.ReadAllFields(&reader);
  fs.close();
}

}  // namespace runtime
}  // namespace tvm
