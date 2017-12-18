/*!
 *  Copyright (c) 2017 by Contributors
 * \file sophon_module.cc
 */
#include "./sophon_module.h"

#if TVM_SOPHON_RUNTIME

#include <tvm/runtime/registry.h>
#include <vector>
#include <array>
#include <string>
#include <mutex>
#include "./sophon_common.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "../meta_data.h"
#include "../file_util.h"

namespace tvm {
namespace runtime {

// bmdnn helper
void bmdnn_run_cmdbuf(bmdnn_handle_t handle,
                      int nodechip_num,
                      int batch_num,
                      uint64_t input_dsize,
                      uint64_t output_dsize,
                      uint64_t weight_bsize,
                      uint64_t neuron_bsize,
                      uint64_t output_offset,
                      TVMArray *input_tvm,
                      TVMArray *output_tvm,
                      TVMArray *weight_tvm,
                      const std::string cmdbuf) {
  LOG(WARNING) << " nodechip_num = " << nodechip_num
               << " batch_num = " << batch_num
               << " input_dsize = " << input_dsize
               << " output_dsize = " << output_dsize
               << " weight_bsize = " << weight_bsize
               << " neuron_bsize = " << neuron_bsize
               << " output_offset = " << output_offset;
  // check params
  CHECK(handle);
  CHECK(nodechip_num > 0);
  CHECK(input_dsize % (batch_num * sizeof(float)) == 0);
  CHECK(output_dsize % (batch_num * sizeof(float)) == 0);

  // alloc neuron mem: broadcast
  ASSERT(neuron_bsize >= output_offset + (output_dsize / nodechip_num));
  ASSERT(neuron_bsize <= 0x400000000ULL); //TODO(wwcai): hardcode 16G
  bmmem_device_t neuron_mem = bmmem_device_alloc_coeff(handle, neuron_bsize / sizeof(float));
  uint64_t global_neuron_base = bmmem_device_get_device_addr(neuron_mem);

  // prealloc input mem: dispatch
  bmmem_device_t input_mem = bmmem_device_prealloc_neuron(
                                handle,
                                global_neuron_base,
                                batch_num,
                                1, 1,
                                input_dsize / batch_num / sizeof(float),
                                false, false, 0, 0, 0);

  // prealloc output mem: dispatch
  bmmem_device_t output_mem = bmmem_device_prealloc_neuron(
                                handle,
                                global_neuron_base + output_offset,
                                batch_num,
                                1, 1,
                                output_dsize / batch_num / sizeof(float),
                                false, false, 0, 0, 0);

  const char *input = static_cast<const char*>(input_tvm->data);
  CHECK(!bm_memcpy_s2d_address(handle, input_mem, const_cast<char*>(input)));

  bmmem_device_t weight_mem = static_cast<bmmem_device_t>(weight_tvm->data);
  uint64_t global_weight_base = bmmem_device_get_device_addr(weight_mem);

  u32 cmdbuf_size = cmdbuf.size();
  void *cmdbuf_data = (void*)&cmdbuf[0];
  bmkernel_relocate_cmdbuf(handle, cmdbuf_data, cmdbuf_size, global_neuron_base, global_weight_base);
  bmkernel_send_cmdbuf(handle, cmdbuf_data, cmdbuf_size);

  char *output = (char*)output_tvm->data;
  CHECK(!bm_memcpy_d2s_address(handle, output, output_mem));

  bmmem_device_free(handle, neuron_mem);
  bmmem_device_prefree(handle, input_mem);
  bmmem_device_prefree(handle, output_mem);
}

int get_nodechip_num(int device_type) {
  if(16802 == device_type) { //TODO(wwcai)
    return 2;
  } else {
    return 1;
  }
}

// Module to support thread-safe multi-TPU execution.
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class SophonModuleNode : public runtime::ModuleNode {
 public:
  explicit SophonModuleNode(std::string data,
                          std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap)
    : data_(data), fmt_(fmt), fmap_(fmap) {
    //std::fill(module_.begin(), module_.end(), nullptr);
    for(auto item : fmap) {
    }
    for(auto f : fmap) {
      LOG(WARNING) << "Loading"
                   << " key =" << f.first
                   << " name = " << f.second.name;
      std::string kernel_file = f.second.sophon_kernel;
      std::string kernel_data;
      LoadBinaryFromFile(kernel_file, &kernel_data);
      kmap_.insert(std::pair<std::string, std::string>(f.first, kernel_data));
    }
  }

  // destructor
  ~SophonModuleNode() {
  }

  const char* type_key() const final {
    return "sophon";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final;

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
  }

#if 0
  std::string GetSource(const std::string& format) final {
    if (format == fmt_) { return data_; }
    //if (format == "llvm") { return hip_source_; }
    //if (format == "asm") { return assembly_; }
    return data_;
    //return "";
  }
#endif

 private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // kernel_file information table.
  std::unordered_map<std::string, std::string> kmap_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed func
class SophonWrappedFunc {
 public:
  // initialize the Sophon function.
  void Init(SophonModuleNode* m,
            std::shared_ptr<ModuleNode> sptr,
            const std::string& func_name,
            size_t num_void_args,
            const std::vector<std::string>& thread_axis_tags,
            int device_type,
            std::string cmdbuf,
            int nodechip_num,
            int batch_num,
            uint64_t input_dsize,
            uint64_t output_dsize,
            uint64_t weight_bsize,
            uint64_t neuron_bsize,
            uint64_t output_offset) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    thread_axis_cfg_.Init(num_void_args, thread_axis_tags);

    // sophon configs
    cmdbuf_ = cmdbuf;
    nodechip_num_ = nodechip_num;
    batch_num_ = batch_num;
    input_dsize_ = input_dsize;
    output_dsize_ = output_dsize;
    weight_bsize_ = weight_bsize;
    neuron_bsize_ = neuron_bsize;
    output_offset_ = output_offset;
  }

  // invoke the function with void arguments
  void operator()(TVMArgs args,
                  TVMRetValue* rv,
                  void* packed_args,
                  size_t packed_nbytes) const {
    TVMArray *input_tvm = (TVMArray*)args[0];
    TVMArray *weight_tvm = (TVMArray*)args[1];
    TVMArray *output_tvm = (TVMArray*)args[2];

    bmdnn_handle_t handle = SophonThreadEntry::ThreadLocal()->stream;
    CHECK(handle != nullptr);

    LOG(WARNING) << "run cmdbuf";
    bmdnn_run_cmdbuf(handle, nodechip_num_, batch_num_,
                     input_dsize_, output_dsize_, weight_bsize_,
                     neuron_bsize_, output_offset_, input_tvm,
                     output_tvm, weight_tvm, cmdbuf_);
  }

 private:
  // internal module
  SophonModuleNode* m_;
  // the resource holder
  std::shared_ptr<ModuleNode> sptr_;
  // The name of the function.
  std::string func_name_;
  // thread axis configuration
  ThreadAxisConfig thread_axis_cfg_;

  // sophon configs
  std::string cmdbuf_;
  int nodechip_num_;
  int batch_num_;
  uint64_t input_dsize_;
  uint64_t output_dsize_;
  uint64_t weight_bsize_;
  uint64_t neuron_bsize_;
  uint64_t output_offset_;
};

// a wrapped function class to get packed func by batch_num
class SophonMainWrappedFunc {
 public:
  // initialize the Sophon function.
  void Init(SophonModuleNode* m,
            std::shared_ptr<ModuleNode> sptr,
            const std::string& func_name,
            std::unordered_map<std::string, FunctionInfo>& fmap,
            std::unordered_map<std::string, std::string>& kmap) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;

    // sophon configs
    fmap_ = fmap;
    kmap_ = kmap;
  }

  // invoke the function with void arguments
  void operator()(TVMArgs args,
                  TVMRetValue* rv,
                  void* packed_args,
                  size_t packed_nbytes) const {
    TVMArray *input_tvm = (TVMArray*)args[0];
    TVMArray *weight_tvm = (TVMArray*)args[1];
    TVMArray *output_tvm = (TVMArray*)args[2];
    CHECK(input_tvm->ndim == 4);
    int input_n = input_tvm->shape[0];
    int input_c = input_tvm->shape[1];
    int input_h = input_tvm->shape[2];
    int input_w = input_tvm->shape[3];

    for(auto f : fmap_) {
      auto function_info = f.second;
      if(function_info.sophon_input_n == input_n
         && function_info.sophon_input_c == input_c
         && function_info.sophon_input_h == input_h
         && function_info.sophon_input_w == input_w) {
        LOG(WARNING) << "run function_info(" << function_info.name
                     << ") shape = (" << input_n
                     << " " << input_c
                     << " " << input_h
                     << " " << input_w
                     << ")";
        int nodechip_num = get_nodechip_num(function_info.sophon_device_type);
        uint64_t input_dsize = function_info.sophon_input_dsize;
        uint64_t output_dsize = function_info.sophon_output_dsize;
        uint64_t weight_bsize = function_info.sophon_weight_bsize;
        uint64_t neuron_bsize = function_info.sophon_neuron_bsize;
        uint64_t output_offset = function_info.sophon_output_offset;
        auto kernel_it = kmap_.find(f.first);

        bmdnn_handle_t handle = SophonThreadEntry::ThreadLocal()->stream;
        CHECK(handle != nullptr);

        bmdnn_run_cmdbuf(handle, nodechip_num, input_n,
                         input_dsize, output_dsize,
                         weight_bsize, neuron_bsize,
                         output_offset, input_tvm,
                         output_tvm, weight_tvm,
                         kernel_it->second);
        return;
      }
    }

    LOG(FATAL) << "cannot find sophon kernel for input"
               << " shape = (" << input_n
               << " " << input_c
               << " " << input_h
               << " " << input_w
               << ")";
  }

 private:
  // internal module
  SophonModuleNode* m_;
  // the resource holder
  std::shared_ptr<ModuleNode> sptr_;
  // The name of the function.
  std::string func_name_;

  // sophon configs
  std::unordered_map<std::string, FunctionInfo> fmap_;
  std::unordered_map<std::string, std::string> kmap_;
};

PackedFunc SophonModuleNode::GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) {
  CHECK_EQ(sptr_to_self.get(), this);
  LOG(WARNING) << ": name = " << name;
  //CHECK_NE(name, symbol::tvm_module_main)
  //    << "Device function do not have main"; //TODO(wwcai)

  if(name == symbol::tvm_module_main) {
    SophonMainWrappedFunc f;
    auto it = fmap_.begin();
    const FunctionInfo& info = it->second; //TODO(wwcai): trick
    f.Init(this, sptr_to_self, name, fmap_, kmap_);
    return PackFuncPackedArg(f, info.arg_types);
  } else {
    auto it = fmap_.find(name);
    if (it == fmap_.end()) return PackedFunc();
    const FunctionInfo& info = it->second;
    auto kernel_it = kmap_.find(name);
    const std::string& kernel_data = kernel_it->second;
    SophonWrappedFunc f;
    int nodechip_num = get_nodechip_num(info.sophon_device_type);
    f.Init(this, sptr_to_self, name,
           info.arg_types.size(), info.thread_axis_tags,
           info.sophon_device_type, kernel_data, nodechip_num,
           info.sophon_input_n, info.sophon_input_dsize,
           info.sophon_output_dsize, info.sophon_weight_bsize,
           info.sophon_neuron_bsize, info.sophon_output_offset);
    return PackFuncPackedArg(f, info.arg_types);
  }
}

Module SophonModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap) {
  std::shared_ptr<SophonModuleNode> n =
    std::make_shared<SophonModuleNode>(data, fmt, fmap);
  return Module(n);
}

// Load module from cmdbuf.
Module SophonModuleLoadFile(const std::string& file_name,
                          const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return SophonModuleCreate(data, fmt, fmap);
}

// Load module from json.
Module SophonModuleLoadMetaInfo(const std::string& metafile_name,
                          const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(metafile_name, format);
  LoadMetaDataFromFile(metafile_name, &fmap);
  for(auto &f : fmap) {
    std::string file_name_full = GetKernelFilePath(metafile_name, f.second.sophon_kernel);
    f.second.sophon_kernel= file_name_full;
  }
  return SophonModuleCreate(data, fmt, fmap);
}

Module SophonModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return SophonModuleCreate(data, fmt, fmap);
}

#if 0
TVM_REGISTER_GLOBAL("module.loadfile_cmdbuf")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = SophonModuleLoadFile(args[0], args[1]);
  });
#endif

TVM_REGISTER_GLOBAL("module.loadfile_json")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = SophonModuleLoadMetaInfo(args[0], args[1]);
  });

#if 0
TVM_REGISTER_GLOBAL("module.loadbinary_sophon")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = SophonModuleLoadBinary(args[0]);
  });
#endif
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_SOPHON_RUNTIME
