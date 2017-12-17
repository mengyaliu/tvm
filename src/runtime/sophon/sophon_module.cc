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
    // alloc neuron mem: broadcast
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

    const char *input = (const char*)input_tvm->data;
    printf("---%s %d: input[0]=0x%x\n", __func__, __LINE__, input[0]);
    bm_memcpy_s2d_address(handle, input_mem, const_cast<char*>(input));

    bmmem_device_t weight_mem = static_cast<bmmem_device_t>(weight_tvm->data);
    uint64_t global_weight_base = bmmem_device_get_device_addr(weight_mem);
    char *weight_mmm = (char*)malloc(weight_bsize);
    bm_memcpy_d2s_address(handle, weight_mmm, weight_mem);
    printf("---%s %d: weight[0]=0x%x\n", __func__, __LINE__, weight_mmm[0]);

    u32 cmdbuf_size = cmdbuf.size();
    std::cout << "cmdbuf_size = " << cmdbuf_size << std::endl;
    void *cmdbuf_data = (void*)&cmdbuf[0];
    bmkernel_relocate_cmdbuf(handle, cmdbuf_data, cmdbuf_size, global_neuron_base, global_weight_base);
    bmkernel_send_cmdbuf(handle, cmdbuf_data, cmdbuf_size);

    char *output = (char*)output_tvm->data;
    bm_memcpy_d2s_address(handle, output, output_mem);
    printf("---%s %d: output[0]=0x%x\n", __func__, __LINE__, output[0]);

    bmmem_device_free(handle, neuron_mem);
    bmmem_device_prefree(handle, input_mem);
    bmmem_device_prefree(handle, output_mem);
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

    std::cout << __FILE__ <<  " "
              << __func__ << " "
              << __LINE__ << ":"
              << " fmt = " << fmt
              << std::endl;
    for(auto item : fmap) {
      std::cout << " key =" << item.first
                << " value = " << item.second.name
                << std::endl;
    }

    for(auto f : fmap) {
      std::string kernel_file = f.second.sophon_kernel;
      std::string kernel_data;
      std::cout << "---" << __func__ << " " << __LINE__ << ": " << f.second.sophon_kernel<< std::endl;
      LoadBinaryFromFile(kernel_file, &kernel_data);
      kmap_.insert(std::pair<std::string, std::string>(f.first, kernel_data));
    }
  }
  // destructor
  ~SophonModuleNode() {
//    for (size_t i = 0; i < module_.size(); ++i) {
//      if (module_[i] != nullptr) {
//        ROCM_CALL(hipSetDevice(static_cast<int>(i)));
//        ROCM_DRIVER_CALL(hipModuleUnload(module_[i]));
//      }
//    }
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
    batch_num_ = batch_num;
    input_dsize_ = input_dsize;
    output_dsize_ = output_dsize;
    weight_bsize_ = weight_bsize;
    neuron_bsize_ = neuron_bsize;
    output_offset_ = output_offset;
    std::cout << __FILE__ <<  " "
              << __func__ << " "
              << __LINE__ << " "
              << " input_dsize_ = " << input_dsize_
              << " output_dsize_ = " << output_dsize_
              << " weight_bsize_ = " << weight_bsize_
              << " neuron_bsize_ = " << neuron_bsize_
              << std::endl;
  }

  // invoke the function with void arguments
  void operator()(TVMArgs args,
                  TVMRetValue* rv,
                  void* packed_args,
                  size_t packed_nbytes) const {
    std::cout << __FILE__ <<  " "
              << __func__ << " "
              << __LINE__
              << std::endl;

    TVMArray *input_tvm = (TVMArray*)args[0];
    TVMArray *weight_tvm = (TVMArray*)args[1];
    TVMArray *output_tvm = (TVMArray*)args[2];

    bmdnn_handle_t handle = SophonThreadEntry::ThreadLocal()->stream;
    CHECK(handle != nullptr);

    bmdnn_run_cmdbuf(handle, batch_num_, input_dsize_, output_dsize_, weight_bsize_,
                     neuron_bsize_, output_offset_, input_tvm, output_tvm, weight_tvm, cmdbuf_);
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

    std::cout << __FILE__ <<  " "
              << __func__ << " "
              << __LINE__ << " "
              << std::endl;
  }

  // invoke the function with void arguments
  void operator()(TVMArgs args,
                  TVMRetValue* rv,
                  void* packed_args,
                  size_t packed_nbytes) const {
    std::cout << __FILE__ <<  " "
              << __func__ << " "
              << __LINE__
              << std::endl;

    TVMArray *input_tvm = (TVMArray*)args[0];
    TVMArray *weight_tvm = (TVMArray*)args[1];
    TVMArray *output_tvm = (TVMArray*)args[2];
    int batch_num = input_tvm->shape[0];

    for(auto f : fmap_) {
      auto function_info = f.second;
      if(function_info.sophon_batch_num == batch_num) {
        std::cout << __FILE__ <<  " "
                  << __func__ << " "
                  << __LINE__ << " "
                  << "run batch_num = " << batch_num
                  << std::endl;
        uint64_t input_dsize = function_info.sophon_input_dsize;
        uint64_t output_dsize = function_info.sophon_output_dsize;
        uint64_t weight_bsize = function_info.sophon_weight_bsize;
        uint64_t neuron_bsize = function_info.sophon_neuron_bsize;
        uint64_t output_offset = function_info.sophon_output_offset;
        auto kernel_it = kmap_.find(f.first);

        bmdnn_handle_t handle = SophonThreadEntry::ThreadLocal()->stream;
        CHECK(handle != nullptr);

        bmdnn_run_cmdbuf(handle, batch_num,
                         input_dsize, output_dsize,
                         weight_bsize, neuron_bsize,
                         output_offset, input_tvm,
                         output_tvm, weight_tvm,
                         kernel_it->second);
        return;
      }
    }

    LOG(FATAL) << "cannot find sophon kernel for batch_num = " << batch_num;
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
  std::cout << __FILE__ <<  " "
            << __func__ << " "
            << __LINE__
            << ": name = " << name
            << std::endl;
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
    f.Init(this, sptr_to_self, name,
           info.arg_types.size(), info.thread_axis_tags,
           info.sophon_device_type, kernel_data,
           info.sophon_batch_num, info.sophon_input_dsize,
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
