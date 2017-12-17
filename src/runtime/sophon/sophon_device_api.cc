/*!
 *  Copyright (c) 2017 by Contributors
 * \file sophon_device_api.cc
 * \brief Bitmain Sophon specific API
 */
#include <tvm/runtime/config.h>
#include <tvm/runtime/device_api.h>

#if TVM_SOPHON_RUNTIME
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include "./sophon_common.h"

namespace tvm {
namespace runtime {

class SophonDeviceAPI final : public DeviceAPI {
 public:
  void *GetDeviceHandle() {
    bmdnn_handle_t handle = SophonThreadEntry::ThreadLocal()->stream;
    CHECK(handle != nullptr);
    return static_cast<void*>(handle);
  }

  void SetDevice(TVMContext ctx) final {
    std::cout << __FILE__ << " " << __func__ << " " << __LINE__ << std::endl;
    //ROCM_CALL(hipSetDevice(ctx.device_id));
  }

  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    std::cout << __FILE__ << " " << __func__ << " " << __LINE__ << std::endl;
  }

  void* AllocDataSpace(TVMContext ctx, size_t size, size_t alignment) final {
    std::cout << __FILE__ << " " << __func__ << " " << __LINE__ << std::endl;

//    ROCM_CALL(hipSetDevice(ctx.device_id));
//    CHECK_EQ(256 % alignment, 0U)
//        << "ROCM space is aligned at 256 bytes";
//    ROCM_CALL(hipMalloc(&ret, size));

    std::cout << __FILE__ << " " << __func__ << " " << __LINE__
              << " size=" << size
              << " alignment=" << alignment
              << std::endl;
    bmdnn_handle_t handle = SophonThreadEntry::ThreadLocal()->stream;
    CHECK(handle != nullptr);
    void *ret = bmmem_device_alloc_coeff(handle, size / sizeof(float));
    printf("---%s %d: ret=0x%x\n", __func__, __LINE__, ret);
    return ret;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    std::cout << __FILE__ << " " << __func__ << " " << __LINE__ << std::endl;
//    ROCM_CALL(hipSetDevice(ctx.device_id));
//    ROCM_CALL(hipFree(ptr));

    CHECK(ptr != nullptr);
    bmdnn_handle_t handle = SophonThreadEntry::ThreadLocal()->stream;
    CHECK(handle != nullptr);
    bmmem_device_free(handle, (bmmem_device_t)ptr);
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMStreamHandle stream) final {
    std::cout << __FILE__ << " " << __func__ << " " << __LINE__ << std::endl;
    CHECK(stream == nullptr);
    bmdnn_handle_t handle = SophonThreadEntry::ThreadLocal()->stream;
    CHECK(handle != nullptr);
    if (ctx_from.device_type == kDLSophon && ctx_to.device_type == kDLSophon) {
    } else if (ctx_from.device_type == kDLSophon && ctx_to.device_type == kDLCPU) {
      std::cout << __FILE__ << " " << __func__ << " " << __LINE__ << std::endl;
      void *to_cpu = static_cast<char*>(to) + to_offset;
      CHECK_EQ(from_offset, 0);
      bm_memcpy_d2s_address(handle, to_cpu, (bmmem_device_t)from); //TODO(wwcai): wrong
    } else if (ctx_from.device_type == kDLCPU && ctx_to.device_type == kDLSophon) {
      std::cout << __FILE__ << " " << __func__ << " " << __LINE__ << std::endl;
      void *from_cpu = const_cast<void*>(from);
      printf("from_cpu[0]=0x%x\n", ((char*)from_cpu)[0]);
      from_cpu = static_cast<char*>(from_cpu) + from_offset;
      CHECK_EQ(to_offset, 0);
      bm_memcpy_s2d_address(handle, (bmmem_device_t)to, from_cpu);
    } else {
      LOG(FATAL) << "expect copy from/to Sophon TPU or between Sophon TPU";
    }
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
    std::cout << __FILE__ << " " << __func__ << " " << __LINE__ << std::endl;
//    ROCM_CALL(hipSetDevice(ctx.device_id));
//    ROCM_CALL(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
  }

  void SetStream(TVMContext ctx, TVMStreamHandle stream) final {
    std::cout << __FILE__ << " " << __func__ << " " << __LINE__ << std::endl;
//    ROCMThreadEntry::ThreadLocal()
//        ->stream = static_cast<hipStream_t>(stream);
  }

  void* AllocWorkspace(TVMContext ctx, size_t size) final {
    std::cout << __FILE__ << " " << __func__ << " " << __LINE__ << std::endl;
    //return ROCMThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, size);
    return NULL;
  }

  void FreeWorkspace(TVMContext ctx, void* data) final {
    std::cout << __FILE__ << " " << __func__ << " " << __LINE__ << std::endl;
    //ROCMThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, data);
  }

  static const std::shared_ptr<SophonDeviceAPI>& Global() {
    static std::shared_ptr<SophonDeviceAPI> inst =
        std::make_shared<SophonDeviceAPI>();
    std::cout << __FILE__ << " " << __func__ << " " << __LINE__ << std::endl;
    return inst;
  }
};

typedef dmlc::ThreadLocalStore<SophonThreadEntry> SophonThreadStore;

SophonThreadEntry::SophonThreadEntry()
    : pool(kDLSophon, SophonDeviceAPI::Global()) {
    bmdnn_handle_t handle;
    bmdnn_init(&handle);
    stream = handle;
}

SophonThreadEntry::~SophonThreadEntry() {
    bmdnn_handle_t handle = static_cast<bmdnn_handle_t>(stream);
    if(handle != nullptr) {
      bmdnn_deinit(handle);
    }
}

SophonThreadEntry* SophonThreadEntry::ThreadLocal() {
  return SophonThreadStore::Get();
}

TVM_REGISTER_GLOBAL("device_api.sophon")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = SophonDeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
  });

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_SOPHON_RUNTIME
