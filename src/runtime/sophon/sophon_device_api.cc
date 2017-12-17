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
    //TODO(wwcai)
    LOG(FATAL) << "Not support now.";
  }

  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    //TODO(wwcai)
    LOG(FATAL) << "Not support now.";
  }

  void* AllocDataSpace(TVMContext ctx, size_t size, size_t alignment) final {
    //TODO(wwcai): check ctx.device_id and stream
    LOG(WARNING) << "size = " << size << " alignment = " << alignment;
    CHECK_EQ(alignment % 32, 0U)
        << "Sophon space is aligned at 32 bytes (support fp32 only now)";
    bmdnn_handle_t handle = SophonThreadEntry::ThreadLocal()->stream;
    CHECK(handle != nullptr);
    void *ret = bmmem_device_alloc_coeff(handle, size / sizeof(float)); //TODO(wwcai): support int8 and other formats
    return ret;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
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
    CHECK(stream == nullptr);
    bmdnn_handle_t handle = SophonThreadEntry::ThreadLocal()->stream;
    CHECK(handle != nullptr);

    if (ctx_from.device_type == kDLSophon && ctx_to.device_type == kDLSophon) {
      LOG(FATAL) << "Not support now.";
    } else if (ctx_from.device_type == kDLSophon && ctx_to.device_type == kDLCPU) {
      void *to_cpu = static_cast<char*>(to) + to_offset;
      CHECK_EQ(from_offset, 0);
      bm_memcpy_d2s_address(handle, to_cpu, (bmmem_device_t)from); //TODO(wwcai): wrong
    } else if (ctx_from.device_type == kDLCPU && ctx_to.device_type == kDLSophon) {
      void *from_cpu = const_cast<void*>(from);
      from_cpu = static_cast<char*>(from_cpu) + from_offset;
      CHECK_EQ(to_offset, 0);
      bm_memcpy_s2d_address(handle, (bmmem_device_t)to, from_cpu);
    } else {
      LOG(FATAL) << "expect copy from/to Sophon TPU or between Sophon TPU";
    }
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
    //TODO(wwcai)
    LOG(FATAL) << "Not support now.";
  }

  void SetStream(TVMContext ctx, TVMStreamHandle stream) final {
    //TODO(wwcai)
    LOG(FATAL) << "Not support now.";
  }

  void* AllocWorkspace(TVMContext ctx, size_t size) final {
    //TODO(wwcai)
    LOG(FATAL) << "Not support now.";
    return NULL;
  }

  void FreeWorkspace(TVMContext ctx, void* data) final {
    //TODO(wwcai)
    LOG(FATAL) << "Not support now.";
  }

  static const std::shared_ptr<SophonDeviceAPI>& Global() {
    static std::shared_ptr<SophonDeviceAPI> inst =
        std::make_shared<SophonDeviceAPI>();
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
