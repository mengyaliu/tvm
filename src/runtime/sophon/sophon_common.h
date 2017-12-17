/*
 * tvm/src/runtime/sophon/sophon_common.h
 *
 * Copyright Bitmain Technologies Inc.
 * Written by:
 *   Wanwei CAI <wanwei.cai@bitmain.com>
 * Created Time: 2017-12-10 14:58
 */

#ifndef _SOPHON_COMMON_H
#define _SOPHON_COMMON_H

#if TVM_CUDA_RUNTIME
#include "../workspace_pool.h"
#include <bm_runtime.h>
#include <bm_memory.h>
#include <bm_memory_common.h>
#include <bmkernel_runtime.h>

namespace tvm {
namespace runtime {

/*! \brief Thread local workspace */
class SophonThreadEntry {
 public:
  /*! \brief The bmdnn stream */
  bmdnn_handle_t stream{nullptr};
  /*! \brief thread local pool*/
  WorkspacePool pool;
  /*! \brief constructor */
  SophonThreadEntry();
  /*! \brief deconstructor */
  ~SophonThreadEntry();
  // get the threadlocal workspace
  static SophonThreadEntry* ThreadLocal();
};
}  // namespace runtime
}  // namespace tvm
#endif
#endif
