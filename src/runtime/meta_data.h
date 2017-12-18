/*!
 *  Copyright (c) 2017 by Contributors
 * \file meta_data.h
 * \brief Meta data related utilities
 */
#ifndef TVM_RUNTIME_META_DATA_H_
#define TVM_RUNTIME_META_DATA_H_

#include <dmlc/json.h>
#include <dmlc/io.h>
#include <tvm/runtime/packed_func.h>
#include <string>
#include <vector>
#include "./runtime_base.h"

namespace tvm {
namespace runtime {

/*! \brief function information needed by device */
struct FunctionInfo {
  std::string name;
  std::vector<TVMType> arg_types;
  std::vector<std::string> thread_axis_tags;

  //Sophon configs TODO(wwcai)
  int sophon_device_type;
  std::string sophon_kernel;
  int sophon_input_n;
  int sophon_input_c;
  int sophon_input_h;
  int sophon_input_w;
  uint64_t sophon_input_dsize; //d: dispatch
  uint64_t sophon_output_dsize;
  uint64_t sophon_weight_bsize; //b: broadcast
  uint64_t sophon_neuron_bsize;
  uint64_t sophon_output_offset;

  void Save(dmlc::JSONWriter *writer) const;
  void Load(dmlc::JSONReader *reader);
  void Save(dmlc::Stream *writer) const;
  bool Load(dmlc::Stream *reader);
};
}  // namespace runtime
}  // namespace tvm

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, ::tvm::runtime::FunctionInfo, true);
}  // namespace dmlc
#endif  // TVM_RUNTIME_META_DATA_H_
