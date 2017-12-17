/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Example code on load and run TVM module.s
 * \file sophon_deploy_example.cc
 */
#include <cstdio>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

int main(void) {
  //TODO(wwcai): just run under current directory to find meta_info
  tvm::runtime::Module mod =
      tvm::runtime::Module::LoadFromFile("../../tutorials/sophon/meta_info/resnet50/resnet50.tvm_meta.json");
  LOG(INFO) << "Verify dynamic loading from resnet50.tvm_meta.json";

  // Get the function from the module.
  std::string fname = "__tvm_main__";
  //std::string fname = "__tvm_reset50_1__";
  tvm::runtime::PackedFunc f = mod.GetFunction(fname);
  CHECK(f != nullptr);
  // Allocate the DLPack data structures.
  //
  // Note that we use TVM runtime API to allocate the DLTensor in this example.
  // TVM accept DLPack compatible DLTensors, so function can be invoked
  // as long as we pass correct pointer to DLTensor array.
  //
  // For more information please refer to dlpack.
  // One thing to notice is that DLPack contains alignment requirement for
  // the data pointer and TVM takes advantage of that.
  // If you plan to use your customized data container, please
  // make sure the DLTensor you pass in meet the alignment requirement.
  //
  DLTensor* input;
  DLTensor* weight;
  DLTensor* output;

  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_id = 0;
  int ndim = 4;
  int64_t shape[4] = {1, 3, 224, 224};
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                kDLCPU, device_id, &input);

  int64_t w_shape[1] = {25610269};
  TVMArrayAlloc(w_shape, 1, dtype_code, dtype_bits, dtype_lanes,
                kDLSophon, device_id, &weight);
  //TODO(wwcai): call TVMArrayCopyFromBytes to upload weight data

  int64_t o_shape[1] = {1000};
  TVMArrayAlloc(w_shape, 1, dtype_code, dtype_bits, dtype_lanes,
                kDLCPU, device_id, &output);

  // Invoke the function
  // PackedFunc is a function that can be invoked via positional argument.
  // The signature of the function is specified in tvm.build
  f(input, weight, output);

  TVMArrayFree(input);
  TVMArrayFree(weight);
  TVMArrayFree(output);
  LOG(INFO) << "Finish verification...";
  return 0;
}
