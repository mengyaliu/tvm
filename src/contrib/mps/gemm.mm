#include "../../runtime/metal/metal_common.h"
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <dmlc/logging.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>

namespace tvm {
namespace contrib {

using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.contrib.mps.matmul")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      DLTensor *A = args[0];
      DLTensor *B = args[1];
      DLTensor *C = args[2];
      bool transa = args[3];
      bool transb = args[4];
      // call gemm for simple compact code.
      CHECK_EQ(A->ndim, 2);
      CHECK_EQ(B->ndim, 2);
      CHECK_EQ(C->ndim, 2);
      CHECK(C->strides == nullptr);
      CHECK(B->strides == nullptr);
      CHECK(A->strides == nullptr);
      CHECK(TypeMatch(A->dtype, kDLFloat, 32));
      CHECK(TypeMatch(B->dtype, kDLFloat, 32));
      CHECK(TypeMatch(C->dtype, kDLFloat, 32));
      // Get Metal device API
      MetalThreadEntry* entry_ptr = MetalThreadEntry::ThreadLocal();
      CHECK_EQ(A->ctx, B->ctx);
      CHECK_EQ(A->ctx, C->ctx);
      id<MTLDevice> dev = entry_ptr->metal_api->GetDevice(A->ctx);
      id<MTLCommandQueue> queue = entry_ptr->metal_api->GetCommandQueue(A->ctx);
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      NSUInteger M = A->shape[0 + transa?1:0];
      NSUInteger N = B->shape[1 - transb?1:0];
      NSUInteger K = B->shape[0 + transb?1:0];
      CHECK_EQ(A->shape[1-transa?1:0], K);
      // mps a
      MPSDataType dtype = MPSType::DLTypeToMPSType(A->dtype);
      MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
          matrixDescriptorWithDimensions:M
                                 columns:K
                                rowBytes:M * sizeof(dtype)
                                dataType:dtype];
      id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)(A->data);
      MPSMatrix *matrixA =
          [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
      // mps b
      MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
          matrixDescriptorWithDimensions:K
                                 columns:N
                                rowBytes:K * sizeof(dtype)
                                dataType:dtype];
      id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)(B->data);
      MPSMatrix *matrixB =
          [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
      // mps c
      MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
          matrixDescriptorWithDimensions:M
                                 columns:N
                                rowBytes:M * sizeof(dtype)
                                dataType:dtype];
      id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)(C->data);
      MPSMatrix *matrixC =
          [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];
      // kernel

      MPSMatrixMultiplication *mul_obj = [[MPSMatrixMultiplication alloc] init];
      MPSMatrixMultiplication *sgemm = [mul_obj initWithDevice:dev
                                                 transposeLeft:transa
                                                transposeRight:transb
                                                    resultRows:M
                                                 resultColumns:N
                                               interiorColumns:K
                                                         alpha:1.0f
                                                          beta:0.0f];
      CHECK(sgemm != nil);
      [sgemm encodeToCommandBuffer:cb
                        leftMatrix:matrixA
                       rightMatrix:matrixB
                      resultMatrix:matrixC];
      [cb commit];
      [mul_obj dealloc];
      [matrixA dealloc];
      [matrixB dealloc];
      [matrixC dealloc];
    });

} // namespace contrib
} // namespace tvm
