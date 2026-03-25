/*
 * nml_backend_metal.m — Apple Metal GPU backend for NML matmul
 *
 * Compiled separately as Objective-C:
 *   clang -x objective-c -DNML_USE_METAL ... -c -o nml_backend_metal.o nml_backend_metal.m
 *
 * nml.c calls nml_backend_metal_matmul() via an extern declaration when
 * NML_USE_METAL is defined. The Tensor type is shared via nml_tensor.h.
 *
 * Only F32 tensors are handled here. F64/I32 workloads fall through to
 * BLAS or the CPU loop in nml.c.
 */

#include "nml_tensor.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

static id<MTLDevice>       _nml_mtl_device = nil;
static id<MTLCommandQueue> _nml_mtl_queue  = nil;

static void nml_metal_init(void) {
    if (!_nml_mtl_device) {
        _nml_mtl_device = MTLCreateSystemDefaultDevice();
        if (_nml_mtl_device)
            _nml_mtl_queue = [_nml_mtl_device newCommandQueue];
    }
}

/*
 * nml_backend_metal_matmul — row-major float32 matmul via Metal Performance Shaders
 *
 * dest: pre-allocated output tensor [m x n], NML_F32
 * a:    input tensor [m x k], NML_F32
 * b:    input tensor [k x n], NML_F32
 *
 * Returns 0 on success, -1 if Metal is unavailable or buffer allocation fails.
 * Caller (tensor_matmul in nml.c) falls through to BLAS/CPU on non-zero return.
 */
int nml_backend_metal_matmul(Tensor *dest, const Tensor *a, const Tensor *b,
                              int m, int k, int n) {
    nml_metal_init();
    if (!_nml_mtl_device || !_nml_mtl_queue) return -1;

    size_t a_bytes = (size_t)m * k * sizeof(float);
    size_t b_bytes = (size_t)k * n * sizeof(float);
    size_t c_bytes = (size_t)m * n * sizeof(float);

    id<MTLBuffer> buf_a = [_nml_mtl_device newBufferWithLength:a_bytes
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_b = [_nml_mtl_device newBufferWithLength:b_bytes
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_c = [_nml_mtl_device newBufferWithLength:c_bytes
                                                       options:MTLResourceStorageModeShared];
    if (!buf_a || !buf_b || !buf_c) return -1;

    float *pa = (float *)[buf_a contents];
    float *pb = (float *)[buf_b contents];
    for (int i = 0; i < m * k; i++) pa[i] = a->data.f32[i];
    for (int i = 0; i < k * n; i++) pb[i] = b->data.f32[i];

    MPSMatrixDescriptor *desc_a = [MPSMatrixDescriptor
        matrixDescriptorWithRows:m columns:k rowBytes:k * sizeof(float)
                        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *desc_b = [MPSMatrixDescriptor
        matrixDescriptorWithRows:k columns:n rowBytes:n * sizeof(float)
                        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *desc_c = [MPSMatrixDescriptor
        matrixDescriptorWithRows:m columns:n rowBytes:n * sizeof(float)
                        dataType:MPSDataTypeFloat32];

    MPSMatrix *mat_a = [[MPSMatrix alloc] initWithBuffer:buf_a descriptor:desc_a];
    MPSMatrix *mat_b = [[MPSMatrix alloc] initWithBuffer:buf_b descriptor:desc_b];
    MPSMatrix *mat_c = [[MPSMatrix alloc] initWithBuffer:buf_c descriptor:desc_c];

    MPSMatrixMultiplication *mmul = [[MPSMatrixMultiplication alloc]
        initWithDevice:_nml_mtl_device resultRows:m resultColumns:n interiorColumns:k];

    id<MTLCommandBuffer> cmd = [_nml_mtl_queue commandBuffer];
    [mmul encodeToCommandBuffer:cmd leftMatrix:mat_a rightMatrix:mat_b resultMatrix:mat_c];
    [cmd commit];
    [cmd waitUntilCompleted];

    float *pc = (float *)[buf_c contents];
    for (int i = 0; i < m * n; i++) dest->data.f32[i] = pc[i];
    return 0;
}
