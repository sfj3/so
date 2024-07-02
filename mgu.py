import numpy as np
from Metal import *
from Foundation import *

def create_metal_device():
    return MTLCreateSystemDefaultDevice()

def create_compute_pipeline(device, kernel_source):
    library = device.newLibraryWithSource_options_error_(kernel_source, None, None)[0]
    function = library.newFunctionWithName_("add_arrays")
    return device.newComputePipelineStateWithFunction_error_(function, None)[0]

def run_compute(device, pipeline, input_a, input_b):
    command_queue = device.newCommandQueue()
    command_buffer = command_queue.commandBuffer()
    compute_encoder = command_buffer.computeCommandEncoder()

    compute_encoder.setComputePipelineState_(pipeline)
    
    buffer_a = device.newBufferWithBytes_length_options_(input_a, input_a.nbytes, MTLResourceStorageModeShared)
    buffer_b = device.newBufferWithBytes_length_options_(input_b, input_b.nbytes, MTLResourceStorageModeShared)
    buffer_result = device.newBufferWithLength_options_(input_a.nbytes, MTLResourceStorageModeShared)

    compute_encoder.setBuffer_offset_atIndex_(buffer_a, 0, 0)
    compute_encoder.setBuffer_offset_atIndex_(buffer_b, 0, 1)
    compute_encoder.setBuffer_offset_atIndex_(buffer_result, 0, 2)

    threads_per_group = (32, 1, 1)
    thread_groups = ((len(input_a) + 31) // 32, 1, 1)
    compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(thread_groups, threads_per_group)

    compute_encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    result_pointer = buffer_result.contents().asarray(input_a.dtype, input_a.size)
    return np.frombuffer(result_pointer, dtype=input_a.dtype)

kernel_source = """
#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(device const float* A,
                       device const float* B,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = A[index] + B[index];
}
"""

device = create_metal_device()
pipeline = create_compute_pipeline(device, kernel_source)

a = np.array([1, 2, 3, 4], dtype=np.float32)
b = np.array([5, 6, 7, 8], dtype=np.float32)

result = run_compute(device, pipeline, a, b)
print("Result:", result)