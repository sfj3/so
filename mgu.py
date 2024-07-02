import numpy as np
from Metal import *
from Foundation import *
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_metal_device():
    device = MTLCreateSystemDefaultDevice()
    logger.debug(f"Created Metal device: {device}")
    return device

def create_compute_pipeline(device, kernel_source):
    logger.debug("Creating compute pipeline")
    library = device.newLibraryWithSource_options_error_(kernel_source, None, None)[0]
    logger.debug(f"Created library: {library}")
    function = library.newFunctionWithName_("add_arrays")
    logger.debug(f"Created function: {function}")
    pipeline = device.newComputePipelineStateWithFunction_error_(function, None)[0]
    logger.debug(f"Created pipeline: {pipeline}")
    return pipeline

def run_compute(device, pipeline, input_a, input_b):
    logger.debug("Starting compute operation")
    command_queue = device.newCommandQueue()
    command_buffer = command_queue.commandBuffer()
    compute_encoder = command_buffer.computeCommandEncoder()

    compute_encoder.setComputePipelineState_(pipeline)
    
    logger.debug(f"Input A: {input_a}")
    logger.debug(f"Input B: {input_b}")
    
    buffer_a = device.newBufferWithBytes_length_options_(input_a, input_a.nbytes, MTLResourceStorageModeShared)
    buffer_b = device.newBufferWithBytes_length_options_(input_b, input_b.nbytes, MTLResourceStorageModeShared)
    buffer_result = device.newBufferWithLength_options_(input_a.nbytes, MTLResourceStorageModeShared)

    compute_encoder.setBuffer_offset_atIndex_(buffer_a, 0, 0)
    compute_encoder.setBuffer_offset_atIndex_(buffer_b, 0, 1)
    compute_encoder.setBuffer_offset_atIndex_(buffer_result, 0, 2)

    threads_per_group = (32, 1, 1)
    thread_groups = ((len(input_a) + 31) // 32, 1, 1)
    logger.debug(f"Dispatching with thread groups: {thread_groups}, threads per group: {threads_per_group}")
    compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(thread_groups, threads_per_group)

    compute_encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    result_pointer = buffer_result.contents()
    logger.debug(f"Result pointer: {result_pointer}")
    logger.debug(f"Result pointer type: {type(result_pointer)}")
    result = 0
    # Try to access the buffer contents directly
    try:
        result = np.frombuffer(buffer_result.contents(), dtype=input_a.dtype, count=len(input_a))

    except Exception as e:
        logger.error(f"Error accessing buffer contents: {e}")
    
    return result

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

logger.debug("Starting main program")
device = create_metal_device()
pipeline = create_compute_pipeline(device, kernel_source)

a = np.array([1, 2, 3, 4], dtype=np.float32)
b = np.array([5, 6, 7, 8], dtype=np.float32)

result = run_compute(device, pipeline, a, b)
logger.debug(f"Final result: {result}")
print("Result:", result)