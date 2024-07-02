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

    threads_per_group = (4, 1, 1)  # Change this to match input size
    thread_groups = (1, 1, 1)
    logger.debug(f"Dispatching with thread groups: {thread_groups}, threads per group: {threads_per_group}")
    compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(thread_groups, threads_per_group)

    compute_encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    result_pointer = buffer_result.contents()
    logger.debug(f"Result pointer: {result_pointer}")
    
    # # Try multiple ways to read the result
    # try:
    #     result = np.frombuffer(buffer_result.contents(), dtype=np.float32, count=len(input_a))
    #     logger.debug(f"Result from np.frombuffer: {result}")
    # except Exception as e:
    #     logger.error(f"Error with np.frombuffer: {e}")

    # try:
    #     result = np.array([result_pointer[i] for i in range(len(input_a))], dtype=np.float32)
    #     logger.debug(f"Result from manual array creation: {result}")
    # except Exception as e:
    #     logger.error(f"Error with manual array creation: {e}")

    # try:
    #     result = buffer_result.contents().to_bytes(input_a.nbytes)
    #     result = np.frombuffer(result, dtype=np.float32)
    #     logger.debug(f"Result from to_bytes: {result}")
    # except Exception as e:
    #     logger.error(f"Error with to_bytes: {e}")
    print('3333')
    abd = [print(x) for x in result_pointer[:4]] 
    return abd

# ... rest of the code remains the same

kernel_source = """
#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(device const float* A,
                       device const float* B,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    if (index < 4) {  // Assuming input size is 4
        result[index] = A[index] + B[index];
    }
}
"""

logger.debug("Starting main program")
device = create_metal_device()
pipeline = create_compute_pipeline(device, kernel_source)

a = np.array([1, 3, 5, 7], dtype=np.float32)
b = np.array([9, 11, 13, 15], dtype=np.float32)

result = run_compute(device, pipeline, a, b)
logger.debug(f"Final result: {result}")
print("Result:", result)