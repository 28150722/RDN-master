#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <math.h>
using namespace tensorflow;

__global__ void ExtractPatchesKernel(typename TTypes<float, 4>::ConstTensor input_images,
                                  	 typename TTypes<float, 3>::ConstTensor lms,
                                  	 typename TTypes<float, 5>::Tensor output_patches, 
	                                 const int  batch_size, 
	                                 const int  num_patches,
	                                 const int  input_height, 
	                                 const int  input_width,  
	                                 const int  patch_height, 
	                                 const int  patch_width, 
	                                 const int  depth){
	if(blockIdx.x < batch_size){
		if(threadIdx.x < num_patches){
			//i = blockIdx.x, n = threadIdx.x
			const float offset_y = lms(blockIdx.x, threadIdx.x, 0);
			const float offset_x = lms(blockIdx.x, threadIdx.x, 1);
			for(int source_x=offset_x-patch_width/2, target_x=0;
                target_x < patch_width;
                ++source_x, ++target_x) {
              for(int source_y=offset_y-patch_height/2, target_y=0;
                target_y < patch_height;
                ++source_y, ++target_y) {
                if (source_x > 0 && source_x < input_width && source_y > 0 && source_y < input_height) {
                	for (int c = 0; c < depth; ++c) {
                		output_patches(blockIdx.x, threadIdx.x, target_y, target_x, c) = input_images(blockIdx.x, source_y, source_x, c);
              		}
            	}
          	  }
        	}
		}
	}
}

void ExtractPatchesKernelLauncher(typename TTypes<float, 4>::ConstTensor input_images,
                                  typename TTypes<float, 3>::ConstTensor lms,
                                  typename TTypes<float, 5>::Tensor output_patches,
                                  const int  batch_size, 
                                  const int  num_patches,
                                  const int  input_height, 
                                  const int  input_width,  
                                  const int  patch_height, 
                                  const int  patch_width, 
                                  const int  depth){
	int BlockSize = 0;
	for(BlockSize = 0; ; BlockSize ++){
		if(pow(2, BlockSize) >= batch_size){
			break;
		}
	}
	BlockSize = pow(2,BlockSize);
	int ThreadSize = 0;
	for(ThreadSize = 0; ; ThreadSize ++){
		if(pow(2, ThreadSize) >= num_patches){
			break;
		}
	}
	ThreadSize = pow(2,ThreadSize);
	ExtractPatchesKernel<<<BlockSize, ThreadSize>>>(input_images,
																	lms,
																	output_patches,
																	batch_size,
																	num_patches,
																	input_height,
																	input_width,
																	patch_height,
																	patch_width,
																	depth);
}

#endif
