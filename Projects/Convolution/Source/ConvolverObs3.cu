#include <cutil_inline.h>

__global__ void kernel1(float *input0,float *result0)
{
	int tidy = threadIdx.y;
	int bidx = blockIdx.x;
	int bidy = blockIdx.y;
	
	
	// Errors to correct:
	// 1) The first comparison is wrong - it should be row dependent.  Fixed.
	// 2) I don't think the destination is correct.  Maybe not.
	//result0[bidx * 32 + threadIdx.x * gridDim.y * blockDim.y + bidy * blockDim.y + tidy - bidx * 32]
	// The line above cancels to:
	result0[threadIdx.x * gridDim.y * blockDim.y + bidy * blockDim.y + tidy]
		=
			input0[(bidx * 32) + ((threadIdx.x + bidx * 32 >= 0) ? (((threadIdx.x - 5) + bidx * 32 < 2048) ? (threadIdx.x - 5) : (2047 - bidx * 32)) : (0 - bidx * 32)) + (tidy + bidy * blockDim.y) * 2048] * 3.5482936e-2 +
			input0[(bidx * 32) + ((threadIdx.x + bidx * 32 >= 0) ? (((threadIdx.x - 4) + bidx * 32 < 2048) ? (threadIdx.x - 4) : (2047 - bidx * 32)) : (0 - bidx * 32)) + (tidy + bidy * blockDim.y) * 2048] * 5.850147e-2 +
			input0[(bidx * 32) + ((threadIdx.x + bidx * 32 >= 0) ? (((threadIdx.x - 3) + bidx * 32 < 2048) ? (threadIdx.x - 3) : (2047 - bidx * 32)) : (0 - bidx * 32)) + (tidy + bidy * blockDim.y) * 2048] * 8.63096e-2 +
			input0[(bidx * 32) + ((threadIdx.x + bidx * 32 >= 0) ? (((threadIdx.x - 2) + bidx * 32 < 2048) ? (threadIdx.x - 2) : (2047 - bidx * 32)) : (0 - bidx * 32)) + (tidy + bidy * blockDim.y) * 2048] * 0.113945305 +
			input0[(bidx * 32) + ((threadIdx.x + bidx * 32 >= 0) ? (((threadIdx.x - 1) + bidx * 32 < 2048) ? (threadIdx.x - 1) : (2047 - bidx * 32)) : (0 - bidx * 32)) + (tidy + bidy * blockDim.y) * 2048] * 0.13461047 +
			input0[(bidx * 32) + ((threadIdx.x + bidx * 32 >= 0) ? (((threadIdx.x    ) + bidx * 32 < 2048) ? (threadIdx.x    ) : (2047 - bidx * 32)) : (0 - bidx * 32)) + (tidy + bidy * blockDim.y) * 2048] * 0.14230047 +
			input0[(bidx * 32) + ((threadIdx.x + bidx * 32 >= 0) ? (((threadIdx.x + 1) + bidx * 32 < 2048) ? (threadIdx.x + 1) : (2047 - bidx * 32)) : (0 - bidx * 32)) + (tidy + bidy * blockDim.y) * 2048] * 0.13461047 +
			input0[(bidx * 32) + ((threadIdx.x + bidx * 32 >= 0) ? (((threadIdx.x + 2) + bidx * 32 < 2048) ? (threadIdx.x + 2) : (2047 - bidx * 32)) : (0 - bidx * 32)) + (tidy + bidy * blockDim.y) * 2048] * 0.113945305 +
			input0[(bidx * 32) + ((threadIdx.x + bidx * 32 >= 0) ? (((threadIdx.x + 3) + bidx * 32 < 2048) ? (threadIdx.x + 3) : (2047 - bidx * 32)) : (0 - bidx * 32)) + (tidy + bidy * blockDim.y) * 2048] * 8.63096e-2 +
			input0[(bidx * 32) + ((threadIdx.x + bidx * 32 >= 0) ? (((threadIdx.x + 4) + bidx * 32 < 2048) ? (threadIdx.x + 4) : (2047 - bidx * 32)) : (0 - bidx * 32)) + (tidy + bidy * blockDim.y) * 2048] * 5.850147e-2 +
			input0[(bidx * 32) + ((threadIdx.x + bidx * 32 >= 0) ? (((threadIdx.x + 5) + bidx * 32 < 2048) ? (threadIdx.x + 5) : (2047 - bidx * 32)) : (0 - bidx * 32)) + (tidy + bidy * blockDim.y) * 2048] * 3.5482936e-2;
  
}

extern "C" void
  DoRows3(float * smoothX, float * res)
{
	dim3
		dimBlocks(2048 / 32, 4096 / 32),
		dimThreads(32, 32);
	kernel1<<<dimBlocks, dimThreads>>>(smoothX, res);
}



__global__ void kernel2(float *input0,float *result0)
{
	int tidy = threadIdx.y;
	int bidx = blockIdx.x;
	int bidy = blockIdx.y;
	
	
	
	result0[((bidx * 32) + (((threadIdx.x * (gridDim.y * blockDim.y)) + ((bidy * blockDim.y) + tidy)) - (bidx * 32)))]
		= (((((((((((input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x - 5) + (bidx * 32)) < 4096) ? (threadIdx.x - 5) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 3.5482936e-2) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x - 4) + (bidx * 32)) < 4096) ? (threadIdx.x - 4) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 5.850147e-2)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x - 3) + (bidx * 32)) < 4096) ? (threadIdx.x - 3) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 8.63096e-2)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x - 2) + (bidx * 32)) < 4096) ? (threadIdx.x - 2) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 0.113945305)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x - 1) + (bidx * 32)) < 4096) ? (threadIdx.x - 1) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 0.13461047)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? (((threadIdx.x + (bidx * 32)) < 4096) ? threadIdx.x : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 0.14230047)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x + 1) + (bidx * 32)) < 4096) ? (threadIdx.x + 1) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 0.13461047)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x + 2) + (bidx * 32)) < 4096) ? (threadIdx.x + 2) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 0.113945305)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x + 3) + (bidx * 32)) < 4096) ? (threadIdx.x + 3) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 8.63096e-2)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x + 4) + (bidx * 32)) < 4096) ? (threadIdx.x + 4) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 5.850147e-2)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x + 5) + (bidx * 32)) < 4096) ? (threadIdx.x + 5) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 3.5482936e-2));
  
}

extern "C" void
  DoCols3(float * smoothY, float * res)
{
	dim3
		dimBlocks(4096 / 32, 2048 / 32),
		dimThreads(32, 32);
	kernel2<<<dimBlocks, dimThreads>>>(smoothY, res);
}

