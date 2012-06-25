#include <cutil_inline.h>

__global__ void kernel1(float *input0,float *result0)
{
	//int bidy = blockIdx.y;
	//int tidy = threadIdx.y;
	//int bidx = blockIdx.x * 32;
	//int tidx = threadIdx.x + bidx;
	int tidx = blockIdx.x * 32 + threadIdx.x;
	int tidy = blockIdx.y * 32 + threadIdx.y;
	int offset = tidy * 2048;
	
	
	
	result0[tidx * 4096 + tidy]
		= 
			input0[((tidx - 5 >= 0) ? ((tidx - 5 < 2048) ? (tidx - 5) : 2047) : 0) + offset] * 3.5482936e-2 + 
			input0[((tidx - 4 >= 0) ? ((tidx - 4 < 2048) ? (tidx - 4) : 2047) : 0) + offset] * 5.850147e-2 + 
			input0[((tidx - 3 >= 0) ? ((tidx - 3 < 2048) ? (tidx - 3) : 2047) : 0) + offset] * 8.63096e-2 + 
			input0[((tidx - 2 >= 0) ? ((tidx - 2 < 2048) ? (tidx - 2) : 2047) : 0) + offset] * 0.113945305 + 
			input0[((tidx - 1 >= 0) ? ((tidx - 1 < 2048) ? (tidx - 1) : 2047) : 0) + offset] * 0.13461047 + 
			input0[((tidx     >= 0) ? ((tidx     < 2048) ? (tidx    ) : 2047) : 0) + offset] * 0.14230047 + 
			input0[((tidx + 1 >= 0) ? ((tidx + 1 < 2048) ? (tidx + 1) : 2047) : 0) + offset] * 0.13461047 + 
			input0[((tidx + 2 >= 0) ? ((tidx + 2 < 2048) ? (tidx + 2) : 2047) : 0) + offset] * 0.113945305 + 
			input0[((tidx + 3 >= 0) ? ((tidx + 3 < 2048) ? (tidx + 3) : 2047) : 0) + offset] * 8.63096e-2 + 
			input0[((tidx + 4 >= 0) ? ((tidx + 4 < 2048) ? (tidx + 4) : 2047) : 0) + offset] * 5.850147e-2 + 
			input0[((tidx + 5 >= 0) ? ((tidx + 5 < 2048) ? (tidx + 5) : 2047) : 0) + offset] * 3.5482936e-2;
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
	//int tidy = threadIdx.y;
	//int bidx = blockIdx.x;
	//int bidy = blockIdx.y;
	
	
	
	//result0[((threadIdx.x * (gridDim.y * blockDim.y)) + ((bidy * blockDim.y) + tidy))] = (((((((((((input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x - 5) + (bidx * 32)) < 4096) ? (threadIdx.x - 5) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 3.5482936e-2) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x - 4) + (bidx * 32)) < 4096) ? (threadIdx.x - 4) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 5.850147e-2)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x - 3) + (bidx * 32)) < 4096) ? (threadIdx.x - 3) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 8.63096e-2)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x - 2) + (bidx * 32)) < 4096) ? (threadIdx.x - 2) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 0.113945305)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x - 1) + (bidx * 32)) < 4096) ? (threadIdx.x - 1) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 0.13461047)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? (((threadIdx.x + (bidx * 32)) < 4096) ? threadIdx.x : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 0.14230047)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x + 1) + (bidx * 32)) < 4096) ? (threadIdx.x + 1) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 0.13461047)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x + 2) + (bidx * 32)) < 4096) ? (threadIdx.x + 2) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 0.113945305)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x + 3) + (bidx * 32)) < 4096) ? (threadIdx.x + 3) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 8.63096e-2)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x + 4) + (bidx * 32)) < 4096) ? (threadIdx.x + 4) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 5.850147e-2)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x + 5) + (bidx * 32)) < 4096) ? (threadIdx.x + 5) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 3.5482936e-2));
	
	
	int tidx = blockIdx.x * 32 + threadIdx.x;
	int tidy = blockIdx.y * 32 + threadIdx.y;
	int offset = tidy * 4096;
	
	
	
	result0[tidx * 2048 + tidy]
		= 
			input0[((tidx - 5 >= 0) ? ((tidx - 5 < 4096) ? (tidx - 5) : 4095) : 0) + offset] * 3.5482936e-2 + 
			input0[((tidx - 4 >= 0) ? ((tidx - 4 < 4096) ? (tidx - 4) : 4095) : 0) + offset] * 5.850147e-2 + 
			input0[((tidx - 3 >= 0) ? ((tidx - 3 < 4096) ? (tidx - 3) : 4095) : 0) + offset] * 8.63096e-2 + 
			input0[((tidx - 2 >= 0) ? ((tidx - 2 < 4096) ? (tidx - 2) : 4095) : 0) + offset] * 0.113945305 + 
			input0[((tidx - 1 >= 0) ? ((tidx - 1 < 4096) ? (tidx - 1) : 4095) : 0) + offset] * 0.13461047 + 
			input0[((tidx     >= 0) ? ((tidx     < 4096) ? (tidx    ) : 4095) : 0) + offset] * 0.14230047 + 
			input0[((tidx + 1 >= 0) ? ((tidx + 1 < 4096) ? (tidx + 1) : 4095) : 0) + offset] * 0.13461047 + 
			input0[((tidx + 2 >= 0) ? ((tidx + 2 < 4096) ? (tidx + 2) : 4095) : 0) + offset] * 0.113945305 + 
			input0[((tidx + 3 >= 0) ? ((tidx + 3 < 4096) ? (tidx + 3) : 4095) : 0) + offset] * 8.63096e-2 + 
			input0[((tidx + 4 >= 0) ? ((tidx + 4 < 4096) ? (tidx + 4) : 4095) : 0) + offset] * 5.850147e-2 + 
			input0[((tidx + 5 >= 0) ? ((tidx + 5 < 4096) ? (tidx + 5) : 4095) : 0) + offset] * 3.5482936e-2;
}

extern "C" void
	DoCols3(float * smoothY, float * res)
{
	dim3
		dimBlocks(4096 / 32, 2048 / 32),
		dimThreads(32, 32);
	kernel2<<<dimBlocks, dimThreads>>>(smoothY, res);
}

