#include <cutil_inline.h>

#define WIDTH (8192)
#define HEIGHT (8192)

__global__ void kernel3(float *input0,float *result0)
{
	//int bidy = blockIdx.y;
	//int tidy = threadIdx.y;
	//int bidx = blockIdx.x * 32;
	//int tidx = threadIdx.x + bidx;
	int tidx = blockIdx.x * 32 + threadIdx.x;
	int tidy = blockIdx.y * 32 + threadIdx.y;
	int offset = tidy * HEIGHT;
	
	
	
	result0[tidx * WIDTH + tidy]
		= 
			input0[((tidx - 5 >= 0) ? ((tidx - 5 < HEIGHT) ? (tidx - 5) : (HEIGHT - 1)) : 0) + offset] * 3.5482936e-2 + 
			input0[((tidx - 4 >= 0) ? ((tidx - 4 < HEIGHT) ? (tidx - 4) : (HEIGHT - 1)) : 0) + offset] * 5.850147e-2 + 
			input0[((tidx - 3 >= 0) ? ((tidx - 3 < HEIGHT) ? (tidx - 3) : (HEIGHT - 1)) : 0) + offset] * 8.63096e-2 + 
			input0[((tidx - 2 >= 0) ? ((tidx - 2 < HEIGHT) ? (tidx - 2) : (HEIGHT - 1)) : 0) + offset] * 0.113945305 + 
			input0[((tidx - 1 >= 0) ? ((tidx - 1 < HEIGHT) ? (tidx - 1) : (HEIGHT - 1)) : 0) + offset] * 0.13461047 + 
			input0[((tidx     >= 0) ? ((tidx     < HEIGHT) ? (tidx    ) : (HEIGHT - 1)) : 0) + offset] * 0.14230047 + 
			input0[((tidx + 1 >= 0) ? ((tidx + 1 < HEIGHT) ? (tidx + 1) : (HEIGHT - 1)) : 0) + offset] * 0.13461047 + 
			input0[((tidx + 2 >= 0) ? ((tidx + 2 < HEIGHT) ? (tidx + 2) : (HEIGHT - 1)) : 0) + offset] * 0.113945305 + 
			input0[((tidx + 3 >= 0) ? ((tidx + 3 < HEIGHT) ? (tidx + 3) : (HEIGHT - 1)) : 0) + offset] * 8.63096e-2 + 
			input0[((tidx + 4 >= 0) ? ((tidx + 4 < HEIGHT) ? (tidx + 4) : (HEIGHT - 1)) : 0) + offset] * 5.850147e-2 + 
			input0[((tidx + 5 >= 0) ? ((tidx + 5 < HEIGHT) ? (tidx + 5) : (HEIGHT - 1)) : 0) + offset] * 3.5482936e-2;
			/*input0[max(0, min(HEIGHT - 1, tidx - 5)) + offset] * 3.5482936e-2 + 
			input0[max(0, min(HEIGHT - 1, tidx - 4)) + offset] * 5.850147e-2 + 
			input0[max(0, min(HEIGHT - 1, tidx - 3)) + offset] * 8.63096e-2 + 
			input0[max(0, min(HEIGHT - 1, tidx - 2)) + offset] * 0.113945305 + 
			input0[max(0, min(HEIGHT - 1, tidx - 1)) + offset] * 0.13461047 + 
			input0[max(0, min(HEIGHT - 1, tidx    )) + offset] * 0.14230047 + 
			input0[max(0, min(HEIGHT - 1, tidx + 1)) + offset] * 0.13461047 + 
			input0[max(0, min(HEIGHT - 1, tidx + 2)) + offset] * 0.113945305 + 
			input0[max(0, min(HEIGHT - 1, tidx + 3)) + offset] * 8.63096e-2 + 
			input0[max(0, min(HEIGHT - 1, tidx + 4)) + offset] * 5.850147e-2 + 
			input0[max(0, min(HEIGHT - 1, tidx + 5)) + offset] * 3.5482936e-2;*/
}

extern "C" void
	DoRows4(float * smoothX, float * res)
{
	dim3
		dimBlocks(HEIGHT / 32, WIDTH / 32),
		dimThreads(32, 32);
	kernel3<<<dimBlocks, dimThreads>>>(smoothX, res);
}



__global__ void kernel4(float *input0,float *result0)
{
	//int tidy = threadIdx.y;
	//int bidx = blockIdx.x;
	//int bidy = blockIdx.y;
	
	
	
	//result0[((threadIdx.x * (gridDim.y * blockDim.y)) + ((bidy * blockDim.y) + tidy))] = (((((((((((input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x - 5) + (bidx * 32)) < 4096) ? (threadIdx.x - 5) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 3.5482936e-2) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x - 4) + (bidx * 32)) < 4096) ? (threadIdx.x - 4) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 5.850147e-2)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x - 3) + (bidx * 32)) < 4096) ? (threadIdx.x - 3) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 8.63096e-2)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x - 2) + (bidx * 32)) < 4096) ? (threadIdx.x - 2) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 0.113945305)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x - 1) + (bidx * 32)) < 4096) ? (threadIdx.x - 1) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 0.13461047)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? (((threadIdx.x + (bidx * 32)) < 4096) ? threadIdx.x : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 0.14230047)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x + 1) + (bidx * 32)) < 4096) ? (threadIdx.x + 1) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 0.13461047)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x + 2) + (bidx * 32)) < 4096) ? (threadIdx.x + 2) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 0.113945305)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x + 3) + (bidx * 32)) < 4096) ? (threadIdx.x + 3) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 8.63096e-2)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x + 4) + (bidx * 32)) < 4096) ? (threadIdx.x + 4) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 5.850147e-2)) + (input0[((bidx * 32) + (((((int)threadIdx.x + (bidx * 32)) >= 0) ? ((((threadIdx.x + 5) + (bidx * 32)) < 4096) ? (threadIdx.x + 5) : (4095 - (bidx * 32))) : (0 - (bidx * 32))) + ((tidy + (bidy * blockDim.y)) * 4096)))] * 3.5482936e-2));
	
	
	int tidx = blockIdx.x * 32 + threadIdx.x;
	int tidy = blockIdx.y * 32 + threadIdx.y;
	int offset = tidy * WIDTH;
	
	
	
	result0[tidx * HEIGHT + tidy]
		= 
			/*input0[max(0, min(WIDTH - 1, tidx - 5)) + offset] * 3.5482936e-2 + 
			input0[max(0, min(WIDTH - 1, tidx - 4)) + offset] * 5.850147e-2 + 
			input0[max(0, min(WIDTH - 1, tidx - 3)) + offset] * 8.63096e-2 + 
			input0[max(0, min(WIDTH - 1, tidx - 2)) + offset] * 0.113945305 + 
			input0[max(0, min(WIDTH - 1, tidx - 1)) + offset] * 0.13461047 + 
			input0[max(0, min(WIDTH - 1, tidx    )) + offset] * 0.14230047 + 
			input0[max(0, min(WIDTH - 1, tidx + 1)) + offset] * 0.13461047 + 
			input0[max(0, min(WIDTH - 1, tidx + 2)) + offset] * 0.113945305 + 
			input0[max(0, min(WIDTH - 1, tidx + 3)) + offset] * 8.63096e-2 + 
			input0[max(0, min(WIDTH - 1, tidx + 4)) + offset] * 5.850147e-2 + 
			input0[max(0, min(WIDTH - 1, tidx + 5)) + offset] * 3.5482936e-2;*/
			input0[((tidx - 5 >= 0) ? ((tidx - 5 < WIDTH) ? (tidx - 5) : (WIDTH - 1)) : 0) + offset] * 3.5482936e-2 + 
			input0[((tidx - 4 >= 0) ? ((tidx - 4 < WIDTH) ? (tidx - 4) : (WIDTH - 1)) : 0) + offset] * 5.850147e-2 + 
			input0[((tidx - 3 >= 0) ? ((tidx - 3 < WIDTH) ? (tidx - 3) : (WIDTH - 1)) : 0) + offset] * 8.63096e-2 + 
			input0[((tidx - 2 >= 0) ? ((tidx - 2 < WIDTH) ? (tidx - 2) : (WIDTH - 1)) : 0) + offset] * 0.113945305 + 
			input0[((tidx - 1 >= 0) ? ((tidx - 1 < WIDTH) ? (tidx - 1) : (WIDTH - 1)) : 0) + offset] * 0.13461047 + 
			input0[((tidx     >= 0) ? ((tidx     < WIDTH) ? (tidx    ) : (WIDTH - 1)) : 0) + offset] * 0.14230047 + 
			input0[((tidx + 1 >= 0) ? ((tidx + 1 < WIDTH) ? (tidx + 1) : (WIDTH - 1)) : 0) + offset] * 0.13461047 + 
			input0[((tidx + 2 >= 0) ? ((tidx + 2 < WIDTH) ? (tidx + 2) : (WIDTH - 1)) : 0) + offset] * 0.113945305 + 
			input0[((tidx + 3 >= 0) ? ((tidx + 3 < WIDTH) ? (tidx + 3) : (WIDTH - 1)) : 0) + offset] * 8.63096e-2 + 
			input0[((tidx + 4 >= 0) ? ((tidx + 4 < WIDTH) ? (tidx + 4) : (WIDTH - 1)) : 0) + offset] * 5.850147e-2 + 
			input0[((tidx + 5 >= 0) ? ((tidx + 5 < WIDTH) ? (tidx + 5) : (WIDTH - 1)) : 0) + offset] * 3.5482936e-2;
}

extern "C" void
	DoCols4(float * smoothY, float * res)
{
	dim3
		dimBlocks(WIDTH / 32, HEIGHT / 32),
		dimThreads(32, 32);
	kernel4<<<dimBlocks, dimThreads>>>(smoothY, res);
}

