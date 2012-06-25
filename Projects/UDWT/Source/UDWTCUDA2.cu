#include <cutil_inline.h>

// Set the global constant memory to something huge and just limit it in code.
__device__ __constant__ float
	gc_fFilter[99];

// CUDA values.  This is constant for now as all known CUDA implementations
// have a warp size of 32 and it makes all the calculations constant.
#define WARP_SIZE                      (32) //(warpSize)
#define HALF_WARP                      (WARP_SIZE / 2)

extern "C" void
	CopyFilter2(
		float *
			filter,
		int
			radius)
{
	cudaMemcpyToSymbol(gc_fFilter, filter, ((radius * 2) + 1) * sizeof (float));
}

__global__ void
	DoOneRow2(
		float * const
			smoothX,
		const float * const
			inputData,
		const int
			height,
		const int
			width,
		const int
			pitch,
		const int
			radius)
{
	// First load some data to do one section of the row.  This should be
	// greater than the max number of threads + twice the max filter radius.
	__shared__ float
		fRowData[1280];
	const float * const
		filter = &gc_fFilter[radius];
	// Set up all the local variables, these should be per block, not per
	// thread as they are mostly all block location dependent.  This code is
	// different to the original, it's more flexible in terms of the number of
	// running threads and has some optimisations to do less runtime processing
	// which was not a problem before as all the numbers were compile time
	// constants.
	const int
		dataStart     = blockIdx.x * blockDim.x, // Always valid               // DATA_START
		apronStart    = dataStart - radius,                                    // THEORETICAL_APRON_START
		apronClamp    = max(apronStart, 0),                                    // ACTUAL_APRON_START
		alignedStart  = apronStart & (-HALF_WARP), // No need for CEILDIV      // LOWEST_ALIGNED_READ
		dataEnd       = dataStart + blockDim.x,                                // THEORETICAL_DATA_END
		apronEnd      = dataEnd + radius,                                      // THEORETICAL_APRON_END
		dataEndClamp  = min(dataEnd, width),                                   // ACTUAL_DATA_END
		apronEndClamp = min(apronEnd, width),                                  // ACTUAL_APRON_END
		apronOffset   = apronStart & (HALF_WARP - 1),                          // DISTANCE_FROM_ALIGNED_START_TO_APRON_START
		dataY         = blockIdx.y * pitch,                                    // Y_COORDINATE_OF_ROW
		maxX          = dataY + width - 1;
	// Now the variables which are threadIdx dependent.
	int
		load = threadIdx.x + alignedStart,                                     // ADDRESS_THIS_THREAD_SHOULD_LOAD
		pos  = threadIdx.x - apronOffset;                                      // ADDRESS_TO_SAVE_THIS_TO
	while (load < apronEnd)
	{
		// These checks haven't really changed.
		if (load >= apronEndClamp)
		{
			// Too high.
			fRowData[pos] = inputData[maxX];
		}
		else if (load >= apronClamp)
		{
			// Just right.
			fRowData[pos] = inputData[dataY + load];
		}
		else if (load >= apronStart)
		{
			// Too low.
			fRowData[pos] = inputData[dataY];
		}
		load += blockDim.x;
		pos  += blockDim.x;
	}
	// All the data is now loaded, synchronise all the threads.
	__syncthreads();
	// Every running thread, assuming the data isn't clamped to less than the
	// number of threads, can now process data.  This code is optimised for
	// processing, not for loading.
	const int
		pixel = dataStart + threadIdx.x;
	if (pixel < dataEndClamp)
	{
		float * const
			dd = fRowData + threadIdx.x + radius;
		float
			total = dd[0] * filter[0];
		for (int i = 0; i != radius; )
		{
			// Take advantage of the symmetry of the filter to do more ops
			// per loop.
			++i;
			total += filter[i] * (dd[-i] + dd[i]);
		}
		smoothX[dataY + pixel] = total;
	}
}

#define CEILDIV(m,n) \
	(((m) + (n) - 1) / (n))

extern "C" void
	DoRows2(
		float *
			smoothX,
		float *
			data,
		int
			height,
	    int
			width,
		int
			threads,
		int
			pitch,
		int
			radius)
{
	dim3
		// Number of blocks to execute in.
		dimBlocks(CEILDIV(width, threads), height),
		// Number of threads per block.
		dimThreads(threads);//ROW_BLOCK_ALIGN + FILTER_RADIUS + ROW_BLOCK_WIDTH);
	DoOneRow2<<<dimBlocks, dimThreads>>>(smoothX, data, height, width, pitch, radius);
}

__global__ void
	DoOneCol2(
		float *
			smoothY,
		float *
			inputData,
		int
			height,
		int
			width,
		int
			pitch,
		int
			radius)
{
	// First load some data to do one section of the row.
	// This data is shared between all the CUDA kernels, and one float is
	// loaded per CUDA thread.
	__shared__ float
		fRowData[1280];
	const float * const
		filter = &gc_fFilter[radius];
	// Now get the area in which this block of threads operate.
	const int
		dataStart     = blockIdx.y * blockDim.y, // Always valid               // DATA_START
		apronStart    = dataStart - radius,                                    // THEORETICAL_APRON_START
		apronClamp    = max(apronStart, 0),                                    // ACTUAL_APRON_START
		dataEnd       = dataStart + blockDim.y,                                // THEORETICAL_DATA_END
		apronEnd      = dataEnd + radius,                                      // THEORETICAL_APRON_END
		dataEndClamp  = min(dataEnd, height),                                  // ACTUAL_DATA_END
		apronEndClamp = min(apronEnd, height),                                 // ACTUAL_APRON_END
		//blockSize     = blockDim.y,
		//dataSize      = blockDim.y * pitch,
		maxY          = (height - 1) * pitch,
		lpitch        = blockDim.y + (2 * radius);                             // LOCAL_PITCH
	// Now the variables which are threadIdx dependent.
	const int
		dataX         = (blockIdx.x * blockDim.x) + threadIdx.x;               // X_COORDINATE_OF_COL
	int
		load = threadIdx.y + apronStart,                                       // ADDRESS_THIS_THREAD_SHOULD_LOAD
		pos  = (threadIdx.x * lpitch) + threadIdx.y;                           // ADDRESS_TO_SAVE_THIS_TO
	// This code transposes the data in shared memory to make the operations
	// acting on it later much simpler and faster.
	while (load < apronEnd)
	{
		// These checks haven't really changed.
		if (load >= apronEndClamp)
		{
			// Too high.
			fRowData[pos] = inputData[dataX + maxY];
		}
		else if (load >= apronClamp)
		{
			// Just right.
			fRowData[pos] = inputData[dataX + load * pitch];
		}
		else if (load >= apronStart)
		{
			// Too low.
			fRowData[pos] = inputData[dataX];
		}
		load += blockDim.y;
		pos  += blockDim.y;
	}
	// All the data is now loaded, synchronise all the threads.
	__syncthreads();
	// Now we can do the calculation for one pixel.  This is mostly the same as
	// the code for the X convolution due to the transposition above, there is
	// just a slight tweak due to the fact that multiple columns are done at
	// once as opposed to only a single row.
	const int
		pixel = dataStart + threadIdx.y;
	if (pixel < dataEndClamp)
	{
		float * const
			dd = fRowData + (threadIdx.x * lpitch) + threadIdx.y + radius;
		float
			total = dd[0] * filter[0];
		for (int i = 0; i != radius; )
		{
			// Take advantage of the symmetry of the filter to do more ops
			// per loop.
			++i;
			total += filter[i] * (dd[-i] + dd[i]);
		}
		smoothY[dataX + (pixel * pitch)] = total;
	}
}

extern "C" void
	DoCols2(
		float *
			smoothY,
		float *
			data,
		int
			height,
	    int
			width,
		int
			threads,
		int
			pitch,
		int
			radius)
{
	dim3
		// Number of blocks to execute in.
		dimBlocks(CEILDIV(width, HALF_WARP), CEILDIV(height, (threads / HALF_WARP))),
		// Number of threads per block.
		dimThreads(HALF_WARP, threads / HALF_WARP);
    DoOneCol2<<<dimBlocks, dimThreads>>>(smoothY, data, height, width, pitch, radius);
}
