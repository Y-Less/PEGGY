#if !defined NO_CUDA || !defined NO_CUDA_OPT
#if 0
#include "ParallelCUDA.h"

// cons
	ParallelCUDA::
	ParallelCUDA() :
		m_warpSize(0),
		m_offset(0),
		m_multiProc(0),
		m_kernel(0)
{
	SetName("CUDA");
}

/*void
	ParallelCUDA::
	Find(
		char *
			kernel)
{
	
}

void
	ParallelCUDA::
	Push*/



void
	ParallelCUDA::
	HWInit(const bool) throw (...)
{
	// Get the fastest device available.
	int
		device = cutGetMaxGflopsDeviceId();
//	std::cout << "Device: " << device << std::endl;
	// Use the first device, compatible with the OpenCL code.
//	int
//		device = 0;
	cudaError
		err;
	// Use this device.
	err = cudaSetDevice(device);
	// Until I find how to close and restart a context, just accept this.
	// Update: You get much more control over contexts in the driver API, which
	// is now used for BOTH CUDA versions, even the runtime API version.
	if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
	{
		throw cudaGetErrorString(err);
	}
	// Get the warp size of this device (probably 32).
//	std::cout << "Middle" << std::endl;
	cudaDeviceProp
		deviceProp;
	err = cudaGetDeviceProperties(&deviceProp, device);
	if (err != cudaSuccess)
	{
		HWClose(false);
		throw cudaGetErrorString(err);
	}
	// Save the warp size for use later.
	m_warpSize = deviceProp.warpSize;
	m_threads = deviceProp.maxThreadsPerBlock;
	//std::cout << "Warp: " << m_warpSize << std::endl;
	//std::cout << "Threads: " << deviceProp.maxThreadsPerBlock << std::endl;
	//std::cout << "Shared: " << deviceProp.sharedMemPerBlock << std::endl;
	//std::cout << "Version: " << deviceProp.major << "." << deviceProp.minor << std::endl;
	//std::cout << "TD: " << deviceProp.maxThreadsDim[0] << "," << deviceProp.maxThreadsDim[1] << "," << deviceProp.maxThreadsDim[2] << std::endl;
	//std::cout << "BD: " << deviceProp.maxGridSize[0] << "," << deviceProp.maxGridSize[1] << "," << deviceProp.maxGridSize[2] << std::endl;
	//std::cout << "Warp: " << m_warpSize << std::endl;
	//cudaGetDevice(&device);
	//std::cout << "Device: " << device << std::endl;
}

void
	ParallelCUDA::
	HWClose(const bool) throw (...)
{
	cudaError
		err = cudaThreadExit();
	if (err != cudaSuccess)
	{
		throw cudaGetErrorString(err);
	}
}

int
	ParallelCUDA::
	GetWarpSize() const
{
	return m_warpSize;
}

int
	ParallelCUDA::
	GetThreads() const
{
	return m_threads;
}

#endif
#endif
