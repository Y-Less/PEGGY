#if !defined NO_CUDA || !defined NO_CUDA_OPT

#include "ParallelCUDADriver.h"
#include <cuda_runtime_api.h>

// cons
	ParallelCUDADriver::
	ParallelCUDADriver() :
		m_context(0),
		m_device(0),
		m_warpSize(0),
		m_threads(0)
{
	SetName("Driver");
}

// Manual gflops calculation.  May want a large destination for this.
// TODO: Add a "source" parameter to this function so that we can add extra
// debug information to see where the problem originated.
inline void
	TryCUDA(
		CUresult                        error)
	throw (...)
{
	//printf("%d\n", error);
	if (error != CUDA_SUCCESS)
	{
		// Throw the error instead of just accepting it.
		//throw cudaGetErrorString(cudaGetLastError());
		static char
			sError[12];
		sprintf(sError, "CUDA %d", error);
		//printf(sError);
		throw sError;
	}
}

// Manual gflops calculation.  May want a large destination for this.
ParallelCUDADriver::gflops_t
	ParallelCUDADriver::
	GetGflops(
		CUdevice                        device)
{
	int
		ret;
	TryCUDA(cuDeviceGetAttribute(&ret, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
	gflops_t
		gflops = ret;
	TryCUDA(cuDeviceGetAttribute(&ret, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
	gflops *= ret;
	TryCUDA(cuDeviceGetAttribute(&ret, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device));
	return gflops * ret;
}

// It seems that you can now mix the driver and runtime APIs - you couldn't when
// I first wrote this code, but that was a while ago now.
CUdevice
	ParallelCUDADriver::
	GetFastestCard()
{
	// I strongly suspect that "cutGetMaxGflopsDeviceId" does exactly the same
	// thing as this function and will still work as it has a prefix of "cut",
	// not "cuda" nor "cu", indicating that it is part of neither the runtime,
	// nor the driver, API; and is instead an additional function.  But this is
	// a good way to get more aquainted with the driver API, so I'll do it
	// anyway (and will waste apparently quite a lot of time writing this
	// comment (something I actually seem to do quite often, meh, comments are
	// always good (as are brackets))).
	int
		deviceCount;
	// Again, I wish I had the PAWN pre-processor instead of the C one so I
	// could write a macro like "TRY:" instead of "TRY()", just seems a litte
	// nicer IMHO.  Maybe I should actually look in to porting it to some sort
	// of stand-alone application which I can use instead of the default (and
	// add some extra features such as __VA_ARGS__ MAYBE) (when I have "free
	// time" (whatever that is)).
	TryCUDA(cuDeviceGetCount(&deviceCount));
	if (deviceCount == 0)
	{
		throw "No CUDA devices found";
	}
	ParallelCUDADriver::gflops_t
		speed = 0;
	CUdevice
		fastest;
	for (int devid = 0; devid != deviceCount; ++devid)
	{
		CUdevice
			device;
		TryCUDA(cuDeviceGet(&device, devid));
		ParallelCUDADriver::gflops_t
			cur = GetGflops(device);
		if (cur > speed)
		{
			speed = cur;
			fastest = devid;
		}
	}
	return fastest;
}

void
	ParallelCUDADriver::
	HWInit(const bool) throw (...)
{
	// This is the second version of the CUDA backend.  This version uses the
	// lower-level driver API instead of the higher-level runtime API.  To be
	// perfectly honest, I've not seen much of a difference in abstraction in
	// either of them yet, we'll see how much my view is influenced by this
	// excercise.
	// Must be called first, always.  What happens if it it called twice - who
	// knows?  Currently there are no flags.
	cuInit(0);
	// Get the fastest device available.
	m_device = GetFastestCard();
	// Set the flags to 0 for now.  Will spin while waiting for results if there
	// are not more threads with CUDA contexts than processors.
	// Note to self: If I do port this to CSP++, make sure to look in to using
	// "CU_CTX_SCHED_BLOCKING_SYNC" instead.
	// TODO: Look in to memory mapping.
	TryCUDA(cuCtxCreate(&m_context, 0, m_device));
	// Get the warp size of this device (probably 32).
	// Save the warp size for use later.  I think this method is actually EASIER
	// than using the runtime API.
	TryCUDA(cuDeviceGetAttribute(&m_warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, m_device));
	TryCUDA(cuDeviceGetAttribute(&m_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, m_device));
}

void
	ParallelCUDADriver::
	HWClose(const bool) throw (...)
{
	TryCUDA(cuCtxDestroy(m_context));
}

int
	ParallelCUDADriver::
	GetWarpSize() const
{
	return m_warpSize;
}

int
	ParallelCUDADriver::
	GetThreads() const
{
	return m_threads;
}

#endif
