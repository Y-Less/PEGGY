#if !defined NO_CUDA_DRIVER

#include "ConvolverCUDADriver.h"
#include <cuda_runtime_api.h>

#define CEILDIV(m,n) \
	(((m) + (n) - 1) / (n))

#define WARP_SIZE                      (32) //(warpSize)
#define HALF_WARP                      (WARP_SIZE / 2)

//#include <tracing.h>
//#include "ConvolverCUDA.tmh"

//extern "C" void
//	CopyFilter(float *, int);

//extern "C" void
//	DoRows(float *, float *, int, int, int, int, int);

//extern "C" void
//	DoCols(float *, float *, int, int, int, int, int);

// cons
	ConvolverCUDADriver::
	ConvolverCUDADriver() :
		Convolver(),
		ParallelCUDADriver()
{
}

void
	ConvolverCUDADriver::
	ConvInit() throw (...)
{
	//printf("Threads: %d", GetThreads());
	// Load the module.
	m_module = 0;
	//printf("1\n");
	CUresult
		r = cuModuleLoad(&m_module, "ConvolverCUDADriver.ptx");
	//if (r != CUDA_ERROR_INVALID_SOURCE)
	if (m_module == 0)
	{
		//printf("FAIL");
		// I don't know why the call returns that value, as the code loads and
		// runs perfectly.  It seems to be something to do with the compute
		// capability specified at run-time.
		TryCUDA(r);
	}
	// Get the two functions.
	//printf("2 %d\n", m_module);
	m_rowKernel = 0;
	TryCUDA(cuModuleGetFunction(&m_rowKernel, m_module, "DoOneRowDriver")); //"_Z14DoOneRowDriverPfPKfiiii")); //
	//printf("3\n");
	m_colKernel = 0;
	TryCUDA(cuModuleGetFunction(&m_colKernel, m_module, "DoOneColDriver")); //"_Z14DoOneColDriverPfS_iiii")); //
	//printf("4\n");
	// Get the data we want.
	m_data = (float *)GetData().ToCUDAArray();
	//printf("5\n");
	// Set up the filter.
	CUdeviceptr
		filter = 0;
	size_t
		size = 0;
	//printf("5\n");
	//printf("6\n");
	TryCUDA(cuModuleGetGlobal(&filter, &size, m_module, "gc_fFilter"));
	//printf("7\n");
	size_t
		len = ((int)GetRadius() * 2 + 1) * sizeof (float);
	//printf("6 %d %d\n", len, size);
	//if (len > size)
	//{
	//	throw "Filter too large";
	//}
	//printf("8\n");
	TryCUDA(cuMemcpyHtoD(filter, GetFilter().GetData(), len));
	//printf("9\n");
	//printf("7\n");
	// Get the temporary arrays.
	m_smoothX = (CUdeviceptr)GetSmoothX().ToCUDAArray(),
	//printf("10 %08x\n", m_smoothX);
	//printf("8\n");
	m_smoothY = (CUdeviceptr)GetSmoothY().ToCUDAArray();
	//printf("11 %08x\n", m_smoothY);
	//printf("9\n");
}

void
	ConvolverCUDADriver::
	ConvExit() throw (...)
{
	cuModuleUnload(m_module);
};

void
	ConvolverCUDADriver::
	Execute() throw (...)
{
	Log("Set");
	size_t
		height = GetHeight(),
		width = GetWidth(),
		pitch = GetPitch(),
		size = width * sizeof (float),
		radius = GetRadius();
	int
		threads = GetThreads();
	End("Set");
	Log("X");
	// Do the X dimension.
	// Do I want to use the deprecated (pre-4.0) execution control which I have
	// apparently used in the past, or the much cleaner 4.0-only version?  The
	// latter is nicer, but also requires CUDA 4.0 (do I care about that fact,
	// I'm pretty much the only person using this code, so I don't think so).
	// Seems like my decision to use the new API was wrong - it's quite
	// convoluted!  The documentation isn't even clear on what order the
	// parameters should appear in - backwards for the order they are generally
	// pushed in, or forwards just because forwards.  OK, having had the code
	// run and work more-or-less FIRST TIME (barring a few issues with
	// "ParallelCUDADriver"), it turns out these are forwards.
	void *
		params0[] =
		{
			(void *)&m_smoothX,
			(void *)&m_data,
			(void *)&height,
			(void *)&width,
			(void *)&pitch,
			(void *)&radius,
		};
	/*int
		params0[] =
		{
			(int)m_smoothX,
			(int)m_data,
			(int)height,
			(int)width,
			(int)pitch,
			(int)radius,*/
		/*{
			(int)radius,
			(int)pitch,
			(int)width,
			(int)height,
			(int)m_data,
			(int)m_smoothX,*/
		/*};
	void *
		extra0[] =
		{
			CU_LAUNCH_PARAM_BUFFER_POINTER, params0,
			CU_LAUNCH_PARAM_BUFFER_SIZE, (void *)(4 * 6),
			CU_LAUNCH_PARAM_END
		};*/
	//TryCUDA(cuLaunchKernel(m_rowKernel, CEILDIV((int)width, threads), (int)height, 1, threads, 1, 1, 0, 0, 0, extra0));
	TryCUDA(cuLaunchKernel(m_rowKernel, CEILDIV(width, threads), height, 1, threads, 1, 1, 0, 0, params0, 0));
	//DoRows(m_smoothX, m_data, (int)height, (int)width, GetThreads(), (int)pitch, (int)radius);
	End("X");
	Log("Y");
	// Do the Y dimension.
	//DoCols(m_smoothY, m_smoothX, (int)height, (int)width, GetThreads(), (int)pitch, (int)radius);
	void *
		params1[] =
		{
			(void *)&m_smoothY,
			(void *)&m_smoothX,
			(void *)&height,
			(void *)&width,
			(void *)&pitch,
			(void *)&radius,
		};
	/*int
		params1[] =
		{
			(int)m_smoothY,
			(int)m_smoothX,
			(int)height,
			(int)width,
			(int)pitch,
			(int)radius,*/
		/*{
			(int)radius,
			(int)pitch,
			(int)width,
			(int)height,
			(int)m_smoothX,
			(int)m_smoothY,*/
		/*};
	void *
		extra1[] =
		{
			CU_LAUNCH_PARAM_BUFFER_POINTER, params1,
			CU_LAUNCH_PARAM_BUFFER_SIZE, (void *)(4 * 6),
			CU_LAUNCH_PARAM_END
		};*/
	//TryCUDA(cuLaunchKernel(m_colKernel, CEILDIV((int)width, threads), (int)height, 1, threads, 1, 1, 0, 0, 0, extra1));
	TryCUDA(cuLaunchKernel(m_colKernel, CEILDIV(width, HALF_WARP), CEILDIV(height, (threads / HALF_WARP)), 1, HALF_WARP, threads / HALF_WARP, 1, 0, 0, params1, 0));
	// Convert the data back.
	End("Y");
	//cudaThreadSynchronize();
	Log("Copy");
	//cudaThreadSynchronize();
	//GetStore()[2] = 6.0;
	//printf("%f %f %f %f %f\n", GetStore()[0], GetStore()[1], GetStore()[2], GetStore()[3], GetStore()[4]);
	//printf("%d\n", cudaMemcpy2D(GetStore(), size, (void *)m_smoothY, pitch * sizeof (float), size, height, cudaMemcpyDeviceToHost));
	cudaMemcpy2D(GetStore(), size, (void *)m_smoothY, pitch * sizeof (float), size, height, cudaMemcpyDeviceToHost);
	// The most nasty bit so far of using the driver API.  Fortunately the
	// parameters actually seem to map quite nicely from the runtime function to
	// the driver structure.  The fact that I seem to have ended up with exactly
	// the same number of the same inputs, just arranged differently, is
	// somewhat reassuring.
	/*CUDA_MEMCPY2D
		memdata = {0};
	memdata.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	memdata.srcDevice = m_smoothY;
	memdata.srcPitch = pitch * sizeof (float);
	memdata.dstMemoryType = CU_MEMORYTYPE_HOST;
	memdata.dstHost = GetStore();
	memdata.dstPitch = size;
	memdata.WidthInBytes = size;
	memdata.Height = height;
	Sleep(0);
	TryCUDA(cuMemcpy2D(&memdata));*/
	End("Copy");
	//printf("%f %f %f %f %f\n", GetStore()[0], GetStore()[1], GetStore()[2], GetStore()[3], GetStore()[4]);
}

#endif
