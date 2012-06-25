#if !defined NO_CUDA

#include "ConvolverCUDA.h"
#include <cuda_runtime_api.h>

//#include <tracing.h>
//#include "ConvolverCUDA.tmh"

extern "C" void
	CopyFilter(float *, int);

extern "C" void
	DoRows(float *, float *, int, int, int, int, int);

extern "C" void
	DoCols(float *, float *, int, int, int, int, int);

// cons
	ConvolverCUDA::
	ConvolverCUDA() :
		Convolver(),
		ParallelCUDADriver()
{
	SetName("CUDA");
}

void
	ConvolverCUDA::
	ConvInit() throw (...)
{
	// Get the data we want.
	m_data = (float *)GetData().ToCUDAArray();
	// Set up the filter.
	CopyFilter((float *)GetFilter().GetData(), (int)GetRadius());
	// Get the temporary arrays.
	cudaError
		ret = cudaGetLastError();
	if (ret != cudaSuccess)
	{
		throw cudaGetErrorString(ret);
	}
	//printf("4");
	m_smoothX = (float *)GetSmoothX().ToCUDAArray(),
	//printf("4");
	m_smoothY = (float *)GetSmoothY().ToCUDAArray();
	//printf("4");
}

void
	ConvolverCUDA::
	Execute() throw (...)
{
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): Set");
	Log("Set");
	//cudaError
		//ret;
	size_t
		height = GetHeight(),
		width = GetWidth(),
		pitch = GetPitch(),
		size = width * sizeof (float),
		radius = GetRadius();
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): Set");
	End("Set");
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): X");
	Log("X");
	// Do the X dimension.
	DoRows(m_smoothX, m_data, (int)height, (int)width, GetThreads(), (int)pitch, (int)radius);
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): X");
	End("X");
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): Y");
	Log("Y");
	// Do the Y dimension.
	//std::cout << "threads: " << height << ", " << width << ", " << GetThreads() << ", " << pitch << ", " << radius << std::endl;
	DoCols(m_smoothY, m_smoothX, (int)height, (int)width, GetThreads(), (int)pitch, (int)radius);
	// Convert the data back.
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): Y");
	End("Y");
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): Copy");
	//cudaThreadSynchronize();
	Log("Copy");
	//cudaThreadSynchronize();
	cudaMemcpy2D(GetStore(), size, m_smoothY, pitch * sizeof (float), size, height, cudaMemcpyDeviceToHost);
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): Copy");
	End("Copy");
}

#endif
