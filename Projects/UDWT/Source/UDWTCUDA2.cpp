#if !defined NO_CUDA_OPT

#include "UDWTCUDA2.h"

extern "C" void
	CopyFilter2(float *, int);

extern "C" void
	DoRows2(float *, float *, int, int, int, int, int);

extern "C" void
	DoCols2(float *, float *, int, int, int, int, int);

// cons
	UDWTCUDA2::
	UDWTCUDA2() :
		UDWT(),
		ParallelCUDA()
{
	SetName("CUD2");
}

void
	UDWTCUDA2::
	ConvInit() throw (...)
{
	// Get the data we want.
	m_data = (float *)GetData().ToCUDAArray();
	// Set up the filter.
	CopyFilter2((float *)GetFilter().GetData(), (int)GetRadius());
	// Get the temporary arrays.
	m_smoothX = (float *)GetSmoothX().ToCUDAArray(),
	m_smoothY = (float *)GetSmoothY().ToCUDAArray();
}

void
	UDWTCUDA2::
	Execute() throw (...)
{
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): Set");
	Log("Set");
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
	DoRows2(m_smoothX, m_data, (int)height, (int)width, GetThreads(), (int)pitch, (int)radius);
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): X");
	End("X");
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): Y");
	Log("Y");
	// Do the Y dimension.
	DoCols2(m_smoothY, m_smoothX, (int)height, (int)width, GetThreads(), (int)pitch, (int)radius);
	// Convert the data back.
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): Y");
	End("Y");
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): Copy");
	Log("Copy");
	//cudaThreadSynchronize();
	cudaMemcpy2D(GetStore(), size, m_smoothY, pitch * sizeof (float), size, height, cudaMemcpyDeviceToHost);
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): Copy");
	End("Copy");
}

#endif
