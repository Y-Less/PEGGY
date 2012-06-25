#if !defined NO_OBS_OPT

#include "ConvolverObsOpt.h"

//#include <tracing.h>
//#include "ConvolverCUDA.tmh"

extern "C" void
	DoRows4(float *, float *);

extern "C" void
	DoCols4(float *, float *);

// cons
	ConvolverObsOpt::
	ConvolverObsOpt() :
		Convolver(),
		ParallelCUDA()
{
	SetName("Hand");
}

void
	ConvolverObsOpt::
	ConvInit() throw (...)
{
	// Get the data we want.
	m_data = (float *)GetData().ToCUDAArray();
	// Set up the filter.
	//CopyFilter((float *)GetFilter().GetData(), (int)GetRadius());
	// Get the temporary arrays.
	cudaError
		ret = cudaGetLastError();
	if (ret != cudaSuccess)
	{
		throw cudaGetErrorString(ret);
	}
	m_smoothX = (float *)GetSmoothX().ToCUDAArray(),
	m_smoothY = (float *)GetSmoothY().ToCUDAArray();
}

void
	ConvolverObsOpt::
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
	DoRows4(m_data, m_smoothX);
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): X");
	End("X");
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): Y");
	Log("Y");
	// Do the Y dimension.
	//std::cout << "threads: " << height << ", " << width << ", " << GetThreads() << ", " << pitch << ", " << radius << std::endl;
	DoCols4(m_smoothX, m_smoothY);
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
