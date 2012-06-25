#if !defined NO_REFERENCE

#include "ConvolverReference.h"

//#include <tracing.h>
//#include "ConvolverReference.tmh"

// cons
	ConvolverReference::
	ConvolverReference() :
		Convolver(),
		ParallelReference()
{
}

void
	ConvolverReference::
	ConvInit() throw (...)
{
	// Get the data we want.
	m_data = (float *)GetData().GetData();
	m_filter = (float *)GetFilter().GetData();
	m_smoothX = (float *)GetSmoothX().GetData();
	m_smoothY = (float *)GetSmoothY().GetData();
}

const inline int
	ConvolverReference::
	Clamp(
		const int
			x,
		const int
			min,
		const int
			max) 
{
		return (x < min) ? min : (x > max) ? max : x;
}

void
	ConvolverReference::
	Execute() throw (...)
{
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): Set");
	Log("Set");
	// Again based on the naive software implementation from Accelerate.
	size_t
		height = GetHeight(),
		width = GetWidth();
	int
		hm1 = (int)height - 1,
		wm1 = (int)width - 1,
		filterHalf = (int)GetRadius();
	// Do the X convolution
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): Set");
	End("Set");
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): X");
	Log("X");
	for (size_t i = 0; i != height; ++i)
	{
		for (size_t j = 0; j != width; ++j)
		{
			float
				sum = 0;
			for (int u = -filterHalf, p = 0; u <= filterHalf; ++u, ++p)
			{
				int
					I = Clamp((int)i + u, 0, hm1);
				sum += m_data[I * width + j] * m_filter[p];
			}
			m_smoothX[i * width + j] = sum;
		}
	}
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): X");
	End("X");
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): Y");
	Log("Y");
	// Do the Y convolution
	for (size_t i = 0; i != height; ++i)
	{
		for (size_t j = 0; j != width; ++j)
		{
			float
				sum = 0;
			for (int u = -filterHalf, p = 0; u <= filterHalf; ++u, ++p)
			{
				int
					J = Clamp((int)j + u, 0, wm1);
				sum += m_smoothX[i * width + J] * m_filter[p];
			}
			m_smoothY[i * width + j] = sum;
		}
	}
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): Y");
	End("Y");
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): Copy");
	Log("Copy");
	// Now store the data back to main memory.
	memcpy(GetStore(), m_smoothY, height * width * sizeof (float));
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): Copy");
	End("Copy");
}

#endif
