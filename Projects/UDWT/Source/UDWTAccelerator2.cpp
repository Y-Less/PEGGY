#if !defined NO_ACCELERATOR_OPT

#include "UDWTAccelerator2.h"

using namespace ParallelArrays;

// cons
	UDWTAccelerator2::
	UDWTAccelerator2(
		const ParallelAcceleratorType
			type
		) :
		UDWTAccelerator(type)
{
	switch (type)
	{
		case PAT_DX9:
			SetName("DX9b");
			break;
		case PAT_X64:
			SetName("X64b");
			break;
	}
}

void
	UDWTAccelerator2::
	Execute() throw (...)
{
	// This version uses a modified version of the algorithm based on the fact
	// that the filter in use is symetrical, note that this code would not
	// work otherwise
	Log("Set");
	size_t
		height = GetHeight(),
		width = GetWidth(),
		dims[] = {height, width};
	intptr_t
		shifts0[] = {0, 0},
		shifts1[] = {0, 0};
	int
		filterHalf = (int)GetRadius();
	float *
		filter = &m_filter[filterHalf];
	// Create the target arrays locally and temporarilly.
	FloatParallelArray
		smoothX = m_data * filter[0];
	End("Set");
	Log("X");
	// Do the X dimension.
	for (int i = 0; i != filterHalf; )
	{
		++i;
		shifts0[1] = -i;
		shifts1[1] = i;
		smoothX += (Shift(m_data, shifts0, 2) + Shift(m_data, shifts1, 2)) * filter[i];
	}
	End("X");
	Log("Y");
	// Reset
	shifts0[1] = 0;
	shifts1[1] = 0;
	FloatParallelArray
		smoothY = smoothX * filter[0];
	// Do the Y dimension.
	for (int i = 0; i != filterHalf; )
	{
		++i;
		shifts0[0] = -i;
		shifts1[0] = i;
		smoothY += (Shift(smoothX, shifts0, 2) + Shift(smoothX, shifts1, 2)) * filter[i];
	}
	End("Y");
	Log("Copy");
	// Save the data.
	GetTarget().ToArray(smoothY, GetStore(), height, width, width * sizeof (float));
	End("Copy");
}

#endif
