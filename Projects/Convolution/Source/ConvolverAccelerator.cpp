#if !defined NO_ACCELERATOR

#include "ConvolverAccelerator.h"
#include <iostream>

//#include <tracing.h>
//#include "ConvolverAccelerator.tmh"

using namespace ParallelArrays;

// cons
	ConvolverAccelerator::
	ConvolverAccelerator(
		const ParallelAcceleratorType
			type
		) :
		Convolver(),
		ParallelAccelerator(type),
		m_filter(0),
		m_data()
{
}

void
	ConvolverAccelerator::
	ConvInit() throw (...)
{
	// Get the data we want.
	m_data = dynamic_cast<FloatParallelArray &>(GetData().ToAcceleratorArray());
	m_filter = (float *)GetFilter().GetData();
	// The accelerator version can't set up here as it needs to zero the
	// accelerator array every itteration.
}

void
	ConvolverAccelerator::
	Execute() throw (...)
{
	Log("Set");
	// Code based on Accelerate v2 convolve example.
	size_t
		height = GetHeight(),
		width = GetWidth(),
		dims[] = {height, width};
	intptr_t
		shifts[] = {0, 0};
	int
		filterHalf = (int)GetRadius();
	// Create the target arrays locally and temporarilly.
	FloatParallelArray
		smoothX(0.0f, dims, 2),
		smoothY(0.0f, dims, 2);
	End("Set");
	Log("X");
	// Do the X dimension.
	for (int i = -filterHalf, j = 0; i <= filterHalf; ++i, ++j)
	{
		shifts[1] = i;
		smoothX += Shift(m_data, shifts, 2) * m_filter[j];
	}
	End("X");
	Log("Y");
	// Reset
	shifts[1] = 0;
	// Do the Y dimension.
	for (int i = -filterHalf, j = 0; i <= filterHalf; ++i, ++j)
	{
		shifts[0] = i;
		smoothY += Shift(smoothX, shifts, 2) * m_filter[j];
	}
	End("Y");
	Log("Copy");
	// Save the data.
	GetTarget().ToArray(smoothY, GetStore(), height, width, width * sizeof (float));
	End("Copy");
}

#endif
