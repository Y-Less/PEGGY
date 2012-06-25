#if !defined NO_ACCELERATOR_OPT

#pragma once

#include "ConvolverAccelerator.h"

class ConvolverAccelerator2 :
	public ConvolverAccelerator
{
public:
	// cons
		ConvolverAccelerator2(const ParallelAcceleratorType);
	
protected:
	virtual void
		Execute() throw (...);
};

#endif
