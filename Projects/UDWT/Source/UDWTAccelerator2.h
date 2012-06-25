#if !defined NO_ACCELERATOR_OPT

#pragma once

#include "UDWTAccelerator.h"

class UDWTAccelerator2 :
	public UDWTAccelerator
{
public:
	// cons
		UDWTAccelerator2(const ParallelAcceleratorType);
	
protected:
	virtual void
		Execute() throw (...);
};

#endif
