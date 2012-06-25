#if !defined NO_REFERENCE_OPT

#pragma once

#include "ConvolverReference.h"

class ConvolverOptimised :
	public ConvolverReference
{
public:
	// cons
		ConvolverOptimised();
	
protected:
	virtual void
		Execute() throw (...);
};

#endif
