#if !defined NO_REFERENCE_OPT

#pragma once

#include "ConvolverReference.h"

class ConvolverCached :
	public ConvolverReference
{
public:
	// cons
		ConvolverCached();
	
protected:
	virtual void
		Execute() throw (...);
};

#endif
