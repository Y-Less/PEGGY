#if !defined NO_REFERENCE || !defined NO_REFERENCE_OPT

#pragma once

#include "ParallelProcessor.h"

class ParallelReference :
	public ParallelProcessor
{
public:
	// cons
		ParallelReference();
	
protected:
	virtual void
		HWInit(const bool) throw (...);
	
	virtual void
		HWClose(const bool) throw (...);
};

#endif
