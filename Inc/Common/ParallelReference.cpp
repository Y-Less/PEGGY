#if !defined NO_REFERENCE || !defined NO_REFERENCE_OPT

#include "ParallelReference.h"

// cons
	ParallelReference::
	ParallelReference()
{
	// This class really does do nothing...
	SetName("Ref ");
}

void
	ParallelReference::
	HWInit(const bool) throw (...)
{
}

void
	ParallelReference::
	HWClose(const bool) throw (...)
{
}

#endif
