#pragma once

#include "ParallelSystem.h"

class ParallelProcessor :
	public virtual ParallelSystem
{
protected:
	virtual void
		HWInit(const bool) throw (...) = 0;
	
	virtual void
		HWClose(const bool) throw (...) = 0;
};
