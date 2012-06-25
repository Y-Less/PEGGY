#pragma once

#include "ParallelSystem.h"

#include <Arrays/DataStore.h>

class ParallelCode :
	public virtual ParallelSystem
{
public:
	virtual DataStore &
		GetResult() const throw (...) = 0;
	
protected:
	virtual void
		Init(const bool) throw (...) = 0;
	
	virtual void
		Execute() throw (...) = 0;
	
	virtual void
		Close(const bool) throw (...) = 0;
};
