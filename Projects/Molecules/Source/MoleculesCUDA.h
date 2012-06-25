#if !defined NO_CUDA

#pragma once

#include <ParallelCUDADriver.h>
#include "Molecules.h"

class MoleculesCUDA :
	public Molecules,
	public ParallelCUDADriver
{
public:
	// cons
		MoleculesCUDA();
	
protected:
	virtual void
		Init(const bool) throw (...);
	
	virtual void
		Execute() throw (...);
};

#endif
