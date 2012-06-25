#if !defined NO_CUDA && !defined NO_HASKELL

#pragma once

#include <ParallelCUDADriver.h>
#include <ParallelHaskell.h>
#include "Molecules.h"

class MoleculesHaskell :
	public Molecules,
	public ParallelCUDADriver,
	public ParallelHaskell
{
public:
	// cons
		MoleculesHaskell();
	
protected:
	virtual void
		Init(const bool) throw (...);
	
	virtual void
		Execute() throw (...);
	
private:
	typedef void (__stdcall * HsFunc__piii_t)(void *, int, int, int);
	
	CUfunction
		m_kernel;
	
	HsFunc__piii_t
		GenerateMolecules;
};

#endif
