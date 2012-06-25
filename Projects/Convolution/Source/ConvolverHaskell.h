#if !defined NO_CUDA && !defined NO_HASKELL

#pragma once

#include <ParallelCUDADriver.h>
#include <ParallelHaskell.h>
#include "Convolver.h"

class ConvolverHaskell :
	public Convolver,
	public ParallelCUDADriver,
	public ParallelHaskell
{
public:
	// cons
		ConvolverHaskell();
	
protected:
	virtual void
		ConvInit() throw (...);
	
	virtual void
		ConvExit() throw (...);
	
	virtual void
		Execute() throw (...);
	
private:
	typedef int (__stdcall * HsFunc_i_piii_t)(void *, int, int, int);
	
	//void
	//	Compile(LPCWSTR);
	
	float
		* m_data;
	
	CUdeviceptr
		m_smoothX,
		m_smoothY;
	
	CUfunction
		m_rowKernel,
		m_colKernel;
	
	HsFunc_i_piii_t
		GenerateConvolution,
		GenerateCachedConvolution;
};

#endif
