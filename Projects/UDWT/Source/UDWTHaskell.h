#if !defined NO_CUDA && !defined NO_HASKELL

#pragma once

#include <ParallelCUDADriver.h>
#include <ParallelHaskell.h>
#include "UDWT.h"

class UDWTHaskell :
	public UDWT,
	public ParallelCUDADriver,
	public ParallelHaskell
{
public:
	// cons
		UDWTHaskell();
	
protected:
	virtual void
		ConvInit() throw (...);
	
	virtual void
		ConvClose() throw (...);
	
	virtual void
		Execute() throw (...);
	
private:
	typedef void (__stdcall * HsFunc__ppppi_t)(void *, void *, void *, void *, int);
	typedef int (__stdcall * HsFunc_i_ii_t)(int, int);
	
	//void
	//	Compile(LPCWSTR);
	
	float
		* m_data,
		* m_lowPassC,
		* m_highPassC,
		* m_lowPass,
		* m_highPass;
	
	CUfunction
		m_lowKernel,
		m_combineKernel;
	
	CUfunction *
		m_highKernel;
	
	HsFunc_i_ii_t
		GenerateUDWT;
	
	HsFunc__ppppi_t
		SetFilters;
};

#endif
