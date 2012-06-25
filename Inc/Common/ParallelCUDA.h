#if !defined NO_CUDA || !defined NO_CUDA_OPT
#if 0

#pragma once

#include "ParallelProcessor.h"

#include <cutil_inline.h>
//#include <cuda.h>
//#include <cutil.h>
//#include <cuda_runtime_api.h>

class ParallelCUDA :
	public ParallelProcessor
{
public:
	// cons
		ParallelCUDA();
	
protected:
	virtual void
		HWInit(const bool) throw (...);
	
	virtual void
		HWClose(const bool) throw (...);
	
	int
		GetWarpSize() const;
	
	int
		GetThreads() const;
	
private:
	int
		m_warpSize,
		m_multiProc,
		m_offset,
		m_threads;
	
	CUfunction
		m_kernel;
};

#endif
#endif
