#if !defined NO_CUDA_OPT

#pragma once

#include <ParallelCUDA.h>
#include "UDWT.h"

class UDWTCUDA2 :
	public UDWT,
	public ParallelCUDA
{
public:
	// cons
		UDWTCUDA2();
	
protected:
	virtual void
		ConvInit() throw (...);
	
	virtual void
		Execute() throw (...);
	
private:
	float
		* m_data,
		* m_smoothX,
		* m_smoothY;
};

#endif
