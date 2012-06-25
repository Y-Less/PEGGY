#if !defined NO_OBS_OPT

#pragma once

#include <ParallelCUDA.h>
#include "Convolver.h"

class ConvolverObsOpt :
	public Convolver,
	public ParallelCUDA
{
public:
	// cons
		ConvolverObsOpt();
	
protected:
	virtual void
		ConvInit() throw (...);
	
	virtual void
		Execute() throw (...);
	
	float
		* m_data,
		* m_smoothX,
		* m_smoothY;
};

#endif
