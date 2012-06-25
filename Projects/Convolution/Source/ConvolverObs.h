#if !defined NO_OBS

#pragma once

#include <ParallelCUDA.h>
#include "Convolver.h"

class ConvolverObs :
	public Convolver,
	public ParallelCUDA
{
public:
	// cons
		ConvolverObs();
	
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
