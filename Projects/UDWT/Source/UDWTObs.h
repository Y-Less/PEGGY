#if !defined NO_OBS

#pragma once

#include <ParallelCUDA.h>
#include "UDWT.h"

class UDWTObs :
	public UDWT,
	public ParallelCUDA
{
public:
	// cons
		UDWTObs();
	
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
