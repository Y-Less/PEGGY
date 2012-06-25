#if !defined NO_CUDA

#pragma once

#include <ParallelCUDADriver.h>
#include "UDWT.h"

//#define Log(n) (void)0
//#define End(n) (void)0

class UDWTCUDA:
	public UDWT,
	public ParallelCUDADriver
{
public:
	// cons
		UDWTCUDA();
	
protected:
	virtual void
		ConvInit() throw (...);
	
	virtual void
		ConvClose() throw (...);
	
	virtual void
		Execute() throw (...);
	
	float
		* m_data,
		* m_lowPassC,
		* m_highPassC,
		* m_lowPass,
		* m_highPass;
	
private:
	//void
	//	MRDWT();
	
	void
		MIRDWT();
};

#endif
