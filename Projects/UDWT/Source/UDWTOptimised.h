#if !defined NO_REFERENCE

#pragma once

#include <ParallelReference.h>
#include "UDWT.h"

//#define Log(n) (void)0
//#define End(n) (void)0

class UDWTOptimised :
	public UDWT,
	public ParallelReference
{
public:
	// cons
		UDWTOptimised();
	
protected:
	virtual void
		ConvInit() throw (...);
	
	virtual void
		ConvClose() throw (...);
	
	virtual void
		Execute() throw (...);
	
	float
		* m_data,
		* m_forward1,
		* m_reverse1,
		* m_forward2,
		* m_reverse2,
		* m_lowPass,
		* m_highPass;
	
private:
	void
		MRDWT();
	
	void
		MIRDWT();
};

#endif
