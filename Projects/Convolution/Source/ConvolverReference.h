#if !defined NO_REFERENCE

#pragma once

#include <ParallelReference.h>
#include "Convolver.h"

//#define Log(n) (void)0
//#define End(n) (void)0

class ConvolverReference :
	public Convolver,
	public ParallelReference
{
public:
	// cons
		ConvolverReference();
	
protected:
	virtual void
		ConvInit() throw (...);
	
	virtual void
		Execute() throw (...);
	
	float
		* m_filter,
		* m_data,
		* m_smoothX,
		* m_smoothY;
	
private:
	static const inline int
		Clamp(const int, const int, const int);
};

#endif
