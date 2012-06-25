#if !defined NO_ACCELERATOR

#pragma once

#include <ParallelAccelerator.h>
#include "UDWT.h"

class UDWTAccelerator :
	public UDWT,
	public ParallelAccelerator
{
public:
	// cons
		UDWTAccelerator(const ParallelAcceleratorType);
	
protected:
	virtual void
		ConvInit() throw (...);
	
	virtual void
		ConvClose() throw (...);
	
	virtual void
		Execute() throw (...);
	
	ParallelArrays::FloatParallelArray
		m_data;
	
	float
		* m_forward1,
		* m_reverse1,
		* m_forward2,
		* m_reverse2,
		* m_dest,
		* m_lowPass,
		* m_highPass;
	
private:
	//ParallelArrays::FloatParallelArray
	//	HardTh(ParallelArrays::FloatParallelArray, float);
	
	ParallelArrays::FloatParallelArray
		MIRDWT(ParallelArrays::FloatParallelArray low, ParallelArrays::FloatParallelArray high);
	
	void
		MRDWT(ParallelArrays::FloatParallelArray, ParallelArrays::FloatParallelArray *, ParallelArrays::FloatParallelArray *, ParallelArrays::FloatParallelArray *);
};

#endif
