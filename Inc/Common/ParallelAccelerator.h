#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT

#pragma once

#include "ParallelProcessor.h"

#include <Accelerator.h>

enum
	ParallelAcceleratorType
{
	PAT_DX9,
	PAT_DX10,
	PAT_DX11,
	PAT_X64,
	PAT_DIRECT,
	PAT_AC_C
};

class ParallelAccelerator :
	public ParallelProcessor
{
public:
	// cons
		ParallelAccelerator(const ParallelAcceleratorType);
	
protected:
	ParallelArrays::Target &
		GetTarget() const throw (...);
	
	virtual void
		HWInit(const bool) throw (...);
	
	virtual void
		HWClose(const bool) throw (...);
	
private:
	void
		SetTarget(ParallelArrays::Target *);
	
	void
		DestroyTarget();
	
	ParallelArrays::Target *
		m_target;
	
	const ParallelAcceleratorType
		m_type;
};

#endif
