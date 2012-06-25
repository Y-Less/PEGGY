#if !defined NO_REFERENCE_OPT

#pragma once

#include <ParallelReference.h>

#include "Molecules.h"

class MoleculesOptimised :
	public Molecules,
	public ParallelReference
{
public:
	// cons
		MoleculesOptimised();
	
protected:
	virtual void
		Init(const bool) throw (...);
	
	virtual void
		Execute() throw (...);
	
	Atoms *
		m_atoms;
};

#endif
