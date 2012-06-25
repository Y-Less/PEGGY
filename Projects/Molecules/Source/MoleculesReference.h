#if !defined NO_REFERENCE

#pragma once

#include <ParallelReference.h>

#include "Molecules.h"

class MoleculesReference :
	public Molecules,
	public ParallelReference
{
public:
	// cons
		MoleculesReference();
	
protected:
	virtual void
		Init(const bool) throw (...);
	
	virtual void
		Execute() throw (...);
	
	Atoms *
		m_atoms;
};

#endif
