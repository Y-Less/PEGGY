#if !defined NO_REFERENCE_OPT

#pragma once

#include <ParallelReference.h>

#include "Molecules.h"

class MoleculesCached :
	public Molecules,
	public ParallelReference
{
public:
	// cons
		MoleculesCached();
	
protected:
	virtual void
		Init(const bool) throw (...);
	
	virtual void
		Execute() throw (...);
	
	Atoms *
		m_atoms;
};

#endif
