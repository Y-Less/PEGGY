#if !defined NO_ACCELERATOR_OPT

#pragma once

#include <ParallelAccelerator.h>

#include "Molecules.h"
#include "Grid.h"

class MoleculesAccelerator2 :
	public Molecules,
	public ParallelAccelerator
{
public:
	// cons
		MoleculesAccelerator2(const ParallelAcceleratorType);
	
protected:
	virtual void
		Init(const bool) throw (...);
	
	virtual void
		Execute() throw (...);
	
	virtual void
		Close(const bool) throw (...);
	
	//ParallelArrays::Float4ParallelArray
	Atoms *
		m_atoms;
	
	Grid *
		m_g;
	
	//ParallelArrays::Float4ParallelArray
	//	m_grid;
};

#endif
