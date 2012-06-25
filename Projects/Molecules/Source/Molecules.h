#pragma once

#include <Templates/CommonProcess2D.h>
#include <Arrays/Data2DFloat.h>

#include "Atoms.h"

class Molecules :
	//public ParallelCode
	public CommonProcess2D<Data2DFloat, float>
{
public:
	// cons
		Molecules();
	
	virtual // dest
		~Molecules();
	
	void
		SetGridSize(const size_t, const size_t);
	
	void
		SetAtoms(Atoms &);
	
protected:
	size_t
		GetHeight() const;
	
	size_t
		GetWidth() const;
	
	Atoms &
		GetAtoms() const;
	
	size_t
		GetCount() const;
	
	virtual void
		Close(const bool) throw (...);
	
private:
	size_t
		m_height,
		m_width;
	
	Atoms
		* m_atoms;
};
