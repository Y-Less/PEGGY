#pragma once

#include <Arrays/Data1DFloat4.h>

class Atoms :
	public Data1DFloat4
{
public:
	// cons
		Atoms(const size_t, const float, const float);
	
	// cons
		Atoms(const size_t, const float, const float, const float, const float);
	
	size_t
		GetCount() const;
	
private:
	void
		GenerateAtoms(const DataType, const void *, const size_t) throw (...);
	
	float
		m_x,
		m_y,
		m_z,
		m_c;
};
