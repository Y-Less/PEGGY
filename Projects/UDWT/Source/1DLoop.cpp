#pragma once

#include <Arrays/Data1DFloat.h>

// I should maybe integrate the 1D wrapping system in to the main generator.
// But I'm not going to at this point as it requires complex run-time analysis
// to determine the maximum and minimum shifts involved.
class Wrapped :
	public Data1DFloat
{
public:
	// cons
		Grid(const size_t, const size_t);
	
private:
	void
		GenerateGrid(const DataType, const void *, const size_t) throw (...);
	
	size_t
		m_x;
};
