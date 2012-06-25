#pragma once

#include <Arrays/Data2DFloat4.h>

class Grid :
	public Data2DFloat4
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
