#include "Grid.h"

// Some member variables are used in the constructor callback.  This means we
// need to GUARANTEE that they will be assigned before the callback is called.
// Using standard initialisations lists, this may not happen.
#define PREASSIGN(m,n) ((int)((m = n) && false)) +

// cons
	Grid::
	Grid(
		const size_t                   height,
		const size_t                   width)
	:
		Data2DFloat4(
				PREASSIGN(m_x, width)
			height, width, GT_Custom, (DataStore::ds_callback)&Grid::GenerateGrid)
{
}

void
	Grid::
	GenerateGrid(
		const DataType                 ,
		const void *                   buffer,
		const size_t                   size)
	throw (...)
{
	float
		* data = (float *)buffer;
	for (size_t j = 0, y = 0; j != size * 4; ++y)
	{
		for (size_t x = 0; x != m_x; ++x)
		{
			data[j++] = (float)x;
			data[j++] = (float)y;
			data[j++] = 1.0f;
			data[j++] = 0.0f;
		}
	}
}
