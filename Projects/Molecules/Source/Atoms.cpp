#include "Atoms.h"

#include <Arrays\StaticData.h>

// Some member variables are used in the constructor callback.  This means we
// need to GUARANTEE that they will be assigned before the callback is called.
// Using standard initialisations lists, this may not happen.
#define PREASSIGN(m,n) ((int)((m = n) && false)) +

// cons
	Atoms::
	Atoms(
		const size_t                   count,
		const float                    charge,
		const float                    area
	) :
		Data1DFloat4(
				PREASSIGN(m_c, charge)PREASSIGN(m_x, area)PREASSIGN(m_y, area)PREASSIGN(m_z, area)
			count, GT_Custom, (DataStore::ds_callback)&Atoms::GenerateAtoms)
{
}

// cons
	Atoms::
	Atoms(
		const size_t                   count,
		const float                    charge,
		const float                    x,
		const float                    y,
		const float                    z
	) :
		Data1DFloat4(
				PREASSIGN(m_c, charge)PREASSIGN(m_x, x)PREASSIGN(m_y, y)PREASSIGN(m_z, z)
			count, GT_Custom, (DataStore::ds_callback)&Atoms::GenerateAtoms)
{
}

void
	Atoms::
	GenerateAtoms(
		const DataType                 ,
		const void *                   buffer,
		const size_t                   size
	) throw (...)
{
	/*float
		* data = (float *)buffer,
		rnd;
	srand(1066);
	for (size_t i = 0, j = -1; i != size; ++i)
	{
		// X
		rnd = rand() / ((float)RAND_MAX + 1);
		data[++j] = rnd * m_x;
		// Y
		rnd = rand() / ((float)RAND_MAX + 1);
		data[++j] = rnd * m_y;
		// Z
		rnd = rand() / ((float)RAND_MAX + 1);
		data[++j] = rnd * m_z;
		// Charge
		rnd = rand() / ((float)RAND_MAX + 1);
		data[++j] = rnd * m_c;
	}*/
	float
		* data = (float *)buffer;
	for (size_t i = 0, j = -1, k = -1; i != size; ++i)
	{
		// X
		data[++j] = gc_fPointData[++k] * m_x;
		// Y
		data[++j] = gc_fPointData[++k] * m_y;
		// Z
		data[++j] = gc_fPointData[++k] * m_z;
		// Charge
		data[++j] = gc_fPointData[++k] * m_c;
	}
}

size_t
	Atoms::
	GetCount() const
{
	return GetWidth();
}
