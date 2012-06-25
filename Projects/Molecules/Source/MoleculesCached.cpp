#if !defined NO_REFERENCE_OPT

#include "MoleculesCached.h"

#include "Grid.h"

#include "math.h"

#include <Accelerator.h>

// cons
	MoleculesCached::
	MoleculesCached() :
		Molecules(),
		ParallelReference()
{
	SetName("Cach");
}

void
	MoleculesCached::
	Init(const bool) throw (...)
{
	m_atoms = &GetAtoms();
}

void
	MoleculesCached::
	Execute() throw (...)
{
	size_t
		height = GetHeight(),
		width = GetWidth(),
		count = GetCount();
	// Target array for the effects.
	float
		* d = GetStore(),
		* atoms = (float *)m_atoms->GetData(),
		* cur;
	for (size_t i = 0; i != height; ++i)
	{
		// Here we have swapped the two inner loops so all grid points in a row
		// get updated for one atom together, before moving on to the next
		// atom.  This allows us, as in the CUDA code, to conglomerate a large
		// amount of processing.  This also reduces the number of calls in to
		// the atoms code.
		cur = atoms;
		for (size_t k = 0; k != count; ++k)
		{
			// Separate the locations and charges of the atoms.
			//ParallelArrays::Float4
			//	f = (*m_atoms)(k);
			const float
				// Calculate the offset from the atom to the grid point.
				ax = -*cur++,
				y = (float)i - *cur++,
				z = 1.0f - *cur++,
				dist = (y * y) + (z * z),
				c = *cur++;
			for (size_t j = 0; j != width; ++j)
			{
				const float
					x = (float)j + ax;
				// Modify the charge's effect.  Somehow this sum doesn't seem
				// accurate given that the charge spreads out over a 3D area,
				// but it is the same as the one in the book.
				d[j] += c / sqrt(dist + (x * x));
			}
		}
		d += width;
	}
}

#endif
