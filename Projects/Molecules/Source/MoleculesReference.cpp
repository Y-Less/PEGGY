#if !defined NO_REFERENCE

#include "MoleculesReference.h"

#include "Grid.h"

#include "math.h"

#include <Accelerator.h>

// cons
	MoleculesReference::
	MoleculesReference() :
		Molecules(),
		ParallelReference()
{
}

void
	MoleculesReference::
	Init(const bool) throw (...)
{
	m_atoms = &GetAtoms();
}

void
	MoleculesReference::
	Execute() throw (...)
{
	size_t
		height = GetHeight(),
		width = GetWidth(),
		count = GetCount();
	// Target array for the effects.
	float *
		d = GetStore();
	for (size_t i = 0; i != height; ++i)
	{
		for (size_t j = 0; j != width; ++j)
		{
			float
				outp = 0.0f;
			for (size_t k = 0; k != count; ++k)
			{
				// Separate the locations and charges of the atoms.
				ParallelArrays::Float4
					f = (*m_atoms)(k);
				float
					c = f.M128.m128_f32[3],
					// Calculate the offset from the atom to the grid point.
					x = (float)j - f.M128.m128_f32[0],
					y = (float)i - f.M128.m128_f32[1],
					z = 1.0f - f.M128.m128_f32[2],
					dist = (x * x) + (y * y) + (z * z);
				// Modify the charge's effect.  Somehow this sum doesn't seem
				// accurate given that the charge spreads out over a 3D area,
				// but it is the same as the one in the book.
				outp += c / sqrt(dist);
			}
			d[i * width + j] = outp;
		}
	}
}

#endif
