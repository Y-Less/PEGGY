#if !defined NO_REFERENCE_OPT

#include "MoleculesOptimised.h"

#include "Grid.h"

#include "math.h"

#include <Accelerator.h>

// cons
	MoleculesOptimised::
	MoleculesOptimised() :
		Molecules(),
		ParallelReference()
{
	SetName("Opt ");
}

void
	MoleculesOptimised::
	Init(const bool) throw (...)
{
	m_atoms = &GetAtoms();
}

void
	MoleculesOptimised::
	Execute() throw (...)
/*{
	// This version is optimised accoring to:
	// http://y-less.com/wordpress/?p=117
	size_t
		height = GetHeight(),
		width = GetWidth(),
		count = GetCount();
	// Target array for the effects.
	float
		* d = GetStore(),
		* atoms = (float *)m_atoms->GetData(),
		* pres = new float [count],
		* cur = atoms;
	for (size_t k = 0; k != count; ++k)
	{
		// Precompute the atom squares.
		float
			temp;
		temp = *cur * *cur;
		++cur;
		temp += *cur * *cur;
		++cur;
		pres[k] = temp + *cur * *cur;
		cur += 2;
	}
	// Do the main loop.
	for (size_t i = 0; i != height; ++i)
	{
		// Do y^2 + z^2
		const float
			total = 1.0f * 1.0f + i * i;
		for (size_t j = 0; j != width; ++j)
		{
			// Do the full calculation for c2.
			const float
				local = total + j * j;
			float
				outp = 0.0f;
			cur = atoms;
			for (size_t k = 0; k != count; ++k)
			{
				// Calculate the offset from the atom to the grid point.
				const float
					x = (float)j * *cur++,
					y = (float)i * *cur++,
					z = *cur++, // * 1.0f
					dist = pres[k] + local - 2 * (x + y + z);
				// Modify the charge's effect.  Somehow this sum doesn't seem
				// accurate given that the charge spreads out over a 3D area,
				// but it is the same as the one in the book.
				outp += *cur++ / sqrt(dist);
			}
			d[j] = outp;
		}
		d += width;
	}
	delete [] pres;
}*/
{
	// This version is optimised accoring to:
	// http://y-less.com/wordpress/?p=117
	// The optimisations at:
	// http://y-less.com/wordpress/?p=122
	// Do not apply here as there are a lot of domain specific optimisations to
	// this code based on the fact that there are two moving points.
	size_t
		height = GetHeight(),
		width = GetWidth(),
		count = GetCount();
	// Target array for the effects.
	float
		* d = GetStore(),
		* atoms = (float *)m_atoms->GetData(),
		* pres = new float [count],
		* cur = atoms;
	for (size_t k = 0; k != count; ++k)
	{
		// Precompute the atom squares.
		float
			temp;
		temp = *cur * *cur;
		++cur;
		temp += *cur * *cur;
		++cur;
		pres[k] = temp + *cur * *cur;
		cur += 2;
	}
	// Do the main loop.
	for (size_t i = 0; i != height; ++i)
	{
		// Do y^2 + z^2
		const float
			total = 1.0f * 1.0f + i * i;
		cur = atoms;
		// Loop over atoms before rows.
		for (size_t k = 0; k != count; ++k)
		{
			// Precompute values constant for all items on this row as
			// functions of the current row and atom.  The distance calcluation
			// is:
			// 
			//  d = c1 + c2 - 2(x1x2 + y1y2 + z1z2)
			//  
			// Much of which is constant for a given row, if we expand we get:
			//  
			//  d = x1x1 + x2x2 + y1y1 + y2y2 + z1z1 + z2z2 - 2x1x2 - 2y1y2 - 2z1z2
			//  
			// We know y and z are constant for any given row, as is x2 for any
			// given atom, so that gives:
			//  
			//  d = (c1b + c2 - 2(y1y2 + z1z2)) + (x1x1 + x1x2b)
			//  d = row_const + x1(x1 + x2b)
			//  
			//  Where:
			//  c1b = c1 - x1x1
			//  x2b = -2x2
			//  
			// This optimisation is implemented in the code below.  Note that
			// currently z1 is 1, so that multiplication is dropped.
			const float
				x = -2 * *cur++,
				y = (float)i * *cur++,
				z = *cur++,
				c = *cur++,
				// This is a partial calculation of the optimised code.
				local = pres[k] + total - 2 * (y + z);
			for (size_t j = 0; j != width; ++j)
			{
				// Combination of old optimised and cached codes.
				d[j] += c / sqrt(local + j * (j + x));
			}
		}
		d += width;
	}
	delete [] pres;
}

#endif
