#if !defined NO_ACCELERATOR_OPT

#include "MoleculesAccelerator2.h"

using namespace ParallelArrays;

// cons
	MoleculesAccelerator2::
	MoleculesAccelerator2(
		const ParallelAcceleratorType
			type
		) :
		Molecules(),
		ParallelAccelerator(type)
{
	switch (type)
	{
		case PAT_DX9:
			SetName("DX9b");
			break;
		case PAT_X64:
			SetName("X64b");
			break;
	}
}

void
	MoleculesAccelerator2::
	Init(const bool) throw (...)
{
	// Get the set of atoms.
	m_atoms = &GetAtoms();
//	Grid
	m_g = new Grid(GetHeight(), GetWidth());
}

void
	MoleculesAccelerator2::
	Execute() throw (...)
{
	// Create an array of array indices.
	Float4ParallelArray
		grid = dynamic_cast<Float4ParallelArray &>(m_g->ToAcceleratorArray());
	size_t
		height = GetHeight(),
		width = GetWidth(),
		dims[] = {height, width},
		count = GetCount();
	// Target array for the effects.
	FloatParallelArray
		outp(0.0f, dims, 2);
	// Calculate the effect of every atom on every element of the grid.
	float
		x0 = 0,
		y0 = 0,
		z0 = 0,
		* data = (float *)m_atoms->GetData();
	FloatParallelArray
		running = SumComponents(grid * grid);
	for (size_t i = 0; i != count; ++i)
	{
		// This code is based on more distance check optimisations found at:
		// http://y-less.com/wordpress/?p=122
		// Note that the value precomputing mentioned there is this bit here
		// due to the way Accelerator works.
		// First precompute the constants for the atom gaps.
		const float
			x1 = *data++,
			y1 = *data++,
			z1 = *data++,
			charge = *data++,
			xm = x0 - x1,
			ym = y0 - y1,
			zm = z0 - z1,
			additive = (xm * xm + ym * ym + zm * zm) - (2 * (x0 * xm + y0 * ym + z0 * zm)) + (2 * zm);
		// These lines do the:
		//  
		//  Px(2(Ax - Bx))
		//  
		// And add all the sections together.
		Float4
			mul(2 * xm, 2 * ym, 0.0f, 0.0f);
		running += Add(SumComponents(grid * mul), additive);
		outp += Multiply(charge, Rsqrt(running));
		x0 = x1;
		y0 = y1;
		z0 = z1;
	}
	// Get the results.
	GetTarget().ToArray(outp, GetStore(), height, width, width * sizeof (float));
	m_g->CleanGPU();
}

void
	MoleculesAccelerator2::
	Close(
		const bool                     save)
	throw (...)
{
	// Clean up the data we don't want.
	GetAtoms().CleanGPU();
	CloseCommon(save);
	delete m_g;
	m_g = 0;
}

#endif
