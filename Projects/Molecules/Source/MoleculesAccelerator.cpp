#if !defined NO_ACCELERATOR

#include "MoleculesAccelerator.h"

using namespace ParallelArrays;

// cons
	MoleculesAccelerator::
	MoleculesAccelerator(
		const ParallelAcceleratorType
			type
		) :
		Molecules(),
		ParallelAccelerator(type)
{
}

void
	MoleculesAccelerator::
	Init(const bool) throw (...)
{
	// Get the set of atoms.
	m_atoms = &GetAtoms();
//	Grid
	m_g = new Grid(GetHeight(), GetWidth());
}

void
	MoleculesAccelerator::
	Execute() throw (...)
{
	// Create an array of array indices.
	//Grid
	//	g(GetHeight(), GetWidth());
	// Convert the value in "m_g" to a Float4ParallelArray.
	Float4ParallelArray
		grid = dynamic_cast<Float4ParallelArray &>(m_g->ToAcceleratorArray());
	size_t
		height = GetHeight(), // Get the height of the data.
		width = GetWidth(), // Get the width of the data.
		dims[] = {height, width},
		count = GetCount(); // Get the number of atoms.
	// Target array for the effects.
	FloatParallelArray
		outp(0.0f, dims, 2);
	// Calculate the effect of every atom on every element of the grid.
	for (size_t i = 0; i != count; ++i)
	{
		// Separate the locations and charges of the atoms.
		Float4
			f = (*m_atoms)(i);
		float
			c = f.M128.m128_f32[3];
		f.M128.m128_f32[3] = 0.0;
		// Calculate the offset from the current atom to all grid points.
		Float4ParallelArray
			diff = Subtract(grid, f);
		FloatParallelArray
			dist = Rsqrt(SumComponents(Multiply(diff, diff)));
		// Add the charge multiplied by the inverse of the distance.
		outp += Multiply(c, dist);
	}
	// Get the results.
	GetTarget().ToArray(outp, GetStore(), height, width, width * sizeof (float));
	m_g->CleanGPU();
}

void
	MoleculesAccelerator::
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
