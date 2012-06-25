#if !defined NO_CUDA

#include "MoleculesCUDA.h"
#include <cuda_runtime_api.h>

//#include <tracing.h>
//#include "MoleculesCUDA.tmh"

extern "C"
void
	CopyAtoms(float *, int);

extern "C"
void
	DoAtoms(float *, int, int, int, int, int);

// cons
	MoleculesCUDA::
	MoleculesCUDA() :
		Molecules(),
		ParallelCUDADriver()
{
	SetName("CUDA");
}

void
	MoleculesCUDA::
	Init(const bool) throw (...)
{
	CopyAtoms((float *)GetAtoms().GetData(), (int)GetCount());
}

void
	MoleculesCUDA::
	Execute() throw (...)
{
	size_t
		height = GetHeight(),
		width = GetWidth(),
		pitch,
		size = width * sizeof (float),
		count = GetCount();
	float *
		grid = 0;
	cudaMallocPitch((void **)&grid, &pitch, size, height);
	// Call the kernel
	DoAtoms(grid, (int)height, (int)width, GetThreads(), (int)pitch / sizeof (float), (int)count);
	cudaMemcpy2D(GetStore(), size, grid, pitch, size, height, cudaMemcpyDeviceToHost);
	cudaFree(grid);
}

#endif
