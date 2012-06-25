#if !defined NO_CUDA && !defined NO_HASKELL

#include "MoleculesHaskell.h"
#include <cuda_runtime_api.h>

#define CEILDIV(m,n) \
	(((m) + (n) - 1) / (n))

#define SHAPE_X 32
#define SHAPE_Y 32

// cons
	MoleculesHaskell::
	MoleculesHaskell() :
		Molecules(),
		ParallelCUDADriver(),
		GenerateMolecules(0)
{
	SetName("Hask");
}

void
	MoleculesHaskell::
	Init(const bool) throw (...)
{
	struct ParallelScript::NamedFunctions_s
		funcs[] =
		{
			{"generateMolecules@16", (void **)&GenerateMolecules},
			{0, 0}
		};
	ScriptInit(L"MoleculesHaskell.dll", funcs);
	GenerateMolecules((float *)GetAtoms().GetData(), (int)GetCount(), GetHeight(), GetWidth());
	CUmodule
		module = 0;
	TryCUDA(cuModuleLoad(&module, "MoleculesHaskell.ptx"));
	m_kernel = 0;
	TryCUDA(cuModuleGetFunction(&m_kernel, module, "DoAtoms"));
}

void
	MoleculesHaskell::
	Execute() throw (...)
{
	//printf("0");
	size_t
		height = GetHeight(),
		width = GetWidth(),
		pitch,
		size = width * sizeof (float);
	//printf("1");
	float *
		grid = 0;
	//printf("2");
	cudaMallocPitch((void **)&grid, &pitch, size, height);
	//printf("3");
	// Call the kernel
	void *
		params0[] =
		{
			(void *)&grid,
		};
	//printf("4");
	TryCUDA(cuLaunchKernel(m_kernel, CEILDIV((int)width, SHAPE_X), CEILDIV((int)height, SHAPE_Y), 1, SHAPE_X, SHAPE_Y, 1, 0, 0, params0, 0));
	//printf("5");
	cudaMemcpy2D(GetStore(), size, grid, pitch, size, height, cudaMemcpyDeviceToHost);
	//printf("6");
	cudaFree(grid);
	//printf("7");
}

#endif
