#if !defined NO_CUDA && !defined NO_HASKELL

#include "UDWTHaskell.h"
#include <cuda_runtime_api.h>

#include <stdio.h>

#define CEILDIV(m,n) \
	(((m) + (n) - 1) / (n))

#define SHAPE_X 1024
#define SHAPE_Y 1

// cons
	UDWTHaskell::
	UDWTHaskell() :
		UDWT(),
		ParallelCUDADriver(),
		ParallelHaskell(),
		SetFilters(0),
		GenerateUDWT(0),
		m_highKernel(0)
{
	SetName("Haskell");
}

void
	UDWTHaskell::
	ConvInit() throw (...)
{
	struct ParallelScript::NamedFunctions_s
		funcs[] =
		{
			{"generateUDWT@8", (void **)&GenerateUDWT},
			{"setFilters@20", (void **)&SetFilters},
			{0, 0}
		};
	ScriptInit(L"UDWTHaskell.dll", funcs);
	SetFilters((float *)GetForward1().GetData(), (float *)GetReverse1().GetData(), (float *)GetForward2().GetData(), (float *)GetReverse2().GetData(), GetRadius());
	int
		levels = GetLevel();
	if (GenerateUDWT(GetWidth(), levels))
	{
		throw "GenerateConvolution returned non-0";
	}
	m_highKernel = new CUfunction [levels];
	// Load the module.
	//printf("Thread: %d\n", GetThreads());
	CUmodule
		module = 0;
	//printf("1\n");
	TryCUDA(cuModuleLoad(&module, "UDWTHaskell.ptx"));
	// Get the two functions.
	//printf("2 %d\n", module);
	m_lowKernel = 0;
	TryCUDA(cuModuleGetFunction(&m_lowKernel, module, "MRDWT"));
	//printf("3\n");
	m_combineKernel = 0;
	TryCUDA(cuModuleGetFunction(&m_combineKernel, module, "MIRDWT"));
	char
		name[7] = "MRDWT\0";
	while (levels--)
	{
		m_highKernel[levels] = 0;
		name[5] = levels + '0';
		TryCUDA(cuModuleGetFunction(&m_highKernel[levels], module, name));
	}
	// Get the data we want.
	m_data = (float *)GetData().ToCUDAArray();
	m_lowPass  = (float *)GetLowPass().GetData();
	m_highPass = (float *)GetHighPass().GetData();
	m_lowPassC  = (float *)GetLowPass().ToCUDAArray();
	m_highPassC = (float *)GetHighPass().ToCUDAArray();
}

void
	Abs(float * data, size_t length);

void
	HardTh(float * data, size_t length, float threshold);

int
	FloatCompare(
		void const *
			a,
		void const *
			b);

void
	UDWTHaskell::
	Execute() throw (...)
{
	float
		TEMP_threshold = 0.5f;
	//MRDWT();
	//printf("1");
	int
		width = GetWidth(),
		size = width * sizeof (float),
		levels = GetLevel();
	void *
		params0[] =
		{
			(void *)&m_data, // SHOULD BE 0, BUT CAN'T BE.  UNUSED.
			(void *)&m_data,
			(void *)&m_lowPassC,
		};
	//printf("1");
	TryCUDA(cuLaunchKernel(m_lowKernel, CEILDIV((int)width, SHAPE_X), 1, 1, SHAPE_X, 1, 1, 0, 0, params0, 0));
	//printf("1");
	for (int ll = 0; ll != levels; ++ll)
	{
		float *
			ptr = m_highPassC + width * ll;
		//printf("2 %d", m_highKernel[ll]);
		params0[2] = (void *)&ptr;
		//printf("2 %d", params0[2]);
		TryCUDA(cuLaunchKernel(m_highKernel[ll], CEILDIV((int)width, SHAPE_X), 1, 1, SHAPE_X, 1, 1, 0, 0, params0, 0));
	}
	//printf("1");
	cudaMemcpy(GetStore(), m_lowPassC, size, cudaMemcpyDeviceToHost);
	//printf("1");
	cudaMemcpy(m_highPass, m_highPassC, levels * size, cudaMemcpyDeviceToHost);
	//printf("1");
	// Now do the common code.  I need to figure out the offset of this.  Maybe
	// I should have a "nothing" target that ONLY does this for timings.  In
	// fact I could reuse the redundant "Cached" target.
	size_t
		length = GetWidth();
	memcpy(m_lowPass, m_highPass, length * sizeof (float));
	Abs(m_lowPass, length);
	qsort(m_lowPass, length, sizeof (float), FloatCompare);
	float
		threshold;
	if (length & 1)
	{
		// Odd.
		threshold = TEMP_threshold * (m_lowPass[length / 2]) / (0.67f);
	}
	else
	{
		// Even.
		threshold = TEMP_threshold * ((m_lowPass[length / 2 - 1] + m_lowPass[length / 2]) / 2) / (0.67f);
	}
	HardTh(m_highPass, length * GetLevel(), threshold);
	HardTh(GetStore(), length, threshold);
	//printf("threshold = %f\n", threshold);
	//MIRDWT();
	cudaMemcpy(m_lowPassC, GetStore(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(m_highPassC, m_highPass, levels * size, cudaMemcpyHostToDevice);
	void **
		params1 = new void * [levels + 3];
	params1[0] = (void *)&m_data;
	params1[1] = (void *)&m_lowPassC;
	float **
		ptrs = new float * [levels];
	for (int ll = 0, lm = levels; ll != levels; ++ll)
	{
		--lm;
		ptrs[ll] = m_highPassC + width * lm;
		params1[ll + 2] = (void *)&ptrs[ll];
	}
	params1[levels + 2] = (void *)&m_data;
	TryCUDA(cuLaunchKernel(m_combineKernel, CEILDIV((int)width, SHAPE_X), 1, 1, SHAPE_X, 1, 1, 0, 0, params1, 0));
	cudaMemcpy(GetStore(), m_data, size, cudaMemcpyDeviceToHost);
	delete [] params1;
	delete [] ptrs;
	//memcpy(GetStore(), m_highPass + width, size);
}

void
	UDWTHaskell::
	ConvClose() throw (...)
{
	//ScriptClose(false);
	delete m_highKernel;
};

#endif
