// Copyright (c) Microsoft Corporation.   All rights reserved.

#if defined NO_ACCELERATOR && defined NO_ACCELERATOR_OPT
	#error "The molecules comparison requires Float4, which requires Accelerator."
#endif

#include "Source/Atoms.h"

#include <RunTest.h>

Atoms *
	g_atoms;

int
	g_height,
	g_width;

bool
	Compare(ParallelCode & a, ParallelCode & b)
{
	// Checks that two Moleculess are close enough.
	Data2DFloat
		& d0 = dynamic_cast<Data2DFloat &>(a.GetResult()),
		& d1 = dynamic_cast<Data2DFloat &>(b.GetResult());
	size_t
		it = d0.GetHeight(),
		jt = d0.GetWidth();
	if (it != d1.GetHeight() || jt != d1.GetWidth())
	{
		return false;
	}
	for (size_t i = 0; i != it; ++i)
	{
		for (size_t j = 0; j != jt; ++j)
		{
			if (!Compare(d0(i, j), d1(i, j), 0.0025f))
			{
				std::cout << i << ", " << j << " : " << d0(i, j) << " <> " << d1(i, j) << std::endl;
				// No point finishing the loop.
				//return false;
			}
		}
	}
	return true;
}

bool
	Print(ParallelCode & a)
{
	// Checks that two convolvers are close enough.
	Data2DFloat
		& d0 = dynamic_cast<Data2DFloat &>(a.GetResult());
	size_t
		it = d0.GetHeight(),
		jt = d0.GetWidth();
	for (size_t i = 0; i != it; ++i)
	{
		for (size_t j = 0; j != jt; ++j)
		{
			std::cout << i << ", " << j << " : " << d0(i, j) << std::endl;
		}
	}
	std::cout << "Data size: " << it << " x " << jt << std::endl;
	return true;
}

void
	Setup(Molecules & c)
{
	try {c.SetAtoms(*g_atoms);}
	catch (...) {}
	try {c.SetGridSize(g_height, g_width);}
	catch (...) {}
}

extern char
	g_fname[32];

#define SIZES                          (sizeof (sizes) / sizeof (int))

int
	main(int argc, char ** argv)
{
	const int
		sizes[] = 
			{
				//1, 2, 4, 8, 16, 23, 32, 39, 42, 64, 71, 128, 135, 256, 263, 512, 519, 555, 700, 1000, 1024, 1500, 2000, 2007, 2048, 4000, 4007, 4096, 4500, 4501, 5000, 10000, 20000
				//32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280, 1312, 1344, 1376, 1408, 1440, 1472, 1504, 1536, 1568, 1600, 1632, 1664, 1696, 1728, 1760, 1792, 1824, 1856, 1888, 1920, 1952, 1984, 2016, 2048, 4096, 8192
				 128,  256,  384,  512,  640,  768,  896, 1024,
				1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
				2176, 2304,	2432, 2560, 2688, 2816, 2944, 3072,
				3200, 3328, 3456, 3584, 3712, 3840, 3968, 4096,
				4224, 4352, 4480, 4608, 4736, 4864, 4992, 5120,
				5248, 5376, 5504, 5632, 5760, 5888, 6016, 6144,
				6272, 6400, 6528, 6656, 6784, 6912, 7040, 7168,
				7296, 7424, 7552, 7680, 7808, 7936, 8064, 8192,
			};
	if (argc < 6) // 5 with atom count
	{
		std::cout << "Usage:" << std::endl;
		std::cout << "[x size index] [y size index] [atom count] <o|c|a|x|r|e|f|d|b|y|m|n> <s|c|v>" << std::endl;
		std::cout << "Any number of items in the angle brackets may be included in a single string to run multiple tests." << std::endl;
		std::cout << "Note however that not all listed targets are currently supported." << std::endl;
		//std::cout << "SIZES: " << SIZES << std::endl;
		return 0;
	}
	int
		i,
		j;
	char
		c;
	int
		rm = (int)E_RUN_NONE,
		cm = (int)E_COMP_NONE,
		atoms = 30;
	i = atoi(argv[1]);
	j = atoi(argv[2]);
	atoms = atoi(argv[3]);
	if (i < 0 || i >= SIZES)
	{
		std::cout << "i out of range (0 - " << (SIZES - 1) << ").";
		return 0;
	}
	else if (j < 0 || j >= SIZES)
	{
		std::cout << "j out of range (0 - " << (SIZES - 1) << ").";
		return 0;
	}
	else if (atoms < 1)
	{
		std::cout << "Please enter a valid atom count.";
		return 0;
	}
	else if (atoms  > 10000)
	{
		std::cout << "Be reasonable!" << std::endl;
	}
	//g_filter = new Filter(radius, FILTER_SIGMA);
	g_height = sizes[i];//i;//
	g_width = sizes[j];//j;//
	g_atoms = new Atoms(atoms, (float)g_height, (float)g_width);
	
	for (size_t k = 0; (c = argv[4][k]); ++k)
	{
		switch (c)
		{
			#if !defined NO_OPEN_CL
				TARGET('O', E_COMP_OPEN_CL);
			#endif
			#if !defined NO_ACCELERATOR
				TARGET('A', E_COMP_ACC);
				TARGET('X', E_COMP_X64);
				TARGET('M', E_COMP_AC_C);
			#endif
			#if !defined NO_ACCELERATOR_OPT
				TARGET('B', E_COMP_ACC2);
				TARGET('Y', E_COMP_X642);
				TARGET('N', E_COMP_AC_C2);
			#endif
			#if !defined NO_CUDA
				TARGET('C', E_COMP_CUDA);
				#if !defined NO_HASKELL
					TARGET('K', E_COMP_HASKELL);
				#endif
			#endif
			#if !defined NO_CUDA_OPT
				TARGET('D', E_COMP_CUDA2);
			#endif
			#if !defined NO_REFERENCE
				TARGET('R', E_COMP_REF);
			#endif
			#if !defined NO_REFERENCE_OPT
				TARGET('E', E_COMP_CACHE);
				TARGET('F', E_COMP_FAST);
			#endif
			default:
				std::cout << "Unknown/unsupported target.";
				return 0;
		}
	}
	for (size_t k = 0; (c = argv[5][k]); ++k)
	{
		switch (c)
		{
			MODE('C', E_RUN_C);
			MODE('S', E_RUN_S);
			MODE('V', E_RUN_V);
			default:
				std::cout << "Unknown run mode.";
				return 0;
		}
	}
	sprintf(g_fname, OUTPUT_DIR "Report2 %d %d.txt", i, j);
	Test((E_RUN)rm, (int)g_height, (int)g_width, (E_COMP)cm);
	return 0;
}
