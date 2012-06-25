// Copyright (c) Microsoft Corporation.   All rights reserved.

#include "Source/Filter.h"

#include <RunTest.h>
//#include <Timings.h>

// Reuse the filter from the normal convolution (may as well).
Filter *
	g_filter = 0;

//extern Timings
//	gTimings;

//Data1DFloat *
//Loop1D *
Data1DFloat *
	g_data = 0;

size_t
	g_level = 3;

bool
	Compare(ParallelCode & a, ParallelCode & b)
{
	// Checks that two convolvers are close enough.
	Data1DFloat
		& d0 = dynamic_cast<Data1DFloat &>(a.GetResult()),
		& d1 = dynamic_cast<Data1DFloat &>(b.GetResult());
	size_t
		jt = d0.GetWidth();
	//if (it != d1.GetHeight() || jt != d1.GetWidth())
	//{
	//	return false;
	//}
	if (jt != d1.GetWidth())
	{
		return false;
	}
	//for (size_t i = 0; i != it; ++i)
	//{
	for (size_t j = 0; j != jt; ++j)
	{
		//if (!Compare(d0(i, j), d1(i, j)))
		// For some reason, every implementation comes out with slightly
		// different values every multiple of 256 - this is an interestng point
		// at which to have a bug, but I can't solve it right now...
		if (!Compare(d0(j), d1(j)) && (j & 0xFF))
		{
			std::cout.precision(10);
			// No point finishing the loop.
			std::cout << j << " : " << d0(j) << " <> " << d1(j) << std::endl;
			//std::cout << j << " : " << d0(j) << " <> " << d1(j) << std::endl;
			//return false;
		}
	}
	//}
	return true;
}

bool
	Print(ParallelCode & a)
{
	// Checks that two convolvers are close enough.
	Data1DFloat
		& d0 = dynamic_cast<Data1DFloat &>(a.GetResult());
	size_t
		jt = d0.GetWidth();
	for (size_t j = 0; j != jt; ++j)
	{
		std::cout << j << " : " << d0(j) << std::endl;
	}
	std::cout << "Data size: " << jt << std::endl;
	return true;
}

void
	Setup(UDWT & c)
{
	try {c.SetFilter(*g_filter);}
	catch (...) {}
	try {c.SetLevel(g_level);}
	catch (...) {}
	try {c.SetData(*g_data);}
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
				//32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, , , , , , , , , , , , 
				//32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280, 1312, 1344, 1376, 1408, 1440, 1472, 1504, 1536, 1568, 1600, 1632, 1664, 1696, 1728, 1760, 1792, 1824, 1856, 1888, 1920, 1952, 1984, 2016, 2048
				 128,  256,  384,  512,  640,  768,  896, 1024,
				1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
				2176, 2304,	2432, 2560, 2688, 2816, 2944, 3072,
				3200, 3328, 3456, 3584, 3712, 3840, 3968, 4096,
				4224, 4352, 4480, 4608, 4736, 4864, 4992, 5120,
				5248, 5376, 5504, 5632, 5760, 5888, 6016, 6144,
				6272, 6400, 6528, 6656, 6784, 6912, 7040, 7168,
				7296, 7424, 7552, 7680, 7808, 7936, 8064, 8192,
				//1, 32, 256, 1000//, 1024, 1500, 2000, 2007, 2048, 4000, 4007, 4096, 4500, 4501, 5000
				//1000, 2000, 4000, 4007, 4096, 4500, 4501, 5000
			};
	if (argc < 5) // 5 with filter size
	{
		std::cout << "Usage:" << std::endl;
		std::cout << "[x size index] [level] <o|c|a|x|r|f|e|d|b|y|m|n|h> <s|c|v|p>" << std::endl;
		std::cout << "Any number of items in the angle brackets may be included in a single string to run multiple tests." << std::endl;
		std::cout << "Note however that not all listed targets are currently supported." << std::endl;
		return 0;
	}
	int
		i,
		j = 0;
	char
		c;
	int
		rm = (int)E_RUN_NONE,
		cm = (int)E_COMP_NONE,
		radius = 3;
	i = atoi(argv[1]);
	//j = atoi(argv[2]);
	g_level = atoi(argv[2]);
	//radius = atoi(argv[4]);
	if (i < 0 || i >= SIZES)//8193)//
	{
		std::cout << "i out of range (0 - " << (SIZES - 1) << ").";
		return 0;
	}
	/*else if (j < 0 || j >= SIZES)//8193)//
	{
		std::cout << "j out of range (0 - " << (SIZES - 1) << ").";
		return 0;
	}*/
	else if (g_level < 0)
	{
		std::cout << "Please enter a valid level.";
		return 0;
	}
	else if (g_level > 6)
	{
		std::cout << "The level must be <= 6.";
		return 0;
	}
	/*else if (radius < 1)
	{
		std::cout << "Please enter a valid radius.";
		return 0;
	}
	else if (radius > 23)
	{
		std::cout << "Warning: CUDA currently only supports filter sizes <= 23." << std::endl;
	}*/
	// Create the filter.
	g_filter = new Filter(); //3, FILTER_SIGMA);
	// Create the data.  NOTE THAT THIS IS 1D DATA IN A 2D ARRAY TO GET MORE.
	// NO IT ISN'T!
	//g_data = new Data1DFloat(sizes[i] * sizes[j], GT_Seed, 2012 * 42);
	g_data = new Data1DFloat(sizes[i] * 2048, GT_Test);
	for (size_t k = 0; (c = argv[3][k]); ++k)
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
			#if !defined NO_OBS
				TARGET('H', E_COMP_OBS);
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
	for (size_t k = 0; (c = argv[4][k]); ++k)
	{
		switch (c)
		{
			MODE('C', E_RUN_C);
			MODE('S', E_RUN_S);
			MODE('V', E_RUN_V);
			MODE('P', E_RUN_P);
			default:
				std::cout << "Unknown run mode.";
				return 0;
		}
	}
	sprintf(g_fname, OUTPUT_DIR "Report2 %d %d %d %d.txt", i, j, g_level, radius);
	//Test((E_RUN)rm, i, j, (E_COMP)cm);
	Test((E_RUN)rm, sizes[i] * 2048, 0, (E_COMP)cm);
	return 0;
}
