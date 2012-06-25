
//#include <DataArrays.h>

#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
	#include <Accelerator.h>
	#include <AcceleratorException.h>
#endif
#include <Timings.h>

#include "RunTest.h"

#include <math.h>
#include <vector>

#include <fstream>

Timings
	gTimings;

bool
	Compare(
		float                          a,
		float                          b,
		float                          acc)
{
	// Checks that two values are close enough.  Based on IsCloseEnough in the
	// Accelerator unit tests.
	//std::cout << "Comparing " << a << " and " << b << std::endl;
	if (a == 0.0f && abs(b) < acc)
	//if (a == 0.0f && abs(b) < 0.1)
	{
		return true;
	}
	float
		flDiff = abs(a - b);
	return abs(flDiff / a) < acc; //025;
	//return flDiff < 0.0001;
	//return flDiff < 0.1;
	// Use a fractional diff system.  Actually, that's what the code I removed
	// used to do!  Silly me!
}

void
	Run(
		TEST_NAME &                    c,
		const bool                     setup)
	throw (...)
{
	try {c.SetItterations(ITTERATIONS);}
	catch (...) {}
	Setup(c);
	c.Run(setup);
	std::cout << std::endl;
}

void
	Verify(
		TEST_NAME &                    c)
	throw (...)
{
	try {c.SetItterations(ITTERATIONS);}
	catch (...) {}
	Setup(c);
	c.Verify();
}

typedef
	std::vector<TEST_NAME *>
	ConvList;

typedef
	ConvList::iterator
	ConvItter;

bool
	Timing(
		ConvList &                     v,
		const bool                     type)
{
	bool
		ret = true;
	// Time everything in "C" (Clean) mode, i.e. do all initialisation outside
	// the timing.
	for (ConvItter i = v.begin(); i != v.end(); ++i)
	{
		if (*i)
		{
			try
			{
				if (type)
				{
					std::cout << **i << " C: ";
				}
				else
				{
					std::cout << **i << " S: ";
				}
				Run(**i, !type);
			}
			catch (std::bad_alloc & e)
			{
				std::cout << " Bad allocation: \"" << e.what() << "\" in timing." << std::endl;
				ret = false;
			}
			#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
				catch (ParallelArrays::AcceleratorException * e)
				{
					std::wcout << " Accelerator Error: \"" << e->GetReasonString() << "\" in timing." << std::endl;
					ret = false;
				}
			#endif
			catch (const char * e)
			{
				std::cout << " Error: \"" << e << "\" in timing." << std::endl;
				ret = false;
			}
			catch (...)
			{
				std::cout << " Other error in timing." << std::endl;
				ret = false;
			}
		}
	}
	return ret;
}

bool
	Verification(
		ConvList &                     v)
{
	bool
		ret = true;
	int
		c = 0;
	// Time everything in "C" (Clean) mode, i.e. do all initialisation outside
	// the timing.
	for (ConvItter i = v.begin(); i != v.end(); ++i)
	{
		if (*i)
		{
			try
			{
				Verify(**i);
			}
			catch (std::bad_alloc & e)
			{
				std::cout << " Bad allocation: \"" << e.what() << "\" in verification." << std::endl;
				ret = false;
			}
			#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
				catch (ParallelArrays::AcceleratorException * e)
				{
					std::wcout << " Accelerator Error: \"" << e->GetReasonString() << "\" in verification." << std::endl;
					ret = false;
				}
			#endif
			catch (const char * e)
			{
				std::cout << " Error: \"" << e << "\" in verification." << std::endl;
				ret = false;
			}
			catch (...)
			{
				std::cout << " Other error in verification." << std::endl;
				ret = false;
			}
		}
		else
		{
			++c;
		}
	}
	if (ret)
	{
		for (ConvItter i = v.begin(); i != v.end(); ++i)
		{
			if (*i)
			{
				for (ConvItter j = i + 1; j != v.end(); ++j)
				{
					if (*j)
					{
						try
						{
							if (!Compare(**i, **j))
							{
								std::cout << **i << " <> " << **j << " comparison failed." << std::endl;
								ret = false;
							}
						}
						#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
							catch (ParallelArrays::AcceleratorException * e)
							{
								std::wcout << " Accelerator Error: \"" << e->GetReasonString() << "\" in verification." << std::endl;
								ret = false;
							}
						#endif
						catch (const char * e)
						{
							std::cout << " Error: \"" << e << "\" in verification." << std::endl;
							ret = false;
						}
						catch (...)
						{
							std::cout << " Other error in verification." << std::endl;
							ret = false;
						}
					}
				}
			}
		}
		if (ret)
		{
			std::cout << "All comparisons passed." << std::endl;
		}
	}
	else
	{
		c = (int)v.size();
	}
	std::cout << c << " target(s) skipped." << std::endl;
	return ret;
}

bool
	Printing(
		ConvList &                     v)
{
	bool
		ret = true;
	int
		c = 0;
	// Time everything in "C" (Clean) mode, i.e. do all initialisation outside
	// the timing.
	for (ConvItter i = v.begin(); i != v.end(); ++i)
	{
		if (*i)
		{
			try
			{
				Verify(**i);
			}
			catch (std::bad_alloc & e)
			{
				std::cout << " Bad allocation: \"" << e.what() << "\" in printing." << std::endl;
				ret = false;
			}
			#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
				catch (ParallelArrays::AcceleratorException * e)
				{
					std::wcout << " Accelerator Error: \"" << e->GetReasonString() << "\" in printing." << std::endl;
					ret = false;
				}
			#endif
			catch (const char * e)
			{
				std::cout << " Error: \"" << e << "\" in printing." << std::endl;
				ret = false;
			}
			catch (...)
			{
				std::cout << " Other error in printing." << std::endl;
				ret = false;
			}
		}
		else
		{
			++c;
		}
	}
	if (ret)
	{
		for (ConvItter i = v.begin(); i != v.end(); ++i)
		{
			if (*i)
			{
				try
				{
					Print(**i);
				}
				#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
					catch (ParallelArrays::AcceleratorException * e)
					{
						std::wcout << " Accelerator Error: \"" << e->GetReasonString() << "\" in printing." << std::endl;
						ret = false;
					}
				#endif
				catch (const char * e)
				{
					std::cout << " Error: \"" << e << "\" in printing." << std::endl;
					ret = false;
				}
				catch (...)
				{
					std::cout << " Other error in printing." << std::endl;
					ret = false;
				}
			}
		}
	}
	return ret;
}

char
	g_fname[64];

bool
	Test(
		const E_RUN                    RunType,
		const int                      height,
		const int                      width,
		E_COMP                         GPUType)
{
//	{
//		std::fstream
//			filestr(g_fname, std::fstream::out);
//		filestr << "Molecules Test Started." << std::endl;
//		filestr.close();
//	}
	std::cout << "========================================" << std::endl << "Testing at " << height << "x" << width << ":" << std::endl;
	bool
		ret = true;
	ConvList
		l;
	// Create all the targets.
	try
	{
		#if !defined NO_ACCELERATOR
			ADD_TYPE_PAR(E_COMP_ACC, Accelerator, PAT_DX9);
			ADD_TYPE_PAR(E_COMP_X64, Accelerator, PAT_X64);
			ADD_TYPE_PAR(E_COMP_AC_C, Accelerator, PAT_AC_C);
		#endif
		#if !defined NO_ACCELERATOR_OPT
			ADD_TYPE_PAR(E_COMP_ACC2, Accelerator2, PAT_DX9);
			ADD_TYPE_PAR(E_COMP_X642, Accelerator2, PAT_X64);
			ADD_TYPE_PAR(E_COMP_AC_C2, Accelerator2, PAT_AC_C);
		#endif
		#if !defined NO_CUDA
			ADD_TYPE(E_COMP_CUDA, CUDA);
			#if !defined NO_HASKELL
				ADD_TYPE(E_COMP_HASKELL, Haskell);
			#endif
		#endif
		#if !defined NO_CUDA_DRIVER
			ADD_TYPE(E_COMP_DRIVER, CUDADriver);
		#endif
		#if !defined NO_CUDA_OPT
			ADD_TYPE(E_COMP_CUDA2, CUDA2);
		#endif
		#if !defined NO_OPEN_CL
			ADD_TYPE_PAR(E_COMP_OPEN_CL, OpenCL, PST_CPU);
		#endif
		#if !defined NO_REFERENCE
			ADD_TYPE(E_COMP_REF, Reference);
		#endif
		#if !defined NO_REFERENCE_OPT
			ADD_TYPE(E_COMP_FAST, Optimised);
			ADD_TYPE(E_COMP_CACHE, Cached);
		#endif
		#if !defined NO_OBS
			ADD_TYPE(E_COMP_OBS, Obs);
		#endif
		#if !defined NO_OBS_OPT
			ADD_TYPE(E_COMP_OBS_OPT, ObsOpt);
		#endif
	}
	catch (...)
	{
		std::cout << "Error creating target" << std::endl;
		return false;
	}
	if (RunType & E_RUN_C)
	{
		std::cout << "Timing " RUN_TEST_STR(TEST_NAME) " clean..." << std::endl;
		ret = Timing(l, true) ? ret : false;
	}
	if (RunType & E_RUN_S)
	{
		std::cout << "Timing " RUN_TEST_STR(TEST_NAME) " setup..." << std::endl;
		ret = Timing(l, false) ? ret : false;
	}
	if (RunType & E_RUN_V)
	{
		std::cout << "Verifying " RUN_TEST_STR(TEST_NAME) "..." << std::endl;
		ret = Verification(l) ? ret : false;
	}
	if (RunType & E_RUN_P)
	{
		std::cout << "Printing " RUN_TEST_STR(TEST_NAME) "..." << std::endl;
		ret = Printing(l) ? ret : false;
	}
	// Output all the collected times.
	{
		std::fstream
			filestr(g_fname, std::fstream::app);//::out);
//		filestr << "========================================" << std::endl << "Testing at " << height << "x" << width << ":" << std::endl;
		filestr << gTimings;
		filestr.close();
	}
	std::cout << "========================================" << std::endl;
	return ret;
}
