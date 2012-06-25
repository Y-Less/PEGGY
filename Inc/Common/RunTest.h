#pragma once

#include <Main.h>

#include <ParallelCode.h>

#if !defined TEST_NAME
	#error Please define the test prefix as "TEST_NAME".
#endif

#if !defined ITTERATIONS
	#error Please define the number of itterations to run.
#endif

#if !defined COMB
	#error Please define a macro to generate class names as "COMB".
#endif

#define RUN_TEST_STR2(m) #m

#define RUN_TEST_STR(m) RUN_TEST_STR2(m)

#define ADD_TYPE(m,n)                                                          \
	if (GPUType & (m))                                                         \
		l.push_back(new COMB(n)())

#define ADD_TYPE_PAR(m,n,o)                                                    \
	if (GPUType & (m))                                                         \
		l.push_back(new COMB(n)(o))

#define TARGET(m,n)                                                            \
	case (m): case ((m) | 0x20):                                               \
		cm |= (int)(n);                                                        \
		break

#define MODE(m,n)                                                              \
	case (m): case ((m) | 0x20):                                               \
		rm |= (int)(n);                                                        \
		break

enum
	E_COMP
{
	E_COMP_NONE    = 0,
	E_COMP_ACC     = (1 << 0),
	E_COMP_CUDA    = (1 << 1),
	E_COMP_X64     = (1 << 2),
	E_COMP_REF     = (1 << 3),
	E_COMP_OPEN_CL = (1 << 4),
	E_COMP_FAST    = (1 << 5),
	E_COMP_CACHE   = (1 << 6),
	E_COMP_ACC2    = (1 << 7),
	E_COMP_X642    = (1 << 8),
	E_COMP_CUDA2   = (1 << 9),
	E_COMP_AC_C    = (1 << 10),
	E_COMP_AC_C2   = (1 << 11),
	E_COMP_OBS     = (1 << 12),
	E_COMP_OBS_OPT = (1 << 13),
	E_COMP_DRIVER  = (1 << 14),
	E_COMP_HASKELL = (1 << 15),
};

enum
	E_RUN
{
	E_RUN_NONE = 0,
	E_RUN_C    = (1 << 0),
	E_RUN_S    = (1 << 1),
	E_RUN_V    = (1 << 2),
	E_RUN_P    = (1 << 3),
};

bool
	Test(const E_RUN, const int, const int, E_COMP);

bool
	Compare(ParallelCode &, ParallelCode &);

bool
	Print(ParallelCode &);

bool
	Compare(float, float, float acc = 0.001f);

void
	Setup(TEST_NAME &);
