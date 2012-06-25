#pragma once

#define OUTPUT_DIR "Output/"

#define TEST_NAME UDWT

#define ITTERATIONS                    (10)

#define COMB(n) UDWT##n

#if !defined NO_ACCELERATOR
	#include "Source/UDWTAccelerator.h"
#endif
#if !defined NO_ACCELERATOR_OPT
	#include "Source/UDWTAccelerator2.h"
#endif
#if !defined NO_CUDA
	#include "Source/UDWTCUDA.h"
	#if !defined NO_HASKELL
		#include "Source/UDWTHaskell.h"
	#endif
#endif
#if !defined NO_CUDA_OPT
	#include "Source/UDWTCUDA2.h"
#endif
#if !defined NO_OPEN_CL
	#include "Source/UDWTOpenCL.h"
#endif
#if !defined NO_OBS
	#include "Source/UDWTObs.h"
#endif

#if !defined NO_REFERENCE
	#include "Source/UDWTReference.h"
#endif
#if !defined NO_REFERENCE_OPT
	#include "Source/UDWTOptimised.h"
	#include "Source/UDWTCached.h"
#endif

#define FILTER_SIGMA                   3.0
