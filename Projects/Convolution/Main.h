#pragma once

#define OUTPUT_DIR "Output/"

#define TEST_NAME Convolver

#define ITTERATIONS                    (10)

#define COMB(n) Convolver##n

#if !defined NO_ACCELERATOR
	#include "Source/ConvolverAccelerator.h"
#endif
#if !defined NO_ACCELERATOR_OPT
	#include "Source/ConvolverAccelerator2.h"
#endif
#if !defined NO_CUDA
	#include "Source/ConvolverCUDA.h"
	#if !defined NO_HASKELL
		#include "Source/ConvolverHaskell.h"
	#endif
#endif
#if !defined NO_CUDA_DRIVER
	#include "Source/ConvolverCUDADriver.h"
#endif
#if !defined NO_CUDA_OPT
	#include "Source/ConvolverCUDA2.h"
#endif
#if !defined NO_OPEN_CL
	#include "Source/ConvolverOpenCL.h"
#endif
#if !defined NO_OBS
	#include "Source/ConvolverObs.h"
#endif
#if !defined NO_OBS_OPT
	#include "Source/ConvolverObsOpt.h"
#endif
#if !defined NO_REFERENCE
	#include "Source/ConvolverReference.h"
#endif
#if !defined NO_REFERENCE_OPT
	#include "Source/ConvolverOptimised.h"
	#include "Source/ConvolverCached.h"
#endif

#define FILTER_SIGMA                   3.0
