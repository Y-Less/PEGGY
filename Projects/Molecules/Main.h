#pragma once

#define OUTPUT_DIR "Output/"

#define TEST_NAME Molecules

#define ITTERATIONS                    (10)

#define COMB(n) Molecules##n

#if !defined NO_ACCELERATOR
	#include "Source/MoleculesAccelerator.h"
#endif

#if !defined NO_ACCELERATOR_OPT
	#include "Source/MoleculesAccelerator2.h"
#endif

#if !defined NO_CUDA
	#include "Source/MoleculesCUDA.h"
	#if !defined NO_HASKELL
		#include "Source/MoleculesHaskell.h"
	#endif
#endif

#if !defined NO_REFERENCE
	#include "Source/MoleculesReference.h"
#endif
#if !defined NO_REFERENCE_OPT
	#include "Source/MoleculesOptimised.h"
	#include "Source/MoleculesCached.h"
#endif
