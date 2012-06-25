// Copyright (c) Microsoft Corporation.   All rights reserved.

#pragma once

#ifdef NO_STATIC_DATA
	#define MAX_DATA_S (1)
#else
	#define MAX_DATA_S 10000 //(sizeof (gc_fPointData) / sizeof (float))
#endif

////////////////////////////////////////////////////////////////////////////////
// Constant known reference data
////////////////////////////////////////////////////////////////////////////////

// Define this externally to avoid repeated compilation.
extern float
	gc_fPointData[MAX_DATA_S];
