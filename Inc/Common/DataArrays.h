// Copyright (c) Microsoft Corporation.   All rights reserved.

#pragma once

// Include all the various types

#include "Arrays/Data1DBool.h"
#include "Arrays/Data1DFloat.h"
#include "Arrays/Data1DInt.h"

#include "Arrays/Data2DBool.h"
#include "Arrays/Data2DFloat.h"
#include "Arrays/Data2DInt.h"

#if !defined NO !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
	#include "Arrays/Data1DFloat4.h"
	#include "Arrays/Data2DFloat4.h"
#endif

/*#include "Arrays/Data3DBool.h"
#include "Arrays/Data3DFloat.h"
#include "Arrays/Data3DFloat4.h"
#include "Arrays/Data3DInt.h"*/
