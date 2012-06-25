#include "StaticData.h"

/*!
 * This file may take a while to compile, but it only needs to ever be compiled
 * once per target.
 */

#ifdef NO_STATIC_DATA
	float
		gc_fPointData[1] = {0.0};
#else
	float
		gc_fPointData[MAX_DATA_S] =
			{
				#include "Data.h"
			};
#endif
