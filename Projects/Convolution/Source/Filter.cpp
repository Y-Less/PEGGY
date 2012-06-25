// Copyright (c) Microsoft Corporation.   All rights reserved.

#include "Filter.h"
#include <memory>
#include <cmath>

// Initialisation lists have no defined execution order, but we need to ensure
// that m_sigma gets allocated before the super constructor is called, so we
// do it in the parameter list without affecting anything.
#define PREASSIGN(n) ((int)((m_sigma = n) && false)) +

// cons
	Filter::
	Filter() throw (...) :
		m_radius(2),
		Data1DFloat(PREASSIGN(2.0) 2 * 2 + 1, GT_Custom, (DataStore::ds_callback)&Filter::GenerateFilter)
{
}

// cons
	Filter::
	Filter(
		const size_t
			filterRadius
		) throw (...) :
		m_radius(filterRadius),
		Data1DFloat(PREASSIGN(2.0) filterRadius * 2 + 1, GT_Custom, (DataStore::ds_callback)&Filter::GenerateFilter)
{
}

// cons
	Filter::
	Filter(
		float
			sigma
		) throw (...) :
		m_radius(2),
		Data1DFloat(PREASSIGN(sigma) 2 * 2 + 1, GT_Custom, (DataStore::ds_callback)&Filter::GenerateFilter)
{
}

// cons
	Filter::
	Filter(
		const size_t
			filterRadius,
		float
			sigma
		) throw (...) :
		m_radius(filterRadius),
		Data1DFloat(PREASSIGN(sigma) filterRadius * 2 + 1, GT_Custom, (DataStore::ds_callback)&Filter::GenerateFilter)
{
}

void
	Filter::
	GenerateFilter(
		const DataType
			type,
		const void *
			buffer,
		const size_t
			size) throw (...)
{
	if (!(size & 1))
	{
		throw "Filter size must be odd";
	}
	if (type != DT_Float)
	{
		throw "Filter type must be DT_Float";
	}
	float
		* newFilter = (float *)buffer,
		sum = 0,
		// -(2 * sigma^2)
		sigma = m_sigma * m_sigma * -2.0f;
	//printf("sigma = %f\n", m_sigma);
	//printf("sigma = %f\n", sigma);
	// This is the only part of this file worth paying any attention to...
	for (int i = 0, j = -((int)size / 2); i != size; ++i, ++j)
	{
		// Most of the calculation here has been moved out of the main
		// loop to the sigma initialisation.
		//printf("j * j * sigma = %f\n", j * j * sigma);
		newFilter[i] = exp(j * j / sigma);
		sum += newFilter[i];
	}
	//printf("Filter:");
	for (int i = 0; i != size; ++i)
	{
		// If your filter is huge you could use Accelerator here...
		//printf("%f = ", newFilter[i]);
		newFilter[i] /= sum;
		//printf("%f ", newFilter[i]);
	}
	//printf("\n");
}

size_t
	Filter::
	GetRadius() const
{
	// Integer division - will round down.
	return m_radius;
}

float
	Filter::
	GetSigma() const
{
	// Integer division - will round down.
	return m_sigma;
}
