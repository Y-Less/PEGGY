// Copyright (c) Microsoft Corporation.   All rights reserved.

#include "Filter.h"
#include <memory>
#include <cmath>

// Initialisation lists have no defined execution order, but we need to ensure
// that m_sigma gets allocated before the super constructor is called, so we
// do it in the parameter list without affecting anything.
//#define PREASSIGN(n) ((int)((m_sigma = n) && false)) +

// cons
	Filter::
	Filter() throw (...) :
		m_radius(8),
		//m_sigma(2.0f),
		m_forward1(8, GT_Garbage),
		m_reverse1(8, GT_Garbage),
		m_forward2(8, GT_Garbage),
		m_reverse2(8, GT_Garbage)
		//Data1DFloat(PREASSIGN(2.0) 2 * 2 + 1, GT_Custom, (DataStore::ds_callback)&Filter::GenerateFilter)
{
	GenerateFilter();
}

void
	Filter::
	GenerateFilter()
{
	// The UDWT filter is pretty much constant.
	float
		* forward1 = (float *)m_forward1.GetData(),
		* reverse1 = (float *)m_reverse1.GetData(),
		* forward2 = (float *)m_forward2.GetData(),
		* reverse2 = (float *)m_reverse2.GetData();
	static double const
		scFilter[] = {0.2304, 0.7148, 0.6309, -0.0280, -0.1870, 0.0308, 0.0329, -0.0106};
	for (size_t eye = 0, jay = 8; eye != 8; ++eye)
	{
		--jay;
		reverse1[eye] = (float)(scFilter[jay]);
		forward2[eye] = (float)(scFilter[eye] / 2);
		if (eye & 1)
		{
			reverse2[eye] = (float)(-scFilter[jay] / 2);
			forward1[eye] = (float)(scFilter[eye]);
		}
		else
		{
			reverse2[eye] = (float)(scFilter[jay] / 2);
			forward1[eye] = (float)(-scFilter[eye]);
		}
	}
}

size_t
	Filter::
	GetRadius() const
{
	// Integer division - will round down.
	return m_radius;
}

Data1DFloat &
	Filter::
	GetForward1()
{
	return m_forward1;
}

Data1DFloat &
	Filter::
	GetReverse1()
{
	return m_reverse1;
}

Data1DFloat &
	Filter::
	GetForward2()
{
	return m_forward2;
}

Data1DFloat &
	Filter::
	GetReverse2()
{
	return m_reverse2;
}

void
	Filter::
	CleanGPU()
{
	m_forward1.CleanGPU();
	m_reverse1.CleanGPU();
	m_forward2.CleanGPU();
	m_reverse2.CleanGPU();
}

// Note to self: From now on only ever use "eye" and "jay" (and "kay" etc) for
// temporary local variables, don't use "i", "j" etc - they're too easy to get
// mixed up.
