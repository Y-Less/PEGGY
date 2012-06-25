// Copyright (c) Microsoft Corporation.   All rights reserved.

#pragma once

#include <Arrays/Data1DFloat.h>

class Filter //:
	//public Data1DFloat
{
public:
	// cons
		Filter();
	
	// cons
		Filter(const size_t);
	
	// cons
		Filter(float);
	
	// cons
		Filter(const size_t, float);
	
	size_t
		GetRadius() const;
	
	Data1DFloat &
		GetForward1();
	
	Data1DFloat &
		GetReverse1();
	
	Data1DFloat &
		GetForward2();
	
	Data1DFloat &
		GetReverse2();
	
	void
		CleanGPU();
	
private:
	void
		GenerateFilter(); //const DataType, const void *, const size_t) throw (...);
	
	float
		m_sigma;
	
	size_t
		m_radius;
	
	Data1DFloat
		m_forward1,
		m_reverse1,
		m_forward2,
		m_reverse2;
};
