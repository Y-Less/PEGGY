// Copyright (c) Microsoft Corporation.   All rights reserved.

#pragma once

#include <Arrays/Data1DFloat.h>

class Filter :
	public Data1DFloat
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
	
	float
		GetSigma() const;
	
private:
	void
		GenerateFilter(const DataType, const void *, const size_t) throw (...);
	
	float
		m_sigma;
	
	size_t
		m_radius;
};
