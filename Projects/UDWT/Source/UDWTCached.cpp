#if !defined NO_REFERENCE

#include "UDWTCached.h"

//#include <tracing.h>
//#include "UDWTReference.tmh"

float
	gLargeData[4096 * 8192];

// cons
	UDWTCached::
	UDWTCached() :
		UDWT(),
		ParallelReference()
{
}

void
	UDWTCached::
	ConvClose() throw (...)
{
}

void
	UDWTCached::
	ConvInit() throw (...)
{
	// Get the data we want.
	m_data = (float *)GetData().GetData();
	m_lowPass  = (float *)GetLowPass().GetData();
	m_highPass = gLargeData; //(float *)GetHighPass().GetData();
	//for (size_t i = 0; i != GetWidth(); ++i)
	//{
	//	m_highPass[i] = (float)random();
	//}
}

void
	Abs(float * data, size_t length);

void
	HardTh(float * data, size_t length, float threshold);

int
	FloatCompare(
		void const *
			a,
		void const *
			b);

void
	UDWTCached::
	Execute() throw (...)
{
	float
		TEMP_threshold = 0.5f;
	size_t
		length = GetWidth();
	memcpy(m_lowPass, m_highPass, length * sizeof (float));
	Abs(m_lowPass, length);
	qsort(m_lowPass, length, sizeof (float), FloatCompare);
	float
		threshold;
	if (length & 1)
	{
		// Odd.
		threshold = TEMP_threshold * (m_lowPass[length / 2]) / (0.67f);
	}
	else
	{
		// Even.
		threshold = TEMP_threshold * ((m_lowPass[length / 2 - 1] + m_lowPass[length / 2]) / 2) / (0.67f);
	}
	HardTh(m_highPass, length * GetLevel(), threshold);
	HardTh(GetStore(), length, threshold);
}

#endif
