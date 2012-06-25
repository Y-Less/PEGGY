#if !defined NO_ACCELERATOR

#include "UDWTAccelerator.h"
#include <iostream>

#include <stdlib.h>

//#include <tracing.h>
//#include "UDWTAccelerator.tmh"

using namespace ParallelArrays;

// cons
	UDWTAccelerator::
	UDWTAccelerator(
		const ParallelAcceleratorType
			type
		) :
		UDWT(),
		ParallelAccelerator(type),
		m_forward1(0),
		m_reverse1(0),
		m_forward2(0),
		m_reverse2(0),
		m_data()
{
}

void
	UDWTAccelerator::
	ConvInit() throw (...)
{
	// Get the data we want.
	m_data = dynamic_cast<FloatParallelArray &>(GetData().ToAcceleratorArray());
	m_forward1 = (float *)GetForward1().GetData();
	m_reverse1 = (float *)GetReverse1().GetData();
	m_forward2 = (float *)GetForward2().GetData();
	m_reverse2 = (float *)GetReverse2().GetData();
	// The accelerator version can't set up here as it needs to zero the
	// accelerator array every itteration.
}

int
	FloatCompare(
		void const *
			a,
		void const *
			b)
{
	float
		ret = (*(float *)a - *(float *)b);
	// We only need the sign and the value, which is fine for casting between
	// float and int.
	return *((int *)(&ret));
}

void
	UDWTAccelerator::
	Execute() throw (...)
{
	static size_t const
		scMaxWidth = 4096;
	size_t
		length = GetWidth(),
		apron = 32,
		usable = scMaxWidth - (apron * 2),
		rows = (length - 1) / usable + 1,
		levels = GetLevel(),
		biglen = levels * length;
	// Parallel arrays for the calculation.
	FloatParallelArray
		xl,
		xh,
		** xhs = new FloatParallelArray * [levels],
		** xhs2 = new FloatParallelArray * [levels];
	// Based on the RWT:
	//  
	//  http://dsp.rice.edu/software/rice-wavelet-toolbox
	//  
	// Don't forget that the arrays must be an exact power of two size.
	MRDWT(m_data, &xl, xhs);
	// This bit can't currently be done in Accelerator - we need a sort and
	// median function.  The latter can be done, the former POSSIBLY can be, but
	// it would be very hard.
	float
		* dest = new float [rows * scMaxWidth],
		* allhigh = new float [biglen];
	// Get the high threshold.
	for (size_t i = 0; i != level; ++i)
	{
		// TODO: Find out which method is fastest.  Calculating the "xhs" parts
		// twice or extracting them once then putting them back in.
		GetTarget().ToArray(*xhs[i], dest, rows, scMaxWidth, scMaxWidth * sizeof (float));
		Data1D::Recompress(allhigh + (length * i), dest, length, apron);
		// For method 2.
		xhs2[i] = new FloatParallelArray(dest, rows, scMaxWidth);
	}
	qsort(allhigh, biglen, sizeof (float), FloatCompare);
	/*float
		highThreshold,
		lowThreshold;
	if (biglen & 1)
	{
		// Odd.
		highThreshold = allhigh[biglen / 2];
	}
	else
	{
		// Even.
		highThreshold = (allhigh[biglen / 2 - 1] + allhigh[biglen / 2]) / 2;
	}
	// Get the low threshold.
	GetTarget().ToArray(xl, dest, rows, scMaxWidth, scMaxWidth * sizeof (float));
	Data1D::Recompress(GetStore(), dest, length, apron);
	qsort(allhigh, biglen, sizeof (float), FloatCompare);
	if (length & 1)
	{
		// Odd.
		highThreshold = allhigh[length / 2];
	}
	else
	{
		// Even.
		highThreshold = (allhigh[length / 2 - 1] + allhigh[length / 2]) / 2;
	}*/
	// Get the threshold as a SHORT FloatParallelArray.  This will need
	// REPLICATING when it is used to match the given array size.  Note that the
	// code here assumes only a single dimension, the original code filtered out
	// parts of the input.
//	FloatParallelArray
//		threshold = Median(Abs(xh)) * (TEMP_threshold * 3 / 2);
//	HardTh(xh, threshold);
//	HardTh(xl, threshold);
//	xd = MIRDWT(xh, xl);
	//FloatParallelArray
	//	xd = MIRDWT(dynamic_cast<FloatParallelArray &>(m_data), dynamic_cast<FloatParallelArray &>(m_data2->ToAcceleratorArray()));
	//printf("7\n");
	// If we had inline functions we could not do the memory
	// allocation here and instead do it in "WrapData", yet still be
	// able to specify more of the logic here and USE the memory.
	//std::cout << "Acc Store: " << GetStore() << std::endl;
	//GetTarget().ToArray(input, GetStore(), length);
	//lowPassRet = new ArrayType(input);
	//highPassRet = new ArrayType(highPass);
	//printf("7\n");
	//GetTarget().ToArray(xl, dest, rows, scMaxWidth, scMaxWidth * sizeof (float));
	//Data1D::Recompress(GetStore(), dest, length, apron);
	for (size_t i = 0; i != level; ++i)
	{
		delete xhs[i];
		delete xhs2[i];
	}
	delete [] xhs;
	delete [] xhs2;
	//GetTarget().ToArray(xl, dest, rows, scMaxWidth, scMaxWidth * sizeof (float));
	//Data1D::Recompress(GetStore(), dest, length, apron);
	delete [] dest;
	delete [] allhigh;
	//FloatParallelArray
	//	xn = input - xd;
	// The code only uses "xd", the filtered smooth version.  I wonder if matlab
	// is lazy, it would make sense to me, but who knows.  Apparently not...
}

FloatParallelArray
	UDWTAccelerator::
	HardTh(
		//float const
		//	threshold,
		FloatParallelArray
			input,
		FloatParallelArray
			threshold)
{
	// Filter only elements over a certain value.
	size_t
		dims = input.GetRank(),
		len[2];
	input.Dimensions(len, dims);
	// Create a comparison array against which to test the threshold.
	FloatParallelArray
		tarr = Replicate(threshold, len, dims),
		zero(0.0, len, dims);
	// abs(input) > threshold ? input : 0
	return Cond(Abs(input) > tarr, input, zero);
}

FloatParallelArray
	UDWTAccelerator::
	Median(
		FloatParallelArray
			input)
{
	// This is not overly nice code to find the median of an array.  Frankly I
	// think finding the mean would be easier.
	// Filter only elements over a certain value.
	size_t
		dims = input.GetRank(),
		len[2],
		slen[2];
	int
		ilen[2];
	if (dims != 1)
	{
		// This could be a problem as our arrays ARE multi-dimensional!
		throw "Median called on multi-dimensional data.";
	}
	input.Dimensions(len, dims);
	if (len[0] & 1)
	{
		// Odd number of elements, get the middle one.
		slen[0] = 1;
		IntParallelArray
			gather(len[0] / 2, slen, 1);
		return Gather(input, &gather, dims);
		//return Replicate(Gather(input, &gather, dims), len, dims);
	}
	else
	{
		// Even number of elements, get the mean of the two middle values.
		ilen[1] = len[0] / 2;
		ilen[0] = ilen[1] - 1;
		// Make a 2 element array.
		IntParallelArray
			gather(ilen, 2);
		FloatParallelArray
			shrunk = Gather(input, &gather, dims);
		// Find the average of the two values.  Technically this finds the
		// average twice, then replicates it many many times.
		slen[0] = 1;
		return (shrunk + Rotate(shrunk, ilen, dims)) / FloatParallelArray(2.0f, len, dims);
		//shrunk += Rotate(shrunk, slen, dims);
		//return Replicate(shrunk / 2, len, dims);
	}
}

FloatParallelArray
	UDWTAccelerator::
	MIRDWT(FloatParallelArray low, FloatParallelArray high)
{
	//printf("1\n");
	static size_t const
		scMaxWidth = 4096;
	size_t
		length = GetWidth();
	size_t
		apron = 32,
		usable = scMaxWidth - (apron * 2),
		rows = (length - 1) / usable + 1,
		levels = GetLevel(),
		// Size of the 1D array wrapped in a 2D array for greater value use.
		dims[] = {rows, scMaxWidth},
		filterLength = GetRadius() * 2 + 1;
	//printf("length = %d, rows = %d, usable = %d\n", length, rows, usable);
	int
		rotate[2] = {0, 0};
	bool *
		hpCond = (bool *)calloc(levels, sizeof (bool));
	//printf("1\n");
	// Get the two filters.
	float
		* forward = (float *)GetFilter().GetForward2().GetData(),
		* reverse = (float *)GetFilter().GetReverse2().GetData();
	//printf("1\n");
	for (size_t level = levels; level-- != 0; )
	{
		//printf("2 %d %d\n", level, rows);
		//printf("2 %d %d\n", high.GetLength(0), high.GetLength(1));
		// Get a single section of the "high" array.
		SectionSpecifier
			sectionX(0, scMaxWidth, scMaxWidth),
			sectionY(level * rows, rows, levels * rows);
		FloatParallelArray
			convL(0.0f, dims, 2),
			convH = Section(high, sectionY, sectionX);
		//printf("3\n");
		// Loop through every element of the filter for the convolution.
		for (size_t filterCur = 0, filterGet = filterLength; filterCur != filterLength; ++filterCur)
		{
			//printf("4 %d\n", filterCur);
			--filterGet;
			// Rotate by on offset.
			rotate[1] = -(int)(filterCur << level);
			// Reverse "filterReverse".
			//printf("%d %d %d %d\n", low.GetLength(0), low.GetLength(1), convH.GetLength(0), convH.GetLength(1));
			convL = convL + (Rotate(low, rotate, 2) * forward[filterCur]) + (Rotate(convH, rotate, 2) * reverse[filterCur]);
		}
		//printf("5\n");
		// Combine the results.
		low = convL;
	}
	//GetTarget().ToArray(input, dest, rows, scMaxWidth, scMaxWidth * sizeof (float));
	//Data1D::Recompress(GetStore(), dest, length, apron);
	//delete [] dest;
	return low;
}

void
	UDWTAccelerator::
	MRDWT(
		FloatParallelArray
			input,
		FloatParallelArray *
			lowRet,
		FloatParallelArray *
			highRet[])
{
	static size_t const
		scMaxWidth = 4096;
	size_t
		length = GetWidth();
	size_t
		apron = 32,
		usable = scMaxWidth - (apron * 2),
		rows = (length - 1) / usable + 1,
		levels = GetLevel(),
		//dims[] = {length},
		dims[] = {rows, scMaxWidth},
		filterLength = GetRadius() * 2 + 1;
	int
		rotate[2] = {0, 0};
	float
		* forward = (float *)GetFilter().GetForward1().GetData(),
		* reverse = (float *)GetFilter().GetReverse1().GetData();
	for (size_t level = 0; level != levels; ++level)
	{
		FloatParallelArray
			convL(0.0f, dims, 2),
			convH(0.0f, dims, 2);
		for (size_t filterCur = 0, filterGet = filterLength; filterCur != filterLength; ++filterCur)
		{
			--filterGet;
			// Rotate by on offset.
			rotate[1] = (int)(filterCur << level);
			//rotate[0] = filterCur * f;
			FloatParallelArray
				rot = Rotate(input, rotate, 2);
			// Reverse "filterReverse".
			convL = convL + rot * reverse[filterGet];
			// Reverse "filterForwards".
			convH = convH + rot * forward[filterGet];
		}
		// Combine the results.  The original highpass code just collected and
		// then redistributed the "high" parts awkwardly, we won't do that.
		input = convL;
		// High-pass.
		highRet[level] = new FloatParallelArray(convH);
	}
	// Return the data.
	*lowRet = input;
}

#endif
