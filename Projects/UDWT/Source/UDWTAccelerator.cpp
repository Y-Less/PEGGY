#if !defined NO_ACCELERATOR

#include "UDWTAccelerator.h"
#include <iostream>

#include <stdlib.h>

//#include <tracing.h>
//#include "UDWTAccelerator.tmh"

using namespace ParallelArrays;

static size_t const
	scMaxWidth = 4096;

extern float
	gLargeData[4096 * 8192];

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
		m_lowPass(0),
		m_highPass(0),
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
	//
	//m_lowPass  = (float *)GetLowPass().GetData();
	//m_highPass = (float *)GetHighPass().GetData();
	// The accelerator version can't set up here as it needs to zero the
	// accelerator array every itteration.
	size_t
		length = GetWidth(),
		apron = 32,
		usable = scMaxWidth - (apron * 2),
		rows = (length - 1) / usable + 1;
	m_highPass = new float [rows * scMaxWidth * GetLevel()];
	m_lowPass = new float [rows * scMaxWidth];
}

void
	UDWTAccelerator::
	ConvClose() throw (...)
{
	//delete [] m_dest;
	delete [] m_highPass;
	delete [] m_lowPass;
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
	size_t
		length = GetWidth(),
		apron = 32,
		usable = scMaxWidth - (apron * 2),
		rows = (length - 1) / usable + 1,
		levels = GetLevel();
	float
		TEMP_threshold = 0.5f;
	// Parallel arrays for the calculation.
	FloatParallelArray
		xl,
		xh; //,
		//xp;
	// Based on the RWT:
	//  
	//  http://dsp.rice.edu/software/rice-wavelet-toolbox
	//  
	// Don't forget that the arrays must be an exact power of two size.
	//printf("1");
	MRDWT(m_data, &xl, &xh, 0); // &xp);
	//printf("2");
	// This bit can't currently be done in Accelerator - we need a sort and
	// median function.  The latter can be done, the former POSSIBLY can be, but
	// it would be very hard.
	//float
	//	* store = (float *)GetStore();
	// Find the threshold from the first chunk of the high parts.
	GetTarget().ToArray(xh, m_highPass, rows * levels, scMaxWidth, scMaxWidth * sizeof (float));
	//GetTarget().ToArray(xp, m_dest, rows, scMaxWidth, scMaxWidth * sizeof (float));
	//printf("4");
	Data1D::Recompress(m_lowPass, m_highPass, length, apron);
	Data1D::Recompress(gLargeData, m_highPass, length, apron);
	//printf("3");
	// Get the high threshold.
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
	GetTarget().ToArray(xl, m_lowPass, rows, scMaxWidth, scMaxWidth * sizeof (float));
	//GetLowPass().CleanGPU();
	//GetHighPass().CleanGPU();
	//printf("threshold = %f\n", threshold);
	//printf("1");
	HardTh(m_highPass, scMaxWidth * rows * GetLevel(), threshold);
	HardTh(m_lowPass, scMaxWidth * rows, threshold);
//	xl = dynamic_cast<FloatParallelArray &>(GetLowPass().ToAcceleratorArray());
//	xh = dynamic_cast<FloatParallelArray &>(GetHighPass().ToAcceleratorArray());
	//printf("2");
	FloatParallelArray
		*xl2 = new ParallelArrays::FloatParallelArray(m_lowPass, rows, scMaxWidth),
		*xh2 = new ParallelArrays::FloatParallelArray(m_highPass, rows * levels, scMaxWidth);
	//printf("3");
	//xh = HardTh(xh, threshold);
	//xl = HardTh(xl, threshold);
	// Get the final result.
	//printf("4");
	GetTarget().ToArray(MIRDWT(*xl2, *xh2), m_lowPass, rows, scMaxWidth, scMaxWidth * sizeof (float));
//	GetTarget().ToArray(xl, dest, rows, scMaxWidth, scMaxWidth * sizeof (float));
	Data1D::Recompress(GetStore(), m_lowPass, length, apron);
	//printf("1");
	delete xl2;
	delete xh2;
	//printf("0");
	
/*	const size_t
		cSize = 10;
	size_t
		dims[] = {cSize};
	float
		v3[] = {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0},
		v5[] = {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0};
	FloatParallelArray
		x51(5, dims, 1),
		x52(v5, cSize),
		x31(3, dims, 1),
		x32(v3, cSize);
	// This bit can't currently be done in Accelerator - we need a sort and
	// median function.  The latter can be done, the former POSSIBLY can be, but
	// it would be very hard.
	float
		* store = (float *)GetStore();
	// Find the threshold from the first chunk of the high parts.
	GetTarget().ToArray(ParallelArrays::Pow(x51, x31), store, cSize);
	std::cout << "51^31" << std::endl;
	for (size_t i = 0; i != cSize; ++i)
	{
		std::cout << i << ": " << store[i] << std::endl;
	}
	GetTarget().ToArray(ParallelArrays::Pow(x51, x32), store, cSize);
	std::cout << "51^32" << std::endl;
	for (size_t i = 0; i != cSize; ++i)
	{
		std::cout << i << ": " << store[i] << std::endl;
	}
	GetTarget().ToArray(ParallelArrays::Pow(x52, x31), store, cSize);
	std::cout << "52^31" << std::endl;
	for (size_t i = 0; i != cSize; ++i)
	{
		std::cout << i << ": " << store[i] << std::endl;
	}
	GetTarget().ToArray(ParallelArrays::Pow(x52, x32), store, cSize);
	std::cout << "52^32" << std::endl;
	for (size_t i = 0; i != cSize; ++i)
	{
		std::cout << i << ": " << store[i] << std::endl;
	}*/
	/*size_t
		dims[] = {rows, scMaxWidth},
		filterLength = GetRadius() * 2 + 1,
		level = 0;
	SectionSpecifier
		sectionX(0, scMaxWidth), //scMaxWidth),
		sectionY(level * rows, rows); //levels * rows);
	printf("Section: (%d, %d) - (%d, %d)\n", 0, level * rows, scMaxWidth, level * rows + rows);
	FloatParallelArray
		convL(0.0f, dims, 2),
		convH = Section(xh, sectionY, sectionX);*/
	/*GetTarget().ToArray(xh, m_dest, rows, scMaxWidth, scMaxWidth * sizeof (float));
	printf("FIRST:\n");
	for (size_t lp = 0; lp != rows; ++lp)
	{
		for (size_t lo = 0; lo != scMaxWidth; ++lo)
		{
			printf("%d, %d = %f\n", lp, lo, m_dest[lp * scMaxWidth + lo]);
		}
	}*/
	//GetTarget().ToArray(convH, m_dest, rows, scMaxWidth, scMaxWidth * sizeof (float));
	/*printf("SECOND:\n");
	for (size_t lp = 0; lp != rows; ++lp)
	{
		for (size_t lo = 0; lo != scMaxWidth; ++lo)
		{
			printf("%d, %d = %f\n", lp, lo, m_dest[lp * scMaxWidth + lo]);
		}
	}*/
//	GetTarget().ToArray(xl, dest, rows, scMaxWidth, scMaxWidth * sizeof (float));
	//Data1D::Recompress(store, m_dest, length, apron);
}

/*FloatParallelArray
	UDWTAccelerator::
	HardTh(
		FloatParallelArray
			input,
		float const
			threshold)
{
	// Filter only elements over a certain value.
	size_t
		dims = input.GetRank(),
		len[2];
	input.Dimensions(len, dims);
	// Create a comparison array against which to test the threshold.
	FloatParallelArray
		tarr(threshold, len, dims),
		zero(0.0, len, dims);
	// abs(input) > threshold ? input : 0
	return Cond(Abs(input) > tarr, input, zero);
}*/

void
	UDWTAccelerator::
	MRDWT(
		FloatParallelArray
			input,
		FloatParallelArray *
			lowRet,
		FloatParallelArray *
			highRet,
		FloatParallelArray *
			partialRet)
{
	size_t
		length = GetWidth();
	size_t
		apron = 32,
		usable = scMaxWidth - (apron * 2),
		rows = (length - 1) / usable + 1,
		levels = GetLevel(),
		// Size of the single huge HighPass array.
		dimsLong[] = {rows * levels, scMaxWidth},
		dims[] = {rows, scMaxWidth},
		filterLength = GetRadius(),
		expBef[] = {0, 0};
	size_t
		expAft[] = {0, 0};
	FloatParallelArray
		highPass(0.0f, dimsLong, 2);
	int
		rotate[2] = {0, 0};
	bool *
		hpCond = (bool *)calloc(levels, sizeof (bool));
	float
		* forward = (float *)GetForward1().GetData(),
		* reverse = (float *)GetReverse1().GetData();
	for (size_t level = 0; level != levels; ++level)
	{
		FloatParallelArray
			convL(0.0f, dims, 2),
			convH(0.0f, dims, 2);
		hpCond[level] = 1;
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
		//if (!level)
		//{
		//	// Just the first "high pass" result for use in the threshold code.
		//	*partialRet = Abs(convH);
		//}
		// Combine the results.  Low-pass is easy, high-pass not so much.
		// Low-pass.
		input = convL;
		// High-pass.
		BoolParallelArray
			cond(hpCond, levels, 1);
		// Create a boolean array with a "1" at the level we want, then stretch
		// the array so that there are "1"s at every required array position.
		cond = Stretch(cond, dims, 2);
		// Replicate "convH" multiple times.
		expAft[0] = rows * (levels - 1);
		FloatParallelArray
			convLong = Expand(convH, expBef, expAft, 2);
		// Now combine "highPass" and "convH" according to "cond".
		highPass = Cond(cond, convLong, highPass);
		hpCond[level] = 0;
	}
	// Return the data.
	*lowRet = input;
	*highRet = highPass;
}

FloatParallelArray
	UDWTAccelerator::
	MIRDWT(
		FloatParallelArray low,
		FloatParallelArray high)
{
	//printf("1\n");
	size_t
		length = GetWidth();
	size_t
		apron = 32,
		usable = scMaxWidth - (apron * 2),
		rows = (length - 1) / usable + 1,
		levels = GetLevel(),
		// Size of the 1D array wrapped in a 2D array for greater value use.
		dims[] = {rows, scMaxWidth},
		filterLength = GetRadius();
	//printf("length = %d, rows = %d, usable = %d\n", length, rows, usable);
	int
		rotate[2] = {0, 0};
	bool *
		hpCond = (bool *)calloc(levels, sizeof (bool));
	//printf("1\n");
	// Get the two filters.
	float
		* forward = (float *)GetForward2().GetData(),
		* reverse = (float *)GetReverse2().GetData();
	//printf("1\n");
	for (size_t level = levels; level-- != 0; )
	{
		//printf("2 %d %d\n", level, rows);
		//printf("2 %d %d\n", high.GetLength(0), high.GetLength(1));
		// Get a single section of the "high" array.
		SectionSpecifier
			sectionX(0, scMaxWidth),
			sectionY(level * rows, rows);
		//printf("Section: (%d, %d) - (%d, %d)\n", 0, level * rows, scMaxWidth, level * rows + rows);
		FloatParallelArray
			convL(0.0f, dims, 2),
			convH = Section(high, sectionY, sectionX);
		//printf("%d %d\n", high.GetLength(0), high.GetLength(1));
		//printf("%d %d\n", low.GetLength(0), low.GetLength(1));
		//printf("%d %d\n", convH.GetLength(0), convH.GetLength(1));
		//printf("3\n");
		// Loop through every element of the filter for the convolution.
		for (size_t filterCur = 0; filterCur != filterLength; ++filterCur)
		{
			// Rotate by on offset.
			rotate[1] = -(int)(filterCur << level);
			// Reverse "filterReverse".
			// THIS CODE DOES NOT USE FILTERGET, THE MECHANICS MEAN THAT THE
			// FILTER USE IS REVERSED.
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

#endif
