#if !defined NO_REFERENCE

#include "UDWTOptimised.h"

//#include <tracing.h>
//#include "UDWTReference.tmh"

// cons
	UDWTOptimised::
	UDWTOptimised() :
		UDWT(),
		ParallelReference()
{
	SetName("Opt ");
}

void
	UDWTOptimised::
	ConvClose() throw (...)
{
	delete [] m_lowPass;
	delete [] m_highPass;
}

void
	UDWTOptimised::
	ConvInit() throw (...)
{
	// Get the data we want.
	m_data = (float *)GetData().GetData();
	m_forward1 = (float *)GetForward1().GetData();
	m_reverse1 = (float *)GetReverse1().GetData();
	m_forward2 = (float *)GetForward2().GetData();
	m_reverse2 = (float *)GetReverse2().GetData();
	size_t
		length = GetWidth();
	//m_lowPass  = (float *)GetLowPass().GetData();
	//m_highPass = (float *)GetHighPass().GetData();
	m_lowPass = new float [length];
	m_highPass = new float [length * GetLevel()];
}

void
	fpconv2Opt(
		// The data that we are doing the convolution on.
		float *     xIn,
		size_t      xLength,
		// The amount we are moving up in the array by for each read.
		size_t      shift,
		float *     filterReverse,
		float *     filterForwards,
		int         filterLength,
		float *     xOutLow,
		float *     xOutHigh)
{
	// TODO: Modify this so it does the edge cases (for which "mod" is currently
	// required) separately and the main core very quickly (or, even quicker
	// than the very quickly it is being done now).  Done.
	size_t
		i = 0,
		end = xLength - filterLength * shift;
	for ( ; i != end; ++i)
	{
		float
			x0 = 0.0,
			x1 = 0.0;
		size_t
			read = i;
		for (size_t j = filterLength; j-- != 0; )
		{
			// TODO: Reverse the filter so this is better still.  Or not, I just
			// reversed the loop instead.
			x0 = x0 + xIn[read] * filterReverse[j]; // j
			x1 = x1 + xIn[read] * filterForwards[j]; // j
			read += shift;
		}
		xOutLow[i] = x0;
		xOutHigh[i] = x1;
	}
	// Tail-end cases (need "mod").
	for ( ; i != xLength; ++i)
	{
		float
			x0 = 0.0,
			x1 = 0.0;
		size_t
			read = i;
		for (size_t j = filterLength; j-- > 0; )
		{
			// TODO: Reverse the filter so this is better still.  Or not, I just
			// reversed the loop instead.
			x0 += xIn[read] * filterReverse[j]; // j
			x1 += xIn[read] * filterForwards[j]; // j
			read = (read + shift) % xLength;
		}
		xOutLow[i] = x0;
		xOutHigh[i] = x1;
	}
}

void
	bpconvOpt(
		float *     xOut,
		size_t      xLength,
		size_t      shift,
		float *     filterForwards,
		float *     filterReverse,
		int         filterLength,
		float *     xInLow,
		float *     xInHigh)
{
	size_t
		i = 0,
		start = filterLength * shift;
	for ( ; i != start; ++i)
	{
		float
			x0 = 0.0;
		int
			read = i;
		for (size_t j = 0; j != filterLength; ++j)
		{
			//x0 += /*xInLow[read] * filterForwards[j]; //+*/ xInHigh[read] * filterReverse[j];
			x0 += xInLow[read] * filterForwards[j] + xInHigh[read] * filterReverse[j];
			read = (read - shift) % xLength;
		}
		xOut[i] = x0;
	}
	// Tail-end cases (need "mod").
	for ( ; i != xLength; ++i)
	{
		float
			x0 = 0.0;
		int
			read = (int)i;
		for (size_t j = 0; j != filterLength; ++j)
		{
			//x0 += /*xInLow[read] * filterForwards[j]; //+*/ xInHigh[read] * filterReverse[j];
			x0 += xInLow[read] * filterForwards[j] + xInHigh[read] * filterReverse[j];
			read -= shift;
		}
		xOut[i] = x0;
	}
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
	UDWTOptimised::
	Execute() throw (...)
{
	float
		TEMP_threshold = 0.5f;
	MRDWT();
	//MIRDWT();
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
	//printf("threshold = %f\n", threshold);
	MIRDWT();
}

void
	UDWTOptimised::
	MRDWT()
{
	size_t
		filterLength = GetRadius(),
		inputLength = GetWidth(),
		levels = GetLevel(),
		shift = 1;
	//printf("Opt\n");
	float
		* lowpassInput = m_data,
		* lowpassOutput = GetStore(),
		* lowpassSwap = (float *)malloc(inputLength * sizeof (float));
	// Now do the main code once per level, each time rotating the input twice
	// as much as last time.
	for (size_t level = 0; level != levels; ++level)
	{
		// No longer need any of those stupid temporary arrays.  We can do it
		// all inline and frankly I expect a HUGE speed increase!  Hmm, we will
		// need ONE temporary array so that we can use the lowpass output as
		// both an input and an output, but that's a minor point and we just
		// need to make sure to use the copy as the input to only copy once.
		// OOORRR!!!  Even better just use two and/ constantly swap them over!
		// No copying or destructive updates at all!  Done...
		fpconv2Opt(lowpassInput, inputLength, shift, m_reverse1, m_forward1, filterLength, lowpassOutput, m_highPass + (level * inputLength));
		shift <<= 1;
		// Swap the lowpass data over.  And YES, I know I could use an XOR swap,
		// but really there's no point any more beyond showing off...
		lowpassInput = lowpassOutput;
		lowpassOutput = lowpassSwap;
		// Yes, swap is LAST and used NEXT time.  This is so that the first time
		// this code we loose the reference to "m_data", which we do NOT want in
		// the swapping.  Instead "lowpassSwap" is initialised to a new array.
		// I know "early optimisation is evil" and all that, but I am actually
		// TRYING to write an optimised version, that's the point of all of this
		// and actually it's not early as we already have a fully working
		// implementation.
		lowpassSwap = lowpassInput;
	}
	if (levels & 1)
	{
		// Odd number of levels.
		free(lowpassOutput);
	}
	else
	{
		// "DataStore()" wasn't the final target (shame).
		memcpy(lowpassOutput, lowpassSwap, inputLength * sizeof (float));
		free(lowpassSwap);
	}
	/*lowpassOutput = GetStore();
	for (int eye = 0; eye != 20; ++eye)
	{
		printf("%d = %d\n", eye, lowpassOutput[eye]);
	}*/
}

void
	UDWTOptimised::
	MIRDWT()
{
	size_t
		filterLength = GetRadius(),
		inputLength = GetWidth(),
		levels = GetLevel(),
		shift = 1 << (levels - 1);
	float
		* lowpassInput = GetStore(),
		* lowpassOutput = (float *)malloc(inputLength * sizeof (float)),
		* lowpassSwap;
	for (size_t level = levels; level-- != 0; )
	{
		bpconvOpt(lowpassOutput, inputLength, shift, m_forward2, m_reverse2, filterLength, lowpassInput, m_highPass + (level * inputLength));
		shift >>= 1;
		lowpassSwap = lowpassInput;
		lowpassInput = lowpassOutput;
		lowpassOutput = lowpassSwap;
	}
	if (levels & 1)
	{
		// If I'm right, this will balance out "MRDWT" as one of them will
		// always require a memcpy, but only one of them.
		memcpy(lowpassOutput, lowpassInput, inputLength * sizeof (float));
		free(lowpassInput);
	}
	else
	{
		free(lowpassSwap);
	}
}

#if 0
// By-hand optimisation, mainly constant propogation.
void
	UDWTOptimised::
	MIRDWT()
{
	//printf("f");
	float
		* lowpass_output = GetStore(),
		* filter_2_forwards = m_forward2,
		* filter_2_reverse = m_reverse2,
		* ydummyll,
		* ydummyhh,
		* xdummyl,
		//* yl = GetStore(),
		* yh = m_highPass; //(float *)m_data2->GetData();
	//printf("g");
	long
		i;
	int
		L = GetLevel(),
		actual_L,
		actual_n,
		c_o_a,
		n_c,
		n_cb,
		yl_length = GetWidth(),
		filter_length = GetRadius() * 2 + 1,
		filter_length_m1;
	//memcpy(lowpass_output, yl, yl_length * sizeof (float));
	int ic, sample_f;
	xdummyl = (float *)calloc(yl_length, sizeof (float));
	ydummyll = (float *)calloc(yl_length + filter_length - 1, sizeof (float));
	ydummyhh = (float *)calloc(yl_length + filter_length - 1, sizeof (float));
	/* analysis lowpass and highpass */





	size_t
		filterLength = GetRadius() * 2 + 1,
		inputLength = GetWidth(),
		levels = GetLevel(),
		shift = 1;
	//printf("Opt\n");
	float
		* lowpassInput = m_data,
		* lowpassOutput = GetStore(),
		* lowpassSwap = (float *)malloc(inputLength * sizeof (float));
	// Yey, more strange filter mangling to work through...
	filter_length_m1 = filter_length - 1;
	// 2 ^ (L - 1)
	// Original comment was wrong!
	shift = 1 << (levels - 1);
	actual_column = yl_length / sample_f;
	/* restore yl in x */
	/* main loop */
	// Again with their wierd nearly-OBOE.
	//for (actual_L = L; actual_L >= 1; --actual_L)
	for (level = levels; level-- > 0; )
	{
		/* actual (level dependent) column offset */
		c_o_a = yl_length * level;
		//c_o_a = yl_length * (actual_L - 1);
		/* go by rows */
		/* # of column blocks per row */
		column_blocks = yl_length / actual_column;
		/* loop within one row */
		for (column = 0; column < column_blocks; column++)
		{
			/* store in dummy variable */
			ic = column - sample_f;
			for  (i = 0; i < actual_column; i++)
			{
				ic = ic + sample_f;
				ydummyll[i + filter_length_m1] = lowpass_output[ic];
				ydummyhh[i + filter_length_m1] = yh[c_o_a + ic];
			}
			/* perform filtering lowpass/highpass */
			bpconvOpt(xdummyl, actual_column, filter_2_forwards, filter_2_reverse, filter_length, ydummyll, ydummyhh);
			/* restore dummy variables in matrices */
			ic = column - sample_f;
			for (i = 0; i < actual_column; i++)
			{
				ic = ic + sample_f;
				lowpass_output[ic] = xdummyl[i];
			}
		}
		shift >>= 1;
		//sample_f = sample_f / 2;
		actual_column = actual_column * 2;
	}
	free(xdummyl);
	free(ydummyll);
	free(ydummyhh);
	//memcpy(lowpass_output, yh, yl_length * sizeof (float));
}
#endif
#endif
