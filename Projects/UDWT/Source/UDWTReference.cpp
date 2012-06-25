#if !defined NO_REFERENCE

#include "UDWTReference.h"

//#include <tracing.h>
//#include "UDWTReference.tmh"

// cons
	UDWTReference::
	UDWTReference() :
		UDWT(),
		ParallelReference()
{
}

void
	UDWTReference::
	ConvClose() throw (...)
{
}

void
	UDWTReference::
	ConvInit() throw (...)
{
	//Sleep(1000);
	// Get the data we want.
	m_data = (float *)GetData().GetData();
	m_forward1 = (float *)GetForward1().GetData();
	m_reverse1 = (float *)GetReverse1().GetData();
	m_forward2 = (float *)GetForward2().GetData();
	m_reverse2 = (float *)GetReverse2().GetData();
	//size_t
	//	length = GetWidth();
	m_lowPass  = (float *)GetLowPass().GetData();
	m_highPass = (float *)GetHighPass().GetData();
	//m_tempLow = new float [length];
	//m_highPass = new float [length * GetLevel()];
}

void
	fpconv2(
		float *
			x_in,
		int
			lx,
		float *
			filter_reverse,
		float *
			filter_forwards,
		int
			filter_length,
		float *
			x_outl,
		float *
			x_outh)
{
	int
		i,
		j;
	float
		x0,
		x1;
	for (i = lx; i < lx + filter_length - 1; i++)
	{
		x_in[i] = x_in[i - lx];
	}
	for (i = 0; i < lx; i++)
	{
		x0 = 0;
		x1 = 0;
		for (j = 0; j < filter_length; j++)
		{
			x0 = x0 + x_in[i + j] * filter_reverse[filter_length - 1 - j]; // j
			x1 = x1 + x_in[i + j] * filter_forwards[filter_length - 1 - j]; // j
		}
		x_outl[i] = x0;
		x_outh[i] = x1;
	}
}

void
	bpconv(float * x_out, int lx, float * filter_2_forwards, float * filter_2_reverse, int filter_length, float * x_inl, float * x_inh)
{
	int i, j;
	float x0;

	for (i=filter_length-2; i > -1; i--)
	{
		x_inl[i] = x_inl[lx+i];
		x_inh[i] = x_inh[lx+i];
	}
	for (i=0; i<lx; i++)
	{
		x0 = 0;
		for (j=0; j<filter_length; j++)
		{
			//x0 = x0 + x_inh[j+i]*filter_2_reverse[filter_length-1-j];
			x0 = x0 + x_inl[j+i]*filter_2_forwards[filter_length-1-j] + x_inh[j+i]*filter_2_reverse[filter_length-1-j];
			//x0 = x0 + x_inl[j+i]*filter_2_forwards[filter_length-1-j] + x_inh[j+i]*filter_2_reverse[filter_length-1-j];
		}
		x_out[i] = x0;
	}
}

void
	Abs(float * data, size_t length)
{
	while (length--)
	{
		data[length] = abs(data[length]);
	}
}

void
	HardTh(float * data, size_t length, float threshold)
{
	//abs(input) > threshold ? input : 0
	while (length--)
	{
		data[length] = (abs(data[length]) > threshold) * data[length];
	}
}

int
	FloatCompare(
		void const *
			a,
		void const *
			b);

void
	UDWTReference::
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
	//memcpy(GetStore(), m_highPass + length, length * sizeof (float));
}

void
	UDWTReference::
	MRDWT()
{
	//printf("Ref\n");
	// 
	// ===============
	// | FIRST STAGE |
	// ===============
	// 
	// This has been DOWNGRADED from "double" to "float".
	float
		* filter_reverse = m_reverse1,
		* filter_forwards = m_forward1,
		* ydummyll,
		* ydummyhh,
		* xdummyl,
		* lowpass_output = GetStore(),
		* highpass_output = m_highPass,
		* input_signal = m_data;
	long
		i;
	int
		levels = GetLevel(),
		level,
		actual_column,
		c_o_a,
		column,
		column_blocks,
		ic,
		sample_f,
		input_columns = GetWidth(),
		filter_length = GetRadius();
	//std::cout << "Ref Store: " << lowpass_output << ", " << input_columns << std::endl;
	// Allocate a whole load of temporary arrays to store a single row or
	// column's worth of data.  Set to 0.
	ydummyll = (float *)calloc(input_columns, sizeof (float));
	ydummyhh = (float *)calloc(input_columns, sizeof (float));
	xdummyl = (float *)calloc(input_columns + filter_length - 1, sizeof (float));
	// Set up for the input.
	actual_column = 2 * input_columns;
	memcpy(lowpass_output, input_signal, input_columns * sizeof (float));
	//for (i = 0; i < input_columns; i++)
	//{
	//	lowpass_output[i] = input_signal[i];
	//}
	// Main loop.
	sample_f = 1;
	// WHY DID THIS GO FROM "1 - <= levels" when the only use is "-1"?
	for (level = 0; level != levels; ++level)
	{
		actual_column /= 2;
		// Actual (level dependent) column offset.
		c_o_a = level * input_columns;
		// Number of column blocks per row.
		column_blocks = input_columns / actual_column;
		// Loop within one row.
		for (column = 0; column < column_blocks; column++)
		{
			// Store in dummy variable.
			//ic = -sample_f + column;
			ic = column - sample_f;
			for (i = 0; i < actual_column; i++)
			{
				//ic = ic + sample_f;
				ic += sample_f;
				//xdummyl[i] = mat(lowpass_output, ir, ic);
				xdummyl[i] = lowpass_output[ic];
			}
			// Perform filtering lowpass/highpass.
			fpconv2(xdummyl, actual_column, filter_reverse, filter_forwards, filter_length, ydummyll, ydummyhh);
			// Restore dummy variables in matrices.
			//ic = -sample_f + column;
			ic = column - sample_f;
			//ic = column;
			for	(i = 0; i < actual_column; i++)
			{
				// WHY DOES THIS LINE EXIST?  Essentially:
				//   
				//   ic = column - sample_f + sample_f;
				//   
				// REPEATEDLY!  NO I AM A MORON!
				//ic = ic + sample_f;
				ic += sample_f;
				//mat(lowpass_output, ir, ic) = ydummyll[i];
				lowpass_output[ic] = ydummyll[i];
				//mat(highpass_output, ir, c_o_a + ic) = ydummyhh[i];
				highpass_output[c_o_a + ic] = ydummyhh[i];
			}
		}
		sample_f = sample_f * 2;
	}
	free(ydummyll);
	free(ydummyhh);
	free(xdummyl);
}

// By-hand optimisation, mainly constant propogation.
void
	UDWTReference::
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
		filter_length = GetRadius(),
		filter_length_m1;
	//memcpy(lowpass_output, yl, yl_length * sizeof (float));
	int ic, sample_f;
	xdummyl = (float *)calloc(yl_length, sizeof (float));
	ydummyll = (float *)calloc(yl_length + filter_length - 1, sizeof (float));
	ydummyhh = (float *)calloc(yl_length + filter_length - 1, sizeof (float));
	/* analysis lowpass and highpass */
	// Yey, more strange filter mangling to work through...
	filter_length_m1 = filter_length - 1;
	/* 2^L */
	sample_f = 1;
	for (i = 1; i < L; i++)
	{
		sample_f = sample_f * 2;
	}
	actual_n = yl_length / sample_f;
	/* restore yl in x */
	/* main loop */
	// Again with their wierd nearly-OBOE.
	//for (actual_L = L; actual_L >= 1; --actual_L)
	for (actual_L = L - 1; actual_L >= 0; --actual_L)
	{
		/* actual (level dependent) column offset */
		c_o_a = yl_length * actual_L;
		//c_o_a = yl_length * (actual_L - 1);
		/* go by rows */
		/* # of column blocks per row */
		n_cb = yl_length / actual_n;
		/* loop within one row */
		for (n_c = 0; n_c < n_cb; n_c++)
		{
			/* store in dummy variable */
			ic = -sample_f + n_c;
			for  (i = 0; i < actual_n; i++)
			{
				ic = ic + sample_f;
				ydummyll[i + filter_length_m1] = lowpass_output[ic];
				ydummyhh[i + filter_length_m1] = yh[c_o_a + ic];
			}
			/* perform filtering lowpass/highpass */
			bpconv(xdummyl, actual_n, filter_2_forwards, filter_2_reverse, filter_length, ydummyll, ydummyhh);
			/* restore dummy variables in matrices */
			ic = -sample_f + n_c;
			for (i = 0; i < actual_n; i++)
			{
				ic = ic + sample_f;
				lowpass_output[ic] = xdummyl[i];
			}
		}
		sample_f = sample_f / 2;
		actual_n = actual_n * 2;
	}
	free(xdummyl);
	free(ydummyll);
	free(ydummyhh);
	//memcpy(lowpass_output, yh, yl_length * sizeof (float));
}

#endif
