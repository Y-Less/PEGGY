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
	delete [] m_lowPass;
	delete [] m_highPass;
}

void
	UDWTReference::
	ConvInit() throw (...)
{
	// Get the data we want.
	m_data = (ValueType *)GetData().GetData();
	m_forward1 = (ValueType *)GetForward1().GetData();
	m_reverse1 = (ValueType *)GetReverse1().GetData();
	m_forward2 = (ValueType *)GetForward2().GetData();
	m_reverse2 = (ValueType *)GetReverse2().GetData();
	size_t
		length = GetWidth();
	//m_lowPass  = (ValueType *)GetLowPass().GetData();
	//m_highPass = (ValueType *)GetHighPass().GetData();
	m_lowPass = new ValueType [length];
	m_highPass = new ValueType [length * GetLevel()];
}

void
	fpconv2(
		ValueType *
			x_in,
		int
			lx,
		ValueType *
			filter_reverse,
		ValueType *
			filter_forwards,
		int
			filter_length,
		ValueType *
			x_outl,
		ValueType *
			x_outh)
{
	int
		i,
		j;
	ValueType
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
	bpconv(ValueType * x_out, int lx, ValueType * filter_2_forwards, ValueType * filter_2_reverse, int filter_length, ValueType * x_inl, ValueType * x_inh)
{
	int i, j;
	ValueType x0;

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
	Abs(ValueType * data, size_t length)
{
	while (length--)
	{
		data[length] = abs(data[length]);
	}
}

void
	HardTh(ValueType * data, size_t length, ValueType threshold)
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
	ValueType
		TEMP_threshold = 0.5f;
	MRDWT();
	//MIRDWT();
	size_t
		length = GetWidth();
	memcpy(m_lowPass, m_highPass, length * sizeof (ValueType));
	Abs(m_lowPass, length);
	qsort(m_lowPass, length, sizeof (ValueType), FloatCompare);
	ValueType
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
	UDWTReference::
	MRDWT()
{
	// 
	// ===============
	// | FIRST STAGE |
	// ===============
	// 
	// This has been DOWNGRADED from "ValueType" to "ValueType".
	ValueType
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
		filter_length = GetRadius() * 2 + 1;
	//std::cout << "Ref Store: " << lowpass_output << ", " << input_columns << std::endl;
	// Allocate a whole load of temporary arrays to store a single row or
	// column's worth of data.  Set to 0.
	ydummyll = (ValueType *)calloc(input_columns, sizeof (ValueType));
	ydummyhh = (ValueType *)calloc(input_columns, sizeof (ValueType));
	xdummyl = (ValueType *)calloc(input_columns + filter_length - 1, sizeof (ValueType));
	// Set up for the input.
	actual_column = 2 * input_columns;
	memcpy(lowpass_output, input_signal, input_columns * sizeof (ValueType));
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
	/*lowpass_output[0] = 42.0f;
	lowpass_output[1] = 42.0f;
	lowpass_output[2] = 42.0f;
	lowpass_output[3] = 42.0f;
	lowpass_output[4] = 42.0f;
	lowpass_output[5] = 42.0f;
	lowpass_output[6] = 42.0f;
	lowpass_output[7] = 42.0f;
	lowpass_output[8] = 42.0f;
	lowpass_output[9] = 42.0f;
	lowpass_output[10] = 42.0f;
	lowpass_output[11] = 42.0f;
	lowpass_output[12] = 42.0f;
	lowpass_output[13] = 42.0f;
	lowpass_output[14] = 42.0f;
	lowpass_output[15] = 42.0f;
	lowpass_output[16] = 42.0f;
	lowpass_output[17] = 42.0f;
	lowpass_output[18] = 42.0f;
	lowpass_output[19] = 42.0f;
	lowpass_output[20] = 42.0f;*/
}

// By-hand optimisation, mainly constant propogation.
void
	UDWTReference::
	MIRDWT()
{
	//printf("f");
	ValueType
		* lowpass_output = GetStore(),
		* filter_2_forwards = m_forward2,
		* filter_2_reverse = m_reverse2,
		* ydummyll,
		* ydummyhh,
		* xdummyl,
		//* yl = GetStore(),
		* yh = m_highPass; //(ValueType *)m_data2->GetData();
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
	//memcpy(lowpass_output, yl, yl_length * sizeof (ValueType));
	int ic, sample_f;
	xdummyl = (ValueType *)calloc(yl_length, sizeof (ValueType));
	ydummyll = (ValueType *)calloc(yl_length + filter_length - 1, sizeof (ValueType));
	ydummyhh = (ValueType *)calloc(yl_length + filter_length - 1, sizeof (ValueType));
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
	//memcpy(lowpass_output, yh, yl_length * sizeof (ValueType));
}

#endif
