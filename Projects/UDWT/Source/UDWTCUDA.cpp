#if !defined NO_CUDA

#include "UDWTCUDA.h"
#include <cuda_runtime_api.h>

//#include <tracing.h>
//#include "UDWTReference.tmh"

extern "C" void
	UDWTCUDAFilter(
		float *     f1,
		float *     r1,
		float *     f2,
		float *     r2,
		int         len);

extern "C" void
	DoMRDWT(
		int     filterLength,
		int     inputLength,
		int     levels,
		float * lowpassInput,
		float * lowpassOutput,
		float * highpassOutput);

extern "C" void
	DoMIRDWT(
		int     filterLength,
		int     inputLength,
		int     levels,
		float * lowpassInput,
		float * lowpassOutput,
		float * highpassInput);

// cons
	UDWTCUDA::
	UDWTCUDA() :
		UDWT(),
		ParallelCUDADriver()
{
	SetName("CUDA ");
}

void
	UDWTCUDA::
	ConvClose() throw (...)
{
	//delete [] m_lowPass;
	//delete [] m_highPass;
}

void
	UDWTCUDA::
	ConvInit() throw (...)
{
	//Sleep(1000);
	// Get the data we want.
	m_data = (float *)GetData().ToCUDAArray();
	//m_data = (float *)GetData().GetData();
	UDWTCUDAFilter(
		(float *)GetForward1().GetData(),
		(float *)GetReverse1().GetData(),
		(float *)GetForward2().GetData(),
		(float *)GetReverse2().GetData(),
		GetRadius());
	//size_t
	//	length = GetWidth();
	m_lowPass  = (float *)GetLowPass().GetData();
	m_highPass = (float *)GetHighPass().GetData();
	m_lowPassC  = (float *)GetLowPass().ToCUDAArray();
	m_highPassC = (float *)GetHighPass().ToCUDAArray();
}

/*void
	bpconvCUDA(float * x_out, int lx, float * filter_2_forwards, float * filter_2_reverse, int filter_length, float * x_inl, float * x_inh)
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
}*/

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
	UDWTCUDA::
	Execute() throw (...)
{
	float
		TEMP_threshold = 0.5f;
	//MRDWT();
	int
		width = GetWidth(),
		size = width * sizeof (float),
		levels = GetLevel();
	DoMRDWT(8, width, levels, m_data, m_lowPassC, m_highPassC);
	cudaMemcpy(GetStore(), m_lowPassC, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(m_highPass, m_highPassC, levels * size, cudaMemcpyDeviceToHost);
	
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
	//MIRDWT();
	cudaMemcpy(m_lowPassC, GetStore(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(m_highPassC, m_highPass, levels * size, cudaMemcpyHostToDevice);
	DoMIRDWT(8, width, levels, m_lowPassC, m_data, m_highPassC);
	cudaMemcpy(GetStore(), m_data, size, cudaMemcpyDeviceToHost);
}

#if 0
// By-hand optimisation, mainly constant propogation.
void
	UDWTCUDA::
	MIRDWT()
{
	//printf("f");
	float
		* lowpass_output = GetStore(),
		//* filter_2_forwards = m_forward2,
		//* filter_2_reverse = m_reverse2,
		* filter_2_forwards = (float *)GetForward2().GetData(),
		* filter_2_reverse = (float *)GetReverse2().GetData(),
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
			bpconvCUDA(xdummyl, actual_n, filter_2_forwards, filter_2_reverse, filter_length, ydummyll, ydummyhh);
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
#endif
