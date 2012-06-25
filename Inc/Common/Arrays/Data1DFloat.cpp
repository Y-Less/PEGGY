// Copyright (c) Microsoft Corporation.   All rights reserved.

#include "Data1DFloat.h"

// cons
	Data1DFloat::
	Data1DFloat(
		const size_t
			width,
		const float * const
			data
		) throw (...) :
		Data1D(width, sizeof (float), (void *)data)
{
}

// cons
	Data1DFloat::
	Data1DFloat(
		const size_t
			width
		) throw (...) :
		Data1D(width, DT_Float)
{
}

 /*!
 * We may not always want RANDOM data, just SOME data.
 * Random data is good for testing, but not precise
 * benchmarking or debugging.
 */
// cons
	Data1DFloat::
	Data1DFloat(
		const size_t
			width,
		const GenType
			gen
		) throw (...) :
		Data1D(width, DT_Float, gen)
{
}

// cons
	Data1DFloat::
	Data1DFloat(
		const size_t
			width,
		const GenType
			gen,
		const unsigned int
			seed
		) throw (...) :
		Data1D(width, DT_Float, gen, seed)
{
}

 /*!
 * We may not always want RANDOM data, just SOME data.
 * Random data is good for testing, but not precise
 * benchmarking or debugging.
 */
// cons
	Data1DFloat::
	Data1DFloat(
		const size_t
			width,
		const GenType
			gen,
		std::istream &
			src
		) throw (...) :
		Data1D(width, DT_Float, gen, src)
{
}

// cons
	Data1DFloat::
	Data1DFloat(
		const size_t
			width,
		const GenType
			gen,
		ds_callback
			func
		) throw (...) :
		Data1D(width, DT_Float, gen, func)
{
}

float
	Data1DFloat::
	operator()(
		const size_t
			idx
		) const throw (...)
{
	if (idx >= GetWidth())
	{
		throw "Index out of bounds";
	}
	return ((float *)GetData())[idx];
}

#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
	ParallelArrays::ParallelArray *
		Data1DFloat::
		_ToAcceleratorArray(
			unsigned int
				apron) throw (...)
	{
		void *
			d = GetData();
		size_t
			width = GetWidth();
		ParallelArrays::ParallelArray *
			fpa;
		if (width > m_scMaxWidth)
		{
			size_t const
				usable = m_scMaxWidth - (apron * 2),
				rows = (width - 1) / usable + 1;
			if (d)
			{
				float *
					dest = new float [rows * m_scMaxWidth];
				// If we had inline functions we could not do the memory
				// allocation here and instead do it in "WrapData", yet still be
				// able to specify more of the logic here and USE the memory.
				WrapData(dest, (float *)d, width, apron);
				fpa = new ParallelArrays::FloatParallelArray(dest, rows, m_scMaxWidth);
				delete [] dest;
			}
			else
			{
				size_t
					r[2] = {rows, m_scMaxWidth};
				fpa = new ParallelArrays::FloatParallelArray(0.0, r, 2);
			}
		}
		else
		{
			if (d)
			{
				fpa = new ParallelArrays::FloatParallelArray((float *)d, width);
			}
			else
			{
				size_t
					r[2] = {width, 0};
				fpa = new ParallelArrays::FloatParallelArray(0.0, r, 1);
			}
		}
		return fpa;
	}
#endif

std::ostream &
	operator<<(
		std::ostream &
			str,
		const Data1DFloat &
			ds)
{
	size_t
		w = ds.GetWidth();
	for (size_t i = 0; i != w; ++i)
	{
		str << i << " : " << ds(i) << std::endl;
	}
	return str;
}
