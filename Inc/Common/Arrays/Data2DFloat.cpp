// Copyright (c) Microsoft Corporation.   All rights reserved.

#include "Data2DFloat.h"

// cons
	Data2DFloat::
	Data2DFloat(
		const size_t
			height,
		const size_t
			width
		) throw (...) :
		Data2D(height, width, DT_Float)
{
}

// cons
	Data2DFloat::
	Data2DFloat(
		const size_t
			height,
		const size_t
			width,
		const float * const
			data
		) throw (...) :
		Data2D(height, width, sizeof (float), (void *)data)
{
}

// cons
	Data2DFloat::
	Data2DFloat(
		const size_t
			height,
		const size_t
			width,
		const GenType
			gen
		) throw (...) :
		Data2D(height, width, DT_Float, gen)
{
}

// cons
	Data2DFloat::
	Data2DFloat(
		const size_t
			height,
		const size_t
			width,
		const GenType
			gen,
		const unsigned int
			seed
		) throw (...) :
		Data2D(height, width, DT_Float, gen, seed)
{
}

// cons
	Data2DFloat::
	Data2DFloat(
		const size_t
			height,
		const size_t
			width,
		const GenType
			gen,
		std::istream &
			src
		) throw (...) :
		Data2D(height, width, DT_Float, gen, src)
{
}

// cons
	Data2DFloat::
	Data2DFloat(
		const size_t
			height,
		const size_t
			width,
		const GenType
			gen,
		ds_callback
			func
		) throw (...) :
		Data2D(height, width, DT_Float, gen, func)
{
}

float
	Data2DFloat::
	operator()(
		const size_t
			idx0,
		const size_t
			idx1
		) const throw (...)
{
	if (idx0 >= GetHeight() || idx1 >= GetWidth())
	{
		throw "Index out of bounds";
	}
	return ((float *)GetData())[(idx0 * GetWidth()) + idx1];
}

#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
	ParallelArrays::ParallelArray *
		Data2DFloat::
		_ToAcceleratorArray(unsigned int) throw (...)
	{
		void *
			d = GetData();
		ParallelArrays::ParallelArray *
			fpa;
		if (d)
		{
			fpa = new ParallelArrays::FloatParallelArray((float *)d, GetHeight(), GetWidth());
		}
		else
		{
			size_t
				r[2];
			r[0] = GetHeight();
			r[1] = GetWidth();
			fpa = new ParallelArrays::FloatParallelArray(0.0, r, 2);
		}
		return fpa;
	}
#endif

std::ostream &
	operator<<(
		std::ostream &
			str,
		const Data2DFloat &
			ds)
{
	size_t
		w = ds.GetWidth(),
		h = ds.GetHeight();
	for (size_t j = 0; j != h; ++j)
	{
		for (size_t i = 0; i != w; ++i)
		{
			str << j << ", " << i << " : " << ds(j, i) << std::endl;
		}
	}
	return str;
}
