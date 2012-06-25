// Copyright (c) Microsoft Corporation.   All rights reserved.

#include "Data2DInt.h"

// cons
	Data2DInt::
	Data2DInt(
		const size_t
			height,
		const size_t
			width
		) throw (...) :
		Data2D(height, width, DT_Int)
{
}

// cons
	Data2DInt::
	Data2DInt(
		const size_t
			height,
		const size_t
			width,
		const __int32 * const
			data
		) throw (...) :
		Data2D(height, width, sizeof (__int32), (void *)data)
{
}

// cons
	Data2DInt::
	Data2DInt(
		const size_t
			height,
		const size_t
			width,
		const GenType
			gen
		) throw (...) :
		Data2D(height, width, DT_Int, gen)
{
}

// cons
	Data2DInt::
	Data2DInt(
		const size_t
			height,
		const size_t
			width,
		const GenType
			gen,
		const unsigned int
			seed
		) throw (...) :
		Data2D(height, width, DT_Int, gen, seed)
{
}

// cons
	Data2DInt::
	Data2DInt(
		const size_t
			height,
		const size_t
			width,
		const GenType
			gen,
		std::istream &
			src
		) throw (...) :
		Data2D(height, width, DT_Int, gen, src)
{
}

// cons
	Data2DInt::
	Data2DInt(
		const size_t
			height,
		const size_t
			width,
		const GenType
			gen,
		ds_callback
			func
		) throw (...) :
		Data2D(height, width, DT_Int, gen, func)
{
}

__int32
	Data2DInt::
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
	return ((__int32 *)GetData())[(idx0 * GetWidth()) + idx1];
}

#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
	ParallelArrays::ParallelArray *
		Data2DInt::
		_ToAcceleratorArray(unsigned int) throw (...)
	{
		void *
			d = GetData();
		ParallelArrays::ParallelArray *
			fpa;
		if (d)
		{
			fpa = new ParallelArrays::IntParallelArray((__int32 *)d, GetHeight(), GetWidth());
		}
		else
		{
			size_t
				r[2];
			r[0] = GetHeight();
			r[1] = GetWidth();
			fpa = new ParallelArrays::IntParallelArray(0, r, 2);
		}
		return fpa;
	}
#endif

std::ostream &
	operator<<(
		std::ostream &
			str,
		const Data2DInt &
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
