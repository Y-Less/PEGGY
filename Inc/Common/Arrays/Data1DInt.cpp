// Copyright (c) Microsoft Corporation.   All rights reserved.

#include "Data1DInt.h"

// cons
	Data1DInt::
	Data1DInt(
		const size_t
			width,
		const __int32 * const
			data
		) throw (...) :
		Data1D(width, sizeof (__int32), (void *)data)
{
}

// cons
	Data1DInt::
	Data1DInt(
		const size_t
			width
		) throw (...) :
		Data1D(width, DT_Int)
{
}

// cons
	Data1DInt::
	Data1DInt(
		const size_t
			width,
		const GenType
			gen
		) throw (...) :
		Data1D(width, DT_Int, gen)
{
}

// cons
	Data1DInt::
	Data1DInt(
		const size_t
			width,
		const GenType
			gen,
		const unsigned int
			seed
		) throw (...) :
		Data1D(width, DT_Int, gen, seed)
{
}

// cons
	Data1DInt::
	Data1DInt(
		const size_t
			width,
		const GenType
			gen,
		std::istream &
			src
		) throw (...) :
		Data1D(width, DT_Int, gen, src)
{
}

// cons
	Data1DInt::
	Data1DInt(
		const size_t
			width,
		const GenType
			gen,
		ds_callback
			func
		) throw (...) :
		Data1D(width, DT_Int, gen, func)
{
}

__int32
	Data1DInt::
	operator()(
		const size_t
			idx
		) const throw (...)
{
	if (idx >= GetWidth())
	{
		throw "Index out of bounds";
	}
	return ((__int32 *)GetData())[idx];
}

#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
	ParallelArrays::ParallelArray *
		Data1DInt::
		_ToAcceleratorArray(unsigned int) throw (...)
	{
		void *
			d = GetData();
		ParallelArrays::ParallelArray *
			fpa;
		if (d)
		{
			fpa = new ParallelArrays::IntParallelArray((__int32 *)d, GetWidth());
		}
		else
		{
			size_t
				r[1];
			r[0] = GetWidth();
			fpa = new ParallelArrays::IntParallelArray(0, r, 1);
		}
		return fpa;
	}
#endif

std::ostream &
	operator<<(
		std::ostream &
			str,
		const Data1DInt &
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
