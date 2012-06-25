// Copyright (c) Microsoft Corporation.   All rights reserved.
#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT

#include "Data1DFloat4.h"

// cons
	Data1DFloat4::
	Data1DFloat4(
		const size_t
			width,
		const ParallelArrays::Float4 * const
			data
		) throw (...) :
		Data1D(width, sizeof (ParallelArrays::Float4) * 4, (void *)data)
{
}

// cons
	Data1DFloat4::
	Data1DFloat4(
		const size_t
			width
		) throw (...) :
		Data1D(width, DT_Float4)
{
}

// cons
	Data1DFloat4::
	Data1DFloat4(
		const size_t
			width,
		const GenType
			gen
		) throw (...) :
		Data1D(width, DT_Float4, gen)
{
}

// cons
	Data1DFloat4::
	Data1DFloat4(
		const size_t
			width,
		const GenType
			gen,
		const unsigned int
			seed
		) throw (...) :
		Data1D(width, DT_Float4, gen, seed)
{
}

// cons
	Data1DFloat4::
	Data1DFloat4(
		const size_t
			width,
		const GenType
			gen,
		std::istream &
			src
		) throw (...) :
		Data1D(width, DT_Float4, gen, src)
{
}

// cons
	Data1DFloat4::
	Data1DFloat4(
		const size_t
			width,
		const GenType
			gen,
		ds_callback
			func
		) throw (...) :
		Data1D(width, DT_Float4, gen, func)
{
}

ParallelArrays::Float4
	Data1DFloat4::
	operator()(
		const size_t
			idx
		) const throw (...)
{
	if (idx >= GetWidth())
	{
		throw "Index out of bounds";
	}
	return ((ParallelArrays::Float4 *)GetData())[idx];
}

ParallelArrays::ParallelArray *
	Data1DFloat4::
	_ToAcceleratorArray(unsigned int) throw (...)
{
	void *
		d = GetData();
	ParallelArrays::ParallelArray *
		fpa;
	if (d)
	{
		fpa = new ParallelArrays::Float4ParallelArray((ParallelArrays::Float4 *)d, GetWidth());
	}
	else
	{
		size_t
			r[1];
		r[0] = GetWidth();
		ParallelArrays::Float4
			val(0.0, 0.0, 0.0, 0.0);
		fpa = new ParallelArrays::Float4ParallelArray(val, r, 1);
	}
	return fpa;
}

std::ostream &
	operator<<(
		std::ostream &
			str,
		const Data1DFloat4 &
			ds)
{
	size_t
		w = ds.GetWidth();
	for (size_t i = 0; i != w; ++i)
	{
			__m128
				f = ds(i).M128;
			str << i << " : " << f.m128_f32[0] << ", " << f.m128_f32[1] << ", " << f.m128_f32[2] << ", " << f.m128_f32[3] << std::endl;
	}
	return str;
}

#endif
