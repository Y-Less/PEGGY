// Copyright (c) Microsoft Corporation.   All rights reserved.
#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT

#include "Data2DFloat4.h"

// cons
	Data2DFloat4::
	Data2DFloat4(
		const size_t
			height,
		const size_t
			width
		) throw (...) :
		Data2D(height, width, DT_Float4)
{
}

// cons
	Data2DFloat4::
	Data2DFloat4(
		const size_t
			height,
		const size_t
			width,
		const ParallelArrays::Float4 * const
			data
		) throw (...) :
		Data2D(height, width, sizeof (ParallelArrays::Float4) * 4, (void *)data)
{
}

// cons
	Data2DFloat4::
	Data2DFloat4(
		const size_t
			height,
		const size_t
			width,
		const GenType
			gen
		) throw (...) :
		Data2D(height, width, DT_Float4, gen)
{
}

// cons
	Data2DFloat4::
	Data2DFloat4(
		const size_t
			height,
		const size_t
			width,
		const GenType
			gen,
		const unsigned int
			seed
		) throw (...) :
		Data2D(height, width, DT_Float4, gen, seed)
{
}

// cons
	Data2DFloat4::
	Data2DFloat4(
		const size_t
			height,
		const size_t
			width,
		const GenType
			gen,
		std::istream &
			src
		) throw (...) :
		Data2D(height, width, DT_Float4, gen, src)
{
}

// cons
	Data2DFloat4::
	Data2DFloat4(
		const size_t
			height,
		const size_t
			width,
		const GenType
			gen,
		ds_callback
			func
		) throw (...) :
		Data2D(height, width, DT_Float4, gen, func)
{
}

ParallelArrays::Float4
	Data2DFloat4::
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
	return ((ParallelArrays::Float4 *)GetData())[(idx0 * GetWidth()) + idx1];
}

#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
	ParallelArrays::ParallelArray *
		Data2DFloat4::
		_ToAcceleratorArray(unsigned int) throw (...)
	{
		void *
			d = GetData();
		ParallelArrays::ParallelArray *
			fpa;
		if (d)
		{
			fpa = new ParallelArrays::Float4ParallelArray((ParallelArrays::Float4 *)d, GetHeight(), GetWidth());
		}
		else
		{
			size_t
				r[2];
			r[0] = GetHeight();
			r[1] = GetWidth();
			ParallelArrays::Float4
				val(0.0, 0.0, 0.0, 0.0);
			fpa = new ParallelArrays::Float4ParallelArray(val, r, 2);
		}
		return fpa;
	}
#endif

std::ostream &
	operator<<(
		std::ostream &
			str,
		const Data2DFloat4 &
			ds)
{
	size_t
		w = ds.GetWidth(),
		h = ds.GetHeight();
	for (size_t j = 0; j != h; ++j)
	{
		for (size_t i = 0; i != w; ++i)
		{
			__m128
				f = ds(j, i).M128;
			str << j << ", " << i << " : " << f.m128_f32[0] << ", " << f.m128_f32[1] << ", " << f.m128_f32[2] << ", " << f.m128_f32[3] << std::endl;
		}
	}
	return str;
}

#endif
