// Copyright (c) Microsoft Corporation.   All rights reserved.

#include "Data1DBool.h"

// cons
	Data1DBool::
	Data1DBool(
		const size_t
			width,
		const bool * const
			data
		) throw (...) :
		Data1D(width, sizeof (bool), (void *)data)
{
}

// cons
	Data1DBool::
	Data1DBool(
		const size_t
			width
		) throw (...) :
		Data1D(width, DT_Bool)
{
}

// cons
	Data1DBool::
	Data1DBool(
		const size_t
			width,
		const GenType
			gen
		) throw (...) :
		Data1D(width, DT_Bool, gen)
{
}

// cons
	Data1DBool::
	Data1DBool(
		const size_t
			width,
		const GenType
			gen,
		const unsigned int
			seed
		) throw (...) :
		Data1D(width, DT_Bool, gen, seed)
{
}

// cons
	Data1DBool::
	Data1DBool(
		const size_t
			width,
		const GenType
			gen,
		std::istream &
			src
		) throw (...) :
		Data1D(width, DT_Bool, gen, src)
{
}

// cons
	Data1DBool::
	Data1DBool(
		const size_t
			width,
		const GenType
			gen,
		ds_callback
			func
		) throw (...) :
		Data1D(width, DT_Bool, gen, func)
{
}

bool
	Data1DBool::
	operator()(
		const size_t
			idx
		) const throw (...)
{
	if (idx >= GetWidth())
	{
		throw "Index out of bounds";
	}
	return ((bool *)GetData())[idx];
}

#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
	ParallelArrays::ParallelArray *
		Data1DBool::
		_ToAcceleratorArray(unsigned int) throw (...)
	{
		void *
			d = GetData();
		ParallelArrays::ParallelArray *
			fpa;
		if (d)
		{
			fpa = new ParallelArrays::BoolParallelArray((bool *)d, GetWidth());
		}
		else
		{
			size_t
				r[1];
			r[0] = GetWidth();
			fpa = new ParallelArrays::BoolParallelArray(false, r, 1);
		}
		return fpa;
	}
#endif

std::ostream &
	operator<<(
		std::ostream &
			str,
		const Data1DBool &
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
