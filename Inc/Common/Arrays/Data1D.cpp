// Copyright (c) Microsoft Corporation.   All rights reserved.

#include "Data1D.h"

// cons
	Data1D::
	Data1D(
		const size_t
			width,
		const size_t
			dw,
		const void * const
			data
		) throw (...) :
		DataStore(width * dw, data),
		m_width(width)/*,
		m_wrapped(0)*/
{
}

// cons
	Data1D::
	Data1D(
		const size_t
			width,
		const DataType
			type
		) throw (...) :
		DataStore(width, type),
		m_width(width)/*,
		m_wrapped(0)*/
{
}

 /*!
 * We may not always want RANDOM data, just SOME data.
 * Random data is good for testing, but not precise
 * benchmarking or debugging.
 */
// cons
	Data1D::
	Data1D(
		const size_t
			width,
		const DataType
			type,
		const GenType
			gen
		) throw (...) :
		DataStore(width, type, gen),
		m_width(width)/*,
		m_wrapped(0)*/
{
}

// cons
	Data1D::
	Data1D(
		const size_t
			width,
		const DataType
			type,
		const GenType
			gen,
		const unsigned int
			seed
		) throw (...) :
		DataStore(width, type, gen, seed),
		m_width(width)/*,
		m_wrapped(0)*/
{
}

// cons
	Data1D::
	Data1D(
		const size_t
			width,
		const DataType
			type,
		const GenType
			gen,
		ds_callback
			func
		) throw (...) :
		DataStore(width, type, gen, func),
		m_width(width)/*,
		m_wrapped(0)*/
{
}

 /*!
 * We may not always want RANDOM data, just SOME data.
 * Random data is good for testing, but not precise
 * benchmarking or debugging.
 */
// cons
	Data1D::
	Data1D(
		const size_t
			width,
		const DataType
			type,
		const GenType
			gen,
		std::istream &
			src
		) throw (...) :
		DataStore(width, type, gen, src),
		m_width(width)/*,
		m_wrapped(0)*/
{
}

size_t
	Data1D::
	GetWidth() const
{
	return m_width;
}

/*template <typename T>
size_t
	Data1D::
	WrapData(
		T *
			dest,
		T const * const
			data,
		size_t const
			width,
		size_t const
			apron)
{
	// This will WRAP the data, not EXTEND the data.
	size_t const
		usable = m_scMaxWidth - (apron * 2),
		rows = (width - 1) / (usable + 1);
	// Because size_t is very helpfully unsigned.
	// UPDATE: Now I do use size_t and just start at a +width offset, which with
	// the "%" gives the same results in the end and also solves the negative
	// values problem.
	size_t
		idx = width - apron,
		extra = apron * 2;
	for (size_t y = 0; y != rows; ++y)
	{
		//size_t const
		//	oy = y * m_scMaxWidth;
		for (size_t x = 0; x != m_scMaxWidth; ++x)
		{
			// Read "idx plus mod mod mod".
			//dest[oy + out_x] = data[(idx + mod) % mod];
			// Hooray for premature optimisation.
			*dest++ = data[idx % width];
			++idx;
		}
		idx -= extra;
	}
	return rows;
}

template <typename T>
void
	Data1D::
	Recompress(
		T *
			dest,
		T const * const
			data,
		size_t const
			width,
		size_t const
			apron)
{
	size_t const
		rowEnd = m_scMaxWidth - apron,
		usable = m_scMaxWidth - (apron * 2),
		rows = (width - 1) / (usable + 1);
	for (size_t y = 0; y != rows; ++y)
	{
		size_t const
			oy = y * m_scMaxWidth;
		for (size_t x = apron; x != rowEnd; ++x)
		{
			// Just extract the middle bit of the array, not the apron.
			*dest++ = data[oy + x];
		}
	}
}*/

/*void * const
	Data1D::
	GetApronStore(
		size_t const
			apron)
{
	if (m_wrapped)
	{
		if (m_apron >= apron)
		{
			m_apron = apron;
			return m_wrapped;
		}
	}
	if (GetSize())
	{
		// This was a placeholder array.
		m_wrapped = new char [(GetSize() - 1) / ((m_scMaxWidth];
		return (void *)m_wrapped;
	}
	else
	{
		throw "Uninitialised data";
	}
}
 
void * const
	Data1D::
	GetApronStore() const
{
	if (!m_wrapped)
	{
		throw "Uninitialised data";
	}
	return m_wrapped;
}*/
