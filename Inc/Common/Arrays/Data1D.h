// Copyright (c) Microsoft Corporation.   All rights reserved.

#pragma once

#include "DataStore.h"

class Data1D :
	public DataStore
{
public:
	size_t
		GetWidth() const;
	
	template <typename T>
	static void
		Recompress(T * dest, T const * const data, size_t const width, size_t const apron = 32)
	{
		size_t const
			rowEnd = m_scMaxWidth - apron,
			usable = m_scMaxWidth - (apron * 2),
			rows = (width - 1) / usable + 1;
		for (size_t y = 0, i = 0; y != rows; ++y)
		{
			size_t const
				oy = y * m_scMaxWidth;
			for (size_t x = apron; x != rowEnd; ++x)
			{
				// Just extract the middle bit of the array, not the apron.
				if (i == width)
				{
					return;
				}
				*dest++ = data[oy + x];
				++i;
			}
		}
	};
	
protected:
	// cons
		Data1D(const size_t, const DataType);
	
	// cons
		Data1D(const size_t, const size_t, const void * const);
	
	// cons
		Data1D(const size_t, const DataType, const GenType);
	
	// cons
		Data1D(const size_t, const DataType, const GenType, const unsigned int);
	
	// cons
		Data1D(const size_t, const DataType, const GenType, ds_callback);
	
	// cons
		Data1D(const size_t, const DataType, const GenType, std::istream &);
	
	template <typename T>
	static size_t
		WrapData(T * dest, T const * const data, size_t const width, size_t const apron = 32)
	{
		// This will WRAP the data, not EXTEND the data.
		size_t const
			usable = m_scMaxWidth - (apron * 2),
			rows = (width - 1) / usable + 1;
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
	};
	
	static size_t const
		m_scMaxWidth = 4096; // Errors claim 8192, I get stuck at 4096.
	
	/*void * const
		GetApronStore();
	
	void * const
		GetApronStore() const;*/
	
private:
	size_t
		m_width;
	
	/*void *
		m_wrapped;*/
};
