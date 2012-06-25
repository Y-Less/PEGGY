// Copyright (c) Microsoft Corporation.   All rights reserved.

#include "DataStore.h"
#include "StaticData.h"

#include <ctime>

#if !defined NO_CUDA
//	#include <cuda_runtime_api.h>
//#endif
//
//#if !defined NO_CUDA_DRIVER
	#include <cuda.h>
	#include "../ParallelCUDADriver.h"
#endif

// cons
	DataStore::
	DataStore(
		const size_t
			size,
		const void * const
			data
		) throw (...) :
		m_size(size),
		m_temp(0),
		#if !defined NO_ACCELERATOR
			m_accArray(0),
		#endif
		#if !defined NO_OPEN_CL
			m_openCLArray(0),
		#endif
		#if !defined NO_CUDA || !defined NO_CUDA_OPT
			m_cudaArray(0),
		#endif
		m_data((void *)new char [size])
{
	// Copy all the data over.
	memcpy(m_data, data, size);
}

// cons
	DataStore::
	DataStore(
		const size_t
			size,
		const DataType
			type
		) throw (...) :
		m_temp(0),
		#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
			m_accArray(0),
		#endif
		#if !defined NO_OPEN_CL
			m_openCLArray(0),
		#endif
		#if !defined NO_CUDA || !defined NO_CUDA_OPT
			m_cudaArray(0),
		#endif
		m_data(0)
{
	GenerateData(type, GT_Random, (unsigned int)0, size);
}

 /*!
 * We may not always want RANDOM data, just SOME data.
 * Random data is good for testing, but not precise
 * benchmarking or debugging.
 */
// cons
	DataStore::
	DataStore(
		const size_t
			size,
		const DataType
			type,
		const GenType
			gen
		) throw (...) :
		m_temp(0),
		#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
			m_accArray(0),
		#endif
		#if !defined NO_OPEN_CL
			m_openCLArray(0),
		#endif
		#if !defined NO_CUDA || !defined NO_CUDA_OPT
			m_cudaArray(0),
		#endif
		m_data(0)
{
	if (gen == GT_Seed)
	{
		throw "GenType::GT_Seed specified, but no seed provided";
	}
	GenerateData(type, gen, (unsigned int)0, size);
}

// cons
	DataStore::
	DataStore(
		const size_t
			size,
		const DataType
			type,
		const GenType
			gen,
		const unsigned int
			seed
		) throw (...) :
		m_temp(0),
		#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
			m_accArray(0),
		#endif
		#if !defined NO_OPEN_CL
			m_openCLArray(0),
		#endif
		#if !defined NO_CUDA || !defined NO_CUDA_OPT
			m_cudaArray(0),
		#endif
		m_data(0)
{
	if (gen != GT_Seed)
	{
		throw "GenType::GT_Seed not specified, but seed provided";
	}
	GenerateData(type, gen, seed, size);
}

// cons
	DataStore::
	DataStore(
		const size_t
			size,
		const DataType
			type,
		const GenType
			gen,
		ds_callback
			func
		) throw (...) :
		m_temp(0),
		#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
			m_accArray(0),
		#endif
		#if !defined NO_OPEN_CL
			m_openCLArray(0),
		#endif
		#if !defined NO_CUDA || !defined NO_CUDA_OPT
			m_cudaArray(0),
		#endif
		m_data(0)
{
	GenerateData(type, gen, func, size);
}

 /*!
 * We may not always want RANDOM data, just SOME data.
 * Random data is good for testing, but not precise
 * benchmarking or debugging.
 */
// cons
	DataStore::
	DataStore(
		const size_t
			size,
		const DataType
			type,
		const GenType
			gen,
		std::istream &
			src
		) throw (...) :
		m_temp(0),
		#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
			m_accArray(0),
		#endif
		#if !defined NO_OPEN_CL
			m_openCLArray(0),
		#endif
		#if !defined NO_CUDA || !defined NO_CUDA_OPT
			m_cudaArray(0),
		#endif
		m_data(0)
{
	GenerateData(type, gen, src, size);
}

// dest
	DataStore::
	~DataStore()
{
	delete [] m_data;
	try
	{
		CleanGPU();
	}
	catch (...)
	{
		// Really need additional reporting here, but throws in
		// destructors are awkward.
	}
}

void
	DataStore::
	GenerateData(
		const DataType
			type,
		const GenType
			gen,
		const unsigned int
			seed,
		const size_t
			size
		) throw (...)
{
	// An istream which provides random numbers would be
	// very useful here.
	if (gen != GT_None)
	{
		GenerateArray(type, size);
	}
	switch (gen)
	{
		case GT_Stream:
			throw "GenType::GT_Stream specified, but no stream provided";
			break;
		case GT_Seed:
			GenerateSeed(type, m_data, size, seed);
			break;
		case GT_Random:
			GenerateRandom(type, m_data, size);
			break;
		case GT_Static:
			GenerateStatic(type, m_data, size);
			break;
		case GT_Custom:
			throw "GenType::GT_Custom specified, but no callback provided";
			break;
		case GT_Blank:
			GenerateBlank(type, m_data, size);
			break;
		case GT_None:
			// Initialise the size.
			switch (type)
			{
				case DT_Float4:
					m_size = size * 4 * sizeof (float);
					break;
				case DT_Float:
					//std::cout << "size: " << size << std::endl;
					m_size = size * sizeof (float);
					break;
				case DT_Int:
					m_size = size * sizeof (__int32);
					break;
				case DT_Bool:
					m_size = size * sizeof (bool);
					break;
			}
			// FALLTHROUGH
		case GT_Garbage:
			// Do nothing.
			break;
		case GT_Test:
			GenerateTest(type, m_data, size);
			break;
	}
}

void
	DataStore::
	GenerateData(
		const DataType
			type,
		const GenType
			gen,
		ds_callback
			func,
		const size_t
			size
		) throw (...)
{
	// An istream which provides random numbers would be
	// very useful here.
	GenerateArray(type, size);
	switch (gen)
	{
		case GT_Custom:
			GenerateCustom(type, m_data, size, func);
			break;
		default:
			throw "GenType::GT_Custom not specified, but callback provided";
			break;
	}
}

void
	DataStore::
	GenerateData(
		const DataType
			type,
		const GenType
			gen,
		std::istream &
			src,
		const size_t
			size
		) throw (...)
{
	// An istream which provides random numbers would be
	// very useful here.
	GenerateArray(type, size);
	switch (gen)
	{
		case GT_Stream:
			GenerateStream(type, m_data, size, src);
			break;
		default:
			throw "GenType::GT_Stream not specified, but stream provided";
			break;
	}
}

void
	DataStore::
	GenerateArray(
		const DataType
			type,
		const size_t
			size
		) throw (...)
{
	// Use C++ style allocators, not malloc!
	void *
		newData;
	// Allocate memory based on required type.
	switch (type)
	{
		case DT_Float4:
			//allocationSize *= sizeof (float) * 4;
			newData = (void *)new float [size * 4];
			m_size = size * 4 * sizeof (float);
			break;
		case DT_Float:
			//allocationSize *= sizeof (float);
			newData = (void *)new float [size];
			m_size = size * sizeof (float);
			break;
		case DT_Int:
			//allocationSize *= sizeof (long);
			newData = (void *)new __int32 [size];
			m_size = size * sizeof (__int32);
			break;
		case DT_Bool:
			//allocationSize *= 1;
			newData = (void *)new bool [size];
			m_size = size * sizeof (bool);
			break;
		default:
			throw "Unknown DataType in DataStore::GenerateArray";
			break;
	}
	delete [] m_data;
	m_data = newData;
}

void
	DataStore::
	GenerateStatic(
		const DataType
			type,
		void * const
			buffer,
		const size_t
			size
		) throw (...)
{
	#ifdef NO_STATIC_DATA
		throw "No static data available";
	#else
		switch (type)
		{
			case DT_Float4:
			{
				// These are the same quite nicely - this will
				// interpret the stored float bit patterns as ints.
				if (size > MAX_DATA_S / 4)
				{
					throw "Insufficient static initialisation data";
				}
				memcpy(buffer, (void *)gc_fPointData, size * 4 * sizeof (float));
				break;
			}
			case DT_Float:
				// FALLTHROUGH
			case DT_Int:
			{
				// These are the same quite nicely - this will
				// interpret the stored float bit patterns as ints.
				if (size > MAX_DATA_S)
				{
					throw "Insufficient static initialisation data";
				}
				memcpy(buffer, (void *)gc_fPointData, size * sizeof (float));
				break;
			}
			case DT_Bool:
			{
				// Could do vast arrays here by using every bit.
				if (size > MAX_DATA_S)
				{
					throw "Insufficient static initialisation data";
				}
				bool *
					data = (bool *)buffer;
				for (int i = 0; i != size; ++i)
				{
					// Every float is between 0 and 1.
					data[i] = gc_fPointData[i] < 0.5f;
				}
				break;
			}
		}
	#endif
}

void
	DataStore::
	GenerateRandom(
		const DataType
			type,
		void * const
			buffer,
		const size_t
			size)
{
	GenerateSeed(type, buffer, size, (unsigned int)time(NULL));
}

void
	DataStore::
	GenerateCustom(
		const DataType
			type,
		void * const
			buffer,
		const size_t
			size,
		ds_callback
			func) throw (...)
{
	(this->*func)(type, buffer, size);
}

void
	DataStore::
	GenerateSeed(
		const DataType
			type,
		void * const
			buffer,
		const size_t
			size,
		const unsigned int
			seed)
{
	srand(seed);
	switch (type)
	{
		case DT_Float4:
		{
			float *
				data = (float *)buffer;
			// Initialize the input.
			for (int i = 0; i != size * 4; ++i)
			{
				data[i] = rand() / (float)RAND_MAX;
			}
			break;
		}
		case DT_Float:
		{
			float *
				data = (float *)buffer;
			// Initialize the input.
			for (int i = 0; i != size; ++i)
			{
				data[i] = rand() / (float)RAND_MAX;
			}
			break;
		}
		case DT_Int:
		{
			__int32 *
				data = (__int32 *)buffer;
			// Initialize the input.
			for (int i = 0; i != size; ++i)
			{
				data[i] = rand();
			}
			break;
		}
		case DT_Bool:
		{
			bool *
				data = (bool *)buffer;
			// Initialize the input.
			for (int i = 0; i != size; ++i)
			{
				// Could do something to use all 32 bits, but
				// then the code below would be required to
				// ensure one bit didn't get priority.  "rand"
				// is better at generating smaller numbers, so
				// there's likely to be more 0's in the upper
				// bits.
				data[i] = rand() & 1;
			}
			/*bool *
				data = (bool *)buffer;
			int
				cur = rand(),
				i = 0,
				j = 0;
			// Tried to use hardware style bit stream filtering to
			// get a better random spread.
			do
			{
				if (cur & 3 == 1)
				{
					data[i] = true;
					++i;
				}
				else if (cur & 3 == 2)
				{
					data[i] = false;
					++i;
				}
				else
				{
					cur >>= 1;
				}
				++j;
				cur >>= 1;
				if (j == 31)
				{
					j = 0;
					int
						lst = cur;
					cur = rand();
					if (i != size && lst & 1 != cur & 1)
					{
						data[i] = lst & 1;
						++i;
					}
				}
			}
			while (i != size);*/
			break;
		}
	}
}

void
	DataStore::
	GenerateStream(
		const DataType
			type,
		void * const
			buffer,
		const size_t
			size,
		std::istream &
			src)
{
	switch (type)
	{
		case DT_Float4:
		{
			float *
				data = (float *)buffer;
			for (int i = 0; i != size * 4; ++i)
			{
				src >> data[i];
			}
			break;
		}
		case DT_Float:
		{
			float *
				data = (float *)buffer;
			for (int i = 0; i != size; ++i)
			{
				src >> data[i];
			}
			break;
		}
		case DT_Int:
		{
			__int32 *
				data = (__int32 *)buffer;
			for (int i = 0; i != size; ++i)
			{
				src >> data[i];
			}
			break;
		}
		case DT_Bool:
		{
			bool *
				data = (bool *)buffer;
			for (int i = 0; i != size; ++i)
			{
				src >> data[i];
			}
			break;
		}
	}
}

void
	DataStore::
	GenerateBlank(
		const DataType
			type,
		void * const
			buffer,
		const size_t
			size)
{
	size_t
		ns;
	switch (type)
	{
		case DT_Float4:
			ns = sizeof (float) * 4;
			break;
		case DT_Float:
			ns = sizeof (float);
			break;
		case DT_Int:
			ns = sizeof (__int32);
			break;
		case DT_Bool:
			ns = sizeof (bool);
			break;
	}
	memset(buffer, 0, ns * size);
}

void
	DataStore::
	GenerateTest(
		const DataType
			type,
		void * const
			buffer,
		const size_t
			size)
{
	switch (type)
	{
		case DT_Float4:
		{
			float *
				data = (float *)buffer;
			for (size_t i = 0; i != size; ++i)
			{
				data[i * 4 + 0] = (float)(i & 255 + 0);
				data[i * 4 + 1] = (float)(i & 255 + 256);
				data[i * 4 + 2] = (float)(i & 255 + 512);
				data[i * 4 + 3] = (float)(i & 255 + 768);
			}
			break;
		}
		case DT_Float:
		{
			float *
				data = (float *)buffer;
			for (size_t i = 0; i != size; ++i)
			{
				data[i] = (float)(i & 255);
			}
			break;
		}
		case DT_Int:
		{
			int *
				data = (int *)buffer;
			for (size_t i = 0; i != size; ++i)
			{
				data[i] = (int)(i & 255);
			}
			break;
		}
		case DT_Bool:
		{
			bool *
				data = (bool *)buffer;
			for (size_t i = 0; i != size; ++i)
			{
				data[i] = (bool)(i & 1);
			}
			break;
		}
	}
}

void *
	DataStore::
	GetData()
{
	if (m_data)
	{
		return m_data;
	}
	else if (m_size)
	{
		// This was a placeholder array.
		m_temp = new char [m_size];
		return (void *)m_temp;
	}
	else
	{
		throw "Uninitialised data a";
	}
}

void *
	DataStore::
	GetData() const
{
	if (!m_data)
	{
		throw "Uninitialised data b";
	}
	return m_data;
}

bool
	DataStore::
	HasData() const
{
	return m_data != 0;
}

#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
	ParallelArrays::ParallelArray &
		DataStore::
		ToAcceleratorArray(unsigned int apron) throw (...)
	{
		#if !defined NO_CUDA || !defined NO_CUDA_OPT
			if (GetCUDAArray())
			{
				throw "GPU memory not clean";
			}
		#endif
		#if !defined NO_OPEN_CL
			if (GetOpenCLArray())
			{
				throw "GPU memory not clean";
			}
		#endif
		// Returns an array of this data in Accelerator format.
		if (!GetAcceleratorArray())
		{
			// If there is no existing data, Accelerator arrays require an
			// initialiser, so default to 0 (or equivalent).
			m_accArray = _ToAcceleratorArray(apron);
		}
		return *GetAcceleratorArray();
	}
	
	ParallelArrays::ParallelArray *
		DataStore::
		GetAcceleratorArray() const
	{
		return m_accArray;
	}
#endif

#if !defined NO_CUDA || !defined NO_CUDA_OPT
	void
		DataStore::
		SetCUDAArray(
			void *
				arr)
	{
		m_cudaArray = arr;
	}
	
	void *
		DataStore::
		GetCUDAArray() const
	{
		//std::cout << "GetCUDAArray: " << m_cudaArray << std::endl;
		return m_cudaArray;
	}
	
	void *
		DataStore::
		ToCUDAArray() throw (...)
	{
		#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
			if (GetAcceleratorArray())
			{
				throw "GPU memory not clean";
			}
		#endif
		#if !defined NO_OPEN_CL
			if (GetOpenCLArray())
			{
				throw "GPU memory not clean";
			}
		#endif
		// Returns an array of this data in CUDA format.
		if (!GetCUDAArray())
		{
			// Get the current device.
			//CUdevice
			//	dev;
			//TryCUDA(cuCtxGetDevice(&dev));
			CUdeviceptr
				mem;
			TryCUDA(cuMemAlloc(&mem, m_size));
			if (m_data)
			{
				// If the real data exists, copy it.
				TryCUDA(cuMemcpyHtoD(mem, m_data, m_size));
			}
			else
			{
				TryCUDA(cuMemsetD8(mem, 0, m_size));
			}
			// The CUDA C Programming Guide specifies this as a safe conversion
			// (at least the opposite is).  Even though it isn't (at least not
			// in 64bit) - "CUdeviceptr" is defined as "unsigned int".
			SetCUDAArray((void *)mem);
			// It has to be said that so-far, the driver API seems quite easy to
			// use actually, possibly actually EASIER than the runtime API.
		}
		return GetCUDAArray();
	}
#endif

#if !defined NO_OPEN_CL
	cl_mem
		DataStore::
		GetOpenCLArray() const
	{
		return m_openCLArray;
	}
	
	cl_mem
		DataStore::
		ToOpenCLArray(
			cl_context
				context) throw (...)
	{
		#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
			if (GetAcceleratorArray())
			{
				throw "GPU memory not clean";
			}
		#endif
		#if !defined NO_CUDA || !defined NO_CUDA_OPT
			if (GetCUDAArray())
			{
				throw "GPU memory not clean";
			}
		#endif
		// Returns an array of this data in OpenCL format.
		if (!GetOpenCLArray())
		{
			cl_mem
				arr = 0;
			cl_int
				err;
			if (m_data)
			{
				// The data really exists.  These arrays can only ever be read
				// from as there is no syncronisation between the CL buffer and
				// the DataStore buffer. We also allocate and copy the data in
				// one move.
				//std::cout << GetSize();
				arr = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, GetSize(), GetData(), &err);
			}
			else
			{
				// Data doesn't exist.  This array will be used for processing,
				// so make it writable.
				arr = clCreateBuffer(context, CL_MEM_READ_WRITE, GetSize(), 0, &err);
			}
			if (err != CL_SUCCESS)
			{
				throw GetOpenCLErrorCodeStr(err);
			}
			else if (arr == 0)
			{
				throw "Could not allocate memory";
			}
			m_openCLArray = arr;
		}
		return GetOpenCLArray();
	}
#endif

void
	DataStore::
	CleanGPU() throw (...)
{
	#if !defined NO_CUDA || !defined NO_CUDA_OPT
		CUresult
			curesult = CUDA_SUCCESS;
		if (m_cudaArray)
		{
			// Remove from the GPU.
			curesult = cuMemFree((CUdeviceptr)m_cudaArray);
			//cudaFree(m_cudaArray);
			m_cudaArray = 0;
		}
	#endif
	#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
		if (m_accArray)
		{
			// Remove from the GPU.
			delete m_accArray;
			m_accArray = 0;
		}
	#endif
	#if !defined NO_OPEN_CL
		cl_int
			err = CL_SUCCESS;
		if (m_openCLArray)
		{
			// Remove from the GPU.
	        err = clReleaseMemObject(m_openCLArray);
			m_openCLArray = 0;
		}
	#endif
	if (m_temp)
	{
		// Delete the temporary array.  Essentially the same as above, but for
		// reference implementations so they can be coded the same.
		delete [] m_temp;
		m_temp = 0;
	}
	#if !defined NO_OBSIDIAN
		if (m_obsidianArray->bFreeH)
		{
			// Not needed by either.
			delete [] m_obsidianArray->data;
			delete m_obsidianArray;
		}
		else
		{
			// Not needed by C++, but still needed by Haskell.
			m_obsidianArray->bFreeC = true;
		}
	#endif
	#if !defined NO_CUDA || !defined NO_CUDA_OPT
		// CUDA went wrong - throw an error after cleaning everything else up.
		TryCUDA(curesult);
	#endif
	#if !defined NO_OPEN_CL
		// OpenCL went wrong - throw an error after cleaning everything else
		// up.
		if (err != CL_SUCCESS)
		{
			throw GetOpenCLErrorCodeStr(err);
		}
	#endif
}

size_t
	DataStore::
	GetSize() const
{
	return m_size;
}

#if !defined NO_OBSIDIAN
	/*ObsidianData *
		DataStore::
		GetObidianArray()
	{
		
	}
	
	ObsidianData *
		DataStore::
		ToObidianArray()
	{
		
	}
	
	extern "C"
	{
		void
			ObsidianFinaliser(
				ObsidianData * const
					dat)
		{
			if (dat->bFreeC)
			{
				// Not needed by either.
				delete [] dat->data;
				delete dat;
			}
			else
			{
				// Not needed by Haskell, but still needed by C++.
				dat->bFreeH = true;
			}
		}
	}*/
#endif
