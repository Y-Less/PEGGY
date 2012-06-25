// Copyright (c) Microsoft Corporation.   All rights reserved.

#include "Data2D.h"

#if !defined NO_CUDA || !defined NO_CUDA_OPT
	#include <cuda.h>
	#include "../ParallelCUDADriver.h"
	#include <cuda_runtime_api.h>
#endif

// cons
	Data2D::
	Data2D(
		const size_t
			height,
		const size_t
			width,
		const size_t
			dw,
		const void * const
			data
		) throw (...) :
		DataStore(height * width * dw, data),
		m_height(height),
		m_width(width)
{
}

// cons
	Data2D::
	Data2D(
		const size_t
			height,
		const size_t
			width,
		const DataType
			type
		) throw (...) :
		DataStore(height * width, type),
		m_height(height),
		m_width(width)
{
}

// cons
	Data2D::
	Data2D(
		const size_t
			height,
		const size_t
			width,
		const DataType
			type,
		const GenType
			gen
		) throw (...) :
		DataStore(height * width, type, gen),
		m_height(height),
		m_width(width)
{
}

// cons
	Data2D::
	Data2D(
		const size_t
			height,
		const size_t
			width,
		const DataType
			type,
		const GenType
			gen,
		const unsigned int
			seed
		) throw (...) :
		DataStore(height * width, type, gen, seed),
		m_height(height),
		m_width(width)
{
}

// cons
	Data2D::
	Data2D(
		const size_t
			height,
		const size_t
			width,
		const DataType
			type,
		const GenType
			gen,
		std::istream &
			src
		) throw (...) :
		DataStore(height * width, type, gen, src),
		m_height(height),
		m_width(width)
{
}

// cons
	Data2D::
	Data2D(
		const size_t
			height,
		const size_t
			width,
		const DataType
			type,
		const GenType
			gen,
		ds_callback
			func
		) throw (...) :
		DataStore(height * width, type, gen, func),
		m_height(height),
		m_width(width)
{
}

size_t
	Data2D::
	GetHeight() const
{
	return m_height;
}

size_t
	Data2D::
	GetWidth() const
{
	return m_width;
}

size_t
	Data2D::
	GetPitch() const
{
	return m_pitch;
}

#if !defined NO_CUDA || !defined NO_CUDA_OPT
	void *
		Data2D::
		ToCUDAArray() throw (...)
	{
		#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
			if (GetAcceleratorArray())
			{
				throw "GPU memory not clean";
			}
		#endif
		// Returns an array of this data in CUDA format.
		//std::cout << std::endl << "To init common";
		//std::cout << "check";
		if (!GetCUDAArray())
		{
			// Get the current device.
			//CUdevice
			//	dev;
			//TryCUDA(cuCtxGetDevice(&dev));
			CUdeviceptr
				mem;
			size_t
				height = GetHeight(),
				width = GetSize() / height,
				pitch,
				element = width / GetWidth();
			TryCUDA(cuMemAllocPitch(&mem, &pitch, width, height, element));
			if (HasData())
			{
				// If the real data exists, copy it.
				cudaMemcpy2D((void *)mem, pitch, GetData(), width, width, height, cudaMemcpyHostToDevice);
				/*CUDA_MEMCPY2D
					memdata = {0};
				memdata.srcMemoryType = CU_MEMORYTYPE_HOST;
				memdata.srcHost = GetData();
				memdata.srcPitch = width;
				memdata.dstMemoryType = CU_MEMORYTYPE_DEVICE;
				memdata.dstDevice = mem;
				memdata.dstPitch = pitch;
				memdata.WidthInBytes = width;
				memdata.Height = height;
				// OK, this is significantly harder in the driver API than in
				// the runtime API.
				TryCUDA(cuMemcpy2D(&memdata));*/
			}
			else
			{
				TryCUDA(cuMemsetD8(mem, 0, pitch * height));
			}
			// The CUDA C Programming Guide specifies this as a safe conversion
			// (at least the opposite is).  Even though it isn't (at least not
			// in 64bit) - "CUdeviceptr" is defined as "unsigned int".
			SetCUDAArray((void *)mem);
			// It has to be said that so-far, the driver API seems quite easy to
			// use actually, possibly actually EASIER than the runtime API.
			m_pitch = pitch / element;
		}
		return GetCUDAArray();
	}
#endif
