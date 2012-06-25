// Copyright (c) Microsoft Corporation.   All rights reserved.

#pragma once

#include <iostream>

#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
	#include <Accelerator.h>
#endif
#if !defined NO_OPEN_CL
	#include <CL/cl.h>
	
	const char * 
		GetOpenCLErrorCodeStr(cl_int);
#endif

enum
	DataType
{
	DT_Float4,
	DT_Float,
	DT_Int,
	DT_Bool
};

enum
	GenType
{
	// Generate the data from an std::istream.
	GT_Stream,
	// Generate the data from a random generator with known seed.
	GT_Seed,
	// Generate random data.
	GT_Random,
	// Use the static existing data.
	GT_Static,
	// Use a custom function to generate the data.
	GT_Custom,
	// Generate an empty array.
	GT_Blank,
	// Do not generate any data, or allocate memory, just placehold.
	GT_None,
	// Constant test pattern 0-255 repeatedly (or true/false alternating).
	GT_Test,
	// Do not initialise the data.
	GT_Garbage
};

struct
	ObsidianData
{
	bool
		bFreeH,
		bFreeC;
	void *
		data;
};

extern "C"
{
	void
		ObsidianFinaliser(ObsidianData * const);
};

class DataStore
{
public:
	typedef
		void (DataStore::*ds_callback)(const DataType, const void *, const size_t) throw (...);
	
	void *
		GetData();
	
	void *
		GetData() const;
	
	bool
		HasData() const;
	
	#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
		ParallelArrays::ParallelArray &
			ToAcceleratorArray(unsigned int = 32) throw (...);
	#endif
	
	#if !defined NO_CUDA || !defined NO_CUDA_OPT
		virtual void *
			ToCUDAArray() throw (...);
	#endif
	
	#if !defined NO_OPEN_CL
		virtual cl_mem
			ToOpenCLArray(cl_context) throw (...);
	#endif
	
	#if !defined NO_OBSIDIAN
		//virtual ObsidianData *
		//	ToObsidianArray() throw (...);
	#endif
	
	void
		CleanGPU() throw (...);
	
protected:
	// cons
		DataStore(const size_t, const DataType);
	
	// cons
		DataStore(const size_t, const void * const);
	
	// cons
		DataStore(const size_t, const DataType, const GenType);
	
	// cons
		DataStore(const size_t, const DataType, const GenType, const unsigned int);
	
	// cons
		DataStore(const size_t, const DataType, const GenType, ds_callback);
	
	// cons
		DataStore(const size_t, const DataType, const GenType, std::istream &);
	
	virtual // dest
		~DataStore();
	
	size_t
		GetSize() const;
	
	#if !defined NO_CUDA
		void
			SetCUDAArray(void *);
		
		void *
			GetCUDAArray() const;
	#endif
	
	#if !defined NO_OPEN_CL
		cl_mem
			GetOpenCLArray() const;
	#endif
	
	#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
		virtual ParallelArrays::ParallelArray *
			_ToAcceleratorArray(unsigned int = 32) = 0;
		
		ParallelArrays::ParallelArray *
			GetAcceleratorArray() const;
	#endif
	
	#if !defined NO_OBSIDIAN
		//ObsidianData *
		//	GetObidianArray() const;
	#endif
	
private:
	void
		GenerateData(const DataType, const GenType, const unsigned int, const size_t);
	
	void
		GenerateData(const DataType, const GenType, std::istream &, const size_t);
	
	void
		GenerateData(const DataType, const GenType, ds_callback, const size_t);
	
	void
		GenerateArray(const DataType, const size_t);
	
	void
		GenerateStatic(const DataType, void * const, const size_t) throw (...);
	
	void
		GenerateRandom(const DataType, void * const, const size_t);
	
	void
		GenerateCustom(const DataType, void * const, const size_t, ds_callback) throw (...);
	
	void
		GenerateBlank(const DataType, void * const, const size_t);
	
	void
		GenerateTest(const DataType, void * const, const size_t);
	
	void
		GenerateSeed(const DataType, void * const, const size_t, const unsigned int);
	
	void
		GenerateStream(const DataType, void * const, const size_t, std::istream &);
	
	void *
		m_data;
	
	char *
		m_temp;
	
	size_t
		m_size;
	
	#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT
		ParallelArrays::ParallelArray *
			m_accArray;
	#endif
	
	#if !defined NO_CUDA || !defined NO_CUDA_OPT
		void *
			m_cudaArray;
	#endif
	
	#if !defined NO_OPEN_CL
		cl_mem
			m_openCLArray;
	#endif
	
	// On reflection, these should be "#ifdef" to allow you to select the parts
	// you want and maintain future compatibility by making things default off.
	#if !defined NO_OBSIDIAN
		//ObsidianData *
		//	m_obsidianArray;
	#endif
};
