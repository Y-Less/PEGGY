#if !defined NO_OPEN_CL

#pragma once

#include "ParallelProcessor.h"

#include <CL/cl.h>

#define KERNEL \
	extern cl_kernel __g_kernel; extern cl_int __g_param;

// Define these as nothing.
#define GLOBAL
#define CONSTANT
#define LOCAL

template <typename T>
void
	KernelPush(//cl_kernel, cl_int *, T)
		cl_kernel
			kernel,
		cl_int *
			pos,
		T
			value)
{
	cl_int
		err = clSetKernelArg(kernel, *pos, sizeof (T), (void *)&value);
	if (err != CL_SUCCESS)
	{
		throw GetOpenCLErrorCodeStr(err);
	}
	++(*pos);
};

void
	KernelCall(cl_kernel);

// Define macros for easy OpenCL calling.
#define KERNEL_PUSH(m) \
	KernelPush(__g_kernel, &__g_param, (m))

#define KERNEL_CALL() \
	KernelCall(__g_kernel)

#define KERNEL_SETUP(m) \
	__g_param = 0, __g_kernel = (m)

const char * 
	GetOpenCLErrorCodeStr(cl_int);

enum
	ParallelOpenCLType
{
	PST_CPU,
	PST_GPU
};

class ParallelOpenCL:
	public ParallelProcessor
{
public:
	// cons
		ParallelOpenCL(const ParallelOpenCLType);
	
protected:
	virtual void
		HWInit(const bool) throw (...);
	
	virtual void
		HWClose(const bool) throw (...);
	
	cl_kernel *
		OpenCode(const char *, const char *, const char ** const, const unsigned int) throw (...);
	
	cl_context
		GetContext() const;
	
private:
	ParallelOpenCLType
		m_type;
	
	cl_command_queue
		m_commands;
	
	cl_device_id *
		m_devices;
	
	cl_int
		m_deviceCount,
		m_kernelCount;
	
	cl_context
		m_context;
	
	cl_program
		m_program;
	
	cl_kernel *
		m_kernels;
};

#endif
