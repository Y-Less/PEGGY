#if !defined NO_OPEN_CL

#include "ParallelOpenCL.h"

#define DEFINE_DIRECTIVES \
	"-D KERNEL=__kernel -D GLOBAL=__global -D CONSTANT=__constant -D LOCAL=__local -I ./"

// Global store for a single kernel.
cl_kernel
	__g_kernel;

cl_int
	__g_param;

void
	KernelCall(cl_kernel)
{
}

// cons
	ParallelOpenCL::
		ParallelOpenCL(
			const ParallelOpenCLType
				type
			) :
			m_type(type),
			m_commands(0),
			m_devices(0),
			m_context(0),
			m_deviceCount(0),
			m_kernels(0),
			m_kernelCount(0),
			m_program(0)
{
	SetName("Strm");
}

void
	ParallelOpenCL::
	HWInit(const bool) throw (...)
{
	// Open the hardware.
	cl_device_type
		type;
	switch (m_type)
	{
		case PST_CPU:
			type = CL_DEVICE_TYPE_CPU;
			break;
		case PST_GPU:
			type = CL_DEVICE_TYPE_GPU;
			break;
	}
	cl_int
		err;
	cl_uint
		numPlatforms;
	err = clGetPlatformIDs(0, 0, &numPlatforms);
	if (err != CL_SUCCESS)
	{
		throw GetOpenCLErrorCodeStr(err);
	}
	if (!numPlatforms)
	{
		throw "No contexts found";
	}
	// Get all the available platforms.
	cl_platform_id *
		platforms = new cl_platform_id[numPlatforms];
	err = clGetPlatformIDs(numPlatforms, platforms, 0);
	if (err != CL_SUCCESS)
	{
		delete [] platforms;
		throw GetOpenCLErrorCodeStr(err);
	}
	// For now just use the first platform.
	cl_platform_id
		targetPlatform = platforms[0];
	delete [] platforms;
	// Create an OpenCL context.
	if (targetPlatform)
	{
		cl_context_properties
			cps[3] = 
				{
					CL_CONTEXT_PLATFORM, 
					(cl_context_properties)targetPlatform, 
					0
				};
		m_context = clCreateContextFromType(cps, type,0, 0, &err);
	}
	else
	{
		m_context = clCreateContextFromType(0, type, 0, 0, &err);
	}
	if (err != CL_SUCCESS)
	{
		throw GetOpenCLErrorCodeStr(err);
	}
	else if (m_context == 0)
	{
		throw "Could not create context.";
	}
	// Get an OpenCL device.
	size_t
		deviceListSize;
    err = clGetContextInfo(m_context, CL_CONTEXT_DEVICES, 0, 0, &deviceListSize);
	if (err != CL_SUCCESS)
	{
		throw GetOpenCLErrorCodeStr(err);
	}
	else if (deviceListSize == 0)
	{
		throw "No devices found.";
	}
	m_devices = (cl_device_id *)malloc(deviceListSize);
	m_deviceCount = (cl_int)(deviceListSize / sizeof (cl_device_id));
	if (m_devices == 0)
	{
		throw "Could not allocate memory.";
	}
	err = clGetContextInfo(m_context, CL_CONTEXT_DEVICES, deviceListSize, m_devices, 0);
	if (err != CL_SUCCESS)
	{
		throw GetOpenCLErrorCodeStr(err);
	}
	// Create the command queue.
	m_commands = clCreateCommandQueue(m_context, m_devices[0], 0, &err);
	if (err != CL_SUCCESS)
	{
		throw GetOpenCLErrorCodeStr(err);
	}
}

void
	ParallelOpenCL::
	HWClose(const bool) throw (...)
{
	cl_int
		err;
	// Destroy any open kernels.
	if (m_kernels)
	{
		unsigned int
			kc = m_kernelCount;
		for (unsigned int i = 0; i != kc; ++i)
		{
			err = clReleaseKernel(m_kernels[i]);
			if (err != CL_SUCCESS)
			{
				throw GetOpenCLErrorCodeStr(err);
			}
		}
		delete [] m_kernels;
		m_kernels = 0;
		m_kernelCount = 0;
	}
	// Destroy any compiled programs.
	if (m_program)
	{
		err = clReleaseProgram(m_program);
		if (err != CL_SUCCESS)
		{
			throw GetOpenCLErrorCodeStr(err);
		}
		m_program = 0;
	}
	// Destroy the command queue.
	err = clReleaseCommandQueue(m_commands);
	if (err != CL_SUCCESS)
	{
		throw GetOpenCLErrorCodeStr(err);
	}
	m_commands = 0;
	// Destroy the context.
	err = clReleaseContext(m_context);
	if (err != CL_SUCCESS)
	{
		throw GetOpenCLErrorCodeStr(err);
	}
	m_context = 0;
	// Free memory.
	if (m_devices)
	{
		free(m_devices);
		m_devices = 0;
		m_deviceCount = 0;
	}
}

cl_kernel *
	ParallelOpenCL::
	OpenCode(
		const char *
			file,
		const char *
			directives,
		const char ** const
			kernels,
		const unsigned int
			kc
		) throw (...)
{
	// Compile the given string to binary.
	cl_int
		err;
	// Open the specified file and read the whole thing in to memory.
	FILE *
		fHnd = fopen(file, "r");
	if (!fHnd)
	{
		throw "Could not open source file";
	}
	fseek(fHnd, 0, SEEK_END);
	size_t
		size = ftell(fHnd);
	rewind(fHnd);
	char *
		code;
	try
	{
		code = new char [size + 1];
	}
	catch (...)
	{
		fclose(fHnd);
		throw;
	}
	// Don't check the size read - new lines (\r\n / \n) mess this up.  They
	// also mess up the reading, so add an explicit NULL at the true end.
	code[fread(code, 1, size, fHnd)] = '\0';
	// Close the file.
	fclose(fHnd);
	// Create a program from the given source code string.
	m_program = clCreateProgramWithSource(m_context, 1, (const char **)&code, NULL, &err);
	// Delete the store for the code string we just created.
	delete [] code;
	if (err != CL_SUCCESS)
	{
		throw GetOpenCLErrorCodeStr(err);
	}
	else if (m_program == 0)
	{
		throw "Could not create program";
	}
	// Compile the program now, do not use a callback.
	err = clBuildProgram(m_program, m_deviceCount, m_devices, /*directives*/ DEFINE_DIRECTIVES, 0, 0);
	if (err != CL_SUCCESS)
	{
		if (err == CL_BUILD_PROGRAM_FAILURE)
		{
			// Build failed - output the errors.
			cl_int
				derr;
			// Get the size of the program build log.
			size_t
				len;
			derr = clGetProgramBuildInfo(m_program, m_devices[0], CL_PROGRAM_BUILD_LOG, 0, 0, &len);
			if (derr != CL_SUCCESS)
			{
				throw GetOpenCLErrorCodeStr(derr);
			}
			// Get the build log.
			char *
				log = new char [len];
			derr = clGetProgramBuildInfo(m_program, m_devices[0], CL_PROGRAM_BUILD_LOG, len, log, 0);
			if (derr != CL_SUCCESS)
			{
				delete [] log;
				throw GetOpenCLErrorCodeStr(derr);
			}
			std::cout << std::endl << log << std::endl;
			delete [] log;
		}
		throw GetOpenCLErrorCodeStr(err);
	}
	// Load all the kernels from the compiled source.
	m_kernels = new cl_kernel [kc];
	m_kernelCount = (cl_int)kc;
	for (unsigned int i = 0; i != kc; ++i)
	{
		m_kernels[i] = clCreateKernel(m_program, kernels[i], &err);
		if (err != CL_SUCCESS)
		{
			throw GetOpenCLErrorCodeStr(err);
		}
	}
	// Return a pointer to the array of kernels.  We don't need to pass the
	// size in any way as the caller passed the size in the first place.
	return m_kernels;
}

cl_context
	ParallelOpenCL::
	GetContext() const
{
	return m_context;
}

const char *
	GetOpenCLErrorCodeStr(
		cl_int
			input)
{
	// Code from ATI streams example library - apprently this isn't in-built!
    switch(input)
    {
        case CL_DEVICE_NOT_FOUND:
            return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE:
            return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
             return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return "CL_INVALID_MIP_LEVEL";
		#ifdef CL_INVALID_GLOBAL_WORK_SIZE
			// Seems to be ATI only.
			case CL_INVALID_GLOBAL_WORK_SIZE:
				return "CL_INVALID_GLOBAL_WORK_SIZE";
		#endif
        default:
            return "unknown error code";
    }
    return "unknown error code";
}

#endif
