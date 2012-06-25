#if !defined NO_CUDA && !defined NO_HASKELL

#include "ConvolverHaskell.h"
#include <cuda_runtime_api.h>

#include <stdio.h>

#define CEILDIV(m,n) \
	(((m) + (n) - 1) / (n))

//#include <tracing.h>
//#include "ConvolverCUDA.tmh"

extern "C" void
	CopyFilter(float *, int);

extern "C" void
	DoRows(float *, float *, int, int, int, int, int);

extern "C" void
	DoCols(float *, float *, int, int, int, int, int);

// cons
	ConvolverHaskell::
	ConvolverHaskell() :
		Convolver(),
		ParallelCUDADriver(),
		ParallelHaskell(),
		GenerateScript(0)
{
	SetName("Haskell");
}

void
	ConvolverHaskell::
	ConvInit() throw (...)
{
	struct ParallelScript::NamedFunctions_s
		funcs[] =
		{
			{"GenerateScript", (void **)&GenerateScript},
			{0, 0}
		};
	/*ScriptInit(L"ConvolverHaskell.dll", funcs);
	if (GenerateScript())
	{
		throw "GenerateScript returned non-0";
	}*/
	// Compile the code.
	Compile(L"ConvolverHaskell");
	// Load the module.
	printf("Thread: %d\n", GetThreads());
	CUmodule
		module = 0;
	TryCUDA(cuModuleLoad(&module, "ConvolverHaskell.cubin"));
	// Get the two functions.
	m_rowKernel = 0;
	TryCUDA(cuModuleGetFunction(&m_rowKernel, module, "DoOneRowDriver"));
	m_colKernel = 0;
	TryCUDA(cuModuleGetFunction(&m_colKernel, module, "DoOneColDriver"));
	// Get the data we want.
	m_data = (float *)GetData().ToCUDAArray();
}

void
	ConvolverHaskell::
	Execute() throw (...)
{
	Log("Set");
	size_t
		height = GetHeight(),
		width = GetWidth(),
		pitch = GetPitch(),
		size = width * sizeof (float),
		radius = GetRadius();
	int
		threads = GetThreads();
	End("Set");
	Log("X");
	// Do the X dimension.
	// Do I want to use the deprecated (pre-4.0) execution control which I have
	// apparently used in the past, or the much cleaner 4.0-only version?  The
	// latter is nicer, but also requires CUDA 4.0 (do I care about that fact,
	// I'm pretty much the only person using this code, so I don't think so).
	// Seems like my decision to use the new API was wrong - it's quite
	// convoluted!  The documentation isn't even clear on what order the
	// parameters should appear in - backwards for the order they are generally
	// pushed in, or forwards just because forwards.  OK, having had the code
	// run and work more-or-less FIRST TIME (barring a few issues with
	// "ParallelCUDADriver"), it turns out these are forwards.
	int
		params0[] =
		{
			(int)m_data,
			(int)m_smoothX,
		};
	void *
		extra0[] =
		{
			CU_LAUNCH_PARAM_BUFFER_POINTER, params0,
			CU_LAUNCH_PARAM_BUFFER_SIZE, (void *)(2 * 4),
			CU_LAUNCH_PARAM_END
		};
	TryCUDA(cuLaunchKernel(m_rowKernel, CEILDIV((int)width, threads), (int)height, 1, threads, 1, 1, 0, 0, 0, extra0));
	//DoRows(m_smoothX, m_data, (int)height, (int)width, GetThreads(), (int)pitch, (int)radius);
	End("X");
	Log("Y");
	// Do the Y dimension.
	//DoCols(m_smoothY, m_smoothX, (int)height, (int)width, GetThreads(), (int)pitch, (int)radius);
	int
		params1[] =
		{
			(int)m_smoothX,
			(int)m_smoothY,
		};
	void *
		extra1[] =
		{
			CU_LAUNCH_PARAM_BUFFER_POINTER, params1,
			CU_LAUNCH_PARAM_BUFFER_SIZE, (void *)(2 * 4),
			CU_LAUNCH_PARAM_END
		};
	TryCUDA(cuLaunchKernel(m_colKernel, CEILDIV((int)width, threads), (int)height, 1, threads, 1, 1, 0, 0, 0, extra1));
	// Convert the data back.
	End("Y");
	//cudaThreadSynchronize();
	Log("Copy");
	//cudaThreadSynchronize();
	//cudaMemcpy2D(GetStore(), size, m_smoothY, pitch * sizeof (float), size, height, cudaMemcpyDeviceToHost);
	// The most nasty bit so far of using the driver API.  Fortunately the
	// parameters actually seem to map quite nicely from the runtime function to
	// the driver structure.  The fact that I seem to have ended up with exactly
	// the same number of the same inputs, just arranged differently, is
	// somewhat reassuring.
	CUDA_MEMCPY2D
		memdata = {0};
	memdata.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	memdata.srcDevice = m_smoothY;
	memdata.srcPitch = pitch * sizeof (float);
	memdata.dstMemoryType = CU_MEMORYTYPE_HOST;
	memdata.dstHost = GetStore();
	memdata.dstPitch = size;
	memdata.WidthInBytes = size;
	memdata.Height = height;
	TryCUDA(cuMemcpy2D(&memdata));
	End("Copy");
}

void
	ConvolverHaskell::
	ConvExit() throw (...)
{
	//ScriptClose(false);
};

void
	ConvolverHaskell::
	Compile(LPCWSTR file)
{
	WCHAR
		commandLine[1024],
		env[256];
	STARTUPINFO
		si = {0};
	si.cb = sizeof (si);
	PROCESS_INFORMATION
		pi = {0};
	// Generate the command-line parameters.
	if (!GetEnvironmentVariable(L"VC_BIN_DIR", env, sizeof (env)))
	{
		if (GetLastError() == ERROR_ENVVAR_NOT_FOUND)
		{
			throw "Could not find $(VC_BIN_DIR)";
		}
		else
		{
			throw "Unknown environment variable error";
		}
	}
	wsprintf(commandLine, L"-gencode=arch=compute_10,code=sm_10 --use-local-env --cl-version 2010 -ccbin \"%s\"", (LPSTR)env);
	if (!GetEnvironmentVariable(L"CUDA_PATH_V4_0", env, sizeof (env)))
	{
		if (GetLastError() == ERROR_ENVVAR_NOT_FOUND)
		{
			throw "Could not find $(CUDA_PATH_V4_0)";
		}
		else
		{
			throw "Unknown environment variable error";
		}
	}
	wsprintf(commandLine + wcslen(commandLine), L" -I\"%sinclude\" --keep-dir \"Release\" -maxrregcount=0 --machine 32 -cubin -o \"%s.cubin\" \"%s.cu\"", env, file, file);
	// Generate the compiler name.
	wcscat(env, L"bin\\nvcc.exe");
	// Create the process.
	//wprintf(L"%ls\n", env);
	//wprintf(L"%ls\n", commandLine);
	if (!CreateProcess(env, commandLine, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi))
	{
		throw "Could not compile Haskell GPU code";
	}
	// Wait for the process.
	WaitForSingleObject(pi.hProcess, INFINITE);
	// Close the process.
	CloseHandle(pi.hProcess);
	CloseHandle(pi.hThread);
}

#endif

#if 0


//-gencode=arch=compute_10,code=sm_10 --use-local-env --cl-version 2010 -ccbin "$(VC_BIN_DIR)" -I"$(CUDA_PATH_V4_0)include" --keep-dir "Release" -maxrregcount=0 --machine 32 -cubin -o "S:\Uni - Bio\Branches\Driver API\Comparisons\Bin\x86\Release\Convolution\ConvolverObsOpt.cubin" "ConvolverObsOpt.cu"

//"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA\v4.0\bin\nvcc.exe" -gencode=arch=compute_10,code=sm_10 --use-local-env --cl-version 2010 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin"  -I"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA\v4.0\include"    --keep-dir "Release" -maxrregcount=0  --machine 32 -cubin  -o "S:\Uni - Bio\Branches\Driver API\Comparisons\Bin\x86\Release\Convolution\ConvolverCUDADriver.cubin" "ConvolverCUDADriver.cu" 

#endif
