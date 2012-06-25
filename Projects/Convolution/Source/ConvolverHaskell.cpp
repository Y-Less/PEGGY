#if !defined NO_CUDA && !defined NO_HASKELL

#include "ConvolverHaskell.h"
#include <cuda_runtime_api.h>

#include <stdio.h>

#define CEILDIV(m,n) \
	(((m) + (n) - 1) / (n))

#define SHAPE_X 1024
#define SHAPE_Y 1

// cons
	ConvolverHaskell::
	ConvolverHaskell() :
		Convolver(),
		ParallelCUDADriver(),
		ParallelHaskell(),
		GenerateConvolution(0),
		GenerateCachedConvolution(0)
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
			{"generateConvolution@16", (void **)&GenerateConvolution},
			//{"generateCachedConvolution@16", (void **)&GenerateCachedConvolution},
			{0, 0}
		};
	ScriptInit(L"ConvolverHaskell.dll", funcs);
	if (GenerateConvolution(GetFilter().GetData(), GetRadius() * 2 + 1, GetHeight(), GetWidth()))
	{
		throw "GenerateConvolution returned non-0";
	}
	// Load the module.
	//printf("Thread: %d\n", GetThreads());
	CUmodule
		module = 0;
	//printf("1\n");
	TryCUDA(cuModuleLoad(&module, "ConvolverHaskell.ptx"));
	// Get the two functions.
	//printf("2 %d\n", module);
	m_rowKernel = 0;
	TryCUDA(cuModuleGetFunction(&m_rowKernel, module, "DoRows"));
	//printf("3\n");
	m_colKernel = 0;
	TryCUDA(cuModuleGetFunction(&m_colKernel, module, "DoCols"));
	//printf("4\n");
	// Get the data we want.
	m_data = (float *)GetData().ToCUDAArray();
	// Get the temporary arrays.
	m_smoothX = (CUdeviceptr)GetSmoothX().ToCUDAArray(),
	//printf("8\n");
	m_smoothY = (CUdeviceptr)GetSmoothY().ToCUDAArray();
	//printf("9\n");
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
	void *
		params0[] =
		{
			(void *)&m_data, // SHOULD BE 0, BUT CAN'T BE.  UNUSED.
			(void *)&m_data,
			(void *)&m_smoothX,
		};
	TryCUDA(cuLaunchKernel(m_rowKernel, CEILDIV((int)width, SHAPE_X), CEILDIV((int)height, SHAPE_Y), 1, SHAPE_X, SHAPE_Y, 1, 0, 0, params0, 0));
	//DoRows(m_smoothX, m_data, (int)height, (int)width, GetThreads(), (int)pitch, (int)radius);
	End("X");
	Log("Y");
	// Do the Y dimension.
	//DoCols(m_smoothY, m_smoothX, (int)height, (int)width, GetThreads(), (int)pitch, (int)radius);
	void *
		params1[] =
		{
			(void *)&m_data, // SHOULD BE 0, BUT CAN'T BE.  UNUSED.
			(void *)&m_smoothX,
			(void *)&m_smoothY,
		};
	TryCUDA(cuLaunchKernel(m_colKernel, CEILDIV((int)height, SHAPE_X), CEILDIV((int)width, SHAPE_Y), 1, SHAPE_X, SHAPE_Y, 1, 0, 0, params1, 0));
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
	/*CUDA_MEMCPY2D
		memdata = {0};
	memdata.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	memdata.srcDevice = m_smoothY;
	memdata.srcPitch = pitch * sizeof (float);
	memdata.dstMemoryType = CU_MEMORYTYPE_HOST;
	memdata.dstHost = GetStore();
	memdata.dstPitch = size;
	memdata.WidthInBytes = size;
	memdata.Height = height;
	TryCUDA(cuMemcpy2D(&memdata));*/
	// TODO: See if the code above still works - I only removed it when I was
	// trying to figure out why my code suddently stopped working.
	cudaMemcpy2D(GetStore(), size, (void *)m_smoothY, pitch * sizeof (float), size, height, cudaMemcpyDeviceToHost);
	End("Copy");
}

void
	ConvolverHaskell::
	ConvExit() throw (...)
{
	//ScriptClose(false);
};

// Compilation is now done on the Haskell side as:
//  a) It's VASTLY simpler and
//  b) There's no reason to think that it will actually NEED compiling, we are
//     getting results from a lazy language, can hapilly cache results.
/*
void
	ConvolverHaskell::
	Compile(LPCWSTR file)
{
	static WCHAR
		commandLine[1024],
		env[256],
		dir[256];
	STARTUPINFO
		si = {0};
	si.cb = sizeof (si);
	PROCESS_INFORMATION
		pi = {0};
	//printf("%d %d\n", pi.dwProcessId, pi.hProcess);
	// Generate the command-line parameters.
	GetEnvironmentVariable(L"CUDA_PATH_V4_0", env, sizeof (env));
	//wsprintf(commandLine, L"\"%sbin\\nvcc.exe\"", (LPSTR)env);
	GetEnvironmentVariable(L"VC_BIN_DIR", dir, sizeof (dir));
	wsprintf(commandLine, L"nvcc.exe -gencode=arch=compute_10,code=sm_10 --use-local-env --cl-version 2010 -ccbin \"%s\" -I\"%sinclude\"    --keep-dir \"Release\" -maxrregcount=0  --machine 32 -cubin  -o \"%s.cubin\" \"%s.cu\"", (LPSTR)dir, (LPSTR)env, file, file);
	//GetEnvironmentVariable(L"CUDA_PATH_V4_0", env, sizeof (env));
	//GetEnvironmentVariable(L"RUN_DIRECTORY", dir, sizeof (dir));
	//wsprintf(commandLine + wcslen(commandLine), L" -I\"%sinclude\" --keep-dir \"Release\" -maxrregcount=0 --machine 32 -cubin -o \"%s%s.cubin\" \"%s%s.cu\"", env, dir, file, dir, file);
	// Generate the compiler name.
	//wcscat(env, L"bin\\nvcc.exe");
	// Create the process.
	//wprintf(L"%ls\n", env);
	//wprintf(L"%ls\n", commandLine);
	//if (!CreateProcess(env, commandLine, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi))
	return;
	if (!CreateProcess(0, commandLine, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi))
	{
		throw "Could not compile Haskell GPU code";
	}
	// Wait for the process.
	WaitForSingleObject(pi.hProcess, INFINITE);
	//printf("done");
	// Close the process.
	CloseHandle(pi.hProcess);
	CloseHandle(pi.hThread);
	//printf("bye\n");
}
*/

#endif

#if 0


//-gencode=arch=compute_10,code=sm_10 --use-local-env --cl-version 2010 -ccbin "$(VC_BIN_DIR)" -I"$(CUDA_PATH_V4_0)include" --keep-dir "Release" -maxrregcount=0 --machine 32 -cubin -o "S:\Uni - Bio\Branches\Driver API\Comparisons\Bin\x86\Release\Convolution\ConvolverObsOpt.cubin" "ConvolverObsOpt.cu"

//"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA\v4.0\bin\nvcc.exe" -gencode=arch=compute_10,code=sm_10 --use-local-env --cl-version 2010 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin"  -I"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA\v4.0\include"    --keep-dir "Release" -maxrregcount=0  --machine 32 -cubin  -o "S:\Uni - Bio\Branches\Driver API\Comparisons\Bin\x86\Release\Convolution\ConvolverCUDADriver.cubin" "ConvolverCUDADriver.cu" 

void
	ConvolverHaskell::
	Compile(LPCWSTR file)
{
	static WCHAR
		commandLine[1024],
		dir[256],
		env[256];
	STARTUPINFO
		si = {0};
	si.cb = sizeof (si);
	PROCESS_INFORMATION
		pi = {0};
	printf("%d %d\n", pi.dwProcessId, pi.hProcess);
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
	if (!GetEnvironmentVariable(L"RUN_DIRECTORY", dir, sizeof (dir)))
	{
		if (GetLastError() == ERROR_ENVVAR_NOT_FOUND)
		{
			throw "Could not find $(RUN_DIRECTORY)";
		}
		else
		{
			throw "Unknown environment variable error";
		}
	}
	wsprintf(commandLine + wcslen(commandLine), L" -I\"%sinclude\" --keep-dir \"Release\" -maxrregcount=0 --machine 32 -cubin -o \"%s%s.cubin\" \"%s%s.cu\"", env, dir, file, dir, file);
	// Generate the compiler name.
	wcscat(env, L"bin\\nvcc.exe");
	// Create the process.
	wprintf(L"%ls\n", env);
	wprintf(L"%ls\n", commandLine);
	//if (!CreateProcess(env, commandLine, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi))
	if (!CreateProcess(env, commandLine, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi))
	{
		throw "Could not compile Haskell GPU code";
	}
	// Wait for the process.
	WaitForSingleObject(pi.hProcess, INFINITE);
	printf("done");
	// Close the process.
	CloseHandle(pi.hProcess);
	CloseHandle(pi.hThread);
	printf("bye\n");
}

#endif
