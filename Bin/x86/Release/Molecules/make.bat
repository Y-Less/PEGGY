@echo off

set filename=%~n2
set filepath=%~p2
set build=%1

set olddir=%cd%

if "%filename%"=="" (
	if "%build%"=="" (
		echo Usage: "make [/clean] file"
		cd "%olddir%"
		goto :eof
	)
	set filename=%~n1
	set filepath=%~dp1
	set build=
)

cd "%filepath%"

:: Set up Visual Studio in the current context.
if NOT "%PATH:~0,52%"=="C:\Program Files (x86)\Microsoft Visual Studio 10.0\" (
	call "C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin\vcvars32.bat"
)

:: Generate the .def files for the various .dll files ("Bridge", "Accelerator",
:: "CUDA" and "cudatarget"), and convert for GCC use.
if "%build%"=="/clean" (
	echo Curently Unsupported
	cd "%olddir%"
	goto :eof
	:: Delete everything.
	for /R "." %%f in (*.hi) do (
		del "%%f"
	)
	:: This would be done during the "clean" stage if there were such a thing.
	:: Which there now is...
	call :DllToDef "Accelerator"
	call :DefToA "Accelerator"
	call :DllToDef "cudatarget"
	call :DefToA "cudatarget"
	call :DllToDef "Bridge"
	call :DefToA "Bridge"
	call :DllToDef "CUDA"
	call :DefToA "CUDA"
)

:: Build the first DLL using the other DLLs.
echo.
echo Building dll

:: Build the file
::
::ghc load.c --make "%filename%.hs" -hide-package monads-fd -o "%filename%.dll" -shared -static
::ghc load.c --make "%filename%.hs" -hide-package monads-fd -no-hs-main -o "%filename%.dll" -shared -static -optl-lAccelerator -optl-lcudatarget -optl-lBridge -optl-lCUDA -optl-L.
::ghc -c load.c
::ghc -c "%filename%.hs"
::ghc -shared -o "%filename%.dll" -shared -static
ghc load.c --make "%filename%.hs" -hide-package monads-fd -no-hs-main -o "%filename%.dll" -shared -static
::ghc load.c --make "%filename%.hs" -no-hs-main -o "%filename%.dll" -shared -static
:: Constrain exports.
::-optdll--def 0optdll%filename%.def

copy "%filename%.dll" "%olddir%"

:: Don't go any further just yet - takes VASTLY too long to generate a full .def
:: and .h file set for Haskell dlls (until export constraints are applied).
cd "%olddir%"
goto :eof


:: Generate the exports for this DLL file (for static inclusion, not dynamic).
"C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin\dumpbin.exe" /EXPORTS "%filename%.dll" > "%filename%.exports"
call :DllToDef "%filename%"
:: Generate the .def and .lib files for Visual Studio.
echo.
echo Generating %filename%.h and %filename%.def

echo.
echo #include "HsFFI.h" > %filename%.h
echo #ifdef __cplusplus >> %filename%.h
echo extern "C" >> %filename%.h
echo { >> %filename%.h
echo #endif >> %filename%.h

:: TODO: Currently this code assumes 2 4-byte parameters (i.e. assumes that
:: there are 8 bytes put on the stack in C.  This is plainly not the case all
:: the time and some method of determining the (approximate) number of
:: parameters is required.
for /f "usebackq tokens=5* delims=( " %%g in (`findstr /R /C:"^[^)]*[) ]*[^(][^(]*.*" "%filename%_stub.h"`) do (
	echo 	#pragma comment^(linker, "/EXPORT:%%g@8=_%%g@8"^) >> %filename%.h
	echo 	extern HsPtr __stdcall %%g^(%%h >> %filename%.h
)

echo 	#pragma comment^(linker, "/EXPORT:HsStart=_HsStart"^) >> %filename%.h
echo 	extern void HsStart^(^); >> %filename%.h
echo 	#pragma comment^(linker, "/EXPORT:HsEnd=_HsEnd"^) >> %filename%.h
echo 	extern void HsEnd^(^); >> %filename%.h
echo #ifdef __cplusplus >> %filename%.h
echo } >> %filename%.h
echo #endif >> %filename%.h

echo Generating %filename%.lib
echo.
lib /MACHINE:X86 /DEF:%filename%.def /OUT:%filename%.lib

:: End of the main script.
cd "%olddir%"
goto :eof

:DllToDef
	set fpath=%~dp1
	set fname=%~n1
	echo.
	echo "%fname%.dll"
	echo.
	echo  ^> %fname%.exports
	"C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin\dumpbin.exe" /EXPORTS "%fname%.dll" > "%fname%.exports"
	echo  ^> %fname%.def
	echo LIBRARY %fname% > %fname%.def
	echo EXPORTS >> %fname%.def
	:: Regular expressions don't seem to have "+".
	for /f "usebackq tokens=4" %%g in (`findstr /R /C:"^[ 	][ 	]*[0-9][0-9]*[ 	][ 	]*[0-9A-F][0-9A-F]*[ 	][ 	]*[0-9A-F][0-9A-F]*[ 	][ 	]*.*" "%fname%.exports"`) do (
		echo 	%%g >> %fname%.def
	)
goto :eof

:DefToA
	set fpath=%~dp1
	set fname=%~n1
	echo  ^> %fname%.a
	"D:\Program Files\Haskell\mingw\bin\dlltool.exe" -d "%fname%.def" -l "lib%fname%.a"
goto :eof
