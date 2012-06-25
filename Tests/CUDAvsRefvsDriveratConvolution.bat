@echo off

cd ..\Bin\x86\Release\Convolution

:: Do 1024xN, 2048xN, 3072xN, 4096xN, 5120xN, 6144xN, 7168xN, 8192xN
:: Need to see if 8192x8192 can even be run on CPU with the memory requirements.
:: Accelerator can only go up to 4096x4096.
FOR /L %%i IN (7,8,63) DO (
	FOR /L %%k in (0,1,63) DO (
		Convolution.exe %%i %%k 5 cj s
	)
)
::dihr
:: This line goes between the "cd" lines
:: Don't know how to do comments in blocks
		::genconv.exe %%i %%k
:: This line goes after them
		::msbuild /p:Configuration=Release Projects\Convolution\Convolution.vcxproj
:: Also don't know how to do "else"
cd ../../../../Tests
