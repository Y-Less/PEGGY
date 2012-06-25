@echo off

cd ..\Bin\x86\Release\Convolution

:: Do 1024xN, 2048xN, 3072xN, 4096xN, 5120xN, 6144xN, 7168xN, 8192xN
:: Need to see if 8192x8192 can even be run on CPU with the memory requirements.
:: Accelerator can only go up to 4096x4096.
FOR /L %%i IN (7,8,63) DO (
	FOR /L %%k in (7,8,63) DO (
		echo =========
		echo  %%i X %%k
		echo =========
		Convolution.exe %%i %%k 5 cderjkax c
		move "Output\Report2 %%i %%k.txt" "Output\Report4 %%i %%k 5 c.txt"
	)
)
::		Convolution.exe %%i %%k 5 cderjkax s
::		move "Output\Report2 %%i %%k.txt" "Output\Report4 %%i %%k 5 s.txt"

:: Also don't know how to do "else"
cd ../../../../Tests
