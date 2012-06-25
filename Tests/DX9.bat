@echo off

cd ..\Bin\x86\Release\Convolution

@call run.bat
:: Do 1024xN, 2048xN, 3072xN, 4096xN, 5120xN, 6144xN, 7168xN, 8192xN
:: Need to see if 8192x8192 can even be run on CPU with the memory requirements.
:: Accelerator can only go up to 4096x4096.
FOR /L %%i IN (0,1,63) DO (
		echo =========
		echo  %%i X 31
		echo =========
		Convolution.exe 31 %%i 5 a c
		move "Output\Report2 31 %%i.txt" "Output\Report4 31 %%i 5 c.txt"
	)
)

cd ..\UDWT

@call run.bat
FOR /L %%i IN (0,1,63) DO (
		echo =====
		echo  %%i
		echo =====
		UDWT.exe %%i 1 a c
		move "Output\Report2 %%i 0.txt" "Output\Report4 %%i 0 1 c.txt"
	)
)

cd ..\Molecules

@call run.bat
FOR /L %%i IN (0,1,63) DO (
		echo =====
		echo  %%i
		echo =====
		Molecules.exe 31 %%i 10 a c
		move "Output\Report2 31 %%i.txt" "Output\Report4 31 %%i 10 c.txt"
	)
)

cd ..\..\..\..\Tests
