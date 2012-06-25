@echo off

S:

cd "S:\SVN - Bio\Branches\Driver API\Comparisons\Tests"

cd ..\Bin\x86\Release\Convolution

@call run.bat
FOR /L %%i IN (8,1,10) DO (
		echo =========
		echo  %%i X 31
		echo =========
		Convolution.exe 31 %%i 5 acrfxk c
		move "Output\Report2 31 %%i.txt" "Output\Report5 31 %%i 5 c.txt"
	)
)
FOR /L %%i IN (32,1,32) DO (
		echo =========
		echo  %%i X 31
		echo =========
		Convolution.exe 31 %%i 5 acrfxk c
		move "Output\Report2 31 %%i.txt" "Output\Report5 31 %%i 5 c.txt"
	)
)

cd ..\..\..\..\Tests
