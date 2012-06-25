@echo off

shift

set VC_BIN_DIR=C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin

@call make Convolver\UDWTHaskell.hs

UDWT.exe %*
