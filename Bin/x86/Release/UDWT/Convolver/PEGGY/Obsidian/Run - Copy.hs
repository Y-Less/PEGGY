{-# LANGUAGE FlexibleInstances #-}

module PEGGY.Obsidian.Run where

import System.Cmd

import PEGGY.Types
import PEGGY.Obsidian.Natives
import PEGGY.Obsidian.Targets

import qualified Obsidian.GCDObsidian.CodeGen.CUDA as CUDA
import Obsidian.GCDObsidian

import Data.List (elemIndex, (!!))

import Data.IORef

import Foreign.Ptr
import Data.Word
import System.Exit


import Control.Monad.State (runState)


import System.Environment
import System.Directory
import System.Cmd
import Prelude hiding (catch)
import System.IO.Error hiding (catch)
import Control.Exception
import Obsidian.GCDObsidian.Program

--main = system "echo hi"

-- ENSURE THAT "vcvarsall.bat" IS CALLED FIRST.
{-pyCompile = unwords [
	"nvcc",
	"--cubin",
	"--gpu-architecture sm_20",
	"--output-file PEGGYKernel.cubin",
	"--cl-version 2010",
	"--optimize 2",
	"--machine 32",
	"--use_fast_math",
	"PEGGYKernel.cu"]-}

pyCompile = unwords [
	"nvcc",
	--"-gencode=arch=compute_10,code=sm_21",
	"-arch=sm_21",
	"--use-local-env",
	"--cl-version 2010",
	"-ccbin \"C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\bin\"",
	"-I\"C:\\Program Files (x86)\\NVIDIA GPU Computing Toolkit\\CUDA\\v4.0\\include\"",
	"--keep-dir \"Release\"",
	"-maxrregcount=0",
	"--machine 32",
	"-cubin",
	"-o \".\\PEGGYKernel.cubin\"",
	"PEGGYKernel.cu"]
{-
pyObRunAllFloat :: PYStorage Float TargetObsidian -> (IO (Ptr Float), Word32)
pyObRunAllFloat (PYStorageObsidian func inp arr leng) = (result, leng)
	where
		-- Generate full valid code for compile.
		config @ (_, _, ins, _) = CUDA.configureKernel (pure $ final . func) inp
		code0 = CUDA.makeKernel "PEGGYKernel" config
		line = lines code0
		-- Get the data sizes
		inputs = length ins
		leng' = fromInteger . toInteger $ leng
		outputPart = unlines [
			"{",
			"  int bidx = blockIdx.x;"]
		final a = mkPullArray (\ ix -> a ! ix) (min (len a) 512)
		-- The finally generated code.
		code1 = "extern \"C\" {" ++ head line ++ outputPart ++ (unlines . tail $ line) ++ "}}"
		--code1 = error $ show inputs
		-- Now that we have the code, write it to a file and compile it.
		result = do
			-- Write.
			pyObOutput code1
			-- Compile.
			comp1 <- system pyCompile
			let
				comp1 = case comp1 of
					ExitSuccess -> undefined
					ExitFailure i -> error ("ExitFailure" ++ show i)
			-- Now start the C plugin
			hs_CUDAFloatStart inputs leng'
			-- Push all parameters.
			pyoPush arr []
			-- Run the code.
			hs_CUDAFloatRun leng'
			-- End the code.
			hs_CUDAFloatEnd

pyObOutput = writeFile "./PEGGYKernel.cu"

-- Run the GPU code through CUDA.
-- CUDAStart count
-- map CUDAPush data
-- CUDAEnd

ppiShow :: ([(Array Pull (Exp Float), Int)], String) -> String
ppiShow = show . snd
-}



outputPart1 = unlines [
	"#define bidx blockIdx.x",
	"",
	"extern \"C\" __global__ void kernel1(float *input0,float *result0)",
	"{",
	""]

outputPart2 = unlines [
	"",
	"extern \"C\" __global__ void kernel2(float *input0,float *result0)",
	"{",
	""]

outputPart3 = ""

--cudaCode func arr = CUDA.genKernel "kernel" (pure func) arr
cudaCode (PYStorageObsidian oldFunc a _ l) = CUDA.genKernel "kernel" (pure (pa . oldFunc)) arrs
	where
		(arrs, _) = runState a 0
		pa (Array (Pull f) len) = (mkPushArray (\ func -> ForAll (\i -> func (variable "blockIdx.y" * variable "PITCH" + variable "blockIdx.x" * variable "X_BLOCK" + i, f (i + variable "blockIdx.x" * variable "X_BLOCK"))) len) len)







--transpose dims (Array (Pull f) len) = mkPushArray (\ func -> ForAll (\i -> func (variable "blockIdx.y" * variable "PITCH" + i, f (i + variable "blockIdx.x" * variable "X_BLOCK"))) len) len


--generateCode obj threads = unlines . tail . lines $ cudaCode obj
generateCode obj threads = cudaCode obj

--generateConvolution = generateConvolutionInternal "ConvolverHaskell"

--generateConvolutionInternal :: String -> Word32 -> Float -> Word32 -> Word32 -> Word32 -> Word32 -> IO Int
generateConvolutionInternal obj2 fname = doCompile (outputPart1 ++ outputPart2 ++ p1 ++ outputPart3) fname
	where
		-- Yes, "w h" then "h w" is correct to transpose twice, as is the NO
		-- transposition of thread counts.
		--p0 = generateCode obj1
		p1 = generateCode obj2 512
		--dimsxy = (Dims w h xThreads yThreads)
		--dimsyx = (Dims h w xThreads yThreads)

doCompile code file = compile' `catch` handler'
	where
		handler' e =
			-- Propgate the error to the calling C code sort of gracefully-ish.
			if isDoesNotExistError e
			then return 1
			else return (-1)
		compile' = do
			if code == []
			then return ()
			else writeFile (".\\" ++ file ++ ".cu") code
			msvc <- getEnv "VC_BIN_DIR"
			nvcc <- getEnv "CUDA_PATH"
			let
				commandLine = unwords [
					"nvcc.exe",
					"-gencode=arch=compute_20,code=\\\"sm_20,compute_20\\\"",
					"--use-local-env",
					"--cl-version 2010",
					"-ccbin \"" ++ msvc ++ "\"",
					"-I\"" ++ nvcc ++ "include\"",
					"--keep-dir \"Release\"",
					"-maxrregcount=0",
					"--machine 32",
					"-ptx",
					"-o \".\\" ++ file ++ ".ptx\"",
					"\".\\" ++ file ++ ".cu\""]
			--putStrLn commandLine
			system commandLine
			return 0

