-- Implementation of the Convolution example in PEGGY.

module Convolution where

import PEGGY
import PEGGY.Obsidian

import Foreign.Marshal.Array
import Foreign.Ptr

--foreign export stdcall generateConvolution :: Ptr Float -> Int -> Int -> Int -> IO Int

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- Main.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

convolve d filter input = foldl (+) 0 convolve'
	where
		convolve' = zipWith doOneShift [-radius .. ] filter
		-- Convert the input to GPU data representation.
		pyInput = pySet input
		-- Get the bounds of the convolution filter.
		radius = length filter `div` 2
		-- Shift once and multiply.
		doOneShift sBy mBy = pyInput .<<<. (sBy : d) * pyConst mBy

dimX = [] :: [Int]
dimY = [0] :: [Int]

generateConvolution filter len height width = do
	filter' <- peekArray len filter
	let
		placeholder = Replicate [height, width] 0.0
		rowCode, colCode :: PYExpr Float TargetObsidian
		rowCode = convolve dimX filter' placeholder
		colCode = convolve dimY filter' placeholder
	rc <- generateObsidianCode rowCode
	cc <- generateObsidianCode colCode
	let
		rows = setKernelName rc "DoRows"
		-- "setKernelName" adds a token to be detected by "compileCUDA" if a
		-- kernel name has been set, as we are concatenating two kernels, we
		-- don't want this token in the second one.
		cols = tail $ setKernelName cc "DoCols"
	--putStrLn (rows ++ cols)
	compileCUDA (rows ++ cols) "ConvolverHaskell"
