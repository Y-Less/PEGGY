{-# LANGUAGE MultiParamTypeClasses       #-}
{-# LANGUAGE FlexibleInstances           #-}
{-# LANGUAGE TypeFamilies                #-}

module PEGGY.Obsidian.Instances where

import PEGGY.Types
import PEGGY.Obsidian.Natives
import PEGGY.Obsidian.Targets
import PEGGY.Obsidian.Run

import Obsidian.GCDObsidian

import System.IO.Unsafe
import Foreign.Marshal.Array

{- 
 - PYExpresible instances.
 -}

-- instance
		-- (dom0 ~ Float, id0 ~ TargetHaskell) =>
		-- PYExpressible (PYNative dom0 id0) Float TargetHaskell
	-- where
		-- pySet (PYNativeObsidian x y) = PYExpr (PYStorageHaskell x) y
		-- pyRun = id
		-- --pyGet (PYNativeHaskell x y) = x

{- 
 - PYDoRun instances.
 -}



--runAll (PYStorageObsidian func inp _) = CUDA.genKernel "PEGGYKernel" (pure func) inp

instance
		Show (PYStorage Float TargetObsidian)
	where
		show (PYStorageObsidian _ _ a l) =
			let
				show' (PYOArr a) = show a
				show' (PYOPair (a, b)) = "(" ++ show' a ++ ", " ++ show' b ++ ")"
			in
				show l ++ ": " ++ show' a
 
instance
		PYExecute Float TargetObsidian
	where
		pyExecute x = PYNativeObsidian (generateObsidianCode x) (pyLength x)
		--	(generateObsidianCode (pyExpr x) "PEGGYOutputCode" (pyLength x))
		--	(pyLength x)

instance PYDim Float TargetObsidian where
	pyDim 0 d = pyObsDimToExpr d $ variable "blockIdx.x" * variable "X_BLOCK" + variable "threadIdx.x"
	pyDim 1 d = pyObsDimToExpr d $ variable "blockIdx.y"
	pyDim 2 d = pyObsDimToExpr d $ variable "blockIdx.z"
	pyDim _ _ = error "Obsidian only supports 3 dimensions"

pyObsDimToExpr :: (Scalar dom, Num dom) => [Int] -> Exp dom -> PYExpr dom TargetObsidian
pyObsDimToExpr d f = PYExpr (PYStorageObsidian pid pya (PYONone 0) (map (fromInteger . toInteger) d)) d
	where
		-- This is never used, but makes the type checker happy.
		pya = return $ mkPullArray (\ix -> BinOp Add (index "constant" ix) f) 1
		--pya = (namedArray "input0" (getLen len)) -- :: Array Pull (Exp dom)
		pid _ = mkPullArray (\ _ -> f) obsidianThreadCount

-- TODO: Hack Obsidian apart so that it ONLY EVER generates one array name so
-- that they are all actually the same.  This is ugly, but tough!
