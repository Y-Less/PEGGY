{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE FlexibleContexts #-}

module PEGGY.Examples where

import PEGGY
--import PEGGY.Accelerator
--import PEGGY.Repa
import PEGGY.Obsidian
import PEGGY.Haskell

--pyRunDX9 :: (PYExecute dom TargetDX9) => PYExpr dom TargetDX9 -> PYNative dom TargetDX9
--pyRunDX9 = pyRun

--pyRunX64 :: (PYExecute dom TargetX64) => PYExpr dom TargetX64 -> PYNative dom TargetX64
--pyRunX64 = pyRun

pyRunHaskell :: (PYExecute Float TargetHaskell) => PYExpr Float TargetHaskell -> PYNative Float TargetHaskell
pyRunHaskell = pyRun

m = pySet [0 .. 9999] -- [100, 200, 300, 400]
--nX = m :: PYExpr Float TargetX64
--nD = m :: PYExpr Float TargetDX9
nH = m :: PYExpr Float TargetHaskell
--nR = m :: PYExpr Float TargetRepa
--O = m :: PYExpr Float TargetHaskell

m2 = pySet $ concat (replicate 128 [0 .. 255]) -- [100, 200, 300, 400]
--n2X = m2 :: PYExpr Float TargetX64
--n2D = m2 :: PYExpr Float TargetDX9
n2H = m2 :: PYExpr Float TargetHaskell
--n2R = m2 :: PYExpr Float TargetRepa
