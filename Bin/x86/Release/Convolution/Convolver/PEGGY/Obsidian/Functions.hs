{-# LANGUAGE MultiParamTypeClasses       #-}
{-# LANGUAGE UndecidableInstances        #-}
{-# LANGUAGE FlexibleInstances           #-}
{-# LANGUAGE FlexibleContexts            #-}
{-# LANGUAGE ViewPatterns                #-}

module PEGGY.Obsidian.Functions where

import PEGGY.Types
import PEGGY.Functions
import PEGGY.Obsidian.Natives
import PEGGY.Obsidian.Targets
import PEGGY.Obsidian.Instances
import qualified Obsidian.GCDObsidian.Types as Type

import Obsidian.GCDObsidian

import Data.Word


liftPYObsidian2 ::
		-- (PYStorable dom TargetObsidian) =>
		(Exp dom -> Exp dom -> Exp dom) -> PYExpr dom TargetObsidian -> PYExpr dom TargetObsidian -> PYExpr dom TargetObsidian
liftPYObsidian2 func x y =
		PYExpr (pyObsMerge2 func (pyExpr x) (pyExpr y)) (pyNewLength (pyLength x) (pyLength y))

liftPYObsidian1 ::
		-- (PYStorable dom TargetObsidian) =>
		(Exp dom -> Exp dom) -> PYExpr dom TargetObsidian -> PYExpr dom TargetObsidian
liftPYObsidian1 func x =
		PYExpr (pyObsMerge1 func (pyExpr x)) (pyLength x)

instance
		(PY2 c0 c1 Float TargetObsidian) =>
		PYAdd c0 c1 Float TargetObsidian
	where
		pyAdd = pyLift2 (liftPYObsidian2 (+))

instance
		(PY2 c0 c1 Float TargetObsidian) =>
		PYSub c0 c1 Float TargetObsidian
	where
		pySub = pyLift2 (liftPYObsidian2 (-))

instance
		(PY2 c0 c1 Float TargetObsidian) =>
		PYMul c0 c1 Float TargetObsidian
	where
		pyMul = pyLift2 (liftPYObsidian2 (*))

instance
		PYAbs c0 Float TargetObsidian
	where
		pyAbs = undefined

instance
		(PY1 c0 Float TargetObsidian) =>
		PYNegate c0 Float TargetObsidian
	where
		pyNegate = pyLift1 (liftPYObsidian1 ((-) (Literal 0)))






instance
		(PY1 c0 Float TargetObsidian) =>
		PYSqrt c0 Float TargetObsidian
	where
		pySqrt = pyLift1 (liftPYObsidian1 sqrt)
		--pySqrt :: (PY1 c0 dom id) => c0 -> PYExpr dom id





{-
instance
		PYMul (PYExpr Float TargetObsidian) (PYExpr Float TargetObsidian) Float TargetObsidian
	where
		pyMul x y = (liftPYObsidian2 (*)) x y

instance
		PYMul (PYExpr Float TargetObsidian) ([] Float) Float TargetObsidian
	where
		pyMul x y = (liftPYObsidian2 (*)) x (pyExpressibleList y)

instance
		PYMul ([] Float) (PYExpr Float TargetObsidian) Float TargetObsidian
	where
		pyMul x y = (liftPYObsidian2 (*)) (pyExpressibleList x) y

instance
		PYMul ([] Float) ([] Float) Float TargetObsidian
	where
		pyMul x y = (liftPYObsidian2 (*)) (pyExpressibleList x) (pyExpressibleList y)

instance
		PYMul (Replicate Float) (PYExpr Float TargetObsidian) Float TargetObsidian
	where
		pyMul (Replicate _ x) y = (liftPYObsidian1 $ (*) (Literal x)) y

instance
		PYMul (PYExpr Float TargetObsidian) (Replicate Float) Float TargetObsidian
	where
		pyMul x (Replicate _ y) = (liftPYObsidian1 $ (*) (Literal y)) x

-- WHY?
instance
		PYMul (Replicate Float) (Replicate Float) Float TargetObsidian
	where
		pyMul (Replicate _ x) y = (liftPYObsidian1 $ (*) (Literal x)) (pyExpressibleReplicate y)

instance
		PYMul (Replicate Float) ([] Float) Float TargetObsidian
	where
		pyMul (Replicate _ x) y = (liftPYObsidian1 $ (*) (Literal x)) (pyExpressibleList y)

instance
		PYMul ([] Float) (Replicate Float) Float TargetObsidian
	where
		pyMul x (Replicate _ y) = (liftPYObsidian1 $ (*) (Literal y)) (pyExpressibleList x)
-}
























instance
		(PY2 c0 c1 Float TargetObsidian) =>
		PYDiv c0 c1 Float TargetObsidian
	where
		pyDiv = pyLift2 (liftPYObsidian2 (/))

instance
		(PY2 c0 c1 Float TargetObsidian) =>
		PYMin c0 c1 Float TargetObsidian
	where
		pyMin = pyLift2 (liftPYObsidian2 min)

instance
		(PY2 c0 c1 Float TargetObsidian) =>
		PYMax c0 c1 Float TargetObsidian
	where
		pyMax = pyLift2 (liftPYObsidian2 max)

instance
		(PY1 c0 Float TargetObsidian) =>
		PYShiftExtend c0 Float TargetObsidian
	where
		--pyShiftExtend [0] = pySet
		pyShiftExtend shift =
			let
				--fi x = Literal . fromInteger . toInteger $ (min x 512)
				shift' = map (fromInteger . toInteger) shift
				
				pyShiftExtend' (PYExpr (PYStorageObsidian func inp arr len1) len2) =
					PYExpr (PYStorageObsidian (func' func len1) inp arr len1) len2
				
				-- Shift the array, taking in to account the fact that Obsidian
				-- adds "bidx * len" on to all shifts.  Also, no longer does a
				-- signed comparison on an unsigned number.  Now doesn't do the
				-- silly "bidx * len" - I removed that from Obsidian.
				func' ff'' trueLength a =
					mkPullArray (\ ix -> ff'' a ! obsClamp trueLength shift' ix) 1024
				
				-- Ok, I need a generic way to shift in any and all dimensions.
				-- I have previously written this for the list backend, but that
				-- id VERY VERY different.
				-- Get a list of dimension sizes and the offset to apply to that
				-- dimension.  Any shift that's 0 is ignored (or pretty much
				-- optimised out)
				--sizeOffsets = zip len2 shift
				
				-- Shift the value in the range of the input array.
				--clamp' x max off = ifThenElse (x + off <=* Literal 2147483647) (ifThenElse (x + off <* max) (x) (max - Literal 1 - off)) (Literal 0 - off)
				-- Old signed version.
				--clamp' x max off = ifThenElse (x + off >=* Literal 0) (ifThenElse (x + off <* max) (x) (max - Literal 1 - off)) (Literal 0 - off)
			in
				pyLift1 pyShiftExtend'

obsClamp :: [Word32] -> [Word32] -> Exp Word32 -> Exp Word32
obsClamp dims' shifts' idx = case dimCount of
		1 -> x idx shifts dims'
		2 -> y idx shifts dims'
		3 -> z idx shifts dims' --(Array (Pull f) n) = mkPullArray f' n
	where
		dimCount = length dims'
		-- Get shifts accurately equal to the number of dimensions.
		shifts = drop (length shifts' + 3 - dimCount) ([0, 0, 0] ++ shifts')
		-- Ironically, after hacking Obsidian to REMOVE the blockDim.x code, I
		-- now need MORE fixes to remove other dimensions.  BUT, this is because
		-- Obsidian doesn't natively support 2D arrays, so everything else is
		-- just a massive load of hacks.
		-- NOT GENERIC IN ANY WAY!
		--x idx (0 : rst) (d : dim) = idx
		x idx (s : rst) (d : dim)
			| s == 0    = idx
			| otherwise = clampOne' (idx + Literal s) d -- variable "blockIdx.x" * variable "X_BLOCK"
		y idx (s : rst) (d : dim)
			| s == 0    = x idx rst dim
			| otherwise = x idx rst dim + clampOne' (variable "blockIdx.y" + Literal s) d * Literal (head dim) - variable "blockIdx.y" * Literal (head dim)
		z idx (s : rst) (d : dim)
			| s == 0    = y idx rst dim
			| otherwise = y idx rst dim + clampOne' (variable "blockIdx.z" + Literal s) d * Literal (head dim) * Literal (head (tail dim))
		-- Clamp an address to a size
		clampOne' :: Exp Word32 -> Word32 -> Exp Word32
		clampOne' i size = ifThenElse (
			CastOp Type.Word32 Type.Int i >=* zero) (
				ifThenElse (i <* Literal size) (
					i) (
					Literal size - Literal 1)) (
				zero)
		zero = Literal 0

instance
		(PY1 c0 Float TargetObsidian) =>
		PYShiftRotate c0 Float TargetObsidian
	where
		--pyShiftExtend [0] = pySet
		pyShiftRotate shift =
			let
				--fi x = Literal . fromInteger . toInteger $ (min x 512)
				shift' = map (fromInteger . toInteger) shift
				
				pyShiftExtend' (PYExpr (PYStorageObsidian func inp arr len1) len2) =
					PYExpr (PYStorageObsidian (func' func len1) inp arr len1) len2
				
				-- Shift the array, taking in to account the fact that Obsidian
				-- adds "bidx * len" on to all shifts.  Also, no longer does a
				-- signed comparison on an unsigned number.  Now doesn't do the
				-- silly "bidx * len" - I removed that from Obsidian.
				func' ff'' trueLength a =
					mkPullArray (\ ix -> ff'' a ! obsRot trueLength shift' ix) 1024
				
				-- Ok, I need a generic way to shift in any and all dimensions.
				-- I have previously written this for the list backend, but that
				-- id VERY VERY different.
				-- Get a list of dimension sizes and the offset to apply to that
				-- dimension.  Any shift that's 0 is ignored (or pretty much
				-- optimised out)
				--sizeOffsets = zip len2 shift
				
				-- Shift the value in the range of the input array.
				--clamp' x max off = ifThenElse (x + off <=* Literal 2147483647) (ifThenElse (x + off <* max) (x) (max - Literal 1 - off)) (Literal 0 - off)
				-- Old signed version.
				--clamp' x max off = ifThenElse (x + off >=* Literal 0) (ifThenElse (x + off <* max) (x) (max - Literal 1 - off)) (Literal 0 - off)
			in
				pyLift1 pyShiftExtend'

obsRot :: [Word32] -> [Word32] -> Exp Word32 -> Exp Word32
obsRot dims' shifts' idx = case dimCount of
		1 -> x idx shifts dims'
		2 -> y idx shifts dims'
		3 -> z idx shifts dims' --(Array (Pull f) n) = mkPullArray f' n
	where
		dimCount = length dims'
		-- Get shifts accurately equal to the number of dimensions.
		shifts = drop (length shifts' + 3 - dimCount) ([0, 0, 0] ++ shifts')
		x idx (s : rst) (d : dim)
			| s == 0    = idx
			| otherwise = (idx + Literal s) `mod` Literal d
		y idx (s : rst) (d : dim)
			| s == 0    = x idx rst dim
			| otherwise = x idx rst dim + ((variable "blockIdx.y" + Literal s) `mod` Literal d) * Literal (head dim) - variable "blockIdx.y" * Literal (head dim)
		z idx (s : rst) (d : dim)
			| s == 0    = y idx rst dim
			| otherwise = y idx rst dim + ((variable "blockIdx.z" + Literal s) `mod` Literal d) * Literal (head dim) * Literal (head (tail dim))
