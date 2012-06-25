{-# LANGUAGE MultiParamTypeClasses       #-}
{-# LANGUAGE UndecidableInstances        #-}
{-# LANGUAGE FlexibleInstances           #-}
{-# LANGUAGE FlexibleContexts            #-}
{-# LANGUAGE ViewPatterns                #-}

module PEGGY.Repa.Functions where

import PEGGY.Types
import PEGGY.Functions
import PEGGY.Repa.Targets
import PEGGY.Repa.Instances
import PEGGY.FloatBound

import Prelude hiding ((++), zipWith, zipWith3, head, last, drop, take, length, map)

import Data.Array.Repa as Repa

pySetRepa' (pyExpr -> PYStorageRepa z) = z

{-liftPYRepaCond2 ::
		(PYStorable dom TargetRepa, IArray UArray dom,
		 PYStorable Bool TargetRepa) =>
		(dom -> dom -> Bool) -> PYExpr dom TargetRepa -> PYExpr dom TargetRepa -> PYExpr Bool TargetRepa
liftPYRepaCond2 func x y =
		pySet $ zipWith func (pySetRepa' x) (pySetRepa' y)-}

pyArrLen (extent -> (_ :. len)) = len

liftPYRepa2 ::
		(PYStorable dom TargetRepa, Elt dom) =>
		(dom -> dom -> dom) -> PYExpr dom TargetRepa -> PYExpr dom TargetRepa -> PYExpr dom TargetRepa
liftPYRepa2 func x y = PYExpr (PYStorageRepa $ zipWith func (pySetRepa' x) (pySetRepa' y)) (pyLength x)

liftPYRepa1 ::
		(PYStorable dom TargetRepa, Elt dom) =>
		(dom -> dom) -> PYExpr dom TargetRepa -> PYExpr dom TargetRepa
liftPYRepa1 func x = PYExpr (PYStorageRepa $ map func (pySetRepa' x)) (pyLength x)

instance
		(PY2 c0 c1 Float TargetRepa) =>
		PYAdd c0 c1 Float TargetRepa
	where
		pyAdd = pyLift2 (liftPYRepa2 (+))

instance
		(PY2 c0 c1 Float TargetRepa) =>
		PYSub c0 c1 Float TargetRepa
	where
		pySub = pyLift2 (liftPYRepa2 (-))

instance
		(PY2 c0 c1 Float TargetRepa) =>
		PYMul c0 c1 Float TargetRepa
	where
		pyMul = pyLift2 (liftPYRepa2 (*))

instance
		(PY2 c0 c1 Float TargetRepa) =>
		PYDiv c0 c1 Float TargetRepa
	where
		pyDiv = pyLift2 (liftPYRepa2 (/))

instance
		(PY2 c0 c1 Float TargetRepa) =>
		PYMin c0 c1 Float TargetRepa
	where
		pyMin = pyLift2 (liftPYRepa2 min)

instance
		(PY2 c0 c1 Float TargetRepa) =>
		PYMax c0 c1 Float TargetRepa
	where
		pyMax = pyLift2 (liftPYRepa2 max)

instance
		(PY1 c0 Float TargetRepa) =>
		PYAbs c0 Float TargetRepa
	where
		pyAbs = pyLift1 (liftPYRepa1 abs)

instance
		(PY1 c0 Float TargetRepa) =>
		PYMaxVal c0 Float TargetRepa
	where
		pyMaxVal =
			let
				maximum' (pySetRepa' -> x) = pySet (Repa.fold max maxBound $ reshape (Z :. pyArrLen x :. 1) x)
			in
				pyLift1 maximum'

instance
		(PY1 c0 Float TargetRepa) =>
		PYMinVal c0 Float TargetRepa
	where
		pyMinVal =
			let
				minimum' (pySetRepa' -> x) = pySet (Repa.fold min minBound $ reshape (Z :. pyArrLen x :. 1) x)
			in
				pyLift1 minimum'

instance
		(PY1 c0 Float TargetRepa) =>
		PYNegate c0 Float TargetRepa
	where
		pyNegate = pyLift1 (liftPYRepa1 negate)

instance
		(PY1 c0 Float TargetRepa) =>
		PYProduct c0 Float TargetRepa
	where
		pyProduct =
			let
				product' (pySetRepa' -> x) = pySet (Repa.fold (*) minBound $ reshape (Z :. pyArrLen x :. 1) x)
			in
				pyLift1 product'

instance
		(PY1 c0 Float TargetRepa) =>
		PYSum c0 Float TargetRepa
	where
		pySum =
			let
				sum' (pySetRepa' -> x) = pySet (Repa.sum $ reshape (Z :. pyArrLen x :. 1) x)
			in
				pyLift1 sum'

instance
		(PY1 c0 Float TargetRepa) =>
		PYSqrt c0 Float TargetRepa
	where
		pySqrt = pyLift1 (liftPYRepa1 sqrt)

{-instance
		(PY1 c0 Float TargetRepa) =>
		PYShift c0 Float TargetRepa
	where
		pyShift q =
			let
				-- First or last element replicated by abs(shift).
				extra' f s x = V.replicate (abs s) (f (pySetRepa' x))
				-- Shift the array, replicating the first or last element.
				pyShift' s x
					| s > 0     = pySet $ drop s (pySetRepa' x) ++ extra' last s x
					| s < 0     = pySet $ extra' head s x ++ take (length (pySetRepa' x) + s) (pySetRepa' x)
					| otherwise = x
			in
				pyLift1 (pyShift' q)

instance
		(PY1 c0 Float TargetRepa) =>
		PYRotate c0 Float TargetRepa
	where
		pyRotate q =
			let
				pyRotate' s x
					| s > 0     = pySet $ drop s (pySetRepa' x) ++ take s (pySetRepa' x)
					| s < 0     = pySet $ drop (length (pySetRepa' x) + s) (pySetRepa' x) ++ take (length (pySetRepa' x) + s) (pySetRepa' x)
					| otherwise = x
			in
				pyLift1 (pyRotate' q)

instance
		(PY0 c0 Bool TargetRepa,
		 PY2 c1 c2 Float TargetRepa) =>
		PYIfThenElse c0 c1 c2 Float TargetRepa
	where
		pyIfThenElse cond =
			let
				pyIfThenElse' (pySetRepa' -> test') (pySetRepa' -> true') (pySetRepa' -> false')
					= pySet (zipWith3 ifThenElse' test' true' false')
				ifThenElse' i t e = if i then t else e
			in
				pyLift2 (pyIfThenElse' (pyLift0 cond :: PYExpr Bool TargetRepa))

type TargetHaskellCond = PYExpr Float TargetRepa -> PYExpr Float TargetRepa -> PYExpr Bool TargetRepa

instance
		(PYStorable Float TargetRepa,
		 PYStorable Bool TargetRepa,
		 PYCond2 c0 c1 Float TargetRepa) =>
		PYGT c0 c1 Float TargetRepa
	where
		pyGT = pyLiftCond2 (liftPYRepaCond2 (>) :: TargetHaskellCond)

instance
		(PYStorable Float TargetRepa,
		 PYStorable Bool TargetRepa,
		 PYCond2 c0 c1 Float TargetRepa) =>
		PYLT c0 c1 Float TargetRepa
	where
		pyLT = pyLiftCond2 (liftPYRepaCond2 (<) :: TargetHaskellCond)

instance
		(PYStorable Float TargetRepa,
		 PYStorable Bool TargetRepa,
		 PYCond2 c0 c1 Float TargetRepa) =>
		PYGTE c0 c1 Float TargetRepa
	where
		pyGTE = pyLiftCond2 (liftPYRepaCond2 (>=) :: TargetHaskellCond)

instance
		(PYStorable Float TargetRepa,
		 PYStorable Bool TargetRepa,
		 PYCond2 c0 c1 Float TargetRepa) =>
		PYLTE c0 c1 Float TargetRepa
	where
		pyLTE = pyLiftCond2 (liftPYRepaCond2 (<=) :: TargetHaskellCond)
-}
