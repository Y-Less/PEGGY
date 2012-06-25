{-# LANGUAGE MultiParamTypeClasses       #-}
{-# LANGUAGE UndecidableInstances        #-}
{-# LANGUAGE FlexibleInstances           #-}
{-# LANGUAGE FlexibleContexts            #-}
{-# LANGUAGE TypeFamilies                #-}

module PEGGY.Functions where

import PEGGY.Types

-- Classes for lifting operations on "PYExpr" to operations on anything that can
-- be converted to PYExpr.
class
		PY0 c0 dom id
	where
		pyLift0 :: c0 -> PYExpr dom id

class
		PY1 c0 dom id
	where
		pyLift1 :: (PYExpr dom id -> PYExpr dom id) -> c0 -> PYExpr dom id

class
		PY2 c0 c1 dom id
	where
		pyLift2 :: (PYExpr dom id -> PYExpr dom id -> PYExpr dom id) -> c0 -> c1 -> PYExpr dom id

class
		PY3 c0 c1 c2 dom id
	where
		pyLift3 :: (PYExpr dom id -> PYExpr dom id -> PYExpr dom id -> PYExpr dom id) -> c0 -> c1 -> c2 -> PYExpr dom id

class
		PYCond1 c0 dom id
	where
		pyLiftCond1 :: (PYExpr dom id -> PYExpr Bool id) -> c0 -> PYExpr Bool id

class
		PYCond2 c0 c1 dom id
	where
		pyLiftCond2 :: (PYExpr dom id -> PYExpr dom id -> PYExpr Bool id) -> c0 -> c1 -> PYExpr Bool id

-- Classes for doing operations.  Can be implemented on a per-target level.
class PYAdd c0 c1 dom id where
	pyAdd :: (PY2 c0 c1 dom id) => c0 -> c1 -> PYExpr dom id

class PYSub c0 c1 dom id where
	pySub :: (PY2 c0 c1 dom id) => c0 -> c1 -> PYExpr dom id

class PYMul c0 c1 dom id where
	pyMul :: (PY2 c0 c1 dom id) => c0 -> c1 -> PYExpr dom id

class PYDiv c0 c1 dom id where
	pyDiv :: (PY2 c0 c1 dom id) => c0 -> c1 -> PYExpr dom id

class PYMin c0 c1 dom id where
	pyMin :: (PY2 c0 c1 dom id) => c0 -> c1 -> PYExpr dom id

class PYMax c0 c1 dom id where
	pyMax :: (PY2 c0 c1 dom id) => c0 -> c1 -> PYExpr dom id

class PYAbs c0 dom id where
	pyAbs :: (PY1 c0 dom id) => c0 -> PYExpr dom id

class PYMaxVal c0 dom id where
	pyMaxVal :: (PY1 c0 dom id) => c0 -> PYExpr dom id

class PYMinVal c0 dom id where
	pyMinVal :: (PY1 c0 dom id) => c0 -> PYExpr dom id

class PYNegate c0 dom id where
	pyNegate :: (PY1 c0 dom id) => c0 -> PYExpr dom id

class PYProduct c0 dom id where
	pyProduct :: (PY1 c0 dom id) => c0 -> PYExpr dom id

class PYSum c0 dom id where
	pySum :: (PY1 c0 dom id) => c0 -> PYExpr dom id

class PYSqrt c0 dom id where
	pySqrt :: (PY1 c0 dom id) => c0 -> PYExpr dom id

class PYShiftExtend c0 dom id where
	pyShiftExtend :: (PY1 c0 dom id) => [] Int -> c0 -> PYExpr dom id

class PYShiftRotate c0 dom id where
	pyShiftRotate :: (PY1 c0 dom id) => [] Int -> c0 -> PYExpr dom id

class PYShiftConst c0 dom id where
	pyShiftConst :: (PY1 c0 dom id) => dom -> [] Int -> c0 -> PYExpr dom id

class PYIfThenElse c0 c1 c2 dom id where
	pyIfThenElse :: (PY1 c0 Bool id, PY2 c1 c2 dom id) => c0 -> c1 -> c2 -> PYExpr dom id

(.+.) :: (PYAdd c0 c1 dom id, PY2 c0 c1 dom id) => c0 -> c1 -> PYExpr dom id
(.+.) = pyAdd

(.-.) :: (PYSub c0 c1 dom id, PY2 c0 c1 dom id) => c0 -> c1 -> PYExpr dom id
(.-.) = pySub

(.*.) :: (PYMul c0 c1 dom id, PY2 c0 c1 dom id) => c0 -> c1 -> PYExpr dom id
(.*.) = pyMul

(./.) :: (PYDiv c0 c1 dom id, PY2 c0 c1 dom id) => c0 -> c1 -> PYExpr dom id
(./.) = pyDiv

(.<.) :: (PYMin c0 c1 dom id, PY2 c0 c1 dom id) => c0 -> c1 -> PYExpr dom id
(.<.) = pyMin

(.>.) :: (PYMax c0 c1 dom id, PY2 c0 c1 dom id) => c0 -> c1 -> PYExpr dom id
(.>.) = pyMax

(.<<<.) :: (PYShiftExtend c0 dom id, PY1 c0 dom id) => c0 -> [] Int -> PYExpr dom id
(.<<<.) a b = pyShiftExtend b a

class PYNEQ c0 c1 dom id where
	pyNEQ :: (PYCond2 c0 c1 dom id) => c0 -> c1 -> PYExpr Bool id

class PYEQ c0 c1 dom id where
	pyEQ :: (PYCond2 c0 c1 dom id) => c0 -> c1 -> PYExpr Bool id

class PYGT c0 c1 dom id where
	pyGT :: (PYCond2 c0 c1 dom id) => c0 -> c1 -> PYExpr Bool id

class PYLT c0 c1 dom id where
	pyLT :: (PYCond2 c0 c1 dom id) => c0 -> c1 -> PYExpr Bool id

class PYGTE c0 c1 dom id where
	pyGTE :: (PYCond2 c0 c1 dom id) => c0 -> c1 -> PYExpr Bool id

class PYLTE c0 c1 dom id where
	pyLTE :: (PYCond2 c0 c1 dom id) => c0 -> c1 -> PYExpr Bool id

class PYPow c0 dom id where
	pyPow :: dom -> c0 -> PYExpr dom id

class PYLog c0 dom id where
	pyLog :: dom -> c0 -> PYExpr dom id

data ConvolutionMovement dom = ShiftLeft | ShiftRight | ShiftRelative | ShiftX Int Int | RotateLeft | RotateRight | RotateRelative | RotateX Int Int | ConstLeft dom | ConstRight dom | ConstRelative dom | ConstX Int Int dom

data ConvolutionFilter dom = Filter ([] Int) ([] dom)

pyMakeFilter1D a = Filter [length a] a
pyMakeFilter2D a = Filter [length a, length (head a)] (concat a)

class PYConvolve c0 dom id where
	pyConvolve :: (PY1 c0 dom id) => ConvolutionMovement dom -> ConvolutionFilter dom -> c0 -> PYExpr dom id

instance
		Eq (PYExpr dom id)
	where
		(==) a b = pyLength a == pyLength b
		(/=) a b = pyLength a /= pyLength b

{-instance
		Show (PYStorage dom id)
	where
		show _ = "Data Specific"-}

instance
		(Num dom,
		 PYStorable dom id,
		 Show (PYExpr dom id),
		 PY1 (PYExpr dom id) dom id,
		 PY2 (PYExpr dom id) (PYExpr dom id) dom id,
		 PYAdd (PYExpr dom id) (PYExpr dom id) dom id,
		 PYMul (PYExpr dom id) (PYExpr dom id) dom id,
		 PYSub (PYExpr dom id) (PYExpr dom id) dom id,
		 PYAbs (PYExpr dom id) dom id,
		 PYNegate (PYExpr dom id) dom id) =>
		 --
		Num (PYExpr dom id)
	where
		(+) = pyAdd
		(*) = pyMul
		(-) = pySub
		negate = pyNegate
		abs = pyAbs
		signum = undefined
		-- I *think* this is the reason why I can do "expr + const",,,
		fromInteger v = PYExpr (pySetNum [] (fromInteger v)) []

instance
		(PYStorable dom id,
		 PY2 (PYExpr dom id) (PYExpr dom id) dom id,
		 Num (PYExpr dom id),
		 Num dom,
		 Fractional dom,
		 PYDiv (PYExpr dom id) (PYExpr dom id) dom id) =>
		 --
		Fractional (PYExpr dom id)
	where
		(/) a b = pyDiv a b
		--recip a = pyDiv ((fromRational 1) :: PYExpr dom id) a
		fromRational v = PYExpr (pySetNum [] (fromRational v)) []

instance
		(PYStorable dom id,
		 Floating dom,
		 Fractional dom,
		 Fractional (PYExpr dom id),
		 PY1 (PYExpr dom id) dom id,
		 PYSqrt (PYExpr dom id) dom id) =>
		 --
		Floating (PYExpr dom id)
	where
		pi = PYExpr (pySetNum [] (pi)) []
		exp = undefined
		sqrt = pySqrt
		log = undefined
		sin = undefined
		cos = undefined
		asin = undefined
		atan = undefined
		acos = undefined
		sinh = undefined
		cosh = undefined
		asinh = undefined
		atanh = undefined
		acosh = undefined
