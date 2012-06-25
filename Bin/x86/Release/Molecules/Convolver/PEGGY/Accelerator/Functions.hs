{-# LANGUAGE MultiParamTypeClasses       #-}
{-# LANGUAGE FlexibleInstances           #-}
{-# LANGUAGE FlexibleContexts            #-}



{-# LANGUAGE TypeFamilies            #-}
{-# LANGUAGE UndecidableInstances    #-}

module PEGGY.Accelerator.Functions where

import PEGGY.Types
import PEGGY.Functions
import PEGGY.Accelerator.Natives
import PEGGY.Accelerator.Conversion
import PEGGY.Accelerator.Targets

import Foreign.ForeignPtr
import qualified Foreign.ForeignPtr.Unsafe as Unsafe

-- DX9
instance
		(PY2 c0 c1 Float TargetDX9) =>
		PYAdd c0 c1 Float TargetDX9
	where
		pyAdd = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAadd)

instance
		(PY2 c0 c1 Float TargetDX9) =>
		PYSub c0 c1 Float TargetDX9
	where
		pySub = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAsubtract)

instance
		(PY2 c0 c1 Float TargetDX9) =>
		PYMul c0 c1 Float TargetDX9
	where
		pyMul = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAmultiply)

instance
		(PY2 c0 c1 Float TargetDX9) =>
		PYDiv c0 c1 Float TargetDX9
	where
		pyDiv = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAdivide)

instance
		(PY2 c0 c1 Float TargetDX9) =>
		PYMin c0 c1 Float TargetDX9
	where
		pyMin = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAmin)

instance
		(PY2 c0 c1 Float TargetDX9) =>
		PYMax c0 c1 Float TargetDX9
	where
		pyMax = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAmax)

instance
		(PY1 c0 Float TargetDX9) =>
		PYAbs c0 Float TargetDX9
	where
		pyAbs = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAabs)

instance
		(PY1 c0 Float TargetDX9) =>
		PYMaxVal c0 Float TargetDX9
	where
		pyMaxVal = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAmaxVal)

instance
		(PY1 c0 Float TargetDX9) =>
		PYMinVal c0 Float TargetDX9
	where
		pyMinVal = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAminVal)

instance
		(PY1 c0 Float TargetDX9) =>
		PYNegate c0 Float TargetDX9
	where
		pyNegate = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAnegate)

instance
		(PY1 c0 Float TargetDX9) =>
		PYProduct c0 Float TargetDX9
	where
		pyProduct = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAproduct)

instance
		(PY1 c0 Float TargetDX9) =>
		PYSum c0 Float TargetDX9
	where
		pySum = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAsum)

instance
		(PY1 c0 Float TargetDX9) =>
		PYShiftExtend c0 Float TargetDX9
	where
		pyShiftExtend s = pyLift1 (liftPYAcc1 $ hs_AcceleratorfloatIAshift (head s))

instance
		(PY1 c0 Float TargetDX9) =>
		PYSqrt c0 Float TargetDX9
	where
		pySqrt = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAsqrt)

instance
		(PY1 c0 Float TargetDX9) =>
		PYPow c0 Float TargetDX9
	where
		pyPow b = pyLift1 (liftPYAcc1 $ hs_AcceleratorfloatIApow b)

instance
		(PY1 c0 Float TargetDX9) =>
		PYLog c0 Float TargetDX9
	where
		pyLog b = pyLift1 (liftPYAcc1 $ hs_AcceleratorfloatIAlog b)

instance
		(PYCond2 c0 c1 Float TargetDX9) =>
		PYGT c0 c1 Float TargetDX9
	where
		pyGT = pyLiftCond2 (liftPYAccCond2 hs_AcceleratorfloatAAgt)

instance
		(PYCond2 c0 c1 Float TargetDX9) =>
		PYLT c0 c1 Float TargetDX9
	where
		pyLT = pyLiftCond2 (liftPYAccCond2 hs_AcceleratorfloatAAlt)

instance
		(PYCond2 c0 c1 Float TargetDX9) =>
		PYGTE c0 c1 Float TargetDX9
	where
		pyGTE = pyLiftCond2 (liftPYAccCond2 hs_AcceleratorfloatAAgeq)

instance
		(PYCond2 c0 c1 Float TargetDX9) =>
		PYLTE c0 c1 Float TargetDX9
	where
		pyLTE = pyLiftCond2 (liftPYAccCond2 hs_AcceleratorfloatAAleq)

instance
		(PYCond2 c0 c1 Float TargetDX9) =>
		PYEQ c0 c1 Float TargetDX9
	where
		pyEQ = pyLiftCond2 (liftPYAccCond2 hs_AcceleratorfloatAAeq)

instance
		(PYCond2 c0 c1 Float TargetDX9) =>
		PYNEQ c0 c1 Float TargetDX9
	where
		pyNEQ = pyLiftCond2 (liftPYAccCond2 hs_AcceleratorfloatAAneq)

instance
		(PY0 c0 Bool TargetDX9,
		 PY2 c1 c2 Float TargetDX9) =>
		PYIfThenElse c0 c1 c2 Float TargetDX9
	where
		-- This is the first function I've written where I've been happy that
		-- the fact that it compiles (and thus has the right types) means that
		-- it is probably correct.
		-- Unit tests save the day!  I found a bug in this function - if the
		-- size of the condition is known, but the size of the options are not,
		-- you will get an error even though it should be able to determine the
		-- size.  This is my fault for trying to be clever and use "pyLift2£ on
		-- a 3 parameter function.
		pyIfThenElse cond =
			let
				(PYStorageDX9 iostor) = pyExpr expr'
				thenElse' thn els = do
					stor <- iostor
					let
						stor' = Unsafe.unsafeForeignPtrToPtr stor
					result' <- hs_AcceleratorfloatAAAcond stor' thn els
					-- So Haskell doesn't clean up the data too early.
					touchForeignPtr stor
					return result'
				-- Shame we need explicit types here.
				expr' = pyLift0 cond :: PYExpr Bool TargetDX9
				len' = pyLength expr'
				-- Use the length of the condition where required.
				relen' dayta' = if pyLength dayta' == [] then PYExpr (pyExpr dayta') len' else dayta'
				-- This function takes two expressions and returns them applied
				-- to another function, with their lengths reified.
				liftedFunc = \ x y -> (liftPYAcc2 thenElse') (relen' x) (relen' y)
			in
				pyLift2 liftedFunc

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--   X64
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
instance
		(PY2 c0 c1 Float TargetX64) =>
		PYAdd c0 c1 Float TargetX64
	where
		pyAdd = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAadd)

instance
		(PY2 c0 c1 Float TargetX64) =>
		PYSub c0 c1 Float TargetX64
	where
		pySub = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAsubtract)

instance
		(PY2 c0 c1 Float TargetX64) =>
		PYMul c0 c1 Float TargetX64
	where
		pyMul = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAmultiply)

instance
		(PY2 c0 c1 Float TargetX64) =>
		PYDiv c0 c1 Float TargetX64
	where
		pyDiv = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAdivide)

instance
		(PY2 c0 c1 Float TargetX64) =>
		PYMin c0 c1 Float TargetX64
	where
		pyMin = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAmin)

instance
		(PY2 c0 c1 Float TargetX64) =>
		PYMax c0 c1 Float TargetX64
	where
		pyMax = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAmax)

instance
		(PY1 c0 Float TargetX64) =>
		PYAbs c0 Float TargetX64
	where
		pyAbs = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAabs)

instance
		(PY1 c0 Float TargetX64) =>
		PYMaxVal c0 Float TargetX64
	where
		pyMaxVal = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAmaxVal)

instance
		(PY1 c0 Float TargetX64) =>
		PYMinVal c0 Float TargetX64
	where
		pyMinVal = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAminVal)

instance
		(PY1 c0 Float TargetX64) =>
		PYNegate c0 Float TargetX64
	where
		pyNegate = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAnegate)

instance
		(PY1 c0 Float TargetX64) =>
		PYProduct c0 Float TargetX64
	where
		pyProduct = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAproduct)

instance
		(PY1 c0 Float TargetX64) =>
		PYSum c0 Float TargetX64
	where
		pySum = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAsum)

instance
		(PY1 c0 Float TargetX64) =>
		PYShiftExtend c0 Float TargetX64
	where
		pyShiftExtend s = pyLift1 (liftPYAcc1 $ hs_AcceleratorfloatIAshift (head s))

instance
		(PY1 c0 Float TargetX64) =>
		PYSqrt c0 Float TargetX64
	where
		pySqrt = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAsqrt)

--instance
--		(PY1 c0 Float TargetX64) =>
--		PYShiftRotate c0 Float TargetX64
--	where
--		pyShiftRotate s = pyLift1 (liftPYAcc1 $ hs_AcceleratorfloatIArotate (head s))

-- FPGA
instance
		(PY2 c0 c1 Float TargetFPGA) =>
		PYAdd c0 c1 Float TargetFPGA
	where
		pyAdd = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAadd)

instance
		(PY2 c0 c1 Float TargetFPGA) =>
		PYSub c0 c1 Float TargetFPGA
	where
		pySub = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAsubtract)

instance
		(PY2 c0 c1 Float TargetFPGA) =>
		PYMul c0 c1 Float TargetFPGA
	where
		pyMul = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAmultiply)

instance
		(PY2 c0 c1 Float TargetFPGA) =>
		PYDiv c0 c1 Float TargetFPGA
	where
		pyDiv = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAdivide)

instance
		(PY2 c0 c1 Float TargetFPGA) =>
		PYMin c0 c1 Float TargetFPGA
	where
		pyMin = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAmin)

instance
		(PY2 c0 c1 Float TargetFPGA) =>
		PYMax c0 c1 Float TargetFPGA
	where
		pyMax = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAmax)

instance
		(PY1 c0 Float TargetFPGA) =>
		PYAbs c0 Float TargetFPGA
	where
		pyAbs = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAabs)

instance
		(PY1 c0 Float TargetFPGA) =>
		PYMaxVal c0 Float TargetFPGA
	where
		pyMaxVal = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAmaxVal)

instance
		(PY1 c0 Float TargetFPGA) =>
		PYMinVal c0 Float TargetFPGA
	where
		pyMinVal = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAminVal)

instance
		(PY1 c0 Float TargetFPGA) =>
		PYNegate c0 Float TargetFPGA
	where
		pyNegate = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAnegate)

instance
		(PY1 c0 Float TargetFPGA) =>
		PYProduct c0 Float TargetFPGA
	where
		pyProduct = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAproduct)

instance
		(PY1 c0 Float TargetFPGA) =>
		PYSum c0 Float TargetFPGA
	where
		pySum = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAsum)

instance
		(PY1 c0 Float TargetFPGA) =>
		PYShiftExtend c0 Float TargetFPGA
	where
		pyShiftExtend s = pyLift1 (liftPYAcc1 $ hs_AcceleratorfloatIAshift (head s))

instance
		(PY1 c0 Float TargetFPGA) =>
		PYSqrt c0 Float TargetFPGA
	where
		pySqrt = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAsqrt)

--instance
--		(PY1 c0 Float TargetFPGA) =>
--		PYShiftRotate c0 Float TargetFPGA
--	where
--		pyShiftRotate s = pyLift1 (liftPYAcc1 $ hs_AcceleratorfloatIArotate (head s))

-- CUDA
instance
		(PY2 c0 c1 Float TargetCUDA) =>
		PYAdd c0 c1 Float TargetCUDA
	where
		pyAdd = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAadd)

instance
		(PY2 c0 c1 Float TargetCUDA) =>
		PYSub c0 c1 Float TargetCUDA
	where
		pySub = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAsubtract)

instance
		(PY2 c0 c1 Float TargetCUDA) =>
		PYMul c0 c1 Float TargetCUDA
	where
		pyMul = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAmultiply)

instance
		(PY2 c0 c1 Float TargetCUDA) =>
		PYDiv c0 c1 Float TargetCUDA
	where
		pyDiv = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAdivide)

instance
		(PY2 c0 c1 Float TargetCUDA) =>
		PYMin c0 c1 Float TargetCUDA
	where
		pyMin = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAmin)

instance
		(PY2 c0 c1 Float TargetCUDA) =>
		PYMax c0 c1 Float TargetCUDA
	where
		pyMax = pyLift2 (liftPYAcc2 hs_AcceleratorfloatAAmax)

instance
		(PY1 c0 Float TargetCUDA) =>
		PYAbs c0 Float TargetCUDA
	where
		pyAbs = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAabs)

instance
		(PY1 c0 Float TargetCUDA) =>
		PYMaxVal c0 Float TargetCUDA
	where
		pyMaxVal = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAmaxVal)

instance
		(PY1 c0 Float TargetCUDA) =>
		PYMinVal c0 Float TargetCUDA
	where
		pyMinVal = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAminVal)

instance
		(PY1 c0 Float TargetCUDA) =>
		PYNegate c0 Float TargetCUDA
	where
		pyNegate = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAnegate)

instance
		(PY1 c0 Float TargetCUDA) =>
		PYProduct c0 Float TargetCUDA
	where
		pyProduct = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAproduct)

instance
		(PY1 c0 Float TargetCUDA) =>
		PYSum c0 Float TargetCUDA
	where
		pySum = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAsum)

instance
		(PY1 c0 Float TargetCUDA) =>
		PYSqrt c0 Float TargetCUDA
	where
		pySqrt = pyLift1 (liftPYAcc1 hs_AcceleratorfloatAsqrt)

instance
		(PY1 c0 Float TargetCUDA) =>
		PYShiftExtend c0 Float TargetCUDA
	where
		pyShiftExtend s = pyLift1 (liftPYAcc1 $ hs_AcceleratorfloatIAshift (head s))

--instance
--		(PY1 c0 Float TargetCUDA) =>
--		PYShiftRotate c0 Float TargetCUDA
--	where
--		pyShiftRotate s = pyLift1 (liftPYAcc1 $ hs_AcceleratorfloatIArotate (head s))
