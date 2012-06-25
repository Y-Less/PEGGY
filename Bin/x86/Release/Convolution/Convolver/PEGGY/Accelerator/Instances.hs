{-# LANGUAGE MultiParamTypeClasses       #-}
{-# LANGUAGE FlexibleInstances           #-}
{-# LANGUAGE TypeFamilies                #-}

module PEGGY.Accelerator.Instances where

import PEGGY.Types
import PEGGY.Accelerator.Targets
import PEGGY.Accelerator.Natives
import PEGGY.Accelerator.Conversion

import Foreign.ForeignPtr

import System.IO.Unsafe
import Foreign.Marshal.Array

--import Data.Vector.Storable

import qualified Data.Vector.Storable as V
--import qualified Data.Vector.Unboxed as VAlt
--import qualified Data.Vector.Generic as VBase

-- This is used in both instances to avoid extra functional dependencies.
pyDoRunAccelerator' constructor process finaliser target dayta =
		constructor result len
	where
		rep = replicate (length len)
		expr = pyExpr   dayta
		len  = pyLength dayta
		--func = process target len
		result = withArray len (\ len' -> case targetAcceleratorUnwrap expr of
			Left x  -> pyDoRunAcc' (process target len') finaliser x
			Right n ->
				if len == rep 1
				then return $ error "Unknown constant size in pyDoRunAccelerator'"
				else pyDoRunAcc' (process target len') finaliser $ pyAcceleratorConst len n)
		--do
		--	let
		--		func = process target len'
			-- Run the code on the target machine, with the length as an array.
		--	pyDoRunAcc' func finaliser (targetAcceleratorUnwrap expr))

pySetAccelerator' x len f = do
	-- "y" is the length of the data (a LIST).
	x' <- x
	-- Convert the list to a TEMPORARY C array, and call the function.  Could be
	-- very shortly rewritten using ">>=".
	withArray (len ++ [0]) (\ length' -> do
		p' <- withForeignPtr x' (f length')
		newAcceleratorPointer p')

pyGetAcceleratorDX9' (PYNativeDX9 x y) =
		unsafePerformIO (x >>= \x' -> withForeignPtr x' (peekArray $ pyRep' y))

pyGetAcceleratorDX9V' (PYNativeDX9 x y) =
		V.convert $ unsafePerformIO (x >>= \x' -> return $ V.unsafeFromForeignPtr0 x' $ pyRep' y)

instance Show (PYNative Float TargetDX9) where
	show x = show (pyGetAcceleratorDX9' x)

--instance Show (PYStorage Float TargetDX9) where
--	show = show . pyExecute

pyGetAcceleratorX64' (PYNativeX64 x y) =
		unsafePerformIO (x >>= \x' -> withForeignPtr x' (peekArray $ pyRep' y))

pyGetAcceleratorX64V' (PYNativeX64 x y) =
		V.convert $ unsafePerformIO (x >>= \x' -> return $ V.unsafeFromForeignPtr0 x' $ pyRep' y)

instance Show (PYNative Float TargetX64) where
	show x = show (pyGetAcceleratorX64' x)

--instance Show (PYStorage Float TargetX64) where
--	show = show . pyExecute

pyGetAcceleratorFPGA' (PYNativeFPGA x y) =
		unsafePerformIO (x >>= \x' -> withForeignPtr x' (peekArray $ pyRep' y))

pyGetAcceleratorFPGAV' (PYNativeFPGA x y) =
		V.convert $ unsafePerformIO (x >>= \x' -> return $ V.unsafeFromForeignPtr0 x' $ pyRep' y)

instance Show (PYNative Float TargetFPGA) where
	show x = show (pyGetAcceleratorFPGA' x)

--instance Show (PYStorage Float TargetFPGA) where
--	show = show . pyExecute

pyGetAcceleratorCUDA' (PYNativeCUDA x y) =
		unsafePerformIO (x >>= \x' -> withForeignPtr x' (peekArray $ pyRep' y))

pyGetAcceleratorCUDAV' (PYNativeCUDA x y) =
		V.convert $ unsafePerformIO (x >>= \x' -> return $ V.unsafeFromForeignPtr0 x' $ pyRep' y)

instance Show (PYNative Float TargetCUDA) where
	show x = show (pyGetAcceleratorCUDA' x)

--instance Show (PYStorage Float TargetCUDA) where
--	show = show . pyExecute

{- 
 - PYExpresible instances.
 -}

instance
		(dom0 ~ Float, id0 ~ TargetDX9) =>
		PYExpressible (PYNative dom0 id0) Float TargetDX9
	where
		pySet (PYNativeDX9 x y) = PYExpr (PYStorageDX9 $ pySetAccelerator' x y hs_AcceleratorFloatCreate) y

instance
		(dom0 ~ Double, id0 ~ TargetDX9) =>
		PYExpressible (PYNative dom0 id0) Double TargetDX9
	where
		pySet (PYNativeDX9 x y) = PYExpr (PYStorageDX9 $ pySetAccelerator' x y hs_AcceleratorDoubleCreate) y

instance
		(dom0 ~ Int, id0 ~ TargetDX9) =>
		PYExpressible (PYNative dom0 id0) Int TargetDX9
	where
		pySet (PYNativeDX9 x y) = PYExpr (PYStorageDX9 $ pySetAccelerator' x y hs_AcceleratorIntCreate) y

instance
		(dom0 ~ Float, id0 ~ TargetX64) =>
		PYExpressible (PYNative dom0 id0) Float TargetX64
	where
		pySet (PYNativeX64 x y) = PYExpr (PYStorageX64 $ pySetAccelerator' x y hs_AcceleratorFloatCreate) y

instance
		(dom0 ~ Double, id0 ~ TargetX64) =>
		PYExpressible (PYNative dom0 id0) Double TargetX64
	where
		pySet (PYNativeX64 x y) = PYExpr (PYStorageX64 $ pySetAccelerator' x y hs_AcceleratorDoubleCreate) y

instance
		(dom0 ~ Int, id0 ~ TargetX64) =>
		PYExpressible (PYNative dom0 id0) Int TargetX64
	where
		pySet (PYNativeX64 x y) = PYExpr (PYStorageX64 $ pySetAccelerator' x y hs_AcceleratorIntCreate) y

instance
		(dom0 ~ Float, id0 ~ TargetFPGA) =>
		PYExpressible (PYNative dom0 id0) Float TargetFPGA
	where
		pySet (PYNativeFPGA x y) = PYExpr (PYStorageFPGA $ pySetAccelerator' x y hs_AcceleratorFloatCreate) y

instance
		(dom0 ~ Double, id0 ~ TargetFPGA) =>
		PYExpressible (PYNative dom0 id0) Double TargetFPGA
	where
		pySet (PYNativeFPGA x y) = PYExpr (PYStorageFPGA $ pySetAccelerator' x y hs_AcceleratorDoubleCreate) y

instance
		(dom0 ~ Int, id0 ~ TargetFPGA) =>
		PYExpressible (PYNative dom0 id0) Int TargetFPGA
	where
		pySet (PYNativeFPGA x y) = PYExpr (PYStorageFPGA $ pySetAccelerator' x y hs_AcceleratorIntCreate) y

instance
		(dom0 ~ Float, id0 ~ TargetCUDA) =>
		PYExpressible (PYNative dom0 id0) Float TargetCUDA
	where
		pySet (PYNativeCUDA x y) = PYExpr (PYStorageCUDA $ pySetAccelerator' x y hs_AcceleratorFloatCreate) y

instance
		(dom0 ~ Double, id0 ~ TargetCUDA) =>
		PYExpressible (PYNative dom0 id0) Double TargetCUDA
	where
		pySet (PYNativeCUDA x y) = PYExpr (PYStorageCUDA $ pySetAccelerator' x y hs_AcceleratorDoubleCreate) y

instance
		(dom0 ~ Int, id0 ~ TargetCUDA) =>
		PYExpressible (PYNative dom0 id0) Int TargetCUDA
	where
		pySet (PYNativeCUDA x y) = PYExpr (PYStorageCUDA $ pySetAccelerator' x y hs_AcceleratorIntCreate) y

{- 
 - PYExecute instances.
 -}

instance
		PYExecute Float TargetDX9
	where
		pyExecute = pyDoRunAccelerator' PYNativeDX9 hs_AcceleratorFloatProcess hs_AcceleratorFloatFree 0
		--pyDoGet = pyGetAcceleratorDX9'

instance
		PYExecute Double TargetDX9
	where
		pyExecute = pyDoRunAccelerator' PYNativeDX9 hs_AcceleratorDoubleProcess hs_AcceleratorDoubleFree 0
		--pyDoGet = pyGetAcceleratorDX9'

instance
		PYExecute Int TargetDX9
	where
		pyExecute = pyDoRunAccelerator' PYNativeDX9 hs_AcceleratorIntProcess hs_AcceleratorIntFree 0
		--pyDoGet = pyGetAcceleratorDX9'

instance
		PYExecute Float TargetX64
	where
		pyExecute = pyDoRunAccelerator' PYNativeX64 hs_AcceleratorFloatProcess hs_AcceleratorFloatFree 1
		--pyDoGet = pyGetAcceleratorX64'

instance
		PYExecute Double TargetX64
	where
		pyExecute = pyDoRunAccelerator' PYNativeX64 hs_AcceleratorDoubleProcess hs_AcceleratorDoubleFree 1
		--pyDoGet = pyGetAcceleratorX64'

instance
		PYExecute Int TargetX64
	where
		pyExecute = pyDoRunAccelerator' PYNativeX64 hs_AcceleratorIntProcess hs_AcceleratorIntFree 1
		--pyDoGet = pyGetAcceleratorX64'

instance
		PYExecute Float TargetFPGA
	where
		pyExecute = pyDoRunAccelerator' PYNativeFPGA hs_AcceleratorFloatProcess hs_AcceleratorFloatFree 2
		--pyDoGet = pyGetAcceleratorFPGA'

instance
		PYExecute Double TargetFPGA
	where
		pyExecute = pyDoRunAccelerator' PYNativeFPGA hs_AcceleratorDoubleProcess hs_AcceleratorDoubleFree 2
		--pyDoGet = pyGetAcceleratorFPGA'

instance
		PYExecute Int TargetFPGA
	where
		pyExecute = pyDoRunAccelerator' PYNativeFPGA hs_AcceleratorIntProcess hs_AcceleratorIntFree 2
		--pyDoGet = pyGetAcceleratorFPGA'

instance
		PYExecute Float TargetCUDA
	where
		pyExecute = pyDoRunAccelerator' PYNativeCUDA hs_AcceleratorFloatProcess hs_AcceleratorFloatFree 3
		--pyDoGet = pyGetAcceleratorCUDA'

instance
		PYExecute Double TargetCUDA
	where
		pyExecute = pyDoRunAccelerator' PYNativeCUDA hs_AcceleratorDoubleProcess hs_AcceleratorDoubleFree 3
		--pyDoGet = pyGetAcceleratorCUDA'

instance
		PYExecute Int TargetCUDA
	where
		pyExecute = pyDoRunAccelerator' PYNativeCUDA hs_AcceleratorIntProcess hs_AcceleratorIntFree 3
		--pyDoGet = pyGetAcceleratorCUDA'

instance
		PYReturn Float TargetDX9
	where
		pyReturnDims (PYNativeDX9 _ n) = n
		pyReturnList = pyGetAcceleratorDX9'
		pyReturnVector = pyGetAcceleratorDX9V'

instance
		PYReturn Double TargetDX9
	where
		pyReturnDims (PYNativeDX9 _ n) = n
		pyReturnList = pyGetAcceleratorDX9'
		pyReturnVector = pyGetAcceleratorDX9V'

instance
		PYReturn Int TargetDX9
	where
		pyReturnDims (PYNativeDX9 _ n) = n
		pyReturnList = pyGetAcceleratorDX9'
		pyReturnVector = pyGetAcceleratorDX9V'

instance
		PYReturn Float TargetX64
	where
		pyReturnDims (PYNativeX64 _ n) = n
		pyReturnList = pyGetAcceleratorX64'
		pyReturnVector = pyGetAcceleratorX64V'

instance
		PYReturn Double TargetX64
	where
		pyReturnDims (PYNativeX64 _ n) = n
		pyReturnList = pyGetAcceleratorX64'
		pyReturnVector = pyGetAcceleratorX64V'

instance
		PYReturn Int TargetX64
	where
		pyReturnDims (PYNativeX64 _ n) = n
		pyReturnList = pyGetAcceleratorX64'
		pyReturnVector = pyGetAcceleratorX64V'

instance
		PYReturn Float TargetFPGA
	where
		pyReturnDims (PYNativeFPGA _ n) = n
		pyReturnList = pyGetAcceleratorFPGA'
		pyReturnVector = pyGetAcceleratorFPGAV'

instance
		PYReturn Double TargetFPGA
	where
		pyReturnDims (PYNativeFPGA _ n) = n
		pyReturnList = pyGetAcceleratorFPGA'
		pyReturnVector = pyGetAcceleratorFPGAV'

instance
		PYReturn Int TargetFPGA
	where
		pyReturnDims (PYNativeFPGA _ n) = n
		pyReturnList = pyGetAcceleratorFPGA'
		pyReturnVector = pyGetAcceleratorFPGAV'

instance
		PYReturn Float TargetCUDA
	where
		pyReturnDims (PYNativeCUDA _ n) = n
		pyReturnList = pyGetAcceleratorCUDA'
		pyReturnVector = pyGetAcceleratorCUDAV'

instance
		PYReturn Double TargetCUDA
	where
		pyReturnDims (PYNativeCUDA _ n) = n
		pyReturnList = pyGetAcceleratorCUDA'
		pyReturnVector = pyGetAcceleratorCUDAV'

instance
		PYReturn Int TargetCUDA
	where
		pyReturnDims (PYNativeCUDA _ n) = n
		pyReturnList = pyGetAcceleratorCUDA'
		pyReturnVector = pyGetAcceleratorCUDAV'
