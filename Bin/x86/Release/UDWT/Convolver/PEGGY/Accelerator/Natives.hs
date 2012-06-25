{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}

module PEGGY.Accelerator.Natives where

import Foreign.C.Types

import Foreign.ForeignPtr
import Foreign.Ptr
import Foreign.Storable

import Foreign.Marshal.Alloc
import Foreign.Marshal.Array

import PEGGY.Types

import qualified Data.Vector.Storable as V
import qualified Data.Vector.Unboxed as VAlt
import qualified Data.Vector.Generic as VBase

data PY_AcceleratorArray a = PY_AcceleratorArray

-- Pointer returned by Haskell memory allocation functions.
type PY_CPtr a = Ptr  a
type PY_IOCPtr a = IO (PY_CPtr a)

-- Pointer returned by Haskell memory allocation functions (with finaliser).
type PY_HPtr a = ForeignPtr  a
type PY_IOHPtr a = IO (PY_HPtr a)

-- "Raw" pointer (one with no attached finaliser).
type PY_RPtr a = Ptr (PY_AcceleratorArray a)
type PY_IORPtr a = IO (PY_RPtr a)

-- Accelerator pointer (with finaliser).
type PY_APtr a = ForeignPtr (PY_AcceleratorArray a)
type PY_IOAPtr a = IO (PY_APtr a)

newAcceleratorPointer :: PY_RPtr a -> PY_IOAPtr a
newAcceleratorPointer = newForeignPtr hs_AcceleratorDestroy

{-
 - Allow lists of certain types to be converted to C arrays (with pointers).
 -}
--class (Storable a) => PY_AConvertible a where
--	ccToArray   :: [a] -> PY_IOAPtr a

listToAcceleratorArray' ::
	(Storable a) =>
	(PY_CPtr Int -> PY_CPtr a -> PY_IORPtr a) -> [a] -> [Int] -> PY_IOAPtr a
listToAcceleratorArray' func array len = do
	--let
	--	length' = fromIntegral . length $ array
	withArray (len ++ [0]) (\ length' -> do
		data' <- withArray array (func length')
		newAcceleratorPointer data')
		--p' <- withForeignPtr x' (f length')
		--newAcceleratorPointer p')

vectorToAcceleratorArray' ::
	(Storable a, VBase.Vector VAlt.Vector a) =>
	(PY_CPtr Int -> PY_CPtr a -> PY_IORPtr a) -> VAlt.Vector a -> [Int] -> PY_IOAPtr a
vectorToAcceleratorArray' func array len = do
	let
		(x', _) = V.unsafeToForeignPtr0 $ V.convert array
	withArray (len ++ [0]) (\ length' -> do
		--withArray y
		data' <- withForeignPtr x' (func length')
		newAcceleratorPointer data')

{-vectorToAcceleratorArray2D' ::
	(Storable a, VBase.Vector VAlt.Vector a, VAlt.Unbox (VAlt.Vector a), VAlt.Unbox a) =>
	(PY_CPtr Int -> PY_CPtr a -> PY_IORPtr a) -> VAlt.Vector (VAlt.Vector a) -> PY_IOAPtr a
vectorToAcceleratorArray2D' func array = do
	-- Concatenate the array to a single block of memory.  I have no idea how
	-- efficient this is.  May be better to do this in C...
	let
		(x', y) = V.unsafeToForeignPtr0 $ V.convert (VAlt.foldl1 (VAlt.++) array)
	p' <- withForeignPtr x' (func y)
	newAcceleratorPointer p'-}

{-instance PY_AConvertible Float  where
	ccToArray   = ccToArray'   hs_AcceleratorFloatCreate

instance PY_AConvertible Double where
	ccToArray   = ccToArray'   hs_AcceleratorDoubleCreate

instance PY_AConvertible Int    where
	ccToArray   = ccToArray'   hs_AcceleratorIntCreate-}

-- Finaliser function for freeing any structures we no longer need.  Accelerator
-- has an additional garbage collector to free up any expression tress no longer
-- needed through destructors, so we need to correctly invoke "delete".
foreign import ccall unsafe "include.h &HS_AcceleratorDestroy"
	hs_AcceleratorDestroy         :: FinalizerPtr a

-- Free memory allocated by Accelerator to store the calculation result.
foreign import ccall unsafe "include.h &HS_AcceleratorFloatFree"
	hs_AcceleratorFloatFree       :: FinalizerPtr Float

-- Free memory allocated by Accelerator to store the calculation result.
foreign import ccall unsafe "include.h &HS_AcceleratorDoubleFree"
	hs_AcceleratorDoubleFree      :: FinalizerPtr Double

-- Free memory allocated by Accelerator to store the calculation result.
foreign import ccall unsafe "include.h &HS_AcceleratorIntFree"
	hs_AcceleratorIntFree         :: FinalizerPtr Int

-- Pass the size of the data.
foreign import ccall unsafe "include.h HS_AcceleratorFloatCreate"
	hs_AcceleratorFloatCreate     :: PY_CPtr Int -> PY_CPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratorDoubleCreate"
	hs_AcceleratorDoubleCreate    :: PY_CPtr Int -> PY_CPtr Double -> PY_IORPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorIntCreate"
	hs_AcceleratorIntCreate       :: PY_CPtr Int -> PY_CPtr Int    -> PY_IORPtr Int

foreign import ccall unsafe "include.h HS_AcceleratorFloatCreateC"
	hs_AcceleratorFloatCreateC    :: PY_CPtr Int -> Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratorDoubleCreateC"
	hs_AcceleratorDoubleCreateC   :: PY_CPtr Int -> Double -> PY_IORPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorIntCreateC"
	hs_AcceleratorIntCreateC      :: PY_CPtr Int -> Int    -> PY_IORPtr Int

-- Pass the processor type as an integer, as well as the size.
foreign import ccall unsafe "include.h HS_AcceleratorFloatProcess"
	hs_AcceleratorFloatProcess    :: Int -> PY_CPtr Int -> PY_RPtr Float  -> PY_IOCPtr Float

foreign import ccall unsafe "include.h HS_AcceleratorDoubleProcess"
	hs_AcceleratorDoubleProcess   :: Int -> PY_CPtr Int -> PY_RPtr Double -> PY_IOCPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorIntProcess"
	hs_AcceleratorIntProcess      :: Int -> PY_CPtr Int -> PY_RPtr Int    -> PY_IOCPtr Int

-- Basic operations.
foreign import ccall unsafe "include.h HS_AcceleratorfloatAAmultiply"
	hs_AcceleratorfloatAAmultiply   :: PY_RPtr Float  -> PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleAAmultiply"
	hs_AcceleratordoubleAAmultiply  :: PY_RPtr Double -> PY_RPtr Double -> PY_IORPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorintAAmultiply"
	hs_AcceleratorintAAmultiply     :: PY_RPtr Int    -> PY_RPtr Int    -> PY_IORPtr Int

foreign import ccall unsafe "include.h HS_AcceleratorfloatAAdivide"
	hs_AcceleratorfloatAAdivide     :: PY_RPtr Float  -> PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleAAdivide"
	hs_AcceleratordoubleAAdivide    :: PY_RPtr Double -> PY_RPtr Double -> PY_IORPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorintAAdivide"
	hs_AcceleratorintAAdivide       :: PY_RPtr Int    -> PY_RPtr Int    -> PY_IORPtr Int

foreign import ccall unsafe "include.h HS_AcceleratorfloatAAmax"
	hs_AcceleratorfloatAAmax        :: PY_RPtr Float  -> PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleAAmax"
	hs_AcceleratordoubleAAmax       :: PY_RPtr Double -> PY_RPtr Double -> PY_IORPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorintAAmax"
	hs_AcceleratorintAAmax          :: PY_RPtr Int    -> PY_RPtr Int    -> PY_IORPtr Int

foreign import ccall unsafe "include.h HS_AcceleratorfloatAAmin"
	hs_AcceleratorfloatAAmin        :: PY_RPtr Float  -> PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleAAmin"
	hs_AcceleratordoubleAAmin       :: PY_RPtr Double -> PY_RPtr Double -> PY_IORPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorintAAmin"
	hs_AcceleratorintAAmin          :: PY_RPtr Int    -> PY_RPtr Int    -> PY_IORPtr Int

foreign import ccall unsafe "include.h HS_AcceleratorfloatAAadd"
	hs_AcceleratorfloatAAadd        :: PY_RPtr Float  -> PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleAAadd"
	hs_AcceleratordoubleAAadd       :: PY_RPtr Double -> PY_RPtr Double -> PY_IORPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorintAAadd"
	hs_AcceleratorintAAadd          :: PY_RPtr Int    -> PY_RPtr Int    -> PY_IORPtr Int

foreign import ccall unsafe "include.h HS_AcceleratorfloatAAsubtract"
	hs_AcceleratorfloatAAsubtract   :: PY_RPtr Float  -> PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleAAsubtract"
	hs_AcceleratordoubleAAsubtract  :: PY_RPtr Double -> PY_RPtr Double -> PY_IORPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorintAAsubtract"
	hs_AcceleratorintAAsubtract     :: PY_RPtr Int    -> PY_RPtr Int    -> PY_IORPtr Int

-- More advanced operations.
{-foreign import ccall "include.h HS_AcceleratorFloatInnerProduct"
	hs_AcceleratorfloatAAXnnerProduct  :: PY_RPtr Float  -> PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall "include.h HS_AcceleratorDoubleInnerProduct"
	hs_AcceleratordoubleAAXnnerProduct :: PY_RPtr Double -> PY_RPtr Double -> PY_IORPtr Double

foreign import ccall "include.h HS_AcceleratorIntInnerProduct"
	hs_AcceleratorintAAXnnerProduct    :: PY_RPtr Int    -> PY_RPtr Int    -> PY_IORPtr Int-}

-- Single element operations.
foreign import ccall unsafe "include.h HS_AcceleratorfloatAabs"
	hs_AcceleratorfloatAabs  :: PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleAabs"
	hs_AcceleratordoubleAabs :: PY_RPtr Double -> PY_IORPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorintAabs"
	hs_AcceleratorintAabs    :: PY_RPtr Int    -> PY_IORPtr Int

foreign import ccall unsafe "include.h HS_AcceleratorfloatAmaxVal"
	hs_AcceleratorfloatAmaxVal  :: PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleAmaxVal"
	hs_AcceleratordoubleAmaxVal :: PY_RPtr Double -> PY_IORPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorintAmaxVal"
	hs_AcceleratorintAmaxVal    :: PY_RPtr Int    -> PY_IORPtr Int

foreign import ccall unsafe "include.h HS_AcceleratorfloatAminVal"
	hs_AcceleratorfloatAminVal  :: PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleAminVal"
	hs_AcceleratordoubleAminVal :: PY_RPtr Double -> PY_IORPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorintAminVal"
	hs_AcceleratorintAminVal    :: PY_RPtr Int    -> PY_IORPtr Int

foreign import ccall unsafe "include.h HS_AcceleratorfloatAnegate"
	hs_AcceleratorfloatAnegate  :: PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleAnegate"
	hs_AcceleratordoubleAnegate :: PY_RPtr Double -> PY_IORPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorintAnegate"
	hs_AcceleratorintAnegate    :: PY_RPtr Int    -> PY_IORPtr Int

foreign import ccall unsafe "include.h HS_AcceleratorfloatAproduct"
	hs_AcceleratorfloatAproduct  :: PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleAproduct"
	hs_AcceleratordoubleAproduct :: PY_RPtr Double -> PY_IORPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorintAproduct"
	hs_AcceleratorintAproduct    :: PY_RPtr Int    -> PY_IORPtr Int

foreign import ccall unsafe "include.h HS_AcceleratorfloatAsum"
	hs_AcceleratorfloatAsum  :: PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleAsum"
	hs_AcceleratordoubleAsum :: PY_RPtr Double -> PY_IORPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorintAsum"
	hs_AcceleratorintAsum    :: PY_RPtr Int    -> PY_IORPtr Int

foreign import ccall unsafe "include.h HS_AcceleratorfloatAsqrt"
	hs_AcceleratorfloatAsqrt  :: PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleAsqrt"
	hs_AcceleratordoubleAsqrt :: PY_RPtr Double -> PY_IORPtr Double

--foreign import ccall "include.h HS_AcceleratorintAsqrt"
--	hs_AcceleratorintAsqrt :: PY_RPtr Int -> PY_IORPtr Int

--foreign import ccall "include.h HS_AcceleratorintAsqrt"
--	hs_AcceleratorintAsqrt    :: PY_RPtr Int    -> PY_IORPtr Int

-- Triple element operations.

foreign import ccall unsafe "include.h HS_AcceleratorfloatAAAselect"
	hs_AcceleratorfloatAAAselect  :: PY_RPtr Float  -> PY_RPtr Float  -> PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleAAAselect"
	hs_AcceleratordoubleAAAselect :: PY_RPtr Double -> PY_RPtr Double -> PY_RPtr Double -> PY_IORPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorintAAAselect"
	hs_AcceleratorintAAAselect    :: PY_RPtr Int    -> PY_RPtr Int    -> PY_RPtr Int    -> PY_IORPtr Int

-- Others
foreign import ccall unsafe "include.h HS_AcceleratorfloatIAshift"
	hs_AcceleratorfloatIAshift  :: Int -> PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleIAshift"
	hs_AcceleratordoubleIAshift :: Int -> PY_RPtr Double -> PY_IORPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorintIAshift"
	hs_AcceleratorintIAshift    :: Int -> PY_RPtr Int    -> PY_IORPtr Int

foreign import ccall unsafe "include.h HS_AcceleratorfloatIApow"
	hs_AcceleratorfloatIApow  :: Float -> PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleIApow"
	hs_AcceleratordoubleIApow :: Double -> PY_RPtr Double -> PY_IORPtr Double

--foreign import ccall "include.h HS_AcceleratorintIApow"
--	hs_AcceleratorintIApow    :: Int -> PY_RPtr Int    -> PY_IORPtr Int

foreign import ccall unsafe "include.h HS_AcceleratorfloatIAlog_"
	hs_AcceleratorfloatIAlog  :: Float -> PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleIAlog_"
	hs_AcceleratordoubleIAlog :: Double -> PY_RPtr Double -> PY_IORPtr Double

--foreign import ccall "include.h HS_AcceleratorintIAlog"
--	hs_AcceleratorintIAlog    :: Int -> PY_RPtr Int    -> PY_IORPtr Int

--foreign import ccall "include.h HS_AcceleratorfloatIArotate"
--	hs_AcceleratorfloatIArotate  :: Int -> PY_RPtr Float  -> PY_IORPtr Float

--foreign import ccall "include.h HS_AcceleratordoubleIArotate"
--	hs_AcceleratordoubleIArotate :: Int -> PY_RPtr Double -> PY_IORPtr Double

--foreign import ccall "include.h HS_AcceleratorintIArotate"
--	hs_AcceleratorintIArotate    :: Int -> PY_RPtr Int    -> PY_IORPtr Int

{-foreign import ccall "include.h HS_AcceleratorFloatXXY"
	hs_AcceleratorfloatAAXXY  :: PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall "include.h HS_AcceleratorDoubleXXY"
	hs_AcceleratordoubleAAXXY :: PY_RPtr Double -> PY_IORPtr Double

foreign import ccall "include.h HS_AcceleratorIntXXY"
	hs_AcceleratorintAAXXY    :: PY_RPtr Int    -> PY_IORPtr Int-}

{-
	Start of Boolean operators.  Need to add greater support to the Accelerator
	target for boolean arrays, as they're now needed all over.
-}

foreign import ccall unsafe "include.h HS_AcceleratorfloatAAeq"
	hs_AcceleratorfloatAAeq :: PY_RPtr Float -> PY_RPtr Float -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratorfloatAAgt"
	hs_AcceleratorfloatAAgt :: PY_RPtr Float -> PY_RPtr Float -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratorfloatAAgeq"
	hs_AcceleratorfloatAAgeq :: PY_RPtr Float -> PY_RPtr Float -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratorfloatAAlt"
	hs_AcceleratorfloatAAlt :: PY_RPtr Float -> PY_RPtr Float -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratorfloatAAleq"
	hs_AcceleratorfloatAAleq :: PY_RPtr Float -> PY_RPtr Float -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratorfloatAAneq"
	hs_AcceleratorfloatAAneq :: PY_RPtr Float -> PY_RPtr Float -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratordoubleAAeq"
	hs_AcceleratordoubleAAeq :: PY_RPtr Double -> PY_RPtr Double -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratordoubleAAgt"
	hs_AcceleratordoubleAAgt :: PY_RPtr Double -> PY_RPtr Double -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratordoubleAAgeq"
	hs_AcceleratordoubleAAgeq :: PY_RPtr Double -> PY_RPtr Double -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratordoubleAAlt"
	hs_AcceleratordoubleAAlt :: PY_RPtr Double -> PY_RPtr Double -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratordoubleAAleq"
	hs_AcceleratordoubleAAleq :: PY_RPtr Double -> PY_RPtr Double -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratordoubleAAneq"
	hs_AcceleratordoubleAAneq :: PY_RPtr Double -> PY_RPtr Double -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratorintAAeq"
	hs_AcceleratorintAAeq :: PY_RPtr Int -> PY_RPtr Int -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratorintAAgt"
	hs_AcceleratorintAAgt :: PY_RPtr Int -> PY_RPtr Int -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratorintAAgeq"
	hs_AcceleratorintAAgeq :: PY_RPtr Int -> PY_RPtr Int -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratorintAAlt"
	hs_AcceleratorintAAlt :: PY_RPtr Int -> PY_RPtr Int -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratorintAAleq"
	hs_AcceleratorintAAleq :: PY_RPtr Int -> PY_RPtr Int -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratorintAAneq"
	hs_AcceleratorintAAneq :: PY_RPtr Int -> PY_RPtr Int -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratorboolAAand_"
	hs_AcceleratorboolAAand :: PY_RPtr Bool -> PY_RPtr Bool -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratorboolAAor_"
	hs_AcceleratorboolAAor :: PY_RPtr Bool -> PY_RPtr Bool -> PY_IORPtr Bool

foreign import ccall unsafe "include.h HS_AcceleratorboolAnot_"
	hs_AcceleratorboolAnot :: PY_RPtr Bool -> PY_RPtr Bool

{-
foreign import ccall "include.h HS_AcceleratorfloatXXY"
	hs_AcceleratorfloatAAXXX :: PY_RPtr Float -> PY_RPtr Float -> PY_IORPtr Float

foreign import ccall "include.h HS_AcceleratordoubleXXY"
	hs_AcceleratordoubleAAXXX :: PY_RPtr Double -> PY_RPtr Double -> PY_IORPtr Double

foreign import ccall "include.h HS_AcceleratorintXXY"
	hs_AcceleratorintAAXXX :: PY_RPtr Int -> PY_RPtr Int -> PY_IORPtr Int
-}

foreign import ccall unsafe "include.h HS_AcceleratorfloatAAAcond"
	hs_AcceleratorfloatAAAcond  :: PY_RPtr Bool -> PY_RPtr Float  -> PY_RPtr Float  -> PY_IORPtr Float

foreign import ccall unsafe "include.h HS_AcceleratordoubleAAAcond"
	hs_AcceleratordoubleAAAcond :: PY_RPtr Bool -> PY_RPtr Double -> PY_RPtr Double -> PY_IORPtr Double

foreign import ccall unsafe "include.h HS_AcceleratorintAAAcond"
	hs_AcceleratorintAAAcond    :: PY_RPtr Bool -> PY_RPtr Int    -> PY_RPtr Int    -> PY_IORPtr Int
