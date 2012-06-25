{-# LANGUAGE MultiParamTypeClasses       #-}
{-# LANGUAGE FlexibleInstances           #-}
{-# LANGUAGE TypeFamilies                #-}

module PEGGY.Accelerator.Targets where

import PEGGY.Types
import PEGGY.Accelerator.Natives

import qualified Data.Vector.Unboxed as V

{-
 - Accelerator DirectX 9 target information.
 -}

data TargetDX9

--type family PYDataStore a

--type instance PYDataStore Float = Float
--type instance PYDataStore Double = Double
--type instance PYDataStore Int = Int
--type instance PYDataStore Bool = Bool
--type instance PYDataStore (V.Vector a) = a

--data instance PYStorage dom TargetDX9 = PYStorageDX9 (PY_IOAPtr (PYDataStore dom))
--data instance PYNative  dom TargetDX9 = PYNativeDX9  (PY_IOHPtr (PYDataStore dom)) Int

data instance PYStorage dom TargetDX9 = PYStorageDX9 (PY_IOAPtr dom) | PYReplicateDX9 dom
data instance PYNative  dom TargetDX9 = PYNativeDX9  (PY_IOHPtr dom) [Int]

{-data instance PYStorage Float TargetDX9 = PYStorageDX9F (PY_IOAPtr Float)
data instance PYNative  Float TargetDX9 = PYNativeDX9F  (PY_IOHPtr Float) Int

data instance PYStorage (Vector Float) TargetDX9 = PYStorageDX9VF (PY_IOAPtr Float)
data instance PYNative  (Vector Float) TargetDX9 = PYNativeDX9VF  (PY_IOHPtr Float) Int

data instance PYStorage Double TargetDX9 = PYStorageDX9D (PY_IOAPtr Double)
data instance PYNative  Double TargetDX9 = PYNativeDX9D  (PY_IOHPtr Double) Int

data instance PYStorage Int TargetDX9 = PYStorageDX9I (PY_IOAPtr Int)
data instance PYNative  Int TargetDX9 = PYNativeDX9I  (PY_IOHPtr Int) Int-}

instance PYStorable Float TargetDX9 where
	pySetVector x = PYStorageDX9 (vectorToAcceleratorArray' hs_AcceleratorFloatCreate x $ [V.length x])
	pySetList   x = PYStorageDX9 (listToAcceleratorArray'   hs_AcceleratorFloatCreate x $ [length x])
	pySetNum  c x = PYReplicateDX9 x --PYStorageDX9 (listToAcceleratorArray'   hs_AcceleratorFloatCreate [x] c)

instance PYStorable Double TargetDX9 where
	pySetVector x = PYStorageDX9 (vectorToAcceleratorArray' hs_AcceleratorDoubleCreate x $ [V.length x])
	pySetList   x = PYStorageDX9 (listToAcceleratorArray'   hs_AcceleratorDoubleCreate x $ [length x])
	pySetNum  c x = PYReplicateDX9 x --PYStorageDX9 (listToAcceleratorArray'   hs_AcceleratorDoubleCreate [x] c)

instance PYStorable Int TargetDX9 where
	pySetVector x = PYStorageDX9 (vectorToAcceleratorArray' hs_AcceleratorIntCreate x $ [V.length x])
	pySetList   x = PYStorageDX9 (listToAcceleratorArray'   hs_AcceleratorIntCreate x $ [length x])
	pySetNum  c x = PYReplicateDX9 x --PYStorageDX9 (listToAcceleratorArray'   hs_AcceleratorIntCreate [x] c)

--instance PYStorable Bool TargetDX9 where
--	pySetVector x = PYStorageDX9 (vectorToAcceleratorArray' hs_AcceleratorIntCreate x $ [V.length x])
--	pySetList   x = PYStorageDX9 (listToAcceleratorArray'   hs_AcceleratorIntCreate x $ [length x])
--	pySetNum  c x = PYStorageDX9 (listToAcceleratorArray'   hs_AcceleratorIntCreate [x] c)

{-
 - Accelerator X64-SSE target information.
 -}

data TargetX64

data instance PYStorage dom TargetX64 = PYStorageX64 (PY_IOAPtr dom) | PYReplicateX64 dom
data instance PYNative  dom TargetX64 = PYNativeX64  (PY_IOHPtr dom) [Int]

instance PYStorable Float TargetX64 where
	pySetVector x = PYStorageX64 (vectorToAcceleratorArray' hs_AcceleratorFloatCreate x $ [V.length x])
	pySetList   x = PYStorageX64 (listToAcceleratorArray'   hs_AcceleratorFloatCreate x $ [length x])
	pySetNum  c x = PYReplicateX64 x --PYStorageX64 (listToAcceleratorArray'   hs_AcceleratorFloatCreate [x] c)

instance PYStorable Double TargetX64 where
	pySetVector x = PYStorageX64 (vectorToAcceleratorArray' hs_AcceleratorDoubleCreate x $ [V.length x])
	pySetList   x = PYStorageX64 (listToAcceleratorArray'   hs_AcceleratorDoubleCreate x $ [length x])
	pySetNum  c x = PYReplicateX64 x --PYStorageX64 (listToAcceleratorArray'   hs_AcceleratorDoubleCreate [x] c)

instance PYStorable Int TargetX64 where
	pySetVector x = PYStorageX64 (vectorToAcceleratorArray' hs_AcceleratorIntCreate x $ [V.length x])
	pySetList   x = PYStorageX64 (listToAcceleratorArray'   hs_AcceleratorIntCreate x $ [length x])
	pySetNum  c x = PYReplicateX64 x --PYStorageX64 (listToAcceleratorArray'   hs_AcceleratorIntCreate [x] c)

{-
 - Accelerator FPGA target information.
 -}

data TargetFPGA

data instance PYStorage dom TargetFPGA = PYStorageFPGA (PY_IOAPtr dom) | PYReplicateFPGA dom
data instance PYNative  dom TargetFPGA = PYNativeFPGA  (PY_IOHPtr dom) [Int]

instance PYStorable Float TargetFPGA where
	pySetVector x = PYStorageFPGA (vectorToAcceleratorArray' hs_AcceleratorFloatCreate x $ [V.length x])
	pySetList   x = PYStorageFPGA (listToAcceleratorArray'   hs_AcceleratorFloatCreate x $ [length x])
	pySetNum  c x = PYReplicateFPGA x --PYStorageFPGA (listToAcceleratorArray'   hs_AcceleratorFloatCreate [x] c)

instance PYStorable Double TargetFPGA where
	pySetVector x = PYStorageFPGA (vectorToAcceleratorArray' hs_AcceleratorDoubleCreate x $ [V.length x])
	pySetList   x = PYStorageFPGA (listToAcceleratorArray'   hs_AcceleratorDoubleCreate x $ [length x])
	pySetNum  c x = PYReplicateFPGA x --PYStorageFPGA (listToAcceleratorArray'   hs_AcceleratorDoubleCreate [x] c)

instance PYStorable Int TargetFPGA where
	pySetVector x = PYStorageFPGA (vectorToAcceleratorArray' hs_AcceleratorIntCreate x $ [V.length x])
	pySetList   x = PYStorageFPGA (listToAcceleratorArray'   hs_AcceleratorIntCreate x $ [length x])
	pySetNum  c x = PYReplicateFPGA x --PYStorageFPGA (listToAcceleratorArray'   hs_AcceleratorIntCreate [x] c)

{-
 - Accelerator FPGA target information.
 -}

data TargetCUDA

data instance PYStorage dom TargetCUDA = PYStorageCUDA (PY_IOAPtr dom) | PYReplicateCUDA dom
data instance PYNative  dom TargetCUDA = PYNativeCUDA  (PY_IOHPtr dom) [Int]

instance PYStorable Float TargetCUDA where
	pySetVector x = PYStorageCUDA (vectorToAcceleratorArray' hs_AcceleratorFloatCreate x $ [V.length x])
	pySetList   x = PYStorageCUDA (listToAcceleratorArray'   hs_AcceleratorFloatCreate x $ [length x])
	pySetNum  c x = PYReplicateCUDA x --PYStorageCUDA (listToAcceleratorArray'   hs_AcceleratorFloatCreate [x] c)

instance PYStorable Double TargetCUDA where
	pySetVector x = PYStorageCUDA (vectorToAcceleratorArray' hs_AcceleratorDoubleCreate x $ [V.length x])
	pySetList   x = PYStorageCUDA (listToAcceleratorArray'   hs_AcceleratorDoubleCreate x $ [length x])
	pySetNum  c x = PYReplicateCUDA x --PYStorageCUDA (listToAcceleratorArray'   hs_AcceleratorDoubleCreate [x] c)

instance PYStorable Int TargetCUDA where
	pySetVector x = PYStorageCUDA (vectorToAcceleratorArray' hs_AcceleratorIntCreate x $ [V.length x])
	pySetList   x = PYStorageCUDA (listToAcceleratorArray'   hs_AcceleratorIntCreate x $ [length x])
	pySetNum  c x = PYReplicateCUDA x --PYStorageCUDA (listToAcceleratorArray'   hs_AcceleratorIntCreate [x] c)

{-
 - Additional declarations designed to replicate the Accelerator API.
 -}

type  FloatParallelArray id = PYExpr Float  id
type DoubleParallelArray id = PYExpr Double id
type    IntParallelArray id = PYExpr Int    id

class TargetAccelerator id where
	targetAcceleratorUnwrap :: PYStorage dom id -> Either (PY_IOAPtr dom) (dom)
	--targetAcceleratorConst  :: PYStorage dom id -> Maybe (dom)
	targetAcceleratorWrap   :: PY_IOAPtr dom -> PYStorage dom id
	--targetAcceleratorID     :: PYStorage dom id -> Int

instance TargetAccelerator TargetDX9 where
	targetAcceleratorUnwrap (PYStorageDX9 x) = Left x
	targetAcceleratorUnwrap (PYReplicateDX9 x) = Right x
	--targetAcceleratorConst (PYReplicateDX9 x) = Just x
	--targetAcceleratorConst (PYStorageDX9 x) = Nothing
	targetAcceleratorWrap   x = (PYStorageDX9 x)
	--targetAcceleratorID _ = 0

instance TargetAccelerator TargetX64 where
	targetAcceleratorUnwrap (PYStorageX64 x) = Left x
	targetAcceleratorUnwrap (PYReplicateX64 x) = Right x
	--targetAcceleratorConst (PYReplicateX64 x) = Just x
	--targetAcceleratorConst (PYStorageX64 x) = Nothing
	targetAcceleratorWrap   x = (PYStorageX64 x)
	--targetAcceleratorID _ = 1

instance TargetAccelerator TargetFPGA where
	targetAcceleratorUnwrap (PYStorageFPGA x) = Left x
	targetAcceleratorUnwrap (PYReplicateFPGA x) = Right x
	--targetAcceleratorConst (PYReplicateFPGA x) = Just x
	--targetAcceleratorConst (PYStorageFPGA x) = Nothing
	targetAcceleratorWrap   x = (PYStorageFPGA x)
	--targetAcceleratorID _ = 2

instance TargetAccelerator TargetCUDA where
	targetAcceleratorUnwrap (PYStorageCUDA x) = Left x
	targetAcceleratorUnwrap (PYReplicateCUDA x) = Right x
	--targetAcceleratorConst (PYReplicateCUDA x) = Just x
	--targetAcceleratorConst (PYStorageCUDA x) = Nothing
	targetAcceleratorWrap   x = (PYStorageCUDA x)
	--targetAcceleratorID _ = 2
