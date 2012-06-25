{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}

module PEGGY.Accelerator.Conversion where

import PEGGY.Types
import PEGGY.Accelerator.Natives
import PEGGY.Accelerator.Targets

import Foreign.ForeignPtr
import qualified Foreign.ForeignPtr.Unsafe as Unsafe
import Foreign.Storable

import Foreign.C.Types
import Foreign.Marshal.Array

import Data.Maybe (fromJust)
import Data.Typeable

{- 
 - Lift functions operating on raw data pointers to functions operating on data
 - pointers with associated finalisers.
 -}
{-pyLiftAcceleratorRToA0 ::
	(PY_RPtr dom -> IO z) -> (PY_APtr dom -> IO z)
pyLiftAcceleratorRToA0 func = \ x -> do
	let
		x' = Unsafe.unsafeForeignPtrToPtr x
	result' <- func x'
	-- So Haskell doesn't clean up the data too early.
	touchForeignPtr x
	--newAcceleratorPointer result'
	return result'-}

pyLiftAcceleratorRToA1 ::
	(PY_RPtr dom -> PY_IORPtr dom) -> (PY_APtr dom -> PY_IOAPtr dom)
pyLiftAcceleratorRToA1 func = \ x -> do
	let
		x' = Unsafe.unsafeForeignPtrToPtr x
	result' <- func x'
	-- So Haskell doesn't clean up the data too early.
	touchForeignPtr x
	newAcceleratorPointer result'

pyLiftAcceleratorRToA2 ::
	(PY_RPtr dom -> PY_RPtr dom -> PY_IORPtr dom) -> (PY_APtr dom -> PY_APtr dom -> PY_IOAPtr dom)
pyLiftAcceleratorRToA2 func = \ x y -> do
	let
		x' = Unsafe.unsafeForeignPtrToPtr x
		y' = Unsafe.unsafeForeignPtrToPtr y
	result' <- func x' y'
	-- So Haskell doesn't clean up the data too early.
	touchForeignPtr x
	touchForeignPtr y
	newAcceleratorPointer result'

pyLiftAcceleratorRToACond2 ::
	(PY_RPtr dom -> PY_RPtr dom -> PY_IORPtr Bool) -> (PY_APtr dom -> PY_APtr dom -> PY_IOAPtr Bool)
pyLiftAcceleratorRToACond2 func = \ x y -> do
	let
		x' = Unsafe.unsafeForeignPtrToPtr x
		y' = Unsafe.unsafeForeignPtrToPtr y
	result' <- func x' y'
	-- So Haskell doesn't clean up the data too early.
	touchForeignPtr x
	touchForeignPtr y
	newAcceleratorPointer result'

pyLiftAcceleratorRToA3 ::
	(PY_RPtr dom -> PY_RPtr dom -> PY_RPtr dom -> PY_IORPtr dom) -> (PY_APtr dom -> PY_APtr dom -> PY_APtr dom -> PY_IOAPtr dom)
pyLiftAcceleratorRToA3 func = \ x y z -> do
	let
		x' = Unsafe.unsafeForeignPtrToPtr x
		y' = Unsafe.unsafeForeignPtrToPtr y
		z' = Unsafe.unsafeForeignPtrToPtr z
	result' <- func x' y' z'
	-- So Haskell doesn't clean up the data too early.
	touchForeignPtr x
	touchForeignPtr y
	touchForeignPtr z
	newAcceleratorPointer result'

{- 
 - Lift functions operating on pointers with associated finalisers to functions
 - operating on PYExpr data.
 -}
{-pyLiftAcceleratorAToE0 ::
	(TargetAccelerator id) =>
	(PY_APtr dom -> IO z) -> (PYExpr dom id -> IO z)
pyLiftAcceleratorAToE0 func = \ x ->
	let
		func' =	do
			x' <- targetAcceleratorUnwrap (pyExpr x)
			func x'
	in
		PYExpr (targetAcceleratorWrap func') (pyLength x)-}

pyLiftAcceleratorAToE1 ::
	(TargetAccelerator id, PYAcceleratorConst dom) =>
	(PY_APtr dom -> PY_IOAPtr dom) -> (PYExpr dom id -> PYExpr dom id)
pyLiftAcceleratorAToE1 func = \ x ->
	let
		func' =	--do
			case targetAcceleratorUnwrap (pyExpr x) of
				Left x'   -> x' >>= func
				Right dat -> if pyNoLength (pyLength x) then error "Unknown constant size in pyLiftAcceleratorAToE1" else pyAcceleratorConst (pyLength x) dat >>= func
				--return $ error "Unknown constant size in pyLiftAcceleratorAToE1"
			--x' <- targetAcceleratorUnwrap (pyExpr x)
			--func x'
	in
		PYExpr (targetAcceleratorWrap func') (pyLength x)

pyLiftAcceleratorAToE2 ::
	(TargetAccelerator id, PYAcceleratorConst dom) =>
	(PY_APtr dom -> PY_APtr dom -> PY_IOAPtr dom) -> (PYExpr dom id -> PYExpr dom id -> PYExpr dom id)
pyLiftAcceleratorAToE2 func = \ x y ->
	let
		-- "Either" values (array/const).
		x'' = case targetAcceleratorUnwrap (pyExpr x) of
			Left  dat -> dat
			Right dat -> pyAcceleratorConst len dat
		y'' = case targetAcceleratorUnwrap (pyExpr y) of
			Left  dat -> dat
			Right dat -> pyAcceleratorConst len dat
		-- Get the result length (must have a value).
		len = case pyGetResultSize x y of
			Just []  -> error "Unknown constant size in pyLiftAcceleratorAToE2"
			Just dat -> if pyNoLength dat then error "Unknown constant size in pyLiftAcceleratorAToE2" else dat
			Nothing  -> error $ "Lengths do not match in pyLiftAcceleratorAToE2 : " ++ show (pyLength x) ++ ", " ++ show (pyLength y)
		func' =	do
			x' <- x''
			y' <- y''
			func x' y'
	in
		PYExpr (targetAcceleratorWrap func') len

pyLiftAcceleratorAToECond2 ::
	(TargetAccelerator id, PYAcceleratorConst dom) =>
	(PY_APtr dom -> PY_APtr dom -> PY_IOAPtr Bool) -> (PYExpr dom id -> PYExpr dom id -> PYExpr Bool id)
pyLiftAcceleratorAToECond2 func = \ x y ->
	{-let
		func' =	do
			x' <- targetAcceleratorUnwrap (pyExpr x)
			y' <- targetAcceleratorUnwrap (pyExpr y)
			func x' y'
	in
		case pyGetResultSize x y of
			Just len -> PYExpr (targetAcceleratorWrap func') len
			Nothing  -> error $ "Lengths do not match in pyLiftAcceleratorAToECond2 : " ++ show (pyLength x) ++ ", " ++ show (pyLength y)
		--if (pyLength x) == (pyLength y) then
		--else-}
	let
		-- "Either" values (array/const).
		x'' = case targetAcceleratorUnwrap (pyExpr x) of
			Left  dat -> dat
			Right dat -> pyAcceleratorConst len dat
		y'' = case targetAcceleratorUnwrap (pyExpr y) of
			Left  dat -> dat
			Right dat -> pyAcceleratorConst len dat
		-- Get the result length (must have a value).
		len = case pyGetResultSize x y of
			Just []  -> error "Unknown constant size in pyLiftAcceleratorAToECond2"
			Just dat -> if pyNoLength dat then error "Unknown constant size in pyLiftAcceleratorAToECond2" else dat
			Nothing  -> error $ "Lengths do not match in pyLiftAcceleratorAToECond2 : " ++ show (pyLength x) ++ ", " ++ show (pyLength y)
		func' =	do
			x' <- x''
			y' <- y''
			func x' y'
	in
		PYExpr (targetAcceleratorWrap func') len

class PYAcceleratorConst dom where
	pyAcceleratorConst :: [] Int -> dom -> PY_IOAPtr dom
	pyAcceleratorConst = undefined

instance PYAcceleratorConst Float where
	pyAcceleratorConst len val = withArray (len ++ [0]) (\ l -> hs_AcceleratorFloatCreateC l val) >>= newAcceleratorPointer

instance PYAcceleratorConst Double where
	pyAcceleratorConst len val = withArray (len ++ [0]) (\ l -> hs_AcceleratorDoubleCreateC l val) >>= newAcceleratorPointer

instance PYAcceleratorConst Int where
	pyAcceleratorConst len val = withArray (len ++ [0]) (\ l -> hs_AcceleratorIntCreateC l val) >>= newAcceleratorPointer

--instance PYAcceleratorConst a where
--	pyAcceleratorConst _ _ = undefined

-- We sort of hacked the types here a little bit...
{-
pyAcceleratorConst :: (Typeable dom) => [] Int -> dom -> PY_IOAPtr dom
pyAcceleratorConst len val = fromJust $ case show (typeOf val) of
		"Float"   -> gcast $ expr' hs_AcceleratorFloatCreateC  float'
		"Double"  -> gcast $ expr' hs_AcceleratorDoubleCreateC double'
		"Int"     -> gcast $ expr' hs_AcceleratorIntCreateC    int'
		otherwise -> undefined
	where
		expr' :: (PY_CPtr Int -> dom -> PY_IORPtr dom) -> dom -> PY_IOAPtr dom
		expr' f v = withArray (len ++ [0]) (\ l -> f l v) >>= newAcceleratorPointer
		float'  = fromJust (cast val :: Maybe Float )
		double' = fromJust (cast val :: Maybe Double)
		int'    = fromJust (cast val :: Maybe Int   )
-}

pyLiftAcceleratorAToE3 ::
	(TargetAccelerator id, PYAcceleratorConst dom) =>
	(PY_APtr dom -> PY_APtr dom -> PY_APtr dom -> PY_IOAPtr dom) -> (PYExpr dom id -> PYExpr dom id -> PYExpr dom id -> PYExpr dom id)
pyLiftAcceleratorAToE3 func = \ x y z ->
	let
		-- "Either" values (array/const).
		x'' = case targetAcceleratorUnwrap (pyExpr x) of
			Left  dat -> dat
			Right dat -> pyAcceleratorConst len dat
		y'' = case targetAcceleratorUnwrap (pyExpr y) of
			Left  dat -> dat
			Right dat -> pyAcceleratorConst len dat
		z'' = case targetAcceleratorUnwrap (pyExpr z) of
			Left  dat -> dat
			Right dat -> pyAcceleratorConst len dat
		--y'' = targetAcceleratorUnwrap (pyExpr y)
		--z'' = targetAcceleratorUnwrap (pyExpr z)
		-- Get the result length (must have a value).
		len = case pyGetResultSize3 x y z of
			Just []  -> error "Unknown constant size in pyLiftAcceleratorAToE3"
			Just dat -> if dat == replicate (length dat) 1 then error "Unknown constant size in pyLiftAcceleratorAToE3" else dat
			Nothing  -> error $ "Lengths do not match in pyLiftAcceleratorAToE3 : " ++ show (pyLength x) ++ ", " ++ show (pyLength y) ++ ", " ++ show (pyLength z)
		func' =	do
			x' <- x''
			y' <- y''
			z' <- z''
			func x' y' z'
	in
		PYExpr (targetAcceleratorWrap func') len

{-liftPYAcc0 ::
	(TargetAccelerator id) =>
	(PY_RPtr dom -> z) -> (PYExpr dom id -> z)
liftPYAcc1 func = \ x ->
	pyLiftAcceleratorAToE0 (pyLiftAcceleratorRToA0 func) x-}

liftPYAcc1 ::
	(TargetAccelerator id, PYAcceleratorConst dom) =>
	(PY_RPtr dom -> PY_IORPtr dom) -> (PYExpr dom id -> PYExpr dom id)
liftPYAcc1 func = \ x ->
	pyLiftAcceleratorAToE1 (pyLiftAcceleratorRToA1 func) x

liftPYAcc2 ::
	(TargetAccelerator id, PYAcceleratorConst dom) =>
	(PY_RPtr dom -> PY_RPtr dom -> PY_IORPtr dom) -> (PYExpr dom id -> PYExpr dom id -> PYExpr dom id)
liftPYAcc2 func = \ x y ->
	pyLiftAcceleratorAToE2 (pyLiftAcceleratorRToA2 func) x y

liftPYAccCond2 ::
	(TargetAccelerator id, PYAcceleratorConst dom) =>
	(PY_RPtr dom -> PY_RPtr dom -> PY_IORPtr Bool) -> (PYExpr dom id -> PYExpr dom id -> PYExpr Bool id)
liftPYAccCond2 func = \ x y ->
	pyLiftAcceleratorAToECond2 (pyLiftAcceleratorRToACond2 func) x y

liftPYAcc3 ::
	(TargetAccelerator id, PYAcceleratorConst dom) =>
	(PY_RPtr dom -> PY_RPtr dom -> PY_RPtr dom -> PY_IORPtr dom) -> (PYExpr dom id -> PYExpr dom id -> PYExpr dom id -> PYExpr dom id)
liftPYAcc3 func = \ x y z ->
	pyLiftAcceleratorAToE3 (pyLiftAcceleratorRToA3 func) x y z

{- 
 - Lift functions operating on PYExpr data to functions operating on any data
 - expressble as a PYExpr.
 -}

-- The type system actually caught an error here before I even wrote it, as in I
-- noticed the types didn't match and went digging.
pyDoRunAcc' :: (PY_RPtr dom -> PY_IOCPtr dom) -> (FinalizerPtr dom) -> PY_IOAPtr dom -> PY_IOHPtr dom
pyDoRunAcc' func finaliser array = do
	let
		func' x = withForeignPtr x func
	array' <- array
	x' <- func' array'
	newForeignPtr finaliser x'
