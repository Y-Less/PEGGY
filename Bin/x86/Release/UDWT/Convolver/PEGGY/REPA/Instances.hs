{-# LANGUAGE MultiParamTypeClasses       #-}
{-# LANGUAGE UndecidableInstances        #-}
{-# LANGUAGE FlexibleInstances           #-}
{-# LANGUAGE FlexibleContexts            #-}
{-# LANGUAGE ViewPatterns                #-}
{-# LANGUAGE TypeFamilies                #-}

module PEGGY.Repa.Instances ( PYExpressible (..), PYExecute (..), PYReturn (..)) where

import PEGGY.Types
import PEGGY.Repa.Targets

import Data.Array.Repa as Repa
import Data.Array.Repa.Shape		as S

pyListLen x = listOfShape (extent x)

pyFlatten x = reshape (Z :. (foldl (*) 1 len)) x
	where
		len = pyListLen x

pyMakeNative shape x = PYNativeRepa (pyFlatten (force (reshape shape x))) (listOfShape shape)

{- 
 - PYExpresible instances.
 -}
instance
		(dom0 ~ dom2, id0 ~ TargetRepa, Elt dom2) =>
		PYExpressible (PYNative dom0 id0) dom2 TargetRepa
	where
		pySet (PYNativeRepa x n) = PYExpr (PYStorageRepa (pyFlatten x)) n
		--pyRun = id
		--pyGet (PYNativeRepa x _) = x

instance
		(dom0 ~ dom2, Elt dom2) =>
		PYExpressible (PYArr dom0) dom2 TargetRepa
	where
		pySet x = PYExpr (PYStorageRepa (pyFlatten x)) (pyListLen x)
		--pyRun = id
		--pyGet (PYNativeRepa x _) = x

{- 
 - PYExecute instances.
 -}
instance
		(Elt dom) =>
		PYExecute dom TargetRepa
	where
		-- Actually do (force) the calculation in Repa.  We need different
		-- intermediate types depending on the length of the dimensions array,
		-- but we can wrap that up so it doesn't affect the external type.
		pyExecute ex
			| length (pyLength ex) == 0 = 
				let
					PYStorageRepa x = pyExpr ex
					shape = shapeOfList (pyLength ex) :: DIM0
				in
					pyMakeNative shape x
		pyExecute ex
			| length (pyLength ex) == 1 = 
				let
					PYStorageRepa x = pyExpr ex
					shape = shapeOfList (pyLength ex) :: DIM1
				in
					pyMakeNative shape x
		pyExecute ex
			| length (pyLength ex) == 2 = 
				let
					PYStorageRepa x = pyExpr ex
					shape = shapeOfList (pyLength ex) :: DIM2
				in
					pyMakeNative shape x
		pyExecute ex
			| length (pyLength ex) == 3 = 
				let
					PYStorageRepa x = pyExpr ex
					shape = shapeOfList (pyLength ex) :: DIM3
				in
					pyMakeNative shape x
		pyExecute ex
			| length (pyLength ex) == 4 = 
				let
					PYStorageRepa x = pyExpr ex
					shape = shapeOfList (pyLength ex) :: DIM4
				in
					pyMakeNative shape x
		pyExecute ex
			| length (pyLength ex) == 5 = 
				let
					PYStorageRepa x = pyExpr ex
					shape = shapeOfList (pyLength ex) :: DIM5
				in
					pyMakeNative shape x
		-- Repa only goes up to DIM5, and we only go up to DIM2.
		pyExecute _ = error "Unsupported dimension in \"TargetRepa pyExecute\""

{- 
 - PYReturn instances.
 -}
instance
		(Elt dom) =>
		PYReturn dom TargetRepa
	where
		pyReturnDims   (PYNativeRepa _ n) = n
		pyReturnList   (PYNativeRepa x _) = toList x
		pyReturnVector (PYNativeRepa x _) = toVector x
