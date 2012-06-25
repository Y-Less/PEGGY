{-# LANGUAGE MultiParamTypeClasses       #-}
{-# LANGUAGE FlexibleInstances           #-}
{-# LANGUAGE FlexibleContexts            #-}
{-# LANGUAGE TypeFamilies                #-}

module PEGGY.Repa.Targets where

import PEGGY.Types

import Data.Array.Repa as Repa
import qualified Data.Vector.Unboxed as V

type PYArr = Array DIM1

{-
 - Haskell target information.
 -}

data TargetRepa

lengthOfList :: [Int] -> DIM1
lengthOfList c = Z :. foldl (*) 1 c

data instance PYStorage dom TargetRepa = PYStorageRepa (PYArr dom)
data instance PYNative  dom TargetRepa = PYNativeRepa  (PYArr dom) ([] Int)

toDIM1 c x
	| length c == 0 = reshape (lengthOfList c) $ fromFunction (shapeOfList c :: DIM0) (\ _ -> x)
	| length c == 1 = reshape (lengthOfList c) $ fromFunction (shapeOfList c :: DIM1) (\ _ -> x)
	| length c == 2 = reshape (lengthOfList c) $ fromFunction (shapeOfList c :: DIM2) (\ _ -> x)
	| length c == 3 = reshape (lengthOfList c) $ fromFunction (shapeOfList c :: DIM3) (\ _ -> x)
	| length c == 4 = reshape (lengthOfList c) $ fromFunction (shapeOfList c :: DIM4) (\ _ -> x)
	| length c == 5 = reshape (lengthOfList c) $ fromFunction (shapeOfList c :: DIM5) (\ _ -> x)

instance
		PYStorable Float TargetRepa
	where
		pySetList    x = PYStorageRepa $ fromList (Z :. length x) x
		pySetVector  x = PYStorageRepa $ fromVector (Z :. V.length x) x
		pySetNum c   x = PYStorageRepa $ toDIM1 c x

instance
		PYStorable Double TargetRepa
	where
		pySetList    x = PYStorageRepa $ fromList (Z :. length x) x
		pySetVector  x = PYStorageRepa $ fromVector (Z :. V.length x) x
		pySetNum c   x = PYStorageRepa $ toDIM1 c x

instance
		PYStorable Int TargetRepa
	where
		pySetList    x = PYStorageRepa $ fromList (Z :. length x) x
		pySetVector  x = PYStorageRepa $ fromVector (Z :. V.length x) x
		pySetNum c   x = PYStorageRepa $ toDIM1 c x

instance
		PYStorable Bool TargetRepa
	where
		pySetList    x = PYStorageRepa $ fromList (Z :. length x) x
		pySetVector  x = PYStorageRepa $ fromVector (Z :. V.length x) x
		pySetNum c   x = PYStorageRepa $ toDIM1 c x

instance
		(Elt dom, Show dom) =>
		Show (PYStorage dom TargetRepa)
	where
		show (PYStorageRepa dat) = show dat --foldl (\ s d -> s ++ show d ++ ", ") "" dat

instance
		(Elt dom, Show dom) =>
		Show (PYNative dom TargetRepa)
	where
		--show (PYNativeRepa dat n) = show dat --foldl (\ s d -> s ++ show d ++ ", ") "" dat
		show (PYNativeRepa x n)
			| length n == 0 = 
				let
					shape = shapeOfList n :: DIM0
				in
					show (reshape shape x)
		show (PYNativeRepa x n)
			| length n == 1 = 
				let
					shape = shapeOfList n :: DIM1
				in
					show (reshape shape x)
		show (PYNativeRepa x n)
			| length n == 2 = 
				let
					shape = shapeOfList n :: DIM2
				in
					show (reshape shape x)
		show (PYNativeRepa x n)
			| length n == 3 = 
				let
					shape = shapeOfList n :: DIM3
				in
					show (reshape shape x)
		show (PYNativeRepa x n)
			| length n == 4 = 
				let
					shape = shapeOfList n :: DIM4
				in
					show (reshape shape x)
		show (PYNativeRepa x n)
			| length n == 5 = 
				let
					shape = shapeOfList n :: DIM5
				in
					show (reshape shape x)
		-- Repa only goes up to DIM5, and we only go up to DIM2.
		show _ = error "Unsupported dimension in \"TargetRepa show\""
