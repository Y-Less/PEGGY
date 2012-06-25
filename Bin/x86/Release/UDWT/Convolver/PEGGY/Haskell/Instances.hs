{-# LANGUAGE MultiParamTypeClasses       #-}
{-# LANGUAGE UndecidableInstances        #-}
{-# LANGUAGE FlexibleInstances           #-}
{-# LANGUAGE FlexibleContexts            #-}
{-# LANGUAGE ViewPatterns                #-}
{-# LANGUAGE TypeFamilies                #-}

module PEGGY.Haskell.Instances where

import PEGGY.Types
import PEGGY.Haskell.Targets

import qualified Data.Vector.Unboxed as V

{- 
 - PYExpresible instances.
 -}
instance
		(dom0 ~ dom2, id0 ~ TargetHaskell) =>
		PYExpressible (PYNative dom0 id0) dom2 TargetHaskell
	where
		pySet (PYNativeHaskell x n) = PYExpr (PYStorageHaskell x) n

{- 
 - PYExecute instances.
 -}
instance
		PYExecute dom TargetHaskell
	where
		pyExecute ex =
			let
				pyExecute' (PYStorageHaskell   x) len = PYNativeHaskell x len
				pyExecute' (PYReplicateHaskell x) len = PYNativeHaskell (replicate (foldl (*) 1 len) x) len
			in
				pyExecute' (pyExpr ex) (pyLength ex)

{- 
 - PYReturn instances.
 -}
instance
		(V.Unbox dom) =>
		PYReturn dom TargetHaskell
	where
		pyReturnDims (PYNativeHaskell _ n) = n
		pyReturnList (PYNativeHaskell x _) = x
