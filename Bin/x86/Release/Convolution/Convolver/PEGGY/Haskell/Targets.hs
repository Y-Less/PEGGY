{-# LANGUAGE MultiParamTypeClasses       #-}
{-# LANGUAGE FlexibleInstances           #-}
{-# LANGUAGE FlexibleContexts            #-}
{-# LANGUAGE TypeFamilies                #-}

module PEGGY.Haskell.Targets where

import PEGGY.Types

import qualified Data.Vector.Unboxed as V

{-
 - Haskell target information.
 -}

data TargetHaskell

data instance PYStorage dom TargetHaskell = PYStorageHaskell ([] dom) | PYReplicateHaskell dom
data instance PYNative  dom TargetHaskell = PYNativeHaskell  ([] dom) ([] Int)

instance
		PYStorable Float TargetHaskell
	where
		pySetList  x = PYStorageHaskell   (x)
		pySetNum _ x = PYReplicateHaskell (x)

instance
		PYStorable Double TargetHaskell
	where
		pySetList  x = PYStorageHaskell   (x)
		pySetNum _ x = PYReplicateHaskell (x)

instance
		PYStorable Int TargetHaskell
	where
		pySetList  x = PYStorageHaskell   (x)
		pySetNum _ x = PYReplicateHaskell (x)

instance
		PYStorable Bool TargetHaskell
	where
		pySetList  x = PYStorageHaskell   (x)
		pySetNum _ x = PYReplicateHaskell (x)

instance
		(Show dom) =>
		Show (PYStorage dom TargetHaskell)
	where
		show (PYStorageHaskell   dat) = show dat --foldl (\ s d -> s ++ show d ++ ", ") "" dat
		show (PYReplicateHaskell dat) = "Replicate " ++ show dat --foldl (\ s d -> s ++ show d ++ ", ") "" dat

instance
		(Show dom) =>
		Show (PYNative dom TargetHaskell)
	where
		show (PYNativeHaskell dat _) = show dat --foldl (\ s d -> s ++ show d ++ ", ") "" dat
