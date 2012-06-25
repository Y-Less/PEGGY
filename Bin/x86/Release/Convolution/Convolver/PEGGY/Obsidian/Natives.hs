{-# LANGUAGE MultiParamTypeClasses       #-}
{-# LANGUAGE FlexibleInstances           #-}
{-# LANGUAGE TypeFamilies                #-}
{-# LANGUAGE ExistentialQuantification   #-}

module PEGGY.Obsidian.Natives where

import PEGGY.Types

import Obsidian.GCDObsidian
import qualified Obsidian.GCDObsidian.CodeGen.InOut as InOut

import Data.Word

import Foreign.ForeignPtr
import Foreign.Ptr
import Foreign.Storable

import Control.Monad.State (State)

import Foreign.Marshal.Alloc
import qualified Foreign.Marshal.Array as Marshal

import Data.List (elemIndex)
import Data.Vector.Unboxed (Vector) --hiding ((++), length, head, concat, replicate, foldl)

{-
 - Obsidian target information.
 -}

data TargetObsidian

data instance PYStorage dom TargetObsidian =
	forall b . InOut.InOut b => PYStorageObsidian (b -> Array Pull (Exp dom)) (IO b) (PYOData dom) [Word32]
--	forall b . InOut.InOut b => PYStorageObsidian (b -> Array Pull (Exp dom)) (State Int b) (PYOData dom) [Word32]

data PYOData a = PYOArr [a] | PYOPair (PYOData a, PYOData a) | PYONone a | PYOVec (Vector a)
{-
instance (Eq a) => Eq (PYOData a) where
	(==) (PYOArr a)        (PYOArr b)        = a == b
	(==) (PYONone a)       (PYONone b)       = a == b
	(==) (PYOArr a)        (PYOPair (c, d))  = (PYOArr a) == c && (PYOArr a) == d
	(==) (PYOPair (a, b))  (PYOArr c)        = a == (PYOArr c) && b == (PYOArr c)
	(==) (PYONone a)       (PYOPair (c, d))  = (PYONone a) == c && (PYONone a) == d
	(==) (PYOPair (a, b))  (PYONone c)       = a == (PYONone c) && b == (PYONone c)
	(==) (PYOPair (a, b))  (PYOPair (c, d))  = a == c && b == d
	(==) _ _ = False
-}
--internalObsidianRun :: forall b . InOut.InOut b => (b -> Array Pull (Exp dom)) (State Int b) (PYOData dom) [Word32]
--internalObsidianRun

{-
type ObsPointer a = IO (ForeignPtr a)
type ObsDictionary a = ([a], ObsPointer a)

-- May need to upgrade this to "Kernel A" and sync things to pass data between
-- everywhere, then work out what goes where at the end.

data instance PYStorage dom TargetObsidian =
	forall b . InOut.InOut b => PYStorageObsidian (b -> Array Pull (Exp dom)) b (PYOData dom) Word32

data PYOData a = PYOArr [a] | PYOPair (PYOData a, PYOData a) | PYONone

-- Smart constructors.

--pyObsCreate1 :: Word32 -> Scalar dom => Array Pull (Exp dom) -> PYStorage dom TargetObsidian
--pyObsCreate1 leng arr = PYStorageObsidian id arr leng

class PYOPush a where
	pyoPush :: PYOData a -> [ObsDictionary a] -> IO [ObsDictionary a]

-- Convert a list to a C array if one doesn't already exist.
instance PYOPush Float where
	pyoPush (PYOArr a) dic = pushPair' hs_CUDAFloatAlloc hs_CUDAFloatFree hs_CUDAFloatPush a dic
	pyoPush (PYOPair (a0, a1)) dic = pyoPush a1 dic >>= pyoPush a0
	pyoPush _ dic = return dic

-- Convert a list to a C array if one doesn't already exist.
--instance PYOPair [Int] Int where
--	pushPair = pushPair' hs_CUDAIntAlloc hs_CUDAIntFree

pushPair' funcC funcD funcP arr dic =
	let
		(lists, ptrs) = unzip dic
		idx' = elemIndex arr lists
		(ptr', dic') = case idx' of
			-- Not in the dictionary already.  Send the data to C and add
			-- the list to the dictionary so we know it has been marshalled.
			Nothing  -> (ptr'', dic'')
				where
					ptr'' = do
						arr' <- Marshal.newArray arr
						ptr''' <- funcC (length arr) arr'
						newForeignPtr funcD ptr'''
					dic'' = (arr, ptr'') : dic
			-- Already marshalled.
			Just idx -> (ptrs !! idx, dic)
	in
		-- Push the data to the GPU in the correct order.  May involve lists
		-- being pushed multiple times if they are used multiple times.
		do
			usptr <- ptr'
			let
				ptr = unsafeForeignPtrToPtr usptr
			funcP ptr
			-- Make the unsafe code always safe (not that it isn't anyway).
			touchForeignPtr usptr
			return dic'

--instance (PYOPair b d, PYOPair c d) => PYOPair (b, c) d where
	--pushPair (arr0, arr1) dic = pushPair arr1 (pushPair arr0 dic)
--	pushPair (arr0, arr1) dic = pushPair arr0 dic >>= pushPair arr1

-- pyObsCreate1 :: Scalar dom => [dom] -> PYStorage dom TargetObsidian
-- pyObsCreate1 arr = PYStorageObsidian id pya leng
	-- where
		-- len = (fromInteger . toInteger . length) arr
		-- pya = namedArray "x" (min 512 len)

--data PYCUDAParams a = PYCUDAParams
--type PYCUDAPtr a = Ptr (PYCUDAParams a)

{-
foreign import ccall "include.h HS_CUDAFloatAlloc"
	hs_CUDAFloatAlloc      :: Int -> Ptr Float -> IO (Ptr Float)

foreign import ccall "include.h &HS_CUDAFloatFree"
	hs_CUDAFloatFree       :: FinalizerPtr Float

foreign import ccall "include.h HS_CUDAFloatPush"
	hs_CUDAFloatPush       :: Ptr Float -> IO ()

foreign import ccall "include.h HS_CUDAFloatStart"
	hs_CUDAFloatStart      :: Int -> Int -> IO ()

foreign import ccall "include.h HS_CUDAFloatRun"
	hs_CUDAFloatRun        :: Int -> IO ()

foreign import ccall "include.h HS_CUDAFloatEnd"
	hs_CUDAFloatEnd        :: IO (Ptr Float)

foreign import ccall "include.h HS_Test"
	hs_Test            :: IO ()
-}
-}
