{-# LANGUAGE ExistentialQuantification   #-}
{-# LANGUAGE NoMonomorphismRestriction   #-}
{-# LANGUAGE MultiParamTypeClasses       #-}
{-# LANGUAGE UndecidableInstances        #-}
{-# LANGUAGE FlexibleInstances           #-}
{-# LANGUAGE FlexibleContexts            #-}
{-# LANGUAGE TypeFamilies                #-}

module PEGGY.Obsidian.Targets where

import PEGGY.Types
import PEGGY.Obsidian.Natives
import Foreign.Ptr

import Obsidian.GCDObsidian
import qualified Obsidian.GCDObsidian.CodeGen.InOut as InOut

import Data.Word

import System.IO.Unsafe
import qualified Foreign.Marshal.Array as Marshal

import Control.Monad.State (get, put, State, runState)

{-
 - Obsidian target information.
 -}

--data TargetObsidian

-- May need to upgrade this to "Kernel A" and sync things to pass data between
-- everywhere, then work out what goes where at the end.

--data instance PYStorage dom TargetObsidian =
--	forall b . InOut.InOut b => PYStorageObsidian (b -> Array Pull (Exp dom)) b Word32

-- Smart constructors.

--pyObsCreate1 :: Word32 -> Scalar dom => Array Pull (Exp dom) -> PYStorage dom TargetObsidian
--pyObsCreate1 leng arr = PYStorageObsidian id arr leng

--class PYOPair a b where
--	pushPair :: a -> ([b], IO (Ptr b)) -> ([b], IO (Ptr b))

--constObs :: Scalar dom => --State Int (Array Pull (Exp dom))
-- This function requires NoMonomorphismRestriction.
--class HasConstatRep dom where
--	constObs :: dom -> State Int (Array Pull (Exp dom)) --return $ namedArray "constants" 0

--constObs = return $ namedArray "constants" 0
constObs n = return $ mkPullArray (\ix -> BinOp Add (index "constant" ix) (Literal n)) 0
-- Requires UndecidableInstances.
--instance (Scalar dom) => HasConstatRep dom where
--	constObs = return $ namedArray "constants" 0

--newObsArray :: (InOut.InOut b) => Word32 -> State Int b
newObsArray :: Scalar dom => Word32 -> State Int (Array Pull (Exp dom))
newObsArray len = do
	num <- get
	put (num + 1)
	return $ namedArray ("input" ++ show num) len

pyObsCreate1 :: Scalar dom => [dom] -> PYStorage dom TargetObsidian
pyObsCreate1 arr = PYStorageObsidian (translateAddr [len]) pya (PYOArr arr) [len]
	where
		len = (fromInteger . toInteger . length) arr
		--pya = namedArray "x" (min 512 len)
		pya = newObsArray (min 512 len)
		--pya = indexArray (min 512 len)


-- indexArray is EXACTLY what is needed for the molecules code!  Not what I want
-- here, but frankly I don't much care!

pyObsCreate0 :: (Scalar dom, Num dom) => dom -> [Int] -> PYStorage dom TargetObsidian
--pyObsCreate0 num len' = PYStorageObsidian id pya PYONone len
pyObsCreate0 num len' = PYStorageObsidian pid (constObs num) (PYONone num) len
	where
		len = map (fromInteger . toInteger) len'
		--pya = mkPullArray (\ _ -> Literal num) (getLen len)
		--pya = mkPullArray (\ _ -> Literal num) (getLen len)
		--pya = mkPullArray (\ix -> BinOp Add (index "input0" ix) (Literal num)) (getLen len)
		--pya = (namedArray "input0" (getLen len)) -- :: Array Pull (Exp dom)
		pid _ = mkPullArray (\ _ -> Literal num) (getLen len)


getLen [] = 512
getLen x = (min 512 (last x))


translateAddr dims (Array (Pull f) l) = mkPullArray f' l
	--dims = reverse len
	where
		f' = case (length dims) of
			1 -> \idx -> f (x idx)
			2 -> \idx -> f ((y * w) + x idx)
			3 -> \idx -> f (((z * h + y) * w) + x idx)
			_ -> error "Obsidian only supports 1, 2, or 3 dimensions"
		x idx = idx --variable "blockIdx.x" * variable "X_BLOCK" + idx
		-- Only "x" threads currently.
		y = variable "blockIdx.y" -- * variable "Y_BLOCK" + variable "threadIdx.y"
		z = variable "blockIdx.z" -- * variable "Z_BLOCK" + variable "threadIdx.z"
		w = variable "PITCH" -- dims !! 0
		h = Literal (dims !! 1)
--pyObsCreate1 leng arr = PYStorageObsidian id arr leng

--pyObsCreate1 (doFromLen x) $ namedArray "x" (min 512 $ doFromLen x)

--pyObsCreate1 :: (Scalar dom, PYOPair [dom] dom) => [dom] -> PYStorage dom TargetObsidian
--pyObsMerge1 ::
--		forall c dom . PYOPair c dom =>
--		(Exp dom -> Exp dom) -> PYStorage dom TargetObsidian -> PYStorage dom TargetObsidian
pyObsMerge1
		func
		(PYStorageObsidian func0 inp0 arr0 l0)
	=
		PYStorageObsidian func' inp0 arr0 l0
			where
				func' a =
					mkPullArray
						(\ ix -> func (func0 a ! ix))
						-- (func (func0 a))
						(getLen l0)

--pyObsMerge2 ::
--		forall c dom . PYOPair c dom =>
--		(Exp dom -> Exp dom -> Exp dom) -> PYStorage dom TargetObsidian -> PYStorage dom TargetObsidian -> PYStorage dom TargetObsidian
-- When Obsidian generates code, it is given a tuple (of tuples (...)) of arrays
-- and a function that takes that type of tuple and returns code.  This means I
-- need some way to represent arrays that are the same as older arrays.  I'm
-- pretty sure that's what "named arrays" should do, but that doesn't seem to
-- work particularly well...  Or indeed at all, maybe I should fix it...
-- Actually, on further reflection I think the best way is to thread state
-- through my code, and run it at code generation time before CUDA code
-- generation time.
pyObsMerge2
		func
		(PYStorageObsidian func0 inp0 arr0 l0)
		(PYStorageObsidian func1 inp1 arr1 l1)
	| arr0 == arr1 = 
		let
			func' a =
				mkPullArray
					(\ ix -> func (func0 a ! ix) (func1 a ! ix))
					(getLen l0)
		in
			PYStorageObsidian func' inp0 arr0 l0
	| otherwise    =
		let
			nuState = do
				num <- get
				let
					(r0, n0) = runState inp0 num
					(r1, n1) = runState inp1 n0
				put n1
				return (r0, r1)
			func' (a, b) =
				mkPullArray
					(\ ix -> func (func0 a ! ix) (func1 b ! ix))
					-- (func (func0 a) (func1 b))
					(min (getLen l0) (getLen l1))
		in
			PYStorageObsidian func' nuState (PYOPair (arr0, arr1)) (Prelude.zipWith min (l0) (l1))

--pyObsMerge3 ::
--		forall c dom . PYOPair c dom =>
--		(Exp dom -> Exp dom -> Exp dom -> Exp dom) -> PYStorage dom TargetObsidian -> PYStorage dom TargetObsidian -> PYStorage dom TargetObsidian -> PYStorage dom TargetObsidian
-- Merge states and compress the output when similar states are found.
pyObsMerge3
		func
		(PYStorageObsidian func0 inp0 arr0 l0)
		(PYStorageObsidian func1 inp1 arr1 l1)
		(PYStorageObsidian func2 inp2 arr2 l2)
	| arr0 == arr1 && arr1 == arr2 =
		let
			func' a =
				mkPullArray
					(\ ix -> func (func0 a ! ix) (func1 a ! ix) (func2 a ! ix))
					(getLen l0)
		in
			PYStorageObsidian func' inp0 arr0 l0
	| arr0 == arr1    =
		let
			nuState = do
				num <- get
				let
					(r0, n0) = runState inp0 num
					(r2, n2) = runState inp2 n0
				put n2
				return (r0, r2)
			func' (a, c) =
				mkPullArray
					(\ ix -> func (func0 a ! ix) (func1 a ! ix) (func2 c ! ix))
					(min (getLen l0) (getLen l2))
		in
			PYStorageObsidian func' nuState (PYOPair (arr0, arr2)) (Prelude.zipWith min (l0) (l2))
	| arr0 == arr2    =
		let
			nuState = do
				num <- get
				let
					(r0, n0) = runState inp0 num
					(r2, n2) = runState inp1 n0
				put n2
				return (r0, r2)
			func' (a, b) =
				mkPullArray
					(\ ix -> func (func0 a ! ix) (func1 b ! ix) (func2 a ! ix))
					(min (getLen l0) (getLen l1))
		in
			PYStorageObsidian func' nuState (PYOPair (arr0, arr1)) (Prelude.zipWith min (l0) (l2))
	| arr1 == arr2    =
		let
			nuState = do
				num <- get
				let
					(r0, n0) = runState inp0 num
					(r2, n2) = runState inp2 n0
				put n2
				return (r0, r2)
			func' (a, c) =
				mkPullArray
					(\ ix -> func (func0 a ! ix) (func1 c ! ix) (func2 c ! ix))
					(min (getLen l0) (getLen l2))
		in
			PYStorageObsidian func' nuState (PYOPair (arr0, arr2)) (Prelude.zipWith min (l0) (l2))
	| otherwise    =
		let
			nuState = do
				num <- get
				let
					(r0, n0) = runState inp0 num
					(r1, n1) = runState inp1 n0
					(r2, n2) = runState inp2 n1
				put n2
				return (r0, (r1, r2))
			min3 a b c = min a (min b c)
			func' (a, (b, c)) =
				mkPullArray
					(\ ix -> func (func0 a ! ix) (func1 b ! ix) (func2 c ! ix))
					-- (func (func0 a) (func1 b) (func2 c))
					(min3 (getLen l0) (getLen l1) (getLen l2))
		in
			PYStorageObsidian func' nuState (PYOPair (arr0, PYOPair (arr1, arr2))) (zipWith3 min3 l0 l1 l2)

--data instance PYStorage dom TargetObsidian = PYStorageObsidian (Array Pull (Exp dom))

data instance PYNative dom TargetObsidian =
	PYNativeObsidian (IO String) [Int]

--data instance PYNative  dom TargetObsidian = PYNativeObsidian  (Array Pull (Exp dom) -> Kernel (Array Pull (Exp dom))) Int

instance Show (PYNative Float TargetObsidian) where
	show (PYNativeObsidian s l) = --show . unsafePerformIO $
		unsafePerformIO $ s >>= readFile . (++ ".cu")
		--s >>= Marshal.peekArray (fromInteger . toInteger $ l)

instance PYStorable Float TargetObsidian where
		pySetList   x = pyObsCreate1 x -- (doFromLen x) $ namedArray "x" (min 512 $ doFromLen x)
		pySetNum  c x = pyObsCreate0 x c
		
		
		--pyObsCreate1 (replicate c x) -- (doFromInt c) $ namedArray "x" (min 512 $ doFromInt c)

-- instance PYStorable Double TargetObsidian where
		-- pySetList   x = PYStorageObsidian $ OB.namedArray "" (length x)
		-- pySetNum  c x = PYStorageObsidian $ OB.namedArray "" c

--instance PYStorable Int TargetObsidian where
--		pySetList   x = pyObsCreate1 x -- (doFromLen x) $ namedArray "x" (min 512 $ doFromLen x)
--		pySetNum  c x = pyObsCreate1 (replicate c x) -- (doFromInt c) $ namedArray "x" (min 512 $ doFromInt c)

--doFromLen = fromInteger . toInteger . length
--doFromInt = fromInteger . toInteger


-- Need to somehow make this generate CUDA code.
-- instance
		-- (Show dom) =>
		-- Show (PYStorage dom TargetObsidian)
	-- where
		-- show (PYStorageHaskell dat) = show dat --foldl (\ s d -> s ++ show d ++ ", ") "" dat

-- instance
		-- (Show dom) =>
		-- Show (PYNative dom TargetObsidian)
	-- where
		-- show (PYNativeHaskell dat _) = show dat --foldl (\ s d -> s ++ show d ++ ", ") "" dat
