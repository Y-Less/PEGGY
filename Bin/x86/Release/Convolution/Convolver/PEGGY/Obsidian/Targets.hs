{-# LANGUAGE ExistentialQuantification   #-}
{-# LANGUAGE MultiParamTypeClasses       #-}
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

import Data.Vector.Unboxed (Vector, Unbox) --hiding ((++), length, head, concat, replicate, foldl)
import qualified Data.Vector.Unboxed as V

import Data.IORef

import Control.DeepSeq

pyNewLength []  y = y
pyNewLength  x [] = x
pyNewLength  x  y = Prelude.zipWith min x y

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

-- OK, this code does use "unsafePerformIO", but most of the time it still uses
-- "IO".  Interestingly, in one place it calls "unsafePerformIO", then wraps the
-- result up in "IO" using "return" - I don't think that's unsafe any more
-- because there is explicit IO threaded through the code.  This is just so that
-- we can use "deepseq" to FORCE a new name to be generated every time one of
-- the "pyObsCreateN" functions is FIRST CALLED, not when they eventually feel
-- like evaluating their arguments after having been copied multiple times.

gName :: IORef Int
--gName = newIORef 0
{-# NOINLINE gName #-}
gName = unsafePerformIO (newIORef 0)

mkNewArray len = do
	--ref <- gName
	num <- readIORef gName
	writeIORef gName (num + 1)
	return $ "input" ++ show num
	--return $ namedArray ("moon" ++ show num) (getLen len)

pyObsCreate1 :: Scalar dom => [dom] -> [Int] -> PYStorage dom TargetObsidian
pyObsCreate1 arr len' = pya `deepseq` PYStorageObsidian (translateAddr len) paa (PYOArr arr) len
	where
		len = map (fromInteger . toInteger) len'
		pya = unsafePerformIO $ mkNewArray len
		paa = (return $ namedArray pya (getLen len))
		--pya = indexArray (min 512 len)


-- indexArray is EXACTLY what is needed for the molecules code!  Not what I want
-- here, but frankly I don't much care!

pyObsCreate0 :: (Scalar dom, Num dom) => dom -> [Int] -> PYStorage dom TargetObsidian
--pyObsCreate0 num len' = PYStorageObsidian id pya PYONone len
pyObsCreate0 num len' = pya `deepseq` PYStorageObsidian pid paa (PYONone num) len
	where
		len = map (fromInteger . toInteger) len'
		--pya = mkPullArray (\ _ -> Literal num) (getLen len)
		--pya = mkPullArray (\ _ -> Literal num) (getLen len)
		--pya = mkPullArray (\ix -> BinOp Add (index "input0" ix) (Literal num)) 512
		pya
			| length len' == 0 = "constant"
			| otherwise        = unsafePerformIO $ mkNewArray len
		pid
			| length len' == 0 = \ _ -> mkPullArray (\ _ -> Literal num) (getLen len)
			| otherwise        = translateAddr len
		paa = (return $ namedArray pya (getLen len))

pyObsCreate2 :: (Scalar dom, Unbox dom) => Vector dom -> [Int] -> PYStorage dom TargetObsidian
pyObsCreate2 vec len' = pya `deepseq` PYStorageObsidian (translateAddr len) paa (PYOVec vec) len
	where
		len = map (fromInteger . toInteger) len'
		pya = unsafePerformIO $ mkNewArray len
		paa = (return $ namedArray pya (getLen len))


getLen [] = 1024
getLen x = (min 1024 (last x))


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
	=
		PYStorageObsidian func' oarr (PYOPair (arr0, arr1)) (pyNewLength l0 l1)
			where
				oarr = do
					inp0' <- inp0
					inp1' <- inp1
					return (inp0', inp1')
				func' (a, b) =
					mkPullArray
						(\ ix -> func (func0 a ! ix) (func1 b ! ix))
						-- (func (func0 a) (func1 b))
						(min (getLen l0) (getLen l1))

--pyObsMerge3 ::
--		forall c dom . PYOPair c dom =>
--		(Exp dom -> Exp dom -> Exp dom -> Exp dom) -> PYStorage dom TargetObsidian -> PYStorage dom TargetObsidian -> PYStorage dom TargetObsidian -> PYStorage dom TargetObsidian
pyObsMerge3
		func
		(PYStorageObsidian func0 inp0 arr0 l0)
		(PYStorageObsidian func1 inp1 arr1 l1)
		(PYStorageObsidian func2 inp2 arr2 l2)
	=
		PYStorageObsidian func' oarr (PYOPair (arr0, PYOPair (arr1, arr2))) (pyNewLength l0 (pyNewLength l1 l2))
			where
				oarr = do
					inp0' <- inp0
					inp1' <- inp1
					inp2' <- inp2
					return (inp0', (inp1', inp2'))
				min3 a b c = min a (min b c)
				func' (a, (b, c)) =
					mkPullArray
						(\ ix -> func (func0 a ! ix) (func1 b ! ix) (func2 c ! ix))
						-- (func (func0 a) (func1 b) (func2 c))
						(min3 (getLen l0) (getLen l1) (getLen l2))

--data instance PYStorage dom TargetObsidian = PYStorageObsidian (Array Pull (Exp dom))

data instance PYNative dom TargetObsidian =
	PYNativeObsidian (IO String) [Int]

--data instance PYNative  dom TargetObsidian = PYNativeObsidian  (Array Pull (Exp dom) -> Kernel (Array Pull (Exp dom))) Int

instance Show (PYNative Float TargetObsidian) where
	show (PYNativeObsidian s l) = --show . unsafePerformIO $
		unsafePerformIO s -- $ s >>= readFile . (++ ".cu")
		--s >>= Marshal.peekArray (fromInteger . toInteger $ l)

instance PYStorable Float TargetObsidian where
	pySetList c x = pyObsCreate1 x c-- (doFromLen x) $ namedArray "x" (min 512 $ doFromLen x)
	pySetNum  c x = pyObsCreate0 x c
	pySetVector c x = pyObsCreate2 x c
		
		
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
