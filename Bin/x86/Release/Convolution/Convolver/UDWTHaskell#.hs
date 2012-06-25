-- Implementation of the UDWT example in PEGGY.

module UDWT where

import PEGGY
import PEGGY.Obsidian

import Data.Bits
import Data.List (sort)
import Foreign.Ptr
import Foreign.Marshal.Array
import Data.IORef

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- Filters.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

--foreign export stdcall setFilters :: Ptr Float -> Ptr Float -> Ptr Float -> Ptr Float -> Int -> IO ()
--foreign export stdcall generateCachedConvolution :: Word32 -> Float -> Word32 -> Word32 -> Word32 -> Word32 -> IO Int

gForward1, gReverse1, gForward2, gReverse2 :: IORef [Float]
{-# NOINLINE gForward1 #-}
{-# NOINLINE gReverse1 #-}
{-# NOINLINE gForward2 #-}
{-# NOINLINE gReverse2 #-}
gForward1 = unsafePerformIO (newIORef [])
gReverse1 = unsafePerformIO (newIORef [])
gForward2 = unsafePerformIO (newIORef [])
gReverse2 = unsafePerformIO (newIORef [])

setFilters forward1 reverse1 forward2 reverse2 len = do
	f1 <- peekArray len forward1
	r1 <- peekArray len reverse1
	f2 <- peekArray len forward2
	r2 <- peekArray len reverse2
	writeIORef gForward1 $ reverse f1
	writeIORef gReverse1 $ reverse r1
	writeIORef gForward2 $ reverse f2
	writeIORef gReverse2 $ reverse r2

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- Main.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

-- And now we make this callable from C.




--main = putStrLn $ show $ udwt n2H 10

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- Code.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

--convolve :: (PYExpressible arr dom id) => Int -> Int -> [Float] -> arr -> PYExpr dom id
convolve dir level filter input = foldl (+) 0 convolve'
	where
		convolve' = zipWith doOneShift [0 - dir .. shiftBounds - dir] filter
		pyInput = pySet input
		shiftBounds = length filter
		doOneShift sBy mBy = pyShiftRotate [sBy `shift` level] pyInput* pyConst mBy

fpconv f1 r1 xl    l = (,)   (convolve 0   l r1 xl) (convolve 0   l f1 xl)

bpconv f2 r2 xl xh l = pyAdd (convolve len l f2 xl) (convolve len l r2 xh)
	where
		len = length r2

mrdwt f1 r1 input levels = mrdwt' (pySet input) [] levels
	where
		mrdwt' inl inh 0     = (inl, inh)
		mrdwt' inl inh level =
			let
				(lo, hi) = fpconv f1 r1 inl (levels - level)
			in
				mrdwt' lo (inh ++ [hi]) (level - 1)

mirdwt f2 r2 xl xh levels = mirdwt' (pySet xl) (reverse . map pySet $ xh) (levels - 1)
	where
		mirdwt' inl []       _     = inl
		mirdwt' inl (x : xh) level =
			let
				res = bpconv f2 r2 inl x level
			in
				mirdwt' res xh (level - 1)





--udwt input levels = mirdwt (hard lowPass) (map hard highPass) levels
--	where
--		(lowPass, highPass) = mrdwt input levels
--		threshold' = threshold 0.5 ((pyGet . pyRun) (head highPass))
--		hard = hardTh threshold'

--threshold mul a = mul / 0.67 * case odd len of
--		True  -> arr !! (len `div` 2)
--		False -> (arr !! (len `div` 2 - 1) + arr !! (len `div` 2)) / 2
--	where
--		arr = (sort (map abs a))
--		len = length arr

--hardTh threshold' dayta = pyIfThenElse (dayta `pyGT` zero') zero' dayta
--	where
--		len   = pyLength dayta
--		zero' = puConst 0
