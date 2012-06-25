module PEGGY.Operations (
		pyNativeConvolve
	) where

import PEGGY
import PEGGY.Functions

-- I have recently modified many of the types and classes in PEGGY given that I
-- now have more experience with what they do and what is needed.  The new types
-- give a very elegant (if not long) type for this function which expresses
-- exactly what is needed to run it:
{-
convolve ::
	(PYAdd (PYExpr dom0 id0) (PYExpr dom0 id0) dom0 id0,
	 PYMul (PYExpr dom0 id0) (Replicate dom0) dom0 id0,
	 PYShift (PYExpr dom0 id0) dom0 id0,
	 PYRotate (PYExpr dom0 id0) dom0 id0,
	 PYStorable dom0 id0,
	 PYExpressible arr dom1 id1,
	 PYExpressible arr dom0 id0,
	 Num dom0) =>
	ConvolutionMovement -> ConvolutionFilter -> arr -> PYExpr dom0 id0
-}
-- Unfortunately, this type cannot by explicitly declared (no idea why).  I'm
-- also fairly certain that "dom1" is always "Bool".
pyNativeConvolve shiftType (Filter filterDims filterValues) dayta = internalConvolve1 shiftType filterDims filterValues dayta--foldl pyAdd (pySet $ Replicate dlen 0) (internalConvolve1 shiftType filterDims filterValues dayta) -- :: [PYExpr Float TargetHaskell])
	where
		--dayta = pySet dayta'
		-- Confirm that the thing
		dlen = pyLength dayta

-- Separable convolver (or just 1D, but either way, it makes no difference).
internalConvolve1 shiftType (filterDim : []) filterValues dayta = doOneLevel dayta allShifts
	where
		-- Get the function that will do the apron extension.
		shiftFunc = sORr shiftType doShift doRotate doConst
		
		-- Generate a full array with only the current filter value.
		dataLengths = pyLength dayta
		rep m = Replicate dataLengths m
		
		-- Get the number of dimensions in every direction.
		numberOfDims = length dataLengths
		highIndex = numberOfDims - 1
		
		--numberOfFilters = length filterDim
		
		-- Make a list of shifts where only one dimension at a time has any
		-- data in it: [[[0, 1], [0]], [[0], [0, 1]]].
		separableShifts =
			let
				onedim n 0 = [replicate (n) 1 ++ [filterDim]]
				onedim n x = [replicate (n - x) 1 ++ filterDim : replicate x 1] ++ onedim n (x - 1)
				--parts x = splitAt x (replicate highIndex [0])
			in
				map (makeOffset shiftType) (onedim highIndex highIndex)
		
		-- All the shifts, one for every convolution side.
		allShifts = map genAllShifts separableShifts
		
		doOneLevel dat []       = dat
		--doOneLevel dat (s : []) = error $ show dat --foldl1 pyAdd $ zipWith shiftFunc s filterValues
		doOneLevel dat (s : ss) = doOneLevel (foldl1 pyAdd $ zipWith (shiftFunc dat) s filterValues) ss
		
		doShift   dat by m = pyShiftExtend  by dat .*. rep m
		doRotate  dat by m = pyShiftRotate  by dat .*. rep m
		doConst n dat by m = pyShiftConst n by dat .*. rep m

internalConvolve1 shiftType filterDims filterValues dayta = foldl1 pyAdd (zipWith (shiftFunc dayta) allShifts filterValues)
	where
		-- Get the function that will do the apron extension.
		shiftFunc = sORr shiftType doShift doRotate doConst
		
		-- Generate a full array with only the current filter value.
		dataLengths = pyLength dayta
		rep m = Replicate dataLengths m
		
		-- Get the number of dimensions in every direction.
		numberOfDims = length dataLengths
		highIndex = numberOfDims - 1
		
		-- Get all the offsets to every 
		allShifts = genShifts shiftType filterDims
		
		-- Two functions for the core algorithms.
		doShift   dat by m = pyShiftExtend  by dat .*. rep m
		doRotate  dat by m = pyShiftRotate  by dat .*. rep m
		doConst n dat by m = pyShiftConst n by dat .*. rep m

-- Make this work on N-D arrays.

--allNumbersEver = [0 .. ]

genShifts s dims = genAllShifts $ makeOffset s dims

-- Generate a list of all possible offsets in every dimension.
makeOffset  ShiftLeft        dims = map (\ n -> [0 .. n - 1]                   ) dims
makeOffset  ShiftRight       dims = map (\ n -> [(1 - n) .. 0]                 ) dims
makeOffset  ShiftRelative    dims = map (\ n -> take n [(1 - n) `div` 2 .. ]   ) dims
makeOffset (ShiftX x y)      dims = map (\ n -> [i * x + y | i <- [0 .. n - 1]]) dims
makeOffset  RotateLeft       dims = map (\ n -> [0 .. n - 1]                   ) dims
makeOffset  RotateRight      dims = map (\ n -> [(1 - n) .. 0]                 ) dims
makeOffset  RotateRelative   dims = map (\ n -> take n [(1 - n) `div` 2 .. ]   ) dims
makeOffset (RotateX x y)     dims = map (\ n -> [i * x + y | i <- [0 .. n - 1]]) dims
makeOffset (ConstLeft _)     dims = map (\ n -> [0 .. n - 1]                   ) dims
makeOffset (ConstRight _)    dims = map (\ n -> [(1 - n) .. 0]                 ) dims
makeOffset (ConstRelative _) dims = map (\ n -> take n [(1 - n) `div` 2 .. ]   ) dims
makeOffset (ConstX x y _)    dims = map (\ n -> [i * x + y | i <- [0 .. n - 1]]) dims

-- Combine all the per-dimension offsets to get every single offset.
genAllShifts (x : []) = map (\ y -> [y]) x
genAllShifts (x : xs) = 
	concat $ zipWith
		(\ a b -> map ((:) a) b)
		x $
		replicate (length x) (genAllShifts xs)

-- Generate a list of all possible offsets in every dimension.
sORr ShiftLeft      func0 _ _ = func0
sORr ShiftRight     func0 _ _ = func0
sORr ShiftRelative  func0 _ _ = func0
sORr (ShiftX _ _)   func0 _ _ = func0
sORr RotateLeft     _ func1 _ = func1
sORr RotateRight    _ func1 _ = func1
sORr RotateRelative _ func1 _ = func1
sORr (RotateX _ _)  _ func1 _ = func1
sORr (ConstLeft n)     _ _ func2 = func2 n
sORr (ConstRight n)    _ _ func2 = func2 n
sORr (ConstRelative n) _ _ func2 = func2 n
sORr (ConstX _ _ n)    _ _ func2 = func2 n
