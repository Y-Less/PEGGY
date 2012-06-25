{- # LANGUAGE FunctionalDependencies      # -}
{-# LANGUAGE MultiParamTypeClasses       #-}
{-# LANGUAGE UndecidableInstances        #-}
{-# LANGUAGE FlexibleInstances           #-}
{-# LANGUAGE FlexibleContexts            #-}
{-# LANGUAGE ViewPatterns                #-}
{- # LANGUAGE TypeFamilies                # -}
{-# LANGUAGE Rank2Types                  #-}

module PEGGY.Haskell.Functions where

import PEGGY.Types
import PEGGY.Functions
import PEGGY.Haskell.Targets
import PEGGY.Haskell.Instances
import PEGGY.Operations

-- Split a list up in to a list of smaller lists, all of the same length.  These
-- are our inner-most dimensions.  It turns out that we don't need the "parts"
-- array.
unconcat _   []    = []
unconcat len dayta = dat0 : unconcat len dat1
	where
		(dat0, dat1) = splitAt len dayta

pySetHaskell' x = case pyExpr x of
					PYStorageHaskell z -> z
					PYReplicateHaskell z -> case length (pyLength x) of
						0 -> [z]
						_ -> replicate (foldl1 (*) (pyLength x)) z
--pySetHaskell' (pyExpr -> PYReplicateHaskell z) = z

-- I forgot to propogate the real dimensions before, now I don't.
liftPYHaskellCond2 ::
		(PYStorable dom TargetHaskell, PYStorable Bool TargetHaskell) =>
		(dom -> dom -> Bool) -> PYExpr dom TargetHaskell -> PYExpr dom TargetHaskell -> PYExpr Bool TargetHaskell
liftPYHaskellCond2 func x y = case pyGetResultSize x y of
				Just l  -> PYExpr (lift' (pyExpr x) (pyExpr y)) l
				Nothing -> error "Lengths do not match in liftPYHaskellCond2"
		where
			lift'  (PYStorageHaskell x')   (PYStorageHaskell y')  =  PYStorageHaskell  $ zipWith func x' y'
			lift'  (PYStorageHaskell x')  (PYReplicateHaskell y') =  PYStorageHaskell  $ map (\ x'' -> func x'' y') x'
			lift' (PYReplicateHaskell x')  (PYStorageHaskell y')  =  PYStorageHaskell  $ map (\ y'' -> func x' y'') y'
			lift' (PYReplicateHaskell x') (PYReplicateHaskell y') = PYReplicateHaskell $ func x' y'

liftPYHaskell2 ::
		(PYStorable dom TargetHaskell) =>
		(dom -> dom -> dom) -> PYExpr dom TargetHaskell -> PYExpr dom TargetHaskell -> PYExpr dom TargetHaskell
liftPYHaskell2 func x y = case pyGetResultSize x y of
				Just l  -> PYExpr (lift' (pyExpr x) (pyExpr y)) l
				Nothing -> error "Lengths do not match in liftPYHaskell2"
		--PYExpr (PYStorageHaskell $ zipWith func (pySetHaskell' x) (pySetHaskell' y)) (pyLength x)
		where
			lift'  (PYStorageHaskell x')   (PYStorageHaskell y')  =  PYStorageHaskell  $ zipWith func x' y'
			lift'  (PYStorageHaskell x')  (PYReplicateHaskell y') =  PYStorageHaskell  $ map (\ x'' -> func x'' y') x'
			lift' (PYReplicateHaskell x')  (PYStorageHaskell y')  =  PYStorageHaskell  $ map (\ y'' -> func x' y'') y'
			lift' (PYReplicateHaskell x') (PYReplicateHaskell y') = PYReplicateHaskell $ func x' y'

liftPYHaskell1 ::
		(PYStorable dom TargetHaskell) =>
		(dom -> dom) -> PYExpr dom TargetHaskell -> PYExpr dom TargetHaskell
liftPYHaskell1 func x = PYExpr (lift' (pyExpr x)) (pyLength x)
		--PYExpr (PYStorageHaskell $ map func (pySetHaskell' x)) (pyLength x)
		where
			lift'  (PYStorageHaskell x')  =  PYStorageHaskell  $ map func x'
			lift' (PYReplicateHaskell x') = PYReplicateHaskell $ func x'

--reducePYHaskell ::
--		(PYStorable dom TargetHaskell) =>
--		([dom] -> dom) -> PYExpr dom TargetHaskell -> PYExpr dom TargetHaskell
reducePYHaskell func =
	let
		doSplit x at = map func (unconcat at x)
		pyReduce' (PYStorageHaskell   x) o _ = PYExpr (PYStorageHaskell (doSplit x o))
		-- Optimise for replicated values.  Just do the operation on one small
		-- segment then replicate the result over higher dimensions.
		pyReduce' (PYReplicateHaskell x) o r = PYExpr (PYStorageHaskell (replicate r . func $ replicate o x))
		s' x = case (pyLength x) of
			[] -> []
			l' -> init l'
		e' x = case (pyLength x) of
			[] -> 1
			l' -> last l'
	in
		-- Single element.  If you pass a replicate of unknown dimensions ([])
		-- this will make it a single dimension, single element, array as there
		-- is no way to correctly evaluate the function in the future with the
		-- known sizes then.
		pyLift1 (\x -> pyReduce' (pyExpr x) (e' x) (foldl (*) 1 (s' x)) (s' x ++ [1]))
		--pyReduce' (pyExpr x) (replicate (length (pyLength x)) 1) (foldl (*) 1 (pyLength x)))

instance
		(PY2 c0 c1 Float TargetHaskell) =>
		PYAdd c0 c1 Float TargetHaskell
	where
		pyAdd = pyLift2 (liftPYHaskell2 (+))

instance
		(PY2 c0 c1 Float TargetHaskell) =>
		PYSub c0 c1 Float TargetHaskell
	where
		pySub = pyLift2 (liftPYHaskell2 (-))

instance
		(PY2 c0 c1 Float TargetHaskell) =>
		PYMul c0 c1 Float TargetHaskell
	where
		pyMul = pyLift2 (liftPYHaskell2 (*))

instance
		(PY2 c0 c1 Float TargetHaskell) =>
		PYDiv c0 c1 Float TargetHaskell
	where
		pyDiv = pyLift2 (liftPYHaskell2 (/))

instance
		(PY2 c0 c1 Int TargetHaskell) =>
		PYDiv c0 c1 Int TargetHaskell
	where
		pyDiv = pyLift2 (liftPYHaskell2 div)

instance
		(PY2 c0 c1 Float TargetHaskell) =>
		PYMin c0 c1 Float TargetHaskell
	where
		pyMin = pyLift2 (liftPYHaskell2 min)

instance
		(PY2 c0 c1 Float TargetHaskell) =>
		PYMax c0 c1 Float TargetHaskell
	where
		pyMax = pyLift2 (liftPYHaskell2 max)

instance
		(PY1 c0 Float TargetHaskell) =>
		PYAbs c0 Float TargetHaskell
	where
		pyAbs = pyLift1 (liftPYHaskell1 abs)

instance
		(PY1 c0 Float TargetHaskell) =>
		PYMaxVal c0 Float TargetHaskell
	where
		pyMaxVal = reducePYHaskell maximum

instance
		(PY1 c0 Float TargetHaskell) =>
		PYMinVal c0 Float TargetHaskell
	where
		pyMinVal = reducePYHaskell minimum

instance
		(PY1 c0 Float TargetHaskell) =>
		PYNegate c0 Float TargetHaskell
	where
		pyNegate = pyLift1 (liftPYHaskell1 negate)

instance
		(PY1 c0 Float TargetHaskell) =>
		PYProduct c0 Float TargetHaskell
	where
		pyProduct = reducePYHaskell product

instance
		(PY1 c0 Float TargetHaskell) =>
		PYSum c0 Float TargetHaskell
	where
		pySum = reducePYHaskell sum

instance
		(PY1 c0 Float TargetHaskell) =>
		PYSqrt c0 Float TargetHaskell
	where
		pySqrt = pyLift1 (liftPYHaskell1 sqrt)

instance
		(PY1 c0 Float TargetHaskell) =>
		PYPow c0 Float TargetHaskell
	where
		-- Need to read this correctly - raises b to n.
		pyPow b = pyLift1 (liftPYHaskell1 ((**) b))

instance
		(PY1 c0 Float TargetHaskell) =>
		PYLog c0 Float TargetHaskell
	where
		pyLog b = pyLift1 (liftPYHaskell1 (logBase b))

-- Split a list up in to chunks.  Map the function with the relevant parameter
-- over the chunks, pass that list of chunks to the next dimension level for
-- processing, then recombine everything.  This code will generate arbitrary
-- dimensions of lists, but again the types will all resolve as it all comes out
-- at the end.
{-mapOverDims func values lengths dayta = mapDims' func (reverse values) (reverse lengths) dayta
	--where
mapDims' :: forall a b . (b -> [a] -> [a]) -> [b] -> [Int] -> [a] -> [a]
mapDims' _ _        []       _ = []
mapDims' f (s : _ ) (n : []) x = concat $ map (f s) (unconcat n x)
mapDims' _ []       (_ : _)  _ = error "Insufficient shifts in \"TargetHaskell pyShift\""
mapDims' f (s : ss) (n : ns) x = concat $ mapDims' f ss ns $ map (f s) (unconcat n x)-}

-- Rank two types.  The function will work on lists and lists of lists etc,
-- which is what we have, but first we need to convice the type system that it
-- WILL work!
foldOverDims :: (forall d . b -> [d] -> [d]) -> [Int] -> [b] -> [a] -> [a]
foldOverDims func dims values dayta = mapDims' func (reverse dims) (reverse values) dayta
	where
		mapDims' :: (forall d . b -> [d] -> [d]) -> [Int] -> [b] -> [a] -> [a]
		mapDims' _ []       _        _ = []
		mapDims' f (n : []) (s : _ ) x = concat $ map (f s) (unconcat n x)
		mapDims' _ (_ : _)  []       _ = error "Insufficient values in \"TargetHaskell foldOverDims\""
		mapDims' f (n : ns) (s : ss) x = concat $ mapDims' f ns ss $ map (f s) (unconcat n x)

-- Annoyingly, it seems I need a separate instance for this.
foldOverDimsForConst :: (forall d . b -> d -> [d] -> [d]) -> a -> [Int] -> [b] -> [a] -> [a]
foldOverDimsForConst func ext dims values dayta = mapDims' func ext (reverse dims) (reverse values) dayta
	where
		mapDims' :: (forall d . b -> d -> [d] -> [d]) -> a -> [Int] -> [b] -> [a] -> [a]
		mapDims' _  _  []       _        _ = []
		mapDims' f ext (n : []) (s : _ ) x = concat $ map (f s ext) (unconcat n x)
		mapDims' _  _  (_ : _)  []       _ = error "Insufficient values in \"TargetHaskell foldOverDims\""
		mapDims' f ext (n : ns) (s : ss) x = concat $ mapDims' f (rep n ext) ns ss $ map (f s ext) (unconcat n x)
		rep n e = replicate n e

-- Rank two types.  The function will work on lists and lists of lists etc,
-- which is what we have, but first we need to convice the type system that it
-- WILL work!
mapOverDims :: (forall d . [d] -> [d]) -> [Int] -> [a] -> [a]
mapOverDims func dims dayta = mapDims' func (reverse dims) dayta
	where
		mapDims' :: (forall d . [d] -> [d]) -> [Int] -> [a] -> [a]
		mapDims' _ []       _ = []
		mapDims' f (n : []) x = concat $ map f (unconcat n x)
		mapDims' f (n : ns) x = concat $ mapDims' f ns $ map f (unconcat n x)
-- Not many functions will need to call "mapOverDims" as for most cases the
-- number of dimensions makes no difference to the code at all.  For example
-- multiplying two arrays is done element-by-element, regardless of where those
-- elements are, so we just operate on the 1D array.  Shifts however are
-- dimension specific (gnostic (pertaining to knowledge), not agnostic).

instance
		(PY1 c0 Float TargetHaskell) =>
		PYShiftExtend c0 Float TargetHaskell
	where
		pyShiftExtend q =
			let
				-- First or last element replicated by abs(shift).
				extra' f s x = replicate (abs s) (f x)
				-- Shift the array, replicating the first or last element.
				doShift s x
					| s > 0     = drop s x ++ extra' last s x
					| s < 0     = extra' head s x ++ take (length x + s) x
					| otherwise = x
				-- Split the list up, shift each part, then combine the results
				-- so that we don't get any bleed from other dimensions.
				pyShift' s x = PYExpr (PYStorageHaskell $ foldOverDims doShift (pyLength x) s (pySetHaskell' x)) (pyLength x)
			in
				pyLift1 (pyShift' q)

instance
		(PY1 c0 Float TargetHaskell) =>
		PYShiftRotate c0 Float TargetHaskell
	where
		pyShiftRotate q =
			let
				doRotate s x
					| s > 0     = drop s x ++ take s x
					| s < 0     = drop (length x + s) x ++ take (length x + s) x
					| otherwise = x
				-- Split the list up, shift each part, then combine the results
				-- so that we don't get any bleed from other dimensions.
				pyRotate' s x = PYExpr (PYStorageHaskell $ foldOverDims doRotate (pyLength x) s (pySetHaskell' x)) (pyLength x)
			in
				pyLift1 (pyRotate' q)

instance
		(PY1 c0 Float TargetHaskell) =>
		PYShiftConst c0 Float TargetHaskell
	where
		pyShiftConst val q =
			let
				--replicate' []       v = v
				--replicate' (x : xs) v = replicate (length x) (replicate' xs )
				-- First or last element replicated by abs(shift).
				--extra' s x = replicate (abs s) $ phlReplicate unsafeCoerce (head x) val
				-- Shift the array, filling the gaps with a constant value.
				doConst s extra x
					| s > 0     = drop s x ++ replicate (abs s) extra
					| s < 0     = replicate (abs s) extra ++ take (length x + s) x
					| otherwise = x
				-- Split the list up, shift each part, then combine the results
				-- so that we don't get any bleed from other dimensions.
				--pyConst' :: Float -> [Int] -> PYExpr Float TargetHaskell -> PYExpr Float TargetHaskell
				pyConst' s x = PYExpr (PYStorageHaskell $ foldOverDimsForConst doConst val (pyLength x) s (pySetHaskell' x)) (pyLength x)
			in
				pyLift1 (pyConst' q)

instance
		(PY0 c0 Bool TargetHaskell,
		 PY2 c1 c2 Float TargetHaskell) =>
		PYIfThenElse c0 c1 c2 Float TargetHaskell
	where
		pyIfThenElse cond =
			let
				pyIfThenElse' (pySetHaskell' -> test') (pySetHaskell' -> true') (pySetHaskell' -> false')
					= pySet (zipWith3 ifThenElse' test' true' false')
				ifThenElse' i t e = if i then t else e
			in
				pyLift2 (pyIfThenElse' (pyLift0 cond :: PYExpr Bool TargetHaskell))

type TargetHaskellCond = PYExpr Float TargetHaskell -> PYExpr Float TargetHaskell -> PYExpr Bool TargetHaskell

instance
		(PYStorable Float TargetHaskell,
		 PYStorable Bool TargetHaskell,
		 PYCond2 c0 c1 Float TargetHaskell) =>
		PYGT c0 c1 Float TargetHaskell
	where
		pyGT = pyLiftCond2 (liftPYHaskellCond2 (>) :: TargetHaskellCond)

instance
		(PYStorable Float TargetHaskell,
		 PYStorable Bool TargetHaskell,
		 PYCond2 c0 c1 Float TargetHaskell) =>
		PYLT c0 c1 Float TargetHaskell
	where
		pyLT = pyLiftCond2 (liftPYHaskellCond2 (<) :: TargetHaskellCond)

instance
		(PYStorable Float TargetHaskell,
		 PYStorable Bool TargetHaskell,
		 PYCond2 c0 c1 Float TargetHaskell) =>
		PYGTE c0 c1 Float TargetHaskell
	where
		pyGTE = pyLiftCond2 (liftPYHaskellCond2 (>=) :: TargetHaskellCond)

instance
		(PYStorable Float TargetHaskell,
		 PYStorable Bool TargetHaskell,
		 PYCond2 c0 c1 Float TargetHaskell) =>
		PYLTE c0 c1 Float TargetHaskell
	where
		pyLTE = pyLiftCond2 (liftPYHaskellCond2 (<=) :: TargetHaskellCond)

instance
		(PYStorable Float TargetHaskell,
		 PYStorable Bool TargetHaskell,
		 PYCond2 c0 c1 Float TargetHaskell) =>
		PYEQ c0 c1 Float TargetHaskell
	where
		pyEQ = pyLiftCond2 (liftPYHaskellCond2 (==) :: TargetHaskellCond)

instance
		(PYStorable Float TargetHaskell,
		 PYStorable Bool TargetHaskell,
		 PYCond2 c0 c1 Float TargetHaskell) =>
		PYNEQ c0 c1 Float TargetHaskell
	where
		pyNEQ = pyLiftCond2 (liftPYHaskellCond2 (==) :: TargetHaskellCond)

instance
		(PY1 c0 Float TargetHaskell) =>
		PYConvolve c0 Float TargetHaskell
	where
		pyConvolve shift filter = pyLift1 $ pyNativeConvolve shift filter

{-
-- We only operate on the lowest dimension.
dims x = length (pyLength x)

-- Work out how many sets of the lowest dimension there are.  An
-- "array" of [3][2][5] will return 6.
parts []            = 0
parts (_ : [])      = 1
parts (x0 : _ : []) = x0
parts (x0 : xs)     = x0 * parts xs
-}
