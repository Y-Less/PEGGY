{-# LANGUAGE MultiParamTypeClasses       #-}
{-# LANGUAGE UndecidableInstances        #-}
{-# LANGUAGE IncoherentInstances         #-}
{-# LANGUAGE FlexibleInstances           #-}
{-# LANGUAGE FlexibleContexts            #-}
{-# LANGUAGE ViewPatterns                #-}
{-# LANGUAGE TypeFamilies                #-}

{-
There are now three main functions, backed by various type classes implemented
at a per-target level.  "pySet", "pyRun" and "pyGet".  The first converts some
type to an expression which can be run.  The second runs an expression and
returns a type still in the native format but only holding that result.  The
last converts a native result to some Haskell type, or to an input to another
expression.  Note that "pySet :: PYNative -> PYExpr" and "pyGet :: PYNative ->
PYExpr" are the same function, the former converts something to "PYExpr", the
latter converts "PYNative" to something, so in this case their uses overlap.
-}

module PEGGY.Types where

import Data.Vector.Unboxed (Vector, Unbox) --hiding ((++), length, head, concat, replicate, foldl)
import qualified Data.Vector.Unboxed as V
--import Data.Array.Unboxed as V

data Replicate dom = Replicate ([] Int) dom

--type PYArray = Vector

{-
type PYFloatArray  = PYArray Float
type PYDoubleArray = PYArray Double
type PYIntArray    = PYArray Int
type PYBoolArray   = PYArray Bool

type PYArray a = StorableArray Int a
-}

{- 
 - PYStorage is a data family to store target specific data on the current
 - syntax tree while passing it around transparently.  Instantiated by every
 - target base file.
 -}
data family PYStorage dom id

{- 
 - PYNative is the type of data passed directly from the backend target system
 - or returned by an execution of a syntax tree.  This means that results aren't
 - marshalled to Haskell if they don't need to be (e.g. if they need to be re-
 - used by a future AST.
 -}
data family PYNative dom id

{-
 - PYExpr holds an expression and is passed around and operated on (immutably
 - of course).  Each of these are a node in a tree (or may be, for example in
 - Accelerator they are just a reference to the Accelerator DAG, and in Haskell
 - they are actually the result of the calculation so far).
 -}
data PYExpr dom id = PYExpr {
	pyExpr   :: PYStorage dom id,
	pyLength :: [] Int}

-- Total number of elements in all dimensions.
pySize = pyRep' . pyLength

-- Number of dimensions.
pyDims = length . pyLength

pyRep' x = (foldr (*) 1 x)

pyNoLength x = x == replicate (length x) 1

pyGetResultSize x y = get'
	where
		lx = pyLength x
		ly = pyLength y
		get'
			-- Must have the same dimensions.
			| length lx == 0                = Just (ly)
			| length ly == 0                = Just (lx)
			| length lx /= length ly        = Nothing
			| lx == ly                      = Just (lx)
			-- One of the elements is a singleton with n dimensions.
			| lx == replicate (length lx) 1 = Just (ly)
			| ly == replicate (length ly) 1 = Just (lx)
			-- Can't combine otherwise - FAR too complex!
			| otherwise                     = Nothing

pyGetResultSize3 x y z = case (l0, l1, l2) of
		-- All inputs of indeterminate length currently.  If any one of them
		-- is not a constant, we no longer exactly know the dimensions.
		([-1], [-1], [-1]) ->
			if p0 == [] || p1 == [] || p2 == []
			then Just []
			else if p0 == p1 && p1 == p2
				-- All dims are the same for constants.
				then Just p1
				else Nothing
		-- One known length.
		([-1], [-1],  z' ) -> Just z'
		([-1],  y' , [-1]) -> Just y'
		( x' , [-1], [-1]) -> Just x'
		-- Two known lengths.
		([-1],  y' ,  z' ) -> if y' == z' then Just z' else Nothing
		( x' , [-1],  z' ) -> if z' == x' then Just x' else Nothing
		( x' ,  y' , [-1]) -> if x' == y' then Just y' else Nothing
		-- All known lengths.
		( x' ,  y' ,  z' ) -> if x' == y' && y' == z' then Just y' else Nothing
	where
		p0 = pyLength x
		p1 = pyLength y
		p2 = pyLength z
		-- "-1" code covers "0 length (unknown)" and "all 1 (constant)" cases.
		rep' n = if n == replicate (length n) 1 then [-1] else n
		l0 = rep' p0
		l1 = rep' p1
		l2 = rep' p2

instance
		(PYExecute dom id,
		 Show (PYNative dom id)) =>
		Show (PYExpr dom id)
	where
		show expr = "Data: " ++ show (pyExecute expr) ++ "\nLength: " ++ show (pyLength expr)

{-
 - PYStorable converts between Haskell types and native types.  Now "PYArray".
 -}
class (Unbox dom) => PYStorable dom id where
	pySetList    :: [] Int -> [] dom -> PYStorage dom id
	--pySetArray1D :: Vector dom -> PYStorage dom id
	pySetVector  :: [] Int -> Vector dom -> PYStorage dom id
	--pySetArray2D :: UArray (Int, Int) dom -> PYStorage dom id
	pySetNum     :: [] Int -> dom -> PYStorage dom id
	
	-- Default implementations.  You now only need "pySetList".
	pySetVector l v = pySetList l (V.toList v)
	pySetNum c n    = pySetList c $ replicate (foldr (*) 1 c) n

{-
 - PYDoRun converts to and from results.
 -}
class
		PYExecute dom id
	where
		pyExecute :: PYExpr dom id -> PYNative dom id
--		pyDoGet :: PYNative dom id -> [] dom --PYArray dom

{-
 - PYReturn converts to and from results.
 -}
class
		(Unbox dom) =>
		PYReturn dom id
	where
		pyReturnDims   :: PYNative dom id -> [] Int
		pyReturnList   :: PYNative dom id -> [] dom
		pyReturnVector :: PYNative dom id -> Vector dom
		
		pyReturnVector = V.fromList . pyReturnList

pyExpressibleList :: (PYStorable dom id) => [] dom -> PYExpr dom id
pyExpressibleList x = PYExpr (pySetList [length x] x) [length x]

--pyExpressibleArray1 :: (PYStorable dom id, Unbox dom) => Vector dom -> PYExpr dom id
--pyExpressibleArray1 x = gb' x
--	where
--		gb' (bounds -> ( 0     ,  x'    )) = PYExpr (pySetArray1D x) x' []
--		gb' _                              = error "Not zero-indexed array."

pyExpressibleVector :: (PYStorable dom id, Unbox dom) => Vector dom -> PYExpr dom id
pyExpressibleVector x = PYExpr (pySetVector [V.length x] x) [V.length x]

--pyExpressibleArray2 :: (PYStorable dom id, Unbox dom) => UArray (Int, Int) dom -> PYExpr dom id
--pyExpressibleArray2 x = gb' x
--	where
--		gb' (bounds -> ((0, 0), (x', y'))) = PYExpr (pySetArray2D x) x' [y']
--		gb' _                              = error "Not zero-indexed array."

pyExpressibleReplicate :: (PYStorable dom id) => Replicate dom -> PYExpr dom id
pyExpressibleReplicate (Replicate x y) = pyExpressibleNumber x y

pyConst n = PYExpr (pySetNum [] n) []

pyExpressibleNumber :: (PYStorable dom id) => [] Int -> dom -> PYExpr dom id
pyExpressibleNumber x y = PYExpr (pySetNum x y) x

{-
 - PYExpressible defines the type class for converting from a Haskell type to a
 - PYExpr.  Uses "PYStorable" to get the native data, but also does all other
 - information that may be stored (for example data size).
 -}
class
		PYExpressible arr dom id
	where
		pySet :: arr -> PYExpr dom id

instance
		(PYStorable dom1 id1, dom0 ~ dom1) =>
		PYExpressible ([] ([] ([] dom0))) dom1 id1
	where
		pySet x =
			let
				lens = [length x, length . head $ x, length . head . head $ x]
			in
				PYExpr (pySetList lens . concat . concat $ x) lens

instance
		(PYStorable dom1 id1, dom0 ~ dom1) =>
		PYExpressible ([] ([] dom0)) dom1 id1
	where
		pySet x =
			let
				lens = [length x, length . head $ x]
			in
				PYExpr (pySetList lens . concat $ x) lens

instance
		(PYStorable dom1 id1, dom0 ~ dom1) =>
		PYExpressible ([] dom0) dom1 id1
	where
		pySet x =
			let
				lens = [length x]
			in
				PYExpr (pySetList lens x) lens

instance
		(dom0 ~ dom1, id0 ~ id1) =>
		PYExpressible (PYExpr dom0 id0) dom1 id1
	where
		pySet = id

instance
		(PYStorable dom1 id1, dom0 ~ dom1) =>
		PYExpressible (Replicate dom0) dom1 id1
	where
		pySet = pyExpressibleReplicate

--instance
--		(PYStorable dom1 id1, dom0 ~ dom1) =>
--		PYExpressible (Replicate dom0) dom1 id1
--	where
--		pySet (Replicate n x) = PYExpr (pySetNum n x) n

instance
		(PYStorable dom1 id1, dom0 ~ dom1, Unbox dom1, Unbox (Vector dom1), Unbox (Vector (Vector dom1))) =>
		PYExpressible (Vector (Vector (Vector dom0))) dom1 id1
	where
		pySet x =
			let
				lens = [V.length x, V.length . V.head $ x, V.length . V.head . V.head $ x]
			in
				PYExpr (pySetVector lens . V.concat . V.toList . V.concat . V.toList $ x) lens

instance
		(PYStorable dom1 id1, dom0 ~ dom1, Unbox dom1, Unbox (Vector dom1)) =>
		PYExpressible (Vector (Vector dom0)) dom1 id1
	where
		pySet x =
			let
				lens = [V.length x, V.length . V.head $ x]
			in
				PYExpr (pySetVector lens . V.concat . V.toList $ x) lens

instance
		(PYStorable dom1 id1, dom0 ~ dom1, Unbox dom1) =>
		PYExpressible (Vector dom0) dom1 id1
	where
		pySet = pyExpressibleVector

-- These are now the only place that "pyRun" and "pyGet" need to be defined.
-- This minor reorganisation has removed "PYExpressible"'s reliance on "PYDoRun"
-- by abstracting the actual reliance out.  This also simplifies several
-- instance declarations by again removing the requirement to have them rely on
-- "PYDoRun" for no apparent reason, and reduces duplicate code when creating
-- new targets as "pyRun" and "pyDoRun" were almost identical for "PYNative".  I
-- just proved the point about new targets by adding "PYArray" (now the native
-- Haskell type, but still added).
pyRun :: (PYExecute dom id, PYExpressible arr dom id) => arr -> PYNative dom id
pyRun = pyExecute . pySet

class
		PYGet arr dom id
	where
		pyGet :: PYNative dom id -> arr

instance
		(dom0 ~ dom1, id0 ~ id1, PYExpressible (PYNative dom1 id1) dom1 id1) =>
		PYGet (PYExpr dom0 id0) dom1 id1
	where
		pyGet = pySet

instance
		(dom0 ~ dom1, id0 ~ id1) =>
		PYGet (PYNative dom0 id0) dom1 id1
	where
		pyGet = id

instance
		(dom0 ~ dom1, PYReturn dom1 id) =>
		PYGet ([] dom0) dom1 id
	where
		pyGet = pyReturnList

{-
-- Doesn't exist.
instance
		(PYReturn dom id) =>
		PYGet (Replicate dom) dom id
	where
		pyGet = pyReturnList
-}

instance
		(dom0 ~ dom1, PYReturn dom1 id) =>
		PYGet (Vector dom0) dom1 id
	where
		pyGet = pyReturnVector

class PYDim dom id where
	pyDim :: Int -> [] Int -> PYExpr dom id
