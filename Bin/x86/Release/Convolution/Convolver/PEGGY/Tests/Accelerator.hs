{-# LANGUAGE NoMonomorphismRestriction #-}

-- Unit tests for the Accelerator PEGGY target (even though I now hate it and
-- wish I had spent more time on the CUDA target - by which I mean "any" time).

module PEGGY.Tests.Accelerator where

import PEGGY
import PEGGY.Accelerator

import Data.Maybe (fromJust, catMaybes)
import Data.List (intercalate)
import Control.DeepSeq
import Control.Monad (zipWithM, mapAndUnzipM)

import Criterion.Main
import Numeric.IEEE

-- Test every function multiple ways.

--type PYTest dom id = (String, PYExpr dom id, [dom])
data PYTest dom id = P (dom -> PYExpr dom id, dom -> [dom]) | F (dom -> PYExpr dom id, dom -> [dom])

type ACCTest dom = PYTest dom TargetDX9

type PYTests dom id = [(String, [PYTest dom id])]

type ACCTests dom = PYTests dom TargetDX9

glData a = [50 + a .. 149 + a]

gfData :: Float -> PYExpr Float TargetDX9
gfData a = pySet (glData a)

mg f a = map f (glData a)

-- No error handling yet...
--floatTests :: [ACCTest Float]
floatTests :: ACCTests Float
floatTests = [
	-- Test "pySet" repeatedly.
{-	("Declare", [
		P (pySet [0 .. 9], [0 .. 9]),
		P (pySet [10 .. 19], [10 .. 19]),
		P (pySet [[10 .. 19], [20 .. 29]], [10 .. 19] ++ [20 .. 29]),
		P (pySet [10 .. 19], [10 .. 19]),
		P (pySet (Replicate [10] 42), replicate 10 42)
	]),
	-- Test both "+" and our unusual "Num" instance by adding loads of different
	-- types of data together directly.
	("Add", [
		P (gfData + gfData, mg (* 2)),
		P (gfData + 2, [52 .. 151]),
		P (5 + gfData, [55 .. 154]),
		P (pyAdd gfData gfData, mg (* 2)),
		P (pyAdd gfData (pyConst 2), mg (+ 2)),
		P (pyAdd (pyConst 5) gfData, mg (+ 5)),
		P (pyAdd gfData [300 .. 399], map (* 2) [175 .. 274]),
		P (pyAdd [300 .. 399] gfData, map (* 2) [175 .. 274]),
		P (pyAdd gfData (Replicate [100] 200), [250 .. 349]),
		P (pyAdd (Replicate [100] 200) gfData, [250 .. 349])
	]),
	--
	("Sub", [
		P (gfData - gfData, replicate 100 0),
		P (gfData - 3, [47 .. 146]),
		P (3 - gfData, reverse [-146 .. -47])
	]),
	--
	("Mul", [
		P (gfData * gfData, mg (** 2)),
		P (gfData * 3, mg (* 3))
	]),
	--
	("Div", [
		P (50 / gfData, mg ((/) 50)),
		P (gfData / 50, mg (/ 50))
	]),
	--
	("Min", [
		P (pyMin (pyConst 10) [0 .. 99], [0 .. 9] ++ replicate 90 10)
	]),
	("Max", [
		P (pyMax (pyConst 10) [0 .. 99], replicate 10 10 ++ [10 .. 99])
	]),
	--
	("Abs", [
		P (pyAbs [-50 .. 49], reverse [1 .. 50] ++ [0 .. 49])
	]),
	("Negate", [
		P (pyNegate [-50 .. 49], reverse [-49 .. 50])
	]),
	("Sqrt", [
		P (pySqrt gfData, mg sqrt)
	]),
	--
	("If Then Else", [
		P (pyIfThenElse (pyEQ gfData gfData) gfData gfData, glData),
		P (pyIfThenElse (pyEQ gfData gfData) (pyConst 0) (pyConst 1), (replicate 100 0)),
		P (pyIfThenElse (pyNEQ gfData gfData) (pyConst 0) (pyConst 1), (replicate 100 1)),
		--
		-- These are essentially "max".
		P (pyIfThenElse (pyGT [-50 .. 49 :: Float] (pyConst 0)) [-50 .. 49] (pyConst 0), replicate 51 0 ++ [1 .. 49]),
		P (pyIfThenElse (pyGTE [-50 .. 49 :: Float] (pyConst 0)) [-50 .. 49] (pyConst 1), replicate 50 1 ++ [0 .. 49]),
		-- These are essentially "min".
		P (pyIfThenElse (pyLT [-50 .. 49 :: Float] (pyConst 0)) [-50 .. 49] (pyConst 0), [-50 .. -1] ++ replicate 50 0),
		P (pyIfThenElse (pyLTE [-50 .. 49 :: Float] (pyConst 0)) [-50 .. 49] (pyConst 1), [-50 .. 0] ++ replicate 49 1)
	]),-}
	--P (pyIfThenElse (gfData == 0) gfData gfData, glData),
	--
	("Pow", [
		P (\ a -> pyPow a (Replicate [10] 3),             replicate 10 . (** 3)),
		P (\ a -> pyPow a [4, 4, 4, 4, 4, 4, 4, 4, 4, 4], replicate 10 . (** 4)),
		P (\ a -> pyPow a [0 .. 5],                       \ a -> map ((**) a) [0 .. 5]),
		P (\ a -> pyPow a (Replicate [10] 4),             replicate 10 . (** 4)),
		P (\ a -> pyPow 2 [0 + a .. 10 + a],              \ a -> map ((**) 2) [0 + a .. 10 + a])
	]),
	--
	("Log", [
		P (\ a -> pyLog (a + 1) [0 .. 5],         \ a -> map (logBase (a + 1)) [0 .. 5]),
		P (\ a -> pyLog 2 [0 + a .. 5 + a], \ a -> map (logBase 2) [0 + a .. 5 + a])
	]),
	--P (pyIfThenElse (pyNeq gfData gfData) (replicate 100 0) (replicate 100 1), (replicate 100 1)),
--	(gfData + Replicate 100 200, [250 .. 349]),
--	(gfData + [300 .. 399], map (* 2) [175 .. 274]),
--	(Replicate 100 200 + gfData, [250 .. 349]),
--	("Add 7", [300 .. 399] + gfData, map (* 2) [175 .. 274])]
	("FAIL", [
		-- First time I've found a use for "const".
		F (gfData, \ a -> [0 + a .. 99 + a]),
		F (gfData, const [])
	])]

{-zipWithMaybe _ [] _ = []
zipWithMaybe _ _ [] = []
zipWithMaybe f (x : xs) (y : ys) = case f x y of
	Just r  -> r ++ zipWithMaybe f xs ys
	Nothing -> zipWithMaybe f xs ys-}

zipWithMaybe f a b = catMaybes $ zipWith f a b

class FuzzyEq dom where
	feq :: dom -> dom -> Bool

instance FuzzyEq Float where
	--feq a b = sameSignificandBits a b > 21
	-- Fuzzy float compare.  There are better ways of doing this, I read a blog
	-- post on it recently, but can't remember where.
	--feq a b = diff < 0.00001 && diff > -0.00001
	--	where
	--		diff = a - b
	-- OK, do it by percentage.
	feq a b = a == b || (diff < sq && diff > -sq)
		where
			diff = (a - b) * (a - b)
			sq = a * a * 0.00001 * 0.00001

instance FuzzyEq Double where
	--feq a b = sameSignificandBits a b > 42
	--feq a b = diff > -0.000001 && diff < 0.000001
	--	where
	--		diff = a - b
	feq a b = a == b || diff <= sq && diff >= -sq
		where
			diff = (a - b) * (a - b)
			sq = a * a * 0.0000001 * 0.0000001

instance FuzzyEq Int where
	feq = (==)

instance (FuzzyEq dom) => FuzzyEq [dom] where
	feq a b = length a == length b && and (zipWith feq a b)

maxSingleTests = 10

runOneTest ::
	(PYReturn dom id, PYExecute dom id, FuzzyEq dom, Show dom) =>
	String -> PYTest dom id -> dom -> IO (Maybe String)
runOneTest name input value = do
		putStrLn $ "Running test " ++ name ++ " (" ++ show value ++ ")"
		let
			-- Print incorrect values.
			disp a b = if a `feq` b == pf input then Nothing else Just (show a ++ " ?= " ++ show b)
			-- Get the result.
			run = pyReturnList (pyExecute test)
			res = case run `feq` result  == pf input of
				True  -> Nothing
				-- Print all the results.
				False -> Just (name ++ " (" ++ show value ++ ")" ++ ": " ++ intercalate ", " (zipWithMaybe disp run result))
		putStrLn $ "Done: " ++ show res
		return res
	where
		-- Get the input data.
		(test', result') = fromPF input
		test = test' value
		result = result' value
		fromPF (P d) = d
		fromPF (F d) = d
		-- Should the test pass or fail?
		pf (P _) = True
		pf (F _) = False

runTest ::
	(PYReturn dom id, PYExecute dom id, FuzzyEq dom, Show dom, Enum dom, Num dom) =>
	String -> PYTest dom id -> IO (Maybe String)
runTest name input = do
	ret <- mapM (runOneTest name input) [0 .. maxSingleTests]
	let
		cat = intercalate "\n" (catMaybes ret)
	case cat of
		[] -> return (Nothing)
		otherwise -> return (Just cat)

makeBench ::
	(PYReturn dom id, PYExecute dom id, FuzzyEq dom, Show dom, Enum dom, Num dom) =>
	String -> PYTest dom id -> [Benchmark]
makeBench name input = map (\ x -> bench (name ++ " (" ++ show x ++ ")") (b x)) [0 .. maxSingleTests]
	where
		(test, _) = fromPF input
		fromPF (P d) = d
		fromPF (F d) = d
		-- I hope this is enough for Criterion.
		--test' _ = test
		--test' x = (test, x)
		b x = whnf test x

runGroup ::
	(PYReturn dom id, PYExecute dom id, FuzzyEq dom, Show dom, Enum dom, Num dom) =>
	(String, [PYTest dom id]) -> IO (Benchmark, [String])
runGroup (groupname, group) = do
		-- Loop through every element in this group and assign a number.
		results <- zipWithM (\ x t -> runTest (mk x) t) [0 .. ] group
		let
			fails = catMaybes results
		return (benches, fails)
	where
		mk n = groupname ++ " " ++ show n
		benches = bgroup groupname blist
		blist = concat $ zipWith (\ x t -> makeBench (mk x) t) [0 .. ] group

runTests ::
	(PYReturn dom id, PYExecute dom id, FuzzyEq dom, Show dom, Enum dom, Num dom) =>
	PYTests dom id -> IO ()
runTests tests = do
		putStrLn "\n*** Running all tests...\n"
		(bms, res) <- results
		let
			res' = concat res
		case res' of
			[]        -> putStrLn "\n*** All tests passed\n"
			otherwise -> putStrLn $ "\n*** Failing tests:" ++ concat (map ((++) "\n\n - ") res') ++ "\n"
		putStrLn "\n*** Timing all tests...\n"
		defaultMain bms
	where
		-- Get a list of all failing tests.
		--just' Nothing  = False
		--just' (Just _) = True
		results = mapAndUnzipM runGroup tests
-- >>= return . catMaybes --filter just'
