{-# LANGUAGE NoMonomorphismRestriction #-}

import PEGGY
import PEGGY.Obsidian
--import PEGGY.Haskell
--import PEGGY.Accelerator
--import PEGGY.Repa

import System.Random

--import Criterion.Main

randomXs, randomYs, randomZs :: Int -> [Float]
randomXs x = randomRs (0, fromIntegral x) (mkStdGen (x + 42))
randomYs y = randomRs (0, fromIntegral y) (mkStdGen (y * 42))
randomZs z = randomRs (0, fromIntegral z) (mkStdGen (z - 42))

randomCharges :: Float -> [Float]
randomCharges c = randomRs (0, c) (mkStdGen (round c))

--gridX, gridY :: Int -> Int -> [[Float]]
--gridX x y z = replicate z $ replicate y [0 .. fromIntegral x - 1]
--gridY x y z = replicate z $ [replicate x i | i <- [0 .. fromIntegral y - 1]]
-- Fake 3D for now (but easy to extend).
--gridZ x y z = [replicate y $ replicate x i | i <- [0 .. fromIntegral z - 1]]

--gridGen :: Int -> Int -> Int -> (PYExpr Float TargetHaskell, PYExpr Float TargetHaskell, PYExpr Float TargetHaskell)
--gridGen x y z = (pySet (gridX x y z), pySet (gridY x y z), pySet (gridZ x y z))

maxX = 400
maxY = 400
maxZ = 400 -- SHOULD BE 1 FOR NOW!
maxCharge = 30.0
atomCount = 300

{-
theGrid = gridGen maxX maxY maxZ
randomX = randomXs maxX
randomY = randomYs maxY
randomZ = randomZs maxZ
randomC = randomCharges maxCharge
-}

data Atom = Atom {
	aX :: Float,
	aY :: Float,
	aZ :: Float,
	aC :: Float}
	deriving (Show, Eq)

{-
Create a list of atom locations.
-}
atoms rx ry rz rc n = [Atom (rz !! i) (ry !! i) (rz !! i) (rc !! i) | i <- [0 .. n - 1]]

--theAtoms = atoms randomX randomY randomZ randomC atomCount

doOneAtom (gridX, gridY, gridZ) (Atom x y z c) = pyConst c / pySqrt distance
	where
		distance = (diff gridX x * diff gridX x) + (diff gridY y * diff gridY y) + (diff gridZ z * diff gridZ z)
		diff arr pos = arr - pyConst pos
		dims = pyLength gridX -- Should be the same as gridY and gridZ.

--chargeSpace :: Int -> PYExpr Float id
chargeSpace n = chargeSpace' maxX maxY maxZ maxCharge n

chargeSpace' x y z c n = foldl pyAdd 0 (map doOneAtom' atoms')
	where
		dims = [y, x]
		doOneAtom' = doOneAtom (pyDim 0 dims, pyDim 1 dims, pyDim 2 dims) --pyConst 0) -- (gridGen x y z)
		-- Generate a random set of atoms spread all over the given grid space
		-- and with random charges.  Don't ask me about the random distribution.
		atoms' = atoms (randomXs x) (randomYs y) (randomZs z) (randomCharges c) n
--foldl1 pyAdd (map doOneAtom theAtoms)

--doBenchmark = bench "Hsakell" $ nf chargeSpace atomCount

--runRepa :: PYExpr Float TargetRepa -> [Float]
--runRepa = pyGet . pyRun
{-
runDX9 :: PYExpr Float TargetDX9 -> [Float]
runDX9 = pyGet . pyRun

runHaskell :: PYExpr Float TargetHaskell -> [Float]
runHaskell = pyGet . pyRun

fuzzSame = all (\ (a, b) -> abs (a - b) < 0.001) $ zip (runDX9 $ chargeSpace atomCount) (runHaskell $ chargeSpace atomCount)

main = putStrLn $ show $ fuzzSame
-}
