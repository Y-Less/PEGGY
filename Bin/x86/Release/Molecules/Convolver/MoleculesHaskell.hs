module MoleculesHaskell where

import PEGGY
import PEGGY.Obsidian

import Foreign.Ptr
import Foreign.Marshal.Array

foreign export stdcall generateMolecules :: Ptr Float -> Int -> Int -> Int -> IO Int

data Atom = Atom {
	aX :: Float,
	aY :: Float,
	aZ :: Float,
	aC :: Float}
	deriving (Show, Eq)

--doOneAtom (gridX, gridY, gridZ) (Atom x y z c) = pyConst c / pySqrt distance
--	where
--		distance = (diff gridX x * diff gridX x) + (diff gridY y * diff gridY y) + (diff gridZ z * diff gridZ z)
--		diff arr pos = arr - pyConst pos

--chargeSpace atoms dims = sum (map doOneAtom' atoms)
--	where
--		doOneAtom' = doOneAtom (pyDim 0 dims, pyDim 1 dims, pyDim 2 dims)

chargeSpace atoms gridPoint = sum $ map oneCharge atoms
	where
		oneCharge a = charge a / distance gridPoint a
		charge (Atom _ _ _ c) = pyConst c
		distance (gridX, gridY, gridZ) (Atom x y z _) = (diff gridX x * diff gridX x) + (diff gridY y * diff gridY y) + (diff gridZ z * diff gridZ z)
		diff arr pos = arr - pyConst pos

generateMolecules atoms'' count height width = do
	atoms' <- peekArray (count * 4) atoms''
	-- Unfold the atoms
	let
		atoms = splitAtoms atoms'
		-- We only do one slice of the input data.
		dims = [height, width]
		code = chargeSpace atoms (pyDim 0 dims, pyDim 1 dims, pyDim 2 dims) --dims
	cc  <- generateObsidianCode code
	--putStrLn (setKernelName cc "DoAtoms(float * result0) { //")
	compileCUDA (setKernelName cc "DoAtoms(float * result0) { //") "MoleculesHaskell"

splitAtoms :: [Float] -> [Atom]
splitAtoms (x : y : z : c : rest) = Atom x y z c : splitAtoms rest
splitAtoms _ = []
