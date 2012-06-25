{-# LANGUAGE MagicHash         #-}

module PEGGY.FloatBound where

-- This module implements "Bounded" for "Float".  This means working at the
-- lowest unboxed type level.

import GHC.Prim
import GHC.Types
import GHC.Int

--import Prelude (Bounded)

-- Use the binary (unboxed) representation of the IEEE 754 -Infinity and
-- Infinity and coerce those bit patterns to unboxed floats.  Then construct a
-- boxed float for use in normal type-safe code.  The hex representations are:
-- 0xFF800000
-- 0x7F800000
instance Bounded Float where
	minBound = F# ((unsafeCoerce# (0xFF800000#)) :: Float#)
	maxBound = F# ((unsafeCoerce# (0x7F800000#)) :: Float#)

--instance Bounded Double where
--	minBound = D# ((unsafeCoerce# (0xFFF0000000000000# :: Int64#)) :: Double#)
--	maxBound = D# ((unsafeCoerce# (0x7FF0000000000000# :: Int64#)) :: Double#)

--instance Bounded Float where
--	minBound = F# ((unsafeCoerce# (0xFF800000# :: Int32#)) :: Float#)
--	maxBound = F# ((unsafeCoerce# (0x7F800000# :: Int32#)) :: Float#)
--
--instance Bounded Double where
--	minBound = D# ((unsafeCoerce# (0xFFF0000000000000# :: Int64#)) :: Double#)
--	maxBound = D# ((unsafeCoerce# (0x7FF0000000000000# :: Int64#)) :: Double#)
