{-# LANGUAGE MultiParamTypeClasses       #-}
{-# LANGUAGE UndecidableInstances        #-}
{-# LANGUAGE FlexibleInstances           #-}
{-# LANGUAGE FlexibleContexts            #-}
{-# LANGUAGE TypeFamilies                #-}

module PEGGY.Instances where

import PEGGY.Types
import PEGGY.Functions

import Data.Vector.Unboxed as V

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--    PY1
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

instance
		(id0 ~ id9, dom0 ~ dom9) =>
		PY0 (PYExpr dom0 id0) dom9 id9
	where
		pyLift0 x = x

instance
		(id0 ~ id9, dom0 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9) =>
		PY0 (PYNative dom0 id0) dom9 id9
	where
		pyLift0 x = (pySet x)

instance
		(id0 ~ id9, dom0 ~ dom9,
		 PYStorable dom9 id9) =>
		PY0 ([] dom0) dom9 id9
	where
		pyLift0 x = (pyExpressibleList x)

instance
		(id0 ~ id9, dom0 ~ dom9,
		 PYStorable dom9 id9) =>
		PY0 (Replicate dom0) dom9 id9
	where
		pyLift0 x = (pyExpressibleReplicate x)

instance
		(id0 ~ id9, dom0 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		PY0 (Vector dom0) dom9 id9
	where
		pyLift0 x = (pyExpressibleVector x)

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--    PY1
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

instance
		(id0 ~ id9, dom0 ~ dom9) =>
		 --
		PY1 (PYExpr dom0 id0) dom9 id9
	where
		pyLift1 func x = func x

instance
		(id0 ~ id9, dom0 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9) =>
		 --
		PY1 (PYNative  dom0 id0) dom9 id9
	where
		pyLift1 func x = func (pySet x)

instance
		(id0 ~ id9, dom0 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY1 ([] dom0) dom9 id9
	where
		pyLift1 func x = func (pyExpressibleList x)

instance
		(id0 ~ id9, dom0 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY1 (Replicate dom0) dom9 id9
	where
		pyLift1 func x = func (pyExpressibleReplicate x)

instance
		(id0 ~ id9, dom0 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY1 (Vector dom0) dom9 id9
	where
		pyLift1 func x = func (pyExpressibleVector x)

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--    PY2
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

{- 
 - This MASS of instance declarations defines every combination of things that
 - can be instances of PYExpressible (through PY2).  In this way, the type
 - inference can very easily figure out the exact return types when typed and
 - untyped elements are used together, even when one is just a list for example:
 - 
 -  (.+.) :: (PYFunc2 dom id, PY2 c0 c1 dom id) => c0 -> c1 -> PYExpr dom id
 -  (.+.) = pyLift2 pyAdd
 -  
 -  m = (pySet [0 .. 1000]) :: PYExpr Float TargetHaskell
 -  q = m .+. ([0 .. 1000] :: [Float])
 - 
 - WITHOUT these instances the type of the above expression would be:
 - 
 -  (PY2 (PYExpr Float TargetHaskell) [Float] dom id, PYFunc2 dom id) =>
 -  PYExpr dom id
 - 
 - WITH these instances the type very nicely becomes:
 - 
 -  PYExpr Float TargetHaskell
 - 
 - And we only need these instances defined once ever.
 - 
 -}

-- PYExpr
instance
		(id0 ~ id9, id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9) =>
		 --
		PY2 (PYExpr dom0 id0) (PYExpr dom1 id1) dom9 id9
	where
		pyLift2 func x y = func x y

instance
		(id0 ~ id9, id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9) =>
		 --
		PY2 (PYExpr dom0 id0) (PYNative dom1 id1) dom9 id9
	where
		pyLift2 func x y = func x (pySet y)

instance
		(id0 ~ id9,            dom0 ~ dom9, dom1 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY2 (PYExpr dom0 id0) ([] dom1)         dom9 id9
	where
		pyLift2 func x y = func x (pyExpressibleList y)

instance
		(id0 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, 
		PYStorable dom9 id9) =>
		 --
		PY2 (PYExpr dom0 id0) (Replicate dom1)  dom9 id9
	where
		pyLift2 func x y = func x (pyExpressibleReplicate y)

instance
		(id0 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, 
		PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY2 (PYExpr dom0 id0) (Vector dom1)  dom9 id9
	where
		pyLift2 func x y = func x (pyExpressibleVector y)

-- PYNative
instance
		(id0 ~ id9, id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9) =>
		 --
		PY2 (PYNative dom0 id0) (PYExpr dom1 id1) dom9 id9
	where
		pyLift2 func x y = func (pySet x) y

instance
		(id0 ~ id9, id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9) =>
		 --
		PY2 (PYNative dom0 id0) (PYNative dom1 id1) dom9 id9
	where
		pyLift2 func x y = func (pySet x) (pySet y)

instance
		(id0 ~ id9,            dom0 ~ dom9, dom1 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY2 (PYNative dom0 id0) ([] dom1)         dom9 id9
	where
		pyLift2 func x y = func (pySet x) (pyExpressibleList y)

instance
		(id0 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, 
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY2 (PYNative dom0 id0) (Replicate dom1)  dom9 id9
	where
		pyLift2 func x y = func (pySet x) (pyExpressibleReplicate y)

instance
		(id0 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, 
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY2 (PYNative dom0 id0) (Vector dom1)  dom9 id9
	where
		pyLift2 func x y = func (pySet x) (pyExpressibleVector y)

-- []
instance
		(           id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY2 ([] dom0) (PYExpr dom1 id1) dom9 id9
	where
		pyLift2 func x y = func (pyExpressibleList x) y

instance
		(           id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY2 ([] dom0) (PYNative dom1 id1) dom9 id9
	where
		pyLift2 func x y = func (pyExpressibleList x) (pySet y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY2 ([] dom0) ([] dom1)         dom9 id9
	where
		pyLift2 func x y = func (pyExpressibleList x) (pyExpressibleList y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9, 
		PYStorable dom9 id9) =>
		 --
		PY2 ([] dom0) (Replicate dom1)  dom9 id9
	where
		pyLift2 func x y = func (pyExpressibleList x) (pyExpressibleReplicate y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9, 
		PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY2 ([] dom0) (Vector dom1)  dom9 id9
	where
		pyLift2 func x y = func (pyExpressibleList x) (pyExpressibleVector y)

-- Replicate
instance
		(           id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY2 (Replicate dom0) (PYExpr dom1 id1) dom9 id9
	where
		pyLift2 func x y = func (pyExpressibleReplicate x) y

instance
		(           id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY2 (Replicate dom0) (PYNative dom1 id1) dom9 id9
	where
		pyLift2 func x y = func (pyExpressibleReplicate x) (pySet y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY2 (Replicate dom0) ([] dom1)         dom9 id9
	where
		pyLift2 func x y = func (pyExpressibleReplicate x) (pyExpressibleList y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9, 
		PYStorable dom9 id9) =>
		 --
		PY2 (Replicate dom0) (Replicate dom1)  dom9 id9
	where
		pyLift2 func x y = func (pyExpressibleReplicate x) (pyExpressibleReplicate y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9, 
		PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY2 (Replicate dom0) (Vector dom1)  dom9 id9
	where
		pyLift2 func x y = func (pyExpressibleReplicate x) (pyExpressibleVector y)

-- Vector
instance
		(           id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY2 (Vector dom0) (PYExpr dom1 id1) dom9 id9
	where
		pyLift2 func x y = func (pyExpressibleVector x) y

instance
		(           id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY2 (Vector dom0) (PYNative dom1 id1) dom9 id9
	where
		pyLift2 func x y = func (pyExpressibleVector x) (pySet y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY2 (Vector dom0) ([] dom1)         dom9 id9
	where
		pyLift2 func x y = func (pyExpressibleVector x) (pyExpressibleList y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9, 
		PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY2 (Vector dom0) (Replicate dom1)  dom9 id9
	where
		pyLift2 func x y = func (pyExpressibleVector x) (pyExpressibleReplicate y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9, 
		PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY2 (Vector dom0) (Vector dom1)  dom9 id9
	where
		pyLift2 func x y = func (pyExpressibleVector x) (pyExpressibleVector y)

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--    PY3
--------------------------------------------------------------------------------
-- There are four types: PYExpr, PYNative, [] and Replicate, thus there are
-- 4 * 4 * 4 = 64 combinations for PY3 - that's a lot of repetitive code...
-- Fortunately it turns out to be not too hard to write.  But now adding a new
-- encoding type will increase the total from 4^3=64 to 5^3=125 (not QUITE
-- double).  There is now a new type - Vector!  Woo!  Whose idea was this?  So
-- now we will end up with 216 if I add another new type...
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
--    PYExpr, X, X
--------------------------------------------------------------------------------
-- PYExpr, PYExpr, X
instance
		(id0 ~ id9, id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9) =>
		 --
		PY3 (PYExpr dom0 id0) (PYExpr dom1 id1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func x y z

instance
		(id0 ~ id9, id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9) =>
		 --
		PY3 (PYExpr dom0 id0) (PYExpr dom1 id1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func x y (pySet z)

instance
		(id0 ~ id9, id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 (PYExpr dom0 id0) (PYExpr dom1 id1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func x y (pyExpressibleList z)

instance
		(id0 ~ id9, id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 (PYExpr dom0 id0) (PYExpr dom1 id1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func x y (pyExpressibleReplicate z)

instance
		(id0 ~ id9, id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYExpr dom0 id0) (PYExpr dom1 id1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func x y (pyExpressibleVector z)

-- PYExpr, PYNative, X
instance
		(id0 ~ id9, id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9) =>
		 --
		PY3 (PYExpr dom0 id0) (PYNative dom1 id1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func x (pySet y) z

instance
		(id0 ~ id9, id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9) =>
		 --
		PY3 (PYExpr dom0 id0) (PYNative dom1 id1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func x (pySet y) (pySet z)

instance
		(id0 ~ id9, id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (PYExpr dom0 id0) (PYNative dom1 id1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func x (pySet y) (pyExpressibleList z)

instance
		(id0 ~ id9, id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (PYExpr dom0 id0) (PYNative dom1 id1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func x (pySet y) (pyExpressibleReplicate z)

instance
		(id0 ~ id9, id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYExpr dom0 id0) (PYNative dom1 id1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func x (pySet y) (pyExpressibleVector z)

-- PYExpr, [], X
instance
		(id0 ~ id9,            id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 (PYExpr dom0 id0) ([] dom1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func x (pyExpressibleList y) z

instance
		(id0 ~ id9,            id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (PYExpr dom0 id0) ([] dom1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func x (pyExpressibleList y) (pySet z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 (PYExpr dom0 id0) ([] dom1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func x (pyExpressibleList y) (pyExpressibleList z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 (PYExpr dom0 id0) ([] dom1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func x (pyExpressibleList y) (pyExpressibleReplicate z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYExpr dom0 id0) ([] dom1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func x (pyExpressibleList y) (pyExpressibleVector z)

-- PYExpr, Replicate, X
instance
		(id0 ~ id9,            id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 (PYExpr dom0 id0) (Replicate dom1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func x (pyExpressibleReplicate y) z

instance
		(id0 ~ id9,            id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (PYExpr dom0 id0) (Replicate dom1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func x (pyExpressibleReplicate y) (pySet z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 (PYExpr dom0 id0) (Replicate dom1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func x (pyExpressibleReplicate y) (pyExpressibleList z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 (PYExpr dom0 id0) (Replicate dom1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func x (pyExpressibleReplicate y) (pyExpressibleReplicate z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYExpr dom0 id0) (Replicate dom1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func x (pyExpressibleReplicate y) (pyExpressibleVector z)

-- PYExpr, Vector, X
instance
		(id0 ~ id9,            id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYExpr dom0 id0) (Vector dom1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func x (pyExpressibleVector y) z

instance
		(id0 ~ id9,            id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYExpr dom0 id0) (Vector dom1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func x (pyExpressibleVector y) (pySet z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYExpr dom0 id0) (Vector dom1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func x (pyExpressibleVector y) (pyExpressibleList z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYExpr dom0 id0) (Vector dom1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func x (pyExpressibleVector y) (pyExpressibleReplicate z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYExpr dom0 id0) (Vector dom1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func x (pyExpressibleVector y) (pyExpressibleVector z)

--------------------------------------------------------------------------------
--    PYNative, X, X
--------------------------------------------------------------------------------
-- PYNative, PYExpr, X
instance
		(id0 ~ id9, id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9) =>
		 --
		PY3 (PYNative dom0 id0) (PYExpr dom1 id1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) y z

instance
		(id0 ~ id9, id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9) =>
		 --
		PY3 (PYNative dom0 id0) (PYExpr dom1 id1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) y (pySet z)

instance
		(id0 ~ id9, id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (PYNative dom0 id0) (PYExpr dom1 id1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) y (pyExpressibleList z)

instance
		(id0 ~ id9, id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (PYNative dom0 id0) (PYExpr dom1 id1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) y (pyExpressibleReplicate z)

instance
		(id0 ~ id9, id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYNative dom0 id0) (PYExpr dom1 id1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) y (pyExpressibleVector z)

-- PYNative, PYNative, X
instance
		(id0 ~ id9, id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9) =>
		 --
		PY3 (PYNative dom0 id0) (PYNative dom1 id1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pySet y) z

instance
		(id0 ~ id9, id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9) =>
		 --
		PY3 (PYNative dom0 id0) (PYNative dom1 id1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pySet y) (pySet z)

instance
		(id0 ~ id9, id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (PYNative dom0 id0) (PYNative dom1 id1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pySet y) (pyExpressibleList z)

instance
		(id0 ~ id9, id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (PYNative dom0 id0) (PYNative dom1 id1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pySet y) (pyExpressibleReplicate z)

instance
		(id0 ~ id9, id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYNative dom0 id0) (PYNative dom1 id1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pySet y) (pyExpressibleVector z)

-- PYNative, [], X
instance
		(id0 ~ id9,            id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (PYNative dom0 id0) ([] dom1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pyExpressibleList y) z

instance
		(id0 ~ id9,            id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (PYNative dom0 id0) ([] dom1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pyExpressibleList y) (pySet z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (PYNative dom0 id0) ([] dom1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pyExpressibleList y) (pyExpressibleList z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (PYNative dom0 id0) ([] dom1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pyExpressibleList y) (pyExpressibleReplicate z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYNative dom0 id0) ([] dom1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pyExpressibleList y) (pyExpressibleVector z)

-- PYNative, Replicate, X
instance
		(id0 ~ id9,            id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (PYNative dom0 id0) (Replicate dom1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pyExpressibleReplicate y) z

instance
		(id0 ~ id9,            id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (PYNative dom0 id0) (Replicate dom1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pyExpressibleReplicate y) (pySet z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (PYNative dom0 id0) (Replicate dom1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pyExpressibleReplicate y) (pyExpressibleList z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (PYNative dom0 id0) (Replicate dom1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pyExpressibleReplicate y) (pyExpressibleReplicate z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYNative dom0 id0) (Replicate dom1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pyExpressibleReplicate y) (pyExpressibleVector z)

-- PYNative, Vector, X
instance
		(id0 ~ id9,            id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYNative dom0 id0) (Vector dom1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pyExpressibleVector y) z

instance
		(id0 ~ id9,            id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYNative dom0 id0) (Vector dom1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pyExpressibleVector y) (pySet z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYNative dom0 id0) (Vector dom1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pyExpressibleVector y) (pyExpressibleList z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYNative dom0 id0) (Vector dom1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pyExpressibleVector y) (pyExpressibleReplicate z)

instance
		(id0 ~ id9,                       dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (PYNative dom0 id0) (Vector dom1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pySet x) (pyExpressibleVector y) (pyExpressibleVector z)

--------------------------------------------------------------------------------
--    [], X, X
--------------------------------------------------------------------------------
-- [], PYExpr, X
instance
		(           id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 ([] dom0) (PYExpr dom1 id1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) y z

instance
		(           id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 ([] dom0) (PYExpr dom1 id1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) y (pySet z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 ([] dom0) (PYExpr dom1 id1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) y (pyExpressibleList z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 ([] dom0) (PYExpr dom1 id1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) y (pyExpressibleReplicate z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 ([] dom0) (PYExpr dom1 id1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) y (pyExpressibleVector z)

-- [], PYNative, X
instance
		(           id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 ([] dom0) (PYNative dom1 id1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pySet y) z

instance
		(           id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 ([] dom0) (PYNative dom1 id1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pySet y) (pySet z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 ([] dom0) (PYNative dom1 id1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pySet y) (pyExpressibleList z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 ([] dom0) (PYNative dom1 id1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pySet y) (pyExpressibleReplicate z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 ([] dom0) (PYNative dom1 id1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pySet y) (pyExpressibleVector z)

-- [], [], X
instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 ([] dom0) ([] dom1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pyExpressibleList y) z

instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 ([] dom0) ([] dom1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pyExpressibleList y) (pySet z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 ([] dom0) ([] dom1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pyExpressibleList y) (pyExpressibleList z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 ([] dom0) ([] dom1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pyExpressibleList y) (pyExpressibleReplicate z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 ([] dom0) ([] dom1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pyExpressibleList y) (pyExpressibleVector z)

-- [], Replicate, X
instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 ([] dom0) (Replicate dom1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pyExpressibleReplicate y) z

instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 ([] dom0) (Replicate dom1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pyExpressibleReplicate y) (pySet z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 ([] dom0) (Replicate dom1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pyExpressibleReplicate y) (pyExpressibleList z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 ([] dom0) (Replicate dom1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pyExpressibleReplicate y) (pyExpressibleReplicate z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 ([] dom0) (Replicate dom1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pyExpressibleReplicate y) (pyExpressibleVector z)

-- [], Vector, X
instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 ([] dom0) (Vector dom1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pyExpressibleVector y) z

instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 ([] dom0) (Vector dom1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pyExpressibleVector y) (pySet z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 ([] dom0) (Vector dom1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pyExpressibleVector y) (pyExpressibleList z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 ([] dom0) (Vector dom1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pyExpressibleVector y) (pyExpressibleReplicate z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 ([] dom0) (Vector dom1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleList x) (pyExpressibleVector y) (pyExpressibleVector z)

--------------------------------------------------------------------------------
--    Replicate, X, X
--------------------------------------------------------------------------------
-- Replicate, PYExpr, X
instance
		(           id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 (Replicate dom0) (PYExpr dom1 id1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) y z

instance
		(           id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (Replicate dom0) (PYExpr dom1 id1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) y (pySet z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 (Replicate dom0) (PYExpr dom1 id1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) y (pyExpressibleList z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 (Replicate dom0) (PYExpr dom1 id1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) y (pyExpressibleReplicate z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Replicate dom0) (PYExpr dom1 id1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) y (pyExpressibleVector z)

-- Replicate, PYNative, X
instance
		(           id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (Replicate dom0) (PYNative dom1 id1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pySet y) z

instance
		(           id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (Replicate dom0) (PYNative dom1 id1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pySet y) (pySet z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (Replicate dom0) (PYNative dom1 id1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pySet y) (pyExpressibleList z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (Replicate dom0) (PYNative dom1 id1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pySet y) (pyExpressibleReplicate z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Replicate dom0) (PYNative dom1 id1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pySet y) (pyExpressibleVector z)

-- Replicate, [], X
instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 (Replicate dom0) ([] dom1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pyExpressibleList y) z

instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (Replicate dom0) ([] dom1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pyExpressibleList y) (pySet z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 (Replicate dom0) ([] dom1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pyExpressibleList y) (pyExpressibleList z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 (Replicate dom0) ([] dom1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pyExpressibleList y) (pyExpressibleReplicate z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Replicate dom0) ([] dom1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pyExpressibleList y) (pyExpressibleVector z)

-- Replicate, Replicate, X
instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 (Replicate dom0) (Replicate dom1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pyExpressibleReplicate y) z

instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PY3 (Replicate dom0) (Replicate dom1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pyExpressibleReplicate y) (pySet z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 (Replicate dom0) (Replicate dom1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pyExpressibleReplicate y) (pyExpressibleList z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PY3 (Replicate dom0) (Replicate dom1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pyExpressibleReplicate y) (pyExpressibleReplicate z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Replicate dom0) (Replicate dom1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pyExpressibleReplicate y) (pyExpressibleVector z)

-- Replicate, Vector, X
instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Replicate dom0) (Vector dom1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pyExpressibleVector y) z

instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Replicate dom0) (Vector dom1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pyExpressibleVector y) (pySet z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Replicate dom0) (Vector dom1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pyExpressibleVector y) (pyExpressibleList z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Replicate dom0) (Vector dom1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pyExpressibleVector y) (pyExpressibleReplicate z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Replicate dom0) (Vector dom1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleReplicate x) (pyExpressibleVector y) (pyExpressibleVector z)

--------------------------------------------------------------------------------
--    Vector, X, X
--------------------------------------------------------------------------------
-- Vector, PYExpr, X
instance
		(           id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (PYExpr dom1 id1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) y z

instance
		(           id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (PYExpr dom1 id1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) y (pySet z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (PYExpr dom1 id1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) y (pyExpressibleList z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (PYExpr dom1 id1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) y (pyExpressibleReplicate z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (PYExpr dom1 id1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) y (pyExpressibleVector z)

-- Vector, PYNative, X
instance
		(           id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (PYNative dom1 id1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pySet y) z

instance
		(           id1 ~ id9, id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (PYNative dom1 id1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pySet y) (pySet z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (PYNative dom1 id1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pySet y) (pyExpressibleList z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (PYNative dom1 id1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pySet y) (pyExpressibleReplicate z)

instance
		(           id1 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (PYNative dom1 id1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pySet y) (pyExpressibleVector z)

-- Vector, [], X
instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) ([] dom1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pyExpressibleList y) z

instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) ([] dom1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pyExpressibleList y) (pySet z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) ([] dom1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pyExpressibleList y) (pyExpressibleList z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) ([] dom1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pyExpressibleList y) (pyExpressibleReplicate z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) ([] dom1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pyExpressibleList y) (pyExpressibleVector z)

-- Vector, Replicate, X
instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (Replicate dom1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pyExpressibleReplicate y) z

instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (Replicate dom1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pyExpressibleReplicate y) (pySet z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (Replicate dom1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pyExpressibleReplicate y) (pyExpressibleList z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (Replicate dom1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pyExpressibleReplicate y) (pyExpressibleReplicate z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (Replicate dom1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pyExpressibleReplicate y) (pyExpressibleVector z)

-- Vector, Vector, X
instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (Vector dom1) (PYExpr dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pyExpressibleVector y) z

instance
		(                      id2 ~ id9, dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (Vector dom1) (PYNative dom2 id2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pyExpressibleVector y) (pySet z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (Vector dom1) ([] dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pyExpressibleVector y) (pyExpressibleList z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (Vector dom1) (Replicate dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pyExpressibleVector y) (pyExpressibleReplicate z)

instance
		(                                 dom0 ~ dom9, dom1 ~ dom9, dom2 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PY3 (Vector dom0) (Vector dom1) (Vector dom2) dom9 id9
	where
		pyLift3 func x y z = func (pyExpressibleVector x) (pyExpressibleVector y) (pyExpressibleVector z)

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--    PYCond
--------------------------------------------------------------------------------
-- Instances for running a test and getting a then/else value in return.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--    PYCond1
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

instance
		(id0 ~ id9, dom0 ~ dom9) =>
		 --
		PYCond1 (PYExpr dom0 id0) dom9 id9
	where
		pyLiftCond1 func x = func x

instance
		(id0 ~ id9, dom0 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9) =>
		 --
		PYCond1 (PYNative  dom0 id0) dom9 id9
	where
		pyLiftCond1 func x = func (pySet x)

instance
		(id0 ~ id9, dom0 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PYCond1 ([] dom0) dom9 id9
	where
		pyLiftCond1 func x = func (pyExpressibleList x)

instance
		(id0 ~ id9, dom0 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PYCond1 (Replicate dom0) dom9 id9
	where
		pyLiftCond1 func x = func (pyExpressibleReplicate x)

instance
		(id0 ~ id9, dom0 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PYCond1 (Vector dom0) dom9 id9
	where
		pyLiftCond1 func x = func (pyExpressibleVector x)

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--    PYCond2
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

-- PYExpr
instance
		(id0 ~ id9, id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9) =>
		 --
		PYCond2 (PYExpr dom0 id0) (PYExpr dom1 id1) dom9 id9
	where
		pyLiftCond2 func x y = func x y

instance
		(id0 ~ id9, id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9) =>
		 --
		PYCond2 (PYExpr dom0 id0) (PYNative dom1 id1) dom9 id9
	where
		pyLiftCond2 func x y = func x (pySet y)

instance
		(id0 ~ id9,            dom0 ~ dom9, dom1 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PYCond2 (PYExpr dom0 id0) ([] dom1)         dom9 id9
	where
		pyLiftCond2 func x y = func x (pyExpressibleList y)

instance
		(id0 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, 
		PYStorable dom9 id9) =>
		 --
		PYCond2 (PYExpr dom0 id0) (Replicate dom1)  dom9 id9
	where
		pyLiftCond2 func x y = func x (pyExpressibleReplicate y)

instance
		(id0 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, 
		PYStorable dom9 id9, Unbox dom9) =>
		 --
		PYCond2 (PYExpr dom0 id0) (Vector dom1)  dom9 id9
	where
		pyLiftCond2 func x y = func x (pyExpressibleVector y)

-- PYNative
instance
		(id0 ~ id9, id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9) =>
		 --
		PYCond2 (PYNative dom0 id0) (PYExpr dom1 id1) dom9 id9
	where
		pyLiftCond2 func x y = func (pySet x) y

instance
		(id0 ~ id9, id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9) =>
		 --
		PYCond2 (PYNative dom0 id0) (PYNative dom1 id1) dom9 id9
	where
		pyLiftCond2 func x y = func (pySet x) (pySet y)

instance
		(id0 ~ id9,            dom0 ~ dom9, dom1 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PYCond2 (PYNative dom0 id0) ([] dom1)         dom9 id9
	where
		pyLiftCond2 func x y = func (pySet x) (pyExpressibleList y)

instance
		(id0 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, 
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PYCond2 (PYNative dom0 id0) (Replicate dom1)  dom9 id9
	where
		pyLiftCond2 func x y = func (pySet x) (pyExpressibleReplicate y)

instance
		(id0 ~ id9,            dom0 ~ dom9, dom1 ~ dom9, 
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PYCond2 (PYNative dom0 id0) (Vector dom1)  dom9 id9
	where
		pyLiftCond2 func x y = func (pySet x) (pyExpressibleVector y)

-- []
instance
		(           id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PYCond2 ([] dom0) (PYExpr dom1 id1) dom9 id9
	where
		pyLiftCond2 func x y = func (pyExpressibleList x) y

instance
		(           id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PYCond2 ([] dom0) (PYNative dom1 id1) dom9 id9
	where
		pyLiftCond2 func x y = func (pyExpressibleList x) (pySet y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PYCond2 ([] dom0) ([] dom1)         dom9 id9
	where
		pyLiftCond2 func x y = func (pyExpressibleList x) (pyExpressibleList y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9, 
		PYStorable dom9 id9) =>
		 --
		PYCond2 ([] dom0) (Replicate dom1)  dom9 id9
	where
		pyLiftCond2 func x y = func (pyExpressibleList x) (pyExpressibleReplicate y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9, 
		PYStorable dom9 id9, Unbox dom9) =>
		 --
		PYCond2 ([] dom0) (Vector dom1)  dom9 id9
	where
		pyLiftCond2 func x y = func (pyExpressibleList x) (pyExpressibleVector y)

-- Replicate
instance
		(           id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PYCond2 (Replicate dom0) (PYExpr dom1 id1) dom9 id9
	where
		pyLiftCond2 func x y = func (pyExpressibleReplicate x) y

instance
		(           id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9) =>
		 --
		PYCond2 (Replicate dom0) (PYNative dom1 id1) dom9 id9
	where
		pyLiftCond2 func x y = func (pyExpressibleReplicate x) (pySet y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9,
		 PYStorable dom9 id9) =>
		 --
		PYCond2 (Replicate dom0) ([] dom1)         dom9 id9
	where
		pyLiftCond2 func x y = func (pyExpressibleReplicate x) (pyExpressibleList y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9, 
		PYStorable dom9 id9) =>
		 --
		PYCond2 (Replicate dom0) (Replicate dom1)  dom9 id9
	where
		pyLiftCond2 func x y = func (pyExpressibleReplicate x) (pyExpressibleReplicate y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9, 
		PYStorable dom9 id9, Unbox dom9) =>
		 --
		PYCond2 (Replicate dom0) (Vector dom1)  dom9 id9
	where
		pyLiftCond2 func x y = func (pyExpressibleReplicate x) (pyExpressibleVector y)

-- Vector
instance
		(           id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PYCond2 (Vector dom0) (PYExpr dom1 id1) dom9 id9
	where
		pyLiftCond2 func x y = func (pyExpressibleVector x) y

instance
		(           id1 ~ id9, dom0 ~ dom9, dom1 ~ dom9,
		 PYExpressible (PYNative dom9 id9) dom9 id9, PYStorable dom9 id9, Unbox dom9) =>
		 --
		PYCond2 (Vector dom0) (PYNative dom1 id1) dom9 id9
	where
		pyLiftCond2 func x y = func (pyExpressibleVector x) (pySet y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9,
		 PYStorable dom9 id9, Unbox dom9) =>
		 --
		PYCond2 (Vector dom0) ([] dom1)         dom9 id9
	where
		pyLiftCond2 func x y = func (pyExpressibleVector x) (pyExpressibleList y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9, 
		PYStorable dom9 id9, Unbox dom9) =>
		 --
		PYCond2 (Vector dom0) (Replicate dom1)  dom9 id9
	where
		pyLiftCond2 func x y = func (pyExpressibleVector x) (pyExpressibleReplicate y)

instance
		(                      dom0 ~ dom9, dom1 ~ dom9, 
		PYStorable dom9 id9, Unbox dom9) =>
		 --
		PYCond2 (Vector dom0) (Vector dom1)  dom9 id9
	where
		pyLiftCond2 func x y = func (pyExpressibleVector x) (pyExpressibleVector y)
