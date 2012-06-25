{-# LANGUAGE FlexibleInstances #-}

-- Various messge types.

import PEGGY
import PEGGY.Haskell
import PEGGY.Accelerator
import PEGGY.FloatBound

--import Results

--import qualified TS as TS

import Data.List (zip4, unzip4, zipWith5)

--data SkillPrior = SkillPrior

--type Target = TargetHaskell
type Target = TargetDX9

doGet :: PYExpr Float Target -> [] Float
doGet = pyGet . pyRun

--scores, winMean, winVar, loseMean, loseVar :: PYExpr Float TargetHaskell
--winMean = pyExpressibleNumber [1000] 25.0
--winVar = pyExpressibleNumber [1000] 11.0
--loseMean = pyExpressibleNumber [1000] 25.0
--loseVar = pyExpressibleNumber [1000] 11.0

--scores = pySet $ concat $ replicate 500 [0, 2]
--scores = pySet [0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2,2,0,2,0,2]

--doRun = zip4 (doGet p0) (doGet p1) (doGet p2) (doGet p3)
--	where
--		(Gaussian p0 p1, Gaussian p2 p3) = doPerformanceUpdate scores (Outcome (Gaussian winMean winVar) (Gaussian loseMean loseVar))

-- "PerformanceUpdate" is the part done in parallel.

data Gaussian arr = Gaussian { mean :: arr, variance :: arr }

c_LN_SQRT_2_pi = 0.9189385332046727417803297364056
c_E = 2.71828182845904523536028747135266249775724709369995

gaussian m v = Gaussian (pySet m) (pySet v)

instance Show (Gaussian Float) where
	show (Gaussian a b) = "Gaussian: mean = " ++ show a ++ ", variance = " ++ show b

instance Eq (Gaussian Float) where
	(==) (Gaussian a b) (Gaussian c d) = a == c && b == d

exp = pyPow c_E
ln = pyLog c_E

data Performance arr = Performance { aPerf :: Gaussian arr, bPerf :: Gaussian arr }
data Outcome arr = Outcome { inWin :: Gaussian arr, inLose :: Gaussian arr }

--data SkillPrior = SkillPrior { players :: [Int], skill :: [Gaussian Float] }
--
--data SkillMarginal = SkillMarginal { players :: [Int], skill :: [Gaussian Float] }
--
--data PerformanceMarginal = PerformanceMarginal

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--                                                                            --
--                                   Tests                                    --
--                                                                            --
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

testSkillPriorUpdate1 = skillPriors $ skillPriorUpdate trueSkillStateInitial
testSkillPriorUpdate2 = skillMarginals $ skillPriorUpdate trueSkillStateInitial

testSkillPerformancePerformanceUpdate1 = skillPerformances $ skillPerformancePerformanceUpdate $ skillPriorUpdate trueSkillStateInitial
testSkillPerformancePerformanceUpdate2 = performanceMarginals $ skillPerformancePerformanceUpdate $ skillPriorUpdate trueSkillStateInitial

testPerformanceUpdate1 = performanceMarginals $ performanceUpdate $ skillPerformancePerformanceUpdate $ skillPriorUpdate trueSkillStateInitial
testPerformanceUpdate2 = scores $ performanceUpdate $ skillPerformancePerformanceUpdate $ skillPriorUpdate trueSkillStateInitial

testSkillPerformanceSkillUpdate1 = skillMarginals $ skillPerformanceSkillUpdate $ performanceUpdate $ skillPerformancePerformanceUpdate $ skillPriorUpdate trueSkillStateInitial
testSkillPerformanceSkillUpdate2 = skillPerformances $ skillPerformanceSkillUpdate $ performanceUpdate $ skillPerformancePerformanceUpdate $ skillPriorUpdate trueSkillStateInitial

testIterations 0 = trueSkillStateInitial
testIterations n = skillPerformanceSkillUpdate $ performanceUpdate $ skillPerformancePerformanceUpdate $ skillPriorUpdate $ testIterations (n - 1)

testAll n = map (\ s -> (mean (skillMarginalSkill s) / variance (skillMarginalSkill s), 1.0 / variance (skillMarginalSkill s))) (skillMarginals (testIterations n))

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--                                                                            --
--                                  Messages                                  --
--                                                                            --
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

-- These would be MUCH simpler if they were structures of lists, not lists of
-- structures.  Note to self: try abstract this data representation in future.
-- If [(a, b)] is better use that; if ([a], [b]) is better, use that instead.
-- No idea how this can be determined programatically, probably can't in all
-- cases without loads of additional annotations.

-- Player, Skill
data SkillPrior = SkillPrior
	{
		skillProirPlayer :: Int,
		skillPriorSkill  :: Gaussian Float
	}
	deriving (Show, Eq)
type SkillPriors = [SkillPrior]

-- Player, Skill
data SkillMarginal = SkillMarginal
	{
		skillMarginalPlayer :: Int,
		skillMarginalSkill  :: Gaussian Float
	}
	deriving (Show, Eq)
type SkillMarginals = [SkillMarginal]

-- Game, Player, Performance
data PerformanceMarginal = PerformanceMarginal
	{
		performanceMarginalGame        :: Int,
		performanceMarginalPlayer      :: Int,
		performanceMarginalPerformance :: Gaussian Float
	}
	deriving (Show, Eq)
type PerformanceMarginals = [PerformanceMarginal]

-- Game, Player, Performance, Skill
data SkillPerformance = SkillPerformance
	{
		skillPerformanceGame        :: Int,
		skillPerformancePlayer      :: Int,
		skillPerformancePerformance :: Gaussian Float,
		skillPerformanceSkill       :: Gaussian Float
	}
	deriving (Show, Eq)
type SkillPerformances = [SkillPerformance]

-- Game, Player A, Player B, Score, Performance A, Performance B
data Score = Score
	{
		scoreGame         :: Int,
		scorePlayerA      :: Int,
		scorePlayerB      :: Int,
		scoreScore        :: Float,
		scorePerformanceA :: Gaussian Float,
		scorePerformanceB :: Gaussian Float
	}
	deriving (Show, Eq)
type Scores = [Score]

data TrueSkillState = TrueSkillState
	{
		skillPriors          :: SkillPriors,
		skillMarginals       :: SkillMarginals,
		performanceMarginals :: PerformanceMarginals,
		skillPerformances    :: SkillPerformances,
		scores               :: Scores
	}
	deriving (Show, Eq)

findOneBy f [] a = error "findOneBy failed."
findOneBy f (x : xs) a = case f x == a of
	True  -> x
	False -> findOneBy f xs a

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--                                                                            --
--                                Constructors                                --
--                                                                            --
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

--skillPriors'          m v = zipWith SkillPrior    [0 .. ] (zipWith Gaussian m v)
--skillMarginals'       m v = zipWith SkillMarginal [0 .. ] (zipWith Gaussian m v)
--performanceMarginals' m v = zipWith (\ n g -> PerformanceMarginal ((n `div` 2) + 1) (n `mod` maxPlayers) g) [0 .. ] (zipWith Gaussian m v)
--skillPerformances' pm pv sm sv = zipWith3 (\ n p s -> SkillPerformance ((n `div` 2) + 1) (n `mod` maxPlayers) p s) [0 .. ] (zipWith Gaussian pm pv) (zipWith Gaussian sm sv)
--scores gid p0 p1 sc m0 v0 m1 v1 = Score gid p0 p1 sc (Gaussian

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--                                                                            --
--                                  Initial                                   --
--                                                                            --
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

maxPlayers = 5
maxGames = 42 * 42

skillPriorsInitial'          players = map (\ p -> SkillPrior    p (Gaussian 0.0 0.0)) [0 .. players - 1]
skillMarginalsInitial'       players = map (\ p -> SkillMarginal p (Gaussian 0.0 0.0)) [0 .. players - 1]
performanceMarginalsInitial' gameDat = map (\ (g, p, _) -> PerformanceMarginal g p (Gaussian 0.0 0.0)) gameDat
skillPerformancesInitial'    gameDat = map (\ (g, p, _) -> SkillPerformance    g p (Gaussian 0.0 0.0) (Gaussian 0.0 0.0)) gameDat

scoresInitial' [] = []
scoresInitial' ((gid, p0, sc) : (_, p1, _) : ps) = Score gid p0 p1 sc (Gaussian 0.0 (maxBound :: Float)) (Gaussian 0.0 (maxBound :: Float)) : scoresInitial' ps

--gameData _ [] = []
--gameData n (s : sc) = (gid, p0, sc) : (gid, p1, 2 - sc) : gameData (n + 2) sc
--	where
--		gid = (n `div` 2) + 1
--		p0 = n `mod` maxPlayers
--		p1 = nextPlayer n 1
--		score = if p0 < p1 then 2 - sc else sc

gameData' n m
	| n >= m    = []
	| otherwise = (gid, p0, sc) : (gid, p1, 2 - sc) : gameData' (n + 2) m
		where
			gid = (n `div` 2) + 1
			p0  = n `mod` maxPlayers
			p1  = nextPlayer n 1
			sc  = if p0 < p1 then 0 else 2
			nextPlayer n q
				| (n + n + q) `mod` maxPlayers == n `mod` maxPlayers = nextPlayer n (q + 1)
				| otherwise = (n + n + q) `mod` maxPlayers

gameData = gameData' 0 (maxGames)
skillPriorsInitial = skillPriorsInitial' maxPlayers
skillMarginalsInitial = skillMarginalsInitial' maxPlayers
performanceMarginalsInitial = performanceMarginalsInitial' gameData
skillPerformancesInitial = skillPerformancesInitial'    gameData

scoresInitial = scoresInitial' gameData

trueSkillStateInitial = TrueSkillState skillPriorsInitial skillMarginalsInitial performanceMarginalsInitial skillPerformancesInitial scoresInitial

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--                                                                            --
--                             Skill Prior Update                             --
--                                                                            --
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

skillPriorUpdate :: TrueSkillState -> TrueSkillState
skillPriorUpdate tss = tss { skillPriors = newPriors, skillMarginals = newMarginals }
	where
		nm = 25.0 / 10.0
		nv = 1.0 / 10.0
		oldMarginals = skillMarginals tss
		oldPriors    = skillPriors    tss
		newMarginals = zipWith doMarginals oldMarginals oldPriors
		--newPriors    = replicate (length oldPriors) (Gaussian (25.0 / 10.0) (1.0 / 10.0))
		newPriors    = map (\ p -> p { skillPriorSkill = Gaussian nm nv }) oldPriors
		doMarginals marg prior = marg { skillMarginalSkill = domarg (skillMarginalSkill marg) (skillPriorSkill prior) }
		domarg g0 g1 = gsub (gadd (Gaussian nm nv) g0) g1

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--                                                                            --
--                    Skill Performance Performance Update                    --
--                                                                            --
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

skillPerformancePerformanceUpdate :: TrueSkillState -> TrueSkillState
skillPerformancePerformanceUpdate tss = tss { skillPerformances = fst parts, performanceMarginals = snd parts }
	where
		-- Combine in to pairs of outputs.
		parts = unzip $ zipWith oneUpdate (skillPerformances tss) (performanceMarginals tss)
		-- Take a matching set of performances and marginals and combine them to
		-- give two outputs (as a pair).
		oneUpdate
				p @ (SkillPerformance _ player g1 g2)
				m @ (PerformanceMarginal _ _ g0) =
			let
				marginal = skillMarginalSkill $ findOneBy skillMarginalPlayer (skillMarginals tss) player
				Gaussian mean' prec' = gsub marginal g2
				g' = Gaussian (mean' / (prec' + 1)) (prec' / (prec' + 1))
			in
				(p { skillPerformancePerformance = g' }, m { performanceMarginalPerformance = gsub (gadd g' g0) g1 })

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--                                                                            --
--                             Performance Update                             --
--                                                                            --
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

performanceUpdate :: TrueSkillState -> TrueSkillState
performanceUpdate tss = tss { performanceMarginals = newPerformanceMarginals, scores = newScores }
	where
		oldScores = scores tss
		oldPerformanceMarginals = performanceMarginals tss
		pmap f (a, b, c, d) = (f a, f b, f c, f d)
		ucur4 f (a, b, c, d) = f a b c d
		-- Find one marginal for the current game ID.
		findMarginals' game [] = (undefined, [])
		findMarginals' game (m : ms) =
			if game == performanceMarginalGame m
			then (performanceMarginalPerformance m, ms)
			else findMarginals' game ms
		-- Split the data up, we have pairs of data for winners and losers.
		getMarginals sc' =
			let
				-- Extract game result data.
				game = scoreGame sc'
				(Gaussian amean avar) = scorePerformanceA sc'
				(Gaussian bmean bvar) = scorePerformanceB sc'
				-- Find other related data.
				(m0, rst) = findMarginals' game oldPerformanceMarginals
				(m1, _) = findMarginals' game rst
				-- Can extend this for more players.
				tmp0 = variance m0 - (1.0 / variance (scorePerformanceA sc'))
				mean0 = (mean m0 - mean (scorePerformanceA sc') / variance (scorePerformanceA sc')) / tmp0
				var0 = 1.0 / tmp0
				tmp1 = variance m1 - (1.0 / variance (scorePerformanceB sc'))
				mean1 = (mean m1 - mean (scorePerformanceB sc') / variance (scorePerformanceB sc')) / tmp1
				var1 = 1.0 / tmp1
			in
				if scoreScore sc' == 2
				then (mean0, var0, mean1, var1) -- (wmean, wvar, lmean, lvar)
				else (mean1, var1, mean0, var0) -- (wmean, wvar, lmean, lvar)
		-- For every game, find the corresponding marginal performances for all
		-- participating players (list of performance tuples).
		oldMarginals = unzip4 $ map getMarginals oldScores
		pyScore :: PYExpr Float Target
		pyScore = pySet $ map scoreScore oldScores
		pyPerformance =
			let
				pyOutcomes :: ([Float], [Float], [Float], [Float]) -> Outcome (PYExpr Float Target)
				pyOutcomes (wm, wv, lm, lv) = Outcome (Gaussian (pySet wm) (pySet wv)) (Gaussian (pySet lm) (pySet lv))
			in
				doPerformanceUpdate pyScore (pyOutcomes oldMarginals)
		allPerformanceUpdates :: ([Float], [Float], [Float], [Float])
		allPerformanceUpdates = pmap (pyGet . pyRun) pyPerformance
		--newPerformanceMarginals = error $ show allPerformanceUpdates
		-- Evaluate and recombine the results.
		-- Map over MARGINALS, not scores as is done in the C++ version.  The
		-- original used mutable updates to accumulate the performance changes,
		-- we can't do that here so need to touch each new marginal only once.
		-- Probably need to get a list of all performance updates affecting each
		-- individual performance marginal (i.e. each player's updates) and do a
		-- fold of some sort over them.  Not sure how yet, just thinking aloud.
		newPerformanceMarginals = map (findAllPerformanceUpdates oldScores allPerformanceUpdates) oldPerformanceMarginals
		--newPerformanceMarginals = zipWith (\ a b -> a { performanceMarginalPerformance = b } ) oldPerformanceMarginals perfs
		newScores = ucur4 (zipWith5 (\ b a0 a1 a2 a3 -> b { scorePerformanceA = Gaussian a0 a1, scorePerformanceB = Gaussian a2 a3 } ) oldScores) allPerformanceUpdates
		-- Turns out that the fold is implicit, we don't call "foldr", instead
		-- we code in the recursion.
		findAllPerformanceUpdates [] _ cur = cur
		findAllPerformanceUpdates
			-- A player can't play against themselves.
			(sc : scs)
			(amean : ams,
			 avar  : avs,
			 bmean : bms,
			 bvar  : bvs)
			cur =
				let 
					-- "Result", "Score", "Old/Other".  Convert to alternate
					-- Gaussian representation.
					doAdd rm rv (Gaussian sm sv) cur' = cur' { performanceMarginalPerformance = Gaussian (rm / rv - sm / sv) (1.0 / rv - 1.0 / sv) `gadd` performanceMarginalPerformance cur' }
				in if scoreGame sc == performanceMarginalGame cur
					then
						if scorePlayerA sc == performanceMarginalPlayer cur
						-- Use the "a" gaussian update.
						then doAdd amean avar (scorePerformanceA sc) (findAllPerformanceUpdates scs (ams, avs, bms, bvs) cur)
						-- Use the "b" gaussian update.
						else doAdd bmean bvar (scorePerformanceB sc) (findAllPerformanceUpdates scs (ams, avs, bms, bvs) cur)
					else findAllPerformanceUpdates scs (ams, avs, bms, bvs) cur
		--newPerformanceMarginals = oldPerformanceMarginals {}

doPerformanceUpdate scores outcomes = (
		{-Gaussian-} (cond mean sumave bave), (cond variance sumave bave),
		{-Gaussian-} (cond mean bave sumave), (cond variance bave sumave))
	where
		out = aAverageConditional (inWin outcomes) (inLose outcomes)
		--back = out
		--sumave = out
		--bave = out
		--back = out
		back = xAverageConditional out
		sumave = sumAverageConditional (inLose outcomes) back
		bave = bAverageConditional (inWin outcomes) back
		cond f a b = pyIfThenElse (pyEQ scores (pyConst 2)) (f a) (f b)

--aAverageConditional :: Gaussian arr -> Gaussian arr -> Gaussian arr
aAverageConditional a b = Gaussian (mean a - mean b) (variance a + variance b)

bAverageConditional = aAverageConditional

sumAverageConditional = gadd--a b = Gaussian (mean a + mean b) (variance a + variance b)

reciprocal = (/) (pyConst 1.0)

rsqrt = reciprocal . sqrt

xAverageConditional a = Gaussian ((weight * tau + alpha) * out) out
--xAverageConditional a = Gaussian z rsq
	where
		rsq = rsqrt (variance a)
		z = (mean a) * rsq
		v = logAverageFactor a
		alpha = Main.exp (z * z * pyConst (-0.5) - pyConst c_LN_SQRT_2_pi - v) * rsq
		tau = (mean a) / (variance a) + alpha
		beta = alpha * tau
		weight = beta / (reciprocal (variance a) - beta)
		out = (reciprocal beta - variance a)

logAverageFactor = getLogAverageOf . isPositiveAverageConditional

getLogAverageOf a = negate (ln (pyConst 1 + Main.exp (negate a)))

isPositiveAverageConditional a = normalCdfLogit ((mean a) / sqrt (variance a))

normalCdfLogit a = normalCdfLn a - normalCdfLn (negate a)

normalCdfLn = ln . normalCdf

-- The bulk of the data processing.
normalCdf a = pyIfThenElse (pyLT a (pyConst 0)) ex (pyConst 1 - ex)
	where
		coe0 = [0.700383064443688, 6.37396220353165, 33.912866078383, 112.079291497871, 221.213596169931, 220.206867912376]
		coe1 = [1.75566716318264, 16.064177579207, 86.7807322029461, 296.564248779674, 637.333633378831, 793.826512519948, 440.413735824752]
		l = abs a
		--build0 = foldWithIndex (\ idx cur val = cur * fromIntegral idx + val) 
		build0 = foldl build' (pyConst 0.0352624965998911) coe0
		build1 = foldl build' (pyConst 0.0883883476483184) coe1
		build' c v = c * l + pyConst v
		--ex = exp (negate (l * l) * fromRational 0.5) * build0 / build1
		ex = Main.exp (l * l * pyConst (-0.5)) * build0 / build1

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--                                                                            --
--                       Skill Performance Skill Update                       --
--                                                                            --
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

skillPerformanceSkillUpdate :: TrueSkillState -> TrueSkillState
skillPerformanceSkillUpdate tss = tss { skillMarginals = newMarginals, skillPerformances = newPerformances }
	where
		sumPlayer _ [] = Gaussian 0.0 0.0
		sumPlayer i (c : cs) =
				if skillPerformancePlayer c == i
				then gadd (sumPlayer i cs) (skillPerformanceSkill c)
				else (sumPlayer i cs)
		oskill = skillPerformances tss
		nskill = zipWith doNewPerformance oskill (performanceMarginals tss)
		doNewPerformance s m =
			let
				os = skillPerformancePerformance s
				om = performanceMarginalPerformance m
				prec' = variance om - variance os
			in
				s { skillPerformanceSkill = Gaussian ((mean om - mean os) / (prec' + 1)) (prec' / (prec' + 1)) }
		newPerformances = nskill
		-- Do the new skill marginals.
		newMarginals = map doNewMarginal (skillMarginals tss)
		doNewMarginal (SkillMarginal player marg) = SkillMarginal player $ gsub (gadd (sumPlayer player nskill) marg) (sumPlayer player oskill)

gadd (Gaussian m1 v1) (Gaussian m2 v2) = Gaussian (m1 + m2) (v1 + v2)
gsub (Gaussian m1 v1) (Gaussian m2 v2) = Gaussian (m1 - m2) (v1 - v2)
