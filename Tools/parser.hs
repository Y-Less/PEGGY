import System.IO

main = do
	--outputFiles "s"
	outputFiles "c"

type INS = (Int, Int, Int, Int, Int, Int, Int, Int)

gt 0 (a, _, _, _, _, _, _, _) = a
gt 1 (_, a, _, _, _, _, _, _) = a
gt 2 (_, _, a, _, _, _, _, _) = a
gt 3 (_, _, _, a, _, _, _, _) = a
gt 4 (_, _, _, _, a, _, _, _) = a
gt 5 (_, _, _, _, _, a, _, _) = a
gt 6 (_, _, _, _, _, _, a, _) = a
gt 7 (_, _, _, _, _, _, _, a) = a
gt _ _ = error "gt used with invalid index"

--outputFiles :: String -> [(Int, [(Int, (Int, Int, Int, Int, Int))])] -> IO ()
outputFiles letter = stage2 >>= mapM_ (outputSection (openFile (".\\dataout_" ++ letter ++ ".csv") AppendMode))
--outputFiles letter = stage2 >>= mapM_ (outputSection (return stdout))
	where
		stage1 :: Int -> IO [(Int, INS)]
		stage1 x = mapM (\ y -> parseInputFile letter x y >>= return . (,) y) [i * 8 + 7 | i <- [0 .. 7]] -- [0 .. 63]
		stage2 :: IO [(Int, [(Int, INS)])]
		stage2 = mapM (\ x -> stage1 x >>= return . (,) x) [i * 8 + 7 | i <- [0 .. 7]]

outputSection :: IO Handle -> (Int, [(Int, INS)]) -> IO ()
outputSection fhnd' (x, byn) = do
		fhnd <- fhnd'
		hPutStrLn fhnd (show x)
		mapM_ (\ (a, _) -> hPutStr fhnd (show a ++ ", ")) byn
		hPutStrLn fhnd ""
		mapM_ (\ n -> mapM_ (\ (_, b) -> hPutStr fhnd (show (gt n b) ++ ", ")) byn >> hPutStrLn fhnd "") [0 .. 7]
		return ()

parseInputFile :: String -> Int -> Int -> IO INS
parseInputFile letter a b =
	do
		file <- openFile ("Output\\Report4 " ++ show a ++ " " ++ show b ++ " 5 " ++ letter ++ ".txt") ReadMode
		l0 <- hGetLine file
		--l1 <- if a < 32 && b < 32 then
		--	-- Has DX9.
		--		hGetLine file
		--	else
		--		return "          0"
		l1 <- hGetLine file
		l2 <- hGetLine file
		l3 <- hGetLine file
		l4 <- hGetLine file
		l5 <- hGetLine file
		l6 <- hGetLine file
		l7 <- hGetLine file
		l8 <- hGetLine file
		-- Because screw generic code!
		hClose file
		return (rd l1, rd l2, rd l3, rd l4, rd l5, rd l6, rd l7, rd l8)
	where
		rd = read . drop 10

--getAllCombinataions :: [a] -> [b] -> [(a, b)]
--getAllCombinataions a b = concat $ map (\n -> map (\m -> (n, m))
