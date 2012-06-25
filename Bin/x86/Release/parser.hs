-- :main .\\Convolution\\Output\\ "Report4 31 ? 5 c.txt"
-- setCurrentDirectory "..\\.."
-- :main .\\UDWT\\Output\\ "Report2 ? 0 1 3.txt"
-- setCurrentDirectory "..\\.."
-- :main .\\Molecules\\Output\\ "Report4 31 ? 10 c.txt"

import System.Environment
import System.Directory
import Control.Monad
import GHC.Exts (sortWith)
import Data.List (transpose)
import System.IO

type OneSize = (Int, [Int])

type AllSize = ([String], [OneSize])

writeOutput :: AllSize -> FilePath -> IO ()
writeOutput (names, times) fname = do
	fhnd <- openFile fname WriteMode
	mapM_ (hPutStr fhnd . (++) "," . show . fst) times
	hPutStrLn fhnd ""
	mapM_ (hPutStr fhnd . (++) "," . show . (*) 4096 . (*) 128 . (+) 1 . fst) times
	let
		allLines = transpose . map snd $ times
	zipWithM_ (\ n' ts' -> do
		hPutStrLn fhnd ""
		hPutStr fhnd n'
		mapM_ (hPutStr fhnd . (++) "," . show) ts')
		names allLines
	hClose fhnd

oneFile :: (FilePath, Int) -> IO OneSize
oneFile (f, n) = do
	contents <- readFile f
	let
		lns = filter ((/=) "") (lines contents)
	return (n, map (read . drop 10) lns)

getNames :: FilePath -> IO [String]
getNames f = do
	contents <- readFile f
	let
		lns = filter ((/=) "") (lines contents)
	return (map (take 4) lns)

parseFiles fs @ ((f, b) : _) = do
	ns <- getNames f
	ts <- mapM oneFile fs
	return (ns, ts)

main = do
	args <- getArgs
	pname <- getProgName
	if (length args < 2)
	then putStrLn $ "Usage: " ++ pname ++ " <directory> <pattern>"
	else do
		let
			pattern' = span ((/=) '?') (args !! 1)
			pattern0 = fst pattern'
			pattern1 = reverse . tail . snd $ pattern'
			-- An input of:
			--  "hello ?.txt"
			-- will give a pattern of:
			--  ("hello ", "txt.")
			matchesPattern f =
				take (length pattern0) f == pattern0 &&
				take (length pattern1) (reverse f) == pattern1
			-- An input of:
			--  "hello 27.txt"
			-- Will give an output of:
			--  ("hello 27.txt", "27")
			reqFile :: FilePath -> IO (Maybe (FilePath, Int))
			reqFile a = doesFileExist a >>= (\ x ->
				if (x && matchesPattern a)
				then return (Just (a, read $ drop (length pattern0) (take (length a - length pattern1) a)))
				else return Nothing)
		setCurrentDirectory (args !! 0)
		-- Get all the valid files and sort them in to order by difference.
		allItems <- getDirectoryContents "."
		allFiles <- mapMaybeM reqFile allItems
		let
			sorted = sortWith snd allFiles
		results <- parseFiles sorted
		--putStrLn (show results)
		writeOutput results "..\\parsed.csv"

--mapMaybeM :: (Monad m) => (a -> m (Maybe b)) -> [a] -> m [b]
mapMaybeM f [] = return []
mapMaybeM f (l : ls) = f l >>= (\ l' ->
	case l' of
		Nothing -> mapMaybeM f ls
		Just a  -> mapMaybeM f ls >>= return . ((:) a))
			--return (a : b))
