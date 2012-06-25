module PEGGY.Obsidian (
	module PEGGY.Obsidian.Targets,
	module PEGGY.Obsidian.Functions,
	module PEGGY.Obsidian.Natives,
	generateObsidianCode,
	setKernelName,
	compileCUDA
	) where

--import PEGGY.Accelerator.Storable
import PEGGY.Obsidian.Natives
import PEGGY.Obsidian.Functions
import PEGGY.Obsidian.Targets
import PEGGY.Obsidian.Instances
import PEGGY.Obsidian.Run
