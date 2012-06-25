#pragma once

#include "ParallelSystem.h"

class ParallelScript
{
protected:
	struct NamedFunctions_s
	{
		char const * const
			Name;
		void ** const
			Pointer;
	};
	
	//typedef
	//	struct NamedFunctions_s *
	//	FunctionList_t;
	
	virtual void
		ScriptInit(LPCWSTR, ParallelScript::NamedFunctions_s const * const) throw (...) = 0;
	
	virtual void
		ScriptClose(const bool) throw (...) = 0;
};
