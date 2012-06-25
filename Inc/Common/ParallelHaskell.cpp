#if !defined NO_HASKELL

#include "ParallelHaskell.h"

// cons
	ParallelHaskell::
	ParallelHaskell()
	:
		m_module(0),
		HsStart(0),
		HsEnd(0)
{
}

void
	ParallelHaskell::
	ScriptInit(
		LPCWSTR                         name,
		ParallelScript::NamedFunctions_s const * const funcs) throw (...)
{
	m_module = LoadLibrary(name);
	if (!m_module)
	{
		throw "Could not load Haskell library";
	}
	// Get the important system functions.
	HsStart = (HsSystemFunc_t)GetProcAddress(m_module, "HsStart"),
	HsEnd = (HsSystemFunc_t)GetProcAddress(m_module, "HsEnd");
	if (!HsStart || !HsEnd)
	{
		throw "Could not find Haskell initialisation functions";
	}
	size_t
		idx = 0;
	while (funcs[idx].Name)
	{
		if (!(*(funcs[idx].Pointer) = GetProcAddress(m_module, funcs[idx].Name)))
		{
			throw "Could not find Haskell function";
		}
		++idx;
	}
	// Set the script running.
	HsStart();
}

void
	ParallelHaskell::
	ScriptClose(const bool) throw (...)
{
	// End the script.
	HsEnd();
}

#endif
