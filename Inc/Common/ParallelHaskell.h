#if !defined NO_HASKELL

#pragma once

/*#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN
#define NOGDICAPMASKS
#define NOVIRTUALKEYCODES
#define NOWINMESSAGES
#define NOWINSTYLES
#define NOSYSMETRICS
#define NOMENUS
#define NOICONS
#define NOKEYSTATES
#define NOSYSCOMMANDS
#define NORASTEROPS
#define NOSHOWWINDOW
#define OEMRESOURCE
#define NOATOM
#define NOCLIPBOARD
#define NOCOLOR
#define NOCTLMGR
#define NODRAWTEXT
#define NOGDI
#define NOKERNEL
#define NOUSER
#define NONLS
#define NOMB
#define NOMEMMGR
#define NOMETAFILE
#define NOMINMAX
#define NOMSG
#define NOOPENFILE
#define NOSCROLL
#define NOSERVICE
#define NOSOUND
#define NOTEXTMETRIC
#define NOWH
#define NOWINOFFSETS
#define NOCOMM
#define NOKANJI
#define NOHELP
#define NOPROFILER
#define NODEFERWINDOWPOS
#define NOMCX*/

#include <windows.h>

#include "Haskell/FFI.h" 

#include "ParallelScript.h"

typedef void (* HsSystemFunc_t)();
typedef int (__stdcall * HsFunc_i__t)();
typedef int (__stdcall * HsFunc_i_i_t)(int);
typedef int (__stdcall * HsFunc_i_ii_t)(int, int);

class ParallelHaskell :
	public ParallelScript
{
public:
	// cons
		ParallelHaskell();
	
protected:
	virtual void
		ScriptInit(LPCWSTR, ParallelScript::NamedFunctions_s const * const) throw (...);
	
	virtual void
		ScriptClose(const bool) throw (...);
	
private:
	HMODULE
		m_module;
	
	HsSystemFunc_t
		HsStart,
		HsEnd;
};

#endif
