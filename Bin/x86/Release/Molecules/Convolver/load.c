
#include <Windows.h>
#include <rts.h>

//extern void __stginit_MorphologicalFilter(void);

//#include "exports.h"

//void __attribute__ ((dllexport))
void __cdecl
	HsStart()
{
	// Initialise all external functions.
	int
		argc = 1;
	char *
		argv[] = {"ghcDll", NULL};
	char **
		args = argv;
	hs_init(&argc, &args);
	//hs_add_root(__stginit_MorphologicalFilter);
	//HsInitCallbacks(NULL);
}

//void __attribute__ ((dllexport))
void __cdecl
	HsEnd()
{
	hs_exit();
}

BOOL APIENTRY DllMain(
	HANDLE hModule, 
	DWORD  ul_reason_for_call, 
	LPVOID lpReserved)
{
	return TRUE;
}
