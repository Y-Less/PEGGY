#ifndef _SOURCE_FFI_H
#define _SOURCE_FFI_H

typedef unsigned int HsChar;  // on 32 bit machine
typedef int HsInt;
typedef unsigned int HsWord;

// Ensure that we use C linkage for HsFunPtr 
extern "C"
{
	typedef
		void
		(* HsFunPtr)
		(void);
}

typedef
	void *
	HsPtr;

typedef
	void *
	HsForeignPtr;

typedef
	void *
	HsStablePtr;

#define HS_BOOL_FALSE 0
#define HS_BOOL_TRUE 1

#endif
