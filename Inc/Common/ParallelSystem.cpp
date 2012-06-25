#include "ParallelProcessor.h"

#include <cstring>

//#include <tracing.h>
//#include "ParallelSystem.tmh"

#include "Timings.h"

extern Timings
	gTimings;

// cons
	ParallelSystem::
	ParallelSystem() :
		m_itterations(0),
		/*m_start(0),
		m_end(0),*/
		m_name(0)
{
}

// cons
	ParallelSystem::
	ParallelSystem(
		const char * const
			name) :
		m_itterations(0),
		/*m_start(0),
		m_end(0),*/
		m_name(0)
{
	SetName(name);
}

// dest
	ParallelSystem::
	~ParallelSystem()
{
	if (m_name)
	{
		free(m_name);
	}
}

void
	ParallelSystem::
	Run(
		const bool
			setup)
{
	if (setup)
	{
		RunSetup();
	}
	else
	{
		RunClean();
	}
}

void
	ParallelSystem::
	RunSetup()
{
	// Includes array creation and destruction in the timings.
	//LARGE_INTEGER
	//	start,
	//	end;
	/*ParallelArrays::AcceleratorInternal::LogTime("Start foo");
	ParallelArrays::AcceleratorInternal::LogTime("Start bar");
	ParallelArrays::AcceleratorInternal::LogTime("End bar");
	ParallelArrays::AcceleratorInternal::LogTime("End foo");*/
	int
		t = m_itterations / 10;
	// Set up anything required in advance.
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): S Start");
	Log("Start");
	try
	{
		//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): S HWInit");
		Log("HWInit");
		try
		{
			HWInit(false);
		}
		catch (...)
		{
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S HWInit");
			End("HWInit");
			throw;
		}
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S HWInit");
		End("HWInit");
		//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): S Init");
		Log("Init");
		try
		{
			Init(false);
		}
		catch (...)
		{
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S Init");
			End("Init");
			throw;
		}
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S Init");
		End("Init");
		// Run the code once for speed's sake.
		//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): S EInit");
		Log("EInit");
		try
		{
			Execute();
		}
		catch (...)
		{
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S EInit");
			End("EInit");
			throw;
		}
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S EInit");
		End("EInit");
		Log("Close");
		try
		{
			Close(false);
		}
		catch (...)
		{
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S Close");
			End("Close");
			throw;
		}
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S Close");
		End("Close");
	}
	catch (...)
	{
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S Start");
		End("Start");
		throw;
	}
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S Start");
	End("Start");
	// Start timing.
	//QueryPerformanceCounter(&start);
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): S Loop");
	Log("Loop");
	try
	{
		for (int i = 0; i != m_itterations; ++i)
		{
			if (!--t)
			{
				std::cout << '|';
				t = m_itterations / 10;
			}
			// Run the code repeatedly.
			//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): S InIn");
			Log("InIn");
			try
			{
				Init(false);
			}
			catch (...)
			{
				//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S InIn");
				End("InIn");
				throw;
			}
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S InIn");
			End("InIn");
			//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): S InEx");
			Log("InEx");
			try
			{
				Execute();
			}
			catch (...)
			{
				//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S InEx");
				End("InEx");
				throw;
			}
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S InEx");
			End("InEx");
			//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): S InCl");
			Log("InCl");
			try
			{
				Close(false);
			}
			catch (...)
			{
				//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S InCl");
				End("InCl");
				throw;
			}
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S InCl");
			End("InCl");
		}
	}
	catch (...)
	{
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S Loop");
		End("Loop");
		throw;
	}
	// Stop timing.
	//QueryPerformanceCounter(&end);
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S Loop");
	End("Loop");
	Log("Shut");
	try
	{
		//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): S Close");
		//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): S HWClose");
		Log("HWClose");
		try
		{
			HWClose(false);
		}
		catch (...)
		{
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S HWClose");
			End("HWClose");
			throw;
		}
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S HWClose");
		End("HWClose");
	}
	catch (...)
	{
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S Shut");
		End("Shut");
		throw;
	}
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): S Shut");
	End("Shut");
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): S Shut");
	//m_start = start.QuadPart;
	//m_end = end.QuadPart;
	//std::cout << gTimings;
}

void
	ParallelSystem::
	RunClean()
{
	// Excludes array creation and destruction from the timings.
	//LARGE_INTEGER
	//	start,
	//	end;
	int
		t = m_itterations / 10;
	// Set up anything required in advance.
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): C Start");
	Log("Start");
	try
	{
		//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): C HWInit");
		Log("HWInit");
		try
		{
			HWInit(false);
		}
		catch (...)
		{
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C HWInit");
			End("HWInit");
			throw;
		}
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C HWInit");
		End("HWInit");
		//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): C Init");
		Log("Init");
		try
		{
			Init(false);
		}
		catch (...)
		{
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C Init");
			End("Init");
			throw;
		}
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C Init");
		End("Init");
		// Run the code once for speed's sake.
		//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): C EInit");
		Log("EInit");
		try
		{
			Execute();
		}
		catch (...)
		{
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C EInit");
			End("EInit");
			throw;
		}
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C EInit");
		End("EInit");
	}
	catch (...)
	{
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C Start");
		End("Start");
		throw;
	}
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C Start");
	End("Start");
	// Start timing.
	//QueryPerformanceCounter(&start);
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): C Loop");
	Log("Loop");
	try
	{
		for (int i = 0; i != m_itterations; ++i)
		{
			if (!--t)
			{
				std::cout << '|';
				t = m_itterations / 10;
			}
			// Run the code repeatedly.
			//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): C InEx");
			Log("InEx");
			try
			{
				Execute();
			}
			catch (...)
			{
				//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C InEx");
				End("InEx");
				throw;
			}
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C InEx");
			End("InEx");
		}
	}
	catch (...)
	{
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C Loop");
		End("Loop");
		throw;
	}
	// Stop timing.
	//QueryPerformanceCounter(&end);
	// Shut down anything requiring it.
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C Loop");
	End("Loop");
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): C Shut");
	Log("Shut");
	try
	{
		//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): C Close");
		Log("Close");
		try
		{
			Close(false);
		}
		catch (...)
		{
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C Close");
			End("Close");
			throw;
		}
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C Close");
		End("Close");
		//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): C HWClose");
		Log("HWClose");
		try
		{
			HWClose(false);
		}
		catch (...)
		{
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C HWClose");
			End("HWClose");
			throw;
		}
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C HWClose");
		End("HWClose");
	}
	catch (...)
	{
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C Shut");
		End("Shut");
		throw;
	}
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): C Shut");
	End("Shut");
	//m_start = start.QuadPart;
	//m_end = end.QuadPart;
	//std::cout << gTimings;
}

void
	ParallelSystem::
	Verify()
{
	// Set up anything required in advance.
	//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): Verify");
	Log("Verify");
	try
	{
		//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): V HWInit");
		Log("HWInit");
		try
		{
			HWInit(true);
		}
		catch (...)
		{
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): V HWInit");
			End("HWInit");
			throw;
		}
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): V HWInit");
		End("HWInit");
		//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): V Init");
		Log("Init");
		try
		{
			Init(true);
		}
		catch (...)
		{
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): V Init");
			End("Init");
			throw;
		}
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): V Init");
		End("Init");
		// Run the code once for speed's sake.
		//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): V EInit");
		Log("Execute");
		try
		{
			Execute();
		}
		catch (...)
		{
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): V EInit");
			End("Execute");
			throw;
		}
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): V EInit");
		End("Execute");
		//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): V Close");
		Log("Close");
		try
		{
			Close(true);
		}
		catch (...)
		{
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): V Close");
			End("Close");
			throw;
		}
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): V Close");
		End("Close");
		//DoTraceMessage(TIMING, "%s%s%s", "Start (", GetName(), "): V HWClose");
		Log("HWClose");
		try
		{
			HWClose(true);
		}
		catch (...)
		{
			//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): V HWClose");
			End("HWClose");
			throw;
		}
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): V HWClose");
		End("HWClose");
	}
	catch (...)
	{
		//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): Verify");
		End("Verify");
		throw;
	}
	//DoTraceMessage(TIMING, "%s%s%s", "End (", GetName(), "): Verify");
	End("Verify");
	//std::cout << gTimings;
}

void
	ParallelSystem::
	SetItterations(
		unsigned int
			itterations)
{
	m_itterations = itterations;
}

unsigned int
	ParallelSystem::
	GetItterations() const
{
	return m_itterations;
}

/*unsigned __int64
	ParallelSystem::
	GetTicks() const
{
	return m_end - m_start;
}

unsigned __int64
	ParallelSystem::
	GetFrequency()
{
	LARGE_INTEGER
		freq;
	QueryPerformanceFrequency(&freq);
	return freq.QuadPart;
}*/

void
	ParallelSystem::
	SetName(
		char const * const
			name)
{
	char *
		old = m_name;
	size_t
		len = strlen(name);
	m_name = (char *)malloc(len + 1);
	if (m_name)
	{
		memcpy(m_name, name, len + 1);
	}
	if (old)
	{
		free(old);
	}
}

char *
	ParallelSystem::
	GetName() const
{
	return m_name;
}

std::ostream &
	operator<<(
		std::ostream &
			str,
		const ParallelSystem &
			ps)
{
	if (ps.m_name)
	{
		return str << ps.m_name;
	}
	else
	{
		return str;
	}
}

void
	ParallelSystem::
	Log(
		char const * const
			msg) const
{
	/*size_t
		len0 = strlen(msg),
		len1 = strlen(m_name);
	char *
		t = (char *)malloc(len0 + len1 + 11);
	if (!t)
	{
		throw "";
	}
	//sprintf(t, "Start (%s): %s", m_name, msg);
	sprintf(t, "Start %s", msg);
	ParallelArrays::AcceleratorInternal::LogTime(t);
	free(t);*/
	//if (!strcmp(msg, "Loop"))
	//{
		gTimings.Log(msg, m_name);
	//}
	//	printf(msg);
}

void
	ParallelSystem::
	End(
		char const * const
			msg) const
{
	/*size_t
		len0 = strlen(msg),
		len1 = strlen(m_name);
	char *
		t = (char *)malloc(len0 + len1 + 9);
	if (!t)
	{
		throw "";
	}
	//sprintf(t, "End (%s): %s", m_name, msg);
	sprintf(t, "End %s", msg);
	ParallelArrays::AcceleratorInternal::LogTime(t);
	free(t);*/
	//if (!strcmp(msg, "Loop"))
	//{
		// How clever is the compiler when only passed constant strings?
		// Not clever enough it seems.
		gTimings.End();
	//}
}
