#include "Timings.h"

#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#include <cstring>

// cons
	Timings::
	Timings() :
		m_timings(*(new TimingList())),
		m_lst(*(new TimingList()))
{
}

void
	Timings::
	Log(
		const char * const
			msg,
		const char * const
			src)
{
	//std::cout << "LOG: " << src << ": " << msg << std::endl;
	LARGE_INTEGER
		time;
	QueryPerformanceCounter(&time);
	try
	{
		Timing *
			timing = new Timing(time.QuadPart, msg, src);
		m_timings.push_back(timing);
		m_lst.push_back(timing);
	}
	catch (...)
	{
	}
}

void
	Timings::
	End()
{
	//std::cout << "END" << std::endl;
	if (!m_lst.empty())
	{
		LARGE_INTEGER
			time;
		QueryPerformanceCounter(&time);
		Timing *
			lst = m_lst.back();
		m_lst.pop_back();
		lst->m_time = time.QuadPart - lst->m_time;
	}
}

// cons
	Timing::
	Timing(
		const __int64
			time,
		const char * const
			msg,
		const char * const
			src
		) throw (...) :
			m_msg(0),
			m_src(0),
			m_time(time)
{
	size_t
		len0 = strlen(msg),
		len1 = strlen(src);
	m_msg = (char *)malloc(len0 + 1);
	if (!m_msg)
	{
		throw "";
	}
	m_src = (char *)malloc(len1 + 1);
	if (!m_src)
	{
		free(m_msg);
		m_msg = 0;
		throw "";
	}
	memcpy(m_msg, msg, len0 + 1);
	memcpy(m_src, src, len1 + 1);
}

// cons
	Timing::
	~Timing()
{
	if (m_msg)
	{
		free(m_msg);
	}
	if (m_src)
	{
		free(m_src);
	}
}

Timing &
	Timing::
	operator=(const Timing &)
{
	return *this;
}

std::ostream &
	operator<<(
		std::ostream &
			str,
		const Timings &
			tim)
{
	for (Timings::TimingIter i = tim.m_timings.begin(); i != tim.m_timings.end(); ++i)
	{
		str << **i;// << std::endl;
	}
	tim.m_timings.clear();
	// This should be empty regardless.
	tim.m_lst.clear();
	return str;
}

char *
	g_lst = 0;

std::ostream &
	operator<<(
		std::ostream &
			str,
		const Timing &
			tim)
{
#if 0
	return str << "(" << tim.m_src << ") " << tim.m_msg << ": " << tim.m_time << std::endl;
#else
	if (tim.m_msg[0] == 'L')
	{
		// Loop variable, the main one.
		/*if (g_lst == 0)
		{
			g_lst = tim.m_src;
			str << g_lst;
		}
		else*/ if (g_lst == 0 || strcmp(g_lst, tim.m_src))
		{
			g_lst = tim.m_src;
			str << std::endl << g_lst;
		}
		return str << '\t' << tim.m_msg << '\t' << tim.m_time;
	}
	return str;
#endif
}
