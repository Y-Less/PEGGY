#include <deque>

#include <iostream>

#include "ParallelSystem.h"

class Timing;

class Timings
{
public:
	typedef
		std::deque<Timing * const>
		TimingList;
	
	typedef
		TimingList::const_iterator
		TimingIter;
	
	// cons
		Timings();
	
	void
		Log(const char * const, const char * const);
	
	void
		End();
	
	friend std::ostream &
		operator<<(std::ostream &, const Timings &);
	
private:
	TimingList
		& m_timings,
		& m_lst;
};

class Timing
{
private:
	// cons
		Timing(const __int64, const char * const, const char * const) throw (...);
		
	// dest
		~Timing();
	
	Timing &
		operator=(const Timing &);
	
	__int64
		m_time;
		
	char
		* m_src,
		* m_msg;
	
	friend std::ostream &
		operator<<(std::ostream &, const Timing &);
	
	friend class Timings;
};
