#pragma once

#include <iostream>

#pragma warning (disable: 4250)

class ParallelSystem
{
public:
	// cons
		ParallelSystem();
	
	// cons
		ParallelSystem(const char * const);
	
	virtual // dest
		~ParallelSystem();
	
	void
		Run(const bool);
	
	void
		Verify();
	
	void
		SetItterations(unsigned int);
	
	unsigned int
		GetItterations() const;
	
	/*unsigned __int64
		GetTicks() const;
	
	static unsigned __int64
		GetFrequency();*/
	
	friend std::ostream &
		operator<<(std::ostream &, const ParallelSystem &);
	
protected:
	virtual void
		Init(const bool) throw (...) = 0;
	
	virtual void
		HWInit(const bool) throw (...) = 0;
	
	virtual void
		Execute() throw (...) = 0;
	
	// I'm sure that at some point in the past there was a reason why all these
	// function calls had bools on them to indicate verification, but it is not
	// used ANYWHERE any more!
	virtual void
		Close(const bool) throw (...) = 0;
	
	virtual void
		HWClose(const bool) throw (...) = 0;
	
	void
		SetName(char const * const);
	
	char *
		GetName(void) const;
	
	void
		Log(char const * const) const;
	
	void
		End(char const * const) const;
	
private:
	void
		RunSetup();
	
	void
		RunClean();
	
	unsigned int
		m_itterations;
	
	/*unsigned __int64
		m_start,
		m_end;*/
	
	char *
		m_name;
};
