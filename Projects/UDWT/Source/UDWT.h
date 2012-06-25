#pragma once

#include <Templates/CommonProcess1D.h>
#include <Arrays/Data1DFloat.h>

#include "Filter.h"

class UDWT :
	public CommonProcess1D<Data1DFloat, float>
	//public ParallelCode
{
public:
	// cons
		UDWT();
	
	virtual // dest
		~UDWT();
	
	void
		SetFilter(Filter &) throw (...);
	
	void
		SetLevel(size_t);
	
	void
		SetData(Data1DFloat &) throw (...);
	
protected:
	Filter &
		GetFilter() const throw (...);
	
	Data1DFloat &
		GetForward1() const throw (...);
	
	Data1DFloat &
		GetReverse1() const throw (...);
	
	Data1DFloat &
		GetForward2() const throw (...);
	
	Data1DFloat &
		GetReverse2() const throw (...);
	
	size_t
		GetLevel() const;
	
	Data1DFloat &
		GetData() const throw (...);
	
	virtual void
		Init(const bool) throw (...);
	
	virtual void
		Close(const bool) throw (...);
	
	virtual void
		ConvInit() throw (...) = 0;
	
	virtual void
		ConvClose() throw (...) = 0;
	
	size_t
		GetRadius() const;
	
	size_t
		GetWidth() const;
	
	//Data2DFloat &
	//	GetSmoothX() const;
	
	//Data2DFloat &
	//	GetSmoothY() const;
	
	Data1DFloat &
		GetLowPass() const;
	
	Data1DFloat &
		GetHighPass() const;
	
//private:
protected:
	// cons
		UDWT(const UDWT &);
	
	UDWT &
		operator=(const UDWT &);
	
	Filter *
		m_filter;
	
	Data1DFloat
		//* m_data2,
		* m_data,
		* m_lowPass,
		* m_highPass;
	
	size_t
		m_level;
};
