#pragma once

#include <Templates/CommonProcess2D.h>
#include <Arrays/Data2DFloat.h>

#include "Filter.h"

class Convolver :
	public CommonProcess2D<Data2DFloat, float>
{
public:
	// cons
		Convolver();
	
	virtual // dest
		~Convolver();
	
	void
		SetFilter(Filter &) throw (...);
	
	void
		SetData(Data2DFloat &) throw (...);
	
protected:
	Filter &
		GetFilter() const throw (...);
	
	Data2DFloat &
		GetData() const throw (...);
	
	virtual void
		Init(const bool) throw (...);
	
	virtual void
		Close(const bool) throw (...);
	
	virtual void
		ConvInit() throw (...) = 0;
	
	virtual void
		ConvExit() throw (...)
	{
	};
	
	size_t
		GetRadius() const;
	
	float
		GetSigma() const;
	
	size_t
		GetWidth() const;
	
	size_t
		GetHeight() const;
	
	size_t
		GetPitch() const;
	
	Data2DFloat &
		GetSmoothX() const;
	
	Data2DFloat &
		GetSmoothY() const;
	
private:
	// cons
		Convolver(const Convolver &);
	
	Convolver &
		operator=(const Convolver &);
	
	Filter *
		m_filter;
	
	Data2DFloat
		* m_data,
		* m_smoothX,
		* m_smoothY;
};
