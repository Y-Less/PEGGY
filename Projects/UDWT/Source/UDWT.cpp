#include "UDWT.h"

// cons
	UDWT::
	UDWT()
	:
		CommonProcess1D(),
		m_filter(0),
		m_data(0),
		m_lowPass(0),
		m_highPass(0),
		m_level(0)
{
}

// dest
	UDWT::
	~UDWT()
{
	delete m_lowPass;
	delete m_highPass;
}

Filter &
	UDWT::
	GetFilter() const throw (...)
{
	if (!m_filter)
	{
		throw "Uninitialised data 1";
	}
	return *m_filter;
}

Data1DFloat &
	UDWT::
	GetForward1() const throw (...)
{
	if (!m_filter)
	{
		throw "Uninitialised data 2";
	}
	return m_filter->GetForward1();
}

Data1DFloat &
	UDWT::
	GetReverse1() const throw (...)
{
	if (!m_filter)
	{
		throw "Uninitialised data 3";
	}
	return m_filter->GetReverse1();
}

Data1DFloat &
	UDWT::
	GetForward2() const throw (...)
{
	if (!m_filter)
	{
		throw "Uninitialised data 4";
	}
	return m_filter->GetForward2();
}

Data1DFloat &
	UDWT::
	GetReverse2() const throw (...)
{
	if (!m_filter)
	{
		throw "Uninitialised data 5";
	}
	return m_filter->GetReverse2();
}

Data1DFloat &
	UDWT::
	GetData() const throw (...)
{
	if (!m_data)
	{
		throw "Uninitialised data 6";
	}
	return *m_data;
}

void
	UDWT::
	SetLevel(
		size_t
			level
		)
{
	m_level = level;
}

size_t
	UDWT::
	GetLevel() const
{
	return m_level;
}

void
	UDWT::
	SetFilter(
		Filter &
			filter
		) throw (...)
{
	if (m_filter)
	{
		throw "Reinitialised data 7";
	}
	m_filter = &filter;
}

void
	UDWT::
	SetData(
		Data1DFloat &
			data
		) throw (...)
{
	if (m_data)
	{
		throw "Reinitialised data 8";
	}
	m_data = &data;
	//m_data2 = new Data1DFloat(4032 * ((GetWidth() / 4031) + 1) * GetLevel(), GT_Random);
	// Initialise the target outside the main loop.
	SetStore();
	m_lowPass = new Data1DFloat(GetWidth(), GT_Garbage);
	m_highPass = new Data1DFloat(GetWidth() * GetLevel(), GT_Garbage);
	//m_smoothX = new Data2DFloat(GetHeight(), GetWidth(), GT_None);
	//m_smoothY = new Data2DFloat(GetHeight(), GetWidth(), GT_None);
}

void
	UDWT::
	Init(const bool) throw (...)
{
	//std::cout << std::endl << "Near init common";
	ConvInit();
	//std::cout << std::endl << "Done init common";
}

void
	UDWT::
	Close(
		const bool
			save
		) throw (...)
{
	//std::cout << std::endl << "Near close common";
	// Clean up the data we don't want.
	GetFilter().CleanGPU();
	//std::cout << std::endl << "1";
	//GetLowPass().CleanGPU();
	//std::cout << std::endl << "0";
	GetData().CleanGPU();
	//std::cout << std::endl << "2";
	//GetHighPass().CleanGPU();
	//std::cout << std::endl << "Call close common";
	CloseCommon(save);
	//std::cout << std::endl << "Done close common";
	ConvClose();
}

size_t
	UDWT::
	GetWidth() const
{
	return m_data->GetWidth();
}

size_t
	UDWT::
	GetRadius() const
{
	return m_filter->GetRadius();
}

Data1DFloat &
	UDWT::
	GetLowPass() const
{
	return *m_lowPass;
}

Data1DFloat &
	UDWT::
	GetHighPass() const
{
	return *m_highPass;
}
