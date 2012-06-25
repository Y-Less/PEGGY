#include "Convolver.h"

// cons
	Convolver::
	Convolver()
	:
		CommonProcess2D(),
		m_filter(0),
		m_data(0),
		m_smoothX(0),
		m_smoothY(0)
{
}

// dest
	Convolver::
	~Convolver()
{
	delete m_smoothX;
	delete m_smoothY;
}

Filter &
	Convolver::
	GetFilter() const throw (...)
{
	if (!m_filter)
	{
		throw "Uninitialised data";
	}
	return *m_filter;
}

Data2DFloat &
	Convolver::
	GetData() const throw (...)
{
	if (!m_data)
	{
		throw "Uninitialised data";
	}
	return *m_data;
}

void
	Convolver::
	SetFilter(
		Filter &
			filter
		) throw (...)
{
	if (m_filter)
	{
		throw "Reinitialised data";
	}
	m_filter = &filter;
}

void
	Convolver::
	SetData(
		Data2DFloat &
			data
		) throw (...)
{
	if (m_data)
	{
		throw "Reinitialised data";
	}
	m_data = &data;
	// Initialise the target outside the main loop.
	SetStore();
	m_smoothX = new Data2DFloat(GetHeight(), GetWidth(), GT_None);
	m_smoothY = new Data2DFloat(GetHeight(), GetWidth(), GT_None);
}

void
	Convolver::
	Init(const bool) throw (...)
{
	//std::cout << std::endl << "Near init common";
	ConvInit();
	//std::cout << std::endl << "Done init common";
}

void
	Convolver::
	Close(
		const bool
			save
		) throw (...)
{
	//std::cout << std::endl << "Near close common";
	// Clean up the data we don't want.
	GetFilter().CleanGPU();
	//std::cout << std::endl << "1";
	GetSmoothX().CleanGPU();
	//std::cout << std::endl << "0";
	GetData().CleanGPU();
	//std::cout << std::endl << "2";
	GetSmoothY().CleanGPU();
	//std::cout << std::endl << "Call close common";
	CloseCommon(save);
	//std::cout << std::endl << "Done close common";
	ConvExit();
}

size_t
	Convolver::
	GetWidth() const
{
	return m_data->GetWidth();
}

size_t
	Convolver::
	GetHeight() const
{
	return m_data->GetHeight();
}

size_t
	Convolver::
	GetPitch() const
{
	return m_data->GetPitch();
}

size_t
	Convolver::
	GetRadius() const
{
	return m_filter->GetRadius();
}

float
	Convolver::
	GetSigma() const
{
	return m_filter->GetSigma();
}

Data2DFloat &
	Convolver::
	GetSmoothX() const
{
	return *m_smoothX;
}

Data2DFloat &
	Convolver::
	GetSmoothY() const
{
	return *m_smoothY;
}
