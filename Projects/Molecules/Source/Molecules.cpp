#include "Molecules.h"

// cons
	Molecules::
	Molecules()
	:
		CommonProcess2D(),
		m_atoms(0),
		m_height(0),
		m_width(0)
{
}

// dest
	Molecules::
	~Molecules()
{
}

void
	Molecules::
	SetGridSize(
		const size_t                   height,
		const size_t                   width)
{
	m_height = height;
	m_width = width;
	SetStore();
}

void
	Molecules::
	SetAtoms(
		Atoms &                        atoms)
{
	m_atoms = &atoms;
}

size_t
	Molecules::
	GetHeight()
	const
{
	return m_height;
}

size_t
	Molecules::
	GetWidth()
	const
{
	return m_width;
}

Atoms &
	Molecules::
	GetAtoms()
	const
{
	return *m_atoms;
}

size_t
	Molecules::
	GetCount()
	const
{
	return m_atoms->GetCount();
}

void
	Molecules::
	Close(
		const bool                     save)
	throw (...)
{
	// Clean up the data we don't want.
	GetAtoms().CleanGPU();
	CloseCommon(save);
}
