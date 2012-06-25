#pragma once

#include "ParallelCode.h"

template <typename T, typename U>
class CommonProcess :
	public ParallelCode
{
public:
	// cons
		CommonProcess()
		:
			m_store(0),
			m_result(0)
	{
	};
	
	virtual // dest
		~CommonProcess()
	{
		delete m_result;
		delete [] m_store;
	};
	
	virtual DataStore &
		GetResult()
		const throw (...)
	{
		if (!m_result)
		{
			throw "No results";
		}
		return *m_result;
	};
	
protected:
	U *
		GetStore()
		const
	{
		return m_store;
	};
	
	virtual void
		SetStore() = 0;
	
	void
		SetStoreP(
			U *                        store)
	{
		m_store = store;
	};
	
	virtual size_t
		GetHeight() const = 0;
	
	virtual size_t
		GetWidth() const = 0;
	
	virtual	T *
		GenDest() const = 0;

	void
		CloseCommon(
			const bool                 save)
		throw (...)
	{
		if (save)
		{
			// Save the final data.
			if (m_result)
			{
				delete m_result;
			}
			m_result = GenDest();
		}
	};
	
private:
	T *
		m_result;
	
	U *
		m_store;
};
