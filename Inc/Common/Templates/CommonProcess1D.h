#pragma once

#include "CommonProcess.h"

template <typename T, typename U>
class CommonProcess1D :
	public CommonProcess<T, U>
{
public:
	// cons
		CommonProcess1D()
		:
			CommonProcess()
	{
	};
	
protected:
	virtual void
		SetStore()
	{
		SetStoreP(new U [GetWidth()]);
	};

	virtual T *
		GenDest()
		const
	{
		return new T(GetWidth(), GetStore());
	};

	virtual size_t
		GetHeight() const
	{
		return 0;
	};
};
