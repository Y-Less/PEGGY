#pragma once

#include "CommonProcess.h"

template <typename T, typename U>
class CommonProcess2D :
	public CommonProcess<T, U>
{
public:
	// cons
		CommonProcess2D()
		:
			CommonProcess()
	{
	};
	
protected:
	virtual void
		SetStore()
	{
		SetStoreP(new U [GetHeight() * GetWidth()]);
	};
	
	virtual T *
		GenDest()
		const
	{
		return new T(GetHeight(), GetWidth(), GetStore());
	};
};
