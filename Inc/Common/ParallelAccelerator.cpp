#if !defined NO_ACCELERATOR || !defined NO_ACCELERATOR_OPT

#include "ParallelAccelerator.h"

#include <D3D9.h>
#include <D3DX9.h>

#include <MulticoreTarget.h>
//#include <DX11Target.h>
#include <DX9Target.h>
#include <CUDATarget.h>

// cons
	ParallelAccelerator::
	ParallelAccelerator(
		const ParallelAcceleratorType
			type
		) :
		m_type(type),
		m_target(0)
{
	switch (type)
	{
		case PAT_DX9:
			SetName("DX9 ");
			break;
		case PAT_DX10:
			SetName("DX10");
			break;
		case PAT_DX11:
			SetName("DX11");
			break;
		case PAT_X64:
			SetName("X64 ");
			break;
		case PAT_DIRECT:
			SetName("DIR ");
			break;
		case PAT_AC_C:
			SetName("AC_C");
			break;
	}
}

ParallelArrays::Target &
	ParallelAccelerator::
	GetTarget() const throw (...)
{
	if (!m_target)
	{
		throw "Target not initialised";
	}
	return *m_target;
}

void
	ParallelAccelerator::
	HWInit(const bool) throw (...)
{
	//DestroyTarget();
	switch (m_type)
	{
		case PAT_X64:
			SetTarget(MicrosoftTargets::CreateMulticoreTarget());
			break;
		/*case PAT_DX11:
			SetTarget(&MicrosoftTargets::CreateDX11Target());
			break;*/
		/*case PAT_DX10:
			SetTarget(&MicrosoftTargets::CreateDX10Target());
			break;*/
		case PAT_DX9:
			SetTarget(MicrosoftTargets::CreateDX9Target());
			break;
		case PAT_AC_C:
			SetTarget(MicrosoftTargets::CreateCUDATarget());
			break;
		/*case PAT_DIRECT:
			SetTarget(&MicrosoftTargets::CreateDirectComputeTarget());
			break;*/
		default:
			throw "Unsupported target";
			break;
	}
}

void
	ParallelAccelerator::
	HWClose(const bool) throw (...)
{
	// Destroy the target.
	DestroyTarget();
}

void
	ParallelAccelerator::
	SetTarget(
		ParallelArrays::Target *
			target) throw (...)
{
	m_target = target;
}

void
	ParallelAccelerator::
	DestroyTarget() throw (...)
{
	if (m_target)
	{
		m_target->Delete();
		m_target = 0;
	}
}

#endif
