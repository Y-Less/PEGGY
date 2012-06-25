
__device__ __constant__ float
	gc_fAtoms[200][4];

// CUDA values.  This is constant for now as all known CUDA implementations
// have a warp size of 32 and it makes all the calculations constant.
#define WARP_SIZE                      (32) //(warpSize)
#define HALF_WARP                      (WARP_SIZE / 2)

#define CEILDIV(m,n) \
	(((m) + (n) - 1) / (n))

#define THREAD_CALCS                   (6)

extern "C"
void
	CopyAtoms(
		float *
			atoms,
		int
			atomsCount)
{
	cudaMemcpyToSymbol(gc_fAtoms, atoms, atomsCount * 4 * sizeof (float));
}

__global__ void
	DoAtomsKernel(
		float *                        p_pfGrid,
		int                            p_iHeight,
		int                            p_iWidth,
		int                            p_iPitch,
		int                            p_iAtomCount)
{
	// This version only calculates one slice, not a full grid.  blockDim.x is
	// HALF_WARP.
	const int
		iy = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (iy >= p_iHeight)
	{
		return;
	}
	const int
		bx = blockIdx.x * HALF_WARP * THREAD_CALCS,
		ix = bx + threadIdx.x;
	// The original version of this code assumed the data was an exact multiple
	// of THREAD_CALCS * HALF_WARP (128) wide - this may not always be the case.
	const float
		x = ix,
		y = iy;
	float * const
		outp = &p_pfGrid[(iy * p_iPitch) + ix];
	float
		ev0 = 0,
		ev1 = 0,
		ev2 = 0,
		ev3 = 0,
		ev4 = 0,
		ev5 = 0;
	for (int atom = 0; atom != p_iAtomCount; ++atom)
	{
		// All the points being done are in a line so y and z are constant.
		const float
			fy = y - gc_fAtoms[atom][1],
			fz = 1.0f - gc_fAtoms[atom][2],
			offset = (fy * fy) + (fz * fz),
			charge = gc_fAtoms[atom][3];
		// Do the same code THREAD_CALCS times.
		const float
			x0 = x - gc_fAtoms[atom][0],
			x1 = x0 + HALF_WARP * 1,
			x2 = x0 + HALF_WARP * 2,
			x3 = x0 + HALF_WARP * 3,
			x4 = x0 + HALF_WARP * 4,
			x5 = x0 + HALF_WARP * 5;
		// Add the effect of this charge to the running total of all current
		// points being compared.
		ev0 += charge * rsqrt((x0 * x0) + offset);
		ev1 += charge * rsqrt((x1 * x1) + offset);
		ev2 += charge * rsqrt((x2 * x2) + offset);
		ev3 += charge * rsqrt((x3 * x3) + offset);
		ev4 += charge * rsqrt((x4 * x4) + offset);
		ev5 += charge * rsqrt((x5 * x5) + offset);
	}
	// We don't need any complex code to half-warp align the writes as
	// we can just launch the threads cleverly to do it all for us.
	if ((p_iWidth - bx) < (HALF_WARP * THREAD_CALCS))
	{
		// If we are in here then the number of grid points is not an exact
		// multiple of blockDim.x * 8, so some results will be discarded.
		// However, calculating and discarding results is probably the best
		// method for speed sake in the large majority of cases.
		if (ix + HALF_WARP * 0 < p_iWidth)
		{
			outp[HALF_WARP * 0] = ev0;
			// The heirarchical structure should mean that warps are better
			// processed.
			if (ix + HALF_WARP * 1 < p_iWidth)
			{
				outp[HALF_WARP * 1] = ev1;
				if (ix + HALF_WARP * 2 < p_iWidth)
				{
					outp[HALF_WARP * 2] = ev2;
					if (ix + HALF_WARP * 3 < p_iWidth)
					{
						outp[HALF_WARP * 3] = ev3;
						if (ix + HALF_WARP * 4 < p_iWidth)
						{
							outp[HALF_WARP * 4] = ev4;
							if (ix + HALF_WARP * 5 < p_iWidth)
							{
								outp[HALF_WARP * 5] = ev5;
							}
						}
					}
				}
			}
		}
	}
	else
	{
		// Fast code for the common case - based on code from "Programming
		// Massively Parallel Processors".
		outp[HALF_WARP * 0] = ev0;
		outp[HALF_WARP * 1] = ev1;
		outp[HALF_WARP * 2] = ev2;
		outp[HALF_WARP * 3] = ev3;
		outp[HALF_WARP * 4] = ev4;
		outp[HALF_WARP * 5] = ev5;
	}
}

extern "C"
void
	DoAtoms(
		float *                        p_pfGrid,
		int                            p_iHeight,
		int                            p_iWidth,
		int                            p_iThreads,
		int                            p_iPitch,
		int                            p_iAtoms)
{
	dim3
		// Number of blocks to execute in.
		dimBlocks(CEILDIV(p_iWidth, HALF_WARP * THREAD_CALCS), CEILDIV(p_iHeight, p_iThreads / HALF_WARP)),
		// Number of threads per block.
		dimThreads(HALF_WARP, p_iThreads / HALF_WARP);
	DoAtomsKernel<<<dimBlocks, dimThreads>>>(p_pfGrid, p_iHeight, p_iWidth, p_iPitch, p_iAtoms);
}
