#ifndef _UTIL_CUH
#define _UTIL_CUH

/*! \file
 *  \brief Util functions that need to be at global scope, that could not be elsewhere.
 */

/**
 * \brief atomicMin for Floats
 *
 * There is not a implementation of atomicMin for floats, this is a possible
 * implementation found on Nvidia Forum that convert the float to int and use
 * atomicCas to do the magic.
 *
 * \param address Base float.
 * \param val comparison float.
 */
__device__ static float atomicMin(float* address, float val)
{
	int* address_as_i = (int*) address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_i, assumed,
		                __float_as_int(fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

__device__ static float atomicMax(float* address, float val)
{
	int* address_as_i = (int*) address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_i, assumed,
		                __float_as_int(fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

inline int ipow(int base, int exp)
{
	int result = 1;
	while (exp)
	{
		if (exp & 1)
			result *= base;
		exp >>= 1;
		base *= base;
	}

	return result;
}


template <typename T>
__host__ __device__ inline T cuda_min(T a, T b);

template <typename T>
__host__ __device__ inline T cuda_pow(T a, T b);

template <>
__host__ __device__ inline float cuda_min<float>(float a, float b)
{
	return fminf(a, b);
}

template <>
__host__ __device__ inline float cuda_pow<float>(float a, float b)
{
	return powf(a, b);
}



#endif
