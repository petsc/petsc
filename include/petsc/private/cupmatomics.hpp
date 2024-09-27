#pragma once

/*====================================================================================*/
/*                             Atomic operations on device                            */
/*====================================================================================*/
#include <petscdevice_cupm.h>
#include <petscsystypes.h>

/* In terms of function overloading, long long int is a different type than int64_t, which PetscInt might be defined to.
   We prefer long long int over PetscInt (int64_t), since CUDA atomics are built around (unsigned) long long int.
 */
typedef long long int          llint;
typedef unsigned long long int ullint;

#if PetscDefined(USING_NVCC)
PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN("-Wunused-function")
/*
  Atomic Insert (exchange) operations

  CUDA C Programming Guide V10.1 Chapter B.12.1.3:

  int atomicExch(int* address, int val);
  unsigned int atomicExch(unsigned int* address, unsigned int val);
  unsigned long long int atomicExch(unsigned long long int* address, unsigned long long int val);
  float atomicExch(float* address, float val);

  reads the 32-bit or 64-bit word old located at the address in global or shared
  memory and stores val back to memory at the same address. These two operations are
  performed in one atomic transaction. The function returns old.

  PETSc notes:

  It may be useful in PetscSFFetchAndOp with op = MPI_REPLACE.

  VecScatter with multiple entries scattered to the same location using INSERT_VALUES does not need
  atomic insertion, since it does not need the old value. A 32-bit or 64-bit store instruction should
  be atomic itself.

  With bs>1 and a unit > 64-bits, the current element-wise atomic approach can not guarantee the whole
  insertion is atomic. Hope no user codes rely on that.
*/
__device__ static double atomicExch(double *address, double val)
{
  return __longlong_as_double(atomicExch((ullint *)address, __double_as_longlong(val)));
}

__device__ static llint atomicExch(llint *address, llint val)
{
  return (llint)(atomicExch((ullint *)address, (ullint)val));
}

template <typename Type>
struct AtomicInsert {
  __device__ Type operator()(Type &x, Type y) const { return atomicExch(&x, y); }
};

  #if defined(PETSC_HAVE_COMPLEX)
    #if defined(PETSC_USE_REAL_DOUBLE)
/* CUDA does not support 128-bit atomics. Users should not insert different 128-bit PetscComplex values to the same location */
template <>
struct AtomicInsert<PetscComplex> {
  __device__ PetscComplex operator()(PetscComplex &x, PetscComplex y) const
  {
    PetscComplex         old, *z = &old;
    double              *xp = (double *)&x, *yp = (double *)&y;
    AtomicInsert<double> op;
    z[0] = op(xp[0], yp[0]);
    z[1] = op(xp[1], yp[1]);
    return old; /* The returned value may not be atomic. It can be mix of two ops. Caller should discard it. */
  }
};
    #elif defined(PETSC_USE_REAL_SINGLE)
template <>
struct AtomicInsert<PetscComplex> {
  __device__ PetscComplex operator()(PetscComplex &x, PetscComplex y) const
  {
    double              *xp = (double *)&x, *yp = (double *)&y;
    AtomicInsert<double> op;
    return op(xp[0], yp[0]);
  }
};
    #endif
  #endif

/*
  Atomic add operations

  CUDA C Programming Guide V10.1 Chapter B.12.1.1:

  int atomicAdd(int* address, int val);
  unsigned int atomicAdd(unsigned int* address,unsigned int val);
  unsigned long long int atomicAdd(unsigned long long int* address,unsigned long long int val);
  float atomicAdd(float* address, float val);
  double atomicAdd(double* address, double val);
  __half2 atomicAdd(__half2 *address, __half2 val);
  __half atomicAdd(__half *address, __half val);

  reads the 16-bit, 32-bit or 64-bit word old located at the address in global or shared memory, computes (old + val),
  and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The
  function returns old.

  The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.
  The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.
  The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and
  higher. The atomicity of the __half2 add operation is guaranteed separately for each of the two __half elements;
  the entire __half2 is not guaranteed to be atomic as a single 32-bit access.
  The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
*/
__device__ static llint atomicAdd(llint *address, llint val)
{
  return (llint)atomicAdd((ullint *)address, (ullint)val);
}

template <typename Type>
struct AtomicAdd {
  __device__ Type operator()(Type &x, Type y) const { return atomicAdd(&x, y); }
};

template <>
struct AtomicAdd<double> {
  __device__ double operator()(double &x, double y) const
  {
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
    return atomicAdd(&x, y);
  #else
    double *address = &x, val = y;
    ullint *address_as_ull = (ullint *)address;
    ullint  old            = *address_as_ull, assumed;
    do {
      assumed = old;
      old     = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
      /* Note: uses integer comparison to avoid hang in case of NaN (since NaN !=NaN) */
    } while (assumed != old);
    return __longlong_as_double(old);
  #endif
  }
};

template <>
struct AtomicAdd<float> {
  __device__ float operator()(float &x, float y) const
  {
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
    return atomicAdd(&x, y);
  #else
    float *address = &x, val = y;
    int   *address_as_int = (int *)address;
    int    old            = *address_as_int, assumed;
    do {
      assumed = old;
      old     = atomicCAS(address_as_int, assumed, __float_as_int(val + __int_as_float(assumed)));
      /* Note: uses integer comparison to avoid hang in case of NaN (since NaN !=NaN) */
    } while (assumed != old);
    return __int_as_float(old);
  #endif
  }
};

  #if defined(PETSC_HAVE_COMPLEX)
template <>
struct AtomicAdd<PetscComplex> {
  __device__ PetscComplex operator()(PetscComplex &x, PetscComplex y) const
  {
    PetscComplex         old, *z = &old;
    PetscReal           *xp = (PetscReal *)&x, *yp = (PetscReal *)&y;
    AtomicAdd<PetscReal> op;
    z[0] = op(xp[0], yp[0]);
    z[1] = op(xp[1], yp[1]);
    return old; /* The returned value may not be atomic. It can be mix of two ops. Caller should discard it. */
  }
};
  #endif

  /*
  Atomic Mult operations:

  CUDA has no atomicMult at all, so we build our own with atomicCAS
 */
  #if defined(PETSC_USE_REAL_DOUBLE)
__device__ static double atomicMult(double *address, double val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    /* Other threads can access and modify value of *address_as_ull after the read above and before the write below */
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val * __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
  #elif defined(PETSC_USE_REAL_SINGLE)
__device__ static float atomicMult(float *address, float val)
{
  int *address_as_int = (int *)(address);
  int  old            = *address_as_int, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_int, assumed, __float_as_int(val * __int_as_float(assumed)));
  } while (assumed != old);
  return __int_as_float(old);
}
  #endif

__device__ static int atomicMult(int *address, int val)
{
  int *address_as_int = (int *)(address);
  int  old            = *address_as_int, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_int, assumed, val * assumed);
  } while (assumed != old);
  return (int)old;
}

__device__ static llint atomicMult(llint *address, llint val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, (ullint)(val * (llint)assumed));
  } while (assumed != old);
  return (llint)old;
}

template <typename Type>
struct AtomicMult {
  __device__ Type operator()(Type &x, Type y) const { return atomicMult(&x, y); }
};

/*
  Atomic Min/Max operations

  CUDA C Programming Guide V10.1 Chapter B.12.1.4~5:

  int atomicMin(int* address, int val);
  unsigned int atomicMin(unsigned int* address,unsigned int val);
  unsigned long long int atomicMin(unsigned long long int* address,unsigned long long int val);

  reads the 32-bit or 64-bit word old located at the address in global or shared
  memory, computes the minimum of old and val, and stores the result back to memory
  at the same address. These three operations are performed in one atomic transaction.
  The function returns old.
  The 64-bit version of atomicMin() is only supported by devices of compute capability 3.5 and higher.

  atomicMax() is similar.
 */

  #if defined(PETSC_USE_REAL_DOUBLE)
__device__ static double atomicMin(double *address, double val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, __double_as_longlong(PetscMin(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__ static double atomicMax(double *address, double val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, __double_as_longlong(PetscMax(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}
  #elif defined(PETSC_USE_REAL_SINGLE)
__device__ static float atomicMin(float *address, float val)
{
  int *address_as_int = (int *)(address);
  int  old            = *address_as_int, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_int, assumed, __float_as_int(PetscMin(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ static float atomicMax(float *address, float val)
{
  int *address_as_int = (int *)(address);
  int  old            = *address_as_int, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_int, assumed, __float_as_int(PetscMax(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}
  #endif

  /*
  atomicMin/Max(long long *, long long) are not in Nvidia's documentation. But on OLCF Summit we found
  atomicMin/Max/And/Or/Xor(long long *, long long) in /sw/summit/cuda/10.1.243/include/sm_32_atomic_functions.h.
  This causes compilation errors with pgi compilers and 64-bit indices:
      error: function "atomicMin(long long *, long long)" has already been defined

  So we add extra conditions defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 320)
*/
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 320)
__device__ static llint atomicMin(llint *address, llint val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, (ullint)(PetscMin(val, (llint)assumed)));
  } while (assumed != old);
  return (llint)old;
}

__device__ static llint atomicMax(llint *address, llint val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, (ullint)(PetscMax(val, (llint)assumed)));
  } while (assumed != old);
  return (llint)old;
}
  #endif

template <typename Type>
struct AtomicMin {
  __device__ Type operator()(Type &x, Type y) const { return atomicMin(&x, y); }
};
template <typename Type>
struct AtomicMax {
  __device__ Type operator()(Type &x, Type y) const { return atomicMax(&x, y); }
};

/*
  Atomic bitwise operations

  CUDA C Programming Guide V10.1 Chapter B.12.2.1 ~ B.12.2.3:

  int atomicAnd(int* address, int val);
  unsigned int atomicAnd(unsigned int* address,unsigned int val);
  unsigned long long int atomicAnd(unsigned long long int* address,unsigned long long int val);

  reads the 32-bit or 64-bit word old located at the address in global or shared
  memory, computes (old & val), and stores the result back to memory at the same
  address. These three operations are performed in one atomic transaction.
  The function returns old.

  The 64-bit version of atomicAnd() is only supported by devices of compute capability 3.5 and higher.

  atomicOr() and atomicXor are similar.
*/

  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 320) /* Why 320? see comments at atomicMin() above */
__device__ static llint atomicAnd(llint *address, llint val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, (ullint)(val & (llint)assumed));
  } while (assumed != old);
  return (llint)old;
}
__device__ static llint atomicOr(llint *address, llint val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, (ullint)(val | (llint)assumed));
  } while (assumed != old);
  return (llint)old;
}

__device__ static llint atomicXor(llint *address, llint val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, (ullint)(val ^ (llint)assumed));
  } while (assumed != old);
  return (llint)old;
}
  #endif

template <typename Type>
struct AtomicBAND {
  __device__ Type operator()(Type &x, Type y) const { return atomicAnd(&x, y); }
};
template <typename Type>
struct AtomicBOR {
  __device__ Type operator()(Type &x, Type y) const { return atomicOr(&x, y); }
};
template <typename Type>
struct AtomicBXOR {
  __device__ Type operator()(Type &x, Type y) const { return atomicXor(&x, y); }
};

/*
  Atomic logical operations:

  CUDA has no atomic logical operations at all. We support them on integer types.
*/

/* A template without definition makes any instantiation not using given specializations erroneous at compile time,
   which is what we want since we only support 32-bit and 64-bit integers.
 */
template <typename Type, class Op, int size /* sizeof(Type) */>
struct AtomicLogical;

template <typename Type, class Op>
struct AtomicLogical<Type, Op, 4> {
  __device__ Type operator()(Type &x, Type y) const
  {
    int *address_as_int = (int *)(&x);
    int  old            = *address_as_int, assumed;
    Op   op;
    do {
      assumed = old;
      old     = atomicCAS(address_as_int, assumed, (int)(op((Type)assumed, y)));
    } while (assumed != old);
    return (Type)old;
  }
};

template <typename Type, class Op>
struct AtomicLogical<Type, Op, 8> {
  __device__ Type operator()(Type &x, Type y) const
  {
    ullint *address_as_ull = (ullint *)(&x);
    ullint  old            = *address_as_ull, assumed;
    Op      op;
    do {
      assumed = old;
      old     = atomicCAS(address_as_ull, assumed, (ullint)(op((Type)assumed, y)));
    } while (assumed != old);
    return (Type)old;
  }
};

/* Note land/lor/lxor below are different from LAND etc above. Here we pass arguments by value and return result of ops (not old value) */
template <typename Type>
struct land {
  __device__ Type operator()(Type x, Type y) { return x && y; }
};
template <typename Type>
struct lor {
  __device__ Type operator()(Type x, Type y) { return x || y; }
};
template <typename Type>
struct lxor {
  __device__ Type operator()(Type x, Type y) { return !x != !y; }
};

template <typename Type>
struct AtomicLAND {
  __device__ Type operator()(Type &x, Type y) const
  {
    AtomicLogical<Type, land<Type>, sizeof(Type)> op;
    return op(x, y);
  }
};
template <typename Type>
struct AtomicLOR {
  __device__ Type operator()(Type &x, Type y) const
  {
    AtomicLogical<Type, lor<Type>, sizeof(Type)> op;
    return op(x, y);
  }
};
template <typename Type>
struct AtomicLXOR {
  __device__ Type operator()(Type &x, Type y) const
  {
    AtomicLogical<Type, lxor<Type>, sizeof(Type)> op;
    return op(x, y);
  }
};
PETSC_PRAGMA_DIAGNOSTIC_IGNORED_END()
#elif PetscDefined(USING_HCC)

  /*
  Atomic Insert (exchange) operations

  See Cuda version
*/
  #if PETSC_PKG_HIP_VERSION_LT(4, 4, 0)
__device__ static double atomicExch(double *address, double val)
{
  return __longlong_as_double(atomicExch((ullint *)address, __double_as_longlong(val)));
}
  #endif

__device__ static inline llint atomicExch(llint *address, llint val)
{
  return (llint)(atomicExch((ullint *)address, (ullint)val));
}

template <typename Type>
struct AtomicInsert {
  __device__ Type operator()(Type &x, Type y) const { return atomicExch(&x, y); }
};

  #if defined(PETSC_HAVE_COMPLEX)
    #if defined(PETSC_USE_REAL_DOUBLE)
template <>
struct AtomicInsert<PetscComplex> {
  __device__ PetscComplex operator()(PetscComplex &x, PetscComplex y) const
  {
    PetscComplex         old, *z = &old;
    double              *xp = (double *)&x, *yp = (double *)&y;
    AtomicInsert<double> op;
    z[0] = op(xp[0], yp[0]);
    z[1] = op(xp[1], yp[1]);
    return old; /* The returned value may not be atomic. It can be mix of two ops. Caller should discard it. */
  }
};
    #elif defined(PETSC_USE_REAL_SINGLE)
template <>
struct AtomicInsert<PetscComplex> {
  __device__ PetscComplex operator()(PetscComplex &x, PetscComplex y) const
  {
    double              *xp = (double *)&x, *yp = (double *)&y;
    AtomicInsert<double> op;
    return op(xp[0], yp[0]);
  }
};
    #endif
  #endif

/*
  Atomic add operations

*/
__device__ static inline llint atomicAdd(llint *address, llint val)
{
  return (llint)atomicAdd((ullint *)address, (ullint)val);
}

template <typename Type>
struct AtomicAdd {
  __device__ Type operator()(Type &x, Type y) const { return atomicAdd(&x, y); }
};

template <>
struct AtomicAdd<double> {
  __device__ double operator()(double &x, double y) const
  {
    /* Cuda version does more checks that may be needed */
    return atomicAdd(&x, y);
  }
};

template <>
struct AtomicAdd<float> {
  __device__ float operator()(float &x, float y) const
  {
    /* Cuda version does more checks that may be needed */
    return atomicAdd(&x, y);
  }
};

  #if defined(PETSC_HAVE_COMPLEX)
template <>
struct AtomicAdd<PetscComplex> {
  __device__ PetscComplex operator()(PetscComplex &x, PetscComplex y) const
  {
    PetscComplex         old, *z = &old;
    PetscReal           *xp = (PetscReal *)&x, *yp = (PetscReal *)&y;
    AtomicAdd<PetscReal> op;
    z[0] = op(xp[0], yp[0]);
    z[1] = op(xp[1], yp[1]);
    return old; /* The returned value may not be atomic. It can be mix of two ops. Caller should discard it. */
  }
};
  #endif

  /*
  Atomic Mult operations:

  HIP has no atomicMult at all, so we build our own with atomicCAS
 */
  #if defined(PETSC_USE_REAL_DOUBLE)
__device__ static inline double atomicMult(double *address, double val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    /* Other threads can access and modify value of *address_as_ull after the read above and before the write below */
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val * __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
  #elif defined(PETSC_USE_REAL_SINGLE)
__device__ static inline float atomicMult(float *address, float val)
{
  int *address_as_int = (int *)(address);
  int  old            = *address_as_int, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_int, assumed, __float_as_int(val * __int_as_float(assumed)));
  } while (assumed != old);
  return __int_as_float(old);
}
  #endif

__device__ static inline int atomicMult(int *address, int val)
{
  int *address_as_int = (int *)(address);
  int  old            = *address_as_int, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_int, assumed, val * assumed);
  } while (assumed != old);
  return (int)old;
}

__device__ static inline llint atomicMult(llint *address, llint val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, (ullint)(val * (llint)assumed));
  } while (assumed != old);
  return (llint)old;
}

template <typename Type>
struct AtomicMult {
  __device__ Type operator()(Type &x, Type y) const { return atomicMult(&x, y); }
};

  /*
  Atomic Min/Max operations

  See CUDA version for comments.
 */
  #if PETSC_PKG_HIP_VERSION_LT(4, 4, 0)
    #if defined(PETSC_USE_REAL_DOUBLE)
__device__ static double atomicMin(double *address, double val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, __double_as_longlong(PetscMin(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__ static double atomicMax(double *address, double val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, __double_as_longlong(PetscMax(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}
    #elif defined(PETSC_USE_REAL_SINGLE)
__device__ static float atomicMin(float *address, float val)
{
  int *address_as_int = (int *)(address);
  int  old            = *address_as_int, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_int, assumed, __float_as_int(PetscMin(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ static float atomicMax(float *address, float val)
{
  int *address_as_int = (int *)(address);
  int  old            = *address_as_int, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_int, assumed, __float_as_int(PetscMax(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}
    #endif
  #endif

  #if PETSC_PKG_HIP_VERSION_LT(5, 7, 0)
__device__ static inline llint atomicMin(llint *address, llint val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, (ullint)(PetscMin(val, (llint)assumed)));
  } while (assumed != old);
  return (llint)old;
}

__device__ static inline llint atomicMax(llint *address, llint val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, (ullint)(PetscMax(val, (llint)assumed)));
  } while (assumed != old);
  return (llint)old;
}
  #endif

template <typename Type>
struct AtomicMin {
  __device__ Type operator()(Type &x, Type y) const { return atomicMin(&x, y); }
};
template <typename Type>
struct AtomicMax {
  __device__ Type operator()(Type &x, Type y) const { return atomicMax(&x, y); }
};

/*
  Atomic bitwise operations
  As of ROCm 3.10, the llint atomicAnd/Or/Xor(llint*, llint) is not supported
*/

__device__ static inline llint atomicAnd(llint *address, llint val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, (ullint)(val & (llint)assumed));
  } while (assumed != old);
  return (llint)old;
}
__device__ static inline llint atomicOr(llint *address, llint val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, (ullint)(val | (llint)assumed));
  } while (assumed != old);
  return (llint)old;
}

__device__ static inline llint atomicXor(llint *address, llint val)
{
  ullint *address_as_ull = (ullint *)(address);
  ullint  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, (ullint)(val ^ (llint)assumed));
  } while (assumed != old);
  return (llint)old;
}

template <typename Type>
struct AtomicBAND {
  __device__ Type operator()(Type &x, Type y) const { return atomicAnd(&x, y); }
};
template <typename Type>
struct AtomicBOR {
  __device__ Type operator()(Type &x, Type y) const { return atomicOr(&x, y); }
};
template <typename Type>
struct AtomicBXOR {
  __device__ Type operator()(Type &x, Type y) const { return atomicXor(&x, y); }
};

/*
  Atomic logical operations:

  CUDA has no atomic logical operations at all. We support them on integer types.
*/

/* A template without definition makes any instantiation not using given specializations erroneous at compile time,
   which is what we want since we only support 32-bit and 64-bit integers.
 */
template <typename Type, class Op, int size /* sizeof(Type) */>
struct AtomicLogical;

template <typename Type, class Op>
struct AtomicLogical<Type, Op, 4> {
  __device__ Type operator()(Type &x, Type y) const
  {
    int *address_as_int = (int *)(&x);
    int  old            = *address_as_int, assumed;
    Op   op;
    do {
      assumed = old;
      old     = atomicCAS(address_as_int, assumed, (int)(op((Type)assumed, y)));
    } while (assumed != old);
    return (Type)old;
  }
};

template <typename Type, class Op>
struct AtomicLogical<Type, Op, 8> {
  __device__ Type operator()(Type &x, Type y) const
  {
    ullint *address_as_ull = (ullint *)(&x);
    ullint  old            = *address_as_ull, assumed;
    Op      op;
    do {
      assumed = old;
      old     = atomicCAS(address_as_ull, assumed, (ullint)(op((Type)assumed, y)));
    } while (assumed != old);
    return (Type)old;
  }
};

/* Note land/lor/lxor below are different from LAND etc above. Here we pass arguments by value and return result of ops (not old value) */
template <typename Type>
struct land {
  __device__ Type operator()(Type x, Type y) { return x && y; }
};
template <typename Type>
struct lor {
  __device__ Type operator()(Type x, Type y) { return x || y; }
};
template <typename Type>
struct lxor {
  __device__ Type operator()(Type x, Type y) { return !x != !y; }
};

template <typename Type>
struct AtomicLAND {
  __device__ Type operator()(Type &x, Type y) const
  {
    AtomicLogical<Type, land<Type>, sizeof(Type)> op;
    return op(x, y);
  }
};
template <typename Type>
struct AtomicLOR {
  __device__ Type operator()(Type &x, Type y) const
  {
    AtomicLogical<Type, lor<Type>, sizeof(Type)> op;
    return op(x, y);
  }
};
template <typename Type>
struct AtomicLXOR {
  __device__ Type operator()(Type &x, Type y) const
  {
    AtomicLogical<Type, lxor<Type>, sizeof(Type)> op;
    return op(x, y);
  }
};
#endif
