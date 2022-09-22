#include <../src/vec/is/sf/impls/basic/sfpack.h>

/* Map a thread id to an index in root/leaf space through a series of 3D subdomains. See PetscSFPackOpt. */
__device__ static inline PetscInt MapTidToIndex(const PetscInt *opt, PetscInt tid)
{
  PetscInt        i, j, k, m, n, r;
  const PetscInt *offset, *start, *dx, *dy, *X, *Y;

  n      = opt[0];
  offset = opt + 1;
  start  = opt + n + 2;
  dx     = opt + 2 * n + 2;
  dy     = opt + 3 * n + 2;
  X      = opt + 5 * n + 2;
  Y      = opt + 6 * n + 2;
  for (r = 0; r < n; r++) {
    if (tid < offset[r + 1]) break;
  }
  m = (tid - offset[r]);
  k = m / (dx[r] * dy[r]);
  j = (m - k * dx[r] * dy[r]) / dx[r];
  i = m - k * dx[r] * dy[r] - j * dx[r];

  return (start[r] + k * X[r] * Y[r] + j * X[r] + i);
}

/*====================================================================================*/
/*  Templated CUDA kernels for pack/unpack. The Op can be regular or atomic           */
/*====================================================================================*/

/* Suppose user calls PetscSFReduce(sf,unit,...) and <unit> is an MPI data type made of 16 PetscReals, then
   <Type> is PetscReal, which is the primitive type we operate on.
   <bs>   is 16, which says <unit> contains 16 primitive types.
   <BS>   is 8, which is the maximal SIMD width we will try to vectorize operations on <unit>.
   <EQ>   is 0, which is (bs == BS ? 1 : 0)

  If instead, <unit> has 8 PetscReals, then bs=8, BS=8, EQ=1, rendering MBS below to a compile time constant.
  For the common case in VecScatter, bs=1, BS=1, EQ=1, MBS=1, the inner for-loops below will be totally unrolled.
*/
template <class Type, PetscInt BS, PetscInt EQ>
__global__ static void d_Pack(PetscInt bs, PetscInt count, PetscInt start, const PetscInt *opt, const PetscInt *idx, const Type *data, Type *buf)
{
  PetscInt       i, s, t, tid = blockIdx.x * blockDim.x + threadIdx.x;
  const PetscInt grid_size = gridDim.x * blockDim.x;
  const PetscInt M         = (EQ) ? 1 : bs / BS; /* If EQ, then M=1 enables compiler's const-propagation */
  const PetscInt MBS       = M * BS;             /* MBS=bs. We turn MBS into a compile-time const when EQ=1. */

  for (; tid < count; tid += grid_size) {
    /* opt != NULL ==> idx == NULL, i.e., the indices have patterns but not contiguous;
       opt == NULL && idx == NULL ==> the indices are contiguous;
     */
    t = (opt ? MapTidToIndex(opt, tid) : (idx ? idx[tid] : start + tid)) * MBS;
    s = tid * MBS;
    for (i = 0; i < MBS; i++) buf[s + i] = data[t + i];
  }
}

template <class Type, class Op, PetscInt BS, PetscInt EQ>
__global__ static void d_UnpackAndOp(PetscInt bs, PetscInt count, PetscInt start, const PetscInt *opt, const PetscInt *idx, Type *data, const Type *buf)
{
  PetscInt       i, s, t, tid = blockIdx.x * blockDim.x + threadIdx.x;
  const PetscInt grid_size = gridDim.x * blockDim.x;
  const PetscInt M = (EQ) ? 1 : bs / BS, MBS = M * BS;
  Op             op;

  for (; tid < count; tid += grid_size) {
    t = (opt ? MapTidToIndex(opt, tid) : (idx ? idx[tid] : start + tid)) * MBS;
    s = tid * MBS;
    for (i = 0; i < MBS; i++) op(data[t + i], buf[s + i]);
  }
}

template <class Type, class Op, PetscInt BS, PetscInt EQ>
__global__ static void d_FetchAndOp(PetscInt bs, PetscInt count, PetscInt rootstart, const PetscInt *rootopt, const PetscInt *rootidx, Type *rootdata, Type *leafbuf)
{
  PetscInt       i, l, r, tid = blockIdx.x * blockDim.x + threadIdx.x;
  const PetscInt grid_size = gridDim.x * blockDim.x;
  const PetscInt M = (EQ) ? 1 : bs / BS, MBS = M * BS;
  Op             op;

  for (; tid < count; tid += grid_size) {
    r = (rootopt ? MapTidToIndex(rootopt, tid) : (rootidx ? rootidx[tid] : rootstart + tid)) * MBS;
    l = tid * MBS;
    for (i = 0; i < MBS; i++) leafbuf[l + i] = op(rootdata[r + i], leafbuf[l + i]);
  }
}

template <class Type, class Op, PetscInt BS, PetscInt EQ>
__global__ static void d_ScatterAndOp(PetscInt bs, PetscInt count, PetscInt srcx, PetscInt srcy, PetscInt srcX, PetscInt srcY, PetscInt srcStart, const PetscInt *srcIdx, const Type *src, PetscInt dstx, PetscInt dsty, PetscInt dstX, PetscInt dstY, PetscInt dstStart, const PetscInt *dstIdx, Type *dst)
{
  PetscInt       i, j, k, s, t, tid = blockIdx.x * blockDim.x + threadIdx.x;
  const PetscInt grid_size = gridDim.x * blockDim.x;
  const PetscInt M = (EQ) ? 1 : bs / BS, MBS = M * BS;
  Op             op;

  for (; tid < count; tid += grid_size) {
    if (!srcIdx) { /* src is either contiguous or 3D */
      k = tid / (srcx * srcy);
      j = (tid - k * srcx * srcy) / srcx;
      i = tid - k * srcx * srcy - j * srcx;
      s = srcStart + k * srcX * srcY + j * srcX + i;
    } else {
      s = srcIdx[tid];
    }

    if (!dstIdx) { /* dst is either contiguous or 3D */
      k = tid / (dstx * dsty);
      j = (tid - k * dstx * dsty) / dstx;
      i = tid - k * dstx * dsty - j * dstx;
      t = dstStart + k * dstX * dstY + j * dstX + i;
    } else {
      t = dstIdx[tid];
    }

    s *= MBS;
    t *= MBS;
    for (i = 0; i < MBS; i++) op(dst[t + i], src[s + i]);
  }
}

template <class Type, class Op, PetscInt BS, PetscInt EQ>
__global__ static void d_FetchAndOpLocal(PetscInt bs, PetscInt count, PetscInt rootstart, const PetscInt *rootopt, const PetscInt *rootidx, Type *rootdata, PetscInt leafstart, const PetscInt *leafopt, const PetscInt *leafidx, const Type *leafdata, Type *leafupdate)
{
  PetscInt       i, l, r, tid = blockIdx.x * blockDim.x + threadIdx.x;
  const PetscInt grid_size = gridDim.x * blockDim.x;
  const PetscInt M = (EQ) ? 1 : bs / BS, MBS = M * BS;
  Op             op;

  for (; tid < count; tid += grid_size) {
    r = (rootopt ? MapTidToIndex(rootopt, tid) : (rootidx ? rootidx[tid] : rootstart + tid)) * MBS;
    l = (leafopt ? MapTidToIndex(leafopt, tid) : (leafidx ? leafidx[tid] : leafstart + tid)) * MBS;
    for (i = 0; i < MBS; i++) leafupdate[l + i] = op(rootdata[r + i], leafdata[l + i]);
  }
}

/*====================================================================================*/
/*                             Regular operations on device                           */
/*====================================================================================*/
template <typename Type>
struct Insert {
  __device__ Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = y;
    return old;
  }
};
template <typename Type>
struct Add {
  __device__ Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x += y;
    return old;
  }
};
template <typename Type>
struct Mult {
  __device__ Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x *= y;
    return old;
  }
};
template <typename Type>
struct Min {
  __device__ Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = PetscMin(x, y);
    return old;
  }
};
template <typename Type>
struct Max {
  __device__ Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = PetscMax(x, y);
    return old;
  }
};
template <typename Type>
struct LAND {
  __device__ Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = x && y;
    return old;
  }
};
template <typename Type>
struct LOR {
  __device__ Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = x || y;
    return old;
  }
};
template <typename Type>
struct LXOR {
  __device__ Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = !x != !y;
    return old;
  }
};
template <typename Type>
struct BAND {
  __device__ Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = x & y;
    return old;
  }
};
template <typename Type>
struct BOR {
  __device__ Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = x | y;
    return old;
  }
};
template <typename Type>
struct BXOR {
  __device__ Type operator()(Type &x, Type y) const
  {
    Type old = x;
    x        = x ^ y;
    return old;
  }
};
template <typename Type>
struct Minloc {
  __device__ Type operator()(Type &x, Type y) const
  {
    Type old = x;
    if (y.a < x.a) x = y;
    else if (y.a == x.a) x.b = min(x.b, y.b);
    return old;
  }
};
template <typename Type>
struct Maxloc {
  __device__ Type operator()(Type &x, Type y) const
  {
    Type old = x;
    if (y.a > x.a) x = y;
    else if (y.a == x.a) x.b = min(x.b, y.b); /* See MPI MAXLOC */
    return old;
  }
};

/*====================================================================================*/
/*                             Atomic operations on device                            */
/*====================================================================================*/

/*
  Atomic Insert (exchange) operations

  CUDA C Programming Guide V10.1 Chapter B.12.1.3:

  int atomicExch(int* address, int val);
  unsigned int atomicExch(unsigned int* address, unsigned int val);
  unsigned long long int atomicExch(unsigned long long int* address, unsigned long long int val);
  float atomicExch(float* address, float val);

  reads the 32-bit or 64-bit word old located at the address address in global or shared
  memory and stores val back to memory at the same address. These two operations are
  performed in one atomic transaction. The function returns old.

  PETSc notes:

  It may be useful in PetscSFFetchAndOp with op = MPI_REPLACE.

  VecScatter with multiple entries scattered to the same location using INSERT_VALUES does not need
  atomic insertion, since it does not need the old value. A 32-bit or 64-bit store instruction should
  be atomic itself.

  With bs>1 and a unit > 64 bits, the current element-wise atomic approach can not guarantee the whole
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

  reads the 16-bit, 32-bit or 64-bit word old located at the address address in global or shared memory, computes (old + val),
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
  int old = *address_as_int, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(val * __int_as_float(assumed)));
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

  reads the 32-bit or 64-bit word old located at the address address in global or shared
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
  int old = *address_as_int, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(PetscMin(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ static float atomicMax(float *address, float val)
{
  int *address_as_int = (int *)(address);
  int old = *address_as_int, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(PetscMax(val, __int_as_float(assumed))));
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

  reads the 32-bit or 64-bit word old located at the address address in global or shared
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
  __device__ Type operator()(Type x, Type y) { return (!x != !y); }
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

/*====================================================================================*/
/*  Wrapper functions of cuda kernels. Function pointers are stored in 'link'         */
/*====================================================================================*/
template <typename Type, PetscInt BS, PetscInt EQ>
static PetscErrorCode Pack(PetscSFLink link, PetscInt count, PetscInt start, PetscSFPackOpt opt, const PetscInt *idx, const void *data, void *buf)
{
  PetscInt        nthreads = 256;
  PetscInt        nblocks  = (count + nthreads - 1) / nthreads;
  const PetscInt *iarray   = opt ? opt->array : NULL;

  PetscFunctionBegin;
  if (!count) PetscFunctionReturn(0);
  if (!opt && !idx) { /* It is a 'CUDA data to nvshmem buf' memory copy */
    PetscCallCUDA(cudaMemcpyAsync(buf, (char *)data + start * link->unitbytes, count * link->unitbytes, cudaMemcpyDeviceToDevice, link->stream));
  } else {
    nblocks = PetscMin(nblocks, link->maxResidentThreadsPerGPU / nthreads);
    d_Pack<Type, BS, EQ><<<nblocks, nthreads, 0, link->stream>>>(link->bs, count, start, iarray, idx, (const Type *)data, (Type *)buf);
    PetscCallCUDA(cudaGetLastError());
  }
  PetscFunctionReturn(0);
}

/* To specialize UnpackAndOp for the cudaMemcpyAsync() below. Usually if this is a contiguous memcpy, we use root/leafdirect and do
   not need UnpackAndOp. Only with nvshmem, we need this 'nvshmem buf to CUDA data' memory copy
*/
template <typename Type, PetscInt BS, PetscInt EQ>
static PetscErrorCode Unpack(PetscSFLink link, PetscInt count, PetscInt start, PetscSFPackOpt opt, const PetscInt *idx, void *data, const void *buf)
{
  PetscInt        nthreads = 256;
  PetscInt        nblocks  = (count + nthreads - 1) / nthreads;
  const PetscInt *iarray   = opt ? opt->array : NULL;

  PetscFunctionBegin;
  if (!count) PetscFunctionReturn(0);
  if (!opt && !idx) { /* It is a 'nvshmem buf to CUDA data' memory copy */
    PetscCallCUDA(cudaMemcpyAsync((char *)data + start * link->unitbytes, buf, count * link->unitbytes, cudaMemcpyDeviceToDevice, link->stream));
  } else {
    nblocks = PetscMin(nblocks, link->maxResidentThreadsPerGPU / nthreads);
    d_UnpackAndOp<Type, Insert<Type>, BS, EQ><<<nblocks, nthreads, 0, link->stream>>>(link->bs, count, start, iarray, idx, (Type *)data, (const Type *)buf);
    PetscCallCUDA(cudaGetLastError());
  }
  PetscFunctionReturn(0);
}

template <typename Type, class Op, PetscInt BS, PetscInt EQ>
static PetscErrorCode UnpackAndOp(PetscSFLink link, PetscInt count, PetscInt start, PetscSFPackOpt opt, const PetscInt *idx, void *data, const void *buf)
{
  PetscInt        nthreads = 256;
  PetscInt        nblocks  = (count + nthreads - 1) / nthreads;
  const PetscInt *iarray   = opt ? opt->array : NULL;

  PetscFunctionBegin;
  if (!count) PetscFunctionReturn(0);
  nblocks = PetscMin(nblocks, link->maxResidentThreadsPerGPU / nthreads);
  d_UnpackAndOp<Type, Op, BS, EQ><<<nblocks, nthreads, 0, link->stream>>>(link->bs, count, start, iarray, idx, (Type *)data, (const Type *)buf);
  PetscCallCUDA(cudaGetLastError());
  PetscFunctionReturn(0);
}

template <typename Type, class Op, PetscInt BS, PetscInt EQ>
static PetscErrorCode FetchAndOp(PetscSFLink link, PetscInt count, PetscInt start, PetscSFPackOpt opt, const PetscInt *idx, void *data, void *buf)
{
  PetscInt        nthreads = 256;
  PetscInt        nblocks  = (count + nthreads - 1) / nthreads;
  const PetscInt *iarray   = opt ? opt->array : NULL;

  PetscFunctionBegin;
  if (!count) PetscFunctionReturn(0);
  nblocks = PetscMin(nblocks, link->maxResidentThreadsPerGPU / nthreads);
  d_FetchAndOp<Type, Op, BS, EQ><<<nblocks, nthreads, 0, link->stream>>>(link->bs, count, start, iarray, idx, (Type *)data, (Type *)buf);
  PetscCallCUDA(cudaGetLastError());
  PetscFunctionReturn(0);
}

template <typename Type, class Op, PetscInt BS, PetscInt EQ>
static PetscErrorCode ScatterAndOp(PetscSFLink link, PetscInt count, PetscInt srcStart, PetscSFPackOpt srcOpt, const PetscInt *srcIdx, const void *src, PetscInt dstStart, PetscSFPackOpt dstOpt, const PetscInt *dstIdx, void *dst)
{
  PetscInt nthreads = 256;
  PetscInt nblocks  = (count + nthreads - 1) / nthreads;
  PetscInt srcx = 0, srcy = 0, srcX = 0, srcY = 0, dstx = 0, dsty = 0, dstX = 0, dstY = 0;

  PetscFunctionBegin;
  if (!count) PetscFunctionReturn(0);
  nblocks = PetscMin(nblocks, link->maxResidentThreadsPerGPU / nthreads);

  /* The 3D shape of source subdomain may be different than that of the destination, which makes it difficult to use CUDA 3D grid and block */
  if (srcOpt) {
    srcx     = srcOpt->dx[0];
    srcy     = srcOpt->dy[0];
    srcX     = srcOpt->X[0];
    srcY     = srcOpt->Y[0];
    srcStart = srcOpt->start[0];
    srcIdx   = NULL;
  } else if (!srcIdx) {
    srcx = srcX = count;
    srcy = srcY = 1;
  }

  if (dstOpt) {
    dstx     = dstOpt->dx[0];
    dsty     = dstOpt->dy[0];
    dstX     = dstOpt->X[0];
    dstY     = dstOpt->Y[0];
    dstStart = dstOpt->start[0];
    dstIdx   = NULL;
  } else if (!dstIdx) {
    dstx = dstX = count;
    dsty = dstY = 1;
  }

  d_ScatterAndOp<Type, Op, BS, EQ><<<nblocks, nthreads, 0, link->stream>>>(link->bs, count, srcx, srcy, srcX, srcY, srcStart, srcIdx, (const Type *)src, dstx, dsty, dstX, dstY, dstStart, dstIdx, (Type *)dst);
  PetscCallCUDA(cudaGetLastError());
  PetscFunctionReturn(0);
}

/* Specialization for Insert since we may use cudaMemcpyAsync */
template <typename Type, PetscInt BS, PetscInt EQ>
static PetscErrorCode ScatterAndInsert(PetscSFLink link, PetscInt count, PetscInt srcStart, PetscSFPackOpt srcOpt, const PetscInt *srcIdx, const void *src, PetscInt dstStart, PetscSFPackOpt dstOpt, const PetscInt *dstIdx, void *dst)
{
  PetscFunctionBegin;
  if (!count) PetscFunctionReturn(0);
  /*src and dst are contiguous */
  if ((!srcOpt && !srcIdx) && (!dstOpt && !dstIdx) && src != dst) {
    PetscCallCUDA(cudaMemcpyAsync((Type *)dst + dstStart * link->bs, (const Type *)src + srcStart * link->bs, count * link->unitbytes, cudaMemcpyDeviceToDevice, link->stream));
  } else {
    PetscCall(ScatterAndOp<Type, Insert<Type>, BS, EQ>(link, count, srcStart, srcOpt, srcIdx, src, dstStart, dstOpt, dstIdx, dst));
  }
  PetscFunctionReturn(0);
}

template <typename Type, class Op, PetscInt BS, PetscInt EQ>
static PetscErrorCode FetchAndOpLocal(PetscSFLink link, PetscInt count, PetscInt rootstart, PetscSFPackOpt rootopt, const PetscInt *rootidx, void *rootdata, PetscInt leafstart, PetscSFPackOpt leafopt, const PetscInt *leafidx, const void *leafdata, void *leafupdate)
{
  PetscInt        nthreads = 256;
  PetscInt        nblocks  = (count + nthreads - 1) / nthreads;
  const PetscInt *rarray   = rootopt ? rootopt->array : NULL;
  const PetscInt *larray   = leafopt ? leafopt->array : NULL;

  PetscFunctionBegin;
  if (!count) PetscFunctionReturn(0);
  nblocks = PetscMin(nblocks, link->maxResidentThreadsPerGPU / nthreads);
  d_FetchAndOpLocal<Type, Op, BS, EQ><<<nblocks, nthreads, 0, link->stream>>>(link->bs, count, rootstart, rarray, rootidx, (Type *)rootdata, leafstart, larray, leafidx, (const Type *)leafdata, (Type *)leafupdate);
  PetscCallCUDA(cudaGetLastError());
  PetscFunctionReturn(0);
}

/*====================================================================================*/
/*  Init various types and instantiate pack/unpack function pointers                  */
/*====================================================================================*/
template <typename Type, PetscInt BS, PetscInt EQ>
static void PackInit_RealType(PetscSFLink link)
{
  /* Pack/unpack for remote communication */
  link->d_Pack            = Pack<Type, BS, EQ>;
  link->d_UnpackAndInsert = Unpack<Type, BS, EQ>;
  link->d_UnpackAndAdd    = UnpackAndOp<Type, Add<Type>, BS, EQ>;
  link->d_UnpackAndMult   = UnpackAndOp<Type, Mult<Type>, BS, EQ>;
  link->d_UnpackAndMin    = UnpackAndOp<Type, Min<Type>, BS, EQ>;
  link->d_UnpackAndMax    = UnpackAndOp<Type, Max<Type>, BS, EQ>;
  link->d_FetchAndAdd     = FetchAndOp<Type, Add<Type>, BS, EQ>;

  /* Scatter for local communication */
  link->d_ScatterAndInsert = ScatterAndInsert<Type, BS, EQ>; /* Has special optimizations */
  link->d_ScatterAndAdd    = ScatterAndOp<Type, Add<Type>, BS, EQ>;
  link->d_ScatterAndMult   = ScatterAndOp<Type, Mult<Type>, BS, EQ>;
  link->d_ScatterAndMin    = ScatterAndOp<Type, Min<Type>, BS, EQ>;
  link->d_ScatterAndMax    = ScatterAndOp<Type, Max<Type>, BS, EQ>;
  link->d_FetchAndAddLocal = FetchAndOpLocal<Type, Add<Type>, BS, EQ>;

  /* Atomic versions when there are data-race possibilities */
  link->da_UnpackAndInsert = UnpackAndOp<Type, AtomicInsert<Type>, BS, EQ>;
  link->da_UnpackAndAdd    = UnpackAndOp<Type, AtomicAdd<Type>, BS, EQ>;
  link->da_UnpackAndMult   = UnpackAndOp<Type, AtomicMult<Type>, BS, EQ>;
  link->da_UnpackAndMin    = UnpackAndOp<Type, AtomicMin<Type>, BS, EQ>;
  link->da_UnpackAndMax    = UnpackAndOp<Type, AtomicMax<Type>, BS, EQ>;
  link->da_FetchAndAdd     = FetchAndOp<Type, AtomicAdd<Type>, BS, EQ>;

  link->da_ScatterAndInsert = ScatterAndOp<Type, AtomicInsert<Type>, BS, EQ>;
  link->da_ScatterAndAdd    = ScatterAndOp<Type, AtomicAdd<Type>, BS, EQ>;
  link->da_ScatterAndMult   = ScatterAndOp<Type, AtomicMult<Type>, BS, EQ>;
  link->da_ScatterAndMin    = ScatterAndOp<Type, AtomicMin<Type>, BS, EQ>;
  link->da_ScatterAndMax    = ScatterAndOp<Type, AtomicMax<Type>, BS, EQ>;
  link->da_FetchAndAddLocal = FetchAndOpLocal<Type, AtomicAdd<Type>, BS, EQ>;
}

/* Have this templated class to specialize for char integers */
template <typename Type, PetscInt BS, PetscInt EQ, PetscInt size /*sizeof(Type)*/>
struct PackInit_IntegerType_Atomic {
  static void Init(PetscSFLink link)
  {
    link->da_UnpackAndInsert = UnpackAndOp<Type, AtomicInsert<Type>, BS, EQ>;
    link->da_UnpackAndAdd    = UnpackAndOp<Type, AtomicAdd<Type>, BS, EQ>;
    link->da_UnpackAndMult   = UnpackAndOp<Type, AtomicMult<Type>, BS, EQ>;
    link->da_UnpackAndMin    = UnpackAndOp<Type, AtomicMin<Type>, BS, EQ>;
    link->da_UnpackAndMax    = UnpackAndOp<Type, AtomicMax<Type>, BS, EQ>;
    link->da_UnpackAndLAND   = UnpackAndOp<Type, AtomicLAND<Type>, BS, EQ>;
    link->da_UnpackAndLOR    = UnpackAndOp<Type, AtomicLOR<Type>, BS, EQ>;
    link->da_UnpackAndLXOR   = UnpackAndOp<Type, AtomicLXOR<Type>, BS, EQ>;
    link->da_UnpackAndBAND   = UnpackAndOp<Type, AtomicBAND<Type>, BS, EQ>;
    link->da_UnpackAndBOR    = UnpackAndOp<Type, AtomicBOR<Type>, BS, EQ>;
    link->da_UnpackAndBXOR   = UnpackAndOp<Type, AtomicBXOR<Type>, BS, EQ>;
    link->da_FetchAndAdd     = FetchAndOp<Type, AtomicAdd<Type>, BS, EQ>;

    link->da_ScatterAndInsert = ScatterAndOp<Type, AtomicInsert<Type>, BS, EQ>;
    link->da_ScatterAndAdd    = ScatterAndOp<Type, AtomicAdd<Type>, BS, EQ>;
    link->da_ScatterAndMult   = ScatterAndOp<Type, AtomicMult<Type>, BS, EQ>;
    link->da_ScatterAndMin    = ScatterAndOp<Type, AtomicMin<Type>, BS, EQ>;
    link->da_ScatterAndMax    = ScatterAndOp<Type, AtomicMax<Type>, BS, EQ>;
    link->da_ScatterAndLAND   = ScatterAndOp<Type, AtomicLAND<Type>, BS, EQ>;
    link->da_ScatterAndLOR    = ScatterAndOp<Type, AtomicLOR<Type>, BS, EQ>;
    link->da_ScatterAndLXOR   = ScatterAndOp<Type, AtomicLXOR<Type>, BS, EQ>;
    link->da_ScatterAndBAND   = ScatterAndOp<Type, AtomicBAND<Type>, BS, EQ>;
    link->da_ScatterAndBOR    = ScatterAndOp<Type, AtomicBOR<Type>, BS, EQ>;
    link->da_ScatterAndBXOR   = ScatterAndOp<Type, AtomicBXOR<Type>, BS, EQ>;
    link->da_FetchAndAddLocal = FetchAndOpLocal<Type, AtomicAdd<Type>, BS, EQ>;
  }
};

/* CUDA does not support atomics on chars. It is TBD in PETSc. */
template <typename Type, PetscInt BS, PetscInt EQ>
struct PackInit_IntegerType_Atomic<Type, BS, EQ, 1> {
  static void Init(PetscSFLink link)
  { /* Nothing to leave function pointers NULL */
  }
};

template <typename Type, PetscInt BS, PetscInt EQ>
static void PackInit_IntegerType(PetscSFLink link)
{
  link->d_Pack            = Pack<Type, BS, EQ>;
  link->d_UnpackAndInsert = Unpack<Type, BS, EQ>;
  link->d_UnpackAndAdd    = UnpackAndOp<Type, Add<Type>, BS, EQ>;
  link->d_UnpackAndMult   = UnpackAndOp<Type, Mult<Type>, BS, EQ>;
  link->d_UnpackAndMin    = UnpackAndOp<Type, Min<Type>, BS, EQ>;
  link->d_UnpackAndMax    = UnpackAndOp<Type, Max<Type>, BS, EQ>;
  link->d_UnpackAndLAND   = UnpackAndOp<Type, LAND<Type>, BS, EQ>;
  link->d_UnpackAndLOR    = UnpackAndOp<Type, LOR<Type>, BS, EQ>;
  link->d_UnpackAndLXOR   = UnpackAndOp<Type, LXOR<Type>, BS, EQ>;
  link->d_UnpackAndBAND   = UnpackAndOp<Type, BAND<Type>, BS, EQ>;
  link->d_UnpackAndBOR    = UnpackAndOp<Type, BOR<Type>, BS, EQ>;
  link->d_UnpackAndBXOR   = UnpackAndOp<Type, BXOR<Type>, BS, EQ>;
  link->d_FetchAndAdd     = FetchAndOp<Type, Add<Type>, BS, EQ>;

  link->d_ScatterAndInsert = ScatterAndInsert<Type, BS, EQ>;
  link->d_ScatterAndAdd    = ScatterAndOp<Type, Add<Type>, BS, EQ>;
  link->d_ScatterAndMult   = ScatterAndOp<Type, Mult<Type>, BS, EQ>;
  link->d_ScatterAndMin    = ScatterAndOp<Type, Min<Type>, BS, EQ>;
  link->d_ScatterAndMax    = ScatterAndOp<Type, Max<Type>, BS, EQ>;
  link->d_ScatterAndLAND   = ScatterAndOp<Type, LAND<Type>, BS, EQ>;
  link->d_ScatterAndLOR    = ScatterAndOp<Type, LOR<Type>, BS, EQ>;
  link->d_ScatterAndLXOR   = ScatterAndOp<Type, LXOR<Type>, BS, EQ>;
  link->d_ScatterAndBAND   = ScatterAndOp<Type, BAND<Type>, BS, EQ>;
  link->d_ScatterAndBOR    = ScatterAndOp<Type, BOR<Type>, BS, EQ>;
  link->d_ScatterAndBXOR   = ScatterAndOp<Type, BXOR<Type>, BS, EQ>;
  link->d_FetchAndAddLocal = FetchAndOpLocal<Type, Add<Type>, BS, EQ>;
  PackInit_IntegerType_Atomic<Type, BS, EQ, sizeof(Type)>::Init(link);
}

#if defined(PETSC_HAVE_COMPLEX)
template <typename Type, PetscInt BS, PetscInt EQ>
static void PackInit_ComplexType(PetscSFLink link)
{
  link->d_Pack            = Pack<Type, BS, EQ>;
  link->d_UnpackAndInsert = Unpack<Type, BS, EQ>;
  link->d_UnpackAndAdd    = UnpackAndOp<Type, Add<Type>, BS, EQ>;
  link->d_UnpackAndMult   = UnpackAndOp<Type, Mult<Type>, BS, EQ>;
  link->d_FetchAndAdd     = FetchAndOp<Type, Add<Type>, BS, EQ>;

  link->d_ScatterAndInsert = ScatterAndInsert<Type, BS, EQ>;
  link->d_ScatterAndAdd    = ScatterAndOp<Type, Add<Type>, BS, EQ>;
  link->d_ScatterAndMult   = ScatterAndOp<Type, Mult<Type>, BS, EQ>;
  link->d_FetchAndAddLocal = FetchAndOpLocal<Type, Add<Type>, BS, EQ>;

  link->da_UnpackAndInsert = UnpackAndOp<Type, AtomicInsert<Type>, BS, EQ>;
  link->da_UnpackAndAdd    = UnpackAndOp<Type, AtomicAdd<Type>, BS, EQ>;
  link->da_UnpackAndMult   = NULL; /* Not implemented yet */
  link->da_FetchAndAdd     = NULL; /* Return value of atomicAdd on complex is not atomic */

  link->da_ScatterAndInsert = ScatterAndOp<Type, AtomicInsert<Type>, BS, EQ>;
  link->da_ScatterAndAdd    = ScatterAndOp<Type, AtomicAdd<Type>, BS, EQ>;
}
#endif

typedef signed char   SignedChar;
typedef unsigned char UnsignedChar;
typedef struct {
  int a;
  int b;
} PairInt;
typedef struct {
  PetscInt a;
  PetscInt b;
} PairPetscInt;

template <typename Type>
static void PackInit_PairType(PetscSFLink link)
{
  link->d_Pack            = Pack<Type, 1, 1>;
  link->d_UnpackAndInsert = Unpack<Type, 1, 1>;
  link->d_UnpackAndMaxloc = UnpackAndOp<Type, Maxloc<Type>, 1, 1>;
  link->d_UnpackAndMinloc = UnpackAndOp<Type, Minloc<Type>, 1, 1>;

  link->d_ScatterAndInsert = ScatterAndOp<Type, Insert<Type>, 1, 1>;
  link->d_ScatterAndMaxloc = ScatterAndOp<Type, Maxloc<Type>, 1, 1>;
  link->d_ScatterAndMinloc = ScatterAndOp<Type, Minloc<Type>, 1, 1>;
  /* Atomics for pair types are not implemented yet */
}

template <typename Type, PetscInt BS, PetscInt EQ>
static void PackInit_DumbType(PetscSFLink link)
{
  link->d_Pack             = Pack<Type, BS, EQ>;
  link->d_UnpackAndInsert  = Unpack<Type, BS, EQ>;
  link->d_ScatterAndInsert = ScatterAndInsert<Type, BS, EQ>;
  /* Atomics for dumb types are not implemented yet */
}

/* Some device-specific utilities */
static PetscErrorCode PetscSFLinkSyncDevice_CUDA(PetscSFLink link)
{
  PetscFunctionBegin;
  PetscCallCUDA(cudaDeviceSynchronize());
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFLinkSyncStream_CUDA(PetscSFLink link)
{
  PetscFunctionBegin;
  PetscCallCUDA(cudaStreamSynchronize(link->stream));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFLinkMemcpy_CUDA(PetscSFLink link, PetscMemType dstmtype, void *dst, PetscMemType srcmtype, const void *src, size_t n)
{
  PetscFunctionBegin;
  enum cudaMemcpyKind kinds[2][2] = {
    {cudaMemcpyHostToHost,   cudaMemcpyHostToDevice  },
    {cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice}
  };

  if (n) {
    if (PetscMemTypeHost(dstmtype) && PetscMemTypeHost(srcmtype)) { /* Separate HostToHost so that pure-cpu code won't call cuda runtime */
      PetscCall(PetscMemcpy(dst, src, n));
    } else {
      int stype = PetscMemTypeDevice(srcmtype) ? 1 : 0;
      int dtype = PetscMemTypeDevice(dstmtype) ? 1 : 0;
      PetscCallCUDA(cudaMemcpyAsync(dst, src, n, kinds[stype][dtype], link->stream));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFMalloc_CUDA(PetscMemType mtype, size_t size, void **ptr)
{
  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) PetscCall(PetscMalloc(size, ptr));
  else if (PetscMemTypeDevice(mtype)) {
    PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
    PetscCallCUDA(cudaMalloc(ptr, size));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Wrong PetscMemType %d", (int)mtype);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFFree_CUDA(PetscMemType mtype, void *ptr)
{
  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) PetscCall(PetscFree(ptr));
  else if (PetscMemTypeDevice(mtype)) PetscCallCUDA(cudaFree(ptr));
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Wrong PetscMemType %d", (int)mtype);
  PetscFunctionReturn(0);
}

/* Destructor when the link uses MPI for communication on CUDA device */
static PetscErrorCode PetscSFLinkDestroy_MPI_CUDA(PetscSF sf, PetscSFLink link)
{
  PetscFunctionBegin;
  for (int i = PETSCSF_LOCAL; i <= PETSCSF_REMOTE; i++) {
    PetscCallCUDA(cudaFree(link->rootbuf_alloc[i][PETSC_MEMTYPE_DEVICE]));
    PetscCallCUDA(cudaFree(link->leafbuf_alloc[i][PETSC_MEMTYPE_DEVICE]));
  }
  PetscFunctionReturn(0);
}

/* Some fields of link are initialized by PetscSFPackSetUp_Host. This routine only does what needed on device */
PetscErrorCode PetscSFLinkSetUp_CUDA(PetscSF sf, PetscSFLink link, MPI_Datatype unit)
{
  PetscInt  nSignedChar = 0, nUnsignedChar = 0, nInt = 0, nPetscInt = 0, nPetscReal = 0;
  PetscBool is2Int, is2PetscInt;
#if defined(PETSC_HAVE_COMPLEX)
  PetscInt nPetscComplex = 0;
#endif

  PetscFunctionBegin;
  if (link->deviceinited) PetscFunctionReturn(0);
  PetscCall(MPIPetsc_Type_compare_contig(unit, MPI_SIGNED_CHAR, &nSignedChar));
  PetscCall(MPIPetsc_Type_compare_contig(unit, MPI_UNSIGNED_CHAR, &nUnsignedChar));
  /* MPI_CHAR is treated below as a dumb type that does not support reduction according to MPI standard */
  PetscCall(MPIPetsc_Type_compare_contig(unit, MPI_INT, &nInt));
  PetscCall(MPIPetsc_Type_compare_contig(unit, MPIU_INT, &nPetscInt));
  PetscCall(MPIPetsc_Type_compare_contig(unit, MPIU_REAL, &nPetscReal));
#if defined(PETSC_HAVE_COMPLEX)
  PetscCall(MPIPetsc_Type_compare_contig(unit, MPIU_COMPLEX, &nPetscComplex));
#endif
  PetscCall(MPIPetsc_Type_compare(unit, MPI_2INT, &is2Int));
  PetscCall(MPIPetsc_Type_compare(unit, MPIU_2INT, &is2PetscInt));

  if (is2Int) {
    PackInit_PairType<PairInt>(link);
  } else if (is2PetscInt) { /* TODO: when is2PetscInt and nPetscInt=2, we don't know which path to take. The two paths support different ops. */
    PackInit_PairType<PairPetscInt>(link);
  } else if (nPetscReal) {
#if !defined(PETSC_HAVE_DEVICE)
    if (nPetscReal == 8) PackInit_RealType<PetscReal, 8, 1>(link);
    else if (nPetscReal % 8 == 0) PackInit_RealType<PetscReal, 8, 0>(link);
    else if (nPetscReal == 4) PackInit_RealType<PetscReal, 4, 1>(link);
    else if (nPetscReal % 4 == 0) PackInit_RealType<PetscReal, 4, 0>(link);
    else if (nPetscReal == 2) PackInit_RealType<PetscReal, 2, 1>(link);
    else if (nPetscReal % 2 == 0) PackInit_RealType<PetscReal, 2, 0>(link);
    else if (nPetscReal == 1) PackInit_RealType<PetscReal, 1, 1>(link);
    else if (nPetscReal % 1 == 0)
#endif
      PackInit_RealType<PetscReal, 1, 0>(link);
  } else if (nPetscInt && sizeof(PetscInt) == sizeof(llint)) {
#if !defined(PETSC_HAVE_DEVICE)
    if (nPetscInt == 8) PackInit_IntegerType<llint, 8, 1>(link);
    else if (nPetscInt % 8 == 0) PackInit_IntegerType<llint, 8, 0>(link);
    else if (nPetscInt == 4) PackInit_IntegerType<llint, 4, 1>(link);
    else if (nPetscInt % 4 == 0) PackInit_IntegerType<llint, 4, 0>(link);
    else if (nPetscInt == 2) PackInit_IntegerType<llint, 2, 1>(link);
    else if (nPetscInt % 2 == 0) PackInit_IntegerType<llint, 2, 0>(link);
    else if (nPetscInt == 1) PackInit_IntegerType<llint, 1, 1>(link);
    else if (nPetscInt % 1 == 0)
#endif
      PackInit_IntegerType<llint, 1, 0>(link);
  } else if (nInt) {
#if !defined(PETSC_HAVE_DEVICE)
    if (nInt == 8) PackInit_IntegerType<int, 8, 1>(link);
    else if (nInt % 8 == 0) PackInit_IntegerType<int, 8, 0>(link);
    else if (nInt == 4) PackInit_IntegerType<int, 4, 1>(link);
    else if (nInt % 4 == 0) PackInit_IntegerType<int, 4, 0>(link);
    else if (nInt == 2) PackInit_IntegerType<int, 2, 1>(link);
    else if (nInt % 2 == 0) PackInit_IntegerType<int, 2, 0>(link);
    else if (nInt == 1) PackInit_IntegerType<int, 1, 1>(link);
    else if (nInt % 1 == 0)
#endif
      PackInit_IntegerType<int, 1, 0>(link);
  } else if (nSignedChar) {
#if !defined(PETSC_HAVE_DEVICE)
    if (nSignedChar == 8) PackInit_IntegerType<SignedChar, 8, 1>(link);
    else if (nSignedChar % 8 == 0) PackInit_IntegerType<SignedChar, 8, 0>(link);
    else if (nSignedChar == 4) PackInit_IntegerType<SignedChar, 4, 1>(link);
    else if (nSignedChar % 4 == 0) PackInit_IntegerType<SignedChar, 4, 0>(link);
    else if (nSignedChar == 2) PackInit_IntegerType<SignedChar, 2, 1>(link);
    else if (nSignedChar % 2 == 0) PackInit_IntegerType<SignedChar, 2, 0>(link);
    else if (nSignedChar == 1) PackInit_IntegerType<SignedChar, 1, 1>(link);
    else if (nSignedChar % 1 == 0)
#endif
      PackInit_IntegerType<SignedChar, 1, 0>(link);
  } else if (nUnsignedChar) {
#if !defined(PETSC_HAVE_DEVICE)
    if (nUnsignedChar == 8) PackInit_IntegerType<UnsignedChar, 8, 1>(link);
    else if (nUnsignedChar % 8 == 0) PackInit_IntegerType<UnsignedChar, 8, 0>(link);
    else if (nUnsignedChar == 4) PackInit_IntegerType<UnsignedChar, 4, 1>(link);
    else if (nUnsignedChar % 4 == 0) PackInit_IntegerType<UnsignedChar, 4, 0>(link);
    else if (nUnsignedChar == 2) PackInit_IntegerType<UnsignedChar, 2, 1>(link);
    else if (nUnsignedChar % 2 == 0) PackInit_IntegerType<UnsignedChar, 2, 0>(link);
    else if (nUnsignedChar == 1) PackInit_IntegerType<UnsignedChar, 1, 1>(link);
    else if (nUnsignedChar % 1 == 0)
#endif
      PackInit_IntegerType<UnsignedChar, 1, 0>(link);
#if defined(PETSC_HAVE_COMPLEX)
  } else if (nPetscComplex) {
  #if !defined(PETSC_HAVE_DEVICE)
    if (nPetscComplex == 8) PackInit_ComplexType<PetscComplex, 8, 1>(link);
    else if (nPetscComplex % 8 == 0) PackInit_ComplexType<PetscComplex, 8, 0>(link);
    else if (nPetscComplex == 4) PackInit_ComplexType<PetscComplex, 4, 1>(link);
    else if (nPetscComplex % 4 == 0) PackInit_ComplexType<PetscComplex, 4, 0>(link);
    else if (nPetscComplex == 2) PackInit_ComplexType<PetscComplex, 2, 1>(link);
    else if (nPetscComplex % 2 == 0) PackInit_ComplexType<PetscComplex, 2, 0>(link);
    else if (nPetscComplex == 1) PackInit_ComplexType<PetscComplex, 1, 1>(link);
    else if (nPetscComplex % 1 == 0)
  #endif
      PackInit_ComplexType<PetscComplex, 1, 0>(link);
#endif
  } else {
    MPI_Aint lb, nbyte;
    PetscCallMPI(MPI_Type_get_extent(unit, &lb, &nbyte));
    PetscCheck(lb == 0, PETSC_COMM_SELF, PETSC_ERR_SUP, "Datatype with nonzero lower bound %ld", (long)lb);
    if (nbyte % sizeof(int)) { /* If the type size is not multiple of int */
#if !defined(PETSC_HAVE_DEVICE)
      if (nbyte == 4) PackInit_DumbType<char, 4, 1>(link);
      else if (nbyte % 4 == 0) PackInit_DumbType<char, 4, 0>(link);
      else if (nbyte == 2) PackInit_DumbType<char, 2, 1>(link);
      else if (nbyte % 2 == 0) PackInit_DumbType<char, 2, 0>(link);
      else if (nbyte == 1) PackInit_DumbType<char, 1, 1>(link);
      else if (nbyte % 1 == 0)
#endif
        PackInit_DumbType<char, 1, 0>(link);
    } else {
      nInt = nbyte / sizeof(int);
#if !defined(PETSC_HAVE_DEVICE)
      if (nInt == 8) PackInit_DumbType<int, 8, 1>(link);
      else if (nInt % 8 == 0) PackInit_DumbType<int, 8, 0>(link);
      else if (nInt == 4) PackInit_DumbType<int, 4, 1>(link);
      else if (nInt % 4 == 0) PackInit_DumbType<int, 4, 0>(link);
      else if (nInt == 2) PackInit_DumbType<int, 2, 1>(link);
      else if (nInt % 2 == 0) PackInit_DumbType<int, 2, 0>(link);
      else if (nInt == 1) PackInit_DumbType<int, 1, 1>(link);
      else if (nInt % 1 == 0)
#endif
        PackInit_DumbType<int, 1, 0>(link);
    }
  }

  if (!sf->maxResidentThreadsPerGPU) { /* Not initialized */
    int                   device;
    struct cudaDeviceProp props;
    PetscCallCUDA(cudaGetDevice(&device));
    PetscCallCUDA(cudaGetDeviceProperties(&props, device));
    sf->maxResidentThreadsPerGPU = props.maxThreadsPerMultiProcessor * props.multiProcessorCount;
  }
  link->maxResidentThreadsPerGPU = sf->maxResidentThreadsPerGPU;

  link->stream       = PetscDefaultCudaStream;
  link->Destroy      = PetscSFLinkDestroy_MPI_CUDA;
  link->SyncDevice   = PetscSFLinkSyncDevice_CUDA;
  link->SyncStream   = PetscSFLinkSyncStream_CUDA;
  link->Memcpy       = PetscSFLinkMemcpy_CUDA;
  link->deviceinited = PETSC_TRUE;
  PetscFunctionReturn(0);
}
