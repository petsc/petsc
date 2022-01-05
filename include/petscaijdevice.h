#if !defined(PETSCAIJDEVICE_H)
#define PETSCAIJDEVICE_H

#include <petscmat.h>

#define CSRDataStructure(datatype)  \
  int         *i; \
  int         *j; \
  datatype    *a;\
  PetscInt    n;\
  PetscInt    ignorezeroentries;

typedef struct {
  CSRDataStructure(PetscScalar)
} PetscCSRDataStructure;

struct _n_SplitCSRMat {
  PetscInt              cstart,cend,rstart,rend;
  PetscCSRDataStructure diag,offdiag;
  int                   *colmap;
  PetscInt              N;
  PetscMPIInt           rank;
};

/* 64-bit floating-point version of atomicAdd() is only natively supported by
   CUDA devices of compute capability 6.x and higher. See also sfcuda.cu
*/
#if defined(PETSC_USE_REAL_DOUBLE) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
  __device__ double atomicAdd(double* x,double y) {
    typedef unsigned long long int ullint;
    double *address = x, val = y;
    ullint *address_as_ull = (ullint*)address;
    ullint old = *address_as_ull, assumed;
    do {
      assumed = old;
      old     = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
  }
#endif

#if defined(KOKKOS_INLINE_FUNCTION)
  #define PetscAtomicAdd(a,b) Kokkos::atomic_fetch_add(a, b)
#elif defined(__CUDA_ARCH__)
  #if defined(PETSC_USE_COMPLEX)
    #define PetscAtomicAdd(a,b) {       \
      PetscReal *_a = (PetscReal*)(a);  \
      PetscReal *_b = (PetscReal*)&(b); \
      atomicAdd(&_a[0],_b[0]);            \
      atomicAdd(&_a[1],_b[1]);            \
    }
  #else
    #define PetscAtomicAdd(a,b) atomicAdd(a,b)
  #endif
#else
  /* TODO: support devices other than CUDA and Kokkos */
  #define PetscAtomicAdd(a,b) *(a) += b
#endif

#define MatSetValues_SeqAIJ_A_Private(row,col,value,addv)              \
  {                                                                    \
  inserted = 0;                                                        \
  if (col <= lastcol1) low1 = 0;                                       \
  else                 high1 = nrow1;                                  \
  lastcol1 = col;                                                      \
  while (high1-low1 > 5) {                                             \
    t = (low1+high1)/2;                                                \
    if (rp1[t] > col) high1 = t;                                       \
    else              low1  = t;                                       \
  }                                                                    \
  for (_i=low1; _i<high1; _i++) {                                      \
    if (rp1[_i] > col) break;                                          \
    if (rp1[_i] == col) {                                              \
      if (addv == ADD_VALUES) {                                        \
        PetscAtomicAdd(&ap1[_i],value);                                \
      }                                                                \
      else ap1[_i] = value;                                            \
      inserted = 1;                                                    \
      break;                                                           \
    }                                                                  \
  }                                                                    \
}

#define MatSetValues_SeqAIJ_B_Private(row,col,value,addv)              \
  {                                                                    \
  inserted = 0;                                                        \
  if (col <= lastcol2) low2 = 0;                                       \
  else high2 = nrow2;                                                  \
  lastcol2 = col;                                                      \
  while (high2-low2 > 5) {                                             \
    t = (low2+high2)/2;                                                \
    if (rp2[t] > col) high2 = t;                                       \
    else              low2  = t;                                       \
  }                                                                    \
  for (_i=low2; _i<high2; _i++) {                                      \
    if (rp2[_i] > col) break;                                          \
    if (rp2[_i] == col) {                                              \
      if (addv == ADD_VALUES) {                                        \
        PetscAtomicAdd(&ap2[_i],value);                                \
      }                                                                \
      else ap2[_i] = value;                                            \
      inserted = 1;                                                    \
      break;                                                           \
    }                                                                  \
  }                                                                    \
}

#if defined(PETSC_USE_DEBUG)
#define SETERR {                                                                                 \
   printf("[%d] ERROR in %s() at %s:%d: Location (%ld,%ld) not found!\n",     \
          d_mat->rank,__func__,__FILE__,__LINE__,(long int)im[i],(long int)in[j]); \
   return PETSC_ERR_ARG_OUTOFRANGE;                                                              \
}
#else
#define SETERR { return PETSC_ERR_ARG_OUTOFRANGE; }
#endif

#if defined(__CUDA_ARCH__)
__device__
#elif defined(KOKKOS_INLINE_FUNCTION)
KOKKOS_INLINE_FUNCTION
#else
static
#endif

/*@C
       MatSetValuesDevice - sets a set of values into a matrix, this may be called by CUDA or KOKKOS kernels

    Input Parameters:
+   d_mat - an object obtained with MatCUSPARSEGetDeviceMatWrite() or MatKokkosGetDeviceMatWrite()
.   m - the number of rows to insert or add to
.   im - the rows to insert or add to
.   n - number of columns to insert or add to
.   in - the columns to insert or add to
.   v - the values to insert or add to the matrix (treated as a  by n row oriented dense array
+   is - either INSERT_VALUES or ADD_VALUES

    Notes:
      Any row or column indices that are outside the bounds of the matrix on the rank are discarded

.seealso: MatSetValues(), MatCreate(), MatCreateDenseCUDA(), MatCreateAIJCUSPARSE(), MatKokkosGetDeviceMatWrite(),
          MatCUSPARSEGetDeviceMatWrite()
@*/
PetscErrorCode MatSetValuesDevice(PetscSplitCSRDataStructure d_mat, PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  MatScalar       value;
  const int       *rp1,*rp2 = NULL,*ai = d_mat->diag.i, *aj = d_mat->diag.j;
  const int       *bi = d_mat->offdiag.i, *bj = d_mat->offdiag.j;
  MatScalar       *ba = d_mat->offdiag.a, *aa = d_mat->diag.a;
  int             nrow1,nrow2 = 0,_i,low1,high1,low2 = 0,high2 = 0,t,lastcol1,lastcol2 = 0,inserted;
  MatScalar       *ap1,*ap2 = NULL;
  PetscBool       roworiented = PETSC_TRUE;
  int             i,j,row,col;
  const PetscInt  rstart = d_mat->rstart,rend = d_mat->rend, cstart = d_mat->rstart,cend = d_mat->rend,N = d_mat->N;

  for (i=0; i<m; i++) {
    if (im[i] >= rstart && im[i] < rend) { // silently ignore off processor rows
      row      = (int)(im[i] - rstart);
      lastcol1 = -1;
      rp1      = aj + ai[row];
      ap1      = aa + ai[row];
      nrow1    = ai[row+1] - ai[row];
      low1     = 0;
      high1    = nrow1;
      if (bj) {
        lastcol2 = -1;
        rp2      = bj + bi[row];
        ap2      = ba + bi[row];
        nrow2    = bi[row+1] - bi[row];
        low2     = 0;
        high2    = nrow2;
      } else {
        high2 = low2 = 0;
      }
      for (j=0; j<n; j++) {
        value = roworiented ? v[i*n+j] : v[i+j*m];
        if (in[j] >= cstart && in[j] < cend) {
          col = (int)(in[j] - cstart);
          MatSetValues_SeqAIJ_A_Private(row,col,value,is);
          if (!inserted) SETERR;
        } else if (in[j] < 0) {
          continue;
        } else if (in[j] >= N) {
          continue;
        } else {
          col = d_mat->colmap[in[j]] - 1;
          if (col < 0) SETERR;
          MatSetValues_SeqAIJ_B_Private(row,col,value,is);
          if (!inserted) SETERR;
        }
      }
    }
  }
  return 0;
}

#undef MatSetValues_SeqAIJ_A_Private
#undef MatSetValues_SeqAIJ_B_Private
#undef SETERR
#undef PetscAtomicAdd
#undef PetscCSRDataStructure_

#endif
