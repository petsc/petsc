#pragma once

/* Shared CUPM (CUDA/HIP) implementations for SeqAIJCUSPARSE and SeqAIJHIPSPARSE
   that do not depend on the cuSPARSE/hipSPARSE library proper.

   Include ordering requirement: the vendor-specific impl header
   (cusparsematimpl.h or hipsparsematimpl.h) must be included before this
   header so that CsrMatrix, THRUSTINTARRAY*, THRUSTARRAY and all device-specific
   struct types are visible when this header is processed.

   Instantiated by:
     aijcusparse.cu      (DeviceType::CUDA, using MatSeqAIJCUSPARSE_Policy)
     aijhipsparse.hip.cxx (DeviceType::HIP,  using MatSeqAIJHIPSPARSE_Policy) */

#include <petsc/private/cupmobject.hpp>
#include <petsc/private/cupmblasinterface.hpp>
#include <petsc/private/matimpl.h>
#include <../src/sys/objects/device/impls/cupm/cupmthrustutility.hpp>
#include <../src/mat/impls/aij/seq/aij.h>

#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/equal.h>

/* Forward declaration of SeqAIJ fallback function used inside template */
PETSC_INTERN PetscErrorCode MatGetDiagonal_SeqAIJ(Mat, Vec);

namespace Petsc
{

namespace mat
{

namespace aij
{

namespace cupm
{

namespace impl
{

/* --------------------------------------------------------------------------
   Shared device functor: left-scale CSR rows.
   cprow[i] gives the logical row index for compressed row i; NULL = identity.
   -------------------------------------------------------------------------- */
struct DiagonalScaleLeft_CSR_Functor {
  const int         *row_ptr;
  PetscScalar       *val_ptr;
  const PetscScalar *lv_ptr;
  const PetscInt    *cprow;

  PETSC_HOSTDEVICE_INLINE_DECL void operator()(int i) const
  {
    const int         row = cprow ? (int)cprow[i] : i;
    const PetscScalar s   = lv_ptr[row];
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) val_ptr[j] *= s;
  }
};

/* --------------------------------------------------------------------------
   Shared device functor: get<1>(t) = get<0>(t).
   Replaces the identical VecCUDAEquals / VecHIPEquals structs.
   -------------------------------------------------------------------------- */
struct VecCUPMEquals {
  template <typename Tuple>
  PETSC_HOSTDEVICE_INLINE_DECL void operator()(Tuple t) const
  {
    thrust::get<1>(t) = thrust::get<0>(t);
  }
};

/* --------------------------------------------------------------------------
   Shared __global__ kernel: accumulate COO values into a CSR array.
   __global__ is valid for both nvcc and hipcc; the body is identical.
   -------------------------------------------------------------------------- */
__global__ static void MatAddCOOValues(const PetscScalar kv[], PetscCount nnz, const PetscCount jmap[], const PetscCount perm[], InsertMode imode, PetscScalar a[])
{
  PetscCount       i         = blockIdx.x * blockDim.x + threadIdx.x;
  const PetscCount grid_size = gridDim.x * blockDim.x;
  for (; i < nnz; i += grid_size) {
    PetscScalar sum = 0.0;
    for (PetscCount k = jmap[i]; k < jmap[i + 1]; k++) sum += kv[perm[k]];
    a[i] = (imode == INSERT_VALUES ? (PetscScalar)0.0 : a[i]) + sum;
  }
}

/* --------------------------------------------------------------------------
   Shared __global__ kernel: extract the CSR diagonal.
   -------------------------------------------------------------------------- */
__global__ static void GetDiagonal_CSR(const int *row, const int *col, const PetscScalar *val, const PetscInt len, PetscScalar *diag)
{
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < (size_t)len) {
    const PetscInt rowx = row[x], num_non0_row = row[x + 1] - rowx;
    PetscScalar    d = 0.0;

    for (PetscInt i = 0; i < num_non0_row; i++) {
      if (col[i + rowx] == (PetscInt)x) {
        d = val[i + rowx];
        break;
      }
    }
    diag[x] = d;
  }
}

/* ==========================================================================
   MatSeqAIJCUSPARSE_CUPM<T, Policy>

   Policy (C++11 traits class) requirements - all static methods:

     // Device struct types
     typedef ... mat_struct_type;       // Mat_SeqAIJCUSPARSE / Mat_SeqAIJHIPSPARSE
     typedef ... mult_struct_type;      // ...MultStruct equivalent

     // Storage-format constants (value of each format enumerator)
     static int storage_format_csr();
     static int storage_format_ell();
     static int storage_format_hyb();

     // Bookkeeping helpers (device-type specific)
     static PetscErrorCode CopyToGPU(Mat);
     static PetscErrorCode CopyFromGPU(Mat);
     static PetscErrorCode InvalidateTranspose(Mat, PetscBool);
     static PetscErrorCode ConvertFromSeqAIJ(Mat, MatType, MatReuse, Mat *);
     static const char    *mat_type_name;   // "seqaijcusparse" / "seqaijhipsparse"

     // Destruction helpers (device-type specific)
     static PetscErrorCode Destroy(Mat);
     static PetscErrorCode TriFactorsDestroy(void **);

     // Compose-function keys that differ between CUDA and HIP
     static const char *set_format_c;          // "MatCUSPARSESetFormat_C"        / "MatHIPSPARSESetFormat_C"
     static const char *set_use_cpu_solve_c;   // "MatCUSPARSESetUseCPUSolve_C"   / "MatHIPSPARSESetUseCPUSolve_C"
     static const char *product_seqdense_device_c; // "...seqdensecuda_C"          / "...seqdensehip_C"
     static const char *product_seqdense_c;    // "...seqdense_C"
     static const char *product_self_c;        // "...seqaijcusparse_C"           / "...seqaijhipsparse_C"
     static const char *seq_convert_hypre_c;   // "MatConvert_seqaijcusparse_hypre_C" / "_seqaijhipsparse_hypre_C"

     // Vec device-array access (device-type specific)
     static PetscErrorCode VecGetArrayRead  (Vec, const PetscScalar **);
     static PetscErrorCode VecRestoreArrayRead(Vec, const PetscScalar **);
     static PetscErrorCode VecGetArrayWrite (Vec, PetscScalar **);
     static PetscErrorCode VecRestoreArrayWrite(Vec, PetscScalar **);
   ========================================================================== */

template <device::cupm::DeviceType T, typename Policy>
struct MatSeqAIJCUSPARSE_CUPM : device::cupm::impl::CUPMObject<T> {
  PETSC_CUPMOBJECT_HEADER(T);

  typedef typename Policy::mat_struct_type  MatStructType;
  typedef typename Policy::mult_struct_type MultStructType;

  /* -------------------------------------------------------------------
     Tier 1 - Trivial
     ------------------------------------------------------------------- */

  /* MatAssemblyEnd: delegation to SeqAIJ */
  static PetscErrorCode AssemblyEnd(Mat A, MatAssemblyType mode) noexcept
  {
    PetscFunctionBegin;
    PetscCall(MatAssemblyEnd_SeqAIJ(A, mode));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatDuplicate */
  static PetscErrorCode Duplicate(Mat A, MatDuplicateOption cpvalues, Mat *B) noexcept
  {
    PetscFunctionBegin;
    PetscCall(MatDuplicate_SeqAIJ(A, cpvalues, B));
    PetscCall(Policy::ConvertFromSeqAIJ(*B, Policy::mat_type_name, MAT_INPLACE_MATRIX, B));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatGetCurrentMemType */
  static PetscErrorCode GetCurrentMemType(PETSC_UNUSED Mat A, PetscMemType *m) noexcept
  {
    PetscFunctionBegin;
    *m = PETSC_MEMTYPE_CUPM();
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatCOOStructDestroy: free device jmap and perm fields */
  static PetscErrorCode COOStructDestroy(PetscCtxRt ctx) noexcept
  {
    MatCOOStruct_SeqAIJ *coo = *(MatCOOStruct_SeqAIJ **)ctx;

    PetscFunctionBegin;
    PetscCallCUPM(cupmFree(coo->perm));
    PetscCallCUPM(cupmFree(coo->jmap));
    PetscCall(PetscFree(coo));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* -------------------------------------------------------------------
     Tier 2 - Straightforward
     ------------------------------------------------------------------- */

  /* MatZeroEntries: fill device CSR values with zero */
  static PetscErrorCode ZeroEntries(Mat A) noexcept
  {
    PetscBool      gpu = PETSC_FALSE;
    Mat_SeqAIJ    *a   = (Mat_SeqAIJ *)A->data;
    MatStructType *spptr;

    PetscFunctionBegin;
    if (A->factortype == MAT_FACTOR_NONE) {
      spptr = (MatStructType *)A->spptr;
      if (spptr->mat) {
        CsrMatrix *matrix = (CsrMatrix *)spptr->mat->mat;
        if (matrix->values) {
          gpu = PETSC_TRUE;
          PetscCallThrust(thrust::fill(thrust::device, matrix->values->begin(), matrix->values->end(), (PetscScalar)0.));
        }
      }
      if (spptr->matTranspose) {
        CsrMatrix *matrix = (CsrMatrix *)spptr->matTranspose->mat;
        if (matrix->values) PetscCallThrust(thrust::fill(thrust::device, matrix->values->begin(), matrix->values->end(), (PetscScalar)0.));
      }
    }
    if (gpu) A->offloadmask = PETSC_OFFLOAD_GPU;
    else {
      PetscCall(PetscArrayzero(a->a, a->i[A->rmap->n]));
      A->offloadmask = PETSC_OFFLOAD_CPU;
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatScale: cupmBlasXscal on the device CSR values */
  static PetscErrorCode Scale(Mat Y, PetscScalar a) noexcept
  {
    Mat_SeqAIJ      *y  = (Mat_SeqAIJ *)Y->data;
    PetscScalar     *ay = nullptr;
    cupmBlasHandle_t blashandle;
    PetscBLASInt     one = 1, bnz = 1;

    PetscFunctionBegin;
    PetscCall(GetArray(Y, &ay));
    PetscCall(GetHandles_(&blashandle));
    PetscCall(PetscBLASIntCast(y->nz, &bnz));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMBLAS(cupmBlasXscal(blashandle, bnz, cupmScalarPtrCast(&a), cupmScalarPtrCast(ay), one));
    PetscCall(PetscLogGpuFlops(bnz));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(RestoreArray(Y, &ay));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatDiagonalScale: Thrust-based left and right scaling of CSR values */
  static PetscErrorCode DiagonalScale(Mat A, Vec ll, Vec rr) noexcept
  {
    Mat_SeqAIJ    *aij = (Mat_SeqAIJ *)A->data;
    MatStructType *devstruct;
    CsrMatrix     *csr;
    PetscScalar   *av = nullptr;
    PetscInt       m, n, nz = aij->nz;
    cupmStream_t   stream;

    PetscFunctionBegin;
    PetscCall(GetHandles_(&stream));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(GetArray(A, &av));
    devstruct = (MatStructType *)A->spptr;
    csr       = (CsrMatrix *)devstruct->mat->mat;
    if (ll) {
      const PetscScalar *lv;
      PetscCall(VecGetLocalSize(ll, &m));
      PetscCheck(m == A->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Left scaling vector wrong length");
      PetscCall(Policy::VecGetArrayRead(ll, &lv));
      {
        const PetscInt               *cprow   = devstruct->mat->cprowIndices ? devstruct->mat->cprowIndices->data().get() : NULL;
        DiagonalScaleLeft_CSR_Functor functor = {csr->row_offsets->data().get(), av, lv, cprow};
        PetscCallThrust(THRUST_CALL(thrust::for_each, stream, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(csr->num_rows), functor));
      }
      PetscCall(Policy::VecRestoreArrayRead(ll, &lv));
      PetscCall(PetscLogGpuFlops(nz));
    }
    if (rr) {
      const PetscScalar *rv;
      PetscCall(VecGetLocalSize(rr, &n));
      PetscCheck(n == A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Right scaling vector wrong length");
      PetscCall(Policy::VecGetArrayRead(rr, &rv));
#if PetscDefined(USING_NVCC) && CCCL_VERSION >= 3001000
      PetscCallThrust(THRUST_CALL(thrust::transform, stream, csr->values->begin(), csr->values->end(), thrust::make_permutation_iterator(thrust::device_pointer_cast(rv), csr->column_indices->begin()), csr->values->begin(), cuda::std::multiplies<PetscScalar>()));
#else
      PetscCallThrust(THRUST_CALL(thrust::transform, stream, csr->values->begin(), csr->values->end(), thrust::make_permutation_iterator(thrust::device_pointer_cast(rv), csr->column_indices->begin()), csr->values->begin(), thrust::multiplies<PetscScalar>()));
#endif
      PetscCall(Policy::VecRestoreArrayRead(rr, &rv));
      PetscCall(PetscLogGpuFlops(nz));
    }
    PetscCall(RestoreArray(A, &av));
    PetscCall(PetscLogGpuTimeEnd());
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatSeqAIJGetIJ: return device CSR row-pointer and column-index arrays */
  static PetscErrorCode GetIJ(Mat A, PetscBool compressed, const int **i, const int **j) noexcept
  {
    MatStructType *cusp = (MatStructType *)A->spptr;
    Mat_SeqAIJ    *a    = (Mat_SeqAIJ *)A->data;
    CsrMatrix     *csr;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
    if (!i || !j) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCheckTypeName(A, Policy::mat_type_name);
    PetscCheck(cusp->format != (decltype(cusp->format))Policy::storage_format_ell() && cusp->format != (decltype(cusp->format))Policy::storage_format_hyb(), PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented");
    PetscCall(Policy::CopyToGPU(A));
    PetscCheck(cusp->mat, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing MultStruct");
    csr = (CsrMatrix *)cusp->mat->mat;
    if (i) {
      if (!compressed && a->compressedrow.use) { /* need full row offset */
        if (!cusp->rowoffsets_gpu) {
          cusp->rowoffsets_gpu = new THRUSTINTARRAY32(A->rmap->n + 1);
          cusp->rowoffsets_gpu->assign(a->i, a->i + A->rmap->n + 1);
          PetscCall(PetscLogCpuToGpu((A->rmap->n + 1) * sizeof(PetscInt)));
        }
        *i = cusp->rowoffsets_gpu->data().get();
      } else *i = csr->row_offsets->data().get();
    }
    if (j) *j = csr->column_indices->data().get();
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatSeqAIJRestoreIJ: nullify the pointers previously obtained with GetIJ */
  static PetscErrorCode RestoreIJ(Mat A, PetscBool compressed, const int **i, const int **j) noexcept
  {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
    PetscCheckTypeName(A, Policy::mat_type_name);
    if (i) *i = NULL;
    if (j) *j = NULL;
    (void)compressed;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatSetPreallocationCOO: copy COO bookkeeping struct to device */
  static PetscErrorCode SetPreallocationCOO(Mat mat, PetscCount coo_n, PetscInt coo_i[], PetscInt coo_j[]) noexcept
  {
    PetscBool            dev_ij = PETSC_FALSE;
    PetscMemType         mtype  = PETSC_MEMTYPE_HOST;
    PetscInt            *i, *j;
    PetscContainer       container_h;
    MatCOOStruct_SeqAIJ *coo_h, *coo_d;

    PetscFunctionBegin;
    PetscCall(PetscGetMemType(coo_i, &mtype));
    if (PetscMemTypeDevice(mtype)) {
      dev_ij = PETSC_TRUE;
      PetscCall(PetscMalloc2(coo_n, &i, coo_n, &j));
      PetscCallCUPM(cupmMemcpy(i, coo_i, coo_n * sizeof(PetscInt), cupmMemcpyDeviceToHost));
      PetscCallCUPM(cupmMemcpy(j, coo_j, coo_n * sizeof(PetscInt), cupmMemcpyDeviceToHost));
    } else {
      i = coo_i;
      j = coo_j;
    }
    PetscCall(MatSetPreallocationCOO_SeqAIJ(mat, coo_n, i, j));
    if (dev_ij) PetscCall(PetscFree2(i, j));
    mat->offloadmask = PETSC_OFFLOAD_CPU;
    /* Create the GPU memory */
    PetscCall(Policy::CopyToGPU(mat));

    /* Copy the COO struct to device */
    PetscCall(PetscObjectQuery((PetscObject)mat, "__PETSc_MatCOOStruct_Host", (PetscObject *)&container_h));
    PetscCall(PetscContainerGetPointer(container_h, (void **)&coo_h));
    PetscCall(PetscMalloc1(1, &coo_d));
    *coo_d = *coo_h; /* shallow copy; device fields amended below */
    PetscCallCUPM(cupmMalloc((void **)&coo_d->jmap, (coo_h->nz + 1) * sizeof(PetscCount)));
    PetscCallCUPM(cupmMemcpy(coo_d->jmap, coo_h->jmap, (coo_h->nz + 1) * sizeof(PetscCount), cupmMemcpyHostToDevice));
    PetscCallCUPM(cupmMalloc((void **)&coo_d->perm, coo_h->Atot * sizeof(PetscCount)));
    PetscCallCUPM(cupmMemcpy(coo_d->perm, coo_h->perm, coo_h->Atot * sizeof(PetscCount), cupmMemcpyHostToDevice));

    PetscCall(PetscObjectContainerCompose((PetscObject)mat, "__PETSc_MatCOOStruct_Device", coo_d, MatSeqAIJCUSPARSE_CUPM::COOStructDestroy));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatSetValuesCOO: launch MatAddCOOValues kernel */
  static PetscErrorCode SetValuesCOO(Mat A, const PetscScalar v[], InsertMode imode) noexcept
  {
    Mat_SeqAIJ          *seq  = (Mat_SeqAIJ *)A->data;
    MatStructType       *dev  = (MatStructType *)A->spptr;
    PetscCount           Annz = seq->nz;
    PetscMemType         memtype;
    const PetscScalar   *v1 = v;
    PetscScalar         *Aa = nullptr;
    PetscContainer       container;
    MatCOOStruct_SeqAIJ *coo;
    cupmStream_t         stream;

    PetscFunctionBegin;
    if (!dev->mat) PetscCall(Policy::CopyToGPU(A));

    PetscCall(PetscObjectQuery((PetscObject)A, "__PETSc_MatCOOStruct_Device", (PetscObject *)&container));
    PetscCall(PetscContainerGetPointer(container, (void **)&coo));

    PetscCall(PetscGetMemType(v, &memtype));
    if (PetscMemTypeHost(memtype)) { /* copy host values to device */
      PetscCallCUPM(cupmMalloc((void **)&v1, coo->n * sizeof(PetscScalar)));
      PetscCallCUPM(cupmMemcpy((void *)v1, v, coo->n * sizeof(PetscScalar), cupmMemcpyHostToDevice));
    }

    if (imode == INSERT_VALUES) PetscCall(GetArrayWrite(A, &Aa));
    else PetscCall(GetArray(A, &Aa));

    PetscCall(GetHandles_(&stream));
    PetscCall(PetscLogGpuTimeBegin());
    if (Annz) {
      PetscCallCUPM(cupmLaunchKernel(MatAddCOOValues, (unsigned int)((Annz + 255) / 256), 256u, (size_t)0, stream, v1, Annz, coo->jmap, coo->perm, imode, Aa));
      PetscCallCUPM(cupmGetLastError());
    }
    PetscCall(PetscLogGpuTimeEnd());

    if (imode == INSERT_VALUES) PetscCall(RestoreArrayWrite(A, &Aa));
    else PetscCall(RestoreArray(A, &Aa));

    if (PetscMemTypeHost(memtype)) {
      void *v1_device = (void *)v1;
      PetscCallCUPM(cupmFree(v1_device));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatSeqAIJCopySubArray: scatter-gather a sub-array of CSR values */
  static PetscErrorCode CopySubArray(Mat A, PetscInt n, const PetscInt idx[], PetscScalar v[]) noexcept
  {
    const PetscScalar *av = nullptr;
    PetscMemType       mtype;
    PetscBool          dmem;

    PetscFunctionBegin;
    PetscCall(PetscCUPMGetMemType(v, &mtype));
    dmem = PetscMemTypeDevice(mtype);
    PetscCall(GetArrayRead(A, &av));
    if (n && idx) {
      THRUSTINTARRAY widx(n);
      widx.assign(idx, idx + n);
      PetscCall(PetscLogCpuToGpu(n * sizeof(PetscInt)));

      THRUSTARRAY                    *w = NULL;
      thrust::device_ptr<PetscScalar> dv;
      if (dmem) {
        dv = thrust::device_pointer_cast(v);
      } else {
        w  = new THRUSTARRAY(n);
        dv = w->data();
      }
      {
        thrust::device_ptr<const PetscScalar> dav   = thrust::device_pointer_cast(av);
        auto                                  zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(dav, widx.begin()), dv));
        auto                                  zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(dav, widx.end()), dv + n));
        PetscCallThrust(thrust::for_each(zibit, zieit, VecCUPMEquals{}));
      }
      if (w) PetscCallCUPM(cupmMemcpy(v, w->data().get(), n * sizeof(PetscScalar), cupmMemcpyDeviceToHost));
      delete w;
    } else {
      PetscCallCUPM(cupmMemcpy(v, av, n * sizeof(PetscScalar), dmem ? cupmMemcpyDeviceToDevice : cupmMemcpyDeviceToHost));
    }
    if (!dmem) PetscCall(PetscLogCpuToGpu(n * sizeof(PetscScalar)));
    PetscCall(RestoreArrayRead(A, &av));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* -------------------------------------------------------------------
     Tier 3 - AXPY shared branches (SAME_NZ and DIFFERENT_NZ only).
     The SUBSET_NZ branch calls cuSPARSE/hipSPARSE and stays in the caller.
     ------------------------------------------------------------------- */

  /* AXPY SAME_NONZERO_PATTERN branch: cupmBlasXaxpy */
  static PetscErrorCode AXPY_SameNZ(Mat Y, PetscScalar a, Mat X) noexcept
  {
    Mat_SeqAIJ        *x  = (Mat_SeqAIJ *)X->data;
    const PetscScalar *ax = nullptr;
    PetscScalar       *ay = nullptr;
    cupmBlasHandle_t   blashandle;
    PetscBLASInt       one = 1, bnz = 1;

    PetscFunctionBegin;
    PetscCall(GetArrayRead(X, &ax));
    PetscCall(GetArray(Y, &ay));
    PetscCall(GetHandles_(&blashandle));
    PetscCall(PetscBLASIntCast(x->nz, &bnz));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMBLAS(cupmBlasXaxpy(blashandle, bnz, cupmScalarPtrCast(&a), cupmScalarPtrCast(ax), one, cupmScalarPtrCast(ay), one));
    PetscCall(PetscLogGpuFlops(2.0 * bnz));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(RestoreArrayRead(X, &ax));
    PetscCall(RestoreArray(Y, &ay));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* GetDiagonal: kernel-based extraction of the CSR diagonal */
  static PetscErrorCode GetDiagonal(Mat A, Vec diag) noexcept
  {
    MatStructType  *devstruct = (MatStructType *)A->spptr;
    MultStructType *matstruct = (MultStructType *)devstruct->mat;
    PetscScalar    *darray;
    cupmStream_t    stream;

    PetscFunctionBegin;
    if (A->offloadmask == PETSC_OFFLOAD_BOTH || A->offloadmask == PETSC_OFFLOAD_GPU) {
      PetscInt   n   = A->rmap->n;
      CsrMatrix *mat = (CsrMatrix *)matstruct->mat;

      PetscCheck(devstruct->format == (decltype(devstruct->format))Policy::storage_format_csr(), PETSC_COMM_SELF, PETSC_ERR_SUP, "Only CSR format supported");
      if (n > 0) {
        PetscCall(Policy::VecGetArrayWrite(diag, &darray));
        PetscCall(GetHandles_(&stream));
        PetscCallCUPM(cupmLaunchKernel(GetDiagonal_CSR, (unsigned int)((n + 255) / 256), 256u, (size_t)0, stream, mat->row_offsets->data().get(), mat->column_indices->data().get(), mat->values->data().get(), n, darray));
        PetscCallCUPM(cupmGetLastError());
        PetscCall(Policy::VecRestoreArrayWrite(diag, &darray));
      }
    } else {
      PetscCall(MatGetDiagonal_SeqAIJ(A, diag));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* -------------------------------------------------------------------
     Tier 4 - Device array access (moved here from vendor files so both
     SeqAIJCUSPARSE and SeqAIJHIPSPARSE share one implementation).
     ------------------------------------------------------------------- */

  /* GetArrayRead: read-only access to device CSR value array */
  static PetscErrorCode GetArrayRead(Mat A, const PetscScalar **a) noexcept
  {
    MatStructType *cusp = (MatStructType *)A->spptr;
    CsrMatrix     *csr;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
    PetscAssertPointer(a, 2);
    PetscCheckTypeName(A, Policy::mat_type_name);
    PetscCheck(cusp->format != (decltype(cusp->format))Policy::storage_format_ell() && cusp->format != (decltype(cusp->format))Policy::storage_format_hyb(), PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented");
    PetscCall(Policy::CopyToGPU(A));
    PetscCheck(cusp->mat, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing MultStruct");
    csr = (CsrMatrix *)cusp->mat->mat;
    PetscCheck(csr->values, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing device memory");
    *a = csr->values->data().get();
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* RestoreArrayRead: release read-only access obtained from GetArrayRead */
  static PetscErrorCode RestoreArrayRead(Mat A, const PetscScalar **a) noexcept
  {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
    PetscAssertPointer(a, 2);
    PetscCheckTypeName(A, Policy::mat_type_name);
    *a = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* GetArray: read-write access to device CSR value array */
  static PetscErrorCode GetArray(Mat A, PetscScalar **a) noexcept
  {
    MatStructType *cusp = (MatStructType *)A->spptr;
    CsrMatrix     *csr;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
    PetscAssertPointer(a, 2);
    PetscCheckTypeName(A, Policy::mat_type_name);
    PetscCheck(cusp->format != (decltype(cusp->format))Policy::storage_format_ell() && cusp->format != (decltype(cusp->format))Policy::storage_format_hyb(), PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented");
    PetscCall(Policy::CopyToGPU(A));
    PetscCheck(cusp->mat, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing MultStruct");
    csr = (CsrMatrix *)cusp->mat->mat;
    PetscCheck(csr->values, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing device memory");
    *a             = csr->values->data().get();
    A->offloadmask = PETSC_OFFLOAD_GPU;
    PetscCall(Policy::InvalidateTranspose(A, PETSC_FALSE));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* RestoreArray: restore read-write access obtained from GetArray */
  static PetscErrorCode RestoreArray(Mat A, PetscScalar **a) noexcept
  {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
    PetscAssertPointer(a, 2);
    PetscCheckTypeName(A, Policy::mat_type_name);
    PetscCall(PetscObjectStateIncrease((PetscObject)A));
    *a = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* GetArrayWrite: write-only access to device CSR value array (no host-to-device copy) */
  static PetscErrorCode GetArrayWrite(Mat A, PetscScalar **a) noexcept
  {
    MatStructType *cusp = (MatStructType *)A->spptr;
    CsrMatrix     *csr;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
    PetscAssertPointer(a, 2);
    PetscCheckTypeName(A, Policy::mat_type_name);
    PetscCheck(cusp->format != (decltype(cusp->format))Policy::storage_format_ell() && cusp->format != (decltype(cusp->format))Policy::storage_format_hyb(), PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented");
    PetscCheck(cusp->mat, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing MultStruct");
    csr = (CsrMatrix *)cusp->mat->mat;
    PetscCheck(csr->values, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing device memory");
    *a             = csr->values->data().get();
    A->offloadmask = PETSC_OFFLOAD_GPU;
    PetscCall(Policy::InvalidateTranspose(A, PETSC_FALSE));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* RestoreArrayWrite: restore write-only access obtained from GetArrayWrite */
  static PetscErrorCode RestoreArrayWrite(Mat A, PetscScalar **a) noexcept
  {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
    PetscAssertPointer(a, 2);
    PetscCheckTypeName(A, Policy::mat_type_name);
    PetscCall(PetscObjectStateIncrease((PetscObject)A));
    *a = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* SeqAIJGetArray: copy GPU-to-CPU then return host value array (ops->getarray) */
  static PetscErrorCode SeqAIJGetArray(Mat A, PetscScalar *array[]) noexcept
  {
    PetscFunctionBegin;
    PetscCall(Policy::CopyFromGPU(A));
    *array = ((Mat_SeqAIJ *)A->data)->a;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* SeqAIJRestoreArray: mark matrix data CPU-valid (ops->restorearray) */
  static PetscErrorCode SeqAIJRestoreArray(Mat A, PetscScalar *array[]) noexcept
  {
    PetscFunctionBegin;
    A->offloadmask = PETSC_OFFLOAD_CPU;
    *array         = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* SeqAIJGetArrayRead: copy GPU-to-CPU then return host value array read-only (ops->getarrayread) */
  static PetscErrorCode SeqAIJGetArrayRead(Mat A, const PetscScalar *array[]) noexcept
  {
    PetscFunctionBegin;
    PetscCall(Policy::CopyFromGPU(A));
    *array = ((Mat_SeqAIJ *)A->data)->a;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* SeqAIJRestoreArrayRead: release read-only host array (ops->restorearrayread) */
  static PetscErrorCode SeqAIJRestoreArrayRead(Mat /*A*/, const PetscScalar *array[]) noexcept
  {
    PetscFunctionBegin;
    *array = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* SeqAIJGetArrayWrite: return host value array for write-only access (ops->getarraywrite) */
  static PetscErrorCode SeqAIJGetArrayWrite(Mat A, PetscScalar *array[]) noexcept
  {
    PetscFunctionBegin;
    *array = ((Mat_SeqAIJ *)A->data)->a;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* SeqAIJRestoreArrayWrite: mark matrix data CPU-valid after write (ops->restorearraywrite) */
  static PetscErrorCode SeqAIJRestoreArrayWrite(Mat A, PetscScalar *array[]) noexcept
  {
    PetscFunctionBegin;
    A->offloadmask = PETSC_OFFLOAD_CPU;
    *array         = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* CreateSeqAIJ: allocate and preallocate a seq sparse matrix of this type */
  static PetscErrorCode CreateSeqAIJ(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt nz, const PetscInt nnz[], Mat *A) noexcept
  {
    PetscFunctionBegin;
    PetscCall(MatCreate(comm, A));
    PetscCall(MatSetSizes(*A, m, n, m, n));
    PetscCall(MatSetType(*A, Policy::mat_type_name));
    PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(*A, nz, (PetscInt *)nnz));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatDestroy: free vendor-specific state, deregister composed functions */
  static PetscErrorCode Destroy(Mat A) noexcept
  {
    PetscFunctionBegin;
    if (A->factortype == MAT_FACTOR_NONE) PetscCall(Policy::Destroy(A));
    else PetscCall(Policy::TriFactorsDestroy(&A->spptr));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSeqAIJCopySubArray_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, Policy::set_format_c, NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, Policy::set_use_cpu_solve_c, NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, Policy::product_seqdense_device_c, NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, Policy::product_seqdense_c, NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, Policy::product_self_c, NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatFactorGetSolverType_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, Policy::seq_convert_hypre_c, NULL));
    PetscCall(MatDestroy_SeqAIJ(A));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

} // namespace impl

} // namespace cupm

} // namespace aij

} // namespace mat

} // namespace Petsc
