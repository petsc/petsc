#pragma once

/* Shared CUPM (CUDA/HIP) implementations for MPIAIJCUSPARSE and MPIAIJHIPSPARSE
   that do not depend on the cuSPARSE/hipSPARSE library proper.

   Include ordering requirement: the vendor-specific MPI impl header
   (mpicusparsematimpl.h or mpihipsparsematimpl.h) must be included before
   this header so that Mat_MPIAIJCUSPARSE/Mat_MPIAIJHIPSPARSE types are visible.

   Instantiated by:
     mpiaijcusparse.cu      (DeviceType::CUDA, using MatMPIAIJCUSPARSE_Policy)
     mpiaijhipsparse.hip.cxx (DeviceType::HIP,  using MatMPIAIJHIPSPARSE_Policy) */

#include <petsc/private/cupmobject.hpp>
#include <petsc/private/matimpl.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscsf.h>

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
   Shared __global__ kernel: pack entries to be sent to remote.
   -------------------------------------------------------------------------- */
__global__ static void MatPackCOOValues_MPI(const PetscScalar kv[], PetscCount nnz, const PetscCount perm[], PetscScalar buf[])
{
  PetscCount       i         = blockIdx.x * blockDim.x + threadIdx.x;
  const PetscCount grid_size = gridDim.x * blockDim.x;
  for (; i < nnz; i += grid_size) buf[i] = kv[perm[i]];
}

/* --------------------------------------------------------------------------
   Shared __global__ kernel: add local COO values to diagonal and off-diagonal.
   -------------------------------------------------------------------------- */
__global__ static void MatAddLocalCOOValues_MPI(const PetscScalar kv[], InsertMode imode, PetscCount Annz, const PetscCount Ajmap1[], const PetscCount Aperm1[], PetscScalar Aa[], PetscCount Bnnz, const PetscCount Bjmap1[], const PetscCount Bperm1[], PetscScalar Ba[])
{
  PetscCount       i         = blockIdx.x * blockDim.x + threadIdx.x;
  const PetscCount grid_size = gridDim.x * blockDim.x;
  for (; i < Annz + Bnnz; i += grid_size) {
    PetscScalar sum = 0.0;
    if (i < Annz) {
      for (PetscCount k = Ajmap1[i]; k < Ajmap1[i + 1]; k++) sum += kv[Aperm1[k]];
      Aa[i] = (imode == INSERT_VALUES ? 0.0 : Aa[i]) + sum;
    } else {
      i -= Annz;
      for (PetscCount k = Bjmap1[i]; k < Bjmap1[i + 1]; k++) sum += kv[Bperm1[k]];
      Ba[i] = (imode == INSERT_VALUES ? 0.0 : Ba[i]) + sum;
    }
  }
}

/* --------------------------------------------------------------------------
   Shared __global__ kernel: add remote COO values to diagonal and off-diagonal.
   -------------------------------------------------------------------------- */
__global__ static void MatAddRemoteCOOValues_MPI(const PetscScalar kv[], PetscCount Annz2, const PetscCount Aimap2[], const PetscCount Ajmap2[], const PetscCount Aperm2[], PetscScalar Aa[], PetscCount Bnnz2, const PetscCount Bimap2[], const PetscCount Bjmap2[], const PetscCount Bperm2[], PetscScalar Ba[])
{
  PetscCount       i         = blockIdx.x * blockDim.x + threadIdx.x;
  const PetscCount grid_size = gridDim.x * blockDim.x;
  for (; i < Annz2 + Bnnz2; i += grid_size) {
    if (i < Annz2) {
      for (PetscCount k = Ajmap2[i]; k < Ajmap2[i + 1]; k++) Aa[Aimap2[i]] += kv[Aperm2[k]];
    } else {
      i -= Annz2;
      for (PetscCount k = Bjmap2[i]; k < Bjmap2[i + 1]; k++) Ba[Bimap2[i]] += kv[Bperm2[k]];
    }
  }
}

/* ==========================================================================
   MatMPIAIJCUSPARSE_CUPM<T, Policy>

   Policy (C++11 traits class) requirements - all static:

     typedef ... mat_struct_type;  // Mat_MPIAIJCUSPARSE / Mat_MPIAIJHIPSPARSE

     static const char *mpi_mat_type;   // MATMPIAIJCUSPARSE / MATMPIAIJHIPSPARSE
     static const char *seq_mat_type;   // MATSEQAIJCUSPARSE / MATSEQAIJHIPSPARSE
     static const char *vec_seq_type;   // VECSEQCUDA / VECSEQHIP

     // Seq sub-matrix device copy
     static PetscErrorCode CopyToGPU(Mat);

     // Seq sub-matrix merge (for GetLocalMatMerge)
     static PetscErrorCode MergeMats(Mat, Mat, MatReuse, Mat *);

     // Seq sub-matrix device array access (for SetValuesCOO)
     static PetscErrorCode GetArray     (Mat, PetscScalar **);
     static PetscErrorCode GetArrayWrite(Mat, PetscScalar **);
     static PetscErrorCode RestoreArray     (Mat, PetscScalar **);
     static PetscErrorCode RestoreArrayWrite(Mat, PetscScalar **);

     // Set cuSPARSE/hipSPARSE storage format on both sub-matrices
     static PetscErrorCode SetSubMatFormats(Mat, Mat, mat_struct_type *);

     // Compose-function keys that differ between CUDA and HIP
     static const char *set_format_c;          // "MatCUSPARSESetFormat_C"            / "MatHIPSPARSESetFormat_C"
     static const char *mpi_convert_hypre_c;   // "MatConvert_mpiaijcusparse_hypre_C" / "_mpiaijhipsparse_hypre_C"
   ========================================================================== */

template <device::cupm::DeviceType T, typename Policy>
struct MatMPIAIJCUSPARSE_CUPM : device::cupm::impl::CUPMObject<T> {
  PETSC_CUPMOBJECT_HEADER(T);

  typedef typename Policy::mat_struct_type MatStructType;

  /* MatCOOStructDestroy: release all device-side COO arrays */
  static PetscErrorCode COOStructDestroy(PetscCtxRt data) noexcept
  {
    MatCOOStruct_MPIAIJ *coo = *(MatCOOStruct_MPIAIJ **)data;

    PetscFunctionBegin;
    PetscCall(PetscSFDestroy(&coo->sf));
    PetscCallCUPM(cupmFree(coo->Ajmap1));
    PetscCallCUPM(cupmFree(coo->Aperm1));
    PetscCallCUPM(cupmFree(coo->Bjmap1));
    PetscCallCUPM(cupmFree(coo->Bperm1));
    PetscCallCUPM(cupmFree(coo->Aimap2));
    PetscCallCUPM(cupmFree(coo->Ajmap2));
    PetscCallCUPM(cupmFree(coo->Aperm2));
    PetscCallCUPM(cupmFree(coo->Bimap2));
    PetscCallCUPM(cupmFree(coo->Bjmap2));
    PetscCallCUPM(cupmFree(coo->Bperm2));
    PetscCallCUPM(cupmFree(coo->Cperm1));
    PetscCallCUPM(cupmFree(coo->sendbuf));
    PetscCallCUPM(cupmFree(coo->recvbuf));
    PetscCall(PetscFree(coo));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatSetPreallocationCOO: copy MPIAIJ COO bookkeeping struct to device */
  static PetscErrorCode SetPreallocationCOO(Mat mat, PetscCount coo_n, PetscInt coo_i[], PetscInt coo_j[]) noexcept
  {
    Mat_MPIAIJ          *mpiaij = (Mat_MPIAIJ *)mat->data;
    PetscBool            dev_ij = PETSC_FALSE;
    PetscMemType         mtype  = PETSC_MEMTYPE_HOST;
    PetscInt            *i, *j;
    PetscContainer       container_h;
    MatCOOStruct_MPIAIJ *coo_h, *coo_d;

    PetscFunctionBegin;
    PetscCall(PetscFree(mpiaij->garray));
    PetscCall(VecDestroy(&mpiaij->lvec));
#if defined(PETSC_USE_CTABLE)
    PetscCall(PetscHMapIDestroy(&mpiaij->colmap));
#else
    PetscCall(PetscFree(mpiaij->colmap));
#endif
    PetscCall(VecScatterDestroy(&mpiaij->Mvctx));
    mat->assembled     = PETSC_FALSE;
    mat->was_assembled = PETSC_FALSE;
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
    PetscCall(MatSetPreallocationCOO_MPIAIJ(mat, coo_n, i, j));
    if (dev_ij) PetscCall(PetscFree2(i, j));
    mat->offloadmask = PETSC_OFFLOAD_CPU;
    /* Create the GPU memory */
    PetscCall(Policy::CopyToGPU(mpiaij->A));
    PetscCall(Policy::CopyToGPU(mpiaij->B));

    /* Copy the COO struct to device */
    PetscCall(PetscObjectQuery((PetscObject)mat, "__PETSc_MatCOOStruct_Host", (PetscObject *)&container_h));
    PetscCall(PetscContainerGetPointer(container_h, (void **)&coo_h));
    PetscCall(PetscMalloc1(1, &coo_d));
    *coo_d = *coo_h; /* shallow copy; device fields amended below */
    PetscCall(PetscObjectReference((PetscObject)coo_d->sf));
    PetscCallCUPM(cupmMalloc((void **)&coo_d->Ajmap1, (coo_h->Annz + 1) * sizeof(PetscCount)));
    PetscCallCUPM(cupmMalloc((void **)&coo_d->Aperm1, coo_h->Atot1 * sizeof(PetscCount)));
    PetscCallCUPM(cupmMalloc((void **)&coo_d->Bjmap1, (coo_h->Bnnz + 1) * sizeof(PetscCount)));
    PetscCallCUPM(cupmMalloc((void **)&coo_d->Bperm1, coo_h->Btot1 * sizeof(PetscCount)));
    PetscCallCUPM(cupmMalloc((void **)&coo_d->Aimap2, coo_h->Annz2 * sizeof(PetscCount)));
    PetscCallCUPM(cupmMalloc((void **)&coo_d->Ajmap2, (coo_h->Annz2 + 1) * sizeof(PetscCount)));
    PetscCallCUPM(cupmMalloc((void **)&coo_d->Aperm2, coo_h->Atot2 * sizeof(PetscCount)));
    PetscCallCUPM(cupmMalloc((void **)&coo_d->Bimap2, coo_h->Bnnz2 * sizeof(PetscCount)));
    PetscCallCUPM(cupmMalloc((void **)&coo_d->Bjmap2, (coo_h->Bnnz2 + 1) * sizeof(PetscCount)));
    PetscCallCUPM(cupmMalloc((void **)&coo_d->Bperm2, coo_h->Btot2 * sizeof(PetscCount)));
    PetscCallCUPM(cupmMalloc((void **)&coo_d->Cperm1, coo_h->sendlen * sizeof(PetscCount)));
    PetscCallCUPM(cupmMalloc((void **)&coo_d->sendbuf, coo_h->sendlen * sizeof(PetscScalar)));
    PetscCallCUPM(cupmMalloc((void **)&coo_d->recvbuf, coo_h->recvlen * sizeof(PetscScalar)));
    PetscCallCUPM(cupmMemcpy(coo_d->Ajmap1, coo_h->Ajmap1, (coo_h->Annz + 1) * sizeof(PetscCount), cupmMemcpyHostToDevice));
    PetscCallCUPM(cupmMemcpy(coo_d->Aperm1, coo_h->Aperm1, coo_h->Atot1 * sizeof(PetscCount), cupmMemcpyHostToDevice));
    PetscCallCUPM(cupmMemcpy(coo_d->Bjmap1, coo_h->Bjmap1, (coo_h->Bnnz + 1) * sizeof(PetscCount), cupmMemcpyHostToDevice));
    PetscCallCUPM(cupmMemcpy(coo_d->Bperm1, coo_h->Bperm1, coo_h->Btot1 * sizeof(PetscCount), cupmMemcpyHostToDevice));
    PetscCallCUPM(cupmMemcpy(coo_d->Aimap2, coo_h->Aimap2, coo_h->Annz2 * sizeof(PetscCount), cupmMemcpyHostToDevice));
    PetscCallCUPM(cupmMemcpy(coo_d->Ajmap2, coo_h->Ajmap2, (coo_h->Annz2 + 1) * sizeof(PetscCount), cupmMemcpyHostToDevice));
    PetscCallCUPM(cupmMemcpy(coo_d->Aperm2, coo_h->Aperm2, coo_h->Atot2 * sizeof(PetscCount), cupmMemcpyHostToDevice));
    PetscCallCUPM(cupmMemcpy(coo_d->Bimap2, coo_h->Bimap2, coo_h->Bnnz2 * sizeof(PetscCount), cupmMemcpyHostToDevice));
    PetscCallCUPM(cupmMemcpy(coo_d->Bjmap2, coo_h->Bjmap2, (coo_h->Bnnz2 + 1) * sizeof(PetscCount), cupmMemcpyHostToDevice));
    PetscCallCUPM(cupmMemcpy(coo_d->Bperm2, coo_h->Bperm2, coo_h->Btot2 * sizeof(PetscCount), cupmMemcpyHostToDevice));
    PetscCallCUPM(cupmMemcpy(coo_d->Cperm1, coo_h->Cperm1, coo_h->sendlen * sizeof(PetscCount), cupmMemcpyHostToDevice));
    /* Put the COO struct in a container and attach it to the matrix */
    PetscCall(PetscObjectContainerCompose((PetscObject)mat, "__PETSc_MatCOOStruct_Device", coo_d, MatMPIAIJCUSPARSE_CUPM::COOStructDestroy));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatSetValuesCOO: launch CUPM kernels for packing and adding local/remote COO values */
  static PetscErrorCode SetValuesCOO(Mat mat, const PetscScalar v[], InsertMode imode) noexcept
  {
    Mat_MPIAIJ          *mpiaij = static_cast<Mat_MPIAIJ *>(mat->data);
    Mat                  A = mpiaij->A, B = mpiaij->B;
    PetscScalar         *Aa, *Ba;
    const PetscScalar   *v1 = v;
    PetscMemType         memtype;
    PetscContainer       container;
    MatCOOStruct_MPIAIJ *coo;
    cupmStream_t         stream;

    PetscFunctionBegin;
    PetscCall(PetscObjectQuery((PetscObject)mat, "__PETSc_MatCOOStruct_Device", (PetscObject *)&container));
    PetscCheck(container, PetscObjectComm((PetscObject)mat), PETSC_ERR_PLIB, "Not found MatCOOStruct on this matrix");
    PetscCall(PetscContainerGetPointer(container, (void **)&coo));

    const auto &Annz   = coo->Annz;
    const auto &Annz2  = coo->Annz2;
    const auto &Bnnz   = coo->Bnnz;
    const auto &Bnnz2  = coo->Bnnz2;
    const auto &vsend  = coo->sendbuf;
    const auto &v2     = coo->recvbuf;
    const auto &Ajmap1 = coo->Ajmap1;
    const auto &Ajmap2 = coo->Ajmap2;
    const auto &Aimap2 = coo->Aimap2;
    const auto &Bjmap1 = coo->Bjmap1;
    const auto &Bjmap2 = coo->Bjmap2;
    const auto &Bimap2 = coo->Bimap2;
    const auto &Aperm1 = coo->Aperm1;
    const auto &Aperm2 = coo->Aperm2;
    const auto &Bperm1 = coo->Bperm1;
    const auto &Bperm2 = coo->Bperm2;
    const auto &Cperm1 = coo->Cperm1;

    PetscCall(PetscGetMemType(v, &memtype));
    if (PetscMemTypeHost(memtype)) {
      PetscCallCUPM(cupmMalloc((void **)&v1, coo->n * sizeof(PetscScalar)));
      PetscCallCUPM(cupmMemcpy((void *)v1, v, coo->n * sizeof(PetscScalar), cupmMemcpyHostToDevice));
      PetscCall(PetscLogCpuToGpu(coo->n * sizeof(PetscScalar)));
    }

    if (imode == INSERT_VALUES) {
      PetscCall(Policy::GetArrayWrite(A, &Aa));
      PetscCall(Policy::GetArrayWrite(B, &Ba));
    } else {
      PetscCall(Policy::GetArray(A, &Aa));
      PetscCall(Policy::GetArray(B, &Ba));
    }

    PetscCall(GetHandles_(&stream));
    PetscCall(PetscLogGpuTimeBegin());
    /* Pack entries to be sent to remote */
    if (coo->sendlen) {
      PetscCallCUPM(cupmLaunchKernel(MatPackCOOValues_MPI, (unsigned int)((coo->sendlen + 255) / 256), 256u, (size_t)0, stream, v1, (PetscCount)coo->sendlen, Cperm1, vsend));
      PetscCallCUPM(cupmGetLastError());
    }
    /* Send remote entries and overlap communication with local computation */
    PetscCall(PetscSFReduceWithMemTypeBegin(coo->sf, MPIU_SCALAR, PETSC_MEMTYPE_CUPM(), vsend, PETSC_MEMTYPE_CUPM(), v2, MPI_REPLACE));
    /* Add local entries to A and B */
    if (Annz + Bnnz > 0) {
      PetscCallCUPM(cupmLaunchKernel(MatAddLocalCOOValues_MPI, (unsigned int)((Annz + Bnnz + 255) / 256), 256u, (size_t)0, stream, v1, imode, Annz, Ajmap1, Aperm1, Aa, Bnnz, Bjmap1, Bperm1, Ba));
      PetscCallCUPM(cupmGetLastError());
    }
    PetscCall(PetscSFReduceEnd(coo->sf, MPIU_SCALAR, vsend, v2, MPI_REPLACE));
    /* Add received remote entries to A and B */
    if (Annz2 + Bnnz2 > 0) {
      PetscCallCUPM(cupmLaunchKernel(MatAddRemoteCOOValues_MPI, (unsigned int)((Annz2 + Bnnz2 + 255) / 256), 256u, (size_t)0, stream, v2, Annz2, Aimap2, Ajmap2, Aperm2, Aa, Bnnz2, Bimap2, Bjmap2, Bperm2, Ba));
      PetscCallCUPM(cupmGetLastError());
    }
    PetscCall(PetscLogGpuTimeEnd());

    if (imode == INSERT_VALUES) {
      PetscCall(Policy::RestoreArrayWrite(A, &Aa));
      PetscCall(Policy::RestoreArrayWrite(B, &Ba));
    } else {
      PetscCall(Policy::RestoreArray(A, &Aa));
      PetscCall(Policy::RestoreArray(B, &Ba));
    }
    if (PetscMemTypeHost(memtype)) {
      void *v1_device = (void *)v1;
      PetscCallCUPM(cupmFree(v1_device));
    }
    mat->offloadmask = PETSC_OFFLOAD_GPU;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatMPIAIJGetLocalMatMerge */
  static PetscErrorCode GetLocalMatMerge(Mat A, MatReuse scall, IS *glob, Mat *A_loc) noexcept
  {
    Mat             Ad, Ao;
    const PetscInt *cmap;

    PetscFunctionBegin;
    PetscCall(MatMPIAIJGetSeqAIJ(A, &Ad, &Ao, &cmap));
    PetscCall(Policy::MergeMats(Ad, Ao, scall, A_loc));
    if (glob) {
      PetscInt cst, i, dn, on, *gidx;

      PetscCall(MatGetLocalSize(Ad, NULL, &dn));
      PetscCall(MatGetLocalSize(Ao, NULL, &on));
      PetscCall(MatGetOwnershipRangeColumn(A, &cst, NULL));
      PetscCall(PetscMalloc1(dn + on, &gidx));
      for (i = 0; i < dn; i++) gidx[i] = cst + i;
      for (i = 0; i < on; i++) gidx[i + dn] = cmap[i];
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)Ad), dn + on, gidx, PETSC_OWN_POINTER, glob));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatMPIAIJSetPreallocation */
  static PetscErrorCode SetPreallocation(Mat B, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[]) noexcept
  {
    Mat_MPIAIJ    *b     = (Mat_MPIAIJ *)B->data;
    MatStructType *spptr = (MatStructType *)b->spptr;
    PetscInt       i;

    PetscFunctionBegin;
    if (B->hash_active) {
      B->ops[0]      = b->cops;
      B->hash_active = PETSC_FALSE;
    }
    PetscCall(PetscLayoutSetUp(B->rmap));
    PetscCall(PetscLayoutSetUp(B->cmap));
    if (PetscDefined(USE_DEBUG) && d_nnz) {
      for (i = 0; i < B->rmap->n; i++) PetscCheck(d_nnz[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "d_nnz cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT, i, d_nnz[i]);
    }
    if (PetscDefined(USE_DEBUG) && o_nnz) {
      for (i = 0; i < B->rmap->n; i++) PetscCheck(o_nnz[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "o_nnz cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT, i, o_nnz[i]);
    }
#if defined(PETSC_USE_CTABLE)
    PetscCall(PetscHMapIDestroy(&b->colmap));
#else
    PetscCall(PetscFree(b->colmap));
#endif
    PetscCall(PetscFree(b->garray));
    PetscCall(VecDestroy(&b->lvec));
    PetscCall(VecScatterDestroy(&b->Mvctx));
    /* Because the B will have been resized we simply destroy it and create a new one each time */
    PetscCall(MatDestroy(&b->B));
    if (!b->A) {
      PetscCall(MatCreate(PETSC_COMM_SELF, &b->A));
      PetscCall(MatSetSizes(b->A, B->rmap->n, B->cmap->n, B->rmap->n, B->cmap->n));
    }
    if (!b->B) {
      PetscMPIInt size;

      PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)B), &size));
      PetscCall(MatCreate(PETSC_COMM_SELF, &b->B));
      PetscCall(MatSetSizes(b->B, B->rmap->n, size > 1 ? B->cmap->N : 0, B->rmap->n, size > 1 ? B->cmap->N : 0));
    }
    PetscCall(MatSetType(b->A, Policy::seq_mat_type));
    PetscCall(MatSetType(b->B, Policy::seq_mat_type));
    PetscCall(MatBindToCPU(b->A, B->boundtocpu));
    PetscCall(MatBindToCPU(b->B, B->boundtocpu));
    PetscCall(MatSeqAIJSetPreallocation(b->A, d_nz, d_nnz));
    PetscCall(MatSeqAIJSetPreallocation(b->B, o_nz, o_nnz));
    PetscCall(Policy::SetSubMatFormats(b->A, b->B, spptr));
    B->preallocated  = PETSC_TRUE;
    B->was_assembled = PETSC_FALSE;
    B->assembled     = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatMult: identical in both CUDA and HIP */
  static PetscErrorCode Mult(Mat A, Vec xx, Vec yy) noexcept
  {
    Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

    PetscFunctionBegin;
    PetscCall(VecScatterBegin(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscUseTypeMethod(a->A, mult, xx, yy);
    PetscCall(VecScatterEnd(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscUseTypeMethod(a->B, multadd, a->lvec, yy, yy);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatZeroEntries: identical in both CUDA and HIP */
  static PetscErrorCode ZeroEntries(Mat A) noexcept
  {
    Mat_MPIAIJ *l = (Mat_MPIAIJ *)A->data;

    PetscFunctionBegin;
    PetscCall(MatZeroEntries(l->A));
    PetscCall(MatZeroEntries(l->B));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatMultAdd: identical in both CUDA and HIP */
  static PetscErrorCode MultAdd(Mat A, Vec xx, Vec yy, Vec zz) noexcept
  {
    Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

    PetscFunctionBegin;
    PetscCall(VecScatterBegin(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscUseTypeMethod(a->A, multadd, xx, yy, zz);
    PetscCall(VecScatterEnd(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscUseTypeMethod(a->B, multadd, a->lvec, zz, zz);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatMultTranspose: identical in both CUDA and HIP */
  static PetscErrorCode MultTranspose(Mat A, Vec xx, Vec yy) noexcept
  {
    Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

    PetscFunctionBegin;
    PetscUseTypeMethod(a->B, multtranspose, xx, a->lvec);
    PetscUseTypeMethod(a->A, multtranspose, xx, yy);
    PetscCall(VecScatterBegin(a->Mvctx, a->lvec, yy, ADD_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(a->Mvctx, a->lvec, yy, ADD_VALUES, SCATTER_REVERSE));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatAssemblyEnd: set lvec type to vendor-appropriate VECSEQ type */
  static PetscErrorCode AssemblyEnd(Mat A, MatAssemblyType mode) noexcept
  {
    Mat_MPIAIJ *mpiaij = (Mat_MPIAIJ *)A->data;

    PetscFunctionBegin;
    PetscCall(MatAssemblyEnd_MPIAIJ(A, mode));
    if (mpiaij->lvec) PetscCall(VecSetType(mpiaij->lvec, Policy::vec_seq_type));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatCreateAIJ: allocate and preallocate a parallel sparse matrix of this type */
  static PetscErrorCode CreateAIJ(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], Mat *A) noexcept
  {
    PetscMPIInt size;

    PetscFunctionBegin;
    PetscCall(MatCreate(comm, A));
    PetscCall(MatSetSizes(*A, m, n, M, N));
    PetscCallMPI(MPI_Comm_size(comm, &size));
    if (size > 1) {
      PetscCall(MatSetType(*A, Policy::mpi_mat_type));
      PetscCall(MatMPIAIJSetPreallocation(*A, d_nz, d_nnz, o_nz, o_nnz));
    } else {
      PetscCall(MatSetType(*A, Policy::seq_mat_type));
      PetscCall(MatSeqAIJSetPreallocation(*A, d_nz, d_nnz));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MatDestroy: free vendor-specific state, deregister composed functions */
  static PetscErrorCode Destroy(Mat A) noexcept
  {
    Mat_MPIAIJ    *aij       = (Mat_MPIAIJ *)A->data;
    MatStructType *mpiStruct = (MatStructType *)aij->spptr;

    PetscFunctionBegin;
    PetscCheck(mpiStruct, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing spptr");
    PetscCallCXX(delete mpiStruct);
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIAIJSetPreallocation_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIAIJGetLocalMatMerge_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, Policy::set_format_c, NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)A, Policy::mpi_convert_hypre_c, NULL));
    PetscCall(MatDestroy_MPIAIJ(A));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

} // namespace impl

} // namespace cupm

} // namespace aij

} // namespace mat

} // namespace Petsc
