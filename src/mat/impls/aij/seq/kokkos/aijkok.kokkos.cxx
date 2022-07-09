#include <petscvec_kokkos.hpp>
#include <petscpkg_version.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/sfimpl.h>
#include <petscsystypes.h>
#include <petscerror.h>

#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include <KokkosKernels_Sorting.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_spiluk.hpp>
#include <KokkosSparse_sptrsv.hpp>
#include <KokkosSparse_spgemm.hpp>
#include <KokkosSparse_spadd.hpp>

#include <../src/mat/impls/aij/seq/kokkos/aijkok.hpp>

static PetscErrorCode MatSetOps_SeqAIJKokkos(Mat); /* Forward declaration */

/* MatAssemblyEnd_SeqAIJKokkos() happens when we finalized nonzeros of the matrix, either after
   we assembled the matrix on host, or after we directly produced the matrix data on device (ex., through MatMatMult).
   In the latter case, it is important to set a_dual's sync state correctly.
 */
static PetscErrorCode MatAssemblyEnd_SeqAIJKokkos(Mat A,MatAssemblyType mode)
{
  Mat_SeqAIJ       *aijseq;
  Mat_SeqAIJKokkos *aijkok;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  PetscCall(MatAssemblyEnd_SeqAIJ(A,mode));

  aijseq = static_cast<Mat_SeqAIJ*>(A->data);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  /* If aijkok does not exist, we just copy i, j to device.
     If aijkok already exists, but the device's nonzero pattern does not match with the host's, we assume the latest data is on host.
     In both cases, we build a new aijkok structure.
  */
  if (!aijkok || aijkok->nonzerostate != A->nonzerostate) { /* aijkok might not exist yet or nonzero pattern has changed */
    delete aijkok;
    aijkok   = new Mat_SeqAIJKokkos(A->rmap->n,A->cmap->n,aijseq->nz,aijseq->i,aijseq->j,aijseq->a,A->nonzerostate,PETSC_FALSE/*don't copy mat values to device*/);
    A->spptr = aijkok;
  }

  if (aijkok->device_mat_d.data()) {
    A->offloadmask = PETSC_OFFLOAD_GPU; // in GPU mode, no going back. MatSetValues checks this
  }
  PetscFunctionReturn(0);
}

/* Sync CSR data to device if not yet */
PETSC_INTERN PetscErrorCode MatSeqAIJKokkosSyncDevice(Mat A)
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  PetscCheck(A->factortype == MAT_FACTOR_NONE,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Cann't sync factorized matrix from host to device");
  PetscCheck(aijkok,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Unexpected NULL (Mat_SeqAIJKokkos*)A->spptr");
  if (aijkok->a_dual.need_sync_device()) {
    aijkok->a_dual.sync_device();
    aijkok->transpose_updated = PETSC_FALSE; /* values of the transpose is out-of-date */
    aijkok->hermitian_updated = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/* Mark the CSR data on device as modified */
PETSC_INTERN PetscErrorCode MatSeqAIJKokkosModifyDevice(Mat A)
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  PetscCheck(A->factortype == MAT_FACTOR_NONE,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Not supported for factorized matries");
  aijkok->a_dual.clear_sync_state();
  aijkok->a_dual.modify_device();
  aijkok->transpose_updated = PETSC_FALSE;
  aijkok->hermitian_updated = PETSC_FALSE;
  PetscCall(MatSeqAIJInvalidateDiagonal(A));
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJKokkosSyncHost(Mat A)
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
  /* We do not expect one needs factors on host  */
  PetscCheck(A->factortype == MAT_FACTOR_NONE,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Cann't sync factorized matrix from device to host");
  PetscCheck(aijkok,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing AIJKOK");
  aijkok->a_dual.sync_host();
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJGetArray_SeqAIJKokkos(Mat A,PetscScalar *array[])
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  /* aijkok contains valid pointers only if the host's nonzerostate matches with the device's.
    Calling MatSeqAIJSetPreallocation() or MatSetValues() on host, where aijseq->{i,j,a} might be
    reallocated, will lead to stale {i,j,a}_dual in aijkok. In both operations, the hosts's nonzerostate
    must have been updated. The stale aijkok will be rebuilt during MatAssemblyEnd.
  */
  if (aijkok && A->nonzerostate == aijkok->nonzerostate) {
    aijkok->a_dual.sync_host();
    *array = aijkok->a_dual.view_host().data();
  } else { /* Happens when calling MatSetValues on a newly created matrix */
    *array = static_cast<Mat_SeqAIJ*>(A->data)->a;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJRestoreArray_SeqAIJKokkos(Mat A,PetscScalar *array[])
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  if (aijkok && A->nonzerostate == aijkok->nonzerostate) aijkok->a_dual.modify_host();
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJGetArrayRead_SeqAIJKokkos(Mat A,const PetscScalar *array[])
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  if (aijkok && A->nonzerostate == aijkok->nonzerostate) {
    aijkok->a_dual.sync_host();
    *array = aijkok->a_dual.view_host().data();
  } else {
    *array = static_cast<Mat_SeqAIJ*>(A->data)->a;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJRestoreArrayRead_SeqAIJKokkos(Mat A,const PetscScalar *array[])
{
  PetscFunctionBegin;
  *array = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJGetArrayWrite_SeqAIJKokkos(Mat A,PetscScalar *array[])
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  if (aijkok && A->nonzerostate == aijkok->nonzerostate) {
    *array = aijkok->a_dual.view_host().data();
  } else { /* Ex. happens with MatZeroEntries on a preallocated but not assembled matrix */
    *array = static_cast<Mat_SeqAIJ*>(A->data)->a;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJRestoreArrayWrite_SeqAIJKokkos(Mat A,PetscScalar *array[])
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  if (aijkok && A->nonzerostate == aijkok->nonzerostate) {
    aijkok->a_dual.clear_sync_state();
    aijkok->a_dual.modify_host();
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJGetCSRAndMemType_SeqAIJKokkos(Mat A,const PetscInt **i,const PetscInt **j,PetscScalar **a,PetscMemType *mtype)
{
  Mat_SeqAIJKokkos  *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  PetscCheck(aijkok != NULL,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"aijkok is NULL");

  if (i) *i = aijkok->i_device_data();
  if (j) *j = aijkok->j_device_data();
  if (a) {
    aijkok->a_dual.sync_device();
    *a = aijkok->a_device_data();
  }
  if (mtype) *mtype = PETSC_MEMTYPE_KOKKOS;
  PetscFunctionReturn(0);
}

// MatSeqAIJKokkosSetDeviceMat takes a PetscSplitCSRDataStructure with device data and copies it to the device. Note, "deep_copy" here is really a shallow copy
PetscErrorCode MatSeqAIJKokkosSetDeviceMat(Mat A, PetscSplitCSRDataStructure h_mat)
{
  Mat_SeqAIJKokkos                             *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  Kokkos::View<SplitCSRMat, Kokkos::HostSpace> h_mat_k(h_mat);

  PetscFunctionBegin;
  PetscCheck(aijkok,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Unexpected NULL (Mat_SeqAIJKokkos*)A->spptr");
  aijkok->device_mat_d = create_mirror(DefaultMemorySpace(),h_mat_k);
  Kokkos::deep_copy (aijkok->device_mat_d, h_mat_k);
  PetscFunctionReturn(0);
}

// MatSeqAIJKokkosGetDeviceMat gets the device if it is here, otherwise it creates a place for it and returns NULL
PetscErrorCode MatSeqAIJKokkosGetDeviceMat(Mat A, PetscSplitCSRDataStructure *d_mat)
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  if (aijkok && aijkok->device_mat_d.data()) {
    *d_mat = aijkok->device_mat_d.data();
  } else {
    PetscCall(MatSeqAIJKokkosSyncDevice(A)); // create aijkok (we are making d_mat now so make a place for it)
    *d_mat  = NULL;
  }
  PetscFunctionReturn(0);
}

/* Generate the transpose on device and cache it internally */
static PetscErrorCode MatSeqAIJKokkosGenerateTranspose_Private(Mat A, KokkosCsrMatrix **csrmatT)
{
  Mat_SeqAIJKokkos                 *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  PetscCheck(aijkok,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Unexpected NULL (Mat_SeqAIJKokkos*)A->spptr");
  if (!aijkok->csrmatT.nnz() || !aijkok->transpose_updated) { /* Generate At for the first time OR just update its values */
    /* FIXME: KK does not separate symbolic/numeric transpose. We could have a permutation array to help value-only update */
    PetscCallCXX(aijkok->a_dual.sync_device());
    PetscCallCXX(aijkok->csrmatT = KokkosKernels::Impl::transpose_matrix(aijkok->csrmat));
    PetscCallCXX(KokkosKernels::sort_crs_matrix(aijkok->csrmatT));
    aijkok->transpose_updated = PETSC_TRUE;
  }
  *csrmatT = &aijkok->csrmatT;
  PetscFunctionReturn(0);
}

/* Generate the Hermitian on device and cache it internally */
static PetscErrorCode MatSeqAIJKokkosGenerateHermitian_Private(Mat A, KokkosCsrMatrix **csrmatH)
{
  Mat_SeqAIJKokkos                 *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCheck(aijkok,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Unexpected NULL (Mat_SeqAIJKokkos*)A->spptr");
  if (!aijkok->csrmatH.nnz() || !aijkok->hermitian_updated) { /* Generate Ah for the first time OR just update its values */
    PetscCallCXX(aijkok->a_dual.sync_device());
    PetscCallCXX(aijkok->csrmatH = KokkosKernels::Impl::transpose_matrix(aijkok->csrmat));
    PetscCallCXX(KokkosKernels::sort_crs_matrix(aijkok->csrmatH));
   #if defined(PETSC_USE_COMPLEX)
    const auto& a = aijkok->csrmatH.values;
    Kokkos::parallel_for(a.extent(0),KOKKOS_LAMBDA(MatRowMapType i) {a(i) = PetscConj(a(i));});
   #endif
    aijkok->hermitian_updated = PETSC_TRUE;
  }
  *csrmatH = &aijkok->csrmatH;
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

/* y = A x */
static PetscErrorCode MatMult_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarKokkosView       xv;
  PetscScalarKokkosView            yv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  PetscCall(VecGetKokkosView(xx,&xv));
  PetscCall(VecGetKokkosViewWrite(yy,&yv));
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("N",1.0/*alpha*/,aijkok->csrmat,xv,0.0/*beta*/,yv); /* y = alpha A x + beta y */
  PetscCall(VecRestoreKokkosView(xx,&xv));
  PetscCall(VecRestoreKokkosViewWrite(yy,&yv));
  /* 2.0*nnz - numRows seems more accurate here but assumes there are no zero-rows. So a little sloppy here. */
  PetscCall(PetscLogGpuFlops(2.0*aijkok->csrmat.nnz()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

/* y = A^T x */
static PetscErrorCode MatMultTranspose_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJKokkos                 *aijkok;
  const char                       *mode;
  ConstPetscScalarKokkosView       xv;
  PetscScalarKokkosView            yv;
  KokkosCsrMatrix                  *csrmat;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  PetscCall(VecGetKokkosView(xx,&xv));
  PetscCall(VecGetKokkosViewWrite(yy,&yv));
  if (A->form_explicit_transpose) {
    PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(A,&csrmat));
    mode = "N";
  } else {
    aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
    csrmat = &aijkok->csrmat;
    mode = "T";
  }
  KokkosSparse::spmv(mode,1.0/*alpha*/,*csrmat,xv,0.0/*beta*/,yv); /* y = alpha A^T x + beta y */
  PetscCall(VecRestoreKokkosView(xx,&xv));
  PetscCall(VecRestoreKokkosViewWrite(yy,&yv));
  PetscCall(PetscLogGpuFlops(2.0*csrmat->nnz()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

/* y = A^H x */
static PetscErrorCode MatMultHermitianTranspose_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJKokkos                 *aijkok;
  const char                       *mode;
  ConstPetscScalarKokkosView       xv;
  PetscScalarKokkosView            yv;
  KokkosCsrMatrix                  *csrmat;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  PetscCall(VecGetKokkosView(xx,&xv));
  PetscCall(VecGetKokkosViewWrite(yy,&yv));
  if (A->form_explicit_transpose) {
    PetscCall(MatSeqAIJKokkosGenerateHermitian_Private(A,&csrmat));
    mode = "N";
  } else {
    aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
    csrmat = &aijkok->csrmat;
    mode = "C";
  }
  KokkosSparse::spmv(mode,1.0/*alpha*/,*csrmat,xv,0.0/*beta*/,yv); /* y = alpha A^H x + beta y */
  PetscCall(VecRestoreKokkosView(xx,&xv));
  PetscCall(VecRestoreKokkosViewWrite(yy,&yv));
  PetscCall(PetscLogGpuFlops(2.0*csrmat->nnz()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

/* z = A x + y */
static PetscErrorCode MatMultAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy, Vec zz)
{
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarKokkosView       xv,yv;
  PetscScalarKokkosView            zv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  PetscCall(VecGetKokkosView(xx,&xv));
  PetscCall(VecGetKokkosView(yy,&yv));
  PetscCall(VecGetKokkosViewWrite(zz,&zv));
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("N",1.0/*alpha*/,aijkok->csrmat,xv,1.0/*beta*/,zv); /* z = alpha A x + beta z */
  PetscCall(VecRestoreKokkosView(xx,&xv));
  PetscCall(VecRestoreKokkosView(yy,&yv));
  PetscCall(VecRestoreKokkosViewWrite(zz,&zv));
  PetscCall(PetscLogGpuFlops(2.0*aijkok->csrmat.nnz()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

/* z = A^T x + y */
static PetscErrorCode MatMultTransposeAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJKokkos                 *aijkok;
  const char                       *mode;
  ConstPetscScalarKokkosView       xv,yv;
  PetscScalarKokkosView            zv;
  KokkosCsrMatrix                  *csrmat;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  PetscCall(VecGetKokkosView(xx,&xv));
  PetscCall(VecGetKokkosView(yy,&yv));
  PetscCall(VecGetKokkosViewWrite(zz,&zv));
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  if (A->form_explicit_transpose) {
    PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(A,&csrmat));
    mode = "N";
  } else {
    aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
    csrmat = &aijkok->csrmat;
    mode = "T";
  }
  KokkosSparse::spmv(mode,1.0/*alpha*/,*csrmat,xv,1.0/*beta*/,zv); /* z = alpha A^T x + beta z */
  PetscCall(VecRestoreKokkosView(xx,&xv));
  PetscCall(VecRestoreKokkosView(yy,&yv));
  PetscCall(VecRestoreKokkosViewWrite(zz,&zv));
  PetscCall(PetscLogGpuFlops(2.0*csrmat->nnz()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

/* z = A^H x + y */
static PetscErrorCode MatMultHermitianTransposeAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJKokkos                 *aijkok;
  const char                       *mode;
  ConstPetscScalarKokkosView       xv,yv;
  PetscScalarKokkosView            zv;
  KokkosCsrMatrix                  *csrmat;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  PetscCall(VecGetKokkosView(xx,&xv));
  PetscCall(VecGetKokkosView(yy,&yv));
  PetscCall(VecGetKokkosViewWrite(zz,&zv));
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  if (A->form_explicit_transpose) {
    PetscCall(MatSeqAIJKokkosGenerateHermitian_Private(A,&csrmat));
    mode = "N";
  } else {
    aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
    csrmat = &aijkok->csrmat;
    mode = "C";
  }
  KokkosSparse::spmv(mode,1.0/*alpha*/,*csrmat,xv,1.0/*beta*/,zv); /* z = alpha A^H x + beta z */
  PetscCall(VecRestoreKokkosView(xx,&xv));
  PetscCall(VecRestoreKokkosView(yy,&yv));
  PetscCall(VecRestoreKokkosViewWrite(zz,&zv));
  PetscCall(PetscLogGpuFlops(2.0*csrmat->nnz()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_SeqAIJKokkos(Mat A,MatOption op,PetscBool flg)
{
  Mat_SeqAIJKokkos          *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  switch (op) {
    case MAT_FORM_EXPLICIT_TRANSPOSE:
      /* need to destroy the transpose matrix if present to prevent from logic errors if flg is set to true later */
      if (A->form_explicit_transpose && !flg && aijkok) PetscCall(aijkok->DestroyMatTranspose());
      A->form_explicit_transpose = flg;
      break;
    default:
      PetscCall(MatSetOption_SeqAIJ(A,op,flg));
      break;
  }
  PetscFunctionReturn(0);
}

/* Depending on reuse, either build a new mat, or use the existing mat */
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJKokkos(Mat A, MatType mtype, MatReuse reuse, Mat* newmat)
{
  Mat_SeqAIJ       *aseq;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  if (reuse == MAT_INITIAL_MATRIX) { /* Build a brand new mat */
    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,newmat)); /* the returned newmat is a SeqAIJKokkos */
  } else if (reuse == MAT_REUSE_MATRIX) { /* Reuse the mat created before */
    PetscCall(MatCopy(A,*newmat,SAME_NONZERO_PATTERN)); /* newmat is already a SeqAIJKokkos */
  } else if (reuse == MAT_INPLACE_MATRIX) { /* newmat is A */
    PetscCheck(A == *newmat,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"A != *newmat with MAT_INPLACE_MATRIX");
    PetscCall(PetscFree(A->defaultvectype));
    PetscCall(PetscStrallocpy(VECKOKKOS,&A->defaultvectype)); /* Allocate and copy the string */
    PetscCall(PetscObjectChangeTypeName((PetscObject)A,MATSEQAIJKOKKOS));
    PetscCall(MatSetOps_SeqAIJKokkos(A));
    aseq = static_cast<Mat_SeqAIJ*>(A->data);
    if (A->assembled) { /* Copy i, j (but not values) to device for an assembled matrix if not yet */
      PetscCheck(!A->spptr,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Expect NULL (Mat_SeqAIJKokkos*)A->spptr");
      A->spptr = new Mat_SeqAIJKokkos(A->rmap->n,A->cmap->n,aseq->nz,aseq->i,aseq->j,aseq->a,A->nonzerostate,PETSC_FALSE);
    }
  }
  PetscFunctionReturn(0);
}

/* MatDuplicate always creates a new matrix. MatDuplicate can be called either on an assembled matrix or
   an unassembled matrix, even though MAT_COPY_VALUES is not allowed for unassembled matrix.
 */
static PetscErrorCode MatDuplicate_SeqAIJKokkos(Mat A,MatDuplicateOption dupOption,Mat *B)
{
  Mat_SeqAIJ            *bseq;
  Mat_SeqAIJKokkos      *akok = static_cast<Mat_SeqAIJKokkos*>(A->spptr),*bkok;
  Mat                   mat;

  PetscFunctionBegin;
  /* Do not copy values on host as A's latest values might be on device. We don't want to do sync blindly */
  PetscCall(MatDuplicate_SeqAIJ(A,MAT_DO_NOT_COPY_VALUES,B));
  mat  = *B;
  if (A->assembled) {
    bseq = static_cast<Mat_SeqAIJ*>(mat->data);
    bkok = new Mat_SeqAIJKokkos(mat->rmap->n,mat->cmap->n,bseq->nz,bseq->i,bseq->j,bseq->a,mat->nonzerostate,PETSC_FALSE);
    bkok->a_dual.clear_sync_state(); /* Clear B's sync state as it will be decided below */
    /* Now copy values to B if needed */
    if (dupOption == MAT_COPY_VALUES) {
      if (akok->a_dual.need_sync_device()) {
        Kokkos::deep_copy(bkok->a_dual.view_host(),akok->a_dual.view_host());
        bkok->a_dual.modify_host();
      } else { /* If device has the latest data, we only copy data on device */
        Kokkos::deep_copy(bkok->a_dual.view_device(),akok->a_dual.view_device());
        bkok->a_dual.modify_device();
      }
    } else { /* MAT_DO_NOT_COPY_VALUES or MAT_SHARE_NONZERO_PATTERN. B's values should be zeroed */
      /* B's values on host should be already zeroed by MatDuplicate_SeqAIJ() */
      bkok->a_dual.modify_host();
    }
    mat->spptr = bkok;
  }

  PetscCall(PetscFree(mat->defaultvectype));
  PetscCall(PetscStrallocpy(VECKOKKOS,&mat->defaultvectype)); /* Allocate and copy the string */
  PetscCall(PetscObjectChangeTypeName((PetscObject)mat,MATSEQAIJKOKKOS));
  PetscCall(MatSetOps_SeqAIJKokkos(mat));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTranspose_SeqAIJKokkos(Mat A,MatReuse reuse,Mat *B)
{
  Mat               At;
  KokkosCsrMatrix   *internT;
  Mat_SeqAIJKokkos  *atkok,*bkok;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(A,&internT)); /* Generate a transpose internally */
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_INPLACE_MATRIX) {
    /* Deep copy internT, as we want to isolate the internal transpose */
    PetscCallCXX(atkok = new Mat_SeqAIJKokkos(KokkosCsrMatrix("csrmat",*internT)));
    PetscCall(MatCreateSeqAIJKokkosWithCSRMatrix(PetscObjectComm((PetscObject)A),atkok,&At));
    if (reuse == MAT_INITIAL_MATRIX) *B = At;
    else PetscCall(MatHeaderReplace(A,&At)); /* Replace A with At inplace */
  } else { /* MAT_REUSE_MATRIX, just need to copy values to B on device */
    if ((*B)->assembled) {
      bkok = static_cast<Mat_SeqAIJKokkos*>((*B)->spptr);
      PetscCallCXX(Kokkos::deep_copy(bkok->a_dual.view_device(),internT->values));
      PetscCall(MatSeqAIJKokkosModifyDevice(*B));
    } else if ((*B)->preallocated) { /* It is ok for B to be only preallocated, as needed in MatTranspose_MPIAIJ */
      Mat_SeqAIJ              *bseq = static_cast<Mat_SeqAIJ*>((*B)->data);
      MatScalarKokkosViewHost a_h(bseq->a,internT->nnz()); /* bseq->nz = 0 if unassembled */
      MatColIdxKokkosViewHost j_h(bseq->j,internT->nnz());
      PetscCallCXX(Kokkos::deep_copy(a_h,internT->values));
      PetscCallCXX(Kokkos::deep_copy(j_h,internT->graph.entries));
    } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"B must be assembled or preallocated");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SeqAIJKokkos(Mat A)
{
  Mat_SeqAIJKokkos           *aijkok;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
    delete aijkok;
  } else {
    delete static_cast<Mat_SeqAIJKokkosTriFactors*>(A->spptr);
  }
  A->spptr = NULL;
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",NULL));
  PetscCall(MatDestroy_SeqAIJ(A));
  PetscFunctionReturn(0);
}

/*MC
   MATSEQAIJKOKKOS - MATAIJKOKKOS = "(seq)aijkokkos" - A matrix type to be used for sparse matrices with Kokkos

   A matrix type type using Kokkos-Kernels CrsMatrix type for portability across different device types

   Options Database Keys:
.  -mat_type aijkokkos - sets the matrix type to "aijkokkos" during a call to MatSetFromOptions()

  Level: beginner

.seealso: `MatCreateSeqAIJKokkos()`, `MATMPIAIJKOKKOS`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJKokkos(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(MatCreate_SeqAIJ(A));
  PetscCall(MatConvert_SeqAIJ_SeqAIJKokkos(A,MATSEQAIJKOKKOS,MAT_INPLACE_MATRIX,&A));
  PetscFunctionReturn(0);
}

/* Merge A, B into a matrix C. A is put before B. C's size would be A->rmap->n by (A->cmap->n + B->cmap->n) */
PetscErrorCode MatSeqAIJKokkosMergeMats(Mat A,Mat B,MatReuse reuse,Mat* C)
{
  Mat_SeqAIJ                   *a,*b;
  Mat_SeqAIJKokkos             *akok,*bkok,*ckok;
  MatScalarKokkosView          aa,ba,ca;
  MatRowMapKokkosView          ai,bi,ci;
  MatColIdxKokkosView          aj,bj,cj;
  PetscInt                     m,n,nnz,aN;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidPointer(C,4);
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
  PetscCheckTypeName(B,MATSEQAIJKOKKOS);
  PetscCheck(A->rmap->n == B->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Invalid number or rows %" PetscInt_FMT " != %" PetscInt_FMT,A->rmap->n,B->rmap->n);
  PetscCheck(reuse != MAT_INPLACE_MATRIX,PETSC_COMM_SELF,PETSC_ERR_SUP,"MAT_INPLACE_MATRIX not supported");

  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  PetscCall(MatSeqAIJKokkosSyncDevice(B));
  a    = static_cast<Mat_SeqAIJ*>(A->data);
  b    = static_cast<Mat_SeqAIJ*>(B->data);
  akok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  bkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  aa   = akok->a_dual.view_device();
  ai   = akok->i_dual.view_device();
  ba   = bkok->a_dual.view_device();
  bi   = bkok->i_dual.view_device();
  m    = A->rmap->n; /* M, N and nnz of C */
  n    = A->cmap->n + B->cmap->n;
  nnz  = a->nz + b->nz;
  aN   = A->cmap->n; /* N of A */
  if (reuse == MAT_INITIAL_MATRIX) {
    aj = akok->j_dual.view_device();
    bj = bkok->j_dual.view_device();
    auto ca_dual = MatScalarKokkosDualView("a",aa.extent(0)+ba.extent(0));
    auto ci_dual = MatRowMapKokkosDualView("i",ai.extent(0));
    auto cj_dual = MatColIdxKokkosDualView("j",aj.extent(0)+bj.extent(0));
    ca = ca_dual.view_device();
    ci = ci_dual.view_device();
    cj = cj_dual.view_device();

    /* Concatenate A and B in parallel using Kokkos hierarchical parallelism */
    Kokkos::parallel_for(Kokkos::TeamPolicy<>(m, Kokkos::AUTO()),KOKKOS_LAMBDA(const KokkosTeamMemberType& t) {
      PetscInt i = t.league_rank(); /* row i */
      PetscInt coffset = ai(i) + bi(i), alen = ai(i+1)-ai(i), blen = bi(i+1)-bi(i);

      Kokkos::single(Kokkos::PerTeam(t), [=]() { /* this side effect only happens once per whole team */
        ci(i) = coffset;
        if (i == m-1) ci(m) = ai(m) + bi(m);
      });

      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, alen+blen), [&](PetscInt k) {
        if (k < alen) {
          ca(coffset+k) = aa(ai(i)+k);
          cj(coffset+k) = aj(ai(i)+k);
        } else {
          ca(coffset+k) = ba(bi(i)+k-alen);
          cj(coffset+k) = bj(bi(i)+k-alen) + aN; /* Entries in B get new column indices in C */
        }
      });
    });
    ca_dual.modify_device();
    ci_dual.modify_device();
    cj_dual.modify_device();
    PetscCallCXX(ckok = new Mat_SeqAIJKokkos(m,n,nnz,ci_dual,cj_dual,ca_dual));
    PetscCall(MatCreateSeqAIJKokkosWithCSRMatrix(PETSC_COMM_SELF,ckok,C));
  } else if (reuse == MAT_REUSE_MATRIX) {
    PetscValidHeaderSpecific(*C,MAT_CLASSID,4);
    PetscCheckTypeName(*C,MATSEQAIJKOKKOS);
    ckok = static_cast<Mat_SeqAIJKokkos*>((*C)->spptr);
    ca   = ckok->a_dual.view_device();
    ci   = ckok->i_dual.view_device();

    Kokkos::parallel_for(Kokkos::TeamPolicy<>(m, Kokkos::AUTO()),KOKKOS_LAMBDA(const KokkosTeamMemberType& t) {
      PetscInt i = t.league_rank(); /* row i */
      PetscInt alen = ai(i+1)-ai(i), blen = bi(i+1)-bi(i);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, alen+blen), [&](PetscInt k) {
        if (k < alen) ca(ci(i)+k) = aa(ai(i)+k);
        else          ca(ci(i)+k) = ba(bi(i)+k-alen);
      });
    });
    PetscCall(MatSeqAIJKokkosModifyDevice(*C));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductDataDestroy_SeqAIJKokkos(void* pdata)
{
  PetscFunctionBegin;
  delete static_cast<MatProductData_SeqAIJKokkos*>(pdata);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_SeqAIJKokkos_SeqAIJKokkos(Mat C)
{
  Mat_Product                    *product = C->product;
  Mat                            A,B;
  bool                           transA,transB; /* use bool, since KK needs this type */
  Mat_SeqAIJKokkos               *akok,*bkok,*ckok;
  Mat_SeqAIJ                     *c;
  MatProductData_SeqAIJKokkos    *pdata;
  KokkosCsrMatrix                *csrmatA,*csrmatB;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  PetscCheck(C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  pdata = static_cast<MatProductData_SeqAIJKokkos*>(C->product->data);

  if (pdata->reusesym) { /* We reached here through e.g., MatMatMult(A,B,MAT_INITIAL_MATRIX,..,C), where symbolic/numeric are combined */
    pdata->reusesym = PETSC_FALSE; /* So that next time when user calls MatMatMult(E,F,MAT_REUSE_MATRIX,..,C), we still do numeric  */
    PetscFunctionReturn(0);
  }

  switch (product->type) {
    case MATPRODUCT_AB:  transA = false; transB = false; break;
    case MATPRODUCT_AtB: transA = true;  transB = false; break;
    case MATPRODUCT_ABt: transA = false; transB = true;  break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Unsupported product type %s",MatProductTypes[product->type]);
  }

  A     = product->A;
  B     = product->B;
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  PetscCall(MatSeqAIJKokkosSyncDevice(B));
  akok  = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  bkok  = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  ckok  = static_cast<Mat_SeqAIJKokkos*>(C->spptr);

  PetscCheck(ckok,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Device data structure spptr is empty");

  csrmatA = &akok->csrmat;
  csrmatB = &bkok->csrmat;

  /* TODO: Once KK spgemm implements transpose, we can get rid of the explicit transpose here */
  if (transA) {
    PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(A,&csrmatA));
    transA = false;
  }

  if (transB) {
    PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(B,&csrmatB));
    transB = false;
  }
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCXX(KokkosSparse::spgemm_numeric(pdata->kh,*csrmatA,transA,*csrmatB,transB,ckok->csrmat));
  PetscCallCXX(KokkosKernels::sort_crs_matrix(ckok->csrmat)); /* without the sort, mat_tests-ex62_14_seqaijkokkos failed */
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatSeqAIJKokkosModifyDevice(C));
  /* shorter version of MatAssemblyEnd_SeqAIJ */
  c = (Mat_SeqAIJ*)C->data;
  PetscCall(PetscInfo(C,"Matrix size: %" PetscInt_FMT " X %" PetscInt_FMT "; storage space: 0 unneeded,%" PetscInt_FMT " used\n",C->rmap->n,C->cmap->n,c->nz));
  PetscCall(PetscInfo(C,"Number of mallocs during MatSetValues() is 0\n"));
  PetscCall(PetscInfo(C,"Maximum nonzeros in any row is %" PetscInt_FMT "\n",c->rmax));
  c->reallocs         = 0;
  C->info.mallocs     = 0;
  C->info.nz_unneeded = 0;
  C->assembled        = C->was_assembled = PETSC_TRUE;
  C->num_ass++;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_SeqAIJKokkos_SeqAIJKokkos(Mat C)
{
  Mat_Product                    *product = C->product;
  MatProductType                 ptype;
  Mat                            A,B;
  bool                           transA,transB;
  Mat_SeqAIJKokkos               *akok,*bkok,*ckok;
  MatProductData_SeqAIJKokkos    *pdata;
  MPI_Comm                       comm;
  KokkosCsrMatrix                *csrmatA,*csrmatB,csrmatC;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  PetscCall(PetscObjectGetComm((PetscObject)C,&comm));
  PetscCheck(!product->data,comm,PETSC_ERR_PLIB,"Product data not empty");
  A       = product->A;
  B       = product->B;
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  PetscCall(MatSeqAIJKokkosSyncDevice(B));
  akok    = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  bkok    = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  csrmatA = &akok->csrmat;
  csrmatB = &bkok->csrmat;

  ptype   = product->type;
  switch (ptype) {
    case MATPRODUCT_AB:  transA = false; transB = false; break;
    case MATPRODUCT_AtB: transA = true;  transB = false; break;
    case MATPRODUCT_ABt: transA = false; transB = true;  break;
    default:
      SETERRQ(comm,PETSC_ERR_PLIB,"Unsupported product type %s",MatProductTypes[product->type]);
  }

  product->data = pdata = new MatProductData_SeqAIJKokkos();
  pdata->kh.set_team_work_size(16);
  pdata->kh.set_dynamic_scheduling(true);
  pdata->reusesym = product->api_user;

  /* TODO: add command line options to select spgemm algorithms */
  auto spgemm_alg = KokkosSparse::SPGEMMAlgorithm::SPGEMM_KK;
  /* CUDA-10.2's spgemm has bugs. As as of 2022-01-19, KK does not support CUDA-11.x's newer spgemm API.
     We can default to SPGEMMAlgorithm::SPGEMM_CUSPARSE with CUDA-11+ when KK adds the support.
   */
  pdata->kh.create_spgemm_handle(spgemm_alg);

  PetscCall(PetscLogGpuTimeBegin());
  /* TODO: Get rid of the explicit transpose once KK-spgemm implements the transpose option */
  if (transA) {
    PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(A,&csrmatA));
    transA = false;
  }

  if (transB) {
    PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(B,&csrmatB));
    transB = false;
  }

  PetscCallCXX(KokkosSparse::spgemm_symbolic(pdata->kh,*csrmatA,transA,*csrmatB,transB,csrmatC));
  /* spgemm_symbolic() only populates C's rowmap, but not C's column indices.
    So we have to do a fake spgemm_numeric() here to get csrmatC.j_d setup, before
    calling new Mat_SeqAIJKokkos().
    TODO: Remove the fake spgemm_numeric() after KK fixed this problem.
  */
  PetscCallCXX(KokkosSparse::spgemm_numeric(pdata->kh,*csrmatA,transA,*csrmatB,transB,csrmatC));
  PetscCallCXX(KokkosKernels::sort_crs_matrix(csrmatC));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCallCXX(ckok = new Mat_SeqAIJKokkos(csrmatC));
  PetscCall(MatSetSeqAIJKokkosWithCSRMatrix(C,ckok));
  C->product->destroy = MatProductDataDestroy_SeqAIJKokkos;
  PetscFunctionReturn(0);
}

/* handles sparse matrix matrix ops */
static PetscErrorCode MatProductSetFromOptions_SeqAIJKokkos(Mat mat)
{
  Mat_Product    *product = mat->product;
  PetscBool      Biskok = PETSC_FALSE,Ciskok = PETSC_TRUE;

  PetscFunctionBegin;
  MatCheckProduct(mat,1);
  PetscCall(PetscObjectTypeCompare((PetscObject)product->B,MATSEQAIJKOKKOS,&Biskok));
  if (product->type == MATPRODUCT_ABC) {
    PetscCall(PetscObjectTypeCompare((PetscObject)product->C,MATSEQAIJKOKKOS,&Ciskok));
  }
  if (Biskok && Ciskok) {
    switch (product->type) {
    case MATPRODUCT_AB:
    case MATPRODUCT_AtB:
    case MATPRODUCT_ABt:
      mat->ops->productsymbolic = MatProductSymbolic_SeqAIJKokkos_SeqAIJKokkos;
      break;
    case MATPRODUCT_PtAP:
    case MATPRODUCT_RARt:
    case MATPRODUCT_ABC:
      mat->ops->productsymbolic = MatProductSymbolic_ABC_Basic;
      break;
    default:
      break;
    }
  } else { /* fallback for AIJ */
    PetscCall(MatProductSetFromOptions_SeqAIJ(mat));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_SeqAIJKokkos(Mat A, PetscScalar a)
{
  Mat_SeqAIJKokkos *aijkok;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosBlas::scal(aijkok->a_dual.view_device(),a,aijkok->a_dual.view_device());
  PetscCall(MatSeqAIJKokkosModifyDevice(A));
  PetscCall(PetscLogGpuFlops(aijkok->a_dual.extent(0)));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_SeqAIJKokkos(Mat A)
{
  Mat_SeqAIJKokkos *aijkok;

  PetscFunctionBegin;
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  if (aijkok) { /* Only zero the device if data is already there */
    KokkosBlas::fill(aijkok->a_dual.view_device(),0.0);
    PetscCall(MatSeqAIJKokkosModifyDevice(A));
  } else { /* Might be preallocated but not assembled */
    PetscCall(MatZeroEntries_SeqAIJ(A));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_SeqAIJKokkos(Mat A,Vec x)
{
  Mat_SeqAIJ                   *aijseq;
  Mat_SeqAIJKokkos             *aijkok;
  PetscInt                     n;
  PetscScalarKokkosView        xv;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(x,&n));
  PetscCheck(n == A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  PetscCheck(A->factortype == MAT_FACTOR_NONE,PETSC_COMM_SELF,PETSC_ERR_SUP,"MatGetDiagonal_SeqAIJKokkos not supported on factored matrices");

  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  if (A->rmap->n && aijkok->diag_dual.extent(0) == 0) { /* Set the diagonal pointer if not already */
    PetscCall(MatMarkDiagonal_SeqAIJ(A));
    aijseq = static_cast<Mat_SeqAIJ*>(A->data);
    aijkok->SetDiagonal(aijseq->diag);
  }

  const auto& Aa = aijkok->a_dual.view_device();
  const auto& Ai = aijkok->i_dual.view_device();
  const auto& Adiag = aijkok->diag_dual.view_device();

  PetscCall(VecGetKokkosViewWrite(x,&xv));
  Kokkos::parallel_for(n,KOKKOS_LAMBDA(const PetscInt i) {
    if (Adiag(i) < Ai(i+1)) xv(i) = Aa(Adiag(i));
    else xv(i) = 0;
  });
  PetscCall(VecRestoreKokkosViewWrite(x,&xv));
  PetscFunctionReturn(0);
}

/* Get a Kokkos View from a mat of type MatSeqAIJKokkos */
PetscErrorCode MatSeqAIJGetKokkosView(Mat A,ConstMatScalarKokkosView* kv)
{
  Mat_SeqAIJKokkos   *aijkok;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(kv,2);
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  *kv    = aijkok->a_dual.view_device();
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJRestoreKokkosView(Mat A,ConstMatScalarKokkosView* kv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(kv,2);
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJGetKokkosView(Mat A,MatScalarKokkosView* kv)
{
  Mat_SeqAIJKokkos   *aijkok;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(kv,2);
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  *kv    = aijkok->a_dual.view_device();
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJRestoreKokkosView(Mat A,MatScalarKokkosView* kv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(kv,2);
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
  PetscCall(MatSeqAIJKokkosModifyDevice(A));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJGetKokkosViewWrite(Mat A,MatScalarKokkosView* kv)
{
  Mat_SeqAIJKokkos   *aijkok;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(kv,2);
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  *kv    = aijkok->a_dual.view_device();
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJRestoreKokkosViewWrite(Mat A,MatScalarKokkosView* kv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(kv,2);
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
  PetscCall(MatSeqAIJKokkosModifyDevice(A));
  PetscFunctionReturn(0);
}

/* Computes Y += alpha X */
static PetscErrorCode MatAXPY_SeqAIJKokkos(Mat Y,PetscScalar alpha,Mat X,MatStructure pattern)
{
  Mat_SeqAIJ                 *x = (Mat_SeqAIJ*)X->data,*y = (Mat_SeqAIJ*)Y->data;
  Mat_SeqAIJKokkos           *xkok,*ykok,*zkok;
  ConstMatScalarKokkosView   Xa;
  MatScalarKokkosView        Ya;

  PetscFunctionBegin;
  PetscCheckTypeName(Y,MATSEQAIJKOKKOS);
  PetscCheckTypeName(X,MATSEQAIJKOKKOS);
  PetscCall(MatSeqAIJKokkosSyncDevice(Y));
  PetscCall(MatSeqAIJKokkosSyncDevice(X));
  PetscCall(PetscLogGpuTimeBegin());

  if (pattern != SAME_NONZERO_PATTERN && x->nz == y->nz) {
    /* We could compare on device, but have to get the comparison result on host. So compare on host instead. */
    PetscBool e;
    PetscCall(PetscArraycmp(x->i,y->i,Y->rmap->n+1,&e));
    if (e) {
      PetscCall(PetscArraycmp(x->j,y->j,y->nz,&e));
      if (e) pattern = SAME_NONZERO_PATTERN;
    }
  }

  /* cusparseDcsrgeam2() computes C = alpha A + beta B. If one knew sparsity pattern of C, one can skip
    cusparseScsrgeam2_bufferSizeExt() / cusparseXcsrgeam2Nnz(), and directly call cusparseScsrgeam2().
    If X is SUBSET_NONZERO_PATTERN of Y, we could take advantage of this cusparse feature. However,
    KokkosSparse::spadd(alpha,A,beta,B,C) has symbolic and numeric phases, MatAXPY does not.
  */
  ykok = static_cast<Mat_SeqAIJKokkos*>(Y->spptr);
  xkok = static_cast<Mat_SeqAIJKokkos*>(X->spptr);
  Xa   = xkok->a_dual.view_device();
  Ya   = ykok->a_dual.view_device();

  if (pattern == SAME_NONZERO_PATTERN) {
    KokkosBlas::axpy(alpha,Xa,Ya);
    PetscCall(MatSeqAIJKokkosModifyDevice(Y));
  } else if (pattern == SUBSET_NONZERO_PATTERN) {
    MatRowMapKokkosView  Xi = xkok->i_dual.view_device(),Yi = ykok->i_dual.view_device();
    MatColIdxKokkosView  Xj = xkok->j_dual.view_device(),Yj = ykok->j_dual.view_device();

    Kokkos::parallel_for(Kokkos::TeamPolicy<>(Y->rmap->n, 1),KOKKOS_LAMBDA(const KokkosTeamMemberType& t) {
      PetscInt i = t.league_rank(); /* row i */
      Kokkos::single(Kokkos::PerTeam(t), [=] () { /* Only one thread works in a team */
        PetscInt p,q = Yi(i);
        for (p=Xi(i); p<Xi(i+1); p++) { /* For each nonzero on row i of X */
          while (Xj(p) != Yj(q) && q < Yi(i+1)) q++; /* find the matching nonzero on row i of Y */
          if (Xj(p) == Yj(q)) { /* Found it */
            Ya(q) += alpha * Xa(p);
            q++;
          } else {
            /* If not found, it indicates the input is wrong (X is not a SUBSET_NONZERO_PATTERN of Y).
               Just insert a NaN at the beginning of row i if it is not empty, to make the result wrong.
            */
            if (Yi(i) != Yi(i+1)) Ya(Yi(i)) =
           #if PETSC_PKG_KOKKOS_VERSION_GE(3,6,99)
              Kokkos::nan("1"); /* auto promote the double NaN if needed */
           #else
              Kokkos::Experimental::nan("1");
           #endif
          }
        }
      });
    });
    PetscCall(MatSeqAIJKokkosModifyDevice(Y));
  } else { /* different nonzero patterns */
    Mat             Z;
    KokkosCsrMatrix zcsr;
    KernelHandle    kh;
    kh.create_spadd_handle(false);
    KokkosSparse::spadd_symbolic(&kh,xkok->csrmat,ykok->csrmat,zcsr);
    KokkosSparse::spadd_numeric(&kh,alpha,xkok->csrmat,(PetscScalar)1.0,ykok->csrmat,zcsr);
    zkok = new Mat_SeqAIJKokkos(zcsr);
    PetscCall(MatCreateSeqAIJKokkosWithCSRMatrix(PETSC_COMM_SELF,zkok,&Z));
    PetscCall(MatHeaderReplace(Y,&Z));
    kh.destroy_spadd_handle();
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(xkok->a_dual.extent(0)*2)); /* Because we scaled X and then added it to Y */
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetPreallocationCOO_SeqAIJKokkos(Mat mat, PetscCount coo_n, const PetscInt coo_i[], const PetscInt coo_j[])
{
  Mat_SeqAIJKokkos *akok;
  Mat_SeqAIJ       *aseq;

  PetscFunctionBegin;
  PetscCall(MatSetPreallocationCOO_SeqAIJ(mat,coo_n,coo_i,coo_j));
  aseq = static_cast<Mat_SeqAIJ*>(mat->data);
  akok = static_cast<Mat_SeqAIJKokkos*>(mat->spptr);
  delete akok;
  mat->spptr = akok = new Mat_SeqAIJKokkos(mat->rmap->n,mat->cmap->n,aseq->nz,aseq->i,aseq->j,aseq->a,mat->nonzerostate+1,PETSC_FALSE);
  PetscCall(MatZeroEntries_SeqAIJKokkos(mat));
  akok->SetUpCOO(aseq);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesCOO_SeqAIJKokkos(Mat A,const PetscScalar v[],InsertMode imode)
{
  Mat_SeqAIJ                  *aseq = static_cast<Mat_SeqAIJ*>(A->data);
  Mat_SeqAIJKokkos            *akok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  PetscCount                  Annz = aseq->nz;
  const PetscCountKokkosView& jmap = akok->jmap_d;
  const PetscCountKokkosView& perm = akok->perm_d;
  MatScalarKokkosView         Aa;
  ConstMatScalarKokkosView    kv;
  PetscMemType                memtype;

  PetscFunctionBegin;
  PetscCall(PetscGetMemType(v,&memtype));
  if (PetscMemTypeHost(memtype)) { /* If user gave v[] in host, we might need to copy it to device if any */
    kv = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),ConstMatScalarKokkosViewHost(v,aseq->coo_n));
  } else {
    kv = ConstMatScalarKokkosView(v,aseq->coo_n); /* Directly use v[]'s memory */
  }

  if (imode == INSERT_VALUES) PetscCall(MatSeqAIJGetKokkosViewWrite(A,&Aa)); /* write matrix values */
  else PetscCall(MatSeqAIJGetKokkosView(A,&Aa)); /* read & write matrix values */

  Kokkos::parallel_for(Annz,KOKKOS_LAMBDA(const PetscCount i) {
    PetscScalar sum = 0.0;
    for (PetscCount k=jmap(i); k<jmap(i+1); k++) sum += kv(perm(k));
    Aa(i) = (imode == INSERT_VALUES? 0.0 : Aa(i)) + sum;
  });

  if (imode == INSERT_VALUES) PetscCall(MatSeqAIJRestoreKokkosViewWrite(A,&Aa));
  else PetscCall(MatSeqAIJRestoreKokkosView(A,&Aa));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatSeqAIJMoveDiagonalValuesFront_SeqAIJKokkos(Mat A,const PetscInt *diag)
{
  Mat_SeqAIJKokkos            *akok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  MatScalarKokkosView         Aa;
  const MatRowMapKokkosView&  Ai = akok->i_dual.view_device();
  PetscInt                    m = A->rmap->n;
  ConstMatRowMapKokkosView    Adiag(diag,m); /* diag is a device pointer */

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetKokkosViewWrite(A,&Aa));
  Kokkos::parallel_for(m,KOKKOS_LAMBDA(const PetscInt i) {
    PetscScalar tmp;
    if (Adiag(i) >= Ai(i) && Adiag(i) < Ai(i+1)) { /* The diagonal element exists */
      tmp          = Aa(Ai(i));
      Aa(Ai(i))    = Aa(Adiag(i));
      Aa(Adiag(i)) = tmp;
    }
  });
  PetscCall(MatSeqAIJRestoreKokkosViewWrite(A,&Aa));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorNumeric_SeqAIJKokkos(Mat B,Mat A,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJKokkosSyncHost(A));
  PetscCall(MatLUFactorNumeric_SeqAIJ(B,A,info));
  B->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetOps_SeqAIJKokkos(Mat A)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  A->offloadmask = PETSC_OFFLOAD_KOKKOS; /* We do not really use this flag */
  A->boundtocpu  = PETSC_FALSE;

  A->ops->assemblyend               = MatAssemblyEnd_SeqAIJKokkos;
  A->ops->destroy                   = MatDestroy_SeqAIJKokkos;
  A->ops->duplicate                 = MatDuplicate_SeqAIJKokkos;
  A->ops->axpy                      = MatAXPY_SeqAIJKokkos;
  A->ops->scale                     = MatScale_SeqAIJKokkos;
  A->ops->zeroentries               = MatZeroEntries_SeqAIJKokkos;
  A->ops->productsetfromoptions     = MatProductSetFromOptions_SeqAIJKokkos;
  A->ops->mult                      = MatMult_SeqAIJKokkos;
  A->ops->multadd                   = MatMultAdd_SeqAIJKokkos;
  A->ops->multtranspose             = MatMultTranspose_SeqAIJKokkos;
  A->ops->multtransposeadd          = MatMultTransposeAdd_SeqAIJKokkos;
  A->ops->multhermitiantranspose    = MatMultHermitianTranspose_SeqAIJKokkos;
  A->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_SeqAIJKokkos;
  A->ops->productnumeric            = MatProductNumeric_SeqAIJKokkos_SeqAIJKokkos;
  A->ops->transpose                 = MatTranspose_SeqAIJKokkos;
  A->ops->setoption                 = MatSetOption_SeqAIJKokkos;
  A->ops->getdiagonal               = MatGetDiagonal_SeqAIJKokkos;
  a->ops->getarray                  = MatSeqAIJGetArray_SeqAIJKokkos;
  a->ops->restorearray              = MatSeqAIJRestoreArray_SeqAIJKokkos;
  a->ops->getarrayread              = MatSeqAIJGetArrayRead_SeqAIJKokkos;
  a->ops->restorearrayread          = MatSeqAIJRestoreArrayRead_SeqAIJKokkos;
  a->ops->getarraywrite             = MatSeqAIJGetArrayWrite_SeqAIJKokkos;
  a->ops->restorearraywrite         = MatSeqAIJRestoreArrayWrite_SeqAIJKokkos;
  a->ops->getcsrandmemtype          = MatSeqAIJGetCSRAndMemType_SeqAIJKokkos;

  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",MatSetPreallocationCOO_SeqAIJKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",MatSetValuesCOO_SeqAIJKokkos));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode  MatSetSeqAIJKokkosWithCSRMatrix(Mat A,Mat_SeqAIJKokkos *akok)
{
  Mat_SeqAIJ *aseq;
  PetscInt    i,m,n;

  PetscFunctionBegin;
  PetscCheck(!A->spptr,PETSC_COMM_SELF,PETSC_ERR_PLIB,"A->spptr is supposed to be empty");

  m = akok->nrows();
  n = akok->ncols();
  PetscCall(MatSetSizes(A,m,n,m,n));
  PetscCall(MatSetType(A,MATSEQAIJKOKKOS));

  /* Set up data structures of A as a MATSEQAIJ */
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(A,MAT_SKIP_ALLOCATION,NULL));
  aseq = (Mat_SeqAIJ*)(A)->data;

  akok->i_dual.sync_host(); /* We always need sync'ed i, j on host */
  akok->j_dual.sync_host();

  aseq->i            = akok->i_host_data();
  aseq->j            = akok->j_host_data();
  aseq->a            = akok->a_host_data();
  aseq->nonew        = -1; /*this indicates that inserting a new value in the matrix that generates a new nonzero is an error*/
  aseq->singlemalloc = PETSC_FALSE;
  aseq->free_a       = PETSC_FALSE;
  aseq->free_ij      = PETSC_FALSE;
  aseq->nz           = akok->nnz();
  aseq->maxnz        = aseq->nz;

  PetscCall(PetscMalloc1(m,&aseq->imax));
  PetscCall(PetscMalloc1(m,&aseq->ilen));
  for (i=0; i<m; i++) {
    aseq->ilen[i] = aseq->imax[i] = aseq->i[i+1] - aseq->i[i];
  }

  /* It is critical to set the nonzerostate, as we use it to check if sparsity pattern (hence data) has changed on host in MatAssemblyEnd */
  akok->nonzerostate = A->nonzerostate;
  A->spptr = akok; /* Set A->spptr before MatAssembly so that A->spptr won't be allocated again there */
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/* Crete a SEQAIJKOKKOS matrix with a Mat_SeqAIJKokkos data structure

   Note we have names like MatSeqAIJSetPreallocationCSR, so I use capitalized CSR
 */
PETSC_INTERN PetscErrorCode  MatCreateSeqAIJKokkosWithCSRMatrix(MPI_Comm comm,Mat_SeqAIJKokkos *akok,Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSeqAIJKokkosWithCSRMatrix(*A,akok));
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/
/*@C
   MatCreateSeqAIJKokkos - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format). This matrix will ultimately be handled by
   Kokkos for calculations. For good matrix
   assembly performance the user should preallocate the matrix storage by setting
   the parameter nz (or the array nnz).  By setting these parameters accurately,
   performance during matrix assembly can be increased by more than a factor of 50.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows
         (possibly different for each row) or NULL

   Output Parameter:
.  A - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradgm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   If nnz is given then nz is ignored

   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  For large problems you MUST preallocate memory or you
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   By default, this format uses inodes (identical nodes) when possible, to
   improve numerical efficiency of matrix-vector products and solves. We
   search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Level: intermediate

.seealso: `MatCreate()`, `MatCreateAIJ()`, `MatSetValues()`, `MatSeqAIJSetColumnIndices()`, `MatCreateSeqAIJWithArrays()`, `MatCreateAIJ()`
@*/
PetscErrorCode  MatCreateSeqAIJKokkos(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,m,n,m,n));
  PetscCall(MatSetType(*A,MATSEQAIJKOKKOS));
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,(PetscInt*)nnz));
  PetscFunctionReturn(0);
}

typedef Kokkos::TeamPolicy<>::member_type team_member;
//
// This factorization exploits block diagonal matrices with "Nf" (not used).
// Use -pc_factor_mat_ordering_type rcm to order decouple blocks of size N/Nf for this optimization
//
static PetscErrorCode MatLUFactorNumeric_SeqAIJKOKKOSDEVICE(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ         *b=(Mat_SeqAIJ*)B->data;
  Mat_SeqAIJKokkos   *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr), *baijkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  IS                 isrow = b->row,isicol = b->icol;
  const PetscInt     *r_h,*ic_h;
  const PetscInt     n=A->rmap->n, *ai_d=aijkok->i_dual.view_device().data(), *aj_d=aijkok->j_dual.view_device().data(), *bi_d=baijkok->i_dual.view_device().data(), *bj_d=baijkok->j_dual.view_device().data(), *bdiag_d = baijkok->diag_d.data();
  const PetscScalar  *aa_d = aijkok->a_dual.view_device().data();
  PetscScalar        *ba_d = baijkok->a_dual.view_device().data();
  PetscBool          row_identity,col_identity;
  PetscInt           nc, Nf=1, nVec=32; // should be a parameter, Nf is batch size - not used

  PetscFunctionBegin;
  PetscCheck(A->rmap->n == n,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"square matrices only supported %" PetscInt_FMT " %" PetscInt_FMT,A->rmap->n,n);
  PetscCall(MatIsStructurallySymmetric(A,&row_identity));
  PetscCheck(row_identity,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"structurally symmetric matrices only supported");
  PetscCall(ISGetIndices(isrow,&r_h));
  PetscCall(ISGetIndices(isicol,&ic_h));
  PetscCall(ISGetSize(isicol,&nc));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  {
#define KOKKOS_SHARED_LEVEL 1
    using scr_mem_t = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using sizet_scr_t = Kokkos::View<size_t, scr_mem_t>;
    using scalar_scr_t = Kokkos::View<PetscScalar, scr_mem_t>;
    const Kokkos::View<const PetscInt*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_r_k (r_h, n);
    Kokkos::View<PetscInt*, Kokkos::LayoutLeft> d_r_k ("r", n);
    const Kokkos::View<const PetscInt*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_ic_k (ic_h, nc);
    Kokkos::View<PetscInt*, Kokkos::LayoutLeft> d_ic_k ("ic", nc);
    size_t flops_h = 0.0;
    Kokkos::View<size_t, Kokkos::HostSpace> h_flops_k (&flops_h);
    Kokkos::View<size_t> d_flops_k ("flops");
    const int conc = Kokkos::DefaultExecutionSpace().concurrency(), team_size = conc > 1 ? 16 : 1; // 8*32 = 256
    const int nloc = n/Nf, Ni = (conc > 8) ? 1 /* some intelegent number of SMs -- but need league_barrier */ : 1;
    Kokkos::deep_copy (d_flops_k, h_flops_k);
    Kokkos::deep_copy (d_r_k, h_r_k);
    Kokkos::deep_copy (d_ic_k, h_ic_k);
    // Fill A --> fact
    Kokkos::parallel_for(Kokkos::TeamPolicy<>(Nf*Ni, team_size, nVec), KOKKOS_LAMBDA (const team_member team) {
        const PetscInt  field = team.league_rank()/Ni, field_block = team.league_rank()%Ni; // use grid.x/y in CUDA
        const PetscInt  nloc_i =  (nloc/Ni + !!(nloc%Ni)), start_i = field*nloc + field_block*nloc_i, end_i = (start_i + nloc_i) > (field+1)*nloc ? (field+1)*nloc : (start_i + nloc_i);
        const PetscInt  *ic = d_ic_k.data(), *r = d_r_k.data();
        // zero rows of B
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, start_i, end_i), [=] (const int &rowb) {
            PetscInt    nzbL = bi_d[rowb+1] - bi_d[rowb], nzbU = bdiag_d[rowb] - bdiag_d[rowb+1]; // with diag
            PetscScalar *baL = ba_d + bi_d[rowb];
            PetscScalar *baU = ba_d + bdiag_d[rowb+1]+1;
            /* zero (unfactored row) */
            for (int j=0;j<nzbL;j++) baL[j] = 0;
            for (int j=0;j<nzbU;j++) baU[j] = 0;
          });
        // copy A into B
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, start_i, end_i), [=] (const int &rowb) {
            PetscInt          rowa = r[rowb], nza = ai_d[rowa+1] - ai_d[rowa];
            const PetscScalar *av    = aa_d + ai_d[rowa];
            const PetscInt    *ajtmp = aj_d + ai_d[rowa];
            /* load in initial (unfactored row) */
            for (int j=0;j<nza;j++) {
              PetscInt    colb = ic[ajtmp[j]];
              PetscScalar vala = av[j];
              if (colb == rowb) {
                *(ba_d + bdiag_d[rowb]) = vala;
              } else {
                const PetscInt    *pbj = bj_d + ((colb > rowb) ? bdiag_d[rowb+1]+1 : bi_d[rowb]);
                PetscScalar       *pba = ba_d + ((colb > rowb) ? bdiag_d[rowb+1]+1 : bi_d[rowb]);
                PetscInt          nz   = (colb > rowb) ? bdiag_d[rowb] - (bdiag_d[rowb+1]+1) : bi_d[rowb+1] - bi_d[rowb], set=0;
                for (int j=0; j<nz ; j++) {
                  if (pbj[j] == colb) {
                    pba[j] = vala;
                    set++;
                    break;
                  }
                }
               #if !defined(PETSC_HAVE_SYCL)
                if (set!=1) printf("\t\t\t ERROR DID NOT SET ?????\n");
               #endif
              }
            }
          });
      });
    Kokkos::fence();

    Kokkos::parallel_for(Kokkos::TeamPolicy<>(Nf*Ni, team_size, nVec).set_scratch_size(KOKKOS_SHARED_LEVEL, Kokkos::PerThread(sizet_scr_t::shmem_size()+scalar_scr_t::shmem_size()), Kokkos::PerTeam(sizet_scr_t::shmem_size())), KOKKOS_LAMBDA (const team_member team) {
        sizet_scr_t     colkIdx(team.thread_scratch(KOKKOS_SHARED_LEVEL));
        scalar_scr_t    L_ki(team.thread_scratch(KOKKOS_SHARED_LEVEL));
        sizet_scr_t     flops(team.team_scratch(KOKKOS_SHARED_LEVEL));
        const PetscInt  field = team.league_rank()/Ni, field_block_idx = team.league_rank()%Ni; // use grid.x/y in CUDA
        const PetscInt  start = field*nloc, end = start + nloc;
        Kokkos::single(Kokkos::PerTeam(team), [=]() { flops() = 0; });
        // A22 panel update for each row A(1,:) and col A(:,1)
        for (int ii=start; ii<end-1; ii++) {
          const PetscInt    *bjUi = bj_d + bdiag_d[ii+1]+1, nzUi = bdiag_d[ii] - (bdiag_d[ii+1]+1); // vector, and vector size, of column indices of U(i,(i+1):end)
          const PetscScalar *baUi = ba_d + bdiag_d[ii+1]+1; // vector of data  U(i,i+1:end)
          const PetscInt    nUi_its = nzUi/Ni + !!(nzUi%Ni);
          const PetscScalar Bii = *(ba_d + bdiag_d[ii]); // diagonal in its special place
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nUi_its), [=] (const int j) {
              PetscInt kIdx = j*Ni + field_block_idx;
              if (kIdx >= nzUi) /* void */ ;
              else {
                const PetscInt myk  = bjUi[kIdx]; // assume symmetric structure, need a transposed meta-data here in general
                const PetscInt *pjL = bj_d + bi_d[myk]; // look for L(myk,ii) in start of row
                const PetscInt nzL  = bi_d[myk+1] - bi_d[myk]; // size of L_k(:)
                size_t         st_idx;
                // find and do L(k,i) = A(:k,i) / A(i,i)
                Kokkos::single(Kokkos::PerThread(team), [&]() { colkIdx() = PETSC_MAX_INT; });
                // get column, there has got to be a better way
                Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,nzL), [&] (const int &j, size_t &idx) {
                    if (pjL[j] == ii) {
                      PetscScalar *pLki = ba_d + bi_d[myk] + j;
                      idx = j; // output
                      *pLki = *pLki/Bii; // column scaling:  L(k,i) = A(:k,i) / A(i,i)
                    }
                }, st_idx);
                Kokkos::single(Kokkos::PerThread(team), [=]() { colkIdx() = st_idx; L_ki() = *(ba_d + bi_d[myk] + st_idx); });
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
                if (colkIdx() == PETSC_MAX_INT) printf("\t\t\t\t\t\t\tERROR: failed to find L_ki(%d,%d)\n",(int)myk,ii); // uses a register
#endif
                // active row k, do  A_kj -= Lki * U_ij; j \in U(i,:) j != i
                // U(i+1,:end)
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,nzUi), [=] (const int &uiIdx) { // index into i (U)
                      PetscScalar Uij = baUi[uiIdx];
                      PetscInt    col = bjUi[uiIdx];
                      if (col==myk) {
                        // A_kk = A_kk - L_ki * U_ij(k)
                        PetscScalar *Akkv = (ba_d + bdiag_d[myk]); // diagonal in its special place
                        *Akkv = *Akkv - L_ki() * Uij; // UiK
                      } else {
                        PetscScalar    *start, *end, *pAkjv=NULL;
                        PetscInt       high, low;
                        const PetscInt *startj;
                        if (col<myk) { // L
                          PetscScalar *pLki = ba_d + bi_d[myk] + colkIdx();
                          PetscInt idx = (pLki+1) - (ba_d + bi_d[myk]); // index into row
                          start = pLki+1; // start at pLki+1, A22(myk,1)
                          startj= bj_d + bi_d[myk] + idx;
                          end   = ba_d + bi_d[myk+1];
                        } else {
                          PetscInt idx = bdiag_d[myk+1]+1;
                          start = ba_d + idx;
                          startj= bj_d + idx;
                          end   = ba_d + bdiag_d[myk];
                        }
                        // search for 'col', use bisection search - TODO
                        low  = 0;
                        high = (PetscInt)(end-start);
                        while (high-low > 5) {
                          int t = (low+high)/2;
                          if (startj[t] > col) high = t;
                          else                 low  = t;
                        }
                        for (pAkjv=start+low; pAkjv<start+high; pAkjv++) {
                          if (startj[pAkjv-start] == col) break;
                        }
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
                        if (pAkjv==start+high) printf("\t\t\t\t\t\t\t\t\t\t\tERROR: *** failed to find Akj(%d,%d)\n",(int)myk,(int)col); // uses a register
#endif
                        *pAkjv = *pAkjv - L_ki() * Uij; // A_kj = A_kj - L_ki * U_ij
                      }
                    });
              }
            });
          team.team_barrier(); // this needs to be a league barrier to use more that one SM per block
          if (field_block_idx==0) Kokkos::single(Kokkos::PerTeam(team), [&]() { Kokkos::atomic_add( flops.data(), (size_t)(2*(nzUi*nzUi)+2)); });
        } /* endof for (i=0; i<n; i++) { */
        Kokkos::single(Kokkos::PerTeam(team), [=]() { Kokkos::atomic_add( &d_flops_k(), flops()); flops() = 0; });
      });
    Kokkos::fence();
    Kokkos::deep_copy (h_flops_k, d_flops_k);
    PetscCall(PetscLogGpuFlops((PetscLogDouble)h_flops_k()));
    Kokkos::parallel_for(Kokkos::TeamPolicy<>(Nf*Ni, 1, 256), KOKKOS_LAMBDA (const team_member team) {
        const PetscInt  lg_rank = team.league_rank(), field = lg_rank/Ni; //, field_offset = lg_rank%Ni;
        const PetscInt  start = field*nloc, end = start + nloc, n_its = (nloc/Ni + !!(nloc%Ni)); // 1/Ni iters
        /* Invert diagonal for simpler triangular solves */
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, n_its), [=] (int outer_index) {
            int i = start + outer_index*Ni + lg_rank%Ni;
            if (i < end) {
              PetscScalar *pv = ba_d + bdiag_d[i];
              *pv = 1.0/(*pv);
            }
          });
      });
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(ISRestoreIndices(isicol,&ic_h));
  PetscCall(ISRestoreIndices(isrow,&r_h));

  PetscCall(ISIdentity(isrow,&row_identity));
  PetscCall(ISIdentity(isicol,&col_identity));
  if (b->inode.size) {
    B->ops->solve = MatSolve_SeqAIJ_Inode;
  } else if (row_identity && col_identity) {
    B->ops->solve = MatSolve_SeqAIJ_NaturalOrdering;
  } else {
    B->ops->solve = MatSolve_SeqAIJ; // at least this needs to be in Kokkos
  }
  B->offloadmask = PETSC_OFFLOAD_GPU;
  PetscCall(MatSeqAIJKokkosSyncHost(B)); // solve on CPU
  B->ops->solveadd          = MatSolveAdd_SeqAIJ; // and this
  B->ops->solvetranspose    = MatSolveTranspose_SeqAIJ;
  B->ops->solvetransposeadd = MatSolveTransposeAdd_SeqAIJ;
  B->ops->matsolve          = MatMatSolve_SeqAIJ;
  B->assembled              = PETSC_TRUE;
  B->preallocated           = PETSC_TRUE;

  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorSymbolic_SeqAIJKokkos(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatLUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info));
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJKokkos;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJKokkosSymbolicSolveCheck(Mat A)
{
  Mat_SeqAIJKokkosTriFactors     *factors = (Mat_SeqAIJKokkosTriFactors*)A->spptr;

  PetscFunctionBegin;
  if (!factors->sptrsv_symbolic_completed) {
    KokkosSparse::Experimental::sptrsv_symbolic(&factors->khU,factors->iU_d,factors->jU_d,factors->aU_d);
    KokkosSparse::Experimental::sptrsv_symbolic(&factors->khL,factors->iL_d,factors->jL_d,factors->aL_d);
    factors->sptrsv_symbolic_completed = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/* Check if we need to update factors etc for transpose solve */
static PetscErrorCode MatSeqAIJKokkosTransposeSolveCheck(Mat A)
{
  Mat_SeqAIJKokkosTriFactors     *factors = (Mat_SeqAIJKokkosTriFactors*)A->spptr;
  MatColIdxType             n = A->rmap->n;

  PetscFunctionBegin;
  if (!factors->transpose_updated) { /* TODO: KK needs to provide functions to do numeric transpose only */
    /* Update L^T and do sptrsv symbolic */
    factors->iLt_d = MatRowMapKokkosView("factors->iLt_d",n+1);
    Kokkos::deep_copy(factors->iLt_d,0); /* KK requires 0 */
    factors->jLt_d = MatColIdxKokkosView("factors->jLt_d",factors->jL_d.extent(0));
    factors->aLt_d = MatScalarKokkosView("factors->aLt_d",factors->aL_d.extent(0));

    KokkosKernels::Impl::transpose_matrix<
      ConstMatRowMapKokkosView,ConstMatColIdxKokkosView,ConstMatScalarKokkosView,
      MatRowMapKokkosView,MatColIdxKokkosView,MatScalarKokkosView,
      MatRowMapKokkosView,DefaultExecutionSpace>(
        n,n,factors->iL_d,factors->jL_d,factors->aL_d,
        factors->iLt_d,factors->jLt_d,factors->aLt_d);

    /* TODO: KK transpose_matrix() does not sort column indices, however cusparse requires sorted indices.
      We have to sort the indices, until KK provides finer control options.
    */
    KokkosKernels::sort_crs_matrix<DefaultExecutionSpace,
      MatRowMapKokkosView,MatColIdxKokkosView,MatScalarKokkosView>(
        factors->iLt_d,factors->jLt_d,factors->aLt_d);

    KokkosSparse::Experimental::sptrsv_symbolic(&factors->khLt,factors->iLt_d,factors->jLt_d,factors->aLt_d);

    /* Update U^T and do sptrsv symbolic */
    factors->iUt_d = MatRowMapKokkosView("factors->iUt_d",n+1);
    Kokkos::deep_copy(factors->iUt_d,0); /* KK requires 0 */
    factors->jUt_d = MatColIdxKokkosView("factors->jUt_d",factors->jU_d.extent(0));
    factors->aUt_d = MatScalarKokkosView("factors->aUt_d",factors->aU_d.extent(0));

    KokkosKernels::Impl::transpose_matrix<
      ConstMatRowMapKokkosView,ConstMatColIdxKokkosView,ConstMatScalarKokkosView,
      MatRowMapKokkosView,MatColIdxKokkosView,MatScalarKokkosView,
      MatRowMapKokkosView,DefaultExecutionSpace>(
        n,n,factors->iU_d, factors->jU_d, factors->aU_d,
        factors->iUt_d,factors->jUt_d,factors->aUt_d);

    /* Sort indices. See comments above */
    KokkosKernels::sort_crs_matrix<DefaultExecutionSpace,
      MatRowMapKokkosView,MatColIdxKokkosView,MatScalarKokkosView>(
        factors->iUt_d,factors->jUt_d,factors->aUt_d);

    KokkosSparse::Experimental::sptrsv_symbolic(&factors->khUt,factors->iUt_d,factors->jUt_d,factors->aUt_d);
    factors->transpose_updated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/* Solve Ax = b, with A = LU */
static PetscErrorCode MatSolve_SeqAIJKokkos(Mat A,Vec b,Vec x)
{
  ConstPetscScalarKokkosView     bv;
  PetscScalarKokkosView          xv;
  Mat_SeqAIJKokkosTriFactors     *factors = (Mat_SeqAIJKokkosTriFactors*)A->spptr;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSymbolicSolveCheck(A));
  PetscCall(VecGetKokkosView(b,&bv));
  PetscCall(VecGetKokkosViewWrite(x,&xv));
  /* Solve L tmpv = b */
  PetscCallCXX(KokkosSparse::Experimental::sptrsv_solve(&factors->khL,factors->iL_d,factors->jL_d,factors->aL_d,bv,factors->workVector));
  /* Solve Ux = tmpv */
  PetscCallCXX(KokkosSparse::Experimental::sptrsv_solve(&factors->khU,factors->iU_d,factors->jU_d,factors->aU_d,factors->workVector,xv));
  PetscCall(VecRestoreKokkosView(b,&bv));
  PetscCall(VecRestoreKokkosViewWrite(x,&xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

/* Solve A^T x = b, where A^T = U^T L^T */
static PetscErrorCode MatSolveTranspose_SeqAIJKokkos(Mat A,Vec b,Vec x)
{
  ConstPetscScalarKokkosView     bv;
  PetscScalarKokkosView          xv;
  Mat_SeqAIJKokkosTriFactors     *factors = (Mat_SeqAIJKokkosTriFactors*)A->spptr;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosTransposeSolveCheck(A));
  PetscCall(VecGetKokkosView(b,&bv));
  PetscCall(VecGetKokkosViewWrite(x,&xv));
  /* Solve U^T tmpv = b */
  KokkosSparse::Experimental::sptrsv_solve(&factors->khUt,factors->iUt_d,factors->jUt_d,factors->aUt_d,bv,factors->workVector);

  /* Solve L^T x = tmpv */
  KokkosSparse::Experimental::sptrsv_solve(&factors->khLt,factors->iLt_d,factors->jLt_d,factors->aLt_d,factors->workVector,xv);
  PetscCall(VecRestoreKokkosView(b,&bv));
  PetscCall(VecRestoreKokkosViewWrite(x,&xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

static PetscErrorCode MatILUFactorNumeric_SeqAIJKokkos(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJKokkos               *aijkok = (Mat_SeqAIJKokkos*)A->spptr;
  Mat_SeqAIJKokkosTriFactors     *factors = (Mat_SeqAIJKokkosTriFactors*)B->spptr;
  PetscInt                       fill_lev = info->levels;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSyncDevice(A));

  auto a_d = aijkok->a_dual.view_device();
  auto i_d = aijkok->i_dual.view_device();
  auto j_d = aijkok->j_dual.view_device();

  KokkosSparse::Experimental::spiluk_numeric(&factors->kh,fill_lev,i_d,j_d,a_d,factors->iL_d,factors->jL_d,factors->aL_d,factors->iU_d,factors->jU_d,factors->aU_d);

  B->assembled                       = PETSC_TRUE;
  B->preallocated                    = PETSC_TRUE;
  B->ops->solve                      = MatSolve_SeqAIJKokkos;
  B->ops->solvetranspose             = MatSolveTranspose_SeqAIJKokkos;
  B->ops->matsolve                   = NULL;
  B->ops->matsolvetranspose          = NULL;
  B->offloadmask                     = PETSC_OFFLOAD_GPU;

  /* Once the factors' value changed, we need to update their transpose and sptrsv handle */
  factors->transpose_updated         = PETSC_FALSE;
  factors->sptrsv_symbolic_completed = PETSC_FALSE;
  /* TODO: log flops, but how to know that? */
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

static PetscErrorCode MatILUFactorSymbolic_SeqAIJKokkos(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  Mat_SeqAIJKokkos               *aijkok;
  Mat_SeqAIJ                     *b;
  Mat_SeqAIJKokkosTriFactors     *factors = (Mat_SeqAIJKokkosTriFactors*)B->spptr;
  PetscInt                       fill_lev = info->levels;
  PetscInt                       nnzA = ((Mat_SeqAIJ*)A->data)->nz,nnzL,nnzU;
  PetscInt                       n = A->rmap->n;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  /* Rebuild factors */
  if (factors) {factors->Destroy();} /* Destroy the old if it exists */
  else {B->spptr = factors = new Mat_SeqAIJKokkosTriFactors(n);}

  /* Create a spiluk handle and then do symbolic factorization */
  nnzL = nnzU = PetscRealIntMultTruncate(info->fill,nnzA);
  factors->kh.create_spiluk_handle(KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1,n,nnzL,nnzU);

  auto spiluk_handle = factors->kh.get_spiluk_handle();

  Kokkos::realloc(factors->iL_d,n+1); /* Free old arrays and realloc */
  Kokkos::realloc(factors->jL_d,spiluk_handle->get_nnzL());
  Kokkos::realloc(factors->iU_d,n+1);
  Kokkos::realloc(factors->jU_d,spiluk_handle->get_nnzU());

  aijkok = (Mat_SeqAIJKokkos*)A->spptr;
  auto i_d = aijkok->i_dual.view_device();
  auto j_d = aijkok->j_dual.view_device();
  KokkosSparse::Experimental::spiluk_symbolic(&factors->kh,fill_lev,i_d,j_d,factors->iL_d,factors->jL_d,factors->iU_d,factors->jU_d);
  /* TODO: if spiluk_symbolic is asynchronous, do we need to sync before calling get_nnzL()? */

  Kokkos::resize (factors->jL_d,spiluk_handle->get_nnzL()); /* Shrink or expand, and retain old value */
  Kokkos::resize (factors->jU_d,spiluk_handle->get_nnzU());
  Kokkos::realloc(factors->aL_d,spiluk_handle->get_nnzL()); /* No need to retain old value */
  Kokkos::realloc(factors->aU_d,spiluk_handle->get_nnzU());

  /* TODO: add options to select sptrsv algorithms */
  /* Create sptrsv handles for L, U and their transpose */
 #if defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE)
  auto sptrsv_alg = KokkosSparse::Experimental::SPTRSVAlgorithm::SPTRSV_CUSPARSE;
 #else
  auto sptrsv_alg = KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1;
 #endif

  factors->khL.create_sptrsv_handle(sptrsv_alg,n,true/* L is lower tri */);
  factors->khU.create_sptrsv_handle(sptrsv_alg,n,false/* U is not lower tri */);
  factors->khLt.create_sptrsv_handle(sptrsv_alg,n,false/* L^T is not lower tri */);
  factors->khUt.create_sptrsv_handle(sptrsv_alg,n,true/* U^T is lower tri */);

  /* Fill fields of the factor matrix B */
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(B,MAT_SKIP_ALLOCATION,NULL));
  b     = (Mat_SeqAIJ*)B->data;
  b->nz = b->maxnz = spiluk_handle->get_nnzL()+spiluk_handle->get_nnzU();
  B->info.fill_ratio_given  = info->fill;
  B->info.fill_ratio_needed = ((PetscReal)b->nz)/((PetscReal)nnzA);

  B->offloadmask            = PETSC_OFFLOAD_GPU;
  B->ops->lufactornumeric   = MatILUFactorNumeric_SeqAIJKokkos;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorSymbolic_SeqAIJKOKKOSDEVICE(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  Mat_SeqAIJ       *b=(Mat_SeqAIJ*)B->data;
  const PetscInt   nrows   = A->rmap->n;

  PetscFunctionBegin;
  PetscCall(MatLUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info));
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJKOKKOSDEVICE;
  // move B data into Kokkos
  PetscCall(MatSeqAIJKokkosSyncDevice(B)); // create aijkok
  PetscCall(MatSeqAIJKokkosSyncDevice(A)); // create aijkok
  {
    Mat_SeqAIJKokkos *baijkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
    if (!baijkok->diag_d.extent(0)) {
      const Kokkos::View<PetscInt*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_diag (b->diag,nrows+1);
      baijkok->diag_d = Kokkos::View<PetscInt*>(Kokkos::create_mirror(DefaultMemorySpace(),h_diag));
      Kokkos::deep_copy (baijkok->diag_d, h_diag);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFactorGetSolverType_SeqAIJKokkos(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERKOKKOS;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFactorGetSolverType_seqaij_kokkos_device(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERKOKKOSDEVICE;
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERKOKKOS = "Kokkos" - A matrix solver type providing triangular solvers for sequential matrices
  on a single GPU of type, SeqAIJKokkos, AIJKokkos.

  Level: beginner

.seealso: `PCFactorSetMatSolverType()`, `MatSolverType`, `MatCreateSeqAIJKokkos()`, `MATAIJKOKKOS`, `MatKokkosSetFormat()`, `MatKokkosStorageFormat`, `MatKokkosFormatOperation`
M*/
PETSC_EXTERN PetscErrorCode MatGetFactor_SeqAIJKokkos_Kokkos(Mat A,MatFactorType ftype,Mat *B) /* MatGetFactor_<MatType>_<MatSolverType> */
{
  PetscInt       n = A->rmap->n;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),B));
  PetscCall(MatSetSizes(*B,n,n,n,n));
  (*B)->factortype = ftype;
  PetscCall(PetscStrallocpy(MATORDERINGND,(char**)&(*B)->preferredordering[MAT_FACTOR_LU]));
  PetscCall(MatSetType(*B,MATSEQAIJKOKKOS));

  if (ftype == MAT_FACTOR_LU) {
    PetscCall(MatSetBlockSizesFromMats(*B,A,A));
    (*B)->canuseordering         = PETSC_TRUE;
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJKokkos;
  } else if (ftype == MAT_FACTOR_ILU) {
    PetscCall(MatSetBlockSizesFromMats(*B,A,A));
    (*B)->canuseordering         = PETSC_FALSE;
    (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqAIJKokkos;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatFactorType %s is not supported by MatType SeqAIJKokkos", MatFactorTypes[ftype]);

  PetscCall(MatSeqAIJSetPreallocation(*B,MAT_SKIP_ALLOCATION,NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)(*B),"MatFactorGetSolverType_C",MatFactorGetSolverType_SeqAIJKokkos));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaijkokkos_kokkos_device(Mat A,MatFactorType ftype,Mat *B)
{
  PetscInt       n = A->rmap->n;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),B));
  PetscCall(MatSetSizes(*B,n,n,n,n));
  (*B)->factortype = ftype;
  (*B)->canuseordering = PETSC_TRUE;
  PetscCall(PetscStrallocpy(MATORDERINGND,(char**)&(*B)->preferredordering[MAT_FACTOR_LU]));
  PetscCall(MatSetType(*B,MATSEQAIJKOKKOS));

  if (ftype == MAT_FACTOR_LU) {
    PetscCall(MatSetBlockSizesFromMats(*B,A,A));
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJKOKKOSDEVICE;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported for KOKKOS Matrix Types");

  PetscCall(MatSeqAIJSetPreallocation(*B,MAT_SKIP_ALLOCATION,NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)(*B),"MatFactorGetSolverType_C",MatFactorGetSolverType_seqaij_kokkos_device));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_KOKKOS(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERKOKKOS,MATSEQAIJKOKKOS,MAT_FACTOR_LU,MatGetFactor_SeqAIJKokkos_Kokkos));
  PetscCall(MatSolverTypeRegister(MATSOLVERKOKKOS,MATSEQAIJKOKKOS,MAT_FACTOR_ILU,MatGetFactor_SeqAIJKokkos_Kokkos));
  PetscCall(MatSolverTypeRegister(MATSOLVERKOKKOSDEVICE,MATSEQAIJKOKKOS,MAT_FACTOR_LU,MatGetFactor_seqaijkokkos_kokkos_device));
  PetscFunctionReturn(0);
}

/* Utility to print out a KokkosCsrMatrix for debugging */
PETSC_INTERN PetscErrorCode PrintCsrMatrix(const KokkosCsrMatrix& csrmat)
{
  const auto&       iv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),csrmat.graph.row_map);
  const auto&       jv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),csrmat.graph.entries);
  const auto&       av = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),csrmat.values);
  const PetscInt    *i = iv.data();
  const PetscInt    *j = jv.data();
  const PetscScalar *a = av.data();
  PetscInt          m = csrmat.numRows(),n = csrmat.numCols(),nnz = csrmat.nnz();

  PetscFunctionBegin;
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT " x %" PetscInt_FMT " SeqAIJKokkos, with %" PetscInt_FMT " nonzeros\n",m,n,nnz));
  for (PetscInt k=0; k<m; k++) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT ": ",k));
    for (PetscInt p=i[k]; p<i[k+1]; p++) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT "(%.1f), ",j[p],(double)PetscRealPart(a[p])));
    }
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n"));
  }
  PetscFunctionReturn(0);
}
