#include <petscvec_kokkos.hpp>
#include <petsc/private/petscimpl.h>
#include <petscsystypes.h>
#include <petscerror.h>

#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_spiluk.hpp>
#include <KokkosSparse_sptrsv.hpp>

#include <../src/mat/impls/aij/seq/aij.h>

#include <../src/mat/impls/aij/seq/kokkos/aijkokkosimpl.hpp>
#include <petscmat.h>

static PetscErrorCode MatSetOps_SeqAIJKokkos(Mat); /* Forward declaration */

static PetscErrorCode MatAssemblyEnd_SeqAIJKokkos(Mat A,MatAssemblyType mode)
{
  PetscErrorCode    ierr;
  Mat_SeqAIJKokkos  *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_SeqAIJ(A,mode);CHKERRQ(ierr);
  A->offloadmask = PETSC_OFFLOAD_CPU;
  if (aijkok && aijkok->device_mat_d.data()) {
    A->offloadmask = PETSC_OFFLOAD_GPU; // in GPU mode, no going back. MatSetValues checks this
  }
  /* Don't build (or update) the Mat_SeqAIJKokkos struct. We delay it to the very last moment until we need it. */
  PetscFunctionReturn(0);
}

/* Sync CSR data to device if not yet */
static PetscErrorCode MatSeqAIJKokkosSyncDevice(Mat A)
{
  Mat_SeqAIJ                *aijseq = static_cast<Mat_SeqAIJ*>(A->data);
  Mat_SeqAIJKokkos          *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  PetscInt                  nrows   = A->rmap->n,ncols = A->cmap->n,nnz = aijseq->nz;

  PetscFunctionBegin;
  if (A->factortype != MAT_FACTOR_NONE) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Cann't sync factorized matrix from host to device");
  /* If aijkok is not built yet OR the nonzero pattern on CPU has changed, build aijkok from the scratch */
  if (!aijkok || aijkok->nonzerostate != A->nonzerostate) {
    delete aijkok;
    aijkok               = new Mat_SeqAIJKokkos(nrows,ncols,nnz,aijseq->i,aijseq->j,aijseq->a);
    aijkok->nonzerostate = A->nonzerostate;
    A->spptr             = aijkok;
  } else if (A->offloadmask == PETSC_OFFLOAD_CPU) { /* Copy values only */
    aijkok->a_dual.clear_sync_state();
    aijkok->a_dual.modify_host(); /* Mark host is modified */
    aijkok->a_dual.sync_device(); /* Sync the device */
    aijkok->transpose_updated = PETSC_FALSE;
    aijkok->hermitian_updated = PETSC_FALSE;
  }
  A->offloadmask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

/* Mark the CSR data on device is modified */
static PetscErrorCode MatSeqAIJKokkosSetDeviceModified(Mat A)
{
  PetscErrorCode   ierr;
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  if (A->factortype != MAT_FACTOR_NONE) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Not supported for factorized matries");
  aijkok->a_dual.clear_sync_state();
  aijkok->a_dual.modify_device();
  aijkok->transpose_updated = PETSC_FALSE;
  aijkok->hermitian_updated = PETSC_FALSE;
  A->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJKokkosSyncHost(Mat A)
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
   /* We do not expect one needs factors on host  */
  if (A->factortype != MAT_FACTOR_NONE) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Cann't sync factorized matrix from device to host");
  if (A->offloadmask == PETSC_OFFLOAD_GPU) {
    if (!aijkok) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing AIJKOK");
    aijkok->a_dual.clear_sync_state();
    aijkok->a_dual.modify_device(); /* Mark device is modified */
    aijkok->a_dual.sync_host(); /* Sync the host */
    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJGetArray_SeqAIJKokkos(Mat A,PetscScalar *array[])
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncHost(A);CHKERRQ(ierr);
  *array = a->a;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

// MatSeqAIJKokkosSetDeviceMat takes a PetscSplitCSRDataStructure with device data and copies it to the device. Note, "deep_copy" here is really a shallow copy
PETSC_EXTERN PetscErrorCode MatSeqAIJKokkosSetDeviceMat(Mat A, PetscSplitCSRDataStructure *h_mat)
{
  Mat_SeqAIJKokkos *aijkok;
  Kokkos::View<PetscSplitCSRDataStructure, Kokkos::HostSpace> h_mat_k(h_mat);

  PetscFunctionBegin;
  // ierr    = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  if (!aijkok) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"no Mat_SeqAIJKokkos");
  aijkok->device_mat_d = create_mirror(DefaultMemorySpace(),h_mat_k);
  Kokkos::deep_copy (aijkok->device_mat_d, h_mat_k);
  PetscFunctionReturn(0);
}

// MatSeqAIJKokkosGetDeviceMat gets the device if it is here, otherwise it creates a place for it and returns NULL
PETSC_EXTERN PetscErrorCode MatSeqAIJKokkosGetDeviceMat(Mat A, PetscSplitCSRDataStructure **d_mat)
{
  Mat_SeqAIJKokkos *aijkok;

  PetscFunctionBegin;
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  if (aijkok && aijkok->device_mat_d.data()) {
    *d_mat = aijkok->device_mat_d.data();
  } else {
    PetscErrorCode   ierr;
    ierr    = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr); // create aijkok (we are making d_mat now so make a place for it)
    *d_mat  = NULL;
  }
  PetscFunctionReturn(0);
}
static PetscErrorCode MatSeqAIJKokkosGenerateTranspose(Mat A)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  if (!aijkok->At) { /* Generate At for the first time */
    ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&aijkok->At);CHKERRQ(ierr);
    ierr = MatSeqAIJKokkosSyncDevice(aijkok->At);CHKERRQ(ierr);
  } else if (!aijkok->transpose_updated) { /* Only update At values */
    ierr = MatTranspose(A,MAT_REUSE_MATRIX,&aijkok->At);CHKERRQ(ierr);
    ierr = MatSeqAIJKokkosSyncDevice(aijkok->At);CHKERRQ(ierr);
  }
  aijkok->transpose_updated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJKokkosGenerateHermitian(Mat A)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  if (!aijkok->Ah) { /* Generate Ah for the first time */
    ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&aijkok->Ah);CHKERRQ(ierr);
    ierr = MatConjugate(aijkok->Ah);CHKERRQ(ierr);
    ierr = MatSeqAIJKokkosSyncDevice(aijkok->Ah);CHKERRQ(ierr);
  } else if (!aijkok->hermitian_updated) { /* Only update Ah values */
    ierr = MatTranspose(A,MAT_REUSE_MATRIX,&aijkok->Ah);CHKERRQ(ierr);
    ierr = MatConjugate(aijkok->Ah);CHKERRQ(ierr);
    ierr = MatSeqAIJKokkosSyncDevice(aijkok->Ah);CHKERRQ(ierr);
  }
  aijkok->hermitian_updated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* y = A x */
static PetscErrorCode MatMult_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarKokkosView       xv;
  PetscScalarKokkosView            yv;

  PetscFunctionBegin;
  ierr   = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr   = VecGetKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr   = VecGetKokkosView(yy,&yv);CHKERRQ(ierr);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("N",1.0/*alpha*/,aijkok->csrmat,xv,0.0/*beta*/,yv); /* y = alpha A x + beta y */
  ierr   = VecRestoreKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr   = VecRestoreKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr   = WaitForKokkos();CHKERRQ(ierr);
  /* 2.0*aijkok->csrmat.nnz()-aijkok->csrmat.numRows() seems more accurate here but assumes there are no zero-rows. So a little sloopy here. */
  ierr   = PetscLogGpuFlops(2.0*aijkok->csrmat.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* y = A^T x */
static PetscErrorCode MatMultTranspose_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  Mat                              B;
  const char                       *mode;
  ConstPetscScalarKokkosView       xv;
  PetscScalarKokkosView            yv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecGetKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(yy,&yv);CHKERRQ(ierr);
  if (A->form_explicit_transpose) {
    ierr = MatSeqAIJKokkosGenerateTranspose(A);CHKERRQ(ierr);
    B    = static_cast<Mat_SeqAIJKokkos*>(A->spptr)->At;
    mode = "N";
  } else {
    B    = A;
    mode = "T";
  }
  aijkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  KokkosSparse::spmv(mode,1.0/*alpha*/,aijkok->csrmat,xv,0.0/*beta*/,yv); /* y = alpha A^T x + beta y */
  ierr = VecRestoreKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csrmat.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* y = A^H x */
static PetscErrorCode MatMultHermitianTranspose_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  Mat                              B;
  const char                       *mode;
  ConstPetscScalarKokkosView       xv;
  PetscScalarKokkosView            yv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecGetKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(yy,&yv);CHKERRQ(ierr);
  if (A->form_explicit_transpose) {
    ierr = MatSeqAIJKokkosGenerateHermitian(A);CHKERRQ(ierr);
    B    = static_cast<Mat_SeqAIJKokkos*>(A->spptr)->Ah;
    mode = "N";
  } else {
    B    = A;
    mode = "C";
  }
  aijkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  KokkosSparse::spmv(mode,1.0/*alpha*/,aijkok->csrmat,xv,0.0/*beta*/,yv); /* y = alpha A^H x + beta y */
  ierr = VecRestoreKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csrmat.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* z = A x + y */
static PetscErrorCode MatMultAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy, Vec zz)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarKokkosView       xv,yv;
  PetscScalarKokkosView            zv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecGetKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(zz,&zv);CHKERRQ(ierr);
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("N",1.0/*alpha*/,aijkok->csrmat,xv,1.0/*beta*/,zv); /* z = alpha A x + beta z */
  ierr = VecRestoreKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(zz,&zv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csrmat.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* z = A^T x + y */
static PetscErrorCode MatMultTransposeAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  Mat                              B;
  const char                       *mode;
  ConstPetscScalarKokkosView       xv,yv;
  PetscScalarKokkosView            zv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecGetKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(zz,&zv);CHKERRQ(ierr);
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  if (A->form_explicit_transpose) {
    ierr = MatSeqAIJKokkosGenerateTranspose(A);CHKERRQ(ierr);
    B    = static_cast<Mat_SeqAIJKokkos*>(A->spptr)->At;
    mode = "N";
  } else {
    B    = A;
    mode = "T";
  }
  aijkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  KokkosSparse::spmv(mode,1.0/*alpha*/,aijkok->csrmat,xv,1.0/*beta*/,zv); /* z = alpha A^T x + beta z */
  ierr = VecRestoreKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(zz,&zv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csrmat.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* z = A^H x + y */
static PetscErrorCode MatMultHermitianTransposeAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  Mat                              B;
  const char                       *mode;
  ConstPetscScalarKokkosView       xv,yv;
  PetscScalarKokkosView            zv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecGetKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(zz,&zv);CHKERRQ(ierr);
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  if (A->form_explicit_transpose) {
    ierr = MatSeqAIJKokkosGenerateHermitian(A);CHKERRQ(ierr);
    B    = static_cast<Mat_SeqAIJKokkos*>(A->spptr)->Ah;
    mode = "N";
  } else {
    B    = A;
    mode = "C";
  }
  aijkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  KokkosSparse::spmv(mode,1.0/*alpha*/,aijkok->csrmat,xv,1.0/*beta*/,zv); /* z = alpha A^H x + beta z */
  ierr = VecRestoreKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(zz,&zv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csrmat.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_SeqAIJKokkos(Mat A,MatOption op,PetscBool flg)
{
  PetscErrorCode            ierr;
  Mat_SeqAIJKokkos          *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  switch (op) {
    case MAT_FORM_EXPLICIT_TRANSPOSE:
      /* need to destroy the transpose matrix if present to prevent from logic errors if flg is set to true later */
      if (A->form_explicit_transpose && !flg && aijkok) {ierr = aijkok->DestroyMatTranspose();CHKERRQ(ierr);}
      A->form_explicit_transpose = flg;
      break;
    default:
      ierr = MatSetOption_SeqAIJ(A,op,flg);CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJKokkos(Mat A, MatType mtype, MatReuse reuse, Mat* newmat)
{
  PetscErrorCode ierr;
  Mat            B;
  Mat_SeqAIJ     *aij;

  PetscFunctionBegin;
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) { /* Build a new mat */
    ierr = MatDuplicate(A,MAT_COPY_VALUES,newmat);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) { /* Reuse the mat created before */
    ierr = MatCopy(A,*newmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }

  B    = *newmat;
  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECKOKKOS,&B->defaultvectype);CHKERRQ(ierr); /* Allocate and copy the string */

  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJKOKKOS);CHKERRQ(ierr);
  ierr = MatSetOps_SeqAIJKokkos(B);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqAIJGetArray_C",MatSeqAIJGetArray_SeqAIJKokkos);CHKERRQ(ierr);
  /* TODO: see ViennaCL and CUSPARSE once we have a BindToCPU? */
  aij  = (Mat_SeqAIJ*)B->data;
  aij->inode.use = PETSC_FALSE;

  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_SeqAIJKokkos(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDuplicate_SeqAIJ(A,cpvalues,B);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJKokkos(*B,MATSEQAIJKOKKOS,MAT_INPLACE_MATRIX,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SeqAIJKokkos(Mat A)
{
  PetscErrorCode             ierr;
  Mat_SeqAIJKokkos           *aijkok;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
    if (aijkok) {
      if (aijkok->device_mat_d.data()) {
        delete aijkok->colmap_d;
        delete aijkok->i_uncompressed_d;
      }
      if (aijkok->diag_d) delete aijkok->diag_d;
    }
    delete aijkok;
  } else {
    delete static_cast<Mat_SeqAIJKokkosTriFactors*>(A->spptr);
  }
  ierr     = PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJGetArray_C",NULL);CHKERRQ(ierr);
  ierr     = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL);CHKERRQ(ierr);
  A->spptr = NULL;
  ierr     = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJKokkos(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = MatCreate_SeqAIJ(A);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJKokkos(A,MATSEQAIJKOKKOS,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if 0
static PetscErrorCode MatMatKernelHandleDestroy_Private(void* data)
{
  MatMatKernelHandle_t *kh = static_cast<MatMatKernelHandle_t *>(data);

  PetscFunctionBegin;
  delete kh;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_SeqAIJKokkos_SeqAIJKokkos(Mat C)
{
  Mat_Product          *product = C->product;
  Mat                  A,B;
  MatProductType       ptype;
  Mat_SeqAIJKokkos     *akok,*bkok,*ckok;
  bool                 tA,tB;
  PetscErrorCode       ierr;
  MatMatKernelHandle_t *kh;
  Mat_SeqAIJ           *c;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  A = product->A;
  B = product->B;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = MatSeqAIJKokkosSyncDevice(B);CHKERRQ(ierr);
  akok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  bkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  ckok = static_cast<Mat_SeqAIJKokkos*>(C->spptr);
  kh   = static_cast<MatMatKernelHandle_t*>(C->product->data);
  ptype = product->type;
  if (A->symmetric && ptype == MATPRODUCT_AtB) ptype = MATPRODUCT_AB;
  if (B->symmetric && ptype == MATPRODUCT_ABt) ptype = MATPRODUCT_AB;
  switch (ptype) {
  case MATPRODUCT_AB:
    tA = false;
    tB = false;
    break;
  case MATPRODUCT_AtB:
    tA = true;
    tB = false;
    break;
  case MATPRODUCT_ABt:
    tA = false;
    tB = true;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Unsupported product type %s",MatProductTypes[product->type]);
  }

  KokkosSparse::spgemm_numeric(*kh, akok->csr, tA, bkok->csr, tB, ckok->csr);
  C->offloadmask = PETSC_OFFLOAD_GPU;
  /* shorter version of MatAssemblyEnd_SeqAIJ */
  c = (Mat_SeqAIJ*)C->data;
  ierr = PetscInfo3(C,"Matrix size: %D X %D; storage space: 0 unneeded,%D used\n",C->rmap->n,C->cmap->n,c->nz);CHKERRQ(ierr);
  ierr = PetscInfo(C,"Number of mallocs during MatSetValues() is 0\n");CHKERRQ(ierr);
  ierr = PetscInfo1(C,"Maximum nonzeros in any row is %D\n",c->rmax);CHKERRQ(ierr);
  c->reallocs         = 0;
  C->info.mallocs    += 0;
  C->info.nz_unneeded = 0;
  C->assembled = C->was_assembled = PETSC_TRUE;
  C->num_ass++;
  /* we can remove these calls when MatSeqAIJGetArray operations are used everywhere! */
  // TODO JZ, copy from device to host since most of Petsc code for AIJ matrices does not use MatSeqAIJGetArray()
  C->offloadmask = PETSC_OFFLOAD_BOTH;
  // Also, we should add support to copy back from device to host
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_SeqAIJKokkos_SeqAIJKokkos(Mat C)
{
  Mat_Product          *product = C->product;
  Mat                  A,B;
  MatProductType       ptype;
  Mat_SeqAIJKokkos     *akok,*bkok,*ckok;
  PetscInt             m,n,k;
  bool                 tA,tB;
  PetscErrorCode       ierr;
  Mat_SeqAIJ           *c;
  MatMatKernelHandle_t *kh;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  A = product->A;
  B = product->B;
  // TODO only copy the i,j data, not the values
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = MatSeqAIJKokkosSyncDevice(B);CHKERRQ(ierr);
  akok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  bkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  ptype = product->type;
  if (A->symmetric && ptype == MATPRODUCT_AtB) ptype = MATPRODUCT_AB;
  if (B->symmetric && ptype == MATPRODUCT_ABt) ptype = MATPRODUCT_AB;
  switch (ptype) {
  case MATPRODUCT_AB:
    tA = false;
    tB = false;
    m = A->rmap->n;
    n = B->cmap->n;
    break;
  case MATPRODUCT_AtB:
    tA = true;
    tB = false;
    m = A->cmap->n;
    n = B->cmap->n;
    break;
  case MATPRODUCT_ABt:
    tA = false;
    tB = true;
    m = A->rmap->n;
    n = B->rmap->n;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Unsupported product type %s",MatProductTypes[product->type]);
  }
  ierr = MatSetSizes(C,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(C,MATSEQAIJKOKKOS);CHKERRQ(ierr);
  c = (Mat_SeqAIJ*)C->data;

  kh = new MatMatKernelHandle_t;
  // TODO SZ: ADD RUNTIME SELECTION OF THESE
  kh->set_team_work_size(16);
  kh->set_dynamic_scheduling(true);
  // Select an spgemm algorithm, limited by configuration at compile-time and
  // set via the handle Some options: {SPGEMM_KK_MEMORY, SPGEMM_KK_SPEED,
  // SPGEMM_KK_MEMSPEED, /*SPGEMM_CUSPARSE, */ SPGEMM_MKL}
  std::string myalg("SPGEMM_KK_MEMORY");
  kh->create_spgemm_handle(KokkosSparse::StringToSPGEMMAlgorithm(myalg));

  /////////////////////////////////////
  // TODO JZ
  ckok = NULL; //new Mat_SeqAIJKokkos();
  C->spptr = ckok;
  KokkosCsrMatrix_t ccsr; // here only to have the code compile
  KokkosSparse::spgemm_symbolic(*kh, akok->csr, tA, bkok->csr, tB, ccsr);
  //cerr = WaitForKOKKOS();CHKERRCUDA(cerr);
  //c->nz = get_nnz_from_ccsr
  //////////////////////////////////////
  c->singlemalloc = PETSC_FALSE;
  c->free_a       = PETSC_TRUE;
  c->free_ij      = PETSC_TRUE;
  ierr = PetscMalloc1(m+1,&c->i);CHKERRQ(ierr);
  ierr = PetscMalloc1(c->nz,&c->j);CHKERRQ(ierr);
  ierr = PetscMalloc1(c->nz,&c->a);CHKERRQ(ierr);
  ////////////////////////////////////
  // TODO JZ copy from device to c->i and c->j
  ////////////////////////////////////
  ierr = PetscMalloc1(m,&c->ilen);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&c->imax);CHKERRQ(ierr);
  c->maxnz = c->nz;
  c->nonzerorowcnt = 0;
  c->rmax = 0;
  for (k = 0; k < m; k++) {
    const PetscInt nn = c->i[k+1] - c->i[k];
    c->ilen[k] = c->imax[k] = nn;
    c->nonzerorowcnt += (PetscInt)!!nn;
    c->rmax = PetscMax(c->rmax,nn);
  }

  C->nonzerostate++;
  ierr = PetscLayoutSetUp(C->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(C->cmap);CHKERRQ(ierr);
  ierr = MatMarkDiagonal_SeqAIJ(C);CHKERRQ(ierr);
  ckok->nonzerostate = C->nonzerostate;
  C->offloadmask   = PETSC_OFFLOAD_UNALLOCATED;
  C->preallocated  = PETSC_TRUE;
  C->assembled     = PETSC_FALSE;
  C->was_assembled = PETSC_FALSE;

  C->ops->productnumeric = MatProductNumeric_SeqAIJKokkos_SeqAIJKokkos;
  C->product->data = kh;
  C->product->destroy = MatMatKernelHandleDestroy_Private;
  PetscFunctionReturn(0);
}

/* handles sparse matrix matrix ops */
PETSC_UNUSED static PetscErrorCode MatProductSetFromOptions_SeqAIJKokkos(Mat mat)
{
  Mat_Product    *product = mat->product;
  PetscErrorCode ierr;
  PetscBool      Biskok = PETSC_FALSE,Ciskok = PETSC_TRUE;

  PetscFunctionBegin;
  MatCheckProduct(mat,1);
  ierr = PetscObjectTypeCompare((PetscObject)product->B,MATSEQAIJKOKKOS,&Biskok);CHKERRQ(ierr);
  if (product->type == MATPRODUCT_ABC) {
    ierr = PetscObjectTypeCompare((PetscObject)product->C,MATSEQAIJKOKKOS,&Ciskok);CHKERRQ(ierr);
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
    ierr = MatProductSetFromOptions_SeqAIJ(mat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode MatSetValues_SeqAIJKokkos(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  PetscErrorCode    ierr;
  Mat_SeqAIJKokkos  *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  PetscFunctionBegin;
  if (aijkok && aijkok->device_mat_d.data()) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mixing GPU and non-GPU assembly not supported");
  ierr = MatSetValues_SeqAIJ(A,m,im,n,in,v,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_SeqAIJKokkos(Mat A, PetscScalar a)
{
  PetscErrorCode   ierr;
  Mat_SeqAIJKokkos *aijkok;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosBlas::scal(aijkok->a_d,a,aijkok->a_d);
  A->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(aijkok->a_d.size());CHKERRQ(ierr);
  // TODO Remove: this can be removed once we implement matmat operations with KOKKOS
  ierr = MatSeqAIJKokkosSyncHost(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_SeqAIJKokkos(Mat A)
{
  PetscErrorCode   ierr;
  PetscBool        both = PETSC_FALSE;
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  Mat_SeqAIJ       *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  if (aijkok && aijkok->a_d.data()) {
    KokkosBlas::fill(aijkok->a_d,0.0);
    both = PETSC_TRUE;
  }
  ierr = PetscArrayzero(a->a,a->i[A->rmap->n]);CHKERRQ(ierr);
  ierr = MatSeqAIJInvalidateDiagonal(A);CHKERRQ(ierr);
  if (both) A->offloadmask = PETSC_OFFLOAD_BOTH;
  else A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

/* Get a Kokkos View from a mat of type MatSeqAIJKokkos */
PetscErrorCode MatSeqAIJGetKokkosView(Mat A,ConstPetscScalarKokkosView* kv)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJKokkos   *aijkok;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(kv,2);
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
  ierr   = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  *kv    = aijkok->a_d;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJRestoreKokkosView(Mat A,ConstPetscScalarKokkosView* kv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(kv,2);
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJGetKokkosView(Mat A,PetscScalarKokkosView* kv)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJKokkos   *aijkok;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(kv,2);
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
  ierr   = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  *kv    = aijkok->a_d;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJRestoreKokkosView(Mat A,PetscScalarKokkosView* kv)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(kv,2);
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
  ierr = MatSeqAIJKokkosSetDeviceModified(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJGetKokkosViewWrite(Mat A,PetscScalarKokkosView* kv)
{
  Mat_SeqAIJKokkos   *aijkok;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(kv,2);
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  *kv    = aijkok->a_d;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJRestoreKokkosViewWrite(Mat A,PetscScalarKokkosView* kv)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(kv,2);
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
  ierr = MatSeqAIJKokkosSetDeviceModified(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Computes Y += a*X */
static PetscErrorCode MatAXPY_SeqAIJKokkos(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *x = (Mat_SeqAIJ*)X->data,*y = (Mat_SeqAIJ*)Y->data;

  PetscFunctionBegin;
  if (X->ops->axpy != Y->ops->axpy) {
    ierr = MatAXPY_SeqAIJ(Y,a,X,str);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (str != SAME_NONZERO_PATTERN && x->nz == y->nz) {
    PetscBool e;
    ierr = PetscArraycmp(x->i,y->i,Y->rmap->n+1,&e);CHKERRQ(ierr);
    if (e) {
      ierr = PetscArraycmp(x->j,y->j,y->nz,&e);CHKERRQ(ierr);
      if (e) str = SAME_NONZERO_PATTERN;
    }
  }

  if (str != SAME_NONZERO_PATTERN) {
    /* TODO: do MatAXPY on device */
    ierr = MatAXPY_SeqAIJ(Y,a,X,str);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else {
    ConstPetscScalarKokkosView xv;
    PetscScalarKokkosView      yv;

    ierr = MatSeqAIJGetKokkosView(X,&xv);CHKERRQ(ierr);
    ierr = MatSeqAIJGetKokkosView(Y,&yv);CHKERRQ(ierr);
    KokkosBlas::axpy(a,xv,yv);
    ierr = MatSeqAIJRestoreKokkosView(X,&xv);CHKERRQ(ierr);
    ierr = MatSeqAIJRestoreKokkosView(Y,&yv);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorNumeric_SeqAIJKokkos(Mat B,Mat A,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncHost(A);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric_SeqAIJ(B,A,info);CHKERRQ(ierr);
  B->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetOps_SeqAIJKokkos(Mat A)
{
  PetscFunctionBegin;
  A->boundtocpu = PETSC_FALSE;

  A->ops->setvalues                 = MatSetValues_SeqAIJKokkos; /* protect with DEBUG, but MatSeqAIJSetTotalPreallocation defeats this ??? */
  A->ops->assemblyend               = MatAssemblyEnd_SeqAIJKokkos;
  A->ops->destroy                   = MatDestroy_SeqAIJKokkos;
  A->ops->duplicate                 = MatDuplicate_SeqAIJKokkos;
  A->ops->axpy                      = MatAXPY_SeqAIJKokkos;
  A->ops->scale                     = MatScale_SeqAIJKokkos;
  A->ops->zeroentries               = MatZeroEntries_SeqAIJKokkos;
  //A->ops->productsetfromoptions     = MatProductSetFromOptions_SeqAIJKokkos;
  A->ops->mult                      = MatMult_SeqAIJKokkos;
  A->ops->multadd                   = MatMultAdd_SeqAIJKokkos;
  A->ops->multtranspose             = MatMultTranspose_SeqAIJKokkos;
  A->ops->multtransposeadd          = MatMultTransposeAdd_SeqAIJKokkos;
  A->ops->multhermitiantranspose    = MatMultHermitianTranspose_SeqAIJKokkos;
  A->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_SeqAIJKokkos;
  A->ops->setoption                 = MatSetOption_SeqAIJKokkos;
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

.seealso: MatCreate(), MatCreateAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays(), MatCreateAIJ(), MATSeqAIJKokkos, MATAIJKOKKOS
@*/
PetscErrorCode  MatCreateSeqAIJKokkos(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJKOKKOS);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// factorizations
typedef Kokkos::TeamPolicy<>::member_type team_member;
//
// This factorization exploits block diagonal matrices with "Nf" attached to the matrix in a container.
// Use -pc_factor_mat_ordering_type rcm to order decouple blocks of size N/Nf for this optimization
//
static PetscErrorCode MatLUFactorNumeric_SeqAIJKOKKOSDEVICE(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ         *b=(Mat_SeqAIJ*)B->data;
  Mat_SeqAIJKokkos   *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr), *baijkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  IS                 isrow = b->row,isicol = b->icol;
  PetscErrorCode     ierr;
  const PetscInt     *r_h,*ic_h;
  const PetscInt     n=A->rmap->n, *ai_d=aijkok->i_d.data(), *aj_d=aijkok->j_d.data(), *bi_d=baijkok->i_d.data(), *bj_d=baijkok->j_d.data(), *bdiag_d = baijkok->diag_d->data();
  const PetscScalar  *aa_d = aijkok->a_d.data();
  PetscScalar        *ba_d = baijkok->a_d.data();
  PetscBool          row_identity,col_identity;
  PetscInt           nc, Nf, nVec=32; // should be a parameter
  PetscContainer     container;

  PetscFunctionBegin;
  if (A->rmap->n != n) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"square matrices only supported %D %D",A->rmap->n,n);
  ierr = MatGetOption(A,MAT_STRUCTURALLY_SYMMETRIC,&row_identity);CHKERRQ(ierr);
  if (!row_identity) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"structurally symmetric matrices only supported");
  ierr = PetscObjectQuery((PetscObject) A, "Nf", (PetscObject *) &container);CHKERRQ(ierr);
  if (container) {
    PetscInt *pNf=NULL, nv;
    ierr = PetscContainerGetPointer(container, (void **) &pNf);CHKERRQ(ierr);
    Nf = (*pNf)%1000;
    nv = (*pNf)/1000;
    if (nv>0) nVec = nv;
  } else Nf = 1;
  if (n%Nf) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"n % Nf != 0 %D %D",n,Nf);
  ierr = ISGetIndices(isrow,&r_h);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic_h);CHKERRQ(ierr);
  ierr = ISGetSize(isicol,&nc);CHKERRQ(ierr);
#if defined(PETSC_HAVE_DEVICE) && defined(PETSC_USE_LOG)
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
#endif
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
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
        const PetscInt  field = team.league_rank()/Ni, field_block = team.league_rank()%Ni; // use grid.x/y in Cuda
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
                if (set!=1) printf("\t\t\t ERROR DID NOT SET ?????\n");
              }
            }
          });
      });
    Kokkos::fence();

    Kokkos::parallel_for(Kokkos::TeamPolicy<>(Nf*Ni, team_size, nVec).set_scratch_size(KOKKOS_SHARED_LEVEL, Kokkos::PerThread(sizet_scr_t::shmem_size()+scalar_scr_t::shmem_size()), Kokkos::PerTeam(sizet_scr_t::shmem_size())), KOKKOS_LAMBDA (const team_member team) {
        sizet_scr_t     colkIdx(team.thread_scratch(KOKKOS_SHARED_LEVEL));
        scalar_scr_t    L_ki(team.thread_scratch(KOKKOS_SHARED_LEVEL));
        sizet_scr_t     flops(team.team_scratch(KOKKOS_SHARED_LEVEL));
        const PetscInt  field = team.league_rank()/Ni, field_block_idx = team.league_rank()%Ni; // use grid.x/y in Cuda
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
                    //printf("\t\t%d) in vector loop, testing col %d =? %d (ii)\n",j,pjL[j],ii);
                    if (pjL[j] == ii) {
                      PetscScalar *pLki = ba_d + bi_d[myk] + j;
                      idx = j; // output
                      *pLki = *pLki/Bii; // column scaling:  L(k,i) = A(:k,i) / A(i,i)
                    }
                  }, st_idx);
                Kokkos::single(Kokkos::PerThread(team), [=]() { colkIdx() = st_idx; L_ki() = *(ba_d + bi_d[myk] + st_idx); });
                if (colkIdx() == PETSC_MAX_INT) printf("\t\t\t\t\t\t\tERROR: failed to find L_ki(%d,%d)\n",myk,ii);
                else { // active row k, do  A_kj -= Lki * U_ij; j \in U(i,:) j != i
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
                        if (pAkjv==start+high) printf("\t\t\t\t\t\t\t\t\t\t\tERROR: *** failed to find Akj(%d,%d)\n",myk,col);
                        *pAkjv = *pAkjv - L_ki() * Uij; // A_kj = A_kj - L_ki * U_ij
                      }
                    });
                }
              }
            });
          team.team_barrier(); // this needs to be a league barrier to use more that one SM per block
          if (field_block_idx==0) Kokkos::single(Kokkos::PerTeam(team), [&]() { Kokkos::atomic_add( flops.data(), (size_t)(2*(nzUi*nzUi)+2)); });
        } /* endof for (i=0; i<n; i++) { */
        Kokkos::single(Kokkos::PerTeam(team), [=]() { Kokkos::atomic_add( &d_flops_k(), flops()); flops() = 0; });
      });
    Kokkos::fence();
    Kokkos::deep_copy (h_flops_k, d_flops_k);
#if defined(PETSC_HAVE_DEVICE) && defined(PETSC_USE_LOG)
    ierr = PetscLogGpuFlops((PetscLogDouble)h_flops_k());CHKERRQ(ierr);
#elif  defined(PETSC_USE_LOG)
    ierr = PetscLogFlops((PetscLogDouble)h_flops_k());CHKERRQ(ierr);
#endif
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
#if defined(PETSC_HAVE_DEVICE) && defined(PETSC_USE_LOG)
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
#endif
  ierr = ISRestoreIndices(isicol,&ic_h);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r_h);CHKERRQ(ierr);

  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(isicol,&col_identity);CHKERRQ(ierr);
  if (b->inode.size) {
    B->ops->solve = MatSolve_SeqAIJ_Inode;
  } else if (row_identity && col_identity) {
    B->ops->solve = MatSolve_SeqAIJ_NaturalOrdering;
  } else {
    B->ops->solve = MatSolve_SeqAIJ; // at least this needs to be in Kokkos
  }
  B->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = MatSeqAIJKokkosSyncHost(B);CHKERRQ(ierr); // solve on CPU
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatLUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info);CHKERRQ(ierr);
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
  MatColumnIndexType             n = A->rmap->n;

  PetscFunctionBegin;
  if (!factors->transpose_updated) { /* TODO: KK needs to provide functions to do numeric transpose only */
    /* Update L^T and do sptrsv symbolic */
    factors->iLt_d = MatRowOffsetKokkosView("factors->iLt_d",n+1);
    Kokkos::deep_copy(factors->iLt_d,0); /* KK requires 0 */
    factors->jLt_d = MatColumnIndexKokkosView("factors->jLt_d",factors->jL_d.extent(0));
    factors->aLt_d = MatValueKokkosView("factors->aLt_d",factors->aL_d.extent(0));

    KokkosKernels::Impl::transpose_matrix<
      ConstMatRowOffsetKokkosView,ConstMatColumnIndexKokkosView,ConstMatValueKokkosView,
      MatRowOffsetKokkosView,MatColumnIndexKokkosView,MatValueKokkosView,
      MatRowOffsetKokkosView,DefaultExecutionSpace>(
        n,n,factors->iL_d,factors->jL_d,factors->aL_d,
        factors->iLt_d,factors->jLt_d,factors->aLt_d);

    /* TODO: KK transpose_matrix() does not sort column indices, however cusparse requires sorted indices.
      We have to sort the indices, until KK provides finer control options.
    */
    KokkosKernels::Impl::sort_crs_matrix<DefaultExecutionSpace,
      MatRowOffsetKokkosView,MatColumnIndexKokkosView,MatValueKokkosView>(
        factors->iLt_d,factors->jLt_d,factors->aLt_d);

    KokkosSparse::Experimental::sptrsv_symbolic(&factors->khLt,factors->iLt_d,factors->jLt_d,factors->aLt_d);

    /* Update U^T and do sptrsv symbolic */
    factors->iUt_d = MatRowOffsetKokkosView("factors->iUt_d",n+1);
    Kokkos::deep_copy(factors->iUt_d,0); /* KK requires 0 */
    factors->jUt_d = MatColumnIndexKokkosView("factors->jUt_d",factors->jU_d.extent(0));
    factors->aUt_d = MatValueKokkosView("factors->aUt_d",factors->aU_d.extent(0));

    KokkosKernels::Impl::transpose_matrix<
      ConstMatRowOffsetKokkosView,ConstMatColumnIndexKokkosView,ConstMatValueKokkosView,
      MatRowOffsetKokkosView,MatColumnIndexKokkosView,MatValueKokkosView,
      MatRowOffsetKokkosView,DefaultExecutionSpace>(
        n,n,factors->iU_d, factors->jU_d, factors->aU_d,
        factors->iUt_d,factors->jUt_d,factors->aUt_d);

    /* Sort indices. See comments above */
    KokkosKernels::Impl::sort_crs_matrix<DefaultExecutionSpace,
      MatRowOffsetKokkosView,MatColumnIndexKokkosView,MatValueKokkosView>(
        factors->iUt_d,factors->jUt_d,factors->aUt_d);

    KokkosSparse::Experimental::sptrsv_symbolic(&factors->khUt,factors->iUt_d,factors->jUt_d,factors->aUt_d);
    factors->transpose_updated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/* Solve Ax = b, with A = LU */
static PetscErrorCode MatSolve_SeqAIJKokkos(Mat A,Vec b,Vec x)
{
  PetscErrorCode                 ierr;
  ConstPetscScalarKokkosView     bv;
  PetscScalarKokkosView          xv;
  Mat_SeqAIJKokkosTriFactors     *factors = (Mat_SeqAIJKokkosTriFactors*)A->spptr;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSymbolicSolveCheck(A);CHKERRQ(ierr);
  ierr = VecGetKokkosView(x,&xv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(b,&bv);CHKERRQ(ierr);
  /* Solve L tmpv = b */
  KokkosSparse::Experimental::sptrsv_solve(&factors->khL,factors->iL_d,factors->jL_d,factors->aL_d,bv,factors->workVector);
  /* Solve Ux = tmpv */
  KokkosSparse::Experimental::sptrsv_solve(&factors->khU,factors->iU_d,factors->jU_d,factors->aU_d,factors->workVector,xv);
  ierr = VecRestoreKokkosView(x,&xv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(b,&bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Solve A^T x = b, with A^T = U^T L^T */
static PetscErrorCode MatSolveTranspose_SeqAIJKokkos(Mat A,Vec b,Vec x)
{
  PetscErrorCode                 ierr;
  ConstPetscScalarKokkosView     bv;
  PetscScalarKokkosView          xv;
  Mat_SeqAIJKokkosTriFactors     *factors = (Mat_SeqAIJKokkosTriFactors*)A->spptr;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosTransposeSolveCheck(A);CHKERRQ(ierr);
  ierr = VecGetKokkosView(x,&xv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(b,&bv);CHKERRQ(ierr);
  /* Solve U^T tmpv = b */
  KokkosSparse::Experimental::sptrsv_solve(&factors->khUt,factors->iUt_d,factors->jUt_d,factors->aUt_d,bv,factors->workVector);

  /* Solve L^T x = tmpv */
  KokkosSparse::Experimental::sptrsv_solve(&factors->khLt,factors->iLt_d,factors->jLt_d,factors->aLt_d,factors->workVector,xv);
  ierr = VecRestoreKokkosView(x,&xv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(b,&bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatILUFactorNumeric_SeqAIJKokkos(Mat B,Mat A,const MatFactorInfo *info)
{
  PetscErrorCode                 ierr;
  Mat_SeqAIJKokkos               *aijkok = (Mat_SeqAIJKokkos*)A->spptr;
  Mat_SeqAIJKokkosTriFactors     *factors = (Mat_SeqAIJKokkosTriFactors*)B->spptr;
  PetscInt                       fill_lev = info->levels;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  KokkosSparse::Experimental::spiluk_numeric(&factors->kh,fill_lev,aijkok->i_d,aijkok->j_d,aijkok->a_d,factors->iL_d,factors->jL_d,factors->aL_d,factors->iU_d,factors->jU_d,factors->aU_d);

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
  PetscFunctionReturn(0);
}

static PetscErrorCode MatILUFactorSymbolic_SeqAIJKokkos(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  PetscErrorCode                 ierr;
  Mat_SeqAIJKokkos               *aijkok;
  Mat_SeqAIJ                     *b;
  Mat_SeqAIJKokkosTriFactors     *factors = (Mat_SeqAIJKokkosTriFactors*)B->spptr;
  PetscInt                       fill_lev = info->levels;
  PetscInt                       nnzA = ((Mat_SeqAIJ*)A->data)->nz,nnzL,nnzU;
  PetscInt                       n = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
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
  KokkosSparse::Experimental::spiluk_symbolic(&factors->kh,fill_lev,aijkok->i_d,aijkok->j_d,factors->iL_d,factors->jL_d,factors->iU_d,factors->jU_d);
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
  ierr  = MatSeqAIJSetPreallocation_SeqAIJ(B,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);
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
  PetscErrorCode   ierr;
  Mat_SeqAIJ       *b=(Mat_SeqAIJ*)B->data;
  const PetscInt   nrows   = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatLUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info);CHKERRQ(ierr);
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJKOKKOSDEVICE;
  // move B data into Kokkos
  ierr = MatSeqAIJKokkosSyncDevice(B);CHKERRQ(ierr); // create aijkok
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr); // create aijkok
  {
    Mat_SeqAIJKokkos *baijkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
    if (!baijkok->diag_d) {
      const Kokkos::View<PetscInt*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_diag (b->diag,nrows+1);
      baijkok->diag_d = new Kokkos::View<PetscInt*>(Kokkos::create_mirror(DefaultMemorySpace(),h_diag));
      Kokkos::deep_copy (*baijkok->diag_d, h_diag);
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

// use -pc_factor_mat_solver_type kokkosdevice
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

.seealso: PCFactorSetMatSolverType(), MatSolverType, MatCreateSeqAIJKokkos(), MATAIJKOKKOS, MatCreateAIJKokkos(), MatKokkosSetFormat(), MatKokkosStorageFormat, MatKokkosFormatOperation
M*/
PETSC_EXTERN PetscErrorCode MatGetFactor_SeqAIJKokkos_Kokkos(Mat A,MatFactorType ftype,Mat *B) /* MatGetFactor_<MatType>_<MatSolverType> */
{
  PetscErrorCode ierr;
  PetscInt       n = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,n,n,n,n);CHKERRQ(ierr);
  (*B)->factortype = ftype;
  ierr = PetscStrallocpy(MATORDERINGND,(char**)&(*B)->preferredordering[MAT_FACTOR_LU]);CHKERRQ(ierr);
  ierr = MatSetType(*B,MATSEQAIJKOKKOS);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_LU) {
    ierr = MatSetBlockSizesFromMats(*B,A,A);CHKERRQ(ierr);
    (*B)->canuseordering         = PETSC_TRUE;
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJKokkos;
  } else if (ftype == MAT_FACTOR_ILU) {
    ierr = MatSetBlockSizesFromMats(*B,A,A);CHKERRQ(ierr);
    (*B)->canuseordering         = PETSC_FALSE;
    (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqAIJKokkos;
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatFactorType %s is not supported by MatType SeqAIJKokkos", MatFactorTypes[ftype]);

  ierr = MatSeqAIJSetPreallocation(*B,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)(*B),"MatFactorGetSolverType_C",MatFactorGetSolverType_SeqAIJKokkos);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaijkokkos_kokkos_device(Mat A,MatFactorType ftype,Mat *B)
{
  PetscErrorCode ierr;
  PetscInt       n = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,n,n,n,n);CHKERRQ(ierr);
  (*B)->factortype = ftype;
  (*B)->canuseordering = PETSC_TRUE;
  ierr = PetscStrallocpy(MATORDERINGND,(char**)&(*B)->preferredordering[MAT_FACTOR_LU]);CHKERRQ(ierr);
  ierr = MatSetType(*B,MATSEQAIJKOKKOS);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_LU) {
    ierr = MatSetBlockSizesFromMats(*B,A,A);CHKERRQ(ierr);
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJKOKKOSDEVICE;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported for KOKKOS Matrix Types");

  ierr = MatSeqAIJSetPreallocation(*B,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)(*B),"MatFactorGetSolverType_C",MatFactorGetSolverType_seqaij_kokkos_device);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_KOKKOS(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverTypeRegister(MATSOLVERKOKKOS,MATSEQAIJKOKKOS,MAT_FACTOR_LU,MatGetFactor_SeqAIJKokkos_Kokkos);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERKOKKOS,MATSEQAIJKOKKOS,MAT_FACTOR_ILU,MatGetFactor_SeqAIJKokkos_Kokkos);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERKOKKOSDEVICE,MATSEQAIJKOKKOS,MAT_FACTOR_LU,MatGetFactor_seqaijkokkos_kokkos_device);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

