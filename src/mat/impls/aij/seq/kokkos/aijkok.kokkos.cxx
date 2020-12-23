#include "petsc/private/petscimpl.h"
#include <petscsystypes.h>
#include <petscerror.h>
#include <petscveckokkos.hpp>

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_spmv.hpp>
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
  if (aijkok && aijkok->device_mat_d.data()) {
    A->offloadmask = PETSC_OFFLOAD_GPU; // in GPU mode, no going back. MatSetValues checks this
  }
  /* Don't build (or update) the Mat_SeqAIJKokkos struct. We delay it to the very last moment until we need it. */
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJKokkosSyncDevice(Mat A)
{
  Mat_SeqAIJ                *aijseq = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJKokkos          *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  PetscInt                  nrows   = A->rmap->n,ncols = A->cmap->n,nnz = aijseq->nz;

  PetscFunctionBegin;
  /* If aijkok is not built yet or the nonzero pattern on CPU has changed, build aijkok from the scratch */
  if (!aijkok || aijkok->nonzerostate != A->nonzerostate) {
    delete aijkok;
    aijkok               = new Mat_SeqAIJKokkos(nrows,ncols,nnz,aijseq->i,aijseq->j,aijseq->a);
    aijkok->nonzerostate = A->nonzerostate;
    A->spptr             = aijkok;
  } else if (A->offloadmask == PETSC_OFFLOAD_CPU) { /* Copy values only */
    Kokkos::deep_copy(aijkok->a_d,aijkok->a_h);
  }
  A->offloadmask = PETSC_OFFLOAD_BOTH;
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
  aijkok->device_mat_d = create_mirror(DeviceMemorySpace(),h_mat_k);
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

/* y = A x */
static PetscErrorCode MatMult_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv;
  PetscScalarViewDevice_t          yv;

  PetscFunctionBegin;
  ierr   = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr   = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr   = VecKokkosGetDeviceView(yy,&yv);CHKERRQ(ierr);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("N",1.0/*alpha*/,aijkok->csr,xv,0.0/*beta*/,yv); /* y = alpha A x + beta y */
  ierr   = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr   = VecKokkosRestoreDeviceView(yy,&yv);CHKERRQ(ierr);
  /* 2.0*aijkok->csr.nnz()-aijkok->csr.numRows() seems more accurate here but assumes there are no zero-rows. So a little sloopy here. */
  ierr   = WaitForKokkos();CHKERRQ(ierr);
  ierr   = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* y = A^T x */
static PetscErrorCode MatMultTranspose_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv;
  PetscScalarViewDevice_t          yv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(yy,&yv);CHKERRQ(ierr);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("T",1.0/*alpha*/,aijkok->csr,xv,0.0/*beta*/,yv); /* y = alpha A^T x + beta y */
  ierr = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(yy,&yv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* y = A^H x */
static PetscErrorCode MatMultHermitianTranspose_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv;
  PetscScalarViewDevice_t          yv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(yy,&yv);CHKERRQ(ierr);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("C",1.0/*alpha*/,aijkok->csr,xv,0.0/*beta*/,yv); /* y = alpha A^H x + beta y */
  ierr = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(yy,&yv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* z = A x + y */
static PetscErrorCode MatMultAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy, Vec zz)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv,yv;
  PetscScalarViewDevice_t          zv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(zz,&zv);CHKERRQ(ierr);
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("N",1.0/*alpha*/,aijkok->csr,xv,1.0/*beta*/,zv); /* z = alpha A x + beta z */
  ierr = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(zz,&zv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* z = A^T x + y */
static PetscErrorCode MatMultTransposeAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv,yv;
  PetscScalarViewDevice_t          zv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(zz,&zv);CHKERRQ(ierr);
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("T",1.0/*alpha*/,aijkok->csr,xv,1.0/*beta*/,zv); /* z = alpha A^T x + beta z */
  ierr = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(zz,&zv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* z = A^H x + y */
static PetscErrorCode MatMultHermitianTransposeAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv,yv;
  PetscScalarViewDevice_t          zv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(zz,&zv);CHKERRQ(ierr);
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("C",1.0/*alpha*/,aijkok->csr,xv,1.0/*beta*/,zv); /* z = alpha A^H x + beta z */
  ierr = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(zz,&zv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJKokkos(Mat A, MatType mtype, MatReuse reuse, Mat* newmat)
{
  PetscErrorCode   ierr;
  Mat              B;
  Mat_SeqAIJ       *aij;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) { /* Build a new mat */
    ierr = MatDuplicate(A,MAT_COPY_VALUES,newmat);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) { /* Reuse the mat created before */
    ierr = MatCopy(A,*newmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }

  B    = *newmat;
  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECKOKKOS,&B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJKOKKOS);CHKERRQ(ierr);
  ierr = MatSetOps_SeqAIJKokkos(B);CHKERRQ(ierr);
  /* TODO: see ViennaCL and CUSPARSE once we have a BindToCPU? */
  aij  = (Mat_SeqAIJ*)B->data;
  aij->inode.use = PETSC_FALSE;

  B->offloadmask = PETSC_OFFLOAD_CPU;
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
  PetscErrorCode        ierr;
  Mat_SeqAIJKokkos      *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  if (aijkok && aijkok->device_mat_d.data()) {
    delete aijkok->colmap_d;
    delete aijkok->i_uncompressed_d;
  }
  delete aijkok;
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
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

static PetscErrorCode MatSetValues_SeqAIJKokkos(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  PetscErrorCode    ierr;
  Mat_SeqAIJKokkos  *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  PetscFunctionBegin;
  if (aijkok && aijkok->device_mat_d.data()) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mixing GPU and non-GPU assembly not supported");
  }
  ierr = MatSetValues_SeqAIJ(A,m,im,n,in,v,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_SeqAIJKokkos(Mat A)
{
  PetscErrorCode             ierr;
  PetscBool                  both = PETSC_FALSE;
  Mat_SeqAIJKokkos           *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  Mat_SeqAIJ                 *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  if (aijkok && aijkok->a_d.data()) {
    Kokkos::parallel_for (aijkok->a_d.size(), KOKKOS_LAMBDA (int i) { aijkok->a_d(i) = 0; });
    both = PETSC_TRUE;
  }
  ierr = PetscArrayzero(a->a,a->i[A->rmap->n]);CHKERRQ(ierr);
  ierr = MatSeqAIJInvalidateDiagonal(A);CHKERRQ(ierr);
  if (both) A->offloadmask = PETSC_OFFLOAD_BOTH;
  else A->offloadmask = PETSC_OFFLOAD_CPU;

  PetscFunctionReturn(0);
}

static PetscErrorCode MatAXPY_SeqAIJKokkos(Mat Y,PetscScalar a,Mat X,MatStructure str) // put axpy in aijcusparse, etc.
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *x = (Mat_SeqAIJ*)X->data,*y = (Mat_SeqAIJ*)Y->data;
  PetscBool      flgx,flgy;

  PetscFunctionBegin;
  if (a == (PetscScalar)0.0) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(Y,MAT_CLASSID,1);
  PetscValidHeaderSpecific(X,MAT_CLASSID,3);
  ierr = PetscObjectTypeCompare((PetscObject)Y,MATSEQAIJKOKKOS,&flgy);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)X,MATSEQAIJKOKKOS,&flgx);CHKERRQ(ierr);
  if (!flgx || !flgy) {
    ierr = MatAXPY_SeqAIJ( Y, a, X, str);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (str == DIFFERENT_NONZERO_PATTERN) {
    if (x->nz == y->nz) {
      PetscBool e;
      ierr = PetscArraycmp(x->i,y->i,Y->rmap->n+1,&e);CHKERRQ(ierr);
      if (e) {
        ierr = PetscArraycmp(x->j,y->j,y->nz,&e);CHKERRQ(ierr);
        if (e) {
          str = SAME_NONZERO_PATTERN;
        }
      }
    }
  }
  if (str != SAME_NONZERO_PATTERN) {
    ierr = MatAXPY_SeqAIJ( Y, a, X, str);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else {
    if (Y->offloadmask==PETSC_OFFLOAD_CPU && X->offloadmask==PETSC_OFFLOAD_CPU) {
      ierr = MatAXPY_SeqAIJ( Y, a, X, str);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    } else if ((Y->offloadmask==PETSC_OFFLOAD_GPU || Y->offloadmask==PETSC_OFFLOAD_BOTH) && X->offloadmask == PETSC_OFFLOAD_CPU) {
      ierr    = MatSeqAIJKokkosSyncDevice(X);CHKERRQ(ierr);
    } else if ((X->offloadmask==PETSC_OFFLOAD_GPU || X->offloadmask==PETSC_OFFLOAD_BOTH) && Y->offloadmask == PETSC_OFFLOAD_CPU) {
      ierr    = MatSeqAIJKokkosSyncDevice(Y);CHKERRQ(ierr); /* promote Y */
    }
    {
      Mat_SeqAIJKokkos *aijkokY = static_cast<Mat_SeqAIJKokkos*>(Y->spptr);
      Mat_SeqAIJKokkos *aijkokX = static_cast<Mat_SeqAIJKokkos*>(X->spptr);
      if (aijkokY && aijkokX && aijkokY->a_d.data() && aijkokX->a_d.data()) {
        Kokkos::parallel_for (aijkokY->a_d.size(), KOKKOS_LAMBDA (int i) { aijkokY->a_d(i) += a*aijkokX->a_d(i); });
        Kokkos::deep_copy(aijkokY->a_h,aijkokY->a_d); // we could not copy and keep GPU
        Y->offloadmask = PETSC_OFFLOAD_BOTH;
        ierr = PetscLogGpuFlops(2.0*aijkokY->a_d.size());CHKERRQ(ierr);
      } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"no Mat_SeqAIJKokkos ???");
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetOps_SeqAIJKokkos(Mat A)
{
  PetscFunctionBegin;
  A->ops->setvalues                 = MatSetValues_SeqAIJKokkos; /* protect with DEBUG, but MatSeqAIJSetTotalPreallocation defeats this ??? */
  A->ops->assemblyend               = MatAssemblyEnd_SeqAIJKokkos;
  A->ops->destroy                   = MatDestroy_SeqAIJKokkos;
  A->ops->duplicate                 = MatDuplicate_SeqAIJKokkos;

  A->ops->axpy                      = MatAXPY_SeqAIJKokkos;
  A->ops->zeroentries               = MatZeroEntries_SeqAIJKokkos;
  A->ops->mult                      = MatMult_SeqAIJKokkos;
  A->ops->multadd                   = MatMultAdd_SeqAIJKokkos;
  A->ops->multtranspose             = MatMultTranspose_SeqAIJKokkos;
  A->ops->multtransposeadd          = MatMultTransposeAdd_SeqAIJKokkos;
  A->ops->multhermitiantranspose    = MatMultHermitianTranspose_SeqAIJKokkos;
  A->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_SeqAIJKokkos;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/
/*C
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
