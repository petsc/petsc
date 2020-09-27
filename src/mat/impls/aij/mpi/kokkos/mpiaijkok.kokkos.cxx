#include <petscconf.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

PetscErrorCode MatAssemblyEnd_MPIAIJKokkos(Mat A,MatAssemblyType mode)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *mpiaij = (Mat_MPIAIJ*)A->data;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_MPIAIJ(A,mode);CHKERRQ(ierr);
  if (!A->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = VecSetType(mpiaij->lvec,VECSEQKOKKOS);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  MatMPIAIJSetPreallocation_MPIAIJKokkos(Mat mat,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscErrorCode     ierr;
  PetscInt           i;
  Mat_MPIAIJ         *mpiaij = (Mat_MPIAIJ*)mat->data;


  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(mat->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(mat->cmap);CHKERRQ(ierr);
  if (d_nnz) {
    for (i=0; i<mat->rmap->n; i++) {
      if (d_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than 0: local row %D value %D",i,d_nnz[i]);
    }
  }
  if (o_nnz) {
    for (i=0; i<mat->rmap->n; i++) {
      if (o_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than 0: local row %D value %D",i,o_nnz[i]);
    }
  }
  if (!mat->preallocated) {
    /* Explicitly create 2 MATSEQAIJKOKKOS matrices. */
    ierr = MatCreate(PETSC_COMM_SELF,&mpiaij->A);CHKERRQ(ierr);
    ierr = MatSetSizes(mpiaij->A,mat->rmap->n,mat->cmap->n,mat->rmap->n,mat->cmap->n);CHKERRQ(ierr);
    ierr = MatSetType(mpiaij->A,MATSEQAIJKOKKOS);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)mpiaij->A);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&mpiaij->B);CHKERRQ(ierr);
    ierr = MatSetSizes(mpiaij->B,mat->rmap->n,mat->cmap->N,mat->rmap->n,mat->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(mpiaij->B,MATSEQAIJKOKKOS);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)mpiaij->B);CHKERRQ(ierr);
  }
  ierr = MatSeqAIJSetPreallocation(mpiaij->A,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mpiaij->B,o_nz,o_nnz);CHKERRQ(ierr);

  mat->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPIAIJKokkos(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIAIJ     *mpiaij = (Mat_MPIAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != mat->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of mat (%D) and xx (%D)",mat->cmap->n,nt);
  ierr = VecScatterBegin(mpiaij->Mvctx,xx,mpiaij->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*mpiaij->A->ops->mult)(mpiaij->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterEnd(mpiaij->Mvctx,xx,mpiaij->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*mpiaij->B->ops->multadd)(mpiaij->B,mpiaij->lvec,yy,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPIAIJKokkos(Mat mat,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ     *mpiaij = (Mat_MPIAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != mat->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of mat (%D) and xx (%D)",mat->cmap->n,nt);
  ierr = VecScatterBegin(mpiaij->Mvctx,xx,mpiaij->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*mpiaij->A->ops->multadd)(mpiaij->A,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(mpiaij->Mvctx,xx,mpiaij->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*mpiaij->B->ops->multadd)(mpiaij->B,mpiaij->lvec,zz,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_MPIAIJKokkos(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIAIJ     *mpiaij = (Mat_MPIAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != mat->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of mat (%D) and xx (%D)",mat->rmap->n,nt);
  ierr = (*mpiaij->B->ops->multtranspose)(mpiaij->B,xx,mpiaij->lvec);CHKERRQ(ierr);
  ierr = (*mpiaij->A->ops->multtranspose)(mpiaij->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterBegin(mpiaij->Mvctx,mpiaij->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(mpiaij->Mvctx,mpiaij->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJKokkos(Mat A, MatType mtype, MatReuse reuse, Mat* newmat)
{
  PetscErrorCode     ierr;
  Mat                B;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,newmat);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatCopy(A,*newmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  B = *newmat;

  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECKOKKOS,&B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATMPIAIJKOKKOS);CHKERRQ(ierr);

  B->ops->assemblyend    = MatAssemblyEnd_MPIAIJKokkos;
  B->ops->mult           = MatMult_MPIAIJKokkos;
  B->ops->multadd        = MatMultAdd_MPIAIJKokkos;
  B->ops->multtranspose  = MatMultTranspose_MPIAIJKokkos;

  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",MatMPIAIJSetPreallocation_MPIAIJKokkos);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJKokkos(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = MatCreate_MPIAIJ(A);CHKERRQ(ierr);
  ierr = MatConvert_MPIAIJ_MPIAIJKokkos(A,MATMPIAIJKOKKOS,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatCreateAIJKokkos - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format).  This matrix will ultimately pushed down
   to Kokkos for calculations. For good matrix
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
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
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

.seealso: MatCreate(), MatCreateAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays(), MatCreateAIJ(), MATMPIAIJKOKKOS, MATAIJKokkos
@*/
PetscErrorCode  MatCreateAIJKokkos(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIAIJKOKKOS);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*A,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQAIJKOKKOS);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(*A,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

