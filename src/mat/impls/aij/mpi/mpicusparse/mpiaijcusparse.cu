#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_CXX_COMPLEX_FIX
#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1

#include <petscconf.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>   /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/seqcusparse/cusparsematimpl.h>
#include <../src/mat/impls/aij/mpi/mpicusparse/mpicusparsematimpl.h>

PetscErrorCode  MatMPIAIJSetPreallocation_MPIAIJCUSPARSE(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  Mat_MPIAIJ         *b               = (Mat_MPIAIJ*)B->data;
  Mat_MPIAIJCUSPARSE * cusparseStruct = (Mat_MPIAIJCUSPARSE*)b->spptr;
  PetscErrorCode     ierr;
  PetscInt           i;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  if (d_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      if (d_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than 0: local row %D value %D",i,d_nnz[i]);
    }
  }
  if (o_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      if (o_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than 0: local row %D value %D",i,o_nnz[i]);
    }
  }
  if (!B->preallocated) {
    /* Explicitly create 2 MATSEQAIJCUSPARSE matrices. */
    ierr = MatCreate(PETSC_COMM_SELF,&b->A);CHKERRQ(ierr);
    ierr = MatBindToCPU(b->A,B->boundtocpu);CHKERRQ(ierr);
    ierr = MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n);CHKERRQ(ierr);
    ierr = MatSetType(b->A,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->A);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&b->B);CHKERRQ(ierr);
    ierr = MatBindToCPU(b->B,B->boundtocpu);CHKERRQ(ierr);
    ierr = MatSetSizes(b->B,B->rmap->n,B->cmap->N,B->rmap->n,B->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(b->B,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->B);CHKERRQ(ierr);
  }
  ierr = MatSeqAIJSetPreallocation(b->A,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(b->B,o_nz,o_nnz);CHKERRQ(ierr);
  ierr = MatCUSPARSESetFormat(b->A,MAT_CUSPARSE_MULT,cusparseStruct->diagGPUMatFormat);CHKERRQ(ierr);
  ierr = MatCUSPARSESetFormat(b->B,MAT_CUSPARSE_MULT,cusparseStruct->offdiagGPUMatFormat);CHKERRQ(ierr);
  ierr = MatCUSPARSESetHandle(b->A,cusparseStruct->handle);CHKERRQ(ierr);
  ierr = MatCUSPARSESetHandle(b->B,cusparseStruct->handle);CHKERRQ(ierr);
  ierr = MatCUSPARSESetStream(b->A,cusparseStruct->stream);CHKERRQ(ierr);
  ierr = MatCUSPARSESetStream(b->B,cusparseStruct->stream);CHKERRQ(ierr);

  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->cmap->n,nt);
  ierr = VecScatterInitializeForGPU(a->Mvctx,xx);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->A->ops->mult)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,yy,yy);CHKERRQ(ierr);
  ierr = VecScatterFinalizeForGPU(a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_MPIAIJCUSPARSE(Mat A)
{
  Mat_MPIAIJ     *l = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_MPIAIJCUSPARSE *spptr = (Mat_MPIAIJCUSPARSE*)l->spptr;
    PetscSplitCSRDataStructure *d_mat = spptr->deviceMat;
    if (d_mat) {
      Mat_SeqAIJ   *a = (Mat_SeqAIJ*)l->A->data;
      Mat_SeqAIJ   *b = (Mat_SeqAIJ*)l->B->data;
      PetscInt     n = A->rmap->n, nnza = a->i[n], nnzb = b->i[n];
      cudaError_t  err;
      PetscScalar  *vals;
      ierr = PetscInfo(A,"Zero device matrix diag and offfdiag\n");CHKERRQ(ierr);
      err = cudaMemcpy( &vals, &d_mat->diag.a, sizeof(PetscScalar*), cudaMemcpyDeviceToHost);CHKERRCUDA(err);
      err = cudaMemset( vals, 0, (nnza)*sizeof(PetscScalar));CHKERRCUDA(err);
      err = cudaMemcpy( &vals, &d_mat->offdiag.a, sizeof(PetscScalar*), cudaMemcpyDeviceToHost);CHKERRCUDA(err);
      err = cudaMemset( vals, 0, (nnzb)*sizeof(PetscScalar));CHKERRCUDA(err);
    }
  }
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
PetscErrorCode MatMultAdd_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->cmap->n,nt);
  ierr = VecScatterInitializeForGPU(a->Mvctx,xx);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,zz,zz);CHKERRQ(ierr);
  ierr = VecScatterFinalizeForGPU(a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->rmap->n,nt);
  ierr = VecScatterInitializeForGPU(a->Mvctx,a->lvec);CHKERRQ(ierr);
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  ierr = (*a->A->ops->multtranspose)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterFinalizeForGPU(a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCUSPARSESetFormat_MPIAIJCUSPARSE(Mat A,MatCUSPARSEFormatOperation op,MatCUSPARSEStorageFormat format)
{
  Mat_MPIAIJ         *a               = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE * cusparseStruct = (Mat_MPIAIJCUSPARSE*)a->spptr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_CUSPARSE_MULT_DIAG:
    cusparseStruct->diagGPUMatFormat = format;
    break;
  case MAT_CUSPARSE_MULT_OFFDIAG:
    cusparseStruct->offdiagGPUMatFormat = format;
    break;
  case MAT_CUSPARSE_ALL:
    cusparseStruct->diagGPUMatFormat    = format;
    cusparseStruct->offdiagGPUMatFormat = format;
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unsupported operation %d for MatCUSPARSEFormatOperation. Only MAT_CUSPARSE_MULT_DIAG, MAT_CUSPARSE_MULT_DIAG, and MAT_CUSPARSE_MULT_ALL are currently supported.",op);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetFromOptions_MPIAIJCUSPARSE(PetscOptionItems *PetscOptionsObject,Mat A)
{
  MatCUSPARSEStorageFormat format;
  PetscErrorCode           ierr;
  PetscBool                flg;
  Mat_MPIAIJ               *a = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE       *cusparseStruct = (Mat_MPIAIJCUSPARSE*)a->spptr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"MPIAIJCUSPARSE options");CHKERRQ(ierr);
  if (A->factortype==MAT_FACTOR_NONE) {
    ierr = PetscOptionsEnum("-mat_cusparse_mult_diag_storage_format","sets storage format of the diagonal blocks of (mpi)aijcusparse gpu matrices for SpMV",
                            "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparseStruct->diagGPUMatFormat,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatCUSPARSESetFormat(A,MAT_CUSPARSE_MULT_DIAG,format);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnum("-mat_cusparse_mult_offdiag_storage_format","sets storage format of the off-diagonal blocks (mpi)aijcusparse gpu matrices for SpMV",
                            "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparseStruct->offdiagGPUMatFormat,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatCUSPARSESetFormat(A,MAT_CUSPARSE_MULT_OFFDIAG,format);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnum("-mat_cusparse_storage_format","sets storage format of the diagonal and off-diagonal blocks (mpi)aijcusparse gpu matrices for SpMV",
                            "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparseStruct->diagGPUMatFormat,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatCUSPARSESetFormat(A,MAT_CUSPARSE_ALL,format);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MPIAIJCUSPARSE(Mat A,MatAssemblyType mode)
{
  PetscErrorCode             ierr;
  Mat_MPIAIJ                 *mpiaij = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE         *cusparseStruct = (Mat_MPIAIJCUSPARSE*)mpiaij->spptr;
  PetscSplitCSRDataStructure *d_mat = cusparseStruct->deviceMat;
  PetscInt                    nnz_state = A->nonzerostate;
  PetscFunctionBegin;
  if (d_mat) {
    cudaError_t                err;
    err = cudaMemcpy( &nnz_state, &d_mat->nonzerostate, sizeof(PetscInt), cudaMemcpyDeviceToHost);CHKERRCUDA(err);
  }
  ierr = MatAssemblyEnd_MPIAIJ(A,mode);CHKERRQ(ierr);
  if (!A->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = VecSetType(mpiaij->lvec,VECSEQCUDA);CHKERRQ(ierr);
  }
  if (nnz_state > A->nonzerostate) {
    A->offloadmask = PETSC_OFFLOAD_GPU; // if we assembled on the device
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIAIJCUSPARSE(Mat A)
{
  PetscErrorCode     ierr;
  Mat_MPIAIJ         *aij            = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE*)aij->spptr;
  cudaError_t        err;
  cusparseStatus_t   stat;

  PetscFunctionBegin;
  if (cusparseStruct->deviceMat) {
    Mat_SeqAIJ                 *jaca = (Mat_SeqAIJ*)aij->A->data;
    Mat_SeqAIJ                 *jacb = (Mat_SeqAIJ*)aij->B->data;
    PetscSplitCSRDataStructure *d_mat = cusparseStruct->deviceMat, h_mat;
    ierr = PetscInfo(A,"Have device matrix\n");CHKERRQ(ierr);
    err = cudaMemcpy( &h_mat, d_mat, sizeof(PetscSplitCSRDataStructure), cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    if (jaca->compressedrow.use) {
      err = cudaFree(h_mat.diag.i);CHKERRCUDA(err);
    }
    if (jacb->compressedrow.use) {
      err = cudaFree(h_mat.offdiag.i);CHKERRCUDA(err);
    }
    err = cudaFree(h_mat.colmap);CHKERRCUDA(err);
    err = cudaFree(d_mat);CHKERRCUDA(err);
  }
  try {
    if (aij->A) { ierr = MatCUSPARSEClearHandle(aij->A);CHKERRQ(ierr); }
    if (aij->B) { ierr = MatCUSPARSEClearHandle(aij->B);CHKERRQ(ierr); }
    stat = cusparseDestroy(cusparseStruct->handle);CHKERRCUSPARSE(stat);
    if (cusparseStruct->stream) {
      err = cudaStreamDestroy(cusparseStruct->stream);CHKERRCUDA(err);
    }
    delete cusparseStruct;
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Mat_MPIAIJCUSPARSE error: %s", ex);
  }
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCUSPARSESetFormat_C",NULL);CHKERRQ(ierr);
  ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJCUSPARSE(Mat B, MatType mtype, MatReuse reuse, Mat* newmat)
{
  PetscErrorCode     ierr;
  Mat_MPIAIJ         *a;
  Mat_MPIAIJCUSPARSE *cusparseStruct;
  cusparseStatus_t   stat;
  Mat                A;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(B,MAT_COPY_VALUES,newmat);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatCopy(B,*newmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  A = *newmat;

  ierr = PetscFree(A->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&A->defaultvectype);CHKERRQ(ierr);

  a = (Mat_MPIAIJ*)A->data;
  if (reuse != MAT_REUSE_MATRIX && !a->spptr) {
    a->spptr = new Mat_MPIAIJCUSPARSE;

    cusparseStruct                      = (Mat_MPIAIJCUSPARSE*)a->spptr;
    cusparseStruct->diagGPUMatFormat    = MAT_CUSPARSE_CSR;
    cusparseStruct->offdiagGPUMatFormat = MAT_CUSPARSE_CSR;
    cusparseStruct->stream              = 0;
    stat = cusparseCreate(&(cusparseStruct->handle));CHKERRCUSPARSE(stat);
    cusparseStruct->deviceMat = NULL;
  }

  A->ops->assemblyend    = MatAssemblyEnd_MPIAIJCUSPARSE;
  A->ops->mult           = MatMult_MPIAIJCUSPARSE;
  A->ops->multadd        = MatMultAdd_MPIAIJCUSPARSE;
  A->ops->multtranspose  = MatMultTranspose_MPIAIJCUSPARSE;
  A->ops->setfromoptions = MatSetFromOptions_MPIAIJCUSPARSE;
  A->ops->destroy        = MatDestroy_MPIAIJCUSPARSE;
  A->ops->zeroentries    = MatZeroEntries_MPIAIJCUSPARSE;

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATMPIAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",MatMPIAIJSetPreallocation_MPIAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCUSPARSESetFormat_C",MatCUSPARSESetFormat_MPIAIJCUSPARSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCUSPARSE(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr);
  ierr = MatCreate_MPIAIJ(A);CHKERRQ(ierr);
  ierr = MatConvert_MPIAIJ_MPIAIJCUSPARSE(A,MATMPIAIJCUSPARSE,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatCreateAIJCUSPARSE - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format).  This matrix will ultimately pushed down
   to NVidia GPUs and use the CUSPARSE library for calculations. For good matrix
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

.seealso: MatCreate(), MatCreateAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays(), MatCreateAIJ(), MATMPIAIJCUSPARSE, MATAIJCUSPARSE
@*/
PetscErrorCode  MatCreateAIJCUSPARSE(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIAIJCUSPARSE);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*A,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(*A,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   MATAIJCUSPARSE - MATMPIAIJCUSPARSE = "aijcusparse" = "mpiaijcusparse" - A matrix type to be used for sparse matrices.

   A matrix type type whose data resides on Nvidia GPUs. These matrices can be in either
   CSR, ELL, or Hybrid format. The ELL and HYB formats require CUDA 4.2 or later.
   All matrix calculations are performed on Nvidia GPUs using the CUSPARSE library.

   This matrix type is identical to MATSEQAIJCUSPARSE when constructed with a single process communicator,
   and MATMPIAIJCUSPARSE otherwise.  As a result, for single process communicators,
   MatSeqAIJSetPreallocation is supported, and similarly MatMPIAIJSetPreallocation is supported
   for communicators controlling multiple processes.  It is recommended that you call both of
   the above preallocation routines for simplicity.

   Options Database Keys:
+  -mat_type mpiaijcusparse - sets the matrix type to "mpiaijcusparse" during a call to MatSetFromOptions()
.  -mat_cusparse_storage_format csr - sets the storage format of diagonal and off-diagonal matrices during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).
.  -mat_cusparse_mult_diag_storage_format csr - sets the storage format of diagonal matrix during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).
-  -mat_cusparse_mult_offdiag_storage_format csr - sets the storage format of off-diagonal matrix during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).

  Level: beginner

 .seealso: MatCreateAIJCUSPARSE(), MATSEQAIJCUSPARSE, MatCreateSeqAIJCUSPARSE(), MatCUSPARSESetFormat(), MatCUSPARSEStorageFormat, MatCUSPARSEFormatOperation
M
M*/

// get GPU pointer to stripped down Mat. For both Seq and MPI Mat.
PetscErrorCode MatCUSPARSEGetDeviceMatWrite(Mat A, PetscSplitCSRDataStructure **B)
{
#if defined(PETSC_USE_CTABLE)
  SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Device metadata does not support ctable (--with-ctable=0)");
#else
  PetscSplitCSRDataStructure **p_d_mat;
  PetscMPIInt                size,rank;
  MPI_Comm                   comm;
  PetscErrorCode             ierr;
  int                        *ai,*bi,*aj,*bj;
  PetscScalar                *aa,*ba;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_NONE) {
    CsrMatrix *matrixA,*matrixB=NULL;
    if (size == 1) {
      Mat_SeqAIJCUSPARSE *cusparsestruct = (Mat_SeqAIJCUSPARSE*)A->spptr;
      p_d_mat = &cusparsestruct->deviceMat;
      Mat_SeqAIJCUSPARSEMultStruct *matstruct = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestruct->mat;
      if (cusparsestruct->format==MAT_CUSPARSE_CSR) {
	matrixA = (CsrMatrix*)matstruct->mat;
	bi = bj = NULL; ba = NULL;
      } else {
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Device Mat needs MAT_CUSPARSE_CSR");
      }
    } else {
      Mat_MPIAIJ         *aij = (Mat_MPIAIJ*)A->data;
      Mat_MPIAIJCUSPARSE *spptr = (Mat_MPIAIJCUSPARSE*)aij->spptr;
      p_d_mat = &spptr->deviceMat;
      Mat_SeqAIJCUSPARSE *cusparsestructA = (Mat_SeqAIJCUSPARSE*)aij->A->spptr;
      Mat_SeqAIJCUSPARSE *cusparsestructB = (Mat_SeqAIJCUSPARSE*)aij->B->spptr;
      Mat_SeqAIJCUSPARSEMultStruct *matstructA = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestructA->mat;
      Mat_SeqAIJCUSPARSEMultStruct *matstructB = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestructB->mat;
      if (cusparsestructA->format==MAT_CUSPARSE_CSR) {
	if (cusparsestructB->format!=MAT_CUSPARSE_CSR) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Device Mat B needs MAT_CUSPARSE_CSR");
	matrixA = (CsrMatrix*)matstructA->mat;
	matrixB = (CsrMatrix*)matstructB->mat;
	bi = thrust::raw_pointer_cast(matrixB->row_offsets->data());
	bj = thrust::raw_pointer_cast(matrixB->column_indices->data());
	ba = thrust::raw_pointer_cast(matrixB->values->data());
	if (rank==-1) {
	  for(unsigned int i = 0; i < matrixB->row_offsets->size(); i++)
	    std::cout << "\trow_offsets[" << i << "] = " << (*matrixB->row_offsets)[i] << std::endl;
	  for(unsigned int i = 0; i < matrixB->column_indices->size(); i++)
	    std::cout << "\tcolumn_indices[" << i << "] = " << (*matrixB->column_indices)[i] << std::endl;
	  for(unsigned int i = 0; i < matrixB->values->size(); i++)
	    std::cout << "\tvalues[" << i << "] = " << (*matrixB->values)[i] << std::endl;
	}
      } else {
	SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Device Mat A needs MAT_CUSPARSE_CSR");
      }
    }
    ai = thrust::raw_pointer_cast(matrixA->row_offsets->data());
    aj = thrust::raw_pointer_cast(matrixA->column_indices->data());
    aa = thrust::raw_pointer_cast(matrixA->values->data());
  } else {
    *B = NULL;
    PetscFunctionReturn(0);
  }
  // act like MatSetValues because not called on host
  if (A->assembled) {
    if (A->was_assembled) {
      ierr = PetscInfo(A,"Assemble more than once already\n");CHKERRQ(ierr);
    }
    A->was_assembled = PETSC_TRUE; // this is done (lazy) in MatAssemble but we are not calling it anymore - done in AIJ AssemblyEnd, need here?
  } else {
    SETERRQ(comm,PETSC_ERR_SUP,"Need assemble matrix");
  }
  if (!*p_d_mat) {
    cudaError_t                 err;
    PetscSplitCSRDataStructure  *d_mat, h_mat;
    Mat_SeqAIJ                  *jaca;
    PetscInt                    n = A->rmap->n, nnz;
    // create and copy
    ierr = PetscInfo(A,"Create device matrix\n");CHKERRQ(ierr);
    err = cudaMalloc((void **)&d_mat, sizeof(PetscSplitCSRDataStructure));CHKERRCUDA(err);
    err = cudaMemset( d_mat, 0,       sizeof(PetscSplitCSRDataStructure));CHKERRCUDA(err);
    *B = *p_d_mat = d_mat; // return it, set it in Mat, and set it up
    if (size == 1) {
      jaca = (Mat_SeqAIJ*)A->data;
      h_mat.nonzerostate = A->nonzerostate;
      h_mat.rstart = 0; h_mat.rend = A->rmap->n;
      h_mat.cstart = 0; h_mat.cend = A->cmap->n;
      h_mat.offdiag.i = h_mat.offdiag.j = NULL;
      h_mat.offdiag.a = NULL;
      h_mat.seq = PETSC_TRUE;
    } else {
      Mat_MPIAIJ  *aij = (Mat_MPIAIJ*)A->data;
      Mat_SeqAIJ  *jacb;
      h_mat.seq = PETSC_FALSE; // for MatAssemblyEnd_SeqAIJCUSPARSE
      jaca = (Mat_SeqAIJ*)aij->A->data;
      jacb = (Mat_SeqAIJ*)aij->B->data;
      h_mat.nonzerostate = aij->A->nonzerostate; // just keep one nonzero state?
      if (!aij->garray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MPIAIJ Matrix was assembled but is missing garray");
      if (aij->B->rmap->n != aij->A->rmap->n) SETERRQ(comm,PETSC_ERR_SUP,"Only support aij->B->rmap->n == aij->A->rmap->n");
      // create colmap - this is ussually done (lazy) in MatSetValues
      aij->donotstash = PETSC_TRUE;
      aij->A->nooffprocentries = aij->B->nooffprocentries = A->nooffprocentries = PETSC_TRUE;
      jaca->nonew = jacb->nonew = PETSC_TRUE; // no more dissassembly
      ierr = PetscCalloc1(A->cmap->N+1,&aij->colmap);CHKERRQ(ierr);
      aij->colmap[A->cmap->N] = -9;
      ierr = PetscLogObjectMemory((PetscObject)A,(A->cmap->N+1)*sizeof(PetscInt));CHKERRQ(ierr);
      {
	PetscInt ii;
	for (ii=0; ii<aij->B->cmap->n; ii++) aij->colmap[aij->garray[ii]] = ii+1;
      }
      if(aij->colmap[A->cmap->N] != -9) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"aij->colmap[A->cmap->N] != -9");
      // allocate B copy data
      h_mat.rstart = A->rmap->rstart; h_mat.rend = A->rmap->rend;
      h_mat.cstart = A->cmap->rstart; h_mat.cend = A->cmap->rend;
      nnz = jacb->i[n];

      if (jacb->compressedrow.use) {
	err = cudaMalloc((void **)&h_mat.offdiag.i,               (n+1)*sizeof(int));CHKERRCUDA(err); // kernel input
	err = cudaMemcpy(          h_mat.offdiag.i,    jacb->i,   (n+1)*sizeof(int), cudaMemcpyHostToDevice);CHKERRCUDA(err);
      } else {
	h_mat.offdiag.i = bi;
      }
      h_mat.offdiag.j = bj;
      h_mat.offdiag.a = ba;

      err = cudaMalloc((void **)&h_mat.colmap,                  (A->cmap->N+1)*sizeof(PetscInt));CHKERRCUDA(err); // kernel output
      err = cudaMemcpy(          h_mat.colmap,    aij->colmap,  (A->cmap->N+1)*sizeof(PetscInt), cudaMemcpyHostToDevice);CHKERRCUDA(err);
      h_mat.offdiag.ignorezeroentries = jacb->ignorezeroentries;
      h_mat.offdiag.n = n;
    }
    // allocate A copy data
    nnz = jaca->i[n];
    h_mat.diag.n = n;
    h_mat.diag.ignorezeroentries = jaca->ignorezeroentries;
    ierr = MPI_Comm_rank(comm,&h_mat.rank);CHKERRQ(ierr);
    if (jaca->compressedrow.use) {
      err = cudaMalloc((void **)&h_mat.diag.i,               (n+1)*sizeof(int));CHKERRCUDA(err); // kernel input
      err = cudaMemcpy(          h_mat.diag.i,    jaca->i,   (n+1)*sizeof(int), cudaMemcpyHostToDevice);CHKERRCUDA(err);
    } else {
      h_mat.diag.i = ai;
    }
    h_mat.diag.j = aj;
    h_mat.diag.a = aa;
    // copy pointers and metdata to device
    err = cudaMemcpy(          d_mat, &h_mat, sizeof(PetscSplitCSRDataStructure), cudaMemcpyHostToDevice);CHKERRCUDA(err);
    ierr = PetscInfo2(A,"Create device Mat n=%D nnz=%D\n",h_mat.diag.n, nnz);CHKERRQ(ierr);
  } else {
    *B = *p_d_mat;
  }
  A->assembled = PETSC_FALSE; // ready to write with matsetvalues - this done (lazy) in normal MatSetValues
  PetscFunctionReturn(0);
#endif
}
