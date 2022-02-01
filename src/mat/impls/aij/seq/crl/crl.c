
/*
  Defines a matrix-vector product for the MATSEQAIJCRL matrix class.
  This class is derived from the MATSEQAIJ class and retains the
  compressed row storage (aka Yale sparse matrix format) but augments
  it with a column oriented storage that is more efficient for
  matrix vector products on Vector machines.

  CRL stands for constant row length (that is the same number of columns
  is kept (padded with zeros) for each row of the sparse matrix.
*/
#include <../src/mat/impls/aij/seq/crl/crl.h>

PetscErrorCode MatDestroy_SeqAIJCRL(Mat A)
{
  PetscErrorCode ierr;
  Mat_AIJCRL     *aijcrl = (Mat_AIJCRL*) A->spptr;

  PetscFunctionBegin;
  /* Free everything in the Mat_AIJCRL data structure. */
  if (aijcrl) {
    ierr = PetscFree2(aijcrl->acols,aijcrl->icols);CHKERRQ(ierr);
  }
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A, MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_AIJCRL(Mat A, MatDuplicateOption op, Mat *M)
{
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot duplicate AIJCRL matrices yet");
}

PetscErrorCode MatSeqAIJCRL_create_aijcrl(Mat A)
{
  Mat_SeqAIJ     *a      = (Mat_SeqAIJ*)(A)->data;
  Mat_AIJCRL     *aijcrl = (Mat_AIJCRL*) A->spptr;
  PetscInt       m       = A->rmap->n; /* Number of rows in the matrix. */
  PetscInt       *aj     = a->j; /* From the CSR representation; points to the beginning  of each row. */
  PetscInt       i, j,rmax = a->rmax,*icols, *ilen = a->ilen;
  MatScalar      *aa = a->a;
  PetscScalar    *acols;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  aijcrl->nz   = a->nz;
  aijcrl->m    = A->rmap->n;
  aijcrl->rmax = rmax;

  ierr  = PetscFree2(aijcrl->acols,aijcrl->icols);CHKERRQ(ierr);
  ierr  = PetscMalloc2(rmax*m,&aijcrl->acols,rmax*m,&aijcrl->icols);CHKERRQ(ierr);
  acols = aijcrl->acols;
  icols = aijcrl->icols;
  for (i=0; i<m; i++) {
    for (j=0; j<ilen[i]; j++) {
      acols[j*m+i] = *aa++;
      icols[j*m+i] = *aj++;
    }
    for (; j<rmax; j++) { /* empty column entries */
      acols[j*m+i] = 0.0;
      icols[j*m+i] = (j) ? icols[(j-1)*m+i] : 0;  /* handle case where row is EMPTY */
    }
  }
  ierr = PetscInfo2(A,"Percentage of 0's introduced for vectorized multiply %g. Rmax= %" PetscInt_FMT "\n",1.0-((double)a->nz)/((double)(rmax*m)),rmax);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_SeqAIJCRL(Mat A, MatAssemblyType mode)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  a->inode.use = PETSC_FALSE;

  ierr = MatAssemblyEnd_SeqAIJ(A,mode);CHKERRQ(ierr);
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  /* Now calculate the permutation and grouping information. */
  ierr = MatSeqAIJCRL_create_aijcrl(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/seq/crl/ftn-kernels/fmultcrl.h>

/*
    Shared by both sequential and parallel versions of CRL matrix: MATMPIAIJCRL and MATSEQAIJCRL
    - the scatter is used only in the parallel version

*/
PetscErrorCode MatMult_AIJCRL(Mat A,Vec xx,Vec yy)
{
  Mat_AIJCRL        *aijcrl = (Mat_AIJCRL*) A->spptr;
  PetscInt          m       = aijcrl->m; /* Number of rows in the matrix. */
  PetscInt          rmax    = aijcrl->rmax,*icols = aijcrl->icols;
  PetscScalar       *acols  = aijcrl->acols;
  PetscErrorCode    ierr;
  PetscScalar       *y;
  const PetscScalar *x;
#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTCRL)
  PetscInt          i,j,ii;
#endif

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aa)
#endif

  PetscFunctionBegin;
  if (aijcrl->xscat) {
    ierr = VecCopy(xx,aijcrl->xwork);CHKERRQ(ierr);
    /* get remote values needed for local part of multiply */
    ierr = VecScatterBegin(aijcrl->xscat,xx,aijcrl->fwork,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(aijcrl->xscat,xx,aijcrl->fwork,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    xx   = aijcrl->xwork;
  }

  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

#if defined(PETSC_USE_FORTRAN_KERNEL_MULTCRL)
  fortranmultcrl_(&m,&rmax,x,y,icols,acols);
#else

  /* first column */
  for (j=0; j<m; j++) y[j] = acols[j]*x[icols[j]];

  /* other columns */
#if defined(PETSC_HAVE_CRAY_VECTOR)
#pragma _CRI preferstream
#endif
  for (i=1; i<rmax; i++) {
    ii = i*m;
#if defined(PETSC_HAVE_CRAY_VECTOR)
#pragma _CRI prefervector
#endif
    for (j=0; j<m; j++) y[j] = y[j] + acols[ii+j]*x[icols[ii+j]];
  }
#if defined(PETSC_HAVE_CRAY_VECTOR)
#pragma _CRI ivdep
#endif

#endif
  ierr = PetscLogFlops(2.0*aijcrl->nz - m);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* MatConvert_SeqAIJ_SeqAIJCRL converts a SeqAIJ matrix into a
 * SeqAIJCRL matrix.  This routine is called by the MatCreate_SeqAIJCRL()
 * routine, but can also be used to convert an assembled SeqAIJ matrix
 * into a SeqAIJCRL one. */
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJCRL(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B = *newmat;
  Mat_AIJCRL     *aijcrl;
  PetscBool      sametype;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)A,type,&sametype);CHKERRQ(ierr);
  if (sametype) PetscFunctionReturn(0);

  ierr     = PetscNewLog(B,&aijcrl);CHKERRQ(ierr);
  B->spptr = (void*) aijcrl;

  /* Set function pointers for methods that we inherit from AIJ but override. */
  B->ops->duplicate   = MatDuplicate_AIJCRL;
  B->ops->assemblyend = MatAssemblyEnd_SeqAIJCRL;
  B->ops->destroy     = MatDestroy_SeqAIJCRL;
  B->ops->mult        = MatMult_AIJCRL;

  /* If A has already been assembled, compute the permutation. */
  if (A->assembled) {
    ierr = MatSeqAIJCRL_create_aijcrl(B);CHKERRQ(ierr);
  }
  ierr    = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJCRL);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}

/*@C
   MatCreateSeqAIJCRL - Creates a sparse matrix of type SEQAIJCRL.
   This type inherits from AIJ, but stores some additional
   information that is used to allow better vectorization of
   the matrix-vector product. At the cost of increased storage, the AIJ formatted
   matrix can be copied to a format in which pieces of the matrix are
   stored in ELLPACK format, allowing the vectorized matrix multiply
   routine to use stride-1 memory accesses.  As with the AIJ type, it is
   important to preallocate matrix storage in order to get good assembly
   performance.

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

   Notes:
   If nnz is given then nz is ignored

   Level: intermediate

.seealso: MatCreate(), MatCreateMPIAIJPERM(), MatSetValues()
@*/
PetscErrorCode  MatCreateSeqAIJCRL(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJCRL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCRL(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJCRL(A,MATSEQAIJCRL,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

