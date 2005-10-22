#define PETSCMAT_DLL

/*
  Defines a matrix-vector product for the MATSEQAIJCRL matrix class.
  This class is derived from the MATSEQAIJ class and retains the 
  compressed row storage (aka Yale sparse matrix format) but augments 
  it with a column oriented storage that is more efficient for 
  matrix vector products on Vector machines.

  CRL stands for constant row length (that is the same number of columns
  is kept (padded with zeros) for each row of the sparse matrix.
*/

#include "src/mat/impls/aij/seq/aij.h"

typedef struct {
  PetscInt    ncols;    /* number of columns in each row */
  PetscInt    *icols;   /* columns of nonzeros, stored one column at a time */ 
  PetscScalar *acols;   /* values of nonzeros, stored as icols */

  /* We need to keep a pointer to MatAssemblyEnd_SeqAIJ because we 
   * actually want to call this function from within the 
   * MatAssemblyEnd_SeqCRL function.  Similarly, we also need 
   * MatDestroy_SeqAIJ and MatDuplicate_SeqAIJ. */
  PetscErrorCode (*AssemblyEnd_SeqAIJ)(Mat,MatAssemblyType);
  PetscErrorCode (*MatDestroy_SeqAIJ)(Mat);
  PetscErrorCode (*MatDuplicate_SeqAIJ)(Mat,MatDuplicateOption,Mat*);
} Mat_SeqCRL;

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqCRL"
PetscErrorCode MatDestroy_SeqCRL(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqCRL     *crl = (Mat_SeqCRL *) A->spptr;

  /* We are going to convert A back into a SEQAIJ matrix, since we are 
   * eventually going to use MatDestroy_SeqAIJ() to destroy everything 
   * that is not specific to CRL.
   * In preparation for this, reset the operations pointers in A to 
   * their SeqAIJ versions. */
  A->ops->assemblyend = crl->AssemblyEnd_SeqAIJ;
  A->ops->destroy     = crl->MatDestroy_SeqAIJ;
  A->ops->duplicate   = crl->MatDuplicate_SeqAIJ;

  /* Free everything in the Mat_SeqCRL data structure. */
  if (crl->icols) {
    ierr = PetscFree2(crl->acols,crl->icols);CHKERRQ(ierr);
  }
  /* Free the Mat_SeqCRL struct itself. */
  ierr = PetscFree(crl);CHKERRQ(ierr);

  /* Change the type of A back to SEQAIJ and use MatDestroy_SeqAIJ() 
   * to destroy everything that remains. */
  ierr = PetscObjectChangeTypeName( (PetscObject)A, MATSEQAIJ);CHKERRQ(ierr);
  /* Note that I don't call MatSetType().  I believe this is because that 
   * is only to be called when *building* a matrix. */
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqCRL(Mat A, MatDuplicateOption op, Mat *M) 
{
  PetscErrorCode ierr;
  Mat_SeqCRL     *crl = (Mat_SeqCRL *) A->spptr;

  PetscFunctionBegin;
  ierr = (*crl->MatDuplicate_SeqAIJ)(A,op,M);CHKERRQ(ierr);
  SETERRQ(PETSC_ERR_SUP,"Cannot duplicate CRL matrices yet");    
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SeqCRL_create_crl"
PetscErrorCode SeqCRL_create_crl(Mat A)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ *)(A)->data;
  Mat_SeqCRL     *crl = (Mat_SeqCRL*) A->spptr;
  PetscInt       m = A->m;  /* Number of rows in the matrix. */
  PetscInt       *aj = a->j;  /* From the CSR representation; points to the beginning  of each row. */
  PetscInt       i, j,rmax = a->rmax,*icols, *ilen = a->ilen;
  PetscScalar    *aa = a->a,*acols;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr  = PetscMalloc2(rmax*m,PetscScalar,&crl->acols,rmax*m,PetscInt,&crl->icols);CHKERRQ(ierr);
  acols = crl->acols;
  icols = crl->icols;
  for (i=0; i<m; i++) {
    for (j=0; j<ilen[i]; j++) {
      acols[j*m+i] = *aa++;
      icols[j*m+i] = *aj++;
    }
    for (;j<rmax; j++) { /* empty column entries */
      acols[j*m+i] = 0.0;
      icols[j*m+i] = (j) ? icols[(j-1)*m+i] : 0;  /* handle case where row is EMPTY */
    }
  }
  ierr = PetscLogInfo((A,"SeqCRL_create_crl: Percentage of 0's introduced for vectorized multiply %g\n",1.0-((double)a->nz)/((double)(rmax*m))));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SeqCRL"
PetscErrorCode MatAssemblyEnd_SeqCRL(Mat A, MatAssemblyType mode)
{
  PetscErrorCode ierr;
  Mat_SeqCRL     *crl = (Mat_SeqCRL*) A->spptr;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;

  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  
  /* Since a MATSEQCRL matrix is really just a MATSEQAIJ with some 
   * extra information, call the AssemblyEnd routine for a MATSEQAIJ. 
   * I'm not sure if this is the best way to do this, but it avoids 
   * a lot of code duplication.
   * I also note that currently MATSEQCRL doesn't know anything about 
   * the Mat_CompressedRow data structure that SeqAIJ now uses when there 
   * are many zero rows.  If the SeqAIJ assembly end routine decides to use 
   * this, this may break things.  (Don't know... haven't looked at it.) */
  a->inode.use = PETSC_FALSE;
  (*crl->AssemblyEnd_SeqAIJ)(A, mode);

  /* Now calculate the permutation and grouping information. */
  ierr = SeqCRL_create_crl(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqCRL"
PetscErrorCode MatMult_SeqCRL(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ *)(A)->data;
  Mat_SeqCRL     *crl = (Mat_SeqCRL*) A->spptr;
  PetscInt       m = A->m;  /* Number of rows in the matrix. */
  PetscInt       rmax = a->rmax,*icols = crl->icols;
  PetscScalar    *acols = crl->acols;
  PetscErrorCode ierr;
  PetscScalar    *x,*y;
#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTCRL)
  PetscInt       i,j,ii;
#endif


#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aa)
#endif

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

#if defined(PETSC_USE_FORTRAN_KERNEL_MULTCRL)
  fortranmultcrl_(&m,&rmax,x,y,icols,acols);
#else

  /* first column */
  for (j=0; j<m; j++) { 
    y[j] = acols[j]*x[icols[j]];
  }

  /* other columns */
#if defined(PETSC_HAVE_CRAYC)
#pragma _CRI preferstream
#endif
  for (i=1; i<rmax; i++) {
    ii = i*m;
#if defined(PETSC_HAVE_CRAYC)
#pragma _CRI prefervector
#endif
    for (j=0; j<m; j++) { 
      y[j] = y[j] + acols[ii+j]*x[icols[ii+j]];
    }
  }
#if defined(PETSC_HAVE_CRAYC)
#pragma _CRI ivdep
#endif

#endif
  ierr = PetscLogFlops(2*a->nz - m);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* MatConvert_SeqAIJ_SeqCRL converts a SeqAIJ matrix into a 
 * SeqCRL matrix.  This routine is called by the MatCreate_SeqCRL() 
 * routine, but can also be used to convert an assembled SeqAIJ matrix 
 * into a SeqCRL one. */
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqAIJ_SeqCRL"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SeqAIJ_SeqCRL(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  /* This routine is only called to convert to MATSEQCRL
   * from MATSEQAIJ, so we can ignore 'MatType Type'. */
  PetscErrorCode ierr;
  Mat            B = *newmat;
  Mat_SeqCRL     *crl;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr = PetscNew(Mat_SeqCRL,&crl);CHKERRQ(ierr);
  B->spptr = (void *) crl;

  /* Save a pointer to the original SeqAIJ assembly end routine, because we 
   * will want to use it later in the CRL assembly end routine. 
   * Also, save a pointer to the original SeqAIJ Destroy routine, because we 
   * will want to use it in the CRL destroy routine. */
  crl->AssemblyEnd_SeqAIJ  = A->ops->assemblyend;
  crl->MatDestroy_SeqAIJ   = A->ops->destroy;
  crl->MatDuplicate_SeqAIJ = A->ops->duplicate;

  /* Set function pointers for methods that we inherit from AIJ but 
   * override. */
  B->ops->duplicate   = MatDuplicate_SeqCRL;
  B->ops->assemblyend = MatAssemblyEnd_SeqCRL;
  B->ops->destroy     = MatDestroy_SeqCRL;
  B->ops->mult        = MatMult_SeqCRL;

  /* If A has already been assembled, compute the permutation. */
  if (A->assembled == PETSC_TRUE) {
    ierr = SeqCRL_create_crl(B);CHKERRQ(ierr);
  }
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQCRL);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__
#define __FUNCT__ "MatCreateSeqCRL"
/*@C
   MatCreateSeqCRL - Creates a sparse matrix of type SEQCRL.
   This type inherits from AIJ, but stores some additional
   information that is used to allow better vectorization of 
   the matrix-vector product. At the cost of increased storage, the AIJ formatted 
   matrix can be copied to a format in which pieces of the matrix are 
   stored in ELLPACK format, allowing the vectorized matrix multiply 
   routine to use stride-1 memory accesses.  As with the AIJ type, it is 
   important to preallocate matrix storage in order to get good assembly 
   performance.
   
   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows 
         (possibly different for each row) or PETSC_NULL

   Output Parameter:
.  A - the matrix 

   Notes:
   If nnz is given then nz is ignored

   Level: intermediate

.keywords: matrix, cray, sparse, parallel

.seealso: MatCreate(), MatCreateMPICSRPERM(), MatSetValues()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqCRL(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQCRL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqCRL"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqCRL(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Change the type name before calling MatSetType() to force proper construction of SeqAIJ 
     and MATSEQCRL types. */
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATSEQCRL);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqCRL(A,MATSEQCRL,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

