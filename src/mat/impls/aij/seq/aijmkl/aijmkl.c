/*
  Defines basic operations for the MATSEQAIJMKL matrix class.
  This class is derived from the MATSEQAIJ class and retains the
  compressed row storage (aka Yale sparse matrix format) but uses 
  sparse BLAS operations from the Intel Math Kernel Library (MKL) 
  wherever possible.
*/

#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/seq/aijmkl/aijmkl.h>

/* MKL include files. */
#include <mkl_spblas.h>  /* Sparse BLAS */

typedef struct {
  /* "Handle" used by SpMV2 inspector-executor routines. */
  sparse_matrix_t csrA;
} Mat_SeqAIJMKL;

extern PetscErrorCode MatAssemblyEnd_SeqAIJ(Mat,MatAssemblyType);

#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqAIJMKL_SeqAIJ"
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJMKL_SeqAIJ(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  /* This routine is only called to convert a MATAIJMKL to its base PETSc type, */
  /* so we will ignore 'MatType type'. */
  PetscErrorCode ierr;
  Mat            B       = *newmat;
  Mat_SeqAIJMKL *aijmkl=(Mat_SeqAIJMKL*)A->spptr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  /* Reset the original function pointers. */
  B->ops->duplicate        = MatDuplicate_SeqAIJ;
  B->ops->assemblyend      = MatAssemblyEnd_SeqAIJ;
  B->ops->destroy          = MatDestroy_SeqAIJ;
  B->ops->mult             = MatMult_SeqAIJ;
  B->ops->multtranspose    = MatMultTranspose_SeqAIJ;
  B->ops->multadd          = MatMultAdd_SeqAIJ;
  B->ops->multtransposeadd = MatMultTransposeAdd_SeqAIJ;

  /* Free everything in the Mat_SeqAIJMKL data structure.
   * We don't free the Mat_SeqAIJMKL struct itself, as this will
   * cause problems later when MatDestroy() tries to free it. */
  /* Actually there is nothing to do here right now.
   * When I've added use of the MKL SpMV2 inspector-executor routines, I should 
   * see if there is some way to clean up the "handle" used by SpMV2. */
  
  /* Change the type of B to MATSEQAIJ. */
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATSEQAIJ);CHKERRQ(ierr);

  *newmat = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJMKL"
PetscErrorCode MatDestroy_SeqAIJMKL(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqAIJMKL *aijmkl = (Mat_SeqAIJMKL*) A->spptr;

  PetscFunctionBegin;
  /* Clean up everything in the Mat_SeqAIJMKL data structure, then free A->spptr. */
  mkl_sparse_destroy(aijmkl->csrA);
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);

  /* Change the type of A back to SEQAIJ and use MatDestroy_SeqAIJ()
   * to destroy everything that remains. */
  ierr = PetscObjectChangeTypeName((PetscObject)A, MATSEQAIJ);CHKERRQ(ierr);
  /* Note that I don't call MatSetType().  I believe this is because that
   * is only to be called when *building* a matrix.  I could be wrong, but
   * that is how things work for the SuperLU matrix class. */
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_SeqAIJMKL"
PetscErrorCode MatDuplicate_SeqAIJMKL(Mat A, MatDuplicateOption op, Mat *M)
{
  PetscErrorCode ierr;
  Mat_SeqAIJMKL *aijmkl      = (Mat_SeqAIJMKL*) A->spptr;
  Mat_SeqAIJMKL *aijmkl_dest = (Mat_SeqAIJMKL*) (*M)->spptr;

  PetscFunctionBegin;
  ierr = MatDuplicate_SeqAIJ(A,op,M);CHKERRQ(ierr);
  ierr = PetscMemcpy((*M)->spptr,aijmkl,sizeof(Mat_SeqAIJMKL));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SeqAIJMKL"
PetscErrorCode MatAssemblyEnd_SeqAIJMKL(Mat A, MatAssemblyType mode)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  /* Since a MATSEQAIJMKL matrix is really just a MATSEQAIJ with some
   * extra information and some different methods, call the AssemblyEnd 
   * routine for a MATSEQAIJ.
   * I'm not sure if this is the best way to do this, but it avoids
   * a lot of code duplication.
   * I also note that currently MATSEQAIJMKL doesn't know anything about
   * the Mat_CompressedRow data structure that SeqAIJ now uses when there
   * are many zero rows.  If the SeqAIJ assembly end routine decides to use
   * this, this may break things.  (Don't know... haven't looked at it. 
   * Do I need to disable this somehow?) */
  a->inode.use = PETSC_FALSE;  /* Must disable: otherwise the MKL routines won't get used. */
  ierr         = MatAssemblyEnd_SeqAIJ(A, mode);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_SeqAIJMKL"
PetscErrorCode MatMult_SeqAIJMKL(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  const PetscScalar *x;
  PetscScalar       *y;
  const MatScalar   *aa;
  PetscErrorCode    ierr;
  PetscInt          m=A->rmap->n;
  const PetscInt    *aj,*ai;
  PetscInt          i;

  /* Variables not in MatMult_SeqAIJ. */
  char transa = 'n';  /* Used to indicate to MKL that we are not computing the transpose product. */

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  aj   = a->j;  /* aj[k] gives column index for element aa[k]. */
  aa   = a->a;  /* Nonzero elements stored row-by-row. */
  ai   = a->i;  /* ai[k] is the position in aa and aj where row k starts. */

  /* Call MKL sparse BLAS routine to do the MatMult. */
  mkl_cspblas_xcsrgemv(&transa,&m,aa,ai,aj,x,y);

  ierr = PetscLogFlops(2.0*a->nz - a->nonzerorowcnt);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_SeqAIJMKL"
PetscErrorCode MatMultTranspose_SeqAIJMKL(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  const PetscScalar *x;
  PetscScalar       *y;
  const MatScalar   *aa;
  PetscErrorCode    ierr;
  PetscInt          m=A->rmap->n;
  const PetscInt    *aj,*ai;
  PetscInt          i;

  /* Variables not in MatMultTranspose_SeqAIJ. */
  char transa = 't';  /* Used to indicate to MKL that we are computing the transpose product. */

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  aj   = a->j;  /* aj[k] gives column index for element aa[k]. */
  aa   = a->a;  /* Nonzero elements stored row-by-row. */
  ai   = a->i;  /* ai[k] is the position in aa and aj where row k starts. */

  /* Call MKL sparse BLAS routine to do the MatMult. */
  mkl_cspblas_xcsrgemv(&transa,&m,aa,ai,aj,x,y);

  ierr = PetscLogFlops(2.0*a->nz - a->nonzerorowcnt);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_SeqAIJMKL"
PetscErrorCode MatMultAdd_SeqAIJMKL(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  const PetscScalar *x;
  PetscScalar       *y,*z;
  const MatScalar   *aa;
  PetscErrorCode    ierr;
  PetscInt          m=A->rmap->n;
  const PetscInt    *aj,*ai;
  PetscInt          i;

  /* Variables not in MatMultAdd_SeqAIJ. */
  char transa = 'n';  /* Used to indicate to MKL that we are not computing the transpose product. */

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
  aj   = a->j;  /* aj[k] gives column index for element aa[k]. */
  aa   = a->a;  /* Nonzero elements stored row-by-row. */
  ai   = a->i;  /* ai[k] is the position in aa and aj where row k starts. */

  /* Call MKL sparse BLAS routine to do the MatMult. */
  mkl_cspblas_xcsrgemv(&transa,&m,aa,ai,aj,x,z);

  /* Now add the vector y; MKL sparse BLAS does not have a MatMultAdd equivalent. */
  for (i=0; i<m; i++) {
    z[i] += y[i];
  }

  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_SeqAIJMKL"
PetscErrorCode MatMultTransposeAdd_SeqAIJMKL(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  const PetscScalar *x;
  PetscScalar       *y,*z;
  const MatScalar   *aa;
  PetscErrorCode    ierr;
  PetscInt          m=A->rmap->n;
  const PetscInt    *aj,*ai;
  PetscInt          i;

  /* Variables not in MatMultTransposeAdd_SeqAIJ. */
  char transa = 't';  /* Used to indicate to MKL that we are computing the transpose product. */

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
  aj   = a->j;  /* aj[k] gives column index for element aa[k]. */
  aa   = a->a;  /* Nonzero elements stored row-by-row. */
  ai   = a->i;  /* ai[k] is the position in aa and aj where row k starts. */

  /* Call MKL sparse BLAS routine to do the MatMult. */
  mkl_cspblas_xcsrgemv(&transa,&m,aa,ai,aj,x,z);

  /* Now add the vector y; MKL sparse BLAS does not have a MatMultAdd equivalent. */
  for (i=0; i<m; i++) {
    z[i] += y[i];
  }

  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* MatConvert_SeqAIJ_SeqAIJMKL converts a SeqAIJ matrix into a
 * SeqAIJMKL matrix.  This routine is called by the MatCreate_SeqMKLAIJ()
 * routine, but can also be used to convert an assembled SeqAIJ matrix
 * into a SeqAIJMKL one. */
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqAIJ_SeqAIJMKL"
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJMKL(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B = *newmat;
  Mat_SeqAIJMKL *aijmkl;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr     = PetscNewLog(B,&aijmkl);CHKERRQ(ierr);
  B->spptr = (void*) aijmkl;

  /* Set function pointers for methods that we inherit from AIJ but override. */
  B->ops->duplicate        = MatDuplicate_SeqAIJMKL;
  B->ops->assemblyend      = MatAssemblyEnd_SeqAIJMKL;
  B->ops->destroy          = MatDestroy_SeqAIJMKL;
  B->ops->mult             = MatMult_SeqAIJMKL;
  B->ops->multtranspose    = MatMultTranspose_SeqAIJMKL;
  B->ops->multadd          = MatMultAdd_SeqAIJMKL;
  B->ops->multtransposeadd = MatMultTransposeAdd_SeqAIJMKL;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqaijmkl_seqaij_C",MatConvert_SeqAIJMKL_SeqAIJ);CHKERRQ(ierr);

  ierr    = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJMKL);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateSeqAIJMKL"
/*@C
   MatCreateSeqAIJMKL - Creates a sparse matrix of type SEQAIJMKL.
   This type inherits from AIJ and is largely identical, but uses sparse BLAS 
   routines from Intel MKL whenever possible.
   Collective on MPI_Comm

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

.keywords: matrix, cray, sparse, parallel

.seealso: MatCreate(), MatCreateMPIAIJMKL(), MatSetValues()
@*/
PetscErrorCode  MatCreateSeqAIJMKL(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJMKL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqAIJMKL"
PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJMKL(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJMKL(A,MATSEQAIJMKL,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


