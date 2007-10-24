#define PETSCMAT_DLL

/*
    Creates hypre ijmatrix from PETSc matrix
*/

#include "include/private/matimpl.h"          /*I "petscmat.h" I*/
EXTERN_C_BEGIN
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatHYPRE_IJMatrixPreallocate"
PetscErrorCode MatHYPRE_IJMatrixPreallocate(Mat A_d, Mat A_o,HYPRE_IJMatrix ij)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscInt       n_d,*ia_d,n_o,*ia_o;
  PetscTruth     done_d=PETSC_FALSE,done_o=PETSC_FALSE;
  PetscInt       *nnz_d=PETSC_NULL,*nnz_o=PETSC_NULL;
  
  PetscFunctionBegin;
  if (A_d) { /* determine number of nonzero entries in local diagonal part */
    ierr = MatGetRowIJ(A_d,0,PETSC_FALSE,PETSC_FALSE,&n_d,&ia_d,PETSC_NULL,&done_d);CHKERRQ(ierr);
    CHKMEMQ;
    if (done_d) {
      ierr = PetscMalloc(n_d*sizeof(PetscInt),&nnz_d);CHKERRQ(ierr);
      for (i=0; i<n_d; i++) {
        nnz_d[i] = ia_d[i+1] - ia_d[i];
      }
    }
    CHKMEMQ;
    ierr = MatRestoreRowIJ(A_d,0,PETSC_FALSE,PETSC_FALSE,&n_d,&ia_d,PETSC_NULL,&done_d);CHKERRQ(ierr);
  }
  if (A_o) { /* determine number of nonzero entries in local off-diagonal part */
    ierr = MatGetRowIJ(A_o,0,PETSC_FALSE,PETSC_FALSE,&n_o,&ia_o,PETSC_NULL,&done_o);CHKERRQ(ierr);
    if (done_o) {
      ierr = PetscMalloc(n_o*sizeof(PetscInt),&nnz_o);CHKERRQ(ierr);
      for (i=0; i<n_o; i++) {
        nnz_o[i] = ia_o[i+1] - ia_o[i];
      }
    }
    ierr = MatRestoreRowIJ(A_o,0,PETSC_FALSE,PETSC_FALSE,&n_o,&ia_o,PETSC_NULL,&done_o);CHKERRQ(ierr);
  }
  if (done_d) {    /* set number of nonzeros in HYPRE IJ matrix */
    if (!done_o) { /* only diagonal part */
      ierr = HYPRE_IJMatrixSetRowSizes(ij,nnz_d);CHKERRQ(ierr);
    } else {       /* diagonal and off-diagonal part */
      ierr = HYPRE_IJMatrixSetDiagOffdSizes(ij,nnz_d,nnz_o);CHKERRQ(ierr);
    }
    ierr = PetscFree(nnz_d);CHKERRQ(ierr);
    ierr = PetscFree(nnz_o);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatHYPRE_IJMatrixCreate"
PetscErrorCode MatHYPRE_IJMatrixCreate(Mat A,HYPRE_IJMatrix *ij)
{
  PetscErrorCode ierr;
  int            rstart,rend,cstart,cend;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  PetscValidPointer(ij,2);
  ierr = MatPreallocated(A);CHKERRQ(ierr);
  rstart = A->rmap.rstart;
  rend   = A->rmap.rend;
  cstart = A->cmap.rstart;
  cend   = A->cmap.rend;
  ierr = HYPRE_IJMatrixCreate(((PetscObject)A)->comm,rstart,rend-1,cstart,cend-1,ij);CHKERRQ(ierr);
  ierr = HYPRE_IJMatrixSetObjectType(*ij,HYPRE_PARCSR);CHKERRQ(ierr);
  {
    PetscTruth  same;
    Mat         A_d,A_o;
    PetscInt    *colmap;
    ierr = PetscTypeCompare((PetscObject)A,MATMPIAIJ,&same);CHKERRQ(ierr);
    if (same) {
      ierr = MatMPIAIJGetSeqAIJ(A,&A_d,&A_o,&colmap);CHKERRQ(ierr);
      ierr = MatHYPRE_IJMatrixPreallocate(A_d,A_o,*ij);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = PetscTypeCompare((PetscObject)A,MATMPIBAIJ,&same);CHKERRQ(ierr);
    if (same) {
      ierr = MatMPIBAIJGetSeqBAIJ(A,&A_d,&A_o,&colmap);CHKERRQ(ierr);
      ierr = MatHYPRE_IJMatrixPreallocate(A_d,A_o,*ij);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&same);CHKERRQ(ierr);
    if (same) {
      ierr = MatHYPRE_IJMatrixPreallocate(A,PETSC_NULL,*ij);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = PetscTypeCompare((PetscObject)A,MATSEQBAIJ,&same);CHKERRQ(ierr);
    if (same) {
      ierr = MatHYPRE_IJMatrixPreallocate(A,PETSC_NULL,*ij);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}

/*
  Currently this only works with the MPIAIJ PETSc matrices to make
  the conversion efficient
*/

#undef __FUNCT__
#define __FUNCT__ "MatHYPRE_IJMatrixCopy"
PetscErrorCode MatHYPRE_IJMatrixCopy(Mat A,HYPRE_IJMatrix ij)
{
  PetscErrorCode    ierr;
  int               i,rstart,rend,ncols;
  const PetscScalar *values;
  const int         *cols;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(MAT_Convert,A,0,0,0);CHKERRQ(ierr);
  ierr = HYPRE_IJMatrixInitialize(ij);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(A,i,&ncols,&cols,&values);CHKERRQ(ierr);
    ierr = HYPRE_IJMatrixSetValues(ij,1,&ncols,&i,cols,values);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&ncols,&cols,&values);CHKERRQ(ierr);
  }
  ierr = HYPRE_IJMatrixAssemble(ij);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Convert,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
