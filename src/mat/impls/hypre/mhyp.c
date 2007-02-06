#define PETSCMAT_DLL

/*
    Creates hypre ijmatrix from PETSc matrix
*/

#include "include/private/matimpl.h"          /*I "petscmat.h" I*/
EXTERN_C_BEGIN
#include "HYPRE.h"
#include "IJ_mv.h"
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatHYPRE_IJMatrixPreallocate"
PetscErrorCode MatHYPRE_IJMatrixPreallocate(Mat A_d, Mat A_o,HYPRE_IJMatrix ij)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscInt       bs,n_d,*ia_d,*ja_d,n_o,*ia_o,*ja_o;
  PetscTruth     done_d=PETSC_FALSE,done_o=PETSC_FALSE;
  PetscInt       nz_d,nz_o,*nnz_d=PETSC_NULL,*nnz_o=PETSC_NULL;
  
  PetscFunctionBegin;
  if (A_d) { /* determine number of nonzero entries in local diagonal part */
    ierr = MatGetBlockSize(A_d,&bs);CHKERRQ(ierr);
    ierr = MatGetRowIJ(A_d,0,PETSC_FALSE,&n_d,&ia_d,&ja_d,&done_d);CHKERRQ(ierr);
    if (done_d) {
      ierr = PetscMalloc(n_d*bs*sizeof(PetscInt),&nnz_d);CHKERRQ(ierr);
      for (i=0; i<n_d; i++) {
	nz_d = (ia_d[i+1] - ia_d[i])*bs;
	for (j=0; j<bs; j++) nnz_d[i+j*bs] = nz_d;
      }
    }
    ierr = MatRestoreRowIJ(A_d,0,PETSC_FALSE,&n_d,&ia_d,&ja_d,&done_d);CHKERRQ(ierr);
  }
  if (A_o) { /* determine number of nonzero entries in local off-diagonal part */
    ierr = MatGetBlockSize(A_o,&bs);CHKERRQ(ierr);
    ierr = MatGetRowIJ(A_o,0,PETSC_FALSE,&n_o,&ia_o,&ja_o,&done_o);CHKERRQ(ierr);
    if (done_o) {
      ierr = PetscMalloc(n_o*bs*sizeof(PetscInt),&nnz_o);CHKERRQ(ierr);
      for (i=0; i<n_o; i++) {
	nz_o = (ia_o[i+1] - ia_o[i])*bs;
	for (j=0; j<bs; j++) nnz_o[i+j*bs] = nz_o;
      }
    }
    ierr = MatRestoreRowIJ(A_o,0,PETSC_FALSE,&n_o,&ia_o,&ja_o,&done_o);CHKERRQ(ierr);
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
  int rstart,rend,cstart,cend;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  PetscValidPointer(ij,2);
  ierr = MatPreallocated(A);CHKERRQ(ierr);
  rstart = A->rmap.rstart;
  rend   = A->rmap.rend;
  cstart = A->cmap.rstart;
  cend   = A->cmap.rend;
  ierr = HYPRE_IJMatrixCreate(A->comm,rstart,rend-1,cstart,cend-1,ij);CHKERRQ(ierr);
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
  ierr = HYPRE_IJMatrixInitialize(ij);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(A,i,&ncols,&cols,&values);CHKERRQ(ierr);
    ierr = HYPRE_IJMatrixSetValues(ij,1,&ncols,&i,cols,values);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&ncols,&cols,&values);CHKERRQ(ierr);
  }
  ierr = HYPRE_IJMatrixAssemble(ij);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
