#define PETSCMAT_DLL

/*
    Creates hypre ijvector from PETSc vector
*/

#include "src/mat/matimpl.h"          /*I "petscvec.h" I*/
EXTERN_C_BEGIN
#include "HYPRE.h"
#include "IJ_mv.h"
EXTERN_C_END

PetscErrorCode MatHYPRE_IJMatrixCreate(Mat v,HYPRE_IJMatrix *ij)
{
  PetscErrorCode ierr;
  int rstart,rend,cstart,cend;
  
  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(v,&rstart,&rend);CHKERRQ(ierr);
  ierr = PetscMapGetLocalRange(v->cmap,&cstart,&cend);CHKERRQ(ierr);
  ierr = HYPRE_IJMatrixCreate(v->comm,rstart,rend-1,cstart,cend-1,ij);CHKERRQ(ierr);
  ierr = HYPRE_IJMatrixSetObjectType(*ij,HYPRE_PARCSR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
      Currently this only works with the MPIAIJ PETSc matrices to make
the conversion efficient
*/
/* #include "src/mat/impls/aij/mpi/mpiaij.h" */
/*  Mat_MPIAIJ  *aij = (Mat_MPIAIJ *)v->data; */

PetscErrorCode MatHYPRE_IJMatrixCopy(Mat v,HYPRE_IJMatrix ij)
{
  PetscErrorCode    ierr;
  int               i,rstart,rend,ncols;
  const PetscScalar *values;
  const int         *cols;

  PetscFunctionBegin;
  ierr = HYPRE_IJMatrixInitialize(ij);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(v,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(v,i,&ncols,&cols,&values);CHKERRQ(ierr);
    ierr = HYPRE_IJMatrixSetValues(ij,1,&ncols,&i,cols,values);CHKERRQ(ierr);
    ierr = MatRestoreRow(v,i,&ncols,&cols,&values);CHKERRQ(ierr);
  }
  ierr = HYPRE_IJMatrixAssemble(ij);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
