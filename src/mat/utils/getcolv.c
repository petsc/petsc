
#include <petsc-private/matimpl.h>  /*I   "petscmat.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnVector"
/*@
   MatGetColumnVector - Gets the values from a given column of a matrix.

   Not Collective

   Input Parameters:
+  A - the matrix
.  yy - the vector
-  c - the column requested (in global numbering)

   Level: advanced

   Notes:
   Each processor for which this is called gets the values for its rows.

   Since PETSc matrices are usually stored in compressed row format, this routine
   will generally be slow.

   The vector must have the same parallel row layout as the matrix.

   Contributed by: Denis Vanderstraeten

.keywords: matrix, column, get 

.seealso: MatGetRow(), MatGetDiagonal()

@*/
PetscErrorCode  MatGetColumnVector(Mat A,Vec yy,PetscInt col)
{
  PetscScalar        *y;
  const PetscScalar  *v;
  PetscErrorCode     ierr;
  PetscInt           i,j,nz,N,Rs,Re,rs,re;
  const PetscInt     *idx;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1); 
  PetscValidHeaderSpecific(yy,VEC_CLASSID,2); 
  if (col < 0)  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Requested negative column: %D",col);
  ierr = MatGetSize(A,PETSC_NULL,&N);CHKERRQ(ierr);
  if (col >= N)  SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Requested column %D larger than number columns in matrix %D",col,N);
  ierr = MatGetOwnershipRange(A,&Rs,&Re);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(yy,&rs,&re);CHKERRQ(ierr);
  if (Rs != rs || Re != re) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Matrix %D %D does not have same ownership range (size) as vector %D %D",Rs,Re,rs,re);

  if (A->ops->getcolumnvector) {
    ierr = (*A->ops->getcolumnvector)(A,yy,col);CHKERRQ(ierr);
  } else {
    ierr = VecSet(yy,0.0);CHKERRQ(ierr);
    ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
    
    for (i=Rs; i<Re; i++) {
      ierr = MatGetRow(A,i,&nz,&idx,&v);CHKERRQ(ierr);
      if (nz && idx[0] <= col) {
	/*
          Should use faster search here 
	*/
	for (j=0; j<nz; j++) {
	  if (idx[j] >= col) {
	    if (idx[j] == col) y[i-rs] = v[j];
	    break;
	  }
	}
      }
      ierr = MatRestoreRow(A,i,&nz,&idx,&v);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);  
  }
  PetscFunctionReturn(0);
}




#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnNorms"
/*@
    MatGetColumnNorms - Gets the norms of each column of a sparse or dense matrix.

  Input Parameter:
+  A - the matrix
-  type - NORM_2, NORM_1 or NORM_INFINITY

  Output Parameter:
.  norms - an array as large as the TOTAL number of columns in the matrix

   Level: intermediate

   Notes: Each process has ALL the column norms after the call. Because of the way this is computed each process gets all the values,
    if each process wants only some of the values it should extract the ones it wants from the array.

.seealso: MatGetColumns()

@*/
PetscErrorCode MatGetColumnNorms(Mat A,NormType type,PetscReal *norms)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  if (A->ops->getcolumnnorms) {
    ierr = (*A->ops->getcolumnnorms)(A,type,norms);CHKERRQ(ierr);
  } else SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"Not coded for this matrix type");
  PetscFunctionReturn(0);
}
  
