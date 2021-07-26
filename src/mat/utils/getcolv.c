
#include <petsc/private/matimpl.h>  /*I   "petscmat.h"  I*/

/*@
   MatGetColumnVector - Gets the values from a given column of a matrix.

   Not Collective

   Input Parameters:
+  A - the matrix
.  yy - the vector
-  col - the column requested (in global numbering)

   Level: advanced

   Notes:
   If a Mat type does not implement the operation, each processor for which this is called
   gets the values for its rows using MatGetRow().

   The vector must have the same parallel row layout as the matrix.

   Contributed by: Denis Vanderstraeten

.seealso: MatGetRow(), MatGetDiagonal(), MatMult()

@*/
PetscErrorCode  MatGetColumnVector(Mat A,Vec yy,PetscInt col)
{
  PetscScalar       *y;
  const PetscScalar *v;
  PetscErrorCode    ierr;
  PetscInt          i,j,nz,N,Rs,Re,rs,re;
  const PetscInt    *idx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(yy,VEC_CLASSID,2);
  PetscValidLogicalCollectiveInt(A,col,3);
  if (col < 0) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_OUTOFRANGE,"Requested negative column: %D",col);
  ierr = MatGetSize(A,NULL,&N);CHKERRQ(ierr);
  if (col >= N) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_OUTOFRANGE,"Requested column %D larger than number columns in matrix %D",col,N);
  ierr = MatGetOwnershipRange(A,&Rs,&Re);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(yy,&rs,&re);CHKERRQ(ierr);
  if (Rs != rs || Re != re) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Matrix %D %D does not have same ownership range (size) as vector %D %D",Rs,Re,rs,re);

  if (A->ops->getcolumnvector) {
    ierr = (*A->ops->getcolumnvector)(A,yy,col);CHKERRQ(ierr);
  } else {
    ierr = VecSet(yy,0.0);CHKERRQ(ierr);
    ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
    /* TODO for general matrices */
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

/*@
    MatGetColumnNorms - Gets the norms of each column of a sparse or dense matrix.

  Input Parameter:
+  A - the matrix
-  type - NORM_2, NORM_1 or NORM_INFINITY

  Output Parameter:
.  norms - an array as large as the TOTAL number of columns in the matrix

   Level: intermediate

   Notes:
    Each process has ALL the column norms after the call. Because of the way this is computed each process gets all the values,
    if each process wants only some of the values it should extract the ones it wants from the array.

.seealso: NormType, MatNorm()

@*/
PetscErrorCode MatGetColumnNorms(Mat A,NormType type,PetscReal norms[])
{
  PetscErrorCode ierr;
  ReductionType reductiontype;

  PetscFunctionBegin;
  switch(type) {
    case NORM_2:
      reductiontype = REDUCTION_NORM_2;
      break;
    case NORM_1:
      reductiontype = REDUCTION_NORM_1;
      break;
    case NORM_FROBENIUS:
      reductiontype = REDUCTION_NORM_FROBENIUS;
      break;
    case NORM_INFINITY:
      reductiontype = REDUCTION_NORM_INFINITY;
      break;
    case NORM_1_AND_2:
      reductiontype = REDUCTION_NORM_1_AND_2;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Unknown NormType");
  }
  ierr = MatGetColumnReductions(A,reductiontype,norms);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    MatGetColumnReductions - Gets the reductions of each column of a sparse or dense matrix.

  Input Parameter:
+  A - the matrix
-  type - NORM_2, NORM_1, NORM_INFINITY, SUM, MEAN

  Output Parameter:
.  reductions - an array as large as the TOTAL number of columns in the matrix

   Level: intermediate

   Notes:
    Each process has ALL the column reductions after the call. Because of the way this is computed each process gets all the values,
    if each process wants only some of the values it should extract the ones it wants from the array.

  Developer Note:
    MatGetColumnNorms() is now implemented using this routine.

.seealso: NormType, MatGetColumnNorms()

@*/
PetscErrorCode MatGetColumnReductions(Mat A,ReductionType type,PetscReal reductions[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  if (A->ops->getcolumnreductions) {
    ierr = (*A->ops->getcolumnreductions)(A,type,reductions);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not coded for this matrix type");
  PetscFunctionReturn(0);
}
