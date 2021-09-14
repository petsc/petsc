
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
  /* NOTE: MatGetColumnNorms() could simply be a macro that calls MatGetColumnReductions().
   * I've kept this as a function because it allows slightly more in the way of error checking,
   * erroring out if MatGetColumnNorms() is not called with a valid NormType. */
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_1 || type == NORM_FROBENIUS || type == NORM_INFINITY || type == NORM_1_AND_2) {
    ierr = MatGetColumnReductions(A,type,norms);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Unknown NormType");
  PetscFunctionReturn(0);
}

/*@
   MatGetColumnSumsRealPart - Gets the sums of the real part of each column of a sparse or dense matrix.

   Input Parameter:
.  A - the matrix

   Output Parameter:
.  sums - an array as large as the TOTAL number of columns in the matrix

   Level: intermediate

   Notes:
    Each process has ALL the column sums after the call. Because of the way this is computed each process gets all the values,
    if each process wants only some of the values it should extract the ones it wants from the array.

.seealso: MatGetColumnSumsImaginaryPart(), VecSum(), MatGetColumnMeans(), MatGetColumnNorms(), MatGetColumnReductions()

@*/
PetscErrorCode MatGetColumnSumsRealPart(Mat A,PetscReal sums[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetColumnReductions(A,REDUCTION_SUM_REALPART,sums);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatGetColumnSumsImaginaryPart - Gets the sums of the imaginary part of each column of a sparse or dense matrix.

   Input Parameter:
.  A - the matrix

   Output Parameter:
.  sums - an array as large as the TOTAL number of columns in the matrix

   Level: intermediate

   Notes:
    Each process has ALL the column sums after the call. Because of the way this is computed each process gets all the values,
    if each process wants only some of the values it should extract the ones it wants from the array.

.seealso: MatGetColumnSumsRealPart(), VecSum(), MatGetColumnMeans(), MatGetColumnNorms(), MatGetColumnReductions()

@*/
PetscErrorCode MatGetColumnSumsImaginaryPart(Mat A,PetscReal sums[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetColumnReductions(A,REDUCTION_SUM_IMAGINARYPART,sums);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatGetColumnSums - Gets the sums of each column of a sparse or dense matrix.

   Input Parameter:
.  A - the matrix

   Output Parameter:
.  sums - an array as large as the TOTAL number of columns in the matrix

   Level: intermediate

   Notes:
    Each process has ALL the column sums after the call. Because of the way this is computed each process gets all the values,
    if each process wants only some of the values it should extract the ones it wants from the array.

.seealso: VecSum(), MatGetColumnMeans(), MatGetColumnNorms(), MatGetColumnReductions()

@*/
PetscErrorCode MatGetColumnSums(Mat A,PetscScalar sums[])
{
  PetscErrorCode ierr;
#if defined(PETSC_USE_COMPLEX)
  PetscInt       i,n;
  PetscReal      *work;
#endif

  PetscFunctionBegin;

#if !defined(PETSC_USE_COMPLEX)
  ierr = MatGetColumnSumsRealPart(A,sums);CHKERRQ(ierr);
#else
  ierr = MatGetSize(A,NULL,&n);CHKERRQ(ierr);
  ierr = PetscArrayzero(sums,n);CHKERRQ(ierr);
  ierr = PetscCalloc1(n,&work);CHKERRQ(ierr);
  ierr = MatGetColumnSumsRealPart(A,work);CHKERRQ(ierr);
  for (i=0; i<n; i++) sums[i] = work[i];
  ierr = MatGetColumnSumsImaginaryPart(A,work);CHKERRQ(ierr);
  for (i=0; i<n; i++) sums[i] += work[i]*PETSC_i;
  ierr = PetscFree(work);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*@
   MatGetColumnMeansRealPart - Gets the arithmetic means of the real part of each column of a sparse or dense matrix.

   Input Parameter:
.  A - the matrix

   Output Parameter:
.  sums - an array as large as the TOTAL number of columns in the matrix

   Level: intermediate

   Notes:
    Each process has ALL the column means after the call. Because of the way this is computed each process gets all the values,
    if each process wants only some of the values it should extract the ones it wants from the array.

.seealso: MatGetColumnMeansImaginaryPart(), VecSum(), MatGetColumnSums(), MatGetColumnNorms(), MatGetColumnReductions()

@*/
PetscErrorCode MatGetColumnMeansRealPart(Mat A,PetscReal means[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetColumnReductions(A,REDUCTION_MEAN_REALPART,means);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatGetColumnMeansImaginaryPart - Gets the arithmetic means of the imaginary part of each column of a sparse or dense matrix.

   Input Parameter:
.  A - the matrix

   Output Parameter:
.  sums - an array as large as the TOTAL number of columns in the matrix

   Level: intermediate

   Notes:
    Each process has ALL the column means after the call. Because of the way this is computed each process gets all the values,
    if each process wants only some of the values it should extract the ones it wants from the array.

.seealso: MatGetColumnMeansRealPart(), VecSum(), MatGetColumnSums(), MatGetColumnNorms(), MatGetColumnReductions()

@*/
PetscErrorCode MatGetColumnMeansImaginaryPart(Mat A,PetscReal means[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetColumnReductions(A,REDUCTION_MEAN_IMAGINARYPART,means);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatGetColumnMeans - Gets the arithmetic means of each column of a sparse or dense matrix.

   Input Parameter:
.  A - the matrix

   Output Parameter:
.  means - an array as large as the TOTAL number of columns in the matrix

   Level: intermediate

   Notes:
    Each process has ALL the column means after the call. Because of the way this is computed each process gets all the values,
    if each process wants only some of the values it should extract the ones it wants from the array.

.seealso: VecSum(), MatGetColumnSums(), MatGetColumnNorms(), MatGetColumnReductions()

@*/
PetscErrorCode MatGetColumnMeans(Mat A,PetscScalar means[])
{
  PetscErrorCode ierr;
#if defined(PETSC_USE_COMPLEX)
  PetscInt       i,n;
  PetscReal      *work;
#endif

  PetscFunctionBegin;

#if !defined(PETSC_USE_COMPLEX)
  ierr = MatGetColumnMeansRealPart(A,means);CHKERRQ(ierr);
#else
  ierr = MatGetSize(A,NULL,&n);CHKERRQ(ierr);
  ierr = PetscArrayzero(means,n);CHKERRQ(ierr);
  ierr = PetscCalloc1(n,&work);CHKERRQ(ierr);
  ierr = MatGetColumnMeansRealPart(A,work);CHKERRQ(ierr);
  for (i=0; i<n; i++) means[i] = work[i];
  ierr = MatGetColumnMeansImaginaryPart(A,work);CHKERRQ(ierr);
  for (i=0; i<n; i++) means[i] += work[i]*PETSC_i;
  ierr = PetscFree(work);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*@
    MatGetColumnReductions - Gets the reductions of each column of a sparse or dense matrix.

  Input Parameter:
+  A - the matrix
-  type - A constant defined in NormType or ReductionType: NORM_2, NORM_1, NORM_INFINITY, REDUCTION_SUM_REALPART,
          REDUCTION_SUM_IMAGINARYPART, REDUCTION_MEAN_REALPART, REDUCTION_MEAN_IMAGINARYPART

  Output Parameter:
.  reductions - an array as large as the TOTAL number of columns in the matrix

   Level: developer

   Notes:
    Each process has ALL the column reductions after the call. Because of the way this is computed each process gets all the values,
    if each process wants only some of the values it should extract the ones it wants from the array.

  Developer Note:
    This routine is primarily intended as a back-end.
    MatGetColumnNorms(), MatGetColumnSums(), and MatGetColumnMeans() are implemented using this routine.

.seealso: ReductionType, NormType, MatGetColumnNorms(), MatGetColumnSums(), MatGetColumnMeans()

@*/
PetscErrorCode MatGetColumnReductions(Mat A,PetscInt type,PetscReal reductions[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  if (A->ops->getcolumnreductions) {
    ierr = (*A->ops->getcolumnreductions)(A,type,reductions);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not coded for this matrix type");
  PetscFunctionReturn(0);
}
