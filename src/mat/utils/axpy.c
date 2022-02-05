
#include <petsc/private/matimpl.h>  /*I   "petscmat.h"  I*/

static PetscErrorCode MatTransposeAXPY_Private(Mat Y,PetscScalar a,Mat X,MatStructure str,Mat T)
{
  PetscErrorCode ierr,(*f)(Mat,Mat*);
  Mat            A,F;

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)T,"MatTransposeGetMat_C",&f);CHKERRQ(ierr);
  if (f) {
    ierr = MatTransposeGetMat(T,&A);CHKERRQ(ierr);
    if (T == X) {
      ierr = PetscInfo(NULL,"Explicitly transposing X of type MATTRANSPOSEMAT to perform MatAXPY()\n");CHKERRQ(ierr);
      ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&F);CHKERRQ(ierr);
      A = Y;
    } else {
      ierr = PetscInfo(NULL,"Transposing X because Y of type MATTRANSPOSEMAT to perform MatAXPY()\n");CHKERRQ(ierr);
      ierr = MatTranspose(X,MAT_INITIAL_MATRIX,&F);CHKERRQ(ierr);
    }
  } else {
    ierr = MatHermitianTransposeGetMat(T,&A);CHKERRQ(ierr);
    if (T == X) {
      ierr = PetscInfo(NULL,"Explicitly Hermitian transposing X of type MATTRANSPOSEMAT to perform MatAXPY()\n");CHKERRQ(ierr);
      ierr = MatHermitianTranspose(A,MAT_INITIAL_MATRIX,&F);CHKERRQ(ierr);
      A = Y;
    } else {
      ierr = PetscInfo(NULL,"Hermitian transposing X because Y of type MATTRANSPOSEMAT to perform MatAXPY()\n");CHKERRQ(ierr);
      ierr = MatHermitianTranspose(X,MAT_INITIAL_MATRIX,&F);CHKERRQ(ierr);
    }
  }
  ierr = MatAXPY(A,a,F,str);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatAXPY - Computes Y = a*X + Y.

   Logically  Collective on Mat

   Input Parameters:
+  a - the scalar multiplier
.  X - the first matrix
.  Y - the second matrix
-  str - either SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN, UNKNOWN_NONZERO_PATTERN, or SUBSET_NONZERO_PATTERN (nonzeros of X is a subset of Y's)

   Notes: No operation is performed when a is zero.

   Level: intermediate

.seealso: MatAYPX()
 @*/
PetscErrorCode MatAXPY(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscErrorCode ierr;
  PetscInt       M1,M2,N1,N2;
  PetscInt       m1,m2,n1,n2;
  MatType        t1,t2;
  PetscBool      sametype,transpose;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,MAT_CLASSID,1);
  PetscValidLogicalCollectiveScalar(Y,a,2);
  PetscValidHeaderSpecific(X,MAT_CLASSID,3);
  PetscCheckSameComm(Y,1,X,3);
  ierr = MatGetSize(X,&M1,&N1);CHKERRQ(ierr);
  ierr = MatGetSize(Y,&M2,&N2);CHKERRQ(ierr);
  ierr = MatGetLocalSize(X,&m1,&n1);CHKERRQ(ierr);
  ierr = MatGetLocalSize(Y,&m2,&n2);CHKERRQ(ierr);
  if (M1 != M2 || N1 != N2) SETERRQ4(PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_SIZ,"Non conforming matrix add: global sizes X: %" PetscInt_FMT " x %" PetscInt_FMT ", Y: %" PetscInt_FMT " x %" PetscInt_FMT,M1,N1,M2,N2);
  if (m1 != m2 || n1 != n2) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Non conforming matrix add: local sizes X: %" PetscInt_FMT " x %" PetscInt_FMT ", Y: %" PetscInt_FMT " x %" PetscInt_FMT,m1,n1,m2,n2);
  if (!Y->assembled) SETERRQ(PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix (Y)");
  if (!X->assembled) SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix (X)");
  if (a == (PetscScalar)0.0) PetscFunctionReturn(0);
  if (Y == X) {
    ierr = MatScale(Y,1.0 + a);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = MatGetType(X,&t1);CHKERRQ(ierr);
  ierr = MatGetType(Y,&t2);CHKERRQ(ierr);
  ierr = PetscStrcmp(t1,t2,&sametype);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(MAT_AXPY,Y,0,0,0);CHKERRQ(ierr);
  if (Y->ops->axpy && (sametype || X->ops->axpy == Y->ops->axpy)) {
    ierr = (*Y->ops->axpy)(Y,a,X,str);CHKERRQ(ierr);
  } else {
    ierr = PetscStrcmp(t1,MATTRANSPOSEMAT,&transpose);CHKERRQ(ierr);
    if (transpose) {
      ierr = MatTransposeAXPY_Private(Y,a,X,str,X);CHKERRQ(ierr);
    } else {
      ierr = PetscStrcmp(t2,MATTRANSPOSEMAT,&transpose);CHKERRQ(ierr);
      if (transpose) {
        ierr = MatTransposeAXPY_Private(Y,a,X,str,Y);CHKERRQ(ierr);
      } else {
        ierr = MatAXPY_Basic(Y,a,X,str);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscLogEventEnd(MAT_AXPY,Y,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_Basic_Preallocate(Mat Y, Mat X, Mat *B)
{
  PetscErrorCode ierr;
  PetscErrorCode (*preall)(Mat,Mat,Mat*) = NULL;

  PetscFunctionBegin;
  /* look for any available faster alternative to the general preallocator */
  ierr = PetscObjectQueryFunction((PetscObject)Y,"MatAXPYGetPreallocation_C",&preall);CHKERRQ(ierr);
  if (preall) {
    ierr = (*preall)(Y,X,B);CHKERRQ(ierr);
  } else { /* Use MatPrellocator, assumes same row-col distribution */
    Mat      preallocator;
    PetscInt r,rstart,rend;
    PetscInt m,n,M,N;

    ierr = MatGetRowUpperTriangular(Y);CHKERRQ(ierr);
    ierr = MatGetRowUpperTriangular(X);CHKERRQ(ierr);
    ierr = MatGetSize(Y,&M,&N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(Y,&m,&n);CHKERRQ(ierr);
    ierr = MatCreate(PetscObjectComm((PetscObject)Y),&preallocator);CHKERRQ(ierr);
    ierr = MatSetType(preallocator,MATPREALLOCATOR);CHKERRQ(ierr);
    ierr = MatSetLayouts(preallocator,Y->rmap,Y->cmap);CHKERRQ(ierr);
    ierr = MatSetUp(preallocator);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(preallocator,&rstart,&rend);CHKERRQ(ierr);
    for (r = rstart; r < rend; ++r) {
      PetscInt          ncols;
      const PetscInt    *row;
      const PetscScalar *vals;

      ierr = MatGetRow(Y,r,&ncols,&row,&vals);CHKERRQ(ierr);
      ierr = MatSetValues(preallocator,1,&r,ncols,row,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(Y,r,&ncols,&row,&vals);CHKERRQ(ierr);
      ierr = MatGetRow(X,r,&ncols,&row,&vals);CHKERRQ(ierr);
      ierr = MatSetValues(preallocator,1,&r,ncols,row,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(X,r,&ncols,&row,&vals);CHKERRQ(ierr);
    }
    ierr = MatSetOption(preallocator,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(preallocator,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(preallocator,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatRestoreRowUpperTriangular(Y);CHKERRQ(ierr);
    ierr = MatRestoreRowUpperTriangular(X);CHKERRQ(ierr);

    ierr = MatCreate(PetscObjectComm((PetscObject)Y),B);CHKERRQ(ierr);
    ierr = MatSetType(*B,((PetscObject)Y)->type_name);CHKERRQ(ierr);
    ierr = MatSetLayouts(*B,Y->rmap,Y->cmap);CHKERRQ(ierr);
    ierr = MatPreallocatorPreallocate(preallocator,PETSC_FALSE,*B);CHKERRQ(ierr);
    ierr = MatDestroy(&preallocator);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_Basic(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscErrorCode ierr;
  PetscBool      isshell,isdense,isnest;

  PetscFunctionBegin;
  ierr = MatIsShell(Y,&isshell);CHKERRQ(ierr);
  if (isshell) { /* MatShell has special support for AXPY */
    PetscErrorCode (*f)(Mat,PetscScalar,Mat,MatStructure);

    ierr = MatGetOperation(Y,MATOP_AXPY,(void (**)(void))&f);CHKERRQ(ierr);
    if (f) {
      ierr = (*f)(Y,a,X,str);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }
  /* no need to preallocate if Y is dense */
  ierr = PetscObjectBaseTypeCompareAny((PetscObject)Y,&isdense,MATSEQDENSE,MATMPIDENSE,"");CHKERRQ(ierr);
  if (isdense) {
    ierr = PetscObjectTypeCompare((PetscObject)X,MATNEST,&isnest);CHKERRQ(ierr);
    if (isnest) {
      ierr = MatAXPY_Dense_Nest(Y,a,X);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    if (str == DIFFERENT_NONZERO_PATTERN || str == UNKNOWN_NONZERO_PATTERN) str = SUBSET_NONZERO_PATTERN;
  }
  if (str != DIFFERENT_NONZERO_PATTERN && str != UNKNOWN_NONZERO_PATTERN) {
    PetscInt          i,start,end,j,ncols,m,n;
    const PetscInt    *row;
    PetscScalar       *val;
    const PetscScalar *vals;

    ierr = MatGetSize(X,&m,&n);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(X,&start,&end);CHKERRQ(ierr);
    ierr = MatGetRowUpperTriangular(X);CHKERRQ(ierr);
    if (a == 1.0) {
      for (i = start; i < end; i++) {
        ierr = MatGetRow(X,i,&ncols,&row,&vals);CHKERRQ(ierr);
        ierr = MatSetValues(Y,1,&i,ncols,row,vals,ADD_VALUES);CHKERRQ(ierr);
        ierr = MatRestoreRow(X,i,&ncols,&row,&vals);CHKERRQ(ierr);
      }
    } else {
      PetscInt vs = 100;
      /* realloc if needed, as this function may be used in parallel */
      ierr = PetscMalloc1(vs,&val);CHKERRQ(ierr);
      for (i=start; i<end; i++) {
        ierr = MatGetRow(X,i,&ncols,&row,&vals);CHKERRQ(ierr);
        if (vs < ncols) {
          vs   = PetscMin(2*ncols,n);
          ierr = PetscRealloc(vs*sizeof(*val),&val);CHKERRQ(ierr);
        }
        for (j=0; j<ncols; j++) val[j] = a*vals[j];
        ierr = MatSetValues(Y,1,&i,ncols,row,val,ADD_VALUES);CHKERRQ(ierr);
        ierr = MatRestoreRow(X,i,&ncols,&row,&vals);CHKERRQ(ierr);
      }
      ierr = PetscFree(val);CHKERRQ(ierr);
    }
    ierr = MatRestoreRowUpperTriangular(X);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else {
    Mat B;

    ierr = MatAXPY_Basic_Preallocate(Y,X,&B);CHKERRQ(ierr);
    ierr = MatAXPY_BasicWithPreallocation(B,Y,a,X,str);CHKERRQ(ierr);
    ierr = MatHeaderMerge(Y,&B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_BasicWithPreallocation(Mat B,Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscInt          i,start,end,j,ncols,m,n;
  PetscErrorCode    ierr;
  const PetscInt    *row;
  PetscScalar       *val;
  const PetscScalar *vals;

  PetscFunctionBegin;
  ierr = MatGetSize(X,&m,&n);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(X,&start,&end);CHKERRQ(ierr);
  ierr = MatGetRowUpperTriangular(Y);CHKERRQ(ierr);
  ierr = MatGetRowUpperTriangular(X);CHKERRQ(ierr);
  if (a == 1.0) {
    for (i = start; i < end; i++) {
      ierr = MatGetRow(Y,i,&ncols,&row,&vals);CHKERRQ(ierr);
      ierr = MatSetValues(B,1,&i,ncols,row,vals,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(Y,i,&ncols,&row,&vals);CHKERRQ(ierr);

      ierr = MatGetRow(X,i,&ncols,&row,&vals);CHKERRQ(ierr);
      ierr = MatSetValues(B,1,&i,ncols,row,vals,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(X,i,&ncols,&row,&vals);CHKERRQ(ierr);
    }
  } else {
    PetscInt vs = 100;
    /* realloc if needed, as this function may be used in parallel */
    ierr = PetscMalloc1(vs,&val);CHKERRQ(ierr);
    for (i=start; i<end; i++) {
      ierr = MatGetRow(Y,i,&ncols,&row,&vals);CHKERRQ(ierr);
      ierr = MatSetValues(B,1,&i,ncols,row,vals,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(Y,i,&ncols,&row,&vals);CHKERRQ(ierr);

      ierr = MatGetRow(X,i,&ncols,&row,&vals);CHKERRQ(ierr);
      if (vs < ncols) {
        vs   = PetscMin(2*ncols,n);
        ierr = PetscRealloc(vs*sizeof(*val),&val);CHKERRQ(ierr);
      }
      for (j=0; j<ncols; j++) val[j] = a*vals[j];
      ierr = MatSetValues(B,1,&i,ncols,row,val,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(X,i,&ncols,&row,&vals);CHKERRQ(ierr);
    }
    ierr = PetscFree(val);CHKERRQ(ierr);
  }
  ierr = MatRestoreRowUpperTriangular(Y);CHKERRQ(ierr);
  ierr = MatRestoreRowUpperTriangular(X);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatShift - Computes Y =  Y + a I, where a is a PetscScalar and I is the identity matrix.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  Y - the matrices
-  a - the PetscScalar

   Level: intermediate

   Notes:
    If the matrix Y is missing some diagonal entries this routine can be very slow. To make it fast one should initially
   fill the matrix so that all diagonal entries have a value (with a value of zero for those locations that would not have an
   entry). No operation is performed when a is zero.

   To form Y = Y + diag(V) use MatDiagonalSet()

.seealso: MatDiagonalSet(), MatScale(), MatDiagonalScale()
 @*/
PetscErrorCode  MatShift(Mat Y,PetscScalar a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,MAT_CLASSID,1);
  if (!Y->assembled) SETERRQ(PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (Y->factortype) SETERRQ(PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(Y,1);
  if (a == 0.0) PetscFunctionReturn(0);

  if (Y->ops->shift) {
    ierr = (*Y->ops->shift)(Y,a);CHKERRQ(ierr);
  } else {
    ierr = MatShift_Basic(Y,a);CHKERRQ(ierr);
  }

  ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatDiagonalSet_Default(Mat Y,Vec D,InsertMode is)
{
  PetscErrorCode    ierr;
  PetscInt          i,start,end;
  const PetscScalar *v;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(Y,&start,&end);CHKERRQ(ierr);
  ierr = VecGetArrayRead(D,&v);CHKERRQ(ierr);
  for (i=start; i<end; i++) {
    ierr = MatSetValues(Y,1,&i,1,&i,v+i-start,is);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(D,&v);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatDiagonalSet - Computes Y = Y + D, where D is a diagonal matrix
   that is represented as a vector. Or Y[i,i] = D[i] if InsertMode is
   INSERT_VALUES.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  Y - the input matrix
.  D - the diagonal matrix, represented as a vector
-  i - INSERT_VALUES or ADD_VALUES

   Notes:
    If the matrix Y is missing some diagonal entries this routine can be very slow. To make it fast one should initially
   fill the matrix so that all diagonal entries have a value (with a value of zero for those locations that would not have an
   entry).

   Level: intermediate

.seealso: MatShift(), MatScale(), MatDiagonalScale()
@*/
PetscErrorCode  MatDiagonalSet(Mat Y,Vec D,InsertMode is)
{
  PetscErrorCode ierr;
  PetscInt       matlocal,veclocal;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,MAT_CLASSID,1);
  PetscValidHeaderSpecific(D,VEC_CLASSID,2);
  ierr = MatGetLocalSize(Y,&matlocal,NULL);CHKERRQ(ierr);
  ierr = VecGetLocalSize(D,&veclocal);CHKERRQ(ierr);
  if (matlocal != veclocal) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number local rows of matrix %" PetscInt_FMT " does not match that of vector for diagonal %" PetscInt_FMT,matlocal,veclocal);
  if (Y->ops->diagonalset) {
    ierr = (*Y->ops->diagonalset)(Y,D,is);CHKERRQ(ierr);
  } else {
    ierr = MatDiagonalSet_Default(Y,D,is);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatAYPX - Computes Y = a*Y + X.

   Logically on Mat

   Input Parameters:
+  a - the PetscScalar multiplier
.  Y - the first matrix
.  X - the second matrix
-  str - either SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN, UNKNOWN_NONZERO_PATTERN, or SUBSET_NONZERO_PATTERN (nonzeros of X is a subset of Y's)

   Level: intermediate

.seealso: MatAXPY()
 @*/
PetscErrorCode  MatAYPX(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatScale(Y,a);CHKERRQ(ierr);
  ierr = MatAXPY(Y,1.0,X,str);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    MatComputeOperator - Computes the explicit matrix

    Collective on Mat

    Input Parameters:
+   inmat - the matrix
-   mattype - the matrix type for the explicit operator

    Output Parameter:
.   mat - the explicit  operator

    Notes:
    This computation is done by applying the operators to columns of the identity matrix.
    This routine is costly in general, and is recommended for use only with relatively small systems.
    Currently, this routine uses a dense matrix format if mattype == NULL.

    Level: advanced

@*/
PetscErrorCode  MatComputeOperator(Mat inmat,MatType mattype,Mat *mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(inmat,MAT_CLASSID,1);
  PetscValidPointer(mat,3);
  ierr = MatConvert_Shell(inmat,mattype ? mattype : MATDENSE,MAT_INITIAL_MATRIX,mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    MatComputeOperatorTranspose - Computes the explicit matrix representation of
        a give matrix that can apply MatMultTranspose()

    Collective on Mat

    Input Parameters:
+   inmat - the matrix
-   mattype - the matrix type for the explicit operator

    Output Parameter:
.   mat - the explicit  operator transposed

    Notes:
    This computation is done by applying the transpose of the operator to columns of the identity matrix.
    This routine is costly in general, and is recommended for use only with relatively small systems.
    Currently, this routine uses a dense matrix format if mattype == NULL.

    Level: advanced

@*/
PetscErrorCode  MatComputeOperatorTranspose(Mat inmat,MatType mattype,Mat *mat)
{
  Mat            A;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(inmat,MAT_CLASSID,1);
  PetscValidPointer(mat,3);
  ierr = MatCreateTranspose(inmat,&A);CHKERRQ(ierr);
  ierr = MatConvert_Shell(A,mattype ? mattype : MATDENSE,MAT_INITIAL_MATRIX,mat);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  MatChop - Set all values in the matrix less than the tolerance to zero

  Input Parameters:
+ A   - The matrix
- tol - The zero tolerance

  Output Parameters:
. A - The chopped matrix

  Level: intermediate

.seealso: MatCreate(), MatZeroEntries()
 @*/
PetscErrorCode MatChop(Mat A, PetscReal tol)
{
  Mat            a;
  PetscScalar    *newVals;
  PetscInt       *newCols, rStart, rEnd, numRows, maxRows, r, colMax = 0;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectBaseTypeCompareAny((PetscObject)A, &flg, MATSEQDENSE, MATMPIDENSE, "");CHKERRQ(ierr);
  if (flg) {
    ierr = MatDenseGetLocalMatrix(A, &a);CHKERRQ(ierr);
    ierr = MatDenseGetLDA(a, &r);CHKERRQ(ierr);
    ierr = MatGetSize(a, &rStart, &rEnd);CHKERRQ(ierr);
    ierr = MatDenseGetArray(a, &newVals);CHKERRQ(ierr);
    for (; colMax < rEnd; ++colMax) {
      for (maxRows = 0; maxRows < rStart; ++maxRows) {
        newVals[maxRows + colMax * r] = PetscAbsScalar(newVals[maxRows + colMax * r]) < tol ? 0.0 : newVals[maxRows + colMax * r];
      }
    }
    ierr = MatDenseRestoreArray(a, &newVals);CHKERRQ(ierr);
  } else {
    ierr = MatGetOwnershipRange(A, &rStart, &rEnd);CHKERRQ(ierr);
    ierr = MatGetRowUpperTriangular(A);CHKERRQ(ierr);
    for (r = rStart; r < rEnd; ++r) {
      PetscInt ncols;

      ierr   = MatGetRow(A, r, &ncols, NULL, NULL);CHKERRQ(ierr);
      colMax = PetscMax(colMax, ncols);CHKERRQ(ierr);
      ierr   = MatRestoreRow(A, r, &ncols, NULL, NULL);CHKERRQ(ierr);
    }
    numRows = rEnd - rStart;
    ierr    = MPIU_Allreduce(&numRows, &maxRows, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
    ierr    = PetscMalloc2(colMax, &newCols, colMax, &newVals);CHKERRQ(ierr);
    ierr    = MatGetOption(A, MAT_NO_OFF_PROC_ENTRIES, &flg);CHKERRQ(ierr); /* cache user-defined value */
    ierr    = MatSetOption(A, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE);CHKERRQ(ierr);
    /* short-circuit code in MatAssemblyBegin() and MatAssemblyEnd()             */
    /* that are potentially called many times depending on the distribution of A */
    for (r = rStart; r < rStart+maxRows; ++r) {
      const PetscScalar *vals;
      const PetscInt    *cols;
      PetscInt           ncols, newcols, c;

      if (r < rEnd) {
        ierr = MatGetRow(A, r, &ncols, &cols, &vals);CHKERRQ(ierr);
        for (c = 0; c < ncols; ++c) {
          newCols[c] = cols[c];
          newVals[c] = PetscAbsScalar(vals[c]) < tol ? 0.0 : vals[c];
        }
        newcols = ncols;
        ierr = MatRestoreRow(A, r, &ncols, &cols, &vals);CHKERRQ(ierr);
        ierr = MatSetValues(A, 1, &r, newcols, newCols, newVals, INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    ierr = MatRestoreRowUpperTriangular(A);CHKERRQ(ierr);
    ierr = PetscFree2(newCols, newVals);CHKERRQ(ierr);
    ierr = MatSetOption(A, MAT_NO_OFF_PROC_ENTRIES, flg);CHKERRQ(ierr); /* reset option to its user-defined value */
  }
  PetscFunctionReturn(0);
}
