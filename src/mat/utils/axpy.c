
#include <petsc/private/matimpl.h>  /*I   "petscmat.h"  I*/

static PetscErrorCode MatTransposeAXPY_Private(Mat Y,PetscScalar a,Mat X,MatStructure str,Mat T)
{
  Mat            A,F;
  PetscErrorCode (*f)(Mat,Mat*);

  PetscFunctionBegin;
  CHKERRQ(PetscObjectQueryFunction((PetscObject)T,"MatTransposeGetMat_C",&f));
  if (f) {
    CHKERRQ(MatTransposeGetMat(T,&A));
    if (T == X) {
      CHKERRQ(PetscInfo(NULL,"Explicitly transposing X of type MATTRANSPOSEMAT to perform MatAXPY()\n"));
      CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX,&F));
      A = Y;
    } else {
      CHKERRQ(PetscInfo(NULL,"Transposing X because Y of type MATTRANSPOSEMAT to perform MatAXPY()\n"));
      CHKERRQ(MatTranspose(X,MAT_INITIAL_MATRIX,&F));
    }
  } else {
    CHKERRQ(MatHermitianTransposeGetMat(T,&A));
    if (T == X) {
      CHKERRQ(PetscInfo(NULL,"Explicitly Hermitian transposing X of type MATTRANSPOSEMAT to perform MatAXPY()\n"));
      CHKERRQ(MatHermitianTranspose(A,MAT_INITIAL_MATRIX,&F));
      A = Y;
    } else {
      CHKERRQ(PetscInfo(NULL,"Hermitian transposing X because Y of type MATTRANSPOSEMAT to perform MatAXPY()\n"));
      CHKERRQ(MatHermitianTranspose(X,MAT_INITIAL_MATRIX,&F));
    }
  }
  CHKERRQ(MatAXPY(A,a,F,str));
  CHKERRQ(MatDestroy(&F));
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
  PetscInt       M1,M2,N1,N2;
  PetscInt       m1,m2,n1,n2;
  MatType        t1,t2;
  PetscBool      sametype,transpose;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,MAT_CLASSID,1);
  PetscValidLogicalCollectiveScalar(Y,a,2);
  PetscValidHeaderSpecific(X,MAT_CLASSID,3);
  PetscCheckSameComm(Y,1,X,3);
  CHKERRQ(MatGetSize(X,&M1,&N1));
  CHKERRQ(MatGetSize(Y,&M2,&N2));
  CHKERRQ(MatGetLocalSize(X,&m1,&n1));
  CHKERRQ(MatGetLocalSize(Y,&m2,&n2));
  PetscCheck(M1 == M2 && N1 == N2,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_SIZ,"Non conforming matrix add: global sizes X %" PetscInt_FMT " x %" PetscInt_FMT ", Y %" PetscInt_FMT " x %" PetscInt_FMT,M1,N1,M2,N2);
  PetscCheck(m1 == m2 && n1 == n2,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Non conforming matrix add: local sizes X %" PetscInt_FMT " x %" PetscInt_FMT ", Y %" PetscInt_FMT " x %" PetscInt_FMT,m1,n1,m2,n2);
  PetscCheck(Y->assembled,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix (Y)");
  PetscCheck(X->assembled,PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix (X)");
  if (a == (PetscScalar)0.0) PetscFunctionReturn(0);
  if (Y == X) {
    CHKERRQ(MatScale(Y,1.0 + a));
    PetscFunctionReturn(0);
  }
  CHKERRQ(MatGetType(X,&t1));
  CHKERRQ(MatGetType(Y,&t2));
  CHKERRQ(PetscStrcmp(t1,t2,&sametype));
  CHKERRQ(PetscLogEventBegin(MAT_AXPY,Y,0,0,0));
  if (Y->ops->axpy && (sametype || X->ops->axpy == Y->ops->axpy)) {
    CHKERRQ((*Y->ops->axpy)(Y,a,X,str));
  } else {
    CHKERRQ(PetscStrcmp(t1,MATTRANSPOSEMAT,&transpose));
    if (transpose) {
      CHKERRQ(MatTransposeAXPY_Private(Y,a,X,str,X));
    } else {
      CHKERRQ(PetscStrcmp(t2,MATTRANSPOSEMAT,&transpose));
      if (transpose) {
        CHKERRQ(MatTransposeAXPY_Private(Y,a,X,str,Y));
      } else {
        CHKERRQ(MatAXPY_Basic(Y,a,X,str));
      }
    }
  }
  CHKERRQ(PetscLogEventEnd(MAT_AXPY,Y,0,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_Basic_Preallocate(Mat Y, Mat X, Mat *B)
{
  PetscErrorCode (*preall)(Mat,Mat,Mat*) = NULL;

  PetscFunctionBegin;
  /* look for any available faster alternative to the general preallocator */
  CHKERRQ(PetscObjectQueryFunction((PetscObject)Y,"MatAXPYGetPreallocation_C",&preall));
  if (preall) {
    CHKERRQ((*preall)(Y,X,B));
  } else { /* Use MatPrellocator, assumes same row-col distribution */
    Mat      preallocator;
    PetscInt r,rstart,rend;
    PetscInt m,n,M,N;

    CHKERRQ(MatGetRowUpperTriangular(Y));
    CHKERRQ(MatGetRowUpperTriangular(X));
    CHKERRQ(MatGetSize(Y,&M,&N));
    CHKERRQ(MatGetLocalSize(Y,&m,&n));
    CHKERRQ(MatCreate(PetscObjectComm((PetscObject)Y),&preallocator));
    CHKERRQ(MatSetType(preallocator,MATPREALLOCATOR));
    CHKERRQ(MatSetLayouts(preallocator,Y->rmap,Y->cmap));
    CHKERRQ(MatSetUp(preallocator));
    CHKERRQ(MatGetOwnershipRange(preallocator,&rstart,&rend));
    for (r = rstart; r < rend; ++r) {
      PetscInt          ncols;
      const PetscInt    *row;
      const PetscScalar *vals;

      CHKERRQ(MatGetRow(Y,r,&ncols,&row,&vals));
      CHKERRQ(MatSetValues(preallocator,1,&r,ncols,row,vals,INSERT_VALUES));
      CHKERRQ(MatRestoreRow(Y,r,&ncols,&row,&vals));
      CHKERRQ(MatGetRow(X,r,&ncols,&row,&vals));
      CHKERRQ(MatSetValues(preallocator,1,&r,ncols,row,vals,INSERT_VALUES));
      CHKERRQ(MatRestoreRow(X,r,&ncols,&row,&vals));
    }
    CHKERRQ(MatSetOption(preallocator,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE));
    CHKERRQ(MatAssemblyBegin(preallocator,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(preallocator,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatRestoreRowUpperTriangular(Y));
    CHKERRQ(MatRestoreRowUpperTriangular(X));

    CHKERRQ(MatCreate(PetscObjectComm((PetscObject)Y),B));
    CHKERRQ(MatSetType(*B,((PetscObject)Y)->type_name));
    CHKERRQ(MatSetLayouts(*B,Y->rmap,Y->cmap));
    CHKERRQ(MatPreallocatorPreallocate(preallocator,PETSC_FALSE,*B));
    CHKERRQ(MatDestroy(&preallocator));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_Basic(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscBool      isshell,isdense,isnest;

  PetscFunctionBegin;
  CHKERRQ(MatIsShell(Y,&isshell));
  if (isshell) { /* MatShell has special support for AXPY */
    PetscErrorCode (*f)(Mat,PetscScalar,Mat,MatStructure);

    CHKERRQ(MatGetOperation(Y,MATOP_AXPY,(void (**)(void))&f));
    if (f) {
      CHKERRQ((*f)(Y,a,X,str));
      PetscFunctionReturn(0);
    }
  }
  /* no need to preallocate if Y is dense */
  CHKERRQ(PetscObjectBaseTypeCompareAny((PetscObject)Y,&isdense,MATSEQDENSE,MATMPIDENSE,""));
  if (isdense) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)X,MATNEST,&isnest));
    if (isnest) {
      CHKERRQ(MatAXPY_Dense_Nest(Y,a,X));
      PetscFunctionReturn(0);
    }
    if (str == DIFFERENT_NONZERO_PATTERN || str == UNKNOWN_NONZERO_PATTERN) str = SUBSET_NONZERO_PATTERN;
  }
  if (str != DIFFERENT_NONZERO_PATTERN && str != UNKNOWN_NONZERO_PATTERN) {
    PetscInt          i,start,end,j,ncols,m,n;
    const PetscInt    *row;
    PetscScalar       *val;
    const PetscScalar *vals;

    CHKERRQ(MatGetSize(X,&m,&n));
    CHKERRQ(MatGetOwnershipRange(X,&start,&end));
    CHKERRQ(MatGetRowUpperTriangular(X));
    if (a == 1.0) {
      for (i = start; i < end; i++) {
        CHKERRQ(MatGetRow(X,i,&ncols,&row,&vals));
        CHKERRQ(MatSetValues(Y,1,&i,ncols,row,vals,ADD_VALUES));
        CHKERRQ(MatRestoreRow(X,i,&ncols,&row,&vals));
      }
    } else {
      PetscInt vs = 100;
      /* realloc if needed, as this function may be used in parallel */
      CHKERRQ(PetscMalloc1(vs,&val));
      for (i=start; i<end; i++) {
        CHKERRQ(MatGetRow(X,i,&ncols,&row,&vals));
        if (vs < ncols) {
          vs   = PetscMin(2*ncols,n);
          CHKERRQ(PetscRealloc(vs*sizeof(*val),&val));
        }
        for (j=0; j<ncols; j++) val[j] = a*vals[j];
        CHKERRQ(MatSetValues(Y,1,&i,ncols,row,val,ADD_VALUES));
        CHKERRQ(MatRestoreRow(X,i,&ncols,&row,&vals));
      }
      CHKERRQ(PetscFree(val));
    }
    CHKERRQ(MatRestoreRowUpperTriangular(X));
    CHKERRQ(MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY));
  } else {
    Mat B;

    CHKERRQ(MatAXPY_Basic_Preallocate(Y,X,&B));
    CHKERRQ(MatAXPY_BasicWithPreallocation(B,Y,a,X,str));
    CHKERRQ(MatHeaderMerge(Y,&B));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_BasicWithPreallocation(Mat B,Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscInt          i,start,end,j,ncols,m,n;
  const PetscInt    *row;
  PetscScalar       *val;
  const PetscScalar *vals;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(X,&m,&n));
  CHKERRQ(MatGetOwnershipRange(X,&start,&end));
  CHKERRQ(MatGetRowUpperTriangular(Y));
  CHKERRQ(MatGetRowUpperTriangular(X));
  if (a == 1.0) {
    for (i = start; i < end; i++) {
      CHKERRQ(MatGetRow(Y,i,&ncols,&row,&vals));
      CHKERRQ(MatSetValues(B,1,&i,ncols,row,vals,ADD_VALUES));
      CHKERRQ(MatRestoreRow(Y,i,&ncols,&row,&vals));

      CHKERRQ(MatGetRow(X,i,&ncols,&row,&vals));
      CHKERRQ(MatSetValues(B,1,&i,ncols,row,vals,ADD_VALUES));
      CHKERRQ(MatRestoreRow(X,i,&ncols,&row,&vals));
    }
  } else {
    PetscInt vs = 100;
    /* realloc if needed, as this function may be used in parallel */
    CHKERRQ(PetscMalloc1(vs,&val));
    for (i=start; i<end; i++) {
      CHKERRQ(MatGetRow(Y,i,&ncols,&row,&vals));
      CHKERRQ(MatSetValues(B,1,&i,ncols,row,vals,ADD_VALUES));
      CHKERRQ(MatRestoreRow(Y,i,&ncols,&row,&vals));

      CHKERRQ(MatGetRow(X,i,&ncols,&row,&vals));
      if (vs < ncols) {
        vs   = PetscMin(2*ncols,n);
        CHKERRQ(PetscRealloc(vs*sizeof(*val),&val));
      }
      for (j=0; j<ncols; j++) val[j] = a*vals[j];
      CHKERRQ(MatSetValues(B,1,&i,ncols,row,val,ADD_VALUES));
      CHKERRQ(MatRestoreRow(X,i,&ncols,&row,&vals));
    }
    CHKERRQ(PetscFree(val));
  }
  CHKERRQ(MatRestoreRowUpperTriangular(Y));
  CHKERRQ(MatRestoreRowUpperTriangular(X));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,MAT_CLASSID,1);
  PetscCheck(Y->assembled,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!Y->factortype,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(Y,1);
  if (a == 0.0) PetscFunctionReturn(0);

  if (Y->ops->shift) CHKERRQ((*Y->ops->shift)(Y,a));
  else CHKERRQ(MatShift_Basic(Y,a));

  CHKERRQ(PetscObjectStateIncrease((PetscObject)Y));
  PetscFunctionReturn(0);
}

PetscErrorCode  MatDiagonalSet_Default(Mat Y,Vec D,InsertMode is)
{
  PetscInt          i,start,end;
  const PetscScalar *v;

  PetscFunctionBegin;
  CHKERRQ(MatGetOwnershipRange(Y,&start,&end));
  CHKERRQ(VecGetArrayRead(D,&v));
  for (i=start; i<end; i++) {
    CHKERRQ(MatSetValues(Y,1,&i,1,&i,v+i-start,is));
  }
  CHKERRQ(VecRestoreArrayRead(D,&v));
  CHKERRQ(MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY));
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
  PetscInt       matlocal,veclocal;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,MAT_CLASSID,1);
  PetscValidHeaderSpecific(D,VEC_CLASSID,2);
  CHKERRQ(MatGetLocalSize(Y,&matlocal,NULL));
  CHKERRQ(VecGetLocalSize(D,&veclocal));
  PetscCheck(matlocal == veclocal,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number local rows of matrix %" PetscInt_FMT " does not match that of vector for diagonal %" PetscInt_FMT,matlocal,veclocal);
  if (Y->ops->diagonalset) {
    CHKERRQ((*Y->ops->diagonalset)(Y,D,is));
  } else {
    CHKERRQ(MatDiagonalSet_Default(Y,D,is));
  }
  CHKERRQ(PetscObjectStateIncrease((PetscObject)Y));
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
  PetscFunctionBegin;
  CHKERRQ(MatScale(Y,a));
  CHKERRQ(MatAXPY(Y,1.0,X,str));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(inmat,MAT_CLASSID,1);
  PetscValidPointer(mat,3);
  CHKERRQ(MatConvert_Shell(inmat,mattype ? mattype : MATDENSE,MAT_INITIAL_MATRIX,mat));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(inmat,MAT_CLASSID,1);
  PetscValidPointer(mat,3);
  CHKERRQ(MatCreateTranspose(inmat,&A));
  CHKERRQ(MatConvert_Shell(A,mattype ? mattype : MATDENSE,MAT_INITIAL_MATRIX,mat));
  CHKERRQ(MatDestroy(&A));
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

  PetscFunctionBegin;
  CHKERRQ(PetscObjectBaseTypeCompareAny((PetscObject)A, &flg, MATSEQDENSE, MATMPIDENSE, ""));
  if (flg) {
    CHKERRQ(MatDenseGetLocalMatrix(A, &a));
    CHKERRQ(MatDenseGetLDA(a, &r));
    CHKERRQ(MatGetSize(a, &rStart, &rEnd));
    CHKERRQ(MatDenseGetArray(a, &newVals));
    for (; colMax < rEnd; ++colMax) {
      for (maxRows = 0; maxRows < rStart; ++maxRows) {
        newVals[maxRows + colMax * r] = PetscAbsScalar(newVals[maxRows + colMax * r]) < tol ? 0.0 : newVals[maxRows + colMax * r];
      }
    }
    CHKERRQ(MatDenseRestoreArray(a, &newVals));
  } else {
    CHKERRQ(MatGetOwnershipRange(A, &rStart, &rEnd));
    CHKERRQ(MatGetRowUpperTriangular(A));
    for (r = rStart; r < rEnd; ++r) {
      PetscInt ncols;

      CHKERRQ(MatGetRow(A, r, &ncols, NULL, NULL));
      colMax = PetscMax(colMax, ncols);
      CHKERRQ(MatRestoreRow(A, r, &ncols, NULL, NULL));
    }
    numRows = rEnd - rStart;
    CHKERRMPI(MPIU_Allreduce(&numRows, &maxRows, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)A)));
    CHKERRQ(PetscMalloc2(colMax, &newCols, colMax, &newVals));
    CHKERRQ(MatGetOption(A, MAT_NO_OFF_PROC_ENTRIES, &flg)); /* cache user-defined value */
    CHKERRQ(MatSetOption(A, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
    /* short-circuit code in MatAssemblyBegin() and MatAssemblyEnd()             */
    /* that are potentially called many times depending on the distribution of A */
    for (r = rStart; r < rStart+maxRows; ++r) {
      const PetscScalar *vals;
      const PetscInt    *cols;
      PetscInt           ncols, newcols, c;

      if (r < rEnd) {
        CHKERRQ(MatGetRow(A, r, &ncols, &cols, &vals));
        for (c = 0; c < ncols; ++c) {
          newCols[c] = cols[c];
          newVals[c] = PetscAbsScalar(vals[c]) < tol ? 0.0 : vals[c];
        }
        newcols = ncols;
        CHKERRQ(MatRestoreRow(A, r, &ncols, &cols, &vals));
        CHKERRQ(MatSetValues(A, 1, &r, newcols, newCols, newVals, INSERT_VALUES));
      }
      CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    }
    CHKERRQ(MatRestoreRowUpperTriangular(A));
    CHKERRQ(PetscFree2(newCols, newVals));
    CHKERRQ(MatSetOption(A, MAT_NO_OFF_PROC_ENTRIES, flg)); /* reset option to its user-defined value */
  }
  PetscFunctionReturn(0);
}
