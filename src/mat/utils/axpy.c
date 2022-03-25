
#include <petsc/private/matimpl.h>  /*I   "petscmat.h"  I*/

static PetscErrorCode MatTransposeAXPY_Private(Mat Y,PetscScalar a,Mat X,MatStructure str,Mat T)
{
  Mat            A,F;
  PetscErrorCode (*f)(Mat,Mat*);

  PetscFunctionBegin;
  PetscCall(PetscObjectQueryFunction((PetscObject)T,"MatTransposeGetMat_C",&f));
  if (f) {
    PetscCall(MatTransposeGetMat(T,&A));
    if (T == X) {
      PetscCall(PetscInfo(NULL,"Explicitly transposing X of type MATTRANSPOSEMAT to perform MatAXPY()\n"));
      PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX,&F));
      A = Y;
    } else {
      PetscCall(PetscInfo(NULL,"Transposing X because Y of type MATTRANSPOSEMAT to perform MatAXPY()\n"));
      PetscCall(MatTranspose(X,MAT_INITIAL_MATRIX,&F));
    }
  } else {
    PetscCall(MatHermitianTransposeGetMat(T,&A));
    if (T == X) {
      PetscCall(PetscInfo(NULL,"Explicitly Hermitian transposing X of type MATTRANSPOSEMAT to perform MatAXPY()\n"));
      PetscCall(MatHermitianTranspose(A,MAT_INITIAL_MATRIX,&F));
      A = Y;
    } else {
      PetscCall(PetscInfo(NULL,"Hermitian transposing X because Y of type MATTRANSPOSEMAT to perform MatAXPY()\n"));
      PetscCall(MatHermitianTranspose(X,MAT_INITIAL_MATRIX,&F));
    }
  }
  PetscCall(MatAXPY(A,a,F,str));
  PetscCall(MatDestroy(&F));
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
  PetscCall(MatGetSize(X,&M1,&N1));
  PetscCall(MatGetSize(Y,&M2,&N2));
  PetscCall(MatGetLocalSize(X,&m1,&n1));
  PetscCall(MatGetLocalSize(Y,&m2,&n2));
  PetscCheck(M1 == M2 && N1 == N2,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_SIZ,"Non conforming matrix add: global sizes X %" PetscInt_FMT " x %" PetscInt_FMT ", Y %" PetscInt_FMT " x %" PetscInt_FMT,M1,N1,M2,N2);
  PetscCheck(m1 == m2 && n1 == n2,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Non conforming matrix add: local sizes X %" PetscInt_FMT " x %" PetscInt_FMT ", Y %" PetscInt_FMT " x %" PetscInt_FMT,m1,n1,m2,n2);
  PetscCheck(Y->assembled,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix (Y)");
  PetscCheck(X->assembled,PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix (X)");
  if (a == (PetscScalar)0.0) PetscFunctionReturn(0);
  if (Y == X) {
    PetscCall(MatScale(Y,1.0 + a));
    PetscFunctionReturn(0);
  }
  PetscCall(MatGetType(X,&t1));
  PetscCall(MatGetType(Y,&t2));
  PetscCall(PetscStrcmp(t1,t2,&sametype));
  PetscCall(PetscLogEventBegin(MAT_AXPY,Y,0,0,0));
  if (Y->ops->axpy && (sametype || X->ops->axpy == Y->ops->axpy)) {
    PetscCall((*Y->ops->axpy)(Y,a,X,str));
  } else {
    PetscCall(PetscStrcmp(t1,MATTRANSPOSEMAT,&transpose));
    if (transpose) {
      PetscCall(MatTransposeAXPY_Private(Y,a,X,str,X));
    } else {
      PetscCall(PetscStrcmp(t2,MATTRANSPOSEMAT,&transpose));
      if (transpose) {
        PetscCall(MatTransposeAXPY_Private(Y,a,X,str,Y));
      } else {
        PetscCall(MatAXPY_Basic(Y,a,X,str));
      }
    }
  }
  PetscCall(PetscLogEventEnd(MAT_AXPY,Y,0,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_Basic_Preallocate(Mat Y, Mat X, Mat *B)
{
  PetscErrorCode (*preall)(Mat,Mat,Mat*) = NULL;

  PetscFunctionBegin;
  /* look for any available faster alternative to the general preallocator */
  PetscCall(PetscObjectQueryFunction((PetscObject)Y,"MatAXPYGetPreallocation_C",&preall));
  if (preall) {
    PetscCall((*preall)(Y,X,B));
  } else { /* Use MatPrellocator, assumes same row-col distribution */
    Mat      preallocator;
    PetscInt r,rstart,rend;
    PetscInt m,n,M,N;

    PetscCall(MatGetRowUpperTriangular(Y));
    PetscCall(MatGetRowUpperTriangular(X));
    PetscCall(MatGetSize(Y,&M,&N));
    PetscCall(MatGetLocalSize(Y,&m,&n));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)Y),&preallocator));
    PetscCall(MatSetType(preallocator,MATPREALLOCATOR));
    PetscCall(MatSetLayouts(preallocator,Y->rmap,Y->cmap));
    PetscCall(MatSetUp(preallocator));
    PetscCall(MatGetOwnershipRange(preallocator,&rstart,&rend));
    for (r = rstart; r < rend; ++r) {
      PetscInt          ncols;
      const PetscInt    *row;
      const PetscScalar *vals;

      PetscCall(MatGetRow(Y,r,&ncols,&row,&vals));
      PetscCall(MatSetValues(preallocator,1,&r,ncols,row,vals,INSERT_VALUES));
      PetscCall(MatRestoreRow(Y,r,&ncols,&row,&vals));
      PetscCall(MatGetRow(X,r,&ncols,&row,&vals));
      PetscCall(MatSetValues(preallocator,1,&r,ncols,row,vals,INSERT_VALUES));
      PetscCall(MatRestoreRow(X,r,&ncols,&row,&vals));
    }
    PetscCall(MatSetOption(preallocator,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE));
    PetscCall(MatAssemblyBegin(preallocator,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(preallocator,MAT_FINAL_ASSEMBLY));
    PetscCall(MatRestoreRowUpperTriangular(Y));
    PetscCall(MatRestoreRowUpperTriangular(X));

    PetscCall(MatCreate(PetscObjectComm((PetscObject)Y),B));
    PetscCall(MatSetType(*B,((PetscObject)Y)->type_name));
    PetscCall(MatSetLayouts(*B,Y->rmap,Y->cmap));
    PetscCall(MatPreallocatorPreallocate(preallocator,PETSC_FALSE,*B));
    PetscCall(MatDestroy(&preallocator));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_Basic(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscBool      isshell,isdense,isnest;

  PetscFunctionBegin;
  PetscCall(MatIsShell(Y,&isshell));
  if (isshell) { /* MatShell has special support for AXPY */
    PetscErrorCode (*f)(Mat,PetscScalar,Mat,MatStructure);

    PetscCall(MatGetOperation(Y,MATOP_AXPY,(void (**)(void))&f));
    if (f) {
      PetscCall((*f)(Y,a,X,str));
      PetscFunctionReturn(0);
    }
  }
  /* no need to preallocate if Y is dense */
  PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)Y,&isdense,MATSEQDENSE,MATMPIDENSE,""));
  if (isdense) {
    PetscCall(PetscObjectTypeCompare((PetscObject)X,MATNEST,&isnest));
    if (isnest) {
      PetscCall(MatAXPY_Dense_Nest(Y,a,X));
      PetscFunctionReturn(0);
    }
    if (str == DIFFERENT_NONZERO_PATTERN || str == UNKNOWN_NONZERO_PATTERN) str = SUBSET_NONZERO_PATTERN;
  }
  if (str != DIFFERENT_NONZERO_PATTERN && str != UNKNOWN_NONZERO_PATTERN) {
    PetscInt          i,start,end,j,ncols,m,n;
    const PetscInt    *row;
    PetscScalar       *val;
    const PetscScalar *vals;

    PetscCall(MatGetSize(X,&m,&n));
    PetscCall(MatGetOwnershipRange(X,&start,&end));
    PetscCall(MatGetRowUpperTriangular(X));
    if (a == 1.0) {
      for (i = start; i < end; i++) {
        PetscCall(MatGetRow(X,i,&ncols,&row,&vals));
        PetscCall(MatSetValues(Y,1,&i,ncols,row,vals,ADD_VALUES));
        PetscCall(MatRestoreRow(X,i,&ncols,&row,&vals));
      }
    } else {
      PetscInt vs = 100;
      /* realloc if needed, as this function may be used in parallel */
      PetscCall(PetscMalloc1(vs,&val));
      for (i=start; i<end; i++) {
        PetscCall(MatGetRow(X,i,&ncols,&row,&vals));
        if (vs < ncols) {
          vs   = PetscMin(2*ncols,n);
          PetscCall(PetscRealloc(vs*sizeof(*val),&val));
        }
        for (j=0; j<ncols; j++) val[j] = a*vals[j];
        PetscCall(MatSetValues(Y,1,&i,ncols,row,val,ADD_VALUES));
        PetscCall(MatRestoreRow(X,i,&ncols,&row,&vals));
      }
      PetscCall(PetscFree(val));
    }
    PetscCall(MatRestoreRowUpperTriangular(X));
    PetscCall(MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY));
  } else {
    Mat B;

    PetscCall(MatAXPY_Basic_Preallocate(Y,X,&B));
    PetscCall(MatAXPY_BasicWithPreallocation(B,Y,a,X,str));
    PetscCall(MatHeaderMerge(Y,&B));
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
  PetscCall(MatGetSize(X,&m,&n));
  PetscCall(MatGetOwnershipRange(X,&start,&end));
  PetscCall(MatGetRowUpperTriangular(Y));
  PetscCall(MatGetRowUpperTriangular(X));
  if (a == 1.0) {
    for (i = start; i < end; i++) {
      PetscCall(MatGetRow(Y,i,&ncols,&row,&vals));
      PetscCall(MatSetValues(B,1,&i,ncols,row,vals,ADD_VALUES));
      PetscCall(MatRestoreRow(Y,i,&ncols,&row,&vals));

      PetscCall(MatGetRow(X,i,&ncols,&row,&vals));
      PetscCall(MatSetValues(B,1,&i,ncols,row,vals,ADD_VALUES));
      PetscCall(MatRestoreRow(X,i,&ncols,&row,&vals));
    }
  } else {
    PetscInt vs = 100;
    /* realloc if needed, as this function may be used in parallel */
    PetscCall(PetscMalloc1(vs,&val));
    for (i=start; i<end; i++) {
      PetscCall(MatGetRow(Y,i,&ncols,&row,&vals));
      PetscCall(MatSetValues(B,1,&i,ncols,row,vals,ADD_VALUES));
      PetscCall(MatRestoreRow(Y,i,&ncols,&row,&vals));

      PetscCall(MatGetRow(X,i,&ncols,&row,&vals));
      if (vs < ncols) {
        vs   = PetscMin(2*ncols,n);
        PetscCall(PetscRealloc(vs*sizeof(*val),&val));
      }
      for (j=0; j<ncols; j++) val[j] = a*vals[j];
      PetscCall(MatSetValues(B,1,&i,ncols,row,val,ADD_VALUES));
      PetscCall(MatRestoreRow(X,i,&ncols,&row,&vals));
    }
    PetscCall(PetscFree(val));
  }
  PetscCall(MatRestoreRowUpperTriangular(Y));
  PetscCall(MatRestoreRowUpperTriangular(X));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
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

  if (Y->ops->shift) PetscCall((*Y->ops->shift)(Y,a));
  else PetscCall(MatShift_Basic(Y,a));

  PetscCall(PetscObjectStateIncrease((PetscObject)Y));
  PetscFunctionReturn(0);
}

PetscErrorCode  MatDiagonalSet_Default(Mat Y,Vec D,InsertMode is)
{
  PetscInt          i,start,end;
  const PetscScalar *v;

  PetscFunctionBegin;
  PetscCall(MatGetOwnershipRange(Y,&start,&end));
  PetscCall(VecGetArrayRead(D,&v));
  for (i=start; i<end; i++) {
    PetscCall(MatSetValues(Y,1,&i,1,&i,v+i-start,is));
  }
  PetscCall(VecRestoreArrayRead(D,&v));
  PetscCall(MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY));
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
  PetscCall(MatGetLocalSize(Y,&matlocal,NULL));
  PetscCall(VecGetLocalSize(D,&veclocal));
  PetscCheck(matlocal == veclocal,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number local rows of matrix %" PetscInt_FMT " does not match that of vector for diagonal %" PetscInt_FMT,matlocal,veclocal);
  if (Y->ops->diagonalset) {
    PetscCall((*Y->ops->diagonalset)(Y,D,is));
  } else {
    PetscCall(MatDiagonalSet_Default(Y,D,is));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)Y));
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
  PetscCall(MatScale(Y,a));
  PetscCall(MatAXPY(Y,1.0,X,str));
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
  PetscCall(MatConvert_Shell(inmat,mattype ? mattype : MATDENSE,MAT_INITIAL_MATRIX,mat));
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
  PetscCall(MatCreateTranspose(inmat,&A));
  PetscCall(MatConvert_Shell(A,mattype ? mattype : MATDENSE,MAT_INITIAL_MATRIX,mat));
  PetscCall(MatDestroy(&A));
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
  PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)A, &flg, MATSEQDENSE, MATMPIDENSE, ""));
  if (flg) {
    PetscCall(MatDenseGetLocalMatrix(A, &a));
    PetscCall(MatDenseGetLDA(a, &r));
    PetscCall(MatGetSize(a, &rStart, &rEnd));
    PetscCall(MatDenseGetArray(a, &newVals));
    for (; colMax < rEnd; ++colMax) {
      for (maxRows = 0; maxRows < rStart; ++maxRows) {
        newVals[maxRows + colMax * r] = PetscAbsScalar(newVals[maxRows + colMax * r]) < tol ? 0.0 : newVals[maxRows + colMax * r];
      }
    }
    PetscCall(MatDenseRestoreArray(a, &newVals));
  } else {
    PetscCall(MatGetOwnershipRange(A, &rStart, &rEnd));
    PetscCall(MatGetRowUpperTriangular(A));
    for (r = rStart; r < rEnd; ++r) {
      PetscInt ncols;

      PetscCall(MatGetRow(A, r, &ncols, NULL, NULL));
      colMax = PetscMax(colMax, ncols);
      PetscCall(MatRestoreRow(A, r, &ncols, NULL, NULL));
    }
    numRows = rEnd - rStart;
    PetscCallMPI(MPIU_Allreduce(&numRows, &maxRows, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)A)));
    PetscCall(PetscMalloc2(colMax, &newCols, colMax, &newVals));
    PetscCall(MatGetOption(A, MAT_NO_OFF_PROC_ENTRIES, &flg)); /* cache user-defined value */
    PetscCall(MatSetOption(A, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
    /* short-circuit code in MatAssemblyBegin() and MatAssemblyEnd()             */
    /* that are potentially called many times depending on the distribution of A */
    for (r = rStart; r < rStart+maxRows; ++r) {
      const PetscScalar *vals;
      const PetscInt    *cols;
      PetscInt           ncols, newcols, c;

      if (r < rEnd) {
        PetscCall(MatGetRow(A, r, &ncols, &cols, &vals));
        for (c = 0; c < ncols; ++c) {
          newCols[c] = cols[c];
          newVals[c] = PetscAbsScalar(vals[c]) < tol ? 0.0 : vals[c];
        }
        newcols = ncols;
        PetscCall(MatRestoreRow(A, r, &ncols, &cols, &vals));
        PetscCall(MatSetValues(A, 1, &r, newcols, newCols, newVals, INSERT_VALUES));
      }
      PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    }
    PetscCall(MatRestoreRowUpperTriangular(A));
    PetscCall(PetscFree2(newCols, newVals));
    PetscCall(MatSetOption(A, MAT_NO_OFF_PROC_ENTRIES, flg)); /* reset option to its user-defined value */
  }
  PetscFunctionReturn(0);
}
