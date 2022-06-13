/*
   This is where the abstract matrix operations are defined
*/

#include <petsc/private/matimpl.h>        /*I "petscmat.h" I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>

/* Logging support */
PetscClassId MAT_CLASSID;
PetscClassId MAT_COLORING_CLASSID;
PetscClassId MAT_FDCOLORING_CLASSID;
PetscClassId MAT_TRANSPOSECOLORING_CLASSID;

PetscLogEvent MAT_Mult, MAT_Mults, MAT_MultAdd, MAT_MultTranspose;
PetscLogEvent MAT_MultTransposeAdd, MAT_Solve, MAT_Solves, MAT_SolveAdd, MAT_SolveTranspose, MAT_MatSolve,MAT_MatTrSolve;
PetscLogEvent MAT_SolveTransposeAdd, MAT_SOR, MAT_ForwardSolve, MAT_BackwardSolve, MAT_LUFactor, MAT_LUFactorSymbolic;
PetscLogEvent MAT_LUFactorNumeric, MAT_CholeskyFactor, MAT_CholeskyFactorSymbolic, MAT_CholeskyFactorNumeric, MAT_ILUFactor;
PetscLogEvent MAT_ILUFactorSymbolic, MAT_ICCFactorSymbolic, MAT_Copy, MAT_Convert, MAT_Scale, MAT_AssemblyBegin;
PetscLogEvent MAT_QRFactorNumeric, MAT_QRFactorSymbolic, MAT_QRFactor;
PetscLogEvent MAT_AssemblyEnd, MAT_SetValues, MAT_GetValues, MAT_GetRow, MAT_GetRowIJ, MAT_CreateSubMats, MAT_GetOrdering, MAT_RedundantMat, MAT_GetSeqNonzeroStructure;
PetscLogEvent MAT_IncreaseOverlap, MAT_Partitioning, MAT_PartitioningND, MAT_Coarsen, MAT_ZeroEntries, MAT_Load, MAT_View, MAT_AXPY, MAT_FDColoringCreate;
PetscLogEvent MAT_FDColoringSetUp, MAT_FDColoringApply,MAT_Transpose,MAT_FDColoringFunction, MAT_CreateSubMat;
PetscLogEvent MAT_TransposeColoringCreate;
PetscLogEvent MAT_MatMult, MAT_MatMultSymbolic, MAT_MatMultNumeric;
PetscLogEvent MAT_PtAP, MAT_PtAPSymbolic, MAT_PtAPNumeric,MAT_RARt, MAT_RARtSymbolic, MAT_RARtNumeric;
PetscLogEvent MAT_MatTransposeMult, MAT_MatTransposeMultSymbolic, MAT_MatTransposeMultNumeric;
PetscLogEvent MAT_TransposeMatMult, MAT_TransposeMatMultSymbolic, MAT_TransposeMatMultNumeric;
PetscLogEvent MAT_MatMatMult, MAT_MatMatMultSymbolic, MAT_MatMatMultNumeric;
PetscLogEvent MAT_MultHermitianTranspose,MAT_MultHermitianTransposeAdd;
PetscLogEvent MAT_Getsymtranspose, MAT_Getsymtransreduced, MAT_GetBrowsOfAcols;
PetscLogEvent MAT_GetBrowsOfAocols, MAT_Getlocalmat, MAT_Getlocalmatcondensed, MAT_Seqstompi, MAT_Seqstompinum, MAT_Seqstompisym;
PetscLogEvent MAT_Applypapt, MAT_Applypapt_numeric, MAT_Applypapt_symbolic, MAT_GetSequentialNonzeroStructure;
PetscLogEvent MAT_GetMultiProcBlock;
PetscLogEvent MAT_CUSPARSECopyToGPU, MAT_CUSPARSECopyFromGPU, MAT_CUSPARSEGenerateTranspose, MAT_CUSPARSESolveAnalysis;
PetscLogEvent MAT_PreallCOO, MAT_SetVCOO;
PetscLogEvent MAT_SetValuesBatch;
PetscLogEvent MAT_ViennaCLCopyToGPU;
PetscLogEvent MAT_DenseCopyToGPU, MAT_DenseCopyFromGPU;
PetscLogEvent MAT_Merge,MAT_Residual,MAT_SetRandom;
PetscLogEvent MAT_FactorFactS,MAT_FactorInvS;
PetscLogEvent MATCOLORING_Apply,MATCOLORING_Comm,MATCOLORING_Local,MATCOLORING_ISCreate,MATCOLORING_SetUp,MATCOLORING_Weights;
PetscLogEvent MAT_H2Opus_Build,MAT_H2Opus_Compress,MAT_H2Opus_Orthog,MAT_H2Opus_LR;

const char *const MatFactorTypes[] = {"NONE","LU","CHOLESKY","ILU","ICC","ILUDT","QR","MatFactorType","MAT_FACTOR_",NULL};

/*@
   MatSetRandom - Sets all components of a matrix to random numbers. For sparse matrices that have been preallocated but not been assembled it randomly selects appropriate locations,
                  for sparse matrices that already have locations it fills the locations with random numbers

   Logically Collective on Mat

   Input Parameters:
+  x  - the matrix
-  rctx - the random number context, formed by PetscRandomCreate(), or NULL and
          it will create one internally.

   Output Parameter:
.  x  - the matrix

   Example of Usage:
.vb
     PetscRandomCreate(PETSC_COMM_WORLD,&rctx);
     MatSetRandom(x,rctx);
     PetscRandomDestroy(rctx);
.ve

   Level: intermediate

.seealso: `MatZeroEntries()`, `MatSetValues()`, `PetscRandomCreate()`, `PetscRandomDestroy()`
@*/
PetscErrorCode MatSetRandom(Mat x,PetscRandom rctx)
{
  PetscRandom    randObj = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,MAT_CLASSID,1);
  if (rctx) PetscValidHeaderSpecific(rctx,PETSC_RANDOM_CLASSID,2);
  PetscValidType(x,1);
  MatCheckPreallocated(x,1);

  PetscCheck(x->ops->setrandom,PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"Mat type %s",((PetscObject)x)->type_name);

  if (!rctx) {
    MPI_Comm comm;
    PetscCall(PetscObjectGetComm((PetscObject)x,&comm));
    PetscCall(PetscRandomCreate(comm,&randObj));
    PetscCall(PetscRandomSetFromOptions(randObj));
    rctx = randObj;
  }
  PetscCall(PetscLogEventBegin(MAT_SetRandom,x,rctx,0,0));
  PetscCall((*x->ops->setrandom)(x,rctx));
  PetscCall(PetscLogEventEnd(MAT_SetRandom,x,rctx,0,0));

  PetscCall(MatAssemblyBegin(x,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(x,MAT_FINAL_ASSEMBLY));
  PetscCall(PetscRandomDestroy(&randObj));
  PetscFunctionReturn(0);
}

/*@
   MatFactorGetErrorZeroPivot - returns the pivot value that was determined to be zero and the row it occurred in

   Logically Collective on Mat

   Input Parameter:
.  mat - the factored matrix

   Output Parameters:
+  pivot - the pivot value computed
-  row - the row that the zero pivot occurred. Note that this row must be interpreted carefully due to row reorderings and which processes
         the share the matrix

   Level: advanced

   Notes:
    This routine does not work for factorizations done with external packages.

    This routine should only be called if MatGetFactorError() returns a value of MAT_FACTOR_NUMERIC_ZEROPIVOT

    This can be called on non-factored matrices that come from, for example, matrices used in SOR.

.seealso: `MatZeroEntries()`, `MatFactor()`, `MatGetFactor()`, `MatLUFactorSymbolic()`, `MatCholeskyFactorSymbolic()`, `MatFactorClearError()`, `MatFactorGetErrorZeroPivot()`
@*/
PetscErrorCode MatFactorGetErrorZeroPivot(Mat mat,PetscReal *pivot,PetscInt *row)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidRealPointer(pivot,2);
  PetscValidIntPointer(row,3);
  *pivot = mat->factorerror_zeropivot_value;
  *row   = mat->factorerror_zeropivot_row;
  PetscFunctionReturn(0);
}

/*@
   MatFactorGetError - gets the error code from a factorization

   Logically Collective on Mat

   Input Parameters:
.  mat - the factored matrix

   Output Parameter:
.  err  - the error code

   Level: advanced

   Notes:
    This can be called on non-factored matrices that come from, for example, matrices used in SOR.

.seealso: `MatZeroEntries()`, `MatFactor()`, `MatGetFactor()`, `MatLUFactorSymbolic()`, `MatCholeskyFactorSymbolic()`, `MatFactorClearError()`, `MatFactorGetErrorZeroPivot()`
@*/
PetscErrorCode MatFactorGetError(Mat mat,MatFactorError *err)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(err,2);
  *err = mat->factorerrortype;
  PetscFunctionReturn(0);
}

/*@
   MatFactorClearError - clears the error code in a factorization

   Logically Collective on Mat

   Input Parameter:
.  mat - the factored matrix

   Level: developer

   Notes:
    This can be called on non-factored matrices that come from, for example, matrices used in SOR.

.seealso: `MatZeroEntries()`, `MatFactor()`, `MatGetFactor()`, `MatLUFactorSymbolic()`, `MatCholeskyFactorSymbolic()`, `MatFactorGetError()`, `MatFactorGetErrorZeroPivot()`
@*/
PetscErrorCode MatFactorClearError(Mat mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  mat->factorerrortype             = MAT_FACTOR_NOERROR;
  mat->factorerror_zeropivot_value = 0.0;
  mat->factorerror_zeropivot_row   = 0;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatFindNonzeroRowsOrCols_Basic(Mat mat,PetscBool cols,PetscReal tol,IS *nonzero)
{
  Vec               r,l;
  const PetscScalar *al;
  PetscInt          i,nz,gnz,N,n;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(mat,&r,&l));
  if (!cols) { /* nonzero rows */
    PetscCall(MatGetSize(mat,&N,NULL));
    PetscCall(MatGetLocalSize(mat,&n,NULL));
    PetscCall(VecSet(l,0.0));
    PetscCall(VecSetRandom(r,NULL));
    PetscCall(MatMult(mat,r,l));
    PetscCall(VecGetArrayRead(l,&al));
  } else { /* nonzero columns */
    PetscCall(MatGetSize(mat,NULL,&N));
    PetscCall(MatGetLocalSize(mat,NULL,&n));
    PetscCall(VecSet(r,0.0));
    PetscCall(VecSetRandom(l,NULL));
    PetscCall(MatMultTranspose(mat,l,r));
    PetscCall(VecGetArrayRead(r,&al));
  }
  if (tol <= 0.0) { for (i=0,nz=0;i<n;i++) if (al[i] != 0.0) nz++; }
  else { for (i=0,nz=0;i<n;i++) if (PetscAbsScalar(al[i]) > tol) nz++; }
  PetscCall(MPIU_Allreduce(&nz,&gnz,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mat)));
  if (gnz != N) {
    PetscInt *nzr;
    PetscCall(PetscMalloc1(nz,&nzr));
    if (nz) {
      if (tol < 0) { for (i=0,nz=0;i<n;i++) if (al[i] != 0.0) nzr[nz++] = i; }
      else { for (i=0,nz=0;i<n;i++) if (PetscAbsScalar(al[i]) > tol) nzr[nz++] = i; }
    }
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)mat),nz,nzr,PETSC_OWN_POINTER,nonzero));
  } else *nonzero = NULL;
  if (!cols) { /* nonzero rows */
    PetscCall(VecRestoreArrayRead(l,&al));
  } else {
    PetscCall(VecRestoreArrayRead(r,&al));
  }
  PetscCall(VecDestroy(&l));
  PetscCall(VecDestroy(&r));
  PetscFunctionReturn(0);
}

/*@
      MatFindNonzeroRows - Locate all rows that are not completely zero in the matrix

  Input Parameter:
.    A  - the matrix

  Output Parameter:
.    keptrows - the rows that are not completely zero

  Notes:
    keptrows is set to NULL if all rows are nonzero.

  Level: intermediate

 @*/
PetscErrorCode MatFindNonzeroRows(Mat mat,IS *keptrows)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidPointer(keptrows,2);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  if (mat->ops->findnonzerorows) {
    PetscCall((*mat->ops->findnonzerorows)(mat,keptrows));
  } else {
    PetscCall(MatFindNonzeroRowsOrCols_Basic(mat,PETSC_FALSE,0.0,keptrows));
  }
  PetscFunctionReturn(0);
}

/*@
      MatFindZeroRows - Locate all rows that are completely zero in the matrix

  Input Parameter:
.    A  - the matrix

  Output Parameter:
.    zerorows - the rows that are completely zero

  Notes:
    zerorows is set to NULL if no rows are zero.

  Level: intermediate

 @*/
PetscErrorCode MatFindZeroRows(Mat mat,IS *zerorows)
{
  IS       keptrows;
  PetscInt m, n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidPointer(zerorows,2);
  PetscCall(MatFindNonzeroRows(mat, &keptrows));
  /* MatFindNonzeroRows sets keptrows to NULL if there are no zero rows.
     In keeping with this convention, we set zerorows to NULL if there are no zero
     rows. */
  if (keptrows == NULL) {
    *zerorows = NULL;
  } else {
    PetscCall(MatGetOwnershipRange(mat,&m,&n));
    PetscCall(ISComplement(keptrows,m,n,zerorows));
    PetscCall(ISDestroy(&keptrows));
  }
  PetscFunctionReturn(0);
}

/*@
   MatGetDiagonalBlock - Returns the part of the matrix associated with the on-process coupling

   Not Collective

   Input Parameters:
.   A - the matrix

   Output Parameters:
.   a - the diagonal part (which is a SEQUENTIAL matrix)

   Notes:
    see the manual page for MatCreateAIJ() for more information on the "diagonal part" of the matrix.
          Use caution, as the reference count on the returned matrix is not incremented and it is used as
          part of the containing MPI Mat's normal operation.

   Level: advanced

@*/
PetscErrorCode MatGetDiagonalBlock(Mat A,Mat *a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(a,2);
  PetscCheck(!A->factortype,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  if (A->ops->getdiagonalblock) {
    PetscCall((*A->ops->getdiagonalblock)(A,a));
  } else {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
    PetscCheck(size == 1,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not for parallel matrix type %s",((PetscObject)A)->type_name);
    *a = A;
  }
  PetscFunctionReturn(0);
}

/*@
   MatGetTrace - Gets the trace of a matrix. The sum of the diagonal entries.

   Collective on Mat

   Input Parameters:
.  mat - the matrix

   Output Parameter:
.   trace - the sum of the diagonal entries

   Level: advanced

@*/
PetscErrorCode MatGetTrace(Mat mat,PetscScalar *trace)
{
  Vec diag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidScalarPointer(trace,2);
  PetscCall(MatCreateVecs(mat,&diag,NULL));
  PetscCall(MatGetDiagonal(mat,diag));
  PetscCall(VecSum(diag,trace));
  PetscCall(VecDestroy(&diag));
  PetscFunctionReturn(0);
}

/*@
   MatRealPart - Zeros out the imaginary part of the matrix

   Logically Collective on Mat

   Input Parameters:
.  mat - the matrix

   Level: advanced

.seealso: `MatImaginaryPart()`
@*/
PetscErrorCode MatRealPart(Mat mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->ops->realpart,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,1);
  PetscCall((*mat->ops->realpart)(mat));
  PetscFunctionReturn(0);
}

/*@C
   MatGetGhosts - Get the global index of all ghost nodes defined by the sparse matrix

   Collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+   nghosts - number of ghosts (note for BAIJ matrices there is one ghost for each block)
-   ghosts - the global indices of the ghost points

   Notes:
    the nghosts and ghosts are suitable to pass into VecCreateGhost()

   Level: advanced

@*/
PetscErrorCode MatGetGhosts(Mat mat,PetscInt *nghosts,const PetscInt *ghosts[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  if (mat->ops->getghosts) {
    PetscCall((*mat->ops->getghosts)(mat,nghosts,ghosts));
  } else {
    if (nghosts) *nghosts = 0;
    if (ghosts)  *ghosts  = NULL;
  }
  PetscFunctionReturn(0);
}

/*@
   MatImaginaryPart - Moves the imaginary part of the matrix to the real part and zeros the imaginary part

   Logically Collective on Mat

   Input Parameters:
.  mat - the matrix

   Level: advanced

.seealso: `MatRealPart()`
@*/
PetscErrorCode MatImaginaryPart(Mat mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->ops->imaginarypart,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,1);
  PetscCall((*mat->ops->imaginarypart)(mat));
  PetscFunctionReturn(0);
}

/*@
   MatMissingDiagonal - Determine if sparse matrix is missing a diagonal entry (or block entry for BAIJ matrices)

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  missing - is any diagonal missing
-  dd - first diagonal entry that is missing (optional) on this process

   Level: advanced

.seealso: `MatRealPart()`
@*/
PetscErrorCode MatMissingDiagonal(Mat mat,PetscBool *missing,PetscInt *dd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidBoolPointer(missing,2);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix %s",((PetscObject)mat)->type_name);
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->ops->missingdiagonal,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  PetscCall((*mat->ops->missingdiagonal)(mat,missing,dd));
  PetscFunctionReturn(0);
}

/*@C
   MatGetRow - Gets a row of a matrix.  You MUST call MatRestoreRow()
   for each row that you get to ensure that your application does
   not bleed memory.

   Not Collective

   Input Parameters:
+  mat - the matrix
-  row - the row to get

   Output Parameters:
+  ncols -  if not NULL, the number of nonzeros in the row
.  cols - if not NULL, the column numbers
-  vals - if not NULL, the values

   Notes:
   This routine is provided for people who need to have direct access
   to the structure of a matrix.  We hope that we provide enough
   high-level matrix routines that few users will need it.

   MatGetRow() always returns 0-based column indices, regardless of
   whether the internal representation is 0-based (default) or 1-based.

   For better efficiency, set cols and/or vals to NULL if you do
   not wish to extract these quantities.

   The user can only examine the values extracted with MatGetRow();
   the values cannot be altered.  To change the matrix entries, one
   must use MatSetValues().

   You can only have one call to MatGetRow() outstanding for a particular
   matrix at a time, per processor. MatGetRow() can only obtain rows
   associated with the given processor, it cannot get rows from the
   other processors; for that we suggest using MatCreateSubMatrices(), then
   MatGetRow() on the submatrix. The row index passed to MatGetRow()
   is in the global number of rows.

   Fortran Notes:
   The calling sequence from Fortran is
.vb
   MatGetRow(matrix,row,ncols,cols,values,ierr)
         Mat     matrix (input)
         integer row    (input)
         integer ncols  (output)
         integer cols(maxcols) (output)
         double precision (or double complex) values(maxcols) output
.ve
   where maxcols >= maximum nonzeros in any row of the matrix.

   Caution:
   Do not try to change the contents of the output arrays (cols and vals).
   In some cases, this may corrupt the matrix.

   Level: advanced

.seealso: `MatRestoreRow()`, `MatSetValues()`, `MatGetValues()`, `MatCreateSubMatrices()`, `MatGetDiagonal()`
@*/
PetscErrorCode MatGetRow(Mat mat,PetscInt row,PetscInt *ncols,const PetscInt *cols[],const PetscScalar *vals[])
{
  PetscInt incols;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscCheck(mat->assembled,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->ops->getrow,PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,1);
  PetscCheck(row >= mat->rmap->rstart && row < mat->rmap->rend,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Only for local rows, %" PetscInt_FMT " not in [%" PetscInt_FMT ",%" PetscInt_FMT ")",row,mat->rmap->rstart,mat->rmap->rend);
  PetscCall(PetscLogEventBegin(MAT_GetRow,mat,0,0,0));
  PetscCall((*mat->ops->getrow)(mat,row,&incols,(PetscInt**)cols,(PetscScalar**)vals));
  if (ncols) *ncols = incols;
  PetscCall(PetscLogEventEnd(MAT_GetRow,mat,0,0,0));
  PetscFunctionReturn(0);
}

/*@
   MatConjugate - replaces the matrix values with their complex conjugates

   Logically Collective on Mat

   Input Parameters:
.  mat - the matrix

   Level: advanced

.seealso: `VecConjugate()`
@*/
PetscErrorCode MatConjugate(Mat mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (PetscDefined(USE_COMPLEX)) {
    PetscCheck(mat->ops->conjugate,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Not provided for matrix type %s, send email to petsc-maint@mcs.anl.gov",((PetscObject)mat)->type_name);
    PetscCall((*mat->ops->conjugate)(mat));
  }
  PetscFunctionReturn(0);
}

/*@C
   MatRestoreRow - Frees any temporary space allocated by MatGetRow().

   Not Collective

   Input Parameters:
+  mat - the matrix
.  row - the row to get
.  ncols, cols - the number of nonzeros and their columns
-  vals - if nonzero the column values

   Notes:
   This routine should be called after you have finished examining the entries.

   This routine zeros out ncols, cols, and vals. This is to prevent accidental
   us of the array after it has been restored. If you pass NULL, it will
   not zero the pointers.  Use of cols or vals after MatRestoreRow is invalid.

   Fortran Notes:
   The calling sequence from Fortran is
.vb
   MatRestoreRow(matrix,row,ncols,cols,values,ierr)
      Mat     matrix (input)
      integer row    (input)
      integer ncols  (output)
      integer cols(maxcols) (output)
      double precision (or double complex) values(maxcols) output
.ve
   Where maxcols >= maximum nonzeros in any row of the matrix.

   In Fortran MatRestoreRow() MUST be called after MatGetRow()
   before another call to MatGetRow() can be made.

   Level: advanced

.seealso: `MatGetRow()`
@*/
PetscErrorCode MatRestoreRow(Mat mat,PetscInt row,PetscInt *ncols,const PetscInt *cols[],const PetscScalar *vals[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (ncols) PetscValidIntPointer(ncols,3);
  PetscCheck(mat->assembled,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (!mat->ops->restorerow) PetscFunctionReturn(0);
  PetscCall((*mat->ops->restorerow)(mat,row,ncols,(PetscInt **)cols,(PetscScalar **)vals));
  if (ncols) *ncols = 0;
  if (cols)  *cols = NULL;
  if (vals)  *vals = NULL;
  PetscFunctionReturn(0);
}

/*@
   MatGetRowUpperTriangular - Sets a flag to enable calls to MatGetRow() for matrix in MATSBAIJ format.
   You should call MatRestoreRowUpperTriangular() after calling MatGetRow/MatRestoreRow() to disable the flag.

   Not Collective

   Input Parameters:
.  mat - the matrix

   Notes:
   The flag is to ensure that users are aware of MatGetRow() only provides the upper triangular part of the row for the matrices in MATSBAIJ format.

   Level: advanced

.seealso: `MatRestoreRowUpperTriangular()`
@*/
PetscErrorCode MatGetRowUpperTriangular(Mat mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscCheck(mat->assembled,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);
  if (!mat->ops->getrowuppertriangular) PetscFunctionReturn(0);
  PetscCall((*mat->ops->getrowuppertriangular)(mat));
  PetscFunctionReturn(0);
}

/*@
   MatRestoreRowUpperTriangular - Disable calls to MatGetRow() for matrix in MATSBAIJ format.

   Not Collective

   Input Parameters:
.  mat - the matrix

   Notes:
   This routine should be called after you have finished MatGetRow/MatRestoreRow().

   Level: advanced

.seealso: `MatGetRowUpperTriangular()`
@*/
PetscErrorCode MatRestoreRowUpperTriangular(Mat mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscCheck(mat->assembled,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);
  if (!mat->ops->restorerowuppertriangular) PetscFunctionReturn(0);
  PetscCall((*mat->ops->restorerowuppertriangular)(mat));
  PetscFunctionReturn(0);
}

/*@C
   MatSetOptionsPrefix - Sets the prefix used for searching for all
   Mat options in the database.

   Logically Collective on Mat

   Input Parameters:
+  A - the Mat context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   This is NOT used for options for the factorization of the matrix. Normally the
   prefix is automatically passed in from the PC calling the factorization. To set
   it directly use  `MatSetOptionsPrefixFactor()`

   Level: advanced

.seealso: `MatSetFromOptions()`, `MatSetOptionsPrefixFactor()`
@*/
PetscErrorCode MatSetOptionsPrefix(Mat A,const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)A,prefix));
  PetscFunctionReturn(0);
}

/*@C
   MatSetOptionsPrefixFactor - Sets the prefix used for searching for all Mat factor options in the database for
   for matrices created with `MatGetFactor()`

   Logically Collective on Mat

   Input Parameters:
+  A - the Mat context
-  prefix - the prefix to prepend to all option names for the factored matrix

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Normally the prefix is automatically passed in from the PC calling the factorization. To set
   it directly when not using `KSP`/`PC` use  `MatSetOptionsPrefixFactor()`

   Level: developer

.seealso: `MatSetFromOptions()`, `MatSetOptionsPrefix()`, `MatAppendOptionsPrefixFactor()`
@*/
PetscErrorCode MatSetOptionsPrefixFactor(Mat A,const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  if (prefix) {
    PetscValidCharPointer(prefix,2);
    PetscCheck(prefix[0] != '-',PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Options prefix should not begin with a hyphen");
    if (prefix != A->factorprefix) {
      PetscCall(PetscFree(A->factorprefix));
      PetscCall(PetscStrallocpy(prefix,&A->factorprefix));
    }
  } else PetscCall(PetscFree(A->factorprefix));
  PetscFunctionReturn(0);
}

/*@C
   MatAppendOptionsPrefixFactor - Appends to the prefix used for searching for all Mat factor options in the database for
   for matrices created with `MatGetFactor()`

   Logically Collective on Mat

   Input Parameters:
+  A - the Mat context
-  prefix - the prefix to prepend to all option names for the factored matrix

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Normally the prefix is automatically passed in from the PC calling the factorization. To set
   it directly when not using `KSP`/`PC` use  `MatAppendOptionsPrefixFactor()`

   Level: developer
   .seealso: `PetscOptionsCreate()`, `PetscOptionsDestroy()`, `PetscObjectSetOptionsPrefix()`, `PetscObjectPrependOptionsPrefix()`,
             `PetscObjectGetOptionsPrefix()`, `TSAppendOptionsPrefix()`, `SNESAppendOptionsPrefix()`, `KSPAppendOptionsPrefix()`, `MatSetOptionsPrefixFactor()`,
             `MatSetOptionsPrefix()`
@*/
PetscErrorCode MatAppendOptionsPrefixFactor(Mat A,const char prefix[])
{
  char           *buf = A->factorprefix;
  size_t         len1,len2;

  PetscFunctionBegin;
  PetscValidHeader(A,1);
  if (!prefix) PetscFunctionReturn(0);
  if (!buf) {
    PetscCall(MatSetOptionsPrefixFactor(A,prefix));
    PetscFunctionReturn(0);
  }
  PetscCheck(prefix[0] != '-',PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Options prefix should not begin with a hyphen");

  PetscCall(PetscStrlen(prefix,&len1));
  PetscCall(PetscStrlen(buf,&len2));
  PetscCall(PetscMalloc1(1+len1+len2,&A->factorprefix));
  PetscCall(PetscStrcpy(A->factorprefix,buf));
  PetscCall(PetscStrcat(A->factorprefix,prefix));
  PetscCall(PetscFree(buf));
  PetscFunctionReturn(0);
}

/*@C
   MatAppendOptionsPrefix - Appends to the prefix used for searching for all
   Mat options in the database.

   Logically Collective on Mat

   Input Parameters:
+  A - the Mat context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: `MatGetOptionsPrefix()`
@*/
PetscErrorCode MatAppendOptionsPrefix(Mat A,const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)A,prefix));
  PetscFunctionReturn(0);
}

/*@C
   MatGetOptionsPrefix - Gets the prefix used for searching for all
   Mat options in the database.

   Not Collective

   Input Parameter:
.  A - the Mat context

   Output Parameter:
.  prefix - pointer to the prefix string used

   Notes:
    On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: `MatAppendOptionsPrefix()`
@*/
PetscErrorCode MatGetOptionsPrefix(Mat A,const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(prefix,2);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)A,prefix));
  PetscFunctionReturn(0);
}

/*@
   MatResetPreallocation - Reset mat to use the original nonzero pattern provided by users.

   Collective on Mat

   Input Parameters:
.  A - the Mat context

   Notes:
   The allocated memory will be shrunk after calling MatAssembly with MAT_FINAL_ASSEMBLY. Users can reset the preallocation to access the original memory.
   Currently support MPIAIJ and SEQAIJ.

   Level: beginner

.seealso: `MatSeqAIJSetPreallocation()`, `MatMPIAIJSetPreallocation()`, `MatXAIJSetPreallocation()`
@*/
PetscErrorCode MatResetPreallocation(Mat A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscUseMethod(A,"MatResetPreallocation_C",(Mat),(A));
  PetscFunctionReturn(0);
}

/*@
   MatSetUp - Sets up the internal matrix data structures for later use.

   Collective on Mat

   Input Parameters:
.  A - the Mat context

   Notes:
   If the user has not set preallocation for this matrix then a default preallocation that is likely to be inefficient is used.

   If a suitable preallocation routine is used, this function does not need to be called.

   See the Performance chapter of the PETSc users manual for how to preallocate matrices

   Level: beginner

.seealso: `MatCreate()`, `MatDestroy()`
@*/
PetscErrorCode MatSetUp(Mat A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  if (!((PetscObject)A)->type_name) {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
    PetscCall(MatSetType(A, size == 1 ? MATSEQAIJ : MATMPIAIJ));
  }
  if (!A->preallocated && A->ops->setup) {
    PetscCall(PetscInfo(A,"Warning not preallocating matrix storage\n"));
    PetscCall((*A->ops->setup)(A));
  }
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  A->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_SAWS)
#include <petscviewersaws.h>
#endif

/*@C
   MatViewFromOptions - View from Options

   Collective on Mat

   Input Parameters:
+  A - the Mat context
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso: `Mat`, `MatView`, `PetscObjectViewFromOptions()`, `MatCreate()`
@*/
PetscErrorCode  MatViewFromOptions(Mat A,PetscObject obj,const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A,obj,name));
  PetscFunctionReturn(0);
}

/*@C
   MatView - Visualizes a matrix object.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  viewer - visualization context

  Notes:
  The available visualization contexts include
+    PETSC_VIEWER_STDOUT_SELF - for sequential matrices
.    PETSC_VIEWER_STDOUT_WORLD - for parallel matrices created on PETSC_COMM_WORLD
.    PETSC_VIEWER_STDOUT_(comm) - for matrices created on MPI communicator comm
-     PETSC_VIEWER_DRAW_WORLD - graphical display of nonzero structure

   The user can open alternative visualization contexts with
+    PetscViewerASCIIOpen() - Outputs matrix to a specified file
.    PetscViewerBinaryOpen() - Outputs matrix in binary to a
         specified file; corresponding input uses MatLoad()
.    PetscViewerDrawOpen() - Outputs nonzero matrix structure to
         an X window display
-    PetscViewerSocketOpen() - Outputs matrix to Socket viewer.
         Currently only the sequential dense and AIJ
         matrix types support the Socket viewer.

   The user can call PetscViewerPushFormat() to specify the output
   format of ASCII printed objects (when using PETSC_VIEWER_STDOUT_SELF,
   PETSC_VIEWER_STDOUT_WORLD and PetscViewerASCIIOpen).  Available formats include
+    PETSC_VIEWER_DEFAULT - default, prints matrix contents
.    PETSC_VIEWER_ASCII_MATLAB - prints matrix contents in Matlab format
.    PETSC_VIEWER_ASCII_DENSE - prints entire matrix including zeros
.    PETSC_VIEWER_ASCII_COMMON - prints matrix contents, using a sparse
         format common among all matrix types
.    PETSC_VIEWER_ASCII_IMPL - prints matrix contents, using an implementation-specific
         format (which is in many cases the same as the default)
.    PETSC_VIEWER_ASCII_INFO - prints basic information about the matrix
         size and structure (not the matrix entries)
-    PETSC_VIEWER_ASCII_INFO_DETAIL - prints more detailed information about
         the matrix structure

   Options Database Keys:
+  -mat_view ::ascii_info - Prints info on matrix at conclusion of MatAssemblyEnd()
.  -mat_view ::ascii_info_detail - Prints more detailed info
.  -mat_view - Prints matrix in ASCII format
.  -mat_view ::ascii_matlab - Prints matrix in Matlab format
.  -mat_view draw - PetscDraws nonzero structure of matrix, using MatView() and PetscDrawOpenX().
.  -display <name> - Sets display name (default is host)
.  -draw_pause <sec> - Sets number of seconds to pause after display
.  -mat_view socket - Sends matrix to socket, can be accessed from Matlab (see Users-Manual: ch_matlab for details)
.  -viewer_socket_machine <machine> -
.  -viewer_socket_port <port> -
.  -mat_view binary - save matrix to file in binary format
-  -viewer_binary_filename <name> -

   Level: beginner

   Notes:
    The ASCII viewers are only recommended for small matrices on at most a moderate number of processes,
    the program will seemingly hang and take hours for larger matrices, for larger matrices one should use the binary format.

    In the debugger you can do "call MatView(mat,0)" to display the matrix. (The same holds for any PETSc object viewer).

    See the manual page for MatLoad() for the exact format of the binary file when the binary
      viewer is used.

      See share/petsc/matlab/PetscBinaryRead.m for a Matlab code that can read in the binary file when the binary
      viewer is used and lib/petsc/bin/PetscBinaryIO.py for loading them into Python.

      One can use '-mat_view draw -draw_pause -1' to pause the graphical display of matrix nonzero structure,
      and then use the following mouse functions.
.vb
  left mouse: zoom in
  middle mouse: zoom out
  right mouse: continue with the simulation
.ve

.seealso: `PetscViewerPushFormat()`, `PetscViewerASCIIOpen()`, `PetscViewerDrawOpen()`,
          `PetscViewerSocketOpen()`, `PetscViewerBinaryOpen()`, `MatLoad()`
@*/
PetscErrorCode MatView(Mat mat,PetscViewer viewer)
{
  PetscInt          rows,cols,rbs,cbs;
  PetscBool         isascii,isstring,issaws;
  PetscViewerFormat format;
  PetscMPIInt       size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)mat),&viewer));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(mat,1,viewer,2);
  MatCheckPreallocated(mat,1);

  PetscCall(PetscViewerGetFormat(viewer,&format));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size));
  if (size == 1 && format == PETSC_VIEWER_LOAD_BALANCE) PetscFunctionReturn(0);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSAWS,&issaws));
  if ((!isascii || (format != PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL)) && mat->factortype) {
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"No viewers for factored matrix except ASCII info or info_detail");
  }

  PetscCall(PetscLogEventBegin(MAT_View,mat,viewer,0,0));
  if (isascii) {
    PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ORDER,"Must call MatAssemblyBegin/End() before viewing matrix");
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)mat,viewer));
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      MatNullSpace nullsp,transnullsp;

      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(MatGetSize(mat,&rows,&cols));
      PetscCall(MatGetBlockSizes(mat,&rbs,&cbs));
      if (rbs != 1 || cbs != 1) {
        if (rbs != cbs) PetscCall(PetscViewerASCIIPrintf(viewer,"rows=%" PetscInt_FMT ", cols=%" PetscInt_FMT ", rbs=%" PetscInt_FMT ", cbs=%" PetscInt_FMT "\n",rows,cols,rbs,cbs));
        else            PetscCall(PetscViewerASCIIPrintf(viewer,"rows=%" PetscInt_FMT ", cols=%" PetscInt_FMT ", bs=%" PetscInt_FMT "\n",rows,cols,rbs));
      } else PetscCall(PetscViewerASCIIPrintf(viewer,"rows=%" PetscInt_FMT ", cols=%" PetscInt_FMT "\n",rows,cols));
      if (mat->factortype) {
        MatSolverType solver;
        PetscCall(MatFactorGetSolverType(mat,&solver));
        PetscCall(PetscViewerASCIIPrintf(viewer,"package used to perform factorization: %s\n",solver));
      }
      if (mat->ops->getinfo) {
        MatInfo info;
        PetscCall(MatGetInfo(mat,MAT_GLOBAL_SUM,&info));
        PetscCall(PetscViewerASCIIPrintf(viewer,"total: nonzeros=%.f, allocated nonzeros=%.f\n",info.nz_used,info.nz_allocated));
        if (!mat->factortype) PetscCall(PetscViewerASCIIPrintf(viewer,"total number of mallocs used during MatSetValues calls=%" PetscInt_FMT "\n",(PetscInt)info.mallocs));
      }
      PetscCall(MatGetNullSpace(mat,&nullsp));
      PetscCall(MatGetTransposeNullSpace(mat,&transnullsp));
      if (nullsp) PetscCall(PetscViewerASCIIPrintf(viewer,"  has attached null space\n"));
      if (transnullsp && transnullsp != nullsp) PetscCall(PetscViewerASCIIPrintf(viewer,"  has attached transposed null space\n"));
      PetscCall(MatGetNearNullSpace(mat,&nullsp));
      if (nullsp) PetscCall(PetscViewerASCIIPrintf(viewer,"  has attached near null space\n"));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(MatProductView(mat,viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  } else if (issaws) {
#if defined(PETSC_HAVE_SAWS)
    PetscMPIInt rank;

    PetscCall(PetscObjectName((PetscObject)mat));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
    if (!((PetscObject)mat)->amsmem && rank == 0) {
      PetscCall(PetscObjectViewSAWs((PetscObject)mat,viewer));
    }
#endif
  } else if (isstring) {
    const char *type;
    PetscCall(MatGetType(mat,&type));
    PetscCall(PetscViewerStringSPrintf(viewer," MatType: %-7.7s",type));
    if (mat->ops->view) PetscCall((*mat->ops->view)(mat,viewer));
  }
  if ((format == PETSC_VIEWER_NATIVE || format == PETSC_VIEWER_LOAD_BALANCE) && mat->ops->viewnative) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall((*mat->ops->viewnative)(mat,viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  } else if (mat->ops->view) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall((*mat->ops->view)(mat,viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  if (isascii) {
    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  }
  PetscCall(PetscLogEventEnd(MAT_View,mat,viewer,0,0));
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DEBUG)
#include <../src/sys/totalview/tv_data_display.h>
PETSC_UNUSED static int TV_display_type(const struct _p_Mat *mat)
{
  TV_add_row("Local rows", "int", &mat->rmap->n);
  TV_add_row("Local columns", "int", &mat->cmap->n);
  TV_add_row("Global rows", "int", &mat->rmap->N);
  TV_add_row("Global columns", "int", &mat->cmap->N);
  TV_add_row("Typename", TV_ascii_string_type, ((PetscObject)mat)->type_name);
  return TV_format_OK;
}
#endif

/*@C
   MatLoad - Loads a matrix that has been stored in binary/HDF5 format
   with MatView().  The matrix format is determined from the options database.
   Generates a parallel MPI matrix if the communicator has more than one
   processor.  The default matrix type is AIJ.

   Collective on PetscViewer

   Input Parameters:
+  mat - the newly loaded matrix, this needs to have been created with MatCreate()
            or some related function before a call to MatLoad()
-  viewer - binary/HDF5 file viewer

   Options Database Keys:
   Used with block matrix formats (MATSEQBAIJ,  ...) to specify
   block size
.    -matload_block_size <bs> - set block size

   Level: beginner

   Notes:
   If the Mat type has not yet been given then MATAIJ is used, call MatSetFromOptions() on the
   Mat before calling this routine if you wish to set it from the options database.

   MatLoad() automatically loads into the options database any options
   given in the file filename.info where filename is the name of the file
   that was passed to the PetscViewerBinaryOpen(). The options in the info
   file will be ignored if you use the -viewer_binary_skip_info option.

   If the type or size of mat is not set before a call to MatLoad, PETSc
   sets the default matrix type AIJ and sets the local and global sizes.
   If type and/or size is already set, then the same are used.

   In parallel, each processor can load a subset of rows (or the
   entire matrix).  This routine is especially useful when a large
   matrix is stored on disk and only part of it is desired on each
   processor.  For example, a parallel solver may access only some of
   the rows from each processor.  The algorithm used here reads
   relatively small blocks of data rather than reading the entire
   matrix and then subsetting it.

   Viewer's PetscViewerType must be either PETSCVIEWERBINARY or PETSCVIEWERHDF5.
   Such viewer can be created using PetscViewerBinaryOpen()/PetscViewerHDF5Open(),
   or the sequence like
$    PetscViewer v;
$    PetscViewerCreate(PETSC_COMM_WORLD,&v);
$    PetscViewerSetType(v,PETSCVIEWERBINARY);
$    PetscViewerSetFromOptions(v);
$    PetscViewerFileSetMode(v,FILE_MODE_READ);
$    PetscViewerFileSetName(v,"datafile");
   The optional PetscViewerSetFromOptions() call allows to override PetscViewerSetType() using option
$ -viewer_type {binary,hdf5}

   See the example src/ksp/ksp/tutorials/ex27.c with the first approach,
   and src/mat/tutorials/ex10.c with the second approach.

   Notes about the PETSc binary format:
   In case of PETSCVIEWERBINARY, a native PETSc binary format is used. Each of the blocks
   is read onto rank 0 and then shipped to its destination rank, one after another.
   Multiple objects, both matrices and vectors, can be stored within the same file.
   Their PetscObject name is ignored; they are loaded in the order of their storage.

   Most users should not need to know the details of the binary storage
   format, since MatLoad() and MatView() completely hide these details.
   But for anyone who's interested, the standard binary matrix storage
   format is

$    PetscInt    MAT_FILE_CLASSID
$    PetscInt    number of rows
$    PetscInt    number of columns
$    PetscInt    total number of nonzeros
$    PetscInt    *number nonzeros in each row
$    PetscInt    *column indices of all nonzeros (starting index is zero)
$    PetscScalar *values of all nonzeros

   PETSc automatically does the byte swapping for
machines that store the bytes reversed, e.g.  DEC alpha, freebsd,
Linux, Microsoft Windows and the Intel Paragon; thus if you write your own binary
read/write routines you have to swap the bytes; see PetscBinaryRead()
and PetscBinaryWrite() to see how this may be done.

   Notes about the HDF5 (MATLAB MAT-File Version 7.3) format:
   In case of PETSCVIEWERHDF5, a parallel HDF5 reader is used.
   Each processor's chunk is loaded independently by its owning rank.
   Multiple objects, both matrices and vectors, can be stored within the same file.
   They are looked up by their PetscObject name.

   As the MATLAB MAT-File Version 7.3 format is also a HDF5 flavor, we decided to use
   by default the same structure and naming of the AIJ arrays and column count
   within the HDF5 file. This means that a MAT file saved with -v7.3 flag, e.g.
$    save example.mat A b -v7.3
   can be directly read by this routine (see Reference 1 for details).
   Note that depending on your MATLAB version, this format might be a default,
   otherwise you can set it as default in Preferences.

   Unless -nocompression flag is used to save the file in MATLAB,
   PETSc must be configured with ZLIB package.

   See also examples src/mat/tutorials/ex10.c and src/ksp/ksp/tutorials/ex27.c

   Current HDF5 (MAT-File) limitations:
   This reader currently supports only real MATSEQAIJ, MATMPIAIJ, MATSEQDENSE and MATMPIDENSE matrices.

   Corresponding MatView() is not yet implemented.

   The loaded matrix is actually a transpose of the original one in MATLAB,
   unless you push PETSC_VIEWER_HDF5_MAT format (see examples above).
   With this format, matrix is automatically transposed by PETSc,
   unless the matrix is marked as SPD or symmetric
   (see MatSetOption(), MAT_SPD, MAT_SYMMETRIC).

   References:
.  * - MATLAB(R) Documentation, manual page of save(), https://www.mathworks.com/help/matlab/ref/save.html#btox10b-1-version

.seealso: `PetscViewerBinaryOpen()`, `PetscViewerSetType()`, `MatView()`, `VecLoad()`

 @*/
PetscErrorCode MatLoad(Mat mat,PetscViewer viewer)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);

  if (!((PetscObject)mat)->type_name) PetscCall(MatSetType(mat,MATAIJ));

  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(((PetscObject)mat)->options,((PetscObject)mat)->prefix,"-matload_symmetric",&flg,NULL));
  if (flg) {
    PetscCall(MatSetOption(mat,MAT_SYMMETRIC,PETSC_TRUE));
    PetscCall(MatSetOption(mat,MAT_SYMMETRY_ETERNAL,PETSC_TRUE));
  }
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(((PetscObject)mat)->options,((PetscObject)mat)->prefix,"-matload_spd",&flg,NULL));
  if (flg) PetscCall(MatSetOption(mat,MAT_SPD,PETSC_TRUE));

  PetscCheck(mat->ops->load,PETSC_COMM_SELF,PETSC_ERR_SUP,"MatLoad is not supported for type %s",((PetscObject)mat)->type_name);
  PetscCall(PetscLogEventBegin(MAT_Load,mat,viewer,0,0));
  PetscCall((*mat->ops->load)(mat,viewer));
  PetscCall(PetscLogEventEnd(MAT_Load,mat,viewer,0,0));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_Redundant(Mat_Redundant **redundant)
{
  Mat_Redundant *redund = *redundant;

  PetscFunctionBegin;
  if (redund) {
    if (redund->matseq) { /* via MatCreateSubMatrices()  */
      PetscCall(ISDestroy(&redund->isrow));
      PetscCall(ISDestroy(&redund->iscol));
      PetscCall(MatDestroySubMatrices(1,&redund->matseq));
    } else {
      PetscCall(PetscFree2(redund->send_rank,redund->recv_rank));
      PetscCall(PetscFree(redund->sbuf_j));
      PetscCall(PetscFree(redund->sbuf_a));
      for (PetscInt i=0; i<redund->nrecvs; i++) {
        PetscCall(PetscFree(redund->rbuf_j[i]));
        PetscCall(PetscFree(redund->rbuf_a[i]));
      }
      PetscCall(PetscFree4(redund->sbuf_nz,redund->rbuf_nz,redund->rbuf_j,redund->rbuf_a));
    }

    if (redund->subcomm) PetscCall(PetscCommDestroy(&redund->subcomm));
    PetscCall(PetscFree(redund));
  }
  PetscFunctionReturn(0);
}

/*@C
   MatDestroy - Frees space taken by a matrix.

   Collective on Mat

   Input Parameter:
.  A - the matrix

   Level: beginner

   Developer Notes:
   Some special arrays of matrices are not destroyed in this routine but instead by the routines called by
   MatDestroySubMatrices(). Thus one must be sure that any changes here must also be made in those routines.
   MatHeaderMerge() and MatHeaderReplace() also manipulate the data in the Mat object and likely need changes
   if changes are needed here.
@*/
PetscErrorCode MatDestroy(Mat *A)
{
  PetscFunctionBegin;
  if (!*A) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*A,MAT_CLASSID,1);
  if (--((PetscObject)(*A))->refct > 0) {*A = NULL; PetscFunctionReturn(0);}

  /* if memory was published with SAWs then destroy it */
  PetscCall(PetscObjectSAWsViewOff((PetscObject)*A));
  if ((*A)->ops->destroy) PetscCall((*(*A)->ops->destroy)(*A));

  PetscCall(PetscFree((*A)->factorprefix));
  PetscCall(PetscFree((*A)->defaultvectype));
  PetscCall(PetscFree((*A)->bsizes));
  PetscCall(PetscFree((*A)->solvertype));
  for (PetscInt i=0; i<MAT_FACTOR_NUM_TYPES; i++) PetscCall(PetscFree((*A)->preferredordering[i]));
  if ((*A)->redundant && (*A)->redundant->matseq[0] == *A) (*A)->redundant->matseq[0] = NULL;
  PetscCall(MatDestroy_Redundant(&(*A)->redundant));
  PetscCall(MatProductClear(*A));
  PetscCall(MatNullSpaceDestroy(&(*A)->nullsp));
  PetscCall(MatNullSpaceDestroy(&(*A)->transnullsp));
  PetscCall(MatNullSpaceDestroy(&(*A)->nearnullsp));
  PetscCall(MatDestroy(&(*A)->schur));
  PetscCall(PetscLayoutDestroy(&(*A)->rmap));
  PetscCall(PetscLayoutDestroy(&(*A)->cmap));
  PetscCall(PetscHeaderDestroy(A));
  PetscFunctionReturn(0);
}

/*@C
   MatSetValues - Inserts or adds a block of values into a matrix.
   These values may be cached, so MatAssemblyBegin() and MatAssemblyEnd()
   MUST be called after all calls to MatSetValues() have been completed.

   Not Collective

   Input Parameters:
+  mat - the matrix
.  v - a logically two-dimensional array of values
.  m, idxm - the number of rows and their global indices
.  n, idxn - the number of columns and their global indices
-  addv - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   If you create the matrix yourself (that is not with a call to DMCreateMatrix()) then you MUST call MatXXXXSetPreallocation() or
      MatSetUp() before using this routine

   By default the values, v, are row-oriented. See MatSetOption() for other options.

   Calls to MatSetValues() with the INSERT_VALUES and ADD_VALUES
   options cannot be mixed without intervening calls to the assembly
   routines.

   MatSetValues() uses 0-based row and column numbers in Fortran
   as well as in C.

   Negative indices may be passed in idxm and idxn, these rows and columns are
   simply ignored. This allows easily inserting element stiffness matrices
   with homogeneous Dirchlet boundary conditions that you don't want represented
   in the matrix.

   Efficiency Alert:
   The routine MatSetValuesBlocked() may offer much better efficiency
   for users of block sparse formats (MATSEQBAIJ and MATMPIBAIJ).

   Level: beginner

   Developer Notes:
    This is labeled with C so does not automatically generate Fortran stubs and interfaces
                    because it requires multiple Fortran interfaces depending on which arguments are scalar or arrays.

.seealso: `MatSetOption()`, `MatAssemblyBegin()`, `MatAssemblyEnd()`, `MatSetValuesBlocked()`, `MatSetValuesLocal()`,
          `InsertMode`, `INSERT_VALUES`, `ADD_VALUES`
@*/
PetscErrorCode MatSetValues(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv)
{
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (!m || !n) PetscFunctionReturn(0); /* no values to insert */
  PetscValidIntPointer(idxm,3);
  PetscValidIntPointer(idxn,5);
  MatCheckPreallocated(mat,1);

  if (mat->insertmode == NOT_SET_VALUES) mat->insertmode = addv;
  else PetscCheck(mat->insertmode == addv,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");

  if (PetscDefined(USE_DEBUG)) {
    PetscInt       i,j;

    PetscCheck(!mat->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
    PetscCheck(mat->ops->setvalues,PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);

    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (mat->erroriffailure && PetscIsInfOrNanScalar(v[i*n+j]))
#if defined(PETSC_USE_COMPLEX)
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Inserting %g+i%g at matrix entry (%" PetscInt_FMT ",%" PetscInt_FMT ")",(double)PetscRealPart(v[i*n+j]),(double)PetscImaginaryPart(v[i*n+j]),idxm[i],idxn[j]);
#else
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Inserting %g at matrix entry (%" PetscInt_FMT ",%" PetscInt_FMT ")",(double)v[i*n+j],idxm[i],idxn[j]);
#endif
      }
    }
    for (i=0; i<m; i++) PetscCheck(idxm[i] < mat->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot insert in row %" PetscInt_FMT ", maximum is %" PetscInt_FMT,idxm[i],mat->rmap->N-1);
    for (i=0; i<n; i++) PetscCheck(idxn[i] < mat->cmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot insert in column %" PetscInt_FMT ", maximum is %" PetscInt_FMT,idxn[i],mat->cmap->N-1);
  }

  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE;
    mat->assembled     = PETSC_FALSE;
  }
  PetscCall(PetscLogEventBegin(MAT_SetValues,mat,0,0,0));
  PetscCall((*mat->ops->setvalues)(mat,m,idxm,n,idxn,v,addv));
  PetscCall(PetscLogEventEnd(MAT_SetValues,mat,0,0,0));
  PetscFunctionReturn(0);
}

/*@C
   MatSetValuesIS - Inserts or adds a block of values into a matrix using IS to indicate the rows and columns
   These values may be cached, so MatAssemblyBegin() and MatAssemblyEnd()
   MUST be called after all calls to MatSetValues() have been completed.

   Not Collective

   Input Parameters:
+  mat - the matrix
.  v - a logically two-dimensional array of values
.  ism - the rows to provide
.  isn - the columns to provide
-  addv - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   If you create the matrix yourself (that is not with a call to DMCreateMatrix()) then you MUST call MatXXXXSetPreallocation() or
      MatSetUp() before using this routine

   By default the values, v, are row-oriented. See MatSetOption() for other options.

   Calls to MatSetValues() with the INSERT_VALUES and ADD_VALUES
   options cannot be mixed without intervening calls to the assembly
   routines.

   MatSetValues() uses 0-based row and column numbers in Fortran
   as well as in C.

   Negative indices may be passed in ism and isn, these rows and columns are
   simply ignored. This allows easily inserting element stiffness matrices
   with homogeneous Dirchlet boundary conditions that you don't want represented
   in the matrix.

   Efficiency Alert:
   The routine MatSetValuesBlocked() may offer much better efficiency
   for users of block sparse formats (MATSEQBAIJ and MATMPIBAIJ).

   Level: beginner

   Developer Notes:
    This is labeled with C so does not automatically generate Fortran stubs and interfaces
                    because it requires multiple Fortran interfaces depending on which arguments are scalar or arrays.

    This is currently not optimized for any particular IS type

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValuesBlocked(), MatSetValuesLocal(),
          InsertMode, INSERT_VALUES, ADD_VALUES, MatSetValues()
@*/
PetscErrorCode MatSetValuesIS(Mat mat,IS ism,IS isn,const PetscScalar v[],InsertMode addv)
{
  PetscInt       m,n;
  const PetscInt *rows,*cols;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscCall(ISGetIndices(ism,&rows));
  PetscCall(ISGetIndices(isn,&cols));
  PetscCall(ISGetLocalSize(ism,&m));
  PetscCall(ISGetLocalSize(isn,&n));
  PetscCall(MatSetValues(mat,m,rows,n,cols,v,addv));
  PetscCall(ISRestoreIndices(ism,&rows));
  PetscCall(ISRestoreIndices(isn,&cols));
  PetscFunctionReturn(0);
}

/*@
   MatSetValuesRowLocal - Inserts a row (block row for BAIJ matrices) of nonzero
        values into a matrix

   Not Collective

   Input Parameters:
+  mat - the matrix
.  row - the (block) row to set
-  v - a logically two-dimensional array of values

   Notes:
   By the values, v, are column-oriented (for the block version) and sorted

   All the nonzeros in the row must be provided

   The matrix must have previously had its column indices set

   The row must belong to this process

   Level: intermediate

.seealso: `MatSetOption()`, `MatAssemblyBegin()`, `MatAssemblyEnd()`, `MatSetValuesBlocked()`, `MatSetValuesLocal()`,
          `InsertMode`, `INSERT_VALUES`, `ADD_VALUES`, `MatSetValues()`, `MatSetValuesRow()`, `MatSetLocalToGlobalMapping()`
@*/
PetscErrorCode MatSetValuesRowLocal(Mat mat,PetscInt row,const PetscScalar v[])
{
  PetscInt globalrow;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidScalarPointer(v,3);
  PetscCall(ISLocalToGlobalMappingApply(mat->rmap->mapping,1,&row,&globalrow));
  PetscCall(MatSetValuesRow(mat,globalrow,v));
  PetscFunctionReturn(0);
}

/*@
   MatSetValuesRow - Inserts a row (block row for BAIJ matrices) of nonzero
        values into a matrix

   Not Collective

   Input Parameters:
+  mat - the matrix
.  row - the (block) row to set
-  v - a logically two-dimensional (column major) array of values for  block matrices with blocksize larger than one, otherwise a one dimensional array of values

   Notes:
   The values, v, are column-oriented for the block version.

   All the nonzeros in the row must be provided

   THE MATRIX MUST HAVE PREVIOUSLY HAD ITS COLUMN INDICES SET. IT IS RARE THAT THIS ROUTINE IS USED, usually MatSetValues() is used.

   The row must belong to this process

   Level: advanced

.seealso: `MatSetOption()`, `MatAssemblyBegin()`, `MatAssemblyEnd()`, `MatSetValuesBlocked()`, `MatSetValuesLocal()`,
          `InsertMode`, `INSERT_VALUES`, `ADD_VALUES`, `MatSetValues()`
@*/
PetscErrorCode MatSetValuesRow(Mat mat,PetscInt row,const PetscScalar v[])
{
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  MatCheckPreallocated(mat,1);
  PetscValidScalarPointer(v,3);
  PetscCheck(mat->insertmode != ADD_VALUES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add and insert values");
  PetscCheck(!mat->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  mat->insertmode = INSERT_VALUES;

  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE;
    mat->assembled     = PETSC_FALSE;
  }
  PetscCall(PetscLogEventBegin(MAT_SetValues,mat,0,0,0));
  PetscCheck(mat->ops->setvaluesrow,PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  PetscCall((*mat->ops->setvaluesrow)(mat,row,v));
  PetscCall(PetscLogEventEnd(MAT_SetValues,mat,0,0,0));
  PetscFunctionReturn(0);
}

/*@
   MatSetValuesStencil - Inserts or adds a block of values into a matrix.
     Using structured grid indexing

   Not Collective

   Input Parameters:
+  mat - the matrix
.  m - number of rows being entered
.  idxm - grid coordinates (and component number when dof > 1) for matrix rows being entered
.  n - number of columns being entered
.  idxn - grid coordinates (and component number when dof > 1) for matrix columns being entered
.  v - a logically two-dimensional array of values
-  addv - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   By default the values, v, are row-oriented.  See MatSetOption() for other options.

   Calls to MatSetValuesStencil() with the INSERT_VALUES and ADD_VALUES
   options cannot be mixed without intervening calls to the assembly
   routines.

   The grid coordinates are across the entire grid, not just the local portion

   MatSetValuesStencil() uses 0-based row and column numbers in Fortran
   as well as in C.

   For setting/accessing vector values via array coordinates you can use the DMDAVecGetArray() routine

   In order to use this routine you must either obtain the matrix with DMCreateMatrix()
   or call MatSetLocalToGlobalMapping() and MatSetStencil() first.

   The columns and rows in the stencil passed in MUST be contained within the
   ghost region of the given process as set with DMDACreateXXX() or MatSetStencil(). For example,
   if you create a DMDA with an overlap of one grid level and on a particular process its first
   local nonghost x logical coordinate is 6 (so its first ghost x logical coordinate is 5) the
   first i index you can use in your column and row indices in MatSetStencil() is 5.

   In Fortran idxm and idxn should be declared as
$     MatStencil idxm(4,m),idxn(4,n)
   and the values inserted using
$    idxm(MatStencil_i,1) = i
$    idxm(MatStencil_j,1) = j
$    idxm(MatStencil_k,1) = k
$    idxm(MatStencil_c,1) = c
   etc

   For periodic boundary conditions use negative indices for values to the left (below 0; that are to be
   obtained by wrapping values from right edge). For values to the right of the last entry using that index plus one
   etc to obtain values that obtained by wrapping the values from the left edge. This does not work for anything but the
   DM_BOUNDARY_PERIODIC boundary type.

   For indices that don't mean anything for your case (like the k index when working in 2d) or the c index when you have
   a single value per point) you can skip filling those indices.

   Inspired by the structured grid interface to the HYPRE package
   (https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods)

   Efficiency Alert:
   The routine MatSetValuesBlockedStencil() may offer much better efficiency
   for users of block sparse formats (MATSEQBAIJ and MATMPIBAIJ).

   Level: beginner

.seealso: `MatSetOption()`, `MatAssemblyBegin()`, `MatAssemblyEnd()`, `MatSetValuesBlocked()`, `MatSetValuesLocal()`
          `MatSetValues()`, `MatSetValuesBlockedStencil()`, `MatSetStencil()`, `DMCreateMatrix()`, `DMDAVecGetArray()`, `MatStencil`
@*/
PetscErrorCode MatSetValuesStencil(Mat mat,PetscInt m,const MatStencil idxm[],PetscInt n,const MatStencil idxn[],const PetscScalar v[],InsertMode addv)
{
  PetscInt       buf[8192],*bufm=NULL,*bufn=NULL,*jdxm,*jdxn;
  PetscInt       j,i,dim = mat->stencil.dim,*dims = mat->stencil.dims+1,tmp;
  PetscInt       *starts = mat->stencil.starts,*dxm = (PetscInt*)idxm,*dxn = (PetscInt*)idxn,sdim = dim - (1 - (PetscInt)mat->stencil.noc);

  PetscFunctionBegin;
  if (!m || !n) PetscFunctionReturn(0); /* no values to insert */
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidPointer(idxm,3);
  PetscValidPointer(idxn,5);

  if ((m+n) <= (PetscInt)(sizeof(buf)/sizeof(PetscInt))) {
    jdxm = buf; jdxn = buf+m;
  } else {
    PetscCall(PetscMalloc2(m,&bufm,n,&bufn));
    jdxm = bufm; jdxn = bufn;
  }
  for (i=0; i<m; i++) {
    for (j=0; j<3-sdim; j++) dxm++;
    tmp = *dxm++ - starts[0];
    for (j=0; j<dim-1; j++) {
      if ((*dxm++ - starts[j+1]) < 0 || tmp < 0) tmp = -1;
      else                                       tmp = tmp*dims[j] + *(dxm-1) - starts[j+1];
    }
    if (mat->stencil.noc) dxm++;
    jdxm[i] = tmp;
  }
  for (i=0; i<n; i++) {
    for (j=0; j<3-sdim; j++) dxn++;
    tmp = *dxn++ - starts[0];
    for (j=0; j<dim-1; j++) {
      if ((*dxn++ - starts[j+1]) < 0 || tmp < 0) tmp = -1;
      else                                       tmp = tmp*dims[j] + *(dxn-1) - starts[j+1];
    }
    if (mat->stencil.noc) dxn++;
    jdxn[i] = tmp;
  }
  PetscCall(MatSetValuesLocal(mat,m,jdxm,n,jdxn,v,addv));
  PetscCall(PetscFree2(bufm,bufn));
  PetscFunctionReturn(0);
}

/*@
   MatSetValuesBlockedStencil - Inserts or adds a block of values into a matrix.
     Using structured grid indexing

   Not Collective

   Input Parameters:
+  mat - the matrix
.  m - number of rows being entered
.  idxm - grid coordinates for matrix rows being entered
.  n - number of columns being entered
.  idxn - grid coordinates for matrix columns being entered
.  v - a logically two-dimensional array of values
-  addv - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   By default the values, v, are row-oriented and unsorted.
   See MatSetOption() for other options.

   Calls to MatSetValuesBlockedStencil() with the INSERT_VALUES and ADD_VALUES
   options cannot be mixed without intervening calls to the assembly
   routines.

   The grid coordinates are across the entire grid, not just the local portion

   MatSetValuesBlockedStencil() uses 0-based row and column numbers in Fortran
   as well as in C.

   For setting/accessing vector values via array coordinates you can use the DMDAVecGetArray() routine

   In order to use this routine you must either obtain the matrix with DMCreateMatrix()
   or call MatSetBlockSize(), MatSetLocalToGlobalMapping() and MatSetStencil() first.

   The columns and rows in the stencil passed in MUST be contained within the
   ghost region of the given process as set with DMDACreateXXX() or MatSetStencil(). For example,
   if you create a DMDA with an overlap of one grid level and on a particular process its first
   local nonghost x logical coordinate is 6 (so its first ghost x logical coordinate is 5) the
   first i index you can use in your column and row indices in MatSetStencil() is 5.

   In Fortran idxm and idxn should be declared as
$     MatStencil idxm(4,m),idxn(4,n)
   and the values inserted using
$    idxm(MatStencil_i,1) = i
$    idxm(MatStencil_j,1) = j
$    idxm(MatStencil_k,1) = k
   etc

   Negative indices may be passed in idxm and idxn, these rows and columns are
   simply ignored. This allows easily inserting element stiffness matrices
   with homogeneous Dirchlet boundary conditions that you don't want represented
   in the matrix.

   Inspired by the structured grid interface to the HYPRE package
   (https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods)

   Level: beginner

.seealso: `MatSetOption()`, `MatAssemblyBegin()`, `MatAssemblyEnd()`, `MatSetValuesBlocked()`, `MatSetValuesLocal()`
          `MatSetValues()`, `MatSetValuesStencil()`, `MatSetStencil()`, `DMCreateMatrix()`, `DMDAVecGetArray()`, `MatStencil`,
          `MatSetBlockSize()`, `MatSetLocalToGlobalMapping()`
@*/
PetscErrorCode MatSetValuesBlockedStencil(Mat mat,PetscInt m,const MatStencil idxm[],PetscInt n,const MatStencil idxn[],const PetscScalar v[],InsertMode addv)
{
  PetscInt       buf[8192],*bufm=NULL,*bufn=NULL,*jdxm,*jdxn;
  PetscInt       j,i,dim = mat->stencil.dim,*dims = mat->stencil.dims+1,tmp;
  PetscInt       *starts = mat->stencil.starts,*dxm = (PetscInt*)idxm,*dxn = (PetscInt*)idxn,sdim = dim - (1 - (PetscInt)mat->stencil.noc);

  PetscFunctionBegin;
  if (!m || !n) PetscFunctionReturn(0); /* no values to insert */
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidPointer(idxm,3);
  PetscValidPointer(idxn,5);
  PetscValidScalarPointer(v,6);

  if ((m+n) <= (PetscInt)(sizeof(buf)/sizeof(PetscInt))) {
    jdxm = buf; jdxn = buf+m;
  } else {
    PetscCall(PetscMalloc2(m,&bufm,n,&bufn));
    jdxm = bufm; jdxn = bufn;
  }
  for (i=0; i<m; i++) {
    for (j=0; j<3-sdim; j++) dxm++;
    tmp = *dxm++ - starts[0];
    for (j=0; j<sdim-1; j++) {
      if ((*dxm++ - starts[j+1]) < 0 || tmp < 0) tmp = -1;
      else                                       tmp = tmp*dims[j] + *(dxm-1) - starts[j+1];
    }
    dxm++;
    jdxm[i] = tmp;
  }
  for (i=0; i<n; i++) {
    for (j=0; j<3-sdim; j++) dxn++;
    tmp = *dxn++ - starts[0];
    for (j=0; j<sdim-1; j++) {
      if ((*dxn++ - starts[j+1]) < 0 || tmp < 0) tmp = -1;
      else                                       tmp = tmp*dims[j] + *(dxn-1) - starts[j+1];
    }
    dxn++;
    jdxn[i] = tmp;
  }
  PetscCall(MatSetValuesBlockedLocal(mat,m,jdxm,n,jdxn,v,addv));
  PetscCall(PetscFree2(bufm,bufn));
  PetscFunctionReturn(0);
}

/*@
   MatSetStencil - Sets the grid information for setting values into a matrix via
        MatSetValuesStencil()

   Not Collective

   Input Parameters:
+  mat - the matrix
.  dim - dimension of the grid 1, 2, or 3
.  dims - number of grid points in x, y, and z direction, including ghost points on your processor
.  starts - starting point of ghost nodes on your processor in x, y, and z direction
-  dof - number of degrees of freedom per node

   Inspired by the structured grid interface to the HYPRE package
   (www.llnl.gov/CASC/hyper)

   For matrices generated with DMCreateMatrix() this routine is automatically called and so not needed by the
   user.

   Level: beginner

.seealso: `MatSetOption()`, `MatAssemblyBegin()`, `MatAssemblyEnd()`, `MatSetValuesBlocked()`, `MatSetValuesLocal()`
          `MatSetValues()`, `MatSetValuesBlockedStencil()`, `MatSetValuesStencil()`
@*/
PetscErrorCode MatSetStencil(Mat mat,PetscInt dim,const PetscInt dims[],const PetscInt starts[],PetscInt dof)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidIntPointer(dims,3);
  PetscValidIntPointer(starts,4);

  mat->stencil.dim = dim + (dof > 1);
  for (PetscInt i=0; i<dim; i++) {
    mat->stencil.dims[i]   = dims[dim-i-1];      /* copy the values in backwards */
    mat->stencil.starts[i] = starts[dim-i-1];
  }
  mat->stencil.dims[dim]   = dof;
  mat->stencil.starts[dim] = 0;
  mat->stencil.noc         = (PetscBool)(dof == 1);
  PetscFunctionReturn(0);
}

/*@C
   MatSetValuesBlocked - Inserts or adds a block of values into a matrix.

   Not Collective

   Input Parameters:
+  mat - the matrix
.  v - a logically two-dimensional array of values
.  m, idxm - the number of block rows and their global block indices
.  n, idxn - the number of block columns and their global block indices
-  addv - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   If you create the matrix yourself (that is not with a call to DMCreateMatrix()) then you MUST call
   MatXXXXSetPreallocation() or MatSetUp() before using this routine.

   The m and n count the NUMBER of blocks in the row direction and column direction,
   NOT the total number of rows/columns; for example, if the block size is 2 and
   you are passing in values for rows 2,3,4,5  then m would be 2 (not 4).
   The values in idxm would be 1 2; that is the first index for each block divided by
   the block size.

   Note that you must call MatSetBlockSize() when constructing this matrix (before
   preallocating it).

   By default the values, v, are row-oriented, so the layout of
   v is the same as for MatSetValues(). See MatSetOption() for other options.

   Calls to MatSetValuesBlocked() with the INSERT_VALUES and ADD_VALUES
   options cannot be mixed without intervening calls to the assembly
   routines.

   MatSetValuesBlocked() uses 0-based row and column numbers in Fortran
   as well as in C.

   Negative indices may be passed in idxm and idxn, these rows and columns are
   simply ignored. This allows easily inserting element stiffness matrices
   with homogeneous Dirchlet boundary conditions that you don't want represented
   in the matrix.

   Each time an entry is set within a sparse matrix via MatSetValues(),
   internal searching must be done to determine where to place the
   data in the matrix storage space.  By instead inserting blocks of
   entries via MatSetValuesBlocked(), the overhead of matrix assembly is
   reduced.

   Example:
$   Suppose m=n=2 and block size(bs) = 2 The array is
$
$   1  2  | 3  4
$   5  6  | 7  8
$   - - - | - - -
$   9  10 | 11 12
$   13 14 | 15 16
$
$   v[] should be passed in like
$   v[] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
$
$  If you are not using row oriented storage of v (that is you called MatSetOption(mat,MAT_ROW_ORIENTED,PETSC_FALSE)) then
$   v[] = [1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16]

   Level: intermediate

.seealso: `MatSetBlockSize()`, `MatSetOption()`, `MatAssemblyBegin()`, `MatAssemblyEnd()`, `MatSetValues()`, `MatSetValuesBlockedLocal()`
@*/
PetscErrorCode MatSetValuesBlocked(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv)
{
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (!m || !n) PetscFunctionReturn(0); /* no values to insert */
  PetscValidIntPointer(idxm,3);
  PetscValidIntPointer(idxn,5);
  MatCheckPreallocated(mat,1);
  if (mat->insertmode == NOT_SET_VALUES) mat->insertmode = addv;
  else PetscCheck(mat->insertmode == addv,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
  if (PetscDefined(USE_DEBUG)) {
    PetscCheck(!mat->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
    PetscCheck(mat->ops->setvaluesblocked || mat->ops->setvalues,PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  }
  if (PetscDefined(USE_DEBUG)) {
    PetscInt rbs,cbs,M,N,i;
    PetscCall(MatGetBlockSizes(mat,&rbs,&cbs));
    PetscCall(MatGetSize(mat,&M,&N));
    for (i=0; i<m; i++) PetscCheck(idxm[i]*rbs < M,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row block index %" PetscInt_FMT " (index %" PetscInt_FMT ") greater than row length %" PetscInt_FMT,i,idxm[i],M);
    for (i=0; i<n; i++) PetscCheck(idxn[i]*cbs < N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column block index %" PetscInt_FMT " (index %" PetscInt_FMT ") great than column length %" PetscInt_FMT,i,idxn[i],N);
  }
  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE;
    mat->assembled     = PETSC_FALSE;
  }
  PetscCall(PetscLogEventBegin(MAT_SetValues,mat,0,0,0));
  if (mat->ops->setvaluesblocked) {
    PetscCall((*mat->ops->setvaluesblocked)(mat,m,idxm,n,idxn,v,addv));
  } else {
    PetscInt buf[8192],*bufr=NULL,*bufc=NULL,*iidxm,*iidxn;
    PetscInt i,j,bs,cbs;

    PetscCall(MatGetBlockSizes(mat,&bs,&cbs));
    if (m*bs+n*cbs <= (PetscInt)(sizeof(buf)/sizeof(PetscInt))) {
      iidxm = buf;
      iidxn = buf + m*bs;
    } else {
      PetscCall(PetscMalloc2(m*bs,&bufr,n*cbs,&bufc));
      iidxm = bufr;
      iidxn = bufc;
    }
    for (i=0; i<m; i++) {
      for (j=0; j<bs; j++) {
        iidxm[i*bs+j] = bs*idxm[i] + j;
      }
    }
    if (m != n || bs != cbs || idxm != idxn) {
      for (i=0; i<n; i++) {
        for (j=0; j<cbs; j++) {
          iidxn[i*cbs+j] = cbs*idxn[i] + j;
        }
      }
    } else iidxn = iidxm;
    PetscCall(MatSetValues(mat,m*bs,iidxm,n*cbs,iidxn,v,addv));
    PetscCall(PetscFree2(bufr,bufc));
  }
  PetscCall(PetscLogEventEnd(MAT_SetValues,mat,0,0,0));
  PetscFunctionReturn(0);
}

/*@C
   MatGetValues - Gets a block of values from a matrix.

   Not Collective; can only return values that are owned by the give process

   Input Parameters:
+  mat - the matrix
.  v - a logically two-dimensional array for storing the values
.  m, idxm - the number of rows and their global indices
-  n, idxn - the number of columns and their global indices

   Notes:
     The user must allocate space (m*n PetscScalars) for the values, v.
     The values, v, are then returned in a row-oriented format,
     analogous to that used by default in MatSetValues().

     MatGetValues() uses 0-based row and column numbers in
     Fortran as well as in C.

     MatGetValues() requires that the matrix has been assembled
     with MatAssemblyBegin()/MatAssemblyEnd().  Thus, calls to
     MatSetValues() and MatGetValues() CANNOT be made in succession
     without intermediate matrix assembly.

     Negative row or column indices will be ignored and those locations in v[] will be
     left unchanged.

     For the standard row-based matrix formats, idxm[] can only contain rows owned by the requesting MPI rank.
     That is, rows with global index greater than or equal to rstart and less than rend where rstart and rend are obtainable
     from MatGetOwnershipRange(mat,&rstart,&rend).

   Level: advanced

.seealso: `MatGetRow()`, `MatCreateSubMatrices()`, `MatSetValues()`, `MatGetOwnershipRange()`, `MatGetValuesLocal()`, `MatGetValue()`
@*/
PetscErrorCode MatGetValues(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],PetscScalar v[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (!m || !n) PetscFunctionReturn(0);
  PetscValidIntPointer(idxm,3);
  PetscValidIntPointer(idxn,5);
  PetscValidScalarPointer(v,6);
  PetscCheck(mat->assembled,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->ops->getvalues,PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,1);

  PetscCall(PetscLogEventBegin(MAT_GetValues,mat,0,0,0));
  PetscCall((*mat->ops->getvalues)(mat,m,idxm,n,idxn,v));
  PetscCall(PetscLogEventEnd(MAT_GetValues,mat,0,0,0));
  PetscFunctionReturn(0);
}

/*@C
   MatGetValuesLocal - retrieves values from certain locations in a matrix using the local numbering of the indices
     defined previously by MatSetLocalToGlobalMapping()

   Not Collective

   Input Parameters:
+  mat - the matrix
.  nrow, irow - number of rows and their local indices
-  ncol, icol - number of columns and their local indices

   Output Parameter:
.  y -  a logically two-dimensional array of values

   Notes:
     If you create the matrix yourself (that is not with a call to DMCreateMatrix()) then you MUST call MatSetLocalToGlobalMapping() before using this routine.

     This routine can only return values that are owned by the requesting MPI rank. That is, for standard matrix formats, rows that, in the global numbering,
     are greater than or equal to rstart and less than rend where rstart and rend are obtainable from MatGetOwnershipRange(mat,&rstart,&rend). One can
     determine if the resulting global row associated with the local row r is owned by the requesting MPI rank by applying the ISLocalToGlobalMapping set
     with MatSetLocalToGlobalMapping().

   Developer Notes:
      This is labelled with C so does not automatically generate Fortran stubs and interfaces
      because it requires multiple Fortran interfaces depending on which arguments are scalar or arrays.

   Level: advanced

.seealso: `MatAssemblyBegin()`, `MatAssemblyEnd()`, `MatSetValues()`, `MatSetLocalToGlobalMapping()`,
          `MatSetValuesLocal()`, `MatGetValues()`
@*/
PetscErrorCode MatGetValuesLocal(Mat mat,PetscInt nrow,const PetscInt irow[],PetscInt ncol,const PetscInt icol[],PetscScalar y[])
{
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  MatCheckPreallocated(mat,1);
  if (!nrow || !ncol) PetscFunctionReturn(0); /* no values to retrieve */
  PetscValidIntPointer(irow,3);
  PetscValidIntPointer(icol,5);
  if (PetscDefined(USE_DEBUG)) {
    PetscCheck(!mat->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
    PetscCheck(mat->ops->getvalueslocal || mat->ops->getvalues,PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  }
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCall(PetscLogEventBegin(MAT_GetValues,mat,0,0,0));
  if (mat->ops->getvalueslocal) {
    PetscCall((*mat->ops->getvalueslocal)(mat,nrow,irow,ncol,icol,y));
  } else {
    PetscInt buf[8192],*bufr=NULL,*bufc=NULL,*irowm,*icolm;
    if ((nrow+ncol) <= (PetscInt)(sizeof(buf)/sizeof(PetscInt))) {
      irowm = buf; icolm = buf+nrow;
    } else {
      PetscCall(PetscMalloc2(nrow,&bufr,ncol,&bufc));
      irowm = bufr; icolm = bufc;
    }
    PetscCheck(mat->rmap->mapping,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"MatGetValuesLocal() cannot proceed without local-to-global row mapping (See MatSetLocalToGlobalMapping()).");
    PetscCheck(mat->cmap->mapping,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"MatGetValuesLocal() cannot proceed without local-to-global column mapping (See MatSetLocalToGlobalMapping()).");
    PetscCall(ISLocalToGlobalMappingApply(mat->rmap->mapping,nrow,irow,irowm));
    PetscCall(ISLocalToGlobalMappingApply(mat->cmap->mapping,ncol,icol,icolm));
    PetscCall(MatGetValues(mat,nrow,irowm,ncol,icolm,y));
    PetscCall(PetscFree2(bufr,bufc));
  }
  PetscCall(PetscLogEventEnd(MAT_GetValues,mat,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  MatSetValuesBatch - Adds (ADD_VALUES) many blocks of values into a matrix at once. The blocks must all be square and
  the same size. Currently, this can only be called once and creates the given matrix.

  Not Collective

  Input Parameters:
+ mat - the matrix
. nb - the number of blocks
. bs - the number of rows (and columns) in each block
. rows - a concatenation of the rows for each block
- v - a concatenation of logically two-dimensional arrays of values

  Notes:
  In the future, we will extend this routine to handle rectangular blocks, and to allow multiple calls for a given matrix.

  Level: advanced

.seealso: `MatSetOption()`, `MatAssemblyBegin()`, `MatAssemblyEnd()`, `MatSetValuesBlocked()`, `MatSetValuesLocal()`,
          `InsertMode`, `INSERT_VALUES`, `ADD_VALUES`, `MatSetValues()`
@*/
PetscErrorCode MatSetValuesBatch(Mat mat, PetscInt nb, PetscInt bs, PetscInt rows[], const PetscScalar v[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidIntPointer(rows,4);
  PetscValidScalarPointer(v,5);
  PetscAssert(!mat->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  PetscCall(PetscLogEventBegin(MAT_SetValuesBatch,mat,0,0,0));
  if (mat->ops->setvaluesbatch) {
    PetscCall((*mat->ops->setvaluesbatch)(mat,nb,bs,rows,v));
  } else {
    for (PetscInt b = 0; b < nb; ++b) PetscCall(MatSetValues(mat, bs, &rows[b*bs], bs, &rows[b*bs], &v[b*bs*bs], ADD_VALUES));
  }
  PetscCall(PetscLogEventEnd(MAT_SetValuesBatch,mat,0,0,0));
  PetscFunctionReturn(0);
}

/*@
   MatSetLocalToGlobalMapping - Sets a local-to-global numbering for use by
   the routine MatSetValuesLocal() to allow users to insert matrix entries
   using a local (per-processor) numbering.

   Not Collective

   Input Parameters:
+  x - the matrix
.  rmapping - row mapping created with ISLocalToGlobalMappingCreate() or ISLocalToGlobalMappingCreateIS()
-  cmapping - column mapping

   Level: intermediate

.seealso: `MatAssemblyBegin()`, `MatAssemblyEnd()`, `MatSetValues()`, `MatSetValuesLocal()`, `MatGetValuesLocal()`
@*/
PetscErrorCode MatSetLocalToGlobalMapping(Mat x,ISLocalToGlobalMapping rmapping,ISLocalToGlobalMapping cmapping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,MAT_CLASSID,1);
  PetscValidType(x,1);
  if (rmapping) PetscValidHeaderSpecific(rmapping,IS_LTOGM_CLASSID,2);
  if (cmapping) PetscValidHeaderSpecific(cmapping,IS_LTOGM_CLASSID,3);
  if (x->ops->setlocaltoglobalmapping) {
    PetscCall((*x->ops->setlocaltoglobalmapping)(x,rmapping,cmapping));
  } else {
    PetscCall(PetscLayoutSetISLocalToGlobalMapping(x->rmap,rmapping));
    PetscCall(PetscLayoutSetISLocalToGlobalMapping(x->cmap,cmapping));
  }
  PetscFunctionReturn(0);
}

/*@
   MatGetLocalToGlobalMapping - Gets the local-to-global numbering set by MatSetLocalToGlobalMapping()

   Not Collective

   Input Parameter:
.  A - the matrix

   Output Parameters:
+ rmapping - row mapping
- cmapping - column mapping

   Level: advanced

.seealso: `MatSetValuesLocal()`
@*/
PetscErrorCode MatGetLocalToGlobalMapping(Mat A,ISLocalToGlobalMapping *rmapping,ISLocalToGlobalMapping *cmapping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  if (rmapping) {
    PetscValidPointer(rmapping,2);
    *rmapping = A->rmap->mapping;
  }
  if (cmapping) {
    PetscValidPointer(cmapping,3);
    *cmapping = A->cmap->mapping;
  }
  PetscFunctionReturn(0);
}

/*@
   MatSetLayouts - Sets the PetscLayout objects for rows and columns of a matrix

   Logically Collective on A

   Input Parameters:
+  A - the matrix
. rmap - row layout
- cmap - column layout

   Level: advanced

.seealso: `MatCreateVecs()`, `MatGetLocalToGlobalMapping()`, `MatGetLayouts()`
@*/
PetscErrorCode MatSetLayouts(Mat A,PetscLayout rmap,PetscLayout cmap)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscCall(PetscLayoutReference(rmap,&A->rmap));
  PetscCall(PetscLayoutReference(cmap,&A->cmap));
  PetscFunctionReturn(0);
}

/*@
   MatGetLayouts - Gets the PetscLayout objects for rows and columns

   Not Collective

   Input Parameter:
.  A - the matrix

   Output Parameters:
+ rmap - row layout
- cmap - column layout

   Level: advanced

.seealso: `MatCreateVecs()`, `MatGetLocalToGlobalMapping()`, `MatSetLayouts()`
@*/
PetscErrorCode MatGetLayouts(Mat A,PetscLayout *rmap,PetscLayout *cmap)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  if (rmap) {
    PetscValidPointer(rmap,2);
    *rmap = A->rmap;
  }
  if (cmap) {
    PetscValidPointer(cmap,3);
    *cmap = A->cmap;
  }
  PetscFunctionReturn(0);
}

/*@C
   MatSetValuesLocal - Inserts or adds values into certain locations of a matrix,
   using a local numbering of the nodes.

   Not Collective

   Input Parameters:
+  mat - the matrix
.  nrow, irow - number of rows and their local indices
.  ncol, icol - number of columns and their local indices
.  y -  a logically two-dimensional array of values
-  addv - either INSERT_VALUES or ADD_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   If you create the matrix yourself (that is not with a call to DMCreateMatrix()) then you MUST call MatXXXXSetPreallocation() or
      MatSetUp() before using this routine

   If you create the matrix yourself (that is not with a call to DMCreateMatrix()) then you MUST call MatSetLocalToGlobalMapping() before using this routine

   Calls to MatSetValuesLocal() with the INSERT_VALUES and ADD_VALUES
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so MatAssemblyBegin() and MatAssemblyEnd()
   MUST be called after all calls to MatSetValuesLocal() have been completed.

   Level: intermediate

   Developer Notes:
    This is labeled with C so does not automatically generate Fortran stubs and interfaces
                    because it requires multiple Fortran interfaces depending on which arguments are scalar or arrays.

.seealso: `MatAssemblyBegin()`, `MatAssemblyEnd()`, `MatSetValues()`, `MatSetLocalToGlobalMapping()`,
          `MatSetValueLocal()`, `MatGetValuesLocal()`
@*/
PetscErrorCode MatSetValuesLocal(Mat mat,PetscInt nrow,const PetscInt irow[],PetscInt ncol,const PetscInt icol[],const PetscScalar y[],InsertMode addv)
{
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  MatCheckPreallocated(mat,1);
  if (!nrow || !ncol) PetscFunctionReturn(0); /* no values to insert */
  PetscValidIntPointer(irow,3);
  PetscValidIntPointer(icol,5);
  if (mat->insertmode == NOT_SET_VALUES) mat->insertmode = addv;
  else PetscCheck(mat->insertmode == addv,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
  if (PetscDefined(USE_DEBUG)) {
    PetscCheck(!mat->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
    PetscCheck(mat->ops->setvalueslocal || mat->ops->setvalues,PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  }

  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE;
    mat->assembled     = PETSC_FALSE;
  }
  PetscCall(PetscLogEventBegin(MAT_SetValues,mat,0,0,0));
  if (mat->ops->setvalueslocal) {
    PetscCall((*mat->ops->setvalueslocal)(mat,nrow,irow,ncol,icol,y,addv));
  } else {
    PetscInt       buf[8192],*bufr=NULL,*bufc=NULL;
    const PetscInt *irowm,*icolm;

    if ((!mat->rmap->mapping && !mat->cmap->mapping) || (nrow+ncol) <= (PetscInt)(sizeof(buf)/sizeof(PetscInt))) {
      bufr  = buf;
      bufc  = buf + nrow;
      irowm = bufr;
      icolm = bufc;
    } else {
      PetscCall(PetscMalloc2(nrow,&bufr,ncol,&bufc));
      irowm = bufr;
      icolm = bufc;
    }
    if (mat->rmap->mapping) PetscCall(ISLocalToGlobalMappingApply(mat->rmap->mapping,nrow,irow,bufr));
    else irowm = irow;
    if (mat->cmap->mapping) {
      if (mat->cmap->mapping != mat->rmap->mapping || ncol != nrow || icol != irow) {
        PetscCall(ISLocalToGlobalMappingApply(mat->cmap->mapping,ncol,icol,bufc));
      } else icolm = irowm;
    } else icolm = icol;
    PetscCall(MatSetValues(mat,nrow,irowm,ncol,icolm,y,addv));
    if (bufr != buf) PetscCall(PetscFree2(bufr,bufc));
  }
  PetscCall(PetscLogEventEnd(MAT_SetValues,mat,0,0,0));
  PetscFunctionReturn(0);
}

/*@C
   MatSetValuesBlockedLocal - Inserts or adds values into certain locations of a matrix,
   using a local ordering of the nodes a block at a time.

   Not Collective

   Input Parameters:
+  x - the matrix
.  nrow, irow - number of rows and their local indices
.  ncol, icol - number of columns and their local indices
.  y -  a logically two-dimensional array of values
-  addv - either INSERT_VALUES or ADD_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   If you create the matrix yourself (that is not with a call to DMCreateMatrix()) then you MUST call MatXXXXSetPreallocation() or
      MatSetUp() before using this routine

   If you create the matrix yourself (that is not with a call to DMCreateMatrix()) then you MUST call MatSetBlockSize() and MatSetLocalToGlobalMapping()
      before using this routineBefore calling MatSetValuesLocal(), the user must first set the

   Calls to MatSetValuesBlockedLocal() with the INSERT_VALUES and ADD_VALUES
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so MatAssemblyBegin() and MatAssemblyEnd()
   MUST be called after all calls to MatSetValuesBlockedLocal() have been completed.

   Level: intermediate

   Developer Notes:
    This is labeled with C so does not automatically generate Fortran stubs and interfaces
                    because it requires multiple Fortran interfaces depending on which arguments are scalar or arrays.

.seealso: `MatSetBlockSize()`, `MatSetLocalToGlobalMapping()`, `MatAssemblyBegin()`, `MatAssemblyEnd()`,
          `MatSetValuesLocal()`, `MatSetValuesBlocked()`
@*/
PetscErrorCode MatSetValuesBlockedLocal(Mat mat,PetscInt nrow,const PetscInt irow[],PetscInt ncol,const PetscInt icol[],const PetscScalar y[],InsertMode addv)
{
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  MatCheckPreallocated(mat,1);
  if (!nrow || !ncol) PetscFunctionReturn(0); /* no values to insert */
  PetscValidIntPointer(irow,3);
  PetscValidIntPointer(icol,5);
  if (mat->insertmode == NOT_SET_VALUES) mat->insertmode = addv;
  else PetscCheck(mat->insertmode == addv,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
  if (PetscDefined(USE_DEBUG)) {
    PetscCheck(!mat->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
    PetscCheck(mat->ops->setvaluesblockedlocal || mat->ops->setvaluesblocked || mat->ops->setvalueslocal || mat->ops->setvalues,PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  }

  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE;
    mat->assembled     = PETSC_FALSE;
  }
  if (PetscUnlikelyDebug(mat->rmap->mapping)) { /* Condition on the mapping existing, because MatSetValuesBlockedLocal_IS does not require it to be set. */
    PetscInt irbs, rbs;
    PetscCall(MatGetBlockSizes(mat, &rbs, NULL));
    PetscCall(ISLocalToGlobalMappingGetBlockSize(mat->rmap->mapping,&irbs));
    PetscCheck(rbs == irbs,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Different row block sizes! mat %" PetscInt_FMT ", row l2g map %" PetscInt_FMT,rbs,irbs);
  }
  if (PetscUnlikelyDebug(mat->cmap->mapping)) {
    PetscInt icbs, cbs;
    PetscCall(MatGetBlockSizes(mat,NULL,&cbs));
    PetscCall(ISLocalToGlobalMappingGetBlockSize(mat->cmap->mapping,&icbs));
    PetscCheck(cbs == icbs,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Different col block sizes! mat %" PetscInt_FMT ", col l2g map %" PetscInt_FMT,cbs,icbs);
  }
  PetscCall(PetscLogEventBegin(MAT_SetValues,mat,0,0,0));
  if (mat->ops->setvaluesblockedlocal) {
    PetscCall((*mat->ops->setvaluesblockedlocal)(mat,nrow,irow,ncol,icol,y,addv));
  } else {
    PetscInt       buf[8192],*bufr=NULL,*bufc=NULL;
    const PetscInt *irowm,*icolm;

    if ((!mat->rmap->mapping && !mat->cmap->mapping) || (nrow+ncol) <= (PetscInt)(sizeof(buf)/sizeof(PetscInt))) {
      bufr  = buf;
      bufc  = buf + nrow;
      irowm = bufr;
      icolm = bufc;
    } else {
      PetscCall(PetscMalloc2(nrow,&bufr,ncol,&bufc));
      irowm = bufr;
      icolm = bufc;
    }
    if (mat->rmap->mapping) PetscCall(ISLocalToGlobalMappingApplyBlock(mat->rmap->mapping,nrow,irow,bufr));
    else irowm = irow;
    if (mat->cmap->mapping) {
      if (mat->cmap->mapping != mat->rmap->mapping || ncol != nrow || icol != irow) {
        PetscCall(ISLocalToGlobalMappingApplyBlock(mat->cmap->mapping,ncol,icol,bufc));
      } else icolm = irowm;
    } else icolm = icol;
    PetscCall(MatSetValuesBlocked(mat,nrow,irowm,ncol,icolm,y,addv));
    if (bufr != buf) PetscCall(PetscFree2(bufr,bufc));
  }
  PetscCall(PetscLogEventEnd(MAT_SetValues,mat,0,0,0));
  PetscFunctionReturn(0);
}

/*@
   MatMultDiagonalBlock - Computes the matrix-vector product, y = Dx. Where D is defined by the inode or block structure of the diagonal

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  x   - the vector to be multiplied

   Output Parameters:
.  y - the result

   Notes:
   The vectors x and y cannot be the same.  I.e., one cannot
   call MatMult(A,y,y).

   Level: developer

.seealso: `MatMultTranspose()`, `MatMultAdd()`, `MatMultTransposeAdd()`
@*/
PetscErrorCode MatMultDiagonalBlock(Mat mat,Vec x,Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);

  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(x != y,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"x and y must be different vectors");
  MatCheckPreallocated(mat,1);

  PetscCheck(mat->ops->multdiagonalblock,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Matrix type %s does not have a multiply defined",((PetscObject)mat)->type_name);
  PetscCall((*mat->ops->multdiagonalblock)(mat,x,y));
  PetscCall(PetscObjectStateIncrease((PetscObject)y));
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------*/
/*@
   MatMult - Computes the matrix-vector product, y = Ax.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat - the matrix
-  x   - the vector to be multiplied

   Output Parameters:
.  y - the result

   Notes:
   The vectors x and y cannot be the same.  I.e., one cannot
   call MatMult(A,y,y).

   Level: beginner

.seealso: `MatMultTranspose()`, `MatMultAdd()`, `MatMultTransposeAdd()`
@*/
PetscErrorCode MatMult(Mat mat,Vec x,Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(x != y,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"x and y must be different vectors");
  PetscCheck(mat->cmap->N == x->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->N,x->map->N);
  PetscCheck(mat->rmap->N == y->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->N,y->map->N);
  PetscCheck(mat->cmap->n == x->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: local dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->n,x->map->n);
  PetscCheck(mat->rmap->n == y->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: local dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->n,y->map->n);
  PetscCall(VecSetErrorIfLocked(y,3));
  if (mat->erroriffailure) PetscCall(VecValidValues(x,2,PETSC_TRUE));
  MatCheckPreallocated(mat,1);

  PetscCall(VecLockReadPush(x));
  PetscCheck(mat->ops->mult,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Matrix type %s does not have a multiply defined",((PetscObject)mat)->type_name);
  PetscCall(PetscLogEventBegin(MAT_Mult,mat,x,y,0));
  PetscCall((*mat->ops->mult)(mat,x,y));
  PetscCall(PetscLogEventEnd(MAT_Mult,mat,x,y,0));
  if (mat->erroriffailure) PetscCall(VecValidValues(y,3,PETSC_FALSE));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(0);
}

/*@
   MatMultTranspose - Computes matrix transpose times a vector y = A^T * x.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat - the matrix
-  x   - the vector to be multiplied

   Output Parameters:
.  y - the result

   Notes:
   The vectors x and y cannot be the same.  I.e., one cannot
   call MatMultTranspose(A,y,y).

   For complex numbers this does NOT compute the Hermitian (complex conjugate) transpose multiple,
   use MatMultHermitianTranspose()

   Level: beginner

.seealso: `MatMult()`, `MatMultAdd()`, `MatMultTransposeAdd()`, `MatMultHermitianTranspose()`, `MatTranspose()`
@*/
PetscErrorCode MatMultTranspose(Mat mat,Vec x,Vec y)
{
  PetscErrorCode (*op)(Mat,Vec,Vec) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);

  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(x != y,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"x and y must be different vectors");
  PetscCheck(mat->cmap->N == y->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->N,y->map->N);
  PetscCheck(mat->rmap->N == x->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->N,x->map->N);
  PetscCheck(mat->cmap->n == y->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: local dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->n,y->map->n);
  PetscCheck(mat->rmap->n == x->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: local dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->n,x->map->n);
  if (mat->erroriffailure) PetscCall(VecValidValues(x,2,PETSC_TRUE));
  MatCheckPreallocated(mat,1);

  if (!mat->ops->multtranspose) {
    if (mat->symmetric == PETSC_BOOL3_TRUE && mat->ops->mult) op = mat->ops->mult;
    PetscCheck(op,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Matrix type %s does not have a multiply transpose defined or is symmetric and does not have a multiply defined",((PetscObject)mat)->type_name);
  } else op = mat->ops->multtranspose;
  PetscCall(PetscLogEventBegin(MAT_MultTranspose,mat,x,y,0));
  PetscCall(VecLockReadPush(x));
  PetscCall((*op)(mat,x,y));
  PetscCall(VecLockReadPop(x));
  PetscCall(PetscLogEventEnd(MAT_MultTranspose,mat,x,y,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)y));
  if (mat->erroriffailure) PetscCall(VecValidValues(y,3,PETSC_FALSE));
  PetscFunctionReturn(0);
}

/*@
   MatMultHermitianTranspose - Computes matrix Hermitian transpose times a vector.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat - the matrix
-  x   - the vector to be multilplied

   Output Parameters:
.  y - the result

   Notes:
   The vectors x and y cannot be the same.  I.e., one cannot
   call MatMultHermitianTranspose(A,y,y).

   Also called the conjugate transpose, complex conjugate transpose, or adjoint.

   For real numbers MatMultTranspose() and MatMultHermitianTranspose() are identical.

   Level: beginner

.seealso: `MatMult()`, `MatMultAdd()`, `MatMultHermitianTransposeAdd()`, `MatMultTranspose()`
@*/
PetscErrorCode MatMultHermitianTranspose(Mat mat,Vec x,Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);

  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(x != y,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"x and y must be different vectors");
  PetscCheck(mat->cmap->N == y->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->N,y->map->N);
  PetscCheck(mat->rmap->N == x->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->N,x->map->N);
  PetscCheck(mat->cmap->n == y->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: local dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->n,y->map->n);
  PetscCheck(mat->rmap->n == x->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: local dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->n,x->map->n);
  MatCheckPreallocated(mat,1);

  PetscCall(PetscLogEventBegin(MAT_MultHermitianTranspose,mat,x,y,0));
#if defined(PETSC_USE_COMPLEX)
  if (mat->ops->multhermitiantranspose || (mat->hermitian == PETSC_BOOL3_TRUE && mat->ops->mult)) {
    PetscCall(VecLockReadPush(x));
    if (mat->ops->multhermitiantranspose) {
      PetscCall((*mat->ops->multhermitiantranspose)(mat,x,y));
    } else {
      PetscCall((*mat->ops->mult)(mat,x,y));
    }
    PetscCall(VecLockReadPop(x));
  } else {
    Vec w;
    PetscCall(VecDuplicate(x,&w));
    PetscCall(VecCopy(x,w));
    PetscCall(VecConjugate(w));
    PetscCall(MatMultTranspose(mat,w,y));
    PetscCall(VecDestroy(&w));
    PetscCall(VecConjugate(y));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)y));
#else
  PetscCall(MatMultTranspose(mat,x,y));
#endif
  PetscCall(PetscLogEventEnd(MAT_MultHermitianTranspose,mat,x,y,0));
  PetscFunctionReturn(0);
}

/*@
    MatMultAdd -  Computes v3 = v2 + A * v1.

    Neighbor-wise Collective on Mat

    Input Parameters:
+   mat - the matrix
-   v1, v2 - the vectors

    Output Parameters:
.   v3 - the result

    Notes:
    The vectors v1 and v3 cannot be the same.  I.e., one cannot
    call MatMultAdd(A,v1,v2,v1).

    Level: beginner

.seealso: `MatMultTranspose()`, `MatMult()`, `MatMultTransposeAdd()`
@*/
PetscErrorCode MatMultAdd(Mat mat,Vec v1,Vec v2,Vec v3)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v1,VEC_CLASSID,2);
  PetscValidHeaderSpecific(v2,VEC_CLASSID,3);
  PetscValidHeaderSpecific(v3,VEC_CLASSID,4);

  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->cmap->N == v1->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec v1: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->N,v1->map->N);
  /* PetscCheck(mat->rmap->N == v2->map->N,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat mat,Vec v2: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->N,v2->map->N);
     PetscCheck(mat->rmap->N == v3->map->N,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat mat,Vec v3: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->N,v3->map->N); */
  PetscCheck(mat->rmap->n == v3->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat mat,Vec v3: local dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->n,v3->map->n);
  PetscCheck(mat->rmap->n == v2->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat mat,Vec v2: local dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->n,v2->map->n);
  PetscCheck(v1 != v3,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_IDN,"v1 and v3 must be different vectors");
  MatCheckPreallocated(mat,1);

  PetscCheck(mat->ops->multadd,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"No MatMultAdd() for matrix type %s",((PetscObject)mat)->type_name);
  PetscCall(PetscLogEventBegin(MAT_MultAdd,mat,v1,v2,v3));
  PetscCall(VecLockReadPush(v1));
  PetscCall((*mat->ops->multadd)(mat,v1,v2,v3));
  PetscCall(VecLockReadPop(v1));
  PetscCall(PetscLogEventEnd(MAT_MultAdd,mat,v1,v2,v3));
  PetscCall(PetscObjectStateIncrease((PetscObject)v3));
  PetscFunctionReturn(0);
}

/*@
   MatMultTransposeAdd - Computes v3 = v2 + A' * v1.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat - the matrix
-  v1, v2 - the vectors

   Output Parameters:
.  v3 - the result

   Notes:
   The vectors v1 and v3 cannot be the same.  I.e., one cannot
   call MatMultTransposeAdd(A,v1,v2,v1).

   Level: beginner

.seealso: `MatMultTranspose()`, `MatMultAdd()`, `MatMult()`
@*/
PetscErrorCode MatMultTransposeAdd(Mat mat,Vec v1,Vec v2,Vec v3)
{
  PetscErrorCode (*op)(Mat,Vec,Vec,Vec) = (!mat->ops->multtransposeadd && mat->symmetric) ? mat->ops->multadd : mat->ops->multtransposeadd;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v1,VEC_CLASSID,2);
  PetscValidHeaderSpecific(v2,VEC_CLASSID,3);
  PetscValidHeaderSpecific(v3,VEC_CLASSID,4);

  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->rmap->N == v1->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec v1: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->N,v1->map->N);
  PetscCheck(mat->cmap->N == v2->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec v2: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->N,v2->map->N);
  PetscCheck(mat->cmap->N == v3->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec v3: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->N,v3->map->N);
  PetscCheck(v1 != v3,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_IDN,"v1 and v3 must be different vectors");
  PetscCheck(op,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,1);

  PetscCall(PetscLogEventBegin(MAT_MultTransposeAdd,mat,v1,v2,v3));
  PetscCall(VecLockReadPush(v1));
  PetscCall((*op)(mat,v1,v2,v3));
  PetscCall(VecLockReadPop(v1));
  PetscCall(PetscLogEventEnd(MAT_MultTransposeAdd,mat,v1,v2,v3));
  PetscCall(PetscObjectStateIncrease((PetscObject)v3));
  PetscFunctionReturn(0);
}

/*@
   MatMultHermitianTransposeAdd - Computes v3 = v2 + A^H * v1.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat - the matrix
-  v1, v2 - the vectors

   Output Parameters:
.  v3 - the result

   Notes:
   The vectors v1 and v3 cannot be the same.  I.e., one cannot
   call MatMultHermitianTransposeAdd(A,v1,v2,v1).

   Level: beginner

.seealso: `MatMultHermitianTranspose()`, `MatMultTranspose()`, `MatMultAdd()`, `MatMult()`
@*/
PetscErrorCode MatMultHermitianTransposeAdd(Mat mat,Vec v1,Vec v2,Vec v3)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v1,VEC_CLASSID,2);
  PetscValidHeaderSpecific(v2,VEC_CLASSID,3);
  PetscValidHeaderSpecific(v3,VEC_CLASSID,4);

  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(v1 != v3,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_IDN,"v1 and v3 must be different vectors");
  PetscCheck(mat->rmap->N == v1->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec v1: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->N,v1->map->N);
  PetscCheck(mat->cmap->N == v2->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec v2: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->N,v2->map->N);
  PetscCheck(mat->cmap->N == v3->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec v3: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->N,v3->map->N);
  MatCheckPreallocated(mat,1);

  PetscCall(PetscLogEventBegin(MAT_MultHermitianTransposeAdd,mat,v1,v2,v3));
  PetscCall(VecLockReadPush(v1));
  if (mat->ops->multhermitiantransposeadd) {
    PetscCall((*mat->ops->multhermitiantransposeadd)(mat,v1,v2,v3));
  } else {
    Vec w,z;
    PetscCall(VecDuplicate(v1,&w));
    PetscCall(VecCopy(v1,w));
    PetscCall(VecConjugate(w));
    PetscCall(VecDuplicate(v3,&z));
    PetscCall(MatMultTranspose(mat,w,z));
    PetscCall(VecDestroy(&w));
    PetscCall(VecConjugate(z));
    if (v2 != v3) {
      PetscCall(VecWAXPY(v3,1.0,v2,z));
    } else {
      PetscCall(VecAXPY(v3,1.0,z));
    }
    PetscCall(VecDestroy(&z));
  }
  PetscCall(VecLockReadPop(v1));
  PetscCall(PetscLogEventEnd(MAT_MultHermitianTransposeAdd,mat,v1,v2,v3));
  PetscCall(PetscObjectStateIncrease((PetscObject)v3));
  PetscFunctionReturn(0);
}

/*@C
   MatGetFactorType - gets the type of factorization it is

   Not Collective

   Input Parameters:
.  mat - the matrix

   Output Parameters:
.  t - the type, one of MAT_FACTOR_NONE, MAT_FACTOR_LU, MAT_FACTOR_CHOLESKY, MAT_FACTOR_ILU, MAT_FACTOR_ICC,MAT_FACTOR_ILUDT

   Level: intermediate

.seealso: `MatFactorType`, `MatGetFactor()`, `MatSetFactorType()`
@*/
PetscErrorCode MatGetFactorType(Mat mat,MatFactorType *t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidPointer(t,2);
  *t = mat->factortype;
  PetscFunctionReturn(0);
}

/*@C
   MatSetFactorType - sets the type of factorization it is

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix
-  t - the type, one of MAT_FACTOR_NONE, MAT_FACTOR_LU, MAT_FACTOR_CHOLESKY, MAT_FACTOR_ILU, MAT_FACTOR_ICC,MAT_FACTOR_ILUDT

   Level: intermediate

.seealso: `MatFactorType`, `MatGetFactor()`, `MatGetFactorType()`
@*/
PetscErrorCode MatSetFactorType(Mat mat, MatFactorType t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  mat->factortype = t;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------*/
/*@C
   MatGetInfo - Returns information about matrix storage (number of
   nonzeros, memory, etc.).

   Collective on Mat if MAT_GLOBAL_MAX or MAT_GLOBAL_SUM is used as the flag

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  flag - flag indicating the type of parameters to be returned
   (MAT_LOCAL - local matrix, MAT_GLOBAL_MAX - maximum over all processors,
   MAT_GLOBAL_SUM - sum over all processors)
-  info - matrix information context

   Notes:
   The MatInfo context contains a variety of matrix data, including
   number of nonzeros allocated and used, number of mallocs during
   matrix assembly, etc.  Additional information for factored matrices
   is provided (such as the fill ratio, number of mallocs during
   factorization, etc.).  Much of this info is printed to PETSC_STDOUT
   when using the runtime options
$       -info -mat_view ::ascii_info

   Example for C/C++ Users:
   See the file ${PETSC_DIR}/include/petscmat.h for a complete list of
   data within the MatInfo context.  For example,
.vb
      MatInfo info;
      Mat     A;
      double  mal, nz_a, nz_u;

      MatGetInfo(A,MAT_LOCAL,&info);
      mal  = info.mallocs;
      nz_a = info.nz_allocated;
.ve

   Example for Fortran Users:
   Fortran users should declare info as a double precision
   array of dimension MAT_INFO_SIZE, and then extract the parameters
   of interest.  See the file ${PETSC_DIR}/include/petsc/finclude/petscmat.h
   a complete list of parameter names.
.vb
      double  precision info(MAT_INFO_SIZE)
      double  precision mal, nz_a
      Mat     A
      integer ierr

      call MatGetInfo(A,MAT_LOCAL,info,ierr)
      mal = info(MAT_INFO_MALLOCS)
      nz_a = info(MAT_INFO_NZ_ALLOCATED)
.ve

    Level: intermediate

    Developer Note: fortran interface is not autogenerated as the f90
    interface definition cannot be generated correctly [due to MatInfo]

.seealso: `MatStashGetInfo()`

@*/
PetscErrorCode MatGetInfo(Mat mat,MatInfoType flag,MatInfo *info)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidPointer(info,3);
  PetscCheck(mat->ops->getinfo,PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,1);
  PetscCall((*mat->ops->getinfo)(mat,flag,info));
  PetscFunctionReturn(0);
}

/*
   This is used by external packages where it is not easy to get the info from the actual
   matrix factorization.
*/
PetscErrorCode MatGetInfo_External(Mat A,MatInfoType flag,MatInfo *info)
{
  PetscFunctionBegin;
  PetscCall(PetscMemzero(info,sizeof(MatInfo)));
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------*/

/*@C
   MatLUFactor - Performs in-place LU factorization of matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  row - row permutation
.  col - column permutation
-  info - options for factorization, includes
$          fill - expected fill as ratio of original fill.
$          dtcol - pivot tolerance (0 no pivot, 1 full column pivoting)
$                   Run with the option -info to determine an optimal value to use

   Notes:
   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   This changes the state of the matrix to a factored matrix; it cannot be used
   for example with MatSetValues() unless one first calls MatSetUnfactored().

   Level: developer

.seealso: `MatLUFactorSymbolic()`, `MatLUFactorNumeric()`, `MatCholeskyFactor()`,
          `MatGetOrdering()`, `MatSetUnfactored()`, `MatFactorInfo`, `MatGetFactor()`

    Developer Note: fortran interface is not autogenerated as the f90
    interface definition cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode MatLUFactor(Mat mat,IS row,IS col,const MatFactorInfo *info)
{
  MatFactorInfo  tinfo;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (row) PetscValidHeaderSpecific(row,IS_CLASSID,2);
  if (col) PetscValidHeaderSpecific(col,IS_CLASSID,3);
  if (info) PetscValidPointer(info,4);
  PetscValidType(mat,1);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->ops->lufactor,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,1);
  if (!info) {
    PetscCall(MatFactorInfoInitialize(&tinfo));
    info = &tinfo;
  }

  PetscCall(PetscLogEventBegin(MAT_LUFactor,mat,row,col,0));
  PetscCall((*mat->ops->lufactor)(mat,row,col,info));
  PetscCall(PetscLogEventEnd(MAT_LUFactor,mat,row,col,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
  PetscFunctionReturn(0);
}

/*@C
   MatILUFactor - Performs in-place ILU factorization of matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  row - row permutation
.  col - column permutation
-  info - structure containing
$      levels - number of levels of fill.
$      expected fill - as ratio of original fill.
$      1 or 0 - indicating force fill on diagonal (improves robustness for matrices
                missing diagonal entries)

   Notes:
   Probably really in-place only when level of fill is zero, otherwise allocates
   new space to store factored matrix and deletes previous memory.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatILUFactorSymbolic()`, `MatLUFactorNumeric()`, `MatCholeskyFactor()`, `MatFactorInfo`

    Developer Note: fortran interface is not autogenerated as the f90
    interface definition cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode MatILUFactor(Mat mat,IS row,IS col,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (row) PetscValidHeaderSpecific(row,IS_CLASSID,2);
  if (col) PetscValidHeaderSpecific(col,IS_CLASSID,3);
  PetscValidPointer(info,4);
  PetscValidType(mat,1);
  PetscCheck(mat->rmap->N == mat->cmap->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONG,"matrix must be square");
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->ops->ilufactor,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,1);

  PetscCall(PetscLogEventBegin(MAT_ILUFactor,mat,row,col,0));
  PetscCall((*mat->ops->ilufactor)(mat,row,col,info));
  PetscCall(PetscLogEventEnd(MAT_ILUFactor,mat,row,col,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
  PetscFunctionReturn(0);
}

/*@C
   MatLUFactorSymbolic - Performs symbolic LU factorization of matrix.
   Call this routine before calling MatLUFactorNumeric().

   Collective on Mat

   Input Parameters:
+  fact - the factor matrix obtained with MatGetFactor()
.  mat - the matrix
.  row, col - row and column permutations
-  info - options for factorization, includes
$          fill - expected fill as ratio of original fill.
$          dtcol - pivot tolerance (0 no pivot, 1 full column pivoting)
$                   Run with the option -info to determine an optimal value to use

   Notes:
    See Users-Manual: ch_mat for additional information about choosing the fill factor for better efficiency.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatLUFactor()`, `MatLUFactorNumeric()`, `MatCholeskyFactor()`, `MatFactorInfo`, `MatFactorInfoInitialize()`

    Developer Note: fortran interface is not autogenerated as the f90
    interface definition cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode MatLUFactorSymbolic(Mat fact,Mat mat,IS row,IS col,const MatFactorInfo *info)
{
  MatFactorInfo  tinfo;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  if (row) PetscValidHeaderSpecific(row,IS_CLASSID,3);
  if (col) PetscValidHeaderSpecific(col,IS_CLASSID,4);
  if (info) PetscValidPointer(info,5);
  PetscValidType(mat,2);
  PetscValidPointer(fact,1);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  if (!(fact)->ops->lufactorsymbolic) {
    MatSolverType stype;
    PetscCall(MatFactorGetSolverType(fact,&stype));
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Matrix type %s symbolic LU using solver package %s",((PetscObject)mat)->type_name,stype);
  }
  MatCheckPreallocated(mat,2);
  if (!info) {
    PetscCall(MatFactorInfoInitialize(&tinfo));
    info = &tinfo;
  }

  if (!fact->trivialsymbolic) PetscCall(PetscLogEventBegin(MAT_LUFactorSymbolic,mat,row,col,0));
  PetscCall((fact->ops->lufactorsymbolic)(fact,mat,row,col,info));
  if (!fact->trivialsymbolic) PetscCall(PetscLogEventEnd(MAT_LUFactorSymbolic,mat,row,col,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)fact));
  PetscFunctionReturn(0);
}

/*@C
   MatLUFactorNumeric - Performs numeric LU factorization of a matrix.
   Call this routine after first calling MatLUFactorSymbolic().

   Collective on Mat

   Input Parameters:
+  fact - the factor matrix obtained with MatGetFactor()
.  mat - the matrix
-  info - options for factorization

   Notes:
   See MatLUFactor() for in-place factorization.  See
   MatCholeskyFactorNumeric() for the symmetric, positive definite case.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatLUFactorSymbolic()`, `MatLUFactor()`, `MatCholeskyFactor()`

    Developer Note: fortran interface is not autogenerated as the f90
    interface definition cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode MatLUFactorNumeric(Mat fact,Mat mat,const MatFactorInfo *info)
{
  MatFactorInfo  tinfo;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  PetscValidType(mat,2);
  PetscValidPointer(fact,1);
  PetscValidHeaderSpecific(fact,MAT_CLASSID,1);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(mat->rmap->N == (fact)->rmap->N && mat->cmap->N == (fact)->cmap->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Mat fact: global dimensions are different %" PetscInt_FMT " should = %" PetscInt_FMT " %" PetscInt_FMT " should = %" PetscInt_FMT,mat->rmap->N,(fact)->rmap->N,mat->cmap->N,(fact)->cmap->N);

  PetscCheck((fact)->ops->lufactornumeric,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s numeric LU",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,2);
  if (!info) {
    PetscCall(MatFactorInfoInitialize(&tinfo));
    info = &tinfo;
  }

  if (!fact->trivialsymbolic) PetscCall(PetscLogEventBegin(MAT_LUFactorNumeric,mat,fact,0,0));
  else PetscCall(PetscLogEventBegin(MAT_LUFactor,mat,fact,0,0));
  PetscCall((fact->ops->lufactornumeric)(fact,mat,info));
  if (!fact->trivialsymbolic) PetscCall(PetscLogEventEnd(MAT_LUFactorNumeric,mat,fact,0,0));
  else PetscCall(PetscLogEventEnd(MAT_LUFactor,mat,fact,0,0));
  PetscCall(MatViewFromOptions(fact,NULL,"-mat_factor_view"));
  PetscCall(PetscObjectStateIncrease((PetscObject)fact));
  PetscFunctionReturn(0);
}

/*@C
   MatCholeskyFactor - Performs in-place Cholesky factorization of a
   symmetric matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  perm - row and column permutations
-  f - expected fill as ratio of original fill

   Notes:
   See MatLUFactor() for the nonsymmetric case.  See also
   MatCholeskyFactorSymbolic(), and MatCholeskyFactorNumeric().

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatLUFactor()`, `MatCholeskyFactorSymbolic()`, `MatCholeskyFactorNumeric()`
          `MatGetOrdering()`

    Developer Note: fortran interface is not autogenerated as the f90
    interface definition cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode MatCholeskyFactor(Mat mat,IS perm,const MatFactorInfo *info)
{
  MatFactorInfo  tinfo;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (perm) PetscValidHeaderSpecific(perm,IS_CLASSID,2);
  if (info) PetscValidPointer(info,3);
  PetscCheck(mat->rmap->N == mat->cmap->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONG,"Matrix must be square");
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->ops->choleskyfactor,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"In-place factorization for Mat type %s is not supported, try out-of-place factorization. See MatCholeskyFactorSymbolic/Numeric",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,1);
  if (!info) {
    PetscCall(MatFactorInfoInitialize(&tinfo));
    info = &tinfo;
  }

  PetscCall(PetscLogEventBegin(MAT_CholeskyFactor,mat,perm,0,0));
  PetscCall((*mat->ops->choleskyfactor)(mat,perm,info));
  PetscCall(PetscLogEventEnd(MAT_CholeskyFactor,mat,perm,0,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
  PetscFunctionReturn(0);
}

/*@C
   MatCholeskyFactorSymbolic - Performs symbolic Cholesky factorization
   of a symmetric matrix.

   Collective on Mat

   Input Parameters:
+  fact - the factor matrix obtained with MatGetFactor()
.  mat - the matrix
.  perm - row and column permutations
-  info - options for factorization, includes
$          fill - expected fill as ratio of original fill.
$          dtcol - pivot tolerance (0 no pivot, 1 full column pivoting)
$                   Run with the option -info to determine an optimal value to use

   Notes:
   See MatLUFactorSymbolic() for the nonsymmetric case.  See also
   MatCholeskyFactor() and MatCholeskyFactorNumeric().

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatLUFactorSymbolic()`, `MatCholeskyFactor()`, `MatCholeskyFactorNumeric()`
          `MatGetOrdering()`

    Developer Note: fortran interface is not autogenerated as the f90
    interface definition cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode MatCholeskyFactorSymbolic(Mat fact,Mat mat,IS perm,const MatFactorInfo *info)
{
  MatFactorInfo  tinfo;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  PetscValidType(mat,2);
  if (perm) PetscValidHeaderSpecific(perm,IS_CLASSID,3);
  if (info) PetscValidPointer(info,4);
  PetscValidPointer(fact,1);
  PetscCheck(mat->rmap->N == mat->cmap->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONG,"Matrix must be square");
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  if (!(fact)->ops->choleskyfactorsymbolic) {
    MatSolverType stype;
    PetscCall(MatFactorGetSolverType(fact,&stype));
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s symbolic factor Cholesky using solver package %s",((PetscObject)mat)->type_name,stype);
  }
  MatCheckPreallocated(mat,2);
  if (!info) {
    PetscCall(MatFactorInfoInitialize(&tinfo));
    info = &tinfo;
  }

  if (!fact->trivialsymbolic) PetscCall(PetscLogEventBegin(MAT_CholeskyFactorSymbolic,mat,perm,0,0));
  PetscCall((fact->ops->choleskyfactorsymbolic)(fact,mat,perm,info));
  if (!fact->trivialsymbolic) PetscCall(PetscLogEventEnd(MAT_CholeskyFactorSymbolic,mat,perm,0,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)fact));
  PetscFunctionReturn(0);
}

/*@C
   MatCholeskyFactorNumeric - Performs numeric Cholesky factorization
   of a symmetric matrix. Call this routine after first calling
   MatCholeskyFactorSymbolic().

   Collective on Mat

   Input Parameters:
+  fact - the factor matrix obtained with MatGetFactor()
.  mat - the initial matrix
.  info - options for factorization
-  fact - the symbolic factor of mat

   Notes:
   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatCholeskyFactorSymbolic()`, `MatCholeskyFactor()`, `MatLUFactorNumeric()`

    Developer Note: fortran interface is not autogenerated as the f90
    interface definition cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode MatCholeskyFactorNumeric(Mat fact,Mat mat,const MatFactorInfo *info)
{
  MatFactorInfo  tinfo;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  PetscValidType(mat,2);
  PetscValidPointer(fact,1);
  PetscValidHeaderSpecific(fact,MAT_CLASSID,1);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck((fact)->ops->choleskyfactornumeric,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s numeric factor Cholesky",((PetscObject)mat)->type_name);
  PetscCheck(mat->rmap->N == (fact)->rmap->N && mat->cmap->N == (fact)->cmap->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Mat fact: global dim %" PetscInt_FMT " should = %" PetscInt_FMT " %" PetscInt_FMT " should = %" PetscInt_FMT,mat->rmap->N,(fact)->rmap->N,mat->cmap->N,(fact)->cmap->N);
  MatCheckPreallocated(mat,2);
  if (!info) {
    PetscCall(MatFactorInfoInitialize(&tinfo));
    info = &tinfo;
  }

  if (!fact->trivialsymbolic) PetscCall(PetscLogEventBegin(MAT_CholeskyFactorNumeric,mat,fact,0,0));
  else PetscCall(PetscLogEventBegin(MAT_CholeskyFactor,mat,fact,0,0));
  PetscCall((fact->ops->choleskyfactornumeric)(fact,mat,info));
  if (!fact->trivialsymbolic) PetscCall(PetscLogEventEnd(MAT_CholeskyFactorNumeric,mat,fact,0,0));
  else PetscCall(PetscLogEventEnd(MAT_CholeskyFactor,mat,fact,0,0));
  PetscCall(MatViewFromOptions(fact,NULL,"-mat_factor_view"));
  PetscCall(PetscObjectStateIncrease((PetscObject)fact));
  PetscFunctionReturn(0);
}

/*@
   MatQRFactor - Performs in-place QR factorization of matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  col - column permutation
-  info - options for factorization, includes
$          fill - expected fill as ratio of original fill.
$          dtcol - pivot tolerance (0 no pivot, 1 full column pivoting)
$                   Run with the option -info to determine an optimal value to use

   Notes:
   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   This changes the state of the matrix to a factored matrix; it cannot be used
   for example with MatSetValues() unless one first calls MatSetUnfactored().

   Level: developer

.seealso: `MatQRFactorSymbolic()`, `MatQRFactorNumeric()`, `MatLUFactor()`,
          `MatSetUnfactored()`, `MatFactorInfo`, `MatGetFactor()`

    Developer Note: fortran interface is not autogenerated as the f90
    interface definition cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode MatQRFactor(Mat mat, IS col, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (col) PetscValidHeaderSpecific(col,IS_CLASSID,2);
  if (info) PetscValidPointer(info,3);
  PetscValidType(mat,1);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);
  PetscCall(PetscLogEventBegin(MAT_QRFactor,mat,col,0,0));
  PetscUseMethod(mat,"MatQRFactor_C", (Mat,IS,const MatFactorInfo*), (mat, col, info));
  PetscCall(PetscLogEventEnd(MAT_QRFactor,mat,col,0,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
  PetscFunctionReturn(0);
}

/*@
   MatQRFactorSymbolic - Performs symbolic QR factorization of matrix.
   Call this routine before calling MatQRFactorNumeric().

   Collective on Mat

   Input Parameters:
+  fact - the factor matrix obtained with MatGetFactor()
.  mat - the matrix
.  col - column permutation
-  info - options for factorization, includes
$          fill - expected fill as ratio of original fill.
$          dtcol - pivot tolerance (0 no pivot, 1 full column pivoting)
$                   Run with the option -info to determine an optimal value to use

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatQRFactor()`, `MatQRFactorNumeric()`, `MatLUFactor()`, `MatFactorInfo`, `MatFactorInfoInitialize()`

    Developer Note: fortran interface is not autogenerated as the f90
    interface definition cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode MatQRFactorSymbolic(Mat fact,Mat mat,IS col,const MatFactorInfo *info)
{
  MatFactorInfo  tinfo;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  if (col) PetscValidHeaderSpecific(col,IS_CLASSID,3);
  if (info) PetscValidPointer(info,4);
  PetscValidType(mat,2);
  PetscValidPointer(fact,1);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,2);
  if (!info) {
    PetscCall(MatFactorInfoInitialize(&tinfo));
    info = &tinfo;
  }

  if (!fact->trivialsymbolic) PetscCall(PetscLogEventBegin(MAT_QRFactorSymbolic,fact,mat,col,0));
  PetscUseMethod(fact,"MatQRFactorSymbolic_C", (Mat,Mat,IS,const MatFactorInfo*), (fact, mat, col, info));
  if (!fact->trivialsymbolic) PetscCall(PetscLogEventEnd(MAT_QRFactorSymbolic,fact,mat,col,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)fact));
  PetscFunctionReturn(0);
}

/*@
   MatQRFactorNumeric - Performs numeric QR factorization of a matrix.
   Call this routine after first calling MatQRFactorSymbolic().

   Collective on Mat

   Input Parameters:
+  fact - the factor matrix obtained with MatGetFactor()
.  mat - the matrix
-  info - options for factorization

   Notes:
   See MatQRFactor() for in-place factorization.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatQRFactorSymbolic()`, `MatLUFactor()`

    Developer Note: fortran interface is not autogenerated as the f90
    interface definition cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode MatQRFactorNumeric(Mat fact,Mat mat,const MatFactorInfo *info)
{
  MatFactorInfo  tinfo;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  PetscValidType(mat,2);
  PetscValidPointer(fact,1);
  PetscValidHeaderSpecific(fact,MAT_CLASSID,1);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(mat->rmap->N == fact->rmap->N && mat->cmap->N == fact->cmap->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Mat fact: global dimensions are different %" PetscInt_FMT " should = %" PetscInt_FMT " %" PetscInt_FMT " should = %" PetscInt_FMT,mat->rmap->N,(fact)->rmap->N,mat->cmap->N,(fact)->cmap->N);

  MatCheckPreallocated(mat,2);
  if (!info) {
    PetscCall(MatFactorInfoInitialize(&tinfo));
    info = &tinfo;
  }

  if (!fact->trivialsymbolic) PetscCall(PetscLogEventBegin(MAT_QRFactorNumeric,mat,fact,0,0));
  else  PetscCall(PetscLogEventBegin(MAT_QRFactor,mat,fact,0,0));
  PetscUseMethod(fact,"MatQRFactorNumeric_C", (Mat,Mat,const MatFactorInfo*), (fact, mat, info));
  if (!fact->trivialsymbolic) PetscCall(PetscLogEventEnd(MAT_QRFactorNumeric,mat,fact,0,0));
  else PetscCall(PetscLogEventEnd(MAT_QRFactor,mat,fact,0,0));
  PetscCall(MatViewFromOptions(fact,NULL,"-mat_factor_view"));
  PetscCall(PetscObjectStateIncrease((PetscObject)fact));
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/
/*@
   MatSolve - Solves A x = b, given a factored matrix.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat - the factored matrix
-  b - the right-hand-side vector

   Output Parameter:
.  x - the result vector

   Notes:
   The vectors b and x cannot be the same.  I.e., one cannot
   call MatSolve(A,x,x).

   Notes:
   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatSolveAdd()`, `MatSolveTranspose()`, `MatSolveTransposeAdd()`
@*/
PetscErrorCode MatSolve(Mat mat,Vec b,Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  PetscCheckSameComm(mat,1,b,2);
  PetscCheckSameComm(mat,1,x,3);
  PetscCheck(x != b,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  PetscCheck(mat->cmap->N == x->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->N,x->map->N);
  PetscCheck(mat->rmap->N == b->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->N,b->map->N);
  PetscCheck(mat->rmap->n == b->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: local dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->n,b->map->n);
  if (!mat->rmap->N && !mat->cmap->N) PetscFunctionReturn(0);
  MatCheckPreallocated(mat,1);

  PetscCall(PetscLogEventBegin(MAT_Solve,mat,b,x,0));
  if (mat->factorerrortype) {
    PetscCall(PetscInfo(mat,"MatFactorError %d\n",mat->factorerrortype));
    PetscCall(VecSetInf(x));
  } else {
    PetscCheck(mat->ops->solve,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
    PetscCall((*mat->ops->solve)(mat,b,x));
  }
  PetscCall(PetscLogEventEnd(MAT_Solve,mat,b,x,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_Basic(Mat A,Mat B,Mat X,PetscBool trans)
{
  Vec            b,x;
  PetscInt       N,i;
  PetscErrorCode (*f)(Mat,Vec,Vec);
  PetscBool      Abound,Bneedconv = PETSC_FALSE,Xneedconv = PETSC_FALSE;

  PetscFunctionBegin;
  if (A->factorerrortype) {
    PetscCall(PetscInfo(A,"MatFactorError %d\n",A->factorerrortype));
    PetscCall(MatSetInf(X));
    PetscFunctionReturn(0);
  }
  f = (!trans || (!A->ops->solvetranspose && A->symmetric)) ? A->ops->solve : A->ops->solvetranspose;
  PetscCheck(f,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Mat type %s",((PetscObject)A)->type_name);
  PetscCall(MatBoundToCPU(A,&Abound));
  if (!Abound) {
    PetscCall(PetscObjectTypeCompareAny((PetscObject)B,&Bneedconv,MATSEQDENSE,MATMPIDENSE,""));
    PetscCall(PetscObjectTypeCompareAny((PetscObject)X,&Xneedconv,MATSEQDENSE,MATMPIDENSE,""));
  }
  if (Bneedconv) {
    PetscCall(MatConvert(B,MATDENSECUDA,MAT_INPLACE_MATRIX,&B));
  }
  if (Xneedconv) {
    PetscCall(MatConvert(X,MATDENSECUDA,MAT_INPLACE_MATRIX,&X));
  }
  PetscCall(MatGetSize(B,NULL,&N));
  for (i=0; i<N; i++) {
    PetscCall(MatDenseGetColumnVecRead(B,i,&b));
    PetscCall(MatDenseGetColumnVecWrite(X,i,&x));
    PetscCall((*f)(A,b,x));
    PetscCall(MatDenseRestoreColumnVecWrite(X,i,&x));
    PetscCall(MatDenseRestoreColumnVecRead(B,i,&b));
  }
  if (Bneedconv) {
    PetscCall(MatConvert(B,MATDENSE,MAT_INPLACE_MATRIX,&B));
  }
  if (Xneedconv) {
    PetscCall(MatConvert(X,MATDENSE,MAT_INPLACE_MATRIX,&X));
  }
  PetscFunctionReturn(0);
}

/*@
   MatMatSolve - Solves A X = B, given a factored matrix.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  A - the factored matrix
-  B - the right-hand-side matrix MATDENSE (or sparse -- when using MUMPS)

   Output Parameter:
.  X - the result matrix (dense matrix)

   Notes:
   If B is a MATDENSE matrix then one can call MatMatSolve(A,B,B) except with MKL_CPARDISO;
   otherwise, B and X cannot be the same.

   Notes:
   Most users should usually employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate(). However KSP can only solve for one vector (column of X)
   at a time.

   Level: developer

.seealso: `MatMatSolveTranspose()`, `MatLUFactor()`, `MatCholeskyFactor()`
@*/
PetscErrorCode MatMatSolve(Mat A,Mat B,Mat X)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidHeaderSpecific(X,MAT_CLASSID,3);
  PetscCheckSameComm(A,1,B,2);
  PetscCheckSameComm(A,1,X,3);
  PetscCheck(A->cmap->N == X->rmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Mat A,Mat X: global dim %" PetscInt_FMT " %" PetscInt_FMT,A->cmap->N,X->rmap->N);
  PetscCheck(A->rmap->N == B->rmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Mat A,Mat B: global dim %" PetscInt_FMT " %" PetscInt_FMT,A->rmap->N,B->rmap->N);
  PetscCheck(X->cmap->N == B->cmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Solution matrix must have same number of columns as rhs matrix");
  if (!A->rmap->N && !A->cmap->N) PetscFunctionReturn(0);
  PetscCheck(A->factortype,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  MatCheckPreallocated(A,1);

  PetscCall(PetscLogEventBegin(MAT_MatSolve,A,B,X,0));
  if (!A->ops->matsolve) {
    PetscCall(PetscInfo(A,"Mat type %s using basic MatMatSolve\n",((PetscObject)A)->type_name));
    PetscCall(MatMatSolve_Basic(A,B,X,PETSC_FALSE));
  } else {
    PetscCall((*A->ops->matsolve)(A,B,X));
  }
  PetscCall(PetscLogEventEnd(MAT_MatSolve,A,B,X,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)X));
  PetscFunctionReturn(0);
}

/*@
   MatMatSolveTranspose - Solves A^T X = B, given a factored matrix.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  A - the factored matrix
-  B - the right-hand-side matrix  (dense matrix)

   Output Parameter:
.  X - the result matrix (dense matrix)

   Notes:
   The matrices B and X cannot be the same.  I.e., one cannot
   call MatMatSolveTranspose(A,X,X).

   Notes:
   Most users should usually employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate(). However KSP can only solve for one vector (column of X)
   at a time.

   When using SuperLU_Dist or MUMPS as a parallel solver, PETSc will use their functionality to solve multiple right hand sides simultaneously.

   Level: developer

.seealso: `MatMatSolve()`, `MatLUFactor()`, `MatCholeskyFactor()`
@*/
PetscErrorCode MatMatSolveTranspose(Mat A,Mat B,Mat X)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidHeaderSpecific(X,MAT_CLASSID,3);
  PetscCheckSameComm(A,1,B,2);
  PetscCheckSameComm(A,1,X,3);
  PetscCheck(X != B,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_IDN,"X and B must be different matrices");
  PetscCheck(A->cmap->N == X->rmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Mat A,Mat X: global dim %" PetscInt_FMT " %" PetscInt_FMT,A->cmap->N,X->rmap->N);
  PetscCheck(A->rmap->N == B->rmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Mat A,Mat B: global dim %" PetscInt_FMT " %" PetscInt_FMT,A->rmap->N,B->rmap->N);
  PetscCheck(A->rmap->n == B->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A,Mat B: local dim %" PetscInt_FMT " %" PetscInt_FMT,A->rmap->n,B->rmap->n);
  PetscCheck(X->cmap->N >= B->cmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Solution matrix must have same number of columns as rhs matrix");
  if (!A->rmap->N && !A->cmap->N) PetscFunctionReturn(0);
  PetscCheck(A->factortype,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  MatCheckPreallocated(A,1);

  PetscCall(PetscLogEventBegin(MAT_MatSolve,A,B,X,0));
  if (!A->ops->matsolvetranspose) {
    PetscCall(PetscInfo(A,"Mat type %s using basic MatMatSolveTranspose\n",((PetscObject)A)->type_name));
    PetscCall(MatMatSolve_Basic(A,B,X,PETSC_TRUE));
  } else {
    PetscCall((*A->ops->matsolvetranspose)(A,B,X));
  }
  PetscCall(PetscLogEventEnd(MAT_MatSolve,A,B,X,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)X));
  PetscFunctionReturn(0);
}

/*@
   MatMatTransposeSolve - Solves A X = B^T, given a factored matrix.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  A - the factored matrix
-  Bt - the transpose of right-hand-side matrix

   Output Parameter:
.  X - the result matrix (dense matrix)

   Notes:
   Most users should usually employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate(). However KSP can only solve for one vector (column of X)
   at a time.

   For MUMPS, it only supports centralized sparse compressed column format on the host processor for right hand side matrix. User must create B^T in sparse compressed row format on the host processor and call MatMatTransposeSolve() to implement MUMPS' MatMatSolve().

   Level: developer

.seealso: `MatMatSolve()`, `MatMatSolveTranspose()`, `MatLUFactor()`, `MatCholeskyFactor()`
@*/
PetscErrorCode MatMatTransposeSolve(Mat A,Mat Bt,Mat X)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidHeaderSpecific(Bt,MAT_CLASSID,2);
  PetscValidHeaderSpecific(X,MAT_CLASSID,3);
  PetscCheckSameComm(A,1,Bt,2);
  PetscCheckSameComm(A,1,X,3);

  PetscCheck(X != Bt,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_IDN,"X and B must be different matrices");
  PetscCheck(A->cmap->N == X->rmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Mat A,Mat X: global dim %" PetscInt_FMT " %" PetscInt_FMT,A->cmap->N,X->rmap->N);
  PetscCheck(A->rmap->N == Bt->cmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Mat A,Mat Bt: global dim %" PetscInt_FMT " %" PetscInt_FMT,A->rmap->N,Bt->cmap->N);
  PetscCheck(X->cmap->N >= Bt->rmap->N,PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_SIZ,"Solution matrix must have same number of columns as row number of the rhs matrix");
  if (!A->rmap->N && !A->cmap->N) PetscFunctionReturn(0);
  PetscCheck(A->factortype,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  MatCheckPreallocated(A,1);

  PetscCheck(A->ops->mattransposesolve,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Mat type %s",((PetscObject)A)->type_name);
  PetscCall(PetscLogEventBegin(MAT_MatTrSolve,A,Bt,X,0));
  PetscCall((*A->ops->mattransposesolve)(A,Bt,X));
  PetscCall(PetscLogEventEnd(MAT_MatTrSolve,A,Bt,X,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)X));
  PetscFunctionReturn(0);
}

/*@
   MatForwardSolve - Solves L x = b, given a factored matrix, A = LU, or
                            U^T*D^(1/2) x = b, given a factored symmetric matrix, A = U^T*D*U,

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat - the factored matrix
-  b - the right-hand-side vector

   Output Parameter:
.  x - the result vector

   Notes:
   MatSolve() should be used for most applications, as it performs
   a forward solve followed by a backward solve.

   The vectors b and x cannot be the same,  i.e., one cannot
   call MatForwardSolve(A,x,x).

   For matrix in seqsbaij format with block size larger than 1,
   the diagonal blocks are not implemented as D = D^(1/2) * D^(1/2) yet.
   MatForwardSolve() solves U^T*D y = b, and
   MatBackwardSolve() solves U x = y.
   Thus they do not provide a symmetric preconditioner.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatSolve()`, `MatBackwardSolve()`
@*/
PetscErrorCode MatForwardSolve(Mat mat,Vec b,Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  PetscCheckSameComm(mat,1,b,2);
  PetscCheckSameComm(mat,1,x,3);
  PetscCheck(x != b,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  PetscCheck(mat->cmap->N == x->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->N,x->map->N);
  PetscCheck(mat->rmap->N == b->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->N,b->map->N);
  PetscCheck(mat->rmap->n == b->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: local dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->n,b->map->n);
  if (!mat->rmap->N && !mat->cmap->N) PetscFunctionReturn(0);
  MatCheckPreallocated(mat,1);

  PetscCheck(mat->ops->forwardsolve,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  PetscCall(PetscLogEventBegin(MAT_ForwardSolve,mat,b,x,0));
  PetscCall((*mat->ops->forwardsolve)(mat,b,x));
  PetscCall(PetscLogEventEnd(MAT_ForwardSolve,mat,b,x,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(0);
}

/*@
   MatBackwardSolve - Solves U x = b, given a factored matrix, A = LU.
                             D^(1/2) U x = b, given a factored symmetric matrix, A = U^T*D*U,

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat - the factored matrix
-  b - the right-hand-side vector

   Output Parameter:
.  x - the result vector

   Notes:
   MatSolve() should be used for most applications, as it performs
   a forward solve followed by a backward solve.

   The vectors b and x cannot be the same.  I.e., one cannot
   call MatBackwardSolve(A,x,x).

   For matrix in seqsbaij format with block size larger than 1,
   the diagonal blocks are not implemented as D = D^(1/2) * D^(1/2) yet.
   MatForwardSolve() solves U^T*D y = b, and
   MatBackwardSolve() solves U x = y.
   Thus they do not provide a symmetric preconditioner.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatSolve()`, `MatForwardSolve()`
@*/
PetscErrorCode MatBackwardSolve(Mat mat,Vec b,Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  PetscCheckSameComm(mat,1,b,2);
  PetscCheckSameComm(mat,1,x,3);
  PetscCheck(x != b,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  PetscCheck(mat->cmap->N == x->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->N,x->map->N);
  PetscCheck(mat->rmap->N == b->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->N,b->map->N);
  PetscCheck(mat->rmap->n == b->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: local dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->n,b->map->n);
  if (!mat->rmap->N && !mat->cmap->N) PetscFunctionReturn(0);
  MatCheckPreallocated(mat,1);

  PetscCheck(mat->ops->backwardsolve,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  PetscCall(PetscLogEventBegin(MAT_BackwardSolve,mat,b,x,0));
  PetscCall((*mat->ops->backwardsolve)(mat,b,x));
  PetscCall(PetscLogEventEnd(MAT_BackwardSolve,mat,b,x,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(0);
}

/*@
   MatSolveAdd - Computes x = y + inv(A)*b, given a factored matrix.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat - the factored matrix
.  b - the right-hand-side vector
-  y - the vector to be added to

   Output Parameter:
.  x - the result vector

   Notes:
   The vectors b and x cannot be the same.  I.e., one cannot
   call MatSolveAdd(A,x,y,x).

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatSolve()`, `MatSolveTranspose()`, `MatSolveTransposeAdd()`
@*/
PetscErrorCode MatSolveAdd(Mat mat,Vec b,Vec y,Vec x)
{
  PetscScalar    one = 1.0;
  Vec            tmp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,4);
  PetscCheckSameComm(mat,1,b,2);
  PetscCheckSameComm(mat,1,y,3);
  PetscCheckSameComm(mat,1,x,4);
  PetscCheck(x != b,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  PetscCheck(mat->cmap->N == x->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->N,x->map->N);
  PetscCheck(mat->rmap->N == b->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->N,b->map->N);
  PetscCheck(mat->rmap->N == y->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->N,y->map->N);
  PetscCheck(mat->rmap->n == b->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: local dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->n,b->map->n);
  PetscCheck(x->map->n == y->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Vec x,Vec y: local dim %" PetscInt_FMT " %" PetscInt_FMT,x->map->n,y->map->n);
  if (!mat->rmap->N && !mat->cmap->N) PetscFunctionReturn(0);
   MatCheckPreallocated(mat,1);

  PetscCall(PetscLogEventBegin(MAT_SolveAdd,mat,b,x,y));
  if (mat->factorerrortype) {

    PetscCall(PetscInfo(mat,"MatFactorError %d\n",mat->factorerrortype));
    PetscCall(VecSetInf(x));
  } else if (mat->ops->solveadd) {
    PetscCall((*mat->ops->solveadd)(mat,b,y,x));
  } else {
    /* do the solve then the add manually */
    if (x != y) {
      PetscCall(MatSolve(mat,b,x));
      PetscCall(VecAXPY(x,one,y));
    } else {
      PetscCall(VecDuplicate(x,&tmp));
      PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)tmp));
      PetscCall(VecCopy(x,tmp));
      PetscCall(MatSolve(mat,b,x));
      PetscCall(VecAXPY(x,one,tmp));
      PetscCall(VecDestroy(&tmp));
    }
  }
  PetscCall(PetscLogEventEnd(MAT_SolveAdd,mat,b,x,y));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(0);
}

/*@
   MatSolveTranspose - Solves A' x = b, given a factored matrix.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat - the factored matrix
-  b - the right-hand-side vector

   Output Parameter:
.  x - the result vector

   Notes:
   The vectors b and x cannot be the same.  I.e., one cannot
   call MatSolveTranspose(A,x,x).

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatSolve()`, `MatSolveAdd()`, `MatSolveTransposeAdd()`
@*/
PetscErrorCode MatSolveTranspose(Mat mat,Vec b,Vec x)
{
  PetscErrorCode (*f)(Mat,Vec,Vec) = (!mat->ops->solvetranspose && mat->symmetric) ? mat->ops->solve : mat->ops->solvetranspose;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  PetscCheckSameComm(mat,1,b,2);
  PetscCheckSameComm(mat,1,x,3);
  PetscCheck(x != b,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  PetscCheck(mat->rmap->N == x->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->N,x->map->N);
  PetscCheck(mat->cmap->N == b->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->N,b->map->N);
  if (!mat->rmap->N && !mat->cmap->N) PetscFunctionReturn(0);
  MatCheckPreallocated(mat,1);
  PetscCall(PetscLogEventBegin(MAT_SolveTranspose,mat,b,x,0));
  if (mat->factorerrortype) {
    PetscCall(PetscInfo(mat,"MatFactorError %d\n",mat->factorerrortype));
    PetscCall(VecSetInf(x));
  } else {
    PetscCheck(f,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Matrix type %s",((PetscObject)mat)->type_name);
    PetscCall((*f)(mat,b,x));
  }
  PetscCall(PetscLogEventEnd(MAT_SolveTranspose,mat,b,x,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(0);
}

/*@
   MatSolveTransposeAdd - Computes x = y + inv(Transpose(A)) b, given a
                      factored matrix.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat - the factored matrix
.  b - the right-hand-side vector
-  y - the vector to be added to

   Output Parameter:
.  x - the result vector

   Notes:
   The vectors b and x cannot be the same.  I.e., one cannot
   call MatSolveTransposeAdd(A,x,y,x).

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatSolve()`, `MatSolveAdd()`, `MatSolveTranspose()`
@*/
PetscErrorCode MatSolveTransposeAdd(Mat mat,Vec b,Vec y,Vec x)
{
  PetscScalar    one = 1.0;
  Vec            tmp;
  PetscErrorCode (*f)(Mat,Vec,Vec,Vec) = (!mat->ops->solvetransposeadd && mat->symmetric) ? mat->ops->solveadd : mat->ops->solvetransposeadd;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,4);
  PetscCheckSameComm(mat,1,b,2);
  PetscCheckSameComm(mat,1,y,3);
  PetscCheckSameComm(mat,1,x,4);
  PetscCheck(x != b,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  PetscCheck(mat->rmap->N == x->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->N,x->map->N);
  PetscCheck(mat->cmap->N == b->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->N,b->map->N);
  PetscCheck(mat->cmap->N == y->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec y: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->N,y->map->N);
  PetscCheck(x->map->n == y->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Vec x,Vec y: local dim %" PetscInt_FMT " %" PetscInt_FMT,x->map->n,y->map->n);
  if (!mat->rmap->N && !mat->cmap->N) PetscFunctionReturn(0);
  MatCheckPreallocated(mat,1);

  PetscCall(PetscLogEventBegin(MAT_SolveTransposeAdd,mat,b,x,y));
  if (mat->factorerrortype) {
    PetscCall(PetscInfo(mat,"MatFactorError %d\n",mat->factorerrortype));
    PetscCall(VecSetInf(x));
  } else if (f) {
    PetscCall((*f)(mat,b,y,x));
  } else {
    /* do the solve then the add manually */
    if (x != y) {
      PetscCall(MatSolveTranspose(mat,b,x));
      PetscCall(VecAXPY(x,one,y));
    } else {
      PetscCall(VecDuplicate(x,&tmp));
      PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)tmp));
      PetscCall(VecCopy(x,tmp));
      PetscCall(MatSolveTranspose(mat,b,x));
      PetscCall(VecAXPY(x,one,tmp));
      PetscCall(VecDestroy(&tmp));
    }
  }
  PetscCall(PetscLogEventEnd(MAT_SolveTransposeAdd,mat,b,x,y));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------*/

/*@
   MatSOR - Computes relaxation (SOR, Gauss-Seidel) sweeps.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat - the matrix
.  b - the right hand side
.  omega - the relaxation factor
.  flag - flag indicating the type of SOR (see below)
.  shift -  diagonal shift
.  its - the number of iterations
-  lits - the number of local iterations

   Output Parameter:
.  x - the solution (can contain an initial guess, use option SOR_ZERO_INITIAL_GUESS to indicate no guess)

   SOR Flags:
+     SOR_FORWARD_SWEEP - forward SOR
.     SOR_BACKWARD_SWEEP - backward SOR
.     SOR_SYMMETRIC_SWEEP - SSOR (symmetric SOR)
.     SOR_LOCAL_FORWARD_SWEEP - local forward SOR
.     SOR_LOCAL_BACKWARD_SWEEP - local forward SOR
.     SOR_LOCAL_SYMMETRIC_SWEEP - local SSOR
.     SOR_APPLY_UPPER, SOR_APPLY_LOWER - applies
         upper/lower triangular part of matrix to
         vector (with omega)
-     SOR_ZERO_INITIAL_GUESS - zero initial guess

   Notes:
   SOR_LOCAL_FORWARD_SWEEP, SOR_LOCAL_BACKWARD_SWEEP, and
   SOR_LOCAL_SYMMETRIC_SWEEP perform separate independent smoothings
   on each processor.

   Application programmers will not generally use MatSOR() directly,
   but instead will employ the KSP/PC interface.

   Notes:
    for BAIJ, SBAIJ, and AIJ matrices with Inodes this does a block SOR smoothing, otherwise it does a pointwise smoothing

   Notes for Advanced Users:
   The flags are implemented as bitwise inclusive or operations.
   For example, use (SOR_ZERO_INITIAL_GUESS | SOR_SYMMETRIC_SWEEP)
   to specify a zero initial guess for SSOR.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Vectors x and b CANNOT be the same

   Developer Note: We should add block SOR support for AIJ matrices with block size set to great than one and no inodes

   Level: developer

@*/
PetscErrorCode MatSOR(Mat mat,Vec b,PetscReal omega,MatSORType flag,PetscReal shift,PetscInt its,PetscInt lits,Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,8);
  PetscCheckSameComm(mat,1,b,2);
  PetscCheckSameComm(mat,1,x,8);
  PetscCheck(mat->ops->sor,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->cmap->N == x->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec x: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->cmap->N,x->map->N);
  PetscCheck(mat->rmap->N == b->map->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: global dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->N,b->map->N);
  PetscCheck(mat->rmap->n == b->map->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat mat,Vec b: local dim %" PetscInt_FMT " %" PetscInt_FMT,mat->rmap->n,b->map->n);
  PetscCheck(its > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %" PetscInt_FMT " positive",its);
  PetscCheck(lits > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires local its %" PetscInt_FMT " positive",lits);
  PetscCheck(b != x,PETSC_COMM_SELF,PETSC_ERR_ARG_IDN,"b and x vector cannot be the same");

  MatCheckPreallocated(mat,1);
  PetscCall(PetscLogEventBegin(MAT_SOR,mat,b,x,0));
  PetscCall((*mat->ops->sor)(mat,b,omega,flag,shift,its,lits,x));
  PetscCall(PetscLogEventEnd(MAT_SOR,mat,b,x,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(0);
}

/*
      Default matrix copy routine.
*/
PetscErrorCode MatCopy_Basic(Mat A,Mat B,MatStructure str)
{
  PetscInt          i,rstart = 0,rend = 0,nz;
  const PetscInt    *cwork;
  const PetscScalar *vwork;

  PetscFunctionBegin;
  if (B->assembled) PetscCall(MatZeroEntries(B));
  if (str == SAME_NONZERO_PATTERN) {
    PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
    for (i=rstart; i<rend; i++) {
      PetscCall(MatGetRow(A,i,&nz,&cwork,&vwork));
      PetscCall(MatSetValues(B,1,&i,nz,cwork,vwork,INSERT_VALUES));
      PetscCall(MatRestoreRow(A,i,&nz,&cwork,&vwork));
    }
  } else {
    PetscCall(MatAYPX(B,0.0,A,str));
  }
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*@
   MatCopy - Copies a matrix to another matrix.

   Collective on Mat

   Input Parameters:
+  A - the matrix
-  str - SAME_NONZERO_PATTERN or DIFFERENT_NONZERO_PATTERN

   Output Parameter:
.  B - where the copy is put

   Notes:
   If you use SAME_NONZERO_PATTERN then the two matrices must have the same nonzero pattern or the routine will crash.

   MatCopy() copies the matrix entries of a matrix to another existing
   matrix (after first zeroing the second matrix).  A related routine is
   MatConvert(), which first creates a new matrix and then copies the data.

   Level: intermediate

.seealso: `MatConvert()`, `MatDuplicate()`
@*/
PetscErrorCode MatCopy(Mat A,Mat B,MatStructure str)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidType(A,1);
  PetscValidType(B,2);
  PetscCheckSameComm(A,1,B,2);
  MatCheckPreallocated(B,2);
  PetscCheck(A->assembled,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!A->factortype,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(A->rmap->N == B->rmap->N && A->cmap->N == B->cmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Mat A,Mat B: global dim (%" PetscInt_FMT ",%" PetscInt_FMT ") (%" PetscInt_FMT ",%" PetscInt_FMT ")",A->rmap->N,B->rmap->N,A->cmap->N,B->cmap->N);
  MatCheckPreallocated(A,1);
  if (A == B) PetscFunctionReturn(0);

  PetscCall(PetscLogEventBegin(MAT_Copy,A,B,0,0));
  if (A->ops->copy) {
    PetscCall((*A->ops->copy)(A,B,str));
  } else { /* generic conversion */
    PetscCall(MatCopy_Basic(A,B,str));
  }

  B->stencil.dim = A->stencil.dim;
  B->stencil.noc = A->stencil.noc;
  for (i=0; i<=A->stencil.dim; i++) {
    B->stencil.dims[i]   = A->stencil.dims[i];
    B->stencil.starts[i] = A->stencil.starts[i];
  }

  PetscCall(PetscLogEventEnd(MAT_Copy,A,B,0,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)B));
  PetscFunctionReturn(0);
}

/*@C
   MatConvert - Converts a matrix to another matrix, either of the same
   or different type.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  newtype - new matrix type.  Use MATSAME to create a new matrix of the
   same type as the original matrix.
-  reuse - denotes if the destination matrix is to be created or reused.
   Use MAT_INPLACE_MATRIX for inplace conversion (that is when you want the input mat to be changed to contain the matrix in the new format), otherwise use
   MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX (can only be used after the first call was made with MAT_INITIAL_MATRIX, causes the matrix space in M to be reused).

   Output Parameter:
.  M - pointer to place new matrix

   Notes:
   MatConvert() first creates a new matrix and then copies the data from
   the first matrix.  A related routine is MatCopy(), which copies the matrix
   entries of one matrix to another already existing matrix context.

   Cannot be used to convert a sequential matrix to parallel or parallel to sequential,
   the MPI communicator of the generated matrix is always the same as the communicator
   of the input matrix.

   Level: intermediate

.seealso: `MatCopy()`, `MatDuplicate()`
@*/
PetscErrorCode MatConvert(Mat mat,MatType newtype,MatReuse reuse,Mat *M)
{
  PetscBool      sametype,issame,flg;
  PetscBool3     issymmetric,ishermitian;
  char           convname[256],mtype[256];
  Mat            B;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidPointer(M,4);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);

  PetscCall(PetscOptionsGetString(((PetscObject)mat)->options,((PetscObject)mat)->prefix,"-matconvert_type",mtype,sizeof(mtype),&flg));
  if (flg) newtype = mtype;

  PetscCall(PetscObjectTypeCompare((PetscObject)mat,newtype,&sametype));
  PetscCall(PetscStrcmp(newtype,"same",&issame));
  PetscCheck(!(reuse == MAT_INPLACE_MATRIX) || !(mat != *M),PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"MAT_INPLACE_MATRIX requires same input and output matrix");
  PetscCheck(!(reuse == MAT_REUSE_MATRIX) || !(mat == *M),PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"MAT_REUSE_MATRIX means reuse matrix in final argument, perhaps you mean MAT_INPLACE_MATRIX");

  if ((reuse == MAT_INPLACE_MATRIX) && (issame || sametype)) {
    PetscCall(PetscInfo(mat,"Early return for inplace %s %d %d\n",((PetscObject)mat)->type_name,sametype,issame));
    PetscFunctionReturn(0);
  }

  /* Cache Mat options because some converters use MatHeaderReplace  */
  issymmetric = mat->symmetric;
  ishermitian = mat->hermitian;

  if ((sametype || issame) && (reuse==MAT_INITIAL_MATRIX) && mat->ops->duplicate) {
    PetscCall(PetscInfo(mat,"Calling duplicate for initial matrix %s %d %d\n",((PetscObject)mat)->type_name,sametype,issame));
    PetscCall((*mat->ops->duplicate)(mat,MAT_COPY_VALUES,M));
  } else {
    PetscErrorCode (*conv)(Mat, MatType,MatReuse,Mat*)=NULL;
    const char     *prefix[3] = {"seq","mpi",""};
    PetscInt       i;
    /*
       Order of precedence:
       0) See if newtype is a superclass of the current matrix.
       1) See if a specialized converter is known to the current matrix.
       2) See if a specialized converter is known to the desired matrix class.
       3) See if a good general converter is registered for the desired class
          (as of 6/27/03 only MATMPIADJ falls into this category).
       4) See if a good general converter is known for the current matrix.
       5) Use a really basic converter.
    */

    /* 0) See if newtype is a superclass of the current matrix.
          i.e mat is mpiaij and newtype is aij */
    for (i=0; i<2; i++) {
      PetscCall(PetscStrncpy(convname,prefix[i],sizeof(convname)));
      PetscCall(PetscStrlcat(convname,newtype,sizeof(convname)));
      PetscCall(PetscStrcmp(convname,((PetscObject)mat)->type_name,&flg));
      PetscCall(PetscInfo(mat,"Check superclass %s %s -> %d\n",convname,((PetscObject)mat)->type_name,flg));
      if (flg) {
        if (reuse == MAT_INPLACE_MATRIX) {
          PetscCall(PetscInfo(mat,"Early return\n"));
          PetscFunctionReturn(0);
        } else if (reuse == MAT_INITIAL_MATRIX && mat->ops->duplicate) {
          PetscCall(PetscInfo(mat,"Calling MatDuplicate\n"));
          PetscCall((*mat->ops->duplicate)(mat,MAT_COPY_VALUES,M));
          PetscFunctionReturn(0);
        } else if (reuse == MAT_REUSE_MATRIX && mat->ops->copy) {
          PetscCall(PetscInfo(mat,"Calling MatCopy\n"));
          PetscCall(MatCopy(mat,*M,SAME_NONZERO_PATTERN));
          PetscFunctionReturn(0);
        }
      }
    }
    /* 1) See if a specialized converter is known to the current matrix and the desired class */
    for (i=0; i<3; i++) {
      PetscCall(PetscStrncpy(convname,"MatConvert_",sizeof(convname)));
      PetscCall(PetscStrlcat(convname,((PetscObject)mat)->type_name,sizeof(convname)));
      PetscCall(PetscStrlcat(convname,"_",sizeof(convname)));
      PetscCall(PetscStrlcat(convname,prefix[i],sizeof(convname)));
      PetscCall(PetscStrlcat(convname,issame ? ((PetscObject)mat)->type_name : newtype,sizeof(convname)));
      PetscCall(PetscStrlcat(convname,"_C",sizeof(convname)));
      PetscCall(PetscObjectQueryFunction((PetscObject)mat,convname,&conv));
      PetscCall(PetscInfo(mat,"Check specialized (1) %s (%s) -> %d\n",convname,((PetscObject)mat)->type_name,!!conv));
      if (conv) goto foundconv;
    }

    /* 2)  See if a specialized converter is known to the desired matrix class. */
    PetscCall(MatCreate(PetscObjectComm((PetscObject)mat),&B));
    PetscCall(MatSetSizes(B,mat->rmap->n,mat->cmap->n,mat->rmap->N,mat->cmap->N));
    PetscCall(MatSetType(B,newtype));
    for (i=0; i<3; i++) {
      PetscCall(PetscStrncpy(convname,"MatConvert_",sizeof(convname)));
      PetscCall(PetscStrlcat(convname,((PetscObject)mat)->type_name,sizeof(convname)));
      PetscCall(PetscStrlcat(convname,"_",sizeof(convname)));
      PetscCall(PetscStrlcat(convname,prefix[i],sizeof(convname)));
      PetscCall(PetscStrlcat(convname,newtype,sizeof(convname)));
      PetscCall(PetscStrlcat(convname,"_C",sizeof(convname)));
      PetscCall(PetscObjectQueryFunction((PetscObject)B,convname,&conv));
      PetscCall(PetscInfo(mat,"Check specialized (2) %s (%s) -> %d\n",convname,((PetscObject)B)->type_name,!!conv));
      if (conv) {
        PetscCall(MatDestroy(&B));
        goto foundconv;
      }
    }

    /* 3) See if a good general converter is registered for the desired class */
    conv = B->ops->convertfrom;
    PetscCall(PetscInfo(mat,"Check convertfrom (%s) -> %d\n",((PetscObject)B)->type_name,!!conv));
    PetscCall(MatDestroy(&B));
    if (conv) goto foundconv;

    /* 4) See if a good general converter is known for the current matrix */
    if (mat->ops->convert) conv = mat->ops->convert;
    PetscCall(PetscInfo(mat,"Check general convert (%s) -> %d\n",((PetscObject)mat)->type_name,!!conv));
    if (conv) goto foundconv;

    /* 5) Use a really basic converter. */
    PetscCall(PetscInfo(mat,"Using MatConvert_Basic\n"));
    conv = MatConvert_Basic;

foundconv:
    PetscCall(PetscLogEventBegin(MAT_Convert,mat,0,0,0));
    PetscCall((*conv)(mat,newtype,reuse,M));
    if (mat->rmap->mapping && mat->cmap->mapping && !(*M)->rmap->mapping && !(*M)->cmap->mapping) {
      /* the block sizes must be same if the mappings are copied over */
      (*M)->rmap->bs = mat->rmap->bs;
      (*M)->cmap->bs = mat->cmap->bs;
      PetscCall(PetscObjectReference((PetscObject)mat->rmap->mapping));
      PetscCall(PetscObjectReference((PetscObject)mat->cmap->mapping));
      (*M)->rmap->mapping = mat->rmap->mapping;
      (*M)->cmap->mapping = mat->cmap->mapping;
    }
    (*M)->stencil.dim = mat->stencil.dim;
    (*M)->stencil.noc = mat->stencil.noc;
    for (i=0; i<=mat->stencil.dim; i++) {
      (*M)->stencil.dims[i]   = mat->stencil.dims[i];
      (*M)->stencil.starts[i] = mat->stencil.starts[i];
    }
    PetscCall(PetscLogEventEnd(MAT_Convert,mat,0,0,0));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)*M));

  /* Copy Mat options */
  if (issymmetric == PETSC_BOOL3_TRUE) PetscCall(MatSetOption(*M,MAT_SYMMETRIC,PETSC_TRUE));
  else if (issymmetric == PETSC_BOOL3_FALSE) PetscCall(MatSetOption(*M,MAT_SYMMETRIC,PETSC_FALSE));
  if (ishermitian == PETSC_BOOL3_TRUE) PetscCall(MatSetOption(*M,MAT_HERMITIAN,PETSC_TRUE));
  else if (ishermitian == PETSC_BOOL3_FALSE) PetscCall(MatSetOption(*M,MAT_HERMITIAN,PETSC_FALSE));
  PetscFunctionReturn(0);
}

/*@C
   MatFactorGetSolverType - Returns name of the package providing the factorization routines

   Not Collective

   Input Parameter:
.  mat - the matrix, must be a factored matrix

   Output Parameter:
.   type - the string name of the package (do not free this string)

   Notes:
      In Fortran you pass in a empty string and the package name will be copied into it.
    (Make sure the string is long enough)

   Level: intermediate

.seealso: `MatCopy()`, `MatDuplicate()`, `MatGetFactorAvailable()`, `MatGetFactor()`
@*/
PetscErrorCode MatFactorGetSolverType(Mat mat, MatSolverType *type)
{
  PetscErrorCode (*conv)(Mat,MatSolverType*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidPointer(type,2);
  PetscCheck(mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscCall(PetscObjectQueryFunction((PetscObject)mat,"MatFactorGetSolverType_C",&conv));
  if (conv) PetscCall((*conv)(mat,type));
  else *type = MATSOLVERPETSC;
  PetscFunctionReturn(0);
}

typedef struct _MatSolverTypeForSpecifcType* MatSolverTypeForSpecifcType;
struct _MatSolverTypeForSpecifcType {
  MatType                        mtype;
  /* no entry for MAT_FACTOR_NONE */
  PetscErrorCode                 (*createfactor[MAT_FACTOR_NUM_TYPES-1])(Mat,MatFactorType,Mat*);
  MatSolverTypeForSpecifcType next;
};

typedef struct _MatSolverTypeHolder* MatSolverTypeHolder;
struct _MatSolverTypeHolder {
  char                        *name;
  MatSolverTypeForSpecifcType handlers;
  MatSolverTypeHolder         next;
};

static MatSolverTypeHolder MatSolverTypeHolders = NULL;

/*@C
   MatSolverTypeRegister - Registers a MatSolverType that works for a particular matrix type

   Input Parameters:
+    package - name of the package, for example petsc or superlu
.    mtype - the matrix type that works with this package
.    ftype - the type of factorization supported by the package
-    createfactor - routine that will create the factored matrix ready to be used

    Level: intermediate

.seealso: `MatCopy()`, `MatDuplicate()`, `MatGetFactorAvailable()`, `MatGetFactor()`
@*/
PetscErrorCode MatSolverTypeRegister(MatSolverType package,MatType mtype,MatFactorType ftype,PetscErrorCode (*createfactor)(Mat,MatFactorType,Mat*))
{
  MatSolverTypeHolder         next = MatSolverTypeHolders,prev = NULL;
  PetscBool                   flg;
  MatSolverTypeForSpecifcType inext,iprev = NULL;

  PetscFunctionBegin;
  PetscCall(MatInitializePackage());
  if (!next) {
    PetscCall(PetscNew(&MatSolverTypeHolders));
    PetscCall(PetscStrallocpy(package,&MatSolverTypeHolders->name));
    PetscCall(PetscNew(&MatSolverTypeHolders->handlers));
    PetscCall(PetscStrallocpy(mtype,(char **)&MatSolverTypeHolders->handlers->mtype));
    MatSolverTypeHolders->handlers->createfactor[(int)ftype-1] = createfactor;
    PetscFunctionReturn(0);
  }
  while (next) {
    PetscCall(PetscStrcasecmp(package,next->name,&flg));
    if (flg) {
      PetscCheck(next->handlers,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatSolverTypeHolder is missing handlers");
      inext = next->handlers;
      while (inext) {
        PetscCall(PetscStrcasecmp(mtype,inext->mtype,&flg));
        if (flg) {
          inext->createfactor[(int)ftype-1] = createfactor;
          PetscFunctionReturn(0);
        }
        iprev = inext;
        inext = inext->next;
      }
      PetscCall(PetscNew(&iprev->next));
      PetscCall(PetscStrallocpy(mtype,(char **)&iprev->next->mtype));
      iprev->next->createfactor[(int)ftype-1] = createfactor;
      PetscFunctionReturn(0);
    }
    prev = next;
    next = next->next;
  }
  PetscCall(PetscNew(&prev->next));
  PetscCall(PetscStrallocpy(package,&prev->next->name));
  PetscCall(PetscNew(&prev->next->handlers));
  PetscCall(PetscStrallocpy(mtype,(char **)&prev->next->handlers->mtype));
  prev->next->handlers->createfactor[(int)ftype-1] = createfactor;
  PetscFunctionReturn(0);
}

/*@C
   MatSolverTypeGet - Gets the function that creates the factor matrix if it exist

   Input Parameters:
+    type - name of the package, for example petsc or superlu
.    ftype - the type of factorization supported by the type
-    mtype - the matrix type that works with this type

   Output Parameters:
+   foundtype - PETSC_TRUE if the type was registered
.   foundmtype - PETSC_TRUE if the type supports the requested mtype
-   createfactor - routine that will create the factored matrix ready to be used or NULL if not found

    Level: intermediate

.seealso: `MatCopy()`, `MatDuplicate()`, `MatGetFactorAvailable()`, `MatSolverTypeRegister()`, `MatGetFactor()`
@*/
PetscErrorCode MatSolverTypeGet(MatSolverType type,MatType mtype,MatFactorType ftype,PetscBool *foundtype,PetscBool *foundmtype,PetscErrorCode (**createfactor)(Mat,MatFactorType,Mat*))
{
  MatSolverTypeHolder         next = MatSolverTypeHolders;
  PetscBool                   flg;
  MatSolverTypeForSpecifcType inext;

  PetscFunctionBegin;
  if (foundtype) *foundtype = PETSC_FALSE;
  if (foundmtype) *foundmtype = PETSC_FALSE;
  if (createfactor) *createfactor = NULL;

  if (type) {
    while (next) {
      PetscCall(PetscStrcasecmp(type,next->name,&flg));
      if (flg) {
        if (foundtype) *foundtype = PETSC_TRUE;
        inext = next->handlers;
        while (inext) {
          PetscCall(PetscStrbeginswith(mtype,inext->mtype,&flg));
          if (flg) {
            if (foundmtype) *foundmtype = PETSC_TRUE;
            if (createfactor)  *createfactor  = inext->createfactor[(int)ftype-1];
            PetscFunctionReturn(0);
          }
          inext = inext->next;
        }
      }
      next = next->next;
    }
  } else {
    while (next) {
      inext = next->handlers;
      while (inext) {
        PetscCall(PetscStrcmp(mtype,inext->mtype,&flg));
        if (flg && inext->createfactor[(int)ftype-1]) {
          if (foundtype) *foundtype = PETSC_TRUE;
          if (foundmtype)   *foundmtype   = PETSC_TRUE;
          if (createfactor) *createfactor = inext->createfactor[(int)ftype-1];
          PetscFunctionReturn(0);
        }
        inext = inext->next;
      }
      next = next->next;
    }
    /* try with base classes inext->mtype */
    next = MatSolverTypeHolders;
    while (next) {
      inext = next->handlers;
      while (inext) {
        PetscCall(PetscStrbeginswith(mtype,inext->mtype,&flg));
        if (flg && inext->createfactor[(int)ftype-1]) {
          if (foundtype) *foundtype = PETSC_TRUE;
          if (foundmtype)   *foundmtype   = PETSC_TRUE;
          if (createfactor) *createfactor = inext->createfactor[(int)ftype-1];
          PetscFunctionReturn(0);
        }
        inext = inext->next;
      }
      next = next->next;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolverTypeDestroy(void)
{
  MatSolverTypeHolder         next = MatSolverTypeHolders,prev;
  MatSolverTypeForSpecifcType inext,iprev;

  PetscFunctionBegin;
  while (next) {
    PetscCall(PetscFree(next->name));
    inext = next->handlers;
    while (inext) {
      PetscCall(PetscFree(inext->mtype));
      iprev = inext;
      inext = inext->next;
      PetscCall(PetscFree(iprev));
    }
    prev = next;
    next = next->next;
    PetscCall(PetscFree(prev));
  }
  MatSolverTypeHolders = NULL;
  PetscFunctionReturn(0);
}

/*@C
   MatFactorGetCanUseOrdering - Indicates if the factorization can use the ordering provided in MatLUFactorSymbolic(), MatCholeskyFactorSymbolic()

   Logically Collective on Mat

   Input Parameters:
.  mat - the matrix

   Output Parameters:
.  flg - PETSC_TRUE if uses the ordering

   Notes:
      Most internal PETSc factorizations use the ordering passed to the factorization routine but external
      packages do not, thus we want to skip generating the ordering when it is not needed or used.

   Level: developer

.seealso: `MatCopy()`, `MatDuplicate()`, `MatGetFactorAvailable()`, `MatGetFactor()`, `MatLUFactorSymbolic()`, `MatCholeskyFactorSymbolic()`
@*/
PetscErrorCode MatFactorGetCanUseOrdering(Mat mat, PetscBool *flg)
{
  PetscFunctionBegin;
  *flg = mat->canuseordering;
  PetscFunctionReturn(0);
}

/*@C
   MatFactorGetPreferredOrdering - The preferred ordering for a particular matrix factor object

   Logically Collective on Mat

   Input Parameters:
.  mat - the matrix

   Output Parameters:
.  otype - the preferred type

   Level: developer

.seealso: `MatCopy()`, `MatDuplicate()`, `MatGetFactorAvailable()`, `MatGetFactor()`, `MatLUFactorSymbolic()`, `MatCholeskyFactorSymbolic()`
@*/
PetscErrorCode MatFactorGetPreferredOrdering(Mat mat, MatFactorType ftype, MatOrderingType *otype)
{
  PetscFunctionBegin;
  *otype = mat->preferredordering[ftype];
  PetscCheck(*otype,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatFactor did not have a preferred ordering");
  PetscFunctionReturn(0);
}

/*@C
   MatGetFactor - Returns a matrix suitable to calls to MatXXFactorSymbolic()

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  type - name of solver type, for example, superlu, petsc (to use PETSc's default)
-  ftype - factor type, MAT_FACTOR_LU, MAT_FACTOR_CHOLESKY, MAT_FACTOR_ICC, MAT_FACTOR_ILU,

   Output Parameters:
.  f - the factor matrix used with MatXXFactorSymbolic() calls

   Options Database Key:
.  -mat_factor_bind_factorization <host, device> - Where to do matrix factorization? Default is device (might consume more device memory.
                                  One can choose host to save device memory). Currently only supported with SEQAIJCUSPARSE matrices.

   Notes:
      Some PETSc matrix formats have alternative solvers available that are contained in alternative packages
     such as pastix, superlu, mumps etc.

      PETSc must have been ./configure to use the external solver, using the option --download-package

      Some of the packages have options for controlling the factorization, these are in the form -prefix_mat_packagename_packageoption
      where prefix is normally obtained from the calling `KSP`/`PC`. If `MatGetFactor()` is called directly one can set
      call `MatSetOptionsPrefixFactor()` on the originating matrix or  `MatSetOptionsPrefix()` on the resulting factor matrix.

   Developer Notes:
      This should actually be called MatCreateFactor() since it creates a new factor object

   Level: intermediate

.seealso: `MatCopy()`, `MatDuplicate()`, `MatGetFactorAvailable()`, `MatFactorGetCanUseOrdering()`, `MatSolverTypeRegister()`
@*/
PetscErrorCode MatGetFactor(Mat mat, MatSolverType type,MatFactorType ftype,Mat *f)
{
  PetscBool      foundtype,foundmtype;
  PetscErrorCode (*conv)(Mat,MatFactorType,Mat*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);

  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);

  PetscCall(MatSolverTypeGet(type,((PetscObject)mat)->type_name,ftype,&foundtype,&foundmtype,&conv));
  if (!foundtype) {
    if (type) {
      SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_MISSING_FACTOR,"Could not locate solver type %s for factorization type %s and matrix type %s. Perhaps you must ./configure with --download-%s",type,MatFactorTypes[ftype],((PetscObject)mat)->type_name,type);
    } else {
      SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_MISSING_FACTOR,"Could not locate a solver type for factorization type %s and matrix type %s.",MatFactorTypes[ftype],((PetscObject)mat)->type_name);
    }
  }
  PetscCheck(foundmtype,PetscObjectComm((PetscObject)mat),PETSC_ERR_MISSING_FACTOR,"MatSolverType %s does not support matrix type %s",type,((PetscObject)mat)->type_name);
  PetscCheck(conv,PetscObjectComm((PetscObject)mat),PETSC_ERR_MISSING_FACTOR,"MatSolverType %s does not support factorization type %s for matrix type %s",type,MatFactorTypes[ftype],((PetscObject)mat)->type_name);

  PetscCall((*conv)(mat,ftype,f));
  if (mat->factorprefix) PetscCall(MatSetOptionsPrefix(*f,mat->factorprefix));
  PetscFunctionReturn(0);
}

/*@C
   MatGetFactorAvailable - Returns a a flag if matrix supports particular type and factor type

   Not Collective

   Input Parameters:
+  mat - the matrix
.  type - name of solver type, for example, superlu, petsc (to use PETSc's default)
-  ftype - factor type, MAT_FACTOR_LU, MAT_FACTOR_CHOLESKY, MAT_FACTOR_ICC, MAT_FACTOR_ILU,

   Output Parameter:
.    flg - PETSC_TRUE if the factorization is available

   Notes:
      Some PETSc matrix formats have alternative solvers available that are contained in alternative packages
     such as pastix, superlu, mumps etc.

      PETSc must have been ./configure to use the external solver, using the option --download-package

   Developer Notes:
      This should actually be called MatCreateFactorAvailable() since MatGetFactor() creates a new factor object

   Level: intermediate

.seealso: `MatCopy()`, `MatDuplicate()`, `MatGetFactor()`, `MatSolverTypeRegister()`
@*/
PetscErrorCode MatGetFactorAvailable(Mat mat, MatSolverType type,MatFactorType ftype,PetscBool  *flg)
{
  PetscErrorCode (*gconv)(Mat,MatFactorType,Mat*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidBoolPointer(flg,4);

  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);

  PetscCall(MatSolverTypeGet(type,((PetscObject)mat)->type_name,ftype,NULL,NULL,&gconv));
  *flg = gconv ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
   MatDuplicate - Duplicates a matrix including the non-zero structure.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  op - One of MAT_DO_NOT_COPY_VALUES, MAT_COPY_VALUES, or MAT_SHARE_NONZERO_PATTERN.
        See the manual page for MatDuplicateOption for an explanation of these options.

   Output Parameter:
.  M - pointer to place new matrix

   Level: intermediate

   Notes:
    You cannot change the nonzero pattern for the parent or child matrix if you use MAT_SHARE_NONZERO_PATTERN.
    May be called with an unassembled input Mat if MAT_DO_NOT_COPY_VALUES is used, in which case the output Mat is unassembled as well.
    When original mat is a product of matrix operation, e.g., an output of MatMatMult() or MatCreateSubMatrix(), only the simple matrix data structure of mat is duplicated and the internal data structures created for the reuse of previous matrix operations are not duplicated. User should not use MatDuplicate() to create new matrix M if M is intended to be reused as the product of matrix operation.

.seealso: `MatCopy()`, `MatConvert()`, `MatDuplicateOption`
@*/
PetscErrorCode MatDuplicate(Mat mat,MatDuplicateOption op,Mat *M)
{
  Mat            B;
  VecType        vtype;
  PetscInt       i;
  PetscObject    dm;
  void           (*viewf)(void);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidPointer(M,3);
  PetscCheck(op != MAT_COPY_VALUES || mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"MAT_COPY_VALUES not allowed for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);

  *M = NULL;
  PetscCheck(mat->ops->duplicate,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Not written for matrix type %s",((PetscObject)mat)->type_name);
  PetscCall(PetscLogEventBegin(MAT_Convert,mat,0,0,0));
  PetscCall((*mat->ops->duplicate)(mat,op,M));
  PetscCall(PetscLogEventEnd(MAT_Convert,mat,0,0,0));
  B    = *M;

  PetscCall(MatGetOperation(mat,MATOP_VIEW,&viewf));
  if (viewf) PetscCall(MatSetOperation(B,MATOP_VIEW,viewf));
  PetscCall(MatGetVecType(mat,&vtype));
  PetscCall(MatSetVecType(B,vtype));

  B->stencil.dim = mat->stencil.dim;
  B->stencil.noc = mat->stencil.noc;
  for (i=0; i<=mat->stencil.dim; i++) {
    B->stencil.dims[i]   = mat->stencil.dims[i];
    B->stencil.starts[i] = mat->stencil.starts[i];
  }

  B->nooffproczerorows = mat->nooffproczerorows;
  B->nooffprocentries  = mat->nooffprocentries;

  PetscCall(PetscObjectQuery((PetscObject) mat, "__PETSc_dm", &dm));
  if (dm) {
    PetscCall(PetscObjectCompose((PetscObject) B, "__PETSc_dm", dm));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)B));
  PetscFunctionReturn(0);
}

/*@
   MatGetDiagonal - Gets the diagonal of a matrix.

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix
-  v - the vector for storing the diagonal

   Output Parameter:
.  v - the diagonal of the matrix

   Level: intermediate

   Note:
   Currently only correct in parallel for square matrices.

.seealso: `MatGetRow()`, `MatCreateSubMatrices()`, `MatCreateSubMatrix()`, `MatGetRowMaxAbs()`
@*/
PetscErrorCode MatGetDiagonal(Mat mat,Vec v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(mat->ops->getdiagonal,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,1);

  PetscCall((*mat->ops->getdiagonal)(mat,v));
  PetscCall(PetscObjectStateIncrease((PetscObject)v));
  PetscFunctionReturn(0);
}

/*@C
   MatGetRowMin - Gets the minimum value (of the real part) of each
        row of the matrix

   Logically Collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  v - the vector for storing the maximums
-  idx - the indices of the column found for each row (optional)

   Level: intermediate

   Notes:
    The result of this call are the same as if one converted the matrix to dense format
      and found the minimum value in each row (i.e. the implicit zeros are counted as zeros).

    This code is only implemented for a couple of matrix formats.

.seealso: `MatGetDiagonal()`, `MatCreateSubMatrices()`, `MatCreateSubMatrix()`, `MatGetRowMaxAbs()`,
          `MatGetRowMax()`
@*/
PetscErrorCode MatGetRowMin(Mat mat,Vec v,PetscInt idx[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");

  if (!mat->cmap->N) {
    PetscCall(VecSet(v,PETSC_MAX_REAL));
    if (idx) {
      PetscInt i,m = mat->rmap->n;
      for (i=0; i<m; i++) idx[i] = -1;
    }
  } else {
    PetscCheck(mat->ops->getrowmin,PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
    MatCheckPreallocated(mat,1);
  }
  PetscCall((*mat->ops->getrowmin)(mat,v,idx));
  PetscCall(PetscObjectStateIncrease((PetscObject)v));
  PetscFunctionReturn(0);
}

/*@C
   MatGetRowMinAbs - Gets the minimum value (in absolute value) of each
        row of the matrix

   Logically Collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  v - the vector for storing the minimums
-  idx - the indices of the column found for each row (or NULL if not needed)

   Level: intermediate

   Notes:
    if a row is completely empty or has only 0.0 values then the idx[] value for that
    row is 0 (the first column).

    This code is only implemented for a couple of matrix formats.

.seealso: `MatGetDiagonal()`, `MatCreateSubMatrices()`, `MatCreateSubMatrix()`, `MatGetRowMax()`, `MatGetRowMaxAbs()`, `MatGetRowMin()`
@*/
PetscErrorCode MatGetRowMinAbs(Mat mat,Vec v,PetscInt idx[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  if (!mat->cmap->N) {
    PetscCall(VecSet(v,0.0));
    if (idx) {
      PetscInt i,m = mat->rmap->n;
      for (i=0; i<m; i++) idx[i] = -1;
    }
  } else {
    PetscCheck(mat->ops->getrowminabs,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
    MatCheckPreallocated(mat,1);
    if (idx) PetscCall(PetscArrayzero(idx,mat->rmap->n));
    PetscCall((*mat->ops->getrowminabs)(mat,v,idx));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)v));
  PetscFunctionReturn(0);
}

/*@C
   MatGetRowMax - Gets the maximum value (of the real part) of each
        row of the matrix

   Logically Collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  v - the vector for storing the maximums
-  idx - the indices of the column found for each row (optional)

   Level: intermediate

   Notes:
    The result of this call are the same as if one converted the matrix to dense format
      and found the minimum value in each row (i.e. the implicit zeros are counted as zeros).

    This code is only implemented for a couple of matrix formats.

.seealso: `MatGetDiagonal()`, `MatCreateSubMatrices()`, `MatCreateSubMatrix()`, `MatGetRowMaxAbs()`, `MatGetRowMin()`
@*/
PetscErrorCode MatGetRowMax(Mat mat,Vec v,PetscInt idx[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");

  if (!mat->cmap->N) {
    PetscCall(VecSet(v,PETSC_MIN_REAL));
    if (idx) {
      PetscInt i,m = mat->rmap->n;
      for (i=0; i<m; i++) idx[i] = -1;
    }
  } else {
    PetscCheck(mat->ops->getrowmax,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
    MatCheckPreallocated(mat,1);
    PetscCall((*mat->ops->getrowmax)(mat,v,idx));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)v));
  PetscFunctionReturn(0);
}

/*@C
   MatGetRowMaxAbs - Gets the maximum value (in absolute value) of each
        row of the matrix

   Logically Collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  v - the vector for storing the maximums
-  idx - the indices of the column found for each row (or NULL if not needed)

   Level: intermediate

   Notes:
    if a row is completely empty or has only 0.0 values then the idx[] value for that
    row is 0 (the first column).

    This code is only implemented for a couple of matrix formats.

.seealso: `MatGetDiagonal()`, `MatCreateSubMatrices()`, `MatCreateSubMatrix()`, `MatGetRowMax()`, `MatGetRowMin()`
@*/
PetscErrorCode MatGetRowMaxAbs(Mat mat,Vec v,PetscInt idx[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");

  if (!mat->cmap->N) {
    PetscCall(VecSet(v,0.0));
    if (idx) {
      PetscInt i,m = mat->rmap->n;
      for (i=0; i<m; i++) idx[i] = -1;
    }
  } else {
    PetscCheck(mat->ops->getrowmaxabs,PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
    MatCheckPreallocated(mat,1);
    if (idx) PetscCall(PetscArrayzero(idx,mat->rmap->n));
    PetscCall((*mat->ops->getrowmaxabs)(mat,v,idx));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)v));
  PetscFunctionReturn(0);
}

/*@
   MatGetRowSum - Gets the sum of each row of the matrix

   Logically or Neighborhood Collective on Mat

   Input Parameters:
.  mat - the matrix

   Output Parameter:
.  v - the vector for storing the sum of rows

   Level: intermediate

   Notes:
    This code is slow since it is not currently specialized for different formats

.seealso: `MatGetDiagonal()`, `MatCreateSubMatrices()`, `MatCreateSubMatrix()`, `MatGetRowMax()`, `MatGetRowMin()`
@*/
PetscErrorCode MatGetRowSum(Mat mat, Vec v)
{
  Vec            ones;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  MatCheckPreallocated(mat,1);
  PetscCall(MatCreateVecs(mat,&ones,NULL));
  PetscCall(VecSet(ones,1.));
  PetscCall(MatMult(mat,ones,v));
  PetscCall(VecDestroy(&ones));
  PetscFunctionReturn(0);
}

/*@
   MatTranspose - Computes an in-place or out-of-place transpose of a matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix to transpose
-  reuse - either MAT_INITIAL_MATRIX, MAT_REUSE_MATRIX, or MAT_INPLACE_MATRIX

   Output Parameter:
.  B - the transpose

   Notes:
     If you use MAT_INPLACE_MATRIX then you must pass in &mat for B

     MAT_REUSE_MATRIX causes the B matrix from a previous call to this function with MAT_INITIAL_MATRIX to be used

     Consider using MatCreateTranspose() instead if you only need a matrix that behaves like the transpose, but don't need the storage to be changed.

   Level: intermediate

.seealso: `MatMultTranspose()`, `MatMultTransposeAdd()`, `MatIsTranspose()`, `MatReuse`
@*/
PetscErrorCode MatTranspose(Mat mat,MatReuse reuse,Mat *B)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->ops->transpose,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  PetscCheck(reuse != MAT_INPLACE_MATRIX || mat == *B,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"MAT_INPLACE_MATRIX requires last matrix to match first");
  PetscCheck(reuse != MAT_REUSE_MATRIX || mat != *B,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Perhaps you mean MAT_INPLACE_MATRIX");
  MatCheckPreallocated(mat,1);

  PetscCall(PetscLogEventBegin(MAT_Transpose,mat,0,0,0));
  PetscCall((*mat->ops->transpose)(mat,reuse,B));
  PetscCall(PetscLogEventEnd(MAT_Transpose,mat,0,0,0));
  if (B) PetscCall(PetscObjectStateIncrease((PetscObject)*B));
  PetscFunctionReturn(0);
}

/*@
   MatIsTranspose - Test whether a matrix is another one's transpose,
        or its own, in which case it tests symmetry.

   Collective on Mat

   Input Parameters:
+  A - the matrix to test
-  B - the matrix to test against, this can equal the first parameter

   Output Parameters:
.  flg - the result

   Notes:
   Only available for SeqAIJ/MPIAIJ matrices. The sequential algorithm
   has a running time of the order of the number of nonzeros; the parallel
   test involves parallel copies of the block-offdiagonal parts of the matrix.

   Level: intermediate

.seealso: `MatTranspose()`, `MatIsSymmetric()`, `MatIsHermitian()`
@*/
PetscErrorCode MatIsTranspose(Mat A,Mat B,PetscReal tol,PetscBool *flg)
{
  PetscErrorCode (*f)(Mat,Mat,PetscReal,PetscBool*),(*g)(Mat,Mat,PetscReal,PetscBool*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidBoolPointer(flg,4);
  PetscCall(PetscObjectQueryFunction((PetscObject)A,"MatIsTranspose_C",&f));
  PetscCall(PetscObjectQueryFunction((PetscObject)B,"MatIsTranspose_C",&g));
  *flg = PETSC_FALSE;
  if (f && g) {
    PetscCheck(f == g,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_NOTSAMETYPE,"Matrices do not have the same comparator for symmetry test");
    PetscCall((*f)(A,B,tol,flg));
  } else {
    MatType mattype;

    PetscCall(MatGetType(f ? B : A,&mattype));
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrix of type %s does not support checking for transpose",mattype);
  }
  PetscFunctionReturn(0);
}

/*@
   MatHermitianTranspose - Computes an in-place or out-of-place transpose of a matrix in complex conjugate.

   Collective on Mat

   Input Parameters:
+  mat - the matrix to transpose and complex conjugate
-  reuse - either MAT_INITIAL_MATRIX, MAT_REUSE_MATRIX, or MAT_INPLACE_MATRIX

   Output Parameter:
.  B - the Hermitian

   Level: intermediate

.seealso: `MatTranspose()`, `MatMultTranspose()`, `MatMultTransposeAdd()`, `MatIsTranspose()`, `MatReuse`
@*/
PetscErrorCode MatHermitianTranspose(Mat mat,MatReuse reuse,Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatTranspose(mat,reuse,B));
#if defined(PETSC_USE_COMPLEX)
  PetscCall(MatConjugate(*B));
#endif
  PetscFunctionReturn(0);
}

/*@
   MatIsHermitianTranspose - Test whether a matrix is another one's Hermitian transpose,

   Collective on Mat

   Input Parameters:
+  A - the matrix to test
-  B - the matrix to test against, this can equal the first parameter

   Output Parameters:
.  flg - the result

   Notes:
   Only available for SeqAIJ/MPIAIJ matrices. The sequential algorithm
   has a running time of the order of the number of nonzeros; the parallel
   test involves parallel copies of the block-offdiagonal parts of the matrix.

   Level: intermediate

.seealso: `MatTranspose()`, `MatIsSymmetric()`, `MatIsHermitian()`, `MatIsTranspose()`
@*/
PetscErrorCode MatIsHermitianTranspose(Mat A,Mat B,PetscReal tol,PetscBool *flg)
{
  PetscErrorCode (*f)(Mat,Mat,PetscReal,PetscBool*),(*g)(Mat,Mat,PetscReal,PetscBool*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidBoolPointer(flg,4);
  PetscCall(PetscObjectQueryFunction((PetscObject)A,"MatIsHermitianTranspose_C",&f));
  PetscCall(PetscObjectQueryFunction((PetscObject)B,"MatIsHermitianTranspose_C",&g));
  if (f && g) {
    PetscCheck(f != g,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_NOTSAMETYPE,"Matrices do not have the same comparator for Hermitian test");
    PetscCall((*f)(A,B,tol,flg));
  }
  PetscFunctionReturn(0);
}

/*@
   MatPermute - Creates a new matrix with rows and columns permuted from the
   original.

   Collective on Mat

   Input Parameters:
+  mat - the matrix to permute
.  row - row permutation, each processor supplies only the permutation for its rows
-  col - column permutation, each processor supplies only the permutation for its columns

   Output Parameters:
.  B - the permuted matrix

   Level: advanced

   Note:
   The index sets map from row/col of permuted matrix to row/col of original matrix.
   The index sets should be on the same communicator as Mat and have the same local sizes.

   Developer Note:
     If you want to implement MatPermute for a matrix type, and your approach doesn't
     exploit the fact that row and col are permutations, consider implementing the
     more general MatCreateSubMatrix() instead.

.seealso: `MatGetOrdering()`, `ISAllGather()`

@*/
PetscErrorCode MatPermute(Mat mat,IS row,IS col,Mat *B)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(row,IS_CLASSID,2);
  PetscValidHeaderSpecific(col,IS_CLASSID,3);
  PetscValidPointer(B,4);
  PetscCheckSameComm(mat,1,row,2);
  if (row != col) PetscCheckSameComm(row,2,col,3);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->ops->permute || mat->ops->createsubmatrix,PETSC_COMM_SELF,PETSC_ERR_SUP,"MatPermute not available for Mat type %s",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,1);

  if (mat->ops->permute) {
    PetscCall((*mat->ops->permute)(mat,row,col,B));
    PetscCall(PetscObjectStateIncrease((PetscObject)*B));
  } else {
    PetscCall(MatCreateSubMatrix(mat, row, col, MAT_INITIAL_MATRIX, B));
  }
  PetscFunctionReturn(0);
}

/*@
   MatEqual - Compares two matrices.

   Collective on Mat

   Input Parameters:
+  A - the first matrix
-  B - the second matrix

   Output Parameter:
.  flg - PETSC_TRUE if the matrices are equal; PETSC_FALSE otherwise.

   Level: intermediate

@*/
PetscErrorCode MatEqual(Mat A,Mat B,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidType(A,1);
  PetscValidType(B,2);
  PetscValidBoolPointer(flg,3);
  PetscCheckSameComm(A,1,B,2);
  MatCheckPreallocated(A,1);
  MatCheckPreallocated(B,2);
  PetscCheck(A->assembled,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(B->assembled,PetscObjectComm((PetscObject)B),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(A->rmap->N == B->rmap->N && A->cmap->N == B->cmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Mat A,Mat B: global dim %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT,A->rmap->N,B->rmap->N,A->cmap->N,B->cmap->N);
  if (A->ops->equal && A->ops->equal == B->ops->equal) {
    PetscCall((*A->ops->equal)(A,B,flg));
  } else {
    PetscCall(MatMultEqual(A,B,10,flg));
  }
  PetscFunctionReturn(0);
}

/*@
   MatDiagonalScale - Scales a matrix on the left and right by diagonal
   matrices that are stored as vectors.  Either of the two scaling
   matrices can be NULL.

   Collective on Mat

   Input Parameters:
+  mat - the matrix to be scaled
.  l - the left scaling vector (or NULL)
-  r - the right scaling vector (or NULL)

   Notes:
   MatDiagonalScale() computes A = LAR, where
   L = a diagonal matrix (stored as a vector), R = a diagonal matrix (stored as a vector)
   The L scales the rows of the matrix, the R scales the columns of the matrix.

   Level: intermediate

.seealso: `MatScale()`, `MatShift()`, `MatDiagonalSet()`
@*/
PetscErrorCode MatDiagonalScale(Mat mat,Vec l,Vec r)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (l) {PetscValidHeaderSpecific(l,VEC_CLASSID,2);PetscCheckSameComm(mat,1,l,2);}
  if (r) {PetscValidHeaderSpecific(r,VEC_CLASSID,3);PetscCheckSameComm(mat,1,r,3);}
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);
  if (!l && !r) PetscFunctionReturn(0);

  PetscCheck(mat->ops->diagonalscale,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  PetscCall(PetscLogEventBegin(MAT_Scale,mat,0,0,0));
  PetscCall((*mat->ops->diagonalscale)(mat,l,r));
  PetscCall(PetscLogEventEnd(MAT_Scale,mat,0,0,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
  if (l != r) mat->symmetric = PETSC_BOOL3_FALSE;
  PetscFunctionReturn(0);
}

/*@
    MatScale - Scales all elements of a matrix by a given number.

    Logically Collective on Mat

    Input Parameters:
+   mat - the matrix to be scaled
-   a  - the scaling value

    Output Parameter:
.   mat - the scaled matrix

    Level: intermediate

.seealso: `MatDiagonalScale()`
@*/
PetscErrorCode MatScale(Mat mat,PetscScalar a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscCheck(a == (PetscScalar)1.0 || mat->ops->scale,PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscValidLogicalCollectiveScalar(mat,a,2);
  MatCheckPreallocated(mat,1);

  PetscCall(PetscLogEventBegin(MAT_Scale,mat,0,0,0));
  if (a != (PetscScalar)1.0) {
    PetscCall((*mat->ops->scale)(mat,a));
    PetscCall(PetscObjectStateIncrease((PetscObject)mat));
  }
  PetscCall(PetscLogEventEnd(MAT_Scale,mat,0,0,0));
  PetscFunctionReturn(0);
}

/*@
   MatNorm - Calculates various norms of a matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  type - the type of norm, NORM_1, NORM_FROBENIUS, NORM_INFINITY

   Output Parameter:
.  nrm - the resulting norm

   Level: intermediate

@*/
PetscErrorCode MatNorm(Mat mat,NormType type,PetscReal *nrm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidRealPointer(nrm,3);

  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->ops->norm,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,1);

  PetscCall((*mat->ops->norm)(mat,type,nrm));
  PetscFunctionReturn(0);
}

/*
     This variable is used to prevent counting of MatAssemblyBegin() that
   are called from within a MatAssemblyEnd().
*/
static PetscInt MatAssemblyEnd_InUse = 0;
/*@
   MatAssemblyBegin - Begins assembling the matrix.  This routine should
   be called after completing all calls to MatSetValues().

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  type - type of assembly, either MAT_FLUSH_ASSEMBLY or MAT_FINAL_ASSEMBLY

   Notes:
   MatSetValues() generally caches the values.  The matrix is ready to
   use only after MatAssemblyBegin() and MatAssemblyEnd() have been called.
   Use MAT_FLUSH_ASSEMBLY when switching between ADD_VALUES and INSERT_VALUES
   in MatSetValues(); use MAT_FINAL_ASSEMBLY for the final assembly before
   using the matrix.

   ALL processes that share a matrix MUST call MatAssemblyBegin() and MatAssemblyEnd() the SAME NUMBER of times, and each time with the
   same flag of MAT_FLUSH_ASSEMBLY or MAT_FINAL_ASSEMBLY for all processes. Thus you CANNOT locally change from ADD_VALUES to INSERT_VALUES, that is
   a global collective operation requring all processes that share the matrix.

   Space for preallocated nonzeros that is not filled by a call to MatSetValues() or a related routine are compressed
   out by assembly. If you intend to use that extra space on a subsequent assembly, be sure to insert explicit zeros
   before MAT_FINAL_ASSEMBLY so the space is not compressed out.

   Level: beginner

.seealso: `MatAssemblyEnd()`, `MatSetValues()`, `MatAssembled()`
@*/
PetscErrorCode MatAssemblyBegin(Mat mat,MatAssemblyType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  MatCheckPreallocated(mat,1);
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix.\nDid you forget to call MatSetUnfactored()?");
  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE;
    mat->assembled     = PETSC_FALSE;
  }

  if (!MatAssemblyEnd_InUse) {
    PetscCall(PetscLogEventBegin(MAT_AssemblyBegin,mat,0,0,0));
    if (mat->ops->assemblybegin) PetscCall((*mat->ops->assemblybegin)(mat,type));
    PetscCall(PetscLogEventEnd(MAT_AssemblyBegin,mat,0,0,0));
  } else if (mat->ops->assemblybegin) PetscCall((*mat->ops->assemblybegin)(mat,type));
  PetscFunctionReturn(0);
}

/*@
   MatAssembled - Indicates if a matrix has been assembled and is ready for
     use; for example, in matrix-vector product.

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  assembled - PETSC_TRUE or PETSC_FALSE

   Level: advanced

.seealso: `MatAssemblyEnd()`, `MatSetValues()`, `MatAssemblyBegin()`
@*/
PetscErrorCode MatAssembled(Mat mat,PetscBool *assembled)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidBoolPointer(assembled,2);
  *assembled = mat->assembled;
  PetscFunctionReturn(0);
}

/*@
   MatAssemblyEnd - Completes assembling the matrix.  This routine should
   be called after MatAssemblyBegin().

   Collective on Mat

   Input Parameters:
+  mat - the matrix
-  type - type of assembly, either MAT_FLUSH_ASSEMBLY or MAT_FINAL_ASSEMBLY

   Options Database Keys:
+  -mat_view ::ascii_info - Prints info on matrix at conclusion of MatEndAssembly()
.  -mat_view ::ascii_info_detail - Prints more detailed info
.  -mat_view - Prints matrix in ASCII format
.  -mat_view ::ascii_matlab - Prints matrix in Matlab format
.  -mat_view draw - PetscDraws nonzero structure of matrix, using MatView() and PetscDrawOpenX().
.  -display <name> - Sets display name (default is host)
.  -draw_pause <sec> - Sets number of seconds to pause after display
.  -mat_view socket - Sends matrix to socket, can be accessed from Matlab (See Users-Manual: ch_matlab)
.  -viewer_socket_machine <machine> - Machine to use for socket
.  -viewer_socket_port <port> - Port number to use for socket
-  -mat_view binary:filename[:append] - Save matrix to file in binary format

   Notes:
   MatSetValues() generally caches the values.  The matrix is ready to
   use only after MatAssemblyBegin() and MatAssemblyEnd() have been called.
   Use MAT_FLUSH_ASSEMBLY when switching between ADD_VALUES and INSERT_VALUES
   in MatSetValues(); use MAT_FINAL_ASSEMBLY for the final assembly before
   using the matrix.

   Space for preallocated nonzeros that is not filled by a call to MatSetValues() or a related routine are compressed
   out by assembly. If you intend to use that extra space on a subsequent assembly, be sure to insert explicit zeros
   before MAT_FINAL_ASSEMBLY so the space is not compressed out.

   Level: beginner

.seealso: `MatAssemblyBegin()`, `MatSetValues()`, `PetscDrawOpenX()`, `PetscDrawCreate()`, `MatView()`, `MatAssembled()`, `PetscViewerSocketOpen()`
@*/
PetscErrorCode MatAssemblyEnd(Mat mat,MatAssemblyType type)
{
  static PetscInt inassm = 0;
  PetscBool       flg    = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);

  inassm++;
  MatAssemblyEnd_InUse++;
  if (MatAssemblyEnd_InUse == 1) { /* Do the logging only the first time through */
    PetscCall(PetscLogEventBegin(MAT_AssemblyEnd,mat,0,0,0));
    if (mat->ops->assemblyend) PetscCall((*mat->ops->assemblyend)(mat,type));
    PetscCall(PetscLogEventEnd(MAT_AssemblyEnd,mat,0,0,0));
  } else if (mat->ops->assemblyend) PetscCall((*mat->ops->assemblyend)(mat,type));

  /* Flush assembly is not a true assembly */
  if (type != MAT_FLUSH_ASSEMBLY) {
    if (mat->num_ass) {
      if (!mat->symmetry_eternal) {
        mat->symmetric              = PETSC_BOOL3_UNKNOWN;
        mat->hermitian              = PETSC_BOOL3_UNKNOWN;
      }
      if (!mat->structural_symmetry_eternal && mat->ass_nonzerostate != mat->nonzerostate) {
        mat->structurally_symmetric = PETSC_BOOL3_UNKNOWN;
      }
      if (!mat->spd_eternal) mat->spd = PETSC_BOOL3_UNKNOWN;
    }
    mat->num_ass++;
    mat->assembled        = PETSC_TRUE;
    mat->ass_nonzerostate = mat->nonzerostate;
  }

  mat->insertmode = NOT_SET_VALUES;
  MatAssemblyEnd_InUse--;
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
  if (inassm == 1 && type != MAT_FLUSH_ASSEMBLY) {
    PetscCall(MatViewFromOptions(mat,NULL,"-mat_view"));

    if (mat->checksymmetryonassembly) {
      PetscCall(MatIsSymmetric(mat,mat->checksymmetrytol,&flg));
      if (flg) {
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)mat),"Matrix is symmetric (tolerance %g)\n",(double)mat->checksymmetrytol));
      } else {
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)mat),"Matrix is not symmetric (tolerance %g)\n",(double)mat->checksymmetrytol));
      }
    }
    if (mat->nullsp && mat->checknullspaceonassembly) {
      PetscCall(MatNullSpaceTest(mat->nullsp,mat,NULL));
    }
  }
  inassm--;
  PetscFunctionReturn(0);
}

/*@
   MatSetOption - Sets a parameter option for a matrix. Some options
   may be specific to certain storage formats.  Some options
   determine how values will be inserted (or added). Sorted,
   row-oriented input will generally assemble the fastest. The default
   is row-oriented.

   Logically Collective on Mat for certain operations, such as MAT_SPD, not collective for MAT_ROW_ORIENTED, see MatOption

   Input Parameters:
+  mat - the matrix
.  option - the option, one of those listed below (and possibly others),
-  flg - turn the option on (PETSC_TRUE) or off (PETSC_FALSE)

  Options Describing Matrix Structure:
+    MAT_SPD - symmetric positive definite
.    MAT_SYMMETRIC - symmetric in terms of both structure and value
.    MAT_HERMITIAN - transpose is the complex conjugation
.    MAT_STRUCTURALLY_SYMMETRIC - symmetric nonzero structure
-    MAT_SYMMETRY_ETERNAL - indicates the symmetry (or Hermitian structure) or its absence will persist through any changes to the matrix
-    MAT_STRUCTURAL_SYMMETRY_ETERNAL - indicates the structural symmetry or its absence will persist through any changes to the matrix

   These are not really options of the matrix, they are knowledge about the structure of the matrix that users may provide so that they
   do not need to be computed (usually at a high cost)

   Options For Use with MatSetValues():
   Insert a logically dense subblock, which can be
.    MAT_ROW_ORIENTED - row-oriented (default)

   Note these options reflect the data you pass in with MatSetValues(); it has
   nothing to do with how the data is stored internally in the matrix
   data structure.

   When (re)assembling a matrix, we can restrict the input for
   efficiency/debugging purposes.  These options include
+    MAT_NEW_NONZERO_LOCATIONS - additional insertions will be allowed if they generate a new nonzero (slow)
.    MAT_FORCE_DIAGONAL_ENTRIES - forces diagonal entries to be allocated
.    MAT_IGNORE_OFF_PROC_ENTRIES - drops off-processor entries
.    MAT_NEW_NONZERO_LOCATION_ERR - generates an error for new matrix entry
.    MAT_USE_HASH_TABLE - uses a hash table to speed up matrix assembly
.    MAT_NO_OFF_PROC_ENTRIES - you know each process will only set values for its own rows, will generate an error if
        any process sets values for another process. This avoids all reductions in the MatAssembly routines and thus improves
        performance for very large process counts.
-    MAT_SUBSET_OFF_PROC_ENTRIES - you know that the first assembly after setting this flag will set a superset
        of the off-process entries required for all subsequent assemblies. This avoids a rendezvous step in the MatAssembly
        functions, instead sending only neighbor messages.

   Notes:
   Except for MAT_UNUSED_NONZERO_LOCATION_ERR and  MAT_ROW_ORIENTED all processes that share the matrix must pass the same value in flg!

   Some options are relevant only for particular matrix types and
   are thus ignored by others.  Other options are not supported by
   certain matrix types and will generate an error message if set.

   If using a Fortran 77 module to compute a matrix, one may need to
   use the column-oriented option (or convert to the row-oriented
   format).

   MAT_NEW_NONZERO_LOCATIONS set to PETSC_FALSE indicates that any add or insertion
   that would generate a new entry in the nonzero structure is instead
   ignored.  Thus, if memory has not alredy been allocated for this particular
   data, then the insertion is ignored. For dense matrices, in which
   the entire array is allocated, no entries are ever ignored.
   Set after the first MatAssemblyEnd(). If this option is set then the MatAssemblyBegin/End() processes has one less global reduction

   MAT_NEW_NONZERO_LOCATION_ERR set to PETSC_TRUE indicates that any add or insertion
   that would generate a new entry in the nonzero structure instead produces
   an error. (Currently supported for AIJ and BAIJ formats only.) If this option is set then the MatAssemblyBegin/End() processes has one less global reduction

   MAT_NEW_NONZERO_ALLOCATION_ERR set to PETSC_TRUE indicates that any add or insertion
   that would generate a new entry that has not been preallocated will
   instead produce an error. (Currently supported for AIJ and BAIJ formats
   only.) This is a useful flag when debugging matrix memory preallocation.
   If this option is set then the MatAssemblyBegin/End() processes has one less global reduction

   MAT_IGNORE_OFF_PROC_ENTRIES set to PETSC_TRUE indicates entries destined for
   other processors should be dropped, rather than stashed.
   This is useful if you know that the "owning" processor is also
   always generating the correct matrix entries, so that PETSc need
   not transfer duplicate entries generated on another processor.

   MAT_USE_HASH_TABLE indicates that a hash table be used to improve the
   searches during matrix assembly. When this flag is set, the hash table
   is created during the first Matrix Assembly. This hash table is
   used the next time through, during MatSetVaules()/MatSetVaulesBlocked()
   to improve the searching of indices. MAT_NEW_NONZERO_LOCATIONS flag
   should be used with MAT_USE_HASH_TABLE flag. This option is currently
   supported by MATMPIBAIJ format only.

   MAT_KEEP_NONZERO_PATTERN indicates when MatZeroRows() is called the zeroed entries
   are kept in the nonzero structure

   MAT_IGNORE_ZERO_ENTRIES - for AIJ/IS matrices this will stop zero values from creating
   a zero location in the matrix

   MAT_USE_INODES - indicates using inode version of the code - works with AIJ matrix types

   MAT_NO_OFF_PROC_ZERO_ROWS - you know each process will only zero its own rows. This avoids all reductions in the
        zero row routines and thus improves performance for very large process counts.

   MAT_IGNORE_LOWER_TRIANGULAR - For SBAIJ matrices will ignore any insertions you make in the lower triangular
        part of the matrix (since they should match the upper triangular part).

   MAT_SORTED_FULL - each process provides exactly its local rows; all column indices for a given row are passed in a
                     single call to MatSetValues(), preallocation is perfect, row oriented, INSERT_VALUES is used. Common
                     with finite difference schemes with non-periodic boundary conditions.

   Level: intermediate

.seealso: `MatOption`, `Mat`, `MatGetOption()`

@*/
PetscErrorCode MatSetOption(Mat mat,MatOption op,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (op > 0) {
    PetscValidLogicalCollectiveEnum(mat,op,2);
    PetscValidLogicalCollectiveBool(mat,flg,3);
  }

  PetscCheck(((int) op) > MAT_OPTION_MIN && ((int) op) < MAT_OPTION_MAX,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_OUTOFRANGE,"Options %d is out of range",(int)op);

  switch (op) {
  case MAT_FORCE_DIAGONAL_ENTRIES:
    mat->force_diagonals = flg;
    PetscFunctionReturn(0);
  case MAT_NO_OFF_PROC_ENTRIES:
    mat->nooffprocentries = flg;
    PetscFunctionReturn(0);
  case MAT_SUBSET_OFF_PROC_ENTRIES:
    mat->assembly_subset = flg;
    if (!mat->assembly_subset) { /* See the same logic in VecAssembly wrt VEC_SUBSET_OFF_PROC_ENTRIES */
#if !defined(PETSC_HAVE_MPIUNI)
      PetscCall(MatStashScatterDestroy_BTS(&mat->stash));
#endif
      mat->stash.first_assembly_done = PETSC_FALSE;
    }
    PetscFunctionReturn(0);
  case MAT_NO_OFF_PROC_ZERO_ROWS:
    mat->nooffproczerorows = flg;
    PetscFunctionReturn(0);
  case MAT_SPD:
    if (flg) {
      mat->spd                     = PETSC_BOOL3_TRUE;
      mat->symmetric               = PETSC_BOOL3_TRUE;
      mat->structurally_symmetric  = PETSC_BOOL3_TRUE;
    } else {
      mat->spd = PETSC_BOOL3_FALSE;
    }
    break;
  case MAT_SYMMETRIC:
    mat->symmetric                       = flg ? PETSC_BOOL3_TRUE : PETSC_BOOL3_FALSE;
    if (flg) mat->structurally_symmetric = PETSC_BOOL3_TRUE;
#if !defined(PETSC_USE_COMPLEX)
    mat->hermitian                       = flg ? PETSC_BOOL3_TRUE : PETSC_BOOL3_FALSE;
#endif
    break;
  case MAT_HERMITIAN:
    mat->hermitian                       = flg ? PETSC_BOOL3_TRUE : PETSC_BOOL3_FALSE;
    if (flg) mat->structurally_symmetric = PETSC_BOOL3_TRUE;
#if !defined(PETSC_USE_COMPLEX)
    mat->symmetric                       = flg ? PETSC_BOOL3_TRUE : PETSC_BOOL3_FALSE;
#endif
    break;
  case MAT_STRUCTURALLY_SYMMETRIC:
    mat->structurally_symmetric = flg ? PETSC_BOOL3_TRUE : PETSC_BOOL3_FALSE;
    break;
  case MAT_SYMMETRY_ETERNAL:
    mat->symmetry_eternal = flg ? PETSC_TRUE : PETSC_FALSE;
    if (flg) mat->structural_symmetry_eternal = PETSC_TRUE;
    break;
  case MAT_STRUCTURAL_SYMMETRY_ETERNAL:
    mat->structural_symmetry_eternal = flg;
    break;
  case MAT_SPD_ETERNAL:
    mat->spd_eternal = flg;
    if (flg) {
      mat->structural_symmetry_eternal = PETSC_TRUE;
      mat->symmetry_eternal            = PETSC_TRUE;
    }
    break;
  case MAT_STRUCTURE_ONLY:
    mat->structure_only = flg;
    break;
  case MAT_SORTED_FULL:
    mat->sortedfull = flg;
    break;
  default:
    break;
  }
  if (mat->ops->setoption) PetscCall((*mat->ops->setoption)(mat,op,flg));
  PetscFunctionReturn(0);
}

/*@
   MatGetOption - Gets a parameter option that has been set for a matrix.

   Logically Collective on Mat for certain operations, such as MAT_SPD, not collective for MAT_ROW_ORIENTED, see MatOption

   Input Parameters:
+  mat - the matrix
-  option - the option, this only responds to certain options, check the code for which ones

   Output Parameter:
.  flg - turn the option on (PETSC_TRUE) or off (PETSC_FALSE)

    Notes:
    Can only be called after MatSetSizes() and MatSetType() have been set.

    Certain option values may be unknown, for those use the routines `MatIsSymmetric()`, `MatIsHermitian()`,  `MatIsStructurallySymmetric()`, or
    `MatIsSymmetricKnown()`, `MatIsHermitianKnown()`,  `MatIsStructurallySymmetricKnown()`

   Level: intermediate

.seealso: `MatOption`, `MatSetOption()`, `MatIsSymmetric()`, `MatIsHermitian()`,  `MatIsStructurallySymmetric()`,
    `MatIsSymmetricKnown()`, `MatIsHermitianKnown()`,  `MatIsStructurallySymmetricKnown()`

@*/
PetscErrorCode MatGetOption(Mat mat,MatOption op,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);

  PetscCheck(((int) op) > MAT_OPTION_MIN && ((int) op) < MAT_OPTION_MAX,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_OUTOFRANGE,"Options %d is out of range",(int)op);
  PetscCheck(((PetscObject)mat)->type_name,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_TYPENOTSET,"Cannot get options until type and size have been set, see MatSetType() and MatSetSizes()");

  switch (op) {
  case MAT_NO_OFF_PROC_ENTRIES:
    *flg = mat->nooffprocentries;
    break;
  case MAT_NO_OFF_PROC_ZERO_ROWS:
    *flg = mat->nooffproczerorows;
    break;
  case MAT_SYMMETRIC:
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Use MatIsSymmetric() or MatIsSymmetricKnown()");
    break;
  case MAT_HERMITIAN:
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Use MatIsHermitian() or MatIsHermitianKnown()");
    break;
  case MAT_STRUCTURALLY_SYMMETRIC:
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Use MatIsStructurallySymmetric() or MatIsStructurallySymmetricKnown()");
    break;
  case MAT_SPD:
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Use MatIsSPDKnown()");
    break;
  case MAT_SYMMETRY_ETERNAL:
    *flg = mat->symmetry_eternal;
    break;
  case MAT_STRUCTURAL_SYMMETRY_ETERNAL:
    *flg = mat->symmetry_eternal;
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}

/*@
   MatZeroEntries - Zeros all entries of a matrix.  For sparse matrices
   this routine retains the old nonzero structure.

   Logically Collective on Mat

   Input Parameters:
.  mat - the matrix

   Level: intermediate

   Notes:
    If the matrix was not preallocated then a default, likely poor preallocation will be set in the matrix, so this should be called after the preallocation phase.
   See the Performance chapter of the users manual for information on preallocating matrices.

.seealso: `MatZeroRows()`
@*/
PetscErrorCode MatZeroEntries(Mat mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->insertmode == NOT_SET_VALUES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for matrices where you have set values but not yet assembled");
  PetscCheck(mat->ops->zeroentries,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,1);

  PetscCall(PetscLogEventBegin(MAT_ZeroEntries,mat,0,0,0));
  PetscCall((*mat->ops->zeroentries)(mat));
  PetscCall(PetscLogEventEnd(MAT_ZeroEntries,mat,0,0,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
  PetscFunctionReturn(0);
}

/*@
   MatZeroRowsColumns - Zeros all entries (except possibly the main diagonal)
   of a set of rows and columns of a matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  numRows - the number of rows to remove
.  rows - the global row indices
.  diag - value put in all diagonals of eliminated rows (0.0 will even eliminate diagonal entry)
.  x - optional vector of solutions for zeroed rows (other entries in vector are not used)
-  b - optional vector of right hand side, that will be adjusted by provided solution

   Notes:
   This does not change the nonzero structure of the matrix, it merely zeros those entries in the matrix.

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing 0.0 as the final argument).

   For the parallel case, all processes that share the matrix (i.e.,
   those in the communicator used for matrix creation) MUST call this
   routine, regardless of whether any rows being zeroed are owned by
   them.

   Each processor can indicate any rows in the entire matrix to be zeroed (i.e. each process does NOT have to
   list only rows local to itself).

   The option MAT_NO_OFF_PROC_ZERO_ROWS does not apply to this routine.

   Level: intermediate

.seealso: `MatZeroRowsIS()`, `MatZeroRows()`, `MatZeroRowsLocalIS()`, `MatZeroRowsStencil()`, `MatZeroEntries()`, `MatZeroRowsLocal()`, `MatSetOption()`,
          `MatZeroRowsColumnsLocal()`, `MatZeroRowsColumnsLocalIS()`, `MatZeroRowsColumnsIS()`, `MatZeroRowsColumnsStencil()`
@*/
PetscErrorCode MatZeroRowsColumns(Mat mat,PetscInt numRows,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (numRows) PetscValidIntPointer(rows,3);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->ops->zerorowscolumns,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,1);

  PetscCall((*mat->ops->zerorowscolumns)(mat,numRows,rows,diag,x,b));
  PetscCall(MatViewFromOptions(mat,NULL,"-mat_view"));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
  PetscFunctionReturn(0);
}

/*@
   MatZeroRowsColumnsIS - Zeros all entries (except possibly the main diagonal)
   of a set of rows and columns of a matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  is - the rows to zero
.  diag - value put in all diagonals of eliminated rows (0.0 will even eliminate diagonal entry)
.  x - optional vector of solutions for zeroed rows (other entries in vector are not used)
-  b - optional vector of right hand side, that will be adjusted by provided solution

   Notes:
   This does not change the nonzero structure of the matrix, it merely zeros those entries in the matrix.

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing 0.0 as the final argument).

   For the parallel case, all processes that share the matrix (i.e.,
   those in the communicator used for matrix creation) MUST call this
   routine, regardless of whether any rows being zeroed are owned by
   them.

   Each processor can indicate any rows in the entire matrix to be zeroed (i.e. each process does NOT have to
   list only rows local to itself).

   The option MAT_NO_OFF_PROC_ZERO_ROWS does not apply to this routine.

   Level: intermediate

.seealso: `MatZeroRowsIS()`, `MatZeroRowsColumns()`, `MatZeroRowsLocalIS()`, `MatZeroRowsStencil()`, `MatZeroEntries()`, `MatZeroRowsLocal()`, `MatSetOption()`,
          `MatZeroRowsColumnsLocal()`, `MatZeroRowsColumnsLocalIS()`, `MatZeroRows()`, `MatZeroRowsColumnsStencil()`
@*/
PetscErrorCode MatZeroRowsColumnsIS(Mat mat,IS is,PetscScalar diag,Vec x,Vec b)
{
  PetscInt       numRows;
  const PetscInt *rows;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  PetscValidType(mat,1);
  PetscValidType(is,2);
  PetscCall(ISGetLocalSize(is,&numRows));
  PetscCall(ISGetIndices(is,&rows));
  PetscCall(MatZeroRowsColumns(mat,numRows,rows,diag,x,b));
  PetscCall(ISRestoreIndices(is,&rows));
  PetscFunctionReturn(0);
}

/*@
   MatZeroRows - Zeros all entries (except possibly the main diagonal)
   of a set of rows of a matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  numRows - the number of rows to remove
.  rows - the global row indices
.  diag - value put in all diagonals of eliminated rows (0.0 will even eliminate diagonal entry)
.  x - optional vector of solutions for zeroed rows (other entries in vector are not used)
-  b - optional vector of right hand side, that will be adjusted by provided solution

   Notes:
   For the AIJ and BAIJ matrix formats this removes the old nonzero structure,
   but does not release memory.  For the dense and block diagonal
   formats this does not alter the nonzero structure.

   If the option MatSetOption(mat,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE) the nonzero structure
   of the matrix is not changed (even for AIJ and BAIJ matrices) the values are
   merely zeroed.

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing 0.0 as the final argument).

   For the parallel case, all processes that share the matrix (i.e.,
   those in the communicator used for matrix creation) MUST call this
   routine, regardless of whether any rows being zeroed are owned by
   them.

   Each processor can indicate any rows in the entire matrix to be zeroed (i.e. each process does NOT have to
   list only rows local to itself).

   You can call MatSetOption(mat,MAT_NO_OFF_PROC_ZERO_ROWS,PETSC_TRUE) if each process indicates only rows it
   owns that are to be zeroed. This saves a global synchronization in the implementation.

   Level: intermediate

.seealso: `MatZeroRowsIS()`, `MatZeroRowsColumns()`, `MatZeroRowsLocalIS()`, `MatZeroRowsStencil()`, `MatZeroEntries()`, `MatZeroRowsLocal()`, `MatSetOption()`,
          `MatZeroRowsColumnsLocal()`, `MatZeroRowsColumnsLocalIS()`, `MatZeroRowsColumnsIS()`, `MatZeroRowsColumnsStencil()`
@*/
PetscErrorCode MatZeroRows(Mat mat,PetscInt numRows,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (numRows) PetscValidIntPointer(rows,3);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->ops->zerorows,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,1);

  PetscCall((*mat->ops->zerorows)(mat,numRows,rows,diag,x,b));
  PetscCall(MatViewFromOptions(mat,NULL,"-mat_view"));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
  PetscFunctionReturn(0);
}

/*@
   MatZeroRowsIS - Zeros all entries (except possibly the main diagonal)
   of a set of rows of a matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  is - index set of rows to remove (if NULL then no row is removed)
.  diag - value put in all diagonals of eliminated rows
.  x - optional vector of solutions for zeroed rows (other entries in vector are not used)
-  b - optional vector of right hand side, that will be adjusted by provided solution

   Notes:
   For the AIJ and BAIJ matrix formats this removes the old nonzero structure,
   but does not release memory.  For the dense and block diagonal
   formats this does not alter the nonzero structure.

   If the option MatSetOption(mat,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE) the nonzero structure
   of the matrix is not changed (even for AIJ and BAIJ matrices) the values are
   merely zeroed.

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing 0.0 as the final argument).

   For the parallel case, all processes that share the matrix (i.e.,
   those in the communicator used for matrix creation) MUST call this
   routine, regardless of whether any rows being zeroed are owned by
   them.

   Each processor can indicate any rows in the entire matrix to be zeroed (i.e. each process does NOT have to
   list only rows local to itself).

   You can call MatSetOption(mat,MAT_NO_OFF_PROC_ZERO_ROWS,PETSC_TRUE) if each process indicates only rows it
   owns that are to be zeroed. This saves a global synchronization in the implementation.

   Level: intermediate

.seealso: `MatZeroRows()`, `MatZeroRowsColumns()`, `MatZeroRowsLocalIS()`, `MatZeroRowsStencil()`, `MatZeroEntries()`, `MatZeroRowsLocal()`, `MatSetOption()`,
          `MatZeroRowsColumnsLocal()`, `MatZeroRowsColumnsLocalIS()`, `MatZeroRowsColumnsIS()`, `MatZeroRowsColumnsStencil()`
@*/
PetscErrorCode MatZeroRowsIS(Mat mat,IS is,PetscScalar diag,Vec x,Vec b)
{
  PetscInt       numRows = 0;
  const PetscInt *rows = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (is) {
    PetscValidHeaderSpecific(is,IS_CLASSID,2);
    PetscCall(ISGetLocalSize(is,&numRows));
    PetscCall(ISGetIndices(is,&rows));
  }
  PetscCall(MatZeroRows(mat,numRows,rows,diag,x,b));
  if (is) {
    PetscCall(ISRestoreIndices(is,&rows));
  }
  PetscFunctionReturn(0);
}

/*@
   MatZeroRowsStencil - Zeros all entries (except possibly the main diagonal)
   of a set of rows of a matrix. These rows must be local to the process.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  numRows - the number of rows to remove
.  rows - the grid coordinates (and component number when dof > 1) for matrix rows
.  diag - value put in all diagonals of eliminated rows (0.0 will even eliminate diagonal entry)
.  x - optional vector of solutions for zeroed rows (other entries in vector are not used)
-  b - optional vector of right hand side, that will be adjusted by provided solution

   Notes:
   For the AIJ and BAIJ matrix formats this removes the old nonzero structure,
   but does not release memory.  For the dense and block diagonal
   formats this does not alter the nonzero structure.

   If the option MatSetOption(mat,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE) the nonzero structure
   of the matrix is not changed (even for AIJ and BAIJ matrices) the values are
   merely zeroed.

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing 0.0 as the final argument).

   For the parallel case, all processes that share the matrix (i.e.,
   those in the communicator used for matrix creation) MUST call this
   routine, regardless of whether any rows being zeroed are owned by
   them.

   Each processor can indicate any rows in the entire matrix to be zeroed (i.e. each process does NOT have to
   list only rows local to itself).

   The grid coordinates are across the entire grid, not just the local portion

   In Fortran idxm and idxn should be declared as
$     MatStencil idxm(4,m)
   and the values inserted using
$    idxm(MatStencil_i,1) = i
$    idxm(MatStencil_j,1) = j
$    idxm(MatStencil_k,1) = k
$    idxm(MatStencil_c,1) = c
   etc

   For periodic boundary conditions use negative indices for values to the left (below 0; that are to be
   obtained by wrapping values from right edge). For values to the right of the last entry using that index plus one
   etc to obtain values that obtained by wrapping the values from the left edge. This does not work for anything but the
   DM_BOUNDARY_PERIODIC boundary type.

   For indices that don't mean anything for your case (like the k index when working in 2d) or the c index when you have
   a single value per point) you can skip filling those indices.

   Level: intermediate

.seealso: `MatZeroRowsIS()`, `MatZeroRowsColumns()`, `MatZeroRowsLocalIS()`, `MatZeroRowsl()`, `MatZeroEntries()`, `MatZeroRowsLocal()`, `MatSetOption()`,
          `MatZeroRowsColumnsLocal()`, `MatZeroRowsColumnsLocalIS()`, `MatZeroRowsColumnsIS()`, `MatZeroRowsColumnsStencil()`
@*/
PetscErrorCode MatZeroRowsStencil(Mat mat,PetscInt numRows,const MatStencil rows[],PetscScalar diag,Vec x,Vec b)
{
  PetscInt       dim     = mat->stencil.dim;
  PetscInt       sdim    = dim - (1 - (PetscInt) mat->stencil.noc);
  PetscInt       *dims   = mat->stencil.dims+1;
  PetscInt       *starts = mat->stencil.starts;
  PetscInt       *dxm    = (PetscInt*) rows;
  PetscInt       *jdxm, i, j, tmp, numNewRows = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (numRows) PetscValidPointer(rows,3);

  PetscCall(PetscMalloc1(numRows, &jdxm));
  for (i = 0; i < numRows; ++i) {
    /* Skip unused dimensions (they are ordered k, j, i, c) */
    for (j = 0; j < 3-sdim; ++j) dxm++;
    /* Local index in X dir */
    tmp = *dxm++ - starts[0];
    /* Loop over remaining dimensions */
    for (j = 0; j < dim-1; ++j) {
      /* If nonlocal, set index to be negative */
      if ((*dxm++ - starts[j+1]) < 0 || tmp < 0) tmp = PETSC_MIN_INT;
      /* Update local index */
      else tmp = tmp*dims[j] + *(dxm-1) - starts[j+1];
    }
    /* Skip component slot if necessary */
    if (mat->stencil.noc) dxm++;
    /* Local row number */
    if (tmp >= 0) {
      jdxm[numNewRows++] = tmp;
    }
  }
  PetscCall(MatZeroRowsLocal(mat,numNewRows,jdxm,diag,x,b));
  PetscCall(PetscFree(jdxm));
  PetscFunctionReturn(0);
}

/*@
   MatZeroRowsColumnsStencil - Zeros all row and column entries (except possibly the main diagonal)
   of a set of rows and columns of a matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  numRows - the number of rows/columns to remove
.  rows - the grid coordinates (and component number when dof > 1) for matrix rows
.  diag - value put in all diagonals of eliminated rows (0.0 will even eliminate diagonal entry)
.  x - optional vector of solutions for zeroed rows (other entries in vector are not used)
-  b - optional vector of right hand side, that will be adjusted by provided solution

   Notes:
   For the AIJ and BAIJ matrix formats this removes the old nonzero structure,
   but does not release memory.  For the dense and block diagonal
   formats this does not alter the nonzero structure.

   If the option MatSetOption(mat,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE) the nonzero structure
   of the matrix is not changed (even for AIJ and BAIJ matrices) the values are
   merely zeroed.

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing 0.0 as the final argument).

   For the parallel case, all processes that share the matrix (i.e.,
   those in the communicator used for matrix creation) MUST call this
   routine, regardless of whether any rows being zeroed are owned by
   them.

   Each processor can indicate any rows in the entire matrix to be zeroed (i.e. each process does NOT have to
   list only rows local to itself, but the row/column numbers are given in local numbering).

   The grid coordinates are across the entire grid, not just the local portion

   In Fortran idxm and idxn should be declared as
$     MatStencil idxm(4,m)
   and the values inserted using
$    idxm(MatStencil_i,1) = i
$    idxm(MatStencil_j,1) = j
$    idxm(MatStencil_k,1) = k
$    idxm(MatStencil_c,1) = c
   etc

   For periodic boundary conditions use negative indices for values to the left (below 0; that are to be
   obtained by wrapping values from right edge). For values to the right of the last entry using that index plus one
   etc to obtain values that obtained by wrapping the values from the left edge. This does not work for anything but the
   DM_BOUNDARY_PERIODIC boundary type.

   For indices that don't mean anything for your case (like the k index when working in 2d) or the c index when you have
   a single value per point) you can skip filling those indices.

   Level: intermediate

.seealso: `MatZeroRowsIS()`, `MatZeroRowsColumns()`, `MatZeroRowsLocalIS()`, `MatZeroRowsStencil()`, `MatZeroEntries()`, `MatZeroRowsLocal()`, `MatSetOption()`,
          `MatZeroRowsColumnsLocal()`, `MatZeroRowsColumnsLocalIS()`, `MatZeroRowsColumnsIS()`, `MatZeroRows()`
@*/
PetscErrorCode MatZeroRowsColumnsStencil(Mat mat,PetscInt numRows,const MatStencil rows[],PetscScalar diag,Vec x,Vec b)
{
  PetscInt       dim     = mat->stencil.dim;
  PetscInt       sdim    = dim - (1 - (PetscInt) mat->stencil.noc);
  PetscInt       *dims   = mat->stencil.dims+1;
  PetscInt       *starts = mat->stencil.starts;
  PetscInt       *dxm    = (PetscInt*) rows;
  PetscInt       *jdxm, i, j, tmp, numNewRows = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (numRows) PetscValidPointer(rows,3);

  PetscCall(PetscMalloc1(numRows, &jdxm));
  for (i = 0; i < numRows; ++i) {
    /* Skip unused dimensions (they are ordered k, j, i, c) */
    for (j = 0; j < 3-sdim; ++j) dxm++;
    /* Local index in X dir */
    tmp = *dxm++ - starts[0];
    /* Loop over remaining dimensions */
    for (j = 0; j < dim-1; ++j) {
      /* If nonlocal, set index to be negative */
      if ((*dxm++ - starts[j+1]) < 0 || tmp < 0) tmp = PETSC_MIN_INT;
      /* Update local index */
      else tmp = tmp*dims[j] + *(dxm-1) - starts[j+1];
    }
    /* Skip component slot if necessary */
    if (mat->stencil.noc) dxm++;
    /* Local row number */
    if (tmp >= 0) {
      jdxm[numNewRows++] = tmp;
    }
  }
  PetscCall(MatZeroRowsColumnsLocal(mat,numNewRows,jdxm,diag,x,b));
  PetscCall(PetscFree(jdxm));
  PetscFunctionReturn(0);
}

/*@C
   MatZeroRowsLocal - Zeros all entries (except possibly the main diagonal)
   of a set of rows of a matrix; using local numbering of rows.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  numRows - the number of rows to remove
.  rows - the local row indices
.  diag - value put in all diagonals of eliminated rows
.  x - optional vector of solutions for zeroed rows (other entries in vector are not used)
-  b - optional vector of right hand side, that will be adjusted by provided solution

   Notes:
   Before calling MatZeroRowsLocal(), the user must first set the
   local-to-global mapping by calling MatSetLocalToGlobalMapping().

   For the AIJ matrix formats this removes the old nonzero structure,
   but does not release memory.  For the dense and block diagonal
   formats this does not alter the nonzero structure.

   If the option MatSetOption(mat,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE) the nonzero structure
   of the matrix is not changed (even for AIJ and BAIJ matrices) the values are
   merely zeroed.

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing 0.0 as the final argument).

   You can call MatSetOption(mat,MAT_NO_OFF_PROC_ZERO_ROWS,PETSC_TRUE) if each process indicates only rows it
   owns that are to be zeroed. This saves a global synchronization in the implementation.

   Level: intermediate

.seealso: `MatZeroRowsIS()`, `MatZeroRowsColumns()`, `MatZeroRowsLocalIS()`, `MatZeroRowsStencil()`, `MatZeroEntries()`, `MatZeroRows()`, `MatSetOption()`,
          `MatZeroRowsColumnsLocal()`, `MatZeroRowsColumnsLocalIS()`, `MatZeroRowsColumnsIS()`, `MatZeroRowsColumnsStencil()`
@*/
PetscErrorCode MatZeroRowsLocal(Mat mat,PetscInt numRows,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (numRows) PetscValidIntPointer(rows,3);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);

  if (mat->ops->zerorowslocal) {
    PetscCall((*mat->ops->zerorowslocal)(mat,numRows,rows,diag,x,b));
  } else {
    IS             is, newis;
    const PetscInt *newRows;

    PetscCheck(mat->rmap->mapping,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Need to provide local to global mapping to matrix first");
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,numRows,rows,PETSC_COPY_VALUES,&is));
    PetscCall(ISLocalToGlobalMappingApplyIS(mat->rmap->mapping,is,&newis));
    PetscCall(ISGetIndices(newis,&newRows));
    PetscCall((*mat->ops->zerorows)(mat,numRows,newRows,diag,x,b));
    PetscCall(ISRestoreIndices(newis,&newRows));
    PetscCall(ISDestroy(&newis));
    PetscCall(ISDestroy(&is));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
  PetscFunctionReturn(0);
}

/*@
   MatZeroRowsLocalIS - Zeros all entries (except possibly the main diagonal)
   of a set of rows of a matrix; using local numbering of rows.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  is - index set of rows to remove
.  diag - value put in all diagonals of eliminated rows
.  x - optional vector of solutions for zeroed rows (other entries in vector are not used)
-  b - optional vector of right hand side, that will be adjusted by provided solution

   Notes:
   Before calling MatZeroRowsLocalIS(), the user must first set the
   local-to-global mapping by calling MatSetLocalToGlobalMapping().

   For the AIJ matrix formats this removes the old nonzero structure,
   but does not release memory.  For the dense and block diagonal
   formats this does not alter the nonzero structure.

   If the option MatSetOption(mat,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE) the nonzero structure
   of the matrix is not changed (even for AIJ and BAIJ matrices) the values are
   merely zeroed.

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing 0.0 as the final argument).

   You can call MatSetOption(mat,MAT_NO_OFF_PROC_ZERO_ROWS,PETSC_TRUE) if each process indicates only rows it
   owns that are to be zeroed. This saves a global synchronization in the implementation.

   Level: intermediate

.seealso: `MatZeroRowsIS()`, `MatZeroRowsColumns()`, `MatZeroRows()`, `MatZeroRowsStencil()`, `MatZeroEntries()`, `MatZeroRowsLocal()`, `MatSetOption()`,
          `MatZeroRowsColumnsLocal()`, `MatZeroRowsColumnsLocalIS()`, `MatZeroRowsColumnsIS()`, `MatZeroRowsColumnsStencil()`
@*/
PetscErrorCode MatZeroRowsLocalIS(Mat mat,IS is,PetscScalar diag,Vec x,Vec b)
{
  PetscInt       numRows;
  const PetscInt *rows;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);

  PetscCall(ISGetLocalSize(is,&numRows));
  PetscCall(ISGetIndices(is,&rows));
  PetscCall(MatZeroRowsLocal(mat,numRows,rows,diag,x,b));
  PetscCall(ISRestoreIndices(is,&rows));
  PetscFunctionReturn(0);
}

/*@
   MatZeroRowsColumnsLocal - Zeros all entries (except possibly the main diagonal)
   of a set of rows and columns of a matrix; using local numbering of rows.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  numRows - the number of rows to remove
.  rows - the global row indices
.  diag - value put in all diagonals of eliminated rows
.  x - optional vector of solutions for zeroed rows (other entries in vector are not used)
-  b - optional vector of right hand side, that will be adjusted by provided solution

   Notes:
   Before calling MatZeroRowsColumnsLocal(), the user must first set the
   local-to-global mapping by calling MatSetLocalToGlobalMapping().

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing 0.0 as the final argument).

   Level: intermediate

.seealso: `MatZeroRowsIS()`, `MatZeroRowsColumns()`, `MatZeroRowsLocalIS()`, `MatZeroRowsStencil()`, `MatZeroEntries()`, `MatZeroRowsLocal()`, `MatSetOption()`,
          `MatZeroRows()`, `MatZeroRowsColumnsLocalIS()`, `MatZeroRowsColumnsIS()`, `MatZeroRowsColumnsStencil()`
@*/
PetscErrorCode MatZeroRowsColumnsLocal(Mat mat,PetscInt numRows,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  IS             is, newis;
  const PetscInt *newRows;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (numRows) PetscValidIntPointer(rows,3);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);

  PetscCheck(mat->cmap->mapping,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Need to provide local to global mapping to matrix first");
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,numRows,rows,PETSC_COPY_VALUES,&is));
  PetscCall(ISLocalToGlobalMappingApplyIS(mat->cmap->mapping,is,&newis));
  PetscCall(ISGetIndices(newis,&newRows));
  PetscCall((*mat->ops->zerorowscolumns)(mat,numRows,newRows,diag,x,b));
  PetscCall(ISRestoreIndices(newis,&newRows));
  PetscCall(ISDestroy(&newis));
  PetscCall(ISDestroy(&is));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
  PetscFunctionReturn(0);
}

/*@
   MatZeroRowsColumnsLocalIS - Zeros all entries (except possibly the main diagonal)
   of a set of rows and columns of a matrix; using local numbering of rows.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  is - index set of rows to remove
.  diag - value put in all diagonals of eliminated rows
.  x - optional vector of solutions for zeroed rows (other entries in vector are not used)
-  b - optional vector of right hand side, that will be adjusted by provided solution

   Notes:
   Before calling MatZeroRowsColumnsLocalIS(), the user must first set the
   local-to-global mapping by calling MatSetLocalToGlobalMapping().

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing 0.0 as the final argument).

   Level: intermediate

.seealso: `MatZeroRowsIS()`, `MatZeroRowsColumns()`, `MatZeroRowsLocalIS()`, `MatZeroRowsStencil()`, `MatZeroEntries()`, `MatZeroRowsLocal()`, `MatSetOption()`,
          `MatZeroRowsColumnsLocal()`, `MatZeroRows()`, `MatZeroRowsColumnsIS()`, `MatZeroRowsColumnsStencil()`
@*/
PetscErrorCode MatZeroRowsColumnsLocalIS(Mat mat,IS is,PetscScalar diag,Vec x,Vec b)
{
  PetscInt       numRows;
  const PetscInt *rows;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);

  PetscCall(ISGetLocalSize(is,&numRows));
  PetscCall(ISGetIndices(is,&rows));
  PetscCall(MatZeroRowsColumnsLocal(mat,numRows,rows,diag,x,b));
  PetscCall(ISRestoreIndices(is,&rows));
  PetscFunctionReturn(0);
}

/*@C
   MatGetSize - Returns the numbers of rows and columns in a matrix.

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  m - the number of global rows
-  n - the number of global columns

   Note: both output parameters can be NULL on input.

   Level: beginner

.seealso: `MatGetLocalSize()`
@*/
PetscErrorCode MatGetSize(Mat mat,PetscInt *m,PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (m) *m = mat->rmap->N;
  if (n) *n = mat->cmap->N;
  PetscFunctionReturn(0);
}

/*@C
   MatGetLocalSize - For most matrix formats, excluding `MATELEMENTAL` and `MATSCALAPACK`, Returns the number of local rows and local columns
   of a matrix. For all matrices this is the local size of the left and right vectors as returned by MatCreateVecs().

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  m - the number of local rows, use `NULL` to not obtain this value
-  n - the number of local columns, use `NULL` to not obtain this value

   Level: beginner

.seealso: `MatGetSize()`
@*/
PetscErrorCode MatGetLocalSize(Mat mat,PetscInt *m,PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (m) PetscValidIntPointer(m,2);
  if (n) PetscValidIntPointer(n,3);
  if (m) *m = mat->rmap->n;
  if (n) *n = mat->cmap->n;
  PetscFunctionReturn(0);
}

/*@C
   MatGetOwnershipRangeColumn - Returns the range of matrix columns associated with rows of a vector one multiplies this matrix by that are owned by
   this processor. (The columns of the "diagonal block" for most sparse matrix formats). See :any:`<sec_matlayout>` for details on matrix layouts.

   Not Collective, unless matrix has not been allocated, then collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  m - the global index of the first local column, use `NULL` to not obtain this value
-  n - one more than the global index of the last local column, use `NULL` to not obtain this value

   Level: developer

.seealso: `MatGetOwnershipRange()`, `MatGetOwnershipRanges()`, `MatGetOwnershipRangesColumn()`, `PetscLayout`

@*/
PetscErrorCode MatGetOwnershipRangeColumn(Mat mat,PetscInt *m,PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (m) PetscValidIntPointer(m,2);
  if (n) PetscValidIntPointer(n,3);
  MatCheckPreallocated(mat,1);
  if (m) *m = mat->cmap->rstart;
  if (n) *n = mat->cmap->rend;
  PetscFunctionReturn(0);
}

/*@C
   MatGetOwnershipRange - For matrices that own values by row, excludes `MATELEMENTAL` and `MATSCALAPACK`, returns the range of matrix rows owned by
   this MPI rank. For all matrices  it returns the range of matrix rows associated with rows of a vector that would contain the result of a matrix
   vector product with this matrix. See :any:`<sec_matlayout>` for details on matrix layouts

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  m - the global index of the first local row, use `NULL` to not obtain this value
-  n - one more than the global index of the last local row, use `NULL` to not obtain this value

   Note:
  This function requires that the matrix be preallocated. If you have not preallocated, consider using
  `PetscSplitOwnership`(`MPI_Comm` comm, `PetscInt` *n, `PetscInt` *N)
  and then `MPI_Scan()` to calculate prefix sums of the local sizes.

   Level: beginner

.seealso: `MatGetOwnershipRanges()`, `MatGetOwnershipRangeColumn()`, `MatGetOwnershipRangesColumn()`, `PetscSplitOwnership()`, `PetscSplitOwnershipBlock()`,
          `PetscLayout`

@*/
PetscErrorCode MatGetOwnershipRange(Mat mat,PetscInt *m,PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (m) PetscValidIntPointer(m,2);
  if (n) PetscValidIntPointer(n,3);
  MatCheckPreallocated(mat,1);
  if (m) *m = mat->rmap->rstart;
  if (n) *n = mat->rmap->rend;
  PetscFunctionReturn(0);
}

/*@C
   MatGetOwnershipRanges - For matrices that own values by row, excludes `MATELEMENTAL` and `MATSCALAPACK`, returns the range of matrix rows owned by
   each process. For all matrices  it returns the ranges of matrix rows associated with rows of a vector that would contain the result of a matrix
   vector product with this matrix. See :any:`<sec_matlayout>` for details on matrix layouts

   Not Collective, unless matrix has not been allocated, then collective on Mat

   Input Parameters:
.  mat - the matrix

   Output Parameters:
.  ranges - start of each processors portion plus one more than the total length at the end

   Level: beginner

.seealso: `MatGetOwnershipRange()`, `MatGetOwnershipRangeColumn()`, `MatGetOwnershipRangesColumn()`, `PetscLayout`

@*/
PetscErrorCode MatGetOwnershipRanges(Mat mat,const PetscInt **ranges)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  MatCheckPreallocated(mat,1);
  PetscCall(PetscLayoutGetRanges(mat->rmap,ranges));
  PetscFunctionReturn(0);
}

/*@C
   MatGetOwnershipRangesColumn - Returns the ranges of matrix columns associated with rows of a vector one multiplies this vector by that are owned by
   each processor. (The columns of the "diagonal blocks", for most sparse matrix formats). See :any:`<sec_matlayout>` for details on matrix layouts.

   Not Collective, unless matrix has not been allocated, then collective on Mat

   Input Parameters:
.  mat - the matrix

   Output Parameters:
.  ranges - start of each processors portion plus one more then the total length at the end

   Level: beginner

.seealso: `MatGetOwnershipRange()`, `MatGetOwnershipRangeColumn()`, `MatGetOwnershipRanges()`

@*/
PetscErrorCode MatGetOwnershipRangesColumn(Mat mat,const PetscInt **ranges)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  MatCheckPreallocated(mat,1);
  PetscCall(PetscLayoutGetRanges(mat->cmap,ranges));
  PetscFunctionReturn(0);
}

/*@C
   MatGetOwnershipIS - Get row and column ownership of a matrices' values as index sets. For most matrices, excluding `MATELEMENTAL` and `MATSCALAPACK`, this
   corresponds to values returned by `MatGetOwnershipRange()`, `MatGetOwnershipRangeColumn()`. For `MATELEMENTAL` and `MATSCALAPACK` the ownership
   is more complicated. See :any:`<sec_matlayout>` for details on matrix layouts.

   Not Collective

   Input Parameter:
.  A - matrix

   Output Parameters:
+  rows - rows in which this process owns elements, , use `NULL` to not obtain this value
-  cols - columns in which this process owns elements, use `NULL` to not obtain this value

   Level: intermediate

.seealso: `MatGetOwnershipRange()`, `MatGetOwnershipRangeColumn()`, `MatSetValues()`, ``MATELEMENTAL``, ``MATSCALAPACK``
@*/
PetscErrorCode MatGetOwnershipIS(Mat A,IS *rows,IS *cols)
{
  PetscErrorCode (*f)(Mat,IS*,IS*);

  PetscFunctionBegin;
  MatCheckPreallocated(A,1);
  PetscCall(PetscObjectQueryFunction((PetscObject)A,"MatGetOwnershipIS_C",&f));
  if (f) {
    PetscCall((*f)(A,rows,cols));
  } else {   /* Create a standard row-based partition, each process is responsible for ALL columns in their row block */
    if (rows) PetscCall(ISCreateStride(PETSC_COMM_SELF,A->rmap->n,A->rmap->rstart,1,rows));
    if (cols) PetscCall(ISCreateStride(PETSC_COMM_SELF,A->cmap->N,0,1,cols));
  }
  PetscFunctionReturn(0);
}

/*@C
   MatILUFactorSymbolic - Performs symbolic ILU factorization of a matrix.
   Uses levels of fill only, not drop tolerance. Use MatLUFactorNumeric()
   to complete the factorization.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  row - row permutation
.  column - column permutation
-  info - structure containing
$      levels - number of levels of fill.
$      expected fill - as ratio of original fill.
$      1 or 0 - indicating force fill on diagonal (improves robustness for matrices
                missing diagonal entries)

   Output Parameters:
.  fact - new matrix that has been symbolically factored

   Notes:
    See Users-Manual: ch_mat for additional information about choosing the fill factor for better efficiency.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatLUFactorSymbolic()`, `MatLUFactorNumeric()`, `MatCholeskyFactor()`
          `MatGetOrdering()`, `MatFactorInfo`

    Note: this uses the definition of level of fill as in Y. Saad, 2003

    Developer Note: fortran interface is not autogenerated as the f90
    interface definition cannot be generated correctly [due to MatFactorInfo]

   References:
.  * - Y. Saad, Iterative methods for sparse linear systems Philadelphia: Society for Industrial and Applied Mathematics, 2003
@*/
PetscErrorCode MatILUFactorSymbolic(Mat fact,Mat mat,IS row,IS col,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  PetscValidType(mat,2);
  if (row) PetscValidHeaderSpecific(row,IS_CLASSID,3);
  if (col) PetscValidHeaderSpecific(col,IS_CLASSID,4);
  PetscValidPointer(info,5);
  PetscValidPointer(fact,1);
  PetscCheck(info->levels >= 0,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_OUTOFRANGE,"Levels of fill negative %" PetscInt_FMT,(PetscInt)info->levels);
  PetscCheck(info->fill >= 1.0,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_OUTOFRANGE,"Expected fill less than 1.0 %g",(double)info->fill);
  if (!fact->ops->ilufactorsymbolic) {
    MatSolverType stype;
    PetscCall(MatFactorGetSolverType(fact,&stype));
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Matrix type %s symbolic ILU using solver type %s",((PetscObject)mat)->type_name,stype);
  }
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,2);

  if (!fact->trivialsymbolic) PetscCall(PetscLogEventBegin(MAT_ILUFactorSymbolic,mat,row,col,0));
  PetscCall((fact->ops->ilufactorsymbolic)(fact,mat,row,col,info));
  if (!fact->trivialsymbolic) PetscCall(PetscLogEventEnd(MAT_ILUFactorSymbolic,mat,row,col,0));
  PetscFunctionReturn(0);
}

/*@C
   MatICCFactorSymbolic - Performs symbolic incomplete
   Cholesky factorization for a symmetric matrix.  Use
   MatCholeskyFactorNumeric() to complete the factorization.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  perm - row and column permutation
-  info - structure containing
$      levels - number of levels of fill.
$      expected fill - as ratio of original fill.

   Output Parameter:
.  fact - the factored matrix

   Notes:
   Most users should employ the KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatCholeskyFactorNumeric()`, `MatCholeskyFactor()`, `MatFactorInfo`

    Note: this uses the definition of level of fill as in Y. Saad, 2003

    Developer Note: fortran interface is not autogenerated as the f90
    interface definition cannot be generated correctly [due to MatFactorInfo]

   References:
.  * - Y. Saad, Iterative methods for sparse linear systems Philadelphia: Society for Industrial and Applied Mathematics, 2003
@*/
PetscErrorCode MatICCFactorSymbolic(Mat fact,Mat mat,IS perm,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  PetscValidType(mat,2);
  if (perm) PetscValidHeaderSpecific(perm,IS_CLASSID,3);
  PetscValidPointer(info,4);
  PetscValidPointer(fact,1);
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(info->levels >= 0,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_OUTOFRANGE,"Levels negative %" PetscInt_FMT,(PetscInt) info->levels);
  PetscCheck(info->fill >= 1.0,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_OUTOFRANGE,"Expected fill less than 1.0 %g",(double)info->fill);
  if (!(fact)->ops->iccfactorsymbolic) {
    MatSolverType stype;
    PetscCall(MatFactorGetSolverType(fact,&stype));
    SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Matrix type %s symbolic ICC using solver type %s",((PetscObject)mat)->type_name,stype);
  }
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  MatCheckPreallocated(mat,2);

  if (!fact->trivialsymbolic) PetscCall(PetscLogEventBegin(MAT_ICCFactorSymbolic,mat,perm,0,0));
  PetscCall((fact->ops->iccfactorsymbolic)(fact,mat,perm,info));
  if (!fact->trivialsymbolic) PetscCall(PetscLogEventEnd(MAT_ICCFactorSymbolic,mat,perm,0,0));
  PetscFunctionReturn(0);
}

/*@C
   MatCreateSubMatrices - Extracts several submatrices from a matrix. If submat
   points to an array of valid matrices, they may be reused to store the new
   submatrices.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  n   - the number of submatrixes to be extracted (on this processor, may be zero)
.  irow, icol - index sets of rows and columns to extract
-  scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

   Output Parameter:
.  submat - the array of submatrices

   Notes:
   MatCreateSubMatrices() can extract ONLY sequential submatrices
   (from both sequential and parallel matrices). Use MatCreateSubMatrix()
   to extract a parallel submatrix.

   Some matrix types place restrictions on the row and column
   indices, such as that they be sorted or that they be equal to each other.

   The index sets may not have duplicate entries.

   When extracting submatrices from a parallel matrix, each processor can
   form a different submatrix by setting the rows and columns of its
   individual index sets according to the local submatrix desired.

   When finished using the submatrices, the user should destroy
   them with MatDestroySubMatrices().

   MAT_REUSE_MATRIX can only be used when the nonzero structure of the
   original matrix has not changed from that last call to MatCreateSubMatrices().

   This routine creates the matrices in submat; you should NOT create them before
   calling it. It also allocates the array of matrix pointers submat.

   For BAIJ matrices the index sets must respect the block structure, that is if they
   request one row/column in a block, they must request all rows/columns that are in
   that block. For example, if the block size is 2 you cannot request just row 0 and
   column 0.

   Fortran Note:
   The Fortran interface is slightly different from that given below; it
   requires one to pass in  as submat a Mat (integer) array of size at least n+1.

   Level: advanced

.seealso: `MatDestroySubMatrices()`, `MatCreateSubMatrix()`, `MatGetRow()`, `MatGetDiagonal()`, `MatReuse`
@*/
PetscErrorCode MatCreateSubMatrices(Mat mat,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *submat[])
{
  PetscInt       i;
  PetscBool      eq;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (n) {
    PetscValidPointer(irow,3);
    for (i=0; i<n; i++) PetscValidHeaderSpecific(irow[i],IS_CLASSID,3);
    PetscValidPointer(icol,4);
    for (i=0; i<n; i++) PetscValidHeaderSpecific(icol[i],IS_CLASSID,4);
  }
  PetscValidPointer(submat,6);
  if (n && scall == MAT_REUSE_MATRIX) {
    PetscValidPointer(*submat,6);
    for (i=0; i<n; i++) PetscValidHeaderSpecific((*submat)[i],MAT_CLASSID,6);
  }
  PetscCheck(mat->ops->createsubmatrices,PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);
  PetscCall(PetscLogEventBegin(MAT_CreateSubMats,mat,0,0,0));
  PetscCall((*mat->ops->createsubmatrices)(mat,n,irow,icol,scall,submat));
  PetscCall(PetscLogEventEnd(MAT_CreateSubMats,mat,0,0,0));
  for (i=0; i<n; i++) {
    (*submat)[i]->factortype = MAT_FACTOR_NONE;  /* in case in place factorization was previously done on submatrix */
    PetscCall(ISEqualUnsorted(irow[i],icol[i],&eq));
    if (eq) {
      PetscCall(MatPropagateSymmetryOptions(mat,(*submat)[i]));
    }
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    if (mat->boundtocpu && mat->bindingpropagates) {
      PetscCall(MatBindToCPU((*submat)[i],PETSC_TRUE));
      PetscCall(MatSetBindingPropagates((*submat)[i],PETSC_TRUE));
    }
#endif
  }
  PetscFunctionReturn(0);
}

/*@C
   MatCreateSubMatricesMPI - Extracts MPI submatrices across a sub communicator of mat (by pairs of IS that may live on subcomms).

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  n   - the number of submatrixes to be extracted
.  irow, icol - index sets of rows and columns to extract
-  scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

   Output Parameter:
.  submat - the array of submatrices

   Level: advanced

.seealso: `MatCreateSubMatrices()`, `MatCreateSubMatrix()`, `MatGetRow()`, `MatGetDiagonal()`, `MatReuse`
@*/
PetscErrorCode MatCreateSubMatricesMPI(Mat mat,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *submat[])
{
  PetscInt       i;
  PetscBool      eq;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (n) {
    PetscValidPointer(irow,3);
    PetscValidHeaderSpecific(*irow,IS_CLASSID,3);
    PetscValidPointer(icol,4);
    PetscValidHeaderSpecific(*icol,IS_CLASSID,4);
  }
  PetscValidPointer(submat,6);
  if (n && scall == MAT_REUSE_MATRIX) {
    PetscValidPointer(*submat,6);
    PetscValidHeaderSpecific(**submat,MAT_CLASSID,6);
  }
  PetscCheck(mat->ops->createsubmatricesmpi,PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);

  PetscCall(PetscLogEventBegin(MAT_CreateSubMats,mat,0,0,0));
  PetscCall((*mat->ops->createsubmatricesmpi)(mat,n,irow,icol,scall,submat));
  PetscCall(PetscLogEventEnd(MAT_CreateSubMats,mat,0,0,0));
  for (i=0; i<n; i++) {
    PetscCall(ISEqualUnsorted(irow[i],icol[i],&eq));
    if (eq) {
      PetscCall(MatPropagateSymmetryOptions(mat,(*submat)[i]));
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   MatDestroyMatrices - Destroys an array of matrices.

   Collective on Mat

   Input Parameters:
+  n - the number of local matrices
-  mat - the matrices (note that this is a pointer to the array of matrices)

   Level: advanced

    Notes:
    Frees not only the matrices, but also the array that contains the matrices
           In Fortran will not free the array.

.seealso: `MatCreateSubMatrices()` `MatDestroySubMatrices()`
@*/
PetscErrorCode MatDestroyMatrices(PetscInt n,Mat *mat[])
{
  PetscInt       i;

  PetscFunctionBegin;
  if (!*mat) PetscFunctionReturn(0);
  PetscCheck(n >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to destroy negative number of matrices %" PetscInt_FMT,n);
  PetscValidPointer(mat,2);

  for (i=0; i<n; i++) {
    PetscCall(MatDestroy(&(*mat)[i]));
  }

  /* memory is allocated even if n = 0 */
  PetscCall(PetscFree(*mat));
  PetscFunctionReturn(0);
}

/*@C
   MatDestroySubMatrices - Destroys a set of matrices obtained with MatCreateSubMatrices().

   Collective on Mat

   Input Parameters:
+  n - the number of local matrices
-  mat - the matrices (note that this is a pointer to the array of matrices, just to match the calling
                       sequence of MatCreateSubMatrices())

   Level: advanced

    Notes:
    Frees not only the matrices, but also the array that contains the matrices
           In Fortran will not free the array.

.seealso: `MatCreateSubMatrices()`
@*/
PetscErrorCode MatDestroySubMatrices(PetscInt n,Mat *mat[])
{
  Mat            mat0;

  PetscFunctionBegin;
  if (!*mat) PetscFunctionReturn(0);
  /* mat[] is an array of length n+1, see MatCreateSubMatrices_xxx() */
  PetscCheck(n >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to destroy negative number of matrices %" PetscInt_FMT,n);
  PetscValidPointer(mat,2);

  mat0 = (*mat)[0];
  if (mat0 && mat0->ops->destroysubmatrices) {
    PetscCall((mat0->ops->destroysubmatrices)(n,mat));
  } else {
    PetscCall(MatDestroyMatrices(n,mat));
  }
  PetscFunctionReturn(0);
}

/*@C
   MatGetSeqNonzeroStructure - Extracts the nonzero structure from a matrix and stores it, in its entirety, on each process

   Collective on Mat

   Input Parameters:
.  mat - the matrix

   Output Parameter:
.  matstruct - the sequential matrix with the nonzero structure of mat

  Level: intermediate

.seealso: `MatDestroySeqNonzeroStructure()`, `MatCreateSubMatrices()`, `MatDestroyMatrices()`
@*/
PetscErrorCode MatGetSeqNonzeroStructure(Mat mat,Mat *matstruct)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(matstruct,2);

  PetscValidType(mat,1);
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);

  PetscCheck(mat->ops->getseqnonzerostructure,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Not for matrix type %s",((PetscObject)mat)->type_name);
  PetscCall(PetscLogEventBegin(MAT_GetSeqNonzeroStructure,mat,0,0,0));
  PetscCall((*mat->ops->getseqnonzerostructure)(mat,matstruct));
  PetscCall(PetscLogEventEnd(MAT_GetSeqNonzeroStructure,mat,0,0,0));
  PetscFunctionReturn(0);
}

/*@C
   MatDestroySeqNonzeroStructure - Destroys matrix obtained with MatGetSeqNonzeroStructure().

   Collective on Mat

   Input Parameters:
.  mat - the matrix (note that this is a pointer to the array of matrices, just to match the calling
                       sequence of MatGetSequentialNonzeroStructure())

   Level: advanced

    Notes:
    Frees not only the matrices, but also the array that contains the matrices

.seealso: `MatGetSeqNonzeroStructure()`
@*/
PetscErrorCode MatDestroySeqNonzeroStructure(Mat *mat)
{
  PetscFunctionBegin;
  PetscValidPointer(mat,1);
  PetscCall(MatDestroy(mat));
  PetscFunctionReturn(0);
}

/*@
   MatIncreaseOverlap - Given a set of submatrices indicated by index sets,
   replaces the index sets by larger ones that represent submatrices with
   additional overlap.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  n   - the number of index sets
.  is  - the array of index sets (these index sets will changed during the call)
-  ov  - the additional overlap requested

   Options Database:
.  -mat_increase_overlap_scalable - use a scalable algorithm to compute the overlap (supported by MPIAIJ matrix)

   Level: developer

   Developer Note:
   Any implementation must preserve block sizes. That is: if the row block size and the column block size of mat are equal to bs, then the output index sets must be compatible with bs.

.seealso: `MatCreateSubMatrices()`
@*/
PetscErrorCode MatIncreaseOverlap(Mat mat,PetscInt n,IS is[],PetscInt ov)
{
  PetscInt       i,bs,cbs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidLogicalCollectiveInt(mat,n,2);
  PetscCheck(n >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must have one or more domains, you have %" PetscInt_FMT,n);
  if (n) {
    PetscValidPointer(is,3);
    for (i = 0; i < n; i++) PetscValidHeaderSpecific(is[i],IS_CLASSID,3);
  }
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);

  if (!ov || !n) PetscFunctionReturn(0);
  PetscCheck(mat->ops->increaseoverlap,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  PetscCall(PetscLogEventBegin(MAT_IncreaseOverlap,mat,0,0,0));
  PetscCall((*mat->ops->increaseoverlap)(mat,n,is,ov));
  PetscCall(PetscLogEventEnd(MAT_IncreaseOverlap,mat,0,0,0));
  PetscCall(MatGetBlockSizes(mat,&bs,&cbs));
  if (bs == cbs) {
    for (i=0; i<n; i++) {
      PetscCall(ISSetBlockSize(is[i],bs));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatIncreaseOverlapSplit_Single(Mat,IS*,PetscInt);

/*@
   MatIncreaseOverlapSplit - Given a set of submatrices indicated by index sets across
   a sub communicator, replaces the index sets by larger ones that represent submatrices with
   additional overlap.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  n   - the number of index sets
.  is  - the array of index sets (these index sets will changed during the call)
-  ov  - the additional overlap requested

   Options Database:
.  -mat_increase_overlap_scalable - use a scalable algorithm to compute the overlap (supported by MPIAIJ matrix)

   Level: developer

.seealso: `MatCreateSubMatrices()`
@*/
PetscErrorCode MatIncreaseOverlapSplit(Mat mat,PetscInt n,IS is[],PetscInt ov)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscCheck(n >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must have one or more domains, you have %" PetscInt_FMT,n);
  if (n) {
    PetscValidPointer(is,3);
    PetscValidHeaderSpecific(*is,IS_CLASSID,3);
  }
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);
  if (!ov) PetscFunctionReturn(0);
  PetscCall(PetscLogEventBegin(MAT_IncreaseOverlap,mat,0,0,0));
  for (i=0; i<n; i++) {
    PetscCall(MatIncreaseOverlapSplit_Single(mat,&is[i],ov));
  }
  PetscCall(PetscLogEventEnd(MAT_IncreaseOverlap,mat,0,0,0));
  PetscFunctionReturn(0);
}

/*@
   MatGetBlockSize - Returns the matrix block size.

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  bs - block size

   Notes:
    Block row formats are MATSEQBAIJ, MATMPIBAIJ, MATSEQSBAIJ, MATMPISBAIJ. These formats ALWAYS have square block storage in the matrix.

   If the block size has not been set yet this routine returns 1.

   Level: intermediate

.seealso: `MatCreateSeqBAIJ()`, `MatCreateBAIJ()`, `MatGetBlockSizes()`
@*/
PetscErrorCode MatGetBlockSize(Mat mat,PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidIntPointer(bs,2);
  *bs = PetscAbs(mat->rmap->bs);
  PetscFunctionReturn(0);
}

/*@
   MatGetBlockSizes - Returns the matrix block row and column sizes.

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  rbs - row block size
-  cbs - column block size

   Notes:
    Block row formats are MATSEQBAIJ, MATMPIBAIJ, MATSEQSBAIJ, MATMPISBAIJ. These formats ALWAYS have square block storage in the matrix.
    If you pass a different block size for the columns than the rows, the row block size determines the square block storage.

   If a block size has not been set yet this routine returns 1.

   Level: intermediate

.seealso: `MatCreateSeqBAIJ()`, `MatCreateBAIJ()`, `MatGetBlockSize()`, `MatSetBlockSize()`, `MatSetBlockSizes()`
@*/
PetscErrorCode MatGetBlockSizes(Mat mat,PetscInt *rbs, PetscInt *cbs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (rbs) PetscValidIntPointer(rbs,2);
  if (cbs) PetscValidIntPointer(cbs,3);
  if (rbs) *rbs = PetscAbs(mat->rmap->bs);
  if (cbs) *cbs = PetscAbs(mat->cmap->bs);
  PetscFunctionReturn(0);
}

/*@
   MatSetBlockSize - Sets the matrix block size.

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix
-  bs - block size

   Notes:
    Block row formats are MATSEQBAIJ, MATMPIBAIJ, MATSEQSBAIJ, MATMPISBAIJ. These formats ALWAYS have square block storage in the matrix.
    This must be called before MatSetUp() or MatXXXSetPreallocation() (or will default to 1) and the block size cannot be changed later.

    For MATMPIAIJ and MATSEQAIJ matrix formats, this function can be called at a later stage, provided that the specified block size
    is compatible with the matrix local sizes.

   Level: intermediate

.seealso: `MatCreateSeqBAIJ()`, `MatCreateBAIJ()`, `MatGetBlockSize()`, `MatSetBlockSizes()`, `MatGetBlockSizes()`
@*/
PetscErrorCode MatSetBlockSize(Mat mat,PetscInt bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(mat,bs,2);
  PetscCall(MatSetBlockSizes(mat,bs,bs));
  PetscFunctionReturn(0);
}

typedef struct {
  PetscInt         n;
  IS               *is;
  Mat              *mat;
  PetscObjectState nonzerostate;
  Mat              C;
} EnvelopeData;

static PetscErrorCode EnvelopeDataDestroy(EnvelopeData *edata)
{
  for (PetscInt i=0; i<edata->n; i++) {
    PetscCall(ISDestroy(&edata->is[i]));
  }
  PetscCall(PetscFree(edata->is));
  PetscCall(PetscFree(edata));
  return 0;
}

/*
   MatComputeVariableBlockEnvelope - Given a matrix whose nonzeros are in blocks along the diagonal this computes and stores
         the sizes of these blocks in the matrix. An individual block may lie over several processes.

   Collective on mat

   Input Parameter:
.  mat - the matrix

   Notes:
     There can be zeros within the blocks

     The blocks can overlap between processes, including laying on more than two processes

*/
static PetscErrorCode MatComputeVariableBlockEnvelope(Mat mat)
{
  PetscInt                    n,*sizes,*starts,i = 0,env = 0, tbs = 0, lblocks = 0,rstart,II,ln = 0,cnt = 0,cstart,cend;
  PetscInt                    *diag,*odiag,sc;
  VecScatter                  scatter;
  PetscScalar                 *seqv;
  const PetscScalar           *parv;
  const PetscInt              *ia,*ja;
  PetscBool                   set,flag,done;
  Mat                         AA = mat,A;
  MPI_Comm                    comm;
  PetscMPIInt                 rank,size,tag;
  MPI_Status                  status;
  PetscContainer              container;
  EnvelopeData                *edata;
  Vec                         seq,par;
  IS                          isglobal;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscCall(MatIsSymmetricKnown(mat,&set,&flag));
  if (!set || !flag) {
    /* TOO: only needs nonzero structure of transpose */
    PetscCall(MatTranspose(mat,MAT_INITIAL_MATRIX,&AA));
    PetscCall(MatAXPY(AA,1.0,mat,DIFFERENT_NONZERO_PATTERN));
  }
  PetscCall(MatAIJGetLocalMat(AA,&A));
  PetscCall(MatGetRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&n,&ia,&ja,&done));
  PetscCheck(done,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Unable to get IJ structure from matrix");

  PetscCall(MatGetLocalSize(mat,&n,NULL));
  PetscCall(PetscObjectGetNewTag((PetscObject)mat,&tag));
  PetscCall(PetscObjectGetComm((PetscObject)mat,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  PetscCall(PetscMalloc2(n,&sizes,n,&starts));

  if (rank > 0) {
    PetscCallMPI(MPI_Recv(&env,1,MPIU_INT,rank-1,tag,comm,&status));
    PetscCallMPI(MPI_Recv(&tbs,1,MPIU_INT,rank-1,tag,comm,&status));
  }
  PetscCall(MatGetOwnershipRange(mat,&rstart,NULL));
  for (i=0; i<n; i++) {
    env = PetscMax(env,ja[ia[i+1]-1]);
    II = rstart + i;
    if (env == II) {
      starts[lblocks]  = tbs;
      sizes[lblocks++] = 1 + II - tbs;
      tbs = 1 + II;
    }
  }
  if (rank < size-1) {
    PetscCallMPI(MPI_Send(&env,1,MPIU_INT,rank+1,tag,comm));
    PetscCallMPI(MPI_Send(&tbs,1,MPIU_INT,rank+1,tag,comm));
  }

  PetscCall(MatRestoreRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&n,&ia,&ja,&done));
  if (!set || !flag) {
    PetscCall(MatDestroy(&AA));
  }
  PetscCall(MatDestroy(&A));

  PetscCall(PetscNew(&edata));
  PetscCall(MatGetNonzeroState(mat,&edata->nonzerostate));
  edata->n = lblocks;
  /* create IS needed for extracting blocks from the original matrix */
  PetscCall(PetscMalloc1(lblocks,&edata->is));
  for (PetscInt i=0; i<lblocks; i++) {
    PetscCall(ISCreateStride(PETSC_COMM_SELF,sizes[i],starts[i],1,&edata->is[i]));
  }

  /* Create the resulting inverse matrix structure with preallocation information */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)mat),&edata->C));
  PetscCall(MatSetSizes(edata->C,mat->rmap->n,mat->cmap->n,mat->rmap->N,mat->cmap->N));
  PetscCall(MatSetBlockSizesFromMats(edata->C,mat,mat));
  PetscCall(MatSetType(edata->C,MATAIJ));

  /* Communicate the start and end of each row, from each block to the correct rank */
  /* TODO: Use PetscSF instead of VecScatter */
  for (PetscInt i=0; i<lblocks; i++) ln += sizes[i];
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,2*ln,&seq));
  PetscCall(VecGetArrayWrite(seq,&seqv));
  for (PetscInt i=0; i<lblocks; i++) {
    for (PetscInt j=0; j<sizes[i]; j++) {
      seqv[cnt]   = starts[i];
      seqv[cnt+1] = starts[i] + sizes[i];
      cnt += 2;
    }
  }
  PetscCall(VecRestoreArrayWrite(seq,&seqv));
  PetscCallMPI(MPI_Scan(&cnt,&sc,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mat)));
  sc -= cnt;
  PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)mat),2*mat->rmap->n,2*mat->rmap->N,&par));
  PetscCall(ISCreateStride(PETSC_COMM_SELF,cnt,sc,1,&isglobal));
  PetscCall(VecScatterCreate(seq, NULL  ,par, isglobal,&scatter));
  PetscCall(ISDestroy(&isglobal));
  PetscCall(VecScatterBegin(scatter,seq,par,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter,seq,par,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(VecDestroy(&seq));
  PetscCall(MatGetOwnershipRangeColumn(mat,&cstart,&cend));
  PetscCall(PetscMalloc2(mat->rmap->n,&diag,mat->rmap->n,&odiag));
  PetscCall(VecGetArrayRead(par,&parv));
  cnt = 0;
  PetscCall(MatGetSize(mat,NULL,&n));
  for (PetscInt i=0; i<mat->rmap->n; i++) {
    PetscInt start,end,d = 0,od = 0;

    start = (PetscInt)PetscRealPart(parv[cnt]);
    end   = (PetscInt)PetscRealPart(parv[cnt+1]);
    cnt  += 2;

    if (start < cstart) {od += cstart - start + n - cend; d += cend - cstart;}
    else if (start < cend) {od += n - cend; d += cend - start;}
    else od += n - start;
    if (end <= cstart) {od -= cstart - end + n - cend; d -= cend - cstart;}
    else if (end < cend) {od -= n - cend; d -= cend - end;}
    else od -= n - end;

    odiag[i] = od;
    diag[i]  = d;
  }
  PetscCall(VecRestoreArrayRead(par,&parv));
  PetscCall(VecDestroy(&par));
  PetscCall(MatXAIJSetPreallocation(edata->C,mat->rmap->bs,diag,odiag,NULL,NULL));
  PetscCall(PetscFree2(diag,odiag));
  PetscCall(PetscFree2(sizes,starts));

  PetscCall(PetscContainerCreate(PETSC_COMM_SELF,&container));
  PetscCall(PetscContainerSetPointer(container,edata));
  PetscCall(PetscContainerSetUserDestroy(container,(PetscErrorCode (*)(void*))EnvelopeDataDestroy));
  PetscCall(PetscObjectCompose((PetscObject)mat,"EnvelopeData",(PetscObject)container));
  PetscCall(PetscObjectDereference((PetscObject)container));
  PetscFunctionReturn(0);
}

/*@
  MatInvertVariableBlockEnvelope - set matrix C to be the inverted block diagonal of matrix A

  Collective on Mat

  Input Parameters:
. A - the matrix

  Output Parameters:
. C - matrix with inverted block diagonal of A.  This matrix should be created and may have its type set.

  Notes:
     For efficiency the matrix A should have all the nonzero entries clustered in smallish blocks along the diagonal.

  Level: advanced

.seealso: MatInvertBlockDiagonal(), MatComputeBlockDiagonal()
@*/
PetscErrorCode MatInvertVariableBlockEnvelope(Mat A,MatReuse reuse, Mat *C)
{
  PetscContainer    container;
  EnvelopeData      *edata;
  PetscObjectState  nonzerostate;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)A,"EnvelopeData",(PetscObject*)&container));
  if (!container) {
    PetscCall(MatComputeVariableBlockEnvelope(A));
    PetscCall(PetscObjectQuery((PetscObject)A,"EnvelopeData",(PetscObject*)&container));
  }
  PetscCall(PetscContainerGetPointer(container,(void**)&edata));
  PetscCall(MatGetNonzeroState(A,&nonzerostate));
  PetscCheck(nonzerostate <= edata->nonzerostate,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Cannot handle changes to matrix nonzero structure");
  PetscCheck(reuse != MAT_REUSE_MATRIX || *C == edata->C,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"C matrix must be the same as previously output");

  PetscCall(MatCreateSubMatrices(A,edata->n,edata->is,edata->is,MAT_INITIAL_MATRIX,&edata->mat));
  *C   = edata->C;

  for (PetscInt i=0; i<edata->n; i++) {
    Mat         D;
    PetscScalar *dvalues;

    PetscCall(MatConvert(edata->mat[i], MATSEQDENSE,MAT_INITIAL_MATRIX,&D));
    PetscCall(MatSetOption(*C,MAT_ROW_ORIENTED,PETSC_FALSE));
    PetscCall(MatSeqDenseInvert(D));
    PetscCall(MatDenseGetArray(D,&dvalues));
    PetscCall(MatSetValuesIS(*C,edata->is[i],edata->is[i],dvalues,INSERT_VALUES));
    PetscCall(MatDestroy(&D));
  }
  PetscCall(MatDestroySubMatrices(edata->n,&edata->mat));
  PetscCall(MatAssemblyBegin(*C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*C,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*@
   MatSetVariableBlockSizes - Sets diagonal point-blocks of the matrix that need not be of the same size

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix
.  nblocks - the number of blocks on this process, each block can only exist on a single process
-  bsizes - the block sizes

   Notes:
    Currently used by PCVPBJACOBI for AIJ matrices

    Each variable point-block set of degrees of freedom must live on a single MPI rank. That is a point block cannot straddle two MPI ranks.

   Level: intermediate

.seealso: `MatCreateSeqBAIJ()`, `MatCreateBAIJ()`, `MatGetBlockSize()`, `MatSetBlockSizes()`, `MatGetBlockSizes()`, `MatGetVariableBlockSizes()`, `MatComputeVariableBlockEnvelope()`, `PCVPBJACOBI`
@*/
PetscErrorCode MatSetVariableBlockSizes(Mat mat,PetscInt nblocks,PetscInt *bsizes)
{
  PetscInt       i,ncnt = 0, nlocal;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscCheck(nblocks >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number of local blocks must be great than or equal to zero");
  PetscCall(MatGetLocalSize(mat,&nlocal,NULL));
  for (i=0; i<nblocks; i++) ncnt += bsizes[i];
  PetscCheck(ncnt == nlocal,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Sum of local block sizes %" PetscInt_FMT " does not equal local size of matrix %" PetscInt_FMT,ncnt,nlocal);
  PetscCall(PetscFree(mat->bsizes));
  mat->nblocks = nblocks;
  PetscCall(PetscMalloc1(nblocks,&mat->bsizes));
  PetscCall(PetscArraycpy(mat->bsizes,bsizes,nblocks));
  PetscFunctionReturn(0);
}

/*@C
   MatGetVariableBlockSizes - Gets a diagonal blocks of the matrix that need not be of the same size

   Logically Collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  nblocks - the number of blocks on this process
-  bsizes - the block sizes

   Notes: Currently not supported from Fortran

   Level: intermediate

.seealso: `MatCreateSeqBAIJ()`, `MatCreateBAIJ()`, `MatGetBlockSize()`, `MatSetBlockSizes()`, `MatGetBlockSizes()`, `MatSetVariableBlockSizes()`, `MatComputeVariableBlockEnvelope()`
@*/
PetscErrorCode MatGetVariableBlockSizes(Mat mat,PetscInt *nblocks,const PetscInt **bsizes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  *nblocks = mat->nblocks;
  *bsizes  = mat->bsizes;
  PetscFunctionReturn(0);
}

/*@
   MatSetBlockSizes - Sets the matrix block row and column sizes.

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix
.  rbs - row block size
-  cbs - column block size

   Notes:
    Block row formats are MATSEQBAIJ, MATMPIBAIJ, MATSEQSBAIJ, MATMPISBAIJ. These formats ALWAYS have square block storage in the matrix.
    If you pass a different block size for the columns than the rows, the row block size determines the square block storage.
    This must be called before MatSetUp() or MatXXXSetPreallocation() (or will default to 1) and the block size cannot be changed later.

    For MATMPIAIJ and MATSEQAIJ matrix formats, this function can be called at a later stage, provided that the specified block sizes
    are compatible with the matrix local sizes.

    The row and column block size determine the blocksize of the "row" and "column" vectors returned by MatCreateVecs().

   Level: intermediate

.seealso: `MatCreateSeqBAIJ()`, `MatCreateBAIJ()`, `MatGetBlockSize()`, `MatSetBlockSize()`, `MatGetBlockSizes()`
@*/
PetscErrorCode MatSetBlockSizes(Mat mat,PetscInt rbs,PetscInt cbs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(mat,rbs,2);
  PetscValidLogicalCollectiveInt(mat,cbs,3);
  if (mat->ops->setblocksizes) PetscCall((*mat->ops->setblocksizes)(mat,rbs,cbs));
  if (mat->rmap->refcnt) {
    ISLocalToGlobalMapping l2g = NULL;
    PetscLayout            nmap = NULL;

    PetscCall(PetscLayoutDuplicate(mat->rmap,&nmap));
    if (mat->rmap->mapping) {
      PetscCall(ISLocalToGlobalMappingDuplicate(mat->rmap->mapping,&l2g));
    }
    PetscCall(PetscLayoutDestroy(&mat->rmap));
    mat->rmap = nmap;
    mat->rmap->mapping = l2g;
  }
  if (mat->cmap->refcnt) {
    ISLocalToGlobalMapping l2g = NULL;
    PetscLayout            nmap = NULL;

    PetscCall(PetscLayoutDuplicate(mat->cmap,&nmap));
    if (mat->cmap->mapping) {
      PetscCall(ISLocalToGlobalMappingDuplicate(mat->cmap->mapping,&l2g));
    }
    PetscCall(PetscLayoutDestroy(&mat->cmap));
    mat->cmap = nmap;
    mat->cmap->mapping = l2g;
  }
  PetscCall(PetscLayoutSetBlockSize(mat->rmap,rbs));
  PetscCall(PetscLayoutSetBlockSize(mat->cmap,cbs));
  PetscFunctionReturn(0);
}

/*@
   MatSetBlockSizesFromMats - Sets the matrix block row and column sizes to match a pair of matrices

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix
.  fromRow - matrix from which to copy row block size
-  fromCol - matrix from which to copy column block size (can be same as fromRow)

   Level: developer

.seealso: `MatCreateSeqBAIJ()`, `MatCreateBAIJ()`, `MatGetBlockSize()`, `MatSetBlockSizes()`
@*/
PetscErrorCode MatSetBlockSizesFromMats(Mat mat,Mat fromRow,Mat fromCol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(fromRow,MAT_CLASSID,2);
  PetscValidHeaderSpecific(fromCol,MAT_CLASSID,3);
  if (fromRow->rmap->bs > 0) PetscCall(PetscLayoutSetBlockSize(mat->rmap,fromRow->rmap->bs));
  if (fromCol->cmap->bs > 0) PetscCall(PetscLayoutSetBlockSize(mat->cmap,fromCol->cmap->bs));
  PetscFunctionReturn(0);
}

/*@
   MatResidual - Default routine to calculate the residual.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  b   - the right-hand-side
-  x   - the approximate solution

   Output Parameter:
.  r - location to store the residual

   Level: developer

.seealso: `PCMGSetResidual()`
@*/
PetscErrorCode MatResidual(Mat mat,Vec b,Vec x,Vec r)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  PetscValidHeaderSpecific(r,VEC_CLASSID,4);
  PetscValidType(mat,1);
  MatCheckPreallocated(mat,1);
  PetscCall(PetscLogEventBegin(MAT_Residual,mat,0,0,0));
  if (!mat->ops->residual) {
    PetscCall(MatMult(mat,x,r));
    PetscCall(VecAYPX(r,-1.0,b));
  } else {
    PetscCall((*mat->ops->residual)(mat,b,x,r));
  }
  PetscCall(PetscLogEventEnd(MAT_Residual,mat,0,0,0));
  PetscFunctionReturn(0);
}

/*@C
    MatGetRowIJ - Returns the compressed row storage i and j indices for the local rows of a sparse matrix

   Collective on Mat

    Input Parameters:
+   mat - the matrix
.   shift -  0 or 1 indicating we want the indices starting at 0 or 1
.   symmetric - PETSC_TRUE or PETSC_FALSE indicating the matrix data structure should be   symmetrized
-   inodecompressed - PETSC_TRUE or PETSC_FALSE  indicating if the nonzero structure of the
                 inodes or the nonzero elements is wanted. For BAIJ matrices the compressed version is
                 always used.

    Output Parameters:
+   n - number of local rows in the (possibly compressed) matrix
.   ia - the row pointers; that is ia[0] = 0, ia[row] = ia[row-1] + number of elements in that row of the matrix
.   ja - the column indices
-   done - indicates if the routine actually worked and returned appropriate ia[] and ja[] arrays; callers
           are responsible for handling the case when done == PETSC_FALSE and ia and ja are not set

    Level: developer

    Notes:
    You CANNOT change any of the ia[] or ja[] values.

    Use MatRestoreRowIJ() when you are finished accessing the ia[] and ja[] values.

    Fortran Notes:
    In Fortran use
$
$      PetscInt ia(1), ja(1)
$      PetscOffset iia, jja
$      call MatGetRowIJ(mat,shift,symmetric,inodecompressed,n,ia,iia,ja,jja,done,ierr)
$      ! Access the ith and jth entries via ia(iia + i) and ja(jja + j)

     or
$
$    PetscInt, pointer :: ia(:),ja(:)
$    call MatGetRowIJF90(mat,shift,symmetric,inodecompressed,n,ia,ja,done,ierr)
$    ! Access the ith and jth entries via ia(i) and ja(j)

.seealso: `MatGetColumnIJ()`, `MatRestoreRowIJ()`, `MatSeqAIJGetArray()`
@*/
PetscErrorCode MatGetRowIJ(Mat mat,PetscInt shift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscBool  *done)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (n) PetscValidIntPointer(n,5);
  if (ia) PetscValidPointer(ia,6);
  if (ja) PetscValidPointer(ja,7);
  if (done) PetscValidBoolPointer(done,8);
  MatCheckPreallocated(mat,1);
  if (!mat->ops->getrowij && done) *done = PETSC_FALSE;
  else {
    if (done) *done = PETSC_TRUE;
    PetscCall(PetscLogEventBegin(MAT_GetRowIJ,mat,0,0,0));
    PetscCall((*mat->ops->getrowij)(mat,shift,symmetric,inodecompressed,n,ia,ja,done));
    PetscCall(PetscLogEventEnd(MAT_GetRowIJ,mat,0,0,0));
  }
  PetscFunctionReturn(0);
}

/*@C
    MatGetColumnIJ - Returns the compressed column storage i and j indices for sequential matrices.

    Collective on Mat

    Input Parameters:
+   mat - the matrix
.   shift - 1 or zero indicating we want the indices starting at 0 or 1
.   symmetric - PETSC_TRUE or PETSC_FALSE indicating the matrix data structure should be
                symmetrized
.   inodecompressed - PETSC_TRUE or PETSC_FALSE indicating if the nonzero structure of the
                 inodes or the nonzero elements is wanted. For BAIJ matrices the compressed version is
                 always used.
.   n - number of columns in the (possibly compressed) matrix
.   ia - the column pointers; that is ia[0] = 0, ia[col] = i[col-1] + number of elements in that col of the matrix
-   ja - the row indices

    Output Parameters:
.   done - PETSC_TRUE or PETSC_FALSE, indicating whether the values have been returned

    Level: developer

.seealso: `MatGetRowIJ()`, `MatRestoreColumnIJ()`
@*/
PetscErrorCode MatGetColumnIJ(Mat mat,PetscInt shift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscBool  *done)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidIntPointer(n,5);
  if (ia) PetscValidPointer(ia,6);
  if (ja) PetscValidPointer(ja,7);
  PetscValidBoolPointer(done,8);
  MatCheckPreallocated(mat,1);
  if (!mat->ops->getcolumnij) *done = PETSC_FALSE;
  else {
    *done = PETSC_TRUE;
    PetscCall((*mat->ops->getcolumnij)(mat,shift,symmetric,inodecompressed,n,ia,ja,done));
  }
  PetscFunctionReturn(0);
}

/*@C
    MatRestoreRowIJ - Call after you are completed with the ia,ja indices obtained with
    MatGetRowIJ().

    Collective on Mat

    Input Parameters:
+   mat - the matrix
.   shift - 1 or zero indicating we want the indices starting at 0 or 1
.   symmetric - PETSC_TRUE or PETSC_FALSE indicating the matrix data structure should be
                symmetrized
.   inodecompressed -  PETSC_TRUE or PETSC_FALSE indicating if the nonzero structure of the
                 inodes or the nonzero elements is wanted. For BAIJ matrices the compressed version is
                 always used.
.   n - size of (possibly compressed) matrix
.   ia - the row pointers
-   ja - the column indices

    Output Parameters:
.   done - PETSC_TRUE or PETSC_FALSE indicated that the values have been returned

    Note:
    This routine zeros out n, ia, and ja. This is to prevent accidental
    us of the array after it has been restored. If you pass NULL, it will
    not zero the pointers.  Use of ia or ja after MatRestoreRowIJ() is invalid.

    Level: developer

.seealso: `MatGetRowIJ()`, `MatRestoreColumnIJ()`
@*/
PetscErrorCode MatRestoreRowIJ(Mat mat,PetscInt shift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscBool  *done)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (ia) PetscValidPointer(ia,6);
  if (ja) PetscValidPointer(ja,7);
  if (done) PetscValidBoolPointer(done,8);
  MatCheckPreallocated(mat,1);

  if (!mat->ops->restorerowij && done) *done = PETSC_FALSE;
  else {
    if (done) *done = PETSC_TRUE;
    PetscCall((*mat->ops->restorerowij)(mat,shift,symmetric,inodecompressed,n,ia,ja,done));
    if (n)  *n = 0;
    if (ia) *ia = NULL;
    if (ja) *ja = NULL;
  }
  PetscFunctionReturn(0);
}

/*@C
    MatRestoreColumnIJ - Call after you are completed with the ia,ja indices obtained with
    MatGetColumnIJ().

    Collective on Mat

    Input Parameters:
+   mat - the matrix
.   shift - 1 or zero indicating we want the indices starting at 0 or 1
.   symmetric - PETSC_TRUE or PETSC_FALSE indicating the matrix data structure should be
                symmetrized
-   inodecompressed - PETSC_TRUE or PETSC_FALSE indicating if the nonzero structure of the
                 inodes or the nonzero elements is wanted. For BAIJ matrices the compressed version is
                 always used.

    Output Parameters:
+   n - size of (possibly compressed) matrix
.   ia - the column pointers
.   ja - the row indices
-   done - PETSC_TRUE or PETSC_FALSE indicated that the values have been returned

    Level: developer

.seealso: `MatGetColumnIJ()`, `MatRestoreRowIJ()`
@*/
PetscErrorCode MatRestoreColumnIJ(Mat mat,PetscInt shift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscBool  *done)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (ia) PetscValidPointer(ia,6);
  if (ja) PetscValidPointer(ja,7);
  PetscValidBoolPointer(done,8);
  MatCheckPreallocated(mat,1);

  if (!mat->ops->restorecolumnij) *done = PETSC_FALSE;
  else {
    *done = PETSC_TRUE;
    PetscCall((*mat->ops->restorecolumnij)(mat,shift,symmetric,inodecompressed,n,ia,ja,done));
    if (n)  *n = 0;
    if (ia) *ia = NULL;
    if (ja) *ja = NULL;
  }
  PetscFunctionReturn(0);
}

/*@C
    MatColoringPatch -Used inside matrix coloring routines that
    use MatGetRowIJ() and/or MatGetColumnIJ().

    Collective on Mat

    Input Parameters:
+   mat - the matrix
.   ncolors - max color value
.   n   - number of entries in colorarray
-   colorarray - array indicating color for each column

    Output Parameters:
.   iscoloring - coloring generated using colorarray information

    Level: developer

.seealso: `MatGetRowIJ()`, `MatGetColumnIJ()`

@*/
PetscErrorCode MatColoringPatch(Mat mat,PetscInt ncolors,PetscInt n,ISColoringValue colorarray[],ISColoring *iscoloring)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidIntPointer(colorarray,4);
  PetscValidPointer(iscoloring,5);
  MatCheckPreallocated(mat,1);

  if (!mat->ops->coloringpatch) {
    PetscCall(ISColoringCreate(PetscObjectComm((PetscObject)mat),ncolors,n,colorarray,PETSC_OWN_POINTER,iscoloring));
  } else {
    PetscCall((*mat->ops->coloringpatch)(mat,ncolors,n,colorarray,iscoloring));
  }
  PetscFunctionReturn(0);
}

/*@
   MatSetUnfactored - Resets a factored matrix to be treated as unfactored.

   Logically Collective on Mat

   Input Parameter:
.  mat - the factored matrix to be reset

   Notes:
   This routine should be used only with factored matrices formed by in-place
   factorization via ILU(0) (or by in-place LU factorization for the MATSEQDENSE
   format).  This option can save memory, for example, when solving nonlinear
   systems with a matrix-free Newton-Krylov method and a matrix-based, in-place
   ILU(0) preconditioner.

   Note that one can specify in-place ILU(0) factorization by calling
.vb
     PCType(pc,PCILU);
     PCFactorSeUseInPlace(pc);
.ve
   or by using the options -pc_type ilu -pc_factor_in_place

   In-place factorization ILU(0) can also be used as a local
   solver for the blocks within the block Jacobi or additive Schwarz
   methods (runtime option: -sub_pc_factor_in_place).  See Users-Manual: ch_pc
   for details on setting local solver options.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `PCFactorSetUseInPlace()`, `PCFactorGetUseInPlace()`

@*/
PetscErrorCode MatSetUnfactored(Mat mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  MatCheckPreallocated(mat,1);
  mat->factortype = MAT_FACTOR_NONE;
  if (!mat->ops->setunfactored) PetscFunctionReturn(0);
  PetscCall((*mat->ops->setunfactored)(mat));
  PetscFunctionReturn(0);
}

/*MC
    MatDenseGetArrayF90 - Accesses a matrix array from Fortran90.

    Synopsis:
    MatDenseGetArrayF90(Mat x,{Scalar, pointer :: xx_v(:,:)},integer ierr)

    Not collective

    Input Parameter:
.   x - matrix

    Output Parameters:
+   xx_v - the Fortran90 pointer to the array
-   ierr - error code

    Example of Usage:
.vb
      PetscScalar, pointer xx_v(:,:)
      ....
      call MatDenseGetArrayF90(x,xx_v,ierr)
      a = xx_v(3)
      call MatDenseRestoreArrayF90(x,xx_v,ierr)
.ve

    Level: advanced

.seealso: `MatDenseRestoreArrayF90()`, `MatDenseGetArray()`, `MatDenseRestoreArray()`, `MatSeqAIJGetArrayF90()`

M*/

/*MC
    MatDenseRestoreArrayF90 - Restores a matrix array that has been
    accessed with MatDenseGetArrayF90().

    Synopsis:
    MatDenseRestoreArrayF90(Mat x,{Scalar, pointer :: xx_v(:,:)},integer ierr)

    Not collective

    Input Parameters:
+   x - matrix
-   xx_v - the Fortran90 pointer to the array

    Output Parameter:
.   ierr - error code

    Example of Usage:
.vb
       PetscScalar, pointer xx_v(:,:)
       ....
       call MatDenseGetArrayF90(x,xx_v,ierr)
       a = xx_v(3)
       call MatDenseRestoreArrayF90(x,xx_v,ierr)
.ve

    Level: advanced

.seealso: `MatDenseGetArrayF90()`, `MatDenseGetArray()`, `MatDenseRestoreArray()`, `MatSeqAIJRestoreArrayF90()`

M*/

/*MC
    MatSeqAIJGetArrayF90 - Accesses a matrix array from Fortran90.

    Synopsis:
    MatSeqAIJGetArrayF90(Mat x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Not collective

    Input Parameter:
.   x - matrix

    Output Parameters:
+   xx_v - the Fortran90 pointer to the array
-   ierr - error code

    Example of Usage:
.vb
      PetscScalar, pointer xx_v(:)
      ....
      call MatSeqAIJGetArrayF90(x,xx_v,ierr)
      a = xx_v(3)
      call MatSeqAIJRestoreArrayF90(x,xx_v,ierr)
.ve

    Level: advanced

.seealso: `MatSeqAIJRestoreArrayF90()`, `MatSeqAIJGetArray()`, `MatSeqAIJRestoreArray()`, `MatDenseGetArrayF90()`

M*/

/*MC
    MatSeqAIJRestoreArrayF90 - Restores a matrix array that has been
    accessed with MatSeqAIJGetArrayF90().

    Synopsis:
    MatSeqAIJRestoreArrayF90(Mat x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Not collective

    Input Parameters:
+   x - matrix
-   xx_v - the Fortran90 pointer to the array

    Output Parameter:
.   ierr - error code

    Example of Usage:
.vb
       PetscScalar, pointer xx_v(:)
       ....
       call MatSeqAIJGetArrayF90(x,xx_v,ierr)
       a = xx_v(3)
       call MatSeqAIJRestoreArrayF90(x,xx_v,ierr)
.ve

    Level: advanced

.seealso: `MatSeqAIJGetArrayF90()`, `MatSeqAIJGetArray()`, `MatSeqAIJRestoreArray()`, `MatDenseRestoreArrayF90()`

M*/

/*@
    MatCreateSubMatrix - Gets a single submatrix on the same number of processors
                      as the original matrix.

    Collective on Mat

    Input Parameters:
+   mat - the original matrix
.   isrow - parallel IS containing the rows this processor should obtain
.   iscol - parallel IS containing all columns you wish to keep. Each process should list the columns that will be in IT's "diagonal part" in the new matrix.
-   cll - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

    Output Parameter:
.   newmat - the new submatrix, of the same type as the old

    Level: advanced

    Notes:
    The submatrix will be able to be multiplied with vectors using the same layout as iscol.

    Some matrix types place restrictions on the row and column indices, such
    as that they be sorted or that they be equal to each other.

    The index sets may not have duplicate entries.

      The first time this is called you should use a cll of MAT_INITIAL_MATRIX,
   the MatCreateSubMatrix() routine will create the newmat for you. Any additional calls
   to this routine with a mat of the same nonzero structure and with a call of MAT_REUSE_MATRIX
   will reuse the matrix generated the first time.  You should call MatDestroy() on newmat when
   you are finished using it.

    The communicator of the newly obtained matrix is ALWAYS the same as the communicator of
    the input matrix.

    If iscol is NULL then all columns are obtained (not supported in Fortran).

   Example usage:
   Consider the following 8x8 matrix with 34 non-zero values, that is
   assembled across 3 processors. Let's assume that proc0 owns 3 rows,
   proc1 owns 3 rows, proc2 owns 2 rows. This division can be shown
   as follows:

.vb
            1  2  0  |  0  3  0  |  0  4
    Proc0   0  5  6  |  7  0  0  |  8  0
            9  0 10  | 11  0  0  | 12  0
    -------------------------------------
           13  0 14  | 15 16 17  |  0  0
    Proc1   0 18  0  | 19 20 21  |  0  0
            0  0  0  | 22 23  0  | 24  0
    -------------------------------------
    Proc2  25 26 27  |  0  0 28  | 29  0
           30  0  0  | 31 32 33  |  0 34
.ve

    Suppose isrow = [0 1 | 4 | 6 7] and iscol = [1 2 | 3 4 5 | 6].  The resulting submatrix is

.vb
            2  0  |  0  3  0  |  0
    Proc0   5  6  |  7  0  0  |  8
    -------------------------------
    Proc1  18  0  | 19 20 21  |  0
    -------------------------------
    Proc2  26 27  |  0  0 28  | 29
            0  0  | 31 32 33  |  0
.ve

.seealso: `MatCreateSubMatrices()`, `MatCreateSubMatricesMPI()`, `MatCreateSubMatrixVirtual()`, `MatSubMatrixVirtualUpdate()`
@*/
PetscErrorCode MatCreateSubMatrix(Mat mat,IS isrow,IS iscol,MatReuse cll,Mat *newmat)
{
  PetscMPIInt    size;
  Mat            *local;
  IS             iscoltmp;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(isrow,IS_CLASSID,2);
  if (iscol) PetscValidHeaderSpecific(iscol,IS_CLASSID,3);
  PetscValidPointer(newmat,5);
  if (cll == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*newmat,MAT_CLASSID,5);
  PetscValidType(mat,1);
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(cll != MAT_IGNORE_MATRIX,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Cannot use MAT_IGNORE_MATRIX");

  MatCheckPreallocated(mat,1);
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size));

  if (!iscol || isrow == iscol) {
    PetscBool   stride;
    PetscMPIInt grabentirematrix = 0,grab;
    PetscCall(PetscObjectTypeCompare((PetscObject)isrow,ISSTRIDE,&stride));
    if (stride) {
      PetscInt first,step,n,rstart,rend;
      PetscCall(ISStrideGetInfo(isrow,&first,&step));
      if (step == 1) {
        PetscCall(MatGetOwnershipRange(mat,&rstart,&rend));
        if (rstart == first) {
          PetscCall(ISGetLocalSize(isrow,&n));
          if (n == rend-rstart) {
            grabentirematrix = 1;
          }
        }
      }
    }
    PetscCall(MPIU_Allreduce(&grabentirematrix,&grab,1,MPI_INT,MPI_MIN,PetscObjectComm((PetscObject)mat)));
    if (grab) {
      PetscCall(PetscInfo(mat,"Getting entire matrix as submatrix\n"));
      if (cll == MAT_INITIAL_MATRIX) {
        *newmat = mat;
        PetscCall(PetscObjectReference((PetscObject)mat));
      }
      PetscFunctionReturn(0);
    }
  }

  if (!iscol) {
    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)mat),mat->cmap->n,mat->cmap->rstart,1,&iscoltmp));
  } else {
    iscoltmp = iscol;
  }

  /* if original matrix is on just one processor then use submatrix generated */
  if (mat->ops->createsubmatrices && !mat->ops->createsubmatrix && size == 1 && cll == MAT_REUSE_MATRIX) {
    PetscCall(MatCreateSubMatrices(mat,1,&isrow,&iscoltmp,MAT_REUSE_MATRIX,&newmat));
    goto setproperties;
  } else if (mat->ops->createsubmatrices && !mat->ops->createsubmatrix && size == 1) {
    PetscCall(MatCreateSubMatrices(mat,1,&isrow,&iscoltmp,MAT_INITIAL_MATRIX,&local));
    *newmat = *local;
    PetscCall(PetscFree(local));
    goto setproperties;
  } else if (!mat->ops->createsubmatrix) {
    /* Create a new matrix type that implements the operation using the full matrix */
    PetscCall(PetscLogEventBegin(MAT_CreateSubMat,mat,0,0,0));
    switch (cll) {
    case MAT_INITIAL_MATRIX:
      PetscCall(MatCreateSubMatrixVirtual(mat,isrow,iscoltmp,newmat));
      break;
    case MAT_REUSE_MATRIX:
      PetscCall(MatSubMatrixVirtualUpdate(*newmat,mat,isrow,iscoltmp));
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_OUTOFRANGE,"Invalid MatReuse, must be either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX");
    }
    PetscCall(PetscLogEventEnd(MAT_CreateSubMat,mat,0,0,0));
    goto setproperties;
  }

  PetscCheck(mat->ops->createsubmatrix,PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  PetscCall(PetscLogEventBegin(MAT_CreateSubMat,mat,0,0,0));
  PetscCall((*mat->ops->createsubmatrix)(mat,isrow,iscoltmp,cll,newmat));
  PetscCall(PetscLogEventEnd(MAT_CreateSubMat,mat,0,0,0));

setproperties:
  PetscCall(ISEqualUnsorted(isrow,iscoltmp,&flg));
  if (flg) PetscCall(MatPropagateSymmetryOptions(mat,*newmat));
  if (!iscol) PetscCall(ISDestroy(&iscoltmp));
  if (*newmat && cll == MAT_INITIAL_MATRIX) PetscCall(PetscObjectStateIncrease((PetscObject)*newmat));
  PetscFunctionReturn(0);
}

/*@
   MatPropagateSymmetryOptions - Propagates symmetry options set on a matrix to another matrix

   Not Collective

   Input Parameters:
+  A - the matrix we wish to propagate options from
-  B - the matrix we wish to propagate options to

   Level: beginner

   Notes:
   Propagates the options associated to `MAT_SYMMETRY_ETERNAL`, `MAT_STRUCTURALLY_SYMMETRIC`, `MAT_HERMITIAN`, `MAT_SPD`, `MAT_SYMMETRIC`, and `MAT_STRUCTURAL_SYMMETRY_ETERNAL`

.seealso: `MatSetOption()`, `MatIsSymmetricKnown()`, `MatIsSPDKnown()`, `MatIsHermitianKnown()`, MatIsStructurallySymmetricKnown()`
@*/
PetscErrorCode MatPropagateSymmetryOptions(Mat A, Mat B)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  B->symmetry_eternal            = A->symmetry_eternal;
  B->structural_symmetry_eternal = A->structural_symmetry_eternal;
  B->symmetric                   = A->symmetric;
  B->structurally_symmetric      = A->structurally_symmetric;
  B->spd                         = A->spd;
  B->hermitian                   = A->hermitian;
  PetscFunctionReturn(0);
}

/*@
   MatStashSetInitialSize - sets the sizes of the matrix stash, that is
   used during the assembly process to store values that belong to
   other processors.

   Not Collective

   Input Parameters:
+  mat   - the matrix
.  size  - the initial size of the stash.
-  bsize - the initial size of the block-stash(if used).

   Options Database Keys:
+   -matstash_initial_size <size> or <size0,size1,...sizep-1>
-   -matstash_block_initial_size <bsize>  or <bsize0,bsize1,...bsizep-1>

   Level: intermediate

   Notes:
     The block-stash is used for values set with MatSetValuesBlocked() while
     the stash is used for values set with MatSetValues()

     Run with the option -info and look for output of the form
     MatAssemblyBegin_MPIXXX:Stash has MM entries, uses nn mallocs.
     to determine the appropriate value, MM, to use for size and
     MatAssemblyBegin_MPIXXX:Block-Stash has BMM entries, uses nn mallocs.
     to determine the value, BMM to use for bsize

.seealso: `MatAssemblyBegin()`, `MatAssemblyEnd()`, `Mat`, `MatStashGetInfo()`

@*/
PetscErrorCode MatStashSetInitialSize(Mat mat,PetscInt size, PetscInt bsize)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscCall(MatStashSetInitialSize_Private(&mat->stash,size));
  PetscCall(MatStashSetInitialSize_Private(&mat->bstash,bsize));
  PetscFunctionReturn(0);
}

/*@
   MatInterpolateAdd - w = y + A*x or A'*x depending on the shape of
     the matrix

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat   - the matrix
.  x,y - the vectors
-  w - where the result is stored

   Level: intermediate

   Notes:
    w may be the same vector as y.

    This allows one to use either the restriction or interpolation (its transpose)
    matrix to do the interpolation

.seealso: `MatMultAdd()`, `MatMultTransposeAdd()`, `MatRestrict()`

@*/
PetscErrorCode MatInterpolateAdd(Mat A,Vec x,Vec y,Vec w)
{
  PetscInt       M,N,Ny;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidHeaderSpecific(w,VEC_CLASSID,4);
  PetscCall(MatGetSize(A,&M,&N));
  PetscCall(VecGetSize(y,&Ny));
  if (M == Ny) {
    PetscCall(MatMultAdd(A,x,y,w));
  } else {
    PetscCall(MatMultTransposeAdd(A,x,y,w));
  }
  PetscFunctionReturn(0);
}

/*@
   MatInterpolate - y = A*x or A'*x depending on the shape of
     the matrix

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat   - the matrix
-  x,y - the vectors

   Level: intermediate

   Notes:
    This allows one to use either the restriction or interpolation (its transpose)
    matrix to do the interpolation

.seealso: `MatMultAdd()`, `MatMultTransposeAdd()`, `MatRestrict()`

@*/
PetscErrorCode MatInterpolate(Mat A,Vec x,Vec y)
{
  PetscInt       M,N,Ny;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscCall(MatGetSize(A,&M,&N));
  PetscCall(VecGetSize(y,&Ny));
  if (M == Ny) {
    PetscCall(MatMult(A,x,y));
  } else {
    PetscCall(MatMultTranspose(A,x,y));
  }
  PetscFunctionReturn(0);
}

/*@
   MatRestrict - y = A*x or A'*x

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat   - the matrix
-  x,y - the vectors

   Level: intermediate

   Notes:
    This allows one to use either the restriction or interpolation (its transpose)
    matrix to do the restriction

.seealso: `MatMultAdd()`, `MatMultTransposeAdd()`, `MatInterpolate()`

@*/
PetscErrorCode MatRestrict(Mat A,Vec x,Vec y)
{
  PetscInt       M,N,Ny;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscCall(MatGetSize(A,&M,&N));
  PetscCall(VecGetSize(y,&Ny));
  if (M == Ny) {
    PetscCall(MatMult(A,x,y));
  } else {
    PetscCall(MatMultTranspose(A,x,y));
  }
  PetscFunctionReturn(0);
}

/*@
   MatMatInterpolateAdd - Y = W + A*X or W + A'*X

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat   - the matrix
-  w, x - the input dense matrices

   Output Parameters:
.  y - the output dense matrix

   Level: intermediate

   Notes:
    This allows one to use either the restriction or interpolation (its transpose)
    matrix to do the interpolation. y matrix can be reused if already created with the proper sizes,
    otherwise it will be recreated. y must be initialized to NULL if not supplied.

.seealso: `MatInterpolateAdd()`, `MatMatInterpolate()`, `MatMatRestrict()`

@*/
PetscErrorCode MatMatInterpolateAdd(Mat A,Mat x,Mat w,Mat *y)
{
  PetscInt       M,N,Mx,Nx,Mo,My = 0,Ny = 0;
  PetscBool      trans = PETSC_TRUE;
  MatReuse       reuse = MAT_INITIAL_MATRIX;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(x,MAT_CLASSID,2);
  PetscValidType(x,2);
  if (w) PetscValidHeaderSpecific(w,MAT_CLASSID,3);
  if (*y) PetscValidHeaderSpecific(*y,MAT_CLASSID,4);
  PetscCall(MatGetSize(A,&M,&N));
  PetscCall(MatGetSize(x,&Mx,&Nx));
  if (N == Mx) trans = PETSC_FALSE;
  else PetscCheck(M == Mx,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Size mismatch: A %" PetscInt_FMT "x%" PetscInt_FMT ", X %" PetscInt_FMT "x%" PetscInt_FMT,M,N,Mx,Nx);
  Mo = trans ? N : M;
  if (*y) {
    PetscCall(MatGetSize(*y,&My,&Ny));
    if (Mo == My && Nx == Ny) { reuse = MAT_REUSE_MATRIX; }
    else {
      PetscCheck(w || *y != w,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Cannot reuse y and w, size mismatch: A %" PetscInt_FMT "x%" PetscInt_FMT ", X %" PetscInt_FMT "x%" PetscInt_FMT ", Y %" PetscInt_FMT "x%" PetscInt_FMT,M,N,Mx,Nx,My,Ny);
      PetscCall(MatDestroy(y));
    }
  }

  if (w && *y == w) { /* this is to minimize changes in PCMG */
    PetscBool flg;

    PetscCall(PetscObjectQuery((PetscObject)*y,"__MatMatIntAdd_w",(PetscObject*)&w));
    if (w) {
      PetscInt My,Ny,Mw,Nw;

      PetscCall(PetscObjectTypeCompare((PetscObject)*y,((PetscObject)w)->type_name,&flg));
      PetscCall(MatGetSize(*y,&My,&Ny));
      PetscCall(MatGetSize(w,&Mw,&Nw));
      if (!flg || My != Mw || Ny != Nw) w = NULL;
    }
    if (!w) {
      PetscCall(MatDuplicate(*y,MAT_COPY_VALUES,&w));
      PetscCall(PetscObjectCompose((PetscObject)*y,"__MatMatIntAdd_w",(PetscObject)w));
      PetscCall(PetscLogObjectParent((PetscObject)*y,(PetscObject)w));
      PetscCall(PetscObjectDereference((PetscObject)w));
    } else {
      PetscCall(MatCopy(*y,w,UNKNOWN_NONZERO_PATTERN));
    }
  }
  if (!trans) {
    PetscCall(MatMatMult(A,x,reuse,PETSC_DEFAULT,y));
  } else {
    PetscCall(MatTransposeMatMult(A,x,reuse,PETSC_DEFAULT,y));
  }
  if (w) PetscCall(MatAXPY(*y,1.0,w,UNKNOWN_NONZERO_PATTERN));
  PetscFunctionReturn(0);
}

/*@
   MatMatInterpolate - Y = A*X or A'*X

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat   - the matrix
-  x - the input dense matrix

   Output Parameters:
.  y - the output dense matrix

   Level: intermediate

   Notes:
    This allows one to use either the restriction or interpolation (its transpose)
    matrix to do the interpolation. y matrix can be reused if already created with the proper sizes,
    otherwise it will be recreated. y must be initialized to NULL if not supplied.

.seealso: `MatInterpolate()`, `MatRestrict()`, `MatMatRestrict()`

@*/
PetscErrorCode MatMatInterpolate(Mat A,Mat x,Mat *y)
{
  PetscFunctionBegin;
  PetscCall(MatMatInterpolateAdd(A,x,NULL,y));
  PetscFunctionReturn(0);
}

/*@
   MatMatRestrict - Y = A*X or A'*X

   Neighbor-wise Collective on Mat

   Input Parameters:
+  mat   - the matrix
-  x - the input dense matrix

   Output Parameters:
.  y - the output dense matrix

   Level: intermediate

   Notes:
    This allows one to use either the restriction or interpolation (its transpose)
    matrix to do the restriction. y matrix can be reused if already created with the proper sizes,
    otherwise it will be recreated. y must be initialized to NULL if not supplied.

.seealso: `MatRestrict()`, `MatInterpolate()`, `MatMatInterpolate()`
@*/
PetscErrorCode MatMatRestrict(Mat A,Mat x,Mat *y)
{
  PetscFunctionBegin;
  PetscCall(MatMatInterpolateAdd(A,x,NULL,y));
  PetscFunctionReturn(0);
}

/*@
   MatGetNullSpace - retrieves the null space of a matrix.

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix
-  nullsp - the null space object

   Level: developer

.seealso: `MatCreate()`, `MatNullSpaceCreate()`, `MatSetNearNullSpace()`, `MatSetNullSpace()`
@*/
PetscErrorCode MatGetNullSpace(Mat mat, MatNullSpace *nullsp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(nullsp,2);
  *nullsp = (mat->symmetric == PETSC_BOOL3_TRUE && !mat->nullsp) ? mat->transnullsp : mat->nullsp;
  PetscFunctionReturn(0);
}

/*@
   MatSetNullSpace - attaches a null space to a matrix.

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix
-  nullsp - the null space object

   Level: advanced

   Notes:
      This null space is used by the KSP linear solvers to solve singular systems.

      Overwrites any previous null space that may have been attached. You can remove the null space from the matrix object by calling this routine with an nullsp of NULL

      For inconsistent singular systems (linear systems where the right hand side is not in the range of the operator) the KSP residuals will not converge to
      to zero but the linear system will still be solved in a least squares sense.

      The fundamental theorem of linear algebra (Gilbert Strang, Introduction to Applied Mathematics, page 72) states that
   the domain of a matrix A (from R^n to R^m (m rows, n columns) R^n = the direct sum of the null space of A, n(A), + the range of A^T, R(A^T).
   Similarly R^m = direct sum n(A^T) + R(A).  Hence the linear system A x = b has a solution only if b in R(A) (or correspondingly b is orthogonal to
   n(A^T)) and if x is a solution then x + alpha n(A) is a solution for any alpha. The minimum norm solution is orthogonal to n(A). For problems without a solution
   the solution that minimizes the norm of the residual (the least squares solution) can be obtained by solving A x = \hat{b} where \hat{b} is b orthogonalized to the n(A^T).
   This  \hat{b} can be obtained by calling MatNullSpaceRemove() with the null space of the transpose of the matrix.

    If the matrix is known to be symmetric because it is an SBAIJ matrix or one as called MatSetOption(mat,MAT_SYMMETRIC or MAT_SYMMETRY_ETERNAL,PETSC_TRUE); this
    routine also automatically calls MatSetTransposeNullSpace().

    The user should call `MatNullSpaceDestroy()`.

.seealso: `MatCreate()`, `MatNullSpaceCreate()`, `MatSetNearNullSpace()`, `MatGetNullSpace()`, `MatSetTransposeNullSpace()`, `MatGetTransposeNullSpace()`, `MatNullSpaceRemove()`,
          `KSPSetPCSide()`
@*/
PetscErrorCode MatSetNullSpace(Mat mat,MatNullSpace nullsp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (nullsp) PetscValidHeaderSpecific(nullsp,MAT_NULLSPACE_CLASSID,2);
  if (nullsp) PetscCall(PetscObjectReference((PetscObject)nullsp));
  PetscCall(MatNullSpaceDestroy(&mat->nullsp));
  mat->nullsp = nullsp;
  if (mat->symmetric == PETSC_BOOL3_TRUE) {
    PetscCall(MatSetTransposeNullSpace(mat,nullsp));
  }
  PetscFunctionReturn(0);
}

/*@
   MatGetTransposeNullSpace - retrieves the null space of the transpose of a matrix.

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix
-  nullsp - the null space object

   Level: developer

.seealso: `MatCreate()`, `MatNullSpaceCreate()`, `MatSetNearNullSpace()`, `MatSetTransposeNullSpace()`, `MatSetNullSpace()`, `MatGetNullSpace()`
@*/
PetscErrorCode MatGetTransposeNullSpace(Mat mat, MatNullSpace *nullsp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidPointer(nullsp,2);
  *nullsp = (mat->symmetric == PETSC_BOOL3_TRUE && !mat->transnullsp) ? mat->nullsp : mat->transnullsp;
  PetscFunctionReturn(0);
}

/*@
   MatSetTransposeNullSpace - attaches the null space of a transpose of a matrix to the matrix

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix
-  nullsp - the null space object

   Level: advanced

   Notes:
      This allows solving singular linear systems defined by the transpose of the matrix using KSP solvers with left preconditioning.

      See MatSetNullSpace()

.seealso: `MatCreate()`, `MatNullSpaceCreate()`, `MatSetNearNullSpace()`, `MatGetNullSpace()`, `MatSetNullSpace()`, `MatGetTransposeNullSpace()`, `MatNullSpaceRemove()`, `KSPSetPCSide()`
@*/
PetscErrorCode MatSetTransposeNullSpace(Mat mat,MatNullSpace nullsp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (nullsp) PetscValidHeaderSpecific(nullsp,MAT_NULLSPACE_CLASSID,2);
  if (nullsp) PetscCall(PetscObjectReference((PetscObject)nullsp));
  PetscCall(MatNullSpaceDestroy(&mat->transnullsp));
  mat->transnullsp = nullsp;
  PetscFunctionReturn(0);
}

/*@
   MatSetNearNullSpace - attaches a null space to a matrix, which is often the null space (rigid body modes) of the operator without boundary conditions
        This null space will be used to provide near null space vectors to a multigrid preconditioner built from this matrix.

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix
-  nullsp - the null space object

   Level: advanced

   Notes:
      Overwrites any previous near null space that may have been attached

      You can remove the null space by calling this routine with an nullsp of NULL

.seealso: `MatCreate()`, `MatNullSpaceCreate()`, `MatSetNullSpace()`, `MatNullSpaceCreateRigidBody()`, `MatGetNearNullSpace()`
@*/
PetscErrorCode MatSetNearNullSpace(Mat mat,MatNullSpace nullsp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (nullsp) PetscValidHeaderSpecific(nullsp,MAT_NULLSPACE_CLASSID,2);
  MatCheckPreallocated(mat,1);
  if (nullsp) PetscCall(PetscObjectReference((PetscObject)nullsp));
  PetscCall(MatNullSpaceDestroy(&mat->nearnullsp));
  mat->nearnullsp = nullsp;
  PetscFunctionReturn(0);
}

/*@
   MatGetNearNullSpace - Get null space attached with MatSetNearNullSpace()

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  nullsp - the null space object, NULL if not set

   Level: developer

.seealso: `MatSetNearNullSpace()`, `MatGetNullSpace()`, `MatNullSpaceCreate()`
@*/
PetscErrorCode MatGetNearNullSpace(Mat mat,MatNullSpace *nullsp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidPointer(nullsp,2);
  MatCheckPreallocated(mat,1);
  *nullsp = mat->nearnullsp;
  PetscFunctionReturn(0);
}

/*@C
   MatICCFactor - Performs in-place incomplete Cholesky factorization of matrix.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  row - row/column permutation
.  fill - expected fill factor >= 1.0
-  level - level of fill, for ICC(k)

   Notes:
   Probably really in-place only when level of fill is zero, otherwise allocates
   new space to store factored matrix and deletes previous memory.

   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatICCFactorSymbolic()`, `MatLUFactorNumeric()`, `MatCholeskyFactor()`

    Developer Note: fortran interface is not autogenerated as the f90
    interface definition cannot be generated correctly [due to MatFactorInfo]

@*/
PetscErrorCode MatICCFactor(Mat mat,IS row,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (row) PetscValidHeaderSpecific(row,IS_CLASSID,2);
  PetscValidPointer(info,3);
  PetscCheck(mat->rmap->N == mat->cmap->N,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONG,"matrix must be square");
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->ops->iccfactor,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,1);
  PetscCall((*mat->ops->iccfactor)(mat,row,info));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
  PetscFunctionReturn(0);
}

/*@
   MatDiagonalScaleLocal - Scales columns of a matrix given the scaling values including the
         ghosted ones.

   Not Collective

   Input Parameters:
+  mat - the matrix
-  diag - the diagonal values, including ghost ones

   Level: developer

   Notes:
    Works only for MPIAIJ and MPIBAIJ matrices

.seealso: `MatDiagonalScale()`
@*/
PetscErrorCode MatDiagonalScaleLocal(Mat mat,Vec diag)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(diag,VEC_CLASSID,2);
  PetscValidType(mat,1);

  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Matrix must be already assembled");
  PetscCall(PetscLogEventBegin(MAT_Scale,mat,0,0,0));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size));
  if (size == 1) {
    PetscInt n,m;
    PetscCall(VecGetSize(diag,&n));
    PetscCall(MatGetSize(mat,NULL,&m));
    if (m == n) {
      PetscCall(MatDiagonalScale(mat,NULL,diag));
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only supported for sequential matrices when no ghost points/periodic conditions");
  } else {
    PetscUseMethod(mat,"MatDiagonalScaleLocal_C",(Mat,Vec),(mat,diag));
  }
  PetscCall(PetscLogEventEnd(MAT_Scale,mat,0,0,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
  PetscFunctionReturn(0);
}

/*@
   MatGetInertia - Gets the inertia from a factored matrix

   Collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+   nneg - number of negative eigenvalues
.   nzero - number of zero eigenvalues
-   npos - number of positive eigenvalues

   Level: advanced

   Notes:
    Matrix must have been factored by MatCholeskyFactor()

@*/
PetscErrorCode MatGetInertia(Mat mat,PetscInt *nneg,PetscInt *nzero,PetscInt *npos)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscCheck(mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Numeric factor mat is not assembled");
  PetscCheck(mat->ops->getinertia,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  PetscCall((*mat->ops->getinertia)(mat,nneg,nzero,npos));
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/
/*@C
   MatSolves - Solves A x = b, given a factored matrix, for a collection of vectors

   Neighbor-wise Collective on Mats

   Input Parameters:
+  mat - the factored matrix
-  b - the right-hand-side vectors

   Output Parameter:
.  x - the result vectors

   Notes:
   The vectors b and x cannot be the same.  I.e., one cannot
   call MatSolves(A,x,x).

   Notes:
   Most users should employ the simplified KSP interface for linear solvers
   instead of working directly with matrix algebra routines such as this.
   See, e.g., KSPCreate().

   Level: developer

.seealso: `MatSolveAdd()`, `MatSolveTranspose()`, `MatSolveTransposeAdd()`, `MatSolve()`
@*/
PetscErrorCode MatSolves(Mat mat,Vecs b,Vecs x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscCheck(x != b,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  PetscCheck(mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  if (!mat->rmap->N && !mat->cmap->N) PetscFunctionReturn(0);

  PetscCheck(mat->ops->solves,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
  MatCheckPreallocated(mat,1);
  PetscCall(PetscLogEventBegin(MAT_Solves,mat,0,0,0));
  PetscCall((*mat->ops->solves)(mat,b,x));
  PetscCall(PetscLogEventEnd(MAT_Solves,mat,0,0,0));
  PetscFunctionReturn(0);
}

/*@
   MatIsSymmetric - Test whether a matrix is symmetric

   Collective on Mat

   Input Parameters:
+  A - the matrix to test
-  tol - difference between value and its transpose less than this amount counts as equal (use 0.0 for exact transpose)

   Output Parameters:
.  flg - the result

   Notes:
    For real numbers MatIsSymmetric() and MatIsHermitian() return identical results

    If the matrix does not yet know if it is symmetric or not this can be an expensive operation, also available `MatIsSymmetricKnown()`

   Level: intermediate

.seealso: `MatTranspose()`, `MatIsTranspose()`, `MatIsHermitian()`, `MatIsStructurallySymmetric()`, `MatSetOption()`, `MatIsSymmetricKnown()`
@*/
PetscErrorCode MatIsSymmetric(Mat A,PetscReal tol,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidBoolPointer(flg,3);

  if (A->symmetric == PETSC_BOOL3_TRUE) *flg = PETSC_TRUE;
  else if (A->symmetric == PETSC_BOOL3_FALSE) *flg = PETSC_FALSE;
  else {
    if (!A->ops->issymmetric) {
      MatType mattype;
      PetscCall(MatGetType(A,&mattype));
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrix of type %s does not support checking for symmetric",mattype);
    }
    PetscCall((*A->ops->issymmetric)(A,tol,flg));
    if (!tol) PetscCall(MatSetOption(A,MAT_SYMMETRIC,*flg));
  }
  PetscFunctionReturn(0);
}

/*@
   MatIsHermitian - Test whether a matrix is Hermitian

   Collective on Mat

   Input Parameters:
+  A - the matrix to test
-  tol - difference between value and its transpose less than this amount counts as equal (use 0.0 for exact Hermitian)

   Output Parameters:
.  flg - the result

   Level: intermediate

   Notes:
    For real numbers MatIsSymmetric() and MatIsHermitian() return identical results

    If the matrix does not yet know if it is hermitian or not this can be an expensive operation, also available `MatIsHermitianKnown()`

.seealso: `MatTranspose()`, `MatIsTranspose()`, `MatIsHermitianKnown()`, `MatIsStructurallySymmetric()`, `MatSetOption()`,
          `MatIsSymmetricKnown()`, `MatIsSymmetric()`
@*/
PetscErrorCode MatIsHermitian(Mat A,PetscReal tol,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidBoolPointer(flg,3);

  if (A->hermitian == PETSC_BOOL3_TRUE) *flg = PETSC_TRUE;
  else if (A->hermitian == PETSC_BOOL3_FALSE) *flg = PETSC_FALSE;
  else {
    if (!A->ops->ishermitian) {
      MatType mattype;
      PetscCall(MatGetType(A,&mattype));
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrix of type %s does not support checking for hermitian",mattype);
    }
    PetscCall((*A->ops->ishermitian)(A,tol,flg));
    if (!tol) PetscCall(MatSetOption(A,MAT_HERMITIAN,*flg));
  }
  PetscFunctionReturn(0);
}

/*@
   MatIsSymmetricKnown - Checks if a matrix knows if it is symmetric or not and its symmetric state

   Not Collective

   Input Parameter:
.  A - the matrix to check

   Output Parameters:
+  set - PETSC_TRUE if the matrix knows its symmetry state (this tells you if the next flag is valid)
-  flg - the result (only valid if set is PETSC_TRUE)

   Level: advanced

   Note: Does not check the matrix values directly, so this may return unknown (set = PETSC_FALSE). Use MatIsSymmetric()
         if you want it explicitly checked

.seealso: `MatTranspose()`, `MatIsTranspose()`, `MatIsHermitian()`, `MatIsStructurallySymmetric()`, `MatSetOption()`, `MatIsSymmetric()`, `MatIsHermitianKnown()`
@*/
PetscErrorCode MatIsSymmetricKnown(Mat A,PetscBool *set,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidBoolPointer(set,2);
  PetscValidBoolPointer(flg,3);
  if (A->symmetric != PETSC_BOOL3_UNKNOWN) {
    *set = PETSC_TRUE;
    *flg = PetscBool3ToBool(A->symmetric);
  } else {
    *set = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*@
   MatIsSPDKnown - Checks if a matrix knows if it is symmetric positive definite or not and its symmetric positive definite state

   Not Collective

   Input Parameter:
.  A - the matrix to check

   Output Parameters:
+  set - PETSC_TRUE if the matrix knows its symmetric positive definite state (this tells you if the next flag is valid)
-  flg - the result (only valid if set is PETSC_TRUE)

   Level: advanced

   Note:
   Does not check the matrix values directly, so this may return unknown (set = PETSC_FALSE).

.seealso: `MatTranspose()`, `MatIsTranspose()`, `MatIsHermitian()`, `MatIsStructurallySymmetric()`, `MatSetOption()`, `MatIsSymmetric()`, `MatIsHermitianKnown()`
@*/
PetscErrorCode MatIsSPDKnown(Mat A,PetscBool *set,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidBoolPointer(set,2);
  PetscValidBoolPointer(flg,3);
  if (A->spd != PETSC_BOOL3_UNKNOWN) {
    *set = PETSC_TRUE;
    *flg = PetscBool3ToBool(A->spd);
  } else {
    *set = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*@
   MatIsHermitianKnown - Checks if a matrix knows if it is Hermitian or not and its Hermitian state

   Not Collective

   Input Parameter:
.  A - the matrix to check

   Output Parameters:
+  set - PETSC_TRUE if the matrix knows its Hermitian state (this tells you if the next flag is valid)
-  flg - the result (only valid if set is PETSC_TRUE)

   Level: advanced

   Note: Does not check the matrix values directly, so this may return unknown (set = PETSC_FALSE). Use MatIsHermitian()
         if you want it explicitly checked

.seealso: `MatTranspose()`, `MatIsTranspose()`, `MatIsHermitian()`, `MatIsStructurallySymmetric()`, `MatSetOption()`, `MatIsSymmetric()`
@*/
PetscErrorCode MatIsHermitianKnown(Mat A,PetscBool *set,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidBoolPointer(set,2);
  PetscValidBoolPointer(flg,3);
  if (A->hermitian  != PETSC_BOOL3_UNKNOWN) {
    *set = PETSC_TRUE;
    *flg = PetscBool3ToBool(A->hermitian);
  } else {
    *set = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*@
   MatIsStructurallySymmetric - Test whether a matrix is structurally symmetric

   Collective on Mat

   Input Parameter:
.  A - the matrix to test

   Output Parameters:
.  flg - the result

   Notes:
      If the matrix does yet know it is structurally symmetric this can be an expensive operation, also available `MatIsStructurallySymmetricKnown()`

   Level: intermediate

.seealso: `MatTranspose()`, `MatIsTranspose()`, `MatIsHermitian()`, `MatIsSymmetric()`, `MatSetOption()`, `MatIsStructurallySymmetricKnown()`
@*/
PetscErrorCode MatIsStructurallySymmetric(Mat A,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  if (A->structurally_symmetric  != PETSC_BOOL3_UNKNOWN) {
    *flg = PetscBool3ToBool(A->structurally_symmetric);
  } else {
    PetscCheck(A->ops->isstructurallysymmetric,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Matrix of type %s does not support checking for structural symmetry",((PetscObject)A)->type_name);
    PetscCall((*A->ops->isstructurallysymmetric)(A,flg));
    PetscCall(MatSetOption(A,MAT_STRUCTURALLY_SYMMETRIC,*flg));
  }
  PetscFunctionReturn(0);
}

/*@
   MatIsStructurallySymmetricKnown - Checks if a matrix knows if it is structurally symmetric or not and its structurally symmetric state

   Not Collective

   Input Parameter:
.  A - the matrix to check

   Output Parameters:
+  set - PETSC_TRUE if the matrix knows its structurally symmetric state (this tells you if the next flag is valid)
-  flg - the result (only valid if set is PETSC_TRUE)

   Level: advanced

.seealso: `MatTranspose()`, `MatIsTranspose()`, `MatIsHermitian()`, `MatIsStructurallySymmetric()`, `MatSetOption()`, `MatIsSymmetric()`, `MatIsHermitianKnown()`
@*/
PetscErrorCode MatIsStructurallySymmetricKnown(Mat A,PetscBool *set,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidBoolPointer(set,2);
  PetscValidBoolPointer(flg,3);
  if (A->structurally_symmetric != PETSC_BOOL3_UNKNOWN) {
    *set = PETSC_TRUE;
    *flg = PetscBool3ToBool(A->structurally_symmetric);
  } else {
    *set = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*@
   MatStashGetInfo - Gets how many values are currently in the matrix stash, i.e. need
       to be communicated to other processors during the MatAssemblyBegin/End() process

    Not collective

   Input Parameter:
.   vec - the vector

   Output Parameters:
+   nstash   - the size of the stash
.   reallocs - the number of additional mallocs incurred.
.   bnstash   - the size of the block stash
-   breallocs - the number of additional mallocs incurred.in the block stash

   Level: advanced

.seealso: `MatAssemblyBegin()`, `MatAssemblyEnd()`, `Mat`, `MatStashSetInitialSize()`

@*/
PetscErrorCode MatStashGetInfo(Mat mat,PetscInt *nstash,PetscInt *reallocs,PetscInt *bnstash,PetscInt *breallocs)
{
  PetscFunctionBegin;
  PetscCall(MatStashGetInfo_Private(&mat->stash,nstash,reallocs));
  PetscCall(MatStashGetInfo_Private(&mat->bstash,bnstash,breallocs));
  PetscFunctionReturn(0);
}

/*@C
   MatCreateVecs - Get vector(s) compatible with the matrix, i.e. with the same
     parallel layout

   Collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+   right - (optional) vector that the matrix can be multiplied against
-   left - (optional) vector that the matrix vector product can be stored in

   Notes:
    The blocksize of the returned vectors is determined by the row and column block sizes set with MatSetBlockSizes() or the single blocksize (same for both) set by MatSetBlockSize().

  Notes:
    These are new vectors which are not owned by the Mat, they should be destroyed in VecDestroy() when no longer needed

  Level: advanced

.seealso: `MatCreate()`, `VecDestroy()`
@*/
PetscErrorCode MatCreateVecs(Mat mat,Vec *right,Vec *left)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  if (mat->ops->getvecs) {
    PetscCall((*mat->ops->getvecs)(mat,right,left));
  } else {
    PetscInt rbs,cbs;
    PetscCall(MatGetBlockSizes(mat,&rbs,&cbs));
    if (right) {
      PetscCheck(mat->cmap->n >= 0,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"PetscLayout for columns not yet setup");
      PetscCall(VecCreate(PetscObjectComm((PetscObject)mat),right));
      PetscCall(VecSetSizes(*right,mat->cmap->n,PETSC_DETERMINE));
      PetscCall(VecSetBlockSize(*right,cbs));
      PetscCall(VecSetType(*right,mat->defaultvectype));
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
      if (mat->boundtocpu && mat->bindingpropagates) {
        PetscCall(VecSetBindingPropagates(*right,PETSC_TRUE));
        PetscCall(VecBindToCPU(*right,PETSC_TRUE));
      }
#endif
      PetscCall(PetscLayoutReference(mat->cmap,&(*right)->map));
    }
    if (left) {
      PetscCheck(mat->rmap->n >= 0,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"PetscLayout for rows not yet setup");
      PetscCall(VecCreate(PetscObjectComm((PetscObject)mat),left));
      PetscCall(VecSetSizes(*left,mat->rmap->n,PETSC_DETERMINE));
      PetscCall(VecSetBlockSize(*left,rbs));
      PetscCall(VecSetType(*left,mat->defaultvectype));
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
      if (mat->boundtocpu && mat->bindingpropagates) {
        PetscCall(VecSetBindingPropagates(*left,PETSC_TRUE));
        PetscCall(VecBindToCPU(*left,PETSC_TRUE));
      }
#endif
      PetscCall(PetscLayoutReference(mat->rmap,&(*left)->map));
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   MatFactorInfoInitialize - Initializes a MatFactorInfo data structure
     with default values.

   Not Collective

   Input Parameters:
.    info - the MatFactorInfo data structure

   Notes:
    The solvers are generally used through the KSP and PC objects, for example
          PCLU, PCILU, PCCHOLESKY, PCICC

   Level: developer

.seealso: `MatFactorInfo`

    Developer Note: fortran interface is not autogenerated as the f90
    interface definition cannot be generated correctly [due to MatFactorInfo]

@*/

PetscErrorCode MatFactorInfoInitialize(MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(PetscMemzero(info,sizeof(MatFactorInfo)));
  PetscFunctionReturn(0);
}

/*@
   MatFactorSetSchurIS - Set indices corresponding to the Schur complement you wish to have computed

   Collective on Mat

   Input Parameters:
+  mat - the factored matrix
-  is - the index set defining the Schur indices (0-based)

   Notes:
    Call MatFactorSolveSchurComplement() or MatFactorSolveSchurComplementTranspose() after this call to solve a Schur complement system.

   You can call MatFactorGetSchurComplement() or MatFactorCreateSchurComplement() after this call.

   Level: developer

.seealso: `MatGetFactor()`, `MatFactorGetSchurComplement()`, `MatFactorRestoreSchurComplement()`, `MatFactorCreateSchurComplement()`, `MatFactorSolveSchurComplement()`,
          `MatFactorSolveSchurComplementTranspose()`, `MatFactorSolveSchurComplement()`

@*/
PetscErrorCode MatFactorSetSchurIS(Mat mat,IS is)
{
  PetscErrorCode (*f)(Mat,IS);

  PetscFunctionBegin;
  PetscValidType(mat,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(is,2);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  PetscCheckSameComm(mat,1,is,2);
  PetscCheck(mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Only for factored matrix");
  PetscCall(PetscObjectQueryFunction((PetscObject)mat,"MatFactorSetSchurIS_C",&f));
  PetscCheck(f,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"The selected MatSolverType does not support Schur complement computation. You should use MATSOLVERMUMPS or MATSOLVERMKL_PARDISO");
  PetscCall(MatDestroy(&mat->schur));
  PetscCall((*f)(mat,is));
  PetscCheck(mat->schur,PetscObjectComm((PetscObject)mat),PETSC_ERR_PLIB,"Schur complement has not been created");
  PetscFunctionReturn(0);
}

/*@
  MatFactorCreateSchurComplement - Create a Schur complement matrix object using Schur data computed during the factorization step

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-MUMPS interface
.  S - location where to return the Schur complement, can be NULL
-  status - the status of the Schur complement matrix, can be NULL

   Notes:
   You must call MatFactorSetSchurIS() before calling this routine.

   The routine provides a copy of the Schur matrix stored within the solver data structures.
   The caller must destroy the object when it is no longer needed.
   If MatFactorInvertSchurComplement() has been called, the routine gets back the inverse.

   Use MatFactorGetSchurComplement() to get access to the Schur complement matrix inside the factored matrix instead of making a copy of it (which this function does)

   Developer Notes:
    The reason this routine exists is because the representation of the Schur complement within the factor matrix may be different than a standard PETSc
   matrix representation and we normally do not want to use the time or memory to make a copy as a regular PETSc matrix.

   See MatCreateSchurComplement() or MatGetSchurComplement() for ways to create virtual or approximate Schur complements.

   Level: advanced

   References:

.seealso: `MatGetFactor()`, `MatFactorSetSchurIS()`, `MatFactorGetSchurComplement()`, `MatFactorSchurStatus`
@*/
PetscErrorCode MatFactorCreateSchurComplement(Mat F,Mat* S,MatFactorSchurStatus* status)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  if (S) PetscValidPointer(S,2);
  if (status) PetscValidPointer(status,3);
  if (S) {
    PetscErrorCode (*f)(Mat,Mat*);

    PetscCall(PetscObjectQueryFunction((PetscObject)F,"MatFactorCreateSchurComplement_C",&f));
    if (f) {
      PetscCall((*f)(F,S));
    } else {
      PetscCall(MatDuplicate(F->schur,MAT_COPY_VALUES,S));
    }
  }
  if (status) *status = F->schur_status;
  PetscFunctionReturn(0);
}

/*@
  MatFactorGetSchurComplement - Gets access to a Schur complement matrix using the current Schur data within a factored matrix

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor()
.  *S - location where to return the Schur complement, can be NULL
-  status - the status of the Schur complement matrix, can be NULL

   Notes:
   You must call MatFactorSetSchurIS() before calling this routine.

   Schur complement mode is currently implemented for sequential matrices.
   The routine returns a the Schur Complement stored within the data strutures of the solver.
   If MatFactorInvertSchurComplement() has previously been called, the returned matrix is actually the inverse of the Schur complement.
   The returned matrix should not be destroyed; the caller should call MatFactorRestoreSchurComplement() when the object is no longer needed.

   Use MatFactorCreateSchurComplement() to create a copy of the Schur complement matrix that is within a factored matrix

   See MatCreateSchurComplement() or MatGetSchurComplement() for ways to create virtual or approximate Schur complements.

   Level: advanced

   References:

.seealso: `MatGetFactor()`, `MatFactorSetSchurIS()`, `MatFactorRestoreSchurComplement()`, `MatFactorCreateSchurComplement()`, `MatFactorSchurStatus`
@*/
PetscErrorCode MatFactorGetSchurComplement(Mat F,Mat* S,MatFactorSchurStatus* status)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  if (S) PetscValidPointer(S,2);
  if (status) PetscValidPointer(status,3);
  if (S) *S = F->schur;
  if (status) *status = F->schur_status;
  PetscFunctionReturn(0);
}

/*@
  MatFactorRestoreSchurComplement - Restore the Schur complement matrix object obtained from a call to MatFactorGetSchurComplement

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor()
.  *S - location where the Schur complement is stored
-  status - the status of the Schur complement matrix (see MatFactorSchurStatus)

   Notes:

   Level: advanced

   References:

.seealso: `MatGetFactor()`, `MatFactorSetSchurIS()`, `MatFactorRestoreSchurComplement()`, `MatFactorCreateSchurComplement()`, `MatFactorSchurStatus`
@*/
PetscErrorCode MatFactorRestoreSchurComplement(Mat F,Mat* S,MatFactorSchurStatus status)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  if (S) {
    PetscValidHeaderSpecific(*S,MAT_CLASSID,2);
    *S = NULL;
  }
  F->schur_status = status;
  PetscCall(MatFactorUpdateSchurStatus_Private(F));
  PetscFunctionReturn(0);
}

/*@
  MatFactorSolveSchurComplementTranspose - Solve the transpose of the Schur complement system computed during the factorization step

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor()
.  rhs - location where the right hand side of the Schur complement system is stored
-  sol - location where the solution of the Schur complement system has to be returned

   Notes:
   The sizes of the vectors should match the size of the Schur complement

   Must be called after MatFactorSetSchurIS()

   Level: advanced

   References:

.seealso: `MatGetFactor()`, `MatFactorSetSchurIS()`, `MatFactorSolveSchurComplement()`
@*/
PetscErrorCode MatFactorSolveSchurComplementTranspose(Mat F, Vec rhs, Vec sol)
{
  PetscFunctionBegin;
  PetscValidType(F,1);
  PetscValidType(rhs,2);
  PetscValidType(sol,3);
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidHeaderSpecific(rhs,VEC_CLASSID,2);
  PetscValidHeaderSpecific(sol,VEC_CLASSID,3);
  PetscCheckSameComm(F,1,rhs,2);
  PetscCheckSameComm(F,1,sol,3);
  PetscCall(MatFactorFactorizeSchurComplement(F));
  switch (F->schur_status) {
  case MAT_FACTOR_SCHUR_FACTORED:
    PetscCall(MatSolveTranspose(F->schur,rhs,sol));
    break;
  case MAT_FACTOR_SCHUR_INVERTED:
    PetscCall(MatMultTranspose(F->schur,rhs,sol));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_SUP,"Unhandled MatFactorSchurStatus %d",F->schur_status);
  }
  PetscFunctionReturn(0);
}

/*@
  MatFactorSolveSchurComplement - Solve the Schur complement system computed during the factorization step

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor()
.  rhs - location where the right hand side of the Schur complement system is stored
-  sol - location where the solution of the Schur complement system has to be returned

   Notes:
   The sizes of the vectors should match the size of the Schur complement

   Must be called after MatFactorSetSchurIS()

   Level: advanced

   References:

.seealso: `MatGetFactor()`, `MatFactorSetSchurIS()`, `MatFactorSolveSchurComplementTranspose()`
@*/
PetscErrorCode MatFactorSolveSchurComplement(Mat F, Vec rhs, Vec sol)
{
  PetscFunctionBegin;
  PetscValidType(F,1);
  PetscValidType(rhs,2);
  PetscValidType(sol,3);
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidHeaderSpecific(rhs,VEC_CLASSID,2);
  PetscValidHeaderSpecific(sol,VEC_CLASSID,3);
  PetscCheckSameComm(F,1,rhs,2);
  PetscCheckSameComm(F,1,sol,3);
  PetscCall(MatFactorFactorizeSchurComplement(F));
  switch (F->schur_status) {
  case MAT_FACTOR_SCHUR_FACTORED:
    PetscCall(MatSolve(F->schur,rhs,sol));
    break;
  case MAT_FACTOR_SCHUR_INVERTED:
    PetscCall(MatMult(F->schur,rhs,sol));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_SUP,"Unhandled MatFactorSchurStatus %d",F->schur_status);
  }
  PetscFunctionReturn(0);
}

/*@
  MatFactorInvertSchurComplement - Invert the Schur complement matrix computed during the factorization step

   Logically Collective on Mat

   Input Parameters:
.  F - the factored matrix obtained by calling MatGetFactor()

   Notes:
    Must be called after MatFactorSetSchurIS().

   Call MatFactorGetSchurComplement() or  MatFactorCreateSchurComplement() AFTER this call to actually compute the inverse and get access to it.

   Level: advanced

   References:

.seealso: `MatGetFactor()`, `MatFactorSetSchurIS()`, `MatFactorGetSchurComplement()`, `MatFactorCreateSchurComplement()`
@*/
PetscErrorCode MatFactorInvertSchurComplement(Mat F)
{
  PetscFunctionBegin;
  PetscValidType(F,1);
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  if (F->schur_status == MAT_FACTOR_SCHUR_INVERTED) PetscFunctionReturn(0);
  PetscCall(MatFactorFactorizeSchurComplement(F));
  PetscCall(MatFactorInvertSchurComplement_Private(F));
  F->schur_status = MAT_FACTOR_SCHUR_INVERTED;
  PetscFunctionReturn(0);
}

/*@
  MatFactorFactorizeSchurComplement - Factorize the Schur complement matrix computed during the factorization step

   Logically Collective on Mat

   Input Parameters:
.  F - the factored matrix obtained by calling MatGetFactor()

   Notes:
    Must be called after MatFactorSetSchurIS().

   Level: advanced

   References:

.seealso: `MatGetFactor()`, `MatFactorSetSchurIS()`, `MatFactorInvertSchurComplement()`
@*/
PetscErrorCode MatFactorFactorizeSchurComplement(Mat F)
{
  PetscFunctionBegin;
  PetscValidType(F,1);
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  if (F->schur_status == MAT_FACTOR_SCHUR_INVERTED || F->schur_status == MAT_FACTOR_SCHUR_FACTORED) PetscFunctionReturn(0);
  PetscCall(MatFactorFactorizeSchurComplement_Private(F));
  F->schur_status = MAT_FACTOR_SCHUR_FACTORED;
  PetscFunctionReturn(0);
}

/*@
   MatPtAP - Creates the matrix product C = P^T * A * P

   Neighbor-wise Collective on Mat

   Input Parameters:
+  A - the matrix
.  P - the projection matrix
.  scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
-  fill - expected fill as ratio of nnz(C)/(nnz(A) + nnz(P)), use PETSC_DEFAULT if you do not have a good estimate
          if the result is a dense matrix this is irrelevant

   Output Parameters:
.  C - the product matrix

   Notes:
   C will be created and must be destroyed by the user with MatDestroy().

   For matrix types without special implementation the function fallbacks to MatMatMult() followed by MatTransposeMatMult().

   Level: intermediate

.seealso: `MatMatMult()`, `MatRARt()`
@*/
PetscErrorCode MatPtAP(Mat A,Mat P,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscFunctionBegin;
  if (scall == MAT_REUSE_MATRIX) MatCheckProduct(*C,5);
  PetscCheck(scall != MAT_INPLACE_MATRIX,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Inplace product not supported");

  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(MatProductCreate(A,P,NULL,C));
    PetscCall(MatProductSetType(*C,MATPRODUCT_PtAP));
    PetscCall(MatProductSetAlgorithm(*C,"default"));
    PetscCall(MatProductSetFill(*C,fill));

    (*C)->product->api_user = PETSC_TRUE;
    PetscCall(MatProductSetFromOptions(*C));
    PetscCheck((*C)->ops->productsymbolic,PetscObjectComm((PetscObject)(*C)),PETSC_ERR_SUP,"MatProduct %s not supported for A %s and P %s",MatProductTypes[MATPRODUCT_PtAP],((PetscObject)A)->type_name,((PetscObject)P)->type_name);
    PetscCall(MatProductSymbolic(*C));
  } else { /* scall == MAT_REUSE_MATRIX */
    PetscCall(MatProductReplaceMats(A,P,NULL,*C));
  }

  PetscCall(MatProductNumeric(*C));
  (*C)->symmetric = A->symmetric;
  (*C)->spd       = A->spd;
  PetscFunctionReturn(0);
}

/*@
   MatRARt - Creates the matrix product C = R * A * R^T

   Neighbor-wise Collective on Mat

   Input Parameters:
+  A - the matrix
.  R - the projection matrix
.  scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
-  fill - expected fill as ratio of nnz(C)/nnz(A), use PETSC_DEFAULT if you do not have a good estimate
          if the result is a dense matrix this is irrelevant

   Output Parameters:
.  C - the product matrix

   Notes:
   C will be created and must be destroyed by the user with MatDestroy().

   This routine is currently only implemented for pairs of AIJ matrices and classes
   which inherit from AIJ. Due to PETSc sparse matrix block row distribution among processes,
   parallel MatRARt is implemented via explicit transpose of R, which could be very expensive.
   We recommend using MatPtAP().

   Level: intermediate

.seealso: `MatMatMult()`, `MatPtAP()`
@*/
PetscErrorCode MatRARt(Mat A,Mat R,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscFunctionBegin;
  if (scall == MAT_REUSE_MATRIX) MatCheckProduct(*C,5);
  PetscCheck(scall != MAT_INPLACE_MATRIX,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Inplace product not supported");

  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(MatProductCreate(A,R,NULL,C));
    PetscCall(MatProductSetType(*C,MATPRODUCT_RARt));
    PetscCall(MatProductSetAlgorithm(*C,"default"));
    PetscCall(MatProductSetFill(*C,fill));

    (*C)->product->api_user = PETSC_TRUE;
    PetscCall(MatProductSetFromOptions(*C));
    PetscCheck((*C)->ops->productsymbolic,PetscObjectComm((PetscObject)(*C)),PETSC_ERR_SUP,"MatProduct %s not supported for A %s and R %s",MatProductTypes[MATPRODUCT_RARt],((PetscObject)A)->type_name,((PetscObject)R)->type_name);
    PetscCall(MatProductSymbolic(*C));
  } else { /* scall == MAT_REUSE_MATRIX */
    PetscCall(MatProductReplaceMats(A,R,NULL,*C));
  }

  PetscCall(MatProductNumeric(*C));
  if (A->symmetric == PETSC_BOOL3_TRUE) {
    PetscCall(MatSetOption(*C,MAT_SYMMETRIC,PETSC_TRUE));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProduct_Private(Mat A,Mat B,MatReuse scall,PetscReal fill,MatProductType ptype, Mat *C)
{
  PetscFunctionBegin;
  PetscCheck(scall != MAT_INPLACE_MATRIX,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Inplace product not supported");

  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(PetscInfo(A,"Calling MatProduct API with MAT_INITIAL_MATRIX and product type %s\n",MatProductTypes[ptype]));
    PetscCall(MatProductCreate(A,B,NULL,C));
    PetscCall(MatProductSetType(*C,ptype));
    PetscCall(MatProductSetAlgorithm(*C,MATPRODUCTALGORITHMDEFAULT));
    PetscCall(MatProductSetFill(*C,fill));

    (*C)->product->api_user = PETSC_TRUE;
    PetscCall(MatProductSetFromOptions(*C));
    PetscCall(MatProductSymbolic(*C));
  } else { /* scall == MAT_REUSE_MATRIX */
    Mat_Product *product = (*C)->product;
    PetscBool isdense;

    PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)(*C),&isdense,MATSEQDENSE,MATMPIDENSE,""));
    if (isdense && product && product->type != ptype) {
      PetscCall(MatProductClear(*C));
      product = NULL;
    }
    PetscCall(PetscInfo(A,"Calling MatProduct API with MAT_REUSE_MATRIX %s product present and product type %s\n",product ? "with" : "without",MatProductTypes[ptype]));
    if (!product) { /* user provide the dense matrix *C without calling MatProductCreate() or reusing it from previous calls */
      if (isdense) {
        PetscCall(MatProductCreate_Private(A,B,NULL,*C));
        product = (*C)->product;
        product->fill     = fill;
        product->api_user = PETSC_TRUE;
        product->clear    = PETSC_TRUE;

        PetscCall(MatProductSetType(*C,ptype));
        PetscCall(MatProductSetFromOptions(*C));
        PetscCheck((*C)->ops->productsymbolic,PetscObjectComm((PetscObject)(*C)),PETSC_ERR_SUP,"MatProduct %s not supported for %s and %s",MatProductTypes[ptype],((PetscObject)A)->type_name,((PetscObject)B)->type_name);
        PetscCall(MatProductSymbolic(*C));
      } else SETERRQ(PetscObjectComm((PetscObject)(*C)),PETSC_ERR_SUP,"Call MatProductCreate() first");
    } else { /* user may change input matrices A or B when REUSE */
      PetscCall(MatProductReplaceMats(A,B,NULL,*C));
    }
  }
  PetscCall(MatProductNumeric(*C));
  PetscFunctionReturn(0);
}

/*@
   MatMatMult - Performs Matrix-Matrix Multiplication C=A*B.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  A - the left matrix
.  B - the right matrix
.  scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
-  fill - expected fill as ratio of nnz(C)/(nnz(A) + nnz(B)), use PETSC_DEFAULT if you do not have a good estimate
          if the result is a dense matrix this is irrelevant

   Output Parameters:
.  C - the product matrix

   Notes:
   Unless scall is MAT_REUSE_MATRIX C will be created.

   MAT_REUSE_MATRIX can only be used if the matrices A and B have the same nonzero pattern as in the previous call and C was obtained from a previous
   call to this function with MAT_INITIAL_MATRIX.

   To determine the correct fill value, run with -info and search for the string "Fill ratio" to see the value actually needed.

   If you have many matrices with the same non-zero structure to multiply, you should use MatProductCreate()/MatProductSymbolic()/MatProductReplaceMats(), and call MatProductNumeric() repeatedly.

   In the special case where matrix B (and hence C) are dense you can create the correctly sized matrix C yourself and then call this routine with MAT_REUSE_MATRIX, rather than first having MatMatMult() create it for you. You can NEVER do this if the matrix C is sparse.

   Example of Usage:
.vb
     MatProductCreate(A,B,NULL,&C);
     MatProductSetType(C,MATPRODUCT_AB);
     MatProductSymbolic(C);
     MatProductNumeric(C); // compute C=A * B
     MatProductReplaceMats(A1,B1,NULL,C); // compute C=A1 * B1
     MatProductNumeric(C);
     MatProductReplaceMats(A2,NULL,NULL,C); // compute C=A2 * B1
     MatProductNumeric(C);
.ve

   Level: intermediate

.seealso: `MatTransposeMatMult()`, `MatMatTransposeMult()`, `MatPtAP()`, `MatProductCreate()`, `MatProductSymbolic()`, `MatProductReplaceMats()`, `MatProductNumeric()`
@*/
PetscErrorCode MatMatMult(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscFunctionBegin;
  PetscCall(MatProduct_Private(A,B,scall,fill,MATPRODUCT_AB,C));
  PetscFunctionReturn(0);
}

/*@
   MatMatTransposeMult - Performs Matrix-Matrix Multiplication C=A*B^T.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  A - the left matrix
.  B - the right matrix
.  scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
-  fill - expected fill as ratio of nnz(C)/(nnz(A) + nnz(B)), use PETSC_DEFAULT if not known

   Output Parameters:
.  C - the product matrix

   Notes:
   C will be created if MAT_INITIAL_MATRIX and must be destroyed by the user with MatDestroy().

   MAT_REUSE_MATRIX can only be used if the matrices A and B have the same nonzero pattern as in the previous call

  To determine the correct fill value, run with -info and search for the string "Fill ratio" to see the value
   actually needed.

   This routine is currently only implemented for pairs of SeqAIJ matrices, for the SeqDense class,
   and for pairs of MPIDense matrices.

   Options Database Keys:
.  -matmattransmult_mpidense_mpidense_via {allgatherv,cyclic} - Choose between algorithms for MPIDense matrices: the
              first redundantly copies the transposed B matrix on each process and requiers O(log P) communication complexity;
              the second never stores more than one portion of the B matrix at a time by requires O(P) communication complexity.

   Level: intermediate

.seealso: `MatMatMult()`, `MatTransposeMatMult()` `MatPtAP()`
@*/
PetscErrorCode MatMatTransposeMult(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscFunctionBegin;
  PetscCall(MatProduct_Private(A,B,scall,fill,MATPRODUCT_ABt,C));
  if (A == B) {
    PetscCall(MatSetOption(*C,MAT_SYMMETRIC,PETSC_TRUE));
  }
  PetscFunctionReturn(0);
}

/*@
   MatTransposeMatMult - Performs Matrix-Matrix Multiplication C=A^T*B.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  A - the left matrix
.  B - the right matrix
.  scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
-  fill - expected fill as ratio of nnz(C)/(nnz(A) + nnz(B)), use PETSC_DEFAULT if not known

   Output Parameters:
.  C - the product matrix

   Notes:
   C will be created if MAT_INITIAL_MATRIX and must be destroyed by the user with MatDestroy().

   MAT_REUSE_MATRIX can only be used if the matrices A and B have the same nonzero pattern as in the previous call.

  To determine the correct fill value, run with -info and search for the string "Fill ratio" to see the value
   actually needed.

   This routine is currently implemented for pairs of AIJ matrices and pairs of SeqDense matrices and classes
   which inherit from SeqAIJ.  C will be of the same type as the input matrices.

   Level: intermediate

.seealso: `MatMatMult()`, `MatMatTransposeMult()`, `MatPtAP()`
@*/
PetscErrorCode MatTransposeMatMult(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscFunctionBegin;
  PetscCall(MatProduct_Private(A,B,scall,fill,MATPRODUCT_AtB,C));
  PetscFunctionReturn(0);
}

/*@
   MatMatMatMult - Performs Matrix-Matrix-Matrix Multiplication D=A*B*C.

   Neighbor-wise Collective on Mat

   Input Parameters:
+  A - the left matrix
.  B - the middle matrix
.  C - the right matrix
.  scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
-  fill - expected fill as ratio of nnz(D)/(nnz(A) + nnz(B)+nnz(C)), use PETSC_DEFAULT if you do not have a good estimate
          if the result is a dense matrix this is irrelevant

   Output Parameters:
.  D - the product matrix

   Notes:
   Unless scall is MAT_REUSE_MATRIX D will be created.

   MAT_REUSE_MATRIX can only be used if the matrices A, B and C have the same nonzero pattern as in the previous call

   To determine the correct fill value, run with -info and search for the string "Fill ratio" to see the value
   actually needed.

   If you have many matrices with the same non-zero structure to multiply, you
   should use MAT_REUSE_MATRIX in all calls but the first

   Level: intermediate

.seealso: `MatMatMult`, `MatPtAP()`, `MatMatTransposeMult()`, `MatTransposeMatMult()`
@*/
PetscErrorCode MatMatMatMult(Mat A,Mat B,Mat C,MatReuse scall,PetscReal fill,Mat *D)
{
  PetscFunctionBegin;
  if (scall == MAT_REUSE_MATRIX) MatCheckProduct(*D,6);
  PetscCheck(scall != MAT_INPLACE_MATRIX,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Inplace product not supported");

  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(MatProductCreate(A,B,C,D));
    PetscCall(MatProductSetType(*D,MATPRODUCT_ABC));
    PetscCall(MatProductSetAlgorithm(*D,"default"));
    PetscCall(MatProductSetFill(*D,fill));

    (*D)->product->api_user = PETSC_TRUE;
    PetscCall(MatProductSetFromOptions(*D));
    PetscCheck((*D)->ops->productsymbolic,PetscObjectComm((PetscObject)(*D)),PETSC_ERR_SUP,"MatProduct %s not supported for A %s, B %s and C %s",MatProductTypes[MATPRODUCT_ABC],((PetscObject)A)->type_name,((PetscObject)B)->type_name,((PetscObject)C)->type_name);
    PetscCall(MatProductSymbolic(*D));
  } else { /* user may change input matrices when REUSE */
    PetscCall(MatProductReplaceMats(A,B,C,*D));
  }
  PetscCall(MatProductNumeric(*D));
  PetscFunctionReturn(0);
}

/*@
   MatCreateRedundantMatrix - Create redundant matrices and put them into processors of subcommunicators.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  nsubcomm - the number of subcommunicators (= number of redundant parallel or sequential matrices)
.  subcomm - MPI communicator split from the communicator where mat resides in (or MPI_COMM_NULL if nsubcomm is used)
-  reuse - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

   Output Parameter:
.  matredundant - redundant matrix

   Notes:
   MAT_REUSE_MATRIX can only be used when the nonzero structure of the
   original matrix has not changed from that last call to MatCreateRedundantMatrix().

   This routine creates the duplicated matrices in the subcommunicators; you should NOT create them before
   calling it.

   Level: advanced

.seealso: `MatDestroy()`
@*/
PetscErrorCode MatCreateRedundantMatrix(Mat mat,PetscInt nsubcomm,MPI_Comm subcomm,MatReuse reuse,Mat *matredundant)
{
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscInt       mloc_sub,nloc_sub,rstart,rend,M=mat->rmap->N,N=mat->cmap->N,bs=mat->rmap->bs;
  Mat_Redundant  *redund=NULL;
  PetscSubcomm   psubcomm=NULL;
  MPI_Comm       subcomm_in=subcomm;
  Mat            *matseq;
  IS             isrow,iscol;
  PetscBool      newsubcomm=PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (nsubcomm && reuse == MAT_REUSE_MATRIX) {
    PetscValidPointer(*matredundant,5);
    PetscValidHeaderSpecific(*matredundant,MAT_CLASSID,5);
  }

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size));
  if (size == 1 || nsubcomm == 1) {
    if (reuse == MAT_INITIAL_MATRIX) {
      PetscCall(MatDuplicate(mat,MAT_COPY_VALUES,matredundant));
    } else {
      PetscCheck(*matredundant != mat,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"MAT_REUSE_MATRIX means reuse the matrix passed in as the final argument, not the original matrix");
      PetscCall(MatCopy(mat,*matredundant,SAME_NONZERO_PATTERN));
    }
    PetscFunctionReturn(0);
  }

  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);

  PetscCall(PetscLogEventBegin(MAT_RedundantMat,mat,0,0,0));
  if (subcomm_in == MPI_COMM_NULL && reuse == MAT_INITIAL_MATRIX) { /* get subcomm if user does not provide subcomm */
    /* create psubcomm, then get subcomm */
    PetscCall(PetscObjectGetComm((PetscObject)mat,&comm));
    PetscCallMPI(MPI_Comm_size(comm,&size));
    PetscCheck(nsubcomm >= 1 && nsubcomm <= size,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"nsubcomm must between 1 and %d",size);

    PetscCall(PetscSubcommCreate(comm,&psubcomm));
    PetscCall(PetscSubcommSetNumber(psubcomm,nsubcomm));
    PetscCall(PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_CONTIGUOUS));
    PetscCall(PetscSubcommSetFromOptions(psubcomm));
    PetscCall(PetscCommDuplicate(PetscSubcommChild(psubcomm),&subcomm,NULL));
    newsubcomm = PETSC_TRUE;
    PetscCall(PetscSubcommDestroy(&psubcomm));
  }

  /* get isrow, iscol and a local sequential matrix matseq[0] */
  if (reuse == MAT_INITIAL_MATRIX) {
    mloc_sub = PETSC_DECIDE;
    nloc_sub = PETSC_DECIDE;
    if (bs < 1) {
      PetscCall(PetscSplitOwnership(subcomm,&mloc_sub,&M));
      PetscCall(PetscSplitOwnership(subcomm,&nloc_sub,&N));
    } else {
      PetscCall(PetscSplitOwnershipBlock(subcomm,bs,&mloc_sub,&M));
      PetscCall(PetscSplitOwnershipBlock(subcomm,bs,&nloc_sub,&N));
    }
    PetscCallMPI(MPI_Scan(&mloc_sub,&rend,1,MPIU_INT,MPI_SUM,subcomm));
    rstart = rend - mloc_sub;
    PetscCall(ISCreateStride(PETSC_COMM_SELF,mloc_sub,rstart,1,&isrow));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,N,0,1,&iscol));
  } else { /* reuse == MAT_REUSE_MATRIX */
    PetscCheck(*matredundant != mat,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"MAT_REUSE_MATRIX means reuse the matrix passed in as the final argument, not the original matrix");
    /* retrieve subcomm */
    PetscCall(PetscObjectGetComm((PetscObject)(*matredundant),&subcomm));
    redund = (*matredundant)->redundant;
    isrow  = redund->isrow;
    iscol  = redund->iscol;
    matseq = redund->matseq;
  }
  PetscCall(MatCreateSubMatrices(mat,1,&isrow,&iscol,reuse,&matseq));

  /* get matredundant over subcomm */
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatCreateMPIMatConcatenateSeqMat(subcomm,matseq[0],nloc_sub,reuse,matredundant));

    /* create a supporting struct and attach it to C for reuse */
    PetscCall(PetscNewLog(*matredundant,&redund));
    (*matredundant)->redundant = redund;
    redund->isrow              = isrow;
    redund->iscol              = iscol;
    redund->matseq             = matseq;
    if (newsubcomm) {
      redund->subcomm          = subcomm;
    } else {
      redund->subcomm          = MPI_COMM_NULL;
    }
  } else {
    PetscCall(MatCreateMPIMatConcatenateSeqMat(subcomm,matseq[0],PETSC_DECIDE,reuse,matredundant));
  }
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
  if (matseq[0]->boundtocpu && matseq[0]->bindingpropagates) {
    PetscCall(MatBindToCPU(*matredundant,PETSC_TRUE));
    PetscCall(MatSetBindingPropagates(*matredundant,PETSC_TRUE));
  }
#endif
  PetscCall(PetscLogEventEnd(MAT_RedundantMat,mat,0,0,0));
  PetscFunctionReturn(0);
}

/*@C
   MatGetMultiProcBlock - Create multiple [bjacobi] 'parallel submatrices' from
   a given 'mat' object. Each submatrix can span multiple procs.

   Collective on Mat

   Input Parameters:
+  mat - the matrix
.  subcomm - the subcommunicator obtained by com_split(comm)
-  scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

   Output Parameter:
.  subMat - 'parallel submatrices each spans a given subcomm

  Notes:
  The submatrix partition across processors is dictated by 'subComm' a
  communicator obtained by MPI_comm_split(). The subComm
  is not restriced to be grouped with consecutive original ranks.

  Due the MPI_Comm_split() usage, the parallel layout of the submatrices
  map directly to the layout of the original matrix [wrt the local
  row,col partitioning]. So the original 'DiagonalMat' naturally maps
  into the 'DiagonalMat' of the subMat, hence it is used directly from
  the subMat. However the offDiagMat looses some columns - and this is
  reconstructed with MatSetValues()

  Level: advanced

.seealso: `MatCreateSubMatrices()`
@*/
PetscErrorCode   MatGetMultiProcBlock(Mat mat, MPI_Comm subComm, MatReuse scall,Mat *subMat)
{
  PetscMPIInt    commsize,subCommSize;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&commsize));
  PetscCallMPI(MPI_Comm_size(subComm,&subCommSize));
  PetscCheck(subCommSize <= commsize,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_OUTOFRANGE,"CommSize %d < SubCommZize %d",commsize,subCommSize);

  PetscCheck(scall != MAT_REUSE_MATRIX || *subMat != mat,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"MAT_REUSE_MATRIX means reuse the matrix passed in as the final argument, not the original matrix");
  PetscCall(PetscLogEventBegin(MAT_GetMultiProcBlock,mat,0,0,0));
  PetscCall((*mat->ops->getmultiprocblock)(mat,subComm,scall,subMat));
  PetscCall(PetscLogEventEnd(MAT_GetMultiProcBlock,mat,0,0,0));
  PetscFunctionReturn(0);
}

/*@
   MatGetLocalSubMatrix - Gets a reference to a submatrix specified in local numbering

   Not Collective

   Input Parameters:
+  mat - matrix to extract local submatrix from
.  isrow - local row indices for submatrix
-  iscol - local column indices for submatrix

   Output Parameter:
.  submat - the submatrix

   Level: intermediate

   Notes:
   The submat should be returned with MatRestoreLocalSubMatrix().

   Depending on the format of mat, the returned submat may not implement MatMult().  Its communicator may be
   the same as mat, it may be PETSC_COMM_SELF, or some other subcomm of mat's.

   The submat always implements MatSetValuesLocal().  If isrow and iscol have the same block size, then
   MatSetValuesBlockedLocal() will also be implemented.

   The mat must have had a ISLocalToGlobalMapping provided to it with MatSetLocalToGlobalMapping(). Note that
   matrices obtained with DMCreateMatrix() generally already have the local to global mapping provided.

.seealso: `MatRestoreLocalSubMatrix()`, `MatCreateLocalRef()`, `MatSetLocalToGlobalMapping()`
@*/
PetscErrorCode MatGetLocalSubMatrix(Mat mat,IS isrow,IS iscol,Mat *submat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(isrow,IS_CLASSID,2);
  PetscValidHeaderSpecific(iscol,IS_CLASSID,3);
  PetscCheckSameComm(isrow,2,iscol,3);
  PetscValidPointer(submat,4);
  PetscCheck(mat->rmap->mapping,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Matrix must have local to global mapping provided before this call");

  if (mat->ops->getlocalsubmatrix) {
    PetscCall((*mat->ops->getlocalsubmatrix)(mat,isrow,iscol,submat));
  } else {
    PetscCall(MatCreateLocalRef(mat,isrow,iscol,submat));
  }
  PetscFunctionReturn(0);
}

/*@
   MatRestoreLocalSubMatrix - Restores a reference to a submatrix specified in local numbering

   Not Collective

   Input Parameters:
+  mat - matrix to extract local submatrix from
.  isrow - local row indices for submatrix
.  iscol - local column indices for submatrix
-  submat - the submatrix

   Level: intermediate

.seealso: `MatGetLocalSubMatrix()`
@*/
PetscErrorCode MatRestoreLocalSubMatrix(Mat mat,IS isrow,IS iscol,Mat *submat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(isrow,IS_CLASSID,2);
  PetscValidHeaderSpecific(iscol,IS_CLASSID,3);
  PetscCheckSameComm(isrow,2,iscol,3);
  PetscValidPointer(submat,4);
  if (*submat) {
    PetscValidHeaderSpecific(*submat,MAT_CLASSID,4);
  }

  if (mat->ops->restorelocalsubmatrix) {
    PetscCall((*mat->ops->restorelocalsubmatrix)(mat,isrow,iscol,submat));
  } else {
    PetscCall(MatDestroy(submat));
  }
  *submat = NULL;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------*/
/*@
   MatFindZeroDiagonals - Finds all the rows of a matrix that have zero or no diagonal entry in the matrix

   Collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  is - if any rows have zero diagonals this contains the list of them

   Level: developer

.seealso: `MatMultTranspose()`, `MatMultAdd()`, `MatMultTransposeAdd()`
@*/
PetscErrorCode MatFindZeroDiagonals(Mat mat,IS *is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  if (!mat->ops->findzerodiagonals) {
    Vec                diag;
    const PetscScalar *a;
    PetscInt          *rows;
    PetscInt           rStart, rEnd, r, nrow = 0;

    PetscCall(MatCreateVecs(mat, &diag, NULL));
    PetscCall(MatGetDiagonal(mat, diag));
    PetscCall(MatGetOwnershipRange(mat, &rStart, &rEnd));
    PetscCall(VecGetArrayRead(diag, &a));
    for (r = 0; r < rEnd-rStart; ++r) if (a[r] == 0.0) ++nrow;
    PetscCall(PetscMalloc1(nrow, &rows));
    nrow = 0;
    for (r = 0; r < rEnd-rStart; ++r) if (a[r] == 0.0) rows[nrow++] = r+rStart;
    PetscCall(VecRestoreArrayRead(diag, &a));
    PetscCall(VecDestroy(&diag));
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject) mat), nrow, rows, PETSC_OWN_POINTER, is));
  } else {
    PetscCall((*mat->ops->findzerodiagonals)(mat, is));
  }
  PetscFunctionReturn(0);
}

/*@
   MatFindOffBlockDiagonalEntries - Finds all the rows of a matrix that have entries outside of the main diagonal block (defined by the matrix block size)

   Collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  is - contains the list of rows with off block diagonal entries

   Level: developer

.seealso: `MatMultTranspose()`, `MatMultAdd()`, `MatMultTransposeAdd()`
@*/
PetscErrorCode MatFindOffBlockDiagonalEntries(Mat mat,IS *is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscCheck(mat->assembled,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  PetscCheck(mat->ops->findoffblockdiagonalentries,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Matrix type %s does not have a find off block diagonal entries defined",((PetscObject)mat)->type_name);
  PetscCall((*mat->ops->findoffblockdiagonalentries)(mat,is));
  PetscFunctionReturn(0);
}

/*@C
  MatInvertBlockDiagonal - Inverts the block diagonal entries.

  Collective on Mat

  Input Parameters:
. mat - the matrix

  Output Parameters:
. values - the block inverses in column major order (FORTRAN-like)

   Note:
     The size of the blocks is determined by the block size of the matrix.

   Fortran Note:
     This routine is not available from Fortran.

  Level: advanced

.seealso: `MatInvertVariableBlockEnvelope()`, `MatInvertBlockDiagonalMat()`
@*/
PetscErrorCode MatInvertBlockDiagonal(Mat mat,const PetscScalar **values)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscCheck(mat->assembled,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->ops->invertblockdiagonal,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not supported for type %s",((PetscObject)mat)->type_name);
  PetscCall((*mat->ops->invertblockdiagonal)(mat,values));
  PetscFunctionReturn(0);
}

/*@C
  MatInvertVariableBlockDiagonal - Inverts the point block diagonal entries.

  Collective on Mat

  Input Parameters:
+ mat - the matrix
. nblocks - the number of blocks on the process, set with MatSetVariableBlockSizes()
- bsizes - the size of each block on the process, set with MatSetVariableBlockSizes()

  Output Parameters:
. values - the block inverses in column major order (FORTRAN-like)

   Note:
   This routine is not available from Fortran.

  Level: advanced

.seealso: `MatInvertBlockDiagonal()`, `MatSetVariableBlockSizes()`, `MatInvertVariableBlockEnvelope()`
@*/
PetscErrorCode MatInvertVariableBlockDiagonal(Mat mat,PetscInt nblocks,const PetscInt *bsizes,PetscScalar *values)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscCheck(mat->assembled,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCheck(!mat->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCheck(mat->ops->invertvariableblockdiagonal,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not supported for type %s",((PetscObject)mat)->type_name);
  PetscCall((*mat->ops->invertvariableblockdiagonal)(mat,nblocks,bsizes,values));
  PetscFunctionReturn(0);
}

/*@
  MatInvertBlockDiagonalMat - set matrix C to be the inverted block diagonal of matrix A

  Collective on Mat

  Input Parameters:
. A - the matrix

  Output Parameters:
. C - matrix with inverted block diagonal of A.  This matrix should be created and may have its type set.

  Notes: the blocksize of the matrix is used to determine the blocks on the diagonal of C

  Level: advanced

.seealso: `MatInvertBlockDiagonal()`
@*/
PetscErrorCode MatInvertBlockDiagonalMat(Mat A,Mat C)
{
  const PetscScalar *vals;
  PetscInt          *dnnz;
  PetscInt           m,rstart,rend,bs,i,j;

  PetscFunctionBegin;
  PetscCall(MatInvertBlockDiagonal(A,&vals));
  PetscCall(MatGetBlockSize(A,&bs));
  PetscCall(MatGetLocalSize(A,&m,NULL));
  PetscCall(MatSetLayouts(C,A->rmap,A->cmap));
  PetscCall(PetscMalloc1(m/bs,&dnnz));
  for (j = 0; j < m/bs; j++) dnnz[j] = 1;
  PetscCall(MatXAIJSetPreallocation(C,bs,dnnz,NULL,NULL,NULL));
  PetscCall(PetscFree(dnnz));
  PetscCall(MatGetOwnershipRange(C,&rstart,&rend));
  PetscCall(MatSetOption(C,MAT_ROW_ORIENTED,PETSC_FALSE));
  for (i = rstart/bs; i < rend/bs; i++) {
    PetscCall(MatSetValuesBlocked(C,1,&i,1,&i,&vals[(i-rstart/bs)*bs*bs],INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(C,MAT_ROW_ORIENTED,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*@C
    MatTransposeColoringDestroy - Destroys a coloring context for matrix product C=A*B^T that was created
    via MatTransposeColoringCreate().

    Collective on MatTransposeColoring

    Input Parameter:
.   c - coloring context

    Level: intermediate

.seealso: `MatTransposeColoringCreate()`
@*/
PetscErrorCode MatTransposeColoringDestroy(MatTransposeColoring *c)
{
  MatTransposeColoring matcolor=*c;

  PetscFunctionBegin;
  if (!matcolor) PetscFunctionReturn(0);
  if (--((PetscObject)matcolor)->refct > 0) {matcolor = NULL; PetscFunctionReturn(0);}

  PetscCall(PetscFree3(matcolor->ncolumns,matcolor->nrows,matcolor->colorforrow));
  PetscCall(PetscFree(matcolor->rows));
  PetscCall(PetscFree(matcolor->den2sp));
  PetscCall(PetscFree(matcolor->colorforcol));
  PetscCall(PetscFree(matcolor->columns));
  if (matcolor->brows>0) PetscCall(PetscFree(matcolor->lstart));
  PetscCall(PetscHeaderDestroy(c));
  PetscFunctionReturn(0);
}

/*@C
    MatTransColoringApplySpToDen - Given a symbolic matrix product C=A*B^T for which
    a MatTransposeColoring context has been created, computes a dense B^T by Apply
    MatTransposeColoring to sparse B.

    Collective on MatTransposeColoring

    Input Parameters:
+   B - sparse matrix B
.   Btdense - symbolic dense matrix B^T
-   coloring - coloring context created with MatTransposeColoringCreate()

    Output Parameter:
.   Btdense - dense matrix B^T

    Level: advanced

     Notes:
    These are used internally for some implementations of MatRARt()

.seealso: `MatTransposeColoringCreate()`, `MatTransposeColoringDestroy()`, `MatTransColoringApplyDenToSp()`

@*/
PetscErrorCode MatTransColoringApplySpToDen(MatTransposeColoring coloring,Mat B,Mat Btdense)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidHeaderSpecific(Btdense,MAT_CLASSID,3);
  PetscValidHeaderSpecific(coloring,MAT_TRANSPOSECOLORING_CLASSID,1);

  PetscCheck(B->ops->transcoloringapplysptoden,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not supported for this matrix type %s",((PetscObject)B)->type_name);
  PetscCall((B->ops->transcoloringapplysptoden)(coloring,B,Btdense));
  PetscFunctionReturn(0);
}

/*@C
    MatTransColoringApplyDenToSp - Given a symbolic matrix product Csp=A*B^T for which
    a MatTransposeColoring context has been created and a dense matrix Cden=A*Btdense
    in which Btdens is obtained from MatTransColoringApplySpToDen(), recover sparse matrix
    Csp from Cden.

    Collective on MatTransposeColoring

    Input Parameters:
+   coloring - coloring context created with MatTransposeColoringCreate()
-   Cden - matrix product of a sparse matrix and a dense matrix Btdense

    Output Parameter:
.   Csp - sparse matrix

    Level: advanced

     Notes:
    These are used internally for some implementations of MatRARt()

.seealso: `MatTransposeColoringCreate()`, `MatTransposeColoringDestroy()`, `MatTransColoringApplySpToDen()`

@*/
PetscErrorCode MatTransColoringApplyDenToSp(MatTransposeColoring matcoloring,Mat Cden,Mat Csp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(matcoloring,MAT_TRANSPOSECOLORING_CLASSID,1);
  PetscValidHeaderSpecific(Cden,MAT_CLASSID,2);
  PetscValidHeaderSpecific(Csp,MAT_CLASSID,3);

  PetscCheck(Csp->ops->transcoloringapplydentosp,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not supported for this matrix type %s",((PetscObject)Csp)->type_name);
  PetscCall((Csp->ops->transcoloringapplydentosp)(matcoloring,Cden,Csp));
  PetscCall(MatAssemblyBegin(Csp,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Csp,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*@C
   MatTransposeColoringCreate - Creates a matrix coloring context for matrix product C=A*B^T.

   Collective on Mat

   Input Parameters:
+  mat - the matrix product C
-  iscoloring - the coloring of the matrix; usually obtained with MatColoringCreate() or DMCreateColoring()

    Output Parameter:
.   color - the new coloring context

    Level: intermediate

.seealso: `MatTransposeColoringDestroy()`, `MatTransColoringApplySpToDen()`,
          `MatTransColoringApplyDenToSp()`
@*/
PetscErrorCode MatTransposeColoringCreate(Mat mat,ISColoring iscoloring,MatTransposeColoring *color)
{
  MatTransposeColoring c;
  MPI_Comm             comm;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(MAT_TransposeColoringCreate,mat,0,0,0));
  PetscCall(PetscObjectGetComm((PetscObject)mat,&comm));
  PetscCall(PetscHeaderCreate(c,MAT_TRANSPOSECOLORING_CLASSID,"MatTransposeColoring","Matrix product C=A*B^T via coloring","Mat",comm,MatTransposeColoringDestroy,NULL));

  c->ctype = iscoloring->ctype;
  if (mat->ops->transposecoloringcreate) {
    PetscCall((*mat->ops->transposecoloringcreate)(mat,iscoloring,c));
  } else SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Code not yet written for matrix type %s",((PetscObject)mat)->type_name);

  *color = c;
  PetscCall(PetscLogEventEnd(MAT_TransposeColoringCreate,mat,0,0,0));
  PetscFunctionReturn(0);
}

/*@
      MatGetNonzeroState - Returns a 64 bit integer representing the current state of nonzeros in the matrix. If the
        matrix has had no new nonzero locations added to the matrix since the previous call then the value will be the
        same, otherwise it will be larger

     Not Collective

  Input Parameter:
.    A  - the matrix

  Output Parameter:
.    state - the current state

  Notes:
    You can only compare states from two different calls to the SAME matrix, you cannot compare calls between
         different matrices

  Level: intermediate

.seealso: `PetscObjectStateGet()`
@*/
PetscErrorCode MatGetNonzeroState(Mat mat,PetscObjectState *state)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  *state = mat->nonzerostate;
  PetscFunctionReturn(0);
}

/*@
      MatCreateMPIMatConcatenateSeqMat - Creates a single large PETSc matrix by concatenating sequential
                 matrices from each processor

    Collective

   Input Parameters:
+    comm - the communicators the parallel matrix will live on
.    seqmat - the input sequential matrices
.    n - number of local columns (or PETSC_DECIDE)
-    reuse - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

   Output Parameter:
.    mpimat - the parallel matrix generated

    Level: advanced

   Notes:
    The number of columns of the matrix in EACH processor MUST be the same.

@*/
PetscErrorCode MatCreateMPIMatConcatenateSeqMat(MPI_Comm comm,Mat seqmat,PetscInt n,MatReuse reuse,Mat *mpimat)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm,&size));
  if (size == 1) {
    if (reuse == MAT_INITIAL_MATRIX) {
      PetscCall(MatDuplicate(seqmat,MAT_COPY_VALUES,mpimat));
    } else {
      PetscCall(MatCopy(seqmat,*mpimat,SAME_NONZERO_PATTERN));
    }
    PetscFunctionReturn(0);
  }

  PetscCheck(seqmat->ops->creatempimatconcatenateseqmat,PetscObjectComm((PetscObject)seqmat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)seqmat)->type_name);
  PetscCheck(reuse != MAT_REUSE_MATRIX || seqmat != *mpimat,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"MAT_REUSE_MATRIX means reuse the matrix passed in as the final argument, not the original matrix");

  PetscCall(PetscLogEventBegin(MAT_Merge,seqmat,0,0,0));
  PetscCall((*seqmat->ops->creatempimatconcatenateseqmat)(comm,seqmat,n,reuse,mpimat));
  PetscCall(PetscLogEventEnd(MAT_Merge,seqmat,0,0,0));
  PetscFunctionReturn(0);
}

/*@
     MatSubdomainsCreateCoalesce - Creates index subdomains by coalescing adjacent
                 ranks' ownership ranges.

    Collective on A

   Input Parameters:
+    A   - the matrix to create subdomains from
-    N   - requested number of subdomains

   Output Parameters:
+    n   - number of subdomains resulting on this rank
-    iss - IS list with indices of subdomains on this rank

    Level: advanced

    Notes:
    number of subdomains must be smaller than the communicator size
@*/
PetscErrorCode MatSubdomainsCreateCoalesce(Mat A,PetscInt N,PetscInt *n,IS *iss[])
{
  MPI_Comm        comm,subcomm;
  PetscMPIInt     size,rank,color;
  PetscInt        rstart,rend,k;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCheck(N >= 1 && N < (PetscInt)size,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"number of subdomains must be > 0 and < %d, got N = %" PetscInt_FMT,size,N);
  *n = 1;
  k = ((PetscInt)size)/N + ((PetscInt)size%N>0); /* There are up to k ranks to a color */
  color = rank/k;
  PetscCallMPI(MPI_Comm_split(comm,color,rank,&subcomm));
  PetscCall(PetscMalloc1(1,iss));
  PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
  PetscCall(ISCreateStride(subcomm,rend-rstart,rstart,1,iss[0]));
  PetscCallMPI(MPI_Comm_free(&subcomm));
  PetscFunctionReturn(0);
}

/*@
   MatGalerkin - Constructs the coarse grid problem via Galerkin projection.

   If the interpolation and restriction operators are the same, uses MatPtAP.
   If they are not the same, use MatMatMatMult.

   Once the coarse grid problem is constructed, correct for interpolation operators
   that are not of full rank, which can legitimately happen in the case of non-nested
   geometric multigrid.

   Input Parameters:
+  restrct - restriction operator
.  dA - fine grid matrix
.  interpolate - interpolation operator
.  reuse - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
-  fill - expected fill, use PETSC_DEFAULT if you do not have a good estimate

   Output Parameters:
.  A - the Galerkin coarse matrix

   Options Database Key:
.  -pc_mg_galerkin <both,pmat,mat,none> - for what matrices the Galerkin process should be used

   Level: developer

.seealso: `MatPtAP()`, `MatMatMatMult()`
@*/
PetscErrorCode  MatGalerkin(Mat restrct, Mat dA, Mat interpolate, MatReuse reuse, PetscReal fill, Mat *A)
{
  IS             zerorows;
  Vec            diag;

  PetscFunctionBegin;
  PetscCheck(reuse != MAT_INPLACE_MATRIX,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Inplace product not supported");
  /* Construct the coarse grid matrix */
  if (interpolate == restrct) {
    PetscCall(MatPtAP(dA,interpolate,reuse,fill,A));
  } else {
    PetscCall(MatMatMatMult(restrct,dA,interpolate,reuse,fill,A));
  }

  /* If the interpolation matrix is not of full rank, A will have zero rows.
     This can legitimately happen in the case of non-nested geometric multigrid.
     In that event, we set the rows of the matrix to the rows of the identity,
     ignoring the equations (as the RHS will also be zero). */

  PetscCall(MatFindZeroRows(*A, &zerorows));

  if (zerorows != NULL) { /* if there are any zero rows */
    PetscCall(MatCreateVecs(*A, &diag, NULL));
    PetscCall(MatGetDiagonal(*A, diag));
    PetscCall(VecISSet(diag, zerorows, 1.0));
    PetscCall(MatDiagonalSet(*A, diag, INSERT_VALUES));
    PetscCall(VecDestroy(&diag));
    PetscCall(ISDestroy(&zerorows));
  }
  PetscFunctionReturn(0);
}

/*@C
    MatSetOperation - Allows user to set a matrix operation for any matrix type

   Logically Collective on Mat

    Input Parameters:
+   mat - the matrix
.   op - the name of the operation
-   f - the function that provides the operation

   Level: developer

    Usage:
$      extern PetscErrorCode usermult(Mat,Vec,Vec);
$      PetscCall(MatCreateXXX(comm,...&A);
$      PetscCall(MatSetOperation(A,MATOP_MULT,(void(*)(void))usermult);

    Notes:
    See the file include/petscmat.h for a complete list of matrix
    operations, which all have the form MATOP_<OPERATION>, where
    <OPERATION> is the name (in all capital letters) of the
    user interface routine (e.g., MatMult() -> MATOP_MULT).

    All user-provided functions (except for MATOP_DESTROY) should have the same calling
    sequence as the usual matrix interface routines, since they
    are intended to be accessed via the usual matrix interface
    routines, e.g.,
$       MatMult(Mat,Vec,Vec) -> usermult(Mat,Vec,Vec)

    In particular each function MUST return an error code of 0 on success and
    nonzero on failure.

    This routine is distinct from MatShellSetOperation() in that it can be called on any matrix type.

.seealso: `MatGetOperation()`, `MatCreateShell()`, `MatShellSetContext()`, `MatShellSetOperation()`
@*/
PetscErrorCode MatSetOperation(Mat mat,MatOperation op,void (*f)(void))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (op == MATOP_VIEW && !mat->ops->viewnative && f != (void (*)(void))(mat->ops->view)) {
    mat->ops->viewnative = mat->ops->view;
  }
  (((void(**)(void))mat->ops)[op]) = f;
  PetscFunctionReturn(0);
}

/*@C
    MatGetOperation - Gets a matrix operation for any matrix type.

    Not Collective

    Input Parameters:
+   mat - the matrix
-   op - the name of the operation

    Output Parameter:
.   f - the function that provides the operation

    Level: developer

    Usage:
$      PetscErrorCode (*usermult)(Mat,Vec,Vec);
$      MatGetOperation(A,MATOP_MULT,(void(**)(void))&usermult);

    Notes:
    See the file include/petscmat.h for a complete list of matrix
    operations, which all have the form MATOP_<OPERATION>, where
    <OPERATION> is the name (in all capital letters) of the
    user interface routine (e.g., MatMult() -> MATOP_MULT).

    This routine is distinct from MatShellGetOperation() in that it can be called on any matrix type.

.seealso: `MatSetOperation()`, `MatCreateShell()`, `MatShellGetContext()`, `MatShellGetOperation()`
@*/
PetscErrorCode MatGetOperation(Mat mat,MatOperation op,void(**f)(void))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  *f = (((void (**)(void))mat->ops)[op]);
  PetscFunctionReturn(0);
}

/*@
    MatHasOperation - Determines whether the given matrix supports the particular
    operation.

   Not Collective

   Input Parameters:
+  mat - the matrix
-  op - the operation, for example, MATOP_GET_DIAGONAL

   Output Parameter:
.  has - either PETSC_TRUE or PETSC_FALSE

   Level: advanced

   Notes:
   See the file include/petscmat.h for a complete list of matrix
   operations, which all have the form MATOP_<OPERATION>, where
   <OPERATION> is the name (in all capital letters) of the
   user-level routine.  E.g., MatNorm() -> MATOP_NORM.

.seealso: `MatCreateShell()`
@*/
PetscErrorCode MatHasOperation(Mat mat,MatOperation op,PetscBool *has)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidBoolPointer(has,3);
  if (mat->ops->hasoperation) {
    PetscCall((*mat->ops->hasoperation)(mat,op,has));
  } else {
    if (((void**)mat->ops)[op]) *has = PETSC_TRUE;
    else {
      *has = PETSC_FALSE;
      if (op == MATOP_CREATE_SUBMATRIX) {
        PetscMPIInt size;

        PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size));
        if (size == 1) {
          PetscCall(MatHasOperation(mat,MATOP_CREATE_SUBMATRICES,has));
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@
    MatHasCongruentLayouts - Determines whether the rows and columns layouts
    of the matrix are congruent

   Collective on mat

   Input Parameters:
.  mat - the matrix

   Output Parameter:
.  cong - either PETSC_TRUE or PETSC_FALSE

   Level: beginner

   Notes:

.seealso: `MatCreate()`, `MatSetSizes()`
@*/
PetscErrorCode MatHasCongruentLayouts(Mat mat,PetscBool *cong)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);
  PetscValidBoolPointer(cong,2);
  if (!mat->rmap || !mat->cmap) {
    *cong = mat->rmap == mat->cmap ? PETSC_TRUE : PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  if (mat->congruentlayouts == PETSC_DECIDE) { /* first time we compare rows and cols layouts */
    PetscCall(PetscLayoutSetUp(mat->rmap));
    PetscCall(PetscLayoutSetUp(mat->cmap));
    PetscCall(PetscLayoutCompare(mat->rmap,mat->cmap,cong));
    if (*cong) mat->congruentlayouts = 1;
    else       mat->congruentlayouts = 0;
  } else *cong = mat->congruentlayouts ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetInf(Mat A)
{
  PetscFunctionBegin;
  PetscCheck(A->ops->setinf,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support for this operation for this matrix type");
  PetscCall((*A->ops->setinf)(A));
  PetscFunctionReturn(0);
}

/*C
   MatCreateGraph - create a scalar matrix, for use in graph algorithms

   Collective on mat

   Input Parameters:
+  A - the matrix
-  sym - PETSC_TRUE indicates that the graph will be symmetrized
.  scale - PETSC_TRUE indicates that the graph will be scaled with the diagonal

   Output Parameter:
.  graph - the resulting graph

   Level: advanced

   Notes:

.seealso: `MatCreate()`, `MatFilter()`
*/
PETSC_EXTERN PetscErrorCode MatCreateGraph(Mat A, PetscBool sym, PetscBool scale, Mat *graph)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(graph,3);
  PetscCheck(A->ops->creategraph,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support for this operation for this matrix type");
  PetscCall((*A->ops->creategraph)(A,sym,scale,graph));
  PetscFunctionReturn(0);
}

/*C
   MatFilter - filters a Mat values with an absolut value equal to or below a give threshold

   Collective on mat

   Input Parameter:
.  value - filter value - < 0: does nothing; == 0: removes only 0.0 entries; otherwise: removes entries <= value

   Input/Output Parameter:
.  A - the Mat to filter in place

   Level: advanced

   Notes:

.seealso: `MatCreate()`, `MatCreateGraph()`
*/
PETSC_EXTERN PetscErrorCode MatFilter(Mat G,PetscReal value,Mat *F)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(G,MAT_CLASSID,1);
  PetscValidType(G,1);
  PetscValidPointer(F,3);
  if (value >= 0.0) {
    PetscCheck(G->ops->filter,PetscObjectComm((PetscObject)G),PETSC_ERR_SUP,"No support for this operation for this matrix type");
    PetscCall((G->ops->filter)(G,value,F));
  }
  PetscFunctionReturn(0);
}
