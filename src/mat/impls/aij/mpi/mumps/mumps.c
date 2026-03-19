/*
    Provides an interface to the MUMPS sparse solver
*/
#include <petsc/private/matimpl.h>
#include <petscmat.h>

/*@
  MatMumpsSetIcntl - Set MUMPS parameter ICNTL() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
. icntl - index of MUMPS parameter array `ICNTL()`
- ival  - value of MUMPS `ICNTL(icntl)`

  Options Database Key:
. -mat_mumps_icntl_ICNTL ival - change the option numbered `icntl` to `ival`, here ICNTL denotes an integer value

  Level: beginner

  Note:
  Ignored if MUMPS is not installed or `F` is not a MUMPS matrix

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsGetIcntl()`, `MatMumpsSetCntl()`, `MatMumpsGetCntl()`, `MatMumpsGetInfo()`, `MatMumpsGetInfog()`, `MatMumpsGetRinfo()`, `MatMumpsGetRinfog()`
@*/
PetscErrorCode MatMumpsSetIcntl(Mat F, PetscInt icntl, PetscInt ival)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscValidLogicalCollectiveInt(F, icntl, 2);
  PetscValidLogicalCollectiveInt(F, ival, 3);
  PetscTryMethod(F, "MatMumpsSetIcntl_C", (Mat, PetscInt, PetscInt), (F, icntl, ival));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetIcntl - Get MUMPS parameter ICNTL() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
- icntl - index of MUMPS parameter array ICNTL()

  Output Parameter:
. ival - value of MUMPS ICNTL(icntl)

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatMumpsSetCntl()`, `MatMumpsGetCntl()`, `MatMumpsGetInfo()`, `MatMumpsGetInfog()`, `MatMumpsGetRinfo()`, `MatMumpsGetRinfog()`
@*/
PetscErrorCode MatMumpsGetIcntl(Mat F, PetscInt icntl, PetscInt *ival)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscValidLogicalCollectiveInt(F, icntl, 2);
  PetscAssertPointer(ival, 3);
  PetscUseMethod(F, "MatMumpsGetIcntl_C", (Mat, PetscInt, PetscInt *), (F, icntl, ival));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsSetCntl - Set MUMPS parameter CNTL() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
. icntl - index of MUMPS parameter array `CNTL()`
- val   - value of MUMPS `CNTL(icntl)`

  Options Database Key:
. -mat_mumps_cntl_icntl val - change the option numbered icntl to ival

  Level: beginner

  Note:
  Ignored if MUMPS is not installed or `F` is not a MUMPS matrix

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatMumpsGetIcntl()`, `MatMumpsGetCntl()`, `MatMumpsGetInfo()`, `MatMumpsGetInfog()`, `MatMumpsGetRinfo()`, `MatMumpsGetRinfog()`
@*/
PetscErrorCode MatMumpsSetCntl(Mat F, PetscInt icntl, PetscReal val)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscValidLogicalCollectiveInt(F, icntl, 2);
  PetscValidLogicalCollectiveReal(F, val, 3);
  PetscTryMethod(F, "MatMumpsSetCntl_C", (Mat, PetscInt, PetscReal), (F, icntl, val));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetCntl - Get MUMPS parameter CNTL() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
- icntl - index of MUMPS parameter array CNTL()

  Output Parameter:
. val - value of MUMPS CNTL(icntl)

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatMumpsGetIcntl()`, `MatMumpsSetCntl()`, `MatMumpsGetInfo()`, `MatMumpsGetInfog()`, `MatMumpsGetRinfo()`, `MatMumpsGetRinfog()`
@*/
PetscErrorCode MatMumpsGetCntl(Mat F, PetscInt icntl, PetscReal *val)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscValidLogicalCollectiveInt(F, icntl, 2);
  PetscAssertPointer(val, 3);
  PetscUseMethod(F, "MatMumpsGetCntl_C", (Mat, PetscInt, PetscReal *), (F, icntl, val));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetInverse - Get user-specified set of entries in inverse of `A` <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameter:
. F - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`

  Output Parameter:
. spRHS - sequential sparse matrix in `MATTRANSPOSEVIRTUAL` format with requested entries of inverse of `A`

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatCreateTranspose()`
@*/
PetscErrorCode MatMumpsGetInverse(Mat F, Mat spRHS)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscUseMethod(F, "MatMumpsGetInverse_C", (Mat, Mat), (F, spRHS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetInverseTranspose - Get user-specified set of entries in inverse of matrix $A^T $ <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameter:
. F - the factored matrix of A obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`

  Output Parameter:
. spRHST - sequential sparse matrix in `MATAIJ` format containing the requested entries of inverse of `A`^T

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatCreateTranspose()`, `MatMumpsGetInverse()`
@*/
PetscErrorCode MatMumpsGetInverseTranspose(Mat F, Mat spRHST)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscCall(PetscObjectTypeCompareAny((PetscObject)spRHST, &flg, MATSEQAIJ, MATMPIAIJ, NULL));
  PetscCheck(flg, PetscObjectComm((PetscObject)spRHST), PETSC_ERR_ARG_WRONG, "Matrix spRHST must be MATAIJ matrix");
  PetscUseMethod(F, "MatMumpsGetInverseTranspose_C", (Mat, Mat), (F, spRHST));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsSetBlk - Set user-specified variable block sizes to be used with `-mat_mumps_icntl_15 1`

  Not collective, only relevant on the first process of the MPI communicator

  Input Parameters:
+ F      - the factored matrix of A obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
. nblk   - the number of blocks
. blkvar - see MUMPS documentation, `blkvar(blkptr(iblk):blkptr(iblk+1)-1)`, (`iblk=1, nblk`) holds the variables associated to block `iblk`
- blkptr - array starting at 1 and of size `nblk + 1` storing the prefix sum of all blocks

  Level: advanced

.seealso: [](ch_matrices), `MATSOLVERMUMPS`, `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatSetVariableBlockSizes()`
@*/
PetscErrorCode MatMumpsSetBlk(Mat F, PetscInt nblk, const PetscInt blkvar[], const PetscInt blkptr[])
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscUseMethod(F, "MatMumpsSetBlk_C", (Mat, PetscInt, const PetscInt[], const PetscInt[]), (F, nblk, blkvar, blkptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetInfo - Get MUMPS parameter INFO() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
- icntl - index of MUMPS parameter array INFO()

  Output Parameter:
. ival - value of MUMPS INFO(icntl)

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatMumpsGetIcntl()`, `MatMumpsSetCntl()`, `MatMumpsGetCntl()`, `MatMumpsGetInfog()`, `MatMumpsGetRinfo()`, `MatMumpsGetRinfog()`
@*/
PetscErrorCode MatMumpsGetInfo(Mat F, PetscInt icntl, PetscInt *ival)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscAssertPointer(ival, 3);
  PetscUseMethod(F, "MatMumpsGetInfo_C", (Mat, PetscInt, PetscInt *), (F, icntl, ival));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetInfog - Get MUMPS parameter INFOG() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
- icntl - index of MUMPS parameter array INFOG()

  Output Parameter:
. ival - value of MUMPS INFOG(icntl)

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatMumpsGetIcntl()`, `MatMumpsSetCntl()`, `MatMumpsGetCntl()`, `MatMumpsGetInfo()`, `MatMumpsGetRinfo()`, `MatMumpsGetRinfog()`
@*/
PetscErrorCode MatMumpsGetInfog(Mat F, PetscInt icntl, PetscInt *ival)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscAssertPointer(ival, 3);
  PetscUseMethod(F, "MatMumpsGetInfog_C", (Mat, PetscInt, PetscInt *), (F, icntl, ival));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetRinfo - Get MUMPS parameter RINFO() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
- icntl - index of MUMPS parameter array RINFO()

  Output Parameter:
. val - value of MUMPS RINFO(icntl)

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatMumpsGetIcntl()`, `MatMumpsSetCntl()`, `MatMumpsGetCntl()`, `MatMumpsGetInfo()`, `MatMumpsGetInfog()`, `MatMumpsGetRinfog()`
@*/
PetscErrorCode MatMumpsGetRinfo(Mat F, PetscInt icntl, PetscReal *val)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscAssertPointer(val, 3);
  PetscUseMethod(F, "MatMumpsGetRinfo_C", (Mat, PetscInt, PetscReal *), (F, icntl, val));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetRinfog - Get MUMPS parameter RINFOG() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
- icntl - index of MUMPS parameter array RINFOG()

  Output Parameter:
. val - value of MUMPS RINFOG(icntl)

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatMumpsGetIcntl()`, `MatMumpsSetCntl()`, `MatMumpsGetCntl()`, `MatMumpsGetInfo()`, `MatMumpsGetInfog()`, `MatMumpsGetRinfo()`
@*/
PetscErrorCode MatMumpsGetRinfog(Mat F, PetscInt icntl, PetscReal *val)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscAssertPointer(val, 3);
  PetscUseMethod(F, "MatMumpsGetRinfog_C", (Mat, PetscInt, PetscReal *), (F, icntl, val));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetNullPivots - Get MUMPS parameter PIVNUL_LIST() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameter:
. F - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`

  Output Parameters:
+ size  - local size of the array. The size of the array is non-zero only on MPI rank 0
- array - array of rows with null pivot, these rows follow 0-based indexing. The array gets allocated within the function and the user is responsible
          for freeing this array.

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatMumpsGetIcntl()`, `MatMumpsSetCntl()`, `MatMumpsGetCntl()`, `MatMumpsGetInfo()`, `MatMumpsGetInfog()`, `MatMumpsGetRinfo()`
@*/
PetscErrorCode MatMumpsGetNullPivots(Mat F, PetscInt *size, PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscAssertPointer(size, 2);
  PetscAssertPointer(array, 3);
  PetscUseMethod(F, "MatMumpsGetNullPivots_C", (Mat, PetscInt *, PetscInt **), (F, size, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}
