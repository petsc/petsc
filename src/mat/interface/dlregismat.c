#define PETSCMAT_DLL

#include "private/matimpl.h"

const char *MatOptions[] = {"ROW_ORIENTED","NEW_NONZERO_LOCATIONS",
              "SYMMETRIC",
              "STRUCTURALLY_SYMMETRIC",
              "NEW_DIAGONALS",
              "IGNORE_OFF_PROC_ENTRIES",
              "NEW_NONZERO_LOCATION_ERR",
              "NEW_NONZERO_ALLOCATION_ERR","USE_HASH_TABLE",
              "KEEP_ZEROED_ROWS","IGNORE_ZERO_ENTRIES","USE_INODES",
              "HERMITIAN",
              "SYMMETRY_ETERNAL",
              "USE_COMPRESSEDROW",
              "IGNORE_LOWER_TRIANGULAR","ERROR_LOWER_TRIANGULAR","GETROW_UPPERTRIANGULAR","MatOption","MAT_",0};
const char *MatFactorShiftTypes[] = {"NONE","NONZERO","POSITIVE_DEFINITE","INBLOCKS","MatFactorShiftType","PC_FACTOR_",0};

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDInitializePackage(const char[]);
static PetscTruth MatPackageInitialized = PETSC_FALSE;
#undef __FUNCT__  
#define __FUNCT__ "MatFinalizePackage"
/*@C
  MatFinalizePackage - This function destroys everything in the Petsc interface to the charactoristics package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, mathematica
.seealso: PetscFinalize()
@*/
PetscErrorCode PETSC_DLLEXPORT MatFinalizePackage(void) 
{
  PetscFunctionBegin;
  MatPackageInitialized            = PETSC_FALSE;
  MatRegisterAllCalled             = PETSC_FALSE;
  MatList                          = PETSC_NULL;
  MatOrderingRegisterAllCalled     = PETSC_FALSE;
  MatOrderingList                  = PETSC_NULL;
  MatColoringList                  = PETSC_NULL;
  MatColoringRegisterAllCalled     = PETSC_FALSE;
  MatPartitioningList              = PETSC_NULL;
  MatPartitioningRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInitializePackage"
/*@C
  MatInitializePackage - This function initializes everything in the Mat package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to MatCreate()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: Mat, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatInitializePackage(const char path[]) 
{
  char              logList[256];
  char              *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (MatPackageInitialized) PetscFunctionReturn(0);
  MatPackageInitialized = PETSC_TRUE;
  /* Inialize subpackage */
  ierr = MatMFFDInitializePackage(path);CHKERRQ(ierr);
  /* Register Classes */
  ierr = PetscCookieRegister("Matrix",&MAT_COOKIE);CHKERRQ(ierr);
  ierr = PetscCookieRegister("Matrix FD Coloring",&MAT_FDCOLORING_COOKIE);CHKERRQ(ierr);
  ierr = PetscCookieRegister("Matrix Partitioning",&MAT_PARTITIONING_COOKIE);CHKERRQ(ierr);
  ierr = PetscCookieRegister("Matrix Null Space",&MAT_NULLSPACE_COOKIE);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = MatRegisterAll(path);CHKERRQ(ierr);
  ierr = MatOrderingRegisterAll(path);CHKERRQ(ierr);
  ierr = MatColoringRegisterAll(path);CHKERRQ(ierr);
  ierr = MatPartitioningRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("MatMult",          MAT_COOKIE,&MAT_Mult);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMults",         MAT_COOKIE,&MAT_Mults);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMultConstr",    MAT_COOKIE,&MAT_MultConstrained);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMultAdd",       MAT_COOKIE,&MAT_MultAdd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMultTranspose", MAT_COOKIE,&MAT_MultTranspose);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMultTrConstr",  MAT_COOKIE,&MAT_MultTransposeConstrained);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMultTrAdd",     MAT_COOKIE,&MAT_MultTransposeAdd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSolve",         MAT_COOKIE,&MAT_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSolves",        MAT_COOKIE,&MAT_Solves);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSolveAdd",      MAT_COOKIE,&MAT_SolveAdd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSolveTranspos", MAT_COOKIE,&MAT_SolveTranspose);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSolveTrAdd",    MAT_COOKIE,&MAT_SolveTransposeAdd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSOR",           MAT_COOKIE,&MAT_SOR);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatForwardSolve",  MAT_COOKIE,&MAT_ForwardSolve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatBackwardSolve", MAT_COOKIE,&MAT_BackwardSolve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatLUFactor",      MAT_COOKIE,&MAT_LUFactor);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatLUFactorSym",   MAT_COOKIE,&MAT_LUFactorSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatLUFactorNum",   MAT_COOKIE,&MAT_LUFactorNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCholeskyFctr",  MAT_COOKIE,&MAT_CholeskyFactor);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCholFctrSym",   MAT_COOKIE,&MAT_CholeskyFactorSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCholFctrNum",   MAT_COOKIE,&MAT_CholeskyFactorNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatILUFactor",     MAT_COOKIE,&MAT_ILUFactor);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatILUFactorSym",  MAT_COOKIE,&MAT_ILUFactorSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatICCFactorSym",  MAT_COOKIE,&MAT_ICCFactorSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCopy",          MAT_COOKIE,&MAT_Copy);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatConvert",       MAT_COOKIE,&MAT_Convert);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatScale",         MAT_COOKIE,&MAT_Scale);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatAssemblyBegin", MAT_COOKIE,&MAT_AssemblyBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatAssemblyEnd",   MAT_COOKIE,&MAT_AssemblyEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSetValues",     MAT_COOKIE,&MAT_SetValues);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetValues",     MAT_COOKIE,&MAT_GetValues);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetRow",        MAT_COOKIE,&MAT_GetRow);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetRowIJ",      MAT_COOKIE,&MAT_GetRowIJ);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetSubMatrice", MAT_COOKIE,&MAT_GetSubMatrices);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetColoring",   MAT_COOKIE,&MAT_GetColoring);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetOrdering",   MAT_COOKIE,&MAT_GetOrdering);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatIncreaseOvrlp", MAT_COOKIE,&MAT_IncreaseOverlap);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatPartitioning",  MAT_PARTITIONING_COOKIE,&MAT_Partitioning);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatZeroEntries",   MAT_COOKIE,&MAT_ZeroEntries);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatLoad",          MAT_COOKIE,&MAT_Load);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatView",          MAT_COOKIE,&MAT_View);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatAXPY",          MAT_COOKIE,&MAT_AXPY);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatFDColorCreate", MAT_FDCOLORING_COOKIE,&MAT_FDColoringCreate);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatFDColorApply",  MAT_FDCOLORING_COOKIE,&MAT_FDColoringApply);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatFDColorFunc",   MAT_FDCOLORING_COOKIE,&MAT_FDColoringFunction);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatTranspose",     MAT_COOKIE,&MAT_Transpose);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatMult",       MAT_COOKIE,&MAT_MatMult);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatSolve",      MAT_COOKIE,&MAT_MatSolve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatMultSym",    MAT_COOKIE,&MAT_MatMultSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatMultNum",    MAT_COOKIE,&MAT_MatMultNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatPtAP",          MAT_COOKIE,&MAT_PtAP);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatPtAPSymbolic",  MAT_COOKIE,&MAT_PtAPSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatPtAPNumeric",   MAT_COOKIE,&MAT_PtAPNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatMultTrans",  MAT_COOKIE,&MAT_MatMultTranspose);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatMultTrnSym" ,MAT_COOKIE,&MAT_MatMultTransposeSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatMultTrnNum", MAT_COOKIE,&MAT_MatMultTransposeNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetRedundant",  MAT_COOKIE,&MAT_GetRedundantMatrix);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetSeqNZStrct", MAT_COOKIE,&MAT_GetSequentialNonzeroStructure);CHKERRQ(ierr);

  /* these may be specific to MPIAIJ matrices */
  ierr = PetscLogEventRegister("MatMerge_SeqsToMPINumeric",MAT_COOKIE,&MAT_Seqstompinum);
  ierr = PetscLogEventRegister("MatMerge_SeqsToMPISymbolic",MAT_COOKIE,&MAT_Seqstompisym);
  ierr = PetscLogEventRegister("MatMerge_SeqsToMPI",MAT_COOKIE,&MAT_Seqstompi);
  ierr = PetscLogEventRegister("MatGetLocalMat",MAT_COOKIE,&MAT_Getlocalmat);
  ierr = PetscLogEventRegister("MatGetLocalMatCondensed",MAT_COOKIE,&MAT_Getlocalmatcondensed);
  ierr = PetscLogEventRegister("MatGetBrowsOfAcols",MAT_COOKIE,&MAT_GetBrowsOfAcols);
  ierr = PetscLogEventRegister("MatGetBrAoCol",MAT_COOKIE,&MAT_GetBrowsOfAocols);

  ierr = PetscLogEventRegister("MatApplyPAPt_Symbolic",MAT_COOKIE,&MAT_Applypapt_symbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatApplyPAPt_Numeric",MAT_COOKIE,&MAT_Applypapt_numeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatApplyPAPt",MAT_COOKIE,&MAT_Applypapt);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("MatGetSymTrans",MAT_COOKIE,&MAT_Getsymtranspose);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetSymTransR",MAT_COOKIE,&MAT_Getsymtransreduced);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatTranspose_SeqAIJ_FAST",MAT_COOKIE,&MAT_Transpose_SeqAIJ);CHKERRQ(ierr);

  /* Turn off high traffic events by default */
  ierr = PetscLogEventSetActiveAll(MAT_SetValues, PETSC_FALSE);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "mat", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(MAT_COOKIE);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "mat", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(MAT_COOKIE);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(MatFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryRegister_petscmat"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the matrix methods that are in the basic PETSc Matrix library.

  Input Parameter:
  path - library path
 */
PetscErrorCode PETSCMAT_DLLEXPORT PetscDLLibraryRegister_petscmat(const char path[])
{
  PetscErrorCode ierr;

  ierr = PetscInitializeNoArguments(); if (ierr) return 1;

  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = MatInitializePackage(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
