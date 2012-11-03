
#include <petsc-private/matimpl.h>

const char *MatOptions[] = {"ROW_ORIENTED","NEW_NONZERO_LOCATIONS",
              "SYMMETRIC",
              "STRUCTURALLY_SYMMETRIC",
              "NEW_DIAGONALS",
              "IGNORE_OFF_PROC_ENTRIES",
              "NEW_NONZERO_LOCATION_ERR",
              "NEW_NONZERO_ALLOCATION_ERR","USE_HASH_TABLE",
              "KEEP_NONZERO_PATTERN","IGNORE_ZERO_ENTRIES","USE_INODES",
              "HERMITIAN",
              "SYMMETRY_ETERNAL",
              "CHECK_COMPRESSED_ROW",
              "IGNORE_LOWER_TRIANGULAR","ERROR_LOWER_TRIANGULAR","GETROW_UPPERTRIANGULAR","SPD","NO_OFF_PROC_ENTRIES","NO_OFF_PROC_ZERO_ROWS","MatOption","MAT_",0};
const char *const MatFactorShiftTypes[] = {"NONE","NONZERO","POSITIVE_DEFINITE","INBLOCKS","MatFactorShiftType","PC_FACTOR_",0};
const char *const MPPTScotchStrategyTypes[] = {"QUALITY","SPEED","BALANCE","SAFETY","SCALABILITY","MPPTScotchStrategyType","MP_PTSCOTCH_",0};
const char *const MPChacoGlobalTypes[] = {"","MULTILEVEL","SPECTRAL","","LINEAR","RANDOM","SCATTERED","MPChacoGlobalType","MP_CHACO_",0};
const char *const MPChacoLocalTypes[] = {"","KERNIGHAN","NONE","MPChacoLocalType","MP_CHACO_",0};
const char *const MPChacoEigenTypes[] = {"LANCZOS","RQI","MPChacoEigenType","MP_CHACO_",0};

extern PetscErrorCode  MatMFFDInitializePackage(const char[]);
static PetscBool  MatPackageInitialized = PETSC_FALSE;
#undef __FUNCT__
#define __FUNCT__ "MatFinalizePackage"
/*@C
  MatFinalizePackage - This function destroys everything in the Petsc interface to the Mat package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, mathematica
.seealso: PetscFinalize()
@*/
PetscErrorCode  MatFinalizePackage(void)
{
  MatBaseName    nnames,names = MatBaseNameList;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while (names) {
    nnames = names->next;
    ierr = PetscFree(names->bname);CHKERRQ(ierr);
    ierr = PetscFree(names->sname);CHKERRQ(ierr);
    ierr = PetscFree(names->mname);CHKERRQ(ierr);
    ierr = PetscFree(names);CHKERRQ(ierr);
    names = nnames;
  }
  MatBaseNameList                  = PETSC_NULL;
  MatPackageInitialized            = PETSC_FALSE;
  MatRegisterAllCalled             = PETSC_FALSE;
  MatList                          = PETSC_NULL;
  MatOrderingRegisterAllCalled     = PETSC_FALSE;
  MatOrderingList                  = PETSC_NULL;
  MatColoringList                  = PETSC_NULL;
  MatColoringRegisterAllCalled     = PETSC_FALSE;
  MatPartitioningList              = PETSC_NULL;
  MatPartitioningRegisterAllCalled = PETSC_FALSE;
  MatCoarsenList              = PETSC_NULL;
  MatCoarsenRegisterAllCalled = PETSC_FALSE;
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
PetscErrorCode  MatInitializePackage(const char path[])
{
  char              logList[256];
  char              *className;
  PetscBool         opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (MatPackageInitialized) PetscFunctionReturn(0);
  MatPackageInitialized = PETSC_TRUE;
  /* Inialize subpackage */
  ierr = MatMFFDInitializePackage(path);CHKERRQ(ierr);
  /* Register Classes */
  ierr = PetscClassIdRegister("Matrix",&MAT_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Matrix FD Coloring",&MAT_FDCOLORING_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Matrix MatTranspose Coloring",&MAT_TRANSPOSECOLORING_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Matrix Partitioning",&MAT_PARTITIONING_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Matrix Coarsen",&MAT_COARSEN_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Matrix Null Space",&MAT_NULLSPACE_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = MatRegisterAll(path);CHKERRQ(ierr);
  ierr = MatOrderingRegisterAll(path);CHKERRQ(ierr);
  ierr = MatColoringRegisterAll(path);CHKERRQ(ierr);
  ierr = MatPartitioningRegisterAll(path);CHKERRQ(ierr);
  ierr = MatCoarsenRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("MatMult",          MAT_CLASSID,&MAT_Mult);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMults",         MAT_CLASSID,&MAT_Mults);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMultConstr",    MAT_CLASSID,&MAT_MultConstrained);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMultAdd",       MAT_CLASSID,&MAT_MultAdd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMultTranspose", MAT_CLASSID,&MAT_MultTranspose);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMultTrConstr",  MAT_CLASSID,&MAT_MultTransposeConstrained);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMultTrAdd",     MAT_CLASSID,&MAT_MultTransposeAdd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSolve",         MAT_CLASSID,&MAT_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSolves",        MAT_CLASSID,&MAT_Solves);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSolveAdd",      MAT_CLASSID,&MAT_SolveAdd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSolveTranspos", MAT_CLASSID,&MAT_SolveTranspose);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSolveTrAdd",    MAT_CLASSID,&MAT_SolveTransposeAdd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSOR",           MAT_CLASSID,&MAT_SOR);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatForwardSolve",  MAT_CLASSID,&MAT_ForwardSolve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatBackwardSolve", MAT_CLASSID,&MAT_BackwardSolve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatLUFactor",      MAT_CLASSID,&MAT_LUFactor);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatLUFactorSym",   MAT_CLASSID,&MAT_LUFactorSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatLUFactorNum",   MAT_CLASSID,&MAT_LUFactorNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCholeskyFctr",  MAT_CLASSID,&MAT_CholeskyFactor);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCholFctrSym",   MAT_CLASSID,&MAT_CholeskyFactorSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCholFctrNum",   MAT_CLASSID,&MAT_CholeskyFactorNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatILUFactor",     MAT_CLASSID,&MAT_ILUFactor);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatILUFactorSym",  MAT_CLASSID,&MAT_ILUFactorSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatICCFactorSym",  MAT_CLASSID,&MAT_ICCFactorSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCopy",          MAT_CLASSID,&MAT_Copy);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatConvert",       MAT_CLASSID,&MAT_Convert);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatScale",         MAT_CLASSID,&MAT_Scale);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatAssemblyBegin", MAT_CLASSID,&MAT_AssemblyBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatAssemblyEnd",   MAT_CLASSID,&MAT_AssemblyEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSetValues",     MAT_CLASSID,&MAT_SetValues);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetValues",     MAT_CLASSID,&MAT_GetValues);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetRow",        MAT_CLASSID,&MAT_GetRow);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetRowIJ",      MAT_CLASSID,&MAT_GetRowIJ);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetSubMatrice", MAT_CLASSID,&MAT_GetSubMatrices);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetColoring",   MAT_CLASSID,&MAT_GetColoring);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetOrdering",   MAT_CLASSID,&MAT_GetOrdering);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatIncreaseOvrlp", MAT_CLASSID,&MAT_IncreaseOverlap);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatPartitioning",  MAT_PARTITIONING_CLASSID,&MAT_Partitioning);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCoarsen",  MAT_COARSEN_CLASSID,&MAT_Coarsen);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatZeroEntries",   MAT_CLASSID,&MAT_ZeroEntries);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatLoad",          MAT_CLASSID,&MAT_Load);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatView",          MAT_CLASSID,&MAT_View);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatAXPY",          MAT_CLASSID,&MAT_AXPY);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatFDColorCreate", MAT_FDCOLORING_CLASSID,&MAT_FDColoringCreate);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatFDColorApply",  MAT_FDCOLORING_CLASSID,&MAT_FDColoringApply);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatFDColorFunc",   MAT_FDCOLORING_CLASSID,&MAT_FDColoringFunction);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatTranspose",     MAT_CLASSID,&MAT_Transpose);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatMult",       MAT_CLASSID,&MAT_MatMult);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatSolve",      MAT_CLASSID,&MAT_MatSolve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatMultSym",    MAT_CLASSID,&MAT_MatMultSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatMultNum",    MAT_CLASSID,&MAT_MatMultNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatMatMult",    MAT_CLASSID,&MAT_MatMatMult);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatMatMultSym", MAT_CLASSID,&MAT_MatMatMultSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatMatMultNum", MAT_CLASSID,&MAT_MatMatMultNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatPtAP",          MAT_CLASSID,&MAT_PtAP);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatPtAPSymbolic",  MAT_CLASSID,&MAT_PtAPSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatPtAPNumeric",   MAT_CLASSID,&MAT_PtAPNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatRARt",          MAT_CLASSID,&MAT_RARt);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatRARtSymbolic",  MAT_CLASSID,&MAT_RARtSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatRARtNumeric",   MAT_CLASSID,&MAT_RARtNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatTransMult",  MAT_CLASSID,&MAT_MatTransposeMult);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatTrnMultSym", MAT_CLASSID,&MAT_MatTransposeMultSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatTrnMultNum", MAT_CLASSID,&MAT_MatTransposeMultNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatTrnMatMult",    MAT_CLASSID,&MAT_TransposeMatMult);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatTrnMatMultSym", MAT_CLASSID,&MAT_TransposeMatMultSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatTrnMatMultNum", MAT_CLASSID,&MAT_TransposeMatMultNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatTrnColorCreate", MAT_CLASSID,&MAT_TransposeColoringCreate);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetRedundant",  MAT_CLASSID,&MAT_GetRedundantMatrix);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetSeqNZStrct", MAT_CLASSID,&MAT_GetSequentialNonzeroStructure);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetMultiProcBlock", MAT_CLASSID,&MAT_GetMultiProcBlock);CHKERRQ(ierr);


  /* these may be specific to MPIAIJ matrices */
  ierr = PetscLogEventRegister("MatMPISumSeqNumeric",MAT_CLASSID,&MAT_Seqstompinum);
  ierr = PetscLogEventRegister("MatMPISumSeqSymbolic",MAT_CLASSID,&MAT_Seqstompisym);
  ierr = PetscLogEventRegister("MatMPISumSeq",MAT_CLASSID,&MAT_Seqstompi);
  ierr = PetscLogEventRegister("MatMPIConcateSeq",MAT_CLASSID,&MAT_Merge);
  ierr = PetscLogEventRegister("MatGetLocalMat",MAT_CLASSID,&MAT_Getlocalmat);
  ierr = PetscLogEventRegister("MatGetLocalMatCondensed",MAT_CLASSID,&MAT_Getlocalmatcondensed);
  ierr = PetscLogEventRegister("MatGetBrowsOfAcols",MAT_CLASSID,&MAT_GetBrowsOfAcols);
  ierr = PetscLogEventRegister("MatGetBrAoCol",MAT_CLASSID,&MAT_GetBrowsOfAocols);

  ierr = PetscLogEventRegister("MatApplyPAPt_Symbolic",MAT_CLASSID,&MAT_Applypapt_symbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatApplyPAPt_Numeric",MAT_CLASSID,&MAT_Applypapt_numeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatApplyPAPt",MAT_CLASSID,&MAT_Applypapt);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("MatGetSymTrans",MAT_CLASSID,&MAT_Getsymtranspose);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetSymTransR",MAT_CLASSID,&MAT_Getsymtransreduced);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatTranspose_SeqAIJ_FAST",MAT_CLASSID,&MAT_Transpose_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCUSPCopyTo",MAT_CLASSID,&MAT_CUSPCopyToGPU);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCUSPARSECopyTo",MAT_CLASSID,&MAT_CUSPARSECopyToGPU);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSetValBatch",MAT_CLASSID,&MAT_SetValuesBatch);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSetValBatch1",MAT_CLASSID,&MAT_SetValuesBatchI);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSetValBatch2",MAT_CLASSID,&MAT_SetValuesBatchII);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSetValBatch3",MAT_CLASSID,&MAT_SetValuesBatchIII);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSetValBatch4",MAT_CLASSID,&MAT_SetValuesBatchIV);CHKERRQ(ierr);

  /* Turn off high traffic events by default */
  ierr = PetscLogEventSetActiveAll(MAT_SetValues, PETSC_FALSE);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "mat", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(MAT_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "mat", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(MAT_CLASSID);CHKERRQ(ierr);
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
PetscErrorCode  PetscDLLibraryRegister_petscmat(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatInitializePackage(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
