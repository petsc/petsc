#define TAOSOLVER_DLL

#include "include/private/taosolver_impl.h"


EXTERN_C_BEGIN
extern PetscErrorCode TaoSolverCreate_LMVM(TaoSolver);
extern PetscErrorCode TaoSolverCreate_NLS(TaoSolver);
extern PetscErrorCode TaoSolverCreate_NTR(TaoSolver);
extern PetscErrorCode TaoSolverCreate_NTL(TaoSolver);
extern PetscErrorCode TaoSolverCreate_NM(TaoSolver);
extern PetscErrorCode TaoSolverCreate_CG(TaoSolver);

extern PetscErrorCode TaoSolverCreate_BLMVM(TaoSolver);
extern PetscErrorCode TaoSolverCreate_GPCG(TaoSolver);
extern PetscErrorCode TaoSolverCreate_BQPIP(TaoSolver);

extern PetscErrorCode TaoSolverCreate_POUNDERS(TaoSolver);
//extern PetscErrorCode TaoSolverCreate_LM(TaoSolver);
/*

extern PetscErrorCode TaoSolverCreate_TRON(TaoSolver);
extern PetscErrorCode TaoSolverCreate_BNLS(TaoSolver);
extern PetscErrorCode TaoSolverCreate_QPIP(TaoSolver);

extern PetscErrorCode TaoSolverCreate_NLSQ(TaoSolver);
extern PetscErrorCode TaoSolverCreate_BLM(TaoSolver);
extern PetscErrorCode TaoSolverCreate_SSILS(TaoSolver);
extern PetscErrorCode TaoSolverCreate_SSFLS(TaoSolver);
extern PetscErrorCode TaoSolverCreate_ASILS(TaoSolver);
extern PetscErrorCode TaoSolverCreate_ASFLS(TaoSolver);
extern PetscErrorCode TaoSolverCreate_ISILS(TaoSolver);
extern PetscErrorCode TaoSolverCreate_KT(TaoSolver);
extern PetscErrorCode TaoSolverCreate_BCG(TaoSolver);
extern PetscErrorCode TaoSolverCreate_RSCS(TaoSolver);
extern PetscErrorCode TaoSolverCreate_ICP(TaoSolver);
extern PetscErrorCode TaoSolverCreate_FD(TaoSolver);
*/
EXTERN_C_END

/* 
   Offset the convergence reasons so negative number represent diverged and
   positive represent converged.
*/
const char *TaoSolverTerminationReasons_Shifted[] = {
    "DIVERGED_USER",
    "DIVERGED_TR_REDUCTION",
    "DIVERGED_LS_FAILURE",
    "DIVERGED_MAXFCN",
    "DIVERGED_NAN",
    "DIVERGED_MAXITS",
    "DIVERGED_FUNCTION_DOMAIN",
 
    "CONTINUE_ITERATING",
    
    " ",
    "CONVERGED_ATOL",
    "CONVERGED_RTOL",
    "CONVERGED_TRTOL",
    "CONVERGED_MINF",
    "CONVERGED_USER" };
const char **TaoSolverTerminationReasons = TaoSolverTerminationReasons_Shifted + 8;

						   


extern PetscBool TaoSolverRegisterAllCalled;

#undef __FUNCT__
#define __FUNCT__ "TaoSolverRegisterAll"
/*@C
  TaoSolverRegisterAll - Registersall of the minimization methods in the TAO
  package.

  Not Collective

  Level: developer

.seealso TaoSolverRegisterDynamic(), TaoSolverRegisterDestroy()
@*/
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverRegisterAll(const char path[])
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  TaoSolverRegisterAllCalled = PETSC_TRUE;
  
  ierr = TaoSolverRegisterDynamic("tao_lmvm",path,"TaoSolverCreate_LMVM",TaoSolverCreate_LMVM); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_nls",path,"TaoSolverCreate_NLS",TaoSolverCreate_NLS); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_ntr",path,"TaoSolverCreate_NTR",TaoSolverCreate_NTR); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_ntl",path,"TaoSolverCreate_NTL",TaoSolverCreate_NTL); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_cg",path,"TaoSolverCreate_CG",TaoSolverCreate_CG); CHKERRQ(ierr);

  ierr = TaoSolverRegisterDynamic("tao_blmvm",path,"TaoSolverCreate_BLMVM",TaoSolverCreate_BLMVM); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_bqpip",path,"TaoSolverCreate_BQPIP",TaoSolverCreate_BQPIP); CHKERRQ(ierr);

  ierr = TaoSolverRegisterDynamic("tao_gpcg",path,"TaoSolverCreate_GPCG",TaoSolverCreate_GPCG); CHKERRQ(ierr);

  ierr = TaoSolverRegisterDynamic("tao_nm",path,"TaoSolverCreate_NM",TaoSolverCreate_NM); CHKERRQ(ierr);

  ierr = TaoSolverRegisterDynamic("tao_pounders",path,"TaoSolverCreate_POUNDERS",TaoSolverCreate_POUNDERS); CHKERRQ(ierr);
//  ierr = TaoSolverRegisterDynamic("tao_lm",path,"TaoSolverCreate_POUNDERS",TaoSolverCreate_LM); CHKERRQ(ierr);
/*
  ierr = TaoSolverRegisterDynamic("tao_bnls",path,"TaoSolverCreate_BNLS",TaoSolverCreate_BNLS); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_tron",path,"TaoSolverCreate_TRON",TaoSolverCreate_TRON); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_nm",path,"TaoSolverCreate_NM",TaoSolverCreate_NM); CHKERRQ(ierr);

  ierr = TaoSolverRegisterDynamic("tao_ssils",path,"TaoSolverCreate_SSILS",TaoSolverCreate_SSILS); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_ssfls",path,"TaoSolverCreate_SSFLS",TaoSolverCreate_SSFLS); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_asils",path,"TaoSolverCreate_ASILS",TaoSolverCreate_ASILS); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_asfls",path,"TaoSolverCreate_ASFLS",TaoSolverCreate_ASFLS); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_isils",path,"TaoSolverCreate_ISILS",TaoSolverCreate_ISILS); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_kt",path,"TaoSolverCreate_KT",TaoSolverCreate_KT); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_rscs",path,"TaoSolverCreate_RSCS",TaoSolverCreate_RSCS); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_icp",path,"TaoSolverCreate_ICP",TaoSolverCreate_ICP); CHKERRQ(ierr);


  ierr = TaoSolverRegisterDynamic("tao_fd_test",path,"TaoSolverCreate_FD",TaoSolverCreate_FD); CHKERRQ(ierr);
*/
  
  PetscFunctionReturn(0);
}
    

