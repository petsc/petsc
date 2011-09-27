#define TAOSOLVER_DLL

#include "include/private/taosolver_impl.h"


EXTERN_C_BEGIN
extern PetscErrorCode TaoCreate_LMVM(TaoSolver);
extern PetscErrorCode TaoCreate_NLS(TaoSolver);
extern PetscErrorCode TaoCreate_NTR(TaoSolver);
extern PetscErrorCode TaoCreate_NTL(TaoSolver);
extern PetscErrorCode TaoCreate_NM(TaoSolver);
extern PetscErrorCode TaoCreate_CG(TaoSolver);
extern PetscErrorCode TaoCreate_TRON(TaoSolver);

extern PetscErrorCode TaoCreate_BLMVM(TaoSolver);
extern PetscErrorCode TaoCreate_GPCG(TaoSolver);
extern PetscErrorCode TaoCreate_BQPIP(TaoSolver);

extern PetscErrorCode TaoCreate_POUNDERS(TaoSolver);
extern PetscErrorCode TaoCreate_POUNDER(TaoSolver);

extern PetscErrorCode TaoCreate_FD(TaoSolver);

extern PetscErrorCode TaoCreate_LCL(TaoSolver);
//extern PetscErrorCode TaoCreate_LM(TaoSolver);
/*

extern PetscErrorCode TaoCreate_BNLS(TaoSolver);
extern PetscErrorCode TaoCreate_QPIP(TaoSolver);

extern PetscErrorCode TaoCreate_NLSQ(TaoSolver);
extern PetscErrorCode TaoCreate_BLM(TaoSolver);
extern PetscErrorCode TaoCreate_SSILS(TaoSolver);
extern PetscErrorCode TaoCreate_SSFLS(TaoSolver);
extern PetscErrorCode TaoCreate_ASILS(TaoSolver);
extern PetscErrorCode TaoCreate_ASFLS(TaoSolver);
extern PetscErrorCode TaoCreate_ISILS(TaoSolver);
extern PetscErrorCode TaoCreate_KT(TaoSolver);
extern PetscErrorCode TaoCreate_RSCS(TaoSolver);
extern PetscErrorCode TaoCreate_ICP(TaoSolver);
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
    
    "CONVERGED_FATOL",
    "CONVERGED_FRTOL",
    "CONVERGED_GATOL",
    "CONVERGED_GRTOL",
    "CONVERGED_GTTOL",
    "CONVERGED_STEPTOL",
    "CONVERGED_MINF",
    "CONVERGED_USER" };
const char **TaoSolverTerminationReasons = TaoSolverTerminationReasons_Shifted + 8;

						   


extern PetscBool TaoRegisterAllCalled;

#undef __FUNCT__
#define __FUNCT__ "TaoRegisterAll"
/*@C
  TaoRegisterAll - Registersall of the minimization methods in the TAO
  package.

  Not Collective

  Level: developer

.seealso TaoRegisterDynamic(), TaoRegisterDestroy()
@*/
PetscErrorCode TaoRegisterAll(const char path[])
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  TaoRegisterAllCalled = PETSC_TRUE;
  
  ierr = TaoRegisterDynamic("tao_lmvm",path,"TaoCreate_LMVM",TaoCreate_LMVM); CHKERRQ(ierr);
  ierr = TaoRegisterDynamic("tao_nls",path,"TaoCreate_NLS",TaoCreate_NLS); CHKERRQ(ierr);
  ierr = TaoRegisterDynamic("tao_ntr",path,"TaoCreate_NTR",TaoCreate_NTR); CHKERRQ(ierr);
  ierr = TaoRegisterDynamic("tao_ntl",path,"TaoCreate_NTL",TaoCreate_NTL); CHKERRQ(ierr);
  ierr = TaoRegisterDynamic("tao_cg",path,"TaoCreate_CG",TaoCreate_CG); CHKERRQ(ierr);
  ierr = TaoRegisterDynamic("tao_tron",path,"TaoCreate_TRON",TaoCreate_TRON); CHKERRQ(ierr);

  ierr = TaoRegisterDynamic("tao_blmvm",path,"TaoCreate_BLMVM",TaoCreate_BLMVM); CHKERRQ(ierr);
  ierr = TaoRegisterDynamic("tao_bqpip",path,"TaoCreate_BQPIP",TaoCreate_BQPIP); CHKERRQ(ierr);

  ierr = TaoRegisterDynamic("tao_gpcg",path,"TaoCreate_GPCG",TaoCreate_GPCG); CHKERRQ(ierr);
  ierr = TaoRegisterDynamic("tao_nm",path,"TaoCreate_NM",TaoCreate_NM); CHKERRQ(ierr);

  ierr = TaoRegisterDynamic("tao_pounders",path,"TaoCreate_POUNDERS",TaoCreate_POUNDERS); CHKERRQ(ierr);
  ierr = TaoRegisterDynamic("tao_pounder",path,"TaoCreate_POUNDER",TaoCreate_POUNDER); CHKERRQ(ierr);



  ierr = TaoRegisterDynamic("tao_lcl",path,"TaoCreate_LCL",TaoCreate_LCL); CHKERRQ(ierr);
  ierr = TaoRegisterDynamic("tao_fd_test",path,"TaoCreate_FD",TaoCreate_FD); CHKERRQ(ierr);
  //  ierr = TaoRegisterDynamic("tao_lm",path,"TaoCreate_LM",TaoCreate_LM); CHKERRQ(ierr);
/*
  ierr = TaoRegisterDynamic("tao_bnls",path,"TaoCreate_BNLS",TaoCreate_BNLS); CHKERRQ(ierr);
  ierr = TaoRegisterDynamic("tao_nm",path,"TaoCreate_NM",TaoCreate_NM); CHKERRQ(ierr);

  ierr = TaoRegisterDynamic("tao_ssils",path,"TaoCreate_SSILS",TaoCreate_SSILS); CHKERRQ(ierr);
  ierr = TaoRegisterDynamic("tao_ssfls",path,"TaoCreate_SSFLS",TaoCreate_SSFLS); CHKERRQ(ierr);
  ierr = TaoRegisterDynamic("tao_asils",path,"TaoCreate_ASILS",TaoCreate_ASILS); CHKERRQ(ierr);
  ierr = TaoRegisterDynamic("tao_asfls",path,"TaoCreate_ASFLS",TaoCreate_ASFLS); CHKERRQ(ierr);
  ierr = TaoRegisterDynamic("tao_isils",path,"TaoCreate_ISILS",TaoCreate_ISILS); CHKERRQ(ierr);
  ierr = TaoRegisterDynamic("tao_kt",path,"TaoCreate_KT",TaoCreate_KT); CHKERRQ(ierr);
  ierr = TaoRegisterDynamic("tao_rscs",path,"TaoCreate_RSCS",TaoCreate_RSCS); CHKERRQ(ierr);
  ierr = TaoRegisterDynamic("tao_icp",path,"TaoCreate_ICP",TaoCreate_ICP); CHKERRQ(ierr);

*/

  
  PetscFunctionReturn(0);
}
    

