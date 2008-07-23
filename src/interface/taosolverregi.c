#define TAOSOLVER_DLL

#include "include/private/taosolver_impl.h"


EXTERN_C_BEGIN
extern PetscErrorCode TaoSolverCreate_LMVM(TaoSolver);
/*
extern int TaoCreate_NLS(TAO_SOLVER);
extern int TaoCreate_NTR(TAO_SOLVER);
extern int TaoCreate_NTL(TAO_SOLVER);
extern int TaoCreate_CG(TAO_SOLVER);

extern int TaoCreate_TRON(TAO_SOLVER);
extern int TaoCreate_BQPIP(TAO_SOLVER);
extern int TaoCreate_BLMVM(TAO_SOLVER);
extern int TaoCreate_BNLS(TAO_SOLVER);
extern int TaoCreate_GPCG(TAO_SOLVER);
extern int TaoCreate_QPIP(TAO_SOLVER);

extern int TaoCreate_NLSQ(TAO_SOLVER);
extern int TaoCreate_BLM(TAO_SOLVER);
extern int TaoCreate_SSILS(TAO_SOLVER);
extern int TaoCreate_SSFLS(TAO_SOLVER);
extern int TaoCreate_ASILS(TAO_SOLVER);
extern int TaoCreate_ASFLS(TAO_SOLVER);
extern int TaoCreate_ISILS(TAO_SOLVER);
extern int TaoCreate_KT(TAO_SOLVER);
extern int TaoCreate_BCG(TAO_SOLVER);
extern int TaoCreate_RSCS(TAO_SOLVER);
extern int TaoCreate_ICP(TAO_SOLVER);
extern int TaoCreate_NM(TAO_SOLVER);
extern int TaoCreate_FD(TAO_SOLVER);
*/
EXTERN_C_END

/* 
   Offset the convergence reasons so negative number represent diverged and
   positive represent converged.
*/
const char *TaoSolverConvergedReasons_Shifted[] = {
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
const char **TaoSolverConvergedReasons = TaoSolverConvergedReasons_Shifted + 8;

						   


extern PetscTruth TaoSolverRegisterAllCalled;

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

/*
  ierr = TaoSolverRegisterDynamic("tao_nls",path,"TaoCreate_NLS",TaoCreate_NLS); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_cg",path,"TaoCreate_CG",TaoCreate_CG); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_bqpip",path,"TaoCreate_BQPIP",TaoCreate_BQPIP); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_blmvm",path,"TaoCreate_BLMVM",TaoCreate_BLMVM); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_bnls",path,"TaoCreate_BNLS",TaoCreate_BNLS); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_tron",path,"TaoCreate_TRON",TaoCreate_TRON); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_nm",path,"TaoCreate_NM",TaoCreate_NM); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_ntl",path,"TaoCreate_NTL",TaoCreate_NTL); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_ntr",path,"TaoCreate_NTR",TaoCreate_NTR); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_gpcg",path,"TaoCreate_GPCG",TaoCreate_GPCG); CHKERRQ(ierr);

  ierr = TaoSolverRegisterDynamic("tao_ssils",path,"TaoCreate_SSILS",TaoCreate_SSILS); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_ssfls",path,"TaoCreate_SSFLS",TaoCreate_SSFLS); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_asils",path,"TaoCreate_ASILS",TaoCreate_ASILS); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_asfls",path,"TaoCreate_ASFLS",TaoCreate_ASFLS); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_isils",path,"TaoCreate_ISILS",TaoCreate_ISILS); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_kt",path,"TaoCreate_KT",TaoCreate_KT); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_rscs",path,"TaoCreate_RSCS",TaoCreate_RSCS); CHKERRQ(ierr);
  ierr = TaoSolverRegisterDynamic("tao_icp",path,"TaoCreate_ICP",TaoCreate_ICP); CHKERRQ(ierr);


  ierr = TaoSolverRegisterDynamic("tao_fd_test",path,"TaoCreate_FD",TaoCreate_FD); CHKERRQ(ierr);
*/
  
  PetscFunctionReturn(0);
}
    

