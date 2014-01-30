#define TAOSOLVER_DLL

#include <petsc-private/taosolverimpl.h> /*I "taosolver.h" I*/


PETSC_EXTERN PetscErrorCode TaoCreate_LMVM(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_NLS(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_NTR(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_NTL(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_NM(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_CG(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_TRON(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_OWLQN(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_BMRM(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_BLMVM(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_GPCG(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_BQPIP(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_POUNDERS(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_FD(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_LCL(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_SSILS(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_SSFLS(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_ASILS(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_ASFLS(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoCreate_IPM(TaoSolver);

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
const char **TaoSolverTerminationReasons = TaoSolverTerminationReasons_Shifted + 7;
extern PetscBool TaoSolverRegisterAllCalled;

#undef __FUNCT__
#define __FUNCT__ "TaoSolverRegisterAll"
/*@C
  TaoSolverRegisterAll - Registers all of the minimization methods in the TAO
  package.

  Not Collective

  Level: developer

.seealso TaoSolverRegister(), TaoSolverRegisterDestroy()
@*/
PetscErrorCode TaoSolverRegisterAll()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TaoSolverRegisterAllCalled = PETSC_TRUE;
  ierr = TaoSolverRegister("tao_lmvm",TaoCreate_LMVM);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_nls",TaoCreate_NLS);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_ntr",TaoCreate_NTR);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_ntl",TaoCreate_NTL);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_cg",TaoCreate_CG);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_tron",TaoCreate_TRON);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_owlqn",TaoCreate_OWLQN);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_bmrm",TaoCreate_BMRM);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_blmvm",TaoCreate_BLMVM);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_bqpip",TaoCreate_BQPIP);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_gpcg",TaoCreate_GPCG);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_nm",TaoCreate_NM);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_pounders",TaoCreate_POUNDERS);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_lcl",TaoCreate_LCL);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_ssils",TaoCreate_SSILS);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_ssfls",TaoCreate_SSFLS);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_asils",TaoCreate_ASILS);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_asfls",TaoCreate_ASFLS);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_ipm",TaoCreate_IPM);CHKERRQ(ierr);
  ierr = TaoSolverRegister("tao_fd_test",TaoCreate_FD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


