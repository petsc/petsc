#define TAO_DLL

#include <petsc-private/taoimpl.h> /*I "petsctao.h" I*/


PETSC_EXTERN PetscErrorCode TaoCreate_LMVM(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_NLS(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_NTR(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_NTL(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_NM(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_CG(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_TRON(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_OWLQN(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_BMRM(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_BLMVM(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_GPCG(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_BQPIP(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_POUNDERS(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_FD(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_LCL(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_SSILS(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_SSFLS(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_ASILS(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_ASFLS(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_IPM(Tao);

/*
   Offset the convergence reasons so negative number represent diverged and
   positive represent converged.
*/
const char *TaoTerminationReasons_Shifted[] = {
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
const char **TaoTerminationReasons = TaoTerminationReasons_Shifted + 7;
extern PetscBool TaoRegisterAllCalled;

#undef __FUNCT__
#define __FUNCT__ "TaoRegisterAll"
/*@C
  TaoRegisterAll - Registers all of the minimization methods in the TAO
  package.

  Not Collective

  Level: developer

.seealso TaoRegister(), TaoRegisterDestroy()
@*/
PetscErrorCode TaoRegisterAll()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TaoRegisterAllCalled = PETSC_TRUE;
  ierr = TaoRegister("tao_lmvm",TaoCreate_LMVM);CHKERRQ(ierr);
  ierr = TaoRegister("tao_nls",TaoCreate_NLS);CHKERRQ(ierr);
  ierr = TaoRegister("tao_ntr",TaoCreate_NTR);CHKERRQ(ierr);
  ierr = TaoRegister("tao_ntl",TaoCreate_NTL);CHKERRQ(ierr);
  ierr = TaoRegister("tao_cg",TaoCreate_CG);CHKERRQ(ierr);
  ierr = TaoRegister("tao_tron",TaoCreate_TRON);CHKERRQ(ierr);
  ierr = TaoRegister("tao_owlqn",TaoCreate_OWLQN);CHKERRQ(ierr);
  ierr = TaoRegister("tao_bmrm",TaoCreate_BMRM);CHKERRQ(ierr);
  ierr = TaoRegister("tao_blmvm",TaoCreate_BLMVM);CHKERRQ(ierr);
  ierr = TaoRegister("tao_bqpip",TaoCreate_BQPIP);CHKERRQ(ierr);
  ierr = TaoRegister("tao_gpcg",TaoCreate_GPCG);CHKERRQ(ierr);
  ierr = TaoRegister("tao_nm",TaoCreate_NM);CHKERRQ(ierr);
  ierr = TaoRegister("tao_pounders",TaoCreate_POUNDERS);CHKERRQ(ierr);
  ierr = TaoRegister("tao_lcl",TaoCreate_LCL);CHKERRQ(ierr);
  ierr = TaoRegister("tao_ssils",TaoCreate_SSILS);CHKERRQ(ierr);
  ierr = TaoRegister("tao_ssfls",TaoCreate_SSFLS);CHKERRQ(ierr);
  ierr = TaoRegister("tao_asils",TaoCreate_ASILS);CHKERRQ(ierr);
  ierr = TaoRegister("tao_asfls",TaoCreate_ASFLS);CHKERRQ(ierr);
  ierr = TaoRegister("tao_ipm",TaoCreate_IPM);CHKERRQ(ierr);
  ierr = TaoRegister("tao_fd_test",TaoCreate_FD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


