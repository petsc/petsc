#define TAO_DLL

#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/


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
PETSC_EXTERN PetscErrorCode TaoCreate_BQNLS(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_BNCG(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_BNLS(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_BNTR(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_BNTL(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_BQNKLS(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_BQNKTR(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_BQNKTL(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_GPCG(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_BQPIP(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_POUNDERS(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_BRGN(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_LCL(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_SSILS(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_SSFLS(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_ASILS(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_ASFLS(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_IPM(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_ADMM(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_Shell(Tao);

/*
   Offset the convergence reasons so negative number represent diverged and
   positive represent converged.
*/
const char *TaoConvergedReasons_Shifted[] = {
    "DIVERGED_USER",
    "DIVERGED_TR_REDUCTION",
    "DIVERGED_LS_FAILURE",
    "DIVERGED_MAXFCN",
    "DIVERGED_NAN",
    "DIVERGED_MAXITS",
    "DIVERGED_FUNCTION_DOMAIN",

    "CONTINUE_ITERATING",

    "",
    "",
    "CONVERGED_GATOL",
    "CONVERGED_GRTOL",
    "CONVERGED_GTTOL",
    "CONVERGED_STEPTOL",
    "CONVERGED_MINF",
    "CONVERGED_USER" };
const char **TaoConvergedReasons = TaoConvergedReasons_Shifted + 7;

/*@C
  TaoRegisterAll - Registers all of the minimization methods in the TAO
  package.

  Not Collective

  Level: developer

.seealso TaoRegister(), TaoRegisterDestroy()
@*/
PetscErrorCode TaoRegisterAll(void)
{
#if !defined(PETSC_USE_COMPLEX)
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
  if (TaoRegisterAllCalled) PetscFunctionReturn(0);
  TaoRegisterAllCalled = PETSC_TRUE;
#if !defined(PETSC_USE_COMPLEX)
  ierr = TaoRegister(TAOLMVM,TaoCreate_LMVM);CHKERRQ(ierr);
  ierr = TaoRegister(TAONLS,TaoCreate_NLS);CHKERRQ(ierr);
  ierr = TaoRegister(TAONTR,TaoCreate_NTR);CHKERRQ(ierr);
  ierr = TaoRegister(TAONTL,TaoCreate_NTL);CHKERRQ(ierr);
  ierr = TaoRegister(TAOCG,TaoCreate_CG);CHKERRQ(ierr);
  ierr = TaoRegister(TAOTRON,TaoCreate_TRON);CHKERRQ(ierr);
  ierr = TaoRegister(TAOOWLQN,TaoCreate_OWLQN);CHKERRQ(ierr);
  ierr = TaoRegister(TAOBMRM,TaoCreate_BMRM);CHKERRQ(ierr);
  ierr = TaoRegister(TAOBLMVM,TaoCreate_BLMVM);CHKERRQ(ierr);
  ierr = TaoRegister(TAOBQNLS,TaoCreate_BQNLS);CHKERRQ(ierr);
  ierr = TaoRegister(TAOBNCG,TaoCreate_BNCG);CHKERRQ(ierr);
  ierr = TaoRegister(TAOBNLS,TaoCreate_BNLS);CHKERRQ(ierr);
  ierr = TaoRegister(TAOBNTR,TaoCreate_BNTR);CHKERRQ(ierr);
  ierr = TaoRegister(TAOBNTL,TaoCreate_BNTL);CHKERRQ(ierr);
  ierr = TaoRegister(TAOBQNKLS,TaoCreate_BQNKLS);CHKERRQ(ierr);
  ierr = TaoRegister(TAOBQNKTR,TaoCreate_BQNKTR);CHKERRQ(ierr);
  ierr = TaoRegister(TAOBQNKTL,TaoCreate_BQNKTL);CHKERRQ(ierr);
  ierr = TaoRegister(TAOBQPIP,TaoCreate_BQPIP);CHKERRQ(ierr);
  ierr = TaoRegister(TAOGPCG,TaoCreate_GPCG);CHKERRQ(ierr);
  ierr = TaoRegister(TAONM,TaoCreate_NM);CHKERRQ(ierr);
  ierr = TaoRegister(TAOPOUNDERS,TaoCreate_POUNDERS);CHKERRQ(ierr);
  ierr = TaoRegister(TAOBRGN,TaoCreate_BRGN);CHKERRQ(ierr);
  ierr = TaoRegister(TAOLCL,TaoCreate_LCL);CHKERRQ(ierr);
  ierr = TaoRegister(TAOSSILS,TaoCreate_SSILS);CHKERRQ(ierr);
  ierr = TaoRegister(TAOSSFLS,TaoCreate_SSFLS);CHKERRQ(ierr);
  ierr = TaoRegister(TAOASILS,TaoCreate_ASILS);CHKERRQ(ierr);
  ierr = TaoRegister(TAOASFLS,TaoCreate_ASFLS);CHKERRQ(ierr);
  ierr = TaoRegister(TAOIPM,TaoCreate_IPM);CHKERRQ(ierr);
  ierr = TaoRegister(TAOSHELL,TaoCreate_Shell);CHKERRQ(ierr);
  ierr = TaoRegister(TAOADMM,TaoCreate_ADMM);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
