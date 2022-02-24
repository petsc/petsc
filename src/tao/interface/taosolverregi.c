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
PETSC_EXTERN PetscErrorCode TaoCreate_PDIPM(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_ADMM(Tao);
PETSC_EXTERN PetscErrorCode TaoCreate_ALMM(Tao);
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
    "",
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
const char **TaoConvergedReasons = TaoConvergedReasons_Shifted - TAO_DIVERGED_USER;

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
#endif

  PetscFunctionBegin;
  if (TaoRegisterAllCalled) PetscFunctionReturn(0);
  TaoRegisterAllCalled = PETSC_TRUE;
#if !defined(PETSC_USE_COMPLEX)
  CHKERRQ(TaoRegister(TAOLMVM,TaoCreate_LMVM));
  CHKERRQ(TaoRegister(TAONLS,TaoCreate_NLS));
  CHKERRQ(TaoRegister(TAONTR,TaoCreate_NTR));
  CHKERRQ(TaoRegister(TAONTL,TaoCreate_NTL));
  CHKERRQ(TaoRegister(TAOCG,TaoCreate_CG));
  CHKERRQ(TaoRegister(TAOTRON,TaoCreate_TRON));
  CHKERRQ(TaoRegister(TAOOWLQN,TaoCreate_OWLQN));
  CHKERRQ(TaoRegister(TAOBMRM,TaoCreate_BMRM));
  CHKERRQ(TaoRegister(TAOBLMVM,TaoCreate_BLMVM));
  CHKERRQ(TaoRegister(TAOBQNLS,TaoCreate_BQNLS));
  CHKERRQ(TaoRegister(TAOBNCG,TaoCreate_BNCG));
  CHKERRQ(TaoRegister(TAOBNLS,TaoCreate_BNLS));
  CHKERRQ(TaoRegister(TAOBNTR,TaoCreate_BNTR));
  CHKERRQ(TaoRegister(TAOBNTL,TaoCreate_BNTL));
  CHKERRQ(TaoRegister(TAOBQNKLS,TaoCreate_BQNKLS));
  CHKERRQ(TaoRegister(TAOBQNKTR,TaoCreate_BQNKTR));
  CHKERRQ(TaoRegister(TAOBQNKTL,TaoCreate_BQNKTL));
  CHKERRQ(TaoRegister(TAOBQPIP,TaoCreate_BQPIP));
  CHKERRQ(TaoRegister(TAOGPCG,TaoCreate_GPCG));
  CHKERRQ(TaoRegister(TAONM,TaoCreate_NM));
  CHKERRQ(TaoRegister(TAOPOUNDERS,TaoCreate_POUNDERS));
  CHKERRQ(TaoRegister(TAOBRGN,TaoCreate_BRGN));
  CHKERRQ(TaoRegister(TAOLCL,TaoCreate_LCL));
  CHKERRQ(TaoRegister(TAOSSILS,TaoCreate_SSILS));
  CHKERRQ(TaoRegister(TAOSSFLS,TaoCreate_SSFLS));
  CHKERRQ(TaoRegister(TAOASILS,TaoCreate_ASILS));
  CHKERRQ(TaoRegister(TAOASFLS,TaoCreate_ASFLS));
  CHKERRQ(TaoRegister(TAOIPM,TaoCreate_IPM));
  CHKERRQ(TaoRegister(TAOPDIPM,TaoCreate_PDIPM));
  CHKERRQ(TaoRegister(TAOSHELL,TaoCreate_Shell));
  CHKERRQ(TaoRegister(TAOADMM,TaoCreate_ADMM));
  CHKERRQ(TaoRegister(TAOALMM,TaoCreate_ALMM));
#endif
  PetscFunctionReturn(0);
}
