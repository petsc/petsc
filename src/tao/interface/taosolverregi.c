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
  TaoRegisterAll - Registers all of the optimization methods in the Tao
  package.

  Not Collective

  Level: developer

.seealso `TaoRegister()`, `TaoRegisterDestroy()`
@*/
PetscErrorCode TaoRegisterAll(void)
{
#if !defined(PETSC_USE_COMPLEX)
#endif

  PetscFunctionBegin;
  if (TaoRegisterAllCalled) PetscFunctionReturn(0);
  TaoRegisterAllCalled = PETSC_TRUE;
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(TaoRegister(TAOLMVM,TaoCreate_LMVM));
  PetscCall(TaoRegister(TAONLS,TaoCreate_NLS));
  PetscCall(TaoRegister(TAONTR,TaoCreate_NTR));
  PetscCall(TaoRegister(TAONTL,TaoCreate_NTL));
  PetscCall(TaoRegister(TAOCG,TaoCreate_CG));
  PetscCall(TaoRegister(TAOTRON,TaoCreate_TRON));
  PetscCall(TaoRegister(TAOOWLQN,TaoCreate_OWLQN));
  PetscCall(TaoRegister(TAOBMRM,TaoCreate_BMRM));
  PetscCall(TaoRegister(TAOBLMVM,TaoCreate_BLMVM));
  PetscCall(TaoRegister(TAOBQNLS,TaoCreate_BQNLS));
  PetscCall(TaoRegister(TAOBNCG,TaoCreate_BNCG));
  PetscCall(TaoRegister(TAOBNLS,TaoCreate_BNLS));
  PetscCall(TaoRegister(TAOBNTR,TaoCreate_BNTR));
  PetscCall(TaoRegister(TAOBNTL,TaoCreate_BNTL));
  PetscCall(TaoRegister(TAOBQNKLS,TaoCreate_BQNKLS));
  PetscCall(TaoRegister(TAOBQNKTR,TaoCreate_BQNKTR));
  PetscCall(TaoRegister(TAOBQNKTL,TaoCreate_BQNKTL));
  PetscCall(TaoRegister(TAOBQPIP,TaoCreate_BQPIP));
  PetscCall(TaoRegister(TAOGPCG,TaoCreate_GPCG));
  PetscCall(TaoRegister(TAONM,TaoCreate_NM));
  PetscCall(TaoRegister(TAOPOUNDERS,TaoCreate_POUNDERS));
  PetscCall(TaoRegister(TAOBRGN,TaoCreate_BRGN));
  PetscCall(TaoRegister(TAOLCL,TaoCreate_LCL));
  PetscCall(TaoRegister(TAOSSILS,TaoCreate_SSILS));
  PetscCall(TaoRegister(TAOSSFLS,TaoCreate_SSFLS));
  PetscCall(TaoRegister(TAOASILS,TaoCreate_ASILS));
  PetscCall(TaoRegister(TAOASFLS,TaoCreate_ASFLS));
  PetscCall(TaoRegister(TAOIPM,TaoCreate_IPM));
  PetscCall(TaoRegister(TAOPDIPM,TaoCreate_PDIPM));
  PetscCall(TaoRegister(TAOSHELL,TaoCreate_Shell));
  PetscCall(TaoRegister(TAOADMM,TaoCreate_ADMM));
  PetscCall(TaoRegister(TAOALMM,TaoCreate_ALMM));
#endif
  PetscFunctionReturn(0);
}
