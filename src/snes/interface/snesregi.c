
#include <private/snesimpl.h>     /*I  "petscsnes.h"  I*/

EXTERN_C_BEGIN
extern PetscErrorCode  SNESCreate_LS(SNES);
extern PetscErrorCode  SNESCreate_TR(SNES);
extern PetscErrorCode  SNESCreate_Test(SNES);
extern PetscErrorCode  SNESCreate_Picard(SNES);
extern PetscErrorCode  SNESCreate_KSPONLY(SNES);
extern PetscErrorCode  SNESCreate_VI(SNES);
extern PetscErrorCode  SNESCreate_NGMRES(SNES);
extern PetscErrorCode  SNESCreate_SORQN(SNES);
extern PetscErrorCode  SNESCreate_QN(SNES);
extern PetscErrorCode  SNESCreate_Shell(SNES);
EXTERN_C_END

const char *SNESConvergedReasons_Shifted[]  = {" "," ","DIVERGED_LOCAL_MIN"," ","DIVERGED_LINE_SEARCH","DIVERGED_MAX_IT",
                                               "DIVERGED_FNORM_NAN","DIVERGED_LINEAR_SOLVE","DIVERGED_FUNCTION_COUNT","DIVERGED_FUNCTION_DOMAIN",
                                               "CONVERGED_ITERATING"," ","CONVERGED_FNORM_ABS","CONVERGED_FNORM_RELATIVE",
                                               "CONVERGED_PNORM_RELATIVE","CONVERGED_ITS"," ","CONVERGED_TR_DELTA","SNESConvergedReason","",0};
const char *const*SNESConvergedReasons = SNESConvergedReasons_Shifted + 10;

/*
      This is used by SNESSetType() to make sure that at least one 
    SNESRegisterAll() is called. In general, if there is more than one
    DLL then SNESRegisterAll() may be called several times.
*/
extern PetscBool  SNESRegisterAllCalled;

#undef __FUNCT__  
#define __FUNCT__ "SNESRegisterAll"
/*@C
   SNESRegisterAll - Registers all of the nonlinear solver methods in the SNES package.

   Not Collective

   Level: advanced

.keywords: SNES, register, all

.seealso:  SNESRegisterDestroy()
@*/
PetscErrorCode  SNESRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  SNESRegisterAllCalled = PETSC_TRUE;

  ierr = SNESRegisterDynamic(SNESLS,     path,"SNESCreate_LS",     SNESCreate_LS);CHKERRQ(ierr);
  ierr = SNESRegisterDynamic(SNESTR,     path,"SNESCreate_TR",     SNESCreate_TR);CHKERRQ(ierr);
  ierr = SNESRegisterDynamic(SNESTEST,   path,"SNESCreate_Test",   SNESCreate_Test);CHKERRQ(ierr);
  ierr = SNESRegisterDynamic(SNESPICARD, path,"SNESCreate_Picard", SNESCreate_Picard);CHKERRQ(ierr);
  ierr = SNESRegisterDynamic(SNESKSPONLY,path,"SNESCreate_KSPONLY",SNESCreate_KSPONLY);CHKERRQ(ierr);
  ierr = SNESRegisterDynamic(SNESVI,     path,"SNESCreate_VI",     SNESCreate_VI);CHKERRQ(ierr);
  ierr = SNESRegisterDynamic(SNESNGMRES, path,"SNESCreate_NGMRES", SNESCreate_NGMRES);CHKERRQ(ierr);
  ierr = SNESRegisterDynamic(SNESSORQN,  path,"SNESCreate_SORQN",  SNESCreate_SORQN);CHKERRQ(ierr);
  ierr = SNESRegisterDynamic(SNESQN,     path,"SNESCreate_QN",     SNESCreate_QN);CHKERRQ(ierr);
  ierr = SNESRegisterDynamic(SNESSHELL,  path,"SNESCreate_Shell",  SNESCreate_Shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
