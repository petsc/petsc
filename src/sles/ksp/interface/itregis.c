/*$Id: itregis.c,v 1.46 1999/11/10 03:20:38 bsmith Exp bsmith $*/

#include "src/sles/ksp/kspimpl.h"  /*I "ksp.h" I*/

EXTERN_C_BEGIN
extern int KSPCreate_Richardson(KSP);
extern int KSPCreate_Chebychev(KSP);
extern int KSPCreate_CG(KSP);
extern int KSPCreate_TCQMR(KSP);
extern int KSPCreate_GMRES(KSP);
extern int KSPCreate_BCGS(KSP);
extern int KSPCreate_CGS(KSP);
extern int KSPCreate_TFQMR(KSP);
extern int KSPCreate_LSQR(KSP);
extern int KSPCreate_PREONLY(KSP);
extern int KSPCreate_CR(KSP);
extern int KSPCreate_QCG(KSP);
extern int KSPCreate_BiCG(KSP);
extern int KSPCreate_FGMRES(KSP);
EXTERN_C_END
  
/*
    This is used by KSPSetType() to make sure that at least one 
    KSPRegisterAll() is called. In general, if there is more than one
    DLL, then KSPRegisterAll() may be called several times.
*/
extern PetscTruth KSPRegisterAllCalled;

#undef __FUNC__  
#define __FUNC__ "KSPRegisterAll"
/*@C
  KSPRegisterAll - Registers all of the Krylov subspace methods in the KSP package.

  Not Collective

  Level: advanced

.keywords: KSP, register, all

.seealso:  KSPRegisterDestroy()
@*/
int KSPRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  KSPRegisterAllCalled = PETSC_TRUE;

  ierr = KSPRegisterDynamic(KSPCG,         path,"KSPCreate_CG",        KSPCreate_CG);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPRICHARDSON, path,"KSPCreate_Richardson",KSPCreate_Richardson);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPCHEBYCHEV,  path,"KSPCreate_Chebychev", KSPCreate_Chebychev);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPGMRES,      path,"KSPCreate_GMRES",     KSPCreate_GMRES);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPTCQMR,      path,"KSPCreate_TCQMR",     KSPCreate_TCQMR);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPBCGS,       path,"KSPCreate_BCGS",      KSPCreate_BCGS);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPCGS,        path,"KSPCreate_CGS",       KSPCreate_CGS);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPTFQMR,      path,"KSPCreate_TFQMR",     KSPCreate_TFQMR);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPCR,         path,"KSPCreate_CR",        KSPCreate_CR);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPLSQR,       path,"KSPCreate_LSQR",      KSPCreate_LSQR);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPPREONLY,    path,"KSPCreate_PREONLY",   KSPCreate_PREONLY);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPQCG,        path,"KSPCreate_QCG",       KSPCreate_QCG);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPBICG,       path,"KSPCreate_BiCG",      KSPCreate_BiCG);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPFGMRES,     path,"KSPCreate_FGMRES",    KSPCreate_FGMRES);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
