#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: itregis.c,v 1.33 1998/04/22 14:24:29 curfman Exp curfman $";
#endif

#include "src/ksp/kspimpl.h"  /*I "ksp.h" I*/

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

  
/*
      This is used by KSPSetType() to make sure that at least one 
    KSPRegisterAll() is called. In general, if there is more than one
    DLL then KSPRegisterAll() may be called several times.
*/
extern int KSPRegisterAllCalled;

#undef __FUNC__  
#define __FUNC__ "KSPRegisterAll"
/*@C
  KSPRegisterAll - Registers all of the Krylov subspace methods in the KSP package.

  Not Collective

.keywords: KSP, register, all

.seealso:  KSPRegisterDestroy()
@*/
int KSPRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  KSPRegisterAllCalled = 1;

  ierr = KSPRegister(KSPCG,         path,"KSPCreate_CG",        KSPCreate_CG);CHKERRQ(ierr);
  ierr = KSPRegister(KSPRICHARDSON, path,"KSPCreate_Richardson",KSPCreate_Richardson);CHKERRQ(ierr);
  ierr = KSPRegister(KSPCHEBYCHEV,  path,"KSPCreate_Chebychev", KSPCreate_Chebychev);CHKERRQ(ierr);
  ierr = KSPRegister(KSPGMRES,      path,"KSPCreate_GMRES",     KSPCreate_GMRES);CHKERRQ(ierr);
  ierr = KSPRegister(KSPTCQMR,      path,"KSPCreate_TCQMR",     KSPCreate_TCQMR);CHKERRQ(ierr);
  ierr = KSPRegister(KSPBCGS,       path,"KSPCreate_BCGS",      KSPCreate_BCGS);CHKERRQ(ierr);
  ierr = KSPRegister(KSPCGS,        path,"KSPCreate_CGS",       KSPCreate_CGS);CHKERRQ(ierr);
  ierr = KSPRegister(KSPTFQMR,      path,"KSPCreate_TFQMR",     KSPCreate_TFQMR);CHKERRQ(ierr);
  ierr = KSPRegister(KSPCR,         path,"KSPCreate_CR",        KSPCreate_CR);CHKERRQ(ierr);
  ierr = KSPRegister(KSPLSQR,       path,"KSPCreate_LSQR",      KSPCreate_LSQR);CHKERRQ(ierr);
  ierr = KSPRegister(KSPPREONLY,    path,"KSPCreate_PREONLY",   KSPCreate_PREONLY);CHKERRQ(ierr);
  ierr = KSPRegister(KSPQCG,        path,"KSPCreate_QCG",       KSPCreate_QCG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
