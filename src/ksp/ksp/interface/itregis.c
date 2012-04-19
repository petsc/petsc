
#include <petsc-private/kspimpl.h>  /*I "petscksp.h" I*/

EXTERN_C_BEGIN
extern PetscErrorCode  KSPCreate_Richardson(KSP);
extern PetscErrorCode  KSPCreate_Chebychev(KSP);
extern PetscErrorCode  KSPCreate_CG(KSP);
extern PetscErrorCode  KSPCreate_CGNE(KSP);
extern PetscErrorCode  KSPCreate_NASH(KSP);
extern PetscErrorCode  KSPCreate_STCG(KSP);
extern PetscErrorCode  KSPCreate_GLTR(KSP);
extern PetscErrorCode  KSPCreate_TCQMR(KSP);
extern PetscErrorCode  KSPCreate_GMRES(KSP);
extern PetscErrorCode  KSPCreate_BCGS(KSP);
extern PetscErrorCode  KSPCreate_IBCGS(KSP);
extern PetscErrorCode  KSPCreate_BCGSL(KSP);
extern PetscErrorCode  KSPCreate_CGS(KSP);
extern PetscErrorCode  KSPCreate_TFQMR(KSP);
extern PetscErrorCode  KSPCreate_LSQR(KSP);
extern PetscErrorCode  KSPCreate_PREONLY(KSP);
extern PetscErrorCode  KSPCreate_CR(KSP);
extern PetscErrorCode  KSPCreate_QCG(KSP);
extern PetscErrorCode  KSPCreate_BiCG(KSP);
extern PetscErrorCode  KSPCreate_FGMRES(KSP);
extern PetscErrorCode  KSPCreate_MINRES(KSP);
extern PetscErrorCode  KSPCreate_SYMMLQ(KSP);
extern PetscErrorCode  KSPCreate_LGMRES(KSP);
extern PetscErrorCode  KSPCreate_LCD(KSP);
extern PetscErrorCode  KSPCreate_GCR(KSP);
extern PetscErrorCode  KSPCreate_PGMRES(KSP);
extern PetscErrorCode  KSPCreate_SpecEst(KSP);
#if !defined(PETSC_USE_COMPLEX)
extern PetscErrorCode  KSPCreate_DGMRES(KSP);
#endif
EXTERN_C_END
  
/*
    This is used by KSPSetType() to make sure that at least one 
    KSPRegisterAll() is called. In general, if there is more than one
    DLL, then KSPRegisterAll() may be called several times.
*/
extern PetscBool  KSPRegisterAllCalled;

#undef __FUNCT__  
#define __FUNCT__ "KSPRegisterAll"
/*@C
  KSPRegisterAll - Registers all of the Krylov subspace methods in the KSP package.

  Not Collective

  Level: advanced

.keywords: KSP, register, all

.seealso:  KSPRegisterDestroy()
@*/
PetscErrorCode  KSPRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  KSPRegisterAllCalled = PETSC_TRUE;

  ierr = KSPRegisterDynamic(KSPCG,         path,"KSPCreate_CG",        KSPCreate_CG);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPCGNE,       path,"KSPCreate_CGNE",      KSPCreate_CGNE);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPNASH,       path,"KSPCreate_NASH",      KSPCreate_NASH);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPSTCG,       path,"KSPCreate_STCG",      KSPCreate_STCG);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPGLTR,       path,"KSPCreate_GLTR",      KSPCreate_GLTR);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPRICHARDSON, path,"KSPCreate_Richardson",KSPCreate_Richardson);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPCHEBYCHEV,  path,"KSPCreate_Chebychev", KSPCreate_Chebychev);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPGMRES,      path,"KSPCreate_GMRES",     KSPCreate_GMRES);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPTCQMR,      path,"KSPCreate_TCQMR",     KSPCreate_TCQMR);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPBCGS,       path,"KSPCreate_BCGS",      KSPCreate_BCGS);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPIBCGS,      path,"KSPCreate_IBCGS",     KSPCreate_IBCGS);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPBCGSL,      path,"KSPCreate_BCGSL",     KSPCreate_BCGSL);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPCGS,        path,"KSPCreate_CGS",       KSPCreate_CGS);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPTFQMR,      path,"KSPCreate_TFQMR",     KSPCreate_TFQMR);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPCR,         path,"KSPCreate_CR",        KSPCreate_CR);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPLSQR,       path,"KSPCreate_LSQR",      KSPCreate_LSQR);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPPREONLY,    path,"KSPCreate_PREONLY",   KSPCreate_PREONLY);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPQCG,        path,"KSPCreate_QCG",       KSPCreate_QCG);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPBICG,       path,"KSPCreate_BiCG",      KSPCreate_BiCG);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPFGMRES,     path,"KSPCreate_FGMRES",    KSPCreate_FGMRES);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPMINRES,     path,"KSPCreate_MINRES",    KSPCreate_MINRES);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPSYMMLQ,     path,"KSPCreate_SYMMLQ",    KSPCreate_SYMMLQ);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPLGMRES,     path,"KSPCreate_LGMRES",    KSPCreate_LGMRES);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPLCD,        path,"KSPCreate_LCD",       KSPCreate_LCD);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPGCR,        path,"KSPCreate_GCR",       KSPCreate_GCR);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPPGMRES,     path,"KSPCreate_PGMRES",    KSPCreate_PGMRES);CHKERRQ(ierr);
  ierr = KSPRegisterDynamic(KSPSPECEST,    path,"KSPCreate_SpecEst",  KSPCreate_SpecEst);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = KSPRegisterDynamic(KSPDGMRES,     path,"KSPCreate_DGMRES", KSPCreate_DGMRES); CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

