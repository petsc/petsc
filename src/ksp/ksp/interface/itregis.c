
#include <petsc-private/kspimpl.h>  /*I "petscksp.h" I*/

PETSC_EXTERN PetscErrorCode KSPCreate_Richardson(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_Chebyshev(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_CG(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_GROPPCG(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_PIPECG(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_CGNE(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_NASH(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_STCG(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_GLTR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_TCQMR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_GMRES(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_BCGS(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_IBCGS(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_FBCGS(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_FBCGSR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_BCGSL(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_CGS(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_TFQMR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_LSQR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_PREONLY(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_CR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_PIPECR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_QCG(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_BiCG(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_FGMRES(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_MINRES(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_SYMMLQ(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_LGMRES(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_LCD(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_GCR(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_PGMRES(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_SpecEst(KSP);
#if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN PetscErrorCode KSPCreate_DGMRES(KSP);
#endif

/*
    This is used by KSPSetType() to make sure that at least one
    KSPRegisterAll() is called. In general, if there is more than one
    DLL, then KSPRegisterAll() may be called several times.
*/
extern PetscBool KSPRegisterAllCalled;

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

  ierr = KSPRegister(KSPCG,         path,"KSPCreate_CG",        KSPCreate_CG);CHKERRQ(ierr);
  ierr = KSPRegister(KSPGROPPCG,    path,"KSPCreate_GROPPCG",   KSPCreate_GROPPCG);CHKERRQ(ierr);
  ierr = KSPRegister(KSPPIPECG,     path,"KSPCreate_PIPECG",    KSPCreate_PIPECG);CHKERRQ(ierr);
  ierr = KSPRegister(KSPCGNE,       path,"KSPCreate_CGNE",      KSPCreate_CGNE);CHKERRQ(ierr);
  ierr = KSPRegister(KSPNASH,       path,"KSPCreate_NASH",      KSPCreate_NASH);CHKERRQ(ierr);
  ierr = KSPRegister(KSPSTCG,       path,"KSPCreate_STCG",      KSPCreate_STCG);CHKERRQ(ierr);
  ierr = KSPRegister(KSPGLTR,       path,"KSPCreate_GLTR",      KSPCreate_GLTR);CHKERRQ(ierr);
  ierr = KSPRegister(KSPRICHARDSON, path,"KSPCreate_Richardson",KSPCreate_Richardson);CHKERRQ(ierr);
  ierr = KSPRegister(KSPCHEBYSHEV,  path,"KSPCreate_Chebyshev", KSPCreate_Chebyshev);CHKERRQ(ierr);
  ierr = KSPRegister(KSPGMRES,      path,"KSPCreate_GMRES",     KSPCreate_GMRES);CHKERRQ(ierr);
  ierr = KSPRegister(KSPTCQMR,      path,"KSPCreate_TCQMR",     KSPCreate_TCQMR);CHKERRQ(ierr);
  ierr = KSPRegister(KSPBCGS,       path,"KSPCreate_BCGS",      KSPCreate_BCGS);CHKERRQ(ierr);
  ierr = KSPRegister(KSPIBCGS,      path,"KSPCreate_IBCGS",     KSPCreate_IBCGS);CHKERRQ(ierr);
  ierr = KSPRegister(KSPFBCGS,      path,"KSPCreate_FBCGS",     KSPCreate_FBCGS);CHKERRQ(ierr);
  ierr = KSPRegister(KSPFBCGSR,     path,"KSPCreate_FBCGSR",    KSPCreate_FBCGSR);CHKERRQ(ierr);
  ierr = KSPRegister(KSPBCGSL,      path,"KSPCreate_BCGSL",     KSPCreate_BCGSL);CHKERRQ(ierr);
  ierr = KSPRegister(KSPCGS,        path,"KSPCreate_CGS",       KSPCreate_CGS);CHKERRQ(ierr);
  ierr = KSPRegister(KSPTFQMR,      path,"KSPCreate_TFQMR",     KSPCreate_TFQMR);CHKERRQ(ierr);
  ierr = KSPRegister(KSPCR,         path,"KSPCreate_CR",        KSPCreate_CR);CHKERRQ(ierr);
  ierr = KSPRegister(KSPPIPECR,     path,"KSPCreate_PIPECR",    KSPCreate_PIPECR);CHKERRQ(ierr);
  ierr = KSPRegister(KSPLSQR,       path,"KSPCreate_LSQR",      KSPCreate_LSQR);CHKERRQ(ierr);
  ierr = KSPRegister(KSPPREONLY,    path,"KSPCreate_PREONLY",   KSPCreate_PREONLY);CHKERRQ(ierr);
  ierr = KSPRegister(KSPQCG,        path,"KSPCreate_QCG",       KSPCreate_QCG);CHKERRQ(ierr);
  ierr = KSPRegister(KSPBICG,       path,"KSPCreate_BiCG",      KSPCreate_BiCG);CHKERRQ(ierr);
  ierr = KSPRegister(KSPFGMRES,     path,"KSPCreate_FGMRES",    KSPCreate_FGMRES);CHKERRQ(ierr);
  ierr = KSPRegister(KSPMINRES,     path,"KSPCreate_MINRES",    KSPCreate_MINRES);CHKERRQ(ierr);
  ierr = KSPRegister(KSPSYMMLQ,     path,"KSPCreate_SYMMLQ",    KSPCreate_SYMMLQ);CHKERRQ(ierr);
  ierr = KSPRegister(KSPLGMRES,     path,"KSPCreate_LGMRES",    KSPCreate_LGMRES);CHKERRQ(ierr);
  ierr = KSPRegister(KSPLCD,        path,"KSPCreate_LCD",       KSPCreate_LCD);CHKERRQ(ierr);
  ierr = KSPRegister(KSPGCR,        path,"KSPCreate_GCR",       KSPCreate_GCR);CHKERRQ(ierr);
  ierr = KSPRegister(KSPPGMRES,     path,"KSPCreate_PGMRES",    KSPCreate_PGMRES);CHKERRQ(ierr);
  ierr = KSPRegister(KSPSPECEST,    path,"KSPCreate_SpecEst",  KSPCreate_SpecEst);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = KSPRegister(KSPDGMRES,     path,"KSPCreate_DGMRES", KSPCreate_DGMRES);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

