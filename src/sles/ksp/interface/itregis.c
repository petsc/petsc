
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: itregis.c,v 1.24 1997/10/19 03:23:06 bsmith Exp bsmith $";
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

#undef __FUNC__  
#define __FUNC__ "KSPRegisterAll"
/*@C
  KSPRegisterAll - Registers all of the Krylov subspace methods in the KSP package.

  Adding new methods:
  To add a new method to the registry, copy this routine and modify
  it to incorporate a call to KSPRegister() for the new method.  

  Restricting the choices:
  To prevent all of the methods from being registered and thus 
  save memory, copy this routine and modify it to register only 
  those methods you desire.  Make sure that the replacement routine 
  is linked before libpetscsles.a.

  Notes:
  Currently the default iterative method, KSPGMRES, must be
  registered.

.keywords: KSP, register, all

.seealso:  KSPRegister(), KSPRegisterDestroy()
@*/
int KSPRegisterAll()
{
  PetscFunctionBegin;
  KSPRegisterAllCalled = 1;
  KSPRegister(KSPCG         , "cg",         "KSPCreate_CG",KSPCreate_CG,0);
  KSPRegister(KSPRICHARDSON , "richardson", "KSPCreate_Richardson",KSPCreate_Richardson,0);
  KSPRegister(KSPCHEBYCHEV  , "chebychev",  "KSPCreate_Chebychev",KSPCreate_Chebychev,0);
  KSPRegister(KSPGMRES      , "gmres",      "KSPCreate_GMRES",KSPCreate_GMRES,0);
  KSPRegister(KSPTCQMR      , "tcqmr",      "KSPCreate_TCQMR",KSPCreate_TCQMR,0);
  KSPRegister(KSPBCGS       , "bcgs",       "KSPCreate_BCGS",KSPCreate_BCGS,0);
  KSPRegister(KSPCGS        , "cgs",        "KSPCreate_CGS",KSPCreate_CGS,0);
  KSPRegister(KSPTFQMR      , "tfqmr",      "KSPCreate_TFQMR",KSPCreate_TFQMR,0);
  KSPRegister(KSPCR         , "cr",         "KSPCreate_CR",KSPCreate_CR,0); 
  KSPRegister(KSPLSQR       , "lsqr",       "KSPCreate_LSQR",KSPCreate_LSQR,0);
  KSPRegister(KSPPREONLY    , "preonly",    "KSPCreate_PREONLY",KSPCreate_PREONLY,0);
  KSPRegister(KSPQCG        , "qcg",        "KSPCreate_QCG",KSPCreate_QCG,0);
  PetscFunctionReturn(0);
}
