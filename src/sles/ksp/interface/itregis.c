#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: itregis.c,v 1.21 1997/03/26 01:34:39 bsmith Exp balay $";
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
#define __FUNC__ "KSPRegisterAll" /* ADIC Ignore */
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
  KSPRegisterAllCalled = 1;
  KSPRegister(KSPCG         , 0,"cg",         KSPCreate_CG);
  KSPRegister(KSPRICHARDSON , 0,"richardson", KSPCreate_Richardson);
  KSPRegister(KSPCHEBYCHEV  , 0,"chebychev",  KSPCreate_Chebychev);
  KSPRegister(KSPGMRES      , 0,"gmres",      KSPCreate_GMRES);
  KSPRegister(KSPTCQMR      , 0,"tcqmr",      KSPCreate_TCQMR);
  KSPRegister(KSPBCGS       , 0,"bcgs",       KSPCreate_BCGS);
  KSPRegister(KSPCGS        , 0,"cgs",        KSPCreate_CGS);
  KSPRegister(KSPTFQMR      , 0,"tfqmr",      KSPCreate_TFQMR);
  KSPRegister(KSPCR         , 0,"cr",         KSPCreate_CR); 
  KSPRegister(KSPLSQR       , 0,"lsqr",       KSPCreate_LSQR);
  KSPRegister(KSPPREONLY    , 0,"preonly",    KSPCreate_PREONLY);
  KSPRegister(KSPQCG        , 0,"qcg",        KSPCreate_QCG);
  return 0;
}
