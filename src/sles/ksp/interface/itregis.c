#ifndef lint
static char vcid[] = "$Id: itregis.c,v 1.12 1996/01/01 01:01:38 bsmith Exp bsmith $";
#endif

#include "kspimpl.h"  /*I "ksp.h" I*/

/*@C
  KSPRegisterAll - Registers all of the Krylov subspace methods in the KSP package.

  Adding new methods:
  To add a new method to the registry
$   1.  Copy this routine and modify it to incorporate
$       a call to KSPRegister() for the new method.  
$   2.  Modify the file "PETSCDIR/include/ksp.h"
$       by appending the method's identifier as an
$       enumerator of the KSPType enumeration.
$       As long as the enumerator is appended to
$       the existing list, only the KSPRegisterAll()
$       routine requires recompilation.

  Restricting the choices:
  To prevent all of the methods from being registered and thus 
  save memory, copy this routine and modify it to register only 
  those methods you desire.  Make sure that the replacement routine 
  is linked before libpetscksp.a.

.keywords: KSP, register, all

.seealso:  KSPRegister(), KSPRegisterDestroy()
@*/
int KSPRegisterAll()
{
  KSPRegister(KSPCG         , "cg",         KSPCreate_CG);
  KSPRegister(KSPRICHARDSON , "richardson", KSPCreate_Richardson);
  KSPRegister(KSPCHEBYCHEV  , "chebychev",  KSPCreate_Chebychev);
  KSPRegister(KSPGMRES      , "gmres",      KSPCreate_GMRES);
  KSPRegister(KSPTCQMR      , "tcqmr",      KSPCreate_TCQMR);
  KSPRegister(KSPBCGS       , "bcgs",       KSPCreate_BCGS);
  KSPRegister(KSPCGS        , "cgs",        KSPCreate_CGS);
  KSPRegister(KSPTFQMR      , "tfqmr",      KSPCreate_TFQMR);
  KSPRegister(KSPCR         , "cr",         KSPCreate_CR); 
  KSPRegister(KSPLSQR       , "lsqr",       KSPCreate_LSQR);
  KSPRegister(KSPPREONLY    , "preonly",    KSPCreate_PREONLY);
  KSPRegister(KSPQCG        , "qcg",        KSPCreate_QCG);
  return 0;
}
