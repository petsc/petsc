#ifndef lint
static char vcid[] = "$Id: itregis.c,v 1.8 1995/04/16 00:50:01 curfman Exp bsmith $";
#endif

#include "kspimpl.h"  /*I "ksp.h" I*/

/*@
   KSPRegisterAll - Registers all the iterative methods in KSP.

   Notes:
   To prevent all the methods from being registered and thus save 
   memory, copy this routine and register only those methods desired.

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
  return 0;
}
