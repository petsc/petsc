#ifndef lint
static char vcid[] = "$Id: itregis.c,v 1.4 1995/03/06 04:18:44 bsmith Exp bsmith $";
#endif


#include "petsc.h"
#include "kspimpl.h"


/*@
   KSPRegisterAll - Registers all the iterative methods
  in KSP.

  Notes:
  To prevent all the methods from being
  registered and thus save memory, copy this routine and
  register only those methods desired.
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
