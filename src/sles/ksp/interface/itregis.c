

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
  KSPRegister(KSPCG         , "cg",         KSPiCGCreate);
  KSPRegister(KSPRICHARDSON , "richardson", KSPiRichardsonCreate);
  KSPRegister(KSPCHEBYCHEV  , "chebychev",  KSPiChebychevCreate);
  KSPRegister(KSPGMRES      , "gmres",      KSPiGMRESCreate);
  KSPRegister(KSPTCQMR      , "tcqmr",      KSPiTCQMRCreate);
  KSPRegister(KSPBCGS       , "bcgs",       KSPiBCGSCreate);
  KSPRegister(KSPCGS        , "cgs",        KSPiCGSCreate);
  KSPRegister(KSPTFQMR      , "tfqmr",      KSPiTFQMRCreate);
  KSPRegister(KSPCR         , "cr",         KSPiCRCreate); 
  KSPRegister(KSPLSQR       , "lsqr",       KSPiLSQRCreate);
  KSPRegister(KSPPREONLY    , "preonly",    KSPiPREONLYCreate);
}
