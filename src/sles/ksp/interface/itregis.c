
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
  KSPRegister((int)KSPCG         , "cg",         KSPiCGCreate);
  KSPRegister((int)KSPRICHARDSON , "richardson", KSPiRichardsonCreate);
  KSPRegister((int)KSPCHEBYCHEV  , "chebychev",  KSPiChebychevCreate);
  KSPRegister((int)KSPGMRES      , "gmres",      KSPiGMRESCreate);
  KSPRegister((int)KSPTCQMR      , "tcqmr",      KSPiTCQMRCreate);
  KSPRegister((int)KSPBCGS       , "bcgs",       KSPiBCGSCreate);
  KSPRegister((int)KSPCGS        , "cgs",        KSPiCGSCreate);
  KSPRegister((int)KSPTFQMR      , "tfqmr",      KSPiTFQMRCreate);
  KSPRegister((int)KSPCR         , "cr",         KSPiCRCreate); 
  KSPRegister((int)KSPLSQR       , "lsqr",       KSPiLSQRCreate);
  KSPRegister((int)KSPPREONLY    , "preonly",    KSPiPREONLYCreate);
}
