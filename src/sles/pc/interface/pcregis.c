

#include "petsc.h"
#include "pcimpl.h"

int PCiJacobiCreate(PC);
int PCiBJacobiCreate(PC);
int PCiNoneCreate(PC);
int PCiDirectCreate(PC);
int PCiSORCreate(PC);
int PCiShellCreate(PC);
int PCiMGCreate(PC);

/*@
   PCRegisterAll - Registers all the iterative methods
  in KSP.

  Notes:
  To prevent all the methods from being
  registered and thus save memory, copy this routine and
  register only those methods desired.
@*/
int PCRegisterAll()
{
  PCRegister(PCNONE         , "none",       PCiNoneCreate);
  PCRegister(PCJACOBI       , "jacobi",     PCiJacobiCreate);
  PCRegister(PCBJACOBI      , "bjacobi",    PCiBJacobiCreate);
  PCRegister(PCSOR          , "sor",        PCiSORCreate);
  PCRegister(PCDIRECT       , "direct",     PCiDirectCreate);
  PCRegister(PCSHELL        , "shell",      PCiShellCreate);
  PCRegister(PCMG           , "mg",         PCiMGCreate);
  return 0;
}


