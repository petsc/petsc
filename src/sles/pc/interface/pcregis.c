

#include "petsc.h"
#include "pcimpl.h"

int PCiJacobiCreate(PC);
int PCiNoneCreate(PC);
int PCiDirectCreate(PC);
int PCiSORCreate(PC);

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
  PCRegister(PCSOR          , "sor",        PCiSORCreate);
  PCRegister(PCDIRECT       , "direct",        PCiDirectCreate);
  return 0;
}


