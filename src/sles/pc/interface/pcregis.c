#ifndef lint
static char vcid[] = "$Id: pcregis.c,v 1.11 1995/04/13 22:40:28 bsmith Exp curfman $";
#endif


#include "petsc.h"
#include "pcimpl.h"

extern int PCCreate_Jacobi(PC);
extern int PCCreate_BJacobi(PC);
extern int PCCreate_None(PC);
extern int PCCreate_Direct(PC);
extern int PCCreate_SOR(PC);
extern int PCCreate_Shell(PC);
extern int PCCreate_MG(PC);
extern int PCCreate_Eisenstat(PC);
extern int PCCreate_ILU(PC);
extern int PCCreate_ICC(PC);

/*@
   PCRegisterAll - Registers all the iterative methods
  in KSP.

  Notes:
  To prevent all the methods from being registered and thus save
  memory, copy this routine and register only those methods desired.

.keywords: PC, register, all

.seealso: PCRegister(), PCRegisterDestroy()
@*/
int PCRegisterAll()
{
  PCRegister(PCNONE         , "none",       PCCreate_None);
  PCRegister(PCJACOBI       , "jacobi",     PCCreate_Jacobi);
  PCRegister(PCBJACOBI      , "bjacobi",    PCCreate_BJacobi);
  PCRegister(PCSOR          , "sor",        PCCreate_SOR);
  PCRegister(PCDIRECT       , "direct",     PCCreate_Direct);
  PCRegister(PCSHELL        , "shell",      PCCreate_Shell);
  PCRegister(PCMG           , "mg",         PCCreate_MG);
  PCRegister(PCESOR         , "eisenstat",  PCCreate_Eisenstat);
  PCRegister(PCILU          , "ilu",        PCCreate_ILU);
#if defined(HAVE_BLOCKSOLVE) && !defined(PETSC_COMPLEX)
  PCRegister(PCICC          , "icc",        PCCreate_ICC);
#endif
  return 0;
}


