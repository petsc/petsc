#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pcregis.c,v 1.33 1997/02/22 02:23:59 bsmith Exp balay $";
#endif

#include "petsc.h"
#include "src/pc/pcimpl.h"          /*I   "pc.h"   I*/

extern int PCCreate_Jacobi(PC);
extern int PCCreate_BJacobi(PC);
extern int PCCreate_ILU(PC);
extern int PCCreate_None(PC);
extern int PCCreate_LU(PC);
extern int PCCreate_SOR(PC);
extern int PCCreate_Shell(PC);
extern int PCCreate_MG(PC);
extern int PCCreate_Eisenstat(PC);
extern int PCCreate_ICC(PC);
extern int PCCreate_ASM(PC);
extern int PCCreate_BGS(PC);

#undef __FUNC__  
#define __FUNC__ "PCRegisterAll" /* ADIC Ignore */
/*@C
  PCRegisterAll - Registers all of the preconditioners in the PC package.

  Adding new methods:
  To add new methods to the registry, copy this routine and modify
  it to incorporate a call to PCRegister() for each new method.  

  Restricting the choices:
  To prevent all of the methods from being registered and thus 
  save memory, copy this routine and modify it to register only 
  those methods you desire.  Make sure that the replacement routine 
  is linked before libpetscsles.a.

  Notes: 
  Currently the default preconditioners PCILU (and in parallel PCBJACOBI)
  must be registered.

.keywords: PC, register, all

.seealso: PCRegister(), PCRegisterDestroy()
@*/
int PCRegisterAll()
{
  PCRegisterAllCalled = 1;

  PCRegister(PCNONE         ,0, "none",       PCCreate_None);
  PCRegister(PCJACOBI       ,0, "jacobi",     PCCreate_Jacobi);
  PCRegister(PCBJACOBI      ,0, "bjacobi",    PCCreate_BJacobi);
  PCRegister(PCSOR          ,0, "sor",        PCCreate_SOR);
  PCRegister(PCLU           ,0, "lu",         PCCreate_LU);
  PCRegister(PCSHELL        ,0, "shell",      PCCreate_Shell);
  PCRegister(PCMG           ,0, "mg",         PCCreate_MG);
  PCRegister(PCEISENSTAT    ,0, "eisenstat",  PCCreate_Eisenstat);
  PCRegister(PCILU          ,0, "ilu",        PCCreate_ILU);
  PCRegister(PCICC          ,0, "icc",        PCCreate_ICC);
  PCRegister(PCASM          ,0, "asm",        PCCreate_ASM);
  PCRegister(PCBGS          ,0, "bgs",        PCCreate_BGS);
  return 0;
}


