#ifndef lint
static char vcid[] = "$Id: pcregis.c,v 1.24 1995/12/13 16:10:28 curfman Exp bsmith $";
#endif

#include "petsc.h"
#include "pcimpl.h"          /*I   "pc.h"   I*/

extern int PCCreate_Jacobi(PC);
extern int PCCreate_BJacobi(PC);
extern int PCCreate_None(PC);
extern int PCCreate_LU(PC);
extern int PCCreate_SOR(PC);
extern int PCCreate_Shell(PC);
extern int PCCreate_MG(PC);
extern int PCCreate_Eisenstat(PC);
extern int PCCreate_ILU(PC);
extern int PCCreate_ICC(PC);
extern int PCCreate_ASM(PC);
extern int PCCreate_BGS(PC);

/*@C
  PCRegisterAll - Registers all of the preconditioners in the PC package.

  Adding new methods:
  To add a new method to the registry
$   1.  Copy this routine and modify it to incorporate
$       a call to PCRegister() for the new method.  
$   2.  Modify the file "PETSCDIR/include/pc.h"
$       by appending the method's identifier as an
$       enumerator of the PCMethod enumeration.
$       As long as the enumerator is appended to
$       the existing list, only the PCRegisterAll()
$       routine requires recompilation.

  Restricting the choices:
  To prevent all of the methods from being registered and thus 
  save memory, copy this routine and modify it to register only 
  those methods you desire.  Make sure that the replacement routine 
  is linked before libpetscsles.a.

.keywords: PC, register, all

.seealso: PCRegister(), PCRegisterDestroy()
@*/
int PCRegisterAll()
{
  PCRegister(PCNONE         , "none",       PCCreate_None);
  PCRegister(PCJACOBI       , "jacobi",     PCCreate_Jacobi);
  PCRegister(PCBJACOBI      , "bjacobi",    PCCreate_BJacobi);
  PCRegister(PCSOR          , "sor",        PCCreate_SOR);
  PCRegister(PCLU           , "lu",         PCCreate_LU);
  PCRegister(PCSHELL        , "shell",      PCCreate_Shell);
  PCRegister(PCMG           , "mg",         PCCreate_MG);
  PCRegister(PCEISENSTAT    , "eisenstat",  PCCreate_Eisenstat);
  PCRegister(PCILU          , "ilu",        PCCreate_ILU);
  PCRegister(PCICC          , "icc",        PCCreate_ICC);
  PCRegister(PCASM          , "asm",        PCCreate_ASM);
  PCRegister(PCBGS          , "bgs",        PCCreate_BGS);
  return 0;
}


