#ifndef lint
static char vcid[] = "$Id: snesregi.c,v 1.11 1996/01/01 01:05:05 bsmith Exp curfman $";
#endif

#include "snesimpl.h"     /*I  "snes.h"  I*/
extern int SNESCreate_EQ_LS(SNES);
extern int SNESCreate_EQ_TR(SNES);
extern int SNESCreate_UM_TR(SNES);
extern int SNESCreate_UM_LS(SNES);
extern int SNESCreate_Test(SNES);

/*@C
  SNESRegisterAll - Registers all of the nonlinear solvers in the SNES 
  package. 

  Adding new methods:
  To add a new method to the registry
$   1.  Copy this routine and modify it to incorporate
$       a call to SNESRegister() for the new method.  
$   2.  Modify the file "PETSCDIR/include/snes.h"
$       by appending the method's identifier as an
$       enumerator of the SNESType enumeration.
$       As long as the enumerator is appended to
$       the existing list, only the SNESRegisterAll()
$       routine requires recompilation.

  Restricting the choices:
  To prevent all of the methods from being registered and thus 
  save memory, copy this routine and modify it to register only 
  those methods you desire.  Make sure that the replacement routine 
  is linked before libpetscsnes.a.

.keywords: SNES, nonlinear, register, all

.seealso: SNESRegister(), SNESRegisterDestroy()
@*/
int SNESRegisterAll()
{
  SNESRegister((int)SNES_EQ_LS,         "ls",      SNESCreate_EQ_LS);
  SNESRegister((int)SNES_EQ_TR,         "tr",      SNESCreate_EQ_TR);
  SNESRegister((int)SNES_EQ_TEST,       "test",    SNESCreate_Test);
  SNESRegister((int)SNES_UM_TR,         "umtr",    SNESCreate_UM_TR);
  SNESRegister((int)SNES_UM_LS,         "umls",    SNESCreate_UM_LS);
  return 0;
}
