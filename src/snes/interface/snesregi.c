#ifndef lint
static char vcid[] = "$Id: snesregi.c,v 1.6 1995/06/29 23:54:14 bsmith Exp curfman $";
#endif

#include "snesimpl.h"     /*I  "snes.h"  I*/
extern int SNESCreate_LS(SNES);
extern int SNESCreate_TR(SNES);
extern int SNESCreate_UMTR(SNES);
extern int SNESCreate_UMLS(SNES);
extern int SNESCreate_Test(SNES);

/*@
  SNESRegisterAll - Registers all of the nonlinear solvers in the SNES 
  package. 

  Adding new methods:
  To add a new method to the registry
$   1.  Copy this routine and modify it to incorporate
$       a call to SNESRegister() for the new method.  
$   2.  Modify the file "PETSCDIR/include/snes.h"
$       by appending the method's identifier as an
$       enumerator of the SNESMethod enumeration.
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
  SNESRegister((int)SNES_EQ_NLS,         "ls",      SNESCreate_LS);
  SNESRegister((int)SNES_EQ_NTR,         "tr",      SNESCreate_TR);
  SNESRegister((int)SNES_EQ_NTEST,       "test",    SNESCreate_Test);
  SNESRegister((int)SNES_UM_NTR,         "umtr",    SNESCreate_UMTR);
  SNESRegister((int)SNES_UM_NLS,         "umls",    SNESCreate_UMLS);
  return 0;
}
