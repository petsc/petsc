
#ifndef lint
static char vcid[] = "$Id: tsregall.c,v 1.1 1996/09/30 18:36:07 bsmith Exp curfman $";
#endif

#include "src/ts/tsimpl.h"     /*I  "ts.h"  I*/
extern int TSCreate_Euler(TS);
extern int TSCreate_BEuler(TS);
extern int TSCreate_Pseudo(TS);

/*@C
  TSRegisterAll - Registers all of the timesteppers in the TS 
  package. 

  Adding new methods:
  To add a new method to the registry
$   1.  Copy this routine and modify it to incorporate
$       a call to TSRegister() for the new method.  
$   2.  Modify the file "PETSCDIR/include/ts.h"
$       by appending the method's identifier as an
$       enumerator of the TSType enumeration.
$       As long as the enumerator is appended to
$       the existing list, only the TSRegisterAll()
$       routine requires recompilation.

  Restricting the choices:
  To prevent all of the methods from being registered and thus 
  save memory, copy this routine and modify it to register only 
  those methods you desire.  Make sure that the replacement routine 
  is linked before libpetscsnes.a.

.keywords: TS, timestepper, register, all

.seealso: TSRegister(), TSRegisterDestroy()
@*/
int TSRegisterAll()
{
  TSRegister((int)TS_EULER,         "euler",      TSCreate_Euler);
  TSRegister((int)TS_BEULER,        "beuler",     TSCreate_BEuler);
  TSRegister((int)TS_PSEUDO,        "pseudo",     TSCreate_Pseudo);
  return 0;
}
