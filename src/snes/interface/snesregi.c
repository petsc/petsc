#ifndef lint
static char vcid[] = "$Id: snesregi.c,v 1.3 1995/04/13 14:42:33 bsmith Exp curfman $";
#endif

#include "snesimpl.h"
extern int SNESCreate_LS(SNES);
extern int SNESCreate_TR(SNES);

/*@
  SNESRegisterAll - This routine registers all of the solution methods
  in the SNES package. 

  Notes:
  Methods within the SNES package for solving systems of nonlinear 
  equations follow the naming convention SNES_XXXX, while methods 
  for solving unconstrained minimization problems (within the SUMS 
  component) follow the naming convention SUMS_XXXX.

  Adding new methods:
  To add a new method to the registry
$   1.  Copy this routine and modify it to incorporate
$       a call to SNESRegister() for the new method.  
$   2.  Modify the file "PETSCDIR/include/snes.h"
$       by appending the method's identifier as an
$       enumerator of the SNESMETHOD enumeration.
$       As long as the enumerator is appended to
$       the existing list, only the SNESRegisterAll()
$       routine requires recompilation.

  The procedure for adding new methods is currently being
  revised ... stay tuned for further details.

  Restricting the choices:
  To prevent all of the methods from being registered and thus 
  save memory, copy this routine and modify it to register only 
  those methods you desire.  Make sure that the replacement routine 
  is linked before petsclibsnes.a .

.keywords: SNES, nonlinear, register, all

.seealso: SNESRegister(), SNESRegisterDestroy()
@*/
int SNESRegisterAll()
{
   SNESRegister((int)SNES_NLS,         "ls",      SNESCreate_LS);
   SNESRegister((int)SNES_NTR,         "tr",      SNESCreate_TR);
/*
   SNESRegister((int)SNES_NTR_DOG_LEG, "snes_ndog_leg", SNESCreate_DogLeg);
*/
  return 0;
}
