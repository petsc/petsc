#ifndef lint
static char vcid[] = "$Id: nldregis.c,v 1.1 1995/03/20 00:11:19 bsmith Exp bsmith $";
#endif

#include "snesimpl.h"
extern int SNESNewtonLS1Create(SNES);

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
$       a call to NLRegister() for the new method.  
$   2.  Modify the file "TOOLSDIR/include/snes.h"
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
  is linked before libsnes.a .
@*/
int SNESRegisterAll()
{
   SNESRegister((int)SNES_NLS1,     "enls1",     SNESNewtonLS1Create);
/*
   SNESRegister((int)SNES_NTR1,     "entr1",     SNESNewtonTR1Create);
   SNESRegister((int)SNES_NTR2_DOG, "entr2_dog", SNESNewtonTR2DoglegCreate);
   SNESRegister((int)SNES_NTR2_LIN, "entr2_lin", SNESNewtonTR2LinearCreate);
   SNESRegister((int)SNES_NBASIC,   "enbasic",   SNESNewtonBasicCreate);
   SNESRegister((int)SUMS_NLS1,     "mnls1",     SUMSNewtonLS1Create);
   SNESRegister((int)SUMS_NTR1,     "mntr1",     SUMSNewtonTR1Create);
*/
}
