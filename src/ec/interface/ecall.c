#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ecall.c,v 1.5 1998/04/27 16:09:37 bsmith Exp bsmith $";
#endif

/*
     Do not use dynamic libraries for the EC yet. This is because will require
   an explicit library load during the PetscInitialize() call.
*/
#undef USE_DYNAMIC_LIBRARIES

#include "petsc.h"
#include "src/ec/ecimpl.h"          /*I   "ec.h"   I*/

extern int ECCreate_Lapack(EC);

#undef __FUNC__  
#define __FUNC__ "ECRegisterAll" 
/*@C
  ECRegisterAll - Registers all of the preconditioners in the EC package.

  Not Collective

  Adding new methods:
  To add a new method to the registry
$   1.  Copy this routine and modify it to incorporate
$       a call to ECRegister() for the new method.  
$   2.  Modify the file "PETSCDIR/include/ec.h"
$       by appending the method's identifier as an
$       enumerator of the ECType enumeration.
$       As long as the enumerator is appended to
$       the existing list, only the ECRegisterAll()
$       routine requires recompilation.

  Restricting the choices:
  To prevent all of the methods from being registered and thus 
  save memory, copy this routine and modify it to register only 
  those methods you desire.  Make sure that the replacement routine 
  is linked before libpetscsles.a.

.keywords: EC, register, all

.seealso: ECRegister(), ECRegisterDestroy()
@*/
int ECRegisterAll(char *path)
{
  PetscFunctionBegin;
  ECRegister(EC_LAPACK         ,path, "ECCreate_Lapack",       ECCreate_Lapack);
  PetscFunctionReturn(0);
}




