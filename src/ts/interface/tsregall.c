#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: tsregall.c,v 1.11 1997/08/22 15:16:35 bsmith Exp bsmith $";
#endif

#include "src/ts/tsimpl.h"     /*I  "ts.h"  I*/
extern int TSCreate_Euler(TS);
extern int TSCreate_BEuler(TS);
extern int TSCreate_Pseudo(TS);
extern int TSCreate_PVode(TS);

#undef __FUNC__  
#define __FUNC__ "TSRegisterAll"
/*@C
  TSRegisterAll - Registers all of the timesteppers in the TS 
  package. 

  Adding new methods:
  To add a new method to the registry copy this routine and modify
  it to incorporate a call to TSRegister() for the new method.  

  Restricting the choices:
  To prevent all of the methods from being registered and thus 
  save memory, copy this routine and modify it to register only 
  those methods you desire.  Make sure that the replacement routine 
  is linked before libpetscts.a.

.keywords: TS, timestepper, register, all

.seealso: TSRegister(), TSRegisterDestroy()
@*/
int TSRegisterAll()
{
  PetscFunctionBegin;
  TSRegisterAllCalled = 1;

  TSRegister(TS_EULER,         0,"euler",      TSCreate_Euler);
  TSRegister(TS_BEULER,        0,"beuler",     TSCreate_BEuler);
  TSRegister(TS_PSEUDO,        0,"pseudo",     TSCreate_Pseudo);
#if defined(HAVE_PVODE) && !defined(__cplusplus)
  TSRegister(TS_PVODE,         0,"pvode",      TSCreate_PVode); 
#endif
  PetscFunctionReturn(0);
}
