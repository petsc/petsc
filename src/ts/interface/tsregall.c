#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: tsregall.c,v 1.12 1997/10/19 03:28:16 bsmith Exp bsmith $";
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

  TSRegister(TS_EULER,         "euler",      "TSCreate_Euler",TSCreate_Euler,0);
  TSRegister(TS_BEULER,        "beuler",     "TSCreate_BEuler",TSCreate_BEuler,0);
  TSRegister(TS_PSEUDO,        "pseudo",     "TSCreate_Pseudo",TSCreate_Pseudo,0);
#if defined(HAVE_PVODE) && !defined(__cplusplus)
  TSRegister(TS_PVODE,         "pvode",      "TSCreate_PVode",TSCreate_PVode,0); 
#endif
  PetscFunctionReturn(0);
}
