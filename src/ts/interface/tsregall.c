#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: tsregall.c,v 1.17 1998/04/21 19:40:44 curfman Exp curfman $";
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

   Not Collective

.keywords: TS, timestepper, register, all

.seealso: TSRegisterDestroy()
@*/
int TSRegisterAll(char *path)
{
  PetscFunctionBegin;
  TSRegisterAllCalled = 1;

  TSRegister(TS_EULER,      path,"TSCreate_Euler", TSCreate_Euler);
  TSRegister(TS_BEULER,     path,"TSCreate_BEuler",TSCreate_BEuler);
  TSRegister(TS_PSEUDO,     path,"TSCreate_Pseudo",TSCreate_Pseudo);
#if defined(HAVE_PVODE) && !defined(__cplusplus)
  TSRegister(TS_PVODE,      path,"TSCreate_PVode", TSCreate_PVode); 
#endif
  PetscFunctionReturn(0);
}
