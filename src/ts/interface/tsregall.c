#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: tsregall.c,v 1.21 1998/10/19 22:19:12 bsmith Exp curfman $";
#endif

#include "src/ts/tsimpl.h"     /*I  "ts.h"  I*/
EXTERN_C_BEGIN
extern int TSCreate_Euler(TS);
extern int TSCreate_BEuler(TS);
extern int TSCreate_Pseudo(TS);
extern int TSCreate_PVode(TS);
extern int TSCreate_CN(TS);
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "TSRegisterAll"
/*@C
   TSRegisterAll - Registers all of the timesteppers in the TS package. 

   Not Collective

   Level: advanced

.keywords: TS, timestepper, register, all

.seealso: TSRegisterDestroy()
@*/
int TSRegisterAll(char *path)
{
  PetscFunctionBegin;
  TSRegisterAllCalled = 1;

  TSRegister(TS_EULER,               path,"TSCreate_Euler", TSCreate_Euler);
  TSRegister(TS_BEULER,              path,"TSCreate_BEuler",TSCreate_BEuler);
  TSRegister(TS_CRANK_NICHOLSON,     path,"TSCreate_CN",TSCreate_CN);
  TSRegister(TS_PSEUDO,              path,"TSCreate_Pseudo",TSCreate_Pseudo);
#if defined(HAVE_PVODE) && !defined(__cplusplus)
  TSRegister(TS_PVODE,               path,"TSCreate_PVode", TSCreate_PVode); 
#endif
  PetscFunctionReturn(0);
}
