/*$Id: tsregall.c,v 1.27 2000/04/09 04:39:08 bsmith Exp bsmith $*/

#include "src/ts/tsimpl.h"     /*I  "ts.h"  I*/
EXTERN_C_BEGIN
extern int TSCreate_Euler(TS);
extern int TSCreate_BEuler(TS);
extern int TSCreate_Pseudo(TS);
extern int TSCreate_PVode(TS);
extern int TSCreate_CN(TS);
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"TSRegisterAll"
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
  TSRegisterAllCalled = PETSC_TRUE;

  TSRegisterDynamic(TS_EULER,               path,"TSCreate_Euler", TSCreate_Euler);
  TSRegisterDynamic(TS_BEULER,              path,"TSCreate_BEuler",TSCreate_BEuler);
  TSRegisterDynamic(TS_CRANK_NICHOLSON,     path,"TSCreate_CN",TSCreate_CN);
  TSRegisterDynamic(TS_PSEUDO,              path,"TSCreate_Pseudo",TSCreate_Pseudo);
#if defined(PETSC_HAVE_PVODE) && !defined(__cplusplus)
  TSRegisterDynamic(TS_PVODE,               path,"TSCreate_PVode", TSCreate_PVode); 
#endif
  PetscFunctionReturn(0);
}
