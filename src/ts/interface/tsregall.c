/*$Id: tsregall.c,v 1.32 2001/03/23 23:24:34 balay Exp $*/

#include "src/ts/tsimpl.h"     /*I  "petscts.h"  I*/
EXTERN_C_BEGIN
EXTERN int TSCreate_Euler(TS);
EXTERN int TSCreate_BEuler(TS);
EXTERN int TSCreate_Pseudo(TS);
EXTERN int TSCreate_PVode(TS);
EXTERN int TSCreate_CN(TS);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "TSRegisterAll"
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
