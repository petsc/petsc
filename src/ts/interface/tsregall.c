#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: tsregall.c,v 1.14 1998/03/06 00:17:24 bsmith Exp bsmith $";
#endif

#include "src/ts/tsimpl.h"     /*I  "ts.h"  I*/
extern int TSCreate_Euler(TS);
extern int TSCreate_BEuler(TS);
extern int TSCreate_Pseudo(TS);
extern int TSCreate_PVode(TS);

#if defined(USE_DYNAMIC_LIBRARIES)
#define TSRegister(a,b,c,d) TSRegister_Private(a,b,c,0)
#else
#define TSRegister(a,b,c,d) TSRegister_Private(a,b,c,d)
#endif

#undef __FUNC__  
#define __FUNC__ "TSRegister_Private"
static int TSRegister_Private(char *sname,char *path,char *name,int (*function)(TS))
{
  char fullname[256];

  PetscFunctionBegin;
  PetscStrcpy(fullname,path); PetscStrcat(fullname,":"); PetscStrcat(fullname,name);
  DLRegister(&TSList,sname,fullname,        (int (*)(void*))function);
  PetscFunctionReturn(0);
}

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
