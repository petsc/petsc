#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: tsregall.c,v 1.15 1998/04/13 17:50:17 bsmith Exp curfman $";
#endif

#include "src/ts/tsimpl.h"     /*I  "ts.h"  I*/
extern int TSCreate_Euler(TS);
extern int TSCreate_BEuler(TS);
extern int TSCreate_Pseudo(TS);
extern int TSCreate_PVode(TS);

/*M
   TSRegister - Adds a method to the timestepping solver package.

   Synopsis:
   TSRegister(char *name_solver,char *path,char *name_create,int (*routine_create)(TS))

   Input Parameters:
.  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
.  routine_create - routine to create method context

   Notes:
   TSRegister() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
   TSRegister("my_solver",/home/username/my_lib/lib/libO/solaris/mylib.a,
                "MySolverCreate",MySolverCreate);

   Then, your solver can be chosen with the procedural interface via
$     TSSetType(ts,"my_solver")
$   or at runtime via the option
$     -ts_type my_solver

.keywords: TS, register

.seealso: TSRegisterAll(), TSRegisterDestroy()
M*/

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
