/*$Id: tsregall.c,v 1.32 2001/03/23 23:24:34 balay Exp $*/

#include "src/ts/tsimpl.h"     /*I  "petscts.h"  I*/
EXTERN_C_BEGIN
EXTERN int TSCreate_Euler(TS);
EXTERN int TSCreate_BEuler(TS);
EXTERN int TSCreate_Pseudo(TS);
EXTERN int TSCreate_PVode(TS);
EXTERN int TSCreate_CN(TS);

EXTERN int GTSSerialize_BEuler(MPI_Comm, TS *, PetscViewer, PetscTruth);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "TSRegisterAll"
/*@C
  TSRegisterAll - Registers all of the timesteppers in the TS package. 

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: TS, timestepper, register, all
.seealso: TSCreate(), TSRegister(), TSRegisterDestroy()
@*/
int TSRegisterAll(const char path[])
{
  int ierr;

  PetscFunctionBegin;
  TSRegisterAllCalled = PETSC_TRUE;

  ierr = TSRegisterDynamic(TS_EULER,           path, "TSCreate_Euler", TSCreate_Euler);                   CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TS_BEULER,          path, "TSCreate_BEuler",TSCreate_BEuler);                  CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TS_CRANK_NICHOLSON, path, "TSCreate_CN", TSCreate_CN);                         CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TS_PSEUDO,          path, "TSCreate_Pseudo", TSCreate_Pseudo);                 CHKERRQ(ierr);
#if defined(PETSC_HAVE_PVODE) && !defined(__cplusplus)
  ierr = TSRegisterDynamic(TS_PVODE,           path, "TSCreate_PVode", TSCreate_PVode);                   CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSerializeRegisterAll"
/*@C
  TSSerializeRegisterAll - Registers all of the serialization routines in the TS package. 

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: ts, register, all, serialize
.seealso: TSSerialize(), TSSerializeRegister(), TSSerializeRegisterDestroy()
@*/
int TSSerializeRegisterAll(const char path[])
{
  PetscFunctionBegin;
  TSSerializeRegisterAllCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}
