
#include <petsc/private/petscimpl.h>        /*I    "petscsys.h"   I*/
#include <petscviewersaws.h>
#include <petscsys.h>

/*@C
   PetscObjectSAWsTakeAccess - Take access of the data fields that have been published to SAWs so they may be changed locally

   Collective on PetscObject

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectSetName((PetscObject)mat,name);

   Level: advanced

.seealso: `PetscObjectSetName()`, `PetscObjectSAWsViewOff()`, `PetscObjectSAWsGrantAccess()`

@*/
PetscErrorCode  PetscObjectSAWsTakeAccess(PetscObject obj)
{
  if (obj->amsmem) {
    /* cannot wrap with PetscPushStack() because that also deals with the locks */
    SAWs_Lock();
  }
  return 0;
}

/*@C
   PetscObjectSAWsGrantAccess - Grants access of the data fields that have been published to SAWs to change

   Collective on PetscObject

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectSetName((PetscObject)mat,name);

   Level: advanced

.seealso: `PetscObjectSetName()`, `PetscObjectSAWsViewOff()`, `PetscObjectSAWsTakeAccess()`

@*/
PetscErrorCode  PetscObjectSAWsGrantAccess(PetscObject obj)
{
  if (obj->amsmem) {
    /* cannot wrap with PetscPushStack() because that also deals with the locks */
    SAWs_Unlock();
  }
  return 0;
}

/*@C
   PetscSAWsBlock - Blocks on SAWs until a client (person using the web browser) unblocks

   Not Collective

   Level: advanced

.seealso: `PetscObjectSetName()`, `PetscObjectSAWsViewOff()`, `PetscObjectSAWsSetBlock()`, `PetscObjectSAWsBlock()`

@*/
PetscErrorCode  PetscSAWsBlock(void)
{
  volatile PetscBool block = PETSC_TRUE;

  PetscFunctionBegin;
  PetscStackCallSAWs(SAWs_Register,("__Block",(PetscBool*)&block,1,SAWs_WRITE,SAWs_BOOLEAN));
  SAWs_Lock();
  while (block) {
    SAWs_Unlock();
    PetscCall(PetscInfo(NULL,"Blocking on SAWs\n"));
    PetscCall(PetscSleep(.3));
    SAWs_Lock();
  }
  SAWs_Unlock();
  PetscStackCallSAWs(SAWs_Delete,("__Block"));
  PetscCall(PetscInfo(NULL,"Out of SAWs block\n"));
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectSAWsBlock - Blocks the object if PetscObjectSAWsSetBlock() has been called

   Collective on PetscObject

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectSetName((PetscObject)mat,name);

   Level: advanced

.seealso: `PetscObjectSetName()`, `PetscObjectSAWsViewOff()`, `PetscObjectSAWsSetBlock()`

@*/
PetscErrorCode  PetscObjectSAWsBlock(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);

  if (!obj->amspublishblock || !obj->amsmem) PetscFunctionReturn(0);
  PetscCall(PetscSAWsBlock());
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectSAWsSetBlock - Sets whether an object will block at PetscObjectSAWsBlock()

   Collective on PetscObject

   Input Parameters:
+  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectSetName((PetscObject)mat,name);
-  flg - whether it should block

   Level: advanced

.seealso: `PetscObjectSetName()`, `PetscObjectSAWsViewOff()`, `PetscObjectSAWsBlock()`

@*/
PetscErrorCode  PetscObjectSAWsSetBlock(PetscObject obj,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  obj->amspublishblock = flg;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscObjectSAWsViewOff(PetscObject obj)
{
  char dir[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  if (obj->classid == PETSC_VIEWER_CLASSID) PetscFunctionReturn(0);
  if (!obj->amsmem) PetscFunctionReturn(0);
  PetscCall(PetscSNPrintf(dir,sizeof(dir),"/PETSc/Objects/%s",obj->name));
  PetscStackCallSAWs(SAWs_Delete,(dir));
  PetscFunctionReturn(0);
}
