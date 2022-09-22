
#include <petsc/private/petscimpl.h> /*I    "petscsys.h"   I*/
#include <petscviewersaws.h>
#include <petscsys.h>

/*@C
   PetscObjectSAWsTakeAccess - Take access of the data fields that have been published to SAWs by a `PetscObject` so their values may
   be changed in the computation

   Collective on obj

   Input Parameters:
.  obj - the `PetscObject` variable. This must be cast with a (`PetscObject`), for example, `PetscObjectSAWSTakeAccess`((`PetscObject`)mat);

   Level: advanced

   Developer Note:
   The naming should perhaps be changed to `PetscObjectSAWsGetAccess()` and `PetscObjectSAWsRestoreAccess()`

.seealso: `PetscObjectSetName()`, `PetscObjectSAWsViewOff()`, `PetscObjectSAWsGrantAccess()`
@*/
PetscErrorCode PetscObjectSAWsTakeAccess(PetscObject obj)
{
  if (obj->amsmem) {
    /* cannot wrap with PetscPushStack() because that also deals with the locks */
    SAWs_Lock();
  }
  return 0;
}

/*@C
   PetscObjectSAWsGrantAccess - Grants access of the data fields that have been published to SAWs called when the changes made during
   `PetscObjectSAWsTakeAccess()` are complete. This allows the webserve to change the published values.

   Collective on obj

   Input Parameters:
.  obj - the `PetscObject` variable. This must be cast with a (`PetscObject`), for example, `PetscObjectSAWSRestoreAccess`((`PetscObject`)mat);

   Level: advanced

.seealso: `PetscObjectSetName()`, `PetscObjectSAWsViewOff()`, `PetscObjectSAWsTakeAccess()`
@*/
PetscErrorCode PetscObjectSAWsGrantAccess(PetscObject obj)
{
  if (obj->amsmem) {
    /* cannot wrap with PetscPushStack() because that also deals with the locks */
    SAWs_Unlock();
  }
  return 0;
}

/*@C
   PetscSAWsBlock - Blocks on SAWs until a client (person using the web browser) unblocks it

   Not Collective

   Level: advanced

.seealso: `PetscObjectSetName()`, `PetscObjectSAWsViewOff()`, `PetscObjectSAWsSetBlock()`, `PetscObjectSAWsBlock()`
@*/
PetscErrorCode PetscSAWsBlock(void)
{
  volatile PetscBool block = PETSC_TRUE;

  PetscFunctionBegin;
  PetscCallSAWs(SAWs_Register, ("__Block", (PetscBool *)&block, 1, SAWs_WRITE, SAWs_BOOLEAN));
  SAWs_Lock();
  while (block) {
    SAWs_Unlock();
    PetscCall(PetscInfo(NULL, "Blocking on SAWs\n"));
    PetscCall(PetscSleep(.3));
    SAWs_Lock();
  }
  SAWs_Unlock();
  PetscCallSAWs(SAWs_Delete, ("__Block"));
  PetscCall(PetscInfo(NULL, "Out of SAWs block\n"));
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectSAWsBlock - Blocks the object if `PetscObjectSAWsSetBlock()` has been called

   Collective on obj

   Input Parameters:
.  obj - the PETSc variable

   Level: advanced

.seealso: `PetscObjectSetName()`, `PetscObjectSAWsViewOff()`, `PetscObjectSAWsSetBlock()`, `PetscSAWsBlock()`
@*/
PetscErrorCode PetscObjectSAWsBlock(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);

  if (!obj->amspublishblock || !obj->amsmem) PetscFunctionReturn(0);
  PetscCall(PetscSAWsBlock());
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectSAWsSetBlock - Sets whether an object will block at `PetscObjectSAWsBlock()`

   Collective on obj

   Input Parameters:
+  obj - the PETSc variable
-  flg - whether it should block

   Level: advanced

.seealso: `PetscObjectSetName()`, `PetscObjectSAWsViewOff()`, `PetscObjectSAWsBlock()`, `PetscSAWsBlock()`
@*/
PetscErrorCode PetscObjectSAWsSetBlock(PetscObject obj, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  obj->amspublishblock = flg;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscObjectSAWsViewOff(PetscObject obj)
{
  char dir[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  if (obj->classid == PETSC_VIEWER_CLASSID) PetscFunctionReturn(0);
  if (!obj->amsmem) PetscFunctionReturn(0);
  PetscCall(PetscSNPrintf(dir, sizeof(dir), "/PETSc/Objects/%s", obj->name));
  PetscCallSAWs(SAWs_Delete, (dir));
  PetscFunctionReturn(0);
}
