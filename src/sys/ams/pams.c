
#include <petsc-private/petscimpl.h>        /*I    "petscsys.h"   I*/
#include <petscviewerams.h>
#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "PetscObjectAMSTakeAccess"
/*@C
   PetscObjectAMSTakeAccess - Take access of the data fields that have been published to AMS so they may be changed locally

   Collective on PetscObject

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectSetName((PetscObject)mat,name);

   Level: advanced

   Concepts: publishing object

.seealso: PetscObjectSetName(), PetscObjectAMSViewOff(), PetscObjectAMSGrantAccess()

@*/
PetscErrorCode  PetscObjectAMSTakeAccess(PetscObject obj)
{
  PetscFunctionBegin;
  if (obj->amsmem != -1) {
    PetscStackCallAMS(AMS_Memory_take_access,(obj->amsmem));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectAMSGrantAccess"
/*@C
   PetscObjectAMSGrantAccess - Grants access of the data fields that have been published to AMS to the memory snooper to change

   Collective on PetscObject

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectSetName((PetscObject)mat,name);

   Level: advanced

   Concepts: publishing object

.seealso: PetscObjectSetName(), PetscObjectAMSViewOff(), PetscObjectAMSTakeAccess()

@*/
PetscErrorCode  PetscObjectAMSGrantAccess(PetscObject obj)
{
  PetscFunctionBegin;
  if (obj->amsmem != -1) {
    PetscStackCallAMS(AMS_Memory_grant_access,(obj->amsmem));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectAMSBlock"
/*@C
   PetscObjectAMSBlock - Blocks the object if PetscObjectAMSSetBlock() has been called

   Collective on PetscObject

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectSetName((PetscObject)mat,name);


   Level: advanced

   Concepts: publishing object

.seealso: PetscObjectSetName(), PetscObjectAMSViewOff(), PetscObjectAMSSetBlock()

@*/
PetscErrorCode  PetscObjectAMSBlock(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);

  if (!obj->amspublishblock) PetscFunctionReturn(0);
  /* Eventually this will be fixed to check if the AMS client has changed the lock */
  while (1);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectAMSSetBlock"
/*@C
   PetscObjectAMSSetBlock - Sets whether an object will block at PetscObjectAMSBlock()

   Collective on PetscObject

   Input Parameters:
+  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectSetName((PetscObject)mat,name);
-  flg - whether it should block

   Level: advanced

   Concepts: publishing object

.seealso: PetscObjectSetName(), PetscObjectAMSViewOff(), PetscObjectAMSBlock()

@*/
PetscErrorCode  PetscObjectAMSSetBlock(PetscObject obj,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  obj->amspublishblock = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectAMSViewOff"
PetscErrorCode PetscObjectAMSViewOff(PetscObject obj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (obj->classid == PETSC_VIEWER_CLASSID) PetscFunctionReturn(0);
  if (obj->amsmem == -1) PetscFunctionReturn(0);
  ierr        = AMS_Memory_destroy(obj->amsmem);CHKERRQ(ierr);
  obj->amsmem = -1;
  PetscFunctionReturn(0);
}

