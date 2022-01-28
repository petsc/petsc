#include <petsc/private/dmimpl.h> /*I "petscdm.h" I*/

/*@
   DMGetLocalVector - Gets a PETSc vector that may be used with the DM local routines. This vector has spaces for the ghost values.

   Not Collective

   Input Parameter:
.  dm - the dm

   Output Parameter:
.  g - the local vector

   Level: beginner

   Note:
   The vector values are NOT initialized and may have garbage in them, so you may need
   to zero them.

   The output parameter, g, is a regular PETSc vector that should be returned with
   DMRestoreLocalVector() DO NOT call VecDestroy() on it.

   This is intended to be used for vectors you need for a short time, like within a single function call.
   For vectors that you intend to keep around (for example in a C struct) or pass around large parts of your
   code you should use DMCreateLocalVector().

   VecStride*() operations can be useful when using DM with dof > 1

.seealso: DMCreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMGlobalToLocalBegin(),
          DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMCreateLocalVector(), DMRestoreLocalVector(),
          VecStrideMax(), VecStrideMin(), VecStrideNorm()
@*/
PetscErrorCode  DMGetLocalVector(DM dm,Vec *g)
{
  PetscErrorCode ierr,i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(g,2);
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (dm->localin[i]) {
      DM vdm;

      *g             = dm->localin[i];
      dm->localin[i] = NULL;

      ierr = VecGetDM(*g,&vdm);CHKERRQ(ierr);
      PetscAssertFalse(vdm,PetscObjectComm((PetscObject)vdm),PETSC_ERR_LIB,"Invalid vector");
      ierr = VecSetDM(*g,dm);CHKERRQ(ierr);
      goto alldone;
    }
  }
  ierr = DMCreateLocalVector(dm,g);CHKERRQ(ierr);

alldone:
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (!dm->localout[i]) {
      dm->localout[i] = *g;
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@
   DMRestoreLocalVector - Returns a PETSc vector that was
     obtained from DMGetLocalVector(). Do not use with vector obtained via
     DMCreateLocalVector().

   Not Collective

   Input Parameters:
+  dm - the dm
-  g - the local vector

   Level: beginner

.seealso: DMCreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMGlobalToLocalBegin(),
          DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMCreateLocalVector(), DMGetLocalVector()
@*/
PetscErrorCode  DMRestoreLocalVector(DM dm,Vec *g)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(g,2);
  for (j=0; j<DM_MAX_WORK_VECTORS; j++) {
    if (*g == dm->localout[j]) {
      DM vdm;

      ierr = VecGetDM(*g,&vdm);CHKERRQ(ierr);
      PetscAssertFalse(vdm != dm,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Invalid vector");
      ierr = VecSetDM(*g,NULL);CHKERRQ(ierr);
      dm->localout[j] = NULL;
      for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
        if (!dm->localin[i]) {
          dm->localin[i] = *g;
          goto alldone;
        }
      }
    }
  }
  ierr = VecDestroy(g);CHKERRQ(ierr);
alldone:
  *g = NULL;
  PetscFunctionReturn(0);
}

/*@
   DMGetGlobalVector - Gets a PETSc vector that may be used with the DM global routines.

   Collective on dm

   Input Parameter:
.  dm - the dm

   Output Parameter:
.  g - the global vector

   Level: beginner

   Note:
   The vector values are NOT initialized and may have garbage in them, so you may need
   to zero them.

   The output parameter, g, is a regular PETSc vector that should be returned with
   DMRestoreGlobalVector() DO NOT call VecDestroy() on it.

   This is intended to be used for vectors you need for a short time, like within a single function call.
   For vectors that you intend to keep around (for example in a C struct) or pass around large parts of your
   code you should use DMCreateGlobalVector().

   VecStride*() operations can be useful when using DM with dof > 1

.seealso: DMCreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMGlobalToLocalBegin(),
          DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMCreateLocalVector(), DMRestoreLocalVector()
          VecStrideMax(), VecStrideMin(), VecStrideNorm()
@*/
PetscErrorCode  DMGetGlobalVector(DM dm,Vec *g)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(g,2);
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (dm->globalin[i]) {
      DM vdm;

      *g              = dm->globalin[i];
      dm->globalin[i] = NULL;

      ierr = VecGetDM(*g,&vdm);CHKERRQ(ierr);
      PetscAssertFalse(vdm,PetscObjectComm((PetscObject)vdm),PETSC_ERR_LIB,"Invalid vector");
      ierr = VecSetDM(*g,dm);CHKERRQ(ierr);
      goto alldone;
    }
  }
  ierr = DMCreateGlobalVector(dm,g);CHKERRQ(ierr);

alldone:
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (!dm->globalout[i]) {
      dm->globalout[i] = *g;
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@
   DMClearGlobalVectors - Destroys all the global vectors that have been stashed in this DM

   Collective on dm

   Input Parameter:
.  dm - the dm

   Level: developer

.seealso: DMCreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMGlobalToLocalBegin(),
          DMGlobalToLocalEnd(), DMLocalToGlobalBegin(), DMCreateLocalVector(), DMRestoreLocalVector()
          VecStrideMax(), VecStrideMin(), VecStrideNorm()
@*/
PetscErrorCode  DMClearGlobalVectors(DM dm)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    Vec g;

    PetscAssertFalse(dm->globalout[i],PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Clearing DM of global vectors that has a global vector obtained with DMGetGlobalVector()");
    g = dm->globalin[i];
    dm->globalin[i] = NULL;
    if (g) {
      DM vdm;

      ierr = VecGetDM(g,&vdm);CHKERRQ(ierr);
      PetscAssertFalse(vdm,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Clearing global vector that has a DM attached");
    }
    ierr = VecDestroy(&g);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   DMClearLocalVectors - Destroys all the local vectors that have been stashed in this DM

   Collective on dm

   Input Parameter:
.  dm - the dm

   Level: developer

.seealso: DMCreateLocalVector(), VecDuplicate(), VecDuplicateVecs(),
          DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMLocalToLocalBegin(),
          DMLocalToLocalEnd(), DMLocalToLocalBegin(), DMCreateLocalVector(), DMRestoreLocalVector()
          VecStrideMax(), VecStrideMin(), VecStrideNorm()
@*/
PetscErrorCode  DMClearLocalVectors(DM dm)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    Vec g;

    PetscAssertFalse(dm->localout[i],PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Clearing DM of local vectors that has a local vector obtained with DMGetLocalVector()");
    g = dm->localin[i];
    dm->localin[i] = NULL;
    if (g) {
      DM vdm;

      ierr = VecGetDM(g,&vdm);CHKERRQ(ierr);
      PetscAssertFalse(vdm,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Clearing local vector that has a DM attached");
    }
    ierr = VecDestroy(&g);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   DMRestoreGlobalVector - Returns a PETSc vector that
     obtained from DMGetGlobalVector(). Do not use with vector obtained via
     DMCreateGlobalVector().

   Not Collective

   Input Parameters:
+  dm - the dm
-  g - the global vector

   Level: beginner

.seealso: DMCreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMGlobalToGlobalBegin(),
          DMGlobalToGlobalEnd(), DMGlobalToGlobal(), DMCreateLocalVector(), DMGetGlobalVector()
@*/
PetscErrorCode  DMRestoreGlobalVector(DM dm,Vec *g)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(g,2);
  ierr = VecSetErrorIfLocked(*g, 2);CHKERRQ(ierr);
  for (j=0; j<DM_MAX_WORK_VECTORS; j++) {
    if (*g == dm->globalout[j]) {
      DM vdm;

      ierr = VecGetDM(*g,&vdm);CHKERRQ(ierr);
      PetscAssertFalse(vdm != dm,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Invalid vector");
      ierr = VecSetDM(*g,NULL);CHKERRQ(ierr);
      dm->globalout[j] = NULL;
      for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
        if (!dm->globalin[i]) {
          dm->globalin[i] = *g;
          goto alldone;
        }
      }
    }
  }
  ierr = VecDestroy(g);CHKERRQ(ierr);
alldone:
  *g = NULL;
  PetscFunctionReturn(0);
}

/*@C
   DMHasNamedGlobalVector - check for a named, persistent global vector

   Not Collective

   Input Parameters:
+  dm - DM to hold named vectors
-  name - unique name for Vec

   Output Parameter:
.  exists - true if the vector was previously created

   Level: developer

   Note: If a Vec with the given name does not exist, it is created.

.seealso: DMGetNamedGlobalVector(), DMRestoreNamedLocalVector()
@*/
PetscErrorCode DMHasNamedGlobalVector(DM dm,const char *name,PetscBool *exists)
{
  PetscErrorCode ierr;
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidBoolPointer(exists,3);
  *exists = PETSC_FALSE;
  for (link=dm->namedglobal; link; link=link->next) {
    PetscBool match;
    ierr = PetscStrcmp(name,link->name,&match);CHKERRQ(ierr);
    if (match) {
      *exists = PETSC_TRUE;
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   DMGetNamedGlobalVector - get access to a named, persistent global vector

   Collective on dm

   Input Parameters:
+  dm - DM to hold named vectors
-  name - unique name for Vec

   Output Parameter:
.  X - named Vec

   Level: developer

   Note: If a Vec with the given name does not exist, it is created.

.seealso: DMRestoreNamedGlobalVector()
@*/
PetscErrorCode DMGetNamedGlobalVector(DM dm,const char *name,Vec *X)
{
  PetscErrorCode ierr;
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(X,3);
  for (link=dm->namedglobal; link; link=link->next) {
    PetscBool match;

    ierr = PetscStrcmp(name,link->name,&match);CHKERRQ(ierr);
    if (match) {
      DM vdm;

      PetscAssertFalse(link->status != DMVEC_STATUS_IN,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Vec name '%s' already checked out",name);
      ierr = VecGetDM(link->X,&vdm);CHKERRQ(ierr);
      PetscAssertFalse(vdm,PetscObjectComm((PetscObject)vdm),PETSC_ERR_LIB,"Invalid vector");
      ierr = VecSetDM(link->X,dm);CHKERRQ(ierr);
      goto found;
    }
  }

  /* Create the Vec */
  ierr            = PetscNew(&link);CHKERRQ(ierr);
  ierr            = PetscStrallocpy(name,&link->name);CHKERRQ(ierr);
  ierr            = DMCreateGlobalVector(dm,&link->X);CHKERRQ(ierr);
  link->next      = dm->namedglobal;
  dm->namedglobal = link;

found:
  *X           = link->X;
  link->status = DMVEC_STATUS_OUT;
  PetscFunctionReturn(0);
}

/*@C
   DMRestoreNamedGlobalVector - restore access to a named, persistent global vector

   Collective on dm

   Input Parameters:
+  dm - DM on which the vector was gotten
.  name - name under which the vector was gotten
-  X - Vec to restore

   Level: developer

.seealso: DMGetNamedGlobalVector()
@*/
PetscErrorCode DMRestoreNamedGlobalVector(DM dm,const char *name,Vec *X)
{
  PetscErrorCode ierr;
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(X,3);
  PetscValidHeaderSpecific(*X,VEC_CLASSID,3);
  for (link=dm->namedglobal; link; link=link->next) {
    PetscBool match;

    ierr = PetscStrcmp(name,link->name,&match);CHKERRQ(ierr);
    if (match) {
      DM vdm;

      ierr = VecGetDM(*X,&vdm);CHKERRQ(ierr);
      PetscAssertFalse(link->status != DMVEC_STATUS_OUT,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Vec name '%s' was not checked out",name);
      PetscAssertFalse(link->X != *X,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_INCOMP,"Attempt to restore Vec name '%s', but Vec does not match the cache",name);
      PetscAssertFalse(vdm != dm,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Invalid vector");

      link->status = DMVEC_STATUS_IN;
      ierr         = VecSetDM(link->X,NULL);CHKERRQ(ierr);
      *X           = NULL;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_INCOMP,"Could not find Vec name '%s' to restore",name);
}

/*@C
   DMHasNamedLocalVector - check for a named, persistent local vector

   Not Collective

   Input Parameters:
+  dm - DM to hold named vectors
-  name - unique name for Vec

   Output Parameter:
.  exists - true if the vector was previously created

   Level: developer

   Note: If a Vec with the given name does not exist, it is created.

.seealso: DMGetNamedGlobalVector(), DMRestoreNamedLocalVector()
@*/
PetscErrorCode DMHasNamedLocalVector(DM dm,const char *name,PetscBool *exists)
{
  PetscErrorCode ierr;
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(exists,3);
  *exists = PETSC_FALSE;
  for (link=dm->namedlocal; link; link=link->next) {
    PetscBool match;
    ierr = PetscStrcmp(name,link->name,&match);CHKERRQ(ierr);
    if (match) {
      *exists = PETSC_TRUE;
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   DMGetNamedLocalVector - get access to a named, persistent local vector

   Not Collective

   Input Parameters:
+  dm - DM to hold named vectors
-  name - unique name for Vec

   Output Parameter:
.  X - named Vec

   Level: developer

   Note: If a Vec with the given name does not exist, it is created.

.seealso: DMGetNamedGlobalVector(), DMRestoreNamedLocalVector()
@*/
PetscErrorCode DMGetNamedLocalVector(DM dm,const char *name,Vec *X)
{
  PetscErrorCode ierr;
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(X,3);
  for (link=dm->namedlocal; link; link=link->next) {
    PetscBool match;

    ierr = PetscStrcmp(name,link->name,&match);CHKERRQ(ierr);
    if (match) {
      DM vdm;

      PetscAssertFalse(link->status != DMVEC_STATUS_IN,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Vec name '%s' already checked out",name);
      ierr = VecGetDM(link->X,&vdm);CHKERRQ(ierr);
      PetscAssertFalse(vdm,PetscObjectComm((PetscObject)vdm),PETSC_ERR_LIB,"Invalid vector");
      ierr = VecSetDM(link->X,dm);CHKERRQ(ierr);
      goto found;
    }
  }

  /* Create the Vec */
  ierr           = PetscNew(&link);CHKERRQ(ierr);
  ierr           = PetscStrallocpy(name,&link->name);CHKERRQ(ierr);
  ierr           = DMCreateLocalVector(dm,&link->X);CHKERRQ(ierr);
  link->next     = dm->namedlocal;
  dm->namedlocal = link;

found:
  *X           = link->X;
  link->status = DMVEC_STATUS_OUT;
  PetscFunctionReturn(0);
}

/*@C
   DMRestoreNamedLocalVector - restore access to a named, persistent local vector

   Not Collective

   Input Parameters:
+  dm - DM on which the vector was gotten
.  name - name under which the vector was gotten
-  X - Vec to restore

   Level: developer

.seealso: DMRestoreNamedGlobalVector(), DMGetNamedLocalVector()
@*/
PetscErrorCode DMRestoreNamedLocalVector(DM dm,const char *name,Vec *X)
{
  PetscErrorCode ierr;
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(X,3);
  PetscValidHeaderSpecific(*X,VEC_CLASSID,3);
  for (link=dm->namedlocal; link; link=link->next) {
    PetscBool match;

    ierr = PetscStrcmp(name,link->name,&match);CHKERRQ(ierr);
    if (match) {
      DM vdm;

      ierr = VecGetDM(*X,&vdm);CHKERRQ(ierr);
      PetscAssertFalse(link->status != DMVEC_STATUS_OUT,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Vec name '%s' was not checked out",name);
      PetscAssertFalse(link->X != *X,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_INCOMP,"Attempt to restore Vec name '%s', but Vec does not match the cache",name);
      PetscAssertFalse(vdm != dm,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Invalid vector");

      link->status = DMVEC_STATUS_IN;
      ierr         = VecSetDM(link->X,NULL);CHKERRQ(ierr);
      *X           = NULL;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_INCOMP,"Could not find Vec name '%s' to restore",name);
}
