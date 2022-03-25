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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(g,2);
  for (PetscInt i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (dm->localin[i]) {
      DM vdm;

      *g             = dm->localin[i];
      dm->localin[i] = NULL;

      PetscCall(VecGetDM(*g,&vdm));
      PetscCheck(!vdm,PetscObjectComm((PetscObject)vdm),PETSC_ERR_LIB,"Invalid vector");
      PetscCall(VecSetDM(*g,dm));
      goto alldone;
    }
  }
  PetscCall(DMCreateLocalVector(dm,g));

alldone:
  for (PetscInt i=0; i<DM_MAX_WORK_VECTORS; i++) {
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
  PetscInt       i,j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(g,2);
  for (j=0; j<DM_MAX_WORK_VECTORS; j++) {
    if (*g == dm->localout[j]) {
      DM vdm;

      PetscCall(VecGetDM(*g,&vdm));
      PetscCheckFalse(vdm != dm,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Invalid vector");
      PetscCall(VecSetDM(*g,NULL));
      dm->localout[j] = NULL;
      for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
        if (!dm->localin[i]) {
          dm->localin[i] = *g;
          goto alldone;
        }
      }
    }
  }
  PetscCall(VecDestroy(g));
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
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(g,2);
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    if (dm->globalin[i]) {
      DM vdm;

      *g              = dm->globalin[i];
      dm->globalin[i] = NULL;

      PetscCall(VecGetDM(*g,&vdm));
      PetscCheck(!vdm,PetscObjectComm((PetscObject)vdm),PETSC_ERR_LIB,"Invalid vector");
      PetscCall(VecSetDM(*g,dm));
      goto alldone;
    }
  }
  PetscCall(DMCreateGlobalVector(dm,g));

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
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    Vec g;

    PetscCheckFalse(dm->globalout[i],PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Clearing DM of global vectors that has a global vector obtained with DMGetGlobalVector()");
    g = dm->globalin[i];
    dm->globalin[i] = NULL;
    if (g) {
      DM vdm;

      PetscCall(VecGetDM(g,&vdm));
      PetscCheck(!vdm,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Clearing global vector that has a DM attached");
    }
    PetscCall(VecDestroy(&g));
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
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
    Vec g;

    PetscCheckFalse(dm->localout[i],PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Clearing DM of local vectors that has a local vector obtained with DMGetLocalVector()");
    g = dm->localin[i];
    dm->localin[i] = NULL;
    if (g) {
      DM vdm;

      PetscCall(VecGetDM(g,&vdm));
      PetscCheck(!vdm,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Clearing local vector that has a DM attached");
    }
    PetscCall(VecDestroy(&g));
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
  PetscInt       i,j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(g,2);
  PetscCall(VecSetErrorIfLocked(*g, 2));
  for (j=0; j<DM_MAX_WORK_VECTORS; j++) {
    if (*g == dm->globalout[j]) {
      DM vdm;

      PetscCall(VecGetDM(*g,&vdm));
      PetscCheckFalse(vdm != dm,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Invalid vector");
      PetscCall(VecSetDM(*g,NULL));
      dm->globalout[j] = NULL;
      for (i=0; i<DM_MAX_WORK_VECTORS; i++) {
        if (!dm->globalin[i]) {
          dm->globalin[i] = *g;
          goto alldone;
        }
      }
    }
  }
  PetscCall(VecDestroy(g));
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
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidBoolPointer(exists,3);
  *exists = PETSC_FALSE;
  for (link=dm->namedglobal; link; link=link->next) {
    PetscBool match;
    PetscCall(PetscStrcmp(name,link->name,&match));
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
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(X,3);
  for (link=dm->namedglobal; link; link=link->next) {
    PetscBool match;

    PetscCall(PetscStrcmp(name,link->name,&match));
    if (match) {
      DM vdm;

      PetscCheckFalse(link->status != DMVEC_STATUS_IN,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Vec name '%s' already checked out",name);
      PetscCall(VecGetDM(link->X,&vdm));
      PetscCheck(!vdm,PetscObjectComm((PetscObject)vdm),PETSC_ERR_LIB,"Invalid vector");
      PetscCall(VecSetDM(link->X,dm));
      goto found;
    }
  }

  /* Create the Vec */
  PetscCall(PetscNew(&link));
  PetscCall(PetscStrallocpy(name,&link->name));
  PetscCall(DMCreateGlobalVector(dm,&link->X));
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
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(X,3);
  PetscValidHeaderSpecific(*X,VEC_CLASSID,3);
  for (link=dm->namedglobal; link; link=link->next) {
    PetscBool match;

    PetscCall(PetscStrcmp(name,link->name,&match));
    if (match) {
      DM vdm;

      PetscCall(VecGetDM(*X,&vdm));
      PetscCheckFalse(link->status != DMVEC_STATUS_OUT,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Vec name '%s' was not checked out",name);
      PetscCheckFalse(link->X != *X,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_INCOMP,"Attempt to restore Vec name '%s', but Vec does not match the cache",name);
      PetscCheckFalse(vdm != dm,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Invalid vector");

      link->status = DMVEC_STATUS_IN;
      PetscCall(VecSetDM(link->X,NULL));
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
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidBoolPointer(exists,3);
  *exists = PETSC_FALSE;
  for (link=dm->namedlocal; link; link=link->next) {
    PetscBool match;
    PetscCall(PetscStrcmp(name,link->name,&match));
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
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(X,3);
  for (link=dm->namedlocal; link; link=link->next) {
    PetscBool match;

    PetscCall(PetscStrcmp(name,link->name,&match));
    if (match) {
      DM vdm;

      PetscCheckFalse(link->status != DMVEC_STATUS_IN,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Vec name '%s' already checked out",name);
      PetscCall(VecGetDM(link->X,&vdm));
      PetscCheck(!vdm,PetscObjectComm((PetscObject)vdm),PETSC_ERR_LIB,"Invalid vector");
      PetscCall(VecSetDM(link->X,dm));
      goto found;
    }
  }

  /* Create the Vec */
  PetscCall(PetscNew(&link));
  PetscCall(PetscStrallocpy(name,&link->name));
  PetscCall(DMCreateLocalVector(dm,&link->X));
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
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(X,3);
  PetscValidHeaderSpecific(*X,VEC_CLASSID,3);
  for (link=dm->namedlocal; link; link=link->next) {
    PetscBool match;

    PetscCall(PetscStrcmp(name,link->name,&match));
    if (match) {
      DM vdm;

      PetscCall(VecGetDM(*X,&vdm));
      PetscCheckFalse(link->status != DMVEC_STATUS_OUT,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Vec name '%s' was not checked out",name);
      PetscCheckFalse(link->X != *X,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_INCOMP,"Attempt to restore Vec name '%s', but Vec does not match the cache",name);
      PetscCheckFalse(vdm != dm,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Invalid vector");

      link->status = DMVEC_STATUS_IN;
      PetscCall(VecSetDM(link->X,NULL));
      *X           = NULL;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_INCOMP,"Could not find Vec name '%s' to restore",name);
}
