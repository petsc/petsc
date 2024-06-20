#include <petsc/private/dmimpl.h> /*I "petscdm.h" I*/

/*@
  DMGetLocalVector - Gets a PETSc vector that may be used with the `DM` local routines. This vector has spaces for the ghost values.

  Not Collective

  Input Parameter:
. dm - the `DM`

  Output Parameter:
. g - the local vector

  Level: beginner

  Note:
  The vector values are NOT initialized and may have garbage in them, so you may need
  to zero them.

  The output parameter, `g`, is a regular PETSc vector that should be returned with
  `DMRestoreLocalVector()` DO NOT call `VecDestroy()` on it.

  This is intended to be used for vectors you need for a short time, like within a single function call.
  For vectors that you intend to keep around (for example in a C struct) or pass around large parts of your
  code you should use `DMCreateLocalVector()`.

  VecStride*() operations can be useful when using `DM` with dof > 1

.seealso: `DM`, `DMCreateGlobalVector()`, `VecDuplicate()`, `VecDuplicateVecs()`,
          `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`, `DMGlobalToLocalBegin()`,
          `DMGlobalToLocalEnd()`, `DMLocalToGlobalBegin()`, `DMCreateLocalVector()`, `DMRestoreLocalVector()`,
          `VecStrideMax()`, `VecStrideMin()`, `VecStrideNorm()`, `DMClearLocalVectors()`, `DMGetNamedGlobalVector()`, `DMGetNamedLocalVector()`
@*/
PetscErrorCode DMGetLocalVector(DM dm, Vec *g)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(g, 2);
  for (PetscInt i = 0; i < DM_MAX_WORK_VECTORS; i++) {
    if (dm->localin[i]) {
      DM vdm;

      *g             = dm->localin[i];
      dm->localin[i] = NULL;

      PetscCall(VecGetDM(*g, &vdm));
      PetscCheck(!vdm, PetscObjectComm((PetscObject)vdm), PETSC_ERR_LIB, "Invalid vector");
      PetscCall(VecSetDM(*g, dm));
      goto alldone;
    }
  }
  PetscCall(DMCreateLocalVector(dm, g));

alldone:
  for (PetscInt i = 0; i < DM_MAX_WORK_VECTORS; i++) {
    if (!dm->localout[i]) {
      dm->localout[i] = *g;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMRestoreLocalVector - Returns a PETSc vector that was
  obtained from `DMGetLocalVector()`. Do not use with vector obtained via
  `DMCreateLocalVector()`.

  Not Collective

  Input Parameters:
+ dm - the `DM`
- g  - the local vector

  Level: beginner

.seealso: `DM`, `DMCreateGlobalVector()`, `VecDuplicate()`, `VecDuplicateVecs()`,
          `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`, `DMGlobalToLocalBegin()`,
          `DMGlobalToLocalEnd()`, `DMLocalToGlobalBegin()`, `DMCreateLocalVector()`, `DMGetLocalVector()`, `DMClearLocalVectors()`
@*/
PetscErrorCode DMRestoreLocalVector(DM dm, Vec *g)
{
  PetscInt i, j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(g, 2);
  for (j = 0; j < DM_MAX_WORK_VECTORS; j++) {
    if (*g == dm->localout[j]) {
      DM vdm;

      PetscCall(VecGetDM(*g, &vdm));
      PetscCheck(vdm == dm, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Invalid vector");
      PetscCall(VecSetDM(*g, NULL));
      dm->localout[j] = NULL;
      for (i = 0; i < DM_MAX_WORK_VECTORS; i++) {
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetGlobalVector - Gets a PETSc vector that may be used with the `DM` global routines.

  Collective

  Input Parameter:
. dm - the `DM`

  Output Parameter:
. g - the global vector

  Level: beginner

  Note:
  The vector values are NOT initialized and may have garbage in them, so you may need
  to zero them.

  The output parameter, `g`, is a regular PETSc vector that should be returned with
  `DMRestoreGlobalVector()` DO NOT call `VecDestroy()` on it.

  This is intended to be used for vectors you need for a short time, like within a single function call.
  For vectors that you intend to keep around (for example in a C struct) or pass around large parts of your
  code you should use `DMCreateGlobalVector()`.

  VecStride*() operations can be useful when using `DM` with dof > 1

.seealso: `DM`, `DMCreateGlobalVector()`, `VecDuplicate()`, `VecDuplicateVecs()`,
          `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`, `DMGlobalToLocalBegin()`,
          `DMGlobalToLocalEnd()`, `DMLocalToGlobalBegin()`, `DMCreateLocalVector()`, `DMRestoreLocalVector()`
          `VecStrideMax()`, `VecStrideMin()`, `VecStrideNorm()`, `DMClearGlobalVectors()`, `DMGetNamedGlobalVector()`, `DMGetNamedLocalVector()`
@*/
PetscErrorCode DMGetGlobalVector(DM dm, Vec *g)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(g, 2);
  for (i = 0; i < DM_MAX_WORK_VECTORS; i++) {
    if (dm->globalin[i]) {
      DM vdm;

      *g              = dm->globalin[i];
      dm->globalin[i] = NULL;

      PetscCall(VecGetDM(*g, &vdm));
      PetscCheck(!vdm, PetscObjectComm((PetscObject)vdm), PETSC_ERR_LIB, "Invalid vector");
      PetscCall(VecSetDM(*g, dm));
      goto alldone;
    }
  }
  PetscCall(DMCreateGlobalVector(dm, g));

alldone:
  for (i = 0; i < DM_MAX_WORK_VECTORS; i++) {
    if (!dm->globalout[i]) {
      dm->globalout[i] = *g;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMClearGlobalVectors - Destroys all the global vectors that have been created for `DMGetGlobalVector()` calls in this `DM`

  Collective

  Input Parameter:
. dm - the `DM`

  Level: developer

.seealso: `DM`, `DMCreateGlobalVector()`, `VecDuplicate()`, `VecDuplicateVecs()`,
          `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`, `DMGlobalToLocalBegin()`,
          `DMGlobalToLocalEnd()`, `DMLocalToGlobalBegin()`, `DMCreateLocalVector()`, `DMRestoreLocalVector()`
          `VecStrideMax()`, `VecStrideMin()`, `VecStrideNorm()`, `DMClearLocalVectors()`
@*/
PetscErrorCode DMClearGlobalVectors(DM dm)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for (i = 0; i < DM_MAX_WORK_VECTORS; i++) {
    Vec g;

    PetscCheck(!dm->globalout[i], PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Clearing DM of global vectors that has a global vector obtained with DMGetGlobalVector()");
    g               = dm->globalin[i];
    dm->globalin[i] = NULL;
    if (g) {
      DM vdm;

      PetscCall(VecGetDM(g, &vdm));
      PetscCheck(!vdm, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Clearing global vector that has a DM attached");
    }
    PetscCall(VecDestroy(&g));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMClearLocalVectors - Destroys all the local vectors that have been created for `DMGetLocalVector()` calls in this `DM`

  Collective

  Input Parameter:
. dm - the `DM`

  Level: developer

.seealso: `DM`, `DMCreateLocalVector()`, `VecDuplicate()`, `VecDuplicateVecs()`,
          `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`, `DMLocalToLocalBegin()`,
          `DMLocalToLocalEnd()`, `DMRestoreLocalVector()`
          `VecStrideMax()`, `VecStrideMin()`, `VecStrideNorm()`, `DMClearGlobalVectors()`
@*/
PetscErrorCode DMClearLocalVectors(DM dm)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for (i = 0; i < DM_MAX_WORK_VECTORS; i++) {
    Vec g;

    PetscCheck(!dm->localout[i], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Clearing DM of local vectors that has a local vector obtained with DMGetLocalVector()");
    g              = dm->localin[i];
    dm->localin[i] = NULL;
    if (g) {
      DM vdm;

      PetscCall(VecGetDM(g, &vdm));
      PetscCheck(!vdm, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Clearing local vector that has a DM attached");
    }
    PetscCall(VecDestroy(&g));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMRestoreGlobalVector - Returns a PETSc vector that
  obtained from `DMGetGlobalVector()`. Do not use with vector obtained via
  `DMCreateGlobalVector()`.

  Not Collective

  Input Parameters:
+ dm - the `DM`
- g  - the global vector

  Level: beginner

.seealso: `DM`, `DMCreateGlobalVector()`, `VecDuplicate()`, `VecDuplicateVecs()`,
          `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`, `DMGlobalToGlobalBegin()`,
          `DMGlobalToGlobalEnd()`, `DMGlobalToGlobal()`, `DMCreateLocalVector()`, `DMGetGlobalVector()`, `DMClearGlobalVectors()`
@*/
PetscErrorCode DMRestoreGlobalVector(DM dm, Vec *g)
{
  PetscInt i, j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(g, 2);
  PetscCall(VecSetErrorIfLocked(*g, 2));
  for (j = 0; j < DM_MAX_WORK_VECTORS; j++) {
    if (*g == dm->globalout[j]) {
      DM vdm;

      PetscCall(VecGetDM(*g, &vdm));
      PetscCheck(vdm == dm, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Invalid vector");
      PetscCall(VecSetDM(*g, NULL));
      dm->globalout[j] = NULL;
      for (i = 0; i < DM_MAX_WORK_VECTORS; i++) {
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMClearNamedGlobalVectors - Destroys all the named global vectors that have been created with `DMGetNamedGlobalVector()` in this `DM`

  Collective

  Input Parameter:
. dm - the `DM`

  Level: developer

.seealso: `DM`, `DMGetNamedGlobalVector()`, `DMGetNamedLocalVector()`, `DMClearNamedLocalVectors()`
@*/
PetscErrorCode DMClearNamedGlobalVectors(DM dm)
{
  DMNamedVecLink nnext;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  nnext           = dm->namedglobal;
  dm->namedglobal = NULL;
  for (DMNamedVecLink nlink = nnext; nlink; nlink = nnext) { /* Destroy the named vectors */
    nnext = nlink->next;
    PetscCheck(nlink->status == DMVEC_STATUS_IN, ((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "DM still has global Vec named '%s' checked out", nlink->name);
    PetscCall(PetscFree(nlink->name));
    PetscCall(VecDestroy(&nlink->X));
    PetscCall(PetscFree(nlink));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMClearNamedLocalVectors - Destroys all the named local vectors that have been created with `DMGetNamedLocalVector()` in this `DM`

  Collective

  Input Parameter:
. dm - the `DM`

  Level: developer

.seealso: `DM`, `DMGetNamedGlobalVector()`, `DMGetNamedLocalVector()`, `DMClearNamedGlobalVectors()`
@*/
PetscErrorCode DMClearNamedLocalVectors(DM dm)
{
  DMNamedVecLink nnext;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  nnext          = dm->namedlocal;
  dm->namedlocal = NULL;
  for (DMNamedVecLink nlink = nnext; nlink; nlink = nnext) { /* Destroy the named vectors */
    nnext = nlink->next;
    PetscCheck(nlink->status == DMVEC_STATUS_IN, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "DM still has local Vec named '%s' checked out", nlink->name);
    PetscCall(PetscFree(nlink->name));
    PetscCall(VecDestroy(&nlink->X));
    PetscCall(PetscFree(nlink));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMHasNamedGlobalVector - check for a named, persistent global vector created with `DMGetNamedGlobalVector()`

  Not Collective

  Input Parameters:
+ dm   - `DM` to hold named vectors
- name - unique name for `Vec`

  Output Parameter:
. exists - true if the vector was previously created

  Level: developer

.seealso: `DM`, `DMGetNamedGlobalVector()`, `DMRestoreNamedLocalVector()`, `DMClearNamedGlobalVectors()`
@*/
PetscErrorCode DMHasNamedGlobalVector(DM dm, const char *name, PetscBool *exists)
{
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(exists, 3);
  *exists = PETSC_FALSE;
  for (link = dm->namedglobal; link; link = link->next) {
    PetscBool match;
    PetscCall(PetscStrcmp(name, link->name, &match));
    if (match) {
      *exists = PETSC_TRUE;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetNamedGlobalVector - get access to a named, persistent global vector

  Collective

  Input Parameters:
+ dm   - `DM` to hold named vectors
- name - unique name for `X`

  Output Parameter:
. X - named `Vec`

  Level: developer

  Note:
  If a `Vec` with the given name does not exist, it is created.

.seealso: `DM`, `DMRestoreNamedGlobalVector()`, `DMHasNamedGlobalVector()`, `DMClearNamedGlobalVectors()`, `DMGetGlobalVector()`, `DMGetLocalVector()`
@*/
PetscErrorCode DMGetNamedGlobalVector(DM dm, const char *name, Vec *X)
{
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(X, 3);
  for (link = dm->namedglobal; link; link = link->next) {
    PetscBool match;

    PetscCall(PetscStrcmp(name, link->name, &match));
    if (match) {
      DM vdm;

      PetscCheck(link->status == DMVEC_STATUS_IN, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Vec name '%s' already checked out", name);
      PetscCall(VecGetDM(link->X, &vdm));
      PetscCheck(!vdm, PetscObjectComm((PetscObject)vdm), PETSC_ERR_LIB, "Invalid vector");
      PetscCall(VecSetDM(link->X, dm));
      goto found;
    }
  }

  /* Create the Vec */
  PetscCall(PetscNew(&link));
  PetscCall(PetscStrallocpy(name, &link->name));
  PetscCall(DMCreateGlobalVector(dm, &link->X));
  link->next      = dm->namedglobal;
  dm->namedglobal = link;

found:
  *X           = link->X;
  link->status = DMVEC_STATUS_OUT;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMRestoreNamedGlobalVector - restore access to a named, persistent global vector

  Collective

  Input Parameters:
+ dm   - `DM` on which `X` was gotten
. name - name under which `X` was gotten
- X    - `Vec` to restore

  Level: developer

.seealso: `DM`, `DMGetNamedGlobalVector()`, `DMClearNamedGlobalVectors()`
@*/
PetscErrorCode DMRestoreNamedGlobalVector(DM dm, const char *name, Vec *X)
{
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(X, 3);
  PetscValidHeaderSpecific(*X, VEC_CLASSID, 3);
  for (link = dm->namedglobal; link; link = link->next) {
    PetscBool match;

    PetscCall(PetscStrcmp(name, link->name, &match));
    if (match) {
      DM vdm;

      PetscCall(VecGetDM(*X, &vdm));
      PetscCheck(link->status == DMVEC_STATUS_OUT, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Vec name '%s' was not checked out", name);
      PetscCheck(link->X == *X, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_INCOMP, "Attempt to restore Vec name '%s', but Vec does not match the cache", name);
      PetscCheck(vdm == dm, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Invalid vector");

      link->status = DMVEC_STATUS_IN;
      PetscCall(VecSetDM(link->X, NULL));
      *X = NULL;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_INCOMP, "Could not find Vec name '%s' to restore", name);
}

/*@
  DMHasNamedLocalVector - check for a named, persistent local vector created with `DMGetNamedLocalVector()`

  Not Collective

  Input Parameters:
+ dm   - `DM` to hold named vectors
- name - unique name for `Vec`

  Output Parameter:
. exists - true if the vector was previously created

  Level: developer

  Note:
  If a `Vec` with the given name does not exist, it is created.

.seealso: `DM`, `DMGetNamedGlobalVector()`, `DMRestoreNamedLocalVector()`, `DMClearNamedLocalVectors()`
@*/
PetscErrorCode DMHasNamedLocalVector(DM dm, const char *name, PetscBool *exists)
{
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(exists, 3);
  *exists = PETSC_FALSE;
  for (link = dm->namedlocal; link; link = link->next) {
    PetscBool match;
    PetscCall(PetscStrcmp(name, link->name, &match));
    if (match) {
      *exists = PETSC_TRUE;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetNamedLocalVector - get access to a named, persistent local vector

  Not Collective

  Input Parameters:
+ dm   - `DM` to hold named vectors
- name - unique name for `X`

  Output Parameter:
. X - named `Vec`

  Level: developer

  Note:
  If a `Vec` with the given name does not exist, it is created.

.seealso: `DM`, `DMGetNamedGlobalVector()`, `DMRestoreNamedLocalVector()`, `DMHasNamedLocalVector()`, `DMClearNamedLocalVectors()`, `DMGetGlobalVector()`, `DMGetLocalVector()`
@*/
PetscErrorCode DMGetNamedLocalVector(DM dm, const char *name, Vec *X)
{
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(X, 3);
  for (link = dm->namedlocal; link; link = link->next) {
    PetscBool match;

    PetscCall(PetscStrcmp(name, link->name, &match));
    if (match) {
      DM vdm;

      PetscCheck(link->status == DMVEC_STATUS_IN, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Vec name '%s' already checked out", name);
      PetscCall(VecGetDM(link->X, &vdm));
      PetscCheck(!vdm, PetscObjectComm((PetscObject)vdm), PETSC_ERR_LIB, "Invalid vector");
      PetscCall(VecSetDM(link->X, dm));
      goto found;
    }
  }

  /* Create the Vec */
  PetscCall(PetscNew(&link));
  PetscCall(PetscStrallocpy(name, &link->name));
  PetscCall(DMCreateLocalVector(dm, &link->X));
  link->next     = dm->namedlocal;
  dm->namedlocal = link;

found:
  *X           = link->X;
  link->status = DMVEC_STATUS_OUT;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMRestoreNamedLocalVector - restore access to a named, persistent local vector obtained with `DMGetNamedLocalVector()`

  Not Collective

  Input Parameters:
+ dm   - `DM` on which `X` was gotten
. name - name under which `X` was gotten
- X    - `Vec` to restore

  Level: developer

.seealso: `DM`, `DMRestoreNamedGlobalVector()`, `DMGetNamedLocalVector()`, `DMClearNamedLocalVectors()`
@*/
PetscErrorCode DMRestoreNamedLocalVector(DM dm, const char *name, Vec *X)
{
  DMNamedVecLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscAssertPointer(X, 3);
  PetscValidHeaderSpecific(*X, VEC_CLASSID, 3);
  for (link = dm->namedlocal; link; link = link->next) {
    PetscBool match;

    PetscCall(PetscStrcmp(name, link->name, &match));
    if (match) {
      DM vdm;

      PetscCall(VecGetDM(*X, &vdm));
      PetscCheck(link->status == DMVEC_STATUS_OUT, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Vec name '%s' was not checked out", name);
      PetscCheck(link->X == *X, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_INCOMP, "Attempt to restore Vec name '%s', but Vec does not match the cache", name);
      PetscCheck(vdm == dm, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Invalid vector");

      link->status = DMVEC_STATUS_IN;
      PetscCall(VecSetDM(link->X, NULL));
      *X = NULL;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_INCOMP, "Could not find Vec name '%s' to restore", name);
}
