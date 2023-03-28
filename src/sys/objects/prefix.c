
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petsc/private/petscimpl.h> /*I   "petscsys.h"    I*/

/*@C
   PetscObjectGetOptions - Gets the options database used by the object that has been set with `PetscObjectSetOptions()`

   Collective

   Input Parameter:
.  obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`.

   Output Parameter:
.  options - the options database

  Level: advanced

   Note:
   If this is not called the object will use the default options database

   Developer Note:
   This functionality is not used in PETSc and should, perhaps, be removed

.seealso: `PetscOptionsCreate()`, `PetscOptionsDestroy()`, `PetscObjectSetOptionsPrefix()`, `PetscObjectAppendOptionsPrefix()`, `PetscObjectPrependOptionsPrefix()`,
          `PetscObjectGetOptionsPrefix()`, `PetscObjectSetOptions()`
@*/
PetscErrorCode PetscObjectGetOptions(PetscObject obj, PetscOptions *options)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  *options = obj->options;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscObjectSetOptions - Sets the options database used by the object. Call immediately after creating the object.

   Collective

   Input Parameters:
+  obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`.
-  options - the options database, use NULL for default

  Level: advanced

   Note:
   If this is not called the object will use the default options database

   Developer Note:
   This functionality is not used in PETSc and should, perhaps, be removed

.seealso: `PetscOptionsCreate()`, `PetscOptionsDestroy()`, `PetscObjectSetOptionsPrefix()`, `PetscObjectAppendOptionsPrefix()`, `PetscObjectPrependOptionsPrefix()`,
          `PetscObjectGetOptionsPrefix()`, `PetscObjectGetOptions()`
@*/
PetscErrorCode PetscObjectSetOptions(PetscObject obj, PetscOptions options)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  obj->options = options;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscObjectSetOptionsPrefix - Sets the prefix used for searching for all
   options for the given object in the database.

   Collective

   Input Parameters:
+  obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`.
-  prefix - the prefix string to prepend to option requests of the object.

  Level: advanced

   Note:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

.seealso: `PetscOptionsCreate()`, `PetscOptionsDestroy()`, `PetscObjectAppendOptionsPrefix()`, `PetscObjectPrependOptionsPrefix()`,
          `PetscObjectGetOptionsPrefix()`, `TSSetOptionsPrefix()`, `SNESSetOptionsPrefix()`, `KSPSetOptionsPrefix()`
@*/
PetscErrorCode PetscObjectSetOptionsPrefix(PetscObject obj, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  if (prefix) {
    PetscValidCharPointer(prefix, 2);
    PetscCheck(prefix[0] != '-', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Options prefix should not begin with a hyphen");
    if (prefix != obj->prefix) {
      PetscCall(PetscFree(obj->prefix));
      PetscCall(PetscStrallocpy(prefix, &obj->prefix));
    }
  } else PetscCall(PetscFree(obj->prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscObjectAppendOptionsPrefix - Appends to the prefix used for searching for options for the given object in the database.

   Input Parameters:
+  obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`.
-  prefix - the prefix string to prepend to option requests of the object.

   Note:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

  Level: advanced

.seealso: `PetscOptionsCreate()`, `PetscOptionsDestroy()`, `PetscObjectSetOptionsPrefix()`, `PetscObjectPrependOptionsPrefix()`,
          `PetscObjectGetOptionsPrefix()`, `TSAppendOptionsPrefix()`, `SNESAppendOptionsPrefix()`, `KSPAppendOptionsPrefix()`
@*/
PetscErrorCode PetscObjectAppendOptionsPrefix(PetscObject obj, const char prefix[])
{
  size_t len1, len2, new_len;

  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  if (!prefix) PetscFunctionReturn(PETSC_SUCCESS);
  if (!obj->prefix) {
    PetscCall(PetscObjectSetOptionsPrefix(obj, prefix));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(prefix[0] != '-', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Options prefix should not begin with a hyphen");

  PetscCall(PetscStrlen(obj->prefix, &len1));
  PetscCall(PetscStrlen(prefix, &len2));
  new_len = len1 + len2 + 1;
  PetscCall(PetscRealloc(new_len * sizeof(*(obj->prefix)), &obj->prefix));
  PetscCall(PetscStrncpy(obj->prefix + len1, prefix, len2 + 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscObjectGetOptionsPrefix - Gets the prefix of the `PetscObject` used for searching in the options database

   Input Parameter:
.  obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`.

   Output Parameter:
.  prefix - pointer to the prefix string used is returned

  Level: advanced

.seealso: `PetscOptionsCreate()`, `PetscOptionsDestroy()`, `PetscObjectSetOptionsPrefix()`, `PetscObjectAppendOptionsPrefix()`, `PetscObjectPrependOptionsPrefix()`,
          `TSGetOptionsPrefix()`, `SNESGetOptionsPrefix()`, `KSPGetOptionsPrefix()`
@*/
PetscErrorCode PetscObjectGetOptionsPrefix(PetscObject obj, const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscValidPointer(prefix, 2);
  *prefix = obj->prefix;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscObjectPrependOptionsPrefix - Sets the prefix used for searching for options of for this object in the database.

   Input Parameters:
+  obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`.
-  prefix - the prefix string to prepend to option requests of the object.

   Note:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

  Level: advanced

.seealso: `PetscOptionsCreate()`, `PetscOptionsDestroy()`, `PetscObjectSetOptionsPrefix()`, `PetscObjectAppendOptionsPrefix()`,
          `PetscObjectGetOptionsPrefix()`
@*/
PetscErrorCode PetscObjectPrependOptionsPrefix(PetscObject obj, const char prefix[])
{
  char  *buf;
  size_t len1, len2, new_len;

  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  if (!prefix) PetscFunctionReturn(PETSC_SUCCESS);
  if (!obj->prefix) {
    PetscCall(PetscObjectSetOptionsPrefix(obj, prefix));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(prefix[0] != '-', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Options prefix should not begin with a hyphen");

  PetscCall(PetscStrlen(prefix, &len1));
  PetscCall(PetscStrlen(obj->prefix, &len2));
  buf     = obj->prefix;
  new_len = len1 + len2 + 1;
  PetscCall(PetscMalloc1(new_len, &obj->prefix));
  PetscCall(PetscStrncpy(obj->prefix, prefix, len1 + 1));
  PetscCall(PetscStrncpy(obj->prefix + len1, buf, len2 + 1));
  PetscCall(PetscFree(buf));
  PetscFunctionReturn(PETSC_SUCCESS);
}
