
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petsc/private/petscimpl.h> /*I   "petscsys.h"    I*/

/*@C
   PetscObjectGetOptions - Gets the options database used by the object that has been set with `PetscObjectSetOptions()`

   Collective on obj

   Input Parameter:
.  obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`.

   Output Parameter:
.  options - the options database

   Note:
   If this is not called the object will use the default options database

   Developer Note:
   This functionality is not used in PETSc and should, perhaps, be removed

  Level: advanced

.seealso: `PetscOptionsCreate()`, `PetscOptionsDestroy()`, `PetscObjectSetOptionsPrefix()`, `PetscObjectAppendOptionsPrefix()`, `PetscObjectPrependOptionsPrefix()`,
          `PetscObjectGetOptionsPrefix()`, `PetscObjectSetOptions()`
@*/
PetscErrorCode PetscObjectGetOptions(PetscObject obj, PetscOptions *options)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  *options = obj->options;
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectSetOptions - Sets the options database used by the object. Call immediately after creating the object.

   Collective on obj

   Input Parameters:
+  obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`.
-  options - the options database, use NULL for default

   Note:
   If this is not called the object will use the default options database

   Developer Note:
   This functionality is not used in PETSc and should, perhaps, be removed

  Level: advanced

.seealso: `PetscOptionsCreate()`, `PetscOptionsDestroy()`, `PetscObjectSetOptionsPrefix()`, `PetscObjectAppendOptionsPrefix()`, `PetscObjectPrependOptionsPrefix()`,
          `PetscObjectGetOptionsPrefix()`, `PetscObjectGetOptions()`
@*/
PetscErrorCode PetscObjectSetOptions(PetscObject obj, PetscOptions options)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  obj->options = options;
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectSetOptionsPrefix - Sets the prefix used for searching for all
   options for the given object in the database.

   Collective on obj

   Input Parameters:
+  obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`.
-  prefix - the prefix string to prepend to option requests of the object.

   Note:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

  Level: advanced

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
  PetscFunctionReturn(0);
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
  char  *buf = obj->prefix;
  size_t len1, len2;

  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  if (!prefix) PetscFunctionReturn(0);
  if (!buf) {
    PetscCall(PetscObjectSetOptionsPrefix(obj, prefix));
    PetscFunctionReturn(0);
  }
  PetscCheck(prefix[0] != '-', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Options prefix should not begin with a hyphen");

  PetscCall(PetscStrlen(prefix, &len1));
  PetscCall(PetscStrlen(buf, &len2));
  PetscCall(PetscMalloc1(1 + len1 + len2, &obj->prefix));
  PetscCall(PetscStrcpy(obj->prefix, buf));
  PetscCall(PetscStrcat(obj->prefix, prefix));
  PetscCall(PetscFree(buf));
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectGetOptionsPrefix - Gets the prefix of the `PetscObject` used for searching in the options database

   Input Parameters:
.  obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`.

   Output Parameters:
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
  PetscFunctionReturn(0);
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
  size_t len1, len2;

  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  buf = obj->prefix;
  if (!prefix) PetscFunctionReturn(0);
  if (!buf) {
    PetscCall(PetscObjectSetOptionsPrefix(obj, prefix));
    PetscFunctionReturn(0);
  }
  PetscCheck(prefix[0] != '-', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Options prefix should not begin with a hyphen");

  PetscCall(PetscStrlen(prefix, &len1));
  PetscCall(PetscStrlen(buf, &len2));
  PetscCall(PetscMalloc1(1 + len1 + len2, &obj->prefix));
  PetscCall(PetscStrcpy(obj->prefix, prefix));
  PetscCall(PetscStrcat(obj->prefix, buf));
  PetscCall(PetscFree(buf));
  PetscFunctionReturn(0);
}
