#include <petsc/private/dmlabelimpl.h> /*I      "petscdmlabelephemeral.h"   I*/
#include <petscdmlabelephemeral.h>     /*I      "petscdmplextransform.h"    I*/

/*@
  DMLabelEphemeralGetTransform - Get the transform for this ephemeral label

  Not Collective

  Input Parameter:
. label - the `DMLabel`

  Output Parameter:
. tr - the transform for this ephemeral label

  Level: intermediate

  Note:
  Ephemeral labels are produced automatically from a base label and a `DMPlexTransform`.

.seealso: `DMLabelEphemeralSetTransform()`, `DMLabelEphemeralGetLabel()`, `DMLabelSetType()`
@*/
PetscErrorCode DMLabelEphemeralGetTransform(DMLabel label, DMPlexTransform *tr)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)label, "__transform__", (PetscObject *)tr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLabelEphemeralSetTransform - Set the transform for this ephemeral label

  Not Collective

  Input Parameters:
+ label - the `DMLabel`
- tr    - the transform for this ephemeral label

  Level: intermediate

  Note:
  Ephemeral labels are produced automatically from a base label and a `DMPlexTransform`.

.seealso: `DMLabelEphemeralGetTransform()`, `DMLabelEphemeralSetLabel()`, `DMLabelSetType()`
@*/
PetscErrorCode DMLabelEphemeralSetTransform(DMLabel label, DMPlexTransform tr)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectCompose((PetscObject)label, "__transform__", (PetscObject)tr));
  PetscFunctionReturn(PETSC_SUCCESS);
}
