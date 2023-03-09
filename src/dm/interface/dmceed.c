#include <petsc/private/dmimpl.h> /*I      "petscdm.h"          I*/

#ifdef PETSC_HAVE_LIBCEED
  #include <petscdmceed.h>

/*@C
  DMGetCeed - Get the LibCEED context associated with this `DM`

  Not Collective

  Input Parameter:
. DM   - The `DM`

  Output Parameter:
. ceed - The LibCEED context

  Level: intermediate

.seealso: `DM`, `DMCreate()`
@*/
PetscErrorCode DMGetCeed(DM dm, Ceed *ceed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ceed, 2);
  if (!dm->ceed) {
    char        ceedresource[PETSC_MAX_PATH_LEN]; /* libCEED resource specifier */
    const char *prefix;

    PetscCall(PetscStrncpy(ceedresource, "/cpu/self", sizeof(ceedresource)));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix));
    PetscCall(PetscOptionsGetString(NULL, prefix, "-dm_ceed", ceedresource, sizeof(ceedresource), NULL));
    PetscCallCEED(CeedInit(ceedresource, &dm->ceed));
  }
  *ceed = dm->ceed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
