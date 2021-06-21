#include <petsc/private/dmimpl.h>           /*I      "petscdm.h"          I*/

#ifdef PETSC_HAVE_LIBCEED
#include <petscdmceed.h>

/*@C
  DMGetCeed - Get the LibCEED context associated with this DM

  Not collective

  Input Parameter:
. DM   - The DM

  Output Parameter:
. ceed - The LibCEED context

  Level: intermediate

.seealso: DMCreate()
@*/
PetscErrorCode DMGetCeed(DM dm, Ceed *ceed)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ceed, 2);
  if (!dm->ceed) {
    char        ceedresource[PETSC_MAX_PATH_LEN]; /* libCEED resource specifier */
    const char *prefix;

    ierr = PetscStrcpy(ceedresource, "/cpu/self");CHKERRQ(ierr);
    ierr = PetscObjectGetOptionsPrefix((PetscObject) dm, &prefix);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL, prefix, "-dm_ceed", ceedresource, sizeof(ceedresource), NULL);CHKERRQ(ierr);
    ierr = CeedInit(ceedresource, &dm->ceed);CHKERRQ(ierr);
  }
  *ceed = dm->ceed;
  PetscFunctionReturn(0);
}

#endif
