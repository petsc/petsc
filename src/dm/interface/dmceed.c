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
  PetscAssertPointer(ceed, 2);
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

static CeedMemType PetscMemType2Ceed(PetscMemType mem_type) {
  return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST;
}

PetscErrorCode VecGetCeedVector(Vec X, Ceed ceed, CeedVector *cx)
{
  PetscMemType memtype;
  PetscScalar *x;
  PetscInt     n;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(X, &n));
  PetscCall(VecGetArrayAndMemType(X, &x, &memtype));
  PetscCallCEED(CeedVectorCreate(ceed, n, cx));
  PetscCallCEED(CeedVectorSetArray(*cx, PetscMemType2Ceed(memtype), CEED_USE_POINTER, x));
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreCeedVector(Vec X, CeedVector *cx)
{
  PetscFunctionBegin;
  PetscCall(VecRestoreArrayAndMemType(X, NULL));
  PetscCallCEED(CeedVectorDestroy(cx));
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetCeedVectorRead(Vec X, Ceed ceed, CeedVector *cx)
{
  PetscMemType       memtype;
  const PetscScalar *x;
  PetscInt           n;
  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(X, &n));
  PetscCall(VecGetArrayReadAndMemType(X, &x, &memtype));
  PetscCallCEED(CeedVectorCreate(ceed, n, cx));
  PetscCallCEED(CeedVectorSetArray(*cx, PetscMemType2Ceed(memtype), CEED_USE_POINTER, (PetscScalar*)x));
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreCeedVectorRead(Vec X, CeedVector *cx)
{
  PetscFunctionBegin;
  PetscCall(VecRestoreArrayReadAndMemType(X, NULL));
  PetscCallCEED(CeedVectorDestroy(cx));
  PetscFunctionReturn(0);
}

#endif
