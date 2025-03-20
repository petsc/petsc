#include <petsc/private/dmpleximpl.h> /*I "petscdmplex.h" I*/
#include <petsc/private/tsimpl.h>     /*I "petscts.h" I*/
#include <petsc/private/snesimpl.h>
#include <petscds.h>
#include <petscfv.h>

PetscErrorCode DMPlexTSComputeRHSFunctionFVMCEED(DM dm, PetscReal time, Vec locX, Vec F, void *user)
{
  PetscFV    fv;
  Vec        locF;
  Ceed       ceed;
  DMCeed     sd = dm->dmceed;
  CeedVector clocX, clocF;

  PetscFunctionBegin;
  PetscCall(DMGetCeed(dm, &ceed));
  PetscCheck(sd, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "This DM has no CEED data. Call DMCeedCreate() before computing the residual.");
  if (time == 0.) PetscCall(DMCeedComputeGeometry(dm, sd));
  PetscCall(DMGetField(dm, 0, NULL, (PetscObject *)&fv));
  PetscCall(DMPlexInsertBoundaryValuesFVM(dm, fv, locX, time, NULL));
  PetscCall(DMGetLocalVector(dm, &locF));
  PetscCall(VecZeroEntries(locF));
  PetscCall(VecGetCeedVectorRead(locX, ceed, &clocX));
  PetscCall(VecGetCeedVector(locF, ceed, &clocF));
  PetscCallCEED(CeedOperatorApplyAdd(sd->op, clocX, clocF, CEED_REQUEST_IMMEDIATE));
  PetscCall(VecRestoreCeedVectorRead(locX, &clocX));
  PetscCall(VecRestoreCeedVector(locF, &clocF));
  PetscCall(DMLocalToGlobalBegin(dm, locF, ADD_VALUES, F));
  PetscCall(DMLocalToGlobalEnd(dm, locF, ADD_VALUES, F));
  PetscCall(DMRestoreLocalVector(dm, &locF));
  PetscCall(VecViewFromOptions(F, NULL, "-fv_rhs_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}
