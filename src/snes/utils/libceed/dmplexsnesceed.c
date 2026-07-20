#include <petsc/private/dmpleximpl.h> /*I "petscdmplex.h" I*/
#include <petsc/private/snesimpl.h>   /*I "petscsnes.h"   I*/
#include <petscds.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/petscfeimpl.h>
#include <petscdmceed.h>
#include <petscdmplexceed.h>

/*@C
  DMPlexSNESComputeResidualCEED - Assemble the local residual for a `SNES` on a `DMPLEX` using the libCEED operator attached to the `DM`

  Collective

  Input Parameters:
+ dm   - the `DMPLEX` for which libCEED operators have been created by `DMCeedCreate()`
. locX - local solution vector including ghost values
- ctx  - application context (unused)

  Output Parameter:
. locF - local residual vector to be assembled

  Level: developer

  Note:
  This is normally installed as the `SNES` local residual callback via `DMSNESSetFunctionLocal()` when using libCEED for the finite-element evaluation.

.seealso: [](ch_snes), `SNES`, `DMPLEX`, `DMCeedCreate()`, `DMSNESSetFunctionLocal()`, `DMPlexTSComputeRHSFunctionFVMCEED()`
@*/
PetscErrorCode DMPlexSNESComputeResidualCEED(DM dm, Vec locX, Vec locF, PetscCtx ctx)
{
  Ceed       ceed;
  DMCeed     sd = dm->dmceed;
  CeedVector clocX, clocF;

  PetscFunctionBegin;
  PetscCall(DMGetCeed(dm, &ceed));
  PetscCheck(sd, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "This DM has no CEED data. Call DMCeedCreate() before computing the residual.");
  PetscCall(DMCeedComputeGeometry(dm, sd));

  PetscCall(VecGetCeedVectorRead(locX, ceed, &clocX));
  PetscCall(VecGetCeedVector(locF, ceed, &clocF));
  PetscCallCEED(CeedOperatorApplyAdd(sd->op, clocX, clocF, CEED_REQUEST_IMMEDIATE));
  PetscCall(VecRestoreCeedVectorRead(locX, &clocX));
  PetscCall(VecRestoreCeedVector(locF, &clocF));

  {
    DM_Plex *mesh = (DM_Plex *)dm->data;

    if (mesh->printFEM) {
      PetscSection section;
      Vec          locFbc;
      PetscInt     pStart, pEnd, p, maxDof;
      PetscScalar *zeroes;

      PetscCall(DMGetLocalSection(dm, &section));
      PetscCall(VecDuplicate(locF, &locFbc));
      PetscCall(VecCopy(locF, locFbc));
      PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
      PetscCall(PetscSectionGetMaxDof(section, &maxDof));
      PetscCall(PetscCalloc1(maxDof, &zeroes));
      for (p = pStart; p < pEnd; ++p) PetscCall(VecSetValuesSection(locFbc, section, p, zeroes, INSERT_BC_VALUES));
      PetscCall(PetscFree(zeroes));
      PetscCall(DMPrintLocalVec(dm, "Residual", mesh->printTol, locFbc));
      PetscCall(VecDestroy(&locFbc));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
