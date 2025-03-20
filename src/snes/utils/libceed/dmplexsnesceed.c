#include <petsc/private/dmpleximpl.h> /*I "petscdmplex.h" I*/
#include <petsc/private/snesimpl.h>   /*I "petscsnes.h"   I*/
#include <petscds.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/petscfeimpl.h>
#include <petscdmceed.h>
#include <petscdmplexceed.h>

PetscErrorCode DMPlexSNESComputeResidualCEED(DM dm, Vec locX, Vec locF, void *user)
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
