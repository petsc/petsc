#include <../src/ts/characteristic/impls/da/slda.h>       /*I  "petsccharacteristic.h"  I*/
#include <petscdmda.h>
#include <petscviewer.h>

PetscErrorCode CharacteristicView_DA(Characteristic c, PetscViewer viewer)
{
  Characteristic_DA *da = (Characteristic_DA*) c->data;
  PetscBool         iascii, isstring;

  PetscFunctionBegin;
  /* Pull out field names from DM */
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERSTRING, &isstring));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  DMDA: dummy=%" PetscInt_FMT "\n", da->dummy));
  } else if (isstring) {
    PetscCall(PetscViewerStringSPrintf(viewer,"dummy %" PetscInt_FMT, da->dummy));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CharacteristicDestroy_DA(Characteristic c)
{
  Characteristic_DA *da = (Characteristic_DA*) c->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(da));
  PetscFunctionReturn(0);
}

PetscErrorCode CharacteristicSetUp_DA(Characteristic c)
{
  PetscMPIInt    blockLen[2];
  MPI_Aint       indices[2];
  MPI_Datatype   oldtypes[2];
  PetscInt       dim, numValues;

  PetscCall(DMDAGetInfo(c->velocityDA, &dim, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  if (c->structured) c->numIds = dim;
  else c->numIds = 3;
  PetscCheck(c->numFieldComp <= MAX_COMPONENTS,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "The maximum number of fields allowed is %d, you have %" PetscInt_FMT ". You can recompile after increasing MAX_COMPONENTS.", MAX_COMPONENTS, c->numFieldComp);
  numValues = 4 + MAX_COMPONENTS;

  /* Create new MPI datatype for communication of characteristic point structs */
  blockLen[0] = 1+c->numIds; indices[0] = 0;                              oldtypes[0] = MPIU_INT;
  blockLen[1] = numValues;   indices[1] = (1+c->numIds)*sizeof(PetscInt); oldtypes[1] = MPIU_SCALAR;
  PetscCallMPI(MPI_Type_create_struct(2, blockLen, indices, oldtypes, &c->itemType));
  PetscCallMPI(MPI_Type_commit(&c->itemType));

  /* Initialize the local queue for char foot values */
  PetscCall(VecGetLocalSize(c->velocity, &c->queueMax));
  PetscCall(PetscMalloc1(c->queueMax, &c->queue));
  c->queueSize = 0;

  /* Allocate communication structures */
  PetscCheck(c->numNeighbors > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Invalid number of neighbors %" PetscInt_FMT ". Call CharactersiticSetNeighbors() before setup.", c->numNeighbors);
  PetscCall(PetscMalloc1(c->numNeighbors, &c->needCount));
  PetscCall(PetscMalloc1(c->numNeighbors, &c->localOffsets));
  PetscCall(PetscMalloc1(c->numNeighbors, &c->fillCount));
  PetscCall(PetscMalloc1(c->numNeighbors, &c->remoteOffsets));
  PetscCall(PetscMalloc1(c->numNeighbors-1, &c->request));
  PetscCall(PetscMalloc1(c->numNeighbors-1,  &c->status));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode CharacteristicCreate_DA(Characteristic c)
{
  Characteristic_DA *da;

  PetscFunctionBegin;
  PetscCall(PetscNew(&da));
  PetscCall(PetscMemzero(da, sizeof(Characteristic_DA)));
  PetscCall(PetscLogObjectMemory((PetscObject)c, sizeof(Characteristic_DA)));
  c->data = (void*) da;

  c->ops->setup   = CharacteristicSetUp_DA;
  c->ops->destroy = CharacteristicDestroy_DA;
  c->ops->view    = CharacteristicView_DA;

  da->dummy = 0;
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------
   Checks for periodicity of a DM and Maps points outside of a domain back onto the domain
   using appropriate periodicity. At the moment assumes only a 2-D DMDA.
   ----------------------------------------------------------------------------------------*/
PetscErrorCode DMDAMapCoordsToPeriodicDomain(DM da, PetscScalar *x, PetscScalar *y)
{
  DMBoundaryType bx, by;
  PetscInt       dim, gx, gy;

  PetscFunctionBegin;
  PetscCall(DMDAGetInfo(da, &dim, &gx, &gy, NULL, NULL, NULL, NULL, NULL, NULL, &bx, &by, NULL, NULL));

  if (bx == DM_BOUNDARY_PERIODIC) {
      while (*x >= (PetscScalar)gx) *x -= (PetscScalar)gx;
      while (*x < 0.0)              *x += (PetscScalar)gx;
    }
    if (by == DM_BOUNDARY_PERIODIC) {
      while (*y >= (PetscScalar)gy) *y -= (PetscScalar)gy;
      while (*y < 0.0)              *y += (PetscScalar)gy;
    }
  PetscFunctionReturn(0);
}
