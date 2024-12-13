static char help[] = "Test point location in DA using DMSwarm\n";

#include <petscdmda.h>
#include <petscdmswarm.h>

PetscErrorCode DMSwarmPrint(DM sw)
{
  DMSwarmCellDM celldm;
  PetscReal    *array;
  PetscInt     *pidArray, *cidArray;
  PetscInt      Np, bs, Nfc;
  PetscMPIInt   rank;
  const char  **coordFields, *cellid;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sw), &rank));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCheck(Nfc == 1, PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "We only support a single coordinate field right now, not %" PetscInt_FMT, Nfc);
  PetscCall(DMSwarmGetField(sw, coordFields[0], &bs, NULL, (void **)&array));
  PetscCall(DMSwarmGetField(sw, DMSwarmField_pid, &bs, NULL, (void **)&pidArray));
  PetscCall(DMSwarmGetField(sw, cellid, &bs, NULL, (void **)&cidArray));
  for (PetscInt p = 0; p < Np; ++p) {
    const PetscReal th = PetscAtan2Real(array[2 * p + 1], array[2 * p]) / PETSC_PI;
    const PetscReal r  = PetscSqrtReal(array[2 * p + 1] * array[2 * p + 1] + array[2 * p] * array[2 * p]);
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] p %" PetscInt_FMT " (%+1.4f,%+1.4f) r=%1.2f th=%1.3f*pi cellid=%" PetscInt_FMT "\n", rank, pidArray[p], (double)array[2 * p], (double)array[2 * p + 1], (double)r, (double)th, cidArray[p]));
  }
  PetscCall(DMSwarmRestoreField(sw, coordFields[0], &bs, NULL, (void **)&array));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmField_pid, &bs, NULL, (void **)&pidArray));
  PetscCall(DMSwarmRestoreField(sw, cellid, &bs, NULL, (void **)&pidArray));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM            dm, sw;
  PetscDataType dtype;
  PetscReal    *coords, r, dr;
  PetscInt      Nx = 4, Ny = 4, Np = 8, bs;
  PetscMPIInt   rank, size;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, Nx, Ny, PETSC_DECIDE, PETSC_DECIDE, 1, 2, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMDASetUniformCoordinates(dm, -1., 1., -1., 1., -1., 1.));
  PetscCall(DMViewFromOptions(dm, NULL, "-da_view"));

  PetscCall(DMCreate(PETSC_COMM_WORLD, &sw));
  PetscCall(PetscObjectSetName((PetscObject)sw, "Particle Grid"));
  PetscCall(DMSetType(sw, DMSWARM));
  PetscCall(DMSetDimension(sw, 2));
  PetscCall(DMSwarmSetType(sw, DMSWARM_PIC));
  PetscCall(DMSetFromOptions(sw));
  PetscCall(DMSwarmSetCellDM(sw, dm));
  PetscCall(DMSwarmInitializeFieldRegister(sw));
  PetscCall(DMSwarmRegisterPetscDatatypeField(sw, "u", 1, PETSC_SCALAR));
  PetscCall(DMSwarmFinalizeFieldRegister(sw));
  PetscCall(DMSwarmSetLocalSizes(sw, Np, 2));
  PetscCall(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
  dr = 1.0 / (size + 1);
  r  = (rank + 1) * dr;
  for (PetscInt p = 0; p < Np; ++p) {
    const PetscReal th = (p + 0.5) * 2. * PETSC_PI / Np;

    coords[p * 2 + 0] = r * PetscCosReal(th);
    coords[p * 2 + 1] = r * PetscSinReal(th);
  }
  PetscCall(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
  PetscCall(DMViewFromOptions(sw, NULL, "-swarm_view"));
  PetscCall(DMSwarmPrint(sw));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n... calling DMSwarmMigrate ...\n\n"));
  PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
  PetscCall(DMViewFromOptions(sw, NULL, "-swarm_view"));
  PetscCall(DMSwarmPrint(sw));

  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Swarm does not handle complex or quad
  build:
    requires: !complex double

  test:
    suffix: 0

TEST*/
