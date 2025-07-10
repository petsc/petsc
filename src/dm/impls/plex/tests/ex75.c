static char help[] = "Tests for submesh with both CG and DG coordinates\n\n";

#include <petscdmplex.h>
#include <petsc/private/dmimpl.h>

int main(int argc, char **argv)
{
  PetscInt       dim = 1, d, cStart, cEnd, c, q, degree = 2, coordSize, offset;
  PetscReal      R = 1.0;
  DM             dm, coordDM, subdm;
  PetscSection   coordSec;
  Vec            coordVec;
  PetscScalar   *coords;
  DMLabel        filter;
  const PetscInt filterValue = 1;
  MPI_Comm       comm;
  PetscMPIInt    size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size != 1) {
    PetscCall(PetscPrintf(comm, "This example is specifically designed for comm size == 1.\n"));
    PetscCall(PetscFinalize());
    return 0;
  }
  /* Make a unit circle with 16 elements. */
  PetscCall(DMPlexCreateSphereMesh(comm, dim, PETSC_TRUE, R, &dm));
  PetscUseTypeMethod(dm, createcellcoordinatedm, &coordDM);
  PetscCall(DMSetCellCoordinateDM(dm, coordDM));
  PetscCall(DMDestroy(&coordDM));
  PetscCall(DMGetCellCoordinateSection(dm, &coordSec));
  PetscCall(PetscSectionSetNumFields(coordSec, 1));
  PetscCall(PetscSectionSetFieldComponents(coordSec, 0, dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(PetscSectionSetChart(coordSec, cStart, cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscCall(PetscSectionSetDof(coordSec, c, dim * (degree + 1)));
    PetscCall(PetscSectionSetFieldDof(coordSec, c, 0, dim * (degree + 1)));
  }
  PetscCall(PetscSectionSetUp(coordSec));
  PetscCall(PetscSectionGetStorageSize(coordSec, &coordSize));
  PetscCall(VecCreate(PETSC_COMM_SELF, &coordVec));
  PetscCall(PetscObjectSetName((PetscObject)coordVec, "cellcoordinates"));
  PetscCall(VecSetBlockSize(coordVec, dim));
  PetscCall(VecSetSizes(coordVec, coordSize, PETSC_DETERMINE));
  PetscCall(VecSetType(coordVec, VECSTANDARD));
  PetscCall(VecGetArray(coordVec, &coords));
  for (c = cStart; c < cEnd; ++c) {
    PetscCall(PetscSectionGetOffset(coordSec, c, &offset));
    for (q = 0; q < degree + 1; ++q) {
      /* Make some DG coordinates. Note that dim = 1.*/
      for (d = 0; d < dim; ++d) coords[offset + dim * q + d] = 100. + (PetscScalar)c + (1.0 / (PetscScalar)degree) * (PetscScalar)q;
    }
  }
  PetscCall(VecRestoreArray(coordVec, &coords));
  PetscCall(DMSetCellCoordinatesLocal(dm, coordVec));
  PetscCall(VecDestroy(&coordVec));
  /* Make submesh. */
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "filter", &filter));
  PetscCall(DMLabelSetValue(filter, 15, filterValue)); /* last cell */
  PetscCall(DMPlexFilter(dm, filter, filterValue, PETSC_FALSE, PETSC_FALSE, NULL, &subdm));
  PetscCall(DMLabelDestroy(&filter));
  PetscCall(PetscObjectSetName((PetscObject)subdm, "Example_SubDM"));
  PetscCall(DMViewFromOptions(subdm, NULL, "-dm_view"));
  PetscCall(DMDestroy(&subdm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -dm_view ascii::ascii_info_detail

TEST*/
