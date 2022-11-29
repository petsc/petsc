static char help[] = "Test DMPlexGetCellType\n\n";

#include <petsc.h>

int main(int argc, char **argv)
{
  DM             dm, pdm;
  char           ifilename[PETSC_MAX_PATH_LEN];
  PetscInt       pStart, pEnd, p;
  DMPolytopeType cellType;
  DMLabel        label;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FEM Layout Options", "ex97");
  PetscCall(PetscOptionsString("-i", "Filename to read", "ex97", ifilename, ifilename, sizeof(ifilename), NULL));
  PetscOptionsEnd();

  PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, ifilename, NULL, PETSC_TRUE, &dm));
  PetscCall(DMPlexDistributeSetDefault(dm, PETSC_FALSE));
  PetscCall(DMSetFromOptions(dm));

  PetscCall(DMPlexDistribute(dm, 0, NULL, &pdm));
  if (pdm) {
    PetscCall(DMDestroy(&dm));
    dm = pdm;
  }
  PetscCall(PetscObjectSetName((PetscObject)dm, "ex97"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(DMGetLabel(dm, "celltype", &label));
  PetscCall(DMLabelView(label, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscCall(DMPlexGetCellType(dm, p, &cellType));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "cell: %" PetscInt_FMT " type: %d\n", p, cellType));
  }
  PetscCall(DMDestroy(&dm));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: !complex
  testset:
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh -dm_view
    nsize: 1
    test:
      suffix: 0
      args:
TEST*/
