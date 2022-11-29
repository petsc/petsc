static char help[] = "Tests for VecGetValuesSection / VecSetValuesSection \n\n";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM           dm;
  Vec          v;
  PetscSection section;
  PetscScalar  val[2];
  PetscInt     pStart, pEnd, p;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-d_view"));

  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscSectionSetChart(section, pStart, pEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &pStart, &pEnd));
  for (p = pStart; p < pEnd; p++) PetscCall(PetscSectionSetDof(section, p, 1));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &pStart, &pEnd));
  for (p = pStart; p < pEnd; p++) PetscCall(PetscSectionSetDof(section, p, 2));
  PetscCall(PetscSectionSetUp(section));
  PetscCall(DMSetLocalSection(dm, section));
  PetscCall(PetscSectionViewFromOptions(section, NULL, "-s_view"));

  PetscCall(DMCreateGlobalVector(dm, &v));
  PetscCall(VecViewFromOptions(v, NULL, "-v_view"));

  /* look through all cells and change "cell values" */
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof;

    PetscCall(PetscSectionGetDof(section, p, &dof));
    for (PetscInt d = 0; d < dof; ++d) val[d] = 100 * p + d;
    PetscCall(VecSetValuesSection(v, section, p, val, INSERT_VALUES));
  }
  PetscCall(VecView(v, PETSC_VIEWER_STDOUT_WORLD));

  for (p = pStart; p < pEnd; ++p) {
    PetscScalar *x;
    PetscInt     dof;

    PetscCall(PetscSectionGetDof(section, p, &dof));
    PetscCall(VecGetValuesSection(v, section, p, &x));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Point #%" PetscInt_FMT " %" PetscInt_FMT " dof\n", p, dof));
  }

  PetscCall(VecDestroy(&v));
  PetscCall(PetscSectionDestroy(&section));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/quads-q2.msh

TEST*/
