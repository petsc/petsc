static char help[] = "Test FEM layout with constraints\n\n";

#include <petsc.h>

int main(int argc, char **argv)
{
  DM              dm, pdm;
  PetscSection    section;
  const PetscInt  field = 0;
  char            ifilename[PETSC_MAX_PATH_LEN];
  PetscInt        sdim, s, pStart, pEnd, p, numVS, numPoints;
  PetscInt        constraints[1];
  IS              setIS, pointIS;
  const PetscInt *setID, *pointID;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FEM Layout Options", "ex98");
  PetscCall(PetscOptionsString("-i", "Filename to read", "ex98", ifilename, ifilename, sizeof(ifilename), NULL));
  PetscOptionsEnd();

  PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, ifilename, NULL, PETSC_TRUE, &dm));
  PetscCall(DMPlexDistributeSetDefault(dm, PETSC_FALSE));
  PetscCall(DMSetFromOptions(dm));

  PetscCall(DMPlexDistribute(dm, 0, NULL, &pdm));
  if (pdm) {
    PetscCall(DMDestroy(&dm));
    dm = pdm;
  }
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  /* create a section */
  PetscCall(DMGetDimension(dm, &sdim));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section));
  PetscCall(PetscSectionSetNumFields(section, 1));
  PetscCall(PetscSectionSetFieldName(section, field, "U"));
  PetscCall(PetscSectionSetFieldComponents(section, field, sdim));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscSectionSetChart(section, pStart, pEnd));

  /* initialize the section storage for a P1 field */
  PetscCall(DMPlexGetDepthStratum(dm, 0, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscCall(PetscSectionSetDof(section, p, sdim));
    PetscCall(PetscSectionSetFieldDof(section, p, 0, sdim));
  }

  /* add constraints at all vertices belonging to a vertex set */
  /* first pass is to reserve space                            */
  PetscCall(DMGetLabelSize(dm, "Vertex Sets", &numVS));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "# Vertex set: %" PetscInt_FMT "\n", numVS));
  PetscCall(DMGetLabelIdIS(dm, "Vertex Sets", &setIS));
  PetscCall(ISGetIndices(setIS, &setID));
  for (s = 0; s < numVS; ++s) {
    PetscCall(DMGetStratumIS(dm, "Vertex Sets", setID[s], &pointIS));
    PetscCall(DMGetStratumSize(dm, "Vertex Sets", setID[s], &numPoints));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "set %" PetscInt_FMT " size: %" PetscInt_FMT "\n", s, numPoints));
    PetscCall(ISGetIndices(pointIS, &pointID));
    for (p = 0; p < numPoints; ++p) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\t point %" PetscInt_FMT "\n", pointID[p]));
      PetscCall(PetscSectionSetConstraintDof(section, pointID[p], 1));
      PetscCall(PetscSectionSetFieldConstraintDof(section, pointID[p], field, 1));
    }
    PetscCall(ISRestoreIndices(pointIS, &pointID));
    PetscCall(ISDestroy(&pointIS));
  }

  PetscCall(PetscSectionSetUp(section));

  /* add constraints at all vertices belonging to a vertex set          */
  /* second pass is to assign constraints to a specific component / dof */
  for (s = 0; s < numVS; ++s) {
    PetscCall(DMGetStratumIS(dm, "Vertex Sets", setID[s], &pointIS));
    PetscCall(DMGetStratumSize(dm, "Vertex Sets", setID[s], &numPoints));
    PetscCall(ISGetIndices(pointIS, &pointID));
    for (p = 0; p < numPoints; ++p) {
      constraints[0] = setID[s] % sdim;
      PetscCall(PetscSectionSetConstraintIndices(section, pointID[p], constraints));
      PetscCall(PetscSectionSetFieldConstraintIndices(section, pointID[p], field, constraints));
    }
    PetscCall(ISRestoreIndices(pointIS, &pointID));
    PetscCall(ISDestroy(&pointIS));
  }
  PetscCall(ISRestoreIndices(setIS, &setID));
  PetscCall(ISDestroy(&setIS));
  PetscCall(PetscObjectViewFromOptions((PetscObject)section, NULL, "-dm_section_view"));

  PetscCall(PetscSectionDestroy(&section));
  PetscCall(DMDestroy(&dm));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: exodusii pnetcdf !complex
  testset:
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/SquareFaceSet.exo -dm_view -dm_section_view
    nsize: 1

    test:
      suffix: 0
      args:

TEST*/
