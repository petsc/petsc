#include "petscsystypes.h"
static char help[] = "Test PetscViewer_ExodusII\n\n";

#include <petsc.h>
#include <exodusII.h>

int main(int argc, char **argv)
{
  DM                 dm, pdm;
  PetscInt           ovlp = 0;
  char               ifilename[PETSC_MAX_PATH_LEN], ofilename[PETSC_MAX_PATH_LEN];
  PetscExodusIIInt   numZVars, numNVars;
  PetscExodusIIInt   nNodalVar = 4;
  PetscExodusIIInt   nZonalVar = 3;
  PetscInt           order     = 1;
  PetscViewer        viewer;
  PetscExodusIIInt   index           = -1;
  const char        *nodalVarName[4] = {"U_x", "U_y", "Alpha", "Beta"};
  const char        *zonalVarName[3] = {"Sigma_11", "Sigma_12", "Sigma_22"};
  const char        *testNames[3]    = {"U", "Sigma", "Gamma"};
  const char *const *varNames;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "PetscViewer_ExodusII test", "ex96");
  PetscCall(PetscOptionsString("-i", "Filename to read", "ex96", ifilename, ifilename, sizeof(ifilename), NULL));
  PetscCall(PetscOptionsString("-o", "Filename to write", "ex96", ofilename, ofilename, sizeof(ofilename), NULL));
  PetscOptionsEnd();

#ifdef PETSC_USE_DEBUG
  PetscCallExternal(ex_opts, EX_VERBOSE + EX_DEBUG);
#endif

  PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, ifilename, NULL, PETSC_TRUE, &dm));
  PetscCall(DMPlexDistributeSetDefault(dm, PETSC_FALSE));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(PetscObjectSetName((PetscObject)dm, "ex96"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(PetscViewerExodusIIOpen(PETSC_COMM_WORLD, ofilename, FILE_MODE_WRITE, &viewer));

  /* Save the geometry to the file, erasing all previous content */
  PetscCall(PetscViewerExodusIISetOrder(viewer, order));
  PetscCall(DMView(dm, viewer));
  PetscCall(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerFlush(viewer));

  PetscCall(DMPlexDistribute(dm, ovlp, NULL, &pdm));
  if (!pdm) pdm = dm;

  /* Testing Variable Number*/
  PetscCall(PetscViewerExodusIISetZonalVariable(viewer, nZonalVar));
  nZonalVar = -1;
  PetscCall(PetscViewerExodusIIGetZonalVariable(viewer, &nZonalVar));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of zonal variables: %d\n", nZonalVar));

  PetscCall(PetscViewerExodusIISetNodalVariable(viewer, nNodalVar));
  nNodalVar = -1;
  PetscCall(PetscViewerExodusIIGetNodalVariable(viewer, &nNodalVar));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of nodal variables: %d\n", nNodalVar));
  PetscCall(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD));

  /*
    Test of PetscViewerExodusIISet[Nodal/Zonal]VariableNames
  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Testing PetscViewerExodusIISet[Nodal/Zonal]VariableNames\n"));
  PetscCall(PetscViewerExodusIISetZonalVariableNames(viewer, zonalVarName));
  PetscCall(PetscViewerExodusIISetNodalVariableNames(viewer, nodalVarName));
  PetscCall(PetscViewerView(viewer, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerDestroy(&viewer));

  /*
    Test of PetscViewerExodusIIGet[Nodal/Zonal]VariableNames
  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n\nReopenning the output file in Read-only mode\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Testing PetscViewerExodusIIGet[Nodal/Zonal]VariableNames\n"));
  PetscCall(PetscViewerExodusIIOpen(PETSC_COMM_WORLD, ofilename, FILE_MODE_APPEND, &viewer));
  PetscCall(PetscViewerExodusIISetOrder(viewer, order));

  PetscCall(PetscViewerExodusIIGetNodalVariableNames(viewer, &numNVars, &varNames));
  for (PetscExodusIIInt i = 0; i < numNVars; i++) {
    PetscCall(PetscViewerExodusIIGetNodalVariableIndex(viewer, varNames[i], &index));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Nodal variable %d: %s, index in file %d\n", i, varNames[i], index));
  }
  for (PetscExodusIIInt i = 0; i < 3; i++) {
    PetscCall(PetscViewerExodusIIGetNodalVariableIndex(viewer, testNames[i], &index));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Nodal variable %d: %s, index in file %d\n", i, testNames[i], index));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
  PetscCall(PetscViewerExodusIIGetZonalVariableNames(viewer, &numZVars, &varNames));
  for (PetscExodusIIInt i = 0; i < numZVars; i++) {
    PetscCall(PetscViewerExodusIIGetZonalVariableIndex(viewer, varNames[i], &index));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Zonal variable %d: %s, index in file %d\n", i, varNames[i], index));
  }
  for (PetscExodusIIInt i = 0; i < 3; i++) {
    PetscCall(PetscViewerExodusIIGetZonalVariableIndex(viewer, testNames[i], &index));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Zonal variable %d: %s, index in file %d\n", i, testNames[i], index));
  }

  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: exodusii pnetcdf !complex
  test:
    suffix: 0
    nsize: 1
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareT-large.exo -o FourSquareT-large_out.exo

TEST*/
