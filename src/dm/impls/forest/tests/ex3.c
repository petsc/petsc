static char help[] = "Tests adaptive refinement using DMForest, and uses HDF5.\n\n";

#include <petscdmforest.h>
#include <petscdmplex.h>
#include <petscviewerhdf5.h>

int main (int argc, char **argv)
{
  DM             base, forest, plex;
  PetscSection   s;
  PetscViewer    viewer;
  Vec            g = NULL, g2 = NULL;
  PetscReal      nrm;
  PetscBool      adapt = PETSC_FALSE, userSection = PETSC_FALSE;
  PetscInt       vStart, vEnd, v, i;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-adapt", &adapt, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-user_section", &userSection, NULL));

  /* Create a base DMPlex mesh */
  PetscCall(DMCreate(PETSC_COMM_WORLD, &base));
  PetscCall(DMSetType(base, DMPLEX));
  PetscCall(DMSetFromOptions(base));
  PetscCall(DMViewFromOptions(base, NULL, "-dm_view"));

  /* Covert Plex mesh to Forest and destroy base */
  PetscCall(DMCreate(PETSC_COMM_WORLD, &forest));
  PetscCall(DMSetType(forest, DMP4EST));
  PetscCall(DMForestSetBaseDM(forest, base));
  PetscCall(DMSetUp(forest));
  PetscCall(DMDestroy(&base));
  PetscCall(DMViewFromOptions(forest, NULL, "-dm_view"));

  if (adapt) {
    /* Adaptively refine the cell 0 of the mesh */
    for (i = 0; i < 3; ++i) {
      DM      postforest;
      DMLabel adaptLabel = NULL;

      PetscCall(DMLabelCreate(PETSC_COMM_SELF, "adapt", &adaptLabel));
      PetscCall(DMLabelSetValue(adaptLabel, 0, DM_ADAPT_REFINE));
      PetscCall(DMForestTemplate(forest, PETSC_COMM_WORLD, &postforest));
      PetscCall(DMForestSetAdaptivityLabel(postforest, adaptLabel));
      PetscCall(DMLabelDestroy(&adaptLabel));
      PetscCall(DMSetUp(postforest));
      PetscCall(DMDestroy(&forest));
      forest = postforest;
    }
  } else {
    /* Adaptively refine all cells of the mesh */
    PetscInt cStart, cEnd, c;

    for (i = 0; i < 3; ++i) {
      DM      postforest;
      DMLabel adaptLabel = NULL;

      PetscCall(DMLabelCreate(PETSC_COMM_SELF, "adapt", &adaptLabel));

      PetscCall(DMForestGetCellChart(forest, &cStart, &cEnd));
      for (c = cStart; c < cEnd; ++c) {
        PetscCall(DMLabelSetValue(adaptLabel, c, DM_ADAPT_REFINE));
      }

      PetscCall(DMForestTemplate(forest, PETSC_COMM_WORLD, &postforest));
      PetscCall(DMForestSetAdaptivityLabel(postforest, adaptLabel));
      PetscCall(DMLabelDestroy(&adaptLabel));
      PetscCall(DMSetUp(postforest));
      PetscCall(DMDestroy(&forest));
      forest = postforest;
    }
  }
  PetscCall(DMViewFromOptions(forest, NULL, "-dm_view"));

  /*  Setup the section*/
  if (userSection) {
    PetscCall(DMConvert(forest, DMPLEX, &plex));
    PetscCall(DMViewFromOptions(plex, NULL, "-dm_view"));
    PetscCall(DMPlexGetDepthStratum(plex, 0, &vStart, &vEnd));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Vertices [%" PetscInt_FMT ", %" PetscInt_FMT ")\n", vStart, vEnd));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL));
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject) forest), &s));
    PetscCall(PetscSectionSetNumFields(s, 1));
    PetscCall(PetscSectionSetChart(s, vStart, vEnd));
    for (v = vStart; v < vEnd; ++v) {
      PetscCall(PetscSectionSetDof(s, v, 1));
      PetscCall(PetscSectionSetFieldDof(s, v, 0, 1));
    }
    PetscCall(PetscSectionSetUp(s));
    PetscCall(DMSetLocalSection(forest, s));
    PetscCall(PetscObjectViewFromOptions((PetscObject) s, NULL, "-my_section_view"));
    PetscCall(PetscSectionDestroy(&s));
    PetscCall(DMDestroy(&plex));
  } else {
    PetscFE  fe;
    PetscInt dim;

    PetscCall(DMGetDimension(forest, &dim));
    PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, 1, PETSC_DETERMINE, &fe));
    PetscCall(DMAddField(forest, NULL, (PetscObject) fe));
    PetscCall(PetscFEDestroy(&fe));
    PetscCall(DMCreateDS(forest));
  }

  /* Create the global vector*/
  PetscCall(DMCreateGlobalVector(forest, &g));
  PetscCall(PetscObjectSetName((PetscObject) g, "g"));
  PetscCall(VecSet(g, 1.0));

  /* Test global to local*/
  Vec l;
  PetscCall(DMCreateLocalVector(forest, &l));
  PetscCall(VecZeroEntries(l));
  PetscCall(DMGlobalToLocal(forest, g, INSERT_VALUES, l));
  PetscCall(VecZeroEntries(g));
  PetscCall(DMLocalToGlobal(forest, l, INSERT_VALUES, g));
  PetscCall(VecDestroy(&l));

  /*  Save a vector*/
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, "forestHDF.h5", FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(g, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /* Load another vector to load into*/
  PetscCall(DMCreateGlobalVector(forest, &g2));
  PetscCall(PetscObjectSetName((PetscObject) g2, "g"));
  PetscCall(VecZeroEntries(g2));

  /*  Load a vector*/
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, "forestHDF.h5", FILE_MODE_READ, &viewer));
  PetscCall(VecLoad(g2, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /*  Check if the data is the same*/
  PetscCall(VecAXPY(g2, -1.0, g));
  PetscCall(VecNorm(g2, NORM_INFINITY, &nrm));
  PetscCheck(PetscAbsReal(nrm) <= PETSC_SMALL,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Invalid difference norm %g", (double) nrm);

  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&g2));
  PetscCall(DMDestroy(&forest));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: hdf5 p4est

  test:
    suffix: 0
    nsize: {{1 2 5}}
    args: -adapt -dm_plex_simplex 0 -dm_plex_box_faces 2,2

TEST*/
