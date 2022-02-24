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
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetBool(NULL, NULL, "-adapt", &adapt, NULL));
  CHKERRQ(PetscOptionsGetBool(NULL, NULL, "-user_section", &userSection, NULL));

  /* Create a base DMPlex mesh */
  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &base));
  CHKERRQ(DMSetType(base, DMPLEX));
  CHKERRQ(DMSetFromOptions(base));
  CHKERRQ(DMViewFromOptions(base, NULL, "-dm_view"));

  /* Covert Plex mesh to Forest and destroy base */
  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &forest));
  CHKERRQ(DMSetType(forest, DMP4EST));
  CHKERRQ(DMForestSetBaseDM(forest, base));
  CHKERRQ(DMSetUp(forest));
  CHKERRQ(DMDestroy(&base));
  CHKERRQ(DMViewFromOptions(forest, NULL, "-dm_view"));

  if (adapt) {
    /* Adaptively refine the cell 0 of the mesh */
    for (i = 0; i < 3; ++i) {
      DM      postforest;
      DMLabel adaptLabel = NULL;

      CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "adapt", &adaptLabel));
      CHKERRQ(DMLabelSetValue(adaptLabel, 0, DM_ADAPT_REFINE));
      CHKERRQ(DMForestTemplate(forest, PETSC_COMM_WORLD, &postforest));
      CHKERRQ(DMForestSetAdaptivityLabel(postforest, adaptLabel));
      CHKERRQ(DMLabelDestroy(&adaptLabel));
      CHKERRQ(DMSetUp(postforest));
      CHKERRQ(DMDestroy(&forest));
      forest = postforest;
    }
  } else {
    /* Adaptively refine all cells of the mesh */
    PetscInt cStart, cEnd, c;

    for (i = 0; i < 3; ++i) {
      DM      postforest;
      DMLabel adaptLabel = NULL;

      CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "adapt", &adaptLabel));

      CHKERRQ(DMForestGetCellChart(forest, &cStart, &cEnd));
      for (c = cStart; c < cEnd; ++c) {
        CHKERRQ(DMLabelSetValue(adaptLabel, c, DM_ADAPT_REFINE));
      }

      CHKERRQ(DMForestTemplate(forest, PETSC_COMM_WORLD, &postforest));
      CHKERRQ(DMForestSetAdaptivityLabel(postforest, adaptLabel));
      CHKERRQ(DMLabelDestroy(&adaptLabel));
      CHKERRQ(DMSetUp(postforest));
      CHKERRQ(DMDestroy(&forest));
      forest = postforest;
    }
  }
  CHKERRQ(DMViewFromOptions(forest, NULL, "-dm_view"));

  /*  Setup the section*/
  if (userSection) {
    CHKERRQ(DMConvert(forest, DMPLEX, &plex));
    CHKERRQ(DMViewFromOptions(plex, NULL, "-dm_view"));
    CHKERRQ(DMPlexGetDepthStratum(plex, 0, &vStart, &vEnd));
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Vertices [%D, %D)\n", vStart, vEnd));
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL));
    CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) forest), &s));
    CHKERRQ(PetscSectionSetNumFields(s, 1));
    CHKERRQ(PetscSectionSetChart(s, vStart, vEnd));
    for (v = vStart; v < vEnd; ++v) {
      CHKERRQ(PetscSectionSetDof(s, v, 1));
      CHKERRQ(PetscSectionSetFieldDof(s, v, 0, 1));
    }
    CHKERRQ(PetscSectionSetUp(s));
    CHKERRQ(DMSetLocalSection(forest, s));
    CHKERRQ(PetscObjectViewFromOptions((PetscObject) s, NULL, "-my_section_view"));
    CHKERRQ(PetscSectionDestroy(&s));
    CHKERRQ(DMDestroy(&plex));
  } else {
    PetscFE  fe;
    PetscInt dim;

    CHKERRQ(DMGetDimension(forest, &dim));
    CHKERRQ(PetscFECreateLagrange(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, 1, PETSC_DETERMINE, &fe));
    CHKERRQ(DMAddField(forest, NULL, (PetscObject) fe));
    CHKERRQ(PetscFEDestroy(&fe));
    CHKERRQ(DMCreateDS(forest));
  }

  /* Create the global vector*/
  CHKERRQ(DMCreateGlobalVector(forest, &g));
  CHKERRQ(PetscObjectSetName((PetscObject) g, "g"));
  CHKERRQ(VecSet(g, 1.0));

  /* Test global to local*/
  Vec l;
  CHKERRQ(DMCreateLocalVector(forest, &l));
  CHKERRQ(VecZeroEntries(l));
  CHKERRQ(DMGlobalToLocal(forest, g, INSERT_VALUES, l));
  CHKERRQ(VecZeroEntries(g));
  CHKERRQ(DMLocalToGlobal(forest, l, INSERT_VALUES, g));
  CHKERRQ(VecDestroy(&l));

  /*  Save a vector*/
  CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD, "forestHDF.h5", FILE_MODE_WRITE, &viewer));
  CHKERRQ(VecView(g, viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  /* Load another vector to load into*/
  CHKERRQ(DMCreateGlobalVector(forest, &g2));
  CHKERRQ(PetscObjectSetName((PetscObject) g2, "g"));
  CHKERRQ(VecZeroEntries(g2));

  /*  Load a vector*/
  CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD, "forestHDF.h5", FILE_MODE_READ, &viewer));
  CHKERRQ(VecLoad(g2, viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  /*  Check if the data is the same*/
  CHKERRQ(VecAXPY(g2, -1.0, g));
  CHKERRQ(VecNorm(g2, NORM_INFINITY, &nrm));
  PetscCheckFalse(PetscAbsReal(nrm) > PETSC_SMALL,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Invalid difference norm %g", (double) nrm);

  CHKERRQ(VecDestroy(&g));
  CHKERRQ(VecDestroy(&g2));
  CHKERRQ(DMDestroy(&forest));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: hdf5 p4est

  test:
    suffix: 0
    nsize: {{1 2 5}}
    args: -adapt -dm_plex_simplex 0 -dm_plex_box_faces 2,2

TEST*/
