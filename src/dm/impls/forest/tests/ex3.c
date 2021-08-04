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
  ierr = PetscOptionsGetBool(NULL, NULL, "-adapt", &adapt, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-user_section", &userSection, NULL);CHKERRQ(ierr);

  /* Create a base DMPlex mesh */
  ierr = DMCreate(PETSC_COMM_WORLD, &base);CHKERRQ(ierr);
  ierr = DMSetType(base, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(base);CHKERRQ(ierr);
  ierr = DMViewFromOptions(base, NULL, "-dm_view");CHKERRQ(ierr);

  /* Covert Plex mesh to Forest and destroy base */
  ierr = DMCreate(PETSC_COMM_WORLD, &forest);CHKERRQ(ierr);
  ierr = DMSetType(forest, DMP4EST);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(forest, base);CHKERRQ(ierr);
  ierr = DMSetUp(forest);CHKERRQ(ierr);
  ierr = DMDestroy(&base);CHKERRQ(ierr);
  ierr = DMViewFromOptions(forest, NULL, "-dm_view");CHKERRQ(ierr);

  if (adapt) {
    /* Adaptively refine the cell 0 of the mesh */
    for (i = 0; i < 3; ++i) {
      DM      postforest;
      DMLabel adaptLabel = NULL;

      ierr = DMLabelCreate(PETSC_COMM_SELF, "adapt", &adaptLabel);CHKERRQ(ierr);
      ierr = DMLabelSetValue(adaptLabel, 0, DM_ADAPT_REFINE);CHKERRQ(ierr);
      ierr = DMForestTemplate(forest, PETSC_COMM_WORLD, &postforest);CHKERRQ(ierr);
      ierr = DMForestSetAdaptivityLabel(postforest, adaptLabel);CHKERRQ(ierr);
      ierr = DMLabelDestroy(&adaptLabel);CHKERRQ(ierr);
      ierr = DMSetUp(postforest);CHKERRQ(ierr);
      ierr = DMDestroy(&forest);CHKERRQ(ierr);
      forest = postforest;
    }
  } else {
    /* Adaptively refine all cells of the mesh */
    PetscInt cStart, cEnd, c;

    for (i = 0; i < 3; ++i) {
      DM      postforest;
      DMLabel adaptLabel = NULL;

      ierr = DMLabelCreate(PETSC_COMM_SELF, "adapt", &adaptLabel);CHKERRQ(ierr);

      ierr = DMForestGetCellChart(forest, &cStart, &cEnd);CHKERRQ(ierr);
      for (c = cStart; c < cEnd; ++c) {
        ierr = DMLabelSetValue(adaptLabel, c, DM_ADAPT_REFINE);CHKERRQ(ierr);
      }

      ierr = DMForestTemplate(forest, PETSC_COMM_WORLD, &postforest);CHKERRQ(ierr);
      ierr = DMForestSetAdaptivityLabel(postforest, adaptLabel);CHKERRQ(ierr);
      ierr = DMLabelDestroy(&adaptLabel);CHKERRQ(ierr);
      ierr = DMSetUp(postforest);CHKERRQ(ierr);
      ierr = DMDestroy(&forest);CHKERRQ(ierr);
      forest = postforest;
    }
  }
  ierr = DMViewFromOptions(forest, NULL, "-dm_view");CHKERRQ(ierr);

  /*  Setup the section*/
  if (userSection) {
    ierr = DMConvert(forest, DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMViewFromOptions(plex, NULL, "-dm_view");CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(plex, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Vertices [%D, %D)\n", vStart, vEnd);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) forest), &s);CHKERRQ(ierr);
    ierr = PetscSectionSetNumFields(s, 1);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(s, vStart, vEnd);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      ierr = PetscSectionSetDof(s, v, 1);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(s, v, 0, 1);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(s);CHKERRQ(ierr);
    ierr = DMSetLocalSection(forest, s);CHKERRQ(ierr);
    ierr = PetscObjectViewFromOptions((PetscObject) s, NULL, "-my_section_view");CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
    ierr = DMDestroy(&plex);CHKERRQ(ierr);
  } else {
    PetscFE  fe;
    PetscInt dim;

    ierr = DMGetDimension(forest, &dim);CHKERRQ(ierr);
    ierr = PetscFECreateLagrange(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, 1, PETSC_DETERMINE, &fe);CHKERRQ(ierr);
    ierr = DMAddField(forest, NULL, (PetscObject) fe);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
    ierr = DMCreateDS(forest);CHKERRQ(ierr);
  }

  /* Create the global vector*/
  ierr = DMCreateGlobalVector(forest, &g);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) g, "g");CHKERRQ(ierr);
  ierr = VecSet(g, 1.0);CHKERRQ(ierr);

  /* Test global to local*/
  Vec l;
  ierr = DMCreateLocalVector(forest, &l);CHKERRQ(ierr);
  ierr = VecZeroEntries(l);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(forest, g, INSERT_VALUES, l);CHKERRQ(ierr);
  ierr = VecZeroEntries(g);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(forest, l, INSERT_VALUES, g);CHKERRQ(ierr);
  ierr = VecDestroy(&l);CHKERRQ(ierr);

  /*  Save a vector*/
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, "forestHDF.h5", FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
  ierr = VecView(g, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /* Load another vector to load into*/
  ierr = DMCreateGlobalVector(forest, &g2);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) g2, "g");CHKERRQ(ierr);
  ierr = VecZeroEntries(g2);CHKERRQ(ierr);

  /*  Load a vector*/
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, "forestHDF.h5", FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  ierr = VecLoad(g2, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /*  Check if the data is the same*/
  ierr = VecAXPY(g2, -1.0, g);CHKERRQ(ierr);
  ierr = VecNorm(g2, NORM_INFINITY, &nrm);CHKERRQ(ierr);
  if (PetscAbsReal(nrm) > PETSC_SMALL) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Invalid difference norm %g", (double) nrm);

  ierr = VecDestroy(&g);CHKERRQ(ierr);
  ierr = VecDestroy(&g2);CHKERRQ(ierr);
  ierr = DMDestroy(&forest);CHKERRQ(ierr);
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
