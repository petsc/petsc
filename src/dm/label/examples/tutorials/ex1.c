static char help[] = "Tests DMLabel operations.\n\n";

#include <petscdm.h>
#include <petscdmplex.h>

PetscErrorCode ViewLabels(DM dm, PetscViewer viewer)
{
  DMLabel        label;
  IS             labelIS;
  const char    *labelName;
  PetscInt       numLabels, l;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* query the number and name of labels*/
  ierr = DMGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Number of labels: %d\n", numLabels);CHKERRQ(ierr);
  for (l = 0; l < numLabels; ++l) {
    ierr = DMGetLabelName(dm, l, &labelName);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Label %d: name: %s\n", l, labelName);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "IS of values\n");CHKERRQ(ierr);
    ierr = DMGetLabel(dm, labelName, &label);CHKERRQ(ierr);
    ierr = DMLabelGetValueIS(label, &labelIS);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = ISView(labelIS, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    ierr = ISDestroy(&labelIS);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
  }
  /* Making sure that string literals work */
  ierr = PetscViewerASCIIPrintf(viewer,"\n\nCell Set label IS\n");CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "Cell Sets", &label);CHKERRQ(ierr);
  if (label) {
    ierr = DMLabelGetValueIS(label, &labelIS);CHKERRQ(ierr);
    ierr = ISView(labelIS, viewer);CHKERRQ(ierr);
    ierr = ISDestroy(&labelIS);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, dmDist;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscBool      interpolate = PETSC_FALSE;
  PetscErrorCode ierr;

  /* initialize and get options */
  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "DMLabel ex1 Options", "DMLabel");CHKERRQ(ierr);
  ierr = PetscOptionsString("-i", "filename to read", "ex1.c", filename, filename, sizeof(filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex1.c", interpolate, &interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* create and distribute DM */
  ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, interpolate, &dm);CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);CHKERRQ(ierr);
  if (dmDist) {
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm   = dmDist;
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = ViewLabels(dm, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo -interpolate
    requires: exodusii

TEST*/
