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
  char           filename[PETSC_MAX_PATH_LEN]="";
  PetscBool      interpolate = PETSC_FALSE;
  PetscErrorCode ierr;

  /* initialize and get options */
  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
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

  /* add custom labels to test adding/removal */
  {
    DMLabel label0, label1, label2, label3;
    PetscInt p, pStart, pEnd;
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    /* create label in DM and get from DM */
    ierr = DMCreateLabel(dm, "label0");CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "label0", &label0);CHKERRQ(ierr);
    /* alternative: create standalone label and add to DM; needs to be destroyed */
    ierr = DMLabelCreate(PETSC_COMM_SELF, "label1", &label1);CHKERRQ(ierr);
    ierr = DMAddLabel(dm, label1);CHKERRQ(ierr);

    pEnd = pStart + (pEnd-pStart)/3; /* we will mark the first third of points */
    for (p=pStart; p < pEnd; p++) {
      ierr = DMLabelSetValue(label0, p, 1);CHKERRQ(ierr);
      ierr = DMLabelSetValue(label1, p, 2);CHKERRQ(ierr);
    }
    /* duplicate label */
    ierr = DMLabelDuplicate(label0, &label2);CHKERRQ(ierr);
    ierr = DMLabelDuplicate(label1, &label3);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)label2, "label2");CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)label3, "label3");CHKERRQ(ierr);
    ierr = DMAddLabel(dm, label2);CHKERRQ(ierr);
    ierr = DMAddLabel(dm, label3);CHKERRQ(ierr);
    /* remove the labels in this scope */
    ierr = DMLabelDestroy(&label1);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&label2);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&label3);CHKERRQ(ierr);
  }

  ierr = ViewLabels(dm, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* remove label0 and label1 just to test manual removal; let label3 be removed automatically by DMDestroy() */
  {
    DMLabel label0, label1, label2;
    ierr = DMGetLabel(dm, "label0", &label0);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "label1", &label1);CHKERRQ(ierr);
    if (!label0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label0 must not be NULL now");
    if (!label1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label1 must not be NULL now");
    ierr = DMRemoveLabel(dm, "label1", NULL);CHKERRQ(ierr);
    ierr = DMRemoveLabel(dm, "label2", &label2);CHKERRQ(ierr);
    ierr = DMRemoveLabelBySelf(dm, &label0, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "label0", &label0);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "label1", &label1);CHKERRQ(ierr);
    if (label0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label0 must be NULL now");
    if (label1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label1 must be NULL now");
    if (!label2) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label2 must not be NULL now");
    ierr = DMRemoveLabelBySelf(dm, &label2, PETSC_FALSE);CHKERRQ(ierr); /* this should do nothing */
    if (!label2) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label2 must not be NULL now");
    ierr = DMLabelDestroy(&label2);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "label2", &label2);CHKERRQ(ierr);
    if (label2) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label2 must be NULL now");
  }

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
