static char help[] = "Tests DMLabel operations.\n\n";

#include <petscdm.h>
#include <petscdmplex.h>

PetscErrorCode ViewLabels(DM dm, PetscViewer viewer)
{
  DMLabel        label;
  const char    *labelName;
  PetscInt       numLabels, l;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* query the number and name of labels*/
  ierr = DMGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Number of labels: %d\n", numLabels);CHKERRQ(ierr);
  for (l = 0; l < numLabels; ++l) {
    IS labelIS, tmpIS;

    ierr = DMGetLabelName(dm, l, &labelName);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Label %d: name: %s\n", l, labelName);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "IS of values\n");CHKERRQ(ierr);
    ierr = DMGetLabel(dm, labelName, &label);CHKERRQ(ierr);
    ierr = DMLabelGetValueIS(label, &labelIS);CHKERRQ(ierr);
    ierr = ISOnComm(labelIS, PetscObjectComm((PetscObject)viewer), PETSC_USE_POINTER, &tmpIS);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = ISView(tmpIS, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    ierr = ISDestroy(&tmpIS);CHKERRQ(ierr);
    ierr = ISDestroy(&labelIS);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
  }
  /* Making sure that string literals work */
  ierr = PetscViewerASCIIPrintf(viewer,"\n\nCell Set label IS\n");CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "Cell Sets", &label);CHKERRQ(ierr);
  if (label) {
    IS labelIS, tmpIS;

    ierr = DMLabelGetValueIS(label, &labelIS);CHKERRQ(ierr);
    ierr = ISOnComm(labelIS, PetscObjectComm((PetscObject)viewer), PETSC_USE_POINTER, &tmpIS);CHKERRQ(ierr);
    ierr = ISView(tmpIS, viewer);CHKERRQ(ierr);
    ierr = ISDestroy(&tmpIS);CHKERRQ(ierr);
    ierr = ISDestroy(&labelIS);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CheckLabelsSame(DMLabel label0, DMLabel label1)
{
  const char     *name0, *name1;
  PetscBool       same;
  char           *msg;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetName((PetscObject)label0, &name0);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)label1, &name1);CHKERRQ(ierr);
  ierr = DMLabelCompare(PETSC_COMM_WORLD, label0, label1, &same, &msg);CHKERRQ(ierr);
  PetscAssertFalse(same != (PetscBool) !msg,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "DMLabelCompare returns inconsistent same=%d msg=\"%s\"", same, msg);
  PetscAssertFalse(!same,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Labels \"%s\" and \"%s\" should not differ! Message:\n%s", name0, name1, msg);
  /* Test passing NULL, must not fail */
  ierr = DMLabelCompare(PETSC_COMM_WORLD, label0, label1, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscFree(msg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CheckLabelsNotSame(DMLabel label0, DMLabel label1)
{
  const char     *name0, *name1;
  PetscBool       same;
  char           *msg;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetName((PetscObject)label0, &name0);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)label1, &name1);CHKERRQ(ierr);
  ierr = DMLabelCompare(PETSC_COMM_WORLD, label0, label1, &same, &msg);CHKERRQ(ierr);
  PetscAssertFalse(same != (PetscBool) !msg,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "DMLabelCompare returns inconsistent same=%d msg=\"%s\"", same, msg);
  PetscAssertFalse(same,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Labels \"%s\" and \"%s\" should differ!", name0, name1);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Compare label \"%s\" with \"%s\": %s\n", name0, name1, msg);CHKERRQ(ierr);
  ierr = PetscFree(msg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CheckDMLabelsSame(DM dm0, DM dm1)
{
  const char     *name0, *name1;
  PetscBool       same;
  char           *msg;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetName((PetscObject)dm0, &name0);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)dm1, &name1);CHKERRQ(ierr);
  ierr = DMCompareLabels(dm0, dm1, &same, &msg);CHKERRQ(ierr);
  PetscAssertFalse(same != (PetscBool) !msg,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "DMCompareLabels returns inconsistent same=%d msg=\"%s\"", same, msg);
  PetscAssertFalse(!same,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Labels of DMs \"%s\" and \"%s\" should not differ! Message:\n%s", name0, name1, msg);
  /* Test passing NULL, must not fail */
  ierr = DMCompareLabels(dm0, dm1, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscFree(msg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CheckDMLabelsNotSame(DM dm0, DM dm1)
{
  const char     *name0, *name1;
  PetscBool       same;
  char           *msg;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetName((PetscObject)dm0, &name0);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)dm1, &name1);CHKERRQ(ierr);
  ierr = DMCompareLabels(dm0, dm1, &same, &msg);CHKERRQ(ierr);
  PetscAssertFalse(same != (PetscBool) !msg,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "DMCompareLabels returns inconsistent same=%d msg=\"%s\"", same, msg);
  PetscAssertFalse(same,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Labels of DMs \"%s\" and \"%s\" should differ!", name0, name1);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Labels of DMs \"%s\" and \"%s\" differ: %s\n", name0, name1, msg);CHKERRQ(ierr);
  ierr = PetscFree(msg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(const char name[], DM *newdm)
{
  DM             dm, dmDist;
  char           filename[PETSC_MAX_PATH_LEN]="";
  PetscBool      interpolate = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* initialize and get options */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "DMLabel ex1 Options", "DMLabel");CHKERRQ(ierr);
  ierr = PetscOptionsString("-i", "filename to read", "ex1.c", filename, filename, sizeof(filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex1.c", interpolate, &interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* create and distribute DM */
  ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, "ex1_plex", interpolate, &dm);CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);CHKERRQ(ierr);
  if (dmDist) {
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm   = dmDist;
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)dm, name);CHKERRQ(ierr);
  *newdm = dm;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = CreateMesh("plex0", &dm);CHKERRQ(ierr);
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

    pEnd = PetscMin(pEnd, pStart + 5);
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

  /* do label perturbations and comparisons */
  {
    DMLabel   label0, label1, label2, label3;
    PetscInt  val;
    PetscInt  p, pStart, pEnd;

    ierr = DMGetLabel(dm, "label0", &label0);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "label1", &label1);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "label2", &label2);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "label3", &label3);CHKERRQ(ierr);

    ierr = CheckLabelsNotSame(label0, label1);CHKERRQ(ierr);
    ierr = CheckLabelsSame(label0, label2);CHKERRQ(ierr);
    ierr = CheckLabelsSame(label1, label3);CHKERRQ(ierr);

    ierr = DMLabelGetDefaultValue(label1, &val);CHKERRQ(ierr);
    ierr = DMLabelSetDefaultValue(label1, 333);CHKERRQ(ierr);
    ierr = CheckLabelsNotSame(label1, label3);CHKERRQ(ierr);
    ierr = DMLabelSetDefaultValue(label1, val);CHKERRQ(ierr);
    ierr = CheckLabelsSame(label1, label3);CHKERRQ(ierr);

    ierr = DMLabelGetBounds(label1, &pStart, &pEnd);CHKERRQ(ierr);

    for (p=pStart; p<pEnd; p++) {
      ierr = DMLabelGetValue(label1, p, &val);CHKERRQ(ierr);
      // This is weird. Perhaps we should not need to call DMLabelClearValue()
      ierr = DMLabelClearValue(label1, p, val);CHKERRQ(ierr);
      val++;
      ierr = DMLabelSetValue(label1, p, val);CHKERRQ(ierr);
    }
    ierr = CheckLabelsNotSame(label1, label3);CHKERRQ(ierr);
    for (p=pStart; p<pEnd; p++) {
      ierr = DMLabelGetValue(label1, p, &val);CHKERRQ(ierr);
      // This is weird. Perhaps we should not need to call DMLabelClearValue()
      ierr = DMLabelClearValue(label1, p, val);CHKERRQ(ierr);
      val--;
      ierr = DMLabelSetValue(label1, p, val);CHKERRQ(ierr);
    }
    ierr = CheckLabelsSame(label1, label3);CHKERRQ(ierr);

    ierr = DMLabelGetValue(label3, pEnd-1, &val);CHKERRQ(ierr);
    ierr = DMLabelSetValue(label3, pEnd, val);CHKERRQ(ierr);
    ierr = CheckLabelsNotSame(label1, label3);CHKERRQ(ierr);
    // This is weird. Perhaps we should not need to call DMLabelClearValue()
    ierr = DMLabelClearValue(label3, pEnd, val);CHKERRQ(ierr);
    ierr = CheckLabelsSame(label1, label3);CHKERRQ(ierr);
  }

  {
    DM        dm1;
    DMLabel   label02, label12;
    PetscInt  p = 0, val;

    ierr = CreateMesh("plex1", &dm1);CHKERRQ(ierr);
    ierr = CheckDMLabelsNotSame(dm, dm1);CHKERRQ(ierr);

    ierr = DMCopyLabels(dm, dm1, PETSC_OWN_POINTER, PETSC_FALSE, DM_COPY_LABELS_REPLACE);CHKERRQ(ierr);
    ierr = CheckDMLabelsSame(dm, dm1);CHKERRQ(ierr);

    ierr = DMCopyLabels(dm, dm1, PETSC_COPY_VALUES, PETSC_FALSE, DM_COPY_LABELS_REPLACE);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "label2", &label02);CHKERRQ(ierr);
    ierr = DMGetLabel(dm1, "label2", &label12);CHKERRQ(ierr);
    ierr = CheckLabelsSame(label02, label12);CHKERRQ(ierr);

    ierr = DMLabelGetValue(label12, p, &val);CHKERRQ(ierr);
    // This is weird. Perhaps we should not need to call DMLabelClearValue()
    ierr = DMLabelClearValue(label12, p, val);CHKERRQ(ierr);
    ierr = DMLabelSetValue(label12, p, val+1);CHKERRQ(ierr);
    ierr = CheckLabelsNotSame(label02, label12);CHKERRQ(ierr);
    ierr = CheckDMLabelsNotSame(dm, dm1);CHKERRQ(ierr);

    // This is weird. Perhaps we should not need to call DMLabelClearValue()
    ierr = DMLabelClearValue(label12, p, val+1);CHKERRQ(ierr);
    ierr = DMLabelSetValue(label12, p, val);CHKERRQ(ierr);
    ierr = CheckLabelsSame(label02, label12);CHKERRQ(ierr);
    ierr = CheckDMLabelsSame(dm, dm1);CHKERRQ(ierr);

    ierr = PetscObjectSetName((PetscObject)label12, "label12");CHKERRQ(ierr);
    ierr = CheckDMLabelsNotSame(dm, dm1);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)label12, "label2");CHKERRQ(ierr);
    ierr = CheckDMLabelsSame(dm, dm1);CHKERRQ(ierr);

    ierr = DMDestroy(&dm1);CHKERRQ(ierr);
  }

  /* remove label0 and label1 just to test manual removal; let label3 be removed automatically by DMDestroy() */
  {
    DMLabel label0, label1, label2;
    ierr = DMGetLabel(dm, "label0", &label0);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "label1", &label1);CHKERRQ(ierr);
    PetscAssertFalse(!label0,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label0 must not be NULL now");
    PetscAssertFalse(!label1,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label1 must not be NULL now");
    ierr = DMRemoveLabel(dm, "label1", NULL);CHKERRQ(ierr);
    ierr = DMRemoveLabel(dm, "label2", &label2);CHKERRQ(ierr);
    ierr = DMRemoveLabelBySelf(dm, &label0, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "label0", &label0);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "label1", &label1);CHKERRQ(ierr);
    PetscAssertFalse(label0,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label0 must be NULL now");
    PetscAssertFalse(label1,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label1 must be NULL now");
    PetscAssertFalse(!label2,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label2 must not be NULL now");
    ierr = DMRemoveLabelBySelf(dm, &label2, PETSC_FALSE);CHKERRQ(ierr); /* this should do nothing */
    PetscAssertFalse(!label2,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label2 must not be NULL now");
    ierr = DMLabelDestroy(&label2);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "label2", &label2);CHKERRQ(ierr);
    PetscAssertFalse(label2,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label2 must be NULL now");
  }

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    nsize: {{1 2}separate output}
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo -interpolate
    requires: exodusii

TEST*/
