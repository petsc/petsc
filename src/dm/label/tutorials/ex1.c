static char help[] = "Tests DMLabel operations.\n\n";

#include <petscdm.h>
#include <petscdmplex.h>

PetscErrorCode ViewLabels(DM dm, PetscViewer viewer)
{
  DMLabel        label;
  const char    *labelName;
  PetscInt       numLabels, l;

  PetscFunctionBegin;
  /* query the number and name of labels*/
  CHKERRQ(DMGetNumLabels(dm, &numLabels));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "Number of labels: %d\n", numLabels));
  for (l = 0; l < numLabels; ++l) {
    IS labelIS, tmpIS;

    CHKERRQ(DMGetLabelName(dm, l, &labelName));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Label %d: name: %s\n", l, labelName));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "IS of values\n"));
    CHKERRQ(DMGetLabel(dm, labelName, &label));
    CHKERRQ(DMLabelGetValueIS(label, &labelIS));
    CHKERRQ(ISOnComm(labelIS, PetscObjectComm((PetscObject)viewer), PETSC_USE_POINTER, &tmpIS));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(ISView(tmpIS, viewer));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
    CHKERRQ(ISDestroy(&tmpIS));
    CHKERRQ(ISDestroy(&labelIS));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "\n"));
  }
  /* Making sure that string literals work */
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"\n\nCell Set label IS\n"));
  CHKERRQ(DMGetLabel(dm, "Cell Sets", &label));
  if (label) {
    IS labelIS, tmpIS;

    CHKERRQ(DMLabelGetValueIS(label, &labelIS));
    CHKERRQ(ISOnComm(labelIS, PetscObjectComm((PetscObject)viewer), PETSC_USE_POINTER, &tmpIS));
    CHKERRQ(ISView(tmpIS, viewer));
    CHKERRQ(ISDestroy(&tmpIS));
    CHKERRQ(ISDestroy(&labelIS));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CheckLabelsSame(DMLabel label0, DMLabel label1)
{
  const char     *name0, *name1;
  PetscBool       same;
  char           *msg;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetName((PetscObject)label0, &name0));
  CHKERRQ(PetscObjectGetName((PetscObject)label1, &name1));
  CHKERRQ(DMLabelCompare(PETSC_COMM_WORLD, label0, label1, &same, &msg));
  PetscCheckFalse(same != (PetscBool) !msg,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "DMLabelCompare returns inconsistent same=%d msg=\"%s\"", same, msg);
  PetscCheck(same,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Labels \"%s\" and \"%s\" should not differ! Message:\n%s", name0, name1, msg);
  /* Test passing NULL, must not fail */
  CHKERRQ(DMLabelCompare(PETSC_COMM_WORLD, label0, label1, NULL, NULL));
  CHKERRQ(PetscFree(msg));
  PetscFunctionReturn(0);
}

PetscErrorCode CheckLabelsNotSame(DMLabel label0, DMLabel label1)
{
  const char     *name0, *name1;
  PetscBool       same;
  char           *msg;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetName((PetscObject)label0, &name0));
  CHKERRQ(PetscObjectGetName((PetscObject)label1, &name1));
  CHKERRQ(DMLabelCompare(PETSC_COMM_WORLD, label0, label1, &same, &msg));
  PetscCheckFalse(same != (PetscBool) !msg,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "DMLabelCompare returns inconsistent same=%d msg=\"%s\"", same, msg);
  PetscCheck(!same,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Labels \"%s\" and \"%s\" should differ!", name0, name1);
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Compare label \"%s\" with \"%s\": %s\n", name0, name1, msg));
  CHKERRQ(PetscFree(msg));
  PetscFunctionReturn(0);
}

PetscErrorCode CheckDMLabelsSame(DM dm0, DM dm1)
{
  const char     *name0, *name1;
  PetscBool       same;
  char           *msg;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetName((PetscObject)dm0, &name0));
  CHKERRQ(PetscObjectGetName((PetscObject)dm1, &name1));
  CHKERRQ(DMCompareLabels(dm0, dm1, &same, &msg));
  PetscCheckFalse(same != (PetscBool) !msg,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "DMCompareLabels returns inconsistent same=%d msg=\"%s\"", same, msg);
  PetscCheck(same,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Labels of DMs \"%s\" and \"%s\" should not differ! Message:\n%s", name0, name1, msg);
  /* Test passing NULL, must not fail */
  CHKERRQ(DMCompareLabels(dm0, dm1, NULL, NULL));
  CHKERRQ(PetscFree(msg));
  PetscFunctionReturn(0);
}

PetscErrorCode CheckDMLabelsNotSame(DM dm0, DM dm1)
{
  const char     *name0, *name1;
  PetscBool       same;
  char           *msg;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetName((PetscObject)dm0, &name0));
  CHKERRQ(PetscObjectGetName((PetscObject)dm1, &name1));
  CHKERRQ(DMCompareLabels(dm0, dm1, &same, &msg));
  PetscCheckFalse(same != (PetscBool) !msg,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "DMCompareLabels returns inconsistent same=%d msg=\"%s\"", same, msg);
  PetscCheck(!same,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Labels of DMs \"%s\" and \"%s\" should differ!", name0, name1);
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Labels of DMs \"%s\" and \"%s\" differ: %s\n", name0, name1, msg));
  CHKERRQ(PetscFree(msg));
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
  CHKERRQ(PetscOptionsString("-i", "filename to read", "ex1.c", filename, filename, sizeof(filename), NULL));
  CHKERRQ(PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex1.c", interpolate, &interpolate, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* create and distribute DM */
  CHKERRQ(DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, "ex1_plex", interpolate, &dm));
  CHKERRQ(DMPlexDistribute(dm, 0, NULL, &dmDist));
  if (dmDist) {
    CHKERRQ(DMDestroy(&dm));
    dm   = dmDist;
  }
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(PetscObjectSetName((PetscObject)dm, name));
  *newdm = dm;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  CHKERRQ(CreateMesh("plex0", &dm));
  /* add custom labels to test adding/removal */
  {
    DMLabel label0, label1, label2, label3;
    PetscInt p, pStart, pEnd;
    CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
    /* create label in DM and get from DM */
    CHKERRQ(DMCreateLabel(dm, "label0"));
    CHKERRQ(DMGetLabel(dm, "label0", &label0));
    /* alternative: create standalone label and add to DM; needs to be destroyed */
    CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "label1", &label1));
    CHKERRQ(DMAddLabel(dm, label1));

    pEnd = PetscMin(pEnd, pStart + 5);
    for (p=pStart; p < pEnd; p++) {
      CHKERRQ(DMLabelSetValue(label0, p, 1));
      CHKERRQ(DMLabelSetValue(label1, p, 2));
    }
    /* duplicate label */
    CHKERRQ(DMLabelDuplicate(label0, &label2));
    CHKERRQ(DMLabelDuplicate(label1, &label3));
    CHKERRQ(PetscObjectSetName((PetscObject)label2, "label2"));
    CHKERRQ(PetscObjectSetName((PetscObject)label3, "label3"));
    CHKERRQ(DMAddLabel(dm, label2));
    CHKERRQ(DMAddLabel(dm, label3));
    /* remove the labels in this scope */
    CHKERRQ(DMLabelDestroy(&label1));
    CHKERRQ(DMLabelDestroy(&label2));
    CHKERRQ(DMLabelDestroy(&label3));
  }

  CHKERRQ(ViewLabels(dm, PETSC_VIEWER_STDOUT_WORLD));

  /* do label perturbations and comparisons */
  {
    DMLabel   label0, label1, label2, label3;
    PetscInt  val;
    PetscInt  p, pStart, pEnd;

    CHKERRQ(DMGetLabel(dm, "label0", &label0));
    CHKERRQ(DMGetLabel(dm, "label1", &label1));
    CHKERRQ(DMGetLabel(dm, "label2", &label2));
    CHKERRQ(DMGetLabel(dm, "label3", &label3));

    CHKERRQ(CheckLabelsNotSame(label0, label1));
    CHKERRQ(CheckLabelsSame(label0, label2));
    CHKERRQ(CheckLabelsSame(label1, label3));

    CHKERRQ(DMLabelGetDefaultValue(label1, &val));
    CHKERRQ(DMLabelSetDefaultValue(label1, 333));
    CHKERRQ(CheckLabelsNotSame(label1, label3));
    CHKERRQ(DMLabelSetDefaultValue(label1, val));
    CHKERRQ(CheckLabelsSame(label1, label3));

    CHKERRQ(DMLabelGetBounds(label1, &pStart, &pEnd));

    for (p=pStart; p<pEnd; p++) {
      CHKERRQ(DMLabelGetValue(label1, p, &val));
      // This is weird. Perhaps we should not need to call DMLabelClearValue()
      CHKERRQ(DMLabelClearValue(label1, p, val));
      val++;
      CHKERRQ(DMLabelSetValue(label1, p, val));
    }
    CHKERRQ(CheckLabelsNotSame(label1, label3));
    for (p=pStart; p<pEnd; p++) {
      CHKERRQ(DMLabelGetValue(label1, p, &val));
      // This is weird. Perhaps we should not need to call DMLabelClearValue()
      CHKERRQ(DMLabelClearValue(label1, p, val));
      val--;
      CHKERRQ(DMLabelSetValue(label1, p, val));
    }
    CHKERRQ(CheckLabelsSame(label1, label3));

    CHKERRQ(DMLabelGetValue(label3, pEnd-1, &val));
    CHKERRQ(DMLabelSetValue(label3, pEnd, val));
    CHKERRQ(CheckLabelsNotSame(label1, label3));
    // This is weird. Perhaps we should not need to call DMLabelClearValue()
    CHKERRQ(DMLabelClearValue(label3, pEnd, val));
    CHKERRQ(CheckLabelsSame(label1, label3));
  }

  {
    DM        dm1;
    DMLabel   label02, label12;
    PetscInt  p = 0, val;

    CHKERRQ(CreateMesh("plex1", &dm1));
    CHKERRQ(CheckDMLabelsNotSame(dm, dm1));

    CHKERRQ(DMCopyLabels(dm, dm1, PETSC_OWN_POINTER, PETSC_FALSE, DM_COPY_LABELS_REPLACE));
    CHKERRQ(CheckDMLabelsSame(dm, dm1));

    CHKERRQ(DMCopyLabels(dm, dm1, PETSC_COPY_VALUES, PETSC_FALSE, DM_COPY_LABELS_REPLACE));
    CHKERRQ(DMGetLabel(dm, "label2", &label02));
    CHKERRQ(DMGetLabel(dm1, "label2", &label12));
    CHKERRQ(CheckLabelsSame(label02, label12));

    CHKERRQ(DMLabelGetValue(label12, p, &val));
    // This is weird. Perhaps we should not need to call DMLabelClearValue()
    CHKERRQ(DMLabelClearValue(label12, p, val));
    CHKERRQ(DMLabelSetValue(label12, p, val+1));
    CHKERRQ(CheckLabelsNotSame(label02, label12));
    CHKERRQ(CheckDMLabelsNotSame(dm, dm1));

    // This is weird. Perhaps we should not need to call DMLabelClearValue()
    CHKERRQ(DMLabelClearValue(label12, p, val+1));
    CHKERRQ(DMLabelSetValue(label12, p, val));
    CHKERRQ(CheckLabelsSame(label02, label12));
    CHKERRQ(CheckDMLabelsSame(dm, dm1));

    CHKERRQ(PetscObjectSetName((PetscObject)label12, "label12"));
    CHKERRQ(CheckDMLabelsNotSame(dm, dm1));
    CHKERRQ(PetscObjectSetName((PetscObject)label12, "label2"));
    CHKERRQ(CheckDMLabelsSame(dm, dm1));

    CHKERRQ(DMDestroy(&dm1));
  }

  /* remove label0 and label1 just to test manual removal; let label3 be removed automatically by DMDestroy() */
  {
    DMLabel label0, label1, label2;
    CHKERRQ(DMGetLabel(dm, "label0", &label0));
    CHKERRQ(DMGetLabel(dm, "label1", &label1));
    PetscCheck(label0,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label0 must not be NULL now");
    PetscCheck(label1,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label1 must not be NULL now");
    CHKERRQ(DMRemoveLabel(dm, "label1", NULL));
    CHKERRQ(DMRemoveLabel(dm, "label2", &label2));
    CHKERRQ(DMRemoveLabelBySelf(dm, &label0, PETSC_TRUE));
    CHKERRQ(DMGetLabel(dm, "label0", &label0));
    CHKERRQ(DMGetLabel(dm, "label1", &label1));
    PetscCheck(!label0,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label0 must be NULL now");
    PetscCheck(!label1,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label1 must be NULL now");
    PetscCheck(label2,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label2 must not be NULL now");
    CHKERRQ(DMRemoveLabelBySelf(dm, &label2, PETSC_FALSE)); /* this should do nothing */
    PetscCheck(label2,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label2 must not be NULL now");
    CHKERRQ(DMLabelDestroy(&label2));
    CHKERRQ(DMGetLabel(dm, "label2", &label2));
    PetscCheck(!label2,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label2 must be NULL now");
  }

  CHKERRQ(DMDestroy(&dm));
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
