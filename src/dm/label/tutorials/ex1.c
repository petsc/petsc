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
  PetscCall(DMGetNumLabels(dm, &numLabels));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Number of labels: %" PetscInt_FMT "\n", numLabels));
  for (l = 0; l < numLabels; ++l) {
    IS labelIS, tmpIS;

    PetscCall(DMGetLabelName(dm, l, &labelName));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Label %" PetscInt_FMT ": name: %s\n", l, labelName));
    PetscCall(PetscViewerASCIIPrintf(viewer, "IS of values\n"));
    PetscCall(DMGetLabel(dm, labelName, &label));
    PetscCall(DMLabelGetValueIS(label, &labelIS));
    PetscCall(ISOnComm(labelIS, PetscObjectComm((PetscObject)viewer), PETSC_USE_POINTER, &tmpIS));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(ISView(tmpIS, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(ISDestroy(&tmpIS));
    PetscCall(ISDestroy(&labelIS));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  }
  /* Making sure that string literals work */
  PetscCall(PetscViewerASCIIPrintf(viewer,"\n\nCell Set label IS\n"));
  PetscCall(DMGetLabel(dm, "Cell Sets", &label));
  if (label) {
    IS labelIS, tmpIS;

    PetscCall(DMLabelGetValueIS(label, &labelIS));
    PetscCall(ISOnComm(labelIS, PetscObjectComm((PetscObject)viewer), PETSC_USE_POINTER, &tmpIS));
    PetscCall(ISView(tmpIS, viewer));
    PetscCall(ISDestroy(&tmpIS));
    PetscCall(ISDestroy(&labelIS));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CheckLabelsSame(DMLabel label0, DMLabel label1)
{
  const char     *name0, *name1;
  PetscBool       same;
  char           *msg;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetName((PetscObject)label0, &name0));
  PetscCall(PetscObjectGetName((PetscObject)label1, &name1));
  PetscCall(DMLabelCompare(PETSC_COMM_WORLD, label0, label1, &same, &msg));
  PetscCheck(same == (PetscBool) !msg,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "DMLabelCompare returns inconsistent same=%d msg=\"%s\"", same, msg);
  PetscCheck(same,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Labels \"%s\" and \"%s\" should not differ! Message:\n%s", name0, name1, msg);
  /* Test passing NULL, must not fail */
  PetscCall(DMLabelCompare(PETSC_COMM_WORLD, label0, label1, NULL, NULL));
  PetscCall(PetscFree(msg));
  PetscFunctionReturn(0);
}

PetscErrorCode CheckLabelsNotSame(DMLabel label0, DMLabel label1)
{
  const char     *name0, *name1;
  PetscBool       same;
  char           *msg;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetName((PetscObject)label0, &name0));
  PetscCall(PetscObjectGetName((PetscObject)label1, &name1));
  PetscCall(DMLabelCompare(PETSC_COMM_WORLD, label0, label1, &same, &msg));
  PetscCheck(same == (PetscBool) !msg,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "DMLabelCompare returns inconsistent same=%d msg=\"%s\"", same, msg);
  PetscCheck(!same,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Labels \"%s\" and \"%s\" should differ!", name0, name1);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Compare label \"%s\" with \"%s\": %s\n", name0, name1, msg));
  PetscCall(PetscFree(msg));
  PetscFunctionReturn(0);
}

PetscErrorCode CheckDMLabelsSame(DM dm0, DM dm1)
{
  const char     *name0, *name1;
  PetscBool       same;
  char           *msg;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetName((PetscObject)dm0, &name0));
  PetscCall(PetscObjectGetName((PetscObject)dm1, &name1));
  PetscCall(DMCompareLabels(dm0, dm1, &same, &msg));
  PetscCheck(same == (PetscBool) !msg,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "DMCompareLabels returns inconsistent same=%d msg=\"%s\"", same, msg);
  PetscCheck(same,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Labels of DMs \"%s\" and \"%s\" should not differ! Message:\n%s", name0, name1, msg);
  /* Test passing NULL, must not fail */
  PetscCall(DMCompareLabels(dm0, dm1, NULL, NULL));
  PetscCall(PetscFree(msg));
  PetscFunctionReturn(0);
}

PetscErrorCode CheckDMLabelsNotSame(DM dm0, DM dm1)
{
  const char     *name0, *name1;
  PetscBool       same;
  char           *msg;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetName((PetscObject)dm0, &name0));
  PetscCall(PetscObjectGetName((PetscObject)dm1, &name1));
  PetscCall(DMCompareLabels(dm0, dm1, &same, &msg));
  PetscCheck(same == (PetscBool) !msg,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "DMCompareLabels returns inconsistent same=%d msg=\"%s\"", same, msg);
  PetscCheck(!same,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Labels of DMs \"%s\" and \"%s\" should differ!", name0, name1);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Labels of DMs \"%s\" and \"%s\" differ: %s\n", name0, name1, msg));
  PetscCall(PetscFree(msg));
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(const char name[], DM *newdm)
{
  DM             dm, dmDist;
  char           filename[PETSC_MAX_PATH_LEN]="";
  PetscBool      interpolate = PETSC_FALSE;

  PetscFunctionBegin;
  /* initialize and get options */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "DMLabel ex1 Options", "DMLabel");
  PetscCall(PetscOptionsString("-i", "filename to read", "ex1.c", filename, filename, sizeof(filename), NULL));
  PetscCall(PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex1.c", interpolate, &interpolate, NULL));
  PetscOptionsEnd();

  /* create and distribute DM */
  PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, "ex1_plex", interpolate, &dm));
  PetscCall(DMPlexDistribute(dm, 0, NULL, &dmDist));
  if (dmDist) {
    PetscCall(DMDestroy(&dm));
    dm   = dmDist;
  }
  PetscCall(DMSetFromOptions(dm));
  PetscCall(PetscObjectSetName((PetscObject)dm, name));
  *newdm = dm;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(CreateMesh("plex0", &dm));
  /* add custom labels to test adding/removal */
  {
    DMLabel label0, label1, label2, label3;
    PetscInt p, pStart, pEnd;
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    /* create label in DM and get from DM */
    PetscCall(DMCreateLabel(dm, "label0"));
    PetscCall(DMGetLabel(dm, "label0", &label0));
    /* alternative: create standalone label and add to DM; needs to be destroyed */
    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "label1", &label1));
    PetscCall(DMAddLabel(dm, label1));

    pEnd = PetscMin(pEnd, pStart + 5);
    for (p=pStart; p < pEnd; p++) {
      PetscCall(DMLabelSetValue(label0, p, 1));
      PetscCall(DMLabelSetValue(label1, p, 2));
    }
    /* duplicate label */
    PetscCall(DMLabelDuplicate(label0, &label2));
    PetscCall(DMLabelDuplicate(label1, &label3));
    PetscCall(PetscObjectSetName((PetscObject)label2, "label2"));
    PetscCall(PetscObjectSetName((PetscObject)label3, "label3"));
    PetscCall(DMAddLabel(dm, label2));
    PetscCall(DMAddLabel(dm, label3));
    /* remove the labels in this scope */
    PetscCall(DMLabelDestroy(&label1));
    PetscCall(DMLabelDestroy(&label2));
    PetscCall(DMLabelDestroy(&label3));
  }

  PetscCall(ViewLabels(dm, PETSC_VIEWER_STDOUT_WORLD));

  /* do label perturbations and comparisons */
  {
    DMLabel   label0, label1, label2, label3;
    PetscInt  val;
    PetscInt  p, pStart, pEnd;

    PetscCall(DMGetLabel(dm, "label0", &label0));
    PetscCall(DMGetLabel(dm, "label1", &label1));
    PetscCall(DMGetLabel(dm, "label2", &label2));
    PetscCall(DMGetLabel(dm, "label3", &label3));

    PetscCall(CheckLabelsNotSame(label0, label1));
    PetscCall(CheckLabelsSame(label0, label2));
    PetscCall(CheckLabelsSame(label1, label3));

    PetscCall(DMLabelGetDefaultValue(label1, &val));
    PetscCall(DMLabelSetDefaultValue(label1, 333));
    PetscCall(CheckLabelsNotSame(label1, label3));
    PetscCall(DMLabelSetDefaultValue(label1, val));
    PetscCall(CheckLabelsSame(label1, label3));

    PetscCall(DMLabelGetBounds(label1, &pStart, &pEnd));

    for (p=pStart; p<pEnd; p++) {
      PetscCall(DMLabelGetValue(label1, p, &val));
      // This is weird. Perhaps we should not need to call DMLabelClearValue()
      PetscCall(DMLabelClearValue(label1, p, val));
      val++;
      PetscCall(DMLabelSetValue(label1, p, val));
    }
    PetscCall(CheckLabelsNotSame(label1, label3));
    for (p=pStart; p<pEnd; p++) {
      PetscCall(DMLabelGetValue(label1, p, &val));
      // This is weird. Perhaps we should not need to call DMLabelClearValue()
      PetscCall(DMLabelClearValue(label1, p, val));
      val--;
      PetscCall(DMLabelSetValue(label1, p, val));
    }
    PetscCall(CheckLabelsSame(label1, label3));

    PetscCall(DMLabelGetValue(label3, pEnd-1, &val));
    PetscCall(DMLabelSetValue(label3, pEnd, val));
    PetscCall(CheckLabelsNotSame(label1, label3));
    // This is weird. Perhaps we should not need to call DMLabelClearValue()
    PetscCall(DMLabelClearValue(label3, pEnd, val));
    PetscCall(CheckLabelsSame(label1, label3));
  }

  {
    DM        dm1;
    DMLabel   label02, label12;
    PetscInt  p = 0, val;

    PetscCall(CreateMesh("plex1", &dm1));
    PetscCall(CheckDMLabelsNotSame(dm, dm1));

    PetscCall(DMCopyLabels(dm, dm1, PETSC_OWN_POINTER, PETSC_FALSE, DM_COPY_LABELS_REPLACE));
    PetscCall(CheckDMLabelsSame(dm, dm1));

    PetscCall(DMCopyLabels(dm, dm1, PETSC_COPY_VALUES, PETSC_FALSE, DM_COPY_LABELS_REPLACE));
    PetscCall(DMGetLabel(dm, "label2", &label02));
    PetscCall(DMGetLabel(dm1, "label2", &label12));
    PetscCall(CheckLabelsSame(label02, label12));

    PetscCall(DMLabelGetValue(label12, p, &val));
    // This is weird. Perhaps we should not need to call DMLabelClearValue()
    PetscCall(DMLabelClearValue(label12, p, val));
    PetscCall(DMLabelSetValue(label12, p, val+1));
    PetscCall(CheckLabelsNotSame(label02, label12));
    PetscCall(CheckDMLabelsNotSame(dm, dm1));

    // This is weird. Perhaps we should not need to call DMLabelClearValue()
    PetscCall(DMLabelClearValue(label12, p, val+1));
    PetscCall(DMLabelSetValue(label12, p, val));
    PetscCall(CheckLabelsSame(label02, label12));
    PetscCall(CheckDMLabelsSame(dm, dm1));

    PetscCall(PetscObjectSetName((PetscObject)label12, "label12"));
    PetscCall(CheckDMLabelsNotSame(dm, dm1));
    PetscCall(PetscObjectSetName((PetscObject)label12, "label2"));
    PetscCall(CheckDMLabelsSame(dm, dm1));

    PetscCall(DMDestroy(&dm1));
  }

  /* remove label0 and label1 just to test manual removal; let label3 be removed automatically by DMDestroy() */
  {
    DMLabel label0, label1, label2;
    PetscCall(DMGetLabel(dm, "label0", &label0));
    PetscCall(DMGetLabel(dm, "label1", &label1));
    PetscCheck(label0,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label0 must not be NULL now");
    PetscCheck(label1,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label1 must not be NULL now");
    PetscCall(DMRemoveLabel(dm, "label1", NULL));
    PetscCall(DMRemoveLabel(dm, "label2", &label2));
    PetscCall(DMRemoveLabelBySelf(dm, &label0, PETSC_TRUE));
    PetscCall(DMGetLabel(dm, "label0", &label0));
    PetscCall(DMGetLabel(dm, "label1", &label1));
    PetscCheck(!label0,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label0 must be NULL now");
    PetscCheck(!label1,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label1 must be NULL now");
    PetscCheck(label2,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label2 must not be NULL now");
    PetscCall(DMRemoveLabelBySelf(dm, &label2, PETSC_FALSE)); /* this should do nothing */
    PetscCheck(label2,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label2 must not be NULL now");
    PetscCall(DMLabelDestroy(&label2));
    PetscCall(DMGetLabel(dm, "label2", &label2));
    PetscCheck(!label2,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "label2 must be NULL now");
  }

  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    nsize: {{1 2}separate output}
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo -interpolate
    requires: exodusii

TEST*/
