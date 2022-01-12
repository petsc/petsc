static char help[] = "TDycore Mesh Examples\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscBool adapt; /* Flag for adaptation of the surface mesh */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->adapt = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-adapt", "Flag for adaptation of the surface mesh", "ex10.c", options->adapt, &options->adapt, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateDomainLabel(DM dm)
{
  DMLabel        label;
  PetscInt       cStart, cEnd, c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateLabel(dm, "Cell Sets");CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "Cell Sets", &label);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscReal centroid[3], volume, x, y;

    ierr = DMPlexComputeCellGeometryFVM(dm, c, &volume, centroid, NULL);CHKERRQ(ierr);
    x = centroid[0]; y = centroid[1];
    /* Headwaters are (0.0,0.25)--(0.1,0.75) */
    if ((x >= 0.0 && x <  0.1) && (y >= 0.25 && y <= 0.75)) {ierr = DMLabelSetValue(label, c, 1);CHKERRQ(ierr);continue;}
    /* River channel is (0.1,0.45)--(1.0,0.55) */
    if ((x >= 0.1 && x <= 1.0) && (y >= 0.45 && y <= 0.55)) {ierr = DMLabelSetValue(label, c, 2);CHKERRQ(ierr);continue;}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode AdaptMesh(DM *dm, AppCtx *ctx)
{
  DM              dmCur = *dm;
  DMLabel         label;
  IS              valueIS, vIS;
  PetscBool       hasLabel;
  const PetscInt *values;
  PetscReal      *volConst; /* Volume constraints for each label value */
  PetscReal       ratio;
  PetscInt        dim, Nv, v, cStart, cEnd, c;
  PetscBool       adapt = PETSC_TRUE;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  if (!ctx->adapt) PetscFunctionReturn(0);
  ierr = DMHasLabel(*dm, "Cell Sets", &hasLabel);CHKERRQ(ierr);
  if (!hasLabel) {ierr = CreateDomainLabel(*dm);CHKERRQ(ierr);}
  ierr = DMGetDimension(*dm, &dim);CHKERRQ(ierr);
  ratio = PetscPowRealInt(0.5, dim);
  /* Get volume constraints */
  ierr = DMGetLabel(*dm, "Cell Sets", &label);CHKERRQ(ierr);
  ierr = DMLabelGetValueIS(label, &vIS);CHKERRQ(ierr);
  ierr = ISDuplicate(vIS, &valueIS);CHKERRQ(ierr);
  ierr = ISDestroy(&vIS);CHKERRQ(ierr);
  /* Sorting ruins the label */
  ierr = ISSort(valueIS);CHKERRQ(ierr);
  ierr = ISGetLocalSize(valueIS, &Nv);CHKERRQ(ierr);
  ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nv, &volConst);CHKERRQ(ierr);
  for (v = 0; v < Nv; ++v) {
    char opt[128];

    volConst[v] = PETSC_MAX_REAL;
    ierr = PetscSNPrintf(opt, 128, "-volume_constraint_%d", (int) values[v]);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL, NULL, opt, &volConst[v], NULL);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
  ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  /* Adapt mesh iteratively */
  while (adapt) {
    DM       dmAdapt;
    DMLabel  adaptLabel;
    PetscInt nAdaptLoc[2], nAdapt[2];

    adapt = PETSC_FALSE;
    nAdaptLoc[0] = nAdaptLoc[1] = 0;
    nAdapt[0]    = nAdapt[1]    = 0;
    /* Adaptation is not preserving the domain label */
    ierr = DMHasLabel(dmCur, "Cell Sets", &hasLabel);CHKERRQ(ierr);
    if (!hasLabel) {ierr = CreateDomainLabel(dmCur);CHKERRQ(ierr);}
    ierr = DMGetLabel(dmCur, "Cell Sets", &label);CHKERRQ(ierr);
    ierr = DMLabelGetValueIS(label, &vIS);CHKERRQ(ierr);
    ierr = ISDuplicate(vIS, &valueIS);CHKERRQ(ierr);
    ierr = ISDestroy(&vIS);CHKERRQ(ierr);
    /* Sorting directly the label's value IS would corrupt the label so we duplicate the IS first */
    ierr = ISSort(valueIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(valueIS, &Nv);CHKERRQ(ierr);
    ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
    /* Construct adaptation label */
    ierr = DMLabelCreate(PETSC_COMM_SELF, "adapt", &adaptLabel);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dmCur, 0, &cStart, &cEnd);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      PetscReal volume, centroid[3];
      PetscInt  value, vidx;

      ierr = DMPlexComputeCellGeometryFVM(dmCur, c, &volume, centroid, NULL);CHKERRQ(ierr);
      ierr = DMLabelGetValue(label, c, &value);CHKERRQ(ierr);
      if (value < 0) continue;
      ierr = PetscFindInt(value, Nv, values, &vidx);CHKERRQ(ierr);
      if (vidx < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Value %D for cell %D does not exist in label", value, c);
      if (volume > volConst[vidx])        {ierr = DMLabelSetValue(adaptLabel, c, DM_ADAPT_REFINE);CHKERRQ(ierr);  ++nAdaptLoc[0];}
      if (volume < volConst[vidx]*ratio) {ierr = DMLabelSetValue(adaptLabel, c, DM_ADAPT_COARSEN);CHKERRQ(ierr); ++nAdaptLoc[1];}
    }
    ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
    ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&nAdaptLoc, &nAdapt, 2, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject) dmCur));CHKERRMPI(ierr);
    if (nAdapt[0]) {
      ierr = PetscInfo(dmCur, "Adapted mesh, marking %D cells for refinement, and %D cells for coarsening\n", nAdapt[0], nAdapt[1]);CHKERRQ(ierr);
      ierr = DMAdaptLabel(dmCur, adaptLabel, &dmAdapt);CHKERRQ(ierr);
      ierr = DMDestroy(&dmCur);CHKERRQ(ierr);
      ierr = DMViewFromOptions(dmAdapt, NULL, "-adapt_dm_view");CHKERRQ(ierr);
      dmCur = dmAdapt;
      adapt = PETSC_TRUE;
    }
    ierr = DMLabelDestroy(&adaptLabel);CHKERRQ(ierr);
  }
  ierr = PetscFree(volConst);CHKERRQ(ierr);
  *dm = dmCur;
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Create top surface */
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) *dm, "init_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) *dm, NULL);CHKERRQ(ierr);
  /* Adapt surface */
  ierr = AdaptMesh(dm, user);CHKERRQ(ierr);
  /* Extrude surface to get volume mesh */
  ierr = DMGetDimension(*dm, &dim);CHKERRQ(ierr);
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    requires: triangle
    args: -init_dm_plex_dim 2 -init_dm_plex_box_faces 1,1 -dm_extrude 1 -dm_view

  test: # Regularly refine the surface before extrusion
    suffix: 1
    requires: triangle
    args: -init_dm_plex_dim 2 -init_dm_refine 2 -dm_extrude 1 -dm_view

  test: # Parallel run
    suffix: 2
    requires: triangle
    nsize: 5
    args: -init_dm_plex_dim 2 -init_dm_refine 3 -petscpartitioner_type simple -dm_distribute -dm_extrude 3 -dm_view

  test: # adaptively refine the surface before extrusion
    suffix: 3
    requires: triangle
    args: -init_dm_plex_dim 2 -init_dm_plex_box_faces 5,5 -adapt -volume_constraint_1 0.01 -volume_constraint_2 0.000625 -dm_extrude 10

TEST*/
