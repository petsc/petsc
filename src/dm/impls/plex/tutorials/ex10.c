static char help[] = "TDycore Mesh Examples\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscBool adapt; /* Flag for adaptation of the surface mesh */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->adapt = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-adapt", "Flag for adaptation of the surface mesh", "ex10.c", options->adapt, &options->adapt, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateDomainLabel(DM dm)
{
  DMLabel        label;
  PetscInt       cStart, cEnd, c;

  PetscFunctionBeginUser;
  PetscCall(DMCreateLabel(dm, "Cell Sets"));
  PetscCall(DMGetLabel(dm, "Cell Sets", &label));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscReal centroid[3], volume, x, y;

    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, &volume, centroid, NULL));
    x = centroid[0]; y = centroid[1];
    /* Headwaters are (0.0,0.25)--(0.1,0.75) */
    if ((x >= 0.0 && x <  0.1) && (y >= 0.25 && y <= 0.75)) {PetscCall(DMLabelSetValue(label, c, 1));continue;}
    /* River channel is (0.1,0.45)--(1.0,0.55) */
    if ((x >= 0.1 && x <= 1.0) && (y >= 0.45 && y <= 0.55)) {PetscCall(DMLabelSetValue(label, c, 2));continue;}
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

  PetscFunctionBeginUser;
  if (!ctx->adapt) PetscFunctionReturn(0);
  PetscCall(DMHasLabel(*dm, "Cell Sets", &hasLabel));
  if (!hasLabel) PetscCall(CreateDomainLabel(*dm));
  PetscCall(DMGetDimension(*dm, &dim));
  ratio = PetscPowRealInt(0.5, dim);
  /* Get volume constraints */
  PetscCall(DMGetLabel(*dm, "Cell Sets", &label));
  PetscCall(DMLabelGetValueIS(label, &vIS));
  PetscCall(ISDuplicate(vIS, &valueIS));
  PetscCall(ISDestroy(&vIS));
  /* Sorting ruins the label */
  PetscCall(ISSort(valueIS));
  PetscCall(ISGetLocalSize(valueIS, &Nv));
  PetscCall(ISGetIndices(valueIS, &values));
  PetscCall(PetscMalloc1(Nv, &volConst));
  for (v = 0; v < Nv; ++v) {
    char opt[128];

    volConst[v] = PETSC_MAX_REAL;
    PetscCall(PetscSNPrintf(opt, 128, "-volume_constraint_%d", (int) values[v]));
    PetscCall(PetscOptionsGetReal(NULL, NULL, opt, &volConst[v], NULL));
  }
  PetscCall(ISRestoreIndices(valueIS, &values));
  PetscCall(ISDestroy(&valueIS));
  /* Adapt mesh iteratively */
  while (adapt) {
    DM       dmAdapt;
    DMLabel  adaptLabel;
    PetscInt nAdaptLoc[2], nAdapt[2];

    adapt = PETSC_FALSE;
    nAdaptLoc[0] = nAdaptLoc[1] = 0;
    nAdapt[0]    = nAdapt[1]    = 0;
    /* Adaptation is not preserving the domain label */
    PetscCall(DMHasLabel(dmCur, "Cell Sets", &hasLabel));
    if (!hasLabel) PetscCall(CreateDomainLabel(dmCur));
    PetscCall(DMGetLabel(dmCur, "Cell Sets", &label));
    PetscCall(DMLabelGetValueIS(label, &vIS));
    PetscCall(ISDuplicate(vIS, &valueIS));
    PetscCall(ISDestroy(&vIS));
    /* Sorting directly the label's value IS would corrupt the label so we duplicate the IS first */
    PetscCall(ISSort(valueIS));
    PetscCall(ISGetLocalSize(valueIS, &Nv));
    PetscCall(ISGetIndices(valueIS, &values));
    /* Construct adaptation label */
    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "adapt", &adaptLabel));
    PetscCall(DMPlexGetHeightStratum(dmCur, 0, &cStart, &cEnd));
    for (c = cStart; c < cEnd; ++c) {
      PetscReal volume, centroid[3];
      PetscInt  value, vidx;

      PetscCall(DMPlexComputeCellGeometryFVM(dmCur, c, &volume, centroid, NULL));
      PetscCall(DMLabelGetValue(label, c, &value));
      if (value < 0) continue;
      PetscCall(PetscFindInt(value, Nv, values, &vidx));
      PetscCheck(vidx >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Value %" PetscInt_FMT " for cell %" PetscInt_FMT " does not exist in label", value, c);
      if (volume > volConst[vidx])        {PetscCall(DMLabelSetValue(adaptLabel, c, DM_ADAPT_REFINE));  ++nAdaptLoc[0];}
      if (volume < volConst[vidx]*ratio) {PetscCall(DMLabelSetValue(adaptLabel, c, DM_ADAPT_COARSEN)); ++nAdaptLoc[1];}
    }
    PetscCall(ISRestoreIndices(valueIS, &values));
    PetscCall(ISDestroy(&valueIS));
    PetscCallMPI(MPI_Allreduce(&nAdaptLoc, &nAdapt, 2, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject) dmCur)));
    if (nAdapt[0]) {
      PetscCall(PetscInfo(dmCur, "Adapted mesh, marking %" PetscInt_FMT " cells for refinement, and %" PetscInt_FMT " cells for coarsening\n", nAdapt[0], nAdapt[1]));
      PetscCall(DMAdaptLabel(dmCur, adaptLabel, &dmAdapt));
      PetscCall(DMDestroy(&dmCur));
      PetscCall(DMViewFromOptions(dmAdapt, NULL, "-adapt_dm_view"));
      dmCur = dmAdapt;
      adapt = PETSC_TRUE;
    }
    PetscCall(DMLabelDestroy(&adaptLabel));
  }
  PetscCall(PetscFree(volConst));
  *dm = dmCur;
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim;

  PetscFunctionBeginUser;
  /* Create top surface */
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) *dm, "init_"));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) *dm, NULL));
  /* Adapt surface */
  PetscCall(AdaptMesh(dm, user));
  /* Extrude surface to get volume mesh */
  PetscCall(DMGetDimension(*dm, &dim));
  PetscCall(DMLocalizeCoordinates(*dm));
  PetscCall(PetscObjectSetName((PetscObject) *dm, "Mesh"));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
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
    args: -init_dm_plex_dim 2 -init_dm_refine 3 -petscpartitioner_type simple -dm_extrude 3 -dm_view

  test: # adaptively refine the surface before extrusion
    suffix: 3
    requires: triangle
    args: -init_dm_plex_dim 2 -init_dm_plex_box_faces 5,5 -adapt -volume_constraint_1 0.01 -volume_constraint_2 0.000625 -dm_extrude 10

TEST*/
