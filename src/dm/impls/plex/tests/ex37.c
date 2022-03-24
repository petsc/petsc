static const char help[] = "Test of CAD functionality";

#include <petscdmplex.h>

/* TODO
  - Fix IGES
  - Test tessellation using -dm_plex_egads_with_tess
*/

typedef struct {
  char      filename[PETSC_MAX_PATH_LEN];
  PetscBool volumeMesh;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->filename[0] = '\0';
  options->volumeMesh  = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "EGADSPlex Problem Options", "EGADSLite");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsString("-filename", "The CAD file", "ex37.c", options->filename, options->filename, sizeof(options->filename), NULL));
  CHKERRQ(PetscOptionsBool("-volume_mesh", "Create a volume mesh", "ex37.c", options->volumeMesh, &options->volumeMesh, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeVolume(DM dm)
{
  PetscObject    obj = (PetscObject) dm;
  DMLabel        bodyLabel, faceLabel, edgeLabel;
  double         surface = 0., volume = 0., vol;
  PetscInt       dim, pStart, pEnd, p, pid;
  const char    *name;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  if (dim < 2) PetscFunctionReturn(0);
  CHKERRQ(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  CHKERRQ(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  CHKERRQ(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));

  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    CHKERRQ(DMLabelGetValue(dim == 2 ? faceLabel : bodyLabel, p, &pid));
    if (pid >= 0) {
      CHKERRQ(DMPlexComputeCellGeometryFVM(dm, p, &vol, NULL, NULL));
      volume += vol;
    }
  }
  CHKERRQ(DMPlexGetHeightStratum(dm, 1, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    CHKERRQ(DMLabelGetValue(dim == 2 ? edgeLabel : faceLabel, p, &pid));
    if (pid >= 0) {
      CHKERRQ(DMPlexComputeCellGeometryFVM(dm, p, &vol, NULL, NULL));
      surface += vol;
    }
  }

  CHKERRQ(PetscObjectGetName(obj, &name));
  CHKERRQ(PetscPrintf(PetscObjectComm(obj), "DM %s: Surface Area = %.6e Volume = %.6e\n", name ? name : "", surface, volume));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  DM             surface, dm;
  AppCtx         ctx;
  PetscErrorCode ierr;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  CHKERRQ(DMPlexCreateFromFile(PETSC_COMM_WORLD, ctx.filename, "ex37_plex", PETSC_TRUE, &surface));
  CHKERRQ(PetscObjectSetName((PetscObject) surface, "CAD Surface"));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) surface, "sur_"));
  CHKERRQ(DMSetFromOptions(surface));
  CHKERRQ(DMViewFromOptions(surface, NULL, "-dm_view"));
  CHKERRQ(ComputeVolume(surface));

  if (ctx.volumeMesh) {
    CHKERRQ(DMPlexGenerate(surface, "tetgen", PETSC_TRUE, &dm));
    CHKERRQ(PetscObjectSetName((PetscObject) dm, "CAD Mesh"));
    CHKERRQ(DMPlexSetRefinementUniform(dm, PETSC_TRUE));
    CHKERRQ(DMViewFromOptions(dm, NULL, "-pre_dm_view"));

    CHKERRQ(DMPlexInflateToGeomModel(dm));
    CHKERRQ(DMViewFromOptions(dm, NULL, "-inf_dm_view"));

    CHKERRQ(DMSetFromOptions(dm));
    CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
    CHKERRQ(ComputeVolume(dm));
  }

  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(DMDestroy(&surface));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: egads tetgen

  test:
    suffix: sphere_0
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/unit_sphere.egadslite -dm_refine 1 -sur_dm_view -dm_plex_check_all -dm_plex_egads_print_model -sur_dm_plex_view_labels "EGADS Body ID","EGADS Face ID","EGADS Edge ID"

  test:
    suffix: sphere_egads
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/unit_sphere.egads -dm_refine 1 -sur_dm_view -dm_plex_check_all -dm_plex_egads_print_model -sur_dm_plex_view_labels "EGADS Body ID","EGADS Face ID","EGADS Edge ID"

  test:
    suffix: sphere_iges
    requires: broken
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/unit_sphere.igs -dm_refine 1 -sur_dm_view -dm_plex_check_all -dm_plex_egads_print_model -sur_dm_plex_view_labels "EGADS Body ID","EGADS Face ID","EGADS Edge ID"

  test:
    suffix: sphere_step
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/unit_sphere.stp -dm_refine 1 -sur_dm_view -dm_plex_check_all -dm_plex_egads_print_model -sur_dm_plex_view_labels "EGADS Body ID","EGADS Face ID","EGADS Edge ID"

  test:
    suffix: nozzle_0
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/nozzle.egadslite -sur_dm_refine 1 -sur_dm_view -dm_plex_check_all

  test:
    suffix: nozzle_egads
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/nozzle.egads -sur_dm_refine 1 -sur_dm_view -dm_plex_check_all

  test:
    suffix: nozzle_iges
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/nozzle.igs -sur_dm_refine 1 -sur_dm_view -dm_plex_check_all

  test:
    suffix: nozzle_step
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/nozzle.stp -sur_dm_refine 1 -sur_dm_view -dm_plex_check_all

TEST*/
