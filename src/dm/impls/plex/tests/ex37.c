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
  ierr = PetscOptionsString("-filename", "The CAD file", "ex37.c", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-volume_mesh", "Create a volume mesh", "ex37.c", options->volumeMesh, &options->volumeMesh, NULL);CHKERRQ(ierr);
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
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim < 2) PetscFunctionReturn(0);
  ierr = DMGetLabel(dm, "EGADS Body ID", &bodyLabel);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "EGADS Face ID", &faceLabel);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "EGADS Edge ID", &edgeLabel);CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    ierr = DMLabelGetValue(dim == 2 ? faceLabel : bodyLabel, p, &pid);CHKERRQ(ierr);
    if (pid >= 0) {
      ierr = DMPlexComputeCellGeometryFVM(dm, p, &vol, NULL, NULL);CHKERRQ(ierr);
      volume += vol;
    }
  }
  ierr = DMPlexGetHeightStratum(dm, 1, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    ierr = DMLabelGetValue(dim == 2 ? edgeLabel : faceLabel, p, &pid);CHKERRQ(ierr);
    if (pid >= 0) {
      ierr = DMPlexComputeCellGeometryFVM(dm, p, &vol, NULL, NULL);CHKERRQ(ierr);
      surface += vol;
    }
  }

  ierr = PetscObjectGetName(obj, &name);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm(obj), "DM %s: Surface Area = %.6e Volume = %.6e\n", name ? name : "", surface, volume);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  DM             surface, dm;
  AppCtx         ctx;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, ctx.filename, PETSC_TRUE, &surface);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) surface, "CAD Surface");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) surface, "sur_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(surface);CHKERRQ(ierr);
  ierr = DMViewFromOptions(surface, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = ComputeVolume(surface);CHKERRQ(ierr);

  if (ctx.volumeMesh) {
    ierr = DMPlexGenerate(surface, "tetgen", PETSC_TRUE, &dm);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) dm, "CAD Mesh");CHKERRQ(ierr);
    ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMViewFromOptions(dm, NULL, "-pre_dm_view");CHKERRQ(ierr);

    ierr = DMPlexInflateToGeomModel(dm);CHKERRQ(ierr);
    ierr = DMViewFromOptions(dm, NULL, "-inf_dm_view");CHKERRQ(ierr);

    ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
    ierr = ComputeVolume(dm);CHKERRQ(ierr);
  }

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&surface);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
