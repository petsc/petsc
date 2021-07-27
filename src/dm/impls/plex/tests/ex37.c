static const char help[] = "Test of EGADSLite CAD functionality";

#include <petscdmplex.h>

typedef struct {
  char filename[PETSC_MAX_PATH_LEN];
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->filename[0] = '\0';

  ierr = PetscOptionsBegin(comm, "", "EGADSPlex Problem Options", "EGADSLite");CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The EGADSLite file", "ex9.c", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  DM             dm;
  DMLabel        bodyLabel, faceLabel, edgeLabel;
  AppCtx         ctx;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &ctx);CHKERRQ(ierr);
  ierr = DMPlexCreateFromFile(comm, ctx.filename, PETSC_TRUE, &dm);CHKERRQ(ierr);

  ierr = DMGetLabel(dm, "EGADS Body ID", &bodyLabel);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "EGADS Face ID", &faceLabel);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "EGADS Edge ID", &edgeLabel);CHKERRQ(ierr);
  ierr = DMLabelView(bodyLabel, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMLabelView(faceLabel, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMLabelView(edgeLabel, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: egads

  test:
    suffix: sphere_0
    filter: sed "s/DM_[0-9a-zA-Z]*_0/DM__0/g"
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/unit_sphere.egadslite -dm_refine 1 -dm_view -dm_plex_check_all -dm_plex_egads_print_model

  test:
    suffix: nozzle_0
    filter: sed "s/DM_[0-9a-zA-Z]*_0/DM__0/g"
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/nozzle.egadslite -dm_refine 1 -dm_view -dm_plex_check_all -dm_plex_egads_print_model

TEST*/
