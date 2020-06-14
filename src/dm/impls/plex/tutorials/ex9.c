static char help[] = "Evaluate the shape quality of a mesh\n\n";

#include <petscdmplex.h>

typedef struct {
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
  PetscBool report;                       /* Print a quality report */
  PetscReal condLimit;                    /* Condition number limit for cell output */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->filename[0] = '\0';
  options->report      = PETSC_FALSE;
  options->condLimit   = PETSC_DETERMINE;

  ierr = PetscOptionsBegin(comm, "", "Mesh Quality Evaluation Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex9.c", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-report", "Output a mesh quality report", "ex9.c", options->report, &options->report, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-cond_limit", "Condition number limit for cell output", "ex9.c", options->condLimit, &options->condLimit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscStrlen(options->filename, &len);CHKERRQ(ierr);
  if (!len) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "You must specify an input mesh using -filename");
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         ctx;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, ctx.filename, PETSC_TRUE, &dm);CHKERRQ(ierr);
  ierr = DMPlexCheckCellShape(dm, ctx.report, ctx.condLimit);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    requires: exodusii
    nsize: {{1 2}}
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo -report

  test:
    suffix: 1
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square.msh -report

TEST*/
