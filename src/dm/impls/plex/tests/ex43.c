static char help[] = "Tests DMPlexCreateBoxMesh().\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscBool sparseLocalize; /* Only localize coordinates where necessary */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  options->sparseLocalize = PETSC_TRUE;
  PetscOptionsBegin(comm, "", "DMPlexCreateBoxMesh() Test Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-sparse_localize", "Only localize coordinates where necessary", "ex43.c", options->sparseLocalize, &options->sparseLocalize, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM             dm;
  const PetscInt faces[2]       = {3, 1};
  DMBoundaryType periodicity[2] = {DM_BOUNDARY_PERIODIC, DM_BOUNDARY_NONE};
  AppCtx         user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 2, PETSC_FALSE, faces, NULL, NULL, periodicity, PETSC_TRUE, 0, user.sparseLocalize, &dm));
  PetscCall(PetscObjectSetName((PetscObject)dm, "ExampleBoxMesh"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -sparse_localize 0 -dm_view ascii::ascii_info_detail

  test:
    suffix: 1
    args: -sparse_localize 1 -dm_view ascii::ascii_info_detail

TEST*/
