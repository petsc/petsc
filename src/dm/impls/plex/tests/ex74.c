static char help[] = "Tests for submesh creation for periodic meshes\n\n";

#include <petscdmplex.h>
#include <petsc/private/dmimpl.h>
#include <petscsf.h>

/* 3 x 2 2D mesh periodic in x-direction using nproc = 1.

Cell submesh (subdim = 2):

     12--21--13--22--14--23---~      6--12---7--13---8--14---~
      |       |       |       ~      |       |       |       ~
     25   3  27   4  29   5   ~     15   0  16   1  17   2   ~
      |       |       |       ~      |       |       |       ~
      9--18--10--19--11--20---~  ->  3---9---4--10---5--11---~
      |       |       |       ~
     24   0  26   1  28   2   ~
      |       |       |       ~
      6--15---7--16---8--17---~

Facet submesh (subdim = 1):

     12--21--13--22--14--23---~      3---0---4---1---5---2---~
      |       |       |       ~
     25   3  27   4  29   5   ~
      |       |       |       ~
      9--18--10--19--11--20---~  ->
      |       |       |       ~
     24   0  26   1  28   2   ~
      |       |       |       ~
      6--15---7--16---8--17---~

*/

typedef struct {
  PetscInt subdim; /* The topological dimension of the submesh */
} AppCtx;

PetscErrorCode ProcessOptions(AppCtx *options)
{
  PetscFunctionBegin;
  options->subdim = 2;

  PetscOptionsBegin(PETSC_COMM_SELF, "", "Filtering Problem Options", "DMPLEX");
  PetscCall(PetscOptionsBoundedInt("-subdim", "The topological dimension of the submesh", "ex74.c", options->subdim, &options->subdim, NULL, 0));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt             dim = 2;
  DM                   dm, subdm, coorddm;
  PetscFE              coordfe;
  const PetscInt       faces[2] = {3, 2};
  const PetscReal      lower[2] = {0., 0.}, upper[2] = {3., 2.};
  const DMBoundaryType periodicity[2] = {DM_BOUNDARY_PERIODIC, DM_BOUNDARY_NONE};
  DMLabel              filter;
  const PetscInt       filterValue = 1;
  MPI_Comm             comm;
  PetscMPIInt          size;
  AppCtx               user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(&user));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size != 1) {
    PetscCall(PetscPrintf(comm, "This example is specifically designed for comm size == 1.\n"));
    PetscCall(PetscFinalize());
    return 0;
  }
  PetscCall(DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, faces, lower, upper, periodicity, PETSC_TRUE, 0, PETSC_TRUE, &dm));
  /* Reset DG coordinates so that DMLocalizeCoordinates() will run again */
  /* in DMSetFromOptions() after CG coordinate FE is set.                */
  dm->coordinates[1].dim = -1;
  /* Localize coordinates on facets, too, if we are to make a facet submesh.  */
  /* Otherwise, DG coordinate values will not be copied from the parent mesh. */
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "filter", &filter));
  switch (user.subdim) {
  case 2:
    PetscCall(DMLabelSetValue(filter, 3, filterValue));
    PetscCall(DMLabelSetValue(filter, 4, filterValue));
    PetscCall(DMLabelSetValue(filter, 5, filterValue));
    break;
  case 1:
    PetscCall(DMLabelSetValue(filter, 21, filterValue));
    PetscCall(DMLabelSetValue(filter, 22, filterValue));
    PetscCall(DMLabelSetValue(filter, 23, filterValue));
    break;
  default:
    PetscCall(PetscPrintf(comm, "This example is only for subdim == {1, 2}.\n"));
    PetscCall(PetscFinalize());
    return 0;
  }
  PetscCall(DMPlexFilter(dm, filter, filterValue, PETSC_FALSE, PETSC_FALSE, NULL, &subdm));
  PetscCall(DMLabelDestroy(&filter));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscObjectSetName((PetscObject)subdm, "Example_SubDM"));
  PetscCall(DMViewFromOptions(subdm, NULL, "-dm_view"));
  PetscCall(DMGetCellCoordinateDM(subdm, &coorddm));
  PetscCall(DMGetField(coorddm, 0, NULL, (PetscObject *)&coordfe));
  PetscCall(PetscFEView(coordfe, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDestroy(&subdm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -dm_coord_space 1 -dm_coord_petscspace_degree 1 -dm_localize 1 -dm_view ascii::ascii_info_detail

    test:
      suffix: 0
      args: -dm_sparse_localize 0 -dm_localize_height 0 -subdim 2

    test:
      suffix: 1
      args: -dm_sparse_localize 1 -dm_localize_height 0 -subdim 2

    test:
      suffix: 2
      args: -dm_sparse_localize 0 -dm_localize_height 1 -subdim 1

    test:
      suffix: 3
      args: -dm_sparse_localize 1 -dm_localize_height 1 -subdim 1

TEST*/
