
static char help[] = "Takes a patch of a large DMDA vector to one process.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmpatch.h>
#include <petscsf.h>

typedef struct {
  PetscScalar x, y;
} Field;

int main(int argc, char **argv)
{
  Vec         xy, sxy;
  DM          da, sda = NULL;
  PetscSF     sf;
  PetscInt    m = 10, n = 10, dof = 2;
  MatStencil  lower = {0, 3, 2, 0}, upper = {0, 7, 8, 0}; /* These are in the order of the z, y, x, logical coordinates, the fourth entry is ignored */
  MPI_Comm    comm;
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /* create the large DMDA and set coordinates (which we will copy down to the small DA). */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, m, n, PETSC_DECIDE, PETSC_DECIDE, dof, 1, 0, 0, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  /* Just as a simple example we use the coordinates as the variables in the vectors we wish to examine. */
  PetscCall(DMGetCoordinates(da, &xy));
  /* The vector entries are displayed in the "natural" ordering on the two dimensional grid; interlaced x and y with with the x variable increasing more rapidly than the y */
  PetscCall(VecView(xy, 0));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank == 0) comm = MPI_COMM_SELF;
  else comm = MPI_COMM_NULL;

  PetscCall(DMPatchZoom(da, lower, upper, comm, &sda, NULL, &sf));
  if (rank == 0) {
    PetscCall(DMCreateGlobalVector(sda, &sxy));
  } else {
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, 0, &sxy));
  }
  /*  A PetscSF can also be used as a VecScatter context */
  PetscCall(VecScatterBegin(sf, xy, sxy, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(sf, xy, sxy, INSERT_VALUES, SCATTER_FORWARD));
  /* Only rank == 0 has the entries of the patch, so run code only at that rank */
  if (rank == 0) {
    Field       **vars;
    DMDALocalInfo info;
    PetscInt      i, j;
    PetscScalar   sum = 0;

    /* The vector entries of the patch are displayed in the "natural" ordering on the two grid; interlaced x and y with with the x variable increasing more rapidly */
    PetscCall(VecView(sxy, PETSC_VIEWER_STDOUT_SELF));
    /* Compute some trivial statistic of the coordinates */
    PetscCall(DMDAGetLocalInfo(sda, &info));
    PetscCall(DMDAVecGetArray(sda, sxy, &vars));
    /* Loop over the patch of the entire domain */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      for (i = info.xs; i < info.xs + info.xm; i++) sum += vars[j][i].x;
    }
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "The sum of the x coordinates is %g\n", (double)PetscRealPart(sum)));
    PetscCall(DMDAVecRestoreArray(sda, sxy, &vars));
  }

  PetscCall(VecDestroy(&sxy));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(DMDestroy(&sda));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
     nsize: 4
     suffix: 2

TEST*/
