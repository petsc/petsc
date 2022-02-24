static char help[] = "Demonstrates the DMLocalToLocal bug in PETSc 3.6.\n\n";

/*
Use the options
     -da_grid_x <nx> - number of grid points in x direction, if M < 0
     -da_grid_y <ny> - number of grid points in y direction, if N < 0
     -da_processors_x <MX> number of processors in x directio
     -da_processors_y <MY> number of processors in x direction

  Contributed by Constantine Khroulev
*/

#include <petscdm.h>
#include <petscdmda.h>

PetscErrorCode PrintVecWithGhosts(DM da, Vec v)
{
  PetscScalar    **p;
  PetscInt       i, j;
  MPI_Comm       com;
  PetscMPIInt    rank;
  DMDALocalInfo  info;

  com = PetscObjectComm((PetscObject)da);
  CHKERRMPI(MPI_Comm_rank(com, &rank));
  CHKERRQ(DMDAGetLocalInfo(da, &info));
  CHKERRQ(PetscSynchronizedPrintf(com, "begin rank %d portion (with ghosts, %D x %D)\n",rank, info.gxm, info.gym));
  CHKERRQ(DMDAVecGetArray(da, v, &p));
  for (i = info.gxs; i < info.gxs + info.gxm; i++) {
    for (j = info.gys; j < info.gys + info.gym; j++) {
      CHKERRQ(PetscSynchronizedPrintf(com, "%g, ", (double) PetscRealPart(p[j][i])));
    }
    CHKERRQ(PetscSynchronizedPrintf(com, "\n"));
  }
  CHKERRQ(DMDAVecRestoreArray(da, v, &p));
  CHKERRQ(PetscSynchronizedPrintf(com, "end rank %d portion\n", rank));
  CHKERRQ(PetscSynchronizedFlush(com, PETSC_STDOUT));
  return 0;
}

/* Set a Vec v to value, but do not touch ghosts. */
PetscErrorCode VecSetOwned(DM da, Vec v, PetscScalar value)
{
  PetscScalar    **p;
  PetscInt         i, j, xs, xm, ys, ym;

  CHKERRQ(DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0));
  CHKERRQ(DMDAVecGetArray(da, v, &p));
  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      p[j][i] = value;
    }
  }
  CHKERRQ(DMDAVecRestoreArray(da, v, &p));
  return 0;
}

int main(int argc, char **argv)
{
  PetscInt         M = 4, N = 3;
  PetscErrorCode   ierr;
  DM               da;
  Vec              local;
  PetscScalar      value = 0.0;
  DMBoundaryType   bx    = DM_BOUNDARY_PERIODIC, by = DM_BOUNDARY_PERIODIC;
  DMDAStencilType  stype = DMDA_STENCIL_BOX;

  ierr = PetscInitialize(&argc, &argv, (char*)0, help);if (ierr) return ierr;
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,M,N,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMCreateLocalVector(da, &local));

  CHKERRQ(VecSet(local, value));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "\nAfter setting all values to %d:\n", (int)PetscRealPart(value)));
  CHKERRQ(PrintVecWithGhosts(da, local));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "done\n"));

  value += 1.0;
  /* set values owned by a process, leaving ghosts alone */
  CHKERRQ(VecSetOwned(da, local, value));

  /* print after re-setting interior values again */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "\nAfter setting interior values to %d:\n", (int)PetscRealPart(value)));
  CHKERRQ(PrintVecWithGhosts(da, local));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "done\n"));

  /* communicate ghosts */
  CHKERRQ(DMLocalToLocalBegin(da, local, INSERT_VALUES, local));
  CHKERRQ(DMLocalToLocalEnd(da, local, INSERT_VALUES, local));

  /* print again */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "\nAfter local-to-local communication:\n"));
  CHKERRQ(PrintVecWithGhosts(da, local));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "done\n"));

  /* Free memory */
  CHKERRQ(VecDestroy(&local));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2

TEST*/
