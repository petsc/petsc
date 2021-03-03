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
  PetscErrorCode ierr;
  MPI_Comm       com;
  PetscMPIInt    rank;
  DMDALocalInfo  info;

  com = PetscObjectComm((PetscObject)da);
  ierr = MPI_Comm_rank(com, &rank);CHKERRMPI(ierr);
  ierr = DMDAGetLocalInfo(da, &info);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(com, "begin rank %d portion (with ghosts, %D x %D)\n",rank, info.gxm, info.gym);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, v, &p);CHKERRQ(ierr);
  for (i = info.gxs; i < info.gxs + info.gxm; i++) {
    for (j = info.gys; j < info.gys + info.gym; j++) {
      ierr = PetscSynchronizedPrintf(com, "%g, ", (double) PetscRealPart(p[j][i]));CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedPrintf(com, "\n");CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArray(da, v, &p);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(com, "end rank %d portion\n", rank);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(com, PETSC_STDOUT);CHKERRQ(ierr);
  return 0;
}

/* Set a Vec v to value, but do not touch ghosts. */
PetscErrorCode VecSetOwned(DM da, Vec v, PetscScalar value)
{
  PetscScalar    **p;
  PetscInt         i, j, xs, xm, ys, ym;
  PetscErrorCode   ierr;

  ierr = DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, v, &p);CHKERRQ(ierr);
  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      p[j][i] = value;
    }
  }
  ierr = DMDAVecRestoreArray(da, v, &p);CHKERRQ(ierr);
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
  ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,M,N,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da, &local);CHKERRQ(ierr);

  ierr  = VecSet(local, value);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\nAfter setting all values to %d:\n", (int)PetscRealPart(value));CHKERRQ(ierr);
  ierr = PrintVecWithGhosts(da, local);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "done\n");CHKERRQ(ierr);

  value += 1.0;
  /* set values owned by a process, leaving ghosts alone */
  ierr = VecSetOwned(da, local, value);CHKERRQ(ierr);

  /* print after re-setting interior values again */
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\nAfter setting interior values to %d:\n", (int)PetscRealPart(value));CHKERRQ(ierr);
  ierr = PrintVecWithGhosts(da, local);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "done\n");CHKERRQ(ierr);

  /* communicate ghosts */
  ierr  = DMLocalToLocalBegin(da, local, INSERT_VALUES, local);CHKERRQ(ierr);
  ierr  = DMLocalToLocalEnd(da, local, INSERT_VALUES, local);CHKERRQ(ierr);

  /* print again */
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\nAfter local-to-local communication:\n");CHKERRQ(ierr);
  ierr = PrintVecWithGhosts(da, local);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "done\n");CHKERRQ(ierr);

  /* Free memory */
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      nsize: 2

TEST*/
