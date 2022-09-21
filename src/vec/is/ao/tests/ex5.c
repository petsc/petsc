/*
 Created by Huy Vo on 12/3/18.
*/
static char help[] = "Test memory scalable AO.\n\n";

#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscao.h>

int main(int argc, char *argv[])
{
  PetscInt       n_global = 16;
  MPI_Comm       comm;
  PetscLayout    layout;
  PetscInt       local_size;
  PetscInt       start, end;
  PetscMPIInt    rank;
  PetscInt      *app_indices, *petsc_indices, *ia, *ia0;
  PetscInt       i;
  AO             app2petsc;
  IS             app_is, petsc_is;
  const PetscInt n_loc = 8;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(PetscLayoutCreate(comm, &layout));
  PetscCall(PetscLayoutSetSize(layout, n_global));
  PetscCall(PetscLayoutSetLocalSize(layout, PETSC_DECIDE));
  PetscCall(PetscLayoutSetUp(layout));
  PetscCall(PetscLayoutGetLocalSize(layout, &local_size));
  PetscCall(PetscLayoutGetRange(layout, &start, &end));

  PetscCall(PetscMalloc1(local_size, &app_indices));
  PetscCall(PetscMalloc1(local_size, &petsc_indices));
  /*  Add values for local indices for usual states */
  for (i = 0; i < local_size; ++i) {
    app_indices[i]   = start + i;
    petsc_indices[i] = end - 1 - i;
  }

  /* Create the AO object that maps from lexicographic ordering to Petsc Vec ordering */
  PetscCall(ISCreateGeneral(comm, local_size, &app_indices[0], PETSC_COPY_VALUES, &app_is));
  PetscCall(ISCreateGeneral(comm, local_size, &petsc_indices[0], PETSC_COPY_VALUES, &petsc_is));
  PetscCall(AOCreate(comm, &app2petsc));
  PetscCall(AOSetIS(app2petsc, app_is, petsc_is));
  PetscCall(AOSetType(app2petsc, AOMEMORYSCALABLE));
  PetscCall(AOSetFromOptions(app2petsc));
  PetscCall(ISDestroy(&app_is));
  PetscCall(ISDestroy(&petsc_is));
  PetscCall(AOView(app2petsc, PETSC_VIEWER_STDOUT_WORLD));

  /* Test AOApplicationToPetsc */
  PetscCall(PetscMalloc1(n_loc, &ia));
  PetscCall(PetscMalloc1(n_loc, &ia0));
  if (rank == 0) {
    ia[0] = 0;
    ia[1] = -1;
    ia[2] = 1;
    ia[3] = 2;
    ia[4] = -1;
    ia[5] = 4;
    ia[6] = 5;
    ia[7] = 6;
  } else {
    ia[0] = -1;
    ia[1] = 8;
    ia[2] = 9;
    ia[3] = 10;
    ia[4] = -1;
    ia[5] = 12;
    ia[6] = 13;
    ia[7] = 14;
  }
  PetscCall(PetscArraycpy(ia0, ia, n_loc));

  PetscCall(AOApplicationToPetsc(app2petsc, n_loc, ia));

  for (i = 0; i < n_loc; ++i) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "proc = %d : %" PetscInt_FMT " -> %" PetscInt_FMT " \n", rank, ia0[i], ia[i]));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  PetscCall(AODestroy(&app2petsc));
  PetscCall(PetscLayoutDestroy(&layout));
  PetscCall(PetscFree(app_indices));
  PetscCall(PetscFree(petsc_indices));
  PetscCall(PetscFree(ia));
  PetscCall(PetscFree(ia0));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     nsize: 2

TEST*/
