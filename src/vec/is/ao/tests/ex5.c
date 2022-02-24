/*
 Created by Huy Vo on 12/3/18.
*/
static char help[] = "Test memory scalable AO.\n\n";

#include<petsc.h>
#include<petscvec.h>
#include<petscmat.h>
#include<petscao.h>

int main(int argc, char *argv[])
{
  PetscInt              ierr;
  PetscInt              n_global = 16;
  MPI_Comm              comm;
  PetscLayout           layout;
  PetscInt              local_size;
  PetscInt              start, end;
  PetscMPIInt           rank;
  PetscInt              *app_indices,*petsc_indices,*ia,*ia0;
  PetscInt              i;
  AO                    app2petsc;
  IS                    app_is, petsc_is;
  const PetscInt        n_loc = 8;

  ierr = PetscInitialize(&argc, &argv, (char *) 0, help); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));

  CHKERRQ(PetscLayoutCreate(comm, &layout));
  CHKERRQ(PetscLayoutSetSize(layout, n_global));
  CHKERRQ(PetscLayoutSetLocalSize(layout, PETSC_DECIDE));
  CHKERRQ(PetscLayoutSetUp(layout));
  CHKERRQ(PetscLayoutGetLocalSize(layout, &local_size));
  CHKERRQ(PetscLayoutGetRange(layout, &start, &end));

  CHKERRQ(PetscMalloc1(local_size,&app_indices));
  CHKERRQ(PetscMalloc1(local_size,&petsc_indices));
  /*  Add values for local indices for usual states */
  for (i = 0; i < local_size; ++i) {
    app_indices[i] = start + i;
    petsc_indices[i] = end -1 - i;
  }

  /* Create the AO object that maps from lexicographic ordering to Petsc Vec ordering */
  CHKERRQ(ISCreateGeneral(comm, local_size, &app_indices[0], PETSC_COPY_VALUES, &app_is));
  CHKERRQ(ISCreateGeneral(comm, local_size, &petsc_indices[0], PETSC_COPY_VALUES, &petsc_is));
  CHKERRQ(AOCreate(comm, &app2petsc));
  CHKERRQ(AOSetIS(app2petsc, app_is, petsc_is));
  CHKERRQ(AOSetType(app2petsc, AOMEMORYSCALABLE));
  CHKERRQ(AOSetFromOptions(app2petsc));
  CHKERRQ(ISDestroy(&app_is));
  CHKERRQ(ISDestroy(&petsc_is));
  CHKERRQ(AOView(app2petsc, PETSC_VIEWER_STDOUT_WORLD));

  /* Test AOApplicationToPetsc */
  CHKERRQ(PetscMalloc1(n_loc,&ia));
  CHKERRQ(PetscMalloc1(n_loc,&ia0));
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
  CHKERRQ(PetscArraycpy(ia0,ia,n_loc));

  CHKERRQ(AOApplicationToPetsc(app2petsc, n_loc, ia));

  for (i=0; i<n_loc; ++i) {
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"proc = %d : %" PetscInt_FMT " -> %" PetscInt_FMT " \n", rank, ia0[i], ia[i]));
  }
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  CHKERRQ(AODestroy(&app2petsc));
  CHKERRQ(PetscLayoutDestroy(&layout));
  CHKERRQ(PetscFree(app_indices));
  CHKERRQ(PetscFree(petsc_indices));
  CHKERRQ(PetscFree(ia));
  CHKERRQ(PetscFree(ia0));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     nsize: 2

TEST*/
