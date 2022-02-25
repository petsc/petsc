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
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);

  ierr = PetscLayoutCreate(comm, &layout);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(layout, n_global);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(layout, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(layout, &local_size);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(layout, &start, &end);CHKERRQ(ierr);

  ierr = PetscMalloc1(local_size,&app_indices);CHKERRQ(ierr);
  ierr = PetscMalloc1(local_size,&petsc_indices);CHKERRQ(ierr);
  /*  Add values for local indices for usual states */
  for (i = 0; i < local_size; ++i) {
    app_indices[i] = start + i;
    petsc_indices[i] = end -1 - i;
  }

  /* Create the AO object that maps from lexicographic ordering to Petsc Vec ordering */
  ierr = ISCreateGeneral(comm, local_size, &app_indices[0], PETSC_COPY_VALUES, &app_is);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, local_size, &petsc_indices[0], PETSC_COPY_VALUES, &petsc_is);CHKERRQ(ierr);
  ierr = AOCreate(comm, &app2petsc);CHKERRQ(ierr);
  ierr = AOSetIS(app2petsc, app_is, petsc_is);CHKERRQ(ierr);
  ierr = AOSetType(app2petsc, AOMEMORYSCALABLE);CHKERRQ(ierr);
  ierr = AOSetFromOptions(app2petsc);CHKERRQ(ierr);
  ierr = ISDestroy(&app_is);CHKERRQ(ierr);
  ierr = ISDestroy(&petsc_is);CHKERRQ(ierr);
  ierr = AOView(app2petsc, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Test AOApplicationToPetsc */
  ierr = PetscMalloc1(n_loc,&ia);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_loc,&ia0);CHKERRQ(ierr);
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
  ierr = PetscArraycpy(ia0,ia,n_loc);CHKERRQ(ierr);

  ierr = AOApplicationToPetsc(app2petsc, n_loc, ia);CHKERRQ(ierr);

  for (i=0; i<n_loc; ++i) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"proc = %d : %" PetscInt_FMT " -> %" PetscInt_FMT " \n", rank, ia0[i], ia[i]);CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  ierr = AODestroy(&app2petsc);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
  ierr = PetscFree(app_indices);CHKERRQ(ierr);
  ierr = PetscFree(petsc_indices);CHKERRQ(ierr);
  ierr = PetscFree(ia);CHKERRQ(ierr);
  ierr = PetscFree(ia0);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     nsize: 2

TEST*/
