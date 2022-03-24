static const char help[] = "Test overlapped communication on a single star forest (PetscSF)\n\n";

#include <petscvec.h>
#include <petscsf.h>
#include <petscviewer.h>

int main(int argc, char **argv)
{
  PetscSF     sf;
  Vec         A,Aout;
  Vec         B,Bout;
  PetscScalar *bufA;
  PetscScalar *bufAout;
  PetscScalar *bufB;
  PetscScalar *bufBout;
  PetscMPIInt rank, size;
  PetscInt    nroots, nleaves;
  PetscInt    i;
  PetscInt    *ilocal;
  PetscSFNode *iremote;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCheckFalse(size != 2,PETSC_COMM_WORLD, PETSC_ERR_USER, "Only coded for two MPI processes");

  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetFromOptions(sf));

  nleaves = 2;
  nroots = 1;
  CHKERRQ(PetscMalloc1(nleaves,&ilocal));

  for (i = 0; i<nleaves; i++) {
    ilocal[i] = i;
  }

  CHKERRQ(PetscMalloc1(nleaves,&iremote));
  if (rank == 0) {
    iremote[0].rank = 0;
    iremote[0].index = 0;
    iremote[1].rank = 1;
    iremote[1].index = 0;
  } else {
    iremote[0].rank = 1;
    iremote[0].index = 0;
    iremote[1].rank = 0;
    iremote[1].index = 0;
  }
  CHKERRQ(PetscSFSetGraph(sf,nroots,nleaves,ilocal,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER));
  CHKERRQ(PetscSFSetUp(sf));
  CHKERRQ(PetscSFView(sf,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(VecSetSizes(A,2,PETSC_DETERMINE));
  CHKERRQ(VecSetFromOptions(A));
  CHKERRQ(VecSetUp(A));

  CHKERRQ(VecDuplicate(A,&B));
  CHKERRQ(VecDuplicate(A,&Aout));
  CHKERRQ(VecDuplicate(A,&Bout));
  CHKERRQ(VecGetArray(A,&bufA));
  CHKERRQ(VecGetArray(B,&bufB));
  for (i=0; i<2; i++) {
    bufA[i] = (PetscScalar)rank;
    bufB[i] = (PetscScalar)(rank) + 10.0;
  }
  CHKERRQ(VecRestoreArray(A,&bufA));
  CHKERRQ(VecRestoreArray(B,&bufB));

  CHKERRQ(VecGetArrayRead(A,(const PetscScalar**)&bufA));
  CHKERRQ(VecGetArrayRead(B,(const PetscScalar**)&bufB));
  CHKERRQ(VecGetArray(Aout,&bufAout));
  CHKERRQ(VecGetArray(Bout,&bufBout));
  CHKERRQ(PetscSFBcastBegin(sf,MPIU_SCALAR,(const void*)bufA,(void *)bufAout,MPI_REPLACE));
  CHKERRQ(PetscSFBcastBegin(sf,MPIU_SCALAR,(const void*)bufB,(void *)bufBout,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sf,MPIU_SCALAR,(const void*)bufA,(void *)bufAout,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sf,MPIU_SCALAR,(const void*)bufB,(void *)bufBout,MPI_REPLACE));
  CHKERRQ(VecRestoreArrayRead(A,(const PetscScalar**)&bufA));
  CHKERRQ(VecRestoreArrayRead(B,(const PetscScalar**)&bufB));
  CHKERRQ(VecRestoreArray(Aout,&bufAout));
  CHKERRQ(VecRestoreArray(Bout,&bufBout));

  CHKERRQ(VecView(Aout,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecView(Bout,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDestroy(&A));
  CHKERRQ(VecDestroy(&B));
  CHKERRQ(VecDestroy(&Aout));
  CHKERRQ(VecDestroy(&Bout));
  CHKERRQ(PetscSFDestroy(&sf));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: basic
      nsize: 2
      filter: grep -v "type" | grep -v "sort"
      args: -sf_type basic

   test:
      suffix: window
      nsize: 2
      filter: grep -v "type" | grep -v "sort"
      output_file: output/ex2_basic.out
      args: -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor {{create dynamic allocate}}
      requires: defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

   # The nightly test suite with MPICH uses ch3:sock, which is broken when winsize == 0 in some of the processes
   test:
      suffix: window_shared
      nsize: 2
      filter: grep -v "type" | grep -v "sort"
      output_file: output/ex2_basic.out
      args: -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor shared
      requires: defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY) !defined(PETSC_HAVE_MPICH_NUMVERSION) defined(PETSC_HAVE_MPI_ONE_SIDED)

TEST*/
