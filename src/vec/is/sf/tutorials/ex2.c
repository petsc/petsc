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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCheck(size == 2,PETSC_COMM_WORLD, PETSC_ERR_USER, "Only coded for two MPI processes");

  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  PetscCall(PetscSFSetFromOptions(sf));

  nleaves = 2;
  nroots = 1;
  PetscCall(PetscMalloc1(nleaves,&ilocal));

  for (i = 0; i<nleaves; i++) {
    ilocal[i] = i;
  }

  PetscCall(PetscMalloc1(nleaves,&iremote));
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
  PetscCall(PetscSFSetGraph(sf,nroots,nleaves,ilocal,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER));
  PetscCall(PetscSFSetUp(sf));
  PetscCall(PetscSFView(sf,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&A));
  PetscCall(VecSetSizes(A,2,PETSC_DETERMINE));
  PetscCall(VecSetFromOptions(A));
  PetscCall(VecSetUp(A));

  PetscCall(VecDuplicate(A,&B));
  PetscCall(VecDuplicate(A,&Aout));
  PetscCall(VecDuplicate(A,&Bout));
  PetscCall(VecGetArray(A,&bufA));
  PetscCall(VecGetArray(B,&bufB));
  for (i=0; i<2; i++) {
    bufA[i] = (PetscScalar)rank;
    bufB[i] = (PetscScalar)(rank) + 10.0;
  }
  PetscCall(VecRestoreArray(A,&bufA));
  PetscCall(VecRestoreArray(B,&bufB));

  PetscCall(VecGetArrayRead(A,(const PetscScalar**)&bufA));
  PetscCall(VecGetArrayRead(B,(const PetscScalar**)&bufB));
  PetscCall(VecGetArray(Aout,&bufAout));
  PetscCall(VecGetArray(Bout,&bufBout));
  PetscCall(PetscSFBcastBegin(sf,MPIU_SCALAR,(const void*)bufA,(void *)bufAout,MPI_REPLACE));
  PetscCall(PetscSFBcastBegin(sf,MPIU_SCALAR,(const void*)bufB,(void *)bufBout,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf,MPIU_SCALAR,(const void*)bufA,(void *)bufAout,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf,MPIU_SCALAR,(const void*)bufB,(void *)bufBout,MPI_REPLACE));
  PetscCall(VecRestoreArrayRead(A,(const PetscScalar**)&bufA));
  PetscCall(VecRestoreArrayRead(B,(const PetscScalar**)&bufB));
  PetscCall(VecRestoreArray(Aout,&bufAout));
  PetscCall(VecRestoreArray(Bout,&bufBout));

  PetscCall(VecView(Aout,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(Bout,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&A));
  PetscCall(VecDestroy(&B));
  PetscCall(VecDestroy(&Aout));
  PetscCall(VecDestroy(&Bout));
  PetscCall(PetscSFDestroy(&sf));

  PetscCall(PetscFinalize());
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
