static const char help[] = "Test freeing of MPI types in PetscSF\n\n";

#include <petscvec.h>
#include <petscsf.h>
#include <petscviewer.h>

int main(int argc, char **argv)
{
  PetscInt    ierr;
  PetscSF     sf;
  Vec         A,Aout;
  PetscScalar *bufA;
  PetscScalar *bufAout;
  PetscMPIInt rank, size;
  PetscInt    nroots, nleaves;
  PetscInt    i;
  PetscInt    *ilocal;
  PetscSFNode *iremote;
  PetscBool   test_dupped_type;
  MPI_Datatype contig;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCheckFalse(size != 1,PETSC_COMM_WORLD, PETSC_ERR_USER, "Only coded for one MPI process");

  ierr             = PetscOptionsBegin(PETSC_COMM_WORLD,"","PetscSF type freeing options","none");CHKERRQ(ierr);
  test_dupped_type = PETSC_FALSE;
  CHKERRQ(PetscOptionsBool("-test_dupped_type", "Test dupped input type","",test_dupped_type,&test_dupped_type,NULL));
  ierr             = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&sf));
  CHKERRQ(PetscSFSetFromOptions(sf));

  nleaves = 1;
  nroots = 1;
  CHKERRQ(PetscMalloc1(nleaves,&ilocal));

  for (i = 0; i<nleaves; i++) {
    ilocal[i] = i;
  }

  CHKERRQ(PetscMalloc1(nleaves,&iremote));
  iremote[0].rank = 0;
  iremote[0].index = 0;
  CHKERRQ(PetscSFSetGraph(sf,nroots,nleaves,ilocal,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER));
  CHKERRQ(PetscSFSetUp(sf));
  CHKERRQ(PetscSFView(sf,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(VecSetSizes(A,4,PETSC_DETERMINE));
  CHKERRQ(VecSetFromOptions(A));
  CHKERRQ(VecSetUp(A));

  CHKERRQ(VecDuplicate(A,&Aout));
  CHKERRQ(VecGetArray(A,&bufA));
  for (i=0; i<4; i++) {
    bufA[i] = (PetscScalar)i;
  }
  CHKERRQ(VecRestoreArray(A,&bufA));

  CHKERRQ(VecGetArrayRead(A,(const PetscScalar**)&bufA));
  CHKERRQ(VecGetArray(Aout,&bufAout));

  CHKERRMPI(MPI_Type_contiguous(4, MPIU_SCALAR, &contig));
  CHKERRMPI(MPI_Type_commit(&contig));

  if (test_dupped_type) {
    MPI_Datatype tmp;
    CHKERRMPI(MPI_Type_dup(contig, &tmp));
    CHKERRMPI(MPI_Type_free(&contig));
    contig = tmp;
  }
  for (i=0;i<10000;i++) {
    CHKERRQ(PetscSFBcastBegin(sf,contig,bufA,bufAout,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(sf,contig,bufA,bufAout,MPI_REPLACE));
  }
  CHKERRQ(VecRestoreArrayRead(A,(const PetscScalar**)&bufA));
  CHKERRQ(VecRestoreArray(Aout,&bufAout));

  CHKERRQ(VecView(Aout,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDestroy(&A));
  CHKERRQ(VecDestroy(&Aout));
  CHKERRQ(PetscSFDestroy(&sf));
  CHKERRMPI(MPI_Type_free(&contig));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: basic
      args: -sf_type basic

   test:
      suffix: basic_dupped
      args: -test_dupped_type -sf_type basic

   test:
      suffix: window
      filter: grep -v "type" | grep -v "sort"
      args: -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor {{create allocate dynamic}}
      requires: defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

   test:
      suffix: window_dupped
      filter: grep -v "type" | grep -v "sort"
      args: -test_dupped_type -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor {{create allocate dynamic}}
      requires: defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

   test:
      suffix: window_shared
      output_file: output/ex3_window.out
      filter: grep -v "type" | grep -v "sort"
      args: -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor shared
      requires: defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY) defined(PETSC_HAVE_MPI_ONE_SIDED) !defined(PETSC_HAVE_I_MPI_NUMVERSION)

   test:
      suffix: window_dupped_shared
      output_file: output/ex3_window_dupped.out
      filter: grep -v "type" | grep -v "sort"
      args: -test_dupped_type -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor shared
      requires: defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY) defined(PETSC_HAVE_MPI_ONE_SIDED) !defined(PETSC_HAVE_I_MPI_NUMVERSION)

TEST*/
