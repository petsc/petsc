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
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  PetscAssertFalse(size != 1,PETSC_COMM_WORLD, PETSC_ERR_USER, "Only coded for one MPI process");

  ierr             = PetscOptionsBegin(PETSC_COMM_WORLD,"","PetscSF type freeing options","none");CHKERRQ(ierr);
  test_dupped_type = PETSC_FALSE;
  ierr             = PetscOptionsBool("-test_dupped_type", "Test dupped input type","",test_dupped_type,&test_dupped_type,NULL);CHKERRQ(ierr);
  ierr             = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);

  nleaves = 1;
  nroots = 1;
  ierr = PetscMalloc1(nleaves,&ilocal);CHKERRQ(ierr);

  for (i = 0; i<nleaves; i++) {
    ilocal[i] = i;
  }

  ierr = PetscMalloc1(nleaves,&iremote);CHKERRQ(ierr);
  iremote[0].rank = 0;
  iremote[0].index = 0;
  ierr = PetscSFSetGraph(sf,nroots,nleaves,ilocal,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscSFView(sf,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = VecSetSizes(A,4,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(A);CHKERRQ(ierr);
  ierr = VecSetUp(A);CHKERRQ(ierr);

  ierr = VecDuplicate(A,&Aout);CHKERRQ(ierr);
  ierr = VecGetArray(A,&bufA);CHKERRQ(ierr);
  for (i=0; i<4; i++) {
    bufA[i] = (PetscScalar)i;
  }
  ierr = VecRestoreArray(A,&bufA);CHKERRQ(ierr);

  ierr = VecGetArrayRead(A,(const PetscScalar**)&bufA);CHKERRQ(ierr);
  ierr = VecGetArray(Aout,&bufAout);CHKERRQ(ierr);

  ierr = MPI_Type_contiguous(4, MPIU_SCALAR, &contig);CHKERRMPI(ierr);
  ierr = MPI_Type_commit(&contig);CHKERRMPI(ierr);

  if (test_dupped_type) {
    MPI_Datatype tmp;
    ierr = MPI_Type_dup(contig, &tmp);CHKERRMPI(ierr);
    ierr = MPI_Type_free(&contig);CHKERRMPI(ierr);
    contig = tmp;
  }
  for (i=0;i<10000;i++) {
    ierr = PetscSFBcastBegin(sf,contig,bufA,bufAout,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,contig,bufA,bufAout,MPI_REPLACE);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(A,(const PetscScalar**)&bufA);CHKERRQ(ierr);
  ierr = VecRestoreArray(Aout,&bufAout);CHKERRQ(ierr);

  ierr = VecView(Aout,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&Aout);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = MPI_Type_free(&contig);CHKERRMPI(ierr);
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
