static const char help[] = "Test freeing of MPI types in PetscSF\n\n";

#include <petscvec.h>
#include <petscsf.h>
#include <petscviewer.h>

int main(int argc, char **argv)
{
  PetscSF      sf;
  Vec          A, Aout;
  PetscScalar *bufA;
  PetscScalar *bufAout;
  PetscMPIInt  rank, size;
  PetscInt     nroots, nleaves;
  PetscInt     i;
  PetscInt    *ilocal;
  PetscSFNode *iremote;
  PetscBool    test_dupped_type;
  MPI_Datatype contig;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_USER, "Only coded for one MPI process");

  PetscOptionsBegin(PETSC_COMM_WORLD, "", "PetscSF type freeing options", "none");
  test_dupped_type = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-test_dupped_type", "Test dupped input type", "", test_dupped_type, &test_dupped_type, NULL));
  PetscOptionsEnd();

  PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sf));
  PetscCall(PetscSFSetFromOptions(sf));

  nleaves = 1;
  nroots  = 1;
  PetscCall(PetscMalloc1(nleaves, &ilocal));

  for (i = 0; i < nleaves; i++) ilocal[i] = i;

  PetscCall(PetscMalloc1(nleaves, &iremote));
  iremote[0].rank  = 0;
  iremote[0].index = 0;
  PetscCall(PetscSFSetGraph(sf, nroots, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetUp(sf));
  PetscCall(PetscSFView(sf, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &A));
  PetscCall(VecSetSizes(A, 4, PETSC_DETERMINE));
  PetscCall(VecSetFromOptions(A));
  PetscCall(VecSetUp(A));

  PetscCall(VecDuplicate(A, &Aout));
  PetscCall(VecGetArray(A, &bufA));
  for (i = 0; i < 4; i++) bufA[i] = (PetscScalar)i;
  PetscCall(VecRestoreArray(A, &bufA));

  PetscCall(VecGetArrayRead(A, (const PetscScalar **)&bufA));
  PetscCall(VecGetArray(Aout, &bufAout));

  PetscCallMPI(MPI_Type_contiguous(4, MPIU_SCALAR, &contig));
  PetscCallMPI(MPI_Type_commit(&contig));

  if (test_dupped_type) {
    MPI_Datatype tmp;
    PetscCallMPI(MPI_Type_dup(contig, &tmp));
    PetscCallMPI(MPI_Type_free(&contig));
    contig = tmp;
  }
  for (i = 0; i < 10000; i++) {
    PetscCall(PetscSFBcastBegin(sf, contig, bufA, bufAout, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, contig, bufA, bufAout, MPI_REPLACE));
  }
  PetscCall(VecRestoreArrayRead(A, (const PetscScalar **)&bufA));
  PetscCall(VecRestoreArray(Aout, &bufAout));

  PetscCall(VecView(Aout, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&A));
  PetscCall(VecDestroy(&Aout));
  PetscCall(PetscSFDestroy(&sf));
  PetscCallMPI(MPI_Type_free(&contig));
  PetscCall(PetscFinalize());
  return 0;
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
