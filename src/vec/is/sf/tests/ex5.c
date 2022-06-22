static char help[]= "Test PetscSFFCompose when the ilocal arrays are not identity nor dense\n\n";

#include <petsc.h>
#include <petscsf.h>

int main(int argc, char **argv)
{
  PetscSF      sfA, sfB, sfBA, sfAAm, sfBBm, sfAm, sfBm;
  PetscInt     nrootsA, nleavesA, nrootsB, nleavesB;
  PetscInt    *ilocalA, *ilocalB;
  PetscSFNode *iremoteA, *iremoteB;
  PetscMPIInt  rank,size;
  PetscInt     i,m,n,k,nl = 2,mA,mB,nldataA,nldataB;
  PetscInt    *rdA,*rdB,*ldA,*ldB;
  PetscBool    inverse    = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nl",&nl,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-explicit_inverse",&inverse,NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sfA));
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sfB));
  PetscCall(PetscSFSetFromOptions(sfA));
  PetscCall(PetscSFSetFromOptions(sfB));

  n = 4*nl*size;
  m = 2*nl;
  k = nl;

  nldataA = rank == 0 ? n : 0;
  nldataB = 3*nl;

  nrootsA  = m;
  nleavesA = rank == 0 ? size*m : 0;
  nrootsB  = rank == 0 ? n : 0;
  nleavesB = k;

  PetscCall(PetscMalloc1(nleavesA, &ilocalA));
  PetscCall(PetscMalloc1(nleavesA, &iremoteA));
  PetscCall(PetscMalloc1(nleavesB, &ilocalB));
  PetscCall(PetscMalloc1(nleavesB, &iremoteB));

  /* sf A bcast is equivalent to a sparse gather on process 0
     process 0 receives data in the middle [nl,3*nl] of the leaf data array for A */
  for (i = 0; i < nleavesA; i++) {
    iremoteA[i].rank = i/m;
    iremoteA[i].index = i%m;
    ilocalA[i] = nl + i/m * 4*nl + i%m;
  }

  /* sf B bcast is equivalent to a sparse scatter from process 0
     process 0 sends data from [nl,2*nl] of the leaf data array for A
     each process receives, in reverse order, in the middle [nl,2*nl] of the leaf data array for B */
  for (i = 0; i < nleavesB; i++) {
    iremoteB[i].rank = 0;
    iremoteB[i].index = rank * 4*nl + nl + i%m;
    ilocalB[i] = 2*nl - i - 1;
  }
  PetscCall(PetscSFSetGraph(sfA, nrootsA, nleavesA, ilocalA, PETSC_OWN_POINTER, iremoteA, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetGraph(sfB, nrootsB, nleavesB, ilocalB, PETSC_OWN_POINTER, iremoteB, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetUp(sfA));
  PetscCall(PetscSFSetUp(sfB));
  PetscCall(PetscObjectSetName((PetscObject)sfA, "sfA"));
  PetscCall(PetscObjectSetName((PetscObject)sfB, "sfB"));
  PetscCall(PetscSFViewFromOptions(sfA, NULL, "-view"));
  PetscCall(PetscSFViewFromOptions(sfB, NULL, "-view"));

  PetscCall(PetscSFGetLeafRange(sfA, NULL, &mA));
  PetscCall(PetscSFGetLeafRange(sfB, NULL, &mB));
  PetscCall(PetscMalloc2(nrootsA, &rdA, nldataA, &ldA));
  PetscCall(PetscMalloc2(nrootsB, &rdB, nldataB, &ldB));
  for (i = 0; i < nrootsA; i++) rdA[i] = m*rank + i;
  for (i = 0; i < nldataA; i++) ldA[i] = -1;
  for (i = 0; i < nldataB; i++) ldB[i] = -1;

  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "BcastB(BcastA)\n"));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "A: root data\n"));
  PetscCall(PetscIntView(nrootsA, rdA, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscSFBcastBegin(sfA, MPIU_INT, rdA, ldA,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sfA, MPIU_INT, rdA, ldA,MPI_REPLACE));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "A: leaf data (all)\n"));
  PetscCall(PetscIntView(nldataA, ldA, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscSFBcastBegin(sfB, MPIU_INT, ldA, ldB,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sfB, MPIU_INT, ldA, ldB,MPI_REPLACE));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "B: leaf data (all)\n"));
  PetscCall(PetscIntView(nldataB, ldB, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscSFCompose(sfA, sfB, &sfBA));
  PetscCall(PetscSFSetFromOptions(sfBA));
  PetscCall(PetscSFSetUp(sfBA));
  PetscCall(PetscObjectSetName((PetscObject)sfBA, "sfBA"));
  PetscCall(PetscSFViewFromOptions(sfBA, NULL, "-view"));

  for (i = 0; i < nldataB; i++) ldB[i] = -1;
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "BcastBA\n"));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "BA: root data\n"));
  PetscCall(PetscIntView(nrootsA, rdA, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscSFBcastBegin(sfBA, MPIU_INT, rdA, ldB,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sfBA, MPIU_INT, rdA, ldB,MPI_REPLACE));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "BA: leaf data (all)\n"));
  PetscCall(PetscIntView(nldataB, ldB, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscSFCreateInverseSF(sfA, &sfAm));
  PetscCall(PetscSFSetFromOptions(sfAm));
  PetscCall(PetscObjectSetName((PetscObject)sfAm, "sfAm"));
  PetscCall(PetscSFViewFromOptions(sfAm, NULL, "-view"));

  if (!inverse) {
    PetscCall(PetscSFComposeInverse(sfA, sfA, &sfAAm));
  } else {
    PetscCall(PetscSFCompose(sfA, sfAm, &sfAAm));
  }
  PetscCall(PetscSFSetFromOptions(sfAAm));
  PetscCall(PetscSFSetUp(sfAAm));
  PetscCall(PetscObjectSetName((PetscObject)sfAAm, "sfAAm"));
  PetscCall(PetscSFViewFromOptions(sfAAm, NULL, "-view"));

  PetscCall(PetscSFCreateInverseSF(sfB, &sfBm));
  PetscCall(PetscSFSetFromOptions(sfBm));
  PetscCall(PetscObjectSetName((PetscObject)sfBm, "sfBm"));
  PetscCall(PetscSFViewFromOptions(sfBm, NULL, "-view"));

  if (!inverse) {
    PetscCall(PetscSFComposeInverse(sfB, sfB, &sfBBm));
  } else {
    PetscCall(PetscSFCompose(sfB, sfBm, &sfBBm));
  }
  PetscCall(PetscSFSetFromOptions(sfBBm));
  PetscCall(PetscSFSetUp(sfBBm));
  PetscCall(PetscObjectSetName((PetscObject)sfBBm, "sfBBm"));
  PetscCall(PetscSFViewFromOptions(sfBBm, NULL, "-view"));

  PetscCall(PetscFree2(rdA, ldA));
  PetscCall(PetscFree2(rdB, ldB));

  PetscCall(PetscSFDestroy(&sfA));
  PetscCall(PetscSFDestroy(&sfB));
  PetscCall(PetscSFDestroy(&sfBA));
  PetscCall(PetscSFDestroy(&sfAm));
  PetscCall(PetscSFDestroy(&sfBm));
  PetscCall(PetscSFDestroy(&sfAAm));
  PetscCall(PetscSFDestroy(&sfBBm));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     suffix: 1
     args: -view -explicit_inverse {{0 1}}

   test:
     nsize: 7
     filter: grep -v "type" | grep -v "sort"
     suffix: 2
     args: -view -nl 5 -explicit_inverse {{0 1}}

   # we cannot test for -sf_window_flavor dynamic because SFCompose with sparse leaves may change the root data pointer only locally, and this is not supported by the dynamic case
   test:
     nsize: 7
     suffix: 2_window
     filter: grep -v "type" | grep -v "sort"
     output_file: output/ex5_2.out
     args: -view -nl 5 -explicit_inverse {{0 1}} -sf_type window -sf_window_sync {{fence lock active}} -sf_window_flavor {{create allocate}}
     requires: defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

   # The nightly test suite with MPICH uses ch3:sock, which is broken when winsize == 0 in some of the processes
   test:
     nsize: 7
     suffix: 2_window_shared
     filter: grep -v "type" | grep -v "sort"
     output_file: output/ex5_2.out
     args: -view -nl 5 -explicit_inverse {{0 1}} -sf_type window -sf_window_sync {{fence lock active}} -sf_window_flavor shared
     requires: defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY) !defined(PETSC_HAVE_MPICH_NUMVERSION) defined(PETSC_HAVE_MPI_ONE_SIDED)

TEST*/
