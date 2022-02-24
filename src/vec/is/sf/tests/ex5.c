static char help[]= "Test PetscSFFCompose when the ilocal arrays are not identity nor dense\n\n";

#include <petsc.h>
#include <petscsf.h>

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  PetscSF        sfA, sfB, sfBA, sfAAm, sfBBm, sfAm, sfBm;
  PetscInt       nrootsA, nleavesA, nrootsB, nleavesB;
  PetscInt       *ilocalA, *ilocalB;
  PetscSFNode    *iremoteA, *iremoteB;
  PetscMPIInt    rank,size;
  PetscInt       i,m,n,k,nl = 2,mA,mB,nldataA,nldataB;
  PetscInt       *rdA,*rdB,*ldA,*ldB;
  PetscBool      inverse = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nl",&nl,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-explicit_inverse",&inverse,NULL));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD, &sfA));
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD, &sfB));
  CHKERRQ(PetscSFSetFromOptions(sfA));
  CHKERRQ(PetscSFSetFromOptions(sfB));

  n = 4*nl*size;
  m = 2*nl;
  k = nl;

  nldataA = rank == 0 ? n : 0;
  nldataB = 3*nl;

  nrootsA  = m;
  nleavesA = rank == 0 ? size*m : 0;
  nrootsB  = rank == 0 ? n : 0;
  nleavesB = k;

  CHKERRQ(PetscMalloc1(nleavesA, &ilocalA));
  CHKERRQ(PetscMalloc1(nleavesA, &iremoteA));
  CHKERRQ(PetscMalloc1(nleavesB, &ilocalB));
  CHKERRQ(PetscMalloc1(nleavesB, &iremoteB));

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
  CHKERRQ(PetscSFSetGraph(sfA, nrootsA, nleavesA, ilocalA, PETSC_OWN_POINTER, iremoteA, PETSC_OWN_POINTER));
  CHKERRQ(PetscSFSetGraph(sfB, nrootsB, nleavesB, ilocalB, PETSC_OWN_POINTER, iremoteB, PETSC_OWN_POINTER));
  CHKERRQ(PetscSFSetUp(sfA));
  CHKERRQ(PetscSFSetUp(sfB));
  CHKERRQ(PetscObjectSetName((PetscObject)sfA, "sfA"));
  CHKERRQ(PetscObjectSetName((PetscObject)sfB, "sfB"));
  CHKERRQ(PetscSFViewFromOptions(sfA, NULL, "-view"));
  CHKERRQ(PetscSFViewFromOptions(sfB, NULL, "-view"));

  CHKERRQ(PetscSFGetLeafRange(sfA, NULL, &mA));
  CHKERRQ(PetscSFGetLeafRange(sfB, NULL, &mB));
  CHKERRQ(PetscMalloc2(nrootsA, &rdA, nldataA, &ldA));
  CHKERRQ(PetscMalloc2(nrootsB, &rdB, nldataB, &ldB));
  for (i = 0; i < nrootsA; i++) rdA[i] = m*rank + i;
  for (i = 0; i < nldataA; i++) ldA[i] = -1;
  for (i = 0; i < nldataB; i++) ldB[i] = -1;

  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "BcastB(BcastA)\n"));
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "A: root data\n"));
  CHKERRQ(PetscIntView(nrootsA, rdA, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscSFBcastBegin(sfA, MPIU_INT, rdA, ldA,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sfA, MPIU_INT, rdA, ldA,MPI_REPLACE));
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "A: leaf data (all)\n"));
  CHKERRQ(PetscIntView(nldataA, ldA, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscSFBcastBegin(sfB, MPIU_INT, ldA, ldB,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sfB, MPIU_INT, ldA, ldB,MPI_REPLACE));
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "B: leaf data (all)\n"));
  CHKERRQ(PetscIntView(nldataB, ldB, PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(PetscSFCompose(sfA, sfB, &sfBA));
  CHKERRQ(PetscSFSetFromOptions(sfBA));
  CHKERRQ(PetscSFSetUp(sfBA));
  CHKERRQ(PetscObjectSetName((PetscObject)sfBA, "sfBA"));
  CHKERRQ(PetscSFViewFromOptions(sfBA, NULL, "-view"));

  for (i = 0; i < nldataB; i++) ldB[i] = -1;
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "BcastBA\n"));
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "BA: root data\n"));
  CHKERRQ(PetscIntView(nrootsA, rdA, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscSFBcastBegin(sfBA, MPIU_INT, rdA, ldB,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sfBA, MPIU_INT, rdA, ldB,MPI_REPLACE));
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "BA: leaf data (all)\n"));
  CHKERRQ(PetscIntView(nldataB, ldB, PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(PetscSFCreateInverseSF(sfA, &sfAm));
  CHKERRQ(PetscSFSetFromOptions(sfAm));
  CHKERRQ(PetscObjectSetName((PetscObject)sfAm, "sfAm"));
  CHKERRQ(PetscSFViewFromOptions(sfAm, NULL, "-view"));

  if (!inverse) {
    CHKERRQ(PetscSFComposeInverse(sfA, sfA, &sfAAm));
  } else {
    CHKERRQ(PetscSFCompose(sfA, sfAm, &sfAAm));
  }
  CHKERRQ(PetscSFSetFromOptions(sfAAm));
  CHKERRQ(PetscSFSetUp(sfAAm));
  CHKERRQ(PetscObjectSetName((PetscObject)sfAAm, "sfAAm"));
  CHKERRQ(PetscSFViewFromOptions(sfAAm, NULL, "-view"));

  CHKERRQ(PetscSFCreateInverseSF(sfB, &sfBm));
  CHKERRQ(PetscSFSetFromOptions(sfBm));
  CHKERRQ(PetscObjectSetName((PetscObject)sfBm, "sfBm"));
  CHKERRQ(PetscSFViewFromOptions(sfBm, NULL, "-view"));

  if (!inverse) {
    CHKERRQ(PetscSFComposeInverse(sfB, sfB, &sfBBm));
  } else {
    CHKERRQ(PetscSFCompose(sfB, sfBm, &sfBBm));
  }
  CHKERRQ(PetscSFSetFromOptions(sfBBm));
  CHKERRQ(PetscSFSetUp(sfBBm));
  CHKERRQ(PetscObjectSetName((PetscObject)sfBBm, "sfBBm"));
  CHKERRQ(PetscSFViewFromOptions(sfBBm, NULL, "-view"));

  CHKERRQ(PetscFree2(rdA, ldA));
  CHKERRQ(PetscFree2(rdB, ldB));

  CHKERRQ(PetscSFDestroy(&sfA));
  CHKERRQ(PetscSFDestroy(&sfB));
  CHKERRQ(PetscSFDestroy(&sfBA));
  CHKERRQ(PetscSFDestroy(&sfAm));
  CHKERRQ(PetscSFDestroy(&sfBm));
  CHKERRQ(PetscSFDestroy(&sfAAm));
  CHKERRQ(PetscSFDestroy(&sfBBm));

  ierr = PetscFinalize();

  return ierr;
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
