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
  ierr = PetscOptionsGetInt(NULL,NULL,"-nl",&nl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-explicit_inverse",&inverse,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  ierr = PetscSFCreate(PETSC_COMM_WORLD, &sfA);CHKERRQ(ierr);
  ierr = PetscSFCreate(PETSC_COMM_WORLD, &sfB);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sfA);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sfB);CHKERRQ(ierr);

  n = 4*nl*size;
  m = 2*nl;
  k = nl;

  nldataA = rank == 0 ? n : 0;
  nldataB = 3*nl;

  nrootsA  = m;
  nleavesA = rank == 0 ? size*m : 0;
  nrootsB  = rank == 0 ? n : 0;
  nleavesB = k;

  ierr = PetscMalloc1(nleavesA, &ilocalA);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleavesA, &iremoteA);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleavesB, &ilocalB);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleavesB, &iremoteB);CHKERRQ(ierr);

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
  ierr = PetscSFSetGraph(sfA, nrootsA, nleavesA, ilocalA, PETSC_OWN_POINTER, iremoteA, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sfB, nrootsB, nleavesB, ilocalB, PETSC_OWN_POINTER, iremoteB, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sfA);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sfB);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)sfA, "sfA");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)sfB, "sfB");CHKERRQ(ierr);
  ierr = PetscSFViewFromOptions(sfA, NULL, "-view");CHKERRQ(ierr);
  ierr = PetscSFViewFromOptions(sfB, NULL, "-view");CHKERRQ(ierr);

  ierr = PetscSFGetLeafRange(sfA, NULL, &mA);CHKERRQ(ierr);
  ierr = PetscSFGetLeafRange(sfB, NULL, &mB);CHKERRQ(ierr);
  ierr = PetscMalloc2(nrootsA, &rdA, nldataA, &ldA);CHKERRQ(ierr);
  ierr = PetscMalloc2(nrootsB, &rdB, nldataB, &ldB);CHKERRQ(ierr);
  for (i = 0; i < nrootsA; i++) rdA[i] = m*rank + i;
  for (i = 0; i < nldataA; i++) ldA[i] = -1;
  for (i = 0; i < nldataB; i++) ldB[i] = -1;

  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "BcastB(BcastA)\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "A: root data\n");CHKERRQ(ierr);
  ierr = PetscIntView(nrootsA, rdA, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sfA, MPIU_INT, rdA, ldA,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sfA, MPIU_INT, rdA, ldA,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "A: leaf data (all)\n");CHKERRQ(ierr);
  ierr = PetscIntView(nldataA, ldA, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sfB, MPIU_INT, ldA, ldB,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sfB, MPIU_INT, ldA, ldB,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "B: leaf data (all)\n");CHKERRQ(ierr);
  ierr = PetscIntView(nldataB, ldB, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscSFCompose(sfA, sfB, &sfBA);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sfBA);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sfBA);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)sfBA, "sfBA");CHKERRQ(ierr);
  ierr = PetscSFViewFromOptions(sfBA, NULL, "-view");CHKERRQ(ierr);

  for (i = 0; i < nldataB; i++) ldB[i] = -1;
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "BcastBA\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "BA: root data\n");CHKERRQ(ierr);
  ierr = PetscIntView(nrootsA, rdA, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sfBA, MPIU_INT, rdA, ldB,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sfBA, MPIU_INT, rdA, ldB,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "BA: leaf data (all)\n");CHKERRQ(ierr);
  ierr = PetscIntView(nldataB, ldB, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscSFCreateInverseSF(sfA, &sfAm);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sfAm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)sfAm, "sfAm");CHKERRQ(ierr);
  ierr = PetscSFViewFromOptions(sfAm, NULL, "-view");CHKERRQ(ierr);

  if (!inverse) {
    ierr = PetscSFComposeInverse(sfA, sfA, &sfAAm);CHKERRQ(ierr);
  } else {
    ierr = PetscSFCompose(sfA, sfAm, &sfAAm);CHKERRQ(ierr);
  }
  ierr = PetscSFSetFromOptions(sfAAm);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sfAAm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)sfAAm, "sfAAm");CHKERRQ(ierr);
  ierr = PetscSFViewFromOptions(sfAAm, NULL, "-view");CHKERRQ(ierr);

  ierr = PetscSFCreateInverseSF(sfB, &sfBm);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sfBm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)sfBm, "sfBm");CHKERRQ(ierr);
  ierr = PetscSFViewFromOptions(sfBm, NULL, "-view");CHKERRQ(ierr);

  if (!inverse) {
    ierr = PetscSFComposeInverse(sfB, sfB, &sfBBm);CHKERRQ(ierr);
  } else {
    ierr = PetscSFCompose(sfB, sfBm, &sfBBm);CHKERRQ(ierr);
  }
  ierr = PetscSFSetFromOptions(sfBBm);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sfBBm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)sfBBm, "sfBBm");CHKERRQ(ierr);
  ierr = PetscSFViewFromOptions(sfBBm, NULL, "-view");CHKERRQ(ierr);

  ierr = PetscFree2(rdA, ldA);CHKERRQ(ierr);
  ierr = PetscFree2(rdB, ldB);CHKERRQ(ierr);

  ierr = PetscSFDestroy(&sfA);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfB);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfBA);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfAm);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfBm);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfAAm);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfBBm);CHKERRQ(ierr);

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
