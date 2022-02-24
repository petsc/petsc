static char help[]= "Test PetscSFFCompose when the ilocal array is not the identity\n\n";

#include <petscsf.h>

int main(int argc, char **argv)
{
  PetscErrorCode     ierr;
  PetscSF            sfA, sfB, sfBA;
  PetscInt           nrootsA, nleavesA, nrootsB, nleavesB;
  PetscInt          *ilocalA, *ilocalB;
  PetscSFNode       *iremoteA, *iremoteB;
  Vec                a, b, ba;
  const PetscScalar *arrayR;
  PetscScalar       *arrayW;
  PetscMPIInt        size;
  PetscInt           i;
  PetscInt           maxleafB;
  PetscBool          flag = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD, PETSC_ERR_USER, "Only coded for one MPI process");

  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD, &sfA));
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD, &sfB));
  CHKERRQ(PetscSFSetFromOptions(sfA));
  CHKERRQ(PetscSFSetFromOptions(sfB));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-sparse_sfB",&flag,NULL));

  if (flag) {
    /* sfA permutes indices, sfB has sparse leaf space. */
    nrootsA = 3;
    nleavesA = 3;
    nrootsB = 3;
    nleavesB = 2;
  } else {
    /* sfA reverses indices, sfB is identity */
    nrootsA = nrootsB = nleavesA = nleavesB = 4;
  }
  CHKERRQ(PetscMalloc1(nleavesA, &ilocalA));
  CHKERRQ(PetscMalloc1(nleavesA, &iremoteA));
  CHKERRQ(PetscMalloc1(nleavesB, &ilocalB));
  CHKERRQ(PetscMalloc1(nleavesB, &iremoteB));

  for (i = 0; i < nleavesA; i++) {
    iremoteA[i].rank = 0;
    iremoteA[i].index = i;
    if (flag) {
      ilocalA[i] = (i + 1) % nleavesA;
    } else {
      ilocalA[i] = nleavesA - i - 1;
    }
  }

  for (i = 0; i < nleavesB; i++) {
    iremoteB[i].rank = 0;
    if (flag) {
      ilocalB[i] = nleavesB - i;
      iremoteB[i].index = nleavesB - i - 1;
    } else {
      ilocalB[i] = i;
      iremoteB[i].index = i;
    }
  }

  CHKERRQ(PetscSFSetGraph(sfA, nrootsA, nleavesA, ilocalA, PETSC_OWN_POINTER, iremoteA, PETSC_OWN_POINTER));
  CHKERRQ(PetscSFSetGraph(sfB, nrootsB, nleavesB, ilocalB, PETSC_OWN_POINTER, iremoteB, PETSC_OWN_POINTER));
  CHKERRQ(PetscSFSetUp(sfA));
  CHKERRQ(PetscSFSetUp(sfB));
  CHKERRQ(PetscObjectSetName((PetscObject)sfA, "sfA"));
  CHKERRQ(PetscObjectSetName((PetscObject)sfB, "sfB"));

  CHKERRQ(VecCreateSeq(PETSC_COMM_WORLD, nrootsA, &a));
  CHKERRQ(VecCreateSeq(PETSC_COMM_WORLD, nleavesA, &b));
  CHKERRQ(PetscSFGetLeafRange(sfB, NULL, &maxleafB));
  CHKERRQ(VecCreateSeq(PETSC_COMM_WORLD, maxleafB+1, &ba));
  CHKERRQ(VecGetArray(a, &arrayW));
  for (i = 0; i < nrootsA; i++) {
    arrayW[i] = (PetscScalar)i;
  }
  CHKERRQ(VecRestoreArray(a, &arrayW));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Initial Vec A\n"));
  CHKERRQ(VecView(a, NULL));
  CHKERRQ(VecGetArrayRead(a, &arrayR));
  CHKERRQ(VecGetArray(b, &arrayW));

  CHKERRQ(PetscSFBcastBegin(sfA, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sfA, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE));
  CHKERRQ(VecRestoreArray(b, &arrayW));
  CHKERRQ(VecRestoreArrayRead(a, &arrayR));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "\nBroadcast A->B over sfA\n"));
  CHKERRQ(VecView(b, NULL));

  CHKERRQ(VecGetArrayRead(b, &arrayR));
  CHKERRQ(VecGetArray(ba, &arrayW));
  arrayW[0] = 10.0;             /* Not touched by bcast */
  CHKERRQ(PetscSFBcastBegin(sfB, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sfB, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE));
  CHKERRQ(VecRestoreArrayRead(b, &arrayR));
  CHKERRQ(VecRestoreArray(ba, &arrayW));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "\nBroadcast B->BA over sfB\n"));
  CHKERRQ(VecView(ba, NULL));

  CHKERRQ(PetscSFCompose(sfA, sfB, &sfBA));
  CHKERRQ(PetscSFSetFromOptions(sfBA));
  CHKERRQ(PetscObjectSetName((PetscObject)sfBA, "(sfB o sfA)"));
  CHKERRQ(VecGetArrayRead(a, &arrayR));
  CHKERRQ(VecGetArray(ba, &arrayW));
  arrayW[0] = 11.0;             /* Not touched by bcast */
  CHKERRQ(PetscSFBcastBegin(sfBA, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sfBA, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE));
  CHKERRQ(VecRestoreArray(ba, &arrayW));
  CHKERRQ(VecRestoreArrayRead(a, &arrayR));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "\nBroadcast A->BA over sfBA (sfB o sfA)\n"));
  CHKERRQ(VecView(ba, NULL));

  CHKERRQ(VecDestroy(&ba));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&a));

  CHKERRQ(PetscSFView(sfA, NULL));
  CHKERRQ(PetscSFView(sfB, NULL));
  CHKERRQ(PetscSFView(sfBA, NULL));
  CHKERRQ(PetscSFDestroy(&sfA));
  CHKERRQ(PetscSFDestroy(&sfB));
  CHKERRQ(PetscSFDestroy(&sfBA));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     suffix: 1

   test:
     suffix: 2
     filter: grep -v "type" | grep -v "sort"
     args: -sparse_sfB

   test:
     suffix: 2_window
     filter: grep -v "type" | grep -v "sort"
     output_file: output/ex4_2.out
     args: -sparse_sfB -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor {{create dynamic allocate}}
     requires: defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

   # The nightly test suite with MPICH uses ch3:sock, which is broken when winsize == 0 in some of the processes
   test:
     suffix: 2_window_shared
     filter: grep -v "type" | grep -v "sort"
     output_file: output/ex4_2.out
     args: -sparse_sfB -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor shared
     requires: defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY) !defined(PETSC_HAVE_MPICH_NUMVERSION) defined(PETSC_HAVE_MPI_ONE_SIDED)

TEST*/
