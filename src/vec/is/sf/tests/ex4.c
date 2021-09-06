static char help[]= "Test PetscSFFCompose when the ilocal array is not the identity\n\n";

#include <petsc.h>
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  if (size != 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Only coded for one MPI process");

  ierr = PetscSFCreate(PETSC_COMM_WORLD, &sfA);CHKERRQ(ierr);
  ierr = PetscSFCreate(PETSC_COMM_WORLD, &sfB);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sfA);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sfB);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(NULL,NULL,"-sparse_sfB",&flag,NULL);CHKERRQ(ierr);

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
  ierr = PetscMalloc1(nleavesA, &ilocalA);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleavesA, &iremoteA);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleavesB, &ilocalB);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleavesB, &iremoteB);CHKERRQ(ierr);

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

  ierr = PetscSFSetGraph(sfA, nrootsA, nleavesA, ilocalA, PETSC_OWN_POINTER, iremoteA, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sfB, nrootsB, nleavesB, ilocalB, PETSC_OWN_POINTER, iremoteB, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sfA);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sfB);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)sfA, "sfA");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)sfB, "sfB");CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_WORLD, nrootsA, &a);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_WORLD, nleavesA, &b);CHKERRQ(ierr);
  ierr = PetscSFGetLeafRange(sfB, NULL, &maxleafB);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_WORLD, maxleafB+1, &ba);CHKERRQ(ierr);
  ierr = VecGetArray(a, &arrayW);CHKERRQ(ierr);
  for (i = 0; i < nrootsA; i++) {
    arrayW[i] = (PetscScalar)i;
  }
  ierr = VecRestoreArray(a, &arrayW);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial Vec A\n");CHKERRQ(ierr);
  ierr = VecView(a, NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(a, &arrayR);CHKERRQ(ierr);
  ierr = VecGetArray(b, &arrayW);CHKERRQ(ierr);

  ierr = PetscSFBcastBegin(sfA, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sfA, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE);CHKERRQ(ierr);
  ierr = VecRestoreArray(b, &arrayW);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(a, &arrayR);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\nBroadcast A->B over sfA\n");CHKERRQ(ierr);
  ierr = VecView(b, NULL);CHKERRQ(ierr);

  ierr = VecGetArrayRead(b, &arrayR);CHKERRQ(ierr);
  ierr = VecGetArray(ba, &arrayW);CHKERRQ(ierr);
  arrayW[0] = 10.0;             /* Not touched by bcast */
  ierr = PetscSFBcastBegin(sfB, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sfB, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(b, &arrayR);CHKERRQ(ierr);
  ierr = VecRestoreArray(ba, &arrayW);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "\nBroadcast B->BA over sfB\n");CHKERRQ(ierr);
  ierr = VecView(ba, NULL);CHKERRQ(ierr);

  ierr = PetscSFCompose(sfA, sfB, &sfBA);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sfBA);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)sfBA, "(sfB o sfA)");CHKERRQ(ierr);
  ierr = VecGetArrayRead(a, &arrayR);CHKERRQ(ierr);
  ierr = VecGetArray(ba, &arrayW);CHKERRQ(ierr);
  arrayW[0] = 11.0;             /* Not touched by bcast */
  ierr = PetscSFBcastBegin(sfBA, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sfBA, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE);CHKERRQ(ierr);
  ierr = VecRestoreArray(ba, &arrayW);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(a, &arrayR);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\nBroadcast A->BA over sfBA (sfB o sfA)\n");CHKERRQ(ierr);
  ierr = VecView(ba, NULL);CHKERRQ(ierr);

  ierr = VecDestroy(&ba);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&a);CHKERRQ(ierr);

  ierr = PetscSFView(sfA, NULL);CHKERRQ(ierr);
  ierr = PetscSFView(sfB, NULL);CHKERRQ(ierr);
  ierr = PetscSFView(sfBA, NULL);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfA);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfB);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfBA);CHKERRQ(ierr);

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
