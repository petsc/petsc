static char help[]= "Test PetscSFFCompose when the ilocal array is not the identity\n\n";

#include <petscsf.h>

int main(int argc, char **argv)
{
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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD, PETSC_ERR_USER, "Only coded for one MPI process");

  PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sfA));
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sfB));
  PetscCall(PetscSFSetFromOptions(sfA));
  PetscCall(PetscSFSetFromOptions(sfB));

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-sparse_sfB",&flag,NULL));

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
  PetscCall(PetscMalloc1(nleavesA, &ilocalA));
  PetscCall(PetscMalloc1(nleavesA, &iremoteA));
  PetscCall(PetscMalloc1(nleavesB, &ilocalB));
  PetscCall(PetscMalloc1(nleavesB, &iremoteB));

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

  PetscCall(PetscSFSetGraph(sfA, nrootsA, nleavesA, ilocalA, PETSC_OWN_POINTER, iremoteA, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetGraph(sfB, nrootsB, nleavesB, ilocalB, PETSC_OWN_POINTER, iremoteB, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetUp(sfA));
  PetscCall(PetscSFSetUp(sfB));
  PetscCall(PetscObjectSetName((PetscObject)sfA, "sfA"));
  PetscCall(PetscObjectSetName((PetscObject)sfB, "sfB"));

  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, nrootsA, &a));
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, nleavesA, &b));
  PetscCall(PetscSFGetLeafRange(sfB, NULL, &maxleafB));
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, maxleafB+1, &ba));
  PetscCall(VecGetArray(a, &arrayW));
  for (i = 0; i < nrootsA; i++) {
    arrayW[i] = (PetscScalar)i;
  }
  PetscCall(VecRestoreArray(a, &arrayW));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial Vec A\n"));
  PetscCall(VecView(a, NULL));
  PetscCall(VecGetArrayRead(a, &arrayR));
  PetscCall(VecGetArray(b, &arrayW));

  PetscCall(PetscSFBcastBegin(sfA, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sfA, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE));
  PetscCall(VecRestoreArray(b, &arrayW));
  PetscCall(VecRestoreArrayRead(a, &arrayR));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nBroadcast A->B over sfA\n"));
  PetscCall(VecView(b, NULL));

  PetscCall(VecGetArrayRead(b, &arrayR));
  PetscCall(VecGetArray(ba, &arrayW));
  arrayW[0] = 10.0;             /* Not touched by bcast */
  PetscCall(PetscSFBcastBegin(sfB, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sfB, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE));
  PetscCall(VecRestoreArrayRead(b, &arrayR));
  PetscCall(VecRestoreArray(ba, &arrayW));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nBroadcast B->BA over sfB\n"));
  PetscCall(VecView(ba, NULL));

  PetscCall(PetscSFCompose(sfA, sfB, &sfBA));
  PetscCall(PetscSFSetFromOptions(sfBA));
  PetscCall(PetscObjectSetName((PetscObject)sfBA, "(sfB o sfA)"));
  PetscCall(VecGetArrayRead(a, &arrayR));
  PetscCall(VecGetArray(ba, &arrayW));
  arrayW[0] = 11.0;             /* Not touched by bcast */
  PetscCall(PetscSFBcastBegin(sfBA, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sfBA, MPIU_SCALAR, arrayR, arrayW,MPI_REPLACE));
  PetscCall(VecRestoreArray(ba, &arrayW));
  PetscCall(VecRestoreArrayRead(a, &arrayR));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nBroadcast A->BA over sfBA (sfB o sfA)\n"));
  PetscCall(VecView(ba, NULL));

  PetscCall(VecDestroy(&ba));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&a));

  PetscCall(PetscSFView(sfA, NULL));
  PetscCall(PetscSFView(sfB, NULL));
  PetscCall(PetscSFView(sfBA, NULL));
  PetscCall(PetscSFDestroy(&sfA));
  PetscCall(PetscSFDestroy(&sfB));
  PetscCall(PetscSFDestroy(&sfBA));

  PetscCall(PetscFinalize());
  return 0;
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
