
static char help[] = "Tests setup PCFIELDSPLIT with blocked IS.\n\n";
/*
 Contributed by Hoang Giang Bui, June 2017.
 */
#include <petscksp.h>

int main(int argc, char *argv[])
{
   Mat            A;
   PetscErrorCode ierr;
   KSP            ksp;
   PC             pc;
   PetscInt       Istart,Iend,local_m,local_n,i;
   PetscMPIInt    rank;
   PetscInt       method=2,mat_size=40,block_size=2,*A_indices=NULL,*B_indices=NULL,A_size=0,B_size=0;
   IS             A_IS, B_IS;

   ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
   ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);CHKERRMPI(ierr);

   ierr = PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-mat_size",&mat_size,PETSC_NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-method",&method,PETSC_NULL);CHKERRQ(ierr);
   ierr = PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-block_size",&block_size,PETSC_NULL);CHKERRQ(ierr);

   if (!rank) {
     ierr = PetscPrintf(PETSC_COMM_SELF,"  matrix size = %D, block size = %D\n",mat_size,block_size);CHKERRQ(ierr);
   }

   ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
   ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,mat_size,mat_size);CHKERRQ(ierr);
   ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
   ierr = MatSetUp(A);CHKERRQ(ierr);

   ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

   for (i = Istart; i < Iend; ++i) {
     ierr = MatSetValue(A,i,i,2,INSERT_VALUES);CHKERRQ(ierr);
     if (i < mat_size-1) {
       ierr = MatSetValue(A,i,i+1,-1,INSERT_VALUES);CHKERRQ(ierr);
     }
     if (i > 0) {
       ierr = MatSetValue(A,i,i-1,-1,INSERT_VALUES);CHKERRQ(ierr);
     }
   }

   ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

   ierr = MatGetLocalSize(A,&local_m,&local_n);CHKERRQ(ierr);

   /* Create Index Sets */
   if (rank == 0) {
     if (method > 1) {
       /* with method > 1, the fieldsplit B is set to zero */
       A_size = Iend-Istart;
       B_size = 0;
     } else {
       /* with method = 1, the fieldsplit A and B is equal. It is noticed that A_size, or N/4, must be divided by block_size */
       A_size = (Iend-Istart)/2;
       B_size = (Iend-Istart)/2;
     }
     ierr = PetscCalloc1(A_size,&A_indices);CHKERRQ(ierr);
     ierr = PetscCalloc1(B_size,&B_indices);CHKERRQ(ierr);
     for (i = 0; i < A_size; ++i) A_indices[i] = Istart + i;
     for (i = 0; i < B_size; ++i) B_indices[i] = Istart + i + A_size;
   } else if (rank == 1) {
     A_size = (Iend-Istart)/2;
     B_size = (Iend-Istart)/2;
     ierr = PetscCalloc1(A_size,&A_indices);CHKERRQ(ierr);
     ierr = PetscCalloc1(B_size,&B_indices);CHKERRQ(ierr);
     for (i = 0; i < A_size; ++i) A_indices[i] = Istart + i;
     for (i = 0; i < B_size; ++i) B_indices[i] = Istart + i + A_size;
   }

   ierr = ISCreateGeneral(PETSC_COMM_WORLD,A_size,A_indices,PETSC_OWN_POINTER,&A_IS);CHKERRQ(ierr);
   ierr = ISCreateGeneral(PETSC_COMM_WORLD,B_size,B_indices,PETSC_OWN_POINTER,&B_IS);CHKERRQ(ierr);
   ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]: A_size = %D, B_size = %D\n",rank,A_size,B_size);CHKERRQ(ierr);
   ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

   /* Solve the system */
   ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
   ierr = KSPSetType(ksp,KSPGMRES);CHKERRQ(ierr);
   ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

   /* Define the fieldsplit for the global matrix */
   ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
   ierr = PCSetType(pc,PCFIELDSPLIT);CHKERRQ(ierr);
   ierr = PCFieldSplitSetIS(pc,"a",A_IS);CHKERRQ(ierr);
   ierr = PCFieldSplitSetIS(pc,"b",B_IS);CHKERRQ(ierr);
   ierr = ISSetBlockSize(A_IS,block_size);CHKERRQ(ierr);
   ierr = ISSetBlockSize(B_IS,block_size);CHKERRQ(ierr);

   ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
   ierr = KSPSetUp(ksp);CHKERRQ(ierr);

   ierr = ISDestroy(&A_IS);CHKERRQ(ierr);
   ierr = ISDestroy(&B_IS);CHKERRQ(ierr);
   ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
   ierr = MatDestroy(&A);CHKERRQ(ierr);
   ierr = PetscFinalize();
   return ierr;
}


/*TEST

   test:
      nsize: 2
      args: -method 1

   test:
      suffix: 2
      nsize: 2
      args: -method 2

TEST*/
