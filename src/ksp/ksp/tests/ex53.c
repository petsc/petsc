
static char help[] = "Tests setup PCFIELDSPLIT with blocked IS.\n\n";
/*
 Contributed by Hoang Giang Bui, June 2017.
 */
#include <petscksp.h>

int main(int argc, char *argv[])
{
   Mat            A;
   KSP            ksp;
   PC             pc;
   PetscInt       Istart,Iend,local_m,local_n,i;
   PetscMPIInt    rank;
   PetscInt       method=2,mat_size=40,block_size=2,*A_indices=NULL,*B_indices=NULL,A_size=0,B_size=0;
   IS             A_IS, B_IS;

   CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
   CHKERRMPI(MPI_Comm_rank(MPI_COMM_WORLD,&rank));

   CHKERRQ(PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-mat_size",&mat_size,PETSC_NULL));
   CHKERRQ(PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-method",&method,PETSC_NULL));
   CHKERRQ(PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-block_size",&block_size,PETSC_NULL));

   if (rank == 0) {
     CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"  matrix size = %D, block size = %D\n",mat_size,block_size));
   }

   CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
   CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,mat_size,mat_size));
   CHKERRQ(MatSetType(A,MATMPIAIJ));
   CHKERRQ(MatSetUp(A));

   CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));

   for (i = Istart; i < Iend; ++i) {
     CHKERRQ(MatSetValue(A,i,i,2,INSERT_VALUES));
     if (i < mat_size-1) {
       CHKERRQ(MatSetValue(A,i,i+1,-1,INSERT_VALUES));
     }
     if (i > 0) {
       CHKERRQ(MatSetValue(A,i,i-1,-1,INSERT_VALUES));
     }
   }

   CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
   CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

   CHKERRQ(MatGetLocalSize(A,&local_m,&local_n));

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
     CHKERRQ(PetscCalloc1(A_size,&A_indices));
     CHKERRQ(PetscCalloc1(B_size,&B_indices));
     for (i = 0; i < A_size; ++i) A_indices[i] = Istart + i;
     for (i = 0; i < B_size; ++i) B_indices[i] = Istart + i + A_size;
   } else if (rank == 1) {
     A_size = (Iend-Istart)/2;
     B_size = (Iend-Istart)/2;
     CHKERRQ(PetscCalloc1(A_size,&A_indices));
     CHKERRQ(PetscCalloc1(B_size,&B_indices));
     for (i = 0; i < A_size; ++i) A_indices[i] = Istart + i;
     for (i = 0; i < B_size; ++i) B_indices[i] = Istart + i + A_size;
   }

   CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,A_size,A_indices,PETSC_OWN_POINTER,&A_IS));
   CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,B_size,B_indices,PETSC_OWN_POINTER,&B_IS));
   CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]: A_size = %D, B_size = %D\n",rank,A_size,B_size));
   CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));

   /* Solve the system */
   CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
   CHKERRQ(KSPSetType(ksp,KSPGMRES));
   CHKERRQ(KSPSetOperators(ksp,A,A));

   /* Define the fieldsplit for the global matrix */
   CHKERRQ(KSPGetPC(ksp,&pc));
   CHKERRQ(PCSetType(pc,PCFIELDSPLIT));
   CHKERRQ(PCFieldSplitSetIS(pc,"a",A_IS));
   CHKERRQ(PCFieldSplitSetIS(pc,"b",B_IS));
   CHKERRQ(ISSetBlockSize(A_IS,block_size));
   CHKERRQ(ISSetBlockSize(B_IS,block_size));

   CHKERRQ(KSPSetFromOptions(ksp));
   CHKERRQ(KSPSetUp(ksp));

   CHKERRQ(ISDestroy(&A_IS));
   CHKERRQ(ISDestroy(&B_IS));
   CHKERRQ(KSPDestroy(&ksp));
   CHKERRQ(MatDestroy(&A));
   CHKERRQ(PetscFinalize());
   return 0;
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
