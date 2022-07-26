
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

   PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
   PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD,&rank));

   PetscCall(PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-mat_size",&mat_size,PETSC_NULL));
   PetscCall(PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-method",&method,PETSC_NULL));
   PetscCall(PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-block_size",&block_size,PETSC_NULL));

   if (rank == 0) {
     PetscCall(PetscPrintf(PETSC_COMM_SELF,"  matrix size = %" PetscInt_FMT ", block size = %" PetscInt_FMT "\n",mat_size,block_size));
   }

   PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
   PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,mat_size,mat_size));
   PetscCall(MatSetType(A,MATMPIAIJ));
   PetscCall(MatSetUp(A));

   PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));

   for (i = Istart; i < Iend; ++i) {
     PetscCall(MatSetValue(A,i,i,2,INSERT_VALUES));
     if (i < mat_size-1) {
       PetscCall(MatSetValue(A,i,i+1,-1,INSERT_VALUES));
     }
     if (i > 0) {
       PetscCall(MatSetValue(A,i,i-1,-1,INSERT_VALUES));
     }
   }

   PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

   PetscCall(MatGetLocalSize(A,&local_m,&local_n));

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
     PetscCall(PetscCalloc1(A_size,&A_indices));
     PetscCall(PetscCalloc1(B_size,&B_indices));
     for (i = 0; i < A_size; ++i) A_indices[i] = Istart + i;
     for (i = 0; i < B_size; ++i) B_indices[i] = Istart + i + A_size;
   } else if (rank == 1) {
     A_size = (Iend-Istart)/2;
     B_size = (Iend-Istart)/2;
     PetscCall(PetscCalloc1(A_size,&A_indices));
     PetscCall(PetscCalloc1(B_size,&B_indices));
     for (i = 0; i < A_size; ++i) A_indices[i] = Istart + i;
     for (i = 0; i < B_size; ++i) B_indices[i] = Istart + i + A_size;
   }

   PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,A_size,A_indices,PETSC_OWN_POINTER,&A_IS));
   PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,B_size,B_indices,PETSC_OWN_POINTER,&B_IS));
   PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]: A_size = %" PetscInt_FMT ", B_size = %" PetscInt_FMT "\n",rank,A_size,B_size));
   PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));

   /* Solve the system */
   PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
   PetscCall(KSPSetType(ksp,KSPGMRES));
   PetscCall(KSPSetOperators(ksp,A,A));

   /* Define the fieldsplit for the global matrix */
   PetscCall(KSPGetPC(ksp,&pc));
   PetscCall(PCSetType(pc,PCFIELDSPLIT));
   PetscCall(PCFieldSplitSetIS(pc,"a",A_IS));
   PetscCall(PCFieldSplitSetIS(pc,"b",B_IS));
   PetscCall(ISSetBlockSize(A_IS,block_size));
   PetscCall(ISSetBlockSize(B_IS,block_size));

   PetscCall(KSPSetFromOptions(ksp));
   PetscCall(KSPSetUp(ksp));

   PetscCall(ISDestroy(&A_IS));
   PetscCall(ISDestroy(&B_IS));
   PetscCall(KSPDestroy(&ksp));
   PetscCall(MatDestroy(&A));
   PetscCall(PetscFinalize());
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
