static char help[] = "Test MatTransposeMatMult() \n\n";

/* Example:
  mpiexec -n 8 ./ex209 -f <inputfile> (e.g., inputfile=ceres_solver_iteration_001_A.petsc)
*/

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,C,AtA,B;
  PetscViewer    fd;
  char           file[PETSC_MAX_PATH_LEN];
  PetscReal      fill = 4.0;
  PetscMPIInt    rank,size;
  PetscBool      equal;
  PetscInt       i,m,n,rstart,rend;
  PetscScalar    one=1.0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),NULL));

  /* Load matrix A */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetType(A,MATAIJ));
  PetscCall(MatLoad(A,fd));
  PetscCall(PetscViewerDestroy(&fd));

  /* Create identity matrix B */
  PetscCall(MatGetLocalSize(A,&m,&n));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,m,m,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(B,MATAIJ));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));

  PetscCall(MatGetOwnershipRange(B,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    PetscCall(MatSetValues(B,1,&i,1,&i,&one,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* Compute AtA = A^T*B*A, B = identity matrix */
  PetscCall(MatPtAP(B,A,MAT_INITIAL_MATRIX,fill,&AtA));
  PetscCall(MatPtAP(B,A,MAT_REUSE_MATRIX,fill,&AtA));
  if (rank == 0) printf("C = A^T*B*A is done...\n");
  PetscCall(MatDestroy(&B));

  /* Compute C = A^T*A */
  PetscCall(MatTransposeMatMult(A,A,MAT_INITIAL_MATRIX,fill,&C));
  if (rank == 0) printf("C = A^T*A is done...\n");
  PetscCall(MatTransposeMatMult(A,A,MAT_REUSE_MATRIX,fill,&C));
  if (rank == 0) printf("REUSE C = A^T*A is done...\n");

  /* Compare C and AtA */
  PetscCall(MatMultEqual(C,AtA,20,&equal));
  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"A^T*A != At*A");
  PetscCall(MatDestroy(&AtA));

  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/arco1

   test:
      suffix: 2
      nsize: 4
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/arco1 -matptap_via nonscalable

TEST*/
