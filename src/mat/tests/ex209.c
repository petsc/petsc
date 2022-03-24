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

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),NULL));

  /* Load matrix A */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  /* Create identity matrix B */
  CHKERRQ(MatGetLocalSize(A,&m,&n));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,m,m,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetType(B,MATAIJ));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));

  CHKERRQ(MatGetOwnershipRange(B,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    CHKERRQ(MatSetValues(B,1,&i,1,&i,&one,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* Compute AtA = A^T*B*A, B = identity matrix */
  CHKERRQ(MatPtAP(B,A,MAT_INITIAL_MATRIX,fill,&AtA));
  CHKERRQ(MatPtAP(B,A,MAT_REUSE_MATRIX,fill,&AtA));
  if (rank == 0) printf("C = A^T*B*A is done...\n");
  CHKERRQ(MatDestroy(&B));

  /* Compute C = A^T*A */
  CHKERRQ(MatTransposeMatMult(A,A,MAT_INITIAL_MATRIX,fill,&C));
  if (rank == 0) printf("C = A^T*A is done...\n");
  CHKERRQ(MatTransposeMatMult(A,A,MAT_REUSE_MATRIX,fill,&C));
  if (rank == 0) printf("REUSE C = A^T*A is done...\n");

  /* Compare C and AtA */
  CHKERRQ(MatMultEqual(C,AtA,20,&equal));
  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"A^T*A != At*A");
  CHKERRQ(MatDestroy(&AtA));

  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
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
