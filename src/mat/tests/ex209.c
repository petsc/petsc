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
  PetscErrorCode ierr;
  PetscReal      fill = 4.0;
  PetscMPIInt    rank,size;
  PetscBool      equal;
  PetscInt       i,m,n,rstart,rend;
  PetscScalar    one=1.0;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),NULL);CHKERRQ(ierr);

  /* Load matrix A */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Create identity matrix B */
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,m,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(B,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(B,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatSetValues(B,1,&i,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Compute AtA = A^T*B*A, B = identity matrix */
  ierr = MatPtAP(B,A,MAT_INITIAL_MATRIX,fill,&AtA);CHKERRQ(ierr);
  ierr = MatPtAP(B,A,MAT_REUSE_MATRIX,fill,&AtA);CHKERRQ(ierr);
  if (!rank) printf("C = A^T*B*A is done...\n");
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  /* Compute C = A^T*A */
  ierr = MatTransposeMatMult(A,A,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
  if (!rank) printf("C = A^T*A is done...\n");
  ierr = MatTransposeMatMult(A,A,MAT_REUSE_MATRIX,fill,&C);CHKERRQ(ierr);
  if (!rank) printf("REUSE C = A^T*A is done...\n");

  /* Compare C and AtA */
  ierr = MatMultEqual(C,AtA,20,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"A^T*A != At*A");
  ierr = MatDestroy(&AtA);CHKERRQ(ierr);

  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/arco1

   test:
      suffix: 2
      nsize: 4
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/arco1 -matptap_via nonscalable

TEST*/
