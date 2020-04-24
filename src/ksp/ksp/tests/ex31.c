
static char help[] = "Test partition. Reads a PETSc matrix and vector from a file and solves a linear system.\n\
This   Input parameters include\n\
  -f <input_file> : file to load \n\
  -partition -mat_partitioning_view \n\\n";

/*T
   Concepts: KSP^solving a linear system
   Processors: n
T*/


#include <petscksp.h>

int main(int argc,char **args)
{
  KSP            ksp;             /* linear solver context */
  Mat            A;               /* matrix */
  Vec            x,b,u;           /* approx solution, RHS, exact solution */
  PetscViewer    fd;              /* viewer */
  char           file[PETSC_MAX_PATH_LEN];     /* input file name */
  PetscBool      flg,partition=PETSC_FALSE,displayIS=PETSC_FALSE,displayMat=PETSC_FALSE;
  PetscErrorCode ierr;
  PetscInt       its,m,n;
  PetscReal      norm;
  PetscMPIInt    size,rank;
  PetscScalar    one = 1.0;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(NULL,NULL,"-partition",&partition,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-displayIS",&displayIS,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-displayMat",&displayMat,NULL);CHKERRQ(ierr);

  /* Determine file from which we read the matrix.*/
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate binary file with the -f option");

  /* - - - - - - - - - - - - - - - - - - - - - - - -
                           Load system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  if (m != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%D, %D)", m, n);

  /* Create rhs vector of all ones */
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = VecSet(b,one);CHKERRQ(ierr);

  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u);CHKERRQ(ierr);
  ierr = VecSet(x,0.0);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - -
                      Test partition
  - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (partition) {
    MatPartitioning mpart;
    IS              mis,nis,is;
    PetscInt        *count;
    Mat             BB;

    if (displayMat) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Before partitioning/reordering, A:\n");CHKERRQ(ierr);
      ierr = MatView(A,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    }

    ierr = PetscMalloc1(size,&count);CHKERRQ(ierr);
    ierr = MatPartitioningCreate(PETSC_COMM_WORLD, &mpart);CHKERRQ(ierr);
    ierr = MatPartitioningSetAdjacency(mpart, A);CHKERRQ(ierr);
    /* ierr = MatPartitioningSetVertexWeights(mpart, weight);CHKERRQ(ierr); */
    ierr = MatPartitioningSetFromOptions(mpart);CHKERRQ(ierr);
    ierr = MatPartitioningApply(mpart, &mis);CHKERRQ(ierr);
    ierr = MatPartitioningDestroy(&mpart);CHKERRQ(ierr);
    if (displayIS) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"mis, new processor assignment:\n");CHKERRQ(ierr);
      ierr = ISView(mis,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    ierr = ISPartitioningToNumbering(mis,&nis);CHKERRQ(ierr);
    if (displayIS) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"nis:\n");CHKERRQ(ierr);
      ierr = ISView(nis,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    ierr = ISPartitioningCount(mis,size,count);CHKERRQ(ierr);
    ierr = ISDestroy(&mis);CHKERRQ(ierr);
    if (displayIS && !rank) {
      PetscInt i;
      ierr = PetscPrintf(PETSC_COMM_SELF,"[ %d ] count:\n",rank);CHKERRQ(ierr);
      for (i=0; i<size; i++) {ierr = PetscPrintf(PETSC_COMM_WORLD," %d",count[i]);CHKERRQ(ierr);}
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
    }

    ierr = ISInvertPermutation(nis, count[rank], &is);CHKERRQ(ierr);
    ierr = PetscFree(count);CHKERRQ(ierr);
    ierr = ISDestroy(&nis);CHKERRQ(ierr);
    ierr = ISSort(is);CHKERRQ(ierr);
    if (displayIS) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"inverse of nis - maps new local rows to old global rows:\n");CHKERRQ(ierr);
      ierr = ISView(is,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    ierr = MatCreateSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&BB);CHKERRQ(ierr);
    if (displayMat) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"After partitioning/reordering, A:\n");CHKERRQ(ierr);
      ierr = MatView(BB,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    }

    /* need to move the vector also */
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    A    = BB;
  }

  /* Create linear solver; set operators; set runtime options.*/
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - -
                           Solve system
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  /* Check error */
  ierr = MatMult(A,x,u);CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",(double)norm);CHKERRQ(ierr);
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL, "-ksp_reason", &flg,NULL);CHKERRQ(ierr);
  if (flg) {
    KSPConvergedReason reason;
    ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"KSPConvergedReason: %D\n", reason);
  }

  /* Free work space.*/
  ierr = MatDestroy(&A);CHKERRQ(ierr); ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr); ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -f ${DATAFILESPATH}/matrices/small -partition -mat_partitioning_type parmetis 
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES) parmetis
      output_file: output/ex31.out
      nsize: 3
 
TEST*/
