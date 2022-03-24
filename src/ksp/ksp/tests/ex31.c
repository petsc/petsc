
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
  PetscInt       its,m,n;
  PetscReal      norm;
  PetscMPIInt    size,rank;
  PetscScalar    one = 1.0;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-partition",&partition,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-displayIS",&displayIS,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-displayMat",&displayMat,NULL));

  /* Determine file from which we read the matrix.*/
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate binary file with the -f option");

  /* - - - - - - - - - - - - - - - - - - - - - - - -
                           Load system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));
  CHKERRQ(MatGetLocalSize(A,&m,&n));
  PetscCheckFalse(m != n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%D, %D)", m, n);

  /* Create rhs vector of all ones */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b));
  CHKERRQ(VecSetSizes(b,m,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(b));
  CHKERRQ(VecSet(b,one));

  CHKERRQ(VecDuplicate(b,&x));
  CHKERRQ(VecDuplicate(b,&u));
  CHKERRQ(VecSet(x,0.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - -
                      Test partition
  - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (partition) {
    MatPartitioning mpart;
    IS              mis,nis,is;
    PetscInt        *count;
    Mat             BB;

    if (displayMat) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Before partitioning/reordering, A:\n"));
      CHKERRQ(MatView(A,PETSC_VIEWER_DRAW_WORLD));
    }

    CHKERRQ(PetscMalloc1(size,&count));
    CHKERRQ(MatPartitioningCreate(PETSC_COMM_WORLD, &mpart));
    CHKERRQ(MatPartitioningSetAdjacency(mpart, A));
    /* CHKERRQ(MatPartitioningSetVertexWeights(mpart, weight)); */
    CHKERRQ(MatPartitioningSetFromOptions(mpart));
    CHKERRQ(MatPartitioningApply(mpart, &mis));
    CHKERRQ(MatPartitioningDestroy(&mpart));
    if (displayIS) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"mis, new processor assignment:\n"));
      CHKERRQ(ISView(mis,PETSC_VIEWER_STDOUT_WORLD));
    }

    CHKERRQ(ISPartitioningToNumbering(mis,&nis));
    if (displayIS) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"nis:\n"));
      CHKERRQ(ISView(nis,PETSC_VIEWER_STDOUT_WORLD));
    }

    CHKERRQ(ISPartitioningCount(mis,size,count));
    CHKERRQ(ISDestroy(&mis));
    if (displayIS && rank == 0) {
      PetscInt i;
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[ %d ] count:\n",rank));
      for (i=0; i<size; i++) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," %d",count[i]));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
    }

    CHKERRQ(ISInvertPermutation(nis, count[rank], &is));
    CHKERRQ(PetscFree(count));
    CHKERRQ(ISDestroy(&nis));
    CHKERRQ(ISSort(is));
    if (displayIS) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"inverse of nis - maps new local rows to old global rows:\n"));
      CHKERRQ(ISView(is,PETSC_VIEWER_STDOUT_WORLD));
    }

    CHKERRQ(MatCreateSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&BB));
    if (displayMat) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After partitioning/reordering, A:\n"));
      CHKERRQ(MatView(BB,PETSC_VIEWER_DRAW_WORLD));
    }

    /* need to move the vector also */
    CHKERRQ(ISDestroy(&is));
    CHKERRQ(MatDestroy(&A));
    A    = BB;
  }

  /* Create linear solver; set operators; set runtime options.*/
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - -
                           Solve system
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(KSPSolve(ksp,b,x));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));

  /* Check error */
  CHKERRQ(MatMult(A,x,u));
  CHKERRQ(VecAXPY(u,-1.0,b));
  CHKERRQ(VecNorm(u,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",(double)norm));
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-ksp_reason", &flg,NULL));
  if (flg) {
    KSPConvergedReason reason;
    CHKERRQ(KSPGetConvergedReason(ksp,&reason));
    PetscPrintf(PETSC_COMM_WORLD,"KSPConvergedReason: %D\n", reason);
  }

  /* Free work space.*/
  CHKERRQ(MatDestroy(&A)); CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&u)); CHKERRQ(VecDestroy(&x));
  CHKERRQ(KSPDestroy(&ksp));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args: -f ${DATAFILESPATH}/matrices/small -partition -mat_partitioning_type parmetis
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES) parmetis
      output_file: output/ex31.out
      nsize: 3

TEST*/
