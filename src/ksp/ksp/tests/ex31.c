
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

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-partition",&partition,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-displayIS",&displayIS,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-displayMat",&displayMat,NULL));

  /* Determine file from which we read the matrix.*/
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate binary file with the -f option");

  /* - - - - - - - - - - - - - - - - - - - - - - - -
                           Load system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatLoad(A,fd));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(MatGetLocalSize(A,&m,&n));
  PetscCheck(m == n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%D, %D)", m, n);

  /* Create rhs vector of all ones */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&b));
  PetscCall(VecSetSizes(b,m,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(b));
  PetscCall(VecSet(b,one));

  PetscCall(VecDuplicate(b,&x));
  PetscCall(VecDuplicate(b,&u));
  PetscCall(VecSet(x,0.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - -
                      Test partition
  - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (partition) {
    MatPartitioning mpart;
    IS              mis,nis,is;
    PetscInt        *count;
    Mat             BB;

    if (displayMat) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Before partitioning/reordering, A:\n"));
      PetscCall(MatView(A,PETSC_VIEWER_DRAW_WORLD));
    }

    PetscCall(PetscMalloc1(size,&count));
    PetscCall(MatPartitioningCreate(PETSC_COMM_WORLD, &mpart));
    PetscCall(MatPartitioningSetAdjacency(mpart, A));
    /* PetscCall(MatPartitioningSetVertexWeights(mpart, weight)); */
    PetscCall(MatPartitioningSetFromOptions(mpart));
    PetscCall(MatPartitioningApply(mpart, &mis));
    PetscCall(MatPartitioningDestroy(&mpart));
    if (displayIS) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"mis, new processor assignment:\n"));
      PetscCall(ISView(mis,PETSC_VIEWER_STDOUT_WORLD));
    }

    PetscCall(ISPartitioningToNumbering(mis,&nis));
    if (displayIS) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"nis:\n"));
      PetscCall(ISView(nis,PETSC_VIEWER_STDOUT_WORLD));
    }

    PetscCall(ISPartitioningCount(mis,size,count));
    PetscCall(ISDestroy(&mis));
    if (displayIS && rank == 0) {
      PetscInt i;
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"[ %d ] count:\n",rank));
      for (i=0; i<size; i++) PetscCall(PetscPrintf(PETSC_COMM_WORLD," %d",count[i]));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n"));
    }

    PetscCall(ISInvertPermutation(nis, count[rank], &is));
    PetscCall(PetscFree(count));
    PetscCall(ISDestroy(&nis));
    PetscCall(ISSort(is));
    if (displayIS) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"inverse of nis - maps new local rows to old global rows:\n"));
      PetscCall(ISView(is,PETSC_VIEWER_STDOUT_WORLD));
    }

    PetscCall(MatCreateSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&BB));
    if (displayMat) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"After partitioning/reordering, A:\n"));
      PetscCall(MatView(BB,PETSC_VIEWER_DRAW_WORLD));
    }

    /* need to move the vector also */
    PetscCall(ISDestroy(&is));
    PetscCall(MatDestroy(&A));
    A    = BB;
  }

  /* Create linear solver; set operators; set runtime options.*/
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - -
                           Solve system
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPSolve(ksp,b,x));
  PetscCall(KSPGetIterationNumber(ksp,&its));

  /* Check error */
  PetscCall(MatMult(A,x,u));
  PetscCall(VecAXPY(u,-1.0,b));
  PetscCall(VecNorm(u,NORM_2,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",(double)norm));
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL, "-ksp_reason", &flg,NULL));
  if (flg) {
    KSPConvergedReason reason;
    PetscCall(KSPGetConvergedReason(ksp,&reason));
    PetscPrintf(PETSC_COMM_WORLD,"KSPConvergedReason: %D\n", reason);
  }

  /* Free work space.*/
  PetscCall(MatDestroy(&A)); PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u)); PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args: -f ${DATAFILESPATH}/matrices/small -partition -mat_partitioning_type parmetis
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES) parmetis
      output_file: output/ex31.out
      nsize: 3

TEST*/
