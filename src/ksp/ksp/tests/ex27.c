
static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Test MatMatSolve().  Input parameters include\n\
  -f <input_file> : file to load \n\n";

/*
  Usage:
     ex27 -f0 <mat_binaryfile>
*/

#include <petscksp.h>
extern PetscErrorCode PCShellApply_Matinv(PC,Vec,Vec);

int main(int argc,char **args)
{
  KSP            ksp;
  Mat            A,B,F,X;
  Vec            x,b,u;          /* approx solution, RHS, exact solution */
  PetscViewer    fd;             /* viewer */
  char           file[1][PETSC_MAX_PATH_LEN];     /* input file name */
  PetscBool      flg;
  PetscErrorCode ierr;
  PetscInt       M,N,i,its;
  PetscReal      norm;
  PetscScalar    val=1.0;
  PetscMPIInt    size;
  PC             pc;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  /* Read matrix and right-hand-side vector */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file[0],sizeof(file[0]),&flg));
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");

  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b));
  CHKERRQ(VecLoad(b,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  /*
     If the loaded matrix is larger than the vector (due to being padded
     to match the block size of the system), then create a new padded vector.
  */
  {
    PetscInt    m,n,j,mvec,start,end,indx;
    Vec         tmp;
    PetscScalar *bold;

    /* Create a new vector b by padding the old one */
    CHKERRQ(MatGetLocalSize(A,&m,&n));
    PetscCheckFalse(m != n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%D, %D)", m, n);
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&tmp));
    CHKERRQ(VecSetSizes(tmp,m,PETSC_DECIDE));
    CHKERRQ(VecSetFromOptions(tmp));
    CHKERRQ(VecGetOwnershipRange(b,&start,&end));
    CHKERRQ(VecGetLocalSize(b,&mvec));
    CHKERRQ(VecGetArray(b,&bold));
    for (j=0; j<mvec; j++) {
      indx = start+j;
      CHKERRQ(VecSetValues(tmp,1,&indx,bold+j,INSERT_VALUES));
    }
    CHKERRQ(VecRestoreArray(b,&bold));
    CHKERRQ(VecDestroy(&b));
    CHKERRQ(VecAssemblyBegin(tmp));
    CHKERRQ(VecAssemblyEnd(tmp));
    b    = tmp;
  }
  CHKERRQ(VecDuplicate(b,&x));
  CHKERRQ(VecDuplicate(b,&u));
  CHKERRQ(VecSet(x,0.0));

  /* Create dense matric B and X. Set B as an identity matrix */
  CHKERRQ(MatGetSize(A,&M,&N));
  CHKERRQ(MatCreate(MPI_COMM_SELF,&B));
  CHKERRQ(MatSetSizes(B,M,N,M,N));
  CHKERRQ(MatSetType(B,MATSEQDENSE));
  CHKERRQ(MatSeqDenseSetPreallocation(B,NULL));
  for (i=0; i<M; i++) {
    CHKERRQ(MatSetValues(B,1,&i,1,&i,&val,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&X));

  /* Compute X=inv(A) by MatMatSolve() */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCLU));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(PCFactorGetMatrix(pc,&F));
  CHKERRQ(MatMatSolve(F,B,X));
  CHKERRQ(MatDestroy(&B));

  /* Now, set X=inv(A) as a preconditioner */
  CHKERRQ(PCSetType(pc,PCSHELL));
  CHKERRQ(PCShellSetContext(pc,X));
  CHKERRQ(PCShellSetApply(pc,PCShellApply_Matinv));
  CHKERRQ(KSPSetFromOptions(ksp));

  /* Solve preconditioned system A*x = b */
  CHKERRQ(KSPSolve(ksp,b,x));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));

  /* Check error */
  CHKERRQ(MatMult(A,x,u));
  CHKERRQ(VecAXPY(u,-1.0,b));
  CHKERRQ(VecNorm(u,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",(double)norm));

  /* Free work space.  */
  CHKERRQ(MatDestroy(&X));
  CHKERRQ(MatDestroy(&A)); CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&u)); CHKERRQ(VecDestroy(&x));
  CHKERRQ(KSPDestroy(&ksp));
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode PCShellApply_Matinv(PC pc,Vec xin,Vec xout)
{
  Mat            X;

  PetscFunctionBeginUser;
  CHKERRQ(PCShellGetContext(pc,&X));
  CHKERRQ(MatMult(X,xin,xout));
  PetscFunctionReturn(0);
}

/*TEST

    test:
      args: -f ${DATAFILESPATH}/matrices/small
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex27.out

TEST*/
