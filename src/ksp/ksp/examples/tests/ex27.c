
static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Test MatMatSolve().  Input parameters include\n\
  -f <input_file> : file to load \n\n";

/*
  Usage:
     ex27 -f0 <mat_binaryfile>  
*/

#include "petscksp.h"
EXTERN PetscErrorCode PCShellApply_Matinv(PC,Vec,Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  KSP            ksp;             
  Mat            A,B,F,X;       
  Vec            x,b,u;          /* approx solution, RHS, exact solution */
  PetscViewer    fd;             /* viewer */
  char           file[1][PETSC_MAX_PATH_LEN];     /* input file name */
  PetscTruth     flg;
  PetscErrorCode ierr;
  PetscInt       M,N,i,its;
  PetscReal      norm;
  PetscScalar    val=1.0;
  PetscMPIInt    size; 
  PC             pc;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_ERR_SUP,"This is a uniprocessor example only!");

  /* Read matrix and right-hand-side vector */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file[0],PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
  if (!flg) {
    SETERRQ(PETSC_ERR_USER,"Must indicate binary file with the -f option");   
  }

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatLoad(fd,MATAIJ,&A);CHKERRQ(ierr);
  ierr = PetscExceptionTry1(VecLoad(fd,PETSC_NULL,&b),PETSC_ERR_FILE_READ);     
  if (PetscExceptionCaught(ierr,PETSC_ERR_FILE_UNEXPECTED) || PetscExceptionCaught(ierr,PETSC_ERR_FILE_READ)) { 
    /* if file contains no RHS, then use a vector of all ones */
    PetscInt    m;
    PetscScalar one = 1.0;
    ierr = PetscInfo(0,"Using vector of ones for RHS\n");CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&m,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
    ierr = VecSetSizes(b,m,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(b);CHKERRQ(ierr);
    ierr = VecSet(b,one);CHKERRQ(ierr);
  } else CHKERRQ(ierr); 
  ierr = PetscViewerDestroy(fd);CHKERRQ(ierr); 

  /* 
     If the loaded matrix is larger than the vector (due to being padded 
     to match the block size of the system), then create a new padded vector.
  */
  { 
    PetscInt    m,n,j,mvec,start,end,indx;
    Vec         tmp;
    PetscScalar *bold;

    /* Create a new vector b by padding the old one */
    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
    if (m != n) {
      SETERRQ2(PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%d, %d)", m, n);
    }
    ierr = VecCreate(PETSC_COMM_WORLD,&tmp);CHKERRQ(ierr);
    ierr = VecSetSizes(tmp,m,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(tmp);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(b,&start,&end);CHKERRQ(ierr);
    ierr = VecGetLocalSize(b,&mvec);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bold);CHKERRQ(ierr);
    for (j=0; j<mvec; j++) {
      indx = start+j;
      ierr  = VecSetValues(tmp,1,&indx,bold+j,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(b,&bold);CHKERRQ(ierr);
    ierr = VecDestroy(b);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(tmp);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tmp);CHKERRQ(ierr);
    b = tmp;
  }
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u);CHKERRQ(ierr);
  ierr = VecSet(x,0.0);CHKERRQ(ierr);

  /* Create dense matric B and X. Set B as an identity matrix */
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatCreate(MPI_COMM_SELF,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,M,N,M,N);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(B,PETSC_NULL);CHKERRQ(ierr);
  for (i=0; i<M; i++){
    ierr = MatSetValues(B,1,&i,1,&i,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&X);CHKERRQ(ierr);
  
  /* Compute X=inv(A) by MatMatSolve() */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);
  ierr = MatMatSolve(F,B,X);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);

  /* Now, set X=inv(A) as a preconditioner */
  ierr = PCSetType(pc,PCSHELL);CHKERRQ(ierr);
  ierr = PCShellSetContext(pc,(void *)X);CHKERRQ(ierr);
  ierr = PCShellSetApply(pc,PCShellApply_Matinv);CHKERRQ(ierr); 
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* Sove preconditioned system A*x = b */
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);   
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  /* Check error */
  ierr = MatMult(A,x,u);CHKERRQ(ierr);   
  ierr = VecAXPY(u,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %A\n",norm);CHKERRQ(ierr);    
  
  /* Free work space.  */
  ierr = MatDestroy(X);CHKERRQ(ierr); 
  ierr = MatDestroy(A);CHKERRQ(ierr); ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr); ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = KSPDestroy(ksp);CHKERRQ(ierr); 
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

PetscErrorCode PCShellApply_Matinv(PC pc,Vec xin,Vec xout)
{
  PetscErrorCode ierr;
  Mat            X;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void**)&X);CHKERRQ(ierr);
  ierr = MatMult(X,xin,xout);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

