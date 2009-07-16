static const char help[] = "Tests MatGetSchurComplement\n";

#include "petscksp.h"


#undef __FUNCT__
#define __FUNCT__ "Create"
PetscErrorCode Create(MPI_Comm comm,Mat *inA,IS *is0,IS *is1)
{
  PetscErrorCode ierr;
  Mat A;
  PetscInt r,rend,M;
  PetscMPIInt rank;

  PetscFunctionBegin;
  *inA = 0;
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,4,4,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&r,&rend);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,PETSC_NULL);CHKERRQ(ierr);

  ierr = ISCreateStride(comm,2,r,1,is0);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,2,r+2,1,is1);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  {
    PetscInt
      rows[] = {r,r+1,r+2,r+3},
      cols0[] = {r+0,r+1,r+3,(r+4)%M,(r+M-4)%M},
      cols1[] = {r+1,r+2,(r+4+1)%M,(r+M-4+1)%M},
      cols2[] = {r,r+2,(r+4+2)%M},
      cols3[] = {r+1,r+3,(r+4+3)%M};
    PetscScalar RR = 1000*rank,
      vals0[] = {RR+1,RR+2,RR+3,RR+4,RR+5},
      vals1[] = {RR+6,RR+7,RR+8,RR+9},
      vals2[] = {RR+10,RR+11,RR+12},
      vals3[] = {RR+13,RR+14,RR+15};
    ierr = MatSetValues(A,1,&rows[0],5,cols0,vals0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(A,1,&rows[1],4,cols1,vals1,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(A,1,&rows[2],3,cols2,vals2,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(A,1,&rows[3],3,cols3,vals3,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *inA = A;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Destroy"
PetscErrorCode Destroy(Mat A,IS is0,IS is1)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = ISDestroy(is0);CHKERRQ(ierr);
  ierr = ISDestroy(is1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char *argv[])
{
  PetscErrorCode ierr;
  Mat            A,S,Sexplicit;
  IS             is0,is1;

  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);

  /* Test the Schur complement one way */
  ierr = Create(PETSC_COMM_WORLD,&A,&is0,&is1);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISView(is0,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISView(is1,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatGetSchurComplement(A,is0,is0,is1,is1,MAT_INITIAL_MATRIX,&S,MAT_IGNORE_MATRIX,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatComputeExplicitOperator(S,&Sexplicit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nExplicit Schur complement of (0,0) in (1,1)\n");CHKERRQ(ierr);
  ierr = MatView(Sexplicit,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = Destroy(A,is0,is1);CHKERRQ(ierr);
  ierr = MatDestroy(S);CHKERRQ(ierr);
  ierr = MatDestroy(Sexplicit);CHKERRQ(ierr);

  /* And the other */
  ierr = Create(PETSC_COMM_WORLD,&A,&is0,&is1);CHKERRQ(ierr);
  ierr = MatGetSchurComplement(A,is1,is1,is0,is0,MAT_INITIAL_MATRIX,&S,MAT_IGNORE_MATRIX,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatComputeExplicitOperator(S,&Sexplicit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nExplicit Schur complement of (1,1) in (0,0)\n");CHKERRQ(ierr);
  ierr = MatView(Sexplicit,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = Destroy(A,is0,is1);CHKERRQ(ierr);
  ierr = MatDestroy(S);CHKERRQ(ierr);
  ierr = MatDestroy(Sexplicit);CHKERRQ(ierr);

  /* This time just the preconditioner */
  ierr = Create(PETSC_COMM_WORLD,&A,&is0,&is1);CHKERRQ(ierr);
  ierr = MatGetSchurComplement(A,is0,is0,is1,is1,MAT_IGNORE_MATRIX,PETSC_NULL,MAT_INITIAL_MATRIX,&S);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nPreconditioning Schur complement of (0,0) in (1,1)\n");CHKERRQ(ierr);
  ierr = MatView(S,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = Destroy(A,is0,is1);CHKERRQ(ierr);
  ierr = MatDestroy(S);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
