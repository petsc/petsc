
static char help[] = "Tests MPI parallel matrix creation.\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            C; 
  MatInfo        info;
  PetscMPIInt    rank,size;
  PetscInt       i,j,m = 3,n = 2,low,high,iglobal;
  PetscInt       Ii,J,ldim;
  PetscErrorCode ierr;
  PetscTruth     flg;
  PetscScalar    v,one = 1.0;
  Vec            u,b;
  PetscInt       bs,ndiag,diag[7];  bs = 1,ndiag = 5;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  n = 2*size;

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);

  diag[0] = n;
  diag[1] = 1;
  diag[2] = 0;
  diag[3] = -1;
  diag[4] = -n;
  if (size>1) {ndiag = 7; diag[5] = 2; diag[6] = -2;}

  /* Create the matrix for the five point stencil, YET AGAIN */
  for (i=0; i<m; i++) { 
    for (j=2*rank; j<2*rank+2; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = Ii + n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = Ii - 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = Ii + 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /* Add extra elements (to illustrate variants of MatGetInfo) */
  Ii = n; J = n-2; v = 100.0;
  ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
  Ii = n-2; J = n; v = 100.0;
  ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Form vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,m*n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecGetLocalSize(u,&ldim);
  ierr = VecGetOwnershipRange(u,&low,&high);CHKERRQ(ierr);
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v = one*((PetscReal)i) + 100.0*rank;
    ierr = VecSetValues(u,1,&iglobal,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);

  ierr = MatMult(C,u,b);CHKERRQ(ierr);

  ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(u);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-view_info",&flg);CHKERRQ(ierr);
  if (flg)  {ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);}
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatGetInfo(C,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix information (global sums):\n\
     nonzeros = %D, allocated nonzeros = %D\n",(PetscInt)info.nz_used,(PetscInt)info.nz_allocated);CHKERRQ(ierr);
  ierr = MatGetInfo (C,MAT_GLOBAL_MAX,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix information (global max):\n\
     nonzeros = %D, allocated nonzeros = %D\n",(PetscInt)info.nz_used,(PetscInt)info.nz_allocated);CHKERRQ(ierr);

  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
