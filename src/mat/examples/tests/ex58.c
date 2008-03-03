
static char help[] = "Tests MatTranspose() and MatEqual() for MPIAIJ matrices.\n\n";

#include "petscmat.h"


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A,B;
  PetscInt       m = 7,n,i,rstart,rend,cols[3];
  PetscErrorCode ierr;
  PetscScalar    v[3];
  PetscTruth     equal;
  const char     *eq[2];

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  n = m;

  /* ------- Assemble matrix, test MatValid() --------- */

  ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,0,0,0,0,&A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  if (!rstart) {
    cols[0] = 0;
    cols[1] = 1;
    v[0]    = 2.0; v[1] = -1.0;
    ierr = MatSetValues(A,1,&rstart,2,cols,v,INSERT_VALUES);CHKERRQ(ierr);
    rstart++;
  }
  if (rend == m) {
    rend--;
    cols[0] = rend-1;
    cols[1] = rend;
    v[0]    = -1.0; v[1] = 2.0;
    ierr = MatSetValues(A,1,&rend,2,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  }
  v[0] = -1.0; v[1] = 2.0; v[2] = -1.0;
  for (i=rstart; i<rend; i++) { 
    cols[0] = i-1;
    cols[1] = i;
    cols[2] = i+1;
    ierr = MatSetValues(A,1,&i,3,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);

  ierr = MatEqual(A,B,&equal);

  eq[0] = "not equal";
  eq[1] = "equal";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrices are %s\n",eq[equal]);CHKERRQ(ierr);

  /* Free data structures */  
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);


  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
