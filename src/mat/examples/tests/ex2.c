/*$Id: ex2.c,v 1.25 2001/08/07 21:30:08 bsmith Exp $*/

static char help[] = "Tests MatTranspose(), MatNorm(), MatValid(), and MatAXPY().\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat          mat,tmat = 0;
  int          m = 7,n,i,j,ierr,size,rank,rstart,rend,rect = 0;
  PetscTruth   flg;
  PetscScalar  v;
  PetscReal    normf,normi,norm1;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  n = m;
  ierr = PetscOptionsHasName(PETSC_NULL,"-rectA",&flg);CHKERRQ(ierr);
  if (flg) {n += 2; rect = 1;}
  ierr = PetscOptionsHasName(PETSC_NULL,"-rectB",&flg);CHKERRQ(ierr);
  if (flg) {n -= 2; rect = 1;}

  /* ------- Assemble matrix, test MatValid() --------- */

  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,&mat);CHKERRQ(ierr);
  ierr = MatSetFromOptions(mat);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mat,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) { 
    for (j=0; j<n; j++) { 
      v=10*i+j; 
      ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Test whether matrix has been corrupted (just to demonstrate this
     routine) not needed in most application codes. */
  ierr = MatValid(mat,(PetscTruth*)&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(1,"Corrupted matrix.");

  /* ----------------- Test MatNorm()  ----------------- */

  ierr = MatNorm(mat,NORM_FROBENIUS,&normf);CHKERRQ(ierr);
  ierr = MatNorm(mat,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = MatNorm(mat,NORM_INFINITY,&normi);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"original: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",
                     normf,norm1,normi);CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* --------------- Test MatTranspose()  -------------- */

  ierr = PetscOptionsHasName(PETSC_NULL,"-in_place",&flg);CHKERRQ(ierr);
  if (!rect && flg) {
    ierr = MatTranspose(mat,0);CHKERRQ(ierr);   /* in-place transpose */
    tmat = mat; mat = 0;
  } else {      /* out-of-place transpose */
    ierr = MatTranspose(mat,&tmat);CHKERRQ(ierr); 
  }

  /* ----------------- Test MatNorm()  ----------------- */

  /* Print info about transpose matrix */
  ierr = MatNorm(tmat,NORM_FROBENIUS,&normf);CHKERRQ(ierr);
  ierr = MatNorm(tmat,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = MatNorm(tmat,NORM_INFINITY,&normi);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"transpose: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",
                     normf,norm1,normi);CHKERRQ(ierr);
  ierr = MatView(tmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* ----------------- Test MatAXPY()  ----------------- */

  if (mat && !rect) {
    PetscScalar alpha = 1.0;
    ierr = PetscOptionsGetScalar(PETSC_NULL,"-alpha",&alpha,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix addition:  B = B + alpha * A\n");CHKERRQ(ierr);
    ierr = MatAXPY(&alpha,mat,tmat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); 
    ierr = MatView(tmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Free data structures */  
  ierr = MatDestroy(tmat);CHKERRQ(ierr);
  if (mat) {ierr = MatDestroy(mat);CHKERRQ(ierr);}

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
