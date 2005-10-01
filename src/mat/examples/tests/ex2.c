
static char help[] = "Tests MatTranspose(), MatNorm(), MatValid(), and MatAXPY().\n\n";

#include "petscmat.h"

#define  TestMatNorm
#define  TestMatTranspose
#define  TestMatNorm_tmat
#define  TestMatAXPY
#undef   TestMatAXPY2

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat              mat,tmat = 0;
  PetscInt         m = 7,n,i,j,rstart,rend,rect = 0;
  PetscErrorCode   ierr;
  PetscMPIInt      size,rank;
  PetscTruth       flg;
  PetscScalar      v, alpha;
  PetscReal        normf,normi,norm1;

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

  ierr = MatCreate(PETSC_COMM_WORLD,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
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
#ifdef TestMatNorm
  ierr = MatNorm(mat,NORM_FROBENIUS,&normf);CHKERRQ(ierr);
  ierr = MatNorm(mat,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = MatNorm(mat,NORM_INFINITY,&normi);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"original: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",
                     normf,norm1,normi);CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif
  /* --------------- Test MatTranspose()  -------------- */
#ifdef TestMatTranspose
  ierr = PetscOptionsHasName(PETSC_NULL,"-in_place",&flg);CHKERRQ(ierr);
  if (!rect && flg) {
    ierr = MatTranspose(mat,0);CHKERRQ(ierr);   /* in-place transpose */
    tmat = mat; mat = 0;
  } else {      /* out-of-place transpose */
    ierr = MatTranspose(mat,&tmat);CHKERRQ(ierr); 
  }
#endif
  /* ----------------- Test MatNorm()  ----------------- */
#ifdef TestMatNorm_tmat
  /* Print info about transpose matrix */
  ierr = MatNorm(tmat,NORM_FROBENIUS,&normf);CHKERRQ(ierr);
  ierr = MatNorm(tmat,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = MatNorm(tmat,NORM_INFINITY,&normi);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"transpose: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",
                     normf,norm1,normi);CHKERRQ(ierr);
  ierr = MatView(tmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif
  /* ----------------- Test MatAXPY()  ----------------- */
#ifdef TestMatAXPY
  if (mat && !rect) {
    alpha = 1.0;
    ierr = PetscOptionsGetScalar(PETSC_NULL,"-alpha",&alpha,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix addition:  B = B + alpha * A\n");CHKERRQ(ierr);
    ierr = MatAXPY(tmat,alpha,mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); 
    ierr = MatView(tmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
#endif 
#ifdef TestMatAXPY2
    Mat matB;
    alpha = 1.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix addition:  B = B + alpha * A, SAME_NONZERO_PATTERN\n");CHKERRQ(ierr);
    ierr = MatDuplicate(mat,MAT_COPY_VALUES,&matB);CHKERRQ(ierr); 
    ierr = MatAXPY(matB,alpha,mat,SAME_NONZERO_PATTERN);CHKERRQ(ierr); 
    ierr = MatView(matB,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
    ierr = MatDestroy(matB);CHKERRQ(ierr);
#ifdef TMP
    /* get matB that has nonzeros of mat in all even numbers of row and col */
    ierr = MatCreate(PETSC_COMM_WORLD,&matB);CHKERRQ(ierr);
    ierr = MatSetSizes(matB,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(matB);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(matB,&rstart,&rend);CHKERRQ(ierr);
    for (i=rstart; i<rend; i += 2) { 
      for (j=0; j<n; j += 2) { 
        v=10*i+j; 
        ierr = MatSetValues(matB,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(matB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(matB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD," B: original matrix:\n");
    ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD," A(a subset of B):\n");
    ierr = MatView(matB,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix addition:  B = B + alpha * A, SUBSET_NONZERO_PATTERN\n");CHKERRQ(ierr);
    ierr = MatAXPY(mat,alpha,matB,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    
    ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix addition:  B = B + alpha * A, SUBSET_NONZERO_PATTERN\n");CHKERRQ(ierr);
    ierr = MatAXPY(mat,alpha,matB,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr); 
    ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    
    ierr = MatDestroy(matB);CHKERRQ(ierr);  
#endif /* TMP */
#endif 
  

  /* Free data structures */  
  if (mat)  {ierr = MatDestroy(mat);CHKERRQ(ierr);}
  if (tmat) {ierr = MatDestroy(tmat);CHKERRQ(ierr);}

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 


