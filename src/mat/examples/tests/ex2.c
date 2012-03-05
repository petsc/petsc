
static char help[] = "Tests MatTranspose(), MatNorm(), MatAXPY() and MatAYPX().\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat              mat,tmat = 0;
  PetscInt         m = 7,n,i,j,rstart,rend,rect = 0;
  PetscErrorCode   ierr;
  PetscMPIInt      size,rank;
  PetscBool        flg;
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
  ierr = MatSetUp(mat);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mat,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) { 
    for (j=0; j<n; j++) { 
      v=10.0*i+j; 
      ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* ----------------- Test MatNorm()  ----------------- */
  ierr = MatNorm(mat,NORM_FROBENIUS,&normf);CHKERRQ(ierr);
  ierr = MatNorm(mat,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = MatNorm(mat,NORM_INFINITY,&normi);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"original A: Frobenious norm = %G, one norm = %G, infinity norm = %G\n",
                     normf,norm1,normi);CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* --------------- Test MatTranspose()  -------------- */
  ierr = PetscOptionsHasName(PETSC_NULL,"-in_place",&flg);CHKERRQ(ierr);
  if (!rect && flg) {
    ierr = MatTranspose(mat,MAT_REUSE_MATRIX,&mat);CHKERRQ(ierr);   /* in-place transpose */
    tmat = mat; mat = 0;
  } else {      /* out-of-place transpose */
    ierr = MatTranspose(mat,MAT_INITIAL_MATRIX,&tmat);CHKERRQ(ierr); 
  }

  /* ----------------- Test MatNorm()  ----------------- */
  /* Print info about transpose matrix */
  ierr = MatNorm(tmat,NORM_FROBENIUS,&normf);CHKERRQ(ierr);
  ierr = MatNorm(tmat,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = MatNorm(tmat,NORM_INFINITY,&normi);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"B = A^T: Frobenious norm = %G, one norm = %G, infinity norm = %G\n",
                     normf,norm1,normi);CHKERRQ(ierr);
  ierr = MatView(tmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* ----------------- Test MatAXPY(), MatAYPX()  ----------------- */
  if (mat && !rect) {
    alpha = 1.0;
    ierr = PetscOptionsGetScalar(PETSC_NULL,"-alpha",&alpha,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  B = B + alpha * A\n");CHKERRQ(ierr);
    ierr = MatAXPY(tmat,alpha,mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); 
    ierr = MatView(tmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAYPX:  B = alpha*B + A\n");CHKERRQ(ierr);
    ierr = MatAYPX(tmat,alpha,mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); 
    ierr = MatView(tmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  {
    Mat C;
    alpha = 1.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  C = C + alpha * A, C=A, SAME_NONZERO_PATTERN\n");CHKERRQ(ierr);
    ierr = MatDuplicate(mat,MAT_COPY_VALUES,&C);CHKERRQ(ierr); 
    ierr = MatAXPY(C,alpha,mat,SAME_NONZERO_PATTERN);CHKERRQ(ierr); 
    ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
    ierr = MatDestroy(&C);CHKERRQ(ierr);
  }

  {
    Mat matB;
    /* get matB that has nonzeros of mat in all even numbers of row and col */
    ierr = MatCreate(PETSC_COMM_WORLD,&matB);CHKERRQ(ierr);
    ierr = MatSetSizes(matB,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(matB);CHKERRQ(ierr);
    ierr = MatSetUp(matB);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(matB,&rstart,&rend);CHKERRQ(ierr);
    if (rstart % 2 != 0) rstart++;   
    for (i=rstart; i<rend; i += 2) { 
      for (j=0; j<n; j += 2) { 
        v=10.0*i+j; 
        ierr = MatSetValues(matB,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(matB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(matB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD," A: original matrix:\n");
    ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD," B(a subset of A):\n");
    ierr = MatView(matB,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  B = B + alpha * A, SUBSET_NONZERO_PATTERN\n");CHKERRQ(ierr);
    ierr = MatAXPY(mat,alpha,matB,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatDestroy(&matB);CHKERRQ(ierr);  
  }

  /* Free data structures */  
  if (mat)  {ierr = MatDestroy(&mat);CHKERRQ(ierr);}
  if (tmat) {ierr = MatDestroy(&tmat);CHKERRQ(ierr);}

  ierr = PetscFinalize();
  return 0;
}
 


