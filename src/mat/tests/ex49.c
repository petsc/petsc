
static char help[] = "Tests MatTranspose(), MatNorm(), and MatAXPY().\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            mat,tmat = 0;
  PetscInt       m = 4,n,i,j;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscInt       rstart,rend,rect = 0;
  PetscBool      flg;
  PetscScalar    v;
  PetscReal      normf,normi,norm1;
  MatInfo        info;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  n    = m;
  ierr = PetscOptionsHasName(NULL,NULL,"-rect1",&flg);CHKERRQ(ierr);
  if (flg) {n += 2; rect = 1;}
  ierr = PetscOptionsHasName(NULL,NULL,"-rect2",&flg);CHKERRQ(ierr);
  if (flg) {n -= 2; rect = 1;}

  /* Create and assemble matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(mat);CHKERRQ(ierr);
  ierr = MatSetUp(mat);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mat,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    for (j=0; j<n; j++) {
      v    = 10*i+j;
      ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Print info about original matrix */
  ierr = MatGetInfo(mat,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"original matrix nonzeros = %D, allocated nonzeros = %D\n",
                     (PetscInt)info.nz_used,(PetscInt)info.nz_allocated);CHKERRQ(ierr);
  ierr = MatNorm(mat,NORM_FROBENIUS,&normf);CHKERRQ(ierr);
  ierr = MatNorm(mat,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = MatNorm(mat,NORM_INFINITY,&normi);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"original: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",(double)normf,(double)norm1,(double)normi);CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Form matrix transpose */
  ierr = PetscOptionsHasName(NULL,NULL,"-in_place",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatTranspose(mat,MAT_INPLACE_MATRIX,&mat);CHKERRQ(ierr);   /* in-place transpose */
    tmat = mat; mat = 0;
  } else {      /* out-of-place transpose */
    ierr = MatTranspose(mat,MAT_INITIAL_MATRIX,&tmat);CHKERRQ(ierr);
  }

  /* Print info about transpose matrix */
  ierr = MatGetInfo(tmat,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"transpose matrix nonzeros = %D, allocated nonzeros = %D\n",
                     (PetscInt)info.nz_used,(PetscInt)info.nz_allocated);CHKERRQ(ierr);
  ierr = MatNorm(tmat,NORM_FROBENIUS,&normf);CHKERRQ(ierr);
  ierr = MatNorm(tmat,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = MatNorm(tmat,NORM_INFINITY,&normi);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"transpose: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",(double)normf,(double)norm1,(double)normi);CHKERRQ(ierr);
  ierr = MatView(tmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  /* Test MatAXPY */
  if (mat && !rect) {
    PetscScalar alpha = 1.0;
    ierr = PetscOptionsGetScalar(NULL,NULL,"-alpha",&alpha,NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix addition:  B = B + alpha * A\n");CHKERRQ(ierr);
    ierr = MatAXPY(tmat,alpha,mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatView(tmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Free data structures */
  ierr = MatDestroy(&tmat);CHKERRQ(ierr);
  if (mat) {ierr = MatDestroy(&mat);CHKERRQ(ierr);}

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   testset:
     args: -rect1
     test:
       suffix: r1
       output_file: output/ex49_r1.out
     test:
       suffix: r1_inplace
       args: -in_place
       output_file: output/ex49_r1.out
     test:
       suffix: r1_par
       nsize: 2
       output_file: output/ex49_r1_par.out
     test:
       suffix: r1_par_inplace
       args: -in_place
       nsize: 2
       output_file: output/ex49_r1_par.out

TEST*/
