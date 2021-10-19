
static char help[] = "Tests cholesky, icc factorization and solve on sequential aij, baij and sbaij matrices. \n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Vec            x,y,b;
  Mat            A;           /* linear system matrix */
  Mat            sA,sC;       /* symmetric part of the matrices */
  PetscInt       n,mbs=16,bs=1,nz=3,prob=1,i,j,col[3],block, row,Ii,J,n1,lvl;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscReal      norm2;
  PetscScalar    neg_one = -1.0,four=4.0,value[3];
  IS             perm,cperm;
  PetscRandom    rdm;
  PetscBool      reorder = PETSC_FALSE,displ = PETSC_FALSE;
  MatFactorInfo  factinfo;
  PetscBool      equal;
  PetscBool      TestAIJ = PETSC_FALSE,TestBAIJ = PETSC_TRUE;
  PetscInt       TestShift=0;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mbs",&mbs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-reorder",&reorder,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testaij",&TestAIJ,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-testShift",&TestShift,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-displ",&displ,NULL);CHKERRQ(ierr);

  n = mbs*bs;
  if (TestAIJ) { /* A is in aij format */
    ierr     = MatCreateSeqAIJ(PETSC_COMM_WORLD,n,n,nz,NULL,&A);CHKERRQ(ierr);
    TestBAIJ = PETSC_FALSE;
  } else { /* A is in baij format */
    ierr    = MatCreateSeqBAIJ(PETSC_COMM_WORLD,bs,n,n,nz,NULL,&A);CHKERRQ(ierr);
    TestAIJ = PETSC_FALSE;
  }

  /* Assemble matrix */
  if (bs == 1) {
    ierr = PetscOptionsGetInt(NULL,NULL,"-test_problem",&prob,NULL);CHKERRQ(ierr);
    if (prob == 1) { /* tridiagonal matrix */
      value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
      for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      }
      i = n - 1; col[0]=0; col[1] = n - 2; col[2] = n - 1;

      value[0]= 0.1; value[1]=-1; value[2]=2;
      ierr    = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);

      i = 0; col[0] = 0; col[1] = 1; col[2]=n-1;

      value[0] = 2.0; value[1] = -1.0; value[2]=0.1;
      ierr     = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    } else if (prob ==2) { /* matrix for the five point stencil */
      n1 = (PetscInt) (PetscSqrtReal((PetscReal)n) + 0.001);
      if (n1*n1 - n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"sqrt(n) must be a positive integer!");
      for (i=0; i<n1; i++) {
        for (j=0; j<n1; j++) {
          Ii = j + n1*i;
          if (i>0) {
            J    = Ii - n1;
            ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          if (i<n1-1) {
            J    = Ii + n1;
            ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          if (j>0) {
            J    = Ii - 1;
            ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          if (j<n1-1) {
            J    = Ii + 1;
            ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          ierr = MatSetValues(A,1,&Ii,1,&Ii,&four,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  } else { /* bs > 1 */
    for (block=0; block<n/bs; block++) {
      /* diagonal blocks */
      value[0] = -1.0; value[1] = 4.0; value[2] = -1.0;
      for (i=1+block*bs; i<bs-1+block*bs; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      }
      i = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;

      value[0]=-1.0; value[1]=4.0;
      ierr    = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);

      i = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs;

      value[0]=4.0; value[1] = -1.0;
      ierr    = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    /* off-diagonal blocks */
    value[0]=-1.0;
    for (i=0; i<(n/bs-1)*bs; i++) {
      col[0]=i+bs;
      ierr  = MatSetValues(A,1,&i,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
      col[0]=i; row=i+bs;
      ierr  = MatSetValues(A,1,&row,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  if (TestShift) {
    /* set diagonals in the 0-th block as 0 for testing shift numerical factor */
    for (i=0; i<bs; i++) {
      row  = i; col[0] = i; value[0] = 0.0;
      ierr = MatSetValues(A,1,&row,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Test MatConvert */
  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatConvert(A,MATSEQSBAIJ,MAT_INITIAL_MATRIX,&sA);CHKERRQ(ierr);
  ierr = MatMultEqual(A,sA,20,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"A != sA");

  /* Test MatGetOwnershipRange() */
  ierr = MatGetOwnershipRange(A,&Ii,&J);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(sA,&i,&j);CHKERRQ(ierr);
  if (i-Ii || j-J) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatGetOwnershipRange() in MatSBAIJ format\n");

  /* Vectors */
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);

  /* Test MatReordering() - not work on sbaij matrix */
  if (reorder) {
    ierr = MatGetOrdering(A,MATORDERINGRCM,&perm,&cperm);CHKERRQ(ierr);
  } else {
    ierr = MatGetOrdering(A,MATORDERINGNATURAL,&perm,&cperm);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&cperm);CHKERRQ(ierr);

  /* initialize factinfo */
  ierr = MatFactorInfoInitialize(&factinfo);CHKERRQ(ierr);
  if (TestShift == 1) {
    factinfo.shifttype   = (PetscReal)MAT_SHIFT_NONZERO;
    factinfo.shiftamount = 0.1;
  } else if (TestShift == 2) {
    factinfo.shifttype = (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE;
  }

  /* Test MatCholeskyFactor(), MatICCFactor() */
  /*------------------------------------------*/
  /* Test aij matrix A */
  if (TestAIJ) {
    if (displ) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"AIJ: \n");CHKERRQ(ierr);
    }
    i = 0;
    for (lvl=-1; lvl<10; lvl++) {
      if (lvl==-1) {  /* Cholesky factor */
        factinfo.fill = 5.0;

        ierr = MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&sC);CHKERRQ(ierr);
        ierr = MatCholeskyFactorSymbolic(sC,A,perm,&factinfo);CHKERRQ(ierr);
      } else {       /* incomplete Cholesky factor */
        factinfo.fill   = 5.0;
        factinfo.levels = lvl;

        ierr = MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_ICC,&sC);CHKERRQ(ierr);
        ierr = MatICCFactorSymbolic(sC,A,perm,&factinfo);CHKERRQ(ierr);
      }
      ierr = MatCholeskyFactorNumeric(sC,A,&factinfo);CHKERRQ(ierr);

      ierr = MatMult(A,x,b);CHKERRQ(ierr);
      ierr = MatSolve(sC,b,y);CHKERRQ(ierr);
      ierr = MatDestroy(&sC);CHKERRQ(ierr);

      /* Check the residual */
      ierr = VecAXPY(y,neg_one,x);CHKERRQ(ierr);
      ierr = VecNorm(y,NORM_2,&norm2);CHKERRQ(ierr);

      if (displ) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"  lvl: %D, residual: %g\n", lvl,(double)norm2);CHKERRQ(ierr);
      }
    }
  }

  /* Test baij matrix A */
  if (TestBAIJ) {
    if (displ) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"BAIJ: \n");CHKERRQ(ierr);
    }
    i = 0;
    for (lvl=-1; lvl<10; lvl++) {
      if (lvl==-1) {  /* Cholesky factor */
        factinfo.fill = 5.0;

        ierr = MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&sC);CHKERRQ(ierr);
        ierr = MatCholeskyFactorSymbolic(sC,A,perm,&factinfo);CHKERRQ(ierr);
      } else {       /* incomplete Cholesky factor */
        factinfo.fill   = 5.0;
        factinfo.levels = lvl;

        ierr = MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_ICC,&sC);CHKERRQ(ierr);
        ierr = MatICCFactorSymbolic(sC,A,perm,&factinfo);CHKERRQ(ierr);
      }
      ierr = MatCholeskyFactorNumeric(sC,A,&factinfo);CHKERRQ(ierr);

      ierr = MatMult(A,x,b);CHKERRQ(ierr);
      ierr = MatSolve(sC,b,y);CHKERRQ(ierr);
      ierr = MatDestroy(&sC);CHKERRQ(ierr);

      /* Check the residual */
      ierr = VecAXPY(y,neg_one,x);CHKERRQ(ierr);
      ierr = VecNorm(y,NORM_2,&norm2);CHKERRQ(ierr);
      if (displ) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"  lvl: %D, residual: %g\n", lvl,(double)norm2);CHKERRQ(ierr);
      }
    }
  }

  /* Test sbaij matrix sA */
  if (displ) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"SBAIJ: \n");CHKERRQ(ierr);
  }
  i = 0;
  for (lvl=-1; lvl<10; lvl++) {
    if (lvl==-1) {  /* Cholesky factor */
      factinfo.fill = 5.0;

      ierr = MatGetFactor(sA,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&sC);CHKERRQ(ierr);
      ierr = MatCholeskyFactorSymbolic(sC,sA,perm,&factinfo);CHKERRQ(ierr);
    } else {       /* incomplete Cholesky factor */
      factinfo.fill   = 5.0;
      factinfo.levels = lvl;

      ierr = MatGetFactor(sA,MATSOLVERPETSC,MAT_FACTOR_ICC,&sC);CHKERRQ(ierr);
      ierr = MatICCFactorSymbolic(sC,sA,perm,&factinfo);CHKERRQ(ierr);
    }
    ierr = MatCholeskyFactorNumeric(sC,sA,&factinfo);CHKERRQ(ierr);

    if (lvl==0 && bs==1) { /* Test inplace ICC(0) for sbaij sA - does not work for new datastructure */
      /*
        Mat B;
        ierr = MatDuplicate(sA,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
        ierr = MatICCFactor(B,perm,&factinfo);CHKERRQ(ierr);
        ierr = MatEqual(sC,B,&equal);CHKERRQ(ierr);
        if (!equal) {
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"in-place Cholesky factor != out-place Cholesky factor");
        }
        ierr = MatDestroy(&B);CHKERRQ(ierr);
      */
    }

    ierr = MatMult(sA,x,b);CHKERRQ(ierr);
    ierr = MatSolve(sC,b,y);CHKERRQ(ierr);

    /* Test MatSolves() */
    if (bs == 1) {
      Vecs xx,bb;
      ierr = VecsCreateSeq(PETSC_COMM_SELF,n,4,&xx);CHKERRQ(ierr);
      ierr = VecsDuplicate(xx,&bb);CHKERRQ(ierr);
      ierr = MatSolves(sC,bb,xx);CHKERRQ(ierr);
      ierr = VecsDestroy(xx);CHKERRQ(ierr);
      ierr = VecsDestroy(bb);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&sC);CHKERRQ(ierr);

    /* Check the residual */
    ierr = VecAXPY(y,neg_one,x);CHKERRQ(ierr);
    ierr = VecNorm(y,NORM_2,&norm2);CHKERRQ(ierr);
    if (displ) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"  lvl: %D, residual: %g\n", lvl,(double)norm2);CHKERRQ(ierr);
    }
  }

  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&sA);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -bs {{1 2 3 4 5 6 7 8}}

   test:
      suffix: 3
      args: -testaij
      output_file: output/ex76_1.out

TEST*/
