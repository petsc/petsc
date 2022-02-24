
static char help[] = "Tests the various sequential routines in MATSEQSBAIJ format.\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;
  Vec            x,y,b,s1,s2;
  Mat            A;                    /* linear system matrix */
  Mat            sA,sB,sFactor,B,C;    /* symmetric matrices */
  PetscInt       n,mbs=16,bs=1,nz=3,prob=1,i,j,k1,k2,col[3],lf,block, row,Ii,J,n1,inc;
  PetscReal      norm1,norm2,rnorm,tol=10*PETSC_SMALL;
  PetscScalar    neg_one=-1.0,four=4.0,value[3];
  IS             perm, iscol;
  PetscRandom    rdm;
  PetscBool      doIcc=PETSC_TRUE,equal;
  MatInfo        minfo1,minfo2;
  MatFactorInfo  factinfo;
  MatType        type;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mbs",&mbs,NULL));

  n    = mbs*bs;
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&A));
  CHKERRQ(MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetType(A,MATSEQBAIJ));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSeqBAIJSetPreallocation(A,bs,nz,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_SELF,&sA));
  CHKERRQ(MatSetSizes(sA,n,n,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetType(sA,MATSEQSBAIJ));
  CHKERRQ(MatSetFromOptions(sA));
  CHKERRQ(MatGetType(sA,&type));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)sA,MATSEQSBAIJ,&doIcc));
  CHKERRQ(MatSeqSBAIJSetPreallocation(sA,bs,nz,NULL));
  CHKERRQ(MatSetOption(sA,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE));

  /* Test MatGetOwnershipRange() */
  CHKERRQ(MatGetOwnershipRange(A,&Ii,&J));
  CHKERRQ(MatGetOwnershipRange(sA,&i,&j));
  if (i-Ii || j-J) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatGetOwnershipRange() in MatSBAIJ format\n"));
  }

  /* Assemble matrix */
  if (bs == 1) {
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-test_problem",&prob,NULL));
    if (prob == 1) { /* tridiagonal matrix */
      value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
      for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
        CHKERRQ(MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES));
      }
      i = n - 1; col[0]=0; col[1] = n - 2; col[2] = n - 1;

      value[0]= 0.1; value[1]=-1; value[2]=2;

      CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
      CHKERRQ(MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES));

      i        = 0;
      col[0]   = n-1;   col[1] = 1;      col[2] = 0;
      value[0] = 0.1; value[1] = -1.0; value[2] = 2;

      CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
      CHKERRQ(MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES));

    } else if (prob == 2) { /* matrix for the five point stencil */
      n1 = (PetscInt) (PetscSqrtReal((PetscReal)n) + 0.001);
      PetscCheckFalse(n1*n1 - n,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"sqrt(n) must be a positive integer!");
      for (i=0; i<n1; i++) {
        for (j=0; j<n1; j++) {
          Ii = j + n1*i;
          if (i>0) {
            J    = Ii - n1;
            CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES));
            CHKERRQ(MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES));
          }
          if (i<n1-1) {
            J    = Ii + n1;
            CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES));
            CHKERRQ(MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES));
          }
          if (j>0) {
            J    = Ii - 1;
            CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES));
            CHKERRQ(MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES));
          }
          if (j<n1-1) {
            J    = Ii + 1;
            CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES));
            CHKERRQ(MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES));
          }
          CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&four,INSERT_VALUES));
          CHKERRQ(MatSetValues(sA,1,&Ii,1,&Ii,&four,INSERT_VALUES));
        }
      }
    }

  } else { /* bs > 1 */
    for (block=0; block<n/bs; block++) {
      /* diagonal blocks */
      value[0] = -1.0; value[1] = 4.0; value[2] = -1.0;
      for (i=1+block*bs; i<bs-1+block*bs; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
        CHKERRQ(MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES));
      }
      i = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;

      value[0]=-1.0; value[1]=4.0;

      CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
      CHKERRQ(MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES));

      i = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs;

      value[0]=4.0; value[1] = -1.0;

      CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
      CHKERRQ(MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES));
    }
    /* off-diagonal blocks */
    value[0]=-1.0;
    for (i=0; i<(n/bs-1)*bs; i++) {
      col[0]=i+bs;

      CHKERRQ(MatSetValues(A,1,&i,1,col,value,INSERT_VALUES));
      CHKERRQ(MatSetValues(sA,1,&i,1,col,value,INSERT_VALUES));

      col[0]=i; row=i+bs;

      CHKERRQ(MatSetValues(A,1,&row,1,col,value,INSERT_VALUES));
      CHKERRQ(MatSetValues(sA,1,&row,1,col,value,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatAssemblyBegin(sA,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(sA,MAT_FINAL_ASSEMBLY));

  /* Test MatGetInfo() of A and sA */
  CHKERRQ(MatGetInfo(A,MAT_LOCAL,&minfo1));
  CHKERRQ(MatGetInfo(sA,MAT_LOCAL,&minfo2));
  i  = (int) (minfo1.nz_used - minfo2.nz_used);
  j  = (int) (minfo1.nz_allocated - minfo2.nz_allocated);
  k1 = (int) (minfo1.nz_allocated - minfo1.nz_used);
  k2 = (int) (minfo2.nz_allocated - minfo2.nz_used);
  if (i < 0 || j < 0 || k1 < 0 || k2 < 0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error (compare A and sA): MatGetInfo()\n"));
  }

  /* Test MatDuplicate() */
  CHKERRQ(MatNorm(A,NORM_FROBENIUS,&norm1));
  CHKERRQ(MatDuplicate(sA,MAT_COPY_VALUES,&sB));
  CHKERRQ(MatEqual(sA,sB,&equal));
  PetscCheckFalse(!equal,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Error in MatDuplicate()");

  /* Test MatNorm() */
  CHKERRQ(MatNorm(A,NORM_FROBENIUS,&norm1));
  CHKERRQ(MatNorm(sB,NORM_FROBENIUS,&norm2));
  rnorm = PetscAbsReal(norm1-norm2)/norm2;
  if (rnorm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_FROBENIUS, NormA=%16.14e NormsB=%16.14e\n",(double)norm1,(double)norm2));
  }
  CHKERRQ(MatNorm(A,NORM_INFINITY,&norm1));
  CHKERRQ(MatNorm(sB,NORM_INFINITY,&norm2));
  rnorm = PetscAbsReal(norm1-norm2)/norm2;
  if (rnorm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_INFINITY(), NormA=%16.14e NormsB=%16.14e\n",(double)norm1,(double)norm2));
  }
  CHKERRQ(MatNorm(A,NORM_1,&norm1));
  CHKERRQ(MatNorm(sB,NORM_1,&norm2));
  rnorm = PetscAbsReal(norm1-norm2)/norm2;
  if (rnorm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_INFINITY(), NormA=%16.14e NormsB=%16.14e\n",(double)norm1,(double)norm2));
  }

  /* Test MatGetInfo(), MatGetSize(), MatGetBlockSize() */
  CHKERRQ(MatGetInfo(A,MAT_LOCAL,&minfo1));
  CHKERRQ(MatGetInfo(sB,MAT_LOCAL,&minfo2));
  i  = (int) (minfo1.nz_used - minfo2.nz_used);
  j  = (int) (minfo1.nz_allocated - minfo2.nz_allocated);
  k1 = (int) (minfo1.nz_allocated - minfo1.nz_used);
  k2 = (int) (minfo2.nz_allocated - minfo2.nz_used);
  if (i < 0 || j < 0 || k1 < 0 || k2 < 0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error(compare A and sB): MatGetInfo()\n"));
  }

  CHKERRQ(MatGetSize(A,&Ii,&J));
  CHKERRQ(MatGetSize(sB,&i,&j));
  if (i-Ii || j-J) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatGetSize()\n"));
  }

  CHKERRQ(MatGetBlockSize(A, &Ii));
  CHKERRQ(MatGetBlockSize(sB, &i));
  if (i-Ii) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatGetBlockSize()\n"));
  }

  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&x));
  CHKERRQ(VecDuplicate(x,&s1));
  CHKERRQ(VecDuplicate(x,&s2));
  CHKERRQ(VecDuplicate(x,&y));
  CHKERRQ(VecDuplicate(x,&b));
  CHKERRQ(VecSetRandom(x,rdm));

  /* Test MatDiagonalScale(), MatGetDiagonal(), MatScale() */
#if !defined(PETSC_USE_COMPLEX)
  /* Scaling matrix with complex numbers results non-spd matrix,
     causing crash of MatForwardSolve() and MatBackwardSolve() */
  CHKERRQ(MatDiagonalScale(A,x,x));
  CHKERRQ(MatDiagonalScale(sB,x,x));
  CHKERRQ(MatMultEqual(A,sB,10,&equal));
  PetscCheckFalse(!equal,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Error in MatDiagonalScale");

  CHKERRQ(MatGetDiagonal(A,s1));
  CHKERRQ(MatGetDiagonal(sB,s2));
  CHKERRQ(VecAXPY(s2,neg_one,s1));
  CHKERRQ(VecNorm(s2,NORM_1,&norm1));
  if (norm1>tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error:MatGetDiagonal(), ||s1-s2||=%g\n",(double)norm1));
  }

  {
    PetscScalar alpha=0.1;
    CHKERRQ(MatScale(A,alpha));
    CHKERRQ(MatScale(sB,alpha));
  }
#endif

  /* Test MatGetRowMaxAbs() */
  CHKERRQ(MatGetRowMaxAbs(A,s1,NULL));
  CHKERRQ(MatGetRowMaxAbs(sB,s2,NULL));
  CHKERRQ(VecNorm(s1,NORM_1,&norm1));
  CHKERRQ(VecNorm(s2,NORM_1,&norm2));
  norm1 -= norm2;
  if (norm1<-tol || norm1>tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error:MatGetRowMaxAbs() \n"));
  }

  /* Test MatMult() */
  for (i=0; i<40; i++) {
    CHKERRQ(VecSetRandom(x,rdm));
    CHKERRQ(MatMult(A,x,s1));
    CHKERRQ(MatMult(sB,x,s2));
    CHKERRQ(VecNorm(s1,NORM_1,&norm1));
    CHKERRQ(VecNorm(s2,NORM_1,&norm2));
    norm1 -= norm2;
    if (norm1<-tol || norm1>tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatMult(), norm1-norm2: %g\n",(double)norm1));
    }
  }

  /* MatMultAdd() */
  for (i=0; i<40; i++) {
    CHKERRQ(VecSetRandom(x,rdm));
    CHKERRQ(VecSetRandom(y,rdm));
    CHKERRQ(MatMultAdd(A,x,y,s1));
    CHKERRQ(MatMultAdd(sB,x,y,s2));
    CHKERRQ(VecNorm(s1,NORM_1,&norm1));
    CHKERRQ(VecNorm(s2,NORM_1,&norm2));
    norm1 -= norm2;
    if (norm1<-tol || norm1>tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error:MatMultAdd(), norm1-norm2: %g\n",(double)norm1));
    }
  }

  /* Test MatMatMult() for sbaij and dense matrices */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,5*n,NULL,&B));
  CHKERRQ(MatSetRandom(B,rdm));
  CHKERRQ(MatMatMult(sA,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C));
  CHKERRQ(MatMatMultEqual(sA,B,C,5*n,&equal));
  PetscCheckFalse(!equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: MatMatMult()");
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&B));

  /* Test MatCholeskyFactor(), MatICCFactor() with natural ordering */
  CHKERRQ(MatGetOrdering(A,MATORDERINGNATURAL,&perm,&iscol));
  CHKERRQ(ISDestroy(&iscol));
  norm1 = tol;
  inc   = bs;

  /* initialize factinfo */
  CHKERRQ(PetscMemzero(&factinfo,sizeof(MatFactorInfo)));

  for (lf=-1; lf<10; lf += inc) {
    if (lf==-1) {  /* Cholesky factor of sB (duplicate sA) */
      factinfo.fill = 5.0;

      CHKERRQ(MatGetFactor(sB,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&sFactor));
      CHKERRQ(MatCholeskyFactorSymbolic(sFactor,sB,perm,&factinfo));
    } else if (!doIcc) break;
    else {       /* incomplete Cholesky factor */
      factinfo.fill   = 5.0;
      factinfo.levels = lf;

      CHKERRQ(MatGetFactor(sB,MATSOLVERPETSC,MAT_FACTOR_ICC,&sFactor));
      CHKERRQ(MatICCFactorSymbolic(sFactor,sB,perm,&factinfo));
    }
    CHKERRQ(MatCholeskyFactorNumeric(sFactor,sB,&factinfo));
    /* MatView(sFactor, PETSC_VIEWER_DRAW_WORLD); */

    /* test MatGetDiagonal on numeric factor */
    /*
    if (lf == -1) {
      CHKERRQ(MatGetDiagonal(sFactor,s1));
      printf(" in ex74.c, diag: \n");
      CHKERRQ(VecView(s1,PETSC_VIEWER_STDOUT_SELF));
    }
    */

    CHKERRQ(MatMult(sB,x,b));

    /* test MatForwardSolve() and MatBackwardSolve() */
    if (lf == -1) {
      CHKERRQ(MatForwardSolve(sFactor,b,s1));
      CHKERRQ(MatBackwardSolve(sFactor,s1,s2));
      CHKERRQ(VecAXPY(s2,neg_one,x));
      CHKERRQ(VecNorm(s2,NORM_2,&norm2));
      if (10*norm1 < norm2) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"MatForwardSolve and BackwardSolve: Norm of error=%g, bs=%" PetscInt_FMT "\n",(double)norm2,bs));
      }
    }

    /* test MatSolve() */
    CHKERRQ(MatSolve(sFactor,b,y));
    CHKERRQ(MatDestroy(&sFactor));
    /* Check the error */
    CHKERRQ(VecAXPY(y,neg_one,x));
    CHKERRQ(VecNorm(y,NORM_2,&norm2));
    if (10*norm1 < norm2 && lf-inc != -1) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"lf=%" PetscInt_FMT ", %" PetscInt_FMT ", Norm of error=%g, %g\n",lf-inc,lf,(double)norm1,(double)norm2));
    }
    norm1 = norm2;
    if (norm2 < tol && lf != -1) break;
  }

#if defined(PETSC_HAVE_MUMPS)
  CHKERRQ(MatGetFactor(sA,MATSOLVERMUMPS,MAT_FACTOR_CHOLESKY,&sFactor));
  CHKERRQ(MatCholeskyFactorSymbolic(sFactor,sA,NULL,NULL));
  CHKERRQ(MatCholeskyFactorNumeric(sFactor,sA,NULL));
  for (i=0; i<10; i++) {
    CHKERRQ(VecSetRandom(b,rdm));
    CHKERRQ(MatSolve(sFactor,b,y));
    /* Check the error */
    CHKERRQ(MatMult(sA,y,x));
    CHKERRQ(VecAXPY(x,neg_one,b));
    CHKERRQ(VecNorm(x,NORM_2,&norm2));
    if (norm2>tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error:MatSolve(), norm2: %g\n",(double)norm2));
    }
  }
  CHKERRQ(MatDestroy(&sFactor));
#endif

  CHKERRQ(ISDestroy(&perm));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&sB));
  CHKERRQ(MatDestroy(&sA));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&s1));
  CHKERRQ(VecDestroy(&s2));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(PetscRandomDestroy(&rdm));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -bs {{1 2 3 4 5 6 7 8}}

TEST*/
