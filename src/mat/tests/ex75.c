
static char help[] = "Tests various routines in MatMPISBAIJ format.\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Vec            x,y,u,s1,s2;
  Mat            A,sA,sB;
  PetscRandom    rctx;
  PetscReal      r1,r2,rnorm,tol = PETSC_SQRT_MACHINE_EPSILON;
  PetscScalar    one=1.0, neg_one=-1.0, value[3], four=4.0,alpha=0.1;
  PetscInt       n,col[3],n1,block,row,i,j,i2,j2,Ii,J,rstart,rend,bs=1,mbs=16,d_nz=3,o_nz=3,prob=1;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mbs",&mbs,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-prob",&prob,NULL));

  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Create a BAIJ matrix A */
  n = mbs*bs;
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetType(A,MATBAIJ));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatMPIBAIJSetPreallocation(A,bs,d_nz,NULL,o_nz,NULL));
  CHKERRQ(MatSeqBAIJSetPreallocation(A,bs,d_nz,NULL));
  CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));

  if (bs == 1) {
    if (prob == 1) { /* tridiagonal matrix */
      value[0] = -1.0; value[1] = 0.0; value[2] = -1.0;
      for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
      }
      i       = n - 1; col[0]=0; col[1] = n - 2; col[2] = n - 1;
      value[0]= 0.1; value[1]=-1.0; value[2]=0.0;
      CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));

      i        = 0; col[0] = 0; col[1] = 1; col[2]=n-1;
      value[0] = 0.0; value[1] = -1.0; value[2]=0.1;
      CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
    } else if (prob ==2) { /* matrix for the five point stencil */
      n1 = (int) PetscSqrtReal((PetscReal)n);
      for (i=0; i<n1; i++) {
        for (j=0; j<n1; j++) {
          Ii = j + n1*i;
          if (i>0)    {J = Ii - n1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES));}
          if (i<n1-1) {J = Ii + n1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES));}
          if (j>0)    {J = Ii - 1;  CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES));}
          if (j<n1-1) {J = Ii + 1;  CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES));}
          CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&four,INSERT_VALUES));
        }
      }
    }
    /* end of if (bs == 1) */
  } else {  /* bs > 1 */
    for (block=0; block<n/bs; block++) {
      /* diagonal blocks */
      value[0] = -1.0; value[1] = 4.0; value[2] = -1.0;
      for (i=1+block*bs; i<bs-1+block*bs; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
      }
      i       = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;
      value[0]=-1.0; value[1]=4.0;
      CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));

      i       = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs;
      value[0]=4.0; value[1] = -1.0;
      CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
    }
    /* off-diagonal blocks */
    value[0]=-1.0;
    for (i=0; i<(n/bs-1)*bs; i++) {
      col[0]=i+bs;
      CHKERRQ(MatSetValues(A,1,&i,1,col,value,INSERT_VALUES));
      col[0]=i; row=i+bs;
      CHKERRQ(MatSetValues(A,1,&row,1,col,value,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));

  /* Get SBAIJ matrix sA from A */
  CHKERRQ(MatConvert(A,MATSBAIJ,MAT_INITIAL_MATRIX,&sA));

  /* Test MatGetSize(), MatGetLocalSize() */
  CHKERRQ(MatGetSize(sA, &i,&j));
  CHKERRQ(MatGetSize(A, &i2,&j2));
  i   -= i2; j -= j2;
  if (i || j) {
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatGetSize()\n",rank));
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  }

  CHKERRQ(MatGetLocalSize(sA, &i,&j));
  CHKERRQ(MatGetLocalSize(A, &i2,&j2));
  i2  -= i; j2 -= j;
  if (i2 || j2) {
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatGetLocalSize()\n",rank));
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  }

  /* vectors */
  /*--------------------*/
  /* i is obtained from MatGetLocalSize() */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,i,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecDuplicate(x,&y));
  CHKERRQ(VecDuplicate(x,&u));
  CHKERRQ(VecDuplicate(x,&s1));
  CHKERRQ(VecDuplicate(x,&s2));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  CHKERRQ(PetscRandomSetFromOptions(rctx));
  CHKERRQ(VecSetRandom(x,rctx));
  CHKERRQ(VecSet(u,one));

  /* Test MatNorm() */
  CHKERRQ(MatNorm(A,NORM_FROBENIUS,&r1));
  CHKERRQ(MatNorm(sA,NORM_FROBENIUS,&r2));
  rnorm = PetscAbsReal(r1-r2)/r2;
  if (rnorm > tol && rank == 0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_FROBENIUS(), Anorm=%16.14e, sAnorm=%16.14e bs=%" PetscInt_FMT "\n",(double)r1,(double)r2,bs));
  }
  CHKERRQ(MatNorm(A,NORM_INFINITY,&r1));
  CHKERRQ(MatNorm(sA,NORM_INFINITY,&r2));
  rnorm = PetscAbsReal(r1-r2)/r2;
  if (rnorm > tol && rank == 0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error: MatNorm_INFINITY(), Anorm=%16.14e, sAnorm=%16.14e bs=%" PetscInt_FMT "\n",(double)r1,(double)r2,bs));
  }
  CHKERRQ(MatNorm(A,NORM_1,&r1));
  CHKERRQ(MatNorm(sA,NORM_1,&r2));
  rnorm = PetscAbsReal(r1-r2)/r2;
  if (rnorm > tol && rank == 0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error: MatNorm_1(), Anorm=%16.14e, sAnorm=%16.14e bs=%" PetscInt_FMT "\n",(double)r1,(double)r2,bs));
  }

  /* Test MatGetOwnershipRange() */
  CHKERRQ(MatGetOwnershipRange(sA,&rstart,&rend));
  CHKERRQ(MatGetOwnershipRange(A,&i2,&j2));
  i2  -= rstart; j2 -= rend;
  if (i2 || j2) {
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MaGetOwnershipRange()\n",rank));
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  }

  /* Test MatDiagonalScale() */
  CHKERRQ(MatDiagonalScale(A,x,x));
  CHKERRQ(MatDiagonalScale(sA,x,x));
  CHKERRQ(MatMultEqual(A,sA,10,&flg));
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Error in MatDiagonalScale");

  /* Test MatGetDiagonal(), MatScale() */
  CHKERRQ(MatGetDiagonal(A,s1));
  CHKERRQ(MatGetDiagonal(sA,s2));
  CHKERRQ(VecNorm(s1,NORM_1,&r1));
  CHKERRQ(VecNorm(s2,NORM_1,&r2));
  r1  -= r2;
  if (r1<-tol || r1>tol) {
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatDiagonalScale() or MatGetDiagonal(), r1=%g \n",rank,(double)r1));
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  }

  CHKERRQ(MatScale(A,alpha));
  CHKERRQ(MatScale(sA,alpha));

  /* Test MatGetRowMaxAbs() */
  CHKERRQ(MatGetRowMaxAbs(A,s1,NULL));
  CHKERRQ(MatGetRowMaxAbs(sA,s2,NULL));

  CHKERRQ(VecNorm(s1,NORM_1,&r1));
  CHKERRQ(VecNorm(s2,NORM_1,&r2));
  r1  -= r2;
  if (r1<-tol || r1>tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatGetRowMaxAbs() \n"));
  }

  /* Test MatMult(), MatMultAdd() */
  CHKERRQ(MatMultEqual(A,sA,10,&flg));
  if (!flg) {
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMult() or MatScale()\n",rank));
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  }

  CHKERRQ(MatMultAddEqual(A,sA,10,&flg));
  if (!flg) {
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMultAdd()\n",rank));
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  }

  /* Test MatMultTranspose(), MatMultTransposeAdd() */
  for (i=0; i<10; i++) {
    CHKERRQ(VecSetRandom(x,rctx));
    CHKERRQ(MatMultTranspose(A,x,s1));
    CHKERRQ(MatMultTranspose(sA,x,s2));
    CHKERRQ(VecNorm(s1,NORM_1,&r1));
    CHKERRQ(VecNorm(s2,NORM_1,&r2));
    r1  -= r2;
    if (r1<-tol || r1>tol) {
      CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMult() or MatScale(), err=%g\n",rank,(double)r1));
      CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
    }
  }
  for (i=0; i<10; i++) {
    CHKERRQ(VecSetRandom(x,rctx));
    CHKERRQ(VecSetRandom(y,rctx));
    CHKERRQ(MatMultTransposeAdd(A,x,y,s1));
    CHKERRQ(MatMultTransposeAdd(sA,x,y,s2));
    CHKERRQ(VecNorm(s1,NORM_1,&r1));
    CHKERRQ(VecNorm(s2,NORM_1,&r2));
    r1  -= r2;
    if (r1<-tol || r1>tol) {
      CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMultAdd(), err=%g \n",rank,(double)r1));
      CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
    }
  }

  /* Test MatDuplicate() */
  CHKERRQ(MatDuplicate(sA,MAT_COPY_VALUES,&sB));
  CHKERRQ(MatEqual(sA,sB,&flg));
  if (!flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Error in MatDuplicate(), sA != sB \n"));
  }
  CHKERRQ(MatMultEqual(sA,sB,5,&flg));
  if (!flg) {
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatDuplicate() or MatMult()\n",rank));
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  }
  CHKERRQ(MatMultAddEqual(sA,sB,5,&flg));
  if (!flg) {
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatDuplicate() or MatMultAdd(()\n",rank));
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  }
  CHKERRQ(MatDestroy(&sB));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&s1));
  CHKERRQ(VecDestroy(&s2));
  CHKERRQ(MatDestroy(&sA));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscRandomDestroy(&rctx));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: {{1 3}}
      args: -bs {{1 2 3  5  7 8}} -mat_ignore_lower_triangular -prob {{1 2}}

TEST*/
