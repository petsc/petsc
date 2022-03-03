static char help[] = "Tests the use of MatTranspose_Nest and MatMatMult_Nest_Dense\n";

#include <petscmat.h>

PetscErrorCode TestInitialMatrix(void)
{
  const PetscInt  nr = 2,nc = 3,nk = 10;
  PetscInt        n,N,m,M;
  const PetscInt  arow[2*3] = { 2,2,2,3,3,3 };
  const PetscInt  acol[2*3] = { 3,2,4,3,2,4 };
  Mat             A,Atranspose,B,C;
  Mat             subs[2*3],**block;
  Vec             x,y,Ax,ATy;
  PetscInt        i,j;
  PetscScalar     dot1,dot2,zero = 0.0,one = 1.0,*valsB,*valsC;
  PetscReal       norm;
  PetscRandom     rctx;
  PetscBool       equal;

  PetscFunctionBegin;
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  /* Force the random numbers to have imaginary part 0 so printed results are the same for --with-scalar-type=real or --with-scalar-type=complex */
  CHKERRQ(PetscRandomSetInterval(rctx,zero,one));
  CHKERRQ(PetscRandomSetFromOptions(rctx));
  for (i=0; i<(nr * nc); i++) {
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_WORLD,arow[i],acol[i],NULL,&subs[i]));
  }
  CHKERRQ(MatCreateNest(PETSC_COMM_WORLD,nr,NULL,nc,NULL,subs,&A));
  CHKERRQ(MatCreateVecs(A, &x, NULL));
  CHKERRQ(MatCreateVecs(A, NULL, &y));
  CHKERRQ(VecDuplicate(x, &ATy));
  CHKERRQ(VecDuplicate(y, &Ax));
  CHKERRQ(MatSetRandom(A,rctx));
  CHKERRQ(MatTranspose(A, MAT_INITIAL_MATRIX, &Atranspose));

  CHKERRQ(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatNestGetSubMats(A, NULL, NULL, &block));
  for (i=0; i<nr; i++) {
    for (j=0; j<nc; j++) {
      CHKERRQ(MatView(block[i][j], PETSC_VIEWER_STDOUT_WORLD));
    }
  }

  CHKERRQ(MatView(Atranspose, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatNestGetSubMats(Atranspose, NULL, NULL, &block));
  for (i=0; i<nc; i++) {
    for (j=0; j<nr; j++) {
      CHKERRQ(MatView(block[i][j], PETSC_VIEWER_STDOUT_WORLD));
    }
  }

  /* Check <Ax, y> = <x, A^Ty> */
  for (i=0; i<10; i++) {
    CHKERRQ(VecSetRandom(x,rctx));
    CHKERRQ(VecSetRandom(y,rctx));

    CHKERRQ(MatMult(A, x, Ax));
    CHKERRQ(VecDot(Ax, y, &dot1));
    CHKERRQ(MatMult(Atranspose, y, ATy));
    CHKERRQ(VecDot(ATy, x, &dot2));

    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "<Ax, y> = %g\n", (double)PetscRealPart(dot1)));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "<x, A^Ty> = %g\n",(double)PetscRealPart(dot2)));
  }
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&Ax));

  CHKERRQ(MatCreateSeqDense(PETSC_COMM_WORLD,acol[0]+acol[nr]+acol[2*nr],nk,NULL,&B));
  CHKERRQ(MatSetRandom(B,rctx));
  CHKERRQ(MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C));
  CHKERRQ(MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C));
  CHKERRQ(MatMatMultEqual(A,B,C,10,&equal));
  PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in C != A*B");

  CHKERRQ(MatGetSize(A,&M,&N));
  CHKERRQ(MatGetLocalSize(A,&m,&n));
  for (i=0; i<nk; i++) {
    CHKERRQ(MatDenseGetColumn(B,i,&valsB));
    CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,N,valsB,&x));
    CHKERRQ(MatCreateVecs(A,NULL,&Ax));
    CHKERRQ(MatMult(A,x,Ax));
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(MatDenseGetColumn(C,i,&valsC));
    CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,m,M,valsC,&y));
    CHKERRQ(VecAXPY(y,-1.0,Ax));
    CHKERRQ(VecDestroy(&Ax));
    CHKERRQ(VecDestroy(&y));
    CHKERRQ(MatDenseRestoreColumn(C,&valsC));
    CHKERRQ(MatDenseRestoreColumn(B,&valsB));
  }
  CHKERRQ(MatNorm(C,NORM_INFINITY,&norm));
  PetscCheckFalse(norm > PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatMatMult(): %g",(double)norm);
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&B));

  for (i=0; i<(nr * nc); i++) {
    CHKERRQ(MatDestroy(&subs[i]));
  }
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&Atranspose));
  CHKERRQ(VecDestroy(&ATy));
  CHKERRQ(PetscRandomDestroy(&rctx));
  PetscFunctionReturn(0);
}

PetscErrorCode TestReuseMatrix(void)
{
  const PetscInt  n = 2;
  Mat             A;
  Mat             subs[2*2],**block;
  PetscInt        i,j;
  PetscRandom     rctx;
  PetscScalar     zero = 0.0, one = 1.0;

  PetscFunctionBegin;
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  CHKERRQ(PetscRandomSetInterval(rctx,zero,one));
  CHKERRQ(PetscRandomSetFromOptions(rctx));
  for (i=0; i<(n * n); i++) {
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_WORLD,n,n,NULL,&subs[i]));
  }
  CHKERRQ(MatCreateNest(PETSC_COMM_WORLD,n,NULL,n,NULL,subs,&A));
  CHKERRQ(MatSetRandom(A,rctx));

  CHKERRQ(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatNestGetSubMats(A, NULL, NULL, &block));
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      CHKERRQ(MatView(block[i][j], PETSC_VIEWER_STDOUT_WORLD));
    }
  }
  CHKERRQ(MatTranspose(A,MAT_INPLACE_MATRIX,&A));
  CHKERRQ(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatNestGetSubMats(A, NULL, NULL, &block));
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      CHKERRQ(MatView(block[i][j], PETSC_VIEWER_STDOUT_WORLD));
    }
  }

  for (i=0; i<(n * n); i++) {
    CHKERRQ(MatDestroy(&subs[i]));
  }
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscRandomDestroy(&rctx));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode      ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(TestInitialMatrix());
  CHKERRQ(TestReuseMatrix());
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -malloc_dump

TEST*/
