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
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  /* Force the random numbers to have imaginary part 0 so printed results are the same for --with-scalar-type=real or --with-scalar-type=complex */
  PetscCall(PetscRandomSetInterval(rctx,zero,one));
  PetscCall(PetscRandomSetFromOptions(rctx));
  for (i=0; i<(nr * nc); i++) {
    PetscCall(MatCreateSeqDense(PETSC_COMM_WORLD,arow[i],acol[i],NULL,&subs[i]));
  }
  PetscCall(MatCreateNest(PETSC_COMM_WORLD,nr,NULL,nc,NULL,subs,&A));
  PetscCall(MatCreateVecs(A, &x, NULL));
  PetscCall(MatCreateVecs(A, NULL, &y));
  PetscCall(VecDuplicate(x, &ATy));
  PetscCall(VecDuplicate(y, &Ax));
  PetscCall(MatSetRandom(A,rctx));
  PetscCall(MatTranspose(A, MAT_INITIAL_MATRIX, &Atranspose));

  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatNestGetSubMats(A, NULL, NULL, &block));
  for (i=0; i<nr; i++) {
    for (j=0; j<nc; j++) {
      PetscCall(MatView(block[i][j], PETSC_VIEWER_STDOUT_WORLD));
    }
  }

  PetscCall(MatView(Atranspose, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatNestGetSubMats(Atranspose, NULL, NULL, &block));
  for (i=0; i<nc; i++) {
    for (j=0; j<nr; j++) {
      PetscCall(MatView(block[i][j], PETSC_VIEWER_STDOUT_WORLD));
    }
  }

  /* Check <Ax, y> = <x, A^Ty> */
  for (i=0; i<10; i++) {
    PetscCall(VecSetRandom(x,rctx));
    PetscCall(VecSetRandom(y,rctx));

    PetscCall(MatMult(A, x, Ax));
    PetscCall(VecDot(Ax, y, &dot1));
    PetscCall(MatMult(Atranspose, y, ATy));
    PetscCall(VecDot(ATy, x, &dot2));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "<Ax, y> = %g\n", (double)PetscRealPart(dot1)));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "<x, A^Ty> = %g\n",(double)PetscRealPart(dot2)));
  }
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&Ax));

  PetscCall(MatCreateSeqDense(PETSC_COMM_WORLD,acol[0]+acol[nr]+acol[2*nr],nk,NULL,&B));
  PetscCall(MatSetRandom(B,rctx));
  PetscCall(MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C));
  PetscCall(MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C));
  PetscCall(MatMatMultEqual(A,B,C,10,&equal));
  PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in C != A*B");

  PetscCall(MatGetSize(A,&M,&N));
  PetscCall(MatGetLocalSize(A,&m,&n));
  for (i=0; i<nk; i++) {
    PetscCall(MatDenseGetColumn(B,i,&valsB));
    PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,N,valsB,&x));
    PetscCall(MatCreateVecs(A,NULL,&Ax));
    PetscCall(MatMult(A,x,Ax));
    PetscCall(VecDestroy(&x));
    PetscCall(MatDenseGetColumn(C,i,&valsC));
    PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,m,M,valsC,&y));
    PetscCall(VecAXPY(y,-1.0,Ax));
    PetscCall(VecDestroy(&Ax));
    PetscCall(VecDestroy(&y));
    PetscCall(MatDenseRestoreColumn(C,&valsC));
    PetscCall(MatDenseRestoreColumn(B,&valsB));
  }
  PetscCall(MatNorm(C,NORM_INFINITY,&norm));
  PetscCheckFalse(norm > PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatMatMult(): %g",(double)norm);
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&B));

  for (i=0; i<(nr * nc); i++) {
    PetscCall(MatDestroy(&subs[i]));
  }
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Atranspose));
  PetscCall(VecDestroy(&ATy));
  PetscCall(PetscRandomDestroy(&rctx));
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
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  PetscCall(PetscRandomSetInterval(rctx,zero,one));
  PetscCall(PetscRandomSetFromOptions(rctx));
  for (i=0; i<(n * n); i++) {
    PetscCall(MatCreateSeqDense(PETSC_COMM_WORLD,n,n,NULL,&subs[i]));
  }
  PetscCall(MatCreateNest(PETSC_COMM_WORLD,n,NULL,n,NULL,subs,&A));
  PetscCall(MatSetRandom(A,rctx));

  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatNestGetSubMats(A, NULL, NULL, &block));
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      PetscCall(MatView(block[i][j], PETSC_VIEWER_STDOUT_WORLD));
    }
  }
  PetscCall(MatTranspose(A,MAT_INPLACE_MATRIX,&A));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatNestGetSubMats(A, NULL, NULL, &block));
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      PetscCall(MatView(block[i][j], PETSC_VIEWER_STDOUT_WORLD));
    }
  }

  for (i=0; i<(n * n); i++) {
    PetscCall(MatDestroy(&subs[i]));
  }
  PetscCall(MatDestroy(&A));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(TestInitialMatrix());
  PetscCall(TestReuseMatrix());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -malloc_dump

TEST*/
