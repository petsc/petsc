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
  PetscErrorCode  ierr;
  PetscBool       equal;

  PetscFunctionBegin;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  /* Force the random numbers to have imaginary part 0 so printed results are the same for --with-scalar-type=real or --with-scalar-type=complex */
  ierr = PetscRandomSetInterval(rctx,zero,one);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  for (i=0; i<(nr * nc); i++) {
    ierr = MatCreateSeqDense(PETSC_COMM_WORLD,arow[i],acol[i],NULL,&subs[i]);CHKERRQ(ierr);
  }
  ierr = MatCreateNest(PETSC_COMM_WORLD,nr,NULL,nc,NULL,subs,&A);CHKERRQ(ierr);
  ierr = MatCreateVecs(A, &x, NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(A, NULL, &y);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &ATy);CHKERRQ(ierr);
  ierr = VecDuplicate(y, &Ax);CHKERRQ(ierr);
  ierr = MatSetRandom(A,rctx);CHKERRQ(ierr);
  ierr = MatTranspose(A, MAT_INITIAL_MATRIX, &Atranspose);CHKERRQ(ierr);

  ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatNestGetSubMats(A, NULL, NULL, &block);CHKERRQ(ierr);
  for (i=0; i<nr; i++) {
    for (j=0; j<nc; j++) {
      ierr = MatView(block[i][j], PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  }

  ierr = MatView(Atranspose, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatNestGetSubMats(Atranspose, NULL, NULL, &block);CHKERRQ(ierr);
  for (i=0; i<nc; i++) {
    for (j=0; j<nr; j++) {
      ierr = MatView(block[i][j], PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  }

  /* Check <Ax, y> = <x, A^Ty> */
  for (i=0; i<10; i++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(y,rctx);CHKERRQ(ierr);

    ierr = MatMult(A, x, Ax);CHKERRQ(ierr);
    ierr = VecDot(Ax, y, &dot1);CHKERRQ(ierr);
    ierr = MatMult(Atranspose, y, ATy);CHKERRQ(ierr);
    ierr = VecDot(ATy, x, &dot2);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD, "<Ax, y> = %g\n", (double)PetscRealPart(dot1));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "<x, A^Ty> = %g\n",(double)PetscRealPart(dot2));CHKERRQ(ierr);
  }
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&Ax);CHKERRQ(ierr);

  ierr = MatCreateSeqDense(PETSC_COMM_WORLD,acol[0]+acol[nr]+acol[2*nr],nk,NULL,&B);CHKERRQ(ierr);
  ierr = MatSetRandom(B,rctx);CHKERRQ(ierr);
  ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C);CHKERRQ(ierr);
  ierr = MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C);CHKERRQ(ierr);
  ierr = MatMatMultEqual(A,B,C,10,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in C != A*B");

  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  for (i=0; i<nk; i++) {
    ierr = MatDenseGetColumn(B,i,&valsB);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,N,valsB,&x);CHKERRQ(ierr);
    ierr = MatCreateVecs(A,NULL,&Ax);CHKERRQ(ierr);
    ierr = MatMult(A,x,Ax);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = MatDenseGetColumn(C,i,&valsC);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,m,M,valsC,&y);CHKERRQ(ierr);
    ierr = VecAXPY(y,-1.0,Ax);CHKERRQ(ierr);
    ierr = VecDestroy(&Ax);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumn(C,&valsC);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumn(B,&valsB);CHKERRQ(ierr);
  }
  ierr = MatNorm(C,NORM_INFINITY,&norm);CHKERRQ(ierr);
  if (norm > PETSC_SMALL) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in MatMatMult(): %g",(double)norm);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  for (i=0; i<(nr * nc); i++) {
    ierr = MatDestroy(&subs[i]);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Atranspose);CHKERRQ(ierr);
  ierr = VecDestroy(&ATy);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TestReuseMatrix(void)
{
  const PetscInt  n = 2;
  Mat             A;
  Mat             subs[2*2],**block;
  PetscInt        i,j;
  PetscRandom     rctx;
  PetscErrorCode  ierr;
  PetscScalar     zero = 0.0, one = 1.0;

  PetscFunctionBegin;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rctx,zero,one);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  for (i=0; i<(n * n); i++) {
    ierr = MatCreateSeqDense(PETSC_COMM_WORLD,n,n,NULL,&subs[i]);CHKERRQ(ierr);
  }
  ierr = MatCreateNest(PETSC_COMM_WORLD,n,NULL,n,NULL,subs,&A);CHKERRQ(ierr);
  ierr = MatSetRandom(A,rctx);CHKERRQ(ierr);

  ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatNestGetSubMats(A, NULL, NULL, &block);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      ierr = MatView(block[i][j], PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  }
  ierr = MatTranspose(A,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatNestGetSubMats(A, NULL, NULL, &block);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      ierr = MatView(block[i][j], PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  }

  for (i=0; i<(n * n); i++) {
    ierr = MatDestroy(&subs[i]);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode      ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = TestInitialMatrix();CHKERRQ(ierr);
  ierr = TestReuseMatrix();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -malloc_dump

TEST*/
