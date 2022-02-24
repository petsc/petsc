static const char help[] = "Tests MatCreateSubMatrix with MatSubMatrix versus MatAIJ, square, shifted (copied from ex97)\n";

#include <petscmat.h>

static PetscErrorCode AssembleMatrix(MPI_Comm comm,Mat *A)
{
  Mat            B;
  PetscInt       i,ms,me;

  PetscFunctionBegin;
  CHKERRQ(MatCreate(comm,&B));
  CHKERRQ(MatSetSizes(B,6,6,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));
  CHKERRQ(MatGetOwnershipRange(B,&ms,&me));
  for (i=ms; i<me; i++) {
    CHKERRQ(MatSetValue(B,i,i,1.0*i,INSERT_VALUES));
  }
  CHKERRQ(MatSetValue(B,me-1,me-1,me*me,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  *A   = B;
  PetscFunctionReturn(0);
}

static PetscErrorCode Compare2(Vec *X,const char *test)
{
  PetscReal      norm;
  Vec            Y;
  PetscInt       verbose = 0;

  PetscFunctionBegin;
  CHKERRQ(VecDuplicate(X[0],&Y));
  CHKERRQ(VecCopy(X[0],Y));
  CHKERRQ(VecAYPX(Y,-1.0,X[1]));
  CHKERRQ(VecNorm(Y,NORM_INFINITY,&norm));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-verbose",&verbose,NULL));
  if (norm < PETSC_SQRT_MACHINE_EPSILON && verbose < 1) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%30s: norm difference < sqrt(eps_machine)\n",test));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%30s: norm difference %g\n",test,(double)norm));
  }
  if (verbose > 1) {
    CHKERRQ(VecView(X[0],PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(VecView(X[1],PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(VecDestroy(&Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckMatrices(Mat A,Mat B,Vec left,Vec right,Vec X,Vec Y,Vec X1,Vec Y1)
{
  Vec            *ltmp,*rtmp;

  PetscFunctionBegin;
  CHKERRQ(VecDuplicateVecs(right,2,&rtmp));
  CHKERRQ(VecDuplicateVecs(left,2,&ltmp));
  CHKERRQ(MatScale(A,PETSC_PI));
  CHKERRQ(MatScale(B,PETSC_PI));
  CHKERRQ(MatDiagonalScale(A,left,right));
  CHKERRQ(MatDiagonalScale(B,left,right));
  CHKERRQ(MatShift(A,PETSC_PI));
  CHKERRQ(MatShift(B,PETSC_PI));

  CHKERRQ(MatMult(A,X,ltmp[0]));
  CHKERRQ(MatMult(B,X,ltmp[1]));
  CHKERRQ(Compare2(ltmp,"MatMult"));

  CHKERRQ(MatMultTranspose(A,Y,rtmp[0]));
  CHKERRQ(MatMultTranspose(B,Y,rtmp[1]));
  CHKERRQ(Compare2(rtmp,"MatMultTranspose"));

  CHKERRQ(VecCopy(Y1,ltmp[0]));
  CHKERRQ(VecCopy(Y1,ltmp[1]));
  CHKERRQ(MatMultAdd(A,X,ltmp[0],ltmp[0]));
  CHKERRQ(MatMultAdd(B,X,ltmp[1],ltmp[1]));
  CHKERRQ(Compare2(ltmp,"MatMultAdd v2==v3"));

  CHKERRQ(MatMultAdd(A,X,Y1,ltmp[0]));
  CHKERRQ(MatMultAdd(B,X,Y1,ltmp[1]));
  CHKERRQ(Compare2(ltmp,"MatMultAdd v2!=v3"));

  CHKERRQ(VecCopy(X1,rtmp[0]));
  CHKERRQ(VecCopy(X1,rtmp[1]));
  CHKERRQ(MatMultTransposeAdd(A,Y,rtmp[0],rtmp[0]));
  CHKERRQ(MatMultTransposeAdd(B,Y,rtmp[1],rtmp[1]));
  CHKERRQ(Compare2(rtmp,"MatMultTransposeAdd v2==v3"));

  CHKERRQ(MatMultTransposeAdd(A,Y,X1,rtmp[0]));
  CHKERRQ(MatMultTransposeAdd(B,Y,X1,rtmp[1]));
  CHKERRQ(Compare2(rtmp,"MatMultTransposeAdd v2!=v3"));

  CHKERRQ(VecDestroyVecs(2,&ltmp));
  CHKERRQ(VecDestroyVecs(2,&rtmp));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  Mat            A,B,Asub,Bsub;
  PetscInt       ms,idxrow[3],idxcol[3];
  Vec            left,right,X,Y,X1,Y1;
  IS             isrow,iscol;
  PetscBool      random = PETSC_TRUE;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  CHKERRQ(AssembleMatrix(PETSC_COMM_WORLD,&A));
  CHKERRQ(AssembleMatrix(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetOperation(B,MATOP_CREATE_SUBMATRIX,NULL));
  CHKERRQ(MatSetOperation(B,MATOP_CREATE_SUBMATRICES,NULL));
  CHKERRQ(MatGetOwnershipRange(A,&ms,NULL));

  idxrow[0] = ms+1;
  idxrow[1] = ms+2;
  idxrow[2] = ms+4;
  CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,3,idxrow,PETSC_USE_POINTER,&isrow));

  idxcol[0] = ms+1;
  idxcol[1] = ms+2;
  idxcol[2] = ms+4;
  CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,3,idxcol,PETSC_USE_POINTER,&iscol));

  CHKERRQ(MatCreateSubMatrix(A,isrow,iscol,MAT_INITIAL_MATRIX,&Asub));
  CHKERRQ(MatCreateSubMatrix(B,isrow,iscol,MAT_INITIAL_MATRIX,&Bsub));

  CHKERRQ(MatCreateVecs(Asub,&right,&left));
  CHKERRQ(VecDuplicate(right,&X));
  CHKERRQ(VecDuplicate(right,&X1));
  CHKERRQ(VecDuplicate(left,&Y));
  CHKERRQ(VecDuplicate(left,&Y1));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-random",&random,NULL));
  if (random) {
    CHKERRQ(VecSetRandom(right,NULL));
    CHKERRQ(VecSetRandom(left,NULL));
    CHKERRQ(VecSetRandom(X,NULL));
    CHKERRQ(VecSetRandom(Y,NULL));
    CHKERRQ(VecSetRandom(X1,NULL));
    CHKERRQ(VecSetRandom(Y1,NULL));
  } else {
    CHKERRQ(VecSet(right,1.0));
    CHKERRQ(VecSet(left,2.0));
    CHKERRQ(VecSet(X,3.0));
    CHKERRQ(VecSet(Y,4.0));
    CHKERRQ(VecSet(X1,3.0));
    CHKERRQ(VecSet(Y1,4.0));
  }
  CHKERRQ(CheckMatrices(Asub,Bsub,left,right,X,Y,X1,Y1));
  CHKERRQ(ISDestroy(&isrow));
  CHKERRQ(ISDestroy(&iscol));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&Asub));
  CHKERRQ(MatDestroy(&Bsub));
  CHKERRQ(VecDestroy(&left));
  CHKERRQ(VecDestroy(&right));
  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&Y));
  CHKERRQ(VecDestroy(&X1));
  CHKERRQ(VecDestroy(&Y1));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 3

TEST*/
