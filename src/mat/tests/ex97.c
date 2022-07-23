static const char help[] = "Tests MatCreateSubMatrix with MatSubMatrix versus MatAIJ, non-square\n";

#include <petscmat.h>

static PetscErrorCode AssembleMatrix(MPI_Comm comm,Mat *A)
{
  Mat            B;
  PetscInt       i,ms,me;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm,&B));
  PetscCall(MatSetSizes(B,5,6,PETSC_DETERMINE,PETSC_DETERMINE));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));
  PetscCall(MatGetOwnershipRange(B,&ms,&me));
  for (i=ms; i<me; i++) {
    PetscCall(MatSetValue(B,i,i,1.0*i,INSERT_VALUES));
  }
  PetscCall(MatSetValue(B,me-1,me,me*me,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  *A   = B;
  PetscFunctionReturn(0);
}

static PetscErrorCode Compare2(Vec *X,const char *test)
{
  PetscReal      norm;
  Vec            Y;
  PetscInt       verbose = 0;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(X[0],&Y));
  PetscCall(VecCopy(X[0],Y));
  PetscCall(VecAYPX(Y,-1.0,X[1]));
  PetscCall(VecNorm(Y,NORM_INFINITY,&norm));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-verbose",&verbose,NULL));
  if (norm < PETSC_SQRT_MACHINE_EPSILON && verbose < 1) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%30s: norm difference < sqrt(eps_machine)\n",test));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%30s: norm difference %g\n",test,(double)norm));
  }
  if (verbose > 1) {
    PetscCall(VecView(X[0],PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecView(X[1],PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(VecDestroy(&Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckMatrices(Mat A,Mat B,Vec left,Vec right,Vec X,Vec Y,Vec X1,Vec Y1)
{
  Vec            *ltmp,*rtmp;

  PetscFunctionBegin;
  PetscCall(VecDuplicateVecs(right,2,&rtmp));
  PetscCall(VecDuplicateVecs(left,2,&ltmp));
  PetscCall(MatScale(A,PETSC_PI));
  PetscCall(MatScale(B,PETSC_PI));
  PetscCall(MatDiagonalScale(A,left,right));
  PetscCall(MatDiagonalScale(B,left,right));

  PetscCall(MatMult(A,X,ltmp[0]));
  PetscCall(MatMult(B,X,ltmp[1]));
  PetscCall(Compare2(ltmp,"MatMult"));

  PetscCall(MatMultTranspose(A,Y,rtmp[0]));
  PetscCall(MatMultTranspose(B,Y,rtmp[1]));
  PetscCall(Compare2(rtmp,"MatMultTranspose"));

  PetscCall(VecCopy(Y1,ltmp[0]));
  PetscCall(VecCopy(Y1,ltmp[1]));
  PetscCall(MatMultAdd(A,X,ltmp[0],ltmp[0]));
  PetscCall(MatMultAdd(B,X,ltmp[1],ltmp[1]));
  PetscCall(Compare2(ltmp,"MatMultAdd v2==v3"));

  PetscCall(MatMultAdd(A,X,Y1,ltmp[0]));
  PetscCall(MatMultAdd(B,X,Y1,ltmp[1]));
  PetscCall(Compare2(ltmp,"MatMultAdd v2!=v3"));

  PetscCall(VecCopy(X1,rtmp[0]));
  PetscCall(VecCopy(X1,rtmp[1]));
  PetscCall(MatMultTransposeAdd(A,Y,rtmp[0],rtmp[0]));
  PetscCall(MatMultTransposeAdd(B,Y,rtmp[1],rtmp[1]));
  PetscCall(Compare2(rtmp,"MatMultTransposeAdd v2==v3"));

  PetscCall(MatMultTransposeAdd(A,Y,X1,rtmp[0]));
  PetscCall(MatMultTransposeAdd(B,Y,X1,rtmp[1]));
  PetscCall(Compare2(rtmp,"MatMultTransposeAdd v2!=v3"));

  PetscCall(VecDestroyVecs(2,&ltmp));
  PetscCall(VecDestroyVecs(2,&rtmp));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  Mat            A,B,Asub,Bsub;
  PetscInt       ms,idxrow[3],idxcol[4];
  Vec            left,right,X,Y,X1,Y1;
  IS             isrow,iscol;
  PetscBool      random = PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCall(AssembleMatrix(PETSC_COMM_WORLD,&A));
  PetscCall(AssembleMatrix(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetOperation(B,MATOP_CREATE_SUBMATRIX,NULL));
  PetscCall(MatSetOperation(B,MATOP_CREATE_SUBMATRICES,NULL));
  PetscCall(MatGetOwnershipRange(A,&ms,NULL));

  idxrow[0] = ms+1;
  idxrow[1] = ms+2;
  idxrow[2] = ms+4;
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,3,idxrow,PETSC_USE_POINTER,&isrow));

  idxcol[0] = ms+1;
  idxcol[1] = ms+2;
  idxcol[2] = ms+4;
  idxcol[3] = ms+5;
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,4,idxcol,PETSC_USE_POINTER,&iscol));

  PetscCall(MatCreateSubMatrix(A,isrow,iscol,MAT_INITIAL_MATRIX,&Asub));
  PetscCall(MatCreateSubMatrix(B,isrow,iscol,MAT_INITIAL_MATRIX,&Bsub));

  PetscCall(MatCreateVecs(Asub,&right,&left));
  PetscCall(VecDuplicate(right,&X));
  PetscCall(VecDuplicate(right,&X1));
  PetscCall(VecDuplicate(left,&Y));
  PetscCall(VecDuplicate(left,&Y1));

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-random",&random,NULL));
  if (random) {
    PetscCall(VecSetRandom(right,NULL));
    PetscCall(VecSetRandom(left,NULL));
    PetscCall(VecSetRandom(X,NULL));
    PetscCall(VecSetRandom(Y,NULL));
    PetscCall(VecSetRandom(X1,NULL));
    PetscCall(VecSetRandom(Y1,NULL));
  } else {
    PetscCall(VecSet(right,1.0));
    PetscCall(VecSet(left,2.0));
    PetscCall(VecSet(X,3.0));
    PetscCall(VecSet(Y,4.0));
    PetscCall(VecSet(X1,3.0));
    PetscCall(VecSet(Y1,4.0));
  }
  PetscCall(CheckMatrices(Asub,Bsub,left,right,X,Y,X1,Y1));
  PetscCall(ISDestroy(&isrow));
  PetscCall(ISDestroy(&iscol));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&Asub));
  PetscCall(MatDestroy(&Bsub));
  PetscCall(VecDestroy(&left));
  PetscCall(VecDestroy(&right));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&Y));
  PetscCall(VecDestroy(&X1));
  PetscCall(VecDestroy(&Y1));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3

TEST*/
