static const char help[] = "Tests MatCreateSubMatrix with MatSubMatrix versus MatAIJ, non-square\n";

#include <petscmat.h>

static PetscErrorCode AssembleMatrix(MPI_Comm comm,Mat *A)
{
  PetscErrorCode ierr;
  Mat            B;
  PetscInt       i,ms,me;

  PetscFunctionBegin;
  ierr = MatCreate(comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,5,6,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(B,&ms,&me);CHKERRQ(ierr);
  for (i=ms; i<me; i++) {
    ierr = MatSetValue(B,i,i,1.0*i,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatSetValue(B,me-1,me,me*me,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *A   = B;
  PetscFunctionReturn(0);
}

static PetscErrorCode Compare2(Vec *X,const char *test)
{
  PetscErrorCode ierr;
  PetscReal      norm;
  Vec            Y;
  PetscInt       verbose = 0;

  PetscFunctionBegin;
  ierr = VecDuplicate(X[0],&Y);CHKERRQ(ierr);
  ierr = VecCopy(X[0],Y);CHKERRQ(ierr);
  ierr = VecAYPX(Y,-1.0,X[1]);CHKERRQ(ierr);
  ierr = VecNorm(Y,NORM_INFINITY,&norm);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(NULL,NULL,"-verbose",&verbose,NULL);CHKERRQ(ierr);
  if (norm < PETSC_SQRT_MACHINE_EPSILON && verbose < 1) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%30s: norm difference < sqrt(eps_machine)\n",test);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%30s: norm difference %g\n",test,(double)norm);CHKERRQ(ierr);
  }
  if (verbose > 1) {
    ierr = VecView(X[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecView(X[1],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecView(Y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckMatrices(Mat A,Mat B,Vec left,Vec right,Vec X,Vec Y,Vec X1,Vec Y1)
{
  PetscErrorCode ierr;
  Vec            *ltmp,*rtmp;

  PetscFunctionBegin;
  ierr = VecDuplicateVecs(right,2,&rtmp);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(left,2,&ltmp);CHKERRQ(ierr);
  ierr = MatScale(A,PETSC_PI);CHKERRQ(ierr);
  ierr = MatScale(B,PETSC_PI);CHKERRQ(ierr);
  ierr = MatDiagonalScale(A,left,right);CHKERRQ(ierr);
  ierr = MatDiagonalScale(B,left,right);CHKERRQ(ierr);

  ierr = MatMult(A,X,ltmp[0]);CHKERRQ(ierr);
  ierr = MatMult(B,X,ltmp[1]);CHKERRQ(ierr);
  ierr = Compare2(ltmp,"MatMult");CHKERRQ(ierr);

  ierr = MatMultTranspose(A,Y,rtmp[0]);CHKERRQ(ierr);
  ierr = MatMultTranspose(B,Y,rtmp[1]);CHKERRQ(ierr);
  ierr = Compare2(rtmp,"MatMultTranspose");CHKERRQ(ierr);

  ierr = VecCopy(Y1,ltmp[0]);CHKERRQ(ierr);
  ierr = VecCopy(Y1,ltmp[1]);CHKERRQ(ierr);
  ierr = MatMultAdd(A,X,ltmp[0],ltmp[0]);CHKERRQ(ierr);
  ierr = MatMultAdd(B,X,ltmp[1],ltmp[1]);CHKERRQ(ierr);
  ierr = Compare2(ltmp,"MatMultAdd v2==v3");CHKERRQ(ierr);

  ierr = MatMultAdd(A,X,Y1,ltmp[0]);CHKERRQ(ierr);
  ierr = MatMultAdd(B,X,Y1,ltmp[1]);CHKERRQ(ierr);
  ierr = Compare2(ltmp,"MatMultAdd v2!=v3");CHKERRQ(ierr);

  ierr = VecCopy(X1,rtmp[0]);CHKERRQ(ierr);
  ierr = VecCopy(X1,rtmp[1]);CHKERRQ(ierr);
  ierr = MatMultTransposeAdd(A,Y,rtmp[0],rtmp[0]);CHKERRQ(ierr);
  ierr = MatMultTransposeAdd(B,Y,rtmp[1],rtmp[1]);CHKERRQ(ierr);
  ierr = Compare2(rtmp,"MatMultTransposeAdd v2==v3");CHKERRQ(ierr);

  ierr = MatMultTransposeAdd(A,Y,X1,rtmp[0]);CHKERRQ(ierr);
  ierr = MatMultTransposeAdd(B,Y,X1,rtmp[1]);CHKERRQ(ierr);
  ierr = Compare2(rtmp,"MatMultTransposeAdd v2!=v3");CHKERRQ(ierr);

  ierr = VecDestroyVecs(2,&ltmp);CHKERRQ(ierr);
  ierr = VecDestroyVecs(2,&rtmp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  Mat            A,B,Asub,Bsub;
  PetscInt       ms,idxrow[3],idxcol[4];
  Vec            left,right,X,Y,X1,Y1;
  IS             isrow,iscol;
  PetscBool      random = PETSC_TRUE;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = AssembleMatrix(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = AssembleMatrix(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetOperation(B,MATOP_CREATE_SUBMATRIX,NULL);CHKERRQ(ierr);
  ierr = MatSetOperation(B,MATOP_CREATE_SUBMATRICES,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&ms,NULL);CHKERRQ(ierr);

  idxrow[0] = ms+1;
  idxrow[1] = ms+2;
  idxrow[2] = ms+4;
  ierr      = ISCreateGeneral(PETSC_COMM_WORLD,3,idxrow,PETSC_USE_POINTER,&isrow);CHKERRQ(ierr);

  idxcol[0] = ms+1;
  idxcol[1] = ms+2;
  idxcol[2] = ms+4;
  idxcol[3] = ms+5;
  ierr      = ISCreateGeneral(PETSC_COMM_WORLD,4,idxcol,PETSC_USE_POINTER,&iscol);CHKERRQ(ierr);

  ierr = MatCreateSubMatrix(A,isrow,iscol,MAT_INITIAL_MATRIX,&Asub);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(B,isrow,iscol,MAT_INITIAL_MATRIX,&Bsub);CHKERRQ(ierr);

  ierr = MatCreateVecs(Asub,&right,&left);CHKERRQ(ierr);
  ierr = VecDuplicate(right,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(right,&X1);CHKERRQ(ierr);
  ierr = VecDuplicate(left,&Y);CHKERRQ(ierr);
  ierr = VecDuplicate(left,&Y1);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(NULL,NULL,"-random",&random,NULL);CHKERRQ(ierr);
  if (random) {
    ierr = VecSetRandom(right,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(left,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(X,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(Y,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(X1,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(Y1,NULL);CHKERRQ(ierr);
  } else {
    ierr = VecSet(right,1.0);CHKERRQ(ierr);
    ierr = VecSet(left,2.0);CHKERRQ(ierr);
    ierr = VecSet(X,3.0);CHKERRQ(ierr);
    ierr = VecSet(Y,4.0);CHKERRQ(ierr);
    ierr = VecSet(X1,3.0);CHKERRQ(ierr);
    ierr = VecSet(Y1,4.0);CHKERRQ(ierr);
  }
  ierr = CheckMatrices(Asub,Bsub,left,right,X,Y,X1,Y1);CHKERRQ(ierr);
  ierr = ISDestroy(&isrow);CHKERRQ(ierr);
  ierr = ISDestroy(&iscol);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&Asub);CHKERRQ(ierr);
  ierr = MatDestroy(&Bsub);CHKERRQ(ierr);
  ierr = VecDestroy(&left);CHKERRQ(ierr);
  ierr = VecDestroy(&right);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  ierr = VecDestroy(&X1);CHKERRQ(ierr);
  ierr = VecDestroy(&Y1);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   test:
      nsize: 3

TEST*/
