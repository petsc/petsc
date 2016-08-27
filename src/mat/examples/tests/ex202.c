static char help[] = "Tests the use of MatTranspose_Nest\n";

#include <petscmat.h>

#define nr  (2)
#define nc  (3)

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  const PetscInt      arow[nr*nc] = { 2, 2, 2, 3, 3, 3 };
  const PetscInt      acol[nr*nc] = { 3, 2, 4, 3, 2, 4 };
  Mat                 A,Atranspose;
  Mat                 subs[nr*nc], **block;
  Vec                 x,y,Ax,ATy;
  PetscInt            i,j;
  PetscScalar         dot1,dot2;
  PetscRandom         rctx;
  PetscErrorCode      ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
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

    ierr = PetscPrintf(PETSC_COMM_WORLD, "<Ax, y> = %g\n", dot1);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "<x, A^Ty> = %g\n", dot2);CHKERRQ(ierr);
  }

  for (i=0; i<(nr * nc); i++) {
    ierr = MatDestroy(&subs[i]);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Atranspose);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&Ax);CHKERRQ(ierr);
  ierr = VecDestroy(&ATy);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
