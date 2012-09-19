
static char help[] = "Test MatFwkAIJ: a block matrix with an AIJ-like datastructure keeping track of nonzero blocks.\n\
Each block is a matrix of (generally) any type.\n\n";

/*
  Include "petscmat.h" so that we can use matrices.
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h    - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscviewer.h - viewers
*/
#include <petscmat.h>
#include <petsc-private/matimpl.h>
extern PetscErrorCode MatSolveTranspose_SeqBAIJ_N(Mat,Vec,Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat                   A,F;
  PetscViewer           fd;               /* viewer */
  char                  file[PETSC_MAX_PATH_LEN];     /* input file name */
  PetscErrorCode        ierr;
  PetscBool             flg;
  Vec                   x,y,w;
  MatFactorInfo         iluinfo;
  IS                    perm;
  PetscInt              m;
  PetscReal             norm;

  PetscInitialize(&argc,&args,(char *)0,help);

  /*
     Determine file from which we read the matrix

  */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f option");


  /*
     Open binary file.  Note that we use FILE_MODE_READ to indicate
     reading from this file.
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);

  /*
    Load the matrix; then destroy the viewer.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecLoad(x,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&w);CHKERRQ(ierr);

  ierr = MatGetFactor(A,"petsc",MAT_FACTOR_ILU,&F);CHKERRQ(ierr);
  iluinfo.fill = 1.0;
  ierr = MatGetSize(A,&m,0);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,m,0,1,&perm);CHKERRQ(ierr);

  ierr = MatLUFactorSymbolic(F,A,perm,perm,&iluinfo);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric(F,A,&iluinfo);CHKERRQ(ierr);
  ierr = MatSolveTranspose(F,x,y);CHKERRQ(ierr);
  F->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_N;
  ierr = MatSolveTranspose(F,x,w);CHKERRQ(ierr);
  //  VecView(w,0);VecView(y,0);
  ierr = VecAXPY(w,-1.0,y);CHKERRQ(ierr);
  ierr = VecNorm(w,NORM_2,&norm);CHKERRQ(ierr);
  if (norm) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Norm of difference is nonzero %g\n",norm);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
