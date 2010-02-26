
static char help[] = "Test MatSolveTranspose() for BAIJ matrix  -f <input_file> : file to load \n\n";

/* 
  Include "petscmat.h" so that we can use matrices.
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h    - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscviewer.h - viewers               
*/
#include "petscmat.h"
#include "private/matimpl.h"
extern PetscErrorCode MatSolveTranspose_SeqBAIJ_N(Mat,Vec,Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat                   A,F;
  PetscViewer           fd;               /* viewer */
  char                  file[PETSC_MAX_PATH_LEN];     /* input file name */
  PetscErrorCode        ierr;
  PetscTruth            flg;
  Vec                   x,y,w;
  MatFactorInfo         iluinfo;
  IS                    perm;
  PetscInt              m;
  PetscReal             norm;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* 
     Determine file from which we read the matrix

  */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(1,"Must indicate binary file with the -f option");


  /* 
     Open binary file.  Note that we use FILE_MODE_READ to indicate
     reading from this file.
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);

  /*
    Load the matrix; then destroy the viewer.
  */
  ierr = MatLoad(fd,MATSEQBAIJ,&A);CHKERRQ(ierr);
  ierr = VecLoad(fd,VECSEQ,&x);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);
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
  ierr = ISDestroy(perm);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(F);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);
  ierr = VecDestroy(w);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
