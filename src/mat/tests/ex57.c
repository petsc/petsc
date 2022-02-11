
static char help[] = "Reads in a binary file, extracts a submatrix from it, and writes to another binary file.\n\
Options:\n\
  -fin  <mat>  : input matrix file\n\
  -fout <mat>  : output marrix file\n\
  -start <row> : the row from where the submat should be extracted\n\
  -m  <sx>  : the size of the submatrix\n";

#include <petscmat.h>
#include <petscvec.h>

int main(int argc,char **args)
{
  char           fin[PETSC_MAX_PATH_LEN],fout[PETSC_MAX_PATH_LEN] ="default.mat";
  PetscViewer    fdin,fdout;
  Vec            b;
  MatType        mtype = MATSEQBAIJ;
  Mat            A,*B;
  PetscErrorCode ierr;
  PetscInt       start=0;
  PetscInt       m;
  IS             isrow,iscol;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetString(NULL,NULL,"-fin",fin,sizeof(fin),&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -fin option");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,fin,FILE_MODE_READ,&fdin);CHKERRQ(ierr);

  ierr = PetscOptionsGetString(NULL,NULL,"-fout",fout,sizeof(fout),&flg);CHKERRQ(ierr);
  if (!flg) {ierr = PetscPrintf(PETSC_COMM_WORLD,"Writing submatrix to file : %s\n",fout);CHKERRQ(ierr);}
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,fout,FILE_MODE_WRITE,&fdout);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,mtype);CHKERRQ(ierr);
  ierr = MatLoad(A,fdin);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fdin);CHKERRQ(ierr);

  ierr  = MatGetSize(A,&m,&m);CHKERRQ(ierr);
  m /= 2;
  ierr  = PetscOptionsGetInt(NULL,NULL,"-start",&start,NULL);CHKERRQ(ierr);
  ierr  = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);

  ierr = ISCreateStride(PETSC_COMM_SELF,m,start,1,&isrow);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,m,start,1,&iscol);CHKERRQ(ierr);
  ierr = MatCreateSubMatrices(A,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatView(B[0],fdout);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_SELF,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = MatView(B[0],fdout);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fdout);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = PetscFree(B);CHKERRQ(ierr);
  ierr = ISDestroy(&iscol);CHKERRQ(ierr);
  ierr = ISDestroy(&isrow);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -fin ${DATAFILESPATH}/matrices/small -fout joe -start 2 -m 4
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/

