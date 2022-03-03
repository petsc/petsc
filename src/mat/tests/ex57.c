
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
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-fin",fin,sizeof(fin),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -fin option");
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_SELF,fin,FILE_MODE_READ,&fdin));

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-fout",fout,sizeof(fout),&flg));
  if (!flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Writing submatrix to file : %s\n",fout));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_SELF,fout,FILE_MODE_WRITE,&fdout));

  CHKERRQ(MatCreate(PETSC_COMM_SELF,&A));
  CHKERRQ(MatSetType(A,mtype));
  CHKERRQ(MatLoad(A,fdin));
  CHKERRQ(PetscViewerDestroy(&fdin));

  CHKERRQ(MatGetSize(A,&m,&m));
  m /= 2;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-start",&start,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));

  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,m,start,1,&isrow));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,m,start,1,&iscol));
  CHKERRQ(MatCreateSubMatrices(A,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&B));
  CHKERRQ(MatView(B[0],fdout));

  CHKERRQ(VecCreate(PETSC_COMM_SELF,&b));
  CHKERRQ(VecSetSizes(b,PETSC_DECIDE,m));
  CHKERRQ(VecSetFromOptions(b));
  CHKERRQ(MatView(B[0],fdout));
  CHKERRQ(PetscViewerDestroy(&fdout));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B[0]));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(PetscFree(B));
  CHKERRQ(ISDestroy(&iscol));
  CHKERRQ(ISDestroy(&isrow));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -fin ${DATAFILESPATH}/matrices/small -fout joe -start 2 -m 4
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/
