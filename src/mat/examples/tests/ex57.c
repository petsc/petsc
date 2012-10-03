
static char help[] = "Reads in a binary file, extracts a submatrix from it, and writes to another binary file.\n\
Options:\n\
  -fin  <mat>  : input matrix file\n\
  -fout <mat>  : output marrix file\n\
  -start <row> : the row from where the submat should be extracted\n\
  -size  <sx>  : the size of the submatrix\n";

#include <petscmat.h>
#include <petscvec.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  char           fin[PETSC_MAX_PATH_LEN],fout[PETSC_MAX_PATH_LEN] ="default.mat";
  PetscViewer    fdin,fdout;
  Vec            b;
  MatType        mtype = MATSEQBAIJ;
  Mat            A,*B;
  PetscErrorCode ierr;
  PetscInt       start=0;
  PetscMPIInt    size;
  IS             isrow,iscol;
  PetscBool      flg;

  PetscInitialize(&argc,&args,(char *)0,help);


  ierr = PetscOptionsGetString(PETSC_NULL,"-fin",fin,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -fin option");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,fin,FILE_MODE_READ,&fdin);CHKERRQ(ierr);

  ierr = PetscOptionsGetString(PETSC_NULL,"-fout",fout,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) {ierr = PetscPrintf(PETSC_COMM_WORLD,"Writing submatrix to file : %s\n",fout);CHKERRQ(ierr);}
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,fout,FILE_MODE_WRITE,&fdout);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,mtype);CHKERRQ(ierr);
  ierr = MatLoad(A,fdin);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fdin);CHKERRQ(ierr);

  ierr = MatGetSize(A,&size,&size);CHKERRQ(ierr);
  size /= 2;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-start",&start,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-size",&size,PETSC_NULL);CHKERRQ(ierr);

  ierr = ISCreateStride(PETSC_COMM_SELF,size,start,1,&isrow);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,size,start,1,&iscol);CHKERRQ(ierr);
  ierr = MatGetSubMatrices(A,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatView(B[0],fdout);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_SELF,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,size);CHKERRQ(ierr);
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
  return 0;
}

