static char help[] = 
"Reads in a binary file, extracts a submatrix from it,\n\
and writes to another binary file\n\
  -fin  <mat>  : input matrix file\n\
  -fout <mat>  : output marrix file\n\
  -start <row> : the row from where the submat should be extracted\n\
  -size  <sx>  : the size of the submatrix\n";

#include "mat.h"
#include "vec.h"

int main(int argc,char **args)
{  
  char       fin[128],fout[128] ="default.mat";
  Viewer     fdin,fdout;               
  Vec        b;   
  MatType    mtype = MATSEQBAIJ;            
  Mat        A,*B;             
  int        flg,ierr,start=0,size;
  IS         isrow,iscol;

  PetscInitialize(&argc,&args,(char *)0,help);


  ierr = OptionsGetString(PETSC_NULL,"-fin",fin,127,&flg); CHKERRA(ierr);
  if (!flg) SETERRA(1,0,"Must indicate binary file with the -fin option");
  ierr = ViewerFileOpenBinary(MPI_COMM_SELF,fin,BINARY_RDONLY,&fdin);CHKERRA(ierr);

  ierr = OptionsGetString(PETSC_NULL,"-fout",fout,127,&flg); CHKERRA(ierr);
  if (!flg) PetscPrintf(MPI_COMM_WORLD,"Writing submatrix to file : %s\n",fout);
  ierr = ViewerFileOpenBinary(MPI_COMM_SELF,fout,BINARY_CREATE,&fdout);CHKERRA(ierr);

  ierr = MatLoad(fdin,mtype,&A); CHKERRA(ierr);
  ierr = ViewerDestroy(fdin); CHKERRA(ierr);
  
  ierr = MatGetSize(A,&size,&size);  CHKERRQ(ierr);
  size /= 2;
  ierr = OptionsGetInt(PETSC_NULL,"-start",&start,&flg); CHKERRQ(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-size", &size,&flg); CHKERRQ(ierr);
  
  ierr = ISCreateStride(MPI_COMM_SELF,size,start,1,&isrow); CHKERRQ(ierr);
  ierr = ISCreateStride(MPI_COMM_SELF,size,start,1,&iscol); CHKERRQ(ierr);
  ierr = MatGetSubMatrices(A,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&B); CHKERRQ(ierr);
  ierr = MatView(B[0],fdout); CHKERRQ(ierr);

  ierr = VecCreate(MPI_COMM_SELF,size,&b); CHKERRQ(ierr);
  ierr = MatView(B[0],fdout); CHKERRQ(ierr);
  ierr = ViewerDestroy(fdout); CHKERRA(ierr);

  MatDestroy(A); 
  MatDestroy(B[0]);
  VecDestroy(b);
  PetscFree(B);
  ISDestroy(iscol);
  ISDestroy(isrow);
  PetscFinalize();
  return 0;
}

