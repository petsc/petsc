#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex57.c,v 1.9 1999/01/31 16:07:27 bsmith Exp bsmith $";
#endif

static char help[] = "Reads in a binary file, extracts a submatrix from it, and writes to another\
 binary file.\n\
Options:\n\
  -fin  <mat>  : input matrix file\n\
  -fout <mat>  : output marrix file\n\
  -start <row> : the row from where the submat should be extracted\n\
  -size  <sx>  : the size of the submatrix\n";

#include "mat.h"
#include "vec.h"

#undef __FUNC__
#define __FUNC__ "main"
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
  ierr = ViewerBinaryOpen(PETSC_COMM_SELF,fin,BINARY_RDONLY,&fdin);CHKERRA(ierr);

  ierr = OptionsGetString(PETSC_NULL,"-fout",fout,127,&flg); CHKERRA(ierr);
  if (!flg) PetscPrintf(PETSC_COMM_WORLD,"Writing submatrix to file : %s\n",fout);
  ierr = ViewerBinaryOpen(PETSC_COMM_SELF,fout,BINARY_CREATE,&fdout);CHKERRA(ierr);

  ierr = MatLoad(fdin,mtype,&A); CHKERRA(ierr);
  ierr = ViewerDestroy(fdin); CHKERRA(ierr);
  
  ierr = MatGetSize(A,&size,&size);  CHKERRA(ierr);
  size /= 2;
  ierr = OptionsGetInt(PETSC_NULL,"-start",&start,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-size", &size,&flg); CHKERRA(ierr);
  
  ierr = ISCreateStride(PETSC_COMM_SELF,size,start,1,&isrow); CHKERRA(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,size,start,1,&iscol); CHKERRA(ierr);
  ierr = MatGetSubMatrices(A,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&B); CHKERRA(ierr);
  ierr = MatView(B[0],fdout); CHKERRA(ierr);

  ierr = VecCreate(PETSC_COMM_SELF,PETSC_DECIDE,size,&b); CHKERRA(ierr);
  ierr = VecSetFromOptions(b); CHKERRA(ierr);
  ierr = MatView(B[0],fdout); CHKERRA(ierr);
  ierr = ViewerDestroy(fdout); CHKERRA(ierr);

  ierr = MatDestroy(A); CHKERRA(ierr);
  ierr = MatDestroy(B[0]);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);
  PetscFree(B);
  ierr = ISDestroy(iscol);CHKERRA(ierr);
  ierr = ISDestroy(isrow);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

