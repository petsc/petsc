#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex61.c,v 1.2 1998/05/29 20:37:45 bsmith Exp bsmith $";
#endif

static char help[] = "Tests MatSeq(B)AIJSetColumnIndices()";

#include "mat.h"

/*
      Generate the following matrix:

         1 0 3 
         1 2 3
         0 0 3
*/
#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         A;
  Scalar      v;
  int         ierr,i,j,rowlens[] = {2,3,1},cols[] = {0,2,0,1,2,2},flag;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = OptionsHasName(PETSC_NULL,"-baij",&flag);CHKERRA(ierr);
  if (flag) {
    ierr = MatCreateSeqBAIJ(PETSC_COMM_WORLD,1,3,3,PETSC_NULL,rowlens,&A);CHKERRA(ierr);
    ierr = MatSeqBAIJSetColumnIndices(A,cols);CHKERRA(ierr);
  } else {
    ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,3,3,PETSC_NULL,rowlens,&A);CHKERRA(ierr);
    ierr = MatSeqAIJSetColumnIndices(A,cols);CHKERRA(ierr);
  }

  i = 0; j = 0; v = 1.0;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES); CHKERRA(ierr);
  i = 0; j = 2; v = 3.0;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES); CHKERRA(ierr);

  i = 1; j = 0; v = 1.0;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES); CHKERRA(ierr);
  i = 1; j = 1; v = 2.0;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES); CHKERRA(ierr);
  i = 1; j = 2; v = 3.0;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES); CHKERRA(ierr);

  i = 2; j = 2; v = 3.0;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES); CHKERRA(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatView(A,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  ierr = MatDestroy(A); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
