/*$Id: ex58.c,v 1.5 1999/05/04 20:33:03 balay Exp bsmith $*/

static char help[] = "Tests MatTranspose() and MatEqual() for MPIAIJ matrices.\n\n";

#include "mat.h"


#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  Mat        A,B;
  int        m = 7, n, i, ierr, rstart, rend,  flg,cols[3];
  Scalar     v[3];
  PetscTruth equal;
  char       *eq[2];

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = ViewerSetFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_COMMON,0);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg);CHKERRA(ierr);
  n = m;

  /* ------- Assemble matrix, test MatValid() --------- */

  ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,0,0,0,0,&A);CHKERRA(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRA(ierr);
  if (rstart == 0) {
    cols[0] = 0;
    cols[1] = 1;
    v[0]    = 2.0; v[1] = -1.0;
    ierr = MatSetValues(A,1,&rstart,2,cols,v,INSERT_VALUES);CHKERRA(ierr);
    rstart++;
  }
  if (rend == m) {
    rend--;
    cols[0] = rend-1;
    cols[1] = rend;
    v[0]    = -1.0; v[1] = 2.0;
    ierr = MatSetValues(A,1,&rend,2,cols,v,INSERT_VALUES);CHKERRA(ierr);
  }
  v[0] = -1.0; v[1] = 2.0; v[2] = -1.0;
  for ( i=rstart; i<rend; i++ ) { 
    cols[0] = i-1;
    cols[1] = i;
    cols[2] = i+1;
    ierr = MatSetValues(A,1,&i,3,cols,v,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  ierr = MatTranspose(A,&B);CHKERRA(ierr);

  ierr = MatEqual(A,B,&equal);

  eq[0] = "not equal";
  eq[1] = "equal";
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrices are %s\n",eq[equal]);CHKERRA(ierr);

  /* Free data structures */  
  ierr = MatDestroy(A);CHKERRA(ierr);
  ierr = MatDestroy(B);CHKERRA(ierr);


  PetscFinalize();
  return 0;
}
 
