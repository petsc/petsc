#ifndef lint
static char vcid[] = "$Id: ex5.c,v 1.73 1997/01/27 18:18:47 bsmith Exp $";
#endif
static char help[] = "Test the use of MatSetValuesBlocked for MatSeqBAIJ";

#include "mat.h"

int main(int argc,char **args)
{
  Mat         A;
  int         bs=3,m=4,i,j,val = 0,ierr;
  Scalar      x[6][9];
  int         row[2],col[3];

  PetscInitialize(&argc,&args,(char *)0,help);

  row[0] =0;  row[1] =2;
  col[0] =0;  col[1] =1;  col[2] =3;
  for (i=0; i<6; i++) {
    for (j =0; j< 9; j++ ) {x[i][j] = (Scalar)val++;}
    val *=-1;
      }
  ierr = MatCreateSeqBAIJ(MPI_COMM_SELF,bs,m*bs,m*bs,PETSC_DEFAULT,PETSC_NULL,&A); CHKERRA(ierr);
  ierr = MatSetValuesBlocked(A,2,row,3,col,&x[0][0],INSERT_VALUES); CHKERRQ(ierr);

  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

  MatDestroy(A);
  PetscFinalize();
  return 0;
}
