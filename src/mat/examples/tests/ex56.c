#ifndef lint
static char vcid[] = "$Id: ex56.c,v 1.4 1997/03/13 22:06:41 balay Exp balay $";
#endif
static char help[] = "Test the use of MatSetValuesBlocked for MatBAIJ";

#include "mat.h"

int main(int argc,char **args)
{
  Mat         A;
  int         bs=3,m=4,i,j,val = 10,ierr,flg,size,rank,rstart;
  Scalar      x[6][9],y[3][3];
  int         row[2],col[3],eval=9;

  PetscInitialize(&argc,&args,(char *)0,help);

  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  ierr = OptionsHasName(PETSC_NULL,"-ass_extern",&flg); CHKERRA(ierr);
  if (flg && (size == 1))    rstart = m*((rank+1)%size);
  else                       rstart = m*(rank);

  row[0] =rstart+0;  row[1] =rstart+2;
  col[0] =rstart+0;  col[1] =rstart+1;  col[2] =rstart+3;
  for (i=0; i<6; i++) {
    for (j =0; j< 9; j++ ) x[i][j] = (Scalar)val++;
  }

  /* 
  for (i=0; i<3; i++) {
    for (j =0; j<3 ; j++ ) x[i][j] = -1.0 ;
  }
  */
  
  if (size == 1) {
    ierr = MatCreateSeqBAIJ(MPI_COMM_SELF,bs,m*bs,m*bs,1,PETSC_NULL,&A); CHKERRA(ierr);
  } else {
    ierr = MatCreateMPIBAIJ(MPI_COMM_WORLD,bs,m*bs,m*bs,PETSC_DECIDE,PETSC_DECIDE,1,
                            PETSC_NULL,1,PETSC_NULL,&A); CHKERRA(ierr);
  }

  ierr = OptionsHasName(PETSC_NULL,"-column_oriented",&flg); CHKERRA(ierr);
  if (flg) { 
    ierr = MatSetOption(A,MAT_COLUMN_ORIENTED); CHKERRA(ierr); 
    eval = 6;
  }
  ierr = MatSetValuesBlocked(A,2,row,3,col,&x[0][0],INSERT_VALUES); CHKERRQ(ierr);

  /* Do another MatSetValues to test the case when only one local block is specified */
  for (i=0; i<3; i++) {
    for (j =0; j<3 ; j++ )  y[i][j] = (Scalar)(10 + i*eval + j);
  }
  ierr = MatSetValuesBlocked(A,1,row,1,col,&y[0][0],INSERT_VALUES); CHKERRQ(ierr);


  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

  MatDestroy(A);
  PetscFinalize();
  return 0;
}
