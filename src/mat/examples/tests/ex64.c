#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex64.c,v 1.5 1999/05/04 20:33:03 balay Exp bsmith $";
#endif

static char help[] = "Saves 4by4 block matrix.\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat     A;
  int     i, j, ierr,size;
  Viewer  fd;
  Scalar  values[16],one = 1.0;
  Vec     x;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  if (size > 1) SETERRA(1,1,"Can only run on one processor");

  /* 
     Open binary file.  Note that we use BINARY_CREATE to indicate
     writing to this file.
  */
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,"4by4",BINARY_CREATE,&fd);CHKERRA(ierr);

  ierr = MatCreateSeqBAIJ(PETSC_COMM_WORLD,4,12,12,0,0,&A);CHKERRA(ierr);

  for ( i=0; i<16; i++ ) values[i] = i; for (i=0; i<4; i++) values[4*i+i] += 5;
  i = 0; j = 0;
  ierr = MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES);CHKERRA(ierr);
  for ( i=0; i<16; i++ ) values[i] = i;
  i = 0; j = 2;
  ierr = MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES);CHKERRA(ierr);
  for ( i=0; i<16; i++ ) values[i] = i;
  i = 1; j = 0;
  ierr = MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES);CHKERRA(ierr);
  for ( i=0; i<16; i++ ) values[i] = i;for (i=0; i<4; i++) values[4*i+i] += 6;
  i = 1; j = 1;
  ierr = MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES);CHKERRA(ierr);
  for ( i=0; i<16; i++ ) values[i] = i;
  i = 2; j = 0;
  ierr = MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES);CHKERRA(ierr);
  for ( i=0; i<16; i++ ) values[i] = i;for (i=0; i<4; i++) values[4*i+i] += 7;
  i = 2; j = 2;
  ierr = MatSetValuesBlocked(A,1,&i,1,&j,values,INSERT_VALUES);CHKERRA(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatView(A,fd);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);

  ierr = VecCreateSeq(PETSC_COMM_WORLD,12,&x);CHKERRA(ierr);
  ierr = VecSet(&one,x);CHKERRA(ierr);
  ierr = VecView(x,fd);CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);

  ierr = ViewerDestroy(fd);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
