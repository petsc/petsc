/*$Id: ex52.c,v 1.10 2000/05/05 22:16:17 balay Exp bsmith $*/

static char help[] = 
"Tests the vatious routines in MatMPIBAIJ format.\n";


#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat        A;
  int        m=2,ierr,bs=1,M,row,col,rank,size,start,end;
  Scalar     data=100;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  /* Test MatSetValues() and MatGetValues() */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mat_size",&m,PETSC_NULL);CHKERRA(ierr);

  M    = m*bs*size;
  ierr = MatCreateMPIBAIJ(PETSC_COMM_WORLD,bs,PETSC_DECIDE,PETSC_DECIDE,M,M,PETSC_DECIDE,PETSC_NULL,PETSC_DECIDE,PETSC_NULL,&A);CHKERRA(ierr);

  ierr = MatGetOwnershipRange(A,&start,&end);CHKERRA(ierr);
  
  for (row=start; row<end; row++) {
    for (col=start; col<end; col++,data+=1) {
      ierr = MatSetValues(A,1,&row,1,&col,&data,INSERT_VALUES);CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* off proc assembly */
  data = 5.0;
  row = (M+start-1)%M;
  for (col=0; col<M; col++) {
    ierr = MatSetValues(A,1,&row,1,&col,&data,ADD_VALUES);CHKERRA(ierr);
  } 
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = MatDestroy(A);
  
  PetscFinalize();
  return 0;
}
 
