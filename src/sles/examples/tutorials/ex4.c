
static char help[] = 
"This example tests MatScatter\n";

#include "mat.h"
#include <stdio.h>

int main(int argc,char **args)
{
  Mat           C, A; 
  int           i,j, m = 4, n = 4, mytid, numtids, low, high, iglobal;
  Scalar        v,  one = 1.0;
  int           I, J, ierr, nz, nzalloc, mem, ldim,Istart,Iend;
  Vec           u,b;
  IS            xr,xc,yr,yc;
  MatScatterCtx ctx;

  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,help);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  
  ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
                           m*n,m*n,5,0,5,0,&C); 
  CHKERRA(ierr);

  /* create the matrix for the five point stencil, YET AGAIN*/
  MatGetOwnershipRange(C,&Istart,&Iend);
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,INSERTVALUES);}
    if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,INSERTVALUES);}
    if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,INSERTVALUES);}
    if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,INSERTVALUES);}
    v = 4.0; MatSetValues(C,1,&I,1,&I,&v,INSERTVALUES);
  }
  ierr = MatAssemblyBegin(C,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,FINAL_ASSEMBLY); CHKERRA(ierr);

  MatView(C,SYNC_STDOUT_VIEWER);

  ierr = MatCreateSequentialAIJ(MPI_COMM_SELF,m*n,m*n,5,0,&A);  CHKERRA(ierr);

  ierr = ISCreateStrideSequential(MPI_COMM_SELF,2,0,1,&xr); CHKERRA(ierr);
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,m*n,0,1,&xc); CHKERRA(ierr);
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,2,0,1,&yr); CHKERRA(ierr);
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,m*n,0,1,&yc); CHKERRA(ierr);

  ierr = MatScatterCtxCreate(C,xr,xc,A,yr,yc,&ctx); CHKERRA(ierr);

  ierr = MatScatterBegin(C,A,INSERTVALUES,ctx); CHKERRA(ierr);
  ierr = MatScatterEnd(C,A,INSERTVALUES,ctx); CHKERRA(ierr);

  MatView(A,STDOUT_VIEWER);

  ierr = ISDestroy(xr); CHKERRA(ierr);
  ierr = ISDestroy(xc); CHKERRA(ierr);
  ierr = ISDestroy(yr); CHKERRA(ierr);
  ierr = ISDestroy(yc); CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);
  ierr = MatScatterCtxDestroy(ctx); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
