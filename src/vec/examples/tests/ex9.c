
static char help[]= 
"This example scatters from a parallel vector to a sequential vector.\n\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int           n = 5, ierr, idx2[3] = {0,2,3}, idx1[3] = {0,1,2};
  int           numtids,mytid,i;
  Scalar        mone = -1.0, value;
  Vec           x,y;
  IS            is1,is2;
  VecScatterCtx ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0); 
  if (OptionsHasName(0,"-help")) fprintf(stderr,"%s",help);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);

  /* create two vectors */
  ierr = VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,numtids*n,&x); CHKERRA(ierr);
  ierr = VecCreateSequential(MPI_COMM_SELF,n,&y); CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateSequential(MPI_COMM_SELF,3,idx1,&is1); CHKERRA(ierr);
  ierr = ISCreateSequential(MPI_COMM_SELF,3,idx2,&is2); CHKERRA(ierr);

  /* fill local part of parallel vector */
  for ( i=n*mytid; i<n*(mytid+1); i++ ) {
    value = (Scalar) i;
    ierr = VecSetValues(x,1,&i,&value,INSERTVALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(x); CHKERRA(ierr);
  ierr = VecAssemblyEnd(x); CHKERRA(ierr);

  ierr = VecView(x,SYNC_STDOUT_VIEWER); CHKERRA(ierr);

  ierr = VecSet(&mone,y); CHKERRA(ierr);

  ierr = VecScatterCtxCreate(x,is1,y,is2,&ctx); CHKERRA(ierr);
  ierr = VecScatterBegin(x,y,INSERTVALUES,SCATTERALL,ctx);
  CHKERRA(ierr);
  ierr = VecScatterEnd(x,y,INSERTVALUES,SCATTERALL,ctx); CHKERRA(ierr);
  ierr = VecScatterCtxDestroy(ctx); CHKERRA(ierr);

  if (!mytid) {
    printf("scattered vector\n"); 
    ierr = VecView(y,STDOUT_VIEWER); CHKERRA(ierr);
  }
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);

  PetscFinalize(); 
  return 0;
}
 
