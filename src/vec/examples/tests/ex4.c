
static char help[] = 
"This example scatters from a parallel vector into seqential vectors.\n\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int           n = 5, ierr, idx1[2] = {0,3}, idx2[2] = {1,4},mytid;
  Scalar        one = 1.0, two = 2.0;
  Vec           x,y;
  IS            is1,is2;
  VecScatterCtx ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,"%s",help);
  OptionsGetInt(0,"-n",&n);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);

  /* create two vectors */
  ierr = VecCreateMPI(MPI_COMM_WORLD,n,PETSC_DECIDE,&x); CHKERRA(ierr);
  ierr = VecCreateSequential(MPI_COMM_SELF,n,&y); CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateSequential(MPI_COMM_SELF,2,idx1,&is1); CHKERRA(ierr);
  ierr = ISCreateSequential(MPI_COMM_SELF,2,idx2,&is2); CHKERRA(ierr);

  ierr = VecSet(&one,x); CHKERRA(ierr);
  ierr = VecSet(&two,y); CHKERRA(ierr);
  ierr = VecScatterCtxCreate(x,is1,y,is2,&ctx); CHKERRA(ierr);
  ierr = VecScatterBegin(x,y,INSERTVALUES,SCATTERALL,ctx);
  CHKERRA(ierr);
  ierr = VecScatterEnd(x,y,INSERTVALUES,SCATTERALL,ctx); CHKERRA(ierr);
  ierr = VecScatterCtxDestroy(ctx); CHKERRA(ierr);
  
  if (!mytid) {VecView(y,STDOUT_VIEWER); CHKERRA(ierr);}

  ierr = ISDestroy(is1); CHKERRA(ierr);
  ierr = ISDestroy(is2); CHKERRA(ierr);

  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);
  PetscFinalize();

  return 0;
}
