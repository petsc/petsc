

static char help[] = "A scatter with a stride and general index set\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "options.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int           n = 6, ierr, loc[6] = {0,1,2,3,4,5};
  Scalar        two = 2.0, vals[6] = {10,11,12,13,14,15};
  Vec           x,y;
  IS            is1,is2;
  VecScatterCtx ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);

  /* create two vector */
  ierr = VecCreateSequential(MPI_COMM_SELF,n,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&y); CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,3,0,2,&is1); CHKERRA(ierr);
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,3,1,2,&is2); CHKERRA(ierr);

  ierr = VecSetValues(x,6,loc,vals,INSERTVALUES); CHKERRA(ierr);
  VecView(x,STDOUT_VIEWER); printf("----\n");
  ierr = VecSet(&two,y);CHKERRA(ierr);
  ierr = VecScatterCtxCreate(x,is1,y,is2,&ctx); CHKERRA(ierr);
  ierr = VecScatterBegin(x,is1,y,is2,INSERTVALUES,SCATTERALL,ctx);
  CHKERRA(ierr);
  ierr = VecScatterEnd(x,is1,y,is2,INSERTVALUES,SCATTERALL,ctx); CHKERRA(ierr);
  ierr = VecScatterCtxDestroy(ctx); CHKERRA(ierr);
  
  VecView(y,STDOUT_VIEWER);

  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
