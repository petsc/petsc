#ifndef lint
static char vcid[] = "$Id: ex7.c,v 1.22 1995/08/22 19:29:36 curfman Exp bsmith $";
#endif

static char help[] = 
"This example demonstrates a scatter with a stride and general index set.\n\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int           n = 6, ierr, idx1[3] = {0,1,2}, loc[6] = {0,1,2,3,4,5};
  Scalar        two = 2.0, vals[6] = {10,11,12,13,14,15};
  Vec           x,y;
  IS            is1,is2;
  VecScatterCtx ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,"-help")) fprintf(stdout,"%s",help);

  /* create two vector */
  ierr = VecCreateSeq(MPI_COMM_SELF,n,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&y); CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateStrideSeq(MPI_COMM_SELF,3,0,2,&is1); CHKERRA(ierr);
  ierr = ISCreateSeq(MPI_COMM_SELF,3,idx1,&is2); CHKERRA(ierr);

  ierr = VecSetValues(x,6,loc,vals,INSERTVALUES); CHKERRA(ierr);
  ierr = VecView(x,STDOUT_VIEWER_SELF); CHKERRA(ierr);
  printf("----\n");
  ierr = VecSet(&two,y);CHKERRA(ierr);
  ierr = VecScatterCtxCreate(x,is1,y,is2,&ctx); CHKERRA(ierr);
  ierr = VecScatterBegin(x,y,INSERTVALUES,SCATTERALL,ctx);
  CHKERRA(ierr);
  ierr = VecScatterEnd(x,y,INSERTVALUES,SCATTERALL,ctx); CHKERRA(ierr);
  ierr = VecScatterCtxDestroy(ctx); CHKERRA(ierr);
  
  ierr = VecView(y,STDOUT_VIEWER_SELF); CHKERRA(ierr);

  ierr = ISDestroy(is1); CHKERRA(ierr);
  ierr = ISDestroy(is2); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
