#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.31 1995/10/22 04:17:25 bsmith Exp bsmith $";
#endif

static char help[] = "Tests vector scatter-gather operations.  Input arguments are\n\
  -n <length> : vector length\n\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int           n = 5, ierr, idx1[2] = {0,3}, idx2[2] = {1,4};
  Scalar        one = 1.0, two = 2.0;
  Vec           x,y;
  IS            is1,is2;
  VecScatter    ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0,help);
  OptionsGetInt(PetscNull,"-n",&n);

  /* create two vector */
  ierr = VecCreateSeq(MPI_COMM_SELF,n,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&y); CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateSeq(MPI_COMM_SELF,2,idx1,&is1); CHKERRA(ierr);
  ierr = ISCreateSeq(MPI_COMM_SELF,2,idx2,&is2); CHKERRA(ierr);

  ierr = VecSet(&one,x); CHKERRA(ierr);
  ierr = VecSet(&two,y); CHKERRA(ierr);
  ierr = VecScatterCreate(x,is1,y,is2,&ctx); CHKERRA(ierr);
  ierr = VecScatterBegin(x,y,INSERT_VALUES,SCATTER_ALL,ctx);CHKERRA(ierr);
  ierr = VecScatterEnd(x,y,INSERT_VALUES,SCATTER_ALL,ctx); CHKERRA(ierr);
  
  ierr = VecView(y,STDOUT_VIEWER_SELF); CHKERRA(ierr);

  ierr = VecScatterBegin(y,x,INSERT_VALUES,SCATTER_ALL,ctx);CHKERRA(ierr);
  ierr = VecScatterEnd(y,x,INSERT_VALUES,SCATTER_ALL,ctx); CHKERRA(ierr);
  ierr = VecScatterDestroy(ctx); CHKERRA(ierr);

  MPIU_printf(MPI_COMM_SELF,"-------\n");
  ierr = VecView(x,STDOUT_VIEWER_SELF); CHKERRA(ierr);

  ierr = ISDestroy(is1); CHKERRA(ierr);
  ierr = ISDestroy(is2); CHKERRA(ierr);

  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
