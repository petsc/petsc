

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
  int           n = 6, ierr, idx1[3] = {0,1,2}, loc[6] = {0,1,2,3,4,5};
  Scalar        two = 2.0, vals[6] = {10,11,12,13,14,15};
  Vec           x,y;
  IS            is1,is2;
  VecScatterCtx ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);

  /* create two vector */
  ierr = VecCreateSequential(n,&x); CHKERR(ierr);
  ierr = VecCreate(x,&y); CHKERR(ierr);

  /* create two index sets */
  ierr = ISCreateStrideSequential(3,0,2,&is1); CHKERR(ierr);
  ierr = ISCreateSequential(3,idx1,&is2); CHKERR(ierr);

  ierr = VecSetValues(x,6,loc,vals,InsertValues); CHKERR(ierr);
  VecView(x,STDOUT_VIEWER); printf("----\n");
  ierr = VecSet(&two,y);CHKERR(ierr);
  ierr = VecScatterCtxCreate(x,is1,y,is2,&ctx); CHKERR(ierr);
  ierr = VecScatterBegin(x,is1,y,is2,InsertValues,ScatterAll,ctx);
  CHKERR(ierr);
  ierr = VecScatterEnd(x,is1,y,is2,InsertValues,ScatterAll,ctx); CHKERR(ierr);
  ierr = VecScatterCtxDestroy(ctx); CHKERR(ierr);
  
  VecView(y,STDOUT_VIEWER);

  ierr = VecDestroy(x);CHKERR(ierr);
  ierr = VecDestroy(y);CHKERR(ierr);

  PetscFinalize();
  return 0;
}
 
