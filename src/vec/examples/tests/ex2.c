/*$Id: ex2.c,v 1.47 2000/01/11 21:00:17 bsmith Exp balay $*/

static char help[] = "Tests vector scatter-gather operations.  Input arguments are\n\
  -n <length> : vector length\n\n";

#include "petscvec.h"
#include "petscsys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int           n = 5,ierr,idx1[2] = {0,3},idx2[2] = {1,4};
  Scalar        one = 1.0,two = 2.0;
  Vec           x,y;
  IS            is1,is2;
  VecScatter    ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);

  /* create two vector */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRA(ierr);
  ierr = VecDuplicate(x,&y);CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,2,idx1,&is1);CHKERRA(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,2,idx2,&is2);CHKERRA(ierr);

  ierr = VecSet(&one,x);CHKERRA(ierr);
  ierr = VecSet(&two,y);CHKERRA(ierr);
  ierr = VecScatterCreate(x,is1,y,is2,&ctx);CHKERRA(ierr);
  ierr = VecScatterBegin(x,y,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRA(ierr);
  ierr = VecScatterEnd(x,y,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRA(ierr);
  
  ierr = VecView(y,VIEWER_STDOUT_SELF);CHKERRA(ierr);

  ierr = VecScatterBegin(y,x,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRA(ierr);
  ierr = VecScatterEnd(y,x,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRA(ierr);
  ierr = VecScatterDestroy(ctx);CHKERRA(ierr);

  ierr = PetscPrintf(PETSC_COMM_SELF,"-------\n");CHKERRA(ierr);
  ierr = VecView(x,VIEWER_STDOUT_SELF);CHKERRA(ierr);

  ierr = ISDestroy(is1);CHKERRA(ierr);
  ierr = ISDestroy(is2);CHKERRA(ierr);

  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
