/*$Id: ex4.c,v 1.46 1999/11/05 14:44:58 bsmith Exp bsmith $*/

static char help[] = "Scatters from a parallel vector into seqential vectors.\n\n";

#include "vec.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int           n = 5,ierr,idx1[2] = {0,3},idx2[2] = {1,4},rank;
  Scalar        one = 1.0,two = 2.0;
  Vec           x,y;
  IS            is1,is2;
  VecScatter    ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);

  /* create two vectors */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,n,PETSC_DECIDE,&x);CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&y);CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,2,idx1,&is1);CHKERRA(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,2,idx2,&is2);CHKERRA(ierr);

  ierr = VecSet(&one,x);CHKERRA(ierr);
  ierr = VecSet(&two,y);CHKERRA(ierr);
  ierr = VecScatterCreate(x,is1,y,is2,&ctx);CHKERRA(ierr);
  ierr = VecScatterBegin(x,y,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRA(ierr);
  ierr = VecScatterEnd(x,y,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRA(ierr);
  ierr = VecScatterDestroy(ctx);CHKERRA(ierr);
  
  if (!rank) {VecView(y,VIEWER_STDOUT_SELF);CHKERRA(ierr);}

  ierr = ISDestroy(is1);CHKERRA(ierr);
  ierr = ISDestroy(is2);CHKERRA(ierr);

  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);
  PetscFinalize();

  return 0;
}
