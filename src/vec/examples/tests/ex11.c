#ifndef lint
static char vcid[] = "$Id: ex11.c,v 1.24 1995/09/11 18:45:48 bsmith Exp bsmith $";
#endif

static char help[] = 
"This example scatters from a parallel vector to a sequential vector.\n\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int           ierr;
  int           numtids,mytid,i,N;
  Scalar        mone = -1.0, value;
  Vec           x,y;
  IS            is1,is2;
  VecScatterCtx ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0); 
  if (OptionsHasName(0,"-help")) fprintf(stdout,"%s",help);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);

  /* create two vectors */
  ierr = VecCreateMPI(MPI_COMM_WORLD,mytid+1,PETSC_DECIDE,&x); CHKERRA(ierr);
  ierr = VecGetSize(x,&N); CHKERRA(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF,N-mytid,&y); CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateStrideSeq(MPI_COMM_SELF,N-mytid,mytid,1,&is1);
  CHKERRA(ierr);
  ierr = ISCreateStrideSeq(MPI_COMM_SELF,N-mytid,0,1,&is2); 
  CHKERRA(ierr);

  /* fill parallel vector: note this is not efficient way*/
  for ( i=0; i<N; i++ ) {
    value = (Scalar) i;
    ierr = VecSetValues(x,1,&i,&value,INSERT_VALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(x); CHKERRA(ierr);
  ierr = VecAssemblyEnd(x); CHKERRA(ierr);
  ierr = VecSet(&mone,y); CHKERRA(ierr);

  ierr = VecView(x,STDOUT_VIEWER_WORLD); CHKERRA(ierr);

  ierr = VecScatterCtxCreate(x,is1,y,is2,&ctx); CHKERRA(ierr);
  ierr = VecScatterBegin(x,y,INSERT_VALUES,SCATTERALL,ctx); CHKERRA(ierr);
  ierr = VecScatterEnd(x,y,INSERT_VALUES,SCATTERALL,ctx); CHKERRA(ierr);
  ierr = VecScatterCtxDestroy(ctx); CHKERRA(ierr);

  if (!mytid) 
    {printf("----\n"); ierr = VecView(y,STDOUT_VIEWER_SELF); CHKERRA(ierr);}

  ierr = ISDestroy(is1); CHKERRA(ierr);
  ierr = ISDestroy(is2); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);

  PetscFinalize(); 
  return 0;
}
 
