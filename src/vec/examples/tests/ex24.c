/*$Id: ex24.c,v 1.11 2000/01/11 21:00:17 bsmith Exp balay $*/

static char help[] = "Scatters from a parallel vector to a sequential vector.\n\
Tests where the local part of the scatter is a copy.\n\n";

#include "petscvec.h"
#include "petscsys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int           n = 5,ierr,size,rank,i,*blks,bs = 1,m = 2;
  Scalar        value;
  Vec           x,y;
  IS            is1,is2;
  VecScatter    ctx = 0;
  Viewer        sviewer;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-bs",&bs,PETSC_NULL);CHKERRA(ierr);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);

  /* create two vectors */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,size*bs*n,&x);CHKERRA(ierr);

  /* create two index sets */
  if (rank < size-1) {
    m = n + 2; 
  } else {
    m = n;
  }
  blks = (int*)PetscMalloc((m)*sizeof(int));CHKPTRA(blks);
  blks[0] = n*rank*bs;
  for (i=1; i<m; i++) {
    blks[i] = blks[i-1] + bs;   
  }
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,m,blks,&is1);CHKERRA(ierr);
  ierr = PetscFree(blks);CHKERRA(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,bs*m,&y);CHKERRA(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,bs*m,0,1,&is2);CHKERRA(ierr);

  /* each processor inserts the entire vector */
  /* this is redundant but tests assembly */
  for (i=0; i<bs*n*size; i++) {
    value = (Scalar) i;
    ierr = VecSetValues(x,1,&i,&value,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRA(ierr);
  ierr = VecAssemblyEnd(x);CHKERRA(ierr);
  ierr = VecView(x,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = VecScatterCreate(x,is1,y,is2,&ctx);CHKERRA(ierr);
  ierr = VecScatterBegin(x,y,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRA(ierr);
  ierr = VecScatterEnd(x,y,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRA(ierr);

  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"----\n");CHKERRA(ierr); 
  ierr = ViewerGetSingleton(VIEWER_STDOUT_WORLD,&sviewer);CHKERRA(ierr);
  ierr = VecView(y,sviewer);CHKERRA(ierr); fflush(stdout);
  ierr = ViewerRestoreSingleton(VIEWER_STDOUT_WORLD,&sviewer);CHKERRA(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRA(ierr);

  ierr = VecScatterDestroy(ctx);CHKERRA(ierr);

  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);
  ierr = ISDestroy(is1);CHKERRA(ierr);
  ierr = ISDestroy(is2);CHKERRA(ierr);

  PetscFinalize(); 
  return 0;
}
 
