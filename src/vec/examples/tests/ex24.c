#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex24.c,v 1.4 1999/04/19 22:11:24 bsmith Exp balay $";
#endif

static char help[] = "Scatters from a parallel vector to a sequential vector.\n\
Tests where the local part of the scatter is a copy.\n\n";

#include "vec.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int           n = 5, ierr, size,rank,i,*blks, bs = 1,flg,m = 2;
  Scalar        value;
  Vec           x,y;
  IS            is1,is2;
  VecScatter    ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,help);

  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);
  OptionsGetInt(PETSC_NULL,"-bs",&bs,&flg);

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
  blks = (int *) PetscMalloc( (m)*sizeof(int) );CHKPTRA(blks);
  blks[0] = n*rank*bs;
  for ( i=1; i<m; i++ ) {
    blks[i] = blks[i-1] + bs;   
  }
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,m,blks,&is1);CHKERRA(ierr);
  PetscFree(blks);

  ierr = VecCreateSeq(PETSC_COMM_SELF,bs*m,&y);CHKERRA(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,bs*m,0,1,&is2);CHKERRA(ierr);

  /* each processor inserts the entire vector */
  /* this is redundant but tests assembly */
  for ( i=0; i<bs*n*size; i++ ) {
    value = (Scalar) i;
    ierr = VecSetValues(x,1,&i,&value,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRA(ierr);
  ierr = VecAssemblyEnd(x);CHKERRA(ierr);
  ierr = VecView(x,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = VecScatterCreate(x,is1,y,is2,&ctx);CHKERRA(ierr);
  ierr = VecScatterBegin(x,y,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRA(ierr);
  ierr = VecScatterEnd(x,y,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRA(ierr);

  
  PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1);
    printf("----\n"); ierr = VecView(y,VIEWER_STDOUT_SELF);CHKERRA(ierr); fflush(stdout);
  PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1);

  ierr = VecScatterDestroy(ctx);CHKERRA(ierr);

  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);
  ierr = ISDestroy(is1);CHKERRA(ierr);
  ierr = ISDestroy(is2);CHKERRA(ierr);

  PetscFinalize(); 
  return 0;
}
 
