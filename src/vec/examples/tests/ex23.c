/*$Id: ex23.c,v 1.12 2000/09/28 21:10:28 bsmith Exp bsmith $*/

static char help[] = "Scatters from a parallel vector to a sequential vector\n\
  using a blocked send and a strided receive.\n\n";

/*
        0 1 2 3 | 4 5 6 7 ||  8 9 10 11 

     Scatter first and third block to first processor and 
     second and third block to second processor
*/
#include "petscvec.h"
#include "petscsys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int           ierr,i;
  int           size,rank,blocks[2],nlocal;
  Scalar        value;
  Vec           x,y;
  IS            is1,is2;
  VecScatter    ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);

  if (size != 2) SETERRQ(1,"Must run with 2 processors");

  /* create two vectors */
  if (!rank) nlocal = 8;
  else nlocal = 4;
  ierr = VecCreateMPI(PETSC_COMM_WORLD,nlocal,12,&x);CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,8,&y);CHKERRA(ierr);

  /* create two index sets */
  if (!rank) {
    blocks[0] = 0; blocks[1] = 8;
  } else {
    blocks[0] = 4; blocks[1] = 8;
  }
  ierr = ISCreateBlock(PETSC_COMM_SELF,4,2,blocks,&is1);CHKERRA(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,8,0,1,&is2);CHKERRA(ierr);

  for (i=0; i<12; i++) {
    value = i;
    ierr = VecSetValues(x,1,&i,&value,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRA(ierr);
  ierr = VecAssemblyEnd(x);CHKERRA(ierr);

  ierr = VecScatterCreate(x,is1,y,is2,&ctx);CHKERRA(ierr);
  ierr = VecScatterBegin(x,y,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRA(ierr);
  ierr = VecScatterEnd(x,y,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRA(ierr);
  ierr = VecScatterDestroy(ctx);CHKERRA(ierr); 
 
  ierr = PetscSleep(2*rank);CHKERRA(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_SELF);CHKERRA(ierr);

  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);
  ierr = ISDestroy(is1);CHKERRA(ierr);
  ierr = ISDestroy(is2);CHKERRA(ierr);

  PetscFinalize(); 
  return 0;
}
 
