

static char help[] = "Tests parallel vector assembly\n";

#include "petsc.h"
#include "comm.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "options.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int          n = 5, ierr, idx1[2] = {0,3},numtids,mytid;
  Scalar       one = 1.0, two = 2.0, three = 3.0, dots[3],dot;
  double       norm;
  Vec          x,y,w,*z;
  int          idx;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  OptionsGetInt(0,0,"-n",&n); if (n < 5) n = 5;
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid); 

  if (numtids < 2) SETERR(1,"Must be run with at least two processors");

  /* create two vector */
  ierr = VecCreateSequential(n,&x); CHKERR(ierr);
  ierr = VecCreateMPI(MPI_COMM_WORLD,n,-1,&y); CHKERR(ierr);
  ierr = VecSet(&one,x);CHKERR(ierr);
  ierr = VecSet(&two,y);CHKERR(ierr);

  if (mytid == 1) {
    idx = 2; ierr = VecSetValues(y,1,&idx,&three,InsertValues); CHKERR(ierr);  
    idx = 0; ierr = VecSetValues(y,1,&idx,&two,InsertValues); CHKERR(ierr); 
    idx = 0; ierr = VecSetValues(y,1,&idx,&one,InsertValues); CHKERR(ierr); 
  }
  else {
    idx = 7; ierr = VecSetValues(y,1,&idx,&three,InsertValues);CHKERR(ierr); 
  } 
  ierr = VecBeginAssembly(y); CHKERR(ierr);
  ierr = VecEndAssembly(y); CHKERR(ierr);

  VecView(y,0);

  ierr = VecDestroy(x);CHKERR(ierr);
  ierr = VecDestroy(y);CHKERR(ierr);

  PetscFinalize();
  return 0;
}
 
