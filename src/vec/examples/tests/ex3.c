

/*
      Example demonstrating some features of the vectors directory.
*/
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

  MPI_Init(&argc,&argv);
  OptionsCreate(argc,argv,(char*)0,(char*)0);
  OptionsGetInt(0,"-n",&n); if (n < 5) n = 5;
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid); 

  /* create two vector */
  ierr = VecCreateSequential(n,&x); CHKERR(ierr);
  ierr = VecCreateMPI(MPI_COMM_WORLD,n,-1,&y); CHKERR(ierr);
  ierr = VecSet(&one,x);CHKERR(ierr);
  ierr = VecSet(&two,y);CHKERR(ierr);

  if (mytid == 1) {
    idx = 2; ierr = VecInsertValues(y,1,&idx,&three); CHKERR(ierr);  
    idx = 0; ierr = VecInsertValues(y,1,&idx,&two); CHKERR(ierr); 
    idx = 0; ierr = VecInsertValues(y,1,&idx,&one); CHKERR(ierr); 
  }
  else {
    idx = 7; ierr = VecInsertValues(y,1,&idx,&three); CHKERR(ierr); 
  } 
  ierr = VecBeginAssembly(y); CHKERR(ierr);
  ierr = VecEndAssembly(y); CHKERR(ierr);

  VecView(y,0);

  ierr = VecDestroy(x);CHKERR(ierr);
  ierr = VecDestroy(y);CHKERR(ierr);

  MPI_Finalize();
  return 0;
}
 
