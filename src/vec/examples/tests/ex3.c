

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
  IS           is1,is2;
  ISScatterCtx ctx = 0;

  MPI_Init(&argc,&argv);
  OptionsCreate(argc,argv,(char*)0,(char*)0);
  OptionsGetInt(0,"-n",&n);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  if (numtids != 2) SETERR(1,"Must be run with 2 processors\n");
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid); 

  /* create two vector */
  ierr = VecCreateSequential(n,&x); CHKERR(ierr);
  ierr = VecCreateMPI(MPI_COMM_WORLD,n,-1,&y); CHKERR(ierr);

  /* create two index sets */
  ierr = ISCreateSequential(2,idx1,&is1); CHKERR(ierr);
  ierr = ISCreateParallel(1,2,&mytid,MPI_COMM_WORLD,&is2); CHKERR(ierr);

ISView(is2,0);

  ierr = VecSet(&one,x);CHKERR(ierr);
  ierr = VecSet(&two,y);CHKERR(ierr);
  ierr = VecScatterBegin(x,is1,y,is2,&ctx); CHKERR(ierr);
  ierr = VecScatterEnd(x,is1,y,is2,&ctx); CHKERR(ierr);
  
  VecView(x,0);
  VecView(y,0);

  ierr = VecDestroy(x);CHKERR(ierr);
  ierr = VecDestroy(y);CHKERR(ierr);

  MPI_Finalize();
  return 0;
}
 
