/*
   Tests MPI parallel AIJ  creation.
*/
#include "comm.h"
#include "vec.h"
#include "mat.h"
#include "options.h"
#include "stdio.h"

int main(int argc,char **args)
{
  Mat         C, LU; 
  int         i,j, m = 3, n = 2, mytid,numtids;
  Scalar      v, mone = -1.0,norm, one = 1.0;
  int         I, J, ierr;
  IS          perm, iperm;
  Vec         x,u,b;

  PetscInitialize(&argc,&args,0,0);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  n = 2*numtids;

  /* create the matrix for the five point stencil, YET AGAIN*/
  ierr = MatCreateSequentialAIJMPI(MPI_COMM_WORLD,-1,-1,m*n,m*n,5,0,5,0,&C); 
  CHKERR(ierr);

/*  for ( i=2*mytid; i<2*mytid+2; i++ ) { */
  for ( i=0; i<m; i++ ) { 
    for ( j=2*mytid; j<2*mytid+2; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,InsertValues);}
      if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,InsertValues);}
      if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,InsertValues);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,InsertValues);}
      v = 4.0; MatSetValues(C,1,&I,1,&I,&v,InsertValues);
    }
  }
  ierr = MatBeginAssembly(C); CHKERR(ierr);
  ierr = MatEndAssembly(C); CHKERR(ierr);

  ierr = VecCreateMPI(MPI_COMM_WORLD,-1,m*n,&u); CHKERR(ierr);
  ierr = VecCreate(u,&b); CHKERR(ierr);
  VecSet(&one,u);

  ierr = MatMult(C,u,b); CHKERR(ierr);

  VecView(b,0); 

  ierr = VecDestroy(u); CHKERR(ierr);
  ierr = VecDestroy(b); CHKERR(ierr);
  ierr = MatView(C,0); CHKERR(ierr);
  ierr = MatDestroy(C); CHKERR(ierr);
  PetscFinalize();
  return 0;
}
