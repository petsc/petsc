#ifndef lint
static char vcid[] = "$Id: mmrow.c,v 1.3 1995/04/15 16:20:07 bsmith Exp $";
#endif


/*
   Support for the MPIROW matrix-vector multiply
*/
#include "mpirow.h"
#include "vec/vecimpl.h"
#include "../seq/row.h"

int MatSetUpMultiply_MPIRow(Mat mat)
{
  Mat_MPIRow *mprow = (Mat_MPIRow *) mat->data;
  Mat_Row    *B = (Mat_Row *) (mprow->B->data);  
  MatiVec    **rs = B->rs;
  int        N = mprow->N,i,j,*indices;
  int        temp,ierr,ec = 0,*garray;
  IS         from,to;
  Vec        gvec;

  /* For the first stab we make an array as long as the number of columns */
  /* mark those columns that are in mprow->B */
  indices = (int *) MALLOC( N*sizeof(int) ); CHKPTR(indices);
  MEMSET(indices,0,N*sizeof(int));
  for ( i=0; i<B->m; i++ ) {
    for ( j=0; j<rs[i]->nz; j++ ) {
     if (!indices[rs[i]->i[j]]) ec++; 
     indices[rs[i]->i[j]] = 1;}
/*   indices[aj[B->i[i] - 1 + j]-1] = 1;}   AIJ format*/
  }

  /* form array of columns we need */
  garray = (int *) MALLOC( ec*sizeof(int) ); CHKPTR(garray);
  ec = 0;
  for ( i=0; i<N; i++ ) {
    if (indices[i]) garray[ec++] = i;
  }

  /* make indices now point into garray */
  for ( i=0; i<ec; i++ ) {
    indices[garray[i]] = i;
/*  indices[garray[i]] = i+1; */
  }

  /* compact out the extra columns in B */
  for ( i=0; i<B->m; i++ ) {
    for ( j=0; j<rs[i]->nz; j++ ) {
      temp = indices[rs[i]->i[j]];
      rs[i]->i[j] = temp;
/*    aj[B->i[i] - 1 + j] = indices[aj[B->i[i] - 1 + j]-1]; */
    }
  }
  B->n = ec;
  FREE(indices);
  
  /* create local vector that is used to scatter into */
  ierr = VecCreateSequential(MPI_COMM_SELF,ec,&mprow->lvec); CHKERR(ierr);

  /* create two temporary Index sets for build scatter gather */
  ierr = ISCreateSequential(MPI_COMM_SELF,ec,garray,&from); CHKERR(ierr);
  ierr = ISCreateStrideSequential(MPI_COMM_SELF,ec,0,1,&to); CHKERR(ierr);

  /* create temporary global vector to generate scatter context */
  /* this is inefficient, but otherwise we must do either 
     1) save garray until the first actual scatter when the vector is known or
     2) have another way of generating a scatter context without a vector.*/
  ierr = VecCreateMPI(mat->comm,mprow->n,mprow->N,&gvec); CHKERR(ierr);

  /* generate the scatter context */
  ierr = VecScatterCtxCreate(gvec,from,mprow->lvec,to,&mprow->Mvctx); CHKERR(ierr);
  PLogObjectParent(mat,mprow->Mvctx);
  PLogObjectParent(mat,mprow->lvec);
  mprow->garray = garray;
  ierr = ISDestroy(from); CHKERR(ierr);
  ierr = ISDestroy(to); CHKERR(ierr);
  ierr = VecDestroy(gvec);
  return 0;
}

