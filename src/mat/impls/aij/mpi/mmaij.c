#ifndef lint
static char vcid[] = "$Id: $";
#endif


/*
   Support for the parallel AIJ matrix vector multiply
*/
#include "mpiaij.h"
#include "vec/vecimpl.h"
#include "../seq/aij.h"

int MPIAIJSetUpMultiply(Mat mat)
{
  Matimpiaij *aij = (Matimpiaij *) mat->data;
  Matiaij    *B = (Matiaij *) (aij->B->data);  
  int        N = aij->N,i,j,*indices,*aj = B->j;
  int        ierr,ec = 0,*garray;
  IS         from,to;
  Vec        gvec;

  /* For the first stab we make an array as long as the number of columns */
  /* mark those columns that are in aij->B */
  indices = (int *) MALLOC( N*sizeof(int) ); CHKPTR(indices);
  MEMSET(indices,0,N*sizeof(int));
  for ( i=0; i<B->m; i++ ) {
    for ( j=0; j<B->ilen[i]; j++ ) {
     if (!indices[aj[B->i[i] - 1 + j]-1]) ec++; 
     indices[aj[B->i[i] - 1 + j]-1] = 1;}
  }

  /* form array of columns we need */
  garray = (int *) MALLOC( ec*sizeof(int) ); CHKPTR(garray);
  ec = 0;
  for ( i=0; i<N; i++ ) {
    if (indices[i]) garray[ec++] = i;
  }

  /* make indices now point into garray */
  for ( i=0; i<ec; i++ ) {
    indices[garray[i]] = i+1;
  }

  /* compact out the extra columns in B */
  for ( i=0; i<B->m; i++ ) {
    for ( j=0; j<B->ilen[i]; j++ ) {
      aj[B->i[i] - 1 + j] = indices[aj[B->i[i] - 1 + j]-1];
    }
  }
  B->n = ec;
  FREE(indices);
  
  /* create local vector that is used to scatter into */
  ierr = VecCreateSequential(ec,&aij->lvec); CHKERR(ierr);

  /* create two temporary Index sets for build scatter gather */
  ierr = ISCreateSequential(ec,garray,&from); CHKERR(ierr);
  ierr = ISCreateStrideSequential(ec,0,1,&to); CHKERR(ierr);

  /* create temporary global vector to generate scatter context */
  /* this is inefficient, but otherwise we must do either 
     1) save garray until the first actual scatter when the vector is known or
     2) have another way of generating a scatter context without a vector.*/
  ierr = VecCreateMPI(mat->comm,aij->n,aij->N,&gvec); CHKERR(ierr);

  /* gnerate the scatter context */
  ierr = VecScatterCtxCreate(gvec,from,aij->lvec,to,&aij->Mvctx); CHKERR(ierr);

  FREE(garray);
  ierr = ISDestroy(from); CHKERR(ierr);
  ierr = ISDestroy(to); CHKERR(ierr);
  ierr = VecDestroy(gvec);
  return 0;
}
