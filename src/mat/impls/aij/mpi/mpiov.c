#ifndef lint
static char vcid[] = "$Id: mpiov.c,v 1.3 1996/01/23 17:16:39 balay Exp balay $";
#endif

#include "mpiaij.h"

int MatIncreaseOverlap_MPIAIJ(Mat A, int is_max, IS *is, int ov)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        **idx, *n, *w1, *w2, *w3, *w4, *rtable, size, rank, m, i, j, ierr ;
  int        row, proc, mct, msz, **outdat, **ptr, *ctr;
                                  /* assume overlap = 1 */
  if (ismax<1) return 0;
  size   = a->size;
  rank   = a->rank;
  m      = a->M;
  idx    = (int **)PetscMalloc((is_max)*sizeof(int *));
  n      = (int *)PetscMalloc((is_max)*sizeof(int ));
  rtable = (int *)PetscMalloc((m+1)*sizeof(int )); /* Hash table for maping row ->proc */
  
  for ( i=0 ; i<is_max ; ++i) {
    ierr = ISGetIndices(is[i],&idx[i]);  CHKERRQ(ierr);
    ierr = ISGetLocalSize(is[i],&n[i]);  CHKERRQ(ierr);
  }
  
  /* Create hash table for the mapping :row -> proc*/
  for( i=0, j=0; i< size; ++i) {
    for (; j <a->rowners[i+1]; ++j) {
      rtable[j] = i;
    }
  }

  /* evaluate communication - mesg to who, length of mesg, and buffer space
     required. Based on this, buffers are allocated, and data copied into them*/
  w1     = (int *)PetscMalloc((size)*2*sizeof(int ));  /* foreach proc mesg size */
  w2     = w1 + size;         /* if w2[i] marked, then a message to proc i*/
  w3     = w2 + size;         /* no of IS that needs to be sent to proc i */
  w4     = w3 + size;         /* temp work space used in determining w1, w2, w3 */
  PetscMemzero(w1,(size)*3*sizeof(int)); /* initialise work vector*/
  for ( i=0;  i<is_max ; ++i) { 
    PetscMemzero(w4,(size)*sizeof(int)); /* initialise work vector*/
    for ( j =0 ; j < n[i] ; ++j) {
      row  = idx[i][j];
      proc = rtable[row];
      w4[proc]++;
    }
    for( j = 0; j < size; ++j){ 
      if( w4[j] ) { w1[j] += w4[j];  w3[j] += 1;} 
    }
  }

  mct      = 0;              /* no of outgoing messages */
  msz      = 0;              /* total mesg length (for all proc */
  w1[rank] = 0;              /* no mesg sent to intself */
  w3[rank] = 0;
  for (j =0; j < size ; ++j) {
    if (w1[j])  { w2[j] = 1; mct++;} /* there exists a message to proc i */
  }
  /* Each message would have a header = 1 + no of IS + data */
  for (j = 0; j<size ; ++j) {
    w1[j] += w2[j] + w3[j];     /*  0+0 or 1 + XX */
    msz   += w1[j];  
  }

  /*  MPI_Allreduce */

  /* Allocate Memory for outgoing messages */
  outdat    = (int **)PetscMalloc( 2*size*sizeof(int*));
  outdat[0] = (int *)PetscMalloc((msz+1) *sizeof (int));
  ptr       = outdat +size;     /* Pointers to the data in outgoing buffers */
  ctr       = (int *)PetscMalloc( size*sizeof(int));  

  for (i = 1; i < size ; ++i) {
    if ( w1[i]) { outdat[i] = outdat[i-1] + w1[ i-1];}
    else { outdat[i] = PETSC_NULL; }
  }

  /* Form the outgoing messages */
  /*plug in the headers*/
  for ( i=0 ; i<size ; ++i) {
    if (w3[i]) {
      outdat[i][0] = 0; /* Updated later*/
      PetscMemzero(outdat[i]+1, w3[i]*sizeof(int));
      ptr[i] = outdat[i] + w3[i];
    }
    else { ptr[i] = PETSC_NULL; }
  }

  /* ??? How to handle the local data computations? */
  for ( i=0 ; i<is_max ; ++i) {
    PetscMemzero(ctr,size*sizeof(int));
    for( j=0;  j<n[i]; ++j) {  /* parse the indices of each IS */
      row  = idx[i][j];
      proc = rtable[row];
      if (proc != rank) {
        ++ctr[proc];
        *ptr[proc] = row;
        ++ptr[proc];
      }
    }
    /*    Update the headers*/
    for( j = 0; j<size; ++j) {
      if (ctr[j]) {
        outdat[j][0] ++;
        outdat[j][ outdat[j][0]] = ctr[j];
      }
    }
  }
  
  /* Check Validity */
  for ( i=0 ; i<size ; ++i) {
    if( w3[i]) {
      if (w3[i] != outdat[i][0]) {SETERRQ(1,"MatIncreaseOverlap_MPIAIJ: Blew it!\n"); }
    }
  }
  
  for ( i=0 ; i<size ; ++i) {
    if( w3[i]) {
      sum = 0;
      for (j = 0; j < w3[i]; ++j) sum+= outdat[i][j]
  printf("Whew!!!\n");



  /*  table = PetscMalloc((ismax)*sizeof(int *)); 
  data  = PetscMalloc((is_max)*sizeof(int *));
  table[0] = PetscMalloc((m+1)*(is_max+1)*2*sizeof(int));
  data [0] = table[0] + (m+1)*(is_max+1);

  for(i = i; i<is_max ; ++i) {
    table[i] = table[0] + (m+1)*i;
    data[i]  = table[0] + (m+1)*i;
  }
  xdata  = table[0] + (m+1)*i;
  PetscMemzero(*table,(m+1)*(is_max+1)*sizeof(int));*/
  
  /* whew! already done? check again :) */
  PetscFree(idx);
  PetscFree(n);
  PetscFree(rtable);
  PetscFree(w1);
  PetscFree(outdat[0]);
  PetscFree(  outdat);
  PetscFree(ctr);
  return 0;
}

