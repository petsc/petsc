static int MatIncreaseOverlap_MPIAIJ(Mat, int id_msx, IS *is, int ov);
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        shift, **table; **data , *xtable, *xdata;, **idx, *n, *w1;
  int        *w2, *w3, *w4, *rtable ;
                                  /* assume overlap = 1 */
  
  size   = a->size;
  rank   = a->rank;
  m      = a->M;
  idx    = PetscMalloc((is_max)*sizeof(int *));
  n      = PetscMalloc((is_max)*sizeof(int ));
  rtable = PetscMalloc((m+1)sizeof(int )); /* Hash table for maping row ->proc */
  
  for ( i=0 ; i<is_max ; ++i) {
    ierr = ISGetIndices(is[i],&idx[i]);  CHKERRQ(ierr);
    ierr = ISGetLocalSize(is[i],&n[i]);  CHKERRQ(ierr);
  }
  
  /* Create hash table for the mapping :row -> proc*/
  for( i=0 j=0; i< size; ++i) {
    for (; j <rowners[i+1]; ++j) {
      rtable[j] = i;
    }
  }

  /* evaluate communication - mesg to who, length of mesg, and buffer space
     required. Based on this, buffers are allocated, and data copied into them*/
  w1     = PetscMalloc((size)*2*sizeof(int ));  /* foreach proc mesg size */
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
  outdat    = PetscMalloc( 2*size*sizeof(int*));
  outdat[0] = PetscMalloc(msz *sizeof (int));
  ptr       = outdat +size;     /* Pointers to the data in outgoing buffers */
  ctr       = PetscMalloc( size*sizeof(int));  

  for (i = 1; i < size ; ++i) {
    if ( w1[i]) { outdat[i] = w1[ i-1];}
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
        ++count[proc];
        *ptr[proc] = row;
        ++ptr[proc];
      }
    }
    /*    Update the headers*/
    for( i = 0; i<size; ++i) {
      if (ctr[i]) {
        outdat[i][0] ++;
        outdat[i][ outdat[i][0]] = ctr[i];
      }
    }
  }

  /* Check Validity */
  for ( i=0 ; i<size ; ++i) {
    if (w3[i] != outdat[i][0]) {SETERRQ(1,"MatIncreaseOverlap_MPIAIJ: Blew it!\n"); }
  }




  table = PetscMalloc((ismax)*sizeof(int *));  /* Hash table for each IS */
  data  = PetscMalloc((is_max)*sizeof(int *)); /* The overlap sol stored here */
  table[0] = PetscMalloc((m+1)*(is_max+1)*2*sizeof(int));
  data [0] = table[0] + (m+1)*(is_max+1);

  for(i = i; i<is_max ; ++i) {
    table[i] = table[0] + (m+1)*i;
    data[i]  = table[0] + (m+1)*i;
  }
  xdata  = table[0] + (m+1)*i;
  PetscMemzero(*table,(m+1)*(is_max+1)*sizeof(int));

  /* whew! already done? check again :) */
  return 0
}
