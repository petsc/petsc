#ifndef lint
static char vcid[] = "$Id: mpiov.c,v 1.6 1996/01/30 16:08:15 balay Exp balay $";
#endif

#include "mpiaij.h"
int MatIncreaseOverlap_MPIAIJ_private(Mat, int, IS *);

int MatIncreaseOverlap_MPIAIJ(Mat C, int is_max, IS *is, int ov)
{
  int i, ierr;
  if (ov < 0){ SETERRQ(1," MatIncreaseOverlap_MPIAIJ: negative overlap specified\n");}
  for (i =0; i<ov; ++i) {
    ierr = MatIncreaseOverlap_MPIAIJ_private(C, is_max, is); CHKERRQ(ierr);
  }
  return 0;
}



int MatIncreaseOverlap_MPIAIJ_private(Mat C, int is_max, IS *is)
{
  Mat_MPIAIJ  *c = (Mat_MPIAIJ *) C->data;
  Mat          A = c->A, B = c->B;
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data,*b = (Mat_SeqAIJ*)B->data;
  int         **idx, *n, *w1, *w2, *w3, *w4, *rtable,**table,**data;
  int         size, rank, m,i,j,k, ierr, **rbuf, row, proc, mct, msz, **outdat, **ptr;
  int         *ctr, sum, *pa, tag, *tmp,bsz, nmsg , *isz, *isz1, **xdata, rstart;
  int         cstart, *ai, *aj, *bi, *bj, *garray, bsz1, **rbuf2;
  MPI_Comm    comm;
  MPI_Request *send_waits,*recv_waits,*send_waits2,*recv_waits2 ;
  MPI_Status  *send_status ,*recv_status;
                                /* assume overlap = 1 */
  comm   = C->comm;
  tag    = C->tag;
  size   = c->size;
  rank   = c->rank;
  m      = c->M;
  rstart = c->rstart;
  cstart = c->cstart;
  ai     = a->i;
  aj     = a->j;
  bi     = b->i;
  bj     = b->j;
  garray = c->garray;

  idx    = (int **)PetscMalloc((is_max+1)*sizeof(int *)); CHKPTRQ(idx);
  n      = (int *)PetscMalloc((is_max+1)*sizeof(int )); CHKPTRQ(n);
  rtable = (int *)PetscMalloc((m+1)*sizeof(int )); CHKPTRQ(rtable);
                                /* Hash table for maping row ->proc */
  
  for ( i=0 ; i<is_max ; i++) {
    ierr = ISGetIndices(is[i],&idx[i]);  CHKERRQ(ierr);
    ierr = ISGetLocalSize(is[i],&n[i]);  CHKERRQ(ierr);
  }
  
  /* Create hash table for the mapping :row -> proc*/
  for( i=0, j=0; i< size; i++) {
    for (; j <c->rowners[i+1]; j++) {
      rtable[j] = i;
    }
  }

  /* evaluate communication - mesg to who, length of mesg, and buffer space
     required. Based on this, buffers are allocated, and data copied into them*/
  w1     = (int *)PetscMalloc((size)*4*sizeof(int )); CHKPTRQ(w1); /*  mesg size */
  w2     = w1 + size;         /* if w2[i] marked, then a message to proc i*/
  w3     = w2 + size;         /* no of IS that needs to be sent to proc i */
  w4     = w3 + size;         /* temp work space used in determining w1, w2, w3 */
  PetscMemzero(w1,(size)*3*sizeof(int)); /* initialise work vector*/
  for ( i=0;  i<is_max ; i++) { 
    PetscMemzero(w4,(size)*sizeof(int)); /* initialise work vector*/
    for ( j =0 ; j < n[i] ; j++) {
      row  = idx[i][j];
      proc = rtable[row];
      w4[proc]++;
    }
    for( j = 0; j < size; j++){ 
      if( w4[j] ) { w1[j] += w4[j];  w3[j] += 1;} 
    }
  }

  mct      = 0;              /* no of outgoing messages */
  msz      = 0;              /* total mesg length (for all proc */
  w1[rank] = 0;              /* no mesg sent to intself */
  w3[rank] = 0;
  for (i =0; i < size ; i++) {
    if (w1[i])  { w2[i] = 1; mct++;} /* there exists a message to proc i */
  }
  pa = (int *)PetscMalloc((mct +1)*sizeof(int)); CHKPTRQ(pa); /* (proc -array) */
  for (i =0, j=0; i < size ; i++) {
    if (w1[i]) { pa[j] = i; j++; }
  } 

  /* Each message would have a header = 1 + 2*(no of IS) + data */
  for (i = 0; i<mct ; i++) {
    j = pa[i];
    w1[j] += w2[j] + 2* w3[j];   
    msz   += w1[j];  
  }

  /* Allocate Memory for outgoing messages */
  outdat    = (int **)PetscMalloc( 2*size*sizeof(int*)); CHKPTRQ(outdat);
  PetscMemzero(outdat,  2*size*sizeof(int*));
  tmp       = (int *)PetscMalloc((msz+1) *sizeof (int)); CHKPTRQ(tmp); /*mrsg arr */
  ptr       = outdat +size;     /* Pointers to the data in outgoing buffers */
  ctr       = (int *)PetscMalloc( size*sizeof(int));   CHKPTRQ(ctr);

  {
    int *iptr = tmp;
    int ict  = 0;
    for (i = 0; i < mct ; i++) {
      j         = pa[i];
      iptr     +=  ict;
      outdat[j] = iptr;
      ict       = w1[j];
    }
  }

  /* Form the outgoing messages */
  /*plug in the headers*/
  for ( i=0 ; i<mct ; i++) {
    j = pa[i];
    outdat[j][0] = 0;
    PetscMemzero(outdat[j]+1, 2 * w3[j]*sizeof(int));
    ptr[j] = outdat[j] + 2*w3[j] +1;
  }
 
  /* Memory for doing local proc's work*/
  table = (int **)PetscMalloc((is_max+1)*sizeof(int *));  CHKPTRQ(table);
  data  = (int **)PetscMalloc((is_max+1)*sizeof(int *)); CHKPTRQ(data);
  table[0] = (int *)PetscMalloc((m+1)*(is_max)*2*sizeof(int)); CHKPTRQ(table[0]);
  data [0] = table[0] + (m+1)*(is_max);
  
  for(i = 1; i<is_max ; i++) {
    table[i] = table[0] + (m+1)*i;
    data[i]  = data[0] + (m+1)*i;
  }

  PetscMemzero((void*)*table,(m+1)*(is_max)*sizeof(int)); 
  isz = (int *)PetscMalloc((is_max+1) *sizeof(int)); CHKPTRQ(isz);
  PetscMemzero((void *)isz,(is_max+1) *sizeof(int));

  /* Parse the IS and update local tables and the outgoing buf with the data*/
  for ( i=0 ; i<is_max ; i++) {
    PetscMemzero(ctr,size*sizeof(int));
    for( j=0;  j<n[i]; j++) {  /* parse the indices of each IS */
      row  = idx[i][j];
      proc = rtable[row];
      if (proc != rank) { /* copy to the outgoing buf*/
        ctr[proc]++;
        *ptr[proc] = row;
        ptr[proc]++;
      }
      else { /* Update the table */
        if(!table[i][row]++) { data[i][isz[i]++] = row;}
      }
    }
    /* Update the headers for the current IS */
    for( j = 0; j<size; j++) { /* Can Optimise this loop too */
      if (ctr[j]) {
        k= ++outdat[j][0];
        outdat[j][2*k]   = ctr[j];
        outdat[j][2*k-1] = i;
      }
    }
  }
  
  /* Check Validity of the outgoing messages */
  for ( i=0 ; i<mct ; i++) {
    j = pa[i];
    if (w3[j] != outdat[j][0]) {SETERRQ(1,"MatIncreaseOverlap_MPIAIJ: Blew it! Header[1] mismatch!\n"); }
  }
  
  for ( i=0 ; i<mct ; i++) {
    j = pa[i];
    sum = 1;
    for (k = 1; k <= w3[j]; k++) sum += outdat[j][2*k]+2; 
    if (sum != w1[j]) { SETERRQ(1,"MatIncreaseOverlap_MPIAIJ: Blew it! Header[2-n] mismatch!  \n"); }
  }


  /* Do a global reduction to determine how many messages to expect*/
  {
    int *rw1, *rw2;
    rw1 = (int *)PetscMalloc(2*size*sizeof(int)); CHKPTRQ(rw1);
    rw2 = rw1+size;
    MPI_Allreduce((void *)w1, rw1, size, MPI_INT, MPI_MAX, comm);
    bsz   = rw1[rank];
    MPI_Allreduce((void *)w2, rw2, size, MPI_INT, MPI_SUM, comm);
    nmsg  = rw2[rank];
    PetscFree(rw1);
  }

  /* Allocate memory for recv buffers . Prob none if nmsg = 0 ???? */ 
  rbuf    = (int**) PetscMalloc((nmsg+1) *sizeof(int*));  CHKPTRQ(rbuf);
  rbuf[0] = (int *) PetscMalloc((nmsg *bsz+1) * sizeof(int));  CHKPTRQ(rbuf[0]);
  for (i=1; i<nmsg ; ++i) rbuf[i] = rbuf[i-1] + bsz;
  
  /* Now post the receives */
  recv_waits = (MPI_Request *) PetscMalloc((nmsg+1)*sizeof(MPI_Request)); 
  CHKPTRQ(recv_waits);
  for ( i=0; i<nmsg; ++i){
    MPI_Irecv((void *)rbuf[i], bsz, MPI_INT, MPI_ANY_SOURCE, tag, comm, recv_waits+i);
  }

  /*  Now  post the sends */
  send_waits = (MPI_Request *) PetscMalloc((mct+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  for( i =0; i< mct; ++i){
    j = pa[i];
    MPI_Isend( (void *)outdat[j], w1[j], MPI_INT, j, tag, comm, send_waits+i);
  }
  
  /* Do Local work*/
  /* Extract the matrices */
  {
    int  start, end, val, max;

    for( i=0; i<is_max; i++) {
      for ( j=0, max =isz[i] ; j< max; j++) {
        row   = data[i][j] - rstart;
        start = ai[row];
        end   = ai[row+1];
        for ( k=start; k < end; k++) { /* Amat */
          val = aj[k] +cstart;
          if(!table[i][val]++) { data[i][isz[i]++] = val;}  
        }
        start = bi[row];
        end   = bi[row+1];
        for ( k=start; k < end; k++) { /* Bmat */
          val = garray[bj[k]] ; 
          if(!table[i][val]++) { data[i][isz[i]++] = val;}  
        } 
      }
    }
  }
  /* Receive messages*/
  {
    int        index;
    
    recv_status = (MPI_Status *) PetscMalloc( (nmsg+1)*sizeof(MPI_Status) );
    CHKPTRQ(recv_status);
    for ( i=0; i< nmsg; ++i) {
      MPI_Waitany(nmsg, recv_waits, &index, recv_status+i);
    }
    
    send_status = (MPI_Status *) PetscMalloc( (mct+1)*sizeof(MPI_Status) );
    CHKPTRQ(send_status);
    MPI_Waitall(mct,send_waits,send_status);
  }
  
  
  /* Do work on recieved messages*/
  {
    int ct, ct1, ct2, *xtable;
    int oct2, l, start, end, val, max1, max2;
  
    
    for (i =0, ct =0; i< nmsg; ++i) ct+= rbuf[i][0];
    
    xdata   = (int **)PetscMalloc((nmsg+1)*sizeof(int *)); CHKPTRQ(xdata);
    xtable  = (int *)PetscMalloc((ct+nmsg+(m+1)*(ct+1))*sizeof(int)); CHKPTRQ(xtable);
    xdata [0] = xtable + (m+1);
    isz1 = (int *)PetscMalloc((nmsg+1) *sizeof(int)); CHKPTRQ(isz1);
    PetscMemzero((void *)isz1,(nmsg+1) *sizeof(int));

    
    for (i =0; i< nmsg; i++) { /* for easch mesg from proc i */
      ct1 = 2*rbuf[i][0]+1;
      ct2 = 2*rbuf[i][0]+1;
      for (j = 1, max1= rbuf[i][0]; j<=max1; j++) { /* for each IS from proc i*/
        PetscMemzero((void *)xtable,(m+1)*sizeof(int));
        oct2 = ct2;
        for (k =0; k < rbuf[i][2*j]; k++, ct1++) { 
          row = rbuf[i][ct1];
          if(!xtable[row]++) { xdata[i][ct2++] = row;}  
        }
        for ( k=oct2, max2 =ct2 ; k< max2; k++) {
          row   = xdata[i][k] - rstart;
          start = ai[row];
          end   = ai[row+1];
          for ( l=start; l < end; l++) {
            val = aj[l] +cstart;
            if(!xtable[val]++) { xdata[i][ct2++] = val;}  
          }
          start = bi[row];
          end   = bi[row+1];
          for ( l=start; l < end; l++) {
            val = garray[bj[l]] ;
            if(!xtable[val]++) { xdata[i][ct2++] = val;}  
          } 
        }
        /* Update the header*/
        xdata[i][2*j]   = ct2-oct2; /* Undo the vector isz1 and use only a var*/
        xdata[i][2*j-1] = rbuf[i][2*j-1];
      }
      xdata[i][0] = rbuf[i][0];
      xdata[i+1]  = xdata[i] +ct2;
      isz1[i]     = ct2; /* size of each message */
    }
  }  
  /* need isz, xdata;*/

  /* Send the data back*/
  /* Do a global reduction to know the buffer space req for incoming messages*/
  {
    int *rw1, *rw2;

    rw1 = (int *)PetscMalloc(2*size*sizeof(int)); CHKPTRQ(rw1);
    PetscMemzero((void*)rw1,2*size*sizeof(int));
    rw2 = rw1+size;
    for (i =0; i < nmsg ; ++i) {
      proc      = recv_status[i].MPI_SOURCE;
      rw1[proc] = isz1[i];
    }
      
    MPI_Allreduce((void *)rw1, (void *)rw2, size, MPI_INT, MPI_MAX, comm);
    bsz1   = rw2[rank];
    PetscFree(rw1);
  }
  
  /* Allocate buffers*/
  
  /* Allocate memory for recv buffers . Prob none if nmsg = 0 ???? */ 
  rbuf2    = (int**) PetscMalloc((mct+1) *sizeof(int*));  CHKPTRQ(rbuf2);
  rbuf2[0] = (int *) PetscMalloc((mct*bsz1+1) * sizeof(int));  CHKPTRQ(rbuf2[0]);
  for (i=1; i<mct ; ++i) rbuf2[i] = rbuf2[i-1] + bsz1;
  
  /* Now post the receives */
  recv_waits2 = (MPI_Request *)PetscMalloc((mct+1)*sizeof(MPI_Request)); CHKPTRQ(recv_waits2)
  CHKPTRQ(recv_waits2);
  for ( i=0; i<mct; ++i){
    MPI_Irecv((void *)rbuf2[i], bsz1, MPI_INT, MPI_ANY_SOURCE, tag, comm, recv_waits2+i);
  }
  
  /*  Now  post the sends */
  send_waits2 = (MPI_Request *) PetscMalloc((nmsg+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits2);
  for( i =0; i< nmsg; ++i){
    j = recv_status[i].MPI_SOURCE;
    MPI_Isend( (void *)xdata[i], isz1[i], MPI_INT, j, tag, comm, send_waits2+i);
  }
  
  /* recieve work done on other processors*/
  {
    int         index, is_no, ct1, max;
    MPI_Status  *send_status2 ,*recv_status2;
     
    recv_status2 = (MPI_Status *) PetscMalloc( (mct+1)*sizeof(MPI_Status) );
    CHKPTRQ(recv_status2);
    

    for ( i=0; i< mct; ++i) {
      MPI_Waitany(mct, recv_waits2, &index, recv_status2+i);
      /* Process the message*/
      ct1 = 2*rbuf2[index][0]+1;
      for (j=1; j<=rbuf2[index][0]; j++) {
        max   = rbuf2[index][2*j];
        is_no = rbuf2[index][2*j-1];
        for (k=0; k < max ; k++, ct1++) {
          row = rbuf2[index][ct1];
          if(!table[is_no][row]++) { data[is_no][isz[is_no]++] = row;}   
        }
      }
    }
    
    
    send_status2 = (MPI_Status *) PetscMalloc( (nmsg+1)*sizeof(MPI_Status) );
    CHKPTRQ(send_status2);
    MPI_Waitall(nmsg,send_waits2,send_status2);
    
    PetscFree(send_status2); PetscFree(recv_status2);
  }
  for( i=0; i< is_max; ++i) {
    ierr = ISRestoreIndices(is[i], idx+i); CHKERRQ(ierr);
  }
  for( i=0; i< is_max; ++i) {
    ierr = ISDestroy(is[i]); CHKERRQ(ierr);
  }
  for ( i=0; i<is_max; ++i) {
    ierr = ISCreateSeq(MPI_COMM_SELF, isz[i], data[i], is+i); CHKERRQ(ierr);
  }
  /* pack up*/

  /* done !!!! */

  /* whew! already done? check again :) */
  PetscFree(idx);
  PetscFree(n);
  PetscFree(rtable);
  PetscFree(w1);
  PetscFree(tmp);
  PetscFree(outdat);
  PetscFree(ctr);
  PetscFree(pa);
  PetscFree(rbuf[0]);
  PetscFree(rbuf);
  PetscFree(send_waits);
  PetscFree(recv_waits);
  PetscFree(table[0]);
  PetscFree(table);
  PetscFree(data);
  PetscFree(send_status);
  /* Dont forget to ISRestoreIndices */
  return 0;
}
