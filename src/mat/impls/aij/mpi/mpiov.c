#ifndef lint
static char vcid[] = "$Id: mpiov.c,v 1.24 1996/02/15 17:28:46 balay Exp balay $";
#endif

#include "mpiaij.h"
#include "inline/bitarray.h"

static int MatIncreaseOverlap_MPIAIJ_private(Mat, int, IS *);
static int FindOverlapLocal(Mat , int , char **,int*, int**);
static int FindOverlapRecievedMesg(Mat , int, int **, int**, int* );

int MatIncreaseOverlap_MPIAIJ(Mat C, int imax, IS *is, int ov)
{
  int i, ierr;
  if (ov < 0){ SETERRQ(1," MatIncreaseOverlap_MPIAIJ: negative overlap specified\n");}
  for (i =0; i<ov; ++i) {
    ierr = MatIncreaseOverlap_MPIAIJ_private(C, imax, is); CHKERRQ(ierr);
  }
  return 0;
}

/*
  Sample message format:
  If a processor A wants processor B to process some elements corresponding
  to index sets 1s[1], is[5]
  mesg [0] = 2   ( no of index sets in the mesg)
  -----------  
  mesg [1] = 1 => is[1]
  mesg [2] = sizeof(is[1]);
  -----------  
  mesg [5] = 5  => is[5]
  mesg [6] = sizeof(is[5]);
  -----------
  mesg [7] 
  mesg [n]  datas[1]
  -----------  
  mesg[n+1]
  mesg[m]  data(is[5])
  -----------  
  
  Notes:
  nrqs - no of requests sent (or to be sent out)
  nrqr - no of requests recieved (which have to be or which have been processed
*/
static int MatIncreaseOverlap_MPIAIJ_private(Mat C, int imax, IS *is)
{
  Mat_MPIAIJ  *c = (Mat_MPIAIJ *) C->data;
  int         **idx, *n, *w1, *w2, *w3, *w4, *rtable,**data;
  int         size, rank, m,i,j,k, ierr, **rbuf, row, proc, nrqs, msz, **outdat, **ptr;
  int         *ctr, *pa, tag, *tmp,bsz, nrqr , *isz, *isz1, **xdata;
  int          bsz1, **rbuf2;
  char        **table;
  MPI_Comm    comm;
  MPI_Request *send_waits,*recv_waits,*send_waits2,*recv_waits2 ;
  MPI_Status  *send_status ,*recv_status;
  double         space, fr, maxs;

  comm   = C->comm;
  tag    = C->tag;
  size   = c->size;
  rank   = c->rank;
  m      = c->M;


  TrSpace( &space, &fr, &maxs );
  /*  MPIU_printf(MPI_COMM_SELF,"[%d] allocated space = %f fragments = %f max ever allocated = %f\n", rank, space, fr, maxs); */
  
  idx    = (int **)PetscMalloc((imax+1)*sizeof(int *)); CHKPTRQ(idx);
  n      = (int *)PetscMalloc((imax+1)*sizeof(int )); CHKPTRQ(n);
  rtable = (int *)PetscMalloc((m+1)*sizeof(int )); CHKPTRQ(rtable);
                                /* Hash table for maping row ->proc */
  
  for ( i=0 ; i<imax ; i++) {
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
  for ( i=0;  i<imax ; i++) { 
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

  nrqs      = 0;              /* no of outgoing messages */
  msz      = 0;              /* total mesg length (for all proc */
  w1[rank] = 0;              /* no mesg sent to intself */
  w3[rank] = 0;
  for (i =0; i < size ; i++) {
    if (w1[i])  { w2[i] = 1; nrqs++;} /* there exists a message to proc i */
  }
  pa = (int *)PetscMalloc((nrqs +1)*sizeof(int)); CHKPTRQ(pa); /* (proc -array) */
  for (i =0, j=0; i < size ; i++) {
    if (w1[i]) { pa[j] = i; j++; }
  } 

  /* Each message would have a header = 1 + 2*(no of IS) + data */
  for (i = 0; i<nrqs ; i++) {
    j = pa[i];
    w1[j] += w2[j] + 2* w3[j];   
    msz   += w1[j];  
  }
  
  
  /* Do a global reduction to determine how many messages to expect*/
  {
    int *rw1, *rw2;
    rw1 = (int *)PetscMalloc(2*size*sizeof(int)); CHKPTRQ(rw1);
    rw2 = rw1+size;
    MPI_Allreduce((void *)w1, rw1, size, MPI_INT, MPI_MAX, comm);
    bsz   = rw1[rank];
    MPI_Allreduce((void *)w2, rw2, size, MPI_INT, MPI_SUM, comm);
    nrqr  = rw2[rank];
    PetscFree(rw1);
  }

  /* Allocate memory for recv buffers . Prob none if nrqr = 0 ???? */ 
  rbuf    = (int**) PetscMalloc((nrqr+1) *sizeof(int*));  CHKPTRQ(rbuf);
  rbuf[0] = (int *) PetscMalloc((nrqr *bsz+1) * sizeof(int));  CHKPTRQ(rbuf[0]);
  for (i=1; i<nrqr ; ++i) rbuf[i] = rbuf[i-1] + bsz;
  
  /* Now post the receives */
  recv_waits = (MPI_Request *) PetscMalloc((nrqr+1)*sizeof(MPI_Request)); 
  CHKPTRQ(recv_waits);
  for ( i=0; i<nrqr; ++i){
    MPI_Irecv((void *)(rbuf[i]), bsz, MPI_INT, MPI_ANY_SOURCE, tag, comm, recv_waits+i);
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
    for (i = 0; i < nrqs ; i++) {
      j         = pa[i];
      iptr     +=  ict;
      outdat[j] = iptr;
      ict       = w1[j];
    }
  }

  /* Form the outgoing messages */
  /*plug in the headers*/
  for ( i=0 ; i<nrqs ; i++) {
    j = pa[i];
    outdat[j][0] = 0;
    PetscMemzero(outdat[j]+1, 2 * w3[j]*sizeof(int));
    ptr[j] = outdat[j] + 2*w3[j] +1;
  }
 
  /* Memory for doing local proc's work*/
  table = (char **)PetscMalloc((imax+1)*sizeof(int *));  CHKPTRQ(table);
  data  = (int **)PetscMalloc((imax+1)*sizeof(int *)); CHKPTRQ(data);
  table[0] = (char *)PetscMalloc((m/BITSPERBYTE +1)*(imax)); CHKPTRQ(table[0]);
  data [0] = (int *)PetscMalloc((m+1)*(imax)*sizeof(int)); CHKPTRQ(data[0]);
  
  for(i = 1; i<imax ; i++) {
    table[i] = table[0] + (m/BITSPERBYTE+1)*i;
    data[i]  = data[0] + (m+1)*i;
  }
  
  PetscMemzero((void*)*table,(m/BITSPERBYTE+1)*(imax)); 
  isz = (int *)PetscMalloc((imax+1) *sizeof(int)); CHKPTRQ(isz);
  PetscMemzero((void *)isz,(imax+1) *sizeof(int));
  
  /* Parse the IS and update local tables and the outgoing buf with the data*/
  for ( i=0 ; i<imax ; i++) {
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
        if ( !BT_LOOKUP(table[i],row)) { data[i][isz[i]++] = row;}
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
  


  /*  Now  post the sends */
  send_waits = (MPI_Request *) PetscMalloc((nrqs+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  for( i =0; i< nrqs; ++i){
    j = pa[i];
    MPI_Isend( (void *)(outdat[j]), w1[j], MPI_INT, j, tag, comm, send_waits+i);
  }
    
  /* I nolonger need the original indices*/
  for( i=0; i< imax; ++i) {
    ierr = ISRestoreIndices(is[i], idx+i); CHKERRQ(ierr);
  }
  PetscFree(idx);
  PetscFree(n);
  PetscFree(rtable);
  for( i=0; i< imax; ++i) {
    ierr = ISDestroy(is[i]); CHKERRQ(ierr);
  }
  
  /* Do Local work*/
  ierr = FindOverlapLocal(C, imax, table,isz, data); CHKERRQ(ierr);
  /* Extract the matrices */

  /* Receive messages*/
  {
    int        index;
    
    recv_status = (MPI_Status *) PetscMalloc( (nrqr+1)*sizeof(MPI_Status) );
    CHKPTRQ(recv_status);
    for ( i=0; i< nrqr; ++i) {
      MPI_Waitany(nrqr, recv_waits, &index, recv_status+i);
    }
    
    send_status = (MPI_Status *) PetscMalloc( (nrqs+1)*sizeof(MPI_Status) );
    CHKPTRQ(send_status);
    MPI_Waitall(nrqs,send_waits,send_status);
  }
  /* Pahse 1 sends are complete - deallocate buffers */
  PetscFree(outdat);
  PetscFree(w1);
  PetscFree(tmp);

  /* int FindOverlapRecievedMesg(Mat C, int imax, int *isz, char **table, int **data)*/
  xdata    = (int **)PetscMalloc((nrqr+1)*sizeof(int *)); CHKPTRQ(xdata);
  isz1     = (int *)PetscMalloc((nrqr+1) *sizeof(int)); CHKPTRQ(isz1);
  ierr = FindOverlapRecievedMesg(C, nrqr, rbuf,xdata,isz1); CHKERRQ(ierr);
 
  /* Nolonger need rbuf. */
  PetscFree(rbuf[0]);
  PetscFree(rbuf);


  /* Send the data back*/
  /* Do a global reduction to know the buffer space req for incoming messages*/
  {
    int *rw1, *rw2;
    
    rw1 = (int *)PetscMalloc(2*size*sizeof(int)); CHKPTRQ(rw1);
    PetscMemzero((void*)rw1,2*size*sizeof(int));
    rw2 = rw1+size;
    for (i =0; i < nrqr ; ++i) {
      proc      = recv_status[i].MPI_SOURCE;
      rw1[proc] = isz1[i];
    }
      
    MPI_Allreduce((void *)rw1, (void *)rw2, size, MPI_INT, MPI_MAX, comm);
    bsz1   = rw2[rank];
    PetscFree(rw1);
  }
  
  /* Allocate buffers*/
  
  /* Allocate memory for recv buffers . Prob none if nrqr = 0 ???? */ 
  rbuf2    = (int**) PetscMalloc((nrqs+1) *sizeof(int*));  CHKPTRQ(rbuf2);
  rbuf2[0] = (int *) PetscMalloc((nrqs*bsz1+1) * sizeof(int));  CHKPTRQ(rbuf2[0]);
  for (i=1; i<nrqs ; ++i) rbuf2[i] = rbuf2[i-1] + bsz1;
  
  /* Now post the receives */
  recv_waits2 = (MPI_Request *)PetscMalloc((nrqs+1)*sizeof(MPI_Request)); CHKPTRQ(recv_waits2)
  CHKPTRQ(recv_waits2);
  for ( i=0; i<nrqs; ++i){
    MPI_Irecv((void *)(rbuf2[i]), bsz1, MPI_INT, MPI_ANY_SOURCE, tag, comm, recv_waits2+i);
  }
  
  /*  Now  post the sends */
  send_waits2 = (MPI_Request *) PetscMalloc((nrqr+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits2);
  for( i =0; i< nrqr; ++i){
    j = recv_status[i].MPI_SOURCE;
    MPI_Isend( (void *)(xdata[i]), isz1[i], MPI_INT, j, tag, comm, send_waits2+i);
  }

  /* recieve work done on other processors*/
  {
    int         index, is_no, ct1, max;
    MPI_Status  *send_status2 ,*recv_status2;
     
    recv_status2 = (MPI_Status *) PetscMalloc( (nrqs+1)*sizeof(MPI_Status) );
    CHKPTRQ(recv_status2);
    

    for ( i=0; i< nrqs; ++i) {
      MPI_Waitany(nrqs, recv_waits2, &index, recv_status2+i);
      /* Process the message*/
      ct1 = 2*rbuf2[index][0]+1;
      for (j=1; j<=rbuf2[index][0]; j++) {
        max   = rbuf2[index][2*j];
        is_no = rbuf2[index][2*j-1];
        for (k=0; k < max ; k++, ct1++) {
          row = rbuf2[index][ct1];
          if(!BT_LOOKUP(table[is_no],row)) { data[is_no][isz[is_no]++] = row;}   
        }
      }
    }
    
    
    send_status2 = (MPI_Status *) PetscMalloc( (nrqr+1)*sizeof(MPI_Status) );
    CHKPTRQ(send_status2);
    MPI_Waitall(nrqr,send_waits2,send_status2);
    
    PetscFree(send_status2); PetscFree(recv_status2);
  }
  
  for ( i=0; i<imax; ++i) {
    ierr = SYIsort(isz[i], data[i]); CHKERRQ(ierr);
    ierr = ISCreateSeq(MPI_COMM_SELF, isz[i], data[i], is+i); CHKERRQ(ierr);
  }
  TrSpace( &space, &fr, &maxs );
  /*  MPIU_printf(MPI_COMM_SELF,"[%d] allocated space = %f fragments = %f max ever allocated = %f\n", rank, space, fr, maxs);*/
  
  PetscFree(ctr);
  PetscFree(pa);
  PetscFree(rbuf2[0]); 
  PetscFree(rbuf2); 
  PetscFree(send_waits);
  PetscFree(recv_waits);
  PetscFree(send_waits2);
  PetscFree(recv_waits2);
  PetscFree(table[0]);
  PetscFree(table);
  PetscFree(send_status);
  PetscFree(recv_status);
  PetscFree(isz1);
  PetscFree(xdata[0]); 
  PetscFree(xdata);
  PetscFree(isz);
  PetscFree(data[0]);
  PetscFree(data);
  
  return 0;
}

/*   FindOverlapLocal() - Called by MatincreaseOverlap, to do the work on
     the local processor.

     Inputs:
      C      - MAT_MPIAIJ;
      imax - total no of index sets processed at a time;
      table  - an array of char - size = m bits.
      
     Output:
      isz    - array containing the count of the solution elements correspondign
               to each index set;
      data   - pointer to the solutions
*/
static int FindOverlapLocal(Mat C, int imax, char **table, int *isz,int **data)
{
  Mat_MPIAIJ *c = (Mat_MPIAIJ *) C->data;
  Mat        A = c->A, B = c->B;
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data,*b = (Mat_SeqAIJ*)B->data;
  int        start, end, val, max, rstart,cstart, ashift, bshift,*ai, *aj;
  int        *bi, *bj, *garray, i, j, k, row;

  rstart = c->rstart;
  cstart = c->cstart;
  ashift = a->indexshift;
  ai     = a->i;
  aj     = a->j +ashift;
  bshift = b->indexshift;
  bi     = b->i;
  bj     = b->j +bshift;
  garray = c->garray;

  
  for( i=0; i<imax; i++) {
    for ( j=0, max =isz[i] ; j< max; j++) {
      row   = data[i][j] - rstart;
      start = ai[row];
      end   = ai[row+1];
      for ( k=start; k < end; k++) { /* Amat */
        val = aj[k] + ashift + cstart;
        if(!BT_LOOKUP(table[i],val)) { data[i][isz[i]++] = val;}  
      }
      start = bi[row];
      end   = bi[row+1];
      for ( k=start; k < end; k++) { /* Bmat */
        val = garray[bj[k]+bshift] ; 
        if(! BT_LOOKUP(table[i],val)) { data[i][isz[i]++] = val;}  
      } 
    }
  }

return 0;
}
/*       FindOverlapRecievedMesg - Process the recieved messages,
         and return the output

         Input:
           C    - the matrix
           nrqr - no of messages being processed.
           rbuf - an array of pointers to the recieved requests
           
         Output:
           xdata - array of messages to be sent back
           isz1  - size of each message
*/
static int FindOverlapRecievedMesg(Mat C, int nrqr, int ** rbuf, int ** xdata, int * isz1 )
{
  Mat_MPIAIJ  *c = (Mat_MPIAIJ *) C->data;
  Mat         A = c->A, B = c->B;
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)A->data,*b = (Mat_SeqAIJ*)B->data;
  int         rstart,cstart, ashift, bshift,*ai, *aj, *bi, *bj, *garray, i, j, k;
  int         row,total_sz,ct, ct1, ct2, ct3,mem_estimate, oct2, l, start, end;
  int         val, max1, max2, rank, m, no_malloc =0, *tmp, new_estimate, ctr;
  char        *xtable;

  rank   = c->rank;
  m      = c->M;
  rstart = c->rstart;
  cstart = c->cstart;
  ashift = a->indexshift;
  ai     = a->i;
  aj     = a->j +ashift;
  bshift = b->indexshift;
  bi     = b->i;
  bj     = b->j +bshift;
  garray = c->garray;
  
  
  for (i =0, ct =0, total_sz =0; i< nrqr; ++i){
    ct+= rbuf[i][0];
    for ( j = 1; j <= rbuf[i][0] ; j++ ) { total_sz += rbuf[i][2*j]; }
    }
  
  max1 = ct*(a->nz +b->nz)/c->m;
  mem_estimate =  3*((total_sz > max1?total_sz:max1)+1);
  xdata[0] = (int *)PetscMalloc(mem_estimate *sizeof(int)); CHKPTRQ(xdata[0]);
  ++no_malloc;
  xtable   = (char *)PetscMalloc((m/BITSPERBYTE+1)); CHKPTRQ(xtable);
  PetscMemzero((void *)isz1,(nrqr+1) *sizeof(int));
  
  ct3 = 0;
  for (i =0; i< nrqr; i++) { /* for easch mesg from proc i */
    ct1 = 2*rbuf[i][0]+1;
    ct2 = ct1;
    ct3+= ct1;
    for (j = 1, max1= rbuf[i][0]; j<=max1; j++) { /* for each IS from proc i*/
      PetscMemzero((void *)xtable,(m/BITSPERBYTE+1));
      oct2 = ct2;
      for (k =0; k < rbuf[i][2*j]; k++, ct1++) { 
        row = rbuf[i][ct1];
        if(!BT_LOOKUP(xtable,row)) { 
          if (!(ct3 < mem_estimate)) {
            new_estimate = (int)(1.5*mem_estimate)+1;
            tmp = (int*) PetscMalloc(new_estimate * sizeof(int)); CHKPTRQ(tmp);
            PetscMemcpy((char *)tmp,(char *)xdata[0],mem_estimate*sizeof(int));
            PetscFree(xdata[0]);
            xdata[0] = tmp;
            mem_estimate = new_estimate; ++no_malloc;
            for (ctr =1; ctr <=i; ctr++) { xdata[ctr] = xdata[ctr-1] + isz1[ctr-1];}
          }
           xdata[i][ct2++] = row;ct3++;
        }
      }
      for ( k=oct2, max2 =ct2 ; k< max2; k++) {
        row   = xdata[i][k] - rstart;
        start = ai[row];
        end   = ai[row+1];
        for ( l=start; l < end; l++) {
          val = aj[l] +ashift + cstart;
          if(!BT_LOOKUP(xtable,val)) {
            if (!(ct3 < mem_estimate)) {
              new_estimate = (int)(1.5*mem_estimate)+1;
              tmp = (int*) PetscMalloc(new_estimate * sizeof(int)); CHKPTRQ(tmp);
              PetscMemcpy((char *)tmp,(char *)xdata[0],mem_estimate*sizeof(int));
              PetscFree(xdata[0]);
              xdata[0] = tmp;
              mem_estimate = new_estimate; ++no_malloc;
              for (ctr =1; ctr <=i; ctr++) { xdata[ctr] = xdata[ctr-1] + isz1[ctr-1];}
            }
            xdata[i][ct2++] = val;ct3++;
          }
        }
        start = bi[row];
        end   = bi[row+1];
        for ( l=start; l < end; l++) {
          val = garray[bj[l]+bshift] ;
          if(!BT_LOOKUP(xtable,val)) { 
            if (!(ct3 < mem_estimate)) { 
              new_estimate = (int)(1.5*mem_estimate)+1;
              tmp = (int*) PetscMalloc(new_estimate * sizeof(int)); CHKPTRQ(tmp);
              PetscMemcpy((char *)tmp,(char *)xdata[0],mem_estimate*sizeof(int));
              PetscFree(xdata[0]);
              xdata[0] = tmp;
              mem_estimate = new_estimate; ++no_malloc;
              for (ctr =1; ctr <=i; ctr++) { xdata[ctr] = xdata[ctr-1] + isz1[ctr-1];}
            }
            xdata[i][ct2++] = val;ct3++;
          }  
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
  PetscFree(xtable);
  PLogInfo(0,"MatIncreaseOverlap_MPIAIJ:[%d] Allocated %d bytes, required %d bytes, no of mallocs = %d\n",rank,mem_estimate, ct3,no_malloc);    
  return 0;
}  


int MatGetSubMatrices_MPIAIJ (Mat C,int ismax, IS *isrow,IS *iscol,MatGetSubMatrixCall 
                              scall, Mat **submat)
{ 
  Mat_MPIAIJ  *c = (Mat_MPIAIJ *) C->data;
  Mat         A = c->A;
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)A->data, *mat;
  int         **irow,**icol,*nrow,*ncol,*w1,*w2,*w3,*w4,*rtable, start, end, size;
  int         **sbuf1,**sbuf2, rank, m,i,j,k,l,ct1,ct2, ierr, **rbuf1, row, proc;
  int         nrqs, msz, **ptr, index, *req_size, *ctr, *pa, tag, *tmp,tcol,bsz, nrqr;
  int         **rbuf3,*req_source,**sbuf_aj, ashift, **rbuf2, max1,max2, **rmap;
  int         **cmap,**lens, is_no, ncols, *cols, mat_i, *mat_j, tmp2;
  MPI_Request *send_waits, *recv_waits, *send_waits2, *recv_waits2, *recv_waits3 ;
  MPI_Request *recv_waits4,*send_waits3,*send_waits4;
  MPI_Status  *recv_status ,*recv_status2,*send_status,*send_status3 ,*send_status2;
  MPI_Status  *recv_status3,*recv_status4,*send_status4;
  MPI_Comm    comm;
  Scalar      **rbuf4, **sbuf_aa, *vals, *mat_a;

  comm   = C->comm;
  tag    = C->tag;
  size   = c->size;
  rank   = c->rank;
  m      = c->M;
  ashift = a->indexshift;
  
  irow   = (int **)PetscMalloc((ismax+1)*sizeof(int *)); CHKPTRQ(irow);
  icol   = (int **)PetscMalloc((ismax+1)*sizeof(int *)); CHKPTRQ(icol);
  nrow   = (int *) PetscMalloc((ismax+1)*sizeof(int )); CHKPTRQ(nrow);
  ncol   = (int *) PetscMalloc((ismax+1)*sizeof(int )); CHKPTRQ(ncol);
  rtable = (int *) PetscMalloc((m+1)*sizeof(int )); CHKPTRQ(rtable);
                                /* Hash table for maping row ->proc */

  for ( i=0 ; i<ismax ; i++) { /* Extract the indicies and sort them */
    ierr = ISGetIndices(isrow[i],&irow[i]);  CHKERRQ(ierr);
    ierr = ISGetIndices(iscol[i],&icol[i]);  CHKERRQ(ierr);
    ierr = ISGetLocalSize(isrow[i],&nrow[i]);  CHKERRQ(ierr);
    ierr = ISGetLocalSize(iscol[i],&ncol[i]);  CHKERRQ(ierr);
    /* Check if the col indices are sorted */
    for (j =1; j< ncol[i]; j++) {
      if (icol[i][j-1]>icol[i][j]) SETERRQ(1,"MatGetSubmatrices_MPIAIJ: col IS is not sorted");
    }
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
  for ( i=0;  i<ismax ; i++) { 
    PetscMemzero(w4,(size)*sizeof(int)); /* initialise work vector*/
    for ( j =0 ; j < nrow[i] ; j++) {
      row  = irow[i][j];
      proc = rtable[row];
      w4[proc]++;
    }
    for( j = 0; j < size; j++){ 
      if( w4[j] ) { w1[j] += w4[j];  w3[j] += 1;} 
    }
  }
  
  nrqs     = 0;              /* no of outgoing messages */
  msz      = 0;              /* total mesg length (for all proc */
  w1[rank] = 0;              /* no mesg sent to intself */
  w3[rank] = 0;
  for (i =0; i < size ; i++) {
    if (w1[i])  { w2[i] = 1; nrqs++;} /* there exists a message to proc i */
  }
  pa = (int *)PetscMalloc((nrqs +1)*sizeof(int)); CHKPTRQ(pa); /* (proc -array) */
  for (i =0, j=0; i < size ; i++) {
    if (w1[i]) { pa[j] = i; j++; }
  } 

  /* Each message would have a header = 1 + 2*(no of IS) + data */
  for (i = 0; i<nrqs ; i++) {
    j = pa[i];
    w1[j] += w2[j] + 2* w3[j];   
    msz   += w1[j];  
  }
  
  /* Do a global reduction to determine how many messages to expect*/
  {
    int *rw1, *rw2;
    rw1 = (int *)PetscMalloc(2*size*sizeof(int)); CHKPTRQ(rw1);
    rw2 = rw1+size;
    MPI_Allreduce((void *)w1, rw1, size, MPI_INT, MPI_MAX, comm);
    bsz   = rw1[rank];
    MPI_Allreduce((void *)w2, rw2, size, MPI_INT, MPI_SUM, comm);
    nrqr  = rw2[rank];
    PetscFree(rw1);
  }

  /* Allocate memory for recv buffers . Prob none if nrqr = 0 ???? */ 
  rbuf1    = (int**) PetscMalloc((nrqr+1) *sizeof(int*));  CHKPTRQ(rbuf1);
  rbuf1[0] = (int *) PetscMalloc((nrqr *bsz+1) * sizeof(int));  CHKPTRQ(rbuf1[0]);
  for (i=1; i<nrqr ; ++i) rbuf1[i] = rbuf1[i-1] + bsz;
  
  /* Now post the receives */
  recv_waits = (MPI_Request *) PetscMalloc((nrqr+1)*sizeof(MPI_Request)); 
  CHKPTRQ(recv_waits);
  for ( i=0; i<nrqr; ++i){
    MPI_Irecv((void *)(rbuf1[i]), bsz, MPI_INT, MPI_ANY_SOURCE, tag, comm, recv_waits+i);
  }

  /* Allocate Memory for outgoing messages */
  sbuf1    = (int **)PetscMalloc( 2*size*sizeof(int*)); CHKPTRQ(sbuf1);
  PetscMemzero(sbuf1,  2*size*sizeof(int*));
  /* allocate memory for outgoing data + buf to recive the first reply */
  tmp       = (int *)PetscMalloc((2*msz+1) *sizeof (int)); CHKPTRQ(tmp); /*mrsg arr */
  ptr       = sbuf1 +size;     /* Pointers to the data in outgoing buffers */
  ctr       = (int *)PetscMalloc( size*sizeof(int));   CHKPTRQ(ctr);

  {
    int *iptr = tmp;
    int ict  = 0;
    for (i = 0; i < nrqs ; i++) {
      j         = pa[i];
      iptr     +=  ict;
      sbuf1[j] = iptr;
      ict       = w1[j];
    }
  }

  /* Form the outgoing messages */
  /* Initialise the header space */
  for ( i=0 ; i<nrqs ; i++) {
    j = pa[i];
    sbuf1[j][0] = 0;
    PetscMemzero(sbuf1[j]+1, 2 * w3[j]*sizeof(int));
    ptr[j] = sbuf1[j] + 2*w3[j] +1;
  }
  
  
  /* Parse the isrow and copy data into outbuf */
  for ( i=0 ; i<ismax ; i++) {
    PetscMemzero(ctr,size*sizeof(int));
    for( j=0;  j<nrow[i]; j++) {  /* parse the indices of each IS */
      row  = irow[i][j];
      proc = rtable[row];
      if (proc != rank) { /* copy to the outgoing buf*/
        ctr[proc]++;
        *ptr[proc] = row;
        ptr[proc]++;
      }
    }
    /* Update the headers for the current IS */
    for( j = 0; j<size; j++) { /* Can Optimise this loop too */
      if (ctr[j]) {
        k= ++sbuf1[j][0];
        sbuf1[j][2*k]   = ctr[j];
        sbuf1[j][2*k-1] = i;
      }
    }
  }

  /*  Now  post the sends */
  send_waits = (MPI_Request *) PetscMalloc((nrqs+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  for( i =0; i< nrqs; ++i){
    j = pa[i];
    /* printf("[%d] Send Req to %d: size %d \n", rank,j, w1[j]); */
    MPI_Isend( (void *)(sbuf1[j]), w1[j], MPI_INT, j, tag, comm, send_waits+i);
  }

  /* Post Recieves to capture the buffer size */
  recv_waits2 = (MPI_Request *) PetscMalloc((nrqs+1)*sizeof(MPI_Request)); 
  CHKPTRQ(recv_waits2);
  rbuf2 = (int**)PetscMalloc((nrqs+1) *sizeof(int *)); CHKPTRQ(rbuf2);
  rbuf2[0] = tmp + msz;
  for( i =1; i< nrqs; ++i){
    j = pa[i];
    rbuf2[i] = rbuf2[i-1]+w1[pa[i-1]];
  }
  for( i =0; i< nrqs; ++i){
    j = pa[i];
    MPI_Irecv( (void *)(rbuf2[i]), w1[j], MPI_INT, j, tag+1, comm, recv_waits2+i);
  }

  /* Send to other procs the buf size they should allocate */
 

  /* Receive messages*/
  send_waits2 = (MPI_Request *) PetscMalloc((nrqr+1)*sizeof(MPI_Request)); 
  CHKPTRQ(send_waits2);
  recv_status = (MPI_Status *) PetscMalloc( (nrqr+1)*sizeof(MPI_Status) );
  CHKPTRQ(recv_status);
  req_size    = (int *) PetscMalloc( (nrqr +1) * sizeof(int)) ; CHKPTRQ(req_size);
  req_source  = (int *) PetscMalloc( (nrqr +1) * sizeof(int)) ; CHKPTRQ(req_source);
  sbuf2       = (int**) PetscMalloc( (nrqr +1) * sizeof(int*)) ; CHKPTRQ(sbuf2);
  
  for ( i=0; i< nrqr; ++i) {
    MPI_Waitany(nrqr, recv_waits, &index, recv_status+i);
    /* req_size[index] = 2*rbuf1[index][0];*/
    req_size[index] = 0;
    start           = 2*rbuf1[index][0] + 1 ;
    MPI_Get_count(recv_status+i,MPI_INT, &end);
    sbuf2 [index] = (int *)PetscMalloc(end*sizeof(int)); CHKPTRQ(sbuf2[index]);
    for (j=start; j< end; j++) {
      ierr = MatGetRow(C,rbuf1[index][j], &ncols,0,0); CHKERRQ(ierr);
      sbuf2[index][j] = ncols;
      req_size[index] += ncols;
    }
    req_source[index] = recv_status[i].MPI_SOURCE;
    /* form the header */
    sbuf2[index][0]   = req_size[index];
    for (j=1; j<start; j++){ sbuf2[index][j] = rbuf1[index][j]; }
    MPI_Isend((void *)(sbuf2[index]),end,MPI_INT,req_source[index],tag+1, comm, send_waits2+i); 
  }

  /*  recv buffer sizes */
 /* Receive messages*/
  
  rbuf3 = (int**)PetscMalloc((nrqs+1) *sizeof(int*)); CHKPTRQ(rbuf3);
  rbuf4 = (Scalar**)PetscMalloc((nrqs+1) *sizeof(Scalar*)); CHKPTRQ(rbuf4);
  recv_waits3 = (MPI_Request *) PetscMalloc((nrqs+1)*sizeof(MPI_Request)); 
  CHKPTRQ(recv_waits3);
  recv_waits4 = (MPI_Request *) PetscMalloc((nrqs+1)*sizeof(MPI_Request)); 
  CHKPTRQ(recv_waits4);
  recv_status2 = (MPI_Status *) PetscMalloc( (nrqs+1)*sizeof(MPI_Status) );
  CHKPTRQ(recv_status2);
  for ( i=0; i< nrqs; ++i) {
    MPI_Waitany(nrqs, recv_waits2, &index, recv_status2+i);
    
    rbuf3[index] = (int *)PetscMalloc(rbuf2[index][0]*sizeof(int)); 
    CHKPTRQ(rbuf3[index]);
    rbuf4[index] = (Scalar *)PetscMalloc(rbuf2[index][0]*sizeof(Scalar));
    CHKPTRQ(rbuf4[index]);
    MPI_Irecv((void *)(rbuf3[index]),rbuf2[index][0], MPI_INT, 
              recv_status2[i].MPI_SOURCE, tag+2, comm, recv_waits3+index); 
    MPI_Irecv((void *)(rbuf4[index]),rbuf2[index][0], MPIU_SCALAR, 
              recv_status2[i].MPI_SOURCE, tag+3, comm, recv_waits4+index); 
  } 
  
  /* Wait on sends1 and sends2 */
    send_status = (MPI_Status *) PetscMalloc( (nrqs+1)*sizeof(MPI_Status) );
    CHKPTRQ(send_status);
    send_status2 = (MPI_Status *) PetscMalloc( (nrqr+1)*sizeof(MPI_Status) );
    CHKPTRQ(send_status2);

  MPI_Waitall(nrqs,send_waits,send_status);
  MPI_Waitall(nrqr,send_waits2,send_status2);
  

  /* Now allocate buffers for a->j, and send them off */
  sbuf_aj = (int **)PetscMalloc((nrqr+1)*sizeof(int *)); CHKPTRQ(sbuf_aj);
  for ( i=0, j =0; i< nrqr; i++) j += req_size[i];
  sbuf_aj[0] = (int*) PetscMalloc((j+1)*sizeof(int)); CHKPTRQ(sbuf_aj[0]);
  for  (i =1; i< nrqr; i++)  sbuf_aj[i] = sbuf_aj[i-1] + req_size[i-1];
  
  send_waits3 = (MPI_Request *) PetscMalloc((nrqr+1)*sizeof(MPI_Request)); 
  CHKPTRQ(send_waits3);
  for (i=0; i<nrqr; i++) {
    ct1 = 2*rbuf1[i][0]+1;
    ct2 = 0;
    for (j=1, max1 = rbuf1[i][0]; j<= max1; j++){
      for( k=0; k< rbuf1[i][2*j]; k++, ct1++) {
        row = rbuf1[i][ct1];
        ierr = MatGetRow(C, row, &ncols, &cols, 0); CHKERRQ(ierr);
        PetscMemcpy(sbuf_aj[i]+ct2, cols, ncols*sizeof(int));
        ct2 += ncols;
        ierr = MatRestoreRow(C,row, &ncols, &cols,0); CHKERRQ(ierr);
      }
    }
    /* no header for this message  */
    MPI_Isend((void *)(sbuf_aj[i]),req_size[i],MPI_INT,req_source[i],tag+2, comm, send_waits3+i);
  } 
  recv_status3 = (MPI_Status *) PetscMalloc( (nrqs+1)*sizeof(MPI_Status) );
  CHKPTRQ(recv_status3);
  send_status3 = (MPI_Status *) PetscMalloc( (nrqr+1)*sizeof(MPI_Status) );
  CHKPTRQ(send_status3);

  /* Now allocate buffers for a->a, and send them off */
  sbuf_aa = (Scalar **)PetscMalloc((nrqr+1)*sizeof(Scalar *)); CHKPTRQ(sbuf_aa);
  for ( i=0, j =0; i< nrqr; i++) j += req_size[i];
  sbuf_aa[0] = (Scalar*) PetscMalloc((j+1)*sizeof(Scalar)); CHKPTRQ(sbuf_aa[0]);
  for  (i =1; i< nrqr; i++)  sbuf_aa[i] = sbuf_aa[i-1] + req_size[i-1];
  
  send_waits4 = (MPI_Request *) PetscMalloc((nrqr+1)*sizeof(MPI_Request)); 
  CHKPTRQ(send_waits4);
  for (i=0; i<nrqr; i++) {
    ct1 = 2*rbuf1[i][0]+1;
    ct2 = 0;
    for (j=1, max1 = rbuf1[i][0]; j<= max1; j++){
      for( k=0; k< rbuf1[i][2*j]; k++, ct1++) {
        row = rbuf1[i][ct1];
        ierr = MatGetRow(C, row, &ncols, 0, &vals); CHKERRQ(ierr);
        PetscMemcpy(sbuf_aa[i]+ct2, vals, ncols*sizeof(Scalar));
        ct2 += ncols;
        ierr = MatRestoreRow(C,row, &ncols,0,&vals); CHKERRQ(ierr);
      }
    }
    /* no header for this message  */
    MPI_Isend((void *)(sbuf_aa[i]),req_size[i],MPIU_SCALAR,req_source[i],tag+3, comm, send_waits4+i);
  } 
  recv_status4 = (MPI_Status *) PetscMalloc( (nrqs+1)*sizeof(MPI_Status) );
  CHKPTRQ(recv_status4);
  send_status4 = (MPI_Status *) PetscMalloc( (nrqr+1)*sizeof(MPI_Status) );
  CHKPTRQ(send_status4);


  /* Form the matrix */
  /* create col map */
  cmap   = (int **) PetscMalloc((1+ ismax)*sizeof(int *)); CHKPTRQ(cmap);
  cmap[0] = (int *)PetscMalloc((1+ ismax*c->N)*sizeof(int)); CHKPTRQ(cmap[0]);
  PetscMemzero((char *)cmap[0],(1+ ismax*c->N)*sizeof(int));
  for (i =1; i<ismax; i++) { cmap[i] = cmap[i-1] + c->N; }
  for (i=0; i< ismax; i++) {
    for ( j=0; j< ncol[i]; j++) { 
      cmap[i][icol[i][j]] = j+1; 
    }
  }
  
  /* Create lens which is required for MatCreate... */
  lens   = (int **)PetscMalloc((1+ ismax)*sizeof(int *)); CHKPTRQ(lens);
  for (i =0, j=0; i<ismax; i++) { j +=nrow[i]; }
  lens[0] = (int *)PetscMalloc((1+ j)*sizeof(int)); CHKPTRQ(lens[0]);
  PetscMemzero((char *)lens[0], (1+ j)*sizeof(int));
  for (i =1; i<ismax; i++) { lens[i] = lens[i-1] +nrow[i-1]; }
  
  /* Update lens from local data */
  for (i=0; i< ismax; i++) {
    for (j =0; j< nrow[i]; j++) {
      row  = irow[i][j] ;
      proc = rtable[row];
      if (proc == rank) {
        ierr = MatGetRow(C,row,&ncols,&cols,0); CHKERRQ(ierr);
        for (k =0; k< ncols; k++) {
          if ( cmap[i][cols[k]]) { lens[i][j]++ ;}
        }
        ierr = MatRestoreRow(C,row,&ncols,&cols,0); CHKERRQ(ierr);
      }
    }
  }
  
  /* Create row map*/
  rmap   = (int **)PetscMalloc((1+ ismax)*sizeof(int *)); CHKPTRQ(rmap);
  rmap[0] = (int *)PetscMalloc((1+ ismax*c->M)*sizeof(int)); CHKPTRQ(rmap[0]);
  PetscMemzero((char *)rmap[0],(1+ ismax*c->M)*sizeof(int));
  for (i =1; i<ismax; i++) { rmap[i] = rmap[i-1] + c->M ;}
  for (i=0; i< ismax; i++) {
    for ( j=0; j< nrow[i]; j++) { 
      rmap[i][irow[i][j]] = j; 
    }
  }
 
  /* Update lens from offproc data */
  for ( tmp2 =0; tmp2 < nrqs; tmp2++) {
    MPI_Waitany(nrqs, recv_waits3, &i, recv_status3+tmp2);
    index = pa[i];
    ct1 = 2*sbuf1[index][0]+1; /* sbuf1, rbuf2*/
    ct2 = 0;               /* rbuf3, rbuf4 */
    for (j =1; j<= sbuf1[index][0]; j++) {
      is_no = sbuf1[index][2*j-1];
      max1   = sbuf1[index][2*j];
      for (k =0; k< max1; k++, ct1++) {
        row  = sbuf1[index][ct1];
        row  = rmap[is_no][row]; /* the val in the new matrix to be */
        max2 = rbuf2[i][ct1];
        for (l=0; l<max2; l++, ct2++) {
          if (cmap[is_no][rbuf3[i][ct2]]) {
            lens[is_no][row]++;
          }
        }
      }
    }
  }    
  MPI_Waitall(nrqr,send_waits3,send_status3); 
 
  /* Create the submatrices */
  if( scall == MAT_REUSE_MATRIX) {
    int n_cols, n_rows;
    for (i=0; i<ismax; i++){
      ierr = MatGetSize((*submat)[i],&n_rows, &n_cols); CHKERRQ(ierr);
      if ((n_rows !=nrow[i]) || (n_cols !=ncol[i])) {
        SETERRQ(1,"MatGetSubmatrices_MPIAIJ:");
      }
    }
  }
  else {
    *submat = (Mat *)PetscMalloc(ismax*sizeof(Mat)); CHKPTRQ(*submat);
    for ( i=0; i<ismax; i++) {
      ierr = MatCreateSeqAIJ(comm, nrow[i],ncol[i],0,lens[i],(*submat)+i); CHKERRQ(ierr);
    }
  }

  /* Assemble the matrices */
  /* First assemble the local rows */
  for (i=0; i< ismax; i++) {
    mat   = (Mat_SeqAIJ *)((*submat)[i]->data);
    for (j =0; j< nrow[i]; j++) {
      row  = irow[i][j] ;
      proc = rtable[row];
      if (proc == rank) {
        ierr = MatGetRow(C,row,&ncols,&cols,&vals); CHKERRQ(ierr);
        row   = rmap[i][row];
        mat_i = mat->i[row] + ashift;
        mat_a = mat->a + mat_i;
        mat_j = mat->j + mat_i;
         for (k =0; k< ncols; k++) {
          if ((tcol = cmap[i][cols[k]])) { 
            *mat_j++ = tcol - (!ashift);
            *mat_a++ = vals[k];
            mat->ilen[row]++;
          }
        }
        ierr = MatRestoreRow(C,row,&ncols,&cols,&vals); CHKERRQ(ierr);
      }
    }
  }

  /*   Now assemble the off proc rows*/
  for(tmp2 =0; tmp2 <nrqs; tmp2++) {
    MPI_Waitany(nrqs, recv_waits4, &i, recv_status4+tmp2);
    index = pa[i];
    ct1 = 2*sbuf1[index][0]+1; /* sbuf1, rbuf2*/
    ct2 = 0;               /* rbuf3, rbuf4 */
    for (j =1; j<= sbuf1[index][0]; j++) {
      is_no = sbuf1[index][2*j-1];
      mat   = (Mat_SeqAIJ *)((*submat)[is_no]->data);
      max1   = sbuf1[index][2*j];
      for (k =0; k< max1; k++, ct1++) {
        row  = sbuf1[index][ct1];
        row  = rmap[is_no][row]; /* the val in the new matrix to be */
        mat_i = mat->i[row] + ashift;
        mat_a = mat->a + mat_i;
        mat_j = mat->j + mat_i;
        max2 = rbuf2[i][ct1];
        for (l=0; l<max2; l++, ct2++) {
          if ((tcol = cmap[is_no][rbuf3[i][ct2]])) {
            *mat_j++ = tcol - (! ashift);
            *mat_a++ = rbuf4[i][ct2];
            mat->ilen[row]++;
          }
        }
      }
    }
  }    
  MPI_Waitall(nrqr,send_waits4,send_status4); 

  /* Packup*/
  for( i=0; i< ismax; i++) {
    ierr = MatAssemblyBegin((*submat)[i], FINAL_ASSEMBLY); CHKERRQ(ierr);
  }
 for( i=0; i< ismax; i++) {
    ierr = MatAssemblyEnd((*submat)[i], FINAL_ASSEMBLY); CHKERRQ(ierr);
  }
  /* Restore the indices */
  for (i=0; i<ismax; i++) {
    ierr = ISRestoreIndices(isrow[i], irow+i); CHKERRQ(ierr);
    ierr = ISRestoreIndices(iscol[i], icol+i); CHKERRQ(ierr);
  }
  /* Destroy allocated memory */
  PetscFree(nrow);
  PetscFree(ncol);
  PetscFree(irow);
  PetscFree(icol);
  PetscFree(rtable);
  PetscFree(w1);
  PetscFree(pa);
  PetscFree(rbuf1[0]);
  PetscFree(rbuf1);
  PetscFree(sbuf1 );
  PetscFree(tmp);
  PetscFree(ctr);
  PetscFree(rbuf2);
  PetscFree(req_size);
  PetscFree(req_source);
  for ( i=0; i< nrqr; ++i) {
    PetscFree(sbuf2[i]);
  }
  for ( i=0; i< nrqs; ++i) {
    PetscFree(rbuf3[i]);
    PetscFree(rbuf4[i]);
  }

  PetscFree( sbuf2 );
  PetscFree(rbuf3);
  PetscFree(rbuf4 );
  PetscFree(sbuf_aj[0]);
  PetscFree(sbuf_aj);
  PetscFree(sbuf_aa[0]);
  PetscFree(sbuf_aa);
  
  PetscFree(cmap[0]);
  PetscFree(rmap[0]);
  PetscFree(cmap);
  PetscFree(rmap);
  PetscFree(lens[0]);
  PetscFree(lens);

  PetscFree(recv_waits );
  PetscFree(recv_waits2);
  PetscFree(recv_waits3);
  PetscFree(recv_waits4);

  PetscFree(recv_status);
  PetscFree(recv_status2);
  PetscFree(recv_status3);
  PetscFree(recv_status4);

  PetscFree(send_waits);
  PetscFree(send_waits2);
  PetscFree(send_waits3);
  PetscFree(send_waits4);

  PetscFree( send_status);
  PetscFree(send_status2);
  PetscFree(send_status3);
  PetscFree(send_status4);

  return 0;
}




