/*$Id: sbaijov.c,v 1.65 2001/08/06 21:15:42 bsmith Exp $*/

/*
   Routines to compute overlapping regions of a parallel MPI matrix.
   Used for finding submatrices that were shared across processors.
*/
#include "src/mat/impls/sbaij/mpi/mpisbaij.h" 
#include "petscbt.h"

static int MatIncreaseOverlap_MPISBAIJ_Once(Mat,int,IS*);
static int MatIncreaseOverlap_MPISBAIJ_Local(Mat,int*,int,int*,PetscBT*);

#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap_MPISBAIJ"
int MatIncreaseOverlap_MPISBAIJ(Mat C,int is_max,IS is[],int ov)
{
  Mat_MPISBAIJ  *c = (Mat_MPISBAIJ*)C->data;
  int           i,ierr,N=C->N, bs=c->bs;
  IS            *is_new;

  PetscFunctionBegin;
  ierr = PetscMalloc(is_max*sizeof(IS),&is_new);CHKERRQ(ierr);
  /* Convert the indices into block format */
  ierr = ISCompressIndicesGeneral(N,bs,is_max,is,is_new);CHKERRQ(ierr);
  if (ov < 0){ SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap specified\n");}
  for (i=0; i<ov; ++i) {
    ierr = MatIncreaseOverlap_MPISBAIJ_Once(C,is_max,is_new);CHKERRQ(ierr);
  }
  for (i=0; i<is_max; i++) {ierr = ISDestroy(is[i]);CHKERRQ(ierr);}
  ierr = ISExpandIndicesGeneral(N,bs,is_max,is_new,is);CHKERRQ(ierr);
  for (i=0; i<is_max; i++) {ierr = ISDestroy(is_new[i]);CHKERRQ(ierr);}
  ierr = PetscFree(is_new);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef enum {MINE,OTHER} WhoseOwner;
/*  data1, odata1 and odata2 are packed in the format (for communication):
       data[0]          = is_max, no of is 
       data[1]          = size of is[0]
        ...
       data[is_max]     = size of is[is_max-1]
       data[is_max + 1] = data(is[0])
        ...
       data[is_max+1+sum(size of is[k]), k=0,...,i-1] = data(is[i])
        ...
   data2 is packed in the format (for creating output is[]):
       data[0]          = is_max, no of is 
       data[1]          = size of is[0]
        ...
       data[is_max]     = size of is[is_max-1]
       data[is_max + 1] = data(is[0])
        ...
       data[is_max + 1 + Mbs*i) = data(is[i])
        ...
*/
#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap_MPISBAIJ_Once"
static int MatIncreaseOverlap_MPISBAIJ_Once(Mat C,int is_max,IS is[])
{
  Mat_MPISBAIJ  *c = (Mat_MPISBAIJ*)C->data;
  int         len,idx,*idx_i,isz,col,*n,*data1,**data1_start,*data2,*data2_i,*data,*data_i,
              size,rank,Mbs,i,j,k,ierr,nrqs,nrqr,*odata1,*odata2,
              tag1,tag2,flag,proc_id,**odata2_ptr,*ctable=0,*btable,len_max,len_est;
  int         *id_r1,*len_r1,proc_end=0,*iwork,*len_s,len_unused,nodata2;
  int         ois_max; /* max no of is[] in each of processor */
  char        *t_p;
  MPI_Comm    comm;
  MPI_Request *s_waits1,*s_waits2,r_req;
  MPI_Status  *s_status,r_status;
  PetscBT     *table=0;  /* mark indices of this processor's is[] */
  PetscBT     table_i;
  PetscBT     otable; /* mark indices of other processors' is[] */ 
  int         bs=c->bs,Bn = c->B->n,Bnbs = Bn/bs,*Bowners;  
  IS          garray_local,garray_gl;

  PetscFunctionBegin;

  comm = C->comm;
  size = c->size;
  rank = c->rank;
  Mbs  = c->Mbs;

  ierr = PetscObjectGetNewTag((PetscObject)C,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag2);CHKERRQ(ierr);

  /* create tables used in
     step 1: table[i] - mark c->garray of proc [i]
     step 2: table[i] - mark indices of is[i] when whose=MINE     
             table[0] - mark incideces of is[] when whose=OTHER */
  len = PetscMax(is_max, size);CHKERRQ(ierr);
  len_max = len*sizeof(PetscBT) + (Mbs/PETSC_BITS_PER_BYTE+1)*len*sizeof(char) + 1;
  ierr = PetscMalloc(len_max,&table);CHKERRQ(ierr);
  t_p  = (char *)(table + len);
  for (i=0; i<len; i++) {
    table[i]  = t_p  + (Mbs/PETSC_BITS_PER_BYTE+1)*i; 
  }

  ierr = MPI_Allreduce(&is_max,&ois_max,1,MPI_INT,MPI_MAX,comm);CHKERRQ(ierr);
  
  /* 1. Send this processor's is[] to other processors */
  /*---------------------------------------------------*/
  /* allocate spaces */
  ierr = PetscMalloc(is_max*sizeof(int),&n);CHKERRQ(ierr);
  len = 0;
  for (i=0; i<is_max; i++) {
    ierr = ISGetLocalSize(is[i],&n[i]);CHKERRQ(ierr);
    len += n[i]; 
  }
  if (len == 0) { 
    is_max = 0;
  } else {
    len += 1 + is_max; /* max length of data1 for one processor */
  }

 
  ierr = PetscMalloc((size*len+1)*sizeof(int),&data1);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(int*),&data1_start);CHKERRQ(ierr);
  for (i=0; i<size; i++) data1_start[i] = data1 + i*len;

  ierr = PetscMalloc((size*4+1)*sizeof(int),&len_s);CHKERRQ(ierr);
  btable  = len_s + size;
  iwork   = btable + size;
  Bowners = iwork + size;

  /* gather c->garray from all processors */
  ierr = ISCreateGeneral(comm,Bnbs,c->garray,&garray_local);CHKERRQ(ierr);
  ierr = ISAllGather(garray_local, &garray_gl);CHKERRQ(ierr);
  ierr = ISDestroy(garray_local);CHKERRQ(ierr);
  ierr = MPI_Allgather(&Bnbs,1,MPI_INT,Bowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  Bowners[0] = 0;
  for (i=0; i<size; i++) Bowners[i+1] += Bowners[i];
  
  if (is_max){ 
    /* hash table ctable which maps c->row to proc_id) */
    ierr = PetscMalloc(Mbs*sizeof(int),&ctable);CHKERRQ(ierr);
    for (proc_id=0,j=0; proc_id<size; proc_id++) {
      for (; j<c->rowners[proc_id+1]; j++) {
        ctable[j] = proc_id;
      }
    }

    /* hash tables marking c->garray */
    ierr = ISGetIndices(garray_gl,&idx_i);
    for (i=0; i<size; i++){
      table_i = table[i]; 
      ierr    = PetscBTMemzero(Mbs,table_i);CHKERRQ(ierr);
      for (j = Bowners[i]; j<Bowners[i+1]; j++){ /* go through B cols of proc[i]*/
        ierr = PetscBTSet(table_i,idx_i[j]);CHKERRQ(ierr);
      }
    }
    ierr = ISRestoreIndices(garray_gl,&idx_i);CHKERRQ(ierr);
  }  /* if (is_max) */
  ierr = ISDestroy(garray_gl);CHKERRQ(ierr); 

  /* evaluate communication - mesg to who, length, and buffer space */
  for (i=0; i<size; i++) len_s[i] = 0;
  
  /* header of data1 */
  for (proc_id=0; proc_id<size; proc_id++){
    iwork[proc_id] = 0;
    *data1_start[proc_id] = is_max; 
    data1_start[proc_id]++;
    for (j=0; j<is_max; j++) { 
      if (proc_id == rank){
        *data1_start[proc_id] = n[j]; 
      } else {
        *data1_start[proc_id] = 0;  
      }
      data1_start[proc_id]++;
    }
  }
  
  for (i=0; i<is_max; i++) { 
    ierr = ISGetIndices(is[i],&idx_i);CHKERRQ(ierr); 
    for (j=0; j<n[i]; j++){
      idx = idx_i[j];
      *data1_start[rank] = idx; data1_start[rank]++; /* for local proccessing */
      proc_end = ctable[idx];
      for (proc_id=0;  proc_id<=proc_end; proc_id++){ /* for others to process */
        if (proc_id == rank ) continue; /* done before this loop */
        if (proc_id < proc_end && !PetscBTLookup(table[proc_id],idx)) 
          continue;   /* no need for sending idx to [proc_id] */
        *data1_start[proc_id] = idx; data1_start[proc_id]++;
        len_s[proc_id]++;
      }
    } 
    /* update header data */
    for (proc_id=0; proc_id<size; proc_id++){ 
      if (proc_id== rank) continue;
      *(data1 + proc_id*len + 1 + i) = len_s[proc_id] - iwork[proc_id];
      iwork[proc_id] = len_s[proc_id] ;
    } 
    ierr = ISRestoreIndices(is[i],&idx_i);CHKERRQ(ierr);
  } 

  nrqs = 0; nrqr = 0;
  for (i=0; i<size; i++){
    data1_start[i] = data1 + i*len;
    if (len_s[i]){
      nrqs++;
      len_s[i] += 1 + is_max; /* add no. of header msg */
    }
  }

  for (i=0; i<is_max; i++) { 
    ierr = ISDestroy(is[i]);CHKERRQ(ierr); 
  }
  ierr = PetscFree(n);CHKERRQ(ierr);
  if (ctable){ierr = PetscFree(ctable);CHKERRQ(ierr);}

  /* Determine the number of messages to expect, their lengths, from from-ids */
  ierr = PetscGatherNumberOfMessages(comm,PETSC_NULL,len_s,&nrqr);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nrqs,nrqr,len_s,&id_r1,&len_r1);CHKERRQ(ierr); 
  /* ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] nrqs: %d, nrqr: %d\n",rank,nrqs,nrqr); */
  
  /*  Now  post the sends */
  ierr = PetscMalloc(2*size*sizeof(MPI_Request),&s_waits1);CHKERRQ(ierr);
  s_waits2 = s_waits1 + size;
  k = 0;
  for (proc_id=0; proc_id<size; proc_id++){  /* send data1 to processor [proc_id] */
    if (len_s[proc_id]){
      ierr = MPI_Isend(data1_start[proc_id],len_s[proc_id],MPI_INT,proc_id,tag1,comm,s_waits1+k);CHKERRQ(ierr);
      k++;
    }
  }
  
  /* 2. Do local work on this processor's is[] */
  /*-------------------------------------------*/
  len_max = is_max*(Mbs+1); /* max space storing all is[] for this processor */
  ierr = PetscMalloc((len_max+1)*sizeof(int),&data);CHKERRQ(ierr);
  ierr = MatIncreaseOverlap_MPISBAIJ_Local(C,data1_start[rank],MINE,data,table);CHKERRQ(ierr);
  ierr = PetscFree(data1_start);CHKERRQ(ierr);
  
  /* 3. Receive other's is[] and process. Then send back */
  /*-----------------------------------------------------*/
  len = 0;
  for (i=0; i<nrqr; i++){
    if (len_r1[i] > len)len = len_r1[i];
    /* ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] expect to recv len=%d from [%d]\n",rank,len_r1[i],id_r1[i]); */
  }
  ierr = PetscMalloc((len+1)*sizeof(int),&odata1);CHKERRQ(ierr);
  ierr = PetscMalloc((size+1)*sizeof(int**),&odata2_ptr);CHKERRQ(ierr);
  ierr = PetscBTCreate(Mbs,otable);CHKERRQ(ierr);

  len_max = ois_max*(Mbs+1);  /* max space storing all is[] for each receive */
  len_est = 2*len_max; /* estimated space of storing is[] for all receiving messages */
  nodata2 = 0;       /* nodata2+1: num of PetscMalloc(,&odata2_ptr[]) called */
  ierr = PetscMalloc((len_est+1)*sizeof(int),&odata2_ptr[nodata2]);CHKERRQ(ierr);
  odata2     = odata2_ptr[nodata2];
  len_unused = len_est; /* unused space in the array odata2_ptr[nodata2]-- needs to be >= len_max  */
  
  k = 0;
  while (k < nrqr){
    /* Receive messages */
    ierr = MPI_Iprobe(MPI_ANY_SOURCE,tag1,comm,&flag,&r_status);CHKERRQ(ierr);
    if (flag){ 
      ierr = MPI_Get_count(&r_status,MPI_INT,&len);CHKERRQ(ierr); 
      proc_id = r_status.MPI_SOURCE;
      ierr = MPI_Irecv(odata1,len,MPI_INT,proc_id,r_status.MPI_TAG,comm,&r_req);CHKERRQ(ierr);
      ierr = MPI_Wait(&r_req,&r_status);CHKERRQ(ierr);
      /* ierr = PetscPrintf(PETSC_COMM_SELF, " [%d] recv %d from [%d]\n",rank,len,proc_id); */

      /*  Process messages */
      /*  check if there is enough unused space in odata2 array */
      if (len_unused < len_max){ /* allocate more space for odata2 */
        ierr = PetscMalloc((len_est+1)*sizeof(int),&odata2_ptr[++nodata2]);CHKERRQ(ierr);
        odata2 = odata2_ptr[nodata2];
        len_unused = len_est;
        /* ierr = PetscPrintf(PETSC_COMM_SELF, " [%d] Malloc odata2, nodata2: %d\n",rank,nodata2); */
      }

      ierr = MatIncreaseOverlap_MPISBAIJ_Local(C,odata1,OTHER,odata2,&otable);CHKERRQ(ierr);
      len = 1 + odata2[0];
      for (i=0; i<odata2[0]; i++){
        len += odata2[1 + i];
      }

      /* Send messages back */
      ierr = MPI_Isend(odata2,len,MPI_INT,proc_id,tag2,comm,s_waits2+k);CHKERRQ(ierr);
      /* ierr = PetscPrintf(PETSC_COMM_SELF," [%d] send %d back to [%d] \n",rank,len,proc_id); */
      k++;
      odata2     += len;
      len_unused -= len;
    } 
  } 
  ierr = PetscFree(odata1);CHKERRQ(ierr); 

  /* 4. Receive work done on other processors, then merge */
  /*------------------------------------------------------*/
  data2 = odata2;
  /* check if there is enough unused space in odata2(=data2) array */
  if (len_unused < len_max){ /* allocate more space for odata2 */
    ierr = PetscMalloc((len_est+1)*sizeof(int),&odata2_ptr[++nodata2]);CHKERRQ(ierr);
    data2 = odata2_ptr[nodata2];
    len_unused = len_est;
    /* ierr = PetscPrintf(PETSC_COMM_SELF, " [%d] Malloc data2, nodata2: %d\n",rank,nodata2); */
  }

  k = 0;
  while (k < nrqs){
    /* Receive messages */
    ierr = MPI_Iprobe(MPI_ANY_SOURCE,tag2,comm,&flag,&r_status);
    if (flag){
      ierr = MPI_Get_count(&r_status,MPI_INT,&len);CHKERRQ(ierr);
      proc_id = r_status.MPI_SOURCE;
      ierr = MPI_Irecv(data2,len,MPI_INT,proc_id,r_status.MPI_TAG,comm,&r_req);CHKERRQ(ierr);
      ierr = MPI_Wait(&r_req,&r_status);CHKERRQ(ierr);
      /* ierr = PetscPrintf(PETSC_COMM_SELF," [%d] recv %d from [%d], data2:\n",rank,len,proc_id); */
      if (len > 1+is_max){ /* Add data2 into data */
        data2_i = data2 + 1 + is_max;
        for (i=0; i<is_max; i++){
          table_i = table[i];
          data_i  = data + 1 + is_max + Mbs*i;
          isz     = data[1+i]; 
          for (j=0; j<data2[1+i]; j++){
            col = data2_i[j];
            if (!PetscBTLookupSet(table_i,col)) {data_i[isz++] = col;}
          }
          data[1+i] = isz;
          if (i < is_max - 1) data2_i += data2[1+i]; 
        } 
      } 
      k++;
    } 
  } 

  /* phase 1 sends are complete */
  ierr = PetscMalloc(size*sizeof(MPI_Status),&s_status);CHKERRQ(ierr);
  if (nrqs){
    ierr = MPI_Waitall(nrqs,s_waits1,s_status);CHKERRQ(ierr);
  }
  ierr = PetscFree(data1);CHKERRQ(ierr); 
       
  /* phase 3 sends are complete */
  if (nrqr){
    ierr = MPI_Waitall(nrqr,s_waits2,s_status);CHKERRQ(ierr);
  }
  for (k=0; k<=nodata2; k++){
    ierr = PetscFree(odata2_ptr[k]);CHKERRQ(ierr); 
  }
  ierr = PetscFree(odata2_ptr);CHKERRQ(ierr);

  /* 5. Create new is[] */
  /*--------------------*/ 
  for (i=0; i<is_max; i++) {
    data_i = data + 1 + is_max + Mbs*i;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,data[1+i],data_i,is+i);CHKERRQ(ierr);
  }
 
  ierr = PetscFree(data);CHKERRQ(ierr); 
  ierr = PetscFree(s_waits1);CHKERRQ(ierr);
  ierr = PetscFree(s_status);CHKERRQ(ierr); 
  if (table) {ierr = PetscFree(table);CHKERRQ(ierr);}
  ierr = PetscBTDestroy(otable);CHKERRQ(ierr); 

  ierr = PetscFree(len_s);CHKERRQ(ierr);
  ierr = PetscFree(id_r1);CHKERRQ(ierr);
  ierr = PetscFree(len_r1);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap_MPISBAIJ_Local"
/*  
   MatIncreaseOverlap_MPISBAIJ_Local - Called by MatIncreaseOverlap, to do 
       the work on the local processor.

     Inputs:
      C      - MAT_MPISBAIJ;
      data   - holds is[]. See MatIncreaseOverlap_MPISBAIJ_Once() for the format.
      whose  - whose is[] to be processed, 
               MINE:  this processor's is[]
               OTHER: other processor's is[]
     Output:  
       nidx  - whose = MINE:
                     holds input and newly found indices in the same format as data
               whose = OTHER:
                     only holds the newly found indices
       table - table[i]: mark the indices of is[i], i=0,...,is_max. Used only in the case 'whose=MINE'.
*/
/* Would computation be reduced by swapping the loop 'for each is' and 'for each row'? */
static int MatIncreaseOverlap_MPISBAIJ_Local(Mat C,int *data,int whose,int *nidx,PetscBT *table)
{
  Mat_MPISBAIJ *c = (Mat_MPISBAIJ*)C->data;
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)(c->A)->data;
  Mat_SeqBAIJ  *b = (Mat_SeqBAIJ*)(c->B)->data;
  int          ierr,row,mbs,Mbs,*nidx_i,col,col_max,isz,isz0,*ai,*aj,*bi,*bj,*garray,rstart,l;
  int          a_start,a_end,b_start,b_end,i,j,k,is_max,*idx_i,n;
  PetscBT      table0;  /* mark the indices of input is[] for look up */
  PetscBT      table_i; /* poits to i-th table. When whose=OTHER, a single table is used for all is[] */
  
  PetscFunctionBegin;
  Mbs = c->Mbs; mbs = a->mbs; 
  ai = a->i; aj = a->j;
  bi = b->i; bj = b->j;
  garray = c->garray;
  rstart = c->rstart;
  is_max = data[0];

  ierr = PetscBTCreate(Mbs,table0);CHKERRQ(ierr);
  
  nidx[0] = is_max; 
  idx_i   = data + is_max + 1; /* ptr to input is[0] array */
  nidx_i  = nidx + is_max + 1; /* ptr to output is[0] array */
  for (i=0; i<is_max; i++) { /* for each is */
    isz  = 0;
    n = data[1+i]; /* size of input is[i] */

    /* initialize and set table_i(mark idx and nidx) and table0(only mark idx) */
    if (whose == MINE){ /* process this processor's is[] */
      table_i = table[i];
      nidx_i  = nidx + 1+ is_max + Mbs*i;
    } else {            /* process other processor's is[] - only use one temp table */
      table_i = table[0];
    }
    ierr = PetscBTMemzero(Mbs,table_i);CHKERRQ(ierr);
    ierr = PetscBTMemzero(Mbs,table0);CHKERRQ(ierr);
    if (n==0) {
       nidx[1+i] = 0; /* size of new is[i] */
       continue; 
    }

    isz0 = 0; col_max = 0;
    for (j=0; j<n; j++){
      col = idx_i[j]; 
      if (col >= Mbs) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"index col %d >= Mbs %d",col,Mbs);
      if(!PetscBTLookupSet(table_i,col)) { 
        ierr = PetscBTSet(table0,col);CHKERRQ(ierr);
        if (whose == MINE) {nidx_i[isz0] = col;}
        if (col_max < col) col_max = col;
        isz0++;
      }
    }
      
    if (whose == MINE) {isz = isz0;}
    k = 0;  /* no. of indices from input is[i] that have been examined */
    for (row=0; row<mbs; row++){ 
      a_start = ai[row]; a_end = ai[row+1];
      b_start = bi[row]; b_end = bi[row+1];
      if (PetscBTLookup(table0,row+rstart)){ /* row is on input is[i]:
                                                do row search: collect all col in this row */
        for (l = a_start; l<a_end ; l++){ /* Amat */
          col = aj[l] + rstart;
          if (!PetscBTLookupSet(table_i,col)) {nidx_i[isz++] = col;}
        }
        for (l = b_start; l<b_end ; l++){ /* Bmat */
          col = garray[bj[l]];
          if (!PetscBTLookupSet(table_i,col)) {nidx_i[isz++] = col;}
        }
        k++;
        if (k >= isz0) break; /* for (row=0; row<mbs; row++) */
      } else { /* row is not on input is[i]:
                  do col serach: add row onto nidx_i if there is a col in nidx_i */
        for (l = a_start; l<a_end ; l++){ /* Amat */
          col = aj[l] + rstart;
          if (col > col_max) break; 
          if (PetscBTLookup(table0,col)){
            if (!PetscBTLookupSet(table_i,row+rstart)) {nidx_i[isz++] = row+rstart;}
            break; /* for l = start; l<end ; l++) */
          }
        } 
        for (l = b_start; l<b_end ; l++){ /* Bmat */
          col = garray[bj[l]];
          if (col > col_max) break; 
          if (PetscBTLookup(table0,col)){
            if (!PetscBTLookupSet(table_i,row+rstart)) {nidx_i[isz++] = row+rstart;}
            break; /* for l = start; l<end ; l++) */
          }
        } 
      }
    } 
    
    if (i < is_max - 1){
      idx_i  += n;   /* ptr to input is[i+1] array */
      nidx_i += isz; /* ptr to output is[i+1] array */
    }
    nidx[1+i] = isz; /* size of new is[i] */
  } /* for each is */
  ierr = PetscBTDestroy(table0);CHKERRQ(ierr); 
  
  PetscFunctionReturn(0);
}


