/*$Id: sbaijov.c,v 1.65 2001/08/06 21:15:42 bsmith Exp $*/

/*
   Routines to compute overlapping regions of a parallel MPI matrix.
   Used for finding submatrices that were shared across processors.
*/
#include "src/mat/impls/sbaij/mpi/mpisbaij.h"
#include "petscbt.h"

static int MatIncreaseOverlap_MPISBAIJ_Once(Mat,int,IS *);
static int MatIncreaseOverlap_MPISBAIJ_Local(Mat,int *,int,int **,PetscBT*);
 
/* this function is sasme as MatCompressIndicesGeneral_MPIBAIJ -- should be removed! */
#undef __FUNCT__  
#define __FUNCT__ "MatCompressIndicesGeneral_MPISBAIJ"
static int MatCompressIndicesGeneral_MPISBAIJ(Mat C,int imax,const IS is_in[],IS is_out[])
{
  Mat_MPISBAIJ        *baij = (Mat_MPISBAIJ*)C->data;
  int                ierr,isz,bs = baij->bs,n,i,j,*idx,ival;
#if defined (PETSC_USE_CTABLE)
  PetscTable         gid1_lid1;
  int                tt, gid1, *nidx;
  PetscTablePosition tpos;
#else
  int                Nbs,*nidx;
  PetscBT            table;
#endif

  PetscFunctionBegin;
  /* printf(" ...MatCompressIndicesGeneral_MPISBAIJ is called ...\n"); */
#if defined (PETSC_USE_CTABLE)
  ierr = PetscTableCreate(baij->mbs,&gid1_lid1);CHKERRQ(ierr);
#else
  Nbs  = baij->Nbs;
  ierr = PetscMalloc((Nbs+1)*sizeof(int),&nidx);CHKERRQ(ierr); 
  ierr = PetscBTCreate(Nbs,table);CHKERRQ(ierr);
#endif
  for (i=0; i<imax; i++) {
    isz  = 0;
#if defined (PETSC_USE_CTABLE)
    ierr = PetscTableRemoveAll(gid1_lid1);CHKERRQ(ierr);
#else
    ierr = PetscBTMemzero(Nbs,table);CHKERRQ(ierr);
#endif
    ierr = ISGetIndices(is_in[i],&idx);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is_in[i],&n);CHKERRQ(ierr);
    for (j=0; j<n ; j++) {
      ival = idx[j]/bs; /* convert the indices into block indices */
#if defined (PETSC_USE_CTABLE)
      ierr = PetscTableFind(gid1_lid1,ival+1,&tt);CHKERRQ(ierr);
      if (!tt) {
	ierr = PetscTableAdd(gid1_lid1,ival+1,isz+1);CHKERRQ(ierr);
        isz++;
      }
#else
      if (ival>Nbs) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"index greater than mat-dim");
      if(!PetscBTLookupSet(table,ival)) { nidx[isz++] = ival;}
#endif
    }
    ierr = ISRestoreIndices(is_in[i],&idx);CHKERRQ(ierr);
#if defined (PETSC_USE_CTABLE)
    ierr = PetscMalloc((isz+1)*sizeof(int),&nidx);CHKERRQ(ierr); 
    ierr = PetscTableGetHeadPosition(gid1_lid1,&tpos);CHKERRQ(ierr); 
    j = 0;
    while (tpos) {  
      ierr = PetscTableGetNext(gid1_lid1,&tpos,&gid1,&tt);CHKERRQ(ierr);
      if (tt-- > isz) { SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"index greater than array-dim"); }
      nidx[tt] = gid1 - 1;
      j++;
    }
    if (j != isz) { SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"table error: jj != isz"); }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,isz,nidx,(is_out+i));CHKERRQ(ierr);
    ierr = PetscFree(nidx);CHKERRQ(ierr);
#else
    ierr = ISCreateGeneral(PETSC_COMM_SELF,isz,nidx,(is_out+i));CHKERRQ(ierr);
#endif
  }
#if defined (PETSC_USE_CTABLE)
  ierr = PetscTableDelete(gid1_lid1);CHKERRQ(ierr);
#else
  ierr = PetscBTDestroy(table);CHKERRQ(ierr);
  ierr = PetscFree(nidx);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatExpandIndices_MPISBAIJ"
static int MatExpandIndices_MPISBAIJ(Mat C,int imax,const IS is_in[],IS is_out[])
{
  Mat_MPISBAIJ  *baij = (Mat_MPISBAIJ*)C->data;
  int          ierr,bs = baij->bs,n,i,j,k,*idx,*nidx;
#if defined (PETSC_USE_CTABLE)
  int          maxsz;
#else
  int          Nbs = baij->Nbs;
#endif

  PetscFunctionBegin;
  /* printf(" ... MatExpandIndices_MPISBAIJ is called ...\n"); */
#if defined (PETSC_USE_CTABLE)
  /* Now check max size */
  for (i=0,maxsz=0; i<imax; i++) {
    ierr = ISGetIndices(is_in[i],&idx);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is_in[i],&n);CHKERRQ(ierr);
    if (n*bs > maxsz) maxsz = n*bs;
  }
  ierr = PetscMalloc((maxsz+1)*sizeof(int),&nidx);CHKERRQ(ierr);   
#else
  ierr = PetscMalloc((Nbs*bs+1)*sizeof(int),&nidx);CHKERRQ(ierr); 
#endif

  for (i=0; i<imax; i++) {
    ierr = ISGetIndices(is_in[i],&idx);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is_in[i],&n);CHKERRQ(ierr);
    for (j=0; j<n ; ++j){
      for (k=0; k<bs; k++)
        nidx[j*bs+k] = idx[j]*bs+k;
    }
    ierr = ISRestoreIndices(is_in[i],&idx);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n*bs,nidx,is_out+i);CHKERRQ(ierr);
  }
  ierr = PetscFree(nidx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap_MPISBAIJ"
int MatIncreaseOverlap_MPISBAIJ(Mat C,int is_max,IS is[],int ov)
{
  int           i,ierr;
  IS            *is_new;

  PetscFunctionBegin;
  ierr = PetscMalloc(is_max*sizeof(IS),&is_new);CHKERRQ(ierr);
  /* Convert the indices into block format */
  ierr = MatCompressIndicesGeneral_MPISBAIJ(C,is_max,is,is_new);CHKERRQ(ierr);
  if (ov < 0){ SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap specified\n");}
  for (i=0; i<ov; ++i) {
    ierr = MatIncreaseOverlap_MPISBAIJ_Once(C,is_max,is_new);CHKERRQ(ierr);
  }
  for (i=0; i<is_max; i++) {ierr = ISDestroy(is[i]);CHKERRQ(ierr);}
  ierr = MatExpandIndices_MPISBAIJ(C,is_max,is_new,is);CHKERRQ(ierr);
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
  int         len,*idx_i,isz,col,*n,*data1,*data2,*data2_i,*data,*data_i,
              size,rank,Mbs,i,j,k,ierr,nrqs,*odata1,*odata2,
              tag1,tag2,flag,proc_id,**odata2_ptr;
  char        *t_p;
  MPI_Comm    comm;
  MPI_Request *s_waits,r_req;
  MPI_Status  *s_status,r_status;
  PetscBT     *table;  /* mark indices of this processor's is[] */
  PetscBT     table_i;
  PetscBT     otable; /* mark indices of other processors' is[] */
  PetscFunctionBegin;

  comm = C->comm;
  size = c->size;
  rank = c->rank;
  Mbs  = c->Mbs;

  ierr = PetscObjectGetNewTag((PetscObject)C,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag2);CHKERRQ(ierr);
 
  /* create tables for marking indices */
  len  = is_max*sizeof(PetscBT) + (Mbs/PETSC_BITS_PER_BYTE+1)*is_max*sizeof(char) + 1;
  ierr = PetscMalloc(len,&table);CHKERRQ(ierr);
  t_p  = (char *)(table + is_max);
  for (i=0; i<is_max; i++) {
    table[i]  = t_p  + (Mbs/PETSC_BITS_PER_BYTE+1)*i; 
  }
  ierr = PetscBTCreate(Mbs,otable);CHKERRQ(ierr);
 
  /* 1. Send this processor's is[] to all other processors */
  /*-------------------------------------------------------*/
  /* Allocate Memory for outgoing messages */
  len  = is_max*sizeof(int); 
  ierr = PetscMalloc(len,&n);CHKERRQ(ierr);

  len = 1 + is_max;
  for (i=0; i<is_max; i++) {
    ierr = ISGetLocalSize(is[i],&n[i]);CHKERRQ(ierr);
    len += n[i]; 
  }
  ierr = PetscMalloc(len*sizeof(int),&data1);CHKERRQ(ierr);

  /* Form the outgoing messages */
  data1[0] = is_max;
  k = is_max + 1;
  for (i=0; i<is_max; i++) { 
    data1[1+i] = n[i];
    ierr = ISGetIndices(is[i],&idx_i);CHKERRQ(ierr);
    for (j=0; j<data1[i+1]; j++){
      data1[k++] = idx_i[j]; 
    }
    ierr = ISRestoreIndices(is[i],&idx_i);CHKERRQ(ierr);
    ierr = ISDestroy(is[i]);CHKERRQ(ierr); 
  }
  if (k != len) SETERRQ2(1,"Error on forming the outgoing messages: k %d != len %d",k,len);
  ierr = PetscFree(n);CHKERRQ(ierr);

  /*  Now  post the sends */
  ierr = PetscMalloc(size*sizeof(MPI_Request),&s_waits);CHKERRQ(ierr);
  k = 0;
  for (proc_id=0; proc_id<size; ++proc_id) { /* send data1 to processor [proc_id] */
    if (proc_id != rank){
      ierr = MPI_Isend(data1,len,MPI_INT,proc_id,tag1,comm,s_waits+k);CHKERRQ(ierr);
      /* printf(" [%d] send %d msg to [%d], data1: \n",rank,len,proc_id); */
      k++;
    }
  }
  
  /* 2. Do local work on this processor's is[] */
  /*-------------------------------------------*/
  ierr = MatIncreaseOverlap_MPISBAIJ_Local(C,data1,MINE,&data,table);CHKERRQ(ierr);

  /* 3. Receive other's is[] and process. Then send back */
  /*-----------------------------------------------------*/
  /* Sending this processor's is[] is done */
  nrqs = size-1;
  ierr = PetscMalloc(size*sizeof(MPI_Status),&s_status);CHKERRQ(ierr);
  ierr = MPI_Waitall(nrqs,s_waits,s_status);CHKERRQ(ierr);
  
  ierr = PetscMalloc(size*sizeof(int**),&odata2_ptr);CHKERRQ(ierr);
  k = 0;
  do {
    /* Receive messages */
    ierr = MPI_Iprobe(MPI_ANY_SOURCE,tag1,comm,&flag,&r_status);CHKERRQ(ierr);
    if (flag){
      ierr = MPI_Get_count(&r_status,MPI_INT,&len);CHKERRQ(ierr);
      proc_id = r_status.MPI_SOURCE;
      ierr = PetscMalloc(len*sizeof(int),&odata1);CHKERRQ(ierr);
      ierr = MPI_Irecv(odata1,len,MPI_INT,proc_id,r_status.MPI_TAG,comm,&r_req);CHKERRQ(ierr);
      /*  printf(" [%d] recv %d msg from [%d]\n",rank,len,proc_id); */

      /*  Process messages */
      ierr = MatIncreaseOverlap_MPISBAIJ_Local(C,odata1,OTHER,&odata2_ptr[k],&otable);CHKERRQ(ierr);
      odata2 = odata2_ptr[k];
      len = 1 + odata2[0];
      for (i=0; i<odata2[0]; i++){
        len += odata2[1 + i];
      }

      /* Send messages back */
      ierr = MPI_Isend(odata2,len,MPI_INT,proc_id,tag2,comm,s_waits+k);CHKERRQ(ierr);
      /* printf(" [%d] send %d msg back to [%d] \n",rank,len,proc_id); */

      ierr = PetscFree(odata1);CHKERRQ(ierr);
      k++;
    }
  } while (k < nrqs);

  /* 4. Receive work done on other processors, then merge */
  /*--------------------------------------------------------*/
  /* Allocate memory for incoming data */
  len = (1+is_max*(Mbs+1));
  ierr = PetscMalloc(len*sizeof(int),&data2);CHKERRQ(ierr); 

  /* Sending others' is[] is done */
  ierr = MPI_Waitall(nrqs,s_waits,s_status);CHKERRQ(ierr);
  for (k=0; k<nrqs; k++){
    ierr = PetscFree(odata2_ptr[k]);CHKERRQ(ierr); 
  }
  ierr = PetscFree(odata2_ptr);CHKERRQ(ierr);

  k = 0;
  do {
    /* Receive messages */
    ierr = MPI_Iprobe(MPI_ANY_SOURCE,tag2,comm,&flag,&r_status);
    if (flag){
      ierr = MPI_Get_count(&r_status,MPI_INT,&len);CHKERRQ(ierr);
      proc_id = r_status.MPI_SOURCE;
      ierr = MPI_Irecv(data2,len,MPI_INT,proc_id,r_status.MPI_TAG,comm,&r_req);CHKERRQ(ierr);
      /* printf(" [%d] recv %d msg from [%d], data2:\n",rank,len,proc_id); */

      /* Add data2 into data */
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
      k++;
    }
  } while (k < nrqs);

  /* 5. Create new is[] */
  /*--------------------*/ 
  for (i=0; i<is_max; i++) {
    data_i = data + 1 + is_max + Mbs*i;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,data[1+i],data_i,is+i);CHKERRQ(ierr);
  }
  ierr = PetscFree(data1);CHKERRQ(ierr); 
  ierr = PetscFree(data2);CHKERRQ(ierr); 
  ierr = PetscFree(data);CHKERRQ(ierr); 
  ierr = PetscFree(s_waits);CHKERRQ(ierr);
  ierr = PetscFree(s_status);CHKERRQ(ierr);
  ierr = PetscFree(table);CHKERRQ(ierr);
  ierr = PetscBTDestroy(otable);CHKERRQ(ierr); 
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
       data_new  - whose = MINE:
                     holds input and newly found indices in the same format as data
                   whose = OTHER:
                     only holds the newly found indices
       table     - table[i]: mark the indices of is[i], i=0,...,is_max. Used only in the case 'whose=MINE'.
*/
static int MatIncreaseOverlap_MPISBAIJ_Local(Mat C,int *data,int whose,int **data_new,PetscBT *table)
{
  Mat_MPISBAIJ *c = (Mat_MPISBAIJ*)C->data;
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)(c->A)->data;
  Mat_SeqBAIJ  *b = (Mat_SeqBAIJ*)(c->B)->data;
  int          ierr,row,mbs,Mbs,*nidx,*nidx_i,col,isz,isz0,*ai,*aj,*bi,*bj,*garray,rstart,l;
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
  
  ierr = PetscMalloc((1+is_max*(Mbs+1))*sizeof(int),&nidx);CHKERRQ(ierr); 
  nidx[0] = is_max; 

  idx_i  = data + is_max + 1; /* ptr to input is[0] array */
  nidx_i = nidx + is_max + 1; /* ptr to output is[0] array */
  for (i=0; i<is_max; i++) { /* for each is */
    isz  = 0;
    n = data[1+i]; /* size of input is[i] */

    /* initialize table_i, set table0 */
    if (whose == MINE){ /* process this processor's is[] */
      table_i = table[i];
      nidx_i  = nidx + 1+ is_max + Mbs*i;
    } else {            /* process other processor's is[] - only use one temp table */
      table_i = *table;
    }
    ierr = PetscBTMemzero(Mbs,table_i);CHKERRQ(ierr);
    ierr = PetscBTMemzero(Mbs,table0);CHKERRQ(ierr);
    if (n > 0) {
      isz0 = 0; 
      for (j=0; j<n; j++){
        col = idx_i[j]; 
        if (col >= Mbs) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"index col %d >= Mbs %d",col,Mbs);
        if(!PetscBTLookupSet(table_i,col)) { 
          ierr = PetscBTSet(table0,col);CHKERRQ(ierr);
          if (whose == MINE) {nidx_i[isz0] = col;}
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
            if (PetscBTLookup(table0,col)){
              if (!PetscBTLookupSet(table_i,row+rstart)) {nidx_i[isz++] = row+rstart;}
              break; /* for l = start; l<end ; l++) */
            }
          } 
          for (l = b_start; l<b_end ; l++){ /* Bmat */
            col = garray[bj[l]];
            if (PetscBTLookup(table0,col)){
              if (!PetscBTLookupSet(table_i,row+rstart)) {nidx_i[isz++] = row+rstart;}
              break; /* for l = start; l<end ; l++) */
            }
          } 
        }
      } 
    } /* if (n > 0) */

    if (i < is_max - 1){
      idx_i  += n;   /* ptr to input is[i+1] array */
      nidx_i += isz; /* ptr to output is[i+1] array */
    }
    nidx[1+i] = isz; /* size of new is[i] */
  } /* for each is */
  *data_new = nidx;
  ierr = PetscBTDestroy(table0);CHKERRQ(ierr); 
  
  PetscFunctionReturn(0);
}


