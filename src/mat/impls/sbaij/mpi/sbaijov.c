/*$Id: sbaijov.c,v 1.65 2001/08/06 21:15:42 bsmith Exp $*/

/*
   Routines to compute overlapping regions of a parallel MPI matrix.
   Used for finding submatrices that were shared across processors.
*/
#include "src/mat/impls/sbaij/mpi/mpisbaij.h"
#include "petscbt.h"

static int MatIncreaseOverlap_MPISBAIJ_Once(Mat,int,IS *);
static int MatIncreaseOverlap_MPISBAIJ_Local(Mat,int *,int **,PetscBT*);
static int MatIncreaseOverlap_MPISBAIJ_Receive(Mat,int,int **,int**,int*);
 
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
  int i,ierr;
  IS  *is_new;

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

/*  All data are packed in the format:
       data[0]          = is_max, no of is 
       data[1]          = size of is[0]
        ...
       data[is_max]     = size of is[is_max-1]
       data[is_max + 1] = data(is[0])
        ...
       data[is_max+1+sum(size of is[k]), k=0,...,i-1] = data(is[i])
        ...
*/
#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap_MPISBAIJ_Once"
static int MatIncreaseOverlap_MPISBAIJ_Once(Mat C,int is_max,IS is[])
{
  Mat_MPISBAIJ  *c = (Mat_MPISBAIJ*)C->data;
  int         len,*idx_i,*nidx_i,isz,col,*n;
  int         size,rank,Mbs,i,j,k,ierr,nrqs,*data1,*data2,*data3,*odata1,*odata2;
  int         *mydata,*mydata_i;
  int         tag1,tag2,flag,proc_id,**odata2_ptr;
  char        *t_p,*ot_p;
  MPI_Comm    comm;
  MPI_Request *s_waits1,*s_waits2,r_req;
  MPI_Status  *s_status,r_status;
  PetscBT     *table;
  PetscBT     table_i;
  PetscBT     *otable;

  PetscFunctionBegin;

  comm = C->comm;
  size = c->size;
  rank = c->rank;
  Mbs  = c->Mbs;

  ierr = PetscObjectGetNewTag((PetscObject)C,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag2);CHKERRQ(ierr);

  /* create tables for marking indices */
  len  = is_max*sizeof(PetscBT) + (Mbs/PETSC_BITS_PER_BYTE+1)*is_max*sizeof(char) + 1;
  ierr = PetscMalloc(2*len,&table);CHKERRQ(ierr);
  otable = (PetscBT *)(table + len);
  t_p  = (char *)(table + is_max);
  ot_p  = (char *)(otable + is_max);
  for (i=0; i<is_max; i++) {
    j = (Mbs/PETSC_BITS_PER_BYTE+1)*i;
    table[i]  = t_p  + j; 
    otable[i] = ot_p + j; 
  }

  /* 1. Send is[] in this processor to all other processors */
  /*--------------------------------------------------------*/
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

  /*  Now  post the sends */
  ierr = PetscMalloc(size*sizeof(MPI_Request),&s_waits1);CHKERRQ(ierr);
  
  k = 0;
  for (proc_id=0; proc_id<size; ++proc_id) { /* send data1 to processor [proc_id] */
    if (proc_id != rank){
      ierr = MPI_Isend(data1,len,MPI_INT,proc_id,tag1,comm,s_waits1+k);CHKERRQ(ierr);
      /* printf(" [%d] send %d msg to [%d] \n",rank,len,proc_id); */
      k++;
    }
  }

  /* 2. Do local work on this processor's is[] */
  /*-------------------------------------------*/
  ierr = MatIncreaseOverlap_MPISBAIJ_Local(C,data1,&data2,table);CHKERRQ(ierr);

  /* 3. Receive other's is[] and process. Then send back */
  /*----------------------------------------------------*/
  /* Send is done */
  nrqs = size-1;
  ierr = PetscMalloc(size*sizeof(MPI_Status),&s_status);CHKERRQ(ierr);
  ierr = MPI_Waitall(nrqs,s_waits1,s_status);CHKERRQ(ierr);

  ierr = PetscFree(data1);CHKERRQ(ierr);  /* keep - move after merge !!! */
  ierr = PetscMalloc(size*sizeof(MPI_Request),&s_waits2);CHKERRQ(ierr);
  
  ierr = PetscMalloc(size*sizeof(int**),&odata2_ptr);
  k = 0;
  do {
    /* Receive messages */
    ierr = MPI_Iprobe(MPI_ANY_SOURCE,tag1,comm,&flag,&r_status);
    if (flag){
      ierr = MPI_Get_count(&r_status,MPI_INT,&len);
      proc_id = r_status.MPI_SOURCE;
      ierr = PetscMalloc(len*sizeof(int),&odata1);CHKERRQ(ierr);
      ierr = MPI_Irecv(odata1,len,MPI_INT,proc_id,r_status.MPI_TAG,comm,&r_req);
      /* printf(" [%d] recv %d msg from [%d]\n",rank,len,proc_id); */

      /*  Process messages */
      ierr = MatIncreaseOverlap_MPISBAIJ_Local(C,odata1,&odata2_ptr[k],otable);CHKERRQ(ierr);
      odata2 = odata2_ptr[k];
      len = 1 + odata2[0];
      for (i=0; i<odata2[0]; i++){
        len += odata2[1 + i];
      }

      /* Send messages back */
      /* printf(" [%d] send %d msg back to [%d] \n",rank,len,proc_id); */
      ierr = MPI_Isend(odata2,len,MPI_INT,proc_id,tag2,comm,&s_waits2[k]);CHKERRQ(ierr);

      ierr = PetscFree(odata1);CHKERRQ(ierr);
      k++;
    }
  } while (k < nrqs);

  /* 4. Receive work done on other processors, then merge */
  /*--------------------------------------------------------*/
  ierr = MPI_Waitall(nrqs,s_waits2,s_status);CHKERRQ(ierr);
  for (k=0; k<nrqs; k++){
    ierr = PetscFree(odata2_ptr[k]);CHKERRQ(ierr); 
  }
  ierr = PetscFree(odata2_ptr);CHKERRQ(ierr);

  /* allocate memory for merged data */
  ierr = PetscMalloc((1+is_max*(Mbs+1))*sizeof(int),&mydata);CHKERRQ(ierr); 

  /* copy data2 into mydata */
  k = is_max + 1;
  for (i=0; i<is_max; i++){
    mydata[1+i] = data2[1+i]; /* size of is[i] before merge */
    mydata_i = mydata + 1 + is_max + Mbs*i;
    for (j=0; j<data2[1+i]; j++){
      mydata_i[j] = data2[k++];
    }
  }

  k = 0;
  do {
    /* Receive messages */
    ierr = MPI_Iprobe(MPI_ANY_SOURCE,tag2,comm,&flag,&r_status);
    if (flag){
      ierr = MPI_Get_count(&r_status,MPI_INT,&len);
      proc_id = r_status.MPI_SOURCE;
      ierr = ierr = PetscMalloc(len*sizeof(int),&data3);CHKERRQ(ierr);
      ierr = MPI_Irecv(data3,len,MPI_INT,proc_id,r_status.MPI_TAG,comm,&r_req);
      /* printf(" [%d] recv %d msg from [%d]\n",rank,len,proc_id); */

      /*  merge data3 into mydata */
      nidx_i   = data3 + 1 + is_max;
      for (i=0; i<is_max; i++){
        table_i  = table[i];
        mydata_i = mydata + 1 + is_max + Mbs*i;
        isz = mydata[1+i]; /* size of is[i] from data2 */
        
        for (j=0; j<data3[1+i]; j++){
          col = nidx_i[j];
          if (!PetscBTLookupSet(table_i,col)) {mydata_i[isz++] = col;}
        }
        mydata[1+i] = isz;
        if (i < is_max - 1){
          nidx_i += data3[1+i]; /* ptr to is[i+1] array from data3 */
        }
      } /* for (i=0; i<is_max; i++) */

      k++;
      ierr = PetscFree(data3);CHKERRQ(ierr);
    }
  } while (k < nrqs);

  /* 5. Create new is[] */
  /*--------------------*/ 
  for (i=0; i<is_max; i++) {
    mydata_i = mydata + 1 + is_max + Mbs*i;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,mydata[1+i],mydata_i,is+i);CHKERRQ(ierr);
  }

  ierr = PetscFree(mydata);CHKERRQ(ierr);
  ierr = PetscFree(data2);CHKERRQ(ierr); 
  ierr = PetscFree(n);CHKERRQ(ierr);
  ierr = PetscFree(s_waits1);CHKERRQ(ierr);
  ierr = PetscFree(s_waits2);CHKERRQ(ierr);
  ierr = PetscFree(s_status);CHKERRQ(ierr);
  ierr = PetscFree(table);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap_MPISBAIJ_Local"
/*  
   MatIncreaseOverlap_MPISBAIJ_Local - Called by MatIncreaseOverlap, to do 
       the work on the local processor.

     Inputs:
      C      - MAT_MPISBAIJ;
      data   - holds is[] in the format:
        data[0]          = is_max, no of is 
        data[1]          = size of is[0]
        ...
        data[is_max]     = size of is[is_max-1]
        data[is_max + 1] = is[0] array
        ...
        data[is_max+1+sum(n[k]), k=0,...,i-1] = is[i] array
        ...
      
     Output:  
       data_new   - holds new is[] in the same format as data
       table      - table[i]: mark the indices of is[i], i=0,...,is_max.
*/
static int MatIncreaseOverlap_MPISBAIJ_Local(Mat C,int *data,int **data_new,PetscBT *table)
{
  Mat_MPISBAIJ *c = (Mat_MPISBAIJ*)C->data;
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)(c->A)->data;
  Mat_SeqBAIJ  *b = (Mat_SeqBAIJ*)(c->B)->data;
  int          ierr,row,mbs,Mbs,*nidx,*nidx_i,col,isz,isz0,*ai,*aj,bs,*bi,*bj,*garray,rstart,l;
  int          a_start,a_end,b_start,b_end,i,j,k,is_max,*idx_i,n;
  PetscBT      table0; 
  PetscBT      table_i; /* poits to i-th table */
  
  PetscFunctionBegin;
  Mbs = c->Mbs; mbs = a->mbs; bs = a->bs;
  ai = a->i; aj = a->j;
  bi = b->i; bj = b->j;
  garray = c->garray;
  rstart = c->rstart;
  is_max = data[0];

  ierr = PetscBTCreate(Mbs,table0);CHKERRQ(ierr);
  
  ierr = PetscMalloc((1+is_max*(Mbs+1))*sizeof(int),&nidx);CHKERRQ(ierr); 
  nidx[0] = is_max; 

  idx_i  = data + is_max + 1; /* ptr to input is[0] array */
  nidx_i = nidx + is_max + 1; /* ptr to active is[0] array */
  for (i=0; i<is_max; i++) { /* for each is */
    isz  = 0;
    table_i = table[i];
    ierr = PetscBTMemzero(Mbs,table_i);CHKERRQ(ierr);
    n = data[1+i]; /* size of input is[i] */
    
    if (n > 0) {
     
      /* Enter input is[i] into active is[i] */
      for (j=0; j<n; j++){
        col = idx_i[j]; 
        if (col >= Mbs) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"index col %d >= Mbs %d",col,Mbs);
        if(!PetscBTLookupSet(table_i,col)) { nidx_i[isz++] = col;}
      }     
  
      /* set table0 for lookup */
      ierr = PetscBTMemzero(Mbs,table0);CHKERRQ(ierr);
      for (l=0; l<isz; l++) PetscBTSet(table0,nidx_i[l]);

      isz0 = isz; /* size of input is[i] after removing repeated indices */
      k = 0;  /* no. of indices from input is[i] that have been examined */
      for (row=0; row<mbs; row++){ 
        a_start = ai[row]; a_end = ai[row+1];
        b_start = bi[row]; b_end = bi[row+1];
        if (PetscBTLookup(table0,row+rstart)){ /* row is on nidx_i - row search: collect all col in this row */
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
        } else { /* row is not on nidx_i - col serach: add row onto nidx_i if there is a col in nidx_i */
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
      } /* for (row=0; row<mbs; row++) */
    } /* if (n > 0) */

    if (i < is_max - 1){
      idx_i  += n;   /* ptr to input is[i+1] array */
      nidx_i += isz; /* ptr to active is[i+1] array */
    }
    nidx[1+i] = isz; /* size of new is[i] */
  } /* /* for each is */
  *data_new = nidx;
  ierr = PetscBTDestroy(table0);CHKERRQ(ierr); 
  
  PetscFunctionReturn(0);
}


