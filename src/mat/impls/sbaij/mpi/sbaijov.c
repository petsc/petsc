/*$Id: sbaijov.c,v 1.65 2001/08/06 21:15:42 bsmith Exp $*/

/*
   Routines to compute overlapping regions of a parallel MPI matrix.
   Used for finding submatrices that were shared across processors.
*/
#include "src/mat/impls/sbaij/mpi/mpisbaij.h"
#include "petscbt.h"

static int MatIncreaseOverlap_MPISBAIJ_Once(Mat,int,IS *);
static int MatIncreaseOverlap_MPISBAIJ_Local(Mat,int *,int **);
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
#define __FUNCT__ "MatCompressIndicesSorted_MPISBAIJ"
static int MatCompressIndicesSorted_MPISBAIJ(Mat C,int imax,const IS is_in[],IS is_out[])
{
  Mat_MPISBAIJ  *baij = (Mat_MPISBAIJ*)C->data;
  int          ierr,bs=baij->bs,i,j,k,val,n,*idx,*nidx,*idx_local;
  PetscTruth   flg;
#if defined (PETSC_USE_CTABLE)
  int maxsz;
#else
  int Nbs=baij->Nbs;
#endif
  PetscFunctionBegin;
  printf(" ... MatCompressIndicesSorted_MPISBAIJ is called ...\n");
  for (i=0; i<imax; i++) {
    ierr = ISSorted(is_in[i],&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Indices are not sorted");
  }
#if defined (PETSC_USE_CTABLE)
  /* Now check max size */
  for (i=0,maxsz=0; i<imax; i++) {
    ierr = ISGetIndices(is_in[i],&idx);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is_in[i],&n);CHKERRQ(ierr);
    if (n%bs !=0) SETERRQ(1,"Indices are not block ordered");
    n = n/bs; /* The reduced index size */
    if (n > maxsz) maxsz = n;
  }
  ierr = PetscMalloc((maxsz+1)*sizeof(int),&nidx);CHKERRQ(ierr);   
#else
  ierr = PetscMalloc((Nbs+1)*sizeof(int),&nidx);CHKERRQ(ierr); 
#endif
  /* Now check if the indices are in block order */
  for (i=0; i<imax; i++) {
    ierr = ISGetIndices(is_in[i],&idx);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is_in[i],&n);CHKERRQ(ierr);
    if (n%bs !=0) SETERRQ(1,"Indices are not block ordered");

    n = n/bs; /* The reduced index size */
    idx_local = idx;
    for (j=0; j<n ; j++) {
      val = idx_local[0];
      if (val%bs != 0) SETERRQ(1,"Indices are not block ordered");
      for (k=0; k<bs; k++) {
        if (val+k != idx_local[k]) SETERRQ(1,"Indices are not block ordered");
      }
      nidx[j] = val/bs;
      idx_local +=bs;
    }
    ierr = ISRestoreIndices(is_in[i],&idx);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n,nidx,(is_out+i));CHKERRQ(ierr);
  }
  ierr = PetscFree(nidx);CHKERRQ(ierr);

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

#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap_MPISBAIJ_Once"
static int MatIncreaseOverlap_MPISBAIJ_Once(Mat C,int is_max,IS is[])
{
  Mat_MPISBAIJ  *c = (Mat_MPISBAIJ*)C->data;
  int         **idx,*n,len,*idx_i;
  int         size,rank,Mbs,i,j,k,ierr,nrqs,msz,*outdat,*indat;
  int         tag1,tag2,flag,proc_id;
  int *nidx,*nidx_i;
  MPI_Comm    comm;
  MPI_Request *s_waits1,*s_waits2,r_req;
  MPI_Status  *s_status,r_status;

  PetscFunctionBegin;

  comm = C->comm;
  size = c->size;
  rank = c->rank;
  Mbs  = c->Mbs;

  /* 1. Send is[] to all other processors */
  /*--------------------------------------*/
  /* This processor sends its is[] to all other processors in the format:
       outdat[0]          = is_max, no of is in this processor
       outdat[1]          = n[0], size of is[0]
        ...
       outdat[is_max]     = n[is_max-1], size of is[is_max-1]
       outdat[is_max + 1] = data(is[0])
        ...
       outdat[is_max+1+sum(n[k]), k=0,...,i-1] = data(is[i])
        ...
  */
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag2);CHKERRQ(ierr);
  printf(" [%d] tags: %d, %d\n",rank,tag1,tag2); 

  len  = (is_max+1)*sizeof(int*)+ (is_max)*sizeof(int); 
  ierr = PetscMalloc(len,&idx);CHKERRQ(ierr);
  n    = (int*)(idx + is_max);
  
  /* Allocate Memory for outgoing messages */
  len = 1 + is_max;
  for (i=0; i<is_max; i++) {
    ierr = ISGetIndices(is[i],&idx[i]);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is[i],&n[i]);CHKERRQ(ierr);
    len += n[i];
  }
  ierr = PetscMalloc(len*sizeof(int),&outdat);CHKERRQ(ierr);
  
  /* Form the outgoing messages */
  outdat[0] = is_max;
  for (i=0; i<is_max; i++) {
    outdat[i+1] = n[i];
  }
  k = is_max + 1;
  for (i=0; i<is_max; i++) { /* for is[i] */
    idx_i = idx[i];
    for (j=0; j<n[i]; j++){
      outdat[k] = *(idx_i); 
      /* if (!rank) printf(" outdat[%d] = %d\n",k,outdat[k] ); */
      k++; idx_i++;
    }
    /* printf(" [%d] n[%d]=%d, k: %d, \n",rank,i,n[i],k); */
  }
  if (k != len) SETERRQ3(1,"[%d] Error on forming the outgoing messages: k %d != len %d",rank,k,len);

  /*  Now  post the sends */
  ierr = PetscMalloc(size*sizeof(MPI_Request),&s_waits1);CHKERRQ(ierr);
  
  k = 0;
  for (proc_id=0; proc_id<size; ++proc_id) { /* send outdat to processor [proc_id] */
    if (proc_id != rank){
      ierr = MPI_Isend(outdat,len,MPI_INT,proc_id,tag1,comm,s_waits1+k);CHKERRQ(ierr);
      printf(" [%d] send %d msg to [%d] \n",rank,len,proc_id); 
      k++;
    }
  }

  /* 2. Do local work on this processor's is[] */
  /*-------------------------------------------*/
  ierr = MatIncreaseOverlap_MPISBAIJ_Local(C,outdat,&nidx);CHKERRQ(ierr);

  for (i=0; i<is_max; i++){
    ierr = ISRestoreIndices(is[i],idx+i);CHKERRQ(ierr);
    ierr = ISDestroy(is[i]);CHKERRQ(ierr); 
  }

  /* 3. Receive other's is[] and process. Then send back */
  /*----------------------------------------------------*/
  /* Send is done */
  nrqs = size-1;
  ierr = PetscMalloc(size*sizeof(MPI_Status),&s_status);CHKERRQ(ierr);
  ierr = MPI_Waitall(nrqs,s_waits1,s_status);CHKERRQ(ierr);

  /* save n[i] */
  for (i=0; i<is_max; i++){
    n[i] = outdat[1+i];
  }
  ierr = PetscFree(outdat);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(MPI_Request),&s_waits2);CHKERRQ(ierr);
  k = 0;
  do {
    /* Receive messages */
    ierr = MPI_Iprobe(MPI_ANY_SOURCE,tag1,comm,&flag,&r_status);
    if (flag){
      ierr = MPI_Get_count(&r_status,MPI_INT,&len);
      proc_id = r_status.MPI_SOURCE;
      ierr = PetscMalloc(len*sizeof(int),&indat);CHKERRQ(ierr);
      ierr = MPI_Irecv(indat,len,MPI_INT,proc_id,r_status.MPI_TAG,comm,&r_req);
      printf(" [%d] recv %d msg from [%d]\n",rank,len,proc_id); 

      /*  Process messages -- not done yet */
      len = indat[0];
      ierr = PetscMalloc(len*sizeof(int),&outdat);CHKERRQ(ierr);
      for (i=0; i<len; i++){outdat[i] = indat[i+1];}

      /* Send messages back */
      printf(" [%d] send %d msg back to [%d] \n",rank,len,proc_id);
      ierr = MPI_Isend(outdat,len,MPI_INT,proc_id,tag2,comm,&s_waits2[k]);CHKERRQ(ierr);

      k++;
      ierr = PetscFree(outdat);CHKERRQ(ierr);
      ierr = PetscFree(indat);CHKERRQ(ierr);
    }
  } while (k < nrqs);

  /* 4. Receive work done on other processors, then process */
  /*--------------------------------------------------------*/
  ierr = MPI_Waitall(nrqs,s_waits2,s_status);CHKERRQ(ierr);
  k = 0;
  do {
    /* Receive messages */
    ierr = MPI_Iprobe(MPI_ANY_SOURCE,tag2,comm,&flag,&r_status);
    if (flag){
      ierr = MPI_Get_count(&r_status,MPI_INT,&len);
      proc_id = r_status.MPI_SOURCE;
      ierr = ierr = PetscMalloc(len*sizeof(int),&indat);CHKERRQ(ierr);
      ierr = MPI_Irecv(indat,len,MPI_INT,proc_id,r_status.MPI_TAG,comm,&r_req);
      printf(" [%d] recv %d msg from [%d]\n",rank,len,proc_id); 

      /*  Process messages -- not done yet */
    

      k++;
      ierr = PetscFree(indat);CHKERRQ(ierr);
    }
  } while (k < nrqs);

  /* 5. Create new is[] */
  /*--------------------*/ 
  for (i=0; i<is_max; i++) {
    nidx_i = nidx+i*Mbs;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n[i],nidx_i,is+i);CHKERRQ(ierr);
  }
  ierr = PetscFree(nidx);CHKERRQ(ierr); 
  ierr = PetscFree(idx);CHKERRQ(ierr);
  ierr = PetscFree(s_waits1);CHKERRQ(ierr);
  ierr = PetscFree(s_waits2);CHKERRQ(ierr);
  ierr = PetscFree(s_status);CHKERRQ(ierr);
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
*/
static int MatIncreaseOverlap_MPISBAIJ_Local(Mat C,int *data,int **data_new)
{
  Mat_MPISBAIJ *c = (Mat_MPISBAIJ*)C->data;
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)(c->A)->data;
  Mat_SeqBAIJ  *b = (Mat_SeqBAIJ*)(c->B)->data;
  int          ierr,row,mbs,Mbs,*nidx,*nidx_i,col,isz,isz0,*ai,*aj,bs,*bi,*bj,*garray,rstart,l;
  int          a_start,a_end,b_start,b_end,i,j,k,is_max,*idx_i,n;
  PetscBT      table;
  PetscBT      table0; 

  PetscFunctionBegin;
  Mbs = c->Mbs; mbs = a->mbs; bs = a->bs;
  ai = a->i; aj = a->j;
  bi = b->i; bj = b->j;
  garray = c->garray;
  rstart = c->rstart;
  is_max = data[0];

  ierr = PetscMalloc(is_max*Mbs*sizeof(int),&nidx);CHKERRQ(ierr); 
  ierr = PetscBTCreate(Mbs,table0);CHKERRQ(ierr);
  ierr = PetscBTCreate(Mbs,table);CHKERRQ(ierr);

  idx_i  = data + is_max + 1; /* ptr to is[0] array */
  for (i=0; i<is_max; i++) { /* for each is */
    isz  = 0;
    ierr = PetscBTMemzero(Mbs,table);CHKERRQ(ierr);
    nidx_i = nidx+i*Mbs;  /*  holds new is[i] array */
    n = data[1+i]; /* size of is[i] */
 
    /* Enter these into the temp arrays i.e mark table[row], enter row into new index */
    for (j=0; j<n; j++){
      col = idx_i[j]; 
      if (col >= Mbs) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"index col %d >= Mbs %d",col,Mbs);
      if(!PetscBTLookupSet(table,col)) { nidx_i[isz++] = col;}
    }    
  
    k = 0;
    /* set table0 for lookup */
    ierr = PetscBTMemzero(mbs,table0);CHKERRQ(ierr);
    for (l=k; l<isz; l++) PetscBTSet(table0,nidx_i[l]);

    isz0 = isz; /* length of nidx_i[] before updating */
    for (row=0; row<mbs; row++){ 
      a_start = ai[row]; a_end = ai[row+1];
      b_start = bi[row]; b_end = bi[row+1];
      if (PetscBTLookup(table0,row+rstart)){ /* row is on nidx_i - row search: collect all col in this row */
        /* printf(" [%d] is[%d] row %d is on nidx_i\n",rank,i,row+rstart); */
        for (l = a_start; l<a_end ; l++){ /* Amat */
          col = aj[l] + rstart;
          if (!PetscBTLookupSet(table,col)) {nidx_i[isz++] = col;}
        }
        for (l = b_start; l<b_end ; l++){ /* Bmat */
          col = garray[bj[l]];
          if (!PetscBTLookupSet(table,col)) {nidx_i[isz++] = col;}
        }
        k++;
        if (k >= isz0) break; /* for (row=0; row<mbs; row++) */
      } else { /* row is not on nidx_i - col serach: add row onto nidx_i if there is a col in nidx_i */
        for (l = a_start; l<a_end ; l++){ /* Amat */
          col = aj[l] + rstart;
          if (PetscBTLookup(table0,col)){
            if (!PetscBTLookupSet(table,row+rstart)) {nidx_i[isz++] = row+rstart;}
            break; /* for l = start; l<end ; l++) */
          }
        } 
        for (l = b_start; l<b_end ; l++){ /* Bmat */
          col = garray[bj[l]];
          if (PetscBTLookup(table0,col)){
            if (!PetscBTLookupSet(table,row+rstart)) {nidx_i[isz++] = row+rstart;}
            break; /* for l = start; l<end ; l++) */
          }
        } 
      }
    } /* for (row=0; row<mbs; row++) */

    if (i < is_max - 1){
      idx_i  += n; /* ptr to is[i+1] array */
    }
    data[1+i] = isz;
  } /* for each is */
  *data_new = nidx;
  ierr = PetscBTDestroy(table);CHKERRQ(ierr);
  ierr = PetscBTDestroy(table0);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap_MPISBAIJ_Receive"
/*     
      MatIncreaseOverlap_MPISBAIJ_Receive - Process the recieved messages,
         and return the output

         Input:
           C    - the matrix
           nrqr - no of messages being processed.
           rbuf - an array of pointers to the recieved requests
           
         Output:
           xdata - array of messages to be sent back
           isz1  - size of each message

  For better efficiency perhaps we should malloc seperately each xdata[i],
then if a remalloc is required we need only copy the data for that one row
rather then all previous rows as it is now where a single large chunck of 
memory is used.

*/
static int MatIncreaseOverlap_MPISBAIJ_Receive(Mat C,int nrqr,int **rbuf,int **xdata,int * isz1)
{
  Mat_MPISBAIJ *c = (Mat_MPISBAIJ*)C->data;
  Mat         A = c->A,B = c->B;
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)A->data;
  Mat_SeqBAIJ  *b = (Mat_SeqBAIJ*)B->data;
  int         rstart,cstart,*ai,*aj,*bi,*bj,*garray,i,j,k;
  int         row,total_sz,ct,ct1,ct2,ct3,mem_estimate,oct2,l,start,end;
  int         val,max1,max2,rank,Mbs,no_malloc =0,*tmp,new_estimate,ctr;
  int         *rbuf_i,kmax,rbuf_0,ierr;
  PetscBT     xtable;

  PetscFunctionBegin;
  rank   = c->rank;
  Mbs    = c->Mbs;
  rstart = c->rstart;
  cstart = c->cstart;
  ai     = a->i;
  aj     = a->j;
  bi     = b->i;
  bj     = b->j;
  garray = c->garray;
  
  
  for (i=0,ct=0,total_sz=0; i<nrqr; ++i) {
    rbuf_i  =  rbuf[i]; 
    rbuf_0  =  rbuf_i[0];
    ct     += rbuf_0;
    for (j=1; j<=rbuf_0; j++) { total_sz += rbuf_i[2*j]; }
  }
  
  if (c->Mbs) max1 = ct*(a->nz +b->nz)/c->Mbs;
  else        max1 = 1;
  mem_estimate = 3*((total_sz > max1 ? total_sz : max1)+1);
  ierr         = PetscMalloc(mem_estimate*sizeof(int),&xdata[0]);CHKERRQ(ierr);
  ++no_malloc;
  ierr         = PetscBTCreate(Mbs,xtable);CHKERRQ(ierr);
  ierr         = PetscMemzero(isz1,nrqr*sizeof(int));CHKERRQ(ierr);
  
  ct3 = 0;
  for (i=0; i<nrqr; i++) { /* for easch mesg from proc i */
    rbuf_i =  rbuf[i]; 
    rbuf_0 =  rbuf_i[0];
    ct1    =  2*rbuf_0+1;
    ct2    =  ct1;
    ct3    += ct1;
    for (j=1; j<=rbuf_0; j++) { /* for each IS from proc i*/
      ierr = PetscBTMemzero(Mbs,xtable);CHKERRQ(ierr);
      oct2 = ct2;
      kmax = rbuf_i[2*j];
      for (k=0; k<kmax; k++,ct1++) { 
        row = rbuf_i[ct1];
        if (!PetscBTLookupSet(xtable,row)) { 
          if (!(ct3 < mem_estimate)) {
            new_estimate = (int)(1.5*mem_estimate)+1;
            ierr = PetscMalloc(new_estimate * sizeof(int),&tmp);CHKERRQ(ierr);
            ierr = PetscMemcpy(tmp,xdata[0],mem_estimate*sizeof(int));CHKERRQ(ierr);
            ierr = PetscFree(xdata[0]);CHKERRQ(ierr);
            xdata[0]     = tmp;
            mem_estimate = new_estimate; ++no_malloc;
            for (ctr=1; ctr<=i; ctr++) { xdata[ctr] = xdata[ctr-1] + isz1[ctr-1];}
          }
          xdata[i][ct2++] = row;
          ct3++;
        }
      }
      for (k=oct2,max2=ct2; k<max2; k++)  {
        row   = xdata[i][k] - rstart;
        start = ai[row];
        end   = ai[row+1];
        for (l=start; l<end; l++) {
          val = aj[l] + cstart;
          if (!PetscBTLookupSet(xtable,val)) {
            if (!(ct3 < mem_estimate)) {
              new_estimate = (int)(1.5*mem_estimate)+1;
              ierr = PetscMalloc(new_estimate * sizeof(int),&tmp);CHKERRQ(ierr);
              ierr = PetscMemcpy(tmp,xdata[0],mem_estimate*sizeof(int));CHKERRQ(ierr);
              ierr = PetscFree(xdata[0]);CHKERRQ(ierr);
              xdata[0]     = tmp;
              mem_estimate = new_estimate; ++no_malloc;
              for (ctr=1; ctr<=i; ctr++) { xdata[ctr] = xdata[ctr-1] + isz1[ctr-1];}
            }
            xdata[i][ct2++] = val;
            ct3++;
          }
        }
        start = bi[row];
        end   = bi[row+1];
        for (l=start; l<end; l++) {
          val = garray[bj[l]];
          if (!PetscBTLookupSet(xtable,val)) { 
            if (!(ct3 < mem_estimate)) { 
              new_estimate = (int)(1.5*mem_estimate)+1;
              ierr = PetscMalloc(new_estimate * sizeof(int),&tmp);CHKERRQ(ierr);
              ierr = PetscMemcpy(tmp,xdata[0],mem_estimate*sizeof(int));CHKERRQ(ierr);
              ierr = PetscFree(xdata[0]);CHKERRQ(ierr);
              xdata[0]     = tmp;
              mem_estimate = new_estimate; ++no_malloc;
              for (ctr =1; ctr <=i; ctr++) { xdata[ctr] = xdata[ctr-1] + isz1[ctr-1];}
            }
            xdata[i][ct2++] = val;
            ct3++;
          }  
        } 
      }
      /* Update the header*/
      xdata[i][2*j]   = ct2 - oct2; /* Undo the vector isz1 and use only a var*/
      xdata[i][2*j-1] = rbuf_i[2*j-1];
    }
    xdata[i][0] = rbuf_0;
    xdata[i+1]  = xdata[i] + ct2;
    isz1[i]     = ct2; /* size of each message */
  }
  ierr = PetscBTDestroy(xtable);CHKERRQ(ierr);
  PetscLogInfo(0,"MatIncreaseOverlap_MPISBAIJ:[%d] Allocated %d bytes, required %d, no of mallocs = %d\n",rank,mem_estimate,ct3,no_malloc);    
  PetscFunctionReturn(0);
}  


