/*$Id: baijov.c,v 1.65 2001/08/06 21:15:42 bsmith Exp $*/

/*
   Routines to compute overlapping regions of a parallel MPI matrix
  and to find submatrices that were shared across processors.
*/
#include "src/mat/impls/baij/mpi/mpibaij.h"
#include "petscbt.h"

static int MatIncreaseOverlap_MPIBAIJ_Once(Mat,int,IS *);
static int MatIncreaseOverlap_MPIBAIJ_Local(Mat,int,char **,int*,int**);
static int MatIncreaseOverlap_MPIBAIJ_Receive(Mat,int,int **,int**,int*);
EXTERN int MatGetRow_MPIBAIJ(Mat,int,int*,int**,PetscScalar**);
EXTERN int MatRestoreRow_MPIBAIJ(Mat,int,int*,int**,PetscScalar**);
 
#undef __FUNCT__  
#define __FUNCT__ "MatCompressIndicesGeneral_MPIBAIJ"
static int MatCompressIndicesGeneral_MPIBAIJ(Mat C,int imax,IS *is_in,IS *is_out)
{
  Mat_MPIBAIJ  *baij = (Mat_MPIBAIJ*)C->data;
  int          ierr,isz,bs = baij->bs,n,i,j,*idx,ival;
#if defined (PETSC_USE_CTABLE)
  PetscTable   gid1_lid1;
  int tt, gid1, *nidx;
  PetscTablePosition tpos;
#else
  int          Nbs,*nidx;
  PetscBT      table;
#endif

  PetscFunctionBegin;
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
#define __FUNCT__ "MatCompressIndicesSorted_MPIBAIJ"
static int MatCompressIndicesSorted_MPIBAIJ(Mat C,int imax,IS *is_in,IS *is_out)
{
  Mat_MPIBAIJ  *baij = (Mat_MPIBAIJ*)C->data;
  int          ierr,bs=baij->bs,i,j,k,val,n,*idx,*nidx,*idx_local;
  PetscTruth   flg;
#if defined (PETSC_USE_CTABLE)
  int maxsz;
#else
  int Nbs=baij->Nbs;
#endif
  PetscFunctionBegin;
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
#define __FUNCT__ "MatExpandIndices_MPIBAIJ"
static int MatExpandIndices_MPIBAIJ(Mat C,int imax,IS *is_in,IS *is_out)
{
  Mat_MPIBAIJ  *baij = (Mat_MPIBAIJ*)C->data;
  int          ierr,bs = baij->bs,n,i,j,k,*idx,*nidx;
#if defined (PETSC_USE_CTABLE)
  int maxsz;
#else
  int Nbs = baij->Nbs;
#endif
  PetscFunctionBegin;
 
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
#define __FUNCT__ "MatIncreaseOverlap_MPIBAIJ"
int MatIncreaseOverlap_MPIBAIJ(Mat C,int imax,IS *is,int ov)
{
  int i,ierr;
  IS  *is_new;

  PetscFunctionBegin;
  ierr = PetscMalloc(imax*sizeof(IS),&is_new);CHKERRQ(ierr);
  /* Convert the indices into block format */
  ierr = MatCompressIndicesGeneral_MPIBAIJ(C,imax,is,is_new);CHKERRQ(ierr);
  if (ov < 0){ SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap specified\n");}
  for (i=0; i<ov; ++i) {
    ierr = MatIncreaseOverlap_MPIBAIJ_Once(C,imax,is_new);CHKERRQ(ierr);
  }
  for (i=0; i<imax; i++) {ierr = ISDestroy(is[i]);CHKERRQ(ierr);}
  ierr = MatExpandIndices_MPIBAIJ(C,imax,is_new,is);CHKERRQ(ierr);
  for (i=0; i<imax; i++) {ierr = ISDestroy(is_new[i]);CHKERRQ(ierr);}
  ierr = PetscFree(is_new);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Sample message format:
  If a processor A wants processor B to process some elements corresponding
  to index sets 1s[1], is[5]
  mesg [0] = 2   (no of index sets in the mesg)
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
#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap_MPIBAIJ_Once"
static int MatIncreaseOverlap_MPIBAIJ_Once(Mat C,int imax,IS *is)
{
  Mat_MPIBAIJ  *c = (Mat_MPIBAIJ*)C->data;
  int         **idx,*n,*w1,*w2,*w3,*w4,*rtable,**data,len,*idx_i;
  int         size,rank,Mbs,i,j,k,ierr,**rbuf,row,proc,nrqs,msz,**outdat,**ptr;
  int         *ctr,*pa,*tmp,nrqr,*isz,*isz1,**xdata,**rbuf2;
  int         *onodes1,*olengths1,tag1,tag2,*onodes2,*olengths2;
  PetscBT     *table;
  MPI_Comm    comm;
  MPI_Request *s_waits1,*r_waits1,*s_waits2,*r_waits2;
  MPI_Status  *s_status,*recv_status;

  PetscFunctionBegin;
  comm   = C->comm;
  size   = c->size;
  rank   = c->rank;
  Mbs    = c->Mbs;

  ierr = PetscObjectGetNewTag((PetscObject)C,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag2);CHKERRQ(ierr);

  len    = (imax+1)*sizeof(int*)+ (imax + Mbs)*sizeof(int);
  ierr   = PetscMalloc(len,&idx);CHKERRQ(ierr);
  n      = (int*)(idx + imax);
  rtable = n + imax;
   
  for (i=0; i<imax; i++) {
    ierr = ISGetIndices(is[i],&idx[i]);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is[i],&n[i]);CHKERRQ(ierr);
  }
  
  /* Create hash table for the mapping :row -> proc*/
  for (i=0,j=0; i<size; i++) {
    len = c->rowners[i+1];  
    for (; j<len; j++) {
      rtable[j] = i;
    }
  }

  /* evaluate communication - mesg to who,length of mesg, and buffer space
     required. Based on this, buffers are allocated, and data copied into them*/
  ierr = PetscMalloc(size*4*sizeof(int),&w1);CHKERRQ(ierr);/*  mesg size */
  w2   = w1 + size;       /* if w2[i] marked, then a message to proc i*/
  w3   = w2 + size;       /* no of IS that needs to be sent to proc i */
  w4   = w3 + size;       /* temp work space used in determining w1, w2, w3 */
  ierr = PetscMemzero(w1,size*3*sizeof(int));CHKERRQ(ierr); /* initialise work vector*/
  for (i=0; i<imax; i++) { 
    ierr  = PetscMemzero(w4,size*sizeof(int));CHKERRQ(ierr); /* initialise work vector*/
    idx_i = idx[i];
    len   = n[i];
    for (j=0; j<len; j++) {
      row  = idx_i[j];
      if (row < 0) {
        SETERRQ(1,"Index set cannot have negative entries");
      }
      proc = rtable[row];
      w4[proc]++;
    }
    for (j=0; j<size; j++){ 
      if (w4[j]) { w1[j] += w4[j]; w3[j]++;} 
    }
  }

  nrqs     = 0;              /* no of outgoing messages */
  msz      = 0;              /* total mesg length (for all proc */
  w1[rank] = 0;              /* no mesg sent to intself */
  w3[rank] = 0;
  for (i=0; i<size; i++) {
    if (w1[i])  {w2[i] = 1; nrqs++;} /* there exists a message to proc i */
  }
  /* pa - is list of processors to communicate with */
  ierr = PetscMalloc((nrqs+1)*sizeof(int),&pa);CHKERRQ(ierr);
  for (i=0,j=0; i<size; i++) {
    if (w1[i]) {pa[j] = i; j++;}
  } 

  /* Each message would have a header = 1 + 2*(no of IS) + data */
  for (i=0; i<nrqs; i++) {
    j      = pa[i];
    w1[j] += w2[j] + 2*w3[j];   
    msz   += w1[j];  
  }
  
  /* Determine the number of messages to expect, their lengths, from from-ids */
  ierr = PetscGatherNumberOfMessages(comm,w2,w1,&nrqr);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nrqs,nrqr,w1,&onodes1,&olengths1);CHKERRQ(ierr);

  /* Now post the Irecvs corresponding to these messages */
  ierr = PetscPostIrecvInt(comm,tag1,nrqr,onodes1,olengths1,&rbuf,&r_waits1);CHKERRQ(ierr);
  
  /* Allocate Memory for outgoing messages */
  len    = 2*size*sizeof(int*) + (size+msz)*sizeof(int);
  ierr = PetscMalloc(len,&outdat);CHKERRQ(ierr);
  ptr    = outdat + size;     /* Pointers to the data in outgoing buffers */
  ierr   = PetscMemzero(outdat,2*size*sizeof(int*));CHKERRQ(ierr);
  tmp    = (int*)(outdat + 2*size);
  ctr    = tmp + msz;

  {
    int *iptr = tmp,ict  = 0;
    for (i=0; i<nrqs; i++) {
      j         = pa[i];
      iptr     +=  ict;
      outdat[j] = iptr;
      ict       = w1[j];
    }
  }

  /* Form the outgoing messages */
  /*plug in the headers*/
  for (i=0; i<nrqs; i++) {
    j            = pa[i];
    outdat[j][0] = 0;
    ierr = PetscMemzero(outdat[j]+1,2*w3[j]*sizeof(int));CHKERRQ(ierr);
    ptr[j]       = outdat[j] + 2*w3[j] + 1;
  }
 
  /* Memory for doing local proc's work*/
  { 
    int  *d_p;
    char *t_p;

    len  = (imax)*(sizeof(PetscBT) + sizeof(int*)+ sizeof(int)) + 
      (Mbs)*imax*sizeof(int)  + (Mbs/PETSC_BITS_PER_BYTE+1)*imax*sizeof(char) + 1;
    ierr = PetscMalloc(len,&table);CHKERRQ(ierr);
    ierr = PetscMemzero(table,len);CHKERRQ(ierr);
    data = (int **)(table + imax);
    isz  = (int  *)(data  + imax);
    d_p  = (int  *)(isz   + imax);
    t_p  = (char *)(d_p   + Mbs*imax);
    for (i=0; i<imax; i++) {
      table[i] = t_p + (Mbs/PETSC_BITS_PER_BYTE+1)*i;
      data[i]  = d_p + (Mbs)*i;
    }
  }

  /* Parse the IS and update local tables and the outgoing buf with the data*/
  {
    int     n_i,*data_i,isz_i,*outdat_j,ctr_j;
    PetscBT table_i;

    for (i=0; i<imax; i++) {
      ierr    = PetscMemzero(ctr,size*sizeof(int));CHKERRQ(ierr);
      n_i     = n[i];
      table_i = table[i];
      idx_i   = idx[i];
      data_i  = data[i];
      isz_i   = isz[i];
      for (j=0;  j<n_i; j++) {  /* parse the indices of each IS */
        row  = idx_i[j];
        proc = rtable[row];
        if (proc != rank) { /* copy to the outgoing buffer */
          ctr[proc]++;
          *ptr[proc] = row;
          ptr[proc]++;
        }
        else { /* Update the local table */
          if (!PetscBTLookupSet(table_i,row)) { data_i[isz_i++] = row;}
        }
      }
      /* Update the headers for the current IS */
      for (j=0; j<size; j++) { /* Can Optimise this loop by using pa[] */
        if ((ctr_j = ctr[j])) {
          outdat_j        = outdat[j];
          k               = ++outdat_j[0];
          outdat_j[2*k]   = ctr_j;
          outdat_j[2*k-1] = i;
        }
      }
      isz[i] = isz_i;
    }
  }
  
  /*  Now  post the sends */
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Request),&s_waits1);CHKERRQ(ierr);
  for (i=0; i<nrqs; ++i) {
    j    = pa[i];
    ierr = MPI_Isend(outdat[j],w1[j],MPI_INT,j,tag1,comm,s_waits1+i);CHKERRQ(ierr);
  }
    
  /* No longer need the original indices*/
  for (i=0; i<imax; ++i) {
    ierr = ISRestoreIndices(is[i],idx+i);CHKERRQ(ierr);
  }
  ierr = PetscFree(idx);CHKERRQ(ierr);

  for (i=0; i<imax; ++i) {
    ierr = ISDestroy(is[i]);CHKERRQ(ierr);
  }
  
  /* Do Local work*/
  ierr = MatIncreaseOverlap_MPIBAIJ_Local(C,imax,table,isz,data);CHKERRQ(ierr);

  /* Receive messages*/
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Status),&recv_status);CHKERRQ(ierr);
  ierr = MPI_Waitall(nrqr,r_waits1,recv_status);CHKERRQ(ierr);
  
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Status),&s_status);CHKERRQ(ierr);
  ierr = MPI_Waitall(nrqs,s_waits1,s_status);CHKERRQ(ierr);

  /* Phase 1 sends are complete - deallocate buffers */
  ierr = PetscFree(outdat);CHKERRQ(ierr);
  ierr = PetscFree(w1);CHKERRQ(ierr);

  ierr = PetscMalloc((nrqr+1)*sizeof(int *),&xdata);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqr+1)*sizeof(int),&isz1);CHKERRQ(ierr);
  ierr = MatIncreaseOverlap_MPIBAIJ_Receive(C,nrqr,rbuf,xdata,isz1);CHKERRQ(ierr);
  ierr = PetscFree(rbuf);CHKERRQ(ierr);

  /* Send the data back*/
  /* Do a global reduction to know the buffer space req for incoming messages*/
  {
    int *rw1;
    
    ierr = PetscMalloc(size*sizeof(int),&rw1);CHKERRQ(ierr);
    ierr = PetscMemzero(rw1,size*sizeof(int));CHKERRQ(ierr);

    for (i=0; i<nrqr; ++i) {
      proc      = recv_status[i].MPI_SOURCE;
      if (proc != onodes1[i]) SETERRQ(1,"MPI_SOURCE mismatch");
      rw1[proc] = isz1[i];
    }
      
    ierr = PetscFree(onodes1);CHKERRQ(ierr);
    ierr = PetscFree(olengths1);CHKERRQ(ierr);

    /* Determine the number of messages to expect, their lengths, from from-ids */
    ierr = PetscGatherMessageLengths(comm,nrqr,nrqs,rw1,&onodes2,&olengths2);CHKERRQ(ierr);
    PetscFree(rw1);
  }
  /* Now post the Irecvs corresponding to these messages */
  ierr = PetscPostIrecvInt(comm,tag2,nrqs,onodes2,olengths2,&rbuf2,&r_waits2);CHKERRQ(ierr);
  
  /*  Now  post the sends */
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Request),&s_waits2);CHKERRQ(ierr);
  for (i=0; i<nrqr; ++i) {
    j    = recv_status[i].MPI_SOURCE;
    ierr = MPI_Isend(xdata[i],isz1[i],MPI_INT,j,tag2,comm,s_waits2+i);CHKERRQ(ierr);
  }

  /* receive work done on other processors*/
  {
    int         index,is_no,ct1,max,*rbuf2_i,isz_i,*data_i,jmax;
    PetscBT     table_i;
    MPI_Status  *status2;
     
    ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Status),&status2);CHKERRQ(ierr);

    for (i=0; i<nrqs; ++i) {
      ierr = MPI_Waitany(nrqs,r_waits2,&index,status2+i);CHKERRQ(ierr);
      /* Process the message*/
      rbuf2_i = rbuf2[index];
      ct1     = 2*rbuf2_i[0]+1;
      jmax    = rbuf2[index][0];
      for (j=1; j<=jmax; j++) {
        max     = rbuf2_i[2*j];
        is_no   = rbuf2_i[2*j-1];
        isz_i   = isz[is_no];
        data_i  = data[is_no];
        table_i = table[is_no];
        for (k=0; k<max; k++,ct1++) {
          row = rbuf2_i[ct1];
          if (!PetscBTLookupSet(table_i,row)) { data_i[isz_i++] = row;}   
        }
        isz[is_no] = isz_i;
      }
    }
    ierr = MPI_Waitall(nrqr,s_waits2,status2);CHKERRQ(ierr);
    ierr = PetscFree(status2);CHKERRQ(ierr);
  }
  
  for (i=0; i<imax; ++i) {
    ierr = ISCreateGeneral(PETSC_COMM_SELF,isz[i],data[i],is+i);CHKERRQ(ierr);
  }
  
  
  ierr = PetscFree(onodes2);CHKERRQ(ierr);
  ierr = PetscFree(olengths2);CHKERRQ(ierr);

  ierr = PetscFree(pa);CHKERRQ(ierr);
  ierr = PetscFree(rbuf2);CHKERRQ(ierr);
  ierr = PetscFree(s_waits1);CHKERRQ(ierr);
  ierr = PetscFree(r_waits1);CHKERRQ(ierr);
  ierr = PetscFree(s_waits2);CHKERRQ(ierr);
  ierr = PetscFree(r_waits2);CHKERRQ(ierr);
  ierr = PetscFree(table);CHKERRQ(ierr);
  ierr = PetscFree(s_status);CHKERRQ(ierr);
  ierr = PetscFree(recv_status);CHKERRQ(ierr);
  ierr = PetscFree(xdata[0]);CHKERRQ(ierr);
  ierr = PetscFree(xdata);CHKERRQ(ierr);
  ierr = PetscFree(isz1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap_MPIBAIJ_Local"
/*  
   MatIncreaseOverlap_MPIBAIJ_Local - Called by MatincreaseOverlap, to do 
       the work on the local processor.

     Inputs:
      C      - MAT_MPIBAIJ;
      imax - total no of index sets processed at a time;
      table  - an array of char - size = Mbs bits.
      
     Output:
      isz    - array containing the count of the solution elements correspondign
               to each index set;
      data   - pointer to the solutions
*/
static int MatIncreaseOverlap_MPIBAIJ_Local(Mat C,int imax,PetscBT *table,int *isz,int **data)
{
  Mat_MPIBAIJ *c = (Mat_MPIBAIJ*)C->data;
  Mat         A = c->A,B = c->B;
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)B->data;
  int         start,end,val,max,rstart,cstart,*ai,*aj;
  int         *bi,*bj,*garray,i,j,k,row,*data_i,isz_i;
  PetscBT     table_i;

  PetscFunctionBegin;
  rstart = c->rstart;
  cstart = c->cstart;
  ai     = a->i;
  aj     = a->j;
  bi     = b->i;
  bj     = b->j;
  garray = c->garray;

  
  for (i=0; i<imax; i++) {
    data_i  = data[i];
    table_i = table[i];
    isz_i   = isz[i];
    for (j=0,max=isz[i]; j<max; j++) {
      row   = data_i[j] - rstart;
      start = ai[row];
      end   = ai[row+1];
      for (k=start; k<end; k++) { /* Amat */
        val = aj[k] + cstart;
        if (!PetscBTLookupSet(table_i,val)) { data_i[isz_i++] = val;}  
      }
      start = bi[row];
      end   = bi[row+1];
      for (k=start; k<end; k++) { /* Bmat */
        val = garray[bj[k]]; 
        if (!PetscBTLookupSet(table_i,val)) { data_i[isz_i++] = val;}  
      } 
    }
    isz[i] = isz_i;
  }
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap_MPIBAIJ_Receive"
/*     
      MatIncreaseOverlap_MPIBAIJ_Receive - Process the recieved messages,
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
static int MatIncreaseOverlap_MPIBAIJ_Receive(Mat C,int nrqr,int **rbuf,int **xdata,int * isz1)
{
  Mat_MPIBAIJ *c = (Mat_MPIBAIJ*)C->data;
  Mat         A = c->A,B = c->B;
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)B->data;
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
  PetscLogInfo(0,"MatIncreaseOverlap_MPIBAIJ:[%d] Allocated %d bytes, required %d, no of mallocs = %d\n",rank,mem_estimate,ct3,no_malloc);    
  PetscFunctionReturn(0);
}  

static int MatGetSubMatrices_MPIBAIJ_local(Mat,int,IS *,IS *,MatReuse,Mat *);

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrices_MPIBAIJ"
int MatGetSubMatrices_MPIBAIJ(Mat C,int ismax,IS *isrow,IS *iscol,MatReuse scall,Mat **submat)
{ 
  IS          *isrow_new,*iscol_new;
  Mat_MPIBAIJ *c = (Mat_MPIBAIJ*)C->data;
  int         nmax,nstages_local,nstages,i,pos,max_no,ierr;

  PetscFunctionBegin;
  /* The compression and expansion should be avoided. Does'nt point
     out errors might change the indices hence buggey */

  ierr = PetscMalloc(2*(ismax+1)*sizeof(IS),&isrow_new);CHKERRQ(ierr);
  iscol_new = isrow_new + ismax;
  ierr = MatCompressIndicesSorted_MPIBAIJ(C,ismax,isrow,isrow_new);CHKERRQ(ierr);
  ierr = MatCompressIndicesSorted_MPIBAIJ(C,ismax,iscol,iscol_new);CHKERRQ(ierr);

  /* Allocate memory to hold all the submatrices */
  if (scall != MAT_REUSE_MATRIX) {
    ierr = PetscMalloc((ismax+1)*sizeof(Mat),submat);CHKERRQ(ierr);
  }
  /* Determine the number of stages through which submatrices are done */
  nmax          = 20*1000000 / (c->Nbs * sizeof(int));
  if (!nmax) nmax = 1;
  nstages_local = ismax/nmax + ((ismax % nmax)?1:0);
  
  /* Make sure every porcessor loops through the nstages */
  ierr = MPI_Allreduce(&nstages_local,&nstages,1,MPI_INT,MPI_MAX,C->comm);CHKERRQ(ierr);
  for (i=0,pos=0; i<nstages; i++) {
    if (pos+nmax <= ismax) max_no = nmax;
    else if (pos == ismax) max_no = 0;
    else                   max_no = ismax-pos;
    ierr = MatGetSubMatrices_MPIBAIJ_local(C,max_no,isrow_new+pos,iscol_new+pos,scall,*submat+pos);CHKERRQ(ierr);
    pos += max_no;
  }
  
  for (i=0; i<ismax; i++) {
    ierr = ISDestroy(isrow_new[i]);CHKERRQ(ierr);
    ierr = ISDestroy(iscol_new[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(isrow_new);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined (PETSC_USE_CTABLE)
#undef __FUNCT__    
#define __FUNCT__ "PetscGetProc" 
int PetscGetProc(const int gid, const int numprocs, const int proc_gnode[], int *proc)
{
  int nGlobalNd = proc_gnode[numprocs];
  int fproc = (int) ((float)gid * (float)numprocs / (float)nGlobalNd + 0.5);
  
  PetscFunctionBegin;
  /* if(fproc < 0) SETERRQ(1,"fproc < 0");*/
  if (fproc > numprocs) fproc = numprocs;
  while (gid < proc_gnode[fproc] || gid >= proc_gnode[fproc+1]) {
    if (gid < proc_gnode[fproc]) fproc--;
    else                         fproc++;
  }
  /* if(fproc<0 || fproc>=numprocs) { SETERRQ(1,"fproc < 0 || fproc >= numprocs"); }*/ 
  *proc = fproc;
  PetscFunctionReturn(0);
}
#endif

/* -------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrices_MPIBAIJ_local"
static int MatGetSubMatrices_MPIBAIJ_local(Mat C,int ismax,IS *isrow,IS *iscol,MatReuse scall,Mat *submats)
{ 
  Mat_MPIBAIJ *c = (Mat_MPIBAIJ*)C->data;
  Mat         A = c->A;
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)c->B->data,*mat;
  int         **irow,**icol,*nrow,*ncol,*w1,*w2,*w3,*w4,start,end,size;
  int         **sbuf1,**sbuf2,rank,i,j,k,l,ct1,ct2,ierr,**rbuf1,row,proc;
  int         nrqs,msz,**ptr,index,*req_size,*ctr,*pa,*tmp,tcol,nrqr;
  int         **rbuf3,*req_source,**sbuf_aj,**rbuf2,max1,max2;
  int         **lens,is_no,ncols,*cols,mat_i,*mat_j,tmp2,jmax,*irow_i;
  int         len,ctr_j,*sbuf1_j,*sbuf_aj_i,*rbuf1_i,kmax,*lens_i;
  int         bs=c->bs,bs2=c->bs2,*a_j=a->j,*b_j=b->j,*cworkA,*cworkB;
  int         cstart = c->cstart,nzA,nzB,*a_i=a->i,*b_i=b->i,imark;
  int         *bmap = c->garray,ctmp,rstart=c->rstart,tag0,tag1,tag2,tag3;
  MPI_Request *s_waits1,*r_waits1,*s_waits2,*r_waits2,*r_waits3;
  MPI_Request *r_waits4,*s_waits3,*s_waits4;
  MPI_Status  *r_status1,*r_status2,*s_status1,*s_status3,*s_status2;
  MPI_Status  *r_status3,*r_status4,*s_status4;
  MPI_Comm    comm;
  MatScalar   **rbuf4,**sbuf_aa,*vals,*mat_a,*sbuf_aa_i,*vworkA,*vworkB;
  MatScalar   *a_a=a->a,*b_a=b->a;
  PetscTruth  flag;
  int         *onodes1,*olengths1;

#if defined (PETSC_USE_CTABLE)
  int tt;
  PetscTable  *rowmaps,*colmaps,lrow1_grow1,lcol1_gcol1;
#else
  int         **cmap,*cmap_i,*rtable,*rmap_i,**rmap, Mbs = c->Mbs;
#endif

  PetscFunctionBegin;
  comm   = C->comm;
  tag0   = C->tag;
  size   = c->size;
  rank   = c->rank;
  
  /* Get some new tags to keep the communication clean */
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag2);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag3);CHKERRQ(ierr);

  /* Check if the col indices are sorted */
  for (i=0; i<ismax; i++) {
    ierr = ISSorted(iscol[i],(PetscTruth*)&j);CHKERRQ(ierr);
    if (!j) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"IS is not sorted");
  }

  len    = (2*ismax+1)*(sizeof(int*)+ sizeof(int));
#if !defined (PETSC_USE_CTABLE)
  len    += (Mbs+1)*sizeof(int);
#endif
  ierr = PetscMalloc(len,&irow);CHKERRQ(ierr);
  icol = irow + ismax;
  nrow = (int*)(icol + ismax);
  ncol = nrow + ismax;
#if !defined (PETSC_USE_CTABLE)
  rtable = ncol + ismax;
  /* Create hash table for the mapping :row -> proc*/
  for (i=0,j=0; i<size; i++) {
    jmax = c->rowners[i+1];
    for (; j<jmax; j++) {
      rtable[j] = i;
    }
  }
#endif
  
  for (i=0; i<ismax; i++) { 
    ierr = ISGetIndices(isrow[i],&irow[i]);CHKERRQ(ierr);
    ierr = ISGetIndices(iscol[i],&icol[i]);CHKERRQ(ierr);
    ierr = ISGetLocalSize(isrow[i],&nrow[i]);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iscol[i],&ncol[i]);CHKERRQ(ierr);
  }

  /* evaluate communication - mesg to who,length of mesg,and buffer space
     required. Based on this, buffers are allocated, and data copied into them*/
  ierr = PetscMalloc(size*4*sizeof(int),&w1);CHKERRQ(ierr); /* mesg size */
  w2   = w1 + size;      /* if w2[i] marked, then a message to proc i*/
  w3   = w2 + size;      /* no of IS that needs to be sent to proc i */
  w4   = w3 + size;      /* temp work space used in determining w1, w2, w3 */
  ierr = PetscMemzero(w1,size*3*sizeof(int));CHKERRQ(ierr); /* initialise work vector*/
  for (i=0; i<ismax; i++) { 
    ierr   = PetscMemzero(w4,size*sizeof(int));CHKERRQ(ierr); /* initialise work vector*/
    jmax   = nrow[i];
    irow_i = irow[i];
    for (j=0; j<jmax; j++) {
      row  = irow_i[j];
#if defined (PETSC_USE_CTABLE)
      ierr = PetscGetProc(row,size,c->rowners,&proc);CHKERRQ(ierr);
#else
      proc = rtable[row];
#endif
      w4[proc]++;
    }
    for (j=0; j<size; j++) { 
      if (w4[j]) { w1[j] += w4[j];  w3[j]++;} 
    }
  }

  nrqs     = 0;              /* no of outgoing messages */
  msz      = 0;              /* total mesg length for all proc */
  w1[rank] = 0;              /* no mesg sent to intself */
  w3[rank] = 0;
  for (i=0; i<size; i++) {
    if (w1[i])  { w2[i] = 1; nrqs++;} /* there exists a message to proc i */
  }
  ierr = PetscMalloc((nrqs+1)*sizeof(int),&pa);CHKERRQ(ierr); /*(proc -array)*/
  for (i=0,j=0; i<size; i++) {
    if (w1[i]) { pa[j] = i; j++; }
  } 

  /* Each message would have a header = 1 + 2*(no of IS) + data */
  for (i=0; i<nrqs; i++) {
    j     = pa[i];
    w1[j] += w2[j] + 2* w3[j];   
    msz   += w1[j];  
  }

  /* Determine the number of messages to expect, their lengths, from from-ids */
  ierr = PetscGatherNumberOfMessages(comm,w2,w1,&nrqr);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nrqs,nrqr,w1,&onodes1,&olengths1);CHKERRQ(ierr);

  /* Now post the Irecvs corresponding to these messages */
  ierr = PetscPostIrecvInt(comm,tag0,nrqr,onodes1,olengths1,&rbuf1,&r_waits1);CHKERRQ(ierr);
  
  ierr = PetscFree(onodes1);CHKERRQ(ierr);
  ierr = PetscFree(olengths1);CHKERRQ(ierr);

  /* Allocate Memory for outgoing messages */
  len  = 2*size*sizeof(int*) + 2*msz*sizeof(int) + size*sizeof(int);
  ierr = PetscMalloc(len,&sbuf1);CHKERRQ(ierr);
  ptr  = sbuf1 + size;   /* Pointers to the data in outgoing buffers */
  ierr = PetscMemzero(sbuf1,2*size*sizeof(int*));CHKERRQ(ierr);
  /* allocate memory for outgoing data + buf to receive the first reply */
  tmp  = (int*)(ptr + size);
  ctr  = tmp + 2*msz;

  {
    int *iptr = tmp,ict = 0;
    for (i=0; i<nrqs; i++) {
      j         = pa[i];
      iptr     += ict;
      sbuf1[j]  = iptr;
      ict       = w1[j];
    }
  }

  /* Form the outgoing messages */
  /* Initialise the header space */
  for (i=0; i<nrqs; i++) {
    j           = pa[i];
    sbuf1[j][0] = 0;
    ierr        = PetscMemzero(sbuf1[j]+1,2*w3[j]*sizeof(int));CHKERRQ(ierr);
    ptr[j]      = sbuf1[j] + 2*w3[j] + 1;
  }
  
  /* Parse the isrow and copy data into outbuf */
  for (i=0; i<ismax; i++) {
    ierr   = PetscMemzero(ctr,size*sizeof(int));CHKERRQ(ierr);
    irow_i = irow[i];
    jmax   = nrow[i];
    for (j=0; j<jmax; j++) {  /* parse the indices of each IS */
      row  = irow_i[j];
#if defined (PETSC_USE_CTABLE)
      ierr = PetscGetProc(row,size,c->rowners,&proc);CHKERRQ(ierr);
#else
      proc = rtable[row];
#endif
      if (proc != rank) { /* copy to the outgoing buf*/
        ctr[proc]++;
        *ptr[proc] = row;
        ptr[proc]++;
      }
    }
    /* Update the headers for the current IS */
    for (j=0; j<size; j++) { /* Can Optimise this loop too */
      if ((ctr_j = ctr[j])) {
        sbuf1_j        = sbuf1[j];
        k              = ++sbuf1_j[0];
        sbuf1_j[2*k]   = ctr_j;
        sbuf1_j[2*k-1] = i;
      }
    }
  }

  /*  Now  post the sends */
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Request),&s_waits1);CHKERRQ(ierr);
  for (i=0; i<nrqs; ++i) {
    j = pa[i];
    ierr = MPI_Isend(sbuf1[j],w1[j],MPI_INT,j,tag0,comm,s_waits1+i);CHKERRQ(ierr);
  }

  /* Post Recieves to capture the buffer size */
  ierr     = PetscMalloc((nrqs+1)*sizeof(MPI_Request),&r_waits2);CHKERRQ(ierr);
  ierr     = PetscMalloc((nrqs+1)*sizeof(int *),&rbuf2);CHKERRQ(ierr);
  rbuf2[0] = tmp + msz;
  for (i=1; i<nrqs; ++i) {
    j        = pa[i];
    rbuf2[i] = rbuf2[i-1]+w1[pa[i-1]];
  }
  for (i=0; i<nrqs; ++i) {
    j    = pa[i];
    ierr = MPI_Irecv(rbuf2[i],w1[j],MPI_INT,j,tag1,comm,r_waits2+i);CHKERRQ(ierr);
  }

  /* Send to other procs the buf size they should allocate */

  /* Receive messages*/
  ierr       = PetscMalloc((nrqr+1)*sizeof(MPI_Request),&s_waits2);CHKERRQ(ierr);
  ierr       = PetscMalloc((nrqr+1)*sizeof(MPI_Status),&r_status1);CHKERRQ(ierr);
  len        = 2*nrqr*sizeof(int) + (nrqr+1)*sizeof(int*);
  ierr       = PetscMalloc(len,&sbuf2);CHKERRQ(ierr);
  req_size   = (int*)(sbuf2 + nrqr);
  req_source = req_size + nrqr;
 
  {
    Mat_SeqBAIJ *sA = (Mat_SeqBAIJ*)c->A->data,*sB = (Mat_SeqBAIJ*)c->B->data;
    int        *sAi = sA->i,*sBi = sB->i,id,*sbuf2_i;

    for (i=0; i<nrqr; ++i) {
      ierr = MPI_Waitany(nrqr,r_waits1,&index,r_status1+i);CHKERRQ(ierr);
      req_size[index] = 0;
      rbuf1_i         = rbuf1[index];
      start           = 2*rbuf1_i[0] + 1;
      ierr            = MPI_Get_count(r_status1+i,MPI_INT,&end);CHKERRQ(ierr);
      ierr            = PetscMalloc(end*sizeof(int),&sbuf2[index]);CHKERRQ(ierr);
      sbuf2_i         = sbuf2[index];
      for (j=start; j<end; j++) {
        id               = rbuf1_i[j] - rstart;
        ncols            = sAi[id+1] - sAi[id] + sBi[id+1] - sBi[id];
        sbuf2_i[j]       = ncols;
        req_size[index] += ncols;
      }
      req_source[index] = r_status1[i].MPI_SOURCE;
      /* form the header */
      sbuf2_i[0]   = req_size[index];
      for (j=1; j<start; j++) { sbuf2_i[j] = rbuf1_i[j]; }
      ierr = MPI_Isend(sbuf2_i,end,MPI_INT,req_source[index],tag1,comm,s_waits2+i);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(r_status1);CHKERRQ(ierr);
  ierr = PetscFree(r_waits1);CHKERRQ(ierr);

  /*  recv buffer sizes */
  /* Receive messages*/

  ierr = PetscMalloc((nrqs+1)*sizeof(int*),&rbuf3);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqs+1)*sizeof(MatScalar*),&rbuf4);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Request),&r_waits3);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Request),&r_waits4);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Status),&r_status2);CHKERRQ(ierr);

  for (i=0; i<nrqs; ++i) {
    ierr = MPI_Waitany(nrqs,r_waits2,&index,r_status2+i);CHKERRQ(ierr);
    ierr = PetscMalloc(rbuf2[index][0]*sizeof(int),&rbuf3[index]);CHKERRQ(ierr);
    ierr = PetscMalloc(rbuf2[index][0]*bs2*sizeof(MatScalar),&rbuf4[index]);CHKERRQ(ierr);
    ierr = MPI_Irecv(rbuf3[index],rbuf2[index][0],MPI_INT,
                     r_status2[i].MPI_SOURCE,tag2,comm,r_waits3+index);CHKERRQ(ierr);
    ierr = MPI_Irecv(rbuf4[index],rbuf2[index][0]*bs2,MPIU_MATSCALAR,
                     r_status2[i].MPI_SOURCE,tag3,comm,r_waits4+index);CHKERRQ(ierr);
  } 
  ierr = PetscFree(r_status2);CHKERRQ(ierr);
  ierr = PetscFree(r_waits2);CHKERRQ(ierr);
  
  /* Wait on sends1 and sends2 */
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Status),&s_status1);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Status),&s_status2);CHKERRQ(ierr);

  ierr = MPI_Waitall(nrqs,s_waits1,s_status1);CHKERRQ(ierr);
  ierr = MPI_Waitall(nrqr,s_waits2,s_status2);CHKERRQ(ierr);
  ierr = PetscFree(s_status1);CHKERRQ(ierr);
  ierr = PetscFree(s_status2);CHKERRQ(ierr);
  ierr = PetscFree(s_waits1);CHKERRQ(ierr);
  ierr = PetscFree(s_waits2);CHKERRQ(ierr);

  /* Now allocate buffers for a->j, and send them off */
  ierr = PetscMalloc((nrqr+1)*sizeof(int *),&sbuf_aj);CHKERRQ(ierr);
  for (i=0,j=0; i<nrqr; i++) j += req_size[i];
  ierr = PetscMalloc((j+1)*sizeof(int),&sbuf_aj[0]);CHKERRQ(ierr);
  for (i=1; i<nrqr; i++)  sbuf_aj[i] = sbuf_aj[i-1] + req_size[i-1];
  
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Request),&s_waits3);CHKERRQ(ierr);
  {
     for (i=0; i<nrqr; i++) {
      rbuf1_i   = rbuf1[i]; 
      sbuf_aj_i = sbuf_aj[i];
      ct1       = 2*rbuf1_i[0] + 1;
      ct2       = 0;
      for (j=1,max1=rbuf1_i[0]; j<=max1; j++) { 
        kmax = rbuf1[i][2*j];
        for (k=0; k<kmax; k++,ct1++) {
          row    = rbuf1_i[ct1] - rstart;
          nzA    = a_i[row+1] - a_i[row];     nzB = b_i[row+1] - b_i[row];
          ncols  = nzA + nzB;
          cworkA = a_j + a_i[row]; cworkB = b_j + b_i[row];

          /* load the column indices for this row into cols*/
          cols  = sbuf_aj_i + ct2;
          for (l=0; l<nzB; l++) {
            if ((ctmp = bmap[cworkB[l]]) < cstart)  cols[l] = ctmp;
            else break;
          }
          imark = l;
          for (l=0; l<nzA; l++)   cols[imark+l] = cstart + cworkA[l];
          for (l=imark; l<nzB; l++) cols[nzA+l] = bmap[cworkB[l]];
          ct2 += ncols;
        }
      }
      ierr = MPI_Isend(sbuf_aj_i,req_size[i],MPI_INT,req_source[i],tag2,comm,s_waits3+i);CHKERRQ(ierr);
    }
  } 
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Status),&r_status3);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Status),&s_status3);CHKERRQ(ierr);

  /* Allocate buffers for a->a, and send them off */
  ierr = PetscMalloc((nrqr+1)*sizeof(MatScalar *),&sbuf_aa);CHKERRQ(ierr);
  for (i=0,j=0; i<nrqr; i++) j += req_size[i];
  ierr = PetscMalloc((j+1)*bs2*sizeof(MatScalar),&sbuf_aa[0]);CHKERRQ(ierr);
  for (i=1; i<nrqr; i++)  sbuf_aa[i] = sbuf_aa[i-1] + req_size[i-1]*bs2;
  
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Request),&s_waits4);CHKERRQ(ierr);
  {
    for (i=0; i<nrqr; i++) {
      rbuf1_i   = rbuf1[i];
      sbuf_aa_i = sbuf_aa[i];
      ct1       = 2*rbuf1_i[0]+1;
      ct2       = 0;
      for (j=1,max1=rbuf1_i[0]; j<=max1; j++) {
        kmax = rbuf1_i[2*j];
        for (k=0; k<kmax; k++,ct1++) {
          row    = rbuf1_i[ct1] - rstart;
          nzA    = a_i[row+1] - a_i[row];     nzB = b_i[row+1] - b_i[row];
          ncols  = nzA + nzB;
          cworkA = a_j + a_i[row];     cworkB = b_j + b_i[row];
          vworkA = a_a + a_i[row]*bs2; vworkB = b_a + b_i[row]*bs2;

          /* load the column values for this row into vals*/
          vals  = sbuf_aa_i+ct2*bs2;
          for (l=0; l<nzB; l++) {
            if ((bmap[cworkB[l]]) < cstart) { 
              ierr = PetscMemcpy(vals+l*bs2,vworkB+l*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
            }
            else break;
          }
          imark = l;
          for (l=0; l<nzA; l++) {
            ierr = PetscMemcpy(vals+(imark+l)*bs2,vworkA+l*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
          }
          for (l=imark; l<nzB; l++) {
            ierr = PetscMemcpy(vals+(nzA+l)*bs2,vworkB+l*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
          }
          ct2 += ncols;
        }
      }
      ierr = MPI_Isend(sbuf_aa_i,req_size[i]*bs2,MPIU_MATSCALAR,req_source[i],tag3,comm,s_waits4+i);CHKERRQ(ierr);
    }
  } 
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Status),&r_status4);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Status),&s_status4);CHKERRQ(ierr);
  ierr = PetscFree(rbuf1);CHKERRQ(ierr);

  /* Form the matrix */
  /* create col map */
  {
    int *icol_i;
#if defined (PETSC_USE_CTABLE)
    /* Create row map*/
    ierr = PetscMalloc((1+ismax)*sizeof(PetscTable),&colmaps);CHKERRQ(ierr);
    for (i=0; i<ismax+1; i++) {
      ierr = PetscTableCreate(((i<ismax) ? ncol[i] : ncol[i-1])+1,&colmaps[i]);CHKERRQ(ierr);
    }
#else
    len     = (1+ismax)*sizeof(int*)+ ismax*c->Nbs*sizeof(int);
    ierr    = PetscMalloc(len,&cmap);CHKERRQ(ierr);
    cmap[0] = (int *)(cmap + ismax);
    ierr    = PetscMemzero(cmap[0],(1+ismax*c->Nbs)*sizeof(int));CHKERRQ(ierr);
    for (i=1; i<ismax; i++) { cmap[i] = cmap[i-1] + c->Nbs; }
#endif
    for (i=0; i<ismax; i++) {
      jmax   = ncol[i];
      icol_i = icol[i];
#if defined (PETSC_USE_CTABLE)
      lcol1_gcol1 = colmaps[i];
      for (j=0; j<jmax; j++) { 
	ierr = PetscTableAdd(lcol1_gcol1,icol_i[j]+1,j+1);CHKERRQ(ierr);
      }
#else
      cmap_i = cmap[i];
      for (j=0; j<jmax; j++) { 
        cmap_i[icol_i[j]] = j+1; 
      }
#endif
    }
  }

  /* Create lens which is required for MatCreate... */
  for (i=0,j=0; i<ismax; i++) { j += nrow[i]; }
  len     = (1+ismax)*sizeof(int*)+ j*sizeof(int);
  ierr    = PetscMalloc(len,&lens);CHKERRQ(ierr);
  lens[0] = (int *)(lens + ismax);
  ierr    = PetscMemzero(lens[0],j*sizeof(int));CHKERRQ(ierr);
  for (i=1; i<ismax; i++) { lens[i] = lens[i-1] + nrow[i-1]; }
  
  /* Update lens from local data */
  for (i=0; i<ismax; i++) {
    jmax   = nrow[i];
#if defined (PETSC_USE_CTABLE)
    lcol1_gcol1 = colmaps[i];
#else
    cmap_i = cmap[i];
#endif
    irow_i = irow[i];
    lens_i = lens[i];
    for (j=0; j<jmax; j++) {
      row  = irow_i[j];
#if defined (PETSC_USE_CTABLE)
      ierr = PetscGetProc(row,size,c->rowners,&proc);CHKERRQ(ierr);
#else
      proc = rtable[row];
#endif
      if (proc == rank) {
        /* Get indices from matA and then from matB */
        row    = row - rstart;
        nzA    = a_i[row+1] - a_i[row];     nzB = b_i[row+1] - b_i[row];
        cworkA =  a_j + a_i[row]; cworkB = b_j + b_i[row];
#if defined (PETSC_USE_CTABLE)
       for (k=0; k<nzA; k++) {
	 ierr = PetscTableFind(lcol1_gcol1,cstart+cworkA[k]+1,&tt);CHKERRQ(ierr);
          if (tt) { lens_i[j]++; }
        }
        for (k=0; k<nzB; k++) {
	  ierr = PetscTableFind(lcol1_gcol1,bmap[cworkB[k]]+1,&tt);CHKERRQ(ierr);
          if (tt) { lens_i[j]++; }
        }
#else
        for (k=0; k<nzA; k++) {
          if (cmap_i[cstart + cworkA[k]]) { lens_i[j]++; }
        }
        for (k=0; k<nzB; k++) {
          if (cmap_i[bmap[cworkB[k]]]) { lens_i[j]++; }
        }
#endif
      }
    }
  } 
#if defined (PETSC_USE_CTABLE)
  /* Create row map*/
  ierr = PetscMalloc((1+ismax)*sizeof(PetscTable),&rowmaps);CHKERRQ(ierr);
  for (i=0; i<ismax+1; i++){ 
    ierr = PetscTableCreate((i<ismax) ? nrow[i] : nrow[i-1],&rowmaps[i]);CHKERRQ(ierr);
  }
#else
  /* Create row map*/
  len     = (1+ismax)*sizeof(int*)+ ismax*Mbs*sizeof(int);
  ierr    = PetscMalloc(len,&rmap);CHKERRQ(ierr);
  rmap[0] = (int *)(rmap + ismax);
  ierr    = PetscMemzero(rmap[0],ismax*Mbs*sizeof(int));CHKERRQ(ierr);
  for (i=1; i<ismax; i++) { rmap[i] = rmap[i-1] + Mbs;}
#endif
  for (i=0; i<ismax; i++) {
    irow_i = irow[i];
    jmax   = nrow[i];
#if defined (PETSC_USE_CTABLE)
    lrow1_grow1 = rowmaps[i];
    for (j=0; j<jmax; j++) { 
      ierr = PetscTableAdd(lrow1_grow1,irow_i[j]+1,j+1);CHKERRQ(ierr); 
    }
#else
    rmap_i = rmap[i];    
    for (j=0; j<jmax; j++) { 
      rmap_i[irow_i[j]] = j; 
    }
#endif
  }

  /* Update lens from offproc data */
  {
    int *rbuf2_i,*rbuf3_i,*sbuf1_i;

    for (tmp2=0; tmp2<nrqs; tmp2++) {
      ierr    = MPI_Waitany(nrqs,r_waits3,&i,r_status3+tmp2);CHKERRQ(ierr);
      index   = pa[i];
      sbuf1_i = sbuf1[index];
      jmax    = sbuf1_i[0];
      ct1     = 2*jmax+1; 
      ct2     = 0;               
      rbuf2_i = rbuf2[i];
      rbuf3_i = rbuf3[i];
      for (j=1; j<=jmax; j++) {
        is_no   = sbuf1_i[2*j-1];
        max1    = sbuf1_i[2*j];
        lens_i  = lens[is_no];
#if defined (PETSC_USE_CTABLE)
	lcol1_gcol1 = colmaps[is_no];
	lrow1_grow1 = rowmaps[is_no];
#else
        cmap_i  = cmap[is_no];
	rmap_i  = rmap[is_no];
#endif
        for (k=0; k<max1; k++,ct1++) {
#if defined (PETSC_USE_CTABLE)
	  ierr = PetscTableFind(lrow1_grow1,sbuf1_i[ct1]+1,&row);CHKERRQ(ierr); 
          row--; 
          if(row < 0) { SETERRQ(1,"row not found in table"); }
#else
          row  = rmap_i[sbuf1_i[ct1]]; /* the val in the new matrix to be */
#endif
          max2 = rbuf2_i[ct1];
          for (l=0; l<max2; l++,ct2++) {
#if defined (PETSC_USE_CTABLE)
	    ierr = PetscTableFind(lcol1_gcol1,rbuf3_i[ct2]+1,&tt);CHKERRQ(ierr);
	    if (tt) {
              lens_i[row]++;
            }
#else
            if (cmap_i[rbuf3_i[ct2]]) {
              lens_i[row]++;
            }
#endif
          }
        }
      }
    }
  }    
  ierr = PetscFree(r_status3);CHKERRQ(ierr);
  ierr = PetscFree(r_waits3);CHKERRQ(ierr);
  ierr = MPI_Waitall(nrqr,s_waits3,s_status3);CHKERRQ(ierr);
  ierr = PetscFree(s_status3);CHKERRQ(ierr);
  ierr = PetscFree(s_waits3);CHKERRQ(ierr);

  /* Create the submatrices */
  if (scall == MAT_REUSE_MATRIX) {
    /*
        Assumes new rows are same length as the old rows, hence bug!
    */
    for (i=0; i<ismax; i++) {
      mat = (Mat_SeqBAIJ *)(submats[i]->data);
      if ((mat->mbs != nrow[i]) || (mat->nbs != ncol[i] || mat->bs != bs)) {
        SETERRQ(PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong size");
      }
      ierr = PetscMemcmp(mat->ilen,lens[i],mat->mbs *sizeof(int),&flag);CHKERRQ(ierr);
      if (flag == PETSC_FALSE) {
        SETERRQ(PETSC_ERR_ARG_INCOMP,"Cannot reuse matrix. wrong no of nonzeros");
      }
      /* Initial matrix as if empty */
      ierr = PetscMemzero(mat->ilen,mat->mbs*sizeof(int));CHKERRQ(ierr);
      submats[i]->factor = C->factor;
    }
  } else {
    for (i=0; i<ismax; i++) {
      ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,a->bs,nrow[i]*bs,ncol[i]*bs,0,lens[i],submats+i);CHKERRQ(ierr);
    }
  }

  /* Assemble the matrices */
  /* First assemble the local rows */
  {
    int       ilen_row,*imat_ilen,*imat_j,*imat_i;
    MatScalar *imat_a;
  
    for (i=0; i<ismax; i++) {
      mat       = (Mat_SeqBAIJ*)submats[i]->data;
      imat_ilen = mat->ilen;
      imat_j    = mat->j;
      imat_i    = mat->i;
      imat_a    = mat->a;

#if defined (PETSC_USE_CTABLE)
      lcol1_gcol1 = colmaps[i];
      lrow1_grow1 = rowmaps[i];
#else
      cmap_i    = cmap[i];
      rmap_i    = rmap[i];
#endif
      irow_i    = irow[i];
      jmax      = nrow[i];
      for (j=0; j<jmax; j++) {
        row      = irow_i[j];
#if defined (PETSC_USE_CTABLE)
	ierr = PetscGetProc(row,size,c->rowners,&proc);CHKERRQ(ierr);
#else
	proc = rtable[row];
#endif
        if (proc == rank) {
          row      = row - rstart;
          nzA      = a_i[row+1] - a_i[row]; 
          nzB      = b_i[row+1] - b_i[row];
          cworkA   = a_j + a_i[row]; 
          cworkB   = b_j + b_i[row];
          vworkA   = a_a + a_i[row]*bs2;
          vworkB   = b_a + b_i[row]*bs2;
#if defined (PETSC_USE_CTABLE)
	  ierr = PetscTableFind(lrow1_grow1,row+rstart+1,&row);CHKERRQ(ierr); 
          row--; 
          if (row < 0) { SETERRQ(1,"row not found in table"); }
#else
          row      = rmap_i[row + rstart];
#endif
          mat_i    = imat_i[row];
          mat_a    = imat_a + mat_i*bs2;
          mat_j    = imat_j + mat_i;
          ilen_row = imat_ilen[row];

          /* load the column indices for this row into cols*/
          for (l=0; l<nzB; l++) {
	    if ((ctmp = bmap[cworkB[l]]) < cstart) {
#if defined (PETSC_USE_CTABLE)
	      ierr = PetscTableFind(lcol1_gcol1,ctmp+1,&tcol);CHKERRQ(ierr);
	      if (tcol) { 
#else
              if ((tcol = cmap_i[ctmp])) { 
#endif
                *mat_j++ = tcol - 1;
                ierr     = PetscMemcpy(mat_a,vworkB+l*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
                mat_a   += bs2;
                ilen_row++;
              }
            } else break;
          } 
          imark = l;
          for (l=0; l<nzA; l++) {
#if defined (PETSC_USE_CTABLE)
	    ierr = PetscTableFind(lcol1_gcol1,cstart+cworkA[l]+1,&tcol);CHKERRQ(ierr);
	    if (tcol) {
#else
	    if ((tcol = cmap_i[cstart + cworkA[l]])) { 
#endif
              *mat_j++ = tcol - 1;
              ierr     = PetscMemcpy(mat_a,vworkA+l*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
              mat_a   += bs2;
              ilen_row++;
            }
          }
          for (l=imark; l<nzB; l++) {
#if defined (PETSC_USE_CTABLE)
	    ierr = PetscTableFind(lcol1_gcol1,bmap[cworkB[l]]+1,&tcol);CHKERRQ(ierr);
	    if (tcol) {
#else
            if ((tcol = cmap_i[bmap[cworkB[l]]])) { 
#endif
              *mat_j++ = tcol - 1;
              ierr     = PetscMemcpy(mat_a,vworkB+l*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
              mat_a   += bs2;
              ilen_row++;
            }
          }
          imat_ilen[row] = ilen_row; 
        }
      }
      
    }
  }

  /*   Now assemble the off proc rows*/
  {
    int       *sbuf1_i,*rbuf2_i,*rbuf3_i,*imat_ilen,ilen;
    int       *imat_j,*imat_i;
    MatScalar *imat_a,*rbuf4_i;

    for (tmp2=0; tmp2<nrqs; tmp2++) {
      ierr    = MPI_Waitany(nrqs,r_waits4,&i,r_status4+tmp2);CHKERRQ(ierr);
      index   = pa[i];
      sbuf1_i = sbuf1[index];
      jmax    = sbuf1_i[0];           
      ct1     = 2*jmax + 1; 
      ct2     = 0;    
      rbuf2_i = rbuf2[i];
      rbuf3_i = rbuf3[i];
      rbuf4_i = rbuf4[i];
      for (j=1; j<=jmax; j++) {
        is_no     = sbuf1_i[2*j-1];
#if defined (PETSC_USE_CTABLE)
	lrow1_grow1 = rowmaps[is_no];
	lcol1_gcol1 = colmaps[is_no];
#else
        rmap_i    = rmap[is_no];
        cmap_i    = cmap[is_no];
#endif
        mat       = (Mat_SeqBAIJ*)submats[is_no]->data;
        imat_ilen = mat->ilen;
        imat_j    = mat->j;
        imat_i    = mat->i;
        imat_a    = mat->a;
        max1      = sbuf1_i[2*j];
        for (k=0; k<max1; k++,ct1++) {
          row   = sbuf1_i[ct1];
#if defined (PETSC_USE_CTABLE)
	  ierr = PetscTableFind(lrow1_grow1,row+1,&row);CHKERRQ(ierr); 
          row--; 
          if(row < 0) { SETERRQ(1,"row not found in table"); }
#else
          row   = rmap_i[row];
#endif
          ilen  = imat_ilen[row];
          mat_i = imat_i[row];
          mat_a = imat_a + mat_i*bs2;
          mat_j = imat_j + mat_i;
          max2 = rbuf2_i[ct1];
          for (l=0; l<max2; l++,ct2++) {
#if defined (PETSC_USE_CTABLE)
	    ierr = PetscTableFind(lcol1_gcol1,rbuf3_i[ct2]+1,&tcol);CHKERRQ(ierr);
	    if (tcol) { 
#else
	    if ((tcol = cmap_i[rbuf3_i[ct2]])) {
#endif
              *mat_j++    = tcol - 1;
              /* *mat_a++= rbuf4_i[ct2]; */
              ierr        = PetscMemcpy(mat_a,rbuf4_i+ct2*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
              mat_a      += bs2;
              ilen++;
            }
          }
          imat_ilen[row] = ilen;
        }
      }
    }
  }    
  ierr = PetscFree(r_status4);CHKERRQ(ierr);
  ierr = PetscFree(r_waits4);CHKERRQ(ierr);
  ierr = MPI_Waitall(nrqr,s_waits4,s_status4);CHKERRQ(ierr);
  ierr = PetscFree(s_waits4);CHKERRQ(ierr);
  ierr = PetscFree(s_status4);CHKERRQ(ierr);

  /* Restore the indices */
  for (i=0; i<ismax; i++) {
    ierr = ISRestoreIndices(isrow[i],irow+i);CHKERRQ(ierr);
    ierr = ISRestoreIndices(iscol[i],icol+i);CHKERRQ(ierr);
  }

  /* Destroy allocated memory */
  ierr = PetscFree(irow);CHKERRQ(ierr);
  ierr = PetscFree(w1);CHKERRQ(ierr);
  ierr = PetscFree(pa);CHKERRQ(ierr);

  ierr = PetscFree(sbuf1);CHKERRQ(ierr);
  ierr = PetscFree(rbuf2);CHKERRQ(ierr);
  for (i=0; i<nrqr; ++i) {
    ierr = PetscFree(sbuf2[i]);CHKERRQ(ierr);
  }
  for (i=0; i<nrqs; ++i) {
    ierr = PetscFree(rbuf3[i]);CHKERRQ(ierr);
    ierr = PetscFree(rbuf4[i]);CHKERRQ(ierr);
  }

  ierr = PetscFree(sbuf2);CHKERRQ(ierr);
  ierr = PetscFree(rbuf3);CHKERRQ(ierr);
  ierr = PetscFree(rbuf4);CHKERRQ(ierr);
  ierr = PetscFree(sbuf_aj[0]);CHKERRQ(ierr);
  ierr = PetscFree(sbuf_aj);CHKERRQ(ierr);
  ierr = PetscFree(sbuf_aa[0]);CHKERRQ(ierr);
  ierr = PetscFree(sbuf_aa);CHKERRQ(ierr);

#if defined (PETSC_USE_CTABLE)
  for (i=0; i<ismax+1; i++){
    ierr = PetscTableDelete(rowmaps[i]);CHKERRQ(ierr);
    ierr = PetscTableDelete(colmaps[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(colmaps);CHKERRQ(ierr);
  ierr = PetscFree(rowmaps);CHKERRQ(ierr);
  /* Mark Adams */
#else
  ierr = PetscFree(rmap);CHKERRQ(ierr);
  ierr = PetscFree(cmap);CHKERRQ(ierr);
#endif
  ierr = PetscFree(lens);CHKERRQ(ierr);

  for (i=0; i<ismax; i++) {
    ierr = MatAssemblyBegin(submats[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(submats[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
} 

