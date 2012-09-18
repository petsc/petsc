
/*
   Routines to compute overlapping regions of a parallel MPI matrix
  and to find submatrices that were shared across processors.
*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscbt.h>

static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Once(Mat,PetscInt,IS *);
static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Local(Mat,PetscInt,char **,PetscInt*,PetscInt**);
static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Receive(Mat,PetscInt,PetscInt **,PetscInt**,PetscInt*);
extern PetscErrorCode MatGetRow_MPIAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);
extern PetscErrorCode MatRestoreRow_MPIAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);

#undef __FUNCT__
#define __FUNCT__ "MatIncreaseOverlap_MPIAIJ"
PetscErrorCode MatIncreaseOverlap_MPIAIJ(Mat C,PetscInt imax,IS is[],PetscInt ov)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (ov < 0) SETERRQ(((PetscObject)C)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap specified");
  for (i=0; i<ov; ++i) {
    ierr = MatIncreaseOverlap_MPIAIJ_Once(C,imax,is);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  Sample message format:
  If a processor A wants processor B to process some elements corresponding
  to index sets is[1],is[5]
  mesg [0] = 2   (no of index sets in the mesg)
  -----------
  mesg [1] = 1 => is[1]
  mesg [2] = sizeof(is[1]);
  -----------
  mesg [3] = 5  => is[5]
  mesg [4] = sizeof(is[5]);
  -----------
  mesg [5]
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
#define __FUNCT__ "MatIncreaseOverlap_MPIAIJ_Once"
static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Once(Mat C,PetscInt imax,IS is[])
{
  Mat_MPIAIJ     *c = (Mat_MPIAIJ*)C->data;
  PetscMPIInt    *w1,*w2,nrqr,*w3,*w4,*onodes1,*olengths1,*onodes2,*olengths2;
  const PetscInt **idx,*idx_i;
  PetscInt       *n,**data,len;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,tag1,tag2;
  PetscInt       M,i,j,k,**rbuf,row,proc = 0,nrqs,msz,**outdat,**ptr;
  PetscInt       *ctr,*pa,*tmp,*isz,*isz1,**xdata,**rbuf2,*d_p;
  PetscBT        *table;
  MPI_Comm       comm;
  MPI_Request    *s_waits1,*r_waits1,*s_waits2,*r_waits2;
  MPI_Status     *s_status,*recv_status;
  char           *t_p;

  PetscFunctionBegin;
  comm   = ((PetscObject)C)->comm;
  size   = c->size;
  rank   = c->rank;
  M      = C->rmap->N;

  ierr = PetscObjectGetNewTag((PetscObject)C,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag2);CHKERRQ(ierr);

  ierr = PetscMalloc2(imax,PetscInt*,&idx,imax,PetscInt,&n);CHKERRQ(ierr);

  for (i=0; i<imax; i++) {
    ierr = ISGetIndices(is[i],&idx[i]);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is[i],&n[i]);CHKERRQ(ierr);
  }

  /* evaluate communication - mesg to who,length of mesg, and buffer space
     required. Based on this, buffers are allocated, and data copied into them*/
  ierr = PetscMalloc4(size,PetscMPIInt,&w1,size,PetscMPIInt,&w2,size,PetscMPIInt,&w3,size,PetscMPIInt,&w4);CHKERRQ(ierr);
  ierr = PetscMemzero(w1,size*sizeof(PetscMPIInt));CHKERRQ(ierr); /* initialise work vector*/
  ierr = PetscMemzero(w2,size*sizeof(PetscMPIInt));CHKERRQ(ierr); /* initialise work vector*/
  ierr = PetscMemzero(w3,size*sizeof(PetscMPIInt));CHKERRQ(ierr); /* initialise work vector*/
  for (i=0; i<imax; i++) {
    ierr  = PetscMemzero(w4,size*sizeof(PetscMPIInt));CHKERRQ(ierr); /* initialise work vector*/
    idx_i = idx[i];
    len   = n[i];
    for (j=0; j<len; j++) {
      row  = idx_i[j];
      if (row < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index set cannot have negative entries");
      ierr = PetscLayoutFindOwner(C->rmap,row,&proc);CHKERRQ(ierr);
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
  ierr = PetscMalloc((nrqs+1)*sizeof(PetscInt),&pa);CHKERRQ(ierr);
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
  ierr = PetscMalloc4(size,PetscInt*,&outdat,size,PetscInt*,&ptr,msz,PetscInt,&tmp,size,PetscInt,&ctr);CHKERRQ(ierr);
  ierr = PetscMemzero(outdat,size*sizeof(PetscInt*));CHKERRQ(ierr);
  ierr = PetscMemzero(ptr,size*sizeof(PetscInt*));CHKERRQ(ierr);

  {
    PetscInt *iptr = tmp,ict  = 0;
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
    ierr         = PetscMemzero(outdat[j]+1,2*w3[j]*sizeof(PetscInt));CHKERRQ(ierr);
    ptr[j]       = outdat[j] + 2*w3[j] + 1;
  }

  /* Memory for doing local proc's work*/
  {
    ierr = PetscMalloc5(imax,PetscBT,&table, imax,PetscInt*,&data, imax,PetscInt,&isz,
                        M*imax,PetscInt,&d_p, (M/PETSC_BITS_PER_BYTE+1)*imax,char,&t_p);CHKERRQ(ierr);
    ierr = PetscMemzero(table,imax*sizeof(PetscBT));CHKERRQ(ierr);
    ierr = PetscMemzero(data,imax*sizeof(PetscInt*));CHKERRQ(ierr);
    ierr = PetscMemzero(isz,imax*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemzero(d_p,M*imax*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemzero(t_p,(M/PETSC_BITS_PER_BYTE+1)*imax*sizeof(char));CHKERRQ(ierr);

    for (i=0; i<imax; i++) {
      table[i] = t_p + (M/PETSC_BITS_PER_BYTE+1)*i;
      data[i]  = d_p + M*i;
    }
  }

  /* Parse the IS and update local tables and the outgoing buf with the data*/
  {
    PetscInt n_i,*data_i,isz_i,*outdat_j,ctr_j;
    PetscBT  table_i;

    for (i=0; i<imax; i++) {
      ierr    = PetscMemzero(ctr,size*sizeof(PetscInt));CHKERRQ(ierr);
      n_i     = n[i];
      table_i = table[i];
      idx_i   = idx[i];
      data_i  = data[i];
      isz_i   = isz[i];
      for (j=0;  j<n_i; j++) {  /* parse the indices of each IS */
        row  = idx_i[j];
        ierr = PetscLayoutFindOwner(C->rmap,row,&proc);CHKERRQ(ierr);
        if (proc != rank) { /* copy to the outgoing buffer */
          ctr[proc]++;
          *ptr[proc] = row;
          ptr[proc]++;
        } else { /* Update the local table */
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
    ierr = MPI_Isend(outdat[j],w1[j],MPIU_INT,j,tag1,comm,s_waits1+i);CHKERRQ(ierr);
  }

  /* No longer need the original indices*/
  for (i=0; i<imax; ++i) {
    ierr = ISRestoreIndices(is[i],idx+i);CHKERRQ(ierr);
  }
  ierr = PetscFree2(idx,n);CHKERRQ(ierr);

  for (i=0; i<imax; ++i) {
    ierr = ISDestroy(&is[i]);CHKERRQ(ierr);
  }

  /* Do Local work*/
  ierr = MatIncreaseOverlap_MPIAIJ_Local(C,imax,table,isz,data);CHKERRQ(ierr);

  /* Receive messages*/
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Status),&recv_status);CHKERRQ(ierr);
  if (nrqr) {ierr = MPI_Waitall(nrqr,r_waits1,recv_status);CHKERRQ(ierr);}

  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Status),&s_status);CHKERRQ(ierr);
  if (nrqs) {ierr = MPI_Waitall(nrqs,s_waits1,s_status);CHKERRQ(ierr);}

  /* Phase 1 sends are complete - deallocate buffers */
  ierr = PetscFree4(outdat,ptr,tmp,ctr);CHKERRQ(ierr);
  ierr = PetscFree4(w1,w2,w3,w4);CHKERRQ(ierr);

  ierr = PetscMalloc((nrqr+1)*sizeof(PetscInt*),&xdata);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqr+1)*sizeof(PetscInt),&isz1);CHKERRQ(ierr);
  ierr = MatIncreaseOverlap_MPIAIJ_Receive(C,nrqr,rbuf,xdata,isz1);CHKERRQ(ierr);
  ierr = PetscFree(rbuf[0]);CHKERRQ(ierr);
  ierr = PetscFree(rbuf);CHKERRQ(ierr);


 /* Send the data back*/
  /* Do a global reduction to know the buffer space req for incoming messages*/
  {
    PetscMPIInt *rw1;

    ierr = PetscMalloc(size*sizeof(PetscMPIInt),&rw1);CHKERRQ(ierr);
    ierr = PetscMemzero(rw1,size*sizeof(PetscMPIInt));CHKERRQ(ierr);

    for (i=0; i<nrqr; ++i) {
      proc      = recv_status[i].MPI_SOURCE;
      if (proc != onodes1[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MPI_SOURCE mismatch");
      rw1[proc] = isz1[i];
    }
    ierr = PetscFree(onodes1);CHKERRQ(ierr);
    ierr = PetscFree(olengths1);CHKERRQ(ierr);

    /* Determine the number of messages to expect, their lengths, from from-ids */
    ierr = PetscGatherMessageLengths(comm,nrqr,nrqs,rw1,&onodes2,&olengths2);CHKERRQ(ierr);
    ierr = PetscFree(rw1);CHKERRQ(ierr);
  }
  /* Now post the Irecvs corresponding to these messages */
  ierr = PetscPostIrecvInt(comm,tag2,nrqs,onodes2,olengths2,&rbuf2,&r_waits2);CHKERRQ(ierr);

  /*  Now  post the sends */
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Request),&s_waits2);CHKERRQ(ierr);
  for (i=0; i<nrqr; ++i) {
    j    = recv_status[i].MPI_SOURCE;
    ierr = MPI_Isend(xdata[i],isz1[i],MPIU_INT,j,tag2,comm,s_waits2+i);CHKERRQ(ierr);
  }

  /* receive work done on other processors*/
  {
    PetscInt    is_no,ct1,max,*rbuf2_i,isz_i,*data_i,jmax;
    PetscMPIInt idex;
    PetscBT     table_i;
    MPI_Status  *status2;

    ierr = PetscMalloc((PetscMax(nrqr,nrqs)+1)*sizeof(MPI_Status),&status2);CHKERRQ(ierr);
    for (i=0; i<nrqs; ++i) {
      ierr = MPI_Waitany(nrqs,r_waits2,&idex,status2+i);CHKERRQ(ierr);
      /* Process the message*/
      rbuf2_i = rbuf2[idex];
      ct1     = 2*rbuf2_i[0]+1;
      jmax    = rbuf2[idex][0];
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

    if (nrqr) {ierr = MPI_Waitall(nrqr,s_waits2,status2);CHKERRQ(ierr);}
    ierr = PetscFree(status2);CHKERRQ(ierr);
  }

  for (i=0; i<imax; ++i) {
    ierr = ISCreateGeneral(PETSC_COMM_SELF,isz[i],data[i],PETSC_COPY_VALUES,is+i);CHKERRQ(ierr);
  }

  ierr = PetscFree(onodes2);CHKERRQ(ierr);
  ierr = PetscFree(olengths2);CHKERRQ(ierr);

  ierr = PetscFree(pa);CHKERRQ(ierr);
  ierr = PetscFree(rbuf2[0]);CHKERRQ(ierr);
  ierr = PetscFree(rbuf2);CHKERRQ(ierr);
  ierr = PetscFree(s_waits1);CHKERRQ(ierr);
  ierr = PetscFree(r_waits1);CHKERRQ(ierr);
  ierr = PetscFree(s_waits2);CHKERRQ(ierr);
  ierr = PetscFree(r_waits2);CHKERRQ(ierr);
  ierr = PetscFree5(table,data,isz,d_p,t_p);CHKERRQ(ierr);
  ierr = PetscFree(s_status);CHKERRQ(ierr);
  ierr = PetscFree(recv_status);CHKERRQ(ierr);
  ierr = PetscFree(xdata[0]);CHKERRQ(ierr);
  ierr = PetscFree(xdata);CHKERRQ(ierr);
  ierr = PetscFree(isz1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatIncreaseOverlap_MPIAIJ_Local"
/*
   MatIncreaseOverlap_MPIAIJ_Local - Called by MatincreaseOverlap, to do
       the work on the local processor.

     Inputs:
      C      - MAT_MPIAIJ;
      imax - total no of index sets processed at a time;
      table  - an array of char - size = m bits.

     Output:
      isz    - array containing the count of the solution elements corresponding
               to each index set;
      data   - pointer to the solutions
*/
static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Local(Mat C,PetscInt imax,PetscBT *table,PetscInt *isz,PetscInt **data)
{
  Mat_MPIAIJ *c = (Mat_MPIAIJ*)C->data;
  Mat        A = c->A,B = c->B;
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data,*b = (Mat_SeqAIJ*)B->data;
  PetscInt   start,end,val,max,rstart,cstart,*ai,*aj;
  PetscInt   *bi,*bj,*garray,i,j,k,row,*data_i,isz_i;
  PetscBT    table_i;

  PetscFunctionBegin;
  rstart = C->rmap->rstart;
  cstart = C->cmap->rstart;
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
#define __FUNCT__ "MatIncreaseOverlap_MPIAIJ_Receive"
/*
      MatIncreaseOverlap_MPIAIJ_Receive - Process the recieved messages,
         and return the output

         Input:
           C    - the matrix
           nrqr - no of messages being processed.
           rbuf - an array of pointers to the recieved requests

         Output:
           xdata - array of messages to be sent back
           isz1  - size of each message

  For better efficiency perhaps we should malloc separately each xdata[i],
then if a remalloc is required we need only copy the data for that one row
rather then all previous rows as it is now where a single large chunck of
memory is used.

*/
static PetscErrorCode MatIncreaseOverlap_MPIAIJ_Receive(Mat C,PetscInt nrqr,PetscInt **rbuf,PetscInt **xdata,PetscInt * isz1)
{
  Mat_MPIAIJ     *c = (Mat_MPIAIJ*)C->data;
  Mat            A = c->A,B = c->B;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data,*b = (Mat_SeqAIJ*)B->data;
  PetscErrorCode ierr;
  PetscInt       rstart,cstart,*ai,*aj,*bi,*bj,*garray,i,j,k;
  PetscInt       row,total_sz,ct,ct1,ct2,ct3,mem_estimate,oct2,l,start,end;
  PetscInt       val,max1,max2,m,no_malloc =0,*tmp,new_estimate,ctr;
  PetscInt       *rbuf_i,kmax,rbuf_0;
  PetscBT        xtable;

  PetscFunctionBegin;
  m      = C->rmap->N;
  rstart = C->rmap->rstart;
  cstart = C->cmap->rstart;
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

  if (C->rmap->n) max1 = ct*(a->nz + b->nz)/C->rmap->n;
  else      max1 = 1;
  mem_estimate = 3*((total_sz > max1 ? total_sz : max1)+1);
  ierr         = PetscMalloc(mem_estimate*sizeof(PetscInt),&xdata[0]);CHKERRQ(ierr);
  ++no_malloc;
  ierr         = PetscBTCreate(m,&xtable);CHKERRQ(ierr);
  ierr         = PetscMemzero(isz1,nrqr*sizeof(PetscInt));CHKERRQ(ierr);

  ct3 = 0;
  for (i=0; i<nrqr; i++) { /* for easch mesg from proc i */
    rbuf_i =  rbuf[i];
    rbuf_0 =  rbuf_i[0];
    ct1    =  2*rbuf_0+1;
    ct2    =  ct1;
    ct3    += ct1;
    for (j=1; j<=rbuf_0; j++) { /* for each IS from proc i*/
      ierr = PetscBTMemzero(m,xtable);CHKERRQ(ierr);
      oct2 = ct2;
      kmax = rbuf_i[2*j];
      for (k=0; k<kmax; k++,ct1++) {
        row = rbuf_i[ct1];
        if (!PetscBTLookupSet(xtable,row)) {
          if (!(ct3 < mem_estimate)) {
            new_estimate = (PetscInt)(1.5*mem_estimate)+1;
            ierr         = PetscMalloc(new_estimate*sizeof(PetscInt),&tmp);CHKERRQ(ierr);
            ierr         = PetscMemcpy(tmp,xdata[0],mem_estimate*sizeof(PetscInt));CHKERRQ(ierr);
            ierr         = PetscFree(xdata[0]);CHKERRQ(ierr);
            xdata[0]     = tmp;
            mem_estimate = new_estimate; ++no_malloc;
            for (ctr=1; ctr<=i; ctr++) { xdata[ctr] = xdata[ctr-1] + isz1[ctr-1];}
          }
          xdata[i][ct2++] = row;
          ct3++;
        }
      }
      for (k=oct2,max2=ct2; k<max2; k++) {
        row   = xdata[i][k] - rstart;
        start = ai[row];
        end   = ai[row+1];
        for (l=start; l<end; l++) {
          val = aj[l] + cstart;
          if (!PetscBTLookupSet(xtable,val)) {
            if (!(ct3 < mem_estimate)) {
              new_estimate = (PetscInt)(1.5*mem_estimate)+1;
              ierr         = PetscMalloc(new_estimate*sizeof(PetscInt),&tmp);CHKERRQ(ierr);
              ierr         = PetscMemcpy(tmp,xdata[0],mem_estimate*sizeof(PetscInt));CHKERRQ(ierr);
              ierr         = PetscFree(xdata[0]);CHKERRQ(ierr);
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
              new_estimate = (PetscInt)(1.5*mem_estimate)+1;
              ierr         = PetscMalloc(new_estimate*sizeof(PetscInt),&tmp);CHKERRQ(ierr);
              ierr         = PetscMemcpy(tmp,xdata[0],mem_estimate*sizeof(PetscInt));CHKERRQ(ierr);
              ierr         = PetscFree(xdata[0]);CHKERRQ(ierr);
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
  ierr = PetscBTDestroy(&xtable);CHKERRQ(ierr);
  ierr = PetscInfo3(C,"Allocated %D bytes, required %D bytes, no of mallocs = %D\n",mem_estimate,ct3,no_malloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------*/
extern PetscErrorCode MatGetSubMatrices_MPIAIJ_Local(Mat,PetscInt,const IS[],const IS[],MatReuse,PetscBool*,Mat*);
extern PetscErrorCode MatAssemblyEnd_SeqAIJ(Mat,MatAssemblyType);
/*
    Every processor gets the entire matrix
*/
#undef __FUNCT__
#define __FUNCT__ "MatGetSubMatrix_MPIAIJ_All"
PetscErrorCode MatGetSubMatrix_MPIAIJ_All(Mat A,MatGetSubMatrixOption flag,MatReuse scall,Mat *Bin[])
{
  Mat            B;
  Mat_MPIAIJ     *a = (Mat_MPIAIJ *)A->data;
  Mat_SeqAIJ     *b,*ad = (Mat_SeqAIJ*)a->A->data,*bd = (Mat_SeqAIJ*)a->B->data;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,*recvcounts = 0,*displs = 0;
  PetscInt       sendcount,i,*rstarts = A->rmap->range,n,cnt,j;
  PetscInt       m,*b_sendj,*garray = a->garray,*lens,*jsendbuf,*a_jsendbuf,*b_jsendbuf;
  MatScalar      *sendbuf,*recvbuf,*a_sendbuf,*b_sendbuf;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)A)->comm,&rank);CHKERRQ(ierr);

  if (scall == MAT_INITIAL_MATRIX) {
    /* ----------------------------------------------------------------
         Tell every processor the number of nonzeros per row
    */
    ierr = PetscMalloc(A->rmap->N*sizeof(PetscInt),&lens);CHKERRQ(ierr);
    for (i=A->rmap->rstart; i<A->rmap->rend; i++) {
      lens[i] = ad->i[i-A->rmap->rstart+1] - ad->i[i-A->rmap->rstart] + bd->i[i-A->rmap->rstart+1] - bd->i[i-A->rmap->rstart];
    }
    sendcount = A->rmap->rend - A->rmap->rstart;
    ierr = PetscMalloc2(size,PetscMPIInt,&recvcounts,size,PetscMPIInt,&displs);CHKERRQ(ierr);
    for (i=0; i<size; i++) {
      recvcounts[i] = A->rmap->range[i+1] - A->rmap->range[i];
      displs[i]     = A->rmap->range[i];
    }
#if defined(PETSC_HAVE_MPI_IN_PLACE)
    ierr  = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,lens,recvcounts,displs,MPIU_INT,((PetscObject)A)->comm);CHKERRQ(ierr);
#else
    ierr  = MPI_Allgatherv(lens+A->rmap->rstart,sendcount,MPIU_INT,lens,recvcounts,displs,MPIU_INT,((PetscObject)A)->comm);CHKERRQ(ierr);
#endif
    /* ---------------------------------------------------------------
         Create the sequential matrix of the same type as the local block diagonal
    */
    ierr  = MatCreate(PETSC_COMM_SELF,&B);CHKERRQ(ierr);
    ierr  = MatSetSizes(B,A->rmap->N,A->cmap->N,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr  = MatSetBlockSizes(B,A->rmap->bs,A->cmap->bs);CHKERRQ(ierr);
    ierr  = MatSetType(B,((PetscObject)a->A)->type_name);CHKERRQ(ierr);
    ierr  = MatSeqAIJSetPreallocation(B,0,lens);CHKERRQ(ierr);
    ierr  = PetscMalloc(sizeof(Mat),Bin);CHKERRQ(ierr);
    **Bin = B;
    b = (Mat_SeqAIJ *)B->data;

    /*--------------------------------------------------------------------
       Copy my part of matrix column indices over
    */
    sendcount  = ad->nz + bd->nz;
    jsendbuf   = b->j + b->i[rstarts[rank]];
    a_jsendbuf = ad->j;
    b_jsendbuf = bd->j;
    n          = A->rmap->rend - A->rmap->rstart;
    cnt        = 0;
    for (i=0; i<n; i++) {

      /* put in lower diagonal portion */
      m = bd->i[i+1] - bd->i[i];
      while (m > 0) {
        /* is it above diagonal (in bd (compressed) numbering) */
        if (garray[*b_jsendbuf] > A->rmap->rstart + i) break;
        jsendbuf[cnt++] = garray[*b_jsendbuf++];
        m--;
      }

      /* put in diagonal portion */
      for (j=ad->i[i]; j<ad->i[i+1]; j++) {
        jsendbuf[cnt++] = A->rmap->rstart + *a_jsendbuf++;
      }

      /* put in upper diagonal portion */
      while (m-- > 0) {
        jsendbuf[cnt++] = garray[*b_jsendbuf++];
      }
    }
    if (cnt != sendcount) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Corrupted PETSc matrix: nz given %D actual nz %D",sendcount,cnt);

    /*--------------------------------------------------------------------
       Gather all column indices to all processors
    */
    for (i=0; i<size; i++) {
      recvcounts[i] = 0;
      for (j=A->rmap->range[i]; j<A->rmap->range[i+1]; j++) {
        recvcounts[i] += lens[j];
      }
    }
    displs[0]  = 0;
    for (i=1; i<size; i++) {
      displs[i] = displs[i-1] + recvcounts[i-1];
    }
#if defined(PETSC_HAVE_MPI_IN_PLACE)
    ierr = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,b->j,recvcounts,displs,MPIU_INT,((PetscObject)A)->comm);CHKERRQ(ierr);
#else
    ierr = MPI_Allgatherv(jsendbuf,sendcount,MPIU_INT,b->j,recvcounts,displs,MPIU_INT,((PetscObject)A)->comm);CHKERRQ(ierr);
#endif
    /*--------------------------------------------------------------------
        Assemble the matrix into useable form (note numerical values not yet set)
    */
    /* set the b->ilen (length of each row) values */
    ierr = PetscMemcpy(b->ilen,lens,A->rmap->N*sizeof(PetscInt));CHKERRQ(ierr);
    /* set the b->i indices */
    b->i[0] = 0;
    for (i=1; i<=A->rmap->N; i++) {
      b->i[i] = b->i[i-1] + lens[i-1];
    }
    ierr = PetscFree(lens);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  } else {
    B  = **Bin;
    b = (Mat_SeqAIJ *)B->data;
  }

  /*--------------------------------------------------------------------
       Copy my part of matrix numerical values into the values location
  */
  if (flag == MAT_GET_VALUES){
    sendcount = ad->nz + bd->nz;
    sendbuf   = b->a + b->i[rstarts[rank]];
    a_sendbuf = ad->a;
    b_sendbuf = bd->a;
    b_sendj   = bd->j;
    n         = A->rmap->rend - A->rmap->rstart;
    cnt       = 0;
    for (i=0; i<n; i++) {

      /* put in lower diagonal portion */
      m = bd->i[i+1] - bd->i[i];
      while (m > 0) {
        /* is it above diagonal (in bd (compressed) numbering) */
        if (garray[*b_sendj] > A->rmap->rstart + i) break;
        sendbuf[cnt++] = *b_sendbuf++;
        m--;
        b_sendj++;
      }

      /* put in diagonal portion */
      for (j=ad->i[i]; j<ad->i[i+1]; j++) {
        sendbuf[cnt++] = *a_sendbuf++;
      }

      /* put in upper diagonal portion */
      while (m-- > 0) {
        sendbuf[cnt++] = *b_sendbuf++;
        b_sendj++;
      }
    }
    if (cnt != sendcount) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Corrupted PETSc matrix: nz given %D actual nz %D",sendcount,cnt);

    /* -----------------------------------------------------------------
       Gather all numerical values to all processors
    */
    if (!recvcounts) {
      ierr   = PetscMalloc2(size,PetscMPIInt,&recvcounts,size,PetscMPIInt,&displs);CHKERRQ(ierr);
    }
    for (i=0; i<size; i++) {
      recvcounts[i] = b->i[rstarts[i+1]] - b->i[rstarts[i]];
    }
    displs[0]  = 0;
    for (i=1; i<size; i++) {
      displs[i] = displs[i-1] + recvcounts[i-1];
    }
    recvbuf   = b->a;
#if defined(PETSC_HAVE_MPI_IN_PLACE)
    ierr = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,recvbuf,recvcounts,displs,MPIU_SCALAR,((PetscObject)A)->comm);CHKERRQ(ierr);
#else
    ierr = MPI_Allgatherv(sendbuf,sendcount,MPIU_SCALAR,recvbuf,recvcounts,displs,MPIU_SCALAR,((PetscObject)A)->comm);CHKERRQ(ierr);
#endif
  }  /* endof (flag == MAT_GET_VALUES) */
  ierr = PetscFree2(recvcounts,displs);CHKERRQ(ierr);

  if (A->symmetric){
    ierr = MatSetOption(B,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  } else if (A->hermitian) {
    ierr = MatSetOption(B,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);
  } else if (A->structurally_symmetric) {
    ierr = MatSetOption(B,MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "MatGetSubMatrices_MPIAIJ"
PetscErrorCode MatGetSubMatrices_MPIAIJ(Mat C,PetscInt ismax,const IS isrow[],const IS iscol[],MatReuse scall,Mat *submat[])
{
  PetscErrorCode ierr;
  PetscInt       nmax,nstages_local,nstages,i,pos,max_no,nrow,ncol;
  PetscBool      rowflag,colflag,wantallmatrix=PETSC_FALSE,twantallmatrix,*allcolumns;

  PetscFunctionBegin;

  /* Currently, unsorted column indices will result in inverted column indices in the resulting submatrices. */
  /* It would make sense to error out in MatGetSubMatrices_MPIAIJ_Local(), the most impl-specific level.
     However, there are more careful users of MatGetSubMatrices_MPIAIJ_Local() -- MatPermute_MPIAIJ() -- that
     take care to order the result correctly by assembling it with MatSetValues() (after preallocating).
   */
  for (i = 0; i < ismax; ++i) {
    PetscBool sorted;
    ierr = ISSorted(iscol[i], &sorted); CHKERRQ(ierr);
    if (!sorted) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Column index set %D not sorted", i);
  }

  /*
       Check for special case: each processor gets entire matrix
  */
  if (ismax == 1 && C->rmap->N == C->cmap->N) {
    ierr = ISIdentity(*isrow,&rowflag);CHKERRQ(ierr);
    ierr = ISIdentity(*iscol,&colflag);CHKERRQ(ierr);
    ierr = ISGetLocalSize(*isrow,&nrow);CHKERRQ(ierr);
    ierr = ISGetLocalSize(*iscol,&ncol);CHKERRQ(ierr);
    if (rowflag && colflag && nrow == C->rmap->N && ncol == C->cmap->N) {
      wantallmatrix = PETSC_TRUE;
      ierr = PetscOptionsGetBool(((PetscObject)C)->prefix,"-use_fast_submatrix",&wantallmatrix,PETSC_NULL);CHKERRQ(ierr);
    }
  }
  ierr = MPI_Allreduce(&wantallmatrix,&twantallmatrix,1,MPI_INT,MPI_MIN,((PetscObject)C)->comm);CHKERRQ(ierr);
  if (twantallmatrix) {
    ierr = MatGetSubMatrix_MPIAIJ_All(C,MAT_GET_VALUES,scall,submat);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* Allocate memory to hold all the submatrices */
  if (scall != MAT_REUSE_MATRIX) {
    ierr = PetscMalloc((ismax+1)*sizeof(Mat),submat);CHKERRQ(ierr);
  }

  /* Check for special case: each processor gets entire matrix columns */
  ierr = PetscMalloc((ismax+1)*sizeof(PetscBool),&allcolumns);CHKERRQ(ierr);
  for (i=0; i<ismax; i++) {
    ierr = ISIdentity(iscol[i],&colflag);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iscol[i],&ncol);CHKERRQ(ierr);
    if (colflag && ncol == C->cmap->N){
      allcolumns[i] = PETSC_TRUE;
    } else {
      allcolumns[i] = PETSC_FALSE;
    }
  }

  /* Determine the number of stages through which submatrices are done */
  nmax          = 20*1000000 / (C->cmap->N * sizeof(PetscInt));

  /*
     Each stage will extract nmax submatrices.
     nmax is determined by the matrix column dimension.
     If the original matrix has 20M columns, only one submatrix per stage is allowed, etc.
  */
  if (!nmax) nmax = 1;
  nstages_local = ismax/nmax + ((ismax % nmax)?1:0);

  /* Make sure every processor loops through the nstages */
  ierr = MPI_Allreduce(&nstages_local,&nstages,1,MPIU_INT,MPI_MAX,((PetscObject)C)->comm);CHKERRQ(ierr);

  for (i=0,pos=0; i<nstages; i++) {
    if (pos+nmax <= ismax) max_no = nmax;
    else if (pos == ismax) max_no = 0;
    else                   max_no = ismax-pos;
    ierr = MatGetSubMatrices_MPIAIJ_Local(C,max_no,isrow+pos,iscol+pos,scall,allcolumns+pos,*submat+pos);CHKERRQ(ierr);
    pos += max_no;
  }

  ierr = PetscFree(allcolumns);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "MatGetSubMatrices_MPIAIJ_Local"
PetscErrorCode MatGetSubMatrices_MPIAIJ_Local(Mat C,PetscInt ismax,const IS isrow[],const IS iscol[],MatReuse scall,PetscBool *allcolumns,Mat *submats)
{
  Mat_MPIAIJ     *c = (Mat_MPIAIJ*)C->data;
  Mat            A = c->A;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data,*b = (Mat_SeqAIJ*)c->B->data,*mat;
  const PetscInt **icol,**irow;
  PetscInt       *nrow,*ncol,start;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size,tag0,tag1,tag2,tag3,*w1,*w2,*w3,*w4,nrqr;
  PetscInt       **sbuf1,**sbuf2,i,j,k,l,ct1,ct2,**rbuf1,row,proc;
  PetscInt       nrqs,msz,**ptr,*req_size,*ctr,*pa,*tmp,tcol;
  PetscInt       **rbuf3,*req_source,**sbuf_aj,**rbuf2,max1,max2;
  PetscInt       **lens,is_no,ncols,*cols,mat_i,*mat_j,tmp2,jmax;
#if defined (PETSC_USE_CTABLE)
  PetscTable     *cmap,cmap_i=PETSC_NULL,*rmap,rmap_i;
#else
  PetscInt       **cmap,*cmap_i=PETSC_NULL,**rmap,*rmap_i;
#endif
  const PetscInt *irow_i;
  PetscInt       ctr_j,*sbuf1_j,*sbuf_aj_i,*rbuf1_i,kmax,*lens_i;
  MPI_Request    *s_waits1,*r_waits1,*s_waits2,*r_waits2,*r_waits3;
  MPI_Request    *r_waits4,*s_waits3,*s_waits4;
  MPI_Status     *r_status1,*r_status2,*s_status1,*s_status3,*s_status2;
  MPI_Status     *r_status3,*r_status4,*s_status4;
  MPI_Comm       comm;
  PetscScalar    **rbuf4,**sbuf_aa,*vals,*mat_a,*sbuf_aa_i;
  PetscMPIInt    *onodes1,*olengths1;
  PetscMPIInt    idex,idex2,end;

  PetscFunctionBegin;

  comm   = ((PetscObject)C)->comm;
  tag0   = ((PetscObject)C)->tag;
  size   = c->size;
  rank   = c->rank;

  /* Get some new tags to keep the communication clean */
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag2);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag3);CHKERRQ(ierr);

  ierr = PetscMalloc4(ismax,const PetscInt*,&irow,ismax,const PetscInt*,&icol,ismax,PetscInt,&nrow,ismax,PetscInt,&ncol);CHKERRQ(ierr);

  for (i=0; i<ismax; i++) {
    ierr = ISGetIndices(isrow[i],&irow[i]);CHKERRQ(ierr);
    ierr = ISGetLocalSize(isrow[i],&nrow[i]);CHKERRQ(ierr);
    if (allcolumns[i]){
      icol[i] = PETSC_NULL;
      ncol[i] = C->cmap->N;
    } else {
      ierr = ISGetIndices(iscol[i],&icol[i]);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iscol[i],&ncol[i]);CHKERRQ(ierr);
    }
  }

  /* evaluate communication - mesg to who, length of mesg, and buffer space
     required. Based on this, buffers are allocated, and data copied into them*/
  ierr   = PetscMalloc4(size,PetscMPIInt,&w1,size,PetscMPIInt,&w2,size,PetscMPIInt,&w3,size,PetscMPIInt,&w4);CHKERRQ(ierr); /* mesg size */
  ierr   = PetscMemzero(w1,size*sizeof(PetscMPIInt));CHKERRQ(ierr); /* initialize work vector*/
  ierr   = PetscMemzero(w2,size*sizeof(PetscMPIInt));CHKERRQ(ierr); /* initialize work vector*/
  ierr   = PetscMemzero(w3,size*sizeof(PetscMPIInt));CHKERRQ(ierr); /* initialize work vector*/
  for (i=0; i<ismax; i++) {
    ierr   = PetscMemzero(w4,size*sizeof(PetscMPIInt));CHKERRQ(ierr); /* initialize work vector*/
    jmax   = nrow[i];
    irow_i = irow[i];
    for (j=0; j<jmax; j++) {
      l = 0;
      row  = irow_i[j];
      while (row >= C->rmap->range[l+1]) l++;
      proc = l;
      w4[proc]++;
    }
    for (j=0; j<size; j++) {
      if (w4[j]) { w1[j] += w4[j];  w3[j]++;}
    }
  }

  nrqs     = 0;              /* no of outgoing messages */
  msz      = 0;              /* total mesg length (for all procs) */
  w1[rank] = 0;              /* no mesg sent to self */
  w3[rank] = 0;
  for (i=0; i<size; i++) {
    if (w1[i])  { w2[i] = 1; nrqs++;} /* there exists a message to proc i */
  }
  ierr = PetscMalloc((nrqs+1)*sizeof(PetscInt),&pa);CHKERRQ(ierr); /*(proc -array)*/
  for (i=0,j=0; i<size; i++) {
    if (w1[i]) { pa[j] = i; j++; }
  }

  /* Each message would have a header = 1 + 2*(no of IS) + data */
  for (i=0; i<nrqs; i++) {
    j     = pa[i];
    w1[j] += w2[j] + 2* w3[j];
    msz   += w1[j];
  }
  ierr = PetscInfo2(C,"Number of outgoing messages %D Total message length %D\n",nrqs,msz);CHKERRQ(ierr);

  /* Determine the number of messages to expect, their lengths, from from-ids */
  ierr = PetscGatherNumberOfMessages(comm,w2,w1,&nrqr);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nrqs,nrqr,w1,&onodes1,&olengths1);CHKERRQ(ierr);

  /* Now post the Irecvs corresponding to these messages */
  ierr = PetscPostIrecvInt(comm,tag0,nrqr,onodes1,olengths1,&rbuf1,&r_waits1);CHKERRQ(ierr);

  ierr = PetscFree(onodes1);CHKERRQ(ierr);
  ierr = PetscFree(olengths1);CHKERRQ(ierr);

  /* Allocate Memory for outgoing messages */
  ierr = PetscMalloc4(size,PetscInt*,&sbuf1,size,PetscInt*,&ptr,2*msz,PetscInt,&tmp,size,PetscInt,&ctr);CHKERRQ(ierr);
  ierr = PetscMemzero(sbuf1,size*sizeof(PetscInt*));CHKERRQ(ierr);
  ierr = PetscMemzero(ptr,size*sizeof(PetscInt*));CHKERRQ(ierr);

  {
    PetscInt *iptr = tmp,ict = 0;
    for (i=0; i<nrqs; i++) {
      j         = pa[i];
      iptr     += ict;
      sbuf1[j]  = iptr;
      ict       = w1[j];
    }
  }

  /* Form the outgoing messages */
  /* Initialize the header space */
  for (i=0; i<nrqs; i++) {
    j           = pa[i];
    sbuf1[j][0] = 0;
    ierr        = PetscMemzero(sbuf1[j]+1,2*w3[j]*sizeof(PetscInt));CHKERRQ(ierr);
    ptr[j]      = sbuf1[j] + 2*w3[j] + 1;
  }

  /* Parse the isrow and copy data into outbuf */
  for (i=0; i<ismax; i++) {
    ierr   = PetscMemzero(ctr,size*sizeof(PetscInt));CHKERRQ(ierr);
    irow_i = irow[i];
    jmax   = nrow[i];
    for (j=0; j<jmax; j++) {  /* parse the indices of each IS */
      l = 0;
      row  = irow_i[j];
      while (row >= C->rmap->range[l+1]) l++;
      proc = l;
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
    j    = pa[i];
    ierr = MPI_Isend(sbuf1[j],w1[j],MPIU_INT,j,tag0,comm,s_waits1+i);CHKERRQ(ierr);
  }

  /* Post Receives to capture the buffer size */
  ierr     = PetscMalloc((nrqs+1)*sizeof(MPI_Request),&r_waits2);CHKERRQ(ierr);
  ierr     = PetscMalloc((nrqs+1)*sizeof(PetscInt*),&rbuf2);CHKERRQ(ierr);
  rbuf2[0] = tmp + msz;
  for (i=1; i<nrqs; ++i) {
    rbuf2[i] = rbuf2[i-1]+w1[pa[i-1]];
  }
  for (i=0; i<nrqs; ++i) {
    j    = pa[i];
    ierr = MPI_Irecv(rbuf2[i],w1[j],MPIU_INT,j,tag1,comm,r_waits2+i);CHKERRQ(ierr);
  }

  /* Send to other procs the buf size they should allocate */


  /* Receive messages*/
  ierr        = PetscMalloc((nrqr+1)*sizeof(MPI_Request),&s_waits2);CHKERRQ(ierr);
  ierr        = PetscMalloc((nrqr+1)*sizeof(MPI_Status),&r_status1);CHKERRQ(ierr);
  ierr        = PetscMalloc3(nrqr,PetscInt*,&sbuf2,nrqr,PetscInt,&req_size,nrqr,PetscInt,&req_source);CHKERRQ(ierr);
  {
    Mat_SeqAIJ  *sA = (Mat_SeqAIJ*)c->A->data,*sB = (Mat_SeqAIJ*)c->B->data;
    PetscInt    *sAi = sA->i,*sBi = sB->i,id,rstart = C->rmap->rstart;
    PetscInt    *sbuf2_i;

    for (i=0; i<nrqr; ++i) {
      ierr = MPI_Waitany(nrqr,r_waits1,&idex,r_status1+i);CHKERRQ(ierr);
      req_size[idex] = 0;
      rbuf1_i         = rbuf1[idex];
      start           = 2*rbuf1_i[0] + 1;
      ierr            = MPI_Get_count(r_status1+i,MPIU_INT,&end);CHKERRQ(ierr);
      ierr            = PetscMalloc((end+1)*sizeof(PetscInt),&sbuf2[idex]);CHKERRQ(ierr);
      sbuf2_i         = sbuf2[idex];
      for (j=start; j<end; j++) {
        id               = rbuf1_i[j] - rstart;
        ncols            = sAi[id+1] - sAi[id] + sBi[id+1] - sBi[id];
        sbuf2_i[j]       = ncols;
        req_size[idex] += ncols;
      }
      req_source[idex] = r_status1[i].MPI_SOURCE;
      /* form the header */
      sbuf2_i[0]   = req_size[idex];
      for (j=1; j<start; j++) { sbuf2_i[j] = rbuf1_i[j]; }
      ierr = MPI_Isend(sbuf2_i,end,MPIU_INT,req_source[idex],tag1,comm,s_waits2+i);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(r_status1);CHKERRQ(ierr);
  ierr = PetscFree(r_waits1);CHKERRQ(ierr);

  /*  recv buffer sizes */
  /* Receive messages*/

  ierr = PetscMalloc((nrqs+1)*sizeof(PetscInt*),&rbuf3);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqs+1)*sizeof(PetscScalar*),&rbuf4);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Request),&r_waits3);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Request),&r_waits4);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Status),&r_status2);CHKERRQ(ierr);

  for (i=0; i<nrqs; ++i) {
    ierr = MPI_Waitany(nrqs,r_waits2,&idex,r_status2+i);CHKERRQ(ierr);
    ierr = PetscMalloc((rbuf2[idex][0]+1)*sizeof(PetscInt),&rbuf3[idex]);CHKERRQ(ierr);
    ierr = PetscMalloc((rbuf2[idex][0]+1)*sizeof(PetscScalar),&rbuf4[idex]);CHKERRQ(ierr);
    ierr = MPI_Irecv(rbuf3[idex],rbuf2[idex][0],MPIU_INT,r_status2[i].MPI_SOURCE,tag2,comm,r_waits3+idex);CHKERRQ(ierr);
    ierr = MPI_Irecv(rbuf4[idex],rbuf2[idex][0],MPIU_SCALAR,r_status2[i].MPI_SOURCE,tag3,comm,r_waits4+idex);CHKERRQ(ierr);
  }
  ierr = PetscFree(r_status2);CHKERRQ(ierr);
  ierr = PetscFree(r_waits2);CHKERRQ(ierr);

  /* Wait on sends1 and sends2 */
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Status),&s_status1);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Status),&s_status2);CHKERRQ(ierr);

  if (nrqs) {ierr = MPI_Waitall(nrqs,s_waits1,s_status1);CHKERRQ(ierr);}
  if (nrqr) {ierr = MPI_Waitall(nrqr,s_waits2,s_status2);CHKERRQ(ierr);}
  ierr = PetscFree(s_status1);CHKERRQ(ierr);
  ierr = PetscFree(s_status2);CHKERRQ(ierr);
  ierr = PetscFree(s_waits1);CHKERRQ(ierr);
  ierr = PetscFree(s_waits2);CHKERRQ(ierr);

  /* Now allocate buffers for a->j, and send them off */
  ierr = PetscMalloc((nrqr+1)*sizeof(PetscInt*),&sbuf_aj);CHKERRQ(ierr);
  for (i=0,j=0; i<nrqr; i++) j += req_size[i];
  ierr = PetscMalloc((j+1)*sizeof(PetscInt),&sbuf_aj[0]);CHKERRQ(ierr);
  for (i=1; i<nrqr; i++)  sbuf_aj[i] = sbuf_aj[i-1] + req_size[i-1];

  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Request),&s_waits3);CHKERRQ(ierr);
  {
    PetscInt nzA,nzB,*a_i = a->i,*b_i = b->i,lwrite;
    PetscInt *cworkA,*cworkB,cstart = C->cmap->rstart,rstart = C->rmap->rstart,*bmap = c->garray;
    PetscInt cend = C->cmap->rend;
    PetscInt *a_j = a->j,*b_j = b->j,ctmp;

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

	  lwrite = 0;
          for (l=0; l<nzB; l++) {
            if ((ctmp = bmap[cworkB[l]]) < cstart)  cols[lwrite++] = ctmp;
          }
          for (l=0; l<nzA; l++)   cols[lwrite++] = cstart + cworkA[l];
          for (l=0; l<nzB; l++) {
            if ((ctmp = bmap[cworkB[l]]) >= cend)  cols[lwrite++] = ctmp;
          }

          ct2 += ncols;
        }
      }
      ierr = MPI_Isend(sbuf_aj_i,req_size[i],MPIU_INT,req_source[i],tag2,comm,s_waits3+i);CHKERRQ(ierr);
    }
  }
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Status),&r_status3);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Status),&s_status3);CHKERRQ(ierr);

  /* Allocate buffers for a->a, and send them off */
  ierr = PetscMalloc((nrqr+1)*sizeof(PetscScalar*),&sbuf_aa);CHKERRQ(ierr);
  for (i=0,j=0; i<nrqr; i++) j += req_size[i];
  ierr = PetscMalloc((j+1)*sizeof(PetscScalar),&sbuf_aa[0]);CHKERRQ(ierr);
  for (i=1; i<nrqr; i++)  sbuf_aa[i] = sbuf_aa[i-1] + req_size[i-1];

  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Request),&s_waits4);CHKERRQ(ierr);
  {
    PetscInt    nzA,nzB,*a_i = a->i,*b_i = b->i, *cworkB,lwrite;
    PetscInt    cstart = C->cmap->rstart,rstart = C->rmap->rstart,*bmap = c->garray;
    PetscInt    cend = C->cmap->rend;
    PetscInt    *b_j = b->j;
    PetscScalar *vworkA,*vworkB,*a_a = a->a,*b_a = b->a;

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
          cworkB = b_j + b_i[row];
          vworkA = a_a + a_i[row];
          vworkB = b_a + b_i[row];

          /* load the column values for this row into vals*/
          vals  = sbuf_aa_i+ct2;

	  lwrite = 0;
          for (l=0; l<nzB; l++) {
            if ((bmap[cworkB[l]]) < cstart)  vals[lwrite++] = vworkB[l];
          }
          for (l=0; l<nzA; l++)   vals[lwrite++] = vworkA[l];
          for (l=0; l<nzB; l++) {
            if ((bmap[cworkB[l]]) >= cend)  vals[lwrite++] = vworkB[l];
          }

          ct2 += ncols;
        }
      }
      ierr = MPI_Isend(sbuf_aa_i,req_size[i],MPIU_SCALAR,req_source[i],tag3,comm,s_waits4+i);CHKERRQ(ierr);
    }
  }
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Status),&r_status4);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Status),&s_status4);CHKERRQ(ierr);
  ierr = PetscFree(rbuf1[0]);CHKERRQ(ierr);
  ierr = PetscFree(rbuf1);CHKERRQ(ierr);

  /* Form the matrix */
  /* create col map: global col of C -> local col of submatrices */
  {
    const PetscInt *icol_i;
#if defined (PETSC_USE_CTABLE)
    ierr = PetscMalloc((1+ismax)*sizeof(PetscTable),&cmap);CHKERRQ(ierr);
    for (i=0; i<ismax; i++) {
      if (!allcolumns[i]){
        ierr = PetscTableCreate(ncol[i]+1,C->cmap->N+1,&cmap[i]);CHKERRQ(ierr);
        jmax   = ncol[i];
        icol_i = icol[i];
        cmap_i = cmap[i];
        for (j=0; j<jmax; j++) {
          ierr = PetscTableAdd(cmap[i],icol_i[j]+1,j+1,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        cmap[i] = PETSC_NULL;
      }
    }
#else
    ierr = PetscMalloc(ismax*sizeof(PetscInt*),&cmap);CHKERRQ(ierr);
    for (i=0; i<ismax; i++) {
      if (!allcolumns[i]){
        ierr = PetscMalloc(C->cmap->N*sizeof(PetscInt),&cmap[i]);CHKERRQ(ierr);
        ierr = PetscMemzero(cmap[i],C->cmap->N*sizeof(PetscInt));CHKERRQ(ierr);
        jmax   = ncol[i];
        icol_i = icol[i];
        cmap_i = cmap[i];
        for (j=0; j<jmax; j++) {
          cmap_i[icol_i[j]] = j+1;
        }
      } else {
        cmap[i] = PETSC_NULL;
      }
    }
#endif
  }

  /* Create lens which is required for MatCreate... */
  for (i=0,j=0; i<ismax; i++) { j += nrow[i]; }
  ierr    = PetscMalloc(ismax*sizeof(PetscInt*),&lens);CHKERRQ(ierr);
  if (ismax) {
    ierr    = PetscMalloc(j*sizeof(PetscInt),&lens[0]);CHKERRQ(ierr);
    ierr    = PetscMemzero(lens[0],j*sizeof(PetscInt));CHKERRQ(ierr);
  }
  for (i=1; i<ismax; i++) { lens[i] = lens[i-1] + nrow[i-1]; }

  /* Update lens from local data */
  for (i=0; i<ismax; i++) {
    jmax   = nrow[i];
    if (!allcolumns[i]) cmap_i = cmap[i];
    irow_i = irow[i];
    lens_i = lens[i];
    for (j=0; j<jmax; j++) {
      l = 0;
      row  = irow_i[j];
      while (row >= C->rmap->range[l+1]) l++;
      proc = l;
      if (proc == rank) {
        ierr = MatGetRow_MPIAIJ(C,row,&ncols,&cols,0);CHKERRQ(ierr);
        if (!allcolumns[i]){
          for (k=0; k<ncols; k++) {
#if defined (PETSC_USE_CTABLE)
            ierr = PetscTableFind(cmap_i,cols[k]+1,&tcol);CHKERRQ(ierr);
#else
            tcol = cmap_i[cols[k]];
#endif
            if (tcol) { lens_i[j]++;}
          }
        } else { /* allcolumns */
          lens_i[j] = ncols;
        }
        ierr = MatRestoreRow_MPIAIJ(C,row,&ncols,&cols,0);CHKERRQ(ierr);
      }
    }
  }

  /* Create row map: global row of C -> local row of submatrices */
#if defined (PETSC_USE_CTABLE)
  ierr = PetscMalloc((1+ismax)*sizeof(PetscTable),&rmap);CHKERRQ(ierr);
    for (i=0; i<ismax; i++) {
      ierr = PetscTableCreate(nrow[i]+1,C->rmap->N+1,&rmap[i]);CHKERRQ(ierr);
      rmap_i = rmap[i];
      irow_i = irow[i];
      jmax   = nrow[i];
      for (j=0; j<jmax; j++) {
        ierr = PetscTableAdd(rmap[i],irow_i[j]+1,j+1,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
#else
  ierr    = PetscMalloc(ismax*sizeof(PetscInt*),&rmap);CHKERRQ(ierr);
  if (ismax) {
    ierr    = PetscMalloc(ismax*C->rmap->N*sizeof(PetscInt),&rmap[0]);CHKERRQ(ierr);
    ierr    = PetscMemzero(rmap[0],ismax*C->rmap->N*sizeof(PetscInt));CHKERRQ(ierr);
  }
  for (i=1; i<ismax; i++) { rmap[i] = rmap[i-1] + C->rmap->N;}
  for (i=0; i<ismax; i++) {
    rmap_i = rmap[i];
    irow_i = irow[i];
    jmax   = nrow[i];
    for (j=0; j<jmax; j++) {
      rmap_i[irow_i[j]] = j;
    }
  }
#endif

  /* Update lens from offproc data */
  {
    PetscInt *rbuf2_i,*rbuf3_i,*sbuf1_i;

    for (tmp2=0; tmp2<nrqs; tmp2++) {
      ierr = MPI_Waitany(nrqs,r_waits3,&idex2,r_status3+tmp2);CHKERRQ(ierr);
      idex   = pa[idex2];
      sbuf1_i = sbuf1[idex];
      jmax    = sbuf1_i[0];
      ct1     = 2*jmax+1;
      ct2     = 0;
      rbuf2_i = rbuf2[idex2];
      rbuf3_i = rbuf3[idex2];
      for (j=1; j<=jmax; j++) {
        is_no   = sbuf1_i[2*j-1];
        max1    = sbuf1_i[2*j];
        lens_i  = lens[is_no];
        if (!allcolumns[is_no]){cmap_i  = cmap[is_no];}
        rmap_i  = rmap[is_no];
        for (k=0; k<max1; k++,ct1++) {
#if defined (PETSC_USE_CTABLE)
          ierr = PetscTableFind(rmap_i,sbuf1_i[ct1]+1,&row);CHKERRQ(ierr);
          row--;
          if (row < 0) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"row not found in table"); }
#else
          row  = rmap_i[sbuf1_i[ct1]]; /* the val in the new matrix to be */
#endif
          max2 = rbuf2_i[ct1];
          for (l=0; l<max2; l++,ct2++) {
            if (!allcolumns[is_no]){
#if defined (PETSC_USE_CTABLE)
              ierr = PetscTableFind(cmap_i,rbuf3_i[ct2]+1,&tcol);CHKERRQ(ierr);
#else
              tcol = cmap_i[rbuf3_i[ct2]];
#endif
              if (tcol) {
                lens_i[row]++;
              }
            } else { /* allcolumns */
              lens_i[row]++; /* lens_i[row] += max2 ? */
            }
          }
        }
      }
    }
  }
  ierr = PetscFree(r_status3);CHKERRQ(ierr);
  ierr = PetscFree(r_waits3);CHKERRQ(ierr);
  if (nrqr) {ierr = MPI_Waitall(nrqr,s_waits3,s_status3);CHKERRQ(ierr);}
  ierr = PetscFree(s_status3);CHKERRQ(ierr);
  ierr = PetscFree(s_waits3);CHKERRQ(ierr);

  /* Create the submatrices */
  if (scall == MAT_REUSE_MATRIX) {
    PetscBool  flag;

    /*
        Assumes new rows are same length as the old rows,hence bug!
    */
    for (i=0; i<ismax; i++) {
      mat = (Mat_SeqAIJ *)(submats[i]->data);
      if ((submats[i]->rmap->n != nrow[i]) || (submats[i]->cmap->n != ncol[i])) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong size");
      ierr = PetscMemcmp(mat->ilen,lens[i],submats[i]->rmap->n*sizeof(PetscInt),&flag);CHKERRQ(ierr);
      if (!flag) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong no of nonzeros");
      /* Initial matrix as if empty */
      ierr = PetscMemzero(mat->ilen,submats[i]->rmap->n*sizeof(PetscInt));CHKERRQ(ierr);
      submats[i]->factortype = C->factortype;
    }
  } else {
    for (i=0; i<ismax; i++) {
      PetscInt rbs,cbs;
      ierr = ISGetBlockSize(isrow[i],&rbs); CHKERRQ(ierr);
      ierr = ISGetBlockSize(iscol[i],&cbs); CHKERRQ(ierr);

      ierr = MatCreate(PETSC_COMM_SELF,submats+i);CHKERRQ(ierr);
      ierr = MatSetSizes(submats[i],nrow[i],ncol[i],PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);

      ierr = MatSetBlockSizes(submats[i],rbs,cbs); CHKERRQ(ierr);
      ierr = MatSetType(submats[i],((PetscObject)A)->type_name);CHKERRQ(ierr);
      ierr = MatSeqAIJSetPreallocation(submats[i],0,lens[i]);CHKERRQ(ierr);
    }
  }

  /* Assemble the matrices */
  /* First assemble the local rows */
  {
    PetscInt    ilen_row,*imat_ilen,*imat_j,*imat_i,old_row;
    PetscScalar *imat_a;

    for (i=0; i<ismax; i++) {
      mat       = (Mat_SeqAIJ*)submats[i]->data;
      imat_ilen = mat->ilen;
      imat_j    = mat->j;
      imat_i    = mat->i;
      imat_a    = mat->a;
      if (!allcolumns[i]) cmap_i = cmap[i];
      rmap_i    = rmap[i];
      irow_i    = irow[i];
      jmax      = nrow[i];
      for (j=0; j<jmax; j++) {
	l = 0;
        row      = irow_i[j];
        while (row >= C->rmap->range[l+1]) l++;
        proc = l;
        if (proc == rank) {
          old_row  = row;
#if defined (PETSC_USE_CTABLE)
          ierr = PetscTableFind(rmap_i,row+1,&row);CHKERRQ(ierr);
          row--;
#else
          row      = rmap_i[row];
#endif
          ilen_row = imat_ilen[row];
          ierr     = MatGetRow_MPIAIJ(C,old_row,&ncols,&cols,&vals);CHKERRQ(ierr);
          mat_i    = imat_i[row] ;
          mat_a    = imat_a + mat_i;
          mat_j    = imat_j + mat_i;
          if (!allcolumns[i]){
            for (k=0; k<ncols; k++) {
#if defined (PETSC_USE_CTABLE)
              ierr = PetscTableFind(cmap_i,cols[k]+1,&tcol);CHKERRQ(ierr);
#else
              tcol = cmap_i[cols[k]];
#endif
              if (tcol){
                *mat_j++ = tcol - 1;
                *mat_a++ = vals[k];
                ilen_row++;
              }
            }
          } else { /* allcolumns */
            for (k=0; k<ncols; k++) {
              *mat_j++ = cols[k] ; /* global col index! */
              *mat_a++ = vals[k];
              ilen_row++;
            }
          }
          ierr = MatRestoreRow_MPIAIJ(C,old_row,&ncols,&cols,&vals);CHKERRQ(ierr);
          imat_ilen[row] = ilen_row;
        }
      }
    }
  }

  /*   Now assemble the off proc rows*/
  {
    PetscInt    *sbuf1_i,*rbuf2_i,*rbuf3_i,*imat_ilen,ilen;
    PetscInt    *imat_j,*imat_i;
    PetscScalar *imat_a,*rbuf4_i;

    for (tmp2=0; tmp2<nrqs; tmp2++) {
      ierr = MPI_Waitany(nrqs,r_waits4,&idex2,r_status4+tmp2);CHKERRQ(ierr);
      idex   = pa[idex2];
      sbuf1_i = sbuf1[idex];
      jmax    = sbuf1_i[0];
      ct1     = 2*jmax + 1;
      ct2     = 0;
      rbuf2_i = rbuf2[idex2];
      rbuf3_i = rbuf3[idex2];
      rbuf4_i = rbuf4[idex2];
      for (j=1; j<=jmax; j++) {
        is_no     = sbuf1_i[2*j-1];
        rmap_i    = rmap[is_no];
        if (!allcolumns[is_no]){cmap_i = cmap[is_no];}
        mat       = (Mat_SeqAIJ*)submats[is_no]->data;
        imat_ilen = mat->ilen;
        imat_j    = mat->j;
        imat_i    = mat->i;
        imat_a    = mat->a;
        max1      = sbuf1_i[2*j];
        for (k=0; k<max1; k++,ct1++) {
          row   = sbuf1_i[ct1];
#if defined (PETSC_USE_CTABLE)
          ierr = PetscTableFind(rmap_i,row+1,&row);CHKERRQ(ierr);
          row--;
#else
          row   = rmap_i[row];
#endif
          ilen  = imat_ilen[row];
          mat_i = imat_i[row] ;
          mat_a = imat_a + mat_i;
          mat_j = imat_j + mat_i;
          max2 = rbuf2_i[ct1];
          if (!allcolumns[is_no]){
            for (l=0; l<max2; l++,ct2++) {

#if defined (PETSC_USE_CTABLE)
              ierr = PetscTableFind(cmap_i,rbuf3_i[ct2]+1,&tcol);CHKERRQ(ierr);
#else
              tcol = cmap_i[rbuf3_i[ct2]];
#endif
              if (tcol) {
                *mat_j++ = tcol - 1;
                *mat_a++ = rbuf4_i[ct2];
                ilen++;
              }
            }
          } else { /* allcolumns */
            for (l=0; l<max2; l++,ct2++) {
              *mat_j++ = rbuf3_i[ct2]; /* same global column index of C */
              *mat_a++ = rbuf4_i[ct2];
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
  if (nrqr) {ierr = MPI_Waitall(nrqr,s_waits4,s_status4);CHKERRQ(ierr);}
  ierr = PetscFree(s_waits4);CHKERRQ(ierr);
  ierr = PetscFree(s_status4);CHKERRQ(ierr);

  /* Restore the indices */
  for (i=0; i<ismax; i++) {
    ierr = ISRestoreIndices(isrow[i],irow+i);CHKERRQ(ierr);
    if (!allcolumns[i]){
      ierr = ISRestoreIndices(iscol[i],icol+i);CHKERRQ(ierr);
    }
  }

  /* Destroy allocated memory */
  ierr = PetscFree4(irow,icol,nrow,ncol);CHKERRQ(ierr);
  ierr = PetscFree4(w1,w2,w3,w4);CHKERRQ(ierr);
  ierr = PetscFree(pa);CHKERRQ(ierr);

  ierr = PetscFree4(sbuf1,ptr,tmp,ctr);CHKERRQ(ierr);
  ierr = PetscFree(rbuf2);CHKERRQ(ierr);
  for (i=0; i<nrqr; ++i) {
    ierr = PetscFree(sbuf2[i]);CHKERRQ(ierr);
  }
  for (i=0; i<nrqs; ++i) {
    ierr = PetscFree(rbuf3[i]);CHKERRQ(ierr);
    ierr = PetscFree(rbuf4[i]);CHKERRQ(ierr);
  }

  ierr = PetscFree3(sbuf2,req_size,req_source);CHKERRQ(ierr);
  ierr = PetscFree(rbuf3);CHKERRQ(ierr);
  ierr = PetscFree(rbuf4);CHKERRQ(ierr);
  ierr = PetscFree(sbuf_aj[0]);CHKERRQ(ierr);
  ierr = PetscFree(sbuf_aj);CHKERRQ(ierr);
  ierr = PetscFree(sbuf_aa[0]);CHKERRQ(ierr);
  ierr = PetscFree(sbuf_aa);CHKERRQ(ierr);

#if defined (PETSC_USE_CTABLE)
  for (i=0; i<ismax; i++) {ierr = PetscTableDestroy((PetscTable*)&rmap[i]);CHKERRQ(ierr);}
#else
  if (ismax) {ierr = PetscFree(rmap[0]);CHKERRQ(ierr);}
#endif
  ierr = PetscFree(rmap);CHKERRQ(ierr);

  for (i=0; i<ismax; i++) {
    if (!allcolumns[i]){
#if defined (PETSC_USE_CTABLE)
      ierr = PetscTableDestroy((PetscTable*)&cmap[i]);CHKERRQ(ierr);
#else
      ierr = PetscFree(cmap[i]);CHKERRQ(ierr);
#endif
    }
  }
  ierr = PetscFree(cmap);CHKERRQ(ierr);
  if (ismax) {ierr = PetscFree(lens[0]);CHKERRQ(ierr);}
  ierr = PetscFree(lens);CHKERRQ(ierr);

  for (i=0; i<ismax; i++) {
    ierr = MatAssemblyBegin(submats[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(submats[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
 Observe that the Seq matrices used to construct this MPI matrix are not increfed.
 Be careful not to destroy them elsewhere.
 */
#undef __FUNCT__
#define __FUNCT__ "MatCreateMPIAIJFromSeqMatrices_Private"
PetscErrorCode MatCreateMPIAIJFromSeqMatrices_Private(MPI_Comm comm, Mat A, Mat B, Mat *C)
{
  /* If making this function public, change the error returned in this function away from _PLIB. */
  PetscErrorCode ierr;
  Mat_MPIAIJ     *aij;
  PetscBool      seqaij;

  PetscFunctionBegin;
  /* Check to make sure the component matrices are compatible with C. */
  ierr = PetscObjectTypeCompare((PetscObject)A, MATSEQAIJ, &seqaij); CHKERRQ(ierr);
  if (!seqaij) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Diagonal matrix is of wrong type");
  ierr = PetscObjectTypeCompare((PetscObject)B, MATSEQAIJ, &seqaij); CHKERRQ(ierr);
  if (!seqaij) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Off-diagonal matrix is of wrong type");
  if (A->rmap->n != B->rmap->n || A->rmap->bs != B->rmap->bs || A->cmap->bs != B->cmap->bs) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incompatible component matrices of an MPIAIJ matrix");

  ierr = MatCreate(comm, C); CHKERRQ(ierr);
  ierr = MatSetSizes(*C,A->rmap->n, A->cmap->n, PETSC_DECIDE, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = MatSetBlockSizes(*C,A->rmap->bs, A->cmap->bs); CHKERRQ(ierr);
  ierr = PetscLayoutSetUp((*C)->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp((*C)->cmap);CHKERRQ(ierr);
  if ((*C)->cmap->N != A->cmap->n + B->cmap->n) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incompatible component matrices of an MPIAIJ matrix");
  ierr = MatSetType(*C, MATMPIAIJ); CHKERRQ(ierr);
  aij = (Mat_MPIAIJ*)((*C)->data);
  aij->A = A;
  aij->B = B;
  ierr = PetscLogObjectParent(*C,A); CHKERRQ(ierr);
  ierr = PetscLogObjectParent(*C,B); CHKERRQ(ierr);
  (*C)->preallocated = (PetscBool)(A->preallocated && B->preallocated);
  (*C)->assembled    = (PetscBool)(A->assembled && B->assembled);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMPIAIJExtractSeqMatrices_Private"
PetscErrorCode MatMPIAIJExtractSeqMatrices_Private(Mat C, Mat *A, Mat *B)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ*) (C->data);
  PetscFunctionBegin;
  PetscValidPointer(A,2);
  PetscValidPointer(B,3);
  *A = aij->A;
  *B = aij->B;
  /* Note that we don't incref *A and *B, so be careful! */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetSubMatricesParallel_MPIXAIJ"
PetscErrorCode MatGetSubMatricesParallel_MPIXAIJ(Mat C,PetscInt ismax,const IS isrow[],const IS iscol[],MatReuse scall,Mat *submat[],
						 PetscErrorCode(*getsubmats_seq)(Mat, PetscInt, const IS[], const IS[], MatReuse, Mat**),
						 PetscErrorCode(*makefromseq)(MPI_Comm, Mat, Mat,Mat*),
						 PetscErrorCode(*extractseq)(Mat, Mat*, Mat*))
{
  PetscErrorCode ierr;
  PetscMPIInt    size, flag;
  PetscInt       i,ii;
  PetscInt       ismax_c;

  PetscFunctionBegin;
  if (!ismax) {
    PetscFunctionReturn(0);
  }
  for (i = 0, ismax_c = 0; i < ismax; ++i) {
    ierr = MPI_Comm_compare(((PetscObject)isrow[i])->comm,((PetscObject)iscol[i])->comm, &flag); CHKERRQ(ierr);
    if (flag != MPI_IDENT) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Row and column index sets must have the same communicator");
    ierr = MPI_Comm_size(((PetscObject)isrow[i])->comm, &size); CHKERRQ(ierr);
    if (size > 1) {
      ++ismax_c;
    }
  }
  if (!ismax_c) { /* Sequential ISs only, so can call the sequential matrix extraction subroutine. */
    ierr = (*getsubmats_seq)(C,ismax,isrow,iscol,scall,submat); CHKERRQ(ierr);
  } else { /* if (ismax_c) */
    Mat         *A,*B;
    IS          *isrow_c, *iscol_c;
    PetscMPIInt size;
    /*
     Allocate the necessary arrays to hold the resulting parallel matrices as well as the intermediate
     array of sequential matrices underlying the resulting parallel matrices.
     Which arrays to allocate is based on the value of MatReuse scall.
     There are as many diag matrices as there are original index sets.
     There are only as many parallel and off-diag matrices, as there are parallel (comm size > 1) index sets.

     Sequential matrix arrays are allocated in any event: even if the array of parallel matrices already exists,
     we need to consolidate the underlying seq matrices into as single array to serve as placeholders into getsubmats_seq
     will deposite the extracted diag and off-diag parts.
     However, if reuse is taking place, we have to allocate the seq matrix arrays here.
     If reuse is NOT taking place, then the seq matrix arrays are allocated by getsubmats_seq.
    */

    /* Parallel matrix array is allocated only if no reuse is taking place. */
    if (scall != MAT_REUSE_MATRIX) {
      ierr = PetscMalloc((ismax)*sizeof(Mat),submat);CHKERRQ(ierr);
    } else {
      ierr = PetscMalloc(ismax*sizeof(Mat), &A); CHKERRQ(ierr);
      ierr = PetscMalloc(ismax_c*sizeof(Mat), &B); CHKERRQ(ierr);
      /* If parallel matrices are being reused, then simply reuse the underlying seq matrices as well. */
      for (i = 0, ii = 0; i < ismax; ++i) {
        ierr = MPI_Comm_size(((PetscObject)isrow[i])->comm, &size); CHKERRQ(ierr);
        if (size > 1) {
          ierr = (*extractseq)((*submat)[i],A+i,B+ii); CHKERRQ(ierr);
          ++ii;
        } else {
          A[i] = (*submat)[i];
        }
      }
    }
    /*
     Construct the complements of the iscol ISs for parallel ISs only.
     These are used to extract the off-diag portion of the resulting parallel matrix.
     The row IS for the off-diag portion is the same as for the diag portion,
     so we merely alias the row IS, while skipping those that are sequential.
    */
    ierr = PetscMalloc2(ismax_c,IS,&isrow_c, ismax_c, IS, &iscol_c); CHKERRQ(ierr);
    for (i = 0, ii = 0; i < ismax; ++i) {
      ierr = MPI_Comm_size(((PetscObject)isrow[i])->comm, &size); CHKERRQ(ierr);
      if (size > 1) {
	isrow_c[ii] = isrow[i];
	ierr = ISGetNonlocalIS(iscol[i], &(iscol_c[ii])); CHKERRQ(ierr);
	++ii;
      }
    }
    /* Now obtain the sequential A and B submatrices separately. */
    ierr = (*getsubmats_seq)(C,ismax,isrow, iscol,scall, &A); CHKERRQ(ierr);
    ierr = (*getsubmats_seq)(C,ismax_c,isrow_c, iscol_c,scall, &B); CHKERRQ(ierr);
    for (ii = 0; ii < ismax_c; ++ii) {
      ierr = ISDestroy(&iscol_c[ii]); CHKERRQ(ierr);
    }
    ierr = PetscFree2(isrow_c, iscol_c); CHKERRQ(ierr);
    /*
     If scall == MAT_REUSE_MATRIX, we are done, since the sequential matrices A & B
     have been extracted directly into the parallel matrices containing them, or
     simply into the sequential matrix identical with the corresponding A (if size == 1).
     Otherwise, make sure that parallel matrices are constructed from A & B, or the
     A is put into the correct submat slot (if size == 1).
     */
    if (scall != MAT_REUSE_MATRIX) {
      for (i = 0, ii = 0; i < ismax; ++i) {
        ierr = MPI_Comm_size(((PetscObject)isrow[i])->comm, &size); CHKERRQ(ierr);
        if (size > 1) {
          /*
           For each parallel isrow[i], create parallel matrices from the extracted sequential matrices.
           */
          /* Construct submat[i] from the Seq pieces A and B. */
          ierr = (*makefromseq)(((PetscObject)isrow[i])->comm, A[i], B[ii], (*submat)+i); CHKERRQ(ierr);

          ++ii;
        } else {
          (*submat)[i] = A[i];
        }
      }
    }
    ierr = PetscFree(A); CHKERRQ(ierr);
    ierr = PetscFree(B); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* MatGetSubMatricesParallel_MPIXAIJ() */



#undef __FUNCT__
#define __FUNCT__ "MatGetSubMatricesParallel_MPIAIJ"
PetscErrorCode MatGetSubMatricesParallel_MPIAIJ(Mat C,PetscInt ismax,const IS isrow[],const IS iscol[],MatReuse scall,Mat *submat[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetSubMatricesParallel_MPIXAIJ(C,ismax,isrow,iscol,scall,submat,MatGetSubMatrices_MPIAIJ,MatCreateMPIAIJFromSeqMatrices_Private,MatMPIAIJExtractSeqMatrices_Private);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
