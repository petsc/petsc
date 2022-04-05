
/*
   Routines to compute overlapping regions of a parallel MPI matrix
  and to find submatrices that were shared across processors.
*/
#include <../src/mat/impls/baij/mpi/mpibaij.h>
#include <petscbt.h>

static PetscErrorCode MatIncreaseOverlap_MPIBAIJ_Local(Mat,PetscInt,char**,PetscInt*,PetscInt**);
static PetscErrorCode MatIncreaseOverlap_MPIBAIJ_Receive(Mat,PetscInt,PetscInt**,PetscInt**,PetscInt*);
extern PetscErrorCode MatGetRow_MPIBAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);
extern PetscErrorCode MatRestoreRow_MPIBAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);

PetscErrorCode MatIncreaseOverlap_MPIBAIJ(Mat C,PetscInt imax,IS is[],PetscInt ov)
{
  PetscInt       i,N=C->cmap->N, bs=C->rmap->bs;
  IS             *is_new;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(imax,&is_new));
  /* Convert the indices into block format */
  PetscCall(ISCompressIndicesGeneral(N,C->rmap->n,bs,imax,is,is_new));
  PetscCheck(ov >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap specified");
  for (i=0; i<ov; ++i) {
    PetscCall(MatIncreaseOverlap_MPIBAIJ_Once(C,imax,is_new));
  }
  for (i=0; i<imax; i++) PetscCall(ISDestroy(&is[i]));
  PetscCall(ISExpandIndicesGeneral(N,N,bs,imax,is_new,is));
  for (i=0; i<imax; i++) PetscCall(ISDestroy(&is_new[i]));
  PetscCall(PetscFree(is_new));
  PetscFunctionReturn(0);
}

/*
  Sample message format:
  If a processor A wants processor B to process some elements corresponding
  to index sets is[1], is[5]
  mesg [0] = 2   (no of index sets in the mesg)
  -----------
  mesg [1] = 1 => is[1]
  mesg [2] = sizeof(is[1]);
  -----------
  mesg [5] = 5  => is[5]
  mesg [6] = sizeof(is[5]);
  -----------
  mesg [7]
  mesg [n]  data(is[1])
  -----------
  mesg[n+1]
  mesg[m]  data(is[5])
  -----------

  Notes:
  nrqs - no of requests sent (or to be sent out)
  nrqr - no of requests received (which have to be or which have been processed)
*/
PetscErrorCode MatIncreaseOverlap_MPIBAIJ_Once(Mat C,PetscInt imax,IS is[])
{
  Mat_MPIBAIJ    *c = (Mat_MPIBAIJ*)C->data;
  const PetscInt **idx,*idx_i;
  PetscInt       *n,*w3,*w4,**data,len;
  PetscMPIInt    size,rank,tag1,tag2,*w2,*w1,nrqr;
  PetscInt       Mbs,i,j,k,**rbuf,row,nrqs,msz,**outdat,**ptr;
  PetscInt       *ctr,*pa,*tmp,*isz,*isz1,**xdata,**rbuf2,*d_p;
  PetscMPIInt    *onodes1,*olengths1,*onodes2,*olengths2,proc=-1;
  PetscBT        *table;
  MPI_Comm       comm,*iscomms;
  MPI_Request    *s_waits1,*r_waits1,*s_waits2,*r_waits2;
  char           *t_p;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)C,&comm));
  size = c->size;
  rank = c->rank;
  Mbs  = c->Mbs;

  PetscCall(PetscObjectGetNewTag((PetscObject)C,&tag1));
  PetscCall(PetscObjectGetNewTag((PetscObject)C,&tag2));

  PetscCall(PetscMalloc2(imax+1,(PetscInt***)&idx,imax,&n));

  for (i=0; i<imax; i++) {
    PetscCall(ISGetIndices(is[i],&idx[i]));
    PetscCall(ISGetLocalSize(is[i],&n[i]));
  }

  /* evaluate communication - mesg to who,length of mesg, and buffer space
     required. Based on this, buffers are allocated, and data copied into them*/
  PetscCall(PetscCalloc4(size,&w1,size,&w2,size,&w3,size,&w4));
  for (i=0; i<imax; i++) {
    PetscCall(PetscArrayzero(w4,size)); /* initialise work vector*/
    idx_i = idx[i];
    len   = n[i];
    for (j=0; j<len; j++) {
      row = idx_i[j];
      PetscCheck(row >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index set cannot have negative entries");
      PetscCall(PetscLayoutFindOwner(C->rmap,row*C->rmap->bs,&proc));
      w4[proc]++;
    }
    for (j=0; j<size; j++) {
      if (w4[j]) { w1[j] += w4[j]; w3[j]++;}
    }
  }

  nrqs     = 0;              /* no of outgoing messages */
  msz      = 0;              /* total mesg length (for all proc */
  w1[rank] = 0;              /* no mesg sent to itself */
  w3[rank] = 0;
  for (i=0; i<size; i++) {
    if (w1[i])  {w2[i] = 1; nrqs++;} /* there exists a message to proc i */
  }
  /* pa - is list of processors to communicate with */
  PetscCall(PetscMalloc1(nrqs,&pa));
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
  PetscCall(PetscGatherNumberOfMessages(comm,w2,w1,&nrqr));
  PetscCall(PetscGatherMessageLengths(comm,nrqs,nrqr,w1,&onodes1,&olengths1));

  /* Now post the Irecvs corresponding to these messages */
  PetscCall(PetscPostIrecvInt(comm,tag1,nrqr,onodes1,olengths1,&rbuf,&r_waits1));

  /* Allocate Memory for outgoing messages */
  PetscCall(PetscMalloc4(size,&outdat,size,&ptr,msz,&tmp,size,&ctr));
  PetscCall(PetscArrayzero(outdat,size));
  PetscCall(PetscArrayzero(ptr,size));
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
    PetscCall(PetscArrayzero(outdat[j]+1,2*w3[j]));
    ptr[j]       = outdat[j] + 2*w3[j] + 1;
  }

  /* Memory for doing local proc's work*/
  {
    PetscCall(PetscCalloc5(imax,&table, imax,&data, imax,&isz, Mbs*imax,&d_p, (Mbs/PETSC_BITS_PER_BYTE+1)*imax,&t_p));

    for (i=0; i<imax; i++) {
      table[i] = t_p + (Mbs/PETSC_BITS_PER_BYTE+1)*i;
      data[i]  = d_p + (Mbs)*i;
    }
  }

  /* Parse the IS and update local tables and the outgoing buf with the data*/
  {
    PetscInt n_i,*data_i,isz_i,*outdat_j,ctr_j;
    PetscBT  table_i;

    for (i=0; i<imax; i++) {
      PetscCall(PetscArrayzero(ctr,size));
      n_i     = n[i];
      table_i = table[i];
      idx_i   = idx[i];
      data_i  = data[i];
      isz_i   = isz[i];
      for (j=0; j<n_i; j++) {   /* parse the indices of each IS */
        row  = idx_i[j];
        PetscCall(PetscLayoutFindOwner(C->rmap,row*C->rmap->bs,&proc));
        if (proc != rank) { /* copy to the outgoing buffer */
          ctr[proc]++;
          *ptr[proc] = row;
          ptr[proc]++;
        } else { /* Update the local table */
          if (!PetscBTLookupSet(table_i,row)) data_i[isz_i++] = row;
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
  PetscCall(PetscMalloc1(nrqs,&s_waits1));
  for (i=0; i<nrqs; ++i) {
    j    = pa[i];
    PetscCallMPI(MPI_Isend(outdat[j],w1[j],MPIU_INT,j,tag1,comm,s_waits1+i));
  }

  /* No longer need the original indices*/
  for (i=0; i<imax; ++i) {
    PetscCall(ISRestoreIndices(is[i],idx+i));
  }
  PetscCall(PetscFree2(*(PetscInt***)&idx,n));

  PetscCall(PetscMalloc1(imax,&iscomms));
  for (i=0; i<imax; ++i) {
    PetscCall(PetscCommDuplicate(PetscObjectComm((PetscObject)is[i]),&iscomms[i],NULL));
    PetscCall(ISDestroy(&is[i]));
  }

  /* Do Local work*/
  PetscCall(MatIncreaseOverlap_MPIBAIJ_Local(C,imax,table,isz,data));

  /* Receive messages*/
  PetscCallMPI(MPI_Waitall(nrqr,r_waits1,MPI_STATUSES_IGNORE));
  PetscCallMPI(MPI_Waitall(nrqs,s_waits1,MPI_STATUSES_IGNORE));

  /* Phase 1 sends are complete - deallocate buffers */
  PetscCall(PetscFree4(outdat,ptr,tmp,ctr));
  PetscCall(PetscFree4(w1,w2,w3,w4));

  PetscCall(PetscMalloc1(nrqr,&xdata));
  PetscCall(PetscMalloc1(nrqr,&isz1));
  PetscCall(MatIncreaseOverlap_MPIBAIJ_Receive(C,nrqr,rbuf,xdata,isz1));
  if (rbuf) {
    PetscCall(PetscFree(rbuf[0]));
    PetscCall(PetscFree(rbuf));
  }

  /* Send the data back*/
  /* Do a global reduction to know the buffer space req for incoming messages*/
  {
    PetscMPIInt *rw1;

    PetscCall(PetscCalloc1(size,&rw1));

    for (i=0; i<nrqr; ++i) {
      proc = onodes1[i];
      rw1[proc] = isz1[i];
    }

    /* Determine the number of messages to expect, their lengths, from from-ids */
    PetscCall(PetscGatherMessageLengths(comm,nrqr,nrqs,rw1,&onodes2,&olengths2));
    PetscCall(PetscFree(rw1));
  }
  /* Now post the Irecvs corresponding to these messages */
  PetscCall(PetscPostIrecvInt(comm,tag2,nrqs,onodes2,olengths2,&rbuf2,&r_waits2));

  /*  Now  post the sends */
  PetscCall(PetscMalloc1(nrqr,&s_waits2));
  for (i=0; i<nrqr; ++i) {
    j    = onodes1[i];
    PetscCallMPI(MPI_Isend(xdata[i],isz1[i],MPIU_INT,j,tag2,comm,s_waits2+i));
  }

  PetscCall(PetscFree(onodes1));
  PetscCall(PetscFree(olengths1));

  /* receive work done on other processors*/
  {
    PetscMPIInt idex;
    PetscInt    is_no,ct1,max,*rbuf2_i,isz_i,*data_i,jmax;
    PetscBT     table_i;

    for (i=0; i<nrqs; ++i) {
      PetscCallMPI(MPI_Waitany(nrqs,r_waits2,&idex,MPI_STATUS_IGNORE));
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
          if (!PetscBTLookupSet(table_i,row)) data_i[isz_i++] = row;
        }
        isz[is_no] = isz_i;
      }
    }
    PetscCallMPI(MPI_Waitall(nrqr,s_waits2,MPI_STATUSES_IGNORE));
  }

  for (i=0; i<imax; ++i) {
    PetscCall(ISCreateGeneral(iscomms[i],isz[i],data[i],PETSC_COPY_VALUES,is+i));
    PetscCall(PetscCommDestroy(&iscomms[i]));
  }

  PetscCall(PetscFree(iscomms));
  PetscCall(PetscFree(onodes2));
  PetscCall(PetscFree(olengths2));

  PetscCall(PetscFree(pa));
  if (rbuf2) {
    PetscCall(PetscFree(rbuf2[0]));
    PetscCall(PetscFree(rbuf2));
  }
  PetscCall(PetscFree(s_waits1));
  PetscCall(PetscFree(r_waits1));
  PetscCall(PetscFree(s_waits2));
  PetscCall(PetscFree(r_waits2));
  PetscCall(PetscFree5(table,data,isz,d_p,t_p));
  if (xdata) {
    PetscCall(PetscFree(xdata[0]));
    PetscCall(PetscFree(xdata));
  }
  PetscCall(PetscFree(isz1));
  PetscFunctionReturn(0);
}

/*
   MatIncreaseOverlap_MPIBAIJ_Local - Called by MatincreaseOverlap, to do
       the work on the local processor.

     Inputs:
      C      - MAT_MPIBAIJ;
      imax - total no of index sets processed at a time;
      table  - an array of char - size = Mbs bits.

     Output:
      isz    - array containing the count of the solution elements corresponding
               to each index set;
      data   - pointer to the solutions
*/
static PetscErrorCode MatIncreaseOverlap_MPIBAIJ_Local(Mat C,PetscInt imax,PetscBT *table,PetscInt *isz,PetscInt **data)
{
  Mat_MPIBAIJ *c = (Mat_MPIBAIJ*)C->data;
  Mat         A  = c->A,B = c->B;
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)B->data;
  PetscInt    start,end,val,max,rstart,cstart,*ai,*aj;
  PetscInt    *bi,*bj,*garray,i,j,k,row,*data_i,isz_i;
  PetscBT     table_i;

  PetscFunctionBegin;
  rstart = c->rstartbs;
  cstart = c->cstartbs;
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
        if (!PetscBTLookupSet(table_i,val)) data_i[isz_i++] = val;
      }
      start = bi[row];
      end   = bi[row+1];
      for (k=start; k<end; k++) { /* Bmat */
        val = garray[bj[k]];
        if (!PetscBTLookupSet(table_i,val)) data_i[isz_i++] = val;
      }
    }
    isz[i] = isz_i;
  }
  PetscFunctionReturn(0);
}
/*
      MatIncreaseOverlap_MPIBAIJ_Receive - Process the received messages,
         and return the output

         Input:
           C    - the matrix
           nrqr - no of messages being processed.
           rbuf - an array of pointers to the received requests

         Output:
           xdata - array of messages to be sent back
           isz1  - size of each message

  For better efficiency perhaps we should malloc separately each xdata[i],
then if a remalloc is required we need only copy the data for that one row
rather than all previous rows as it is now where a single large chunk of
memory is used.

*/
static PetscErrorCode MatIncreaseOverlap_MPIBAIJ_Receive(Mat C,PetscInt nrqr,PetscInt **rbuf,PetscInt **xdata,PetscInt * isz1)
{
  Mat_MPIBAIJ    *c = (Mat_MPIBAIJ*)C->data;
  Mat            A  = c->A,B = c->B;
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)B->data;
  PetscInt       rstart,cstart,*ai,*aj,*bi,*bj,*garray,i,j,k;
  PetscInt       row,total_sz,ct,ct1,ct2,ct3,mem_estimate,oct2,l,start,end;
  PetscInt       val,max1,max2,Mbs,no_malloc =0,*tmp,new_estimate,ctr;
  PetscInt       *rbuf_i,kmax,rbuf_0;
  PetscBT        xtable;

  PetscFunctionBegin;
  Mbs    = c->Mbs;
  rstart = c->rstartbs;
  cstart = c->cstartbs;
  ai     = a->i;
  aj     = a->j;
  bi     = b->i;
  bj     = b->j;
  garray = c->garray;

  for (i=0,ct=0,total_sz=0; i<nrqr; ++i) {
    rbuf_i =  rbuf[i];
    rbuf_0 =  rbuf_i[0];
    ct    += rbuf_0;
    for (j=1; j<=rbuf_0; j++) total_sz += rbuf_i[2*j];
  }

  if (c->Mbs) max1 = ct*(a->nz +b->nz)/c->Mbs;
  else        max1 = 1;
  mem_estimate = 3*((total_sz > max1 ? total_sz : max1)+1);
  if (nrqr) {
    PetscCall(PetscMalloc1(mem_estimate,&xdata[0]));
    ++no_malloc;
  }
  PetscCall(PetscBTCreate(Mbs,&xtable));
  PetscCall(PetscArrayzero(isz1,nrqr));

  ct3 = 0;
  for (i=0; i<nrqr; i++) { /* for easch mesg from proc i */
    rbuf_i =  rbuf[i];
    rbuf_0 =  rbuf_i[0];
    ct1    =  2*rbuf_0+1;
    ct2    =  ct1;
    ct3   += ct1;
    for (j=1; j<=rbuf_0; j++) { /* for each IS from proc i*/
      PetscCall(PetscBTMemzero(Mbs,xtable));
      oct2 = ct2;
      kmax = rbuf_i[2*j];
      for (k=0; k<kmax; k++,ct1++) {
        row = rbuf_i[ct1];
        if (!PetscBTLookupSet(xtable,row)) {
          if (!(ct3 < mem_estimate)) {
            new_estimate = (PetscInt)(1.5*mem_estimate)+1;
            PetscCall(PetscMalloc1(new_estimate,&tmp));
            PetscCall(PetscArraycpy(tmp,xdata[0],mem_estimate));
            PetscCall(PetscFree(xdata[0]));
            xdata[0]     = tmp;
            mem_estimate = new_estimate; ++no_malloc;
            for (ctr=1; ctr<=i; ctr++) xdata[ctr] = xdata[ctr-1] + isz1[ctr-1];
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
              new_estimate = (PetscInt)(1.5*mem_estimate)+1;
              PetscCall(PetscMalloc1(new_estimate,&tmp));
              PetscCall(PetscArraycpy(tmp,xdata[0],mem_estimate));
              PetscCall(PetscFree(xdata[0]));
              xdata[0]     = tmp;
              mem_estimate = new_estimate; ++no_malloc;
              for (ctr=1; ctr<=i; ctr++) xdata[ctr] = xdata[ctr-1] + isz1[ctr-1];
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
              PetscCall(PetscMalloc1(new_estimate,&tmp));
              PetscCall(PetscArraycpy(tmp,xdata[0],mem_estimate));
              PetscCall(PetscFree(xdata[0]));
              xdata[0]     = tmp;
              mem_estimate = new_estimate; ++no_malloc;
              for (ctr =1; ctr <=i; ctr++) xdata[ctr] = xdata[ctr-1] + isz1[ctr-1];
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
    if (i+1<nrqr) xdata[i+1] = xdata[i] + ct2;
    isz1[i]     = ct2; /* size of each message */
  }
  PetscCall(PetscBTDestroy(&xtable));
  PetscCall(PetscInfo(C,"Allocated %" PetscInt_FMT " bytes, required %" PetscInt_FMT ", no of mallocs = %" PetscInt_FMT "\n",mem_estimate,ct3,no_malloc));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrices_MPIBAIJ(Mat C,PetscInt ismax,const IS isrow[],const IS iscol[],MatReuse scall,Mat *submat[])
{
  IS             *isrow_block,*iscol_block;
  Mat_MPIBAIJ    *c = (Mat_MPIBAIJ*)C->data;
  PetscInt       nmax,nstages_local,nstages,i,pos,max_no,N=C->cmap->N,bs=C->rmap->bs;
  Mat_SeqBAIJ    *subc;
  Mat_SubSppt    *smat;

  PetscFunctionBegin;
  /* The compression and expansion should be avoided. Doesn't point
     out errors, might change the indices, hence buggey */
  PetscCall(PetscMalloc2(ismax+1,&isrow_block,ismax+1,&iscol_block));
  PetscCall(ISCompressIndicesGeneral(N,C->rmap->n,bs,ismax,isrow,isrow_block));
  PetscCall(ISCompressIndicesGeneral(N,C->cmap->n,bs,ismax,iscol,iscol_block));

  /* Determine the number of stages through which submatrices are done */
  if (!C->cmap->N) nmax=20*1000000/sizeof(PetscInt);
  else nmax = 20*1000000 / (c->Nbs * sizeof(PetscInt));
  if (!nmax) nmax = 1;

  if (scall == MAT_INITIAL_MATRIX) {
    nstages_local = ismax/nmax + ((ismax % nmax) ? 1 : 0); /* local nstages */

    /* Make sure every processor loops through the nstages */
    PetscCall(MPIU_Allreduce(&nstages_local,&nstages,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)C)));

    /* Allocate memory to hold all the submatrices and dummy submatrices */
    PetscCall(PetscCalloc1(ismax+nstages,submat));
  } else { /* MAT_REUSE_MATRIX */
    if (ismax) {
      subc = (Mat_SeqBAIJ*)((*submat)[0]->data);
      smat   = subc->submatis1;
    } else { /* (*submat)[0] is a dummy matrix */
      smat = (Mat_SubSppt*)(*submat)[0]->data;
    }
    PetscCheck(smat,PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"MatCreateSubMatrices(...,MAT_REUSE_MATRIX,...) requires submat");
    nstages = smat->nstages;
  }

  for (i=0,pos=0; i<nstages; i++) {
    if (pos+nmax <= ismax) max_no = nmax;
    else if (pos >= ismax) max_no = 0;
    else                   max_no = ismax-pos;

    PetscCall(MatCreateSubMatrices_MPIBAIJ_local(C,max_no,isrow_block+pos,iscol_block+pos,scall,*submat+pos));
    if (!max_no) {
      if (scall == MAT_INITIAL_MATRIX) { /* submat[pos] is a dummy matrix */
        smat = (Mat_SubSppt*)(*submat)[pos]->data;
        smat->nstages = nstages;
      }
      pos++; /* advance to next dummy matrix if any */
    } else pos += max_no;
  }

  if (scall == MAT_INITIAL_MATRIX && ismax) {
    /* save nstages for reuse */
    subc = (Mat_SeqBAIJ*)((*submat)[0]->data);
    smat = subc->submatis1;
    smat->nstages = nstages;
  }

  for (i=0; i<ismax; i++) {
    PetscCall(ISDestroy(&isrow_block[i]));
    PetscCall(ISDestroy(&iscol_block[i]));
  }
  PetscCall(PetscFree2(isrow_block,iscol_block));
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_CTABLE)
PetscErrorCode PetscGetProc(const PetscInt row, const PetscMPIInt size, const PetscInt proc_gnode[], PetscMPIInt *rank)
{
  PetscInt       nGlobalNd = proc_gnode[size];
  PetscMPIInt    fproc;

  PetscFunctionBegin;
  PetscCall(PetscMPIIntCast((PetscInt)(((float)row * (float)size / (float)nGlobalNd + 0.5)),&fproc));
  if (fproc > size) fproc = size;
  while (row < proc_gnode[fproc] || row >= proc_gnode[fproc+1]) {
    if (row < proc_gnode[fproc]) fproc--;
    else                         fproc++;
  }
  *rank = fproc;
  PetscFunctionReturn(0);
}
#endif

/* -------------------------------------------------------------------------*/
/* This code is used for BAIJ and SBAIJ matrices (unfortunate dependency) */
PetscErrorCode MatCreateSubMatrices_MPIBAIJ_local(Mat C,PetscInt ismax,const IS isrow[],const IS iscol[],MatReuse scall,Mat *submats)
{
  Mat_MPIBAIJ    *c = (Mat_MPIBAIJ*)C->data;
  Mat            A  = c->A;
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)c->B->data,*subc;
  const PetscInt **icol,**irow;
  PetscInt       *nrow,*ncol,start;
  PetscMPIInt    rank,size,tag0,tag2,tag3,tag4,*w1,*w2,*w3,*w4,nrqr;
  PetscInt       **sbuf1,**sbuf2,*sbuf2_i,i,j,k,l,ct1,ct2,**rbuf1,row,proc=-1;
  PetscInt       nrqs=0,msz,**ptr=NULL,*req_size=NULL,*ctr=NULL,*pa,*tmp=NULL,tcol;
  PetscInt       **rbuf3=NULL,*req_source1=NULL,*req_source2,**sbuf_aj,**rbuf2=NULL,max1,max2;
  PetscInt       **lens,is_no,ncols,*cols,mat_i,*mat_j,tmp2,jmax;
#if defined(PETSC_USE_CTABLE)
  PetscTable     *cmap,cmap_i=NULL,*rmap,rmap_i;
#else
  PetscInt       **cmap,*cmap_i=NULL,**rmap,*rmap_i;
#endif
  const PetscInt *irow_i,*icol_i;
  PetscInt       ctr_j,*sbuf1_j,*sbuf_aj_i,*rbuf1_i,kmax,*lens_i;
  MPI_Request    *s_waits1,*r_waits1,*s_waits2,*r_waits2,*r_waits3;
  MPI_Request    *r_waits4,*s_waits3,*s_waits4;
  MPI_Comm       comm;
  PetscScalar    **rbuf4,*rbuf4_i=NULL,**sbuf_aa,*vals,*mat_a=NULL,*imat_a=NULL,*sbuf_aa_i;
  PetscMPIInt    *onodes1,*olengths1,end;
  PetscInt       **row2proc,*row2proc_i,*imat_ilen,*imat_j,*imat_i;
  Mat_SubSppt    *smat_i;
  PetscBool      *issorted,colflag,iscsorted=PETSC_TRUE;
  PetscInt       *sbuf1_i,*rbuf2_i,*rbuf3_i,ilen;
  PetscInt       bs=C->rmap->bs,bs2=c->bs2,rstart = c->rstartbs;
  PetscBool      ijonly=c->ijonly; /* private flag indicates only matrix data structures are requested */
  PetscInt       nzA,nzB,*a_i=a->i,*b_i=b->i,*a_j = a->j,*b_j = b->j,ctmp,imark,*cworkA,*cworkB;
  PetscScalar    *vworkA=NULL,*vworkB=NULL,*a_a = a->a,*b_a = b->a;
  PetscInt       cstart = c->cstartbs,*bmap = c->garray;
  PetscBool      *allrows,*allcolumns;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)C,&comm));
  size = c->size;
  rank = c->rank;

  PetscCall(PetscMalloc5(ismax,&row2proc,ismax,&cmap,ismax,&rmap,ismax+1,&allcolumns,ismax,&allrows));
  PetscCall(PetscMalloc5(ismax,(PetscInt***)&irow,ismax,(PetscInt***)&icol,ismax,&nrow,ismax,&ncol,ismax,&issorted));

  for (i=0; i<ismax; i++) {
    PetscCall(ISSorted(iscol[i],&issorted[i]));
    if (!issorted[i]) iscsorted = issorted[i]; /* columns are not sorted! */
    PetscCall(ISSorted(isrow[i],&issorted[i]));

    /* Check for special case: allcolumns */
    PetscCall(ISIdentity(iscol[i],&colflag));
    PetscCall(ISGetLocalSize(iscol[i],&ncol[i]));

    if (colflag && ncol[i] == c->Nbs) {
      allcolumns[i] = PETSC_TRUE;
      icol[i]       = NULL;
    } else {
      allcolumns[i] = PETSC_FALSE;
      PetscCall(ISGetIndices(iscol[i],&icol[i]));
    }

    /* Check for special case: allrows */
    PetscCall(ISIdentity(isrow[i],&colflag));
    PetscCall(ISGetLocalSize(isrow[i],&nrow[i]));
    if (colflag && nrow[i] == c->Mbs) {
      allrows[i] = PETSC_TRUE;
      irow[i]    = NULL;
    } else {
      allrows[i] = PETSC_FALSE;
      PetscCall(ISGetIndices(isrow[i],&irow[i]));
    }
  }

  if (scall == MAT_REUSE_MATRIX) {
    /* Assumes new rows are same length as the old rows */
    for (i=0; i<ismax; i++) {
      subc = (Mat_SeqBAIJ*)(submats[i]->data);
      PetscCheckFalse(subc->mbs != nrow[i] || subc->nbs != ncol[i],PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong size");

      /* Initial matrix as if empty */
      PetscCall(PetscArrayzero(subc->ilen,subc->mbs));

      /* Initial matrix as if empty */
      submats[i]->factortype = C->factortype;

      smat_i   = subc->submatis1;

      nrqs        = smat_i->nrqs;
      nrqr        = smat_i->nrqr;
      rbuf1       = smat_i->rbuf1;
      rbuf2       = smat_i->rbuf2;
      rbuf3       = smat_i->rbuf3;
      req_source2 = smat_i->req_source2;

      sbuf1     = smat_i->sbuf1;
      sbuf2     = smat_i->sbuf2;
      ptr       = smat_i->ptr;
      tmp       = smat_i->tmp;
      ctr       = smat_i->ctr;

      pa          = smat_i->pa;
      req_size    = smat_i->req_size;
      req_source1 = smat_i->req_source1;

      allcolumns[i] = smat_i->allcolumns;
      allrows[i]    = smat_i->allrows;
      row2proc[i]   = smat_i->row2proc;
      rmap[i]       = smat_i->rmap;
      cmap[i]       = smat_i->cmap;
    }

    if (!ismax){ /* Get dummy submatrices and retrieve struct submatis1 */
      PetscCheck(submats[0],PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"submats are null, cannot reuse");
      smat_i = (Mat_SubSppt*)submats[0]->data;

      nrqs        = smat_i->nrqs;
      nrqr        = smat_i->nrqr;
      rbuf1       = smat_i->rbuf1;
      rbuf2       = smat_i->rbuf2;
      rbuf3       = smat_i->rbuf3;
      req_source2 = smat_i->req_source2;

      sbuf1       = smat_i->sbuf1;
      sbuf2       = smat_i->sbuf2;
      ptr         = smat_i->ptr;
      tmp         = smat_i->tmp;
      ctr         = smat_i->ctr;

      pa          = smat_i->pa;
      req_size    = smat_i->req_size;
      req_source1 = smat_i->req_source1;

      allcolumns[0] = PETSC_FALSE;
    }
  } else { /* scall == MAT_INITIAL_MATRIX */
    /* Get some new tags to keep the communication clean */
    PetscCall(PetscObjectGetNewTag((PetscObject)C,&tag2));
    PetscCall(PetscObjectGetNewTag((PetscObject)C,&tag3));

    /* evaluate communication - mesg to who, length of mesg, and buffer space
     required. Based on this, buffers are allocated, and data copied into them*/
    PetscCall(PetscCalloc4(size,&w1,size,&w2,size,&w3,size,&w4));   /* mesg size, initialize work vectors */

    for (i=0; i<ismax; i++) {
      jmax   = nrow[i];
      irow_i = irow[i];

      PetscCall(PetscMalloc1(jmax,&row2proc_i));
      row2proc[i] = row2proc_i;

      if (issorted[i]) proc = 0;
      for (j=0; j<jmax; j++) {
        if (!issorted[i]) proc = 0;
        if (allrows[i]) row = j;
        else row = irow_i[j];

        while (row >= c->rangebs[proc+1]) proc++;
        w4[proc]++;
        row2proc_i[j] = proc; /* map row index to proc */
      }
      for (j=0; j<size; j++) {
        if (w4[j]) { w1[j] += w4[j];  w3[j]++; w4[j] = 0;}
      }
    }

    nrqs     = 0;              /* no of outgoing messages */
    msz      = 0;              /* total mesg length (for all procs) */
    w1[rank] = 0;              /* no mesg sent to self */
    w3[rank] = 0;
    for (i=0; i<size; i++) {
      if (w1[i])  { w2[i] = 1; nrqs++;} /* there exists a message to proc i */
    }
    PetscCall(PetscMalloc1(nrqs,&pa)); /*(proc -array)*/
    for (i=0,j=0; i<size; i++) {
      if (w1[i]) { pa[j] = i; j++; }
    }

    /* Each message would have a header = 1 + 2*(no of IS) + data */
    for (i=0; i<nrqs; i++) {
      j      = pa[i];
      w1[j] += w2[j] + 2* w3[j];
      msz   += w1[j];
    }
    PetscCall(PetscInfo(0,"Number of outgoing messages %" PetscInt_FMT " Total message length %" PetscInt_FMT "\n",nrqs,msz));

    /* Determine the number of messages to expect, their lengths, from from-ids */
    PetscCall(PetscGatherNumberOfMessages(comm,w2,w1,&nrqr));
    PetscCall(PetscGatherMessageLengths(comm,nrqs,nrqr,w1,&onodes1,&olengths1));

    /* Now post the Irecvs corresponding to these messages */
    PetscCall(PetscObjectGetNewTag((PetscObject)C,&tag0));
    PetscCall(PetscPostIrecvInt(comm,tag0,nrqr,onodes1,olengths1,&rbuf1,&r_waits1));

    /* Allocate Memory for outgoing messages */
    PetscCall(PetscMalloc4(size,&sbuf1,size,&ptr,2*msz,&tmp,size,&ctr));
    PetscCall(PetscArrayzero(sbuf1,size));
    PetscCall(PetscArrayzero(ptr,size));

    {
      PetscInt *iptr = tmp;
      k    = 0;
      for (i=0; i<nrqs; i++) {
        j        = pa[i];
        iptr    += k;
        sbuf1[j] = iptr;
        k        = w1[j];
      }
    }

    /* Form the outgoing messages. Initialize the header space */
    for (i=0; i<nrqs; i++) {
      j           = pa[i];
      sbuf1[j][0] = 0;
      PetscCall(PetscArrayzero(sbuf1[j]+1,2*w3[j]));
      ptr[j]      = sbuf1[j] + 2*w3[j] + 1;
    }

    /* Parse the isrow and copy data into outbuf */
    for (i=0; i<ismax; i++) {
      row2proc_i = row2proc[i];
      PetscCall(PetscArrayzero(ctr,size));
      irow_i = irow[i];
      jmax   = nrow[i];
      for (j=0; j<jmax; j++) {  /* parse the indices of each IS */
        proc = row2proc_i[j];
        if (allrows[i]) row = j;
        else row = irow_i[j];

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
    PetscCall(PetscMalloc1(nrqs,&s_waits1));
    for (i=0; i<nrqs; ++i) {
      j    = pa[i];
      PetscCallMPI(MPI_Isend(sbuf1[j],w1[j],MPIU_INT,j,tag0,comm,s_waits1+i));
    }

    /* Post Receives to capture the buffer size */
    PetscCall(PetscMalloc1(nrqs,&r_waits2));
    PetscCall(PetscMalloc3(nrqs,&req_source2,nrqs,&rbuf2,nrqs,&rbuf3));
    if (nrqs) rbuf2[0] = tmp + msz;
    for (i=1; i<nrqs; ++i) {
      rbuf2[i] = rbuf2[i-1]+w1[pa[i-1]];
    }
    for (i=0; i<nrqs; ++i) {
      j    = pa[i];
      PetscCallMPI(MPI_Irecv(rbuf2[i],w1[j],MPIU_INT,j,tag2,comm,r_waits2+i));
    }

    /* Send to other procs the buf size they should allocate */
    /* Receive messages*/
    PetscCall(PetscMalloc1(nrqr,&s_waits2));
    PetscCall(PetscMalloc3(nrqr,&sbuf2,nrqr,&req_size,nrqr,&req_source1));

    PetscCallMPI(MPI_Waitall(nrqr,r_waits1,MPI_STATUSES_IGNORE));
    for (i=0; i<nrqr; ++i) {
      req_size[i] = 0;
      rbuf1_i        = rbuf1[i];
      start          = 2*rbuf1_i[0] + 1;
      end            = olengths1[i];
      PetscCall(PetscMalloc1(end,&sbuf2[i]));
      sbuf2_i        = sbuf2[i];
      for (j=start; j<end; j++) {
        row             = rbuf1_i[j] - rstart;
        ncols           = a_i[row+1] - a_i[row] + b_i[row+1] - b_i[row];
        sbuf2_i[j]      = ncols;
        req_size[i] += ncols;
      }
      req_source1[i] = onodes1[i];
      /* form the header */
      sbuf2_i[0] = req_size[i];
      for (j=1; j<start; j++) sbuf2_i[j] = rbuf1_i[j];

      PetscCallMPI(MPI_Isend(sbuf2_i,end,MPIU_INT,req_source1[i],tag2,comm,s_waits2+i));
    }

    PetscCall(PetscFree(onodes1));
    PetscCall(PetscFree(olengths1));

    PetscCall(PetscFree(r_waits1));
    PetscCall(PetscFree4(w1,w2,w3,w4));

    /* Receive messages*/
    PetscCall(PetscMalloc1(nrqs,&r_waits3));

    PetscCallMPI(MPI_Waitall(nrqs,r_waits2,MPI_STATUSES_IGNORE));
    for (i=0; i<nrqs; ++i) {
      PetscCall(PetscMalloc1(rbuf2[i][0],&rbuf3[i]));
      req_source2[i] = pa[i];
      PetscCallMPI(MPI_Irecv(rbuf3[i],rbuf2[i][0],MPIU_INT,req_source2[i],tag3,comm,r_waits3+i));
    }
    PetscCall(PetscFree(r_waits2));

    /* Wait on sends1 and sends2 */
    PetscCallMPI(MPI_Waitall(nrqs,s_waits1,MPI_STATUSES_IGNORE));
    PetscCallMPI(MPI_Waitall(nrqr,s_waits2,MPI_STATUSES_IGNORE));
    PetscCall(PetscFree(s_waits1));
    PetscCall(PetscFree(s_waits2));

    /* Now allocate sending buffers for a->j, and send them off */
    PetscCall(PetscMalloc1(nrqr,&sbuf_aj));
    for (i=0,j=0; i<nrqr; i++) j += req_size[i];
    if (nrqr) PetscCall(PetscMalloc1(j,&sbuf_aj[0]));
    for (i=1; i<nrqr; i++) sbuf_aj[i] = sbuf_aj[i-1] + req_size[i-1];

    PetscCall(PetscMalloc1(nrqr,&s_waits3));
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
            nzA    = a_i[row+1] - a_i[row]; nzB = b_i[row+1] - b_i[row];
            ncols  = nzA + nzB;
            cworkA = a_j + a_i[row]; cworkB = b_j + b_i[row];

            /* load the column indices for this row into cols */
            cols = sbuf_aj_i + ct2;
            for (l=0; l<nzB; l++) {
              if ((ctmp = bmap[cworkB[l]]) < cstart) cols[l] = ctmp;
              else break;
            }
            imark = l;
            for (l=0; l<nzA; l++) {cols[imark+l] = cstart + cworkA[l];}
            for (l=imark; l<nzB; l++) cols[nzA+l] = bmap[cworkB[l]];
            ct2 += ncols;
          }
        }
        PetscCallMPI(MPI_Isend(sbuf_aj_i,req_size[i],MPIU_INT,req_source1[i],tag3,comm,s_waits3+i));
      }
    }

    /* create col map: global col of C -> local col of submatrices */
#if defined(PETSC_USE_CTABLE)
    for (i=0; i<ismax; i++) {
      if (!allcolumns[i]) {
        PetscCall(PetscTableCreate(ncol[i],c->Nbs,&cmap[i]));

        jmax   = ncol[i];
        icol_i = icol[i];
        cmap_i = cmap[i];
        for (j=0; j<jmax; j++) {
          PetscCall(PetscTableAdd(cmap[i],icol_i[j]+1,j+1,INSERT_VALUES));
        }
      } else cmap[i] = NULL;
    }
#else
    for (i=0; i<ismax; i++) {
      if (!allcolumns[i]) {
        PetscCall(PetscCalloc1(c->Nbs,&cmap[i]));
        jmax   = ncol[i];
        icol_i = icol[i];
        cmap_i = cmap[i];
        for (j=0; j<jmax; j++) cmap_i[icol_i[j]] = j+1;
      } else cmap[i] = NULL;
    }
#endif

    /* Create lens which is required for MatCreate... */
    for (i=0,j=0; i<ismax; i++) j += nrow[i];
    PetscCall(PetscMalloc1(ismax,&lens));

    if (ismax) {
      PetscCall(PetscCalloc1(j,&lens[0]));
    }
    for (i=1; i<ismax; i++) lens[i] = lens[i-1] + nrow[i-1];

    /* Update lens from local data */
    for (i=0; i<ismax; i++) {
      row2proc_i = row2proc[i];
      jmax = nrow[i];
      if (!allcolumns[i]) cmap_i = cmap[i];
      irow_i = irow[i];
      lens_i = lens[i];
      for (j=0; j<jmax; j++) {
        if (allrows[i]) row = j;
        else row = irow_i[j]; /* global blocked row of C */

        proc = row2proc_i[j];
        if (proc == rank) {
          /* Get indices from matA and then from matB */
#if defined(PETSC_USE_CTABLE)
          PetscInt   tt;
#endif
          row    = row - rstart;
          nzA    = a_i[row+1] - a_i[row];
          nzB    = b_i[row+1] - b_i[row];
          cworkA =  a_j + a_i[row];
          cworkB = b_j + b_i[row];

          if (!allcolumns[i]) {
#if defined(PETSC_USE_CTABLE)
            for (k=0; k<nzA; k++) {
              PetscCall(PetscTableFind(cmap_i,cstart+cworkA[k]+1,&tt));
              if (tt) lens_i[j]++;
            }
            for (k=0; k<nzB; k++) {
              PetscCall(PetscTableFind(cmap_i,bmap[cworkB[k]]+1,&tt));
              if (tt) lens_i[j]++;
            }

#else
            for (k=0; k<nzA; k++) {
              if (cmap_i[cstart + cworkA[k]]) lens_i[j]++;
            }
            for (k=0; k<nzB; k++) {
              if (cmap_i[bmap[cworkB[k]]]) lens_i[j]++;
            }
#endif
          } else { /* allcolumns */
            lens_i[j] = nzA + nzB;
          }
        }
      }
    }

    /* Create row map: global row of C -> local row of submatrices */
    for (i=0; i<ismax; i++) {
      if (!allrows[i]) {
#if defined(PETSC_USE_CTABLE)
        PetscCall(PetscTableCreate(nrow[i],c->Mbs,&rmap[i]));
        irow_i = irow[i];
        jmax   = nrow[i];
        for (j=0; j<jmax; j++) {
          if (allrows[i]) {
            PetscCall(PetscTableAdd(rmap[i],j+1,j+1,INSERT_VALUES));
          } else {
            PetscCall(PetscTableAdd(rmap[i],irow_i[j]+1,j+1,INSERT_VALUES));
          }
        }
#else
        PetscCall(PetscCalloc1(c->Mbs,&rmap[i]));
        rmap_i = rmap[i];
        irow_i = irow[i];
        jmax   = nrow[i];
        for (j=0; j<jmax; j++) {
          if (allrows[i]) rmap_i[j] = j;
          else rmap_i[irow_i[j]] = j;
        }
#endif
      } else rmap[i] = NULL;
    }

    /* Update lens from offproc data */
    {
      PetscInt *rbuf2_i,*rbuf3_i,*sbuf1_i;

      PetscCallMPI(MPI_Waitall(nrqs,r_waits3,MPI_STATUSES_IGNORE));
      for (tmp2=0; tmp2<nrqs; tmp2++) {
        sbuf1_i = sbuf1[pa[tmp2]];
        jmax    = sbuf1_i[0];
        ct1     = 2*jmax+1;
        ct2     = 0;
        rbuf2_i = rbuf2[tmp2];
        rbuf3_i = rbuf3[tmp2];
        for (j=1; j<=jmax; j++) {
          is_no  = sbuf1_i[2*j-1];
          max1   = sbuf1_i[2*j];
          lens_i = lens[is_no];
          if (!allcolumns[is_no]) cmap_i = cmap[is_no];
          rmap_i = rmap[is_no];
          for (k=0; k<max1; k++,ct1++) {
            if (allrows[is_no]) {
              row = sbuf1_i[ct1];
            } else {
#if defined(PETSC_USE_CTABLE)
              PetscCall(PetscTableFind(rmap_i,sbuf1_i[ct1]+1,&row));
              row--;
              PetscCheck(row >= 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"row not found in table");
#else
              row = rmap_i[sbuf1_i[ct1]]; /* the val in the new matrix to be */
#endif
            }
            max2 = rbuf2_i[ct1];
            for (l=0; l<max2; l++,ct2++) {
              if (!allcolumns[is_no]) {
#if defined(PETSC_USE_CTABLE)
                PetscCall(PetscTableFind(cmap_i,rbuf3_i[ct2]+1,&tcol));
#else
                tcol = cmap_i[rbuf3_i[ct2]];
#endif
                if (tcol) lens_i[row]++;
              } else { /* allcolumns */
                lens_i[row]++; /* lens_i[row] += max2 ? */
              }
            }
          }
        }
      }
    }
    PetscCall(PetscFree(r_waits3));
    PetscCallMPI(MPI_Waitall(nrqr,s_waits3,MPI_STATUSES_IGNORE));
    PetscCall(PetscFree(s_waits3));

    /* Create the submatrices */
    for (i=0; i<ismax; i++) {
      PetscInt bs_tmp;
      if (ijonly) bs_tmp = 1;
      else        bs_tmp = bs;

      PetscCall(MatCreate(PETSC_COMM_SELF,submats+i));
      PetscCall(MatSetSizes(submats[i],nrow[i]*bs_tmp,ncol[i]*bs_tmp,PETSC_DETERMINE,PETSC_DETERMINE));

      PetscCall(MatSetType(submats[i],((PetscObject)A)->type_name));
      PetscCall(MatSeqBAIJSetPreallocation(submats[i],bs_tmp,0,lens[i]));
      PetscCall(MatSeqSBAIJSetPreallocation(submats[i],bs_tmp,0,lens[i])); /* this subroutine is used by SBAIJ routines */

      /* create struct Mat_SubSppt and attached it to submat */
      PetscCall(PetscNew(&smat_i));
      subc = (Mat_SeqBAIJ*)submats[i]->data;
      subc->submatis1 = smat_i;

      smat_i->destroy          = submats[i]->ops->destroy;
      submats[i]->ops->destroy = MatDestroySubMatrix_SeqBAIJ;
      submats[i]->factortype   = C->factortype;

      smat_i->id          = i;
      smat_i->nrqs        = nrqs;
      smat_i->nrqr        = nrqr;
      smat_i->rbuf1       = rbuf1;
      smat_i->rbuf2       = rbuf2;
      smat_i->rbuf3       = rbuf3;
      smat_i->sbuf2       = sbuf2;
      smat_i->req_source2 = req_source2;

      smat_i->sbuf1       = sbuf1;
      smat_i->ptr         = ptr;
      smat_i->tmp         = tmp;
      smat_i->ctr         = ctr;

      smat_i->pa           = pa;
      smat_i->req_size     = req_size;
      smat_i->req_source1  = req_source1;

      smat_i->allcolumns  = allcolumns[i];
      smat_i->allrows     = allrows[i];
      smat_i->singleis    = PETSC_FALSE;
      smat_i->row2proc    = row2proc[i];
      smat_i->rmap        = rmap[i];
      smat_i->cmap        = cmap[i];
    }

    if (!ismax) { /* Create dummy submats[0] for reuse struct subc */
      PetscCall(MatCreate(PETSC_COMM_SELF,&submats[0]));
      PetscCall(MatSetSizes(submats[0],0,0,PETSC_DETERMINE,PETSC_DETERMINE));
      PetscCall(MatSetType(submats[0],MATDUMMY));

      /* create struct Mat_SubSppt and attached it to submat */
      PetscCall(PetscNewLog(submats[0],&smat_i));
      submats[0]->data = (void*)smat_i;

      smat_i->destroy          = submats[0]->ops->destroy;
      submats[0]->ops->destroy = MatDestroySubMatrix_Dummy;
      submats[0]->factortype   = C->factortype;

      smat_i->id          = 0;
      smat_i->nrqs        = nrqs;
      smat_i->nrqr        = nrqr;
      smat_i->rbuf1       = rbuf1;
      smat_i->rbuf2       = rbuf2;
      smat_i->rbuf3       = rbuf3;
      smat_i->sbuf2       = sbuf2;
      smat_i->req_source2 = req_source2;

      smat_i->sbuf1       = sbuf1;
      smat_i->ptr         = ptr;
      smat_i->tmp         = tmp;
      smat_i->ctr         = ctr;

      smat_i->pa           = pa;
      smat_i->req_size     = req_size;
      smat_i->req_source1  = req_source1;

      smat_i->allcolumns  = PETSC_FALSE;
      smat_i->singleis    = PETSC_FALSE;
      smat_i->row2proc    = NULL;
      smat_i->rmap        = NULL;
      smat_i->cmap        = NULL;
    }

    if (ismax) PetscCall(PetscFree(lens[0]));
    PetscCall(PetscFree(lens));
    if (sbuf_aj) {
      PetscCall(PetscFree(sbuf_aj[0]));
      PetscCall(PetscFree(sbuf_aj));
    }

  } /* endof scall == MAT_INITIAL_MATRIX */

  /* Post recv matrix values */
  if (!ijonly) {
    PetscCall(PetscObjectGetNewTag((PetscObject)C,&tag4));
    PetscCall(PetscMalloc1(nrqs,&rbuf4));
    PetscCall(PetscMalloc1(nrqs,&r_waits4));
    for (i=0; i<nrqs; ++i) {
      PetscCall(PetscMalloc1(rbuf2[i][0]*bs2,&rbuf4[i]));
      PetscCallMPI(MPI_Irecv(rbuf4[i],rbuf2[i][0]*bs2,MPIU_SCALAR,req_source2[i],tag4,comm,r_waits4+i));
    }

    /* Allocate sending buffers for a->a, and send them off */
    PetscCall(PetscMalloc1(nrqr,&sbuf_aa));
    for (i=0,j=0; i<nrqr; i++) j += req_size[i];

    if (nrqr) PetscCall(PetscMalloc1(j*bs2,&sbuf_aa[0]));
    for (i=1; i<nrqr; i++) sbuf_aa[i] = sbuf_aa[i-1] + req_size[i-1]*bs2;

    PetscCall(PetscMalloc1(nrqr,&s_waits4));

    for (i=0; i<nrqr; i++) {
      rbuf1_i   = rbuf1[i];
      sbuf_aa_i = sbuf_aa[i];
      ct1       = 2*rbuf1_i[0]+1;
      ct2       = 0;
      for (j=1,max1=rbuf1_i[0]; j<=max1; j++) {
        kmax = rbuf1_i[2*j];
        for (k=0; k<kmax; k++,ct1++) {
          row    = rbuf1_i[ct1] - rstart;
          nzA    = a_i[row+1] - a_i[row];
          nzB    = b_i[row+1] - b_i[row];
          ncols  = nzA + nzB;
          cworkB = b_j + b_i[row];
          vworkA = a_a + a_i[row]*bs2;
          vworkB = b_a + b_i[row]*bs2;

          /* load the column values for this row into vals*/
          vals = sbuf_aa_i+ct2*bs2;
          for (l=0; l<nzB; l++) {
            if ((bmap[cworkB[l]]) < cstart) {
              PetscCall(PetscArraycpy(vals+l*bs2,vworkB+l*bs2,bs2));
            } else break;
          }
          imark = l;
          for (l=0; l<nzA; l++) {
            PetscCall(PetscArraycpy(vals+(imark+l)*bs2,vworkA+l*bs2,bs2));
          }
          for (l=imark; l<nzB; l++) {
            PetscCall(PetscArraycpy(vals+(nzA+l)*bs2,vworkB+l*bs2,bs2));
          }

          ct2 += ncols;
        }
      }
      PetscCallMPI(MPI_Isend(sbuf_aa_i,req_size[i]*bs2,MPIU_SCALAR,req_source1[i],tag4,comm,s_waits4+i));
    }
  }

  /* Assemble the matrices */
  /* First assemble the local rows */
  for (i=0; i<ismax; i++) {
    row2proc_i = row2proc[i];
    subc      = (Mat_SeqBAIJ*)submats[i]->data;
    imat_ilen = subc->ilen;
    imat_j    = subc->j;
    imat_i    = subc->i;
    imat_a    = subc->a;

    if (!allcolumns[i]) cmap_i = cmap[i];
    rmap_i = rmap[i];
    irow_i = irow[i];
    jmax   = nrow[i];
    for (j=0; j<jmax; j++) {
      if (allrows[i]) row = j;
      else row  = irow_i[j];
      proc = row2proc_i[j];

      if (proc == rank) {

        row    = row - rstart;
        nzA    = a_i[row+1] - a_i[row];
        nzB    = b_i[row+1] - b_i[row];
        cworkA = a_j + a_i[row];
        cworkB = b_j + b_i[row];
        if (!ijonly) {
          vworkA = a_a + a_i[row]*bs2;
          vworkB = b_a + b_i[row]*bs2;
        }

        if (allrows[i]) {
          row = row+rstart;
        } else {
#if defined(PETSC_USE_CTABLE)
          PetscCall(PetscTableFind(rmap_i,row+rstart+1,&row));
          row--;

          PetscCheck(row >= 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"row not found in table");
#else
          row = rmap_i[row + rstart];
#endif
        }
        mat_i = imat_i[row];
        if (!ijonly) mat_a = imat_a + mat_i*bs2;
        mat_j    = imat_j + mat_i;
        ilen = imat_ilen[row];

        /* load the column indices for this row into cols*/
        if (!allcolumns[i]) {
          for (l=0; l<nzB; l++) {
            if ((ctmp = bmap[cworkB[l]]) < cstart) {
#if defined(PETSC_USE_CTABLE)
              PetscCall(PetscTableFind(cmap_i,ctmp+1,&tcol));
              if (tcol) {
#else
              if ((tcol = cmap_i[ctmp])) {
#endif
                *mat_j++ = tcol - 1;
                PetscCall(PetscArraycpy(mat_a,vworkB+l*bs2,bs2));
                mat_a   += bs2;
                ilen++;
              }
            } else break;
          }
          imark = l;
          for (l=0; l<nzA; l++) {
#if defined(PETSC_USE_CTABLE)
            PetscCall(PetscTableFind(cmap_i,cstart+cworkA[l]+1,&tcol));
            if (tcol) {
#else
            if ((tcol = cmap_i[cstart + cworkA[l]])) {
#endif
              *mat_j++ = tcol - 1;
              if (!ijonly) {
                PetscCall(PetscArraycpy(mat_a,vworkA+l*bs2,bs2));
                mat_a += bs2;
              }
              ilen++;
            }
          }
          for (l=imark; l<nzB; l++) {
#if defined(PETSC_USE_CTABLE)
            PetscCall(PetscTableFind(cmap_i,bmap[cworkB[l]]+1,&tcol));
            if (tcol) {
#else
            if ((tcol = cmap_i[bmap[cworkB[l]]])) {
#endif
              *mat_j++ = tcol - 1;
              if (!ijonly) {
                PetscCall(PetscArraycpy(mat_a,vworkB+l*bs2,bs2));
                mat_a += bs2;
              }
              ilen++;
            }
          }
        } else { /* allcolumns */
          for (l=0; l<nzB; l++) {
            if ((ctmp = bmap[cworkB[l]]) < cstart) {
              *mat_j++ = ctmp;
              PetscCall(PetscArraycpy(mat_a,vworkB+l*bs2,bs2));
              mat_a   += bs2;
              ilen++;
            } else break;
          }
          imark = l;
          for (l=0; l<nzA; l++) {
            *mat_j++ = cstart+cworkA[l];
            if (!ijonly) {
              PetscCall(PetscArraycpy(mat_a,vworkA+l*bs2,bs2));
              mat_a += bs2;
            }
            ilen++;
          }
          for (l=imark; l<nzB; l++) {
            *mat_j++ = bmap[cworkB[l]];
            if (!ijonly) {
              PetscCall(PetscArraycpy(mat_a,vworkB+l*bs2,bs2));
              mat_a += bs2;
            }
            ilen++;
          }
        }
        imat_ilen[row] = ilen;
      }
    }
  }

  /* Now assemble the off proc rows */
  if (!ijonly) {
    PetscCallMPI(MPI_Waitall(nrqs,r_waits4,MPI_STATUSES_IGNORE));
  }
  for (tmp2=0; tmp2<nrqs; tmp2++) {
    sbuf1_i = sbuf1[pa[tmp2]];
    jmax    = sbuf1_i[0];
    ct1     = 2*jmax + 1;
    ct2     = 0;
    rbuf2_i = rbuf2[tmp2];
    rbuf3_i = rbuf3[tmp2];
    if (!ijonly) rbuf4_i = rbuf4[tmp2];
    for (j=1; j<=jmax; j++) {
      is_no     = sbuf1_i[2*j-1];
      rmap_i    = rmap[is_no];
      if (!allcolumns[is_no]) cmap_i = cmap[is_no];
      subc      = (Mat_SeqBAIJ*)submats[is_no]->data;
      imat_ilen = subc->ilen;
      imat_j    = subc->j;
      imat_i    = subc->i;
      if (!ijonly) imat_a    = subc->a;
      max1      = sbuf1_i[2*j];
      for (k=0; k<max1; k++,ct1++) { /* for each recved block row */
        row = sbuf1_i[ct1];

        if (allrows[is_no]) {
          row = sbuf1_i[ct1];
        } else {
#if defined(PETSC_USE_CTABLE)
          PetscCall(PetscTableFind(rmap_i,row+1,&row));
          row--;
          PetscCheck(row >= 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"row not found in table");
#else
          row = rmap_i[row];
#endif
        }
        ilen  = imat_ilen[row];
        mat_i = imat_i[row];
        if (!ijonly) mat_a = imat_a + mat_i*bs2;
        mat_j = imat_j + mat_i;
        max2  = rbuf2_i[ct1];
        if (!allcolumns[is_no]) {
          for (l=0; l<max2; l++,ct2++) {
#if defined(PETSC_USE_CTABLE)
            PetscCall(PetscTableFind(cmap_i,rbuf3_i[ct2]+1,&tcol));
#else
            tcol = cmap_i[rbuf3_i[ct2]];
#endif
            if (tcol) {
              *mat_j++ = tcol - 1;
              if (!ijonly) {
                PetscCall(PetscArraycpy(mat_a,rbuf4_i+ct2*bs2,bs2));
                mat_a += bs2;
              }
              ilen++;
            }
          }
        } else { /* allcolumns */
          for (l=0; l<max2; l++,ct2++) {
            *mat_j++ = rbuf3_i[ct2]; /* same global column index of C */
            if (!ijonly) {
              PetscCall(PetscArraycpy(mat_a,rbuf4_i+ct2*bs2,bs2));
              mat_a += bs2;
            }
            ilen++;
          }
        }
        imat_ilen[row] = ilen;
      }
    }
  }

  if (!iscsorted) { /* sort column indices of the rows */
    MatScalar *work;

    PetscCall(PetscMalloc1(bs2,&work));
    for (i=0; i<ismax; i++) {
      subc      = (Mat_SeqBAIJ*)submats[i]->data;
      imat_ilen = subc->ilen;
      imat_j    = subc->j;
      imat_i    = subc->i;
      if (!ijonly) imat_a = subc->a;
      if (allcolumns[i]) continue;

      jmax = nrow[i];
      for (j=0; j<jmax; j++) {
        mat_i = imat_i[j];
        mat_j = imat_j + mat_i;
        ilen  = imat_ilen[j];
        if (ijonly) {
          PetscCall(PetscSortInt(ilen,mat_j));
        } else {
          mat_a = imat_a + mat_i*bs2;
          PetscCall(PetscSortIntWithDataArray(ilen,mat_j,mat_a,bs2*sizeof(MatScalar),work));
        }
      }
    }
    PetscCall(PetscFree(work));
  }

  if (!ijonly) {
    PetscCall(PetscFree(r_waits4));
    PetscCallMPI(MPI_Waitall(nrqr,s_waits4,MPI_STATUSES_IGNORE));
    PetscCall(PetscFree(s_waits4));
  }

  /* Restore the indices */
  for (i=0; i<ismax; i++) {
    if (!allrows[i]) {
      PetscCall(ISRestoreIndices(isrow[i],irow+i));
    }
    if (!allcolumns[i]) {
      PetscCall(ISRestoreIndices(iscol[i],icol+i));
    }
  }

  for (i=0; i<ismax; i++) {
    PetscCall(MatAssemblyBegin(submats[i],MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(submats[i],MAT_FINAL_ASSEMBLY));
  }

  PetscCall(PetscFree5(*(PetscInt***)&irow,*(PetscInt***)&icol,nrow,ncol,issorted));
  PetscCall(PetscFree5(row2proc,cmap,rmap,allcolumns,allrows));

  if (!ijonly) {
    if (sbuf_aa) {
      PetscCall(PetscFree(sbuf_aa[0]));
      PetscCall(PetscFree(sbuf_aa));
    }

    for (i=0; i<nrqs; ++i) {
      PetscCall(PetscFree(rbuf4[i]));
    }
    PetscCall(PetscFree(rbuf4));
  }
  c->ijonly = PETSC_FALSE; /* set back to the default */
  PetscFunctionReturn(0);
}
