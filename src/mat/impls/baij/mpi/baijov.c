
/*
   Routines to compute overlapping regions of a parallel MPI matrix
  and to find submatrices that were shared across processors.
*/
#include <../src/mat/impls/baij/mpi/mpibaij.h>
#include <petscbt.h>

static PetscErrorCode MatIncreaseOverlap_MPIBAIJ_Local(Mat,PetscInt,char **,PetscInt*,PetscInt**);
static PetscErrorCode MatIncreaseOverlap_MPIBAIJ_Receive(Mat,PetscInt,PetscInt **,PetscInt**,PetscInt*);
extern PetscErrorCode MatGetRow_MPIBAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);
extern PetscErrorCode MatRestoreRow_MPIBAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);

#undef __FUNCT__
#define __FUNCT__ "MatIncreaseOverlap_MPIBAIJ"
PetscErrorCode MatIncreaseOverlap_MPIBAIJ(Mat C,PetscInt imax,IS is[],PetscInt ov)
{
  PetscErrorCode ierr;
  PetscInt       i,N=C->cmap->N, bs=C->rmap->bs;
  IS             *is_new;

  PetscFunctionBegin;
  ierr = PetscMalloc(imax*sizeof(IS),&is_new);CHKERRQ(ierr);
  /* Convert the indices into block format */
  ierr = ISCompressIndicesGeneral(N,C->rmap->n,bs,imax,is,is_new);CHKERRQ(ierr);
  if (ov < 0){ SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap specified\n");}
  for (i=0; i<ov; ++i) {
    ierr = MatIncreaseOverlap_MPIBAIJ_Once(C,imax,is_new);CHKERRQ(ierr);
  }
  for (i=0; i<imax; i++) {ierr = ISDestroy(&is[i]);CHKERRQ(ierr);}
  ierr = ISExpandIndicesGeneral(N,N,bs,imax,is_new,is);CHKERRQ(ierr);
  for (i=0; i<imax; i++) {ierr = ISDestroy(&is_new[i]);CHKERRQ(ierr);}
  ierr = PetscFree(is_new);CHKERRQ(ierr);
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
  nrqr - no of requests recieved (which have to be or which have been processed
*/
#undef __FUNCT__
#define __FUNCT__ "MatIncreaseOverlap_MPIBAIJ_Once"
PetscErrorCode MatIncreaseOverlap_MPIBAIJ_Once(Mat C,PetscInt imax,IS is[])
{
  Mat_MPIBAIJ    *c = (Mat_MPIBAIJ*)C->data;
  const PetscInt **idx,*idx_i;
  PetscInt       *n,*w3,*w4,**data,len;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,tag1,tag2,*w2,*w1,nrqr;
  PetscInt       Mbs,i,j,k,**rbuf,row,proc=-1,nrqs,msz,**outdat,**ptr;
  PetscInt       *ctr,*pa,*tmp,*isz,*isz1,**xdata,**rbuf2,*d_p;
  PetscMPIInt    *onodes1,*olengths1,*onodes2,*olengths2;
  PetscBT        *table;
  MPI_Comm       comm;
  MPI_Request    *s_waits1,*r_waits1,*s_waits2,*r_waits2;
  MPI_Status     *s_status,*recv_status;
  char           *t_p;

  PetscFunctionBegin;
  comm   = ((PetscObject)C)->comm;
  size   = c->size;
  rank   = c->rank;
  Mbs    = c->Mbs;

  ierr = PetscObjectGetNewTag((PetscObject)C,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag2);CHKERRQ(ierr);

  ierr = PetscMalloc2(imax+1,const PetscInt*,&idx,imax,PetscInt,&n);CHKERRQ(ierr);

  for (i=0; i<imax; i++) {
    ierr = ISGetIndices(is[i],&idx[i]);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is[i],&n[i]);CHKERRQ(ierr);
  }

  /* evaluate communication - mesg to who,length of mesg, and buffer space
     required. Based on this, buffers are allocated, and data copied into them*/
  ierr = PetscMalloc4(size,PetscMPIInt,&w1,size,PetscMPIInt,&w2,size,PetscInt,&w3,size,PetscInt,&w4);CHKERRQ(ierr);
  ierr = PetscMemzero(w1,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMemzero(w2,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMemzero(w3,size*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<imax; i++) {
    ierr  = PetscMemzero(w4,size*sizeof(PetscInt));CHKERRQ(ierr); /* initialise work vector*/
    idx_i = idx[i];
    len   = n[i];
    for (j=0; j<len; j++) {
      row  = idx_i[j];
      if (row < 0) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index set cannot have negative entries");
      }
      ierr = PetscLayoutFindOwner(C->rmap,row*C->rmap->bs,&proc);CHKERRQ(ierr);
      w4[proc]++;
    }
    for (j=0; j<size; j++){
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
    ierr = PetscMemzero(outdat[j]+1,2*w3[j]*sizeof(PetscInt));CHKERRQ(ierr);
    ptr[j]       = outdat[j] + 2*w3[j] + 1;
  }

  /* Memory for doing local proc's work*/
  {
    ierr = PetscMalloc5(imax,PetscBT,&table, imax,PetscInt*,&data, imax,PetscInt,&isz,
                        Mbs*imax,PetscInt,&d_p, (Mbs/PETSC_BITS_PER_BYTE+1)*imax,char,&t_p);CHKERRQ(ierr);
    ierr = PetscMemzero(table,imax*sizeof(PetscBT));CHKERRQ(ierr);
    ierr = PetscMemzero(data,imax*sizeof(PetscInt*));CHKERRQ(ierr);
    ierr = PetscMemzero(isz,imax*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemzero(d_p,Mbs*imax*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemzero(t_p,(Mbs/PETSC_BITS_PER_BYTE+1)*imax*sizeof(char));CHKERRQ(ierr);

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
      ierr    = PetscMemzero(ctr,size*sizeof(PetscInt));CHKERRQ(ierr);
      n_i     = n[i];
      table_i = table[i];
      idx_i   = idx[i];
      data_i  = data[i];
      isz_i   = isz[i];
      for (j=0;  j<n_i; j++) {  /* parse the indices of each IS */
        row  = idx_i[j];
        ierr = PetscLayoutFindOwner(C->rmap,row*C->rmap->bs,&proc);CHKERRQ(ierr);
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
  ierr = MatIncreaseOverlap_MPIBAIJ_Local(C,imax,table,isz,data);CHKERRQ(ierr);

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
  ierr = MatIncreaseOverlap_MPIBAIJ_Receive(C,nrqr,rbuf,xdata,isz1);CHKERRQ(ierr);
  ierr = PetscFree(rbuf[0]);CHKERRQ(ierr);
  ierr = PetscFree(rbuf);CHKERRQ(ierr);

  /* Send the data back*/
  /* Do a global reduction to know the buffer space req for incoming messages*/
  {
    PetscMPIInt *rw1;

    ierr = PetscMalloc(size*sizeof(PetscInt),&rw1);CHKERRQ(ierr);
    ierr = PetscMemzero(rw1,size*sizeof(PetscInt));CHKERRQ(ierr);

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
    PetscMPIInt idex;
    PetscInt    is_no,ct1,max,*rbuf2_i,isz_i,*data_i,jmax;
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
#define __FUNCT__ "MatIncreaseOverlap_MPIBAIJ_Local"
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
  Mat         A = c->A,B = c->B;
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

  For better efficiency perhaps we should malloc separately each xdata[i],
then if a remalloc is required we need only copy the data for that one row
rather than all previous rows as it is now where a single large chunck of
memory is used.

*/
static PetscErrorCode MatIncreaseOverlap_MPIBAIJ_Receive(Mat C,PetscInt nrqr,PetscInt **rbuf,PetscInt **xdata,PetscInt * isz1)
{
  Mat_MPIBAIJ    *c = (Mat_MPIBAIJ*)C->data;
  Mat            A = c->A,B = c->B;
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)B->data;
  PetscErrorCode ierr;
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
    rbuf_i  =  rbuf[i];
    rbuf_0  =  rbuf_i[0];
    ct     += rbuf_0;
    for (j=1; j<=rbuf_0; j++) { total_sz += rbuf_i[2*j]; }
  }

  if (c->Mbs) max1 = ct*(a->nz +b->nz)/c->Mbs;
  else        max1 = 1;
  mem_estimate = 3*((total_sz > max1 ? total_sz : max1)+1);
  ierr         = PetscMalloc(mem_estimate*sizeof(PetscInt),&xdata[0]);CHKERRQ(ierr);
  ++no_malloc;
  ierr         = PetscBTCreate(Mbs,&xtable);CHKERRQ(ierr);
  ierr         = PetscMemzero(isz1,nrqr*sizeof(PetscInt));CHKERRQ(ierr);

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
            new_estimate = (PetscInt)(1.5*mem_estimate)+1;
            ierr = PetscMalloc(new_estimate * sizeof(PetscInt),&tmp);CHKERRQ(ierr);
            ierr = PetscMemcpy(tmp,xdata[0],mem_estimate*sizeof(PetscInt));CHKERRQ(ierr);
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
              new_estimate = (PetscInt)(1.5*mem_estimate)+1;
              ierr = PetscMalloc(new_estimate * sizeof(PetscInt),&tmp);CHKERRQ(ierr);
              ierr = PetscMemcpy(tmp,xdata[0],mem_estimate*sizeof(PetscInt));CHKERRQ(ierr);
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
              new_estimate = (PetscInt)(1.5*mem_estimate)+1;
              ierr = PetscMalloc(new_estimate * sizeof(PetscInt),&tmp);CHKERRQ(ierr);
              ierr = PetscMemcpy(tmp,xdata[0],mem_estimate*sizeof(PetscInt));CHKERRQ(ierr);
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
  ierr = PetscBTDestroy(&xtable);CHKERRQ(ierr);
  ierr = PetscInfo3(C,"Allocated %D bytes, required %D, no of mallocs = %D\n",mem_estimate,ct3,no_malloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetSubMatrices_MPIBAIJ"
PetscErrorCode MatGetSubMatrices_MPIBAIJ(Mat C,PetscInt ismax,const IS isrow[],const IS iscol[],MatReuse scall,Mat *submat[])
{
  IS             *isrow_new,*iscol_new;
  Mat_MPIBAIJ    *c = (Mat_MPIBAIJ*)C->data;
  PetscErrorCode ierr;
  PetscInt       nmax,nstages_local,nstages,i,pos,max_no,ncol,nrow,N=C->cmap->N,bs=C->rmap->bs;
  PetscBool      colflag,*allcolumns,*allrows;

  PetscFunctionBegin;
  /* Currently, unsorted column indices will result in inverted column indices in the resulting submatrices. */
  for (i = 0; i < ismax; ++i) {
    PetscBool sorted;
    ierr = ISSorted(iscol[i], &sorted); CHKERRQ(ierr);
    if (!sorted) SETERRQ1(((PetscObject)iscol[i])->comm, PETSC_ERR_SUP, "Column index set %D not sorted", i);
  }
  /* The compression and expansion should be avoided. Doesn't point
     out errors, might change the indices, hence buggey */
  ierr = PetscMalloc2(ismax+1,IS,&isrow_new,ismax+1,IS,&iscol_new);CHKERRQ(ierr);
  ierr = ISCompressIndicesGeneral(N,C->rmap->n,bs,ismax,isrow,isrow_new);CHKERRQ(ierr);
  ierr = ISCompressIndicesGeneral(N,C->cmap->n,bs,ismax,iscol,iscol_new);CHKERRQ(ierr);

  /* Check for special case: each processor gets entire matrix columns */
  ierr = PetscMalloc2(ismax+1,PetscBool,&allcolumns,ismax+1,PetscBool,&allrows);CHKERRQ(ierr);
  for (i=0; i<ismax; i++) {
    ierr = ISIdentity(iscol[i],&colflag);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iscol[i],&ncol);CHKERRQ(ierr);
    if (colflag && ncol == C->cmap->N){
      allcolumns[i] = PETSC_TRUE;
    } else {
      allcolumns[i] = PETSC_FALSE;
    }

    ierr = ISIdentity(isrow[i],&colflag);CHKERRQ(ierr);
    ierr = ISGetLocalSize(isrow[i],&nrow);CHKERRQ(ierr);
    if (colflag && nrow == C->rmap->N){
      allrows[i] = PETSC_TRUE;
    } else {
      allrows[i] = PETSC_FALSE;
    }
  }

  /* Allocate memory to hold all the submatrices */
  if (scall != MAT_REUSE_MATRIX) {
    ierr = PetscMalloc((ismax+1)*sizeof(Mat),submat);CHKERRQ(ierr);
  }
  /* Determine the number of stages through which submatrices are done */
  nmax          = 20*1000000 / (c->Nbs * sizeof(PetscInt));
  if (!nmax) nmax = 1;
  nstages_local = ismax/nmax + ((ismax % nmax)?1:0);

  /* Make sure every processor loops through the nstages */
  ierr = MPI_Allreduce(&nstages_local,&nstages,1,MPIU_INT,MPI_MAX,((PetscObject)C)->comm);CHKERRQ(ierr);
  for (i=0,pos=0; i<nstages; i++) {
    if (pos+nmax <= ismax) max_no = nmax;
    else if (pos == ismax) max_no = 0;
    else                   max_no = ismax-pos;
    ierr = MatGetSubMatrices_MPIBAIJ_local(C,max_no,isrow_new+pos,iscol_new+pos,scall,allrows+pos,allcolumns+pos,*submat+pos);CHKERRQ(ierr);
    pos += max_no;
  }

  for (i=0; i<ismax; i++) {
    ierr = ISDestroy(&isrow_new[i]);CHKERRQ(ierr);
    ierr = ISDestroy(&iscol_new[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(isrow_new,iscol_new);CHKERRQ(ierr);
  ierr = PetscFree2(allcolumns,allrows);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined (PETSC_USE_CTABLE)
#undef __FUNCT__
#define __FUNCT__ "PetscGetProc"
PetscErrorCode PetscGetProc(const PetscInt row, const PetscMPIInt size, const PetscInt proc_gnode[], PetscMPIInt *rank)
{
  PetscInt    nGlobalNd = proc_gnode[size];
  PetscMPIInt fproc = PetscMPIIntCast( (PetscInt)(((float)row * (float)size / (float)nGlobalNd + 0.5)));

  PetscFunctionBegin;
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
#undef __FUNCT__
#define __FUNCT__ "MatGetSubMatrices_MPIBAIJ_local"
PetscErrorCode MatGetSubMatrices_MPIBAIJ_local(Mat C,PetscInt ismax,const IS isrow[],const IS iscol[],MatReuse scall,PetscBool *allrows,PetscBool *allcolumns,Mat *submats)
{
  Mat_MPIBAIJ    *c = (Mat_MPIBAIJ*)C->data;
  Mat            A = c->A;
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)c->B->data,*mat;
  const PetscInt **irow,**icol,*irow_i;
  PetscInt       *nrow,*ncol,*w3,*w4,start;
  PetscErrorCode ierr;
  PetscMPIInt    size,tag0,tag1,tag2,tag3,*w1,*w2,nrqr,idex,end,proc;
  PetscInt       **sbuf1,**sbuf2,rank,i,j,k,l,ct1,ct2,**rbuf1,row;
  PetscInt       nrqs,msz,**ptr,*req_size,*ctr,*pa,*tmp,tcol;
  PetscInt       **rbuf3,*req_source,**sbuf_aj,**rbuf2,max1,max2;
  PetscInt       **lens,is_no,ncols,*cols,mat_i,*mat_j,tmp2,jmax;
  PetscInt       ctr_j,*sbuf1_j,*sbuf_aj_i,*rbuf1_i,kmax,*lens_i;
  PetscInt       bs=C->rmap->bs,bs2=c->bs2,*a_j=a->j,*b_j=b->j,*cworkA,*cworkB;
  PetscInt       cstart = c->cstartbs,nzA,nzB,*a_i=a->i,*b_i=b->i,imark;
  PetscInt       *bmap = c->garray,ctmp,rstart=c->rstartbs;
  MPI_Request    *s_waits1,*r_waits1,*s_waits2,*r_waits2,*r_waits3,*s_waits3;
  MPI_Status     *r_status1,*r_status2,*s_status1,*s_status3,*s_status2,*r_status3;
  MPI_Comm       comm;
  PetscBool      flag;
  PetscMPIInt    *onodes1,*olengths1;
  PetscBool      ijonly=c->ijonly; /* private flag indicates only matrix data structures are requested */
  /* variables below are used for the matrix numerical values - case of !ijonly */
  MPI_Request    *r_waits4,*s_waits4;
  MPI_Status     *r_status4,*s_status4;
  MatScalar      **rbuf4,**sbuf_aa,*vals,*mat_a = PETSC_NULL,*sbuf_aa_i,*vworkA = PETSC_NULL,*vworkB = PETSC_NULL;
  MatScalar      *a_a=a->a,*b_a=b->a;

#if defined (PETSC_USE_CTABLE)
  PetscInt       tt;
  PetscTable     *rmap,*cmap,rmap_i,cmap_i=PETSC_NULL;
#else
  PetscInt       **cmap,*cmap_i=PETSC_NULL,*rtable,*rmap_i,**rmap, Mbs = c->Mbs;
#endif

  PetscFunctionBegin;
  comm   = ((PetscObject)C)->comm;
  tag0   = ((PetscObject)C)->tag;
  size   = c->size;
  rank   = c->rank;

  /* Get some new tags to keep the communication clean */
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag2);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag3);CHKERRQ(ierr);

#if defined(PETSC_USE_CTABLE)
  ierr = PetscMalloc4(ismax,const PetscInt*,&irow,ismax,const PetscInt*,&icol,ismax,PetscInt,&nrow,ismax,PetscInt,&ncol);CHKERRQ(ierr);
#else
  ierr = PetscMalloc5(ismax,const PetscInt*,&irow,ismax,const PetscInt*,&icol,ismax,PetscInt,&nrow,ismax,PetscInt,&ncol,Mbs+1,PetscInt,&rtable);CHKERRQ(ierr);
  /* Create hash table for the mapping :row -> proc*/
  for (i=0,j=0; i<size; i++) {
    jmax = C->rmap->range[i+1]/bs;
    for (; j<jmax; j++) {
      rtable[j] = i;
    }
  }
#endif

  for (i=0; i<ismax; i++) {
    if (allrows[i]){
      irow[i] = PETSC_NULL;
      nrow[i] = C->rmap->N/bs;
    } else {
      ierr = ISGetIndices(isrow[i],&irow[i]);CHKERRQ(ierr);
      ierr = ISGetLocalSize(isrow[i],&nrow[i]);CHKERRQ(ierr);
    }

    if (allcolumns[i]){
      icol[i] = PETSC_NULL;
      ncol[i] = C->cmap->N/bs;
    } else {
      ierr = ISGetIndices(iscol[i],&icol[i]);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iscol[i],&ncol[i]);CHKERRQ(ierr);
    }
  }

  /* evaluate communication - mesg to who,length of mesg,and buffer space
     required. Based on this, buffers are allocated, and data copied into them*/
  ierr = PetscMalloc4(size,PetscMPIInt,&w1,size,PetscMPIInt,&w2,size,PetscInt,&w3,size,PetscInt,&w4);CHKERRQ(ierr);
  ierr = PetscMemzero(w1,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMemzero(w2,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMemzero(w3,size*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<ismax; i++) {
    ierr   = PetscMemzero(w4,size*sizeof(PetscInt));CHKERRQ(ierr); /* initialise work vector*/
    jmax   = nrow[i];
    irow_i = irow[i];
    for (j=0; j<jmax; j++) {
      if (allrows[i]) {
        row = j;
      } else {
        row  = irow_i[j];
      }
#if defined (PETSC_USE_CTABLE)
      ierr = PetscGetProc(row,size,c->rangebs,&proc);CHKERRQ(ierr);
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
  /* Initialise the header space */
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
      if (allrows[i]){
        row = j;
      } else {
        row  = irow_i[j];
      }
#if defined (PETSC_USE_CTABLE)
      ierr = PetscGetProc(row,size,c->rangebs,&proc);CHKERRQ(ierr);
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
    ierr = MPI_Isend(sbuf1[j],w1[j],MPIU_INT,j,tag0,comm,s_waits1+i);CHKERRQ(ierr);
  }

  /* Post Recieves to capture the buffer size */
  ierr     = PetscMalloc((nrqs+1)*sizeof(MPI_Request),&r_waits2);CHKERRQ(ierr);
  ierr     = PetscMalloc((nrqs+1)*sizeof(PetscInt*),&rbuf2);CHKERRQ(ierr);
  rbuf2[0] = tmp + msz;
  for (i=1; i<nrqs; ++i) {
    j        = pa[i];
    rbuf2[i] = rbuf2[i-1]+w1[pa[i-1]];
  }
  for (i=0; i<nrqs; ++i) {
    j    = pa[i];
    ierr = MPI_Irecv(rbuf2[i],w1[j],MPIU_INT,j,tag1,comm,r_waits2+i);CHKERRQ(ierr);
  }

  /* Send to other procs the buf size they should allocate */

  /* Receive messages*/
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Request),&s_waits2);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Status),&r_status1);CHKERRQ(ierr);
  ierr = PetscMalloc3(nrqr+1,PetscInt*,&sbuf2,nrqr,PetscInt,&req_size,nrqr,PetscInt,&req_source);CHKERRQ(ierr);
  {
    Mat_SeqBAIJ *sA = (Mat_SeqBAIJ*)c->A->data,*sB = (Mat_SeqBAIJ*)c->B->data;
    PetscInt    *sAi = sA->i,*sBi = sB->i,id,*sbuf2_i;

    for (i=0; i<nrqr; ++i) {
      ierr = MPI_Waitany(nrqr,r_waits1,&idex,r_status1+i);CHKERRQ(ierr);
      req_size[idex] = 0;
      rbuf1_i         = rbuf1[idex];
      start           = 2*rbuf1_i[0] + 1;
      ierr            = MPI_Get_count(r_status1+i,MPIU_INT,&end);CHKERRQ(ierr);
      ierr            = PetscMalloc(end*sizeof(PetscInt),&sbuf2[idex]);CHKERRQ(ierr);
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
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Request),&r_waits3);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Status),&r_status2);CHKERRQ(ierr);
  if (!ijonly){
    ierr = PetscMalloc((nrqs+1)*sizeof(MatScalar*),&rbuf4);CHKERRQ(ierr);
    ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Request),&r_waits4);CHKERRQ(ierr);
  }

  for (i=0; i<nrqs; ++i) {
    ierr = MPI_Waitany(nrqs,r_waits2,&idex,r_status2+i);CHKERRQ(ierr);
    ierr = PetscMalloc(rbuf2[idex][0]*sizeof(PetscInt),&rbuf3[idex]);CHKERRQ(ierr);
    ierr = MPI_Irecv(rbuf3[idex],rbuf2[idex][0],MPIU_INT,r_status2[i].MPI_SOURCE,tag2,comm,r_waits3+idex);CHKERRQ(ierr);
    if (!ijonly){
      ierr = PetscMalloc(rbuf2[idex][0]*bs2*sizeof(MatScalar),&rbuf4[idex]);CHKERRQ(ierr);
      ierr = MPI_Irecv(rbuf4[idex],rbuf2[idex][0]*bs2,MPIU_MATSCALAR,r_status2[i].MPI_SOURCE,tag3,comm,r_waits4+idex);CHKERRQ(ierr);
    }
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
      ierr = MPI_Isend(sbuf_aj_i,req_size[i],MPIU_INT,req_source[i],tag2,comm,s_waits3+i);CHKERRQ(ierr);
    }
  }
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Status),&r_status3);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Status),&s_status3);CHKERRQ(ierr);

  /* Allocate buffers for a->a, and send them off */
  if (!ijonly){
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
  }
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
        ierr = PetscTableCreate(ncol[i]+1,c->Nbs+1,&cmap[i]);CHKERRQ(ierr);
        jmax   = ncol[i];
        icol_i = icol[i];
        cmap_i = cmap[i];
        for (j=0; j<jmax; j++) {
          ierr = PetscTableAdd(cmap_i,icol_i[j]+1,j+1,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        cmap[i] = PETSC_NULL;
      }
    }
#else
    ierr    = PetscMalloc(ismax*sizeof(PetscInt*),&cmap);CHKERRQ(ierr);
    for (i=0; i<ismax; i++) {
      if (!allcolumns[i]){
        ierr = PetscMalloc(c->Nbs*sizeof(PetscInt),&cmap[i]);CHKERRQ(ierr);
        ierr = PetscMemzero(cmap[i],c->Nbs*sizeof(PetscInt));CHKERRQ(ierr);
        jmax   = ncol[i];
        icol_i = icol[i];
        cmap_i = cmap[i];
        for (j=0; j<jmax; j++) {
          cmap_i[icol_i[j]] = j+1;
        }
      } else { /* allcolumns[i] */
        cmap[i] = PETSC_NULL;
      }
    }
#endif
  }

  /* Create lens which is required for MatCreate... */
  for (i=0,j=0; i<ismax; i++) { j += nrow[i]; }
  ierr    = PetscMalloc((1+ismax)*sizeof(PetscInt*)+ j*sizeof(PetscInt),&lens);CHKERRQ(ierr);
  lens[0] = (PetscInt*)(lens + ismax);
  ierr    = PetscMemzero(lens[0],j*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=1; i<ismax; i++) { lens[i] = lens[i-1] + nrow[i-1]; }

  /* Update lens from local data */
  for (i=0; i<ismax; i++) {
    jmax   = nrow[i];
    if (!allcolumns[i]) cmap_i = cmap[i];
    irow_i = irow[i];
    lens_i = lens[i];
    for (j=0; j<jmax; j++) {
      if (allrows[i]){
        row = j;
      } else {
        row  = irow_i[j];
      }
#if defined (PETSC_USE_CTABLE)
      ierr = PetscGetProc(row,size,c->rangebs,&proc);CHKERRQ(ierr);
#else
      proc = rtable[row];
#endif
      if (proc == rank) {
        /* Get indices from matA and then from matB */
        row    = row - rstart;
        nzA    = a_i[row+1] - a_i[row];     nzB = b_i[row+1] - b_i[row];
        cworkA =  a_j + a_i[row]; cworkB = b_j + b_i[row];
        if (!allcolumns[i]){
#if defined (PETSC_USE_CTABLE)
          for (k=0; k<nzA; k++) {
            ierr = PetscTableFind(cmap_i,cstart+cworkA[k]+1,&tt);CHKERRQ(ierr);
            if (tt) { lens_i[j]++; }
          }
          for (k=0; k<nzB; k++) {
            ierr = PetscTableFind(cmap_i,bmap[cworkB[k]]+1,&tt);CHKERRQ(ierr);
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
        } else {/* allcolumns */
          lens_i[j] = nzA + nzB;
        }
      }
    }
  }
#if defined (PETSC_USE_CTABLE)
  /* Create row map*/
  ierr = PetscMalloc((1+ismax)*sizeof(PetscTable),&rmap);CHKERRQ(ierr);
  for (i=0; i<ismax; i++){
    ierr = PetscTableCreate(nrow[i]+1,c->Mbs+1,&rmap[i]);CHKERRQ(ierr);
  }
#else
  /* Create row map*/
  ierr    = PetscMalloc((1+ismax)*sizeof(PetscInt*)+ ismax*Mbs*sizeof(PetscInt),&rmap);CHKERRQ(ierr);
  rmap[0] = (PetscInt*)(rmap + ismax);
  ierr    = PetscMemzero(rmap[0],ismax*Mbs*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=1; i<ismax; i++) { rmap[i] = rmap[i-1] + Mbs;}
#endif
  for (i=0; i<ismax; i++) {
    irow_i = irow[i];
    jmax   = nrow[i];
#if defined (PETSC_USE_CTABLE)
    rmap_i = rmap[i];
    for (j=0; j<jmax; j++) {
      if (allrows[i]){
        ierr = PetscTableAdd(rmap_i,j+1,j+1,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        ierr = PetscTableAdd(rmap_i,irow_i[j]+1,j+1,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
#else
    rmap_i = rmap[i];
    for (j=0; j<jmax; j++) {
      if (allrows[i]){
        rmap_i[j] = j;
      } else {
        rmap_i[irow_i[j]] = j;
      }
    }
#endif
  }

  /* Update lens from offproc data */
  {
    PetscInt    *rbuf2_i,*rbuf3_i,*sbuf1_i;
    PetscMPIInt ii;

    for (tmp2=0; tmp2<nrqs; tmp2++) {
      ierr    = MPI_Waitany(nrqs,r_waits3,&ii,r_status3+tmp2);CHKERRQ(ierr);
      idex   = pa[ii];
      sbuf1_i = sbuf1[idex];
      jmax    = sbuf1_i[0];
      ct1     = 2*jmax+1;
      ct2     = 0;
      rbuf2_i = rbuf2[ii];
      rbuf3_i = rbuf3[ii];
      for (j=1; j<=jmax; j++) {
        is_no   = sbuf1_i[2*j-1];
        max1    = sbuf1_i[2*j];
        lens_i  = lens[is_no];
        if (!allcolumns[is_no]) cmap_i = cmap[is_no];
	rmap_i = rmap[is_no];
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
              ierr = PetscTableFind(cmap_i,rbuf3_i[ct2]+1,&tt);CHKERRQ(ierr);
              if (tt) {
                lens_i[row]++;
              }
#else
              if (cmap_i[rbuf3_i[ct2]]) {
                lens_i[row]++;
              }
#endif
            } else { /* allcolumns */
              lens_i[row]++;
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
    if (ijonly) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP," MAT_REUSE_MATRIX and ijonly is not supported yet");
    /*
        Assumes new rows are same length as the old rows, hence bug!
    */
    for (i=0; i<ismax; i++) {
      mat = (Mat_SeqBAIJ *)(submats[i]->data);
      if ((mat->mbs != nrow[i]) || (mat->nbs != ncol[i] || C->rmap->bs != bs)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong size");
      ierr = PetscMemcmp(mat->ilen,lens[i],mat->mbs *sizeof(PetscInt),&flag);CHKERRQ(ierr);
      if (!flag) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cannot reuse matrix. wrong no of nonzeros");
      /* Initial matrix as if empty */
      ierr = PetscMemzero(mat->ilen,mat->mbs*sizeof(PetscInt));CHKERRQ(ierr);
      submats[i]->factortype = C->factortype;
    }
  } else {
    PetscInt bs_tmp;
    if (ijonly){
      bs_tmp = 1;
    } else {
      bs_tmp = bs;
    }
    for (i=0; i<ismax; i++) {
      ierr = MatCreate(PETSC_COMM_SELF,submats+i);CHKERRQ(ierr);
      ierr = MatSetSizes(submats[i],nrow[i]*bs_tmp,ncol[i]*bs_tmp,nrow[i]*bs_tmp,ncol[i]*bs_tmp);CHKERRQ(ierr);
      ierr = MatSetType(submats[i],((PetscObject)A)->type_name);CHKERRQ(ierr);
      ierr = MatSeqBAIJSetPreallocation(submats[i],bs_tmp,0,lens[i]);CHKERRQ(ierr);
      ierr = MatSeqSBAIJSetPreallocation(submats[i],bs_tmp,0,lens[i]);CHKERRQ(ierr); /* this subroutine is used by SBAIJ routines */
    }
  }

  /* Assemble the matrices */
  /* First assemble the local rows */
  {
    PetscInt  ilen_row,*imat_ilen,*imat_j,*imat_i;
    MatScalar *imat_a = PETSC_NULL;

    for (i=0; i<ismax; i++) {
      mat       = (Mat_SeqBAIJ*)submats[i]->data;
      imat_ilen = mat->ilen;
      imat_j    = mat->j;
      imat_i    = mat->i;
      if (!ijonly) imat_a = mat->a;
      if (!allcolumns[i]) cmap_i = cmap[i];
      rmap_i = rmap[i];
      irow_i    = irow[i];
      jmax      = nrow[i];
      for (j=0; j<jmax; j++) {
        if (allrows[i]){
          row = j;
        } else {
          row      = irow_i[j];
        }
#if defined (PETSC_USE_CTABLE)
	ierr = PetscGetProc(row,size,c->rangebs,&proc);CHKERRQ(ierr);
#else
	proc = rtable[row];
#endif
        if (proc == rank) {
          row      = row - rstart;
          nzA      = a_i[row+1] - a_i[row];
          nzB      = b_i[row+1] - b_i[row];
          cworkA   = a_j + a_i[row];
          cworkB   = b_j + b_i[row];
          if (!ijonly){
            vworkA = a_a + a_i[row]*bs2;
            vworkB = b_a + b_i[row]*bs2;
          }
#if defined (PETSC_USE_CTABLE)
	  ierr = PetscTableFind(rmap_i,row+rstart+1,&row);CHKERRQ(ierr);
          row--;
          if (row < 0) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"row not found in table"); }
#else
          row      = rmap_i[row + rstart];
#endif
          mat_i    = imat_i[row];
          if (!ijonly) mat_a = imat_a + mat_i*bs2;
          mat_j    = imat_j + mat_i;
          ilen_row = imat_ilen[row];

          /* load the column indices for this row into cols*/
          if (!allcolumns[i]){
            for (l=0; l<nzB; l++) {
              if ((ctmp = bmap[cworkB[l]]) < cstart) {
#if defined (PETSC_USE_CTABLE)
                ierr = PetscTableFind(cmap_i,ctmp+1,&tcol);CHKERRQ(ierr);
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
              ierr = PetscTableFind(cmap_i,cstart+cworkA[l]+1,&tcol);CHKERRQ(ierr);
              if (tcol) {
#else
              if ((tcol = cmap_i[cstart + cworkA[l]])) {
#endif
                *mat_j++ = tcol - 1;
                if (!ijonly){
                  ierr   = PetscMemcpy(mat_a,vworkA+l*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
                  mat_a += bs2;
                }
                ilen_row++;
              }
            }
            for (l=imark; l<nzB; l++) {
#if defined (PETSC_USE_CTABLE)
              ierr = PetscTableFind(cmap_i,bmap[cworkB[l]]+1,&tcol);CHKERRQ(ierr);
              if (tcol) {
#else
              if ((tcol = cmap_i[bmap[cworkB[l]]])) {
#endif
                *mat_j++ = tcol - 1;
                if (!ijonly){
                  ierr     = PetscMemcpy(mat_a,vworkB+l*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
                  mat_a   += bs2;
                }
                ilen_row++;
              }
            }
          } else { /* allcolumns */
            for (l=0; l<nzB; l++) {
              if ((ctmp = bmap[cworkB[l]]) < cstart) {
                *mat_j++ = ctmp;
                ierr     = PetscMemcpy(mat_a,vworkB+l*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
                mat_a   += bs2;
                ilen_row++;
              } else break;
            }
            imark = l;
            for (l=0; l<nzA; l++) {
              *mat_j++ = cstart+cworkA[l];
              if (!ijonly){
                ierr   = PetscMemcpy(mat_a,vworkA+l*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
                mat_a += bs2;
              }
              ilen_row++;
            }
            for (l=imark; l<nzB; l++) {
              *mat_j++ = bmap[cworkB[l]];
              if (!ijonly){
                ierr     = PetscMemcpy(mat_a,vworkB+l*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
                mat_a   += bs2;
              }
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
    PetscInt    *sbuf1_i,*rbuf2_i,*rbuf3_i,*imat_ilen,ilen;
    PetscInt    *imat_j,*imat_i;
    MatScalar   *imat_a = PETSC_NULL,*rbuf4_i = PETSC_NULL;
    PetscMPIInt ii;

    for (tmp2=0; tmp2<nrqs; tmp2++) {
      if (ijonly){
        ii = tmp2;
      } else {
        ierr = MPI_Waitany(nrqs,r_waits4,&ii,r_status4+tmp2);CHKERRQ(ierr);
      }
      idex   = pa[ii];
      sbuf1_i = sbuf1[idex];
      jmax    = sbuf1_i[0];
      ct1     = 2*jmax + 1;
      ct2     = 0;
      rbuf2_i = rbuf2[ii];
      rbuf3_i = rbuf3[ii];
      if (!ijonly) rbuf4_i = rbuf4[ii];
      for (j=1; j<=jmax; j++) {
        is_no     = sbuf1_i[2*j-1];
        if (!allcolumns[is_no]) cmap_i = cmap[is_no];
	rmap_i    = rmap[is_no];
        mat       = (Mat_SeqBAIJ*)submats[is_no]->data;
        imat_ilen = mat->ilen;
        imat_j    = mat->j;
        imat_i    = mat->i;
        if (!ijonly) imat_a = mat->a;
        max1      = sbuf1_i[2*j];
        for (k=0; k<max1; k++,ct1++) {
          row   = sbuf1_i[ct1];
#if defined (PETSC_USE_CTABLE)
	  ierr = PetscTableFind(rmap_i,row+1,&row);CHKERRQ(ierr);
          row--;
          if (row < 0) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"row not found in table"); }
#else
          row   = rmap_i[row];
#endif
          ilen  = imat_ilen[row];
          mat_i = imat_i[row];
          if (!ijonly) mat_a = imat_a + mat_i*bs2;
          mat_j = imat_j + mat_i;
          max2 = rbuf2_i[ct1];

          if (!allcolumns[is_no]){
            for (l=0; l<max2; l++,ct2++) {
#if defined (PETSC_USE_CTABLE)
              ierr = PetscTableFind(cmap_i,rbuf3_i[ct2]+1,&tcol);CHKERRQ(ierr);
              if (tcol) {
#else
              if ((tcol = cmap_i[rbuf3_i[ct2]])) {
#endif
                *mat_j++    = tcol - 1;
                if (!ijonly){
                  ierr   = PetscMemcpy(mat_a,rbuf4_i+ct2*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
                  mat_a += bs2;
                }
                ilen++;
              }
            }
          } else { /* allcolumns */
            for (l=0; l<max2; l++,ct2++) {
              *mat_j++    = rbuf3_i[ct2];
              if (!ijonly){
                ierr   = PetscMemcpy(mat_a,rbuf4_i+ct2*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
                mat_a += bs2;
              }
              ilen++;
            }
          }
          imat_ilen[row] = ilen;
        }
      }
    }
  }
  if (!ijonly){
    ierr = PetscFree(r_status4);CHKERRQ(ierr);
    ierr = PetscFree(r_waits4);CHKERRQ(ierr);
    if (nrqr) {ierr = MPI_Waitall(nrqr,s_waits4,s_status4);CHKERRQ(ierr);}
    ierr = PetscFree(s_waits4);CHKERRQ(ierr);
    ierr = PetscFree(s_status4);CHKERRQ(ierr);
  }

  /* Restore the indices */
  for (i=0; i<ismax; i++) {
    if (!allrows[i]){
      ierr = ISRestoreIndices(isrow[i],irow+i);CHKERRQ(ierr);
    }
    if (!allcolumns[i]){
      ierr = ISRestoreIndices(iscol[i],icol+i);CHKERRQ(ierr);
    }
  }

  /* Destroy allocated memory */
#if defined(PETSC_USE_CTABLE)
  ierr = PetscFree4(irow,icol,nrow,ncol);CHKERRQ(ierr);
#else
  ierr = PetscFree5(irow,icol,nrow,ncol,rtable);CHKERRQ(ierr);
#endif
  ierr = PetscFree4(w1,w2,w3,w4);CHKERRQ(ierr);
  ierr = PetscFree(pa);CHKERRQ(ierr);

  ierr = PetscFree4(sbuf1,ptr,tmp,ctr);CHKERRQ(ierr);
  ierr = PetscFree(sbuf1);CHKERRQ(ierr);
  ierr = PetscFree(rbuf2);CHKERRQ(ierr);
  for (i=0; i<nrqr; ++i) {
    ierr = PetscFree(sbuf2[i]);CHKERRQ(ierr);
  }
  for (i=0; i<nrqs; ++i) {
    ierr = PetscFree(rbuf3[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree3(sbuf2,req_size,req_source);CHKERRQ(ierr);
  ierr = PetscFree(rbuf3);CHKERRQ(ierr);
  ierr = PetscFree(sbuf_aj[0]);CHKERRQ(ierr);
  ierr = PetscFree(sbuf_aj);CHKERRQ(ierr);
  if (!ijonly) {
    for (i=0; i<nrqs; ++i) {ierr = PetscFree(rbuf4[i]);CHKERRQ(ierr);}
    ierr = PetscFree(rbuf4);CHKERRQ(ierr);
    ierr = PetscFree(sbuf_aa[0]);CHKERRQ(ierr);
    ierr = PetscFree(sbuf_aa);CHKERRQ(ierr);
  }

#if defined (PETSC_USE_CTABLE)
  for (i=0; i<ismax; i++) {
    ierr = PetscTableDestroy((PetscTable*)&rmap[i]);CHKERRQ(ierr);
  }
#endif
  ierr = PetscFree(rmap);CHKERRQ(ierr);

  for (i=0; i<ismax; i++){
    if (!allcolumns[i]){
#if defined (PETSC_USE_CTABLE)
      ierr = PetscTableDestroy((PetscTable*)&cmap[i]);CHKERRQ(ierr);
#else
      ierr = PetscFree(cmap[i]);CHKERRQ(ierr);
#endif
    }
  }
  ierr = PetscFree(cmap);CHKERRQ(ierr);
  ierr = PetscFree(lens);CHKERRQ(ierr);

  for (i=0; i<ismax; i++) {
    ierr = MatAssemblyBegin(submats[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(submats[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  c->ijonly = PETSC_FALSE; /* set back to the default */
  PetscFunctionReturn(0);
}

