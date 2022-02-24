
/*
   Routines to compute overlapping regions of a parallel MPI matrix.
   Used for finding submatrices that were shared across processors.
*/
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h>
#include <petscbt.h>

static PetscErrorCode MatIncreaseOverlap_MPISBAIJ_Once(Mat,PetscInt,IS*);
static PetscErrorCode MatIncreaseOverlap_MPISBAIJ_Local(Mat,PetscInt*,PetscInt,PetscInt*,PetscBT*);

PetscErrorCode MatIncreaseOverlap_MPISBAIJ(Mat C,PetscInt is_max,IS is[],PetscInt ov)
{
  PetscInt       i,N=C->cmap->N, bs=C->rmap->bs,M=C->rmap->N,Mbs=M/bs,*nidx,isz,iov;
  IS             *is_new,*is_row;
  Mat            *submats;
  Mat_MPISBAIJ   *c=(Mat_MPISBAIJ*)C->data;
  Mat_SeqSBAIJ   *asub_i;
  PetscBT        table;
  PetscInt       *ai,brow,nz,nis,l,nmax,nstages_local,nstages,max_no,pos;
  const PetscInt *idx;
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc1(is_max,&is_new));
  /* Convert the indices into block format */
  CHKERRQ(ISCompressIndicesGeneral(N,C->rmap->n,bs,is_max,is,is_new));
  PetscCheckFalse(ov < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap specified");

  /* ----- previous non-scalable implementation ----- */
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-IncreaseOverlap_old", &flg));
  if (flg) { /* previous non-scalable implementation */
    printf("use previous non-scalable implementation...\n");
    for (i=0; i<ov; ++i) {
      CHKERRQ(MatIncreaseOverlap_MPISBAIJ_Once(C,is_max,is_new));
    }
  } else { /* implementation using modified BAIJ routines */

    CHKERRQ(PetscMalloc1(Mbs+1,&nidx));
    CHKERRQ(PetscBTCreate(Mbs,&table)); /* for column search */

    /* Create is_row */
    CHKERRQ(PetscMalloc1(is_max,&is_row));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,Mbs,0,1,&is_row[0]));

    for (i=1; i<is_max; i++) {
      is_row[i]  = is_row[0]; /* reuse is_row[0] */
    }

    /* Allocate memory to hold all the submatrices - Modified from MatCreateSubMatrices_MPIBAIJ() */
    CHKERRQ(PetscMalloc1(is_max+1,&submats));

    /* Determine the number of stages through which submatrices are done */
    nmax = 20*1000000 / (c->Nbs * sizeof(PetscInt));
    if (!nmax) nmax = 1;
    nstages_local = is_max/nmax + ((is_max % nmax) ? 1 : 0);

    /* Make sure every processor loops through the nstages */
    CHKERRMPI(MPIU_Allreduce(&nstages_local,&nstages,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)C)));

    for (iov=0; iov<ov; ++iov) {
      /* 1) Get submats for column search */
      for (i=0,pos=0; i<nstages; i++) {
        if (pos+nmax <= is_max) max_no = nmax;
        else if (pos == is_max) max_no = 0;
        else                    max_no = is_max-pos;
        c->ijonly = PETSC_TRUE; /* only matrix data structures are requested */
        /* The resulting submatrices should be BAIJ, not SBAIJ, hence we change this value to trigger that */
        CHKERRQ(PetscStrcpy(((PetscObject)c->A)->type_name,MATSEQBAIJ));
        CHKERRQ(MatCreateSubMatrices_MPIBAIJ_local(C,max_no,is_row+pos,is_new+pos,MAT_INITIAL_MATRIX,submats+pos));
        CHKERRQ(PetscStrcpy(((PetscObject)c->A)->type_name,MATSEQSBAIJ));
        pos      += max_no;
      }

      /* 2) Row search */
      CHKERRQ(MatIncreaseOverlap_MPIBAIJ_Once(C,is_max,is_new));

      /* 3) Column search */
      for (i=0; i<is_max; i++) {
        asub_i = (Mat_SeqSBAIJ*)submats[i]->data;
        ai     = asub_i->i;

        /* put is_new obtained from MatIncreaseOverlap_MPIBAIJ() to table */
        CHKERRQ(PetscBTMemzero(Mbs,table));

        CHKERRQ(ISGetIndices(is_new[i],&idx));
        CHKERRQ(ISGetLocalSize(is_new[i],&nis));
        for (l=0; l<nis; l++) {
          CHKERRQ(PetscBTSet(table,idx[l]));
          nidx[l] = idx[l];
        }
        isz = nis;

        /* add column entries to table */
        for (brow=0; brow<Mbs; brow++) {
          nz = ai[brow+1] - ai[brow];
          if (nz) {
            if (!PetscBTLookupSet(table,brow)) nidx[isz++] = brow;
          }
        }
        CHKERRQ(ISRestoreIndices(is_new[i],&idx));
        CHKERRQ(ISDestroy(&is_new[i]));

        /* create updated is_new */
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,isz,nidx,PETSC_COPY_VALUES,is_new+i));
      }

      /* Free tmp spaces */
      for (i=0; i<is_max; i++) {
        CHKERRQ(MatDestroy(&submats[i]));
      }
    }

    CHKERRQ(PetscBTDestroy(&table));
    CHKERRQ(PetscFree(submats));
    CHKERRQ(ISDestroy(&is_row[0]));
    CHKERRQ(PetscFree(is_row));
    CHKERRQ(PetscFree(nidx));

  }

  for (i=0; i<is_max; i++) CHKERRQ(ISDestroy(&is[i]));
  CHKERRQ(ISExpandIndicesGeneral(N,N,bs,is_max,is_new,is));

  for (i=0; i<is_max; i++) CHKERRQ(ISDestroy(&is_new[i]));
  CHKERRQ(PetscFree(is_new));
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
static PetscErrorCode MatIncreaseOverlap_MPISBAIJ_Once(Mat C,PetscInt is_max,IS is[])
{
  Mat_MPISBAIJ   *c = (Mat_MPISBAIJ*)C->data;
  PetscMPIInt    size,rank,tag1,tag2,*len_s,nrqr,nrqs,*id_r1,*len_r1,flag,len,*iwork;
  const PetscInt *idx_i;
  PetscInt       idx,isz,col,*n,*data1,**data1_start,*data2,*data2_i,*data,*data_i;
  PetscInt       Mbs,i,j,k,*odata1,*odata2;
  PetscInt       proc_id,**odata2_ptr,*ctable=NULL,*btable,len_max,len_est;
  PetscInt       proc_end=0,len_unused,nodata2;
  PetscInt       ois_max; /* max no of is[] in each of processor */
  char           *t_p;
  MPI_Comm       comm;
  MPI_Request    *s_waits1,*s_waits2,r_req;
  MPI_Status     *s_status,r_status;
  PetscBT        *table;  /* mark indices of this processor's is[] */
  PetscBT        table_i;
  PetscBT        otable; /* mark indices of other processors' is[] */
  PetscInt       bs=C->rmap->bs,Bn = c->B->cmap->n,Bnbs = Bn/bs,*Bowners;
  IS             garray_local,garray_gl;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)C,&comm));
  size = c->size;
  rank = c->rank;
  Mbs  = c->Mbs;

  CHKERRQ(PetscObjectGetNewTag((PetscObject)C,&tag1));
  CHKERRQ(PetscObjectGetNewTag((PetscObject)C,&tag2));

  /* create tables used in
     step 1: table[i] - mark c->garray of proc [i]
     step 3: table[i] - mark indices of is[i] when whose=MINE
             table[0] - mark incideces of is[] when whose=OTHER */
  len  = PetscMax(is_max, size);
  CHKERRQ(PetscMalloc2(len,&table,(Mbs/PETSC_BITS_PER_BYTE+1)*len,&t_p));
  for (i=0; i<len; i++) {
    table[i] = t_p  + (Mbs/PETSC_BITS_PER_BYTE+1)*i;
  }

  CHKERRMPI(MPIU_Allreduce(&is_max,&ois_max,1,MPIU_INT,MPI_MAX,comm));

  /* 1. Send this processor's is[] to other processors */
  /*---------------------------------------------------*/
  /* allocate spaces */
  CHKERRQ(PetscMalloc1(is_max,&n));
  len  = 0;
  for (i=0; i<is_max; i++) {
    CHKERRQ(ISGetLocalSize(is[i],&n[i]));
    len += n[i];
  }
  if (!len) {
    is_max = 0;
  } else {
    len += 1 + is_max; /* max length of data1 for one processor */
  }

  CHKERRQ(PetscMalloc1(size*len+1,&data1));
  CHKERRQ(PetscMalloc1(size,&data1_start));
  for (i=0; i<size; i++) data1_start[i] = data1 + i*len;

  CHKERRQ(PetscMalloc4(size,&len_s,size,&btable,size,&iwork,size+1,&Bowners));

  /* gather c->garray from all processors */
  CHKERRQ(ISCreateGeneral(comm,Bnbs,c->garray,PETSC_COPY_VALUES,&garray_local));
  CHKERRQ(ISAllGather(garray_local, &garray_gl));
  CHKERRQ(ISDestroy(&garray_local));
  CHKERRMPI(MPI_Allgather(&Bnbs,1,MPIU_INT,Bowners+1,1,MPIU_INT,comm));

  Bowners[0] = 0;
  for (i=0; i<size; i++) Bowners[i+1] += Bowners[i];

  if (is_max) {
    /* hash table ctable which maps c->row to proc_id) */
    CHKERRQ(PetscMalloc1(Mbs,&ctable));
    for (proc_id=0,j=0; proc_id<size; proc_id++) {
      for (; j<C->rmap->range[proc_id+1]/bs; j++) ctable[j] = proc_id;
    }

    /* hash tables marking c->garray */
    CHKERRQ(ISGetIndices(garray_gl,&idx_i));
    for (i=0; i<size; i++) {
      table_i = table[i];
      CHKERRQ(PetscBTMemzero(Mbs,table_i));
      for (j = Bowners[i]; j<Bowners[i+1]; j++) { /* go through B cols of proc[i]*/
        CHKERRQ(PetscBTSet(table_i,idx_i[j]));
      }
    }
    CHKERRQ(ISRestoreIndices(garray_gl,&idx_i));
  }  /* if (is_max) */
  CHKERRQ(ISDestroy(&garray_gl));

  /* evaluate communication - mesg to who, length, and buffer space */
  for (i=0; i<size; i++) len_s[i] = 0;

  /* header of data1 */
  for (proc_id=0; proc_id<size; proc_id++) {
    iwork[proc_id]        = 0;
    *data1_start[proc_id] = is_max;
    data1_start[proc_id]++;
    for (j=0; j<is_max; j++) {
      if (proc_id == rank) {
        *data1_start[proc_id] = n[j];
      } else {
        *data1_start[proc_id] = 0;
      }
      data1_start[proc_id]++;
    }
  }

  for (i=0; i<is_max; i++) {
    CHKERRQ(ISGetIndices(is[i],&idx_i));
    for (j=0; j<n[i]; j++) {
      idx                = idx_i[j];
      *data1_start[rank] = idx; data1_start[rank]++; /* for local proccessing */
      proc_end           = ctable[idx];
      for (proc_id=0; proc_id<=proc_end; proc_id++) {  /* for others to process */
        if (proc_id == rank) continue; /* done before this loop */
        if (proc_id < proc_end && !PetscBTLookup(table[proc_id],idx)) continue; /* no need for sending idx to [proc_id] */
        *data1_start[proc_id] = idx; data1_start[proc_id]++;
        len_s[proc_id]++;
      }
    }
    /* update header data */
    for (proc_id=0; proc_id<size; proc_id++) {
      if (proc_id== rank) continue;
      *(data1 + proc_id*len + 1 + i) = len_s[proc_id] - iwork[proc_id];
      iwork[proc_id]                 = len_s[proc_id];
    }
    CHKERRQ(ISRestoreIndices(is[i],&idx_i));
  }

  nrqs = 0; nrqr = 0;
  for (i=0; i<size; i++) {
    data1_start[i] = data1 + i*len;
    if (len_s[i]) {
      nrqs++;
      len_s[i] += 1 + is_max; /* add no. of header msg */
    }
  }

  for (i=0; i<is_max; i++) {
    CHKERRQ(ISDestroy(&is[i]));
  }
  CHKERRQ(PetscFree(n));
  CHKERRQ(PetscFree(ctable));

  /* Determine the number of messages to expect, their lengths, from from-ids */
  CHKERRQ(PetscGatherNumberOfMessages(comm,NULL,len_s,&nrqr));
  CHKERRQ(PetscGatherMessageLengths(comm,nrqs,nrqr,len_s,&id_r1,&len_r1));

  /*  Now  post the sends */
  CHKERRQ(PetscMalloc2(size,&s_waits1,size,&s_waits2));
  k    = 0;
  for (proc_id=0; proc_id<size; proc_id++) {  /* send data1 to processor [proc_id] */
    if (len_s[proc_id]) {
      CHKERRMPI(MPI_Isend(data1_start[proc_id],len_s[proc_id],MPIU_INT,proc_id,tag1,comm,s_waits1+k));
      k++;
    }
  }

  /* 2. Receive other's is[] and process. Then send back */
  /*-----------------------------------------------------*/
  len = 0;
  for (i=0; i<nrqr; i++) {
    if (len_r1[i] > len) len = len_r1[i];
  }
  CHKERRQ(PetscFree(len_r1));
  CHKERRQ(PetscFree(id_r1));

  for (proc_id=0; proc_id<size; proc_id++) len_s[proc_id] = iwork[proc_id] = 0;

  CHKERRQ(PetscMalloc1(len+1,&odata1));
  CHKERRQ(PetscMalloc1(size,&odata2_ptr));
  CHKERRQ(PetscBTCreate(Mbs,&otable));

  len_max = ois_max*(Mbs+1); /* max space storing all is[] for each receive */
  len_est = 2*len_max;       /* estimated space of storing is[] for all receiving messages */
  CHKERRQ(PetscMalloc1(len_est+1,&odata2));
  nodata2 = 0;               /* nodata2+1: num of PetscMalloc(,&odata2_ptr[]) called */

  odata2_ptr[nodata2] = odata2;

  len_unused = len_est; /* unused space in the array odata2_ptr[nodata2]-- needs to be >= len_max  */

  k = 0;
  while (k < nrqr) {
    /* Receive messages */
    CHKERRMPI(MPI_Iprobe(MPI_ANY_SOURCE,tag1,comm,&flag,&r_status));
    if (flag) {
      CHKERRMPI(MPI_Get_count(&r_status,MPIU_INT,&len));
      proc_id = r_status.MPI_SOURCE;
      CHKERRMPI(MPI_Irecv(odata1,len,MPIU_INT,proc_id,r_status.MPI_TAG,comm,&r_req));
      CHKERRMPI(MPI_Wait(&r_req,&r_status));

      /*  Process messages */
      /*  make sure there is enough unused space in odata2 array */
      if (len_unused < len_max) { /* allocate more space for odata2 */
        CHKERRQ(PetscMalloc1(len_est+1,&odata2));

        odata2_ptr[++nodata2] = odata2;

        len_unused = len_est;
      }

      CHKERRQ(MatIncreaseOverlap_MPISBAIJ_Local(C,odata1,OTHER,odata2,&otable));
      len  = 1 + odata2[0];
      for (i=0; i<odata2[0]; i++) len += odata2[1 + i];

      /* Send messages back */
      CHKERRMPI(MPI_Isend(odata2,len,MPIU_INT,proc_id,tag2,comm,s_waits2+k));
      k++;
      odata2        += len;
      len_unused    -= len;
      len_s[proc_id] = len; /* num of messages sending back to [proc_id] by this proc */
    }
  }
  CHKERRQ(PetscFree(odata1));
  CHKERRQ(PetscBTDestroy(&otable));

  /* 3. Do local work on this processor's is[] */
  /*-------------------------------------------*/
  /* make sure there is enough unused space in odata2(=data) array */
  len_max = is_max*(Mbs+1); /* max space storing all is[] for this processor */
  if (len_unused < len_max) { /* allocate more space for odata2 */
    CHKERRQ(PetscMalloc1(len_est+1,&odata2));

    odata2_ptr[++nodata2] = odata2;
  }

  data = odata2;
  CHKERRQ(MatIncreaseOverlap_MPISBAIJ_Local(C,data1_start[rank],MINE,data,table));
  CHKERRQ(PetscFree(data1_start));

  /* 4. Receive work done on other processors, then merge */
  /*------------------------------------------------------*/
  /* get max number of messages that this processor expects to recv */
  CHKERRMPI(MPIU_Allreduce(len_s,iwork,size,MPI_INT,MPI_MAX,comm));
  CHKERRQ(PetscMalloc1(iwork[rank]+1,&data2));
  CHKERRQ(PetscFree4(len_s,btable,iwork,Bowners));

  k = 0;
  while (k < nrqs) {
    /* Receive messages */
    CHKERRMPI(MPI_Iprobe(MPI_ANY_SOURCE,tag2,comm,&flag,&r_status));
    if (flag) {
      CHKERRMPI(MPI_Get_count(&r_status,MPIU_INT,&len));

      proc_id = r_status.MPI_SOURCE;

      CHKERRMPI(MPI_Irecv(data2,len,MPIU_INT,proc_id,r_status.MPI_TAG,comm,&r_req));
      CHKERRMPI(MPI_Wait(&r_req,&r_status));
      if (len > 1+is_max) { /* Add data2 into data */
        data2_i = data2 + 1 + is_max;
        for (i=0; i<is_max; i++) {
          table_i = table[i];
          data_i  = data + 1 + is_max + Mbs*i;
          isz     = data[1+i];
          for (j=0; j<data2[1+i]; j++) {
            col = data2_i[j];
            if (!PetscBTLookupSet(table_i,col)) data_i[isz++] = col;
          }
          data[1+i] = isz;
          if (i < is_max - 1) data2_i += data2[1+i];
        }
      }
      k++;
    }
  }
  CHKERRQ(PetscFree(data2));
  CHKERRQ(PetscFree2(table,t_p));

  /* phase 1 sends are complete */
  CHKERRQ(PetscMalloc1(size,&s_status));
  if (nrqs) CHKERRMPI(MPI_Waitall(nrqs,s_waits1,s_status));
  CHKERRQ(PetscFree(data1));

  /* phase 2 sends are complete */
  if (nrqr) CHKERRMPI(MPI_Waitall(nrqr,s_waits2,s_status));
  CHKERRQ(PetscFree2(s_waits1,s_waits2));
  CHKERRQ(PetscFree(s_status));

  /* 5. Create new is[] */
  /*--------------------*/
  for (i=0; i<is_max; i++) {
    data_i = data + 1 + is_max + Mbs*i;
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,data[1+i],data_i,PETSC_COPY_VALUES,is+i));
  }
  for (k=0; k<=nodata2; k++) {
    CHKERRQ(PetscFree(odata2_ptr[k]));
  }
  CHKERRQ(PetscFree(odata2_ptr));
  PetscFunctionReturn(0);
}

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
static PetscErrorCode MatIncreaseOverlap_MPISBAIJ_Local(Mat C,PetscInt *data,PetscInt whose,PetscInt *nidx,PetscBT *table)
{
  Mat_MPISBAIJ   *c = (Mat_MPISBAIJ*)C->data;
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)(c->A)->data;
  Mat_SeqBAIJ    *b = (Mat_SeqBAIJ*)(c->B)->data;
  PetscInt       row,mbs,Mbs,*nidx_i,col,col_max,isz,isz0,*ai,*aj,*bi,*bj,*garray,rstart,l;
  PetscInt       a_start,a_end,b_start,b_end,i,j,k,is_max,*idx_i,n;
  PetscBT        table0;  /* mark the indices of input is[] for look up */
  PetscBT        table_i; /* poits to i-th table. When whose=OTHER, a single table is used for all is[] */

  PetscFunctionBegin;
  Mbs    = c->Mbs; mbs = a->mbs;
  ai     = a->i; aj = a->j;
  bi     = b->i; bj = b->j;
  garray = c->garray;
  rstart = c->rstartbs;
  is_max = data[0];

  CHKERRQ(PetscBTCreate(Mbs,&table0));

  nidx[0] = is_max;
  idx_i   = data + is_max + 1; /* ptr to input is[0] array */
  nidx_i  = nidx + is_max + 1; /* ptr to output is[0] array */
  for (i=0; i<is_max; i++) { /* for each is */
    isz = 0;
    n   = data[1+i]; /* size of input is[i] */

    /* initialize and set table_i(mark idx and nidx) and table0(only mark idx) */
    if (whose == MINE) { /* process this processor's is[] */
      table_i = table[i];
      nidx_i  = nidx + 1+ is_max + Mbs*i;
    } else {            /* process other processor's is[] - only use one temp table */
      table_i = table[0];
    }
    CHKERRQ(PetscBTMemzero(Mbs,table_i));
    CHKERRQ(PetscBTMemzero(Mbs,table0));
    if (n==0) {
      nidx[1+i] = 0;  /* size of new is[i] */
      continue;
    }

    isz0 = 0; col_max = 0;
    for (j=0; j<n; j++) {
      col = idx_i[j];
      PetscCheckFalse(col >= Mbs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"index col %" PetscInt_FMT " >= Mbs %" PetscInt_FMT,col,Mbs);
      if (!PetscBTLookupSet(table_i,col)) {
        CHKERRQ(PetscBTSet(table0,col));
        if (whose == MINE) nidx_i[isz0] = col;
        if (col_max < col) col_max = col;
        isz0++;
      }
    }

    if (whose == MINE) isz = isz0;
    k = 0;  /* no. of indices from input is[i] that have been examined */
    for (row=0; row<mbs; row++) {
      a_start = ai[row]; a_end = ai[row+1];
      b_start = bi[row]; b_end = bi[row+1];
      if (PetscBTLookup(table0,row+rstart)) { /* row is on input is[i]:
                                                do row search: collect all col in this row */
        for (l = a_start; l<a_end ; l++) { /* Amat */
          col = aj[l] + rstart;
          if (!PetscBTLookupSet(table_i,col)) nidx_i[isz++] = col;
        }
        for (l = b_start; l<b_end ; l++) { /* Bmat */
          col = garray[bj[l]];
          if (!PetscBTLookupSet(table_i,col)) nidx_i[isz++] = col;
        }
        k++;
        if (k >= isz0) break; /* for (row=0; row<mbs; row++) */
      } else { /* row is not on input is[i]:
                  do col serach: add row onto nidx_i if there is a col in nidx_i */
        for (l = a_start; l<a_end; l++) {  /* Amat */
          col = aj[l] + rstart;
          if (col > col_max) break;
          if (PetscBTLookup(table0,col)) {
            if (!PetscBTLookupSet(table_i,row+rstart)) nidx_i[isz++] = row+rstart;
            break; /* for l = start; l<end ; l++) */
          }
        }
        for (l = b_start; l<b_end; l++) {  /* Bmat */
          col = garray[bj[l]];
          if (col > col_max) break;
          if (PetscBTLookup(table0,col)) {
            if (!PetscBTLookupSet(table_i,row+rstart)) nidx_i[isz++] = row+rstart;
            break; /* for l = start; l<end ; l++) */
          }
        }
      }
    }

    if (i < is_max - 1) {
      idx_i  += n;   /* ptr to input is[i+1] array */
      nidx_i += isz; /* ptr to output is[i+1] array */
    }
    nidx[1+i] = isz; /* size of new is[i] */
  } /* for each is */
  CHKERRQ(PetscBTDestroy(&table0));
  PetscFunctionReturn(0);
}
