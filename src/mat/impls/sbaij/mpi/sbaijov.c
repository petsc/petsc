
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
  PetscErrorCode ierr;
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
  ierr = PetscMalloc1(is_max,&is_new);CHKERRQ(ierr);
  /* Convert the indices into block format */
  ierr = ISCompressIndicesGeneral(N,C->rmap->n,bs,is_max,is,is_new);CHKERRQ(ierr);
  if (ov < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap specified\n");

  /* ----- previous non-scalable implementation ----- */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsHasName(NULL,NULL, "-IncreaseOverlap_old", &flg);CHKERRQ(ierr);
  if (flg) { /* previous non-scalable implementation */
    printf("use previous non-scalable implementation...\n");
    for (i=0; i<ov; ++i) {
      ierr = MatIncreaseOverlap_MPISBAIJ_Once(C,is_max,is_new);CHKERRQ(ierr);
    }
  } else { /* implementation using modified BAIJ routines */

    ierr = PetscMalloc1(Mbs+1,&nidx);CHKERRQ(ierr);
    ierr = PetscBTCreate(Mbs,&table);CHKERRQ(ierr); /* for column search */

    /* Create is_row */
    ierr = PetscMalloc1(is_max,&is_row);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,Mbs,0,1,&is_row[0]);CHKERRQ(ierr);

    for (i=1; i<is_max; i++) {
      is_row[i]  = is_row[0]; /* reuse is_row[0] */
    }

    /* Allocate memory to hold all the submatrices - Modified from MatCreateSubMatrices_MPIBAIJ() */
    ierr = PetscMalloc1(is_max+1,&submats);CHKERRQ(ierr);

    /* Determine the number of stages through which submatrices are done */
    nmax = 20*1000000 / (c->Nbs * sizeof(PetscInt));
    if (!nmax) nmax = 1;
    nstages_local = is_max/nmax + ((is_max % nmax) ? 1 : 0);

    /* Make sure every processor loops through the nstages */
    ierr = MPIU_Allreduce(&nstages_local,&nstages,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)C));CHKERRQ(ierr);

    for (iov=0; iov<ov; ++iov) {
      /* 1) Get submats for column search */
      for (i=0,pos=0; i<nstages; i++) {
        if (pos+nmax <= is_max) max_no = nmax;
        else if (pos == is_max) max_no = 0;
        else                    max_no = is_max-pos;
        c->ijonly = PETSC_TRUE; /* only matrix data structures are requested */
        /* The resulting submatrices should be BAIJ, not SBAIJ, hence we change this value to trigger that */
        ierr      = PetscStrcpy(((PetscObject)c->A)->type_name,MATSEQBAIJ);CHKERRQ(ierr);
        ierr      = MatCreateSubMatrices_MPIBAIJ_local(C,max_no,is_row+pos,is_new+pos,MAT_INITIAL_MATRIX,submats+pos);CHKERRQ(ierr);
        ierr      = PetscStrcpy(((PetscObject)c->A)->type_name,MATSEQSBAIJ);CHKERRQ(ierr);
        pos      += max_no;
      }

      /* 2) Row search */
      ierr = MatIncreaseOverlap_MPIBAIJ_Once(C,is_max,is_new);CHKERRQ(ierr);

      /* 3) Column search */
      for (i=0; i<is_max; i++) {
        asub_i = (Mat_SeqSBAIJ*)submats[i]->data;
        ai     = asub_i->i;

        /* put is_new obtained from MatIncreaseOverlap_MPIBAIJ() to table */
        ierr = PetscBTMemzero(Mbs,table);CHKERRQ(ierr);

        ierr = ISGetIndices(is_new[i],&idx);CHKERRQ(ierr);
        ierr = ISGetLocalSize(is_new[i],&nis);CHKERRQ(ierr);
        for (l=0; l<nis; l++) {
          ierr    = PetscBTSet(table,idx[l]);CHKERRQ(ierr);
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
        ierr = ISRestoreIndices(is_new[i],&idx);CHKERRQ(ierr);
        ierr = ISDestroy(&is_new[i]);CHKERRQ(ierr);

        /* create updated is_new */
        ierr = ISCreateGeneral(PETSC_COMM_SELF,isz,nidx,PETSC_COPY_VALUES,is_new+i);CHKERRQ(ierr);
      }

      /* Free tmp spaces */
      for (i=0; i<is_max; i++) {
        ierr = MatDestroy(&submats[i]);CHKERRQ(ierr);
      }
    }

    ierr = PetscBTDestroy(&table);CHKERRQ(ierr);
    ierr = PetscFree(submats);CHKERRQ(ierr);
    ierr = ISDestroy(&is_row[0]);CHKERRQ(ierr);
    ierr = PetscFree(is_row);CHKERRQ(ierr);
    ierr = PetscFree(nidx);CHKERRQ(ierr);

  }

  for (i=0; i<is_max; i++) {ierr = ISDestroy(&is[i]);CHKERRQ(ierr);}
  ierr = ISExpandIndicesGeneral(N,N,bs,is_max,is_new,is);CHKERRQ(ierr);

  for (i=0; i<is_max; i++) {ierr = ISDestroy(&is_new[i]);CHKERRQ(ierr);}
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
static PetscErrorCode MatIncreaseOverlap_MPISBAIJ_Once(Mat C,PetscInt is_max,IS is[])
{
  Mat_MPISBAIJ   *c = (Mat_MPISBAIJ*)C->data;
  PetscErrorCode ierr;
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
  ierr = PetscObjectGetComm((PetscObject)C,&comm);CHKERRQ(ierr);
  size = c->size;
  rank = c->rank;
  Mbs  = c->Mbs;

  ierr = PetscObjectGetNewTag((PetscObject)C,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag2);CHKERRQ(ierr);

  /* create tables used in
     step 1: table[i] - mark c->garray of proc [i]
     step 3: table[i] - mark indices of is[i] when whose=MINE
             table[0] - mark incideces of is[] when whose=OTHER */
  len  = PetscMax(is_max, size);CHKERRQ(ierr);
  ierr = PetscMalloc2(len,&table,(Mbs/PETSC_BITS_PER_BYTE+1)*len,&t_p);CHKERRQ(ierr);
  for (i=0; i<len; i++) {
    table[i] = t_p  + (Mbs/PETSC_BITS_PER_BYTE+1)*i;
  }

  ierr = MPIU_Allreduce(&is_max,&ois_max,1,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);

  /* 1. Send this processor's is[] to other processors */
  /*---------------------------------------------------*/
  /* allocate spaces */
  ierr = PetscMalloc1(is_max,&n);CHKERRQ(ierr);
  len  = 0;
  for (i=0; i<is_max; i++) {
    ierr = ISGetLocalSize(is[i],&n[i]);CHKERRQ(ierr);
    len += n[i];
  }
  if (!len) {
    is_max = 0;
  } else {
    len += 1 + is_max; /* max length of data1 for one processor */
  }


  ierr = PetscMalloc1(size*len+1,&data1);CHKERRQ(ierr);
  ierr = PetscMalloc1(size,&data1_start);CHKERRQ(ierr);
  for (i=0; i<size; i++) data1_start[i] = data1 + i*len;

  ierr = PetscMalloc4(size,&len_s,size,&btable,size,&iwork,size+1,&Bowners);CHKERRQ(ierr);

  /* gather c->garray from all processors */
  ierr = ISCreateGeneral(comm,Bnbs,c->garray,PETSC_COPY_VALUES,&garray_local);CHKERRQ(ierr);
  ierr = ISAllGather(garray_local, &garray_gl);CHKERRQ(ierr);
  ierr = ISDestroy(&garray_local);CHKERRQ(ierr);
  ierr = MPI_Allgather(&Bnbs,1,MPIU_INT,Bowners+1,1,MPIU_INT,comm);CHKERRMPI(ierr);

  Bowners[0] = 0;
  for (i=0; i<size; i++) Bowners[i+1] += Bowners[i];

  if (is_max) {
    /* hash table ctable which maps c->row to proc_id) */
    ierr = PetscMalloc1(Mbs,&ctable);CHKERRQ(ierr);
    for (proc_id=0,j=0; proc_id<size; proc_id++) {
      for (; j<C->rmap->range[proc_id+1]/bs; j++) ctable[j] = proc_id;
    }

    /* hash tables marking c->garray */
    ierr = ISGetIndices(garray_gl,&idx_i);CHKERRQ(ierr);
    for (i=0; i<size; i++) {
      table_i = table[i];
      ierr    = PetscBTMemzero(Mbs,table_i);CHKERRQ(ierr);
      for (j = Bowners[i]; j<Bowners[i+1]; j++) { /* go through B cols of proc[i]*/
        ierr = PetscBTSet(table_i,idx_i[j]);CHKERRQ(ierr);
      }
    }
    ierr = ISRestoreIndices(garray_gl,&idx_i);CHKERRQ(ierr);
  }  /* if (is_max) */
  ierr = ISDestroy(&garray_gl);CHKERRQ(ierr);

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
    ierr = ISGetIndices(is[i],&idx_i);CHKERRQ(ierr);
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
    ierr = ISRestoreIndices(is[i],&idx_i);CHKERRQ(ierr);
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
    ierr = ISDestroy(&is[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(n);CHKERRQ(ierr);
  ierr = PetscFree(ctable);CHKERRQ(ierr);

  /* Determine the number of messages to expect, their lengths, from from-ids */
  ierr = PetscGatherNumberOfMessages(comm,NULL,len_s,&nrqr);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nrqs,nrqr,len_s,&id_r1,&len_r1);CHKERRQ(ierr);

  /*  Now  post the sends */
  ierr = PetscMalloc2(size,&s_waits1,size,&s_waits2);CHKERRQ(ierr);
  k    = 0;
  for (proc_id=0; proc_id<size; proc_id++) {  /* send data1 to processor [proc_id] */
    if (len_s[proc_id]) {
      ierr = MPI_Isend(data1_start[proc_id],len_s[proc_id],MPIU_INT,proc_id,tag1,comm,s_waits1+k);CHKERRMPI(ierr);
      k++;
    }
  }

  /* 2. Receive other's is[] and process. Then send back */
  /*-----------------------------------------------------*/
  len = 0;
  for (i=0; i<nrqr; i++) {
    if (len_r1[i] > len) len = len_r1[i];
  }
  ierr = PetscFree(len_r1);CHKERRQ(ierr);
  ierr = PetscFree(id_r1);CHKERRQ(ierr);

  for (proc_id=0; proc_id<size; proc_id++) len_s[proc_id] = iwork[proc_id] = 0;

  ierr = PetscMalloc1(len+1,&odata1);CHKERRQ(ierr);
  ierr = PetscMalloc1(size,&odata2_ptr);CHKERRQ(ierr);
  ierr = PetscBTCreate(Mbs,&otable);CHKERRQ(ierr);

  len_max = ois_max*(Mbs+1); /* max space storing all is[] for each receive */
  len_est = 2*len_max;       /* estimated space of storing is[] for all receiving messages */
  ierr    = PetscMalloc1(len_est+1,&odata2);CHKERRQ(ierr);
  nodata2 = 0;               /* nodata2+1: num of PetscMalloc(,&odata2_ptr[]) called */

  odata2_ptr[nodata2] = odata2;

  len_unused = len_est; /* unused space in the array odata2_ptr[nodata2]-- needs to be >= len_max  */

  k = 0;
  while (k < nrqr) {
    /* Receive messages */
    ierr = MPI_Iprobe(MPI_ANY_SOURCE,tag1,comm,&flag,&r_status);CHKERRMPI(ierr);
    if (flag) {
      ierr    = MPI_Get_count(&r_status,MPIU_INT,&len);CHKERRQ(ierr);
      proc_id = r_status.MPI_SOURCE;
      ierr    = MPI_Irecv(odata1,len,MPIU_INT,proc_id,r_status.MPI_TAG,comm,&r_req);CHKERRQ(ierr);
      ierr    = MPI_Wait(&r_req,&r_status);CHKERRQ(ierr);

      /*  Process messages */
      /*  make sure there is enough unused space in odata2 array */
      if (len_unused < len_max) { /* allocate more space for odata2 */
        ierr = PetscMalloc1(len_est+1,&odata2);CHKERRQ(ierr);

        odata2_ptr[++nodata2] = odata2;

        len_unused = len_est;
      }

      ierr = MatIncreaseOverlap_MPISBAIJ_Local(C,odata1,OTHER,odata2,&otable);CHKERRQ(ierr);
      len  = 1 + odata2[0];
      for (i=0; i<odata2[0]; i++) len += odata2[1 + i];

      /* Send messages back */
      ierr = MPI_Isend(odata2,len,MPIU_INT,proc_id,tag2,comm,s_waits2+k);CHKERRMPI(ierr);
      k++;
      odata2        += len;
      len_unused    -= len;
      len_s[proc_id] = len; /* num of messages sending back to [proc_id] by this proc */
    }
  }
  ierr = PetscFree(odata1);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&otable);CHKERRQ(ierr);

  /* 3. Do local work on this processor's is[] */
  /*-------------------------------------------*/
  /* make sure there is enough unused space in odata2(=data) array */
  len_max = is_max*(Mbs+1); /* max space storing all is[] for this processor */
  if (len_unused < len_max) { /* allocate more space for odata2 */
    ierr = PetscMalloc1(len_est+1,&odata2);CHKERRQ(ierr);

    odata2_ptr[++nodata2] = odata2;
  }

  data = odata2;
  ierr = MatIncreaseOverlap_MPISBAIJ_Local(C,data1_start[rank],MINE,data,table);CHKERRQ(ierr);
  ierr = PetscFree(data1_start);CHKERRQ(ierr);

  /* 4. Receive work done on other processors, then merge */
  /*------------------------------------------------------*/
  /* get max number of messages that this processor expects to recv */
  ierr = MPIU_Allreduce(len_s,iwork,size,MPI_INT,MPI_MAX,comm);CHKERRQ(ierr);
  ierr = PetscMalloc1(iwork[rank]+1,&data2);CHKERRQ(ierr);
  ierr = PetscFree4(len_s,btable,iwork,Bowners);CHKERRQ(ierr);

  k = 0;
  while (k < nrqs) {
    /* Receive messages */
    ierr = MPI_Iprobe(MPI_ANY_SOURCE,tag2,comm,&flag,&r_status);CHKERRMPI(ierr);
    if (flag) {
      ierr = MPI_Get_count(&r_status,MPIU_INT,&len);CHKERRMPI(ierr);

      proc_id = r_status.MPI_SOURCE;

      ierr = MPI_Irecv(data2,len,MPIU_INT,proc_id,r_status.MPI_TAG,comm,&r_req);CHKERRMPI(ierr);
      ierr = MPI_Wait(&r_req,&r_status);CHKERRMPI(ierr);
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
  ierr = PetscFree(data2);CHKERRQ(ierr);
  ierr = PetscFree2(table,t_p);CHKERRQ(ierr);

  /* phase 1 sends are complete */
  ierr = PetscMalloc1(size,&s_status);CHKERRQ(ierr);
  if (nrqs) {ierr = MPI_Waitall(nrqs,s_waits1,s_status);CHKERRMPI(ierr);}
  ierr = PetscFree(data1);CHKERRQ(ierr);

  /* phase 2 sends are complete */
  if (nrqr) {ierr = MPI_Waitall(nrqr,s_waits2,s_status);CHKERRMPI(ierr);}
  ierr = PetscFree2(s_waits1,s_waits2);CHKERRQ(ierr);
  ierr = PetscFree(s_status);CHKERRQ(ierr);

  /* 5. Create new is[] */
  /*--------------------*/
  for (i=0; i<is_max; i++) {
    data_i = data + 1 + is_max + Mbs*i;
    ierr   = ISCreateGeneral(PETSC_COMM_SELF,data[1+i],data_i,PETSC_COPY_VALUES,is+i);CHKERRQ(ierr);
  }
  for (k=0; k<=nodata2; k++) {
    ierr = PetscFree(odata2_ptr[k]);CHKERRQ(ierr);
  }
  ierr = PetscFree(odata2_ptr);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
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

  ierr = PetscBTCreate(Mbs,&table0);CHKERRQ(ierr);

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
    ierr = PetscBTMemzero(Mbs,table_i);CHKERRQ(ierr);
    ierr = PetscBTMemzero(Mbs,table0);CHKERRQ(ierr);
    if (n==0) {
      nidx[1+i] = 0;  /* size of new is[i] */
      continue;
    }

    isz0 = 0; col_max = 0;
    for (j=0; j<n; j++) {
      col = idx_i[j];
      if (col >= Mbs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"index col %D >= Mbs %D",col,Mbs);
      if (!PetscBTLookupSet(table_i,col)) {
        ierr = PetscBTSet(table0,col);CHKERRQ(ierr);
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
  ierr = PetscBTDestroy(&table0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
