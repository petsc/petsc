
/*
   Routines to compute overlapping regions of a parallel MPI matrix.
   Used for finding submatrices that were shared across processors.
*/
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h> 
#include <petscbt.h>

static PetscErrorCode MatIncreaseOverlap_MPISBAIJ_Once(Mat,PetscInt,IS*);
static PetscErrorCode MatIncreaseOverlap_MPISBAIJ_Local(Mat,PetscInt*,PetscInt,PetscInt*,PetscBT*);

/* -------------------------------------------------------------------------*/
/* This code is modified from MatGetSubMatrices_MPIBAIJ_local() */
#if defined (PETSC_USE_CTABLE)
#undef __FUNCT__    
#define __FUNCT__ "PetscGetProc_ijonly" 
PetscErrorCode PetscGetProc_ijonly(const PetscInt row, const PetscMPIInt size, const PetscInt proc_gnode[], PetscMPIInt *rank)
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

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrices_MPIBAIJ_local_ijonly"
PetscErrorCode MatGetSubMatrices_MPIBAIJ_local_ijonly(Mat C,PetscInt ismax,const IS isrow[],const IS iscol[],MatReuse scall,Mat *submats)
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
  PetscInt       *a_j=a->j,*b_j=b->j,*cworkA,*cworkB; //bs=C->rmap->bs,bs2=c->bs2
  PetscInt       cstart = c->cstartbs,nzA,nzB,*a_i=a->i,*b_i=b->i,imark;
  PetscInt       *bmap = c->garray,ctmp,rstart=c->rstartbs;
  MPI_Request    *s_waits1,*r_waits1,*s_waits2,*r_waits2,*r_waits3,*s_waits3;
  //MPI_Request    *r_waits4,*s_waits4;
  MPI_Status     *r_status1,*r_status2,*s_status1,*s_status3,*s_status2,*r_status3;
  //MPI_Status     *r_status4,*s_status4;
  MPI_Comm       comm;
  //MatScalar      **rbuf4,**sbuf_aa,*vals,*mat_a,*sbuf_aa_i,*vworkA,*vworkB;
  //MatScalar      *a_a=a->a,*b_a=b->a;
  PetscBool      flag;
  PetscMPIInt    *onodes1,*olengths1;

#if defined (PETSC_USE_CTABLE)
  PetscInt       tt;
  PetscTable     *rowmaps,*colmaps,lrow1_grow1,lcol1_gcol1;
#else
  PetscInt       **cmap,*cmap_i,*rtable,*rmap_i,**rmap, Mbs = c->Mbs;
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
  ierr = PetscMalloc4(size,PetscMPIInt,&w1,size,PetscMPIInt,&w2,size,PetscInt,&w3,size,PetscInt,&w4);CHKERRQ(ierr);
  ierr = PetscMemzero(w1,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMemzero(w2,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMemzero(w3,size*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<ismax; i++) { 
    ierr   = PetscMemzero(w4,size*sizeof(PetscInt));CHKERRQ(ierr); /* initialise work vector*/
    jmax   = nrow[i];
    irow_i = irow[i];
    for (j=0; j<jmax; j++) {
      row  = irow_i[j];
#if defined (PETSC_USE_CTABLE)
      ierr = PetscGetProc_ijonly(row,size,c->rangebs,&proc);CHKERRQ(ierr);
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
      row  = irow_i[j];
#if defined (PETSC_USE_CTABLE)
      ierr = PetscGetProc_ijonly(row,size,c->rangebs,&proc);CHKERRQ(ierr);
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
  //ierr = PetscMalloc((nrqs+1)*sizeof(MatScalar*),&rbuf4);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Request),&r_waits3);CHKERRQ(ierr);
  //ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Request),&r_waits4);CHKERRQ(ierr);
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Status),&r_status2);CHKERRQ(ierr);

  for (i=0; i<nrqs; ++i) {
    ierr = MPI_Waitany(nrqs,r_waits2,&idex,r_status2+i);CHKERRQ(ierr);
    ierr = PetscMalloc(rbuf2[idex][0]*sizeof(PetscInt),&rbuf3[idex]);CHKERRQ(ierr);
    //ierr = PetscMalloc(rbuf2[idex][0]*bs2*sizeof(MatScalar),&rbuf4[idex]);CHKERRQ(ierr);
    ierr = MPI_Irecv(rbuf3[idex],rbuf2[idex][0],MPIU_INT,r_status2[i].MPI_SOURCE,tag2,comm,r_waits3+idex);CHKERRQ(ierr);
    //ierr = MPI_Irecv(rbuf4[idex],rbuf2[idex][0]*bs2,MPIU_MATSCALAR,r_status2[i].MPI_SOURCE,tag3,comm,r_waits4+idex);CHKERRQ(ierr);
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

#if defined(MV)
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
#endif
  ierr = PetscFree(rbuf1[0]);CHKERRQ(ierr);
  ierr = PetscFree(rbuf1);CHKERRQ(ierr);

  /* Form the matrix */
  /* create col map */
  {
    const PetscInt *icol_i;
#if defined (PETSC_USE_CTABLE)
    /* Create row map*/
    ierr = PetscMalloc((1+ismax)*sizeof(PetscTable),&colmaps);CHKERRQ(ierr);
    for (i=0; i<ismax; i++) {
      ierr = PetscTableCreate(ncol[i]+1,&colmaps[i]);CHKERRQ(ierr);
    }
#else
    ierr    = PetscMalloc(ismax*sizeof(PetscInt*),&cmap);CHKERRQ(ierr);
    ierr    = PetscMalloc(ismax*c->Nbs*sizeof(PetscInt),&cmap[0]);CHKERRQ(ierr);
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
  ierr    = PetscMalloc((1+ismax)*sizeof(PetscInt*)+ j*sizeof(PetscInt),&lens);CHKERRQ(ierr);
  lens[0] = (PetscInt*)(lens + ismax);
  ierr    = PetscMemzero(lens[0],j*sizeof(PetscInt));CHKERRQ(ierr);
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
      ierr = PetscGetProc_ijonly(row,size,c->rangebs,&proc);CHKERRQ(ierr);
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
  for (i=0; i<ismax; i++){ 
    ierr = PetscTableCreate(nrow[i]+1,&rowmaps[i]);CHKERRQ(ierr);
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
          if (row < 0) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"row not found in table"); }
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
  if (nrqr) {ierr = MPI_Waitall(nrqr,s_waits3,s_status3);CHKERRQ(ierr);}
  ierr = PetscFree(s_status3);CHKERRQ(ierr);
  ierr = PetscFree(s_waits3);CHKERRQ(ierr);

  /* Create the submatrices */
  if (scall == MAT_REUSE_MATRIX) {
    /*
        Assumes new rows are same length as the old rows, hence bug!
    */
    for (i=0; i<ismax; i++) {
      mat = (Mat_SeqBAIJ *)(submats[i]->data);
      //if ((mat->mbs != nrow[i]) || (mat->nbs != ncol[i] || C->rmap->bs != bs)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong size");
      if ((mat->mbs != nrow[i]) || (mat->nbs != ncol[i])) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong size");
      ierr = PetscMemcmp(mat->ilen,lens[i],mat->mbs *sizeof(PetscInt),&flag);CHKERRQ(ierr);
      if (!flag) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cannot reuse matrix. wrong no of nonzeros");
      /* Initial matrix as if empty */
      ierr = PetscMemzero(mat->ilen,mat->mbs*sizeof(PetscInt));CHKERRQ(ierr);
      submats[i]->factortype = C->factortype;
    }
  } else {
    for (i=0; i<ismax; i++) {
      ierr = MatCreate(PETSC_COMM_SELF,submats+i);CHKERRQ(ierr);
      //ierr = MatSetSizes(submats[i],nrow[i]*bs,ncol[i]*bs,nrow[i]*bs,ncol[i]*bs);CHKERRQ(ierr);
      ierr = MatSetSizes(submats[i],nrow[i],ncol[i],nrow[i],ncol[i]);CHKERRQ(ierr);
      ierr = MatSetType(submats[i],((PetscObject)A)->type_name);CHKERRQ(ierr);
      ierr = MatSeqBAIJSetPreallocation(submats[i],C->rmap->bs,0,lens[i]);CHKERRQ(ierr);
      //ierr = MatSeqBAIJSetPreallocation(submats[i],1,0,lens[i]);CHKERRQ(ierr);
      ierr = MatSeqSBAIJSetPreallocation(submats[i],1,0,lens[i]);CHKERRQ(ierr);
      //ierr = MatSeqSBAIJSetPreallocation(submats[i],C->rmap->bs,0,lens[i]);CHKERRQ(ierr); /* this subroutine is used by MatGetSubMatrices_MPISBAIJ()*/
    }
  }

  /* Assemble the matrices */
  /* First assemble the local rows */
  {
    PetscInt       ilen_row,*imat_ilen,*imat_j,*imat_i;
    //MatScalar *imat_a;
  
    for (i=0; i<ismax; i++) {
      mat       = (Mat_SeqBAIJ*)submats[i]->data;
      imat_ilen = mat->ilen;
      imat_j    = mat->j;
      imat_i    = mat->i;
      //imat_a    = mat->a;

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
	ierr = PetscGetProc_ijonly(row,size,c->rangebs,&proc);CHKERRQ(ierr);
#else
	proc = rtable[row];
#endif
        if (proc == rank) {
          row      = row - rstart;
          nzA      = a_i[row+1] - a_i[row]; 
          nzB      = b_i[row+1] - b_i[row];
          cworkA   = a_j + a_i[row]; 
          cworkB   = b_j + b_i[row];
          //vworkA   = a_a + a_i[row]*bs2;
          //vworkB   = b_a + b_i[row]*bs2;
#if defined (PETSC_USE_CTABLE)
	  ierr = PetscTableFind(lrow1_grow1,row+rstart+1,&row);CHKERRQ(ierr); 
          row--; 
          if (row < 0) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"row not found in table"); }
#else
          row      = rmap_i[row + rstart];
#endif
          mat_i    = imat_i[row]; 
          //mat_a    = imat_a + mat_i*bs2;
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
                //ierr     = PetscMemcpy(mat_a,vworkB+l*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
                //mat_a   += bs2;
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
              //ierr     = PetscMemcpy(mat_a,vworkA+l*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
              //mat_a   += bs2;
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
              //ierr     = PetscMemcpy(mat_a,vworkB+l*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
              //mat_a   += bs2;
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
    //MatScalar   *imat_a,*rbuf4_i;
    PetscMPIInt ii;

    for (tmp2=0; tmp2<nrqs; tmp2++) {
      //ierr    = MPI_Waitany(nrqs,r_waits4,&ii,r_status4+tmp2);CHKERRQ(ierr);
      ii = tmp2;  //new
      idex   = pa[ii];
      sbuf1_i = sbuf1[idex];
      jmax    = sbuf1_i[0];           
      ct1     = 2*jmax + 1; 
      ct2     = 0;    
      rbuf2_i = rbuf2[ii];
      rbuf3_i = rbuf3[ii];
      //rbuf4_i = rbuf4[ii];
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
        //imat_a    = mat->a;
        max1      = sbuf1_i[2*j];
        for (k=0; k<max1; k++,ct1++) {
          row   = sbuf1_i[ct1];
#if defined (PETSC_USE_CTABLE)
	  ierr = PetscTableFind(lrow1_grow1,row+1,&row);CHKERRQ(ierr); 
          row--; 
          if(row < 0) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"row not found in table"); }
#else
          row   = rmap_i[row];
#endif
          ilen  = imat_ilen[row];
          mat_i = imat_i[row];
          //mat_a = imat_a + mat_i*bs2;
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
              //ierr        = PetscMemcpy(mat_a,rbuf4_i+ct2*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
              //mat_a      += bs2;
              ilen++;
            }
          }
          imat_ilen[row] = ilen;
        }
      }
    }
  }    
    //ierr = PetscFree(r_status4);CHKERRQ(ierr);
    //ierr = PetscFree(r_waits4);CHKERRQ(ierr);
    //if (nrqr) {ierr = MPI_Waitall(nrqr,s_waits4,s_status4);CHKERRQ(ierr);}
    //ierr = PetscFree(s_waits4);CHKERRQ(ierr);
    //ierr = PetscFree(s_status4);CHKERRQ(ierr);

  /* Restore the indices */
  for (i=0; i<ismax; i++) {
    ierr = ISRestoreIndices(isrow[i],irow+i);CHKERRQ(ierr);
    ierr = ISRestoreIndices(iscol[i],icol+i);CHKERRQ(ierr);
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
    //ierr = PetscFree(rbuf4[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree3(sbuf2,req_size,req_source);CHKERRQ(ierr);
  ierr = PetscFree(rbuf3);CHKERRQ(ierr);
  //ierr = PetscFree(rbuf4);CHKERRQ(ierr);
  ierr = PetscFree(sbuf_aj[0]);CHKERRQ(ierr);
  ierr = PetscFree(sbuf_aj);CHKERRQ(ierr);
  //ierr = PetscFree(sbuf_aa[0]);CHKERRQ(ierr);
  //ierr = PetscFree(sbuf_aa);CHKERRQ(ierr);

#if defined (PETSC_USE_CTABLE)
  for (i=0; i<ismax; i++){
    ierr = PetscTableDestroy(rowmaps[i]);CHKERRQ(ierr);
    ierr = PetscTableDestroy(colmaps[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(colmaps);CHKERRQ(ierr);
  ierr = PetscFree(rowmaps);CHKERRQ(ierr);
#else
  ierr = PetscFree(rmap);CHKERRQ(ierr);
  ierr = PetscFree(cmap[0]);CHKERRQ(ierr);
  ierr = PetscFree(cmap);CHKERRQ(ierr);
#endif
  ierr = PetscFree(lens);CHKERRQ(ierr);

  for (i=0; i<ismax; i++) {
    ierr = MatAssemblyBegin(submats[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(submats[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
} 

/* ------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap_MPISBAIJ"
PetscErrorCode MatIncreaseOverlap_MPISBAIJ(Mat C,PetscInt is_max,IS is[],PetscInt ov)
{
  PetscErrorCode ierr;
  PetscInt       i,N=C->cmap->N, bs=C->rmap->bs;
  IS             *is_new;

  PetscFunctionBegin;
  ierr = PetscMalloc(is_max*sizeof(IS),&is_new);CHKERRQ(ierr);
  /* Convert the indices into block format */
  ierr = ISCompressIndicesGeneral(N,bs,is_max,is,is_new);CHKERRQ(ierr);
  if (ov < 0){ SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap specified\n");}
  
  //------- new ----------
  PetscBool flg=PETSC_FALSE;
  ierr = PetscOptionsHasName(PETSC_NULL, "-IncreaseOverlap_new", &flg);CHKERRQ(ierr);
  if (!flg){ /* previous non-scalable implementation */
    for (i=0; i<ov; ++i) {
      ierr = MatIncreaseOverlap_MPISBAIJ_Once(C,is_max,is_new);CHKERRQ(ierr);
    }
  } else { /* scalable implementation using modified BAIJ routines */

  Mat           *submats;
  IS            *is_row;
  PetscInt      M=C->rmap->N,Mbs=M/bs,*nidx,isz,iov;
  Mat_MPISBAIJ  *c = (Mat_MPISBAIJ*)C->data;
  PetscMPIInt   rank=c->rank;

  
  ierr = PetscMalloc((Mbs+1)*sizeof(PetscInt),&nidx);CHKERRQ(ierr); 

  /* Create is_row */
  ierr = PetscMalloc(is_max*sizeof(IS **),&is_row);CHKERRQ(ierr);
  for (i=0; i<is_max; i++){
    ierr = ISCreateStride(PETSC_COMM_SELF,Mbs,0,1,is_row+i);CHKERRQ(ierr); 
  }

  for (iov=0; iov<ov; ++iov) {
    /* Get submats for column search - replace with MatGetSubMatrices_MPIBAIJ_IJ() */
    // copied from MatGetSubMatrices_MPIBAIJ() ----------------
    /* Allocate memory to hold all the submatrices */
    //if (scall != MAT_REUSE_MATRIX) {
    ierr = PetscMalloc((is_max+1)*sizeof(Mat),&submats);CHKERRQ(ierr);
    //}
    /* Determine the number of stages through which submatrices are done */
    PetscInt nmax,nstages_local,nstages,max_no,pos;
    nmax          = 20*1000000 / (c->Nbs * sizeof(PetscInt));
    if (!nmax) nmax = 1;
    nstages_local = is_max/nmax + ((is_max % nmax)?1:0);
  
    /* Make sure every processor loops through the nstages */
    ierr = MPI_Allreduce(&nstages_local,&nstages,1,MPIU_INT,MPI_MAX,((PetscObject)C)->comm);CHKERRQ(ierr);
    for (i=0,pos=0; i<nstages; i++) {
      if (pos+nmax <= is_max) max_no = nmax;
      else if (pos == is_max) max_no = 0;
      else                   max_no = is_max-pos;
     
      //ierr = MatGetSubMatrices_MPIBAIJ_local(C,max_no,is_row+pos,is_new+pos,MAT_INITIAL_MATRIX,submats+pos);CHKERRQ(ierr);
      ierr = MatGetSubMatrices_MPIBAIJ_local_ijonly(C,max_no,is_row+pos,is_new+pos,MAT_INITIAL_MATRIX,submats+pos);CHKERRQ(ierr);
      if (rank == 10 && !i){
        printf("submats[0]:\n");
        ierr = MatView(submats[0],PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      }

      pos += max_no;
    }
    // end of copy -----------------------------------
   
    /* Row search */
    ierr = MatIncreaseOverlap_MPIBAIJ_Once(C,is_max,is_new);CHKERRQ(ierr);

    /* Column search */
    PetscBT        table;
    Mat_SeqSBAIJ   *asub_i; 
    PetscInt       *ai,brow,nz,nis,l;
    const PetscInt *idx;

    ierr = PetscBTCreate(Mbs,table);CHKERRQ(ierr); 
    for (i=0; i<is_max; i++){ 
      asub_i = (Mat_SeqSBAIJ*)submats[i]->data;
      ai=asub_i->i;;
    
      /* put is_new obtained from MatIncreaseOverlap_MPIBAIJ() to table */
      ierr = PetscBTMemzero(Mbs,table);CHKERRQ(ierr);
    
      ierr = ISGetIndices(is_new[i],&idx);CHKERRQ(ierr);
      ierr = ISGetLocalSize(is_new[i],&nis);CHKERRQ(ierr);
      for (l=0; l<nis; l++) { 
        ierr = PetscBTSet(table,idx[l]);CHKERRQ(ierr); 
        nidx[l] = idx[l];
      }
      isz = nis;

      for (brow=0; brow<Mbs; brow++){
        nz = ai[brow+1] - ai[brow];
        if (nz) {
          if (!PetscBTLookupSet(table,brow)) nidx[isz++] = brow;
        }
      }
      ierr = ISRestoreIndices(is_new[i],&idx);CHKERRQ(ierr);
      ierr = ISDestroy(is_new[i]);CHKERRQ(ierr);

      /* create updated is_new */
      ierr = ISCreateGeneral(PETSC_COMM_SELF,isz,nidx,PETSC_COPY_VALUES,is_new+i);CHKERRQ(ierr);
    }

    /* Free tmp spaces */
    ierr = PetscBTDestroy(table);CHKERRQ(ierr);
    for (i=0; i<is_max; i++){
      if (rank == 10 && !i){
        printf("submats[0]:\n");
        ierr = MatView(submats[0],PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      }
      ierr = MatDestroy(submats[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(submats);CHKERRQ(ierr);
  } 

  for (i=0; i<is_max; i++){
    ierr = ISDestroy(is_row[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(is_row);CHKERRQ(ierr);
  ierr = PetscFree(nidx);CHKERRQ(ierr);
  } 
  //--------end of new----------

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
static PetscErrorCode MatIncreaseOverlap_MPISBAIJ_Once(Mat C,PetscInt is_max,IS is[])
{
  Mat_MPISBAIJ  *c = (Mat_MPISBAIJ*)C->data;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,tag1,tag2,*len_s,nrqr,nrqs,*id_r1,*len_r1,flag,len;
  const PetscInt *idx_i;
  PetscInt       idx,isz,col,*n,*data1,**data1_start,*data2,*data2_i,*data,*data_i,
                 Mbs,i,j,k,*odata1,*odata2,
                 proc_id,**odata2_ptr,*ctable=0,*btable,len_max,len_est;
  PetscInt       proc_end=0,*iwork,len_unused,nodata2;
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
  comm = ((PetscObject)C)->comm;
  size = c->size;
  rank = c->rank;
  Mbs  = c->Mbs;

  ierr = PetscObjectGetNewTag((PetscObject)C,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag2);CHKERRQ(ierr);

  /* create tables used in
     step 1: table[i] - mark c->garray of proc [i]
     step 3: table[i] - mark indices of is[i] when whose=MINE     
             table[0] - mark incideces of is[] when whose=OTHER */
  len = PetscMax(is_max, size);CHKERRQ(ierr);
  ierr = PetscMalloc2(len,PetscBT,&table,(Mbs/PETSC_BITS_PER_BYTE+1)*len,char,&t_p);CHKERRQ(ierr);
  for (i=0; i<len; i++) {
    table[i]  = t_p  + (Mbs/PETSC_BITS_PER_BYTE+1)*i; 
  }

  ierr = MPI_Allreduce(&is_max,&ois_max,1,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);
  
  /* 1. Send this processor's is[] to other processors */
  /*---------------------------------------------------*/
  /* allocate spaces */
  ierr = PetscMalloc(is_max*sizeof(PetscInt),&n);CHKERRQ(ierr);
  len = 0;
  for (i=0; i<is_max; i++) {
    ierr = ISGetLocalSize(is[i],&n[i]);CHKERRQ(ierr);
    len += n[i]; 
  }
  if (!len) { 
    is_max = 0;
  } else {
    len += 1 + is_max; /* max length of data1 for one processor */
  }

 
  ierr = PetscMalloc((size*len+1)*sizeof(PetscInt),&data1);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(PetscInt*),&data1_start);CHKERRQ(ierr);
  for (i=0; i<size; i++) data1_start[i] = data1 + i*len;

  ierr = PetscMalloc4(size,PetscInt,&len_s,size,PetscInt,&btable,size,PetscInt,&iwork,size+1,PetscInt,&Bowners);CHKERRQ(ierr);

  /* gather c->garray from all processors */
  ierr = ISCreateGeneral(comm,Bnbs,c->garray,PETSC_COPY_VALUES,&garray_local);CHKERRQ(ierr);
  ierr = ISAllGather(garray_local, &garray_gl);CHKERRQ(ierr);
  ierr = ISDestroy(garray_local);CHKERRQ(ierr);
  ierr = MPI_Allgather(&Bnbs,1,MPIU_INT,Bowners+1,1,MPIU_INT,comm);CHKERRQ(ierr);
  Bowners[0] = 0;
  for (i=0; i<size; i++) Bowners[i+1] += Bowners[i];
  
  if (is_max){ 
    /* hash table ctable which maps c->row to proc_id) */
    ierr = PetscMalloc(Mbs*sizeof(PetscInt),&ctable);CHKERRQ(ierr);
    for (proc_id=0,j=0; proc_id<size; proc_id++) {
      for (; j<C->rmap->range[proc_id+1]/bs; j++) {
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
  ierr = PetscFree(ctable);CHKERRQ(ierr);

  /* Determine the number of messages to expect, their lengths, from from-ids */
  ierr = PetscGatherNumberOfMessages(comm,PETSC_NULL,len_s,&nrqr);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nrqs,nrqr,len_s,&id_r1,&len_r1);CHKERRQ(ierr); 
  
  /*  Now  post the sends */
  ierr = PetscMalloc2(size,MPI_Request,&s_waits1,size,MPI_Request,&s_waits2);CHKERRQ(ierr);
  k = 0;
  for (proc_id=0; proc_id<size; proc_id++){  /* send data1 to processor [proc_id] */
    if (len_s[proc_id]){
      ierr = MPI_Isend(data1_start[proc_id],len_s[proc_id],MPIU_INT,proc_id,tag1,comm,s_waits1+k);CHKERRQ(ierr);
      k++;
    }
  }

  /* 2. Receive other's is[] and process. Then send back */
  /*-----------------------------------------------------*/
  len = 0;
  for (i=0; i<nrqr; i++){
    if (len_r1[i] > len)len = len_r1[i];
  }
  ierr = PetscFree(len_r1);CHKERRQ(ierr);
  ierr = PetscFree(id_r1);CHKERRQ(ierr);

  for (proc_id=0; proc_id<size; proc_id++)
    len_s[proc_id] = iwork[proc_id] = 0;
  
  ierr = PetscMalloc((len+1)*sizeof(PetscInt),&odata1);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(PetscInt**),&odata2_ptr);CHKERRQ(ierr); 
  ierr = PetscBTCreate(Mbs,otable);CHKERRQ(ierr);

  len_max = ois_max*(Mbs+1);  /* max space storing all is[] for each receive */
  len_est = 2*len_max; /* estimated space of storing is[] for all receiving messages */
  ierr = PetscMalloc((len_est+1)*sizeof(PetscInt),&odata2);CHKERRQ(ierr);
  nodata2 = 0;       /* nodata2+1: num of PetscMalloc(,&odata2_ptr[]) called */
  odata2_ptr[nodata2] = odata2;
  len_unused = len_est; /* unused space in the array odata2_ptr[nodata2]-- needs to be >= len_max  */
  
  k = 0;
  while (k < nrqr){
    /* Receive messages */
    ierr = MPI_Iprobe(MPI_ANY_SOURCE,tag1,comm,&flag,&r_status);CHKERRQ(ierr);
    if (flag){ 
      ierr = MPI_Get_count(&r_status,MPIU_INT,&len);CHKERRQ(ierr); 
      proc_id = r_status.MPI_SOURCE;
      ierr = MPI_Irecv(odata1,len,MPIU_INT,proc_id,r_status.MPI_TAG,comm,&r_req);CHKERRQ(ierr);
      ierr = MPI_Wait(&r_req,&r_status);CHKERRQ(ierr);

      /*  Process messages */
      /*  make sure there is enough unused space in odata2 array */
      if (len_unused < len_max){ /* allocate more space for odata2 */
        ierr = PetscMalloc((len_est+1)*sizeof(PetscInt),&odata2);CHKERRQ(ierr);
        odata2_ptr[++nodata2] = odata2;
        len_unused = len_est;
      }

      ierr = MatIncreaseOverlap_MPISBAIJ_Local(C,odata1,OTHER,odata2,&otable);CHKERRQ(ierr);
      len = 1 + odata2[0];
      for (i=0; i<odata2[0]; i++){
        len += odata2[1 + i];
      }

      /* Send messages back */
      ierr = MPI_Isend(odata2,len,MPIU_INT,proc_id,tag2,comm,s_waits2+k);CHKERRQ(ierr);
      k++;
      odata2     += len;
      len_unused -= len;
      len_s[proc_id] = len; /* num of messages sending back to [proc_id] by this proc */
    } 
  } 
  ierr = PetscFree(odata1);CHKERRQ(ierr); 
  ierr = PetscBTDestroy(otable);CHKERRQ(ierr); 

  /* 3. Do local work on this processor's is[] */
  /*-------------------------------------------*/
  /* make sure there is enough unused space in odata2(=data) array */
  len_max = is_max*(Mbs+1); /* max space storing all is[] for this processor */
  if (len_unused < len_max){ /* allocate more space for odata2 */
    ierr = PetscMalloc((len_est+1)*sizeof(PetscInt),&odata2);CHKERRQ(ierr);
    odata2_ptr[++nodata2] = odata2;
    len_unused = len_est;
  }

  data = odata2;
  ierr = MatIncreaseOverlap_MPISBAIJ_Local(C,data1_start[rank],MINE,data,table);CHKERRQ(ierr);
  ierr = PetscFree(data1_start);CHKERRQ(ierr);

  /* 4. Receive work done on other processors, then merge */
  /*------------------------------------------------------*/
  /* get max number of messages that this processor expects to recv */
  ierr = MPI_Allreduce(len_s,iwork,size,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);
  ierr = PetscMalloc((iwork[rank]+1)*sizeof(PetscInt),&data2);CHKERRQ(ierr);
  ierr = PetscFree4(len_s,btable,iwork,Bowners);CHKERRQ(ierr);

  k = 0;
  while (k < nrqs){
    /* Receive messages */
    ierr = MPI_Iprobe(MPI_ANY_SOURCE,tag2,comm,&flag,&r_status);
    if (flag){
      ierr = MPI_Get_count(&r_status,MPIU_INT,&len);CHKERRQ(ierr);
      proc_id = r_status.MPI_SOURCE;
      ierr = MPI_Irecv(data2,len,MPIU_INT,proc_id,r_status.MPI_TAG,comm,&r_req);CHKERRQ(ierr);
      ierr = MPI_Wait(&r_req,&r_status);CHKERRQ(ierr);
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
  ierr = PetscFree(data2);CHKERRQ(ierr);
  ierr = PetscFree2(table,t_p);CHKERRQ(ierr);

  /* phase 1 sends are complete */
  ierr = PetscMalloc(size*sizeof(MPI_Status),&s_status);CHKERRQ(ierr);
  if (nrqs) {ierr = MPI_Waitall(nrqs,s_waits1,s_status);CHKERRQ(ierr);}
  ierr = PetscFree(data1);CHKERRQ(ierr); 
       
  /* phase 2 sends are complete */
  if (nrqr){ierr = MPI_Waitall(nrqr,s_waits2,s_status);CHKERRQ(ierr);}
  ierr = PetscFree2(s_waits1,s_waits2);CHKERRQ(ierr);
  ierr = PetscFree(s_status);CHKERRQ(ierr); 

  /* 5. Create new is[] */
  /*--------------------*/ 
  for (i=0; i<is_max; i++) {
    data_i = data + 1 + is_max + Mbs*i;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,data[1+i],data_i,PETSC_COPY_VALUES,is+i);CHKERRQ(ierr);
  }
  for (k=0; k<=nodata2; k++){
    ierr = PetscFree(odata2_ptr[k]);CHKERRQ(ierr); 
  }
  ierr = PetscFree(odata2_ptr);CHKERRQ(ierr);

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
  Mbs = c->Mbs; mbs = a->mbs; 
  ai = a->i; aj = a->j;
  bi = b->i; bj = b->j;
  garray = c->garray;
  rstart = c->rstartbs;
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
      if (col >= Mbs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"index col %D >= Mbs %D",col,Mbs);
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


