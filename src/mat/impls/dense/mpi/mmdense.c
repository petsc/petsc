#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mmdense.c,v 1.13 1997/10/19 03:25:11 bsmith Exp balay $";
#endif

/*
   Support for the parallel dense matrix vector multiply
*/
#include "src/mat/impls/dense/mpi/mpidense.h"
#include "src/vec/vecimpl.h"

#undef __FUNC__  
#define __FUNC__ "MatSetUpMultiply_MPIDense"
int MatSetUpMultiply_MPIDense(Mat mat)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr,n;
  IS           tofrom;
  Vec          gvec;

  PetscFunctionBegin;
  /* Create local vector that is used to scatter into */
  ierr = VecCreateSeq(PETSC_COMM_SELF,mdn->N,&mdn->lvec); CHKERRQ(ierr);

  /* Create temporary index set for building scatter gather */
  ierr = ISCreateStride(PETSC_COMM_SELF,mdn->N,0,1,&tofrom); CHKERRQ(ierr);

  /* Create temporary global vector to generate scatter context */
  n    = mdn->cowners[mdn->rank+1] - mdn->cowners[mdn->rank];
  ierr = VecCreateMPI(mat->comm,n,mdn->N,&gvec); CHKERRQ(ierr);

  /* Generate the scatter context */
  ierr = VecScatterCreate(gvec,tofrom,mdn->lvec,tofrom,&mdn->Mvctx); CHKERRQ(ierr);
  PLogObjectParent(mat,mdn->Mvctx);
  PLogObjectParent(mat,mdn->lvec);
  PLogObjectParent(mat,tofrom);
  PLogObjectParent(mat,gvec);

  ierr = ISDestroy(tofrom); CHKERRQ(ierr);
  ierr = VecDestroy(gvec); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined (__JUNK__)

int MatGetSubMatrices_MPIDense_Local(Mat,int,IS*,IS*,MatGetSubMatrixCall,Mat*);
#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrices_MPIDense" 
int MatGetSubMatrices_MPIDense(Mat C,int ismax,IS *isrow,IS *iscol,
                             MatGetSubMatrixCall scall,Mat **submat)
{ 
  Mat_MPIDense  *c = (Mat_MPIDense *) C->data;
  int         nmax,nstages_local,nstages,i,pos,max_no,ierr;

  PetscFunctionBegin;
  /* Allocate memory to hold all the submatrices */
  if (scall != MAT_REUSE_MATRIX) {
    *submat = (Mat *)PetscMalloc((ismax+1)*sizeof(Mat));CHKPTRQ(*submat);
  }
  /* Determine the number of stages through which submatrices are done */
  nmax          = 20*1000000 / (c->N * sizeof(int));
  if (nmax == 0) nmax = 1;
  nstages_local = ismax/nmax + ((ismax % nmax)?1:0);

  /* Make sure every processor loops through the nstages */
  ierr = MPI_Allreduce(&nstages_local,&nstages,1,MPI_INT,MPI_MAX,C->comm);CHKERRQ(ierr);


  for ( i=0,pos=0; i<nstages; i++ ) {
    if (pos+nmax <= ismax) max_no = nmax;
    else if (pos == ismax) max_no = 0;
    else                   max_no = ismax-pos;
    ierr = MatGetSubMatrices_MPIDense_Local(C,max_no,isrow+pos,iscol+pos,scall,*submat+pos);CHKERRQ(ierr);
    pos += max_no;
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrices_MPIDense_Local" 
int MatGetSubMatrices_MPIDense_Local(Mat C,int ismax,IS *isrow,IS *iscol,
                             MatGetSubMatrixCall scall,Mat *submats)
{ 
  Mat_MPIDense  *c = (Mat_MPIDense *) C->data;
  Mat         A = c->A;
  Mat_SeqDense  *a = (Mat_SeqDense*)A->data, *mat;
  int         **irow,**icol,*nrow,*ncol,*w1,*w2,*w3,*w4,*rtable,start,end,size;
  int         **sbuf1,**sbuf2, rank, m,i,j,k,l,ct1,ct2,ierr, **rbuf1,row,proc;
  int         nrqs, msz, **ptr,index,*req_size,*ctr,*pa,*tmp,tcol,bsz,nrqr;
  int         **rbuf3,*req_source,**sbuf_aj, **rbuf2, max1,max2,**rmap;
  int         **cmap,**lens,is_no,ncols,*cols,mat_i,*mat_j,tmp2,jmax,*irow_i;
  int         len,ctr_j,*sbuf1_j,*sbuf_aj_i,*rbuf1_i,kmax,*cmap_i,*lens_i;
  int         *rmap_i,tag0,tag1,tag2,tag3;
  MPI_Request *s_waits1,*r_waits1,*s_waits2,*r_waits2,*r_waits3;
  MPI_Request *r_waits4,*s_waits3,*s_waits4;
  MPI_Status  *r_status1,*r_status2,*s_status1,*s_status3,*s_status2;
  MPI_Status  *r_status3,*r_status4,*s_status4;
  MPI_Comm    comm;
  Scalar      **rbuf4, **sbuf_aa, *vals, *mat_a, *sbuf_aa_i;

  PetscFunctionBegin;
  comm   = C->comm;
  tag0   = C->tag;
  size   = c->size;
  rank   = c->rank;
  m      = c->M;
  
  /* Get some new tags to keep the communication clean */
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag1); CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag2); CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)C,&tag3); CHKERRQ(ierr);

    /* Check if the col indices are sorted */
  for ( i=0; i<ismax; i++ ) {
    ierr = ISSorted(isrow[i],(PetscTruth*)&j);
    if (!j) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"ISrow is not sorted");
    ierr = ISSorted(iscol[i],(PetscTruth*)&j);
    if (!j) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"IScol is not sorted");
  }

  len    = (2*ismax+1)*(sizeof(int *) + sizeof(int)) + (m+1)*sizeof(int);
  irow   = (int **)PetscMalloc(len); CHKPTRQ(irow);
  icol   = irow + ismax;
  nrow   = (int *) (icol + ismax);
  ncol   = nrow + ismax;
  rtable = ncol + ismax;

  for ( i=0; i<ismax; i++ ) { 
    ierr = ISGetIndices(isrow[i],&irow[i]);  CHKERRQ(ierr);
    ierr = ISGetIndices(iscol[i],&icol[i]);  CHKERRQ(ierr);
    ierr = ISGetSize(isrow[i],&nrow[i]);  CHKERRQ(ierr);
    ierr = ISGetSize(iscol[i],&ncol[i]);  CHKERRQ(ierr);
  }

  /* Create hash table for the mapping :row -> proc*/
  for ( i=0,j=0; i<size; i++ ) {
    jmax = c->rowners[i+1];
    for ( ; j<jmax; j++ ) {
      rtable[j] = i;
    }
  }

  /* evaluate communication - mesg to who, length of mesg, and buffer space
     required. Based on this, buffers are allocated, and data copied into them*/
  w1     = (int *)PetscMalloc(size*4*sizeof(int));CHKPTRQ(w1); /* mesg size */
  w2     = w1 + size;      /* if w2[i] marked, then a message to proc i*/
  w3     = w2 + size;      /* no of IS that needs to be sent to proc i */
  w4     = w3 + size;      /* temp work space used in determining w1, w2, w3 */
  PetscMemzero(w1,size*3*sizeof(int)); /* initialize work vector*/
  for ( i=0; i<ismax; i++ ) { 
    PetscMemzero(w4,size*sizeof(int)); /* initialize work vector*/
    jmax   = nrow[i];
    irow_i = irow[i];
    for ( j=0; j<jmax; j++ ) {
      row  = irow_i[j];
      proc = rtable[row];
      w4[proc]++;
    }
    for ( j=0; j<size; j++ ) { 
      if (w4[j]) { w1[j] += w4[j];  w3[j]++;} 
    }
  }
  
  nrqs     = 0;              /* no of outgoing messages */
  msz      = 0;              /* total mesg length (for all procs) */
  w1[rank] = 0;              /* no mesg sent to self */
  w3[rank] = 0;
  for ( i=0; i<size; i++ ) {
    if (w1[i])  { w2[i] = 1; nrqs++;} /* there exists a message to proc i */
  }
  pa = (int *)PetscMalloc((nrqs+1)*sizeof(int));CHKPTRQ(pa); /*(proc -array)*/
  for ( i=0, j=0; i<size; i++ ) {
    if (w1[i]) { pa[j] = i; j++; }
  } 

  /* Each message would have a header = 1 + 2*(no of IS) + data */
  for ( i=0; i<nrqs; i++ ) {
    j     = pa[i];
    w1[j] += w2[j] + 2* w3[j];   
    msz   += w1[j];  
  }
  /* Do a global reduction to determine how many messages to expect*/
  {
    int *rw1, *rw2;
    rw1   = (int *)PetscMalloc(2*size*sizeof(int)); CHKPTRQ(rw1);
    rw2   = rw1+size;
    ierr  = MPI_Allreduce(w1, rw1, size, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);
    bsz   = rw1[rank];
    ierr  = MPI_Allreduce(w2, rw2, size, MPI_INT, MPI_SUM, comm);CHKERRQ(ierr);
    nrqr  = rw2[rank];
    PetscFree(rw1);
  }

  /* Allocate memory for recv buffers . Prob none if nrqr = 0 ???? */ 
  len      = (nrqr+1)*sizeof(int*) + nrqr*bsz*sizeof(int);
  rbuf1    = (int**) PetscMalloc(len);  CHKPTRQ(rbuf1);
  rbuf1[0] = (int *) (rbuf1 + nrqr);
  for ( i=1; i<nrqr; ++i ) rbuf1[i] = rbuf1[i-1] + bsz;
  
  /* Post the receives */
  r_waits1 = (MPI_Request *) PetscMalloc((nrqr+1)*sizeof(MPI_Request));CHKPTRQ(r_waits1);
  for ( i=0; i<nrqr; ++i ) {
    ierr = MPI_Irecv(rbuf1[i],bsz,MPI_INT,MPI_ANY_SOURCE,tag0,comm,r_waits1+i);CHKERRQ(ierr);
  }

  /* Allocate Memory for outgoing messages */
  len      = 2*size*sizeof(int*) + 2*msz*sizeof(int) + size*sizeof(int);
  sbuf1    = (int **)PetscMalloc(len); CHKPTRQ(sbuf1);
  ptr      = sbuf1 + size;   /* Pointers to the data in outgoing buffers */
  PetscMemzero(sbuf1,2*size*sizeof(int*));
  /* allocate memory for outgoing data + buf to receive the first reply */
  tmp      = (int *) (ptr + size);
  ctr      = tmp + 2*msz;

  {
    int *iptr = tmp,ict = 0;
    for ( i=0; i<nrqs; i++ ) {
      j         = pa[i];
      iptr     += ict;
      sbuf1[j]  = iptr;
      ict       = w1[j];
    }
  }

  /* Form the outgoing messages */
  /* Initialize the header space */
  for ( i=0; i<nrqs; i++ ) {
    j           = pa[i];
    sbuf1[j][0] = 0;
    PetscMemzero(sbuf1[j]+1, 2*w3[j]*sizeof(int));
    ptr[j]      = sbuf1[j] + 2*w3[j] + 1;
  }
  
  /* Parse the isrow and copy data into outbuf */
  for ( i=0; i<ismax; i++ ) {
    PetscMemzero(ctr,size*sizeof(int));
    irow_i = irow[i];
    jmax   = nrow[i];
    for ( j=0; j<jmax; j++ ) {  /* parse the indices of each IS */
      row  = irow_i[j];
      proc = rtable[row];
      if (proc != rank) { /* copy to the outgoing buf*/
        ctr[proc]++;
        *ptr[proc] = row;
        ptr[proc]++;
      }
    }
    /* Update the headers for the current IS */
    for ( j=0; j<size; j++ ) { /* Can Optimise this loop too */
      if ((ctr_j = ctr[j])) {
        sbuf1_j        = sbuf1[j];
        k              = ++sbuf1_j[0];
        sbuf1_j[2*k]   = ctr_j;
        sbuf1_j[2*k-1] = i;
      }
    }
  }

  /*  Now  post the sends */
  s_waits1 = (MPI_Request *) PetscMalloc((nrqs+1)*sizeof(MPI_Request));CHKPTRQ(s_waits1);
  for ( i=0; i<nrqs; ++i ) {
    j = pa[i];
    /* printf("[%d] Send Req to %d: size %d \n", rank,j, w1[j]); */
    ierr = MPI_Isend( sbuf1[j], w1[j], MPI_INT, j, tag0, comm, s_waits1+i);CHKERRQ(ierr);
  }

  /* Post Receives to capture the buffer size */
  r_waits2 = (MPI_Request *) PetscMalloc((nrqs+1)*sizeof(MPI_Request));CHKPTRQ(r_waits2);
  rbuf2    = (int**)PetscMalloc((nrqs+1)*sizeof(int *));CHKPTRQ(rbuf2);
  rbuf2[0] = tmp + msz;
  for ( i=1; i<nrqs; ++i ) {
    j        = pa[i];
    rbuf2[i] = rbuf2[i-1]+w1[pa[i-1]];
  }
  for ( i=0; i<nrqs; ++i ) {
    j    = pa[i];
    ierr = MPI_Irecv( rbuf2[i], w1[j], MPI_INT, j, tag1, comm, r_waits2+i);CHKERRQ(ierr);
  }

  /* Send to other procs the buf size they should allocate */
 

  /* Receive messages*/
  s_waits2  = (MPI_Request *) PetscMalloc((nrqr+1)*sizeof(MPI_Request));CHKPTRQ(s_waits2);
  r_status1 = (MPI_Status *) PetscMalloc((nrqr+1)*sizeof(MPI_Status));CHKPTRQ(r_status1);
  len         = 2*nrqr*sizeof(int) + (nrqr+1)*sizeof(int*);
  sbuf2       = (int**) PetscMalloc(len);CHKPTRQ(sbuf2);
  req_size    = (int *) (sbuf2 + nrqr);
  req_source  = req_size + nrqr;
 
  {
    int        *sbuf2_i;

    for ( i=0; i<nrqr; ++i ) {
      ierr = MPI_Waitany(nrqr, r_waits1, &index, r_status1+i);CHKERRQ(ierr);
      req_size[index] = 0;
      rbuf1_i         = rbuf1[index];
      start           = 2*rbuf1_i[0] + 1;
      MPI_Get_count(r_status1+i,MPI_INT, &end);
      sbuf2[index] = (int *)PetscMalloc((end+1)*sizeof(int));CHKPTRQ(sbuf2[index]);
      sbuf2_i      = sbuf2[index];
      for ( j=start; j<end; j++ ) {
        /** ncols            = sAi[id+1] - sAi[id] + sBi[id+1] - sBi[id]; */
        /** ncols is now the whole row in the dense case */
        /** Can get rid of this unnecessary communication */
        ncols            = c->N;
        sbuf2_i[j]       = ncols;
        req_size[index] += ncols;
      }
      req_source[index] = r_status1[i].MPI_SOURCE;
      /* form the header */
      sbuf2_i[0]   = req_size[index];
      for ( j=1; j<start; j++ ) { sbuf2_i[j] = rbuf1_i[j]; }
      ierr = MPI_Isend(sbuf2_i,end,MPI_INT,req_source[index],tag1,comm,s_waits2+i); CHKERRQ(ierr);
    }
  }
  PetscFree(r_status1); PetscFree(r_waits1);

  /*  recv buffer sizes */
  /* Receive messages*/
  
  rbuf3     = (int**)PetscMalloc((nrqs+1)*sizeof(int*)); CHKPTRQ(rbuf3);
  rbuf4     = (Scalar**)PetscMalloc((nrqs+1)*sizeof(Scalar*));CHKPTRQ(rbuf4);
  r_waits3  = (MPI_Request *) PetscMalloc((nrqs+1)*sizeof(MPI_Request));CHKPTRQ(r_waits3);
  r_waits4  = (MPI_Request *) PetscMalloc((nrqs+1)*sizeof(MPI_Request));CHKPTRQ(r_waits4);
  r_status2 = (MPI_Status *) PetscMalloc((nrqs+1)*sizeof(MPI_Status));CHKPTRQ(r_status2);

  for ( i=0; i<nrqs; ++i ) {
    ierr = MPI_Waitany(nrqs, r_waits2, &index, r_status2+i);CHKERRQ(ierr);
    rbuf3[index] = (int *)PetscMalloc((rbuf2[index][0]+1)*sizeof(int));CHKPTRQ(rbuf3[index]);
    rbuf4[index] = (Scalar *)PetscMalloc((rbuf2[index][0]+1)*sizeof(Scalar));CHKPTRQ(rbuf4[index]);
    ierr = MPI_Irecv(rbuf3[index],rbuf2[index][0], MPI_INT, 
              r_status2[i].MPI_SOURCE, tag2, comm, r_waits3+index); CHKERRQ(ierr);
    ierr = MPI_Irecv(rbuf4[index],rbuf2[index][0], MPIU_SCALAR, 
              r_status2[i].MPI_SOURCE, tag3, comm, r_waits4+index); CHKERRQ(ierr);
  } 
  PetscFree(r_status2); PetscFree(r_waits2);
  
  /* Wait on sends1 and sends2 */
  s_status1 = (MPI_Status *) PetscMalloc((nrqs+1)*sizeof(MPI_Status));CHKPTRQ(s_status1);
  s_status2 = (MPI_Status *) PetscMalloc((nrqr+1)*sizeof(MPI_Status));CHKPTRQ(s_status2);

  ierr = MPI_Waitall(nrqs,s_waits1,s_status1);CHKERRQ(ierr);
  ierr = MPI_Waitall(nrqr,s_waits2,s_status2);CHKERRQ(ierr);
  PetscFree(s_status1); PetscFree(s_status2);
  PetscFree(s_waits1); PetscFree(s_waits2);

  /* Now allocate buffers for a->j, and send them off */
  /** No space required for a->j... as the complete row is packed and sent */
  /** sbuf_aj = (int **)PetscMalloc((nrqr+1)*sizeof(int *));CHKPTRQ(sbuf_aj); */
  /** for ( i=0,j=0; i<nrqr; i++ ) j += req_size[i]; */
  /** sbuf_aj[0] = (int*) PetscMalloc((j+1)*sizeof(int)); CHKPTRQ(sbuf_aj[0]); */
  /** for ( i=1; i<nrqr; i++ )  sbuf_aj[i] = sbuf_aj[i-1] + req_size[i-1]; */

  s_waits3 = (MPI_Request *) PetscMalloc((nrqr+1)*sizeof(MPI_Request));CHKPTRQ(s_waits3);
  {
    int nzA, nzB, *a_i = a->i, *b_i = b->i, imark;
    int *cworkA, *cworkB, cstart = c->cstart, rstart = c->rstart, *bmap = c->garray;
    int *a_j = a->j, *b_j = b->j, ctmp, *t_cols;

    for ( i=0; i<nrqr; i++ ) {
      rbuf1_i   = rbuf1[i]; 
      sbuf_aj_i = sbuf_aj[i];
      ct1       = 2*rbuf1_i[0] + 1;
      ct2       = 0;
      for ( j=1,max1=rbuf1_i[0]; j<=max1; j++ ) { 
        kmax = rbuf1[i][2*j];
        for ( k=0; k<kmax; k++,ct1++ ) {
          row    = rbuf1_i[ct1] - rstart;
          nzA    = a_i[row+1] - a_i[row];     nzB = b_i[row+1] - b_i[row];
          ncols  = nzA + nzB;
          cworkA = a_j + a_i[row]; cworkB = b_j + b_i[row];

          /* load the column indices for this row into cols*/
          cols  = sbuf_aj_i + ct2;
          for ( l=0; l<nzB; l++ ) {
            if ((ctmp = bmap[cworkB[l]]) < cstart)  cols[l] = ctmp;
            else break;
          }
          imark = l;
          for ( l=0; l<nzA; l++ )   cols[imark+l] = cstart + cworkA[l];
          for ( l=imark; l<nzB; l++ ) cols[nzA+l] = bmap[cworkB[l]];

          ct2 += ncols;
        }
      }
      ierr = MPI_Isend(sbuf_aj_i,req_size[i],MPI_INT,req_source[i],tag2,comm,s_waits3+i);CHKERRQ(ierr);
    }
  } 
  r_status3 = (MPI_Status *) PetscMalloc((nrqs+1)*sizeof(MPI_Status));CHKPTRQ(r_status3);
  s_status3 = (MPI_Status *) PetscMalloc((nrqr+1)*sizeof(MPI_Status));CHKPTRQ(s_status3);

  /* Allocate buffers for a->a, and send them off */
  sbuf_aa = (Scalar **)PetscMalloc((nrqr+1)*sizeof(Scalar *));CHKPTRQ(sbuf_aa);
  for ( i=0,j=0; i<nrqr; i++ ) j += req_size[i];
  sbuf_aa[0] = (Scalar*) PetscMalloc((j+1)*sizeof(Scalar));CHKPTRQ(sbuf_aa[0]);
  for ( i=1; i<nrqr; i++ )  sbuf_aa[i] = sbuf_aa[i-1] + req_size[i-1];
  
  s_waits4 = (MPI_Request *) PetscMalloc((nrqr+1)*sizeof(MPI_Request));CHKPTRQ(s_waits4);
  {
    int    nzA, nzB, *a_i = a->i, *b_i = b->i,  *cworkB, imark;
    int    cstart = c->cstart, rstart = c->rstart, *bmap = c->garray;
    int    *b_j = b->j;
    Scalar *vworkA, *vworkB, *a_a = a->a, *b_a = b->a,*t_vals;
    
    for ( i=0; i<nrqr; i++ ) {
      rbuf1_i   = rbuf1[i];
      sbuf_aa_i = sbuf_aa[i];
      ct1       = 2*rbuf1_i[0]+1;
      ct2       = 0;
      for ( j=1,max1=rbuf1_i[0]; j<=max1; j++ ) {
        kmax = rbuf1_i[2*j];
        for ( k=0; k<kmax; k++,ct1++ ) {
          row    = rbuf1_i[ct1] - rstart;
          nzA    = a_i[row+1] - a_i[row];     nzB = b_i[row+1] - b_i[row];
          ncols  = nzA + nzB;
          cworkB = b_j + b_i[row];
          vworkA = a_a + a_i[row]; 
          vworkB = b_a + b_i[row];

          /* load the column values for this row into vals*/
          vals  = sbuf_aa_i+ct2;
          for ( l=0; l<nzB; l++ ) {
            if ((bmap[cworkB[l]]) < cstart)  vals[l] = vworkB[l];
            else break;
          }
          imark = l;
          for ( l=0; l<nzA; l++ )   vals[imark+l] = vworkA[l];
          for ( l=imark; l<nzB; l++ ) vals[nzA+l] = vworkB[l];
          ct2 += ncols;
        }
      }
      ierr = MPI_Isend(sbuf_aa_i,req_size[i],MPIU_SCALAR,req_source[i],tag3,comm,s_waits4+i);CHKERRQ(ierr);
    }
  } 
  r_status4 = (MPI_Status *) PetscMalloc((nrqs+1)*sizeof(MPI_Status));CHKPTRQ(r_status4);
  s_status4 = (MPI_Status *) PetscMalloc((nrqr+1)*sizeof(MPI_Status));CHKPTRQ(s_status4);
  PetscFree(rbuf1);

  /* Form the matrix */
  /* create col map */
  {
    int *icol_i;
    
    len     = (1+ismax)*sizeof(int *) + ismax*c->N*sizeof(int);
    cmap    = (int **)PetscMalloc(len); CHKPTRQ(cmap);
    cmap[0] = (int *)(cmap + ismax);
    PetscMemzero(cmap[0],(1+ismax*c->N)*sizeof(int));
    for ( i=1; i<ismax; i++ ) { cmap[i] = cmap[i-1] + c->N; }
    for ( i=0; i<ismax; i++ ) {
      jmax   = ncol[i];
      icol_i = icol[i];
      cmap_i = cmap[i];
      for ( j=0; j<jmax; j++ ) { 
        cmap_i[icol_i[j]] = j+1; 
      }
    }
  }

  /* Create lens which is required for MatCreate... */
  for ( i=0,j=0; i<ismax; i++ ) { j += nrow[i]; }
  len     = (1+ismax)*sizeof(int *) + j*sizeof(int);
  lens    = (int **)PetscMalloc(len); CHKPTRQ(lens);
  lens[0] = (int *)(lens + ismax);
  PetscMemzero(lens[0], j*sizeof(int));
  for ( i=1; i<ismax; i++ ) { lens[i] = lens[i-1] + nrow[i-1]; }
  
  /* Update lens from local data */
  for ( i=0; i<ismax; i++ ) {
    jmax   = nrow[i];
    cmap_i = cmap[i];
    irow_i = irow[i];
    lens_i = lens[i];
    for ( j=0; j<jmax; j++ ) {
      row  = irow_i[j];
      proc = rtable[row];
      if (proc == rank) {
        ierr = MatGetRow_MPIDense(C,row,&ncols,&cols,0); CHKERRQ(ierr);
        for ( k=0; k<ncols; k++ ) {
          if (cmap_i[cols[k]]) { lens_i[j]++;}
        }
        ierr = MatRestoreRow_MPIDense(C,row,&ncols,&cols,0); CHKERRQ(ierr);
      }
    }
  }
  
  /* Create row map*/
  len     = (1+ismax)*sizeof(int *) + ismax*c->M*sizeof(int);
  rmap    = (int **)PetscMalloc(len); CHKPTRQ(rmap);
  rmap[0] = (int *)(rmap + ismax);
  PetscMemzero(rmap[0],ismax*c->M*sizeof(int));
  for ( i=1; i<ismax; i++ ) { rmap[i] = rmap[i-1] + c->M;}
  for ( i=0; i<ismax; i++ ) {
    rmap_i = rmap[i];
    irow_i = irow[i];
    jmax   = nrow[i];
    for ( j=0; j<jmax; j++ ) { 
      rmap_i[irow_i[j]] = j; 
    }
  }
 
  /* Update lens from offproc data */
  {
    int *rbuf2_i, *rbuf3_i, *sbuf1_i;

    for ( tmp2=0; tmp2<nrqs; tmp2++ ) {
      ierr = MPI_Waitany(nrqs, r_waits3, &i, r_status3+tmp2);CHKERRQ(ierr);
      index   = pa[i];
      sbuf1_i = sbuf1[index];
      jmax    = sbuf1_i[0];
      ct1     = 2*jmax+1; 
      ct2     = 0;               
      rbuf2_i = rbuf2[i];
      rbuf3_i = rbuf3[i];
      for ( j=1; j<=jmax; j++ ) {
        is_no   = sbuf1_i[2*j-1];
        max1    = sbuf1_i[2*j];
        lens_i  = lens[is_no];
        cmap_i  = cmap[is_no];
        rmap_i  = rmap[is_no];
        for ( k=0; k<max1; k++,ct1++ ) {
          row  = rmap_i[sbuf1_i[ct1]]; /* the val in the new matrix to be */
          max2 = rbuf2_i[ct1];
          for ( l=0; l<max2; l++,ct2++ ) {
            if (cmap_i[rbuf3_i[ct2]]) {
              lens_i[row]++;
            }
          }
        }
      }
    }
  }    
  PetscFree(r_status3); PetscFree(r_waits3);
  ierr = MPI_Waitall(nrqr,s_waits3,s_status3); CHKERRQ(ierr);
  PetscFree(s_status3); PetscFree(s_waits3);

  /* Create the submatrices */
  if (scall == MAT_REUSE_MATRIX) {
    /*
        Assumes new rows are same length as the old rows, hence bug!
    */
    for ( i=0; i<ismax; i++ ) {
      mat = (Mat_SeqDense *)(submats[i]->data);
      if ((mat->m != nrow[i]) || (mat->n != ncol[i])) {
        SETERRQ(PETSC_ERR_ARG_SIZ,0,"Cannot reuse matrix. wrong size");
      }
      if (PetscMemcmp(mat->ilen,lens[i], mat->m *sizeof(int))) {
        SETERRQ(PETSC_ERR_ARG_SIZ,0,"Cannot reuse matrix. wrong no of nonzeros");
      }
      /* Initial matrix as if empty */
      PetscMemzero(mat->ilen,mat->m*sizeof(int));
      submats[i]->factor = C->factor;
    }
  } else {
    for ( i=0; i<ismax; i++ ) {
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,nrow[i],ncol[i],0,lens[i],submats+i);CHKERRQ(ierr);
    }
  }

  /* Assemble the matrices */
  /* First assemble the local rows */
  {
    int    ilen_row,*imat_ilen, *imat_j, *imat_i,old_row;
    Scalar *imat_a;
  
    for ( i=0; i<ismax; i++ ) {
      mat       = (Mat_SeqDense *) submats[i]->data;
      imat_ilen = mat->ilen;
      imat_j    = mat->j;
      imat_i    = mat->i;
      imat_a    = mat->a;
      cmap_i    = cmap[i];
      rmap_i    = rmap[i];
      irow_i    = irow[i];
      jmax      = nrow[i];
      for ( j=0; j<jmax; j++ ) {
        row      = irow_i[j];
        proc     = rtable[row];
        if (proc == rank) {
          old_row  = row;
          row      = rmap_i[row];
          ilen_row = imat_ilen[row];
          ierr     = MatGetRow_MPIDense(C,old_row,&ncols,&cols,&vals);CHKERRQ(ierr);
          mat_i    = imat_i[row];
          mat_a    = imat_a + mat_i;
          mat_j    = imat_j + mat_i;
          for ( k=0; k<ncols; k++ ) {
            if ((tcol = cmap_i[cols[k]])) { 
              *mat_j++ = tcol - -1;
              *mat_a++ = vals[k];
              ilen_row++;
            }
          }
          ierr = MatRestoreRow_MPIDense(C,old_row,&ncols,&cols,&vals);CHKERRQ(ierr);
          imat_ilen[row] = ilen_row; 
        }
      }
    }
  }

  /*   Now assemble the off proc rows*/
  {
    int    *sbuf1_i,*rbuf2_i,*rbuf3_i,*imat_ilen,ilen;
    int    *imat_j,*imat_i;
    Scalar *imat_a,*rbuf4_i;

    for ( tmp2=0; tmp2<nrqs; tmp2++ ) {
      ierr = MPI_Waitany(nrqs, r_waits4, &i, r_status4+tmp2);CHKERRQ(ierr);
      index   = pa[i];
      sbuf1_i = sbuf1[index];
      jmax    = sbuf1_i[0];           
      ct1     = 2*jmax + 1; 
      ct2     = 0;    
      rbuf2_i = rbuf2[i];
      rbuf3_i = rbuf3[i];
      rbuf4_i = rbuf4[i];
      for ( j=1; j<=jmax; j++ ) {
        is_no     = sbuf1_i[2*j-1];
        rmap_i    = rmap[is_no];
        cmap_i    = cmap[is_no];
        mat       = (Mat_SeqDense *) submats[is_no]->data;
        imat_ilen = mat->ilen;
        imat_j    = mat->j;
        imat_i    = mat->i;
        imat_a    = mat->a;
        max1      = sbuf1_i[2*j];
        for ( k=0; k<max1; k++, ct1++ ) {
          row   = sbuf1_i[ct1];
          row   = rmap_i[row]; 
          ilen  = imat_ilen[row];
          mat_i = imat_i[row];
          mat_a = imat_a + mat_i;
          mat_j = imat_j + mat_i;
          max2 = rbuf2_i[ct1];
          for ( l=0; l<max2; l++,ct2++ ) {
            if ((tcol = cmap_i[rbuf3_i[ct2]])) {
              *mat_j++ = tcol - 1;
              *mat_a++ = rbuf4_i[ct2];
              ilen++;
            }
          }
          imat_ilen[row] = ilen;
        }
      }
    }
  }    
  PetscFree(r_status4); PetscFree(r_waits4);
  ierr = MPI_Waitall(nrqr,s_waits4,s_status4); CHKERRQ(ierr);
  PetscFree(s_waits4); PetscFree(s_status4);

  /* Restore the indices */
  for ( i=0; i<ismax; i++ ) {
    ierr = ISRestoreIndices(isrow[i], irow+i); CHKERRQ(ierr);
    ierr = ISRestoreIndices(iscol[i], icol+i); CHKERRQ(ierr);
  }

  /* Destroy allocated memory */
  PetscFree(irow);
  PetscFree(w1);
  PetscFree(pa);

  PetscFree(sbuf1);
  PetscFree(rbuf2);
  for ( i=0; i<nrqr; ++i ) {
    PetscFree(sbuf2[i]);
  }
  for ( i=0; i<nrqs; ++i ) {
    PetscFree(rbuf3[i]);
    PetscFree(rbuf4[i]);
  }

  PetscFree(sbuf2);
  PetscFree(rbuf3);
  PetscFree(rbuf4 );
  PetscFree(sbuf_aj[0]);
  PetscFree(sbuf_aj);
  PetscFree(sbuf_aa[0]);
  PetscFree(sbuf_aa);
  
  PetscFree(cmap);
  PetscFree(rmap);
  PetscFree(lens);

  ierr = PetscObjectRestoreNewTag((PetscObject)C,&tag3); CHKERRQ(ierr);
  ierr = PetscObjectRestoreNewTag((PetscObject)C,&tag2); CHKERRQ(ierr);
  ierr = PetscObjectRestoreNewTag((PetscObject)C,&tag1); CHKERRQ(ierr);

  for ( i=0; i<ismax; i++ ) {
    ierr = MatAssemblyBegin(submats[i], MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(submats[i], MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#endif
