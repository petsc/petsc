#ifndef lint
static char vcid[] = "$Id: mpirow.c,v 1.24 1995/05/03 17:52:48 bsmith Exp $";
#endif

#include "mpirow.h"
#include "vec/vecimpl.h"
#include "inline/spops.h"

/* It seems like much of this code could/should be shared by the AIJ and
   Row formats.  But for now we just deal with everything separately. */

#define CHUNCKSIZE   100

/*
   This is a simple minded stash. Do a linear search to determine if
 in stash, if not add to end.
*/
static int StashValues(Stash2 *stash,int row,int n, int *idxn,
                       Scalar *values,InsertMode addv)
{
  int    i,j,N = stash->n,found,*n_idx, *n_idy;
  Scalar val,*n_array;

  for ( i=0; i<n; i++ ) {
    found = 0;
    val = *values++;
    for ( j=0; j<N; j++ ) {
      if ( stash->idx[j] == row && stash->idy[j] == idxn[i]) {
        /* found a match */
        if (addv == ADDVALUES) stash->array[j] += val;
        else stash->array[j] = val;
        found = 1;
        break;
      }
    }
    if (!found) { /* not found so add to end */
      if ( stash->n == stash->nmax ) {
        /* allocate a larger stash */
        n_array = (Scalar *) MALLOC( (stash->nmax + CHUNCKSIZE)*(
                                     2*sizeof(int) + sizeof(Scalar)));
        CHKPTR(n_array);
        n_idx = (int *) (n_array + stash->nmax + CHUNCKSIZE);
        n_idy = (int *) (n_idx + stash->nmax + CHUNCKSIZE);
        MEMCPY(n_array,stash->array,stash->nmax*sizeof(Scalar));
        MEMCPY(n_idx,stash->idx,stash->nmax*sizeof(int));
        MEMCPY(n_idy,stash->idy,stash->nmax*sizeof(int));
        if (stash->array) FREE(stash->array);
        stash->array = n_array; stash->idx = n_idx; stash->idy = n_idy;
        stash->nmax += CHUNCKSIZE;
      }
      stash->array[stash->n]   = val;
      stash->idx[stash->n]     = row;
      stash->idy[stash->n++]   = idxn[i];
    }
  }
  return 0;
}

/* local utility routine that creates a mapping from the global column 
number to the local number in the off-diagonal part of the local 
storage of the matrix.  This is done in a non scable way since the 
length of colmap equals the global matrix length. 
*/
static int CreateColmap(Mat mat)
{
  Mat_MPIRow *mrow = (Mat_MPIRow *) mat->data;
  Mat_Row    *B = (Mat_Row*) mrow->B->data;
  int        n = B->n,i;
  mrow->colmap = (int *) MALLOC( mrow->N*sizeof(int) ); CHKPTR(mrow->colmap);
  MEMSET(mrow->colmap,0,mrow->N*sizeof(int));
  for ( i=0; i<n; i++ ) {
    mrow->colmap[mrow->garray[i]] = i;
/*  mrow->colmap[mrow->garray[i]] = i+1; */
  }
  return 0;
}

static int MatSetValues_MPIRow(Mat mat,int m,int *idxm,int n,
                            int *idxn,Scalar *v,InsertMode addv)
{
  Mat_MPIRow *mrow = (Mat_MPIRow *) mat->data;
  int        ierr,i,j, rstart = mrow->rstart, rend = mrow->rend;
  int        cstart = mrow->cstart, cend = mrow->cend,row,col;

  if (mrow->insertmode != NOTSETVALUES && mrow->insertmode != addv) {
    SETERR(1,"You cannot mix inserts and adds");
  }
  mrow->insertmode = addv;
  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) SETERR(1,"Negative row index");
    if (idxm[i] >= mrow->M) SETERR(1,"Row index too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (idxn[j] < 0) SETERR(1,"Negative column index");
        if (idxn[j] >= mrow->N) SETERR(1,"Column index too large");
        if (idxn[j] >= cstart && idxn[j] < cend){
          col = idxn[j] - cstart;
          ierr = MatSetValues(mrow->A,1,&row,1,&col,v+i*n+j,addv);CHKERR(ierr);
        }
        else {
          if (mrow->assembled) {
            if (!mrow->colmap) {ierr = CreateColmap(mat); CHKERR(ierr);}
            col = mrow->colmap[idxn[j]];
/*          col = mrow->colmap[idxn[j]] - 1; */
            if (col < 0) {
              SETERR(1,"Cannot insert new off diagonal block nonzero in\
                     already\
                     assembled matrix. Contact petsc-maint@mcs.anl.gov\
                     if your need this feature");
            }
          }
          else col = idxn[j];
          ierr = MatSetValues(mrow->B,1,&row,1,&col,v+i*n+j,addv);CHKERR(ierr);
        }
      }
    } 
    else {
      ierr = StashValues(&mrow->stash,idxm[i],n,idxn,v+i*n,addv);CHKERR(ierr);
    }
  }
  return 0;
}

/*
    the assembly code is alot like the code for vectors, we should 
    sometime derive a single assembly code that can be used for 
    either case.
*/

static int MatAssemblyBegin_MPIRow(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIRow  *mrow = (Mat_MPIRow *) mat->data;
  MPI_Comm    comm = mat->comm;
  int         numtids = mrow->numtids, *owners = mrow->rowners;
  int         mytid = mrow->mytid;
  MPI_Request *send_waits,*recv_waits;
  int         *nprocs,i,j,idx,*procs,nsends,nreceives,nmax,*work;
  int         tag = 50, *owner,*starts,count;
  InsertMode  addv;
  Scalar      *rvalues,*svalues;

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  MPI_Allreduce((void *) &mrow->insertmode,(void *) &addv,1,MPI_INT,
                MPI_BOR,comm);
  if (addv == (ADDVALUES|INSERTVALUES)) {
    SETERR(1,"Some processors have inserted while others have added");
  }
  mrow->insertmode = addv; /* in case this processor had no cache */

  /*  first count number of contributors to each processor */
  nprocs = (int *) MALLOC( 2*numtids*sizeof(int) ); CHKPTR(nprocs);
  MEMSET(nprocs,0,2*numtids*sizeof(int)); procs = nprocs + numtids;
  owner = (int *) MALLOC( (mrow->stash.n+1)*sizeof(int) ); CHKPTR(owner);
  for ( i=0; i<mrow->stash.n; i++ ) {
    idx = mrow->stash.idx[i];
    for ( j=0; j<numtids; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; break;
      }
    }
  }
  nsends = 0;  for ( i=0; i<numtids; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) MALLOC( numtids*sizeof(int) ); CHKPTR(work);
  MPI_Allreduce((void *) procs,(void *) work,numtids,MPI_INT,MPI_SUM,comm);
  nreceives = work[mytid]; 
  MPI_Allreduce((void *) nprocs,(void *) work,numtids,MPI_INT,MPI_MAX,comm);
  nmax = work[mytid];
  FREE(work);

  /* post receives: 
       1) each message will consist of ordered pairs 
     (global index,value) we store the global index as a double 
     to simplify the message passing. 
       2) since we don't know how long each individual message is we 
     allocate the largest needed buffer for each receive. Potentially 
     this is a lot of wasted space.


       This could be done better.
  */
  rvalues = (Scalar *) MALLOC(3*(nreceives+1)*(nmax+1)*sizeof(Scalar));
  CHKPTR(rvalues);
  recv_waits = (MPI_Request *) MALLOC((nreceives+1)*sizeof(MPI_Request));
  CHKPTR(recv_waits);
  for ( i=0; i<nreceives; i++ ) {
    MPI_Irecv((void *)(rvalues+3*nmax*i),3*nmax,MPI_SCALAR,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (Scalar *) MALLOC( 3*(mrow->stash.n+1)*sizeof(Scalar) );
  CHKPTR(svalues);
  send_waits = (MPI_Request *) MALLOC( (nsends+1)*sizeof(MPI_Request));
  CHKPTR(send_waits);
  starts = (int *) MALLOC( numtids*sizeof(int) ); CHKPTR(starts);
  starts[0] = 0; 
  for ( i=1; i<numtids; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<mrow->stash.n; i++ ) {
    svalues[3*starts[owner[i]]]       = (Scalar)  mrow->stash.idx[i];
    svalues[3*starts[owner[i]]+1]     = (Scalar)  mrow->stash.idy[i];
    svalues[3*(starts[owner[i]]++)+2] =  mrow->stash.array[i];
  }
  FREE(owner);
  starts[0] = 0;
  for ( i=1; i<numtids; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<numtids; i++ ) {
    if (procs[i]) {
      MPI_Isend((void*)(svalues+3*starts[i]),3*nprocs[i],MPI_SCALAR,i,tag,
                comm,send_waits+count++);
    }
  }
  FREE(starts); FREE(nprocs);

  /* Free cache space */
  mrow->stash.nmax = mrow->stash.n = 0;
  if (mrow->stash.array){ FREE(mrow->stash.array); mrow->stash.array = 0;}

  mrow->svalues    = svalues;       mrow->rvalues = rvalues;
  mrow->nsends     = nsends;         mrow->nrecvs = nreceives;
  mrow->send_waits = send_waits; mrow->recv_waits = recv_waits;
  mrow->rmax       = nmax;

  return 0;
}
extern int MatSetUpMultiply_MPIRow(Mat);

static int MatAssemblyEnd_MPIRow(Mat mat,MatAssemblyType mode)
{ 
  int        ierr;
  Mat_MPIRow *mrow = (Mat_MPIRow *) mat->data;

  MPI_Status  *send_status,recv_status;
  int         imdex,nrecvs = mrow->nrecvs, count = nrecvs, i, n;
  int         row,col;
  Scalar      *values,val;
  InsertMode  addv = mrow->insertmode;

  /*  wait on receives */
  while (count) {
    MPI_Waitany(nrecvs,mrow->recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    values = mrow->rvalues + 3*imdex*mrow->rmax;
    MPI_Get_count(&recv_status,MPI_SCALAR,&n);
    n = n/3;
    for ( i=0; i<n; i++ ) {
      row = (int) PETSCREAL(values[3*i]) - mrow->rstart;
      col = (int) PETSCREAL(values[3*i+1]);
      val = values[3*i+2];
      if (col >= mrow->cstart && col < mrow->cend) {
          col -= mrow->cstart;
        MatSetValues(mrow->A,1,&row,1,&col,&val,addv);
      } 
      else {
        if (mrow->assembled) {
          if (!mrow->colmap) {ierr = CreateColmap(mat); CHKERR(ierr);}
          col = mrow->colmap[col];
/*        col = mrow->colmap[col] - 1; */
          if (col < 0) {
            SETERR(1,"Cannot insert new off diagonal block nonzero in\
                     already\
                     assembled matrix. Contact petsc-maint@mcs.anl.gov\
                     if your need this feature");
          }
        }
        MatSetValues(mrow->B,1,&row,1,&col,&val,addv);
      }
    }
    count--;
  }
  FREE(mrow->recv_waits); FREE(mrow->rvalues);
 
  /* wait on sends */
  if (mrow->nsends) {
    send_status = (MPI_Status *) MALLOC( mrow->nsends*sizeof(MPI_Status) );
    CHKPTR(send_status);
    MPI_Waitall(mrow->nsends,mrow->send_waits,send_status);
    FREE(send_status);
  }
  FREE(mrow->send_waits); FREE(mrow->svalues);

  mrow->insertmode = NOTSETVALUES;
  ierr = MatAssemblyBegin(mrow->A,mode); CHKERR(ierr);
  ierr = MatAssemblyEnd(mrow->A,mode); CHKERR(ierr);

  if (!mrow->assembled && mode == FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIRow(mat); CHKERR(ierr);
  }
  ierr = MatAssemblyBegin(mrow->B,mode); CHKERR(ierr);
  ierr = MatAssemblyEnd(mrow->B,mode); CHKERR(ierr);

  mrow->assembled = 1;
  return 0;
}

static int MatZeroEntries_MPIRow(Mat A)
{
  Mat_MPIRow *l = (Mat_MPIRow *) A->data;

  MatZeroEntries(l->A); MatZeroEntries(l->B);
  return 0;
}

/* Since the parallel part of the MPIRow structure is identical to
   that of the MPIAIJ format, this code is identical to MatZeroRows_MPIAIJ.
   What a waste in duplication ... we've got to fix this. */

/* again this uses the same basic stratagy as in the assembly and 
   scatter create routines, we should try to do it systemamatically 
   if we can figure out the proper level of generality. */

/* the code does not do the diagonal entries correctly unless the 
   matrix is square and the column and row owerships are identical.
   This is a BUG. The only way to fix it seems to be to access 
   aij->A and aij->B directly and not through the MatZeroRows() 
   routine. 
*/
static int MatZeroRows_MPIRow(Mat A,IS is,Scalar *diag)
{
  Mat_MPIRow     *l = (Mat_MPIRow *) A->data;
  int            i,ierr,N, *rows,*owners = l->rowners,numtids = l->numtids;
  int            *procs,*nprocs,j,found,idx,nsends,*work;
  int            nmax,*svalues,*starts,*owner,nrecvs,mytid = l->mytid;
  int            *rvalues,tag = 67,count,base,slen,n,*source;
  int            *lens,imdex,*lrows,*values;
  MPI_Comm       comm = A->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
  IS             istmp;

  if (!l->assembled) SETERR(1,"MatZeroRows_MPIAIJ: must assemble matrix first");
  ierr = ISGetLocalSize(is,&N); CHKERR(ierr);
  ierr = ISGetIndices(is,&rows); CHKERR(ierr);

  /*  first count number of contributors to each processor */
  nprocs = (int *) MALLOC( 2*numtids*sizeof(int) ); CHKPTR(nprocs);
  MEMSET(nprocs,0,2*numtids*sizeof(int)); procs = nprocs + numtids;
  owner = (int *) MALLOC((N+1)*sizeof(int)); CHKPTR(owner); /* see note*/
  for ( i=0; i<N; i++ ) {
    idx = rows[i];
    found = 0;
    for ( j=0; j<numtids; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERR(1,"Imdex out of range");
  }
  nsends = 0;  for ( i=0; i<numtids; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) MALLOC( numtids*sizeof(int) ); CHKPTR(work);
  MPI_Allreduce((void *) procs,(void *) work,numtids,MPI_INT,MPI_SUM,comm);
  nrecvs = work[mytid]; 
  MPI_Allreduce((void *) nprocs,(void *) work,numtids,MPI_INT,MPI_MAX,comm);
  nmax = work[mytid];
  FREE(work);

  /* post receives:   */
  rvalues = (int *) MALLOC((nrecvs+1)*(nmax+1)*sizeof(int)); /*see note */
  CHKPTR(rvalues);
  recv_waits = (MPI_Request *) MALLOC((nrecvs+1)*sizeof(MPI_Request));
  CHKPTR(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv((void *)(rvalues+nmax*i),nmax,MPI_INT,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (int *) MALLOC( (N+1)*sizeof(int) ); CHKPTR(svalues);
  send_waits = (MPI_Request *) MALLOC( (nsends+1)*sizeof(MPI_Request));
  CHKPTR(send_waits);
  starts = (int *) MALLOC( (numtids+1)*sizeof(int) ); CHKPTR(starts);
  starts[0] = 0; 
  for ( i=1; i<numtids; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<N; i++ ) {
    svalues[starts[owner[i]]++] = rows[i];
  }
  ISRestoreIndices(is,&rows);

  starts[0] = 0;
  for ( i=1; i<numtids+1; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<numtids; i++ ) {
    if (procs[i]) {
      MPI_Isend((void*)(svalues+starts[i]),nprocs[i],MPI_INT,i,tag,
                comm,send_waits+count++);
    }
  }
  FREE(starts);

  base = owners[mytid];

  /*  wait on receives */
  lens = (int *) MALLOC( 2*(nrecvs+1)*sizeof(int) ); CHKPTR(lens);
  source = lens + nrecvs;
  count = nrecvs; slen = 0;
  while (count) {
    MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    MPI_Get_count(&recv_status,MPI_INT,&n);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]  = n;
    slen += n;
    count--;
  }
  FREE(recv_waits); 
  
  /* move the data into the send scatter */
  lrows = (int *) MALLOC( slen*sizeof(int) ); CHKPTR(lrows);
  count = 0;
  for ( i=0; i<nrecvs; i++ ) {
    values = rvalues + i*nmax;
    for ( j=0; j<lens[i]; j++ ) {
      lrows[count++] = values[j] - base;
    }
  }
  FREE(rvalues); FREE(lens);
  FREE(owner); FREE(nprocs);
    
  /* actually zap the local rows */
  ierr = ISCreateSequential(MPI_COMM_SELF,slen,lrows,&istmp); 
  CHKERR(ierr);  FREE(lrows);
  ierr = MatZeroRows(l->A,istmp,diag); CHKERR(ierr);
  ierr = MatZeroRows(l->B,istmp,0); CHKERR(ierr);
  ierr = ISDestroy(istmp); CHKERR(ierr);

  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status *) MALLOC( nsends*sizeof(MPI_Status) );
    CHKPTR(send_status);
    MPI_Waitall(nsends,send_waits,send_status);
    FREE(send_status);
  }
  FREE(send_waits); FREE(svalues);

  return 0;
}

static int MatMult_MPIRow(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIRow *mrow = (Mat_MPIRow *) mat->data;
  int        ierr;
  if (!mrow->assembled) SETERR(1,"MatMult_MPIRow: Must assemble matrix first.");
  ierr = VecScatterBegin(xx,0,mrow->lvec,0,INSERTVALUES,SCATTERALL,mrow->Mvctx);
  CHKERR(ierr);
  ierr = MatMult(mrow->A,xx,yy); CHKERR(ierr);
  ierr = VecScatterEnd(xx,0,mrow->lvec,0,INSERTVALUES,SCATTERALL,mrow->Mvctx);
  CHKERR(ierr);
  ierr = MatMultAdd(mrow->B,mrow->lvec,yy,yy); CHKERR(ierr);
  return 0;
}

static int MatMultAdd_MPIRow(Mat mat,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIRow *mrow = (Mat_MPIRow *) mat->data;
  int        ierr;
  if (!mrow->assembled) 
    SETERR(1,"MatMultAdd_MPIRow: Must assemble matrix first.");
  ierr = VecScatterBegin(xx,0,mrow->lvec,0,INSERTVALUES,SCATTERALL,mrow->Mvctx);
  CHKERR(ierr);
  ierr = MatMultAdd(mrow->A,xx,yy,zz); CHKERR(ierr);
  ierr = VecScatterEnd(xx,0,mrow->lvec,0,INSERTVALUES,SCATTERALL,mrow->Mvctx);
  CHKERR(ierr);
  ierr = MatMultAdd(mrow->B,mrow->lvec,zz,zz); CHKERR(ierr);
  return 0;
}

static int MatMultTrans_MPIRow(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIRow *mrow = (Mat_MPIRow *) mat->data;
  int        ierr;
  if (!mrow->assembled) 
    SETERR(1,"MatMultTrans_MPIRow: Must assemble matrix first.");
  /* Do nondiagonal part */
  ierr = MatMultTrans(mrow->B,xx,mrow->lvec); CHKERR(ierr);
  /* Send it on its way */
  ierr = VecScatterBegin(mrow->lvec,0,yy,0,ADDVALUES,
          (ScatterMode)(SCATTERALL|SCATTERREVERSE),mrow->Mvctx); CHKERR(ierr);
  /* Do local part */
  ierr = MatMultTrans(mrow->A,xx,yy); CHKERR(ierr);
  /* Receive remote parts:  note this assumes the values are not actually */
  /* inserted in yy until the next line, which is true for my implementation */
  /* but is not perhaps always true. */
  ierr = VecScatterEnd(mrow->lvec,0,yy,0,ADDVALUES,
          (ScatterMode)(SCATTERALL|SCATTERREVERSE),mrow->Mvctx); CHKERR(ierr);
  return 0;
}

static int MatMultTransAdd_MPIRow(Mat mat,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIRow *mrow = (Mat_MPIRow *) mat->data;
  int        ierr;
  if (!mrow->assembled) 
    SETERR(1,"MatMultTransAdd_MPIRow: Must assemble matrix first.");
  /* Do nondiagonal part */
  ierr = MatMultTrans(mrow->B,xx,mrow->lvec); CHKERR(ierr);
  /* Send it on its way */
  ierr = VecScatterBegin(mrow->lvec,0,zz,0,ADDVALUES,
          (ScatterMode)(SCATTERALL|SCATTERREVERSE),mrow->Mvctx); CHKERR(ierr);
  /* Do local part */
  ierr = MatMultTransAdd(mrow->A,xx,yy,zz); CHKERR(ierr);
  /* Receive remote parts:  note this assumes the values are not actually */
  /* inserted in yy until the next line, which is true for my implementation */
  /* but is not perhaps always true. */
  ierr = VecScatterEnd(mrow->lvec,0,zz,0,ADDVALUES,
          (ScatterMode)(SCATTERALL|SCATTERREVERSE),
                         mrow->Mvctx); CHKERR(ierr);
  return 0;
}

static int MatGetInfo_MPIRow(Mat matin,MatInfoType flag,int *nz,
                             int *nzalloc,int *mem)
{
  Mat_MPIRow *mat = (Mat_MPIRow *) matin->data;
  Mat        A = mat->A, B = mat->B;
  int        ierr, isend[3], irecv[3], nzA, nzallocA, memA;

  ierr = MatGetInfo(A,MAT_LOCAL,&nzA,&nzallocA,&memA); CHKERR(ierr);
  ierr = MatGetInfo(B,MAT_LOCAL,&isend[0],&isend[1],&isend[2]); CHKERR(ierr);
  isend[0] += nzA; isend[1] += nzallocA; isend[2] += memA;
  if (flag == MAT_LOCAL) {
    *nz = isend[0]; *nzalloc = isend[1]; *mem = isend[2];
  } else if (flag == MAT_GLOBAL_MAX) {
    MPI_Allreduce((void *) isend,(void *) irecv,3,MPI_INT,MPI_MAX,matin->comm);
    *nz = irecv[0]; *nzalloc = irecv[1]; *mem = irecv[2];
  } else if (flag == MAT_GLOBAL_SUM) {
    MPI_Allreduce((void *) isend,(void *) irecv,3,MPI_INT,MPI_SUM,matin->comm);
    *nz = irecv[0]; *nzalloc = irecv[1]; *mem = irecv[2];
  }
  return 0;
}

/*
  This only works correctly for square matrices where the subblock A->A is the 
   diagonal block
*/
static int MatGetDiagonal_MPIRow(Mat mat,Vec v)
{
  Mat_MPIRow *A = (Mat_MPIRow *) mat->data;
  if (!A->assembled) SETERR(1,"MatGetDiag_MPIRow: Must assemble matrix first.");
  return MatGetDiagonal(A->A,v);
}

static int MatDestroy_MPIRow(PetscObject obj)
{
  Mat        mat = (Mat) obj;
  Mat_MPIRow *mrow = (Mat_MPIRow *) mat->data;
  int        ierr;
#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows %d Cols %d",mrow->M,mrow->N);
#endif
  FREE(mrow->rowners); 
  ierr = MatDestroy(mrow->A); CHKERR(ierr);
  ierr = MatDestroy(mrow->B); CHKERR(ierr);
  if (mrow->colmap) FREE(mrow->colmap);
  if (mrow->garray) FREE(mrow->garray);
  if (mrow->lvec) VecDestroy(mrow->lvec);
  if (mrow->Mvctx) VecScatterCtxDestroy(mrow->Mvctx);
  FREE(mrow); 
  PLogObjectDestroy(mat);
  PETSCHEADERDESTROY(mat);
  return 0;
}

#include "draw.h"
#include "pviewer.h"
static int MatView_MPIRow(PetscObject obj,Viewer viewer)
{
  Mat        mat = (Mat) obj;
  Mat_MPIRow *mrow = (Mat_MPIRow *) mat->data;
  int        ierr, *cols;
  PetscObject vobj = (PetscObject) viewer;

  if (!mrow->assembled)SETERR(1,"MatView_MPIRow: Must assemble matrix first.");
  if (!viewer) { /* so that viewers may be used from debuggers */
    viewer = STDOUT_VIEWER; vobj = (PetscObject) viewer;
  }
  if (vobj->cookie == DRAW_COOKIE && vobj->type == NULLWINDOW) return 0;
  if (vobj->cookie == VIEWER_COOKIE) {
    FILE *fd = ViewerFileGetPointer_Private(viewer);
    if (vobj->type == FILE_VIEWER) {
      MPE_Seq_begin(mat->comm,1);
      fprintf(fd,"[%d] rows %d starts %d ends %d cols %d starts %d ends %d\n",
             mrow->mytid,mrow->m,mrow->rstart,mrow->rend,mrow->n,mrow->cstart,
             mrow->cend);
      ierr = MatView(mrow->A,viewer); CHKERR(ierr);
      ierr = MatView(mrow->B,viewer); CHKERR(ierr);
      fflush(fd);
      MPE_Seq_end(mat->comm,1);
    }
    else if (vobj->type == FILES_VIEWER) {
      int numtids = mrow->numtids, mytid = mrow->mytid; 
      if (numtids == 1) { 
         ierr = MatView(mrow->A,viewer); CHKERR(ierr);
      }
      else {

        /* assemble the entire matrix onto first processor. */
        Mat     A;
        Mat_Row *Amrow;
        MatiVec *vs;
        int     M = mrow->M, N = mrow->N,m,n,row,i,j;

        if (!mytid) {
          ierr = MatCreateMPIRow(mat->comm,M,N,M,N,0,0,0,0,&A);
        }
        else {
          ierr = MatCreateMPIRow(mat->comm,0,0,M,N,0,0,0,0,&A);
        }
        CHKERR(ierr);

        /* copy over the A part */
        Amrow = (Mat_Row*) mrow->A->data;
        m = Amrow->m; n = Amrow->n; row = mrow->rstart;
        for ( i=0; i<m; i++ ) {
          vs = Amrow->rs[i];
          for ( j=0; j<vs->nz; j++ ) {vs->i[j] += mrow->cstart;}
          ierr = MatSetValues(A,1,&row,vs->nz,vs->i,vs->v,INSERTVALUES);
          CHKERR(ierr);
          for ( j=0; j<vs->nz; j++ ) {vs->i[j] -= mrow->cstart;}
          row++;
        } 

        /* copy over the B part */
        Amrow = (Mat_Row*) mrow->B->data;
        m = Amrow->m; n = Amrow->n; row = mrow->rstart;
        cols = (int *) MALLOC( n * sizeof(int) ); CHKPTR(cols);
        for ( i=0; i<m; i++ ) {
          vs = Amrow->rs[i];
          for ( j=0; j<vs->nz; j++ ) {cols[j] = mrow->garray[vs->i[j]];}
          ierr = MatSetValues(A,1,&row,vs->nz,cols,vs->v,INSERTVALUES);
          CHKERR(ierr);
          row++; 
        } 
        FREE(cols);

        ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERR(ierr);
        ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERR(ierr);
        if (!mytid) {
          ierr = MatView(((Mat_MPIRow*)(A->data))->A,viewer); CHKERR(ierr);
        }
        ierr = MatDestroy(A); CHKERR(ierr);
      }
    }
  }
  return 0;
}

static int MatSetOption_MPIRow(Mat mat,MatOption op)
{
  Mat_MPIRow *mrow = (Mat_MPIRow *) mat->data;

  if      (op == NO_NEW_NONZERO_LOCATIONS)  {
    MatSetOption(mrow->A,op);
    MatSetOption(mrow->B,op);
  }
  else if (op == YES_NEW_NONZERO_LOCATIONS) {
    MatSetOption(mrow->A,op);
    MatSetOption(mrow->B,op);
  }
  else if (op == COLUMN_ORIENTED) SETERR(1,"Column oriented not supported");
  return 0;
}

static int MatGetSize_MPIRow(Mat mat,int *m,int *n)
{
  Mat_MPIRow *mrow = (Mat_MPIRow *) mat->data;
  *m = mrow->M; *n = mrow->N;
  return 0;
}

static int MatGetLocalSize_MPIRow(Mat mat,int *m,int *n)
{
  Mat_MPIRow *mrow = (Mat_MPIRow *) mat->data;
  *m = mrow->m; *n = mrow->n;
  return 0;
}

static int MatGetOwnershipRange_MPIRow(Mat matin,int *m,int *n)
{
  Mat_MPIRow *mat = (Mat_MPIRow *) matin->data;
  *m = mat->rstart; *n = mat->rend;
  return 0;
}

static int MatGetRow_MPIRow(Mat matin,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIRow *mat = (Mat_MPIRow *) matin->data;
  Scalar     *vworkA, *vworkB, **pvA, **pvB;
  int        i, ierr, *cworkA, *cworkB, **pcA, **pcB, cstart = mat->cstart;
  int        nztot, nzA, nzB, lrow, rstart = mat->rstart, rend = mat->rend;
   
  if (!mat->assembled) 
    SETERR(1,"MatGetRow_MPIRow: Must assemble matrix first.");
  if (row < rstart || row >= rend) 
    SETERR(1,"MatGetRow_MPIRow: Currently you can get only local rows.")
  lrow = row - rstart;

  pvA = &vworkA; pcA = &cworkA; pvB = &vworkB; pcB = &cworkB;
  if (!v)   {pvA = 0; pvB = 0;}
  if (!idx) {pcA = 0; if (!v) pcB = 0;}
  ierr = MatGetRow(mat->A,lrow,&nzA,pcA,pvA); CHKERR(ierr);
  ierr = MatGetRow(mat->B,lrow,&nzB,pcB,pvB); CHKERR(ierr);
  nztot = nzA + nzB;

  if (v  || idx) {
    if (nztot) {
      /* Sort by increasing column numbers, assuming A and B already sorted */
      int imark, imark2;
      for (i=0; i<nzB; i++) cworkB[i] = mat->garray[cworkB[i]];
      if (v) {
        *v = (Scalar *) MALLOC( (nztot)*sizeof(Scalar) ); CHKPTR(*v);
        for ( i=0; i<nzB; i++ ) {
          if (cworkB[i] < cstart)   (*v)[i] = vworkB[i];
          else break;
        }
        imark = i;
        for ( i=0; i<nzA; i++ )     (*v)[imark+i] = vworkA[i];
        imark2 = imark+nzA;
        for ( i=imark; i<nzB; i++ ) (*v)[imark2+i] = vworkB[i];
      }
      if (idx) {
        *idx = (int *) MALLOC( (nztot)*sizeof(int) ); CHKPTR(*idx);
        for (i=0; i<nzA; i++) cworkA[i] += cstart;
        for ( i=0; i<nzB; i++ ) {
          if (cworkB[i] < cstart)   (*idx)[i] = cworkB[i];
          else break;
        }
        imark = i;
        for ( i=0; i<nzA; i++ )     (*idx)[imark+i] = cworkA[i];
        imark2 = imark+nzA;
        for ( i=imark; i<nzB; i++ ) (*idx)[imark2+i] = cworkB[i];
      } 
    } 
    else {*idx = 0; *v=0;}
  }
  *nz = nztot;
  ierr = MatRestoreRow(mat->A,lrow,&nzA,pcA,pvA); CHKERR(ierr);
  ierr = MatRestoreRow(mat->B,lrow,&nzB,pcB,pvB); CHKERR(ierr);
  return 0;
}

static int MatRestoreRow_MPIRow(Mat mat,int row,int *nz,int **idx,Scalar **v)
{
  if (idx) FREE(*idx);
  if (v) FREE(*v);
  return 0;
}


static int MatCopy_MPIRow_Private(Mat,Mat *);

/* -------------------------------------------------------------------*/

static struct _MatOps MatOps = {MatSetValues_MPIRow,
       MatGetRow_MPIRow,MatRestoreRow_MPIRow,
       MatMult_MPIRow,MatMultAdd_MPIRow, 
       MatMultTrans_MPIRow,MatMultTransAdd_MPIRow,
       0,0,0,0,
       0,0,
       0,
       0,
       MatGetInfo_MPIRow,0,
       MatGetDiagonal_MPIRow,0,0,
       MatAssemblyBegin_MPIRow,MatAssemblyEnd_MPIRow,
       0,
       MatSetOption_MPIRow,MatZeroEntries_MPIRow,MatZeroRows_MPIRow,0,
       0,0,0,0,
       MatGetSize_MPIRow,MatGetLocalSize_MPIRow,MatGetOwnershipRange_MPIRow,
       0,0,
       0,0,
       0,0,0,
       MatCopy_MPIRow_Private};

/*@
   MatCreateMPIRow - Creates a sparse parallel matrix in MPIRow format.

   Input Parameters:
.  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
.  n - number of local columns (or PETSC_DECIDE to have calculated
           if N is given)
.  M - number of global rows (or PETSC_DECIDE to have calculated if m is given)
.  N - number of global columns (or PETSC_DECIDE to have calculated 
           if n is given)
.  d_nz - number of nonzeros per row in diagonal portion of matrix
           (same for all local rows)
.  d_nzz - number of nonzeros per row in diagonal portion of matrix or null
           (possibly different for each row).  You must leave room for the 
           diagonal entry even if it is zero.
.  o_nz - number of nonzeros per row in off-diagonal portion of matrix
           (same for all local rows)
.  o_nzz - number of nonzeros per row in off-diagonal portion of matrix
           or null (possibly different for each row).

   Output Parameter:
.  newmat - the matrix 

   Notes:
   The MPIRow format is a parallel C variant of compressed, sparse row 
   storage.  The stored row and column indices begin at zero, as in
   conventional C storage.

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   The user can set d_nz, d_nnz, o_nz, and o_nnz to zero for PETSc to
   control dynamic memory allocation.

.keywords: Mat, matrix, row, compressed row, sparse, parallel

.seealso: MatCreateSequentialRow(), MatSetValues()
@*/
int MatCreateMPIRow(MPI_Comm comm,int m,int n,int M,int N,
                 int d_nz,int *d_nnz, int o_nz,int *o_nnz,Mat *newmat)
{
  Mat          mat;
  Mat_MPIRow   *mrow;
  int          ierr, i,sum[2],work[2];
  *newmat         = 0;
  PETSCHEADERCREATE(mat,_Mat,MAT_COOKIE,MATMPIROW,comm);
  PLogObjectCreate(mat);
  mat->data	= (void *) (mrow = NEW(Mat_MPIRow)); CHKPTR(mrow);
  mat->ops	= &MatOps;
  mat->destroy	= MatDestroy_MPIRow;
  mat->view	= MatView_MPIRow;
  mat->factor	= 0;
  mat->comm	= comm;

  mrow->row	= 0;
  mrow->col	= 0;
  mrow->insertmode = NOTSETVALUES;
  MPI_Comm_rank(comm,&mrow->mytid);
  MPI_Comm_size(comm,&mrow->numtids);

  if (M == -1 || N == -1) {
    work[0] = m; work[1] = n;
    MPI_Allreduce((void *) work,(void *) sum,2,MPI_INT,MPI_SUM,comm );
    if (M == -1) M = sum[0];
    if (N == -1) N = sum[1];
  }
  if (m == -1) {m = M/mrow->numtids + ((M % mrow->numtids) > mrow->mytid);}
  if (n == -1) {n = N/mrow->numtids + ((N % mrow->numtids) > mrow->mytid);}
  mrow->m       = m;
  mrow->n       = n;
  mrow->N       = N;
  mrow->M       = M;

  /* build local table of row and column ownerships */
  mrow->rowners = (int *) MALLOC(2*(mrow->numtids+2)*sizeof(int)); 
  CHKPTR(mrow->rowners);
  mrow->cowners = mrow->rowners + mrow->numtids + 1;
  MPI_Allgather(&m,1,MPI_INT,mrow->rowners+1,1,MPI_INT,comm);
  mrow->rowners[0] = 0;
  for ( i=2; i<=mrow->numtids; i++ ) {
    mrow->rowners[i] += mrow->rowners[i-1];
  }
  mrow->rstart = mrow->rowners[mrow->mytid]; 
  mrow->rend   = mrow->rowners[mrow->mytid+1]; 
  MPI_Allgather(&n,1,MPI_INT,mrow->cowners+1,1,MPI_INT,comm);
  mrow->cowners[0] = 0;
  for ( i=2; i<=mrow->numtids; i++ ) {
    mrow->cowners[i] += mrow->cowners[i-1];
  }
  mrow->cstart = mrow->cowners[mrow->mytid]; 
  mrow->cend   = mrow->cowners[mrow->mytid+1]; 


  ierr = MatCreateSequentialRow(MPI_COMM_SELF,m,n,d_nz,d_nnz,&mrow->A); 
  CHKERR(ierr);
  PLogObjectParent(mat,mrow->A);
  ierr = MatCreateSequentialRow(MPI_COMM_SELF,m,N,o_nz,o_nnz,&mrow->B); 
  CHKERR(ierr);
  PLogObjectParent(mat,mrow->B);

  /* build cache for off array entries formed */
  mrow->stash.nmax = CHUNCKSIZE; /* completely arbratray number */
  mrow->stash.n    = 0;
  mrow->stash.array = (Scalar *) MALLOC( mrow->stash.nmax*(2*sizeof(int) +
                            sizeof(Scalar))); CHKPTR(mrow->stash.array);
  mrow->stash.idx = (int *) (mrow->stash.array + mrow->stash.nmax);
  mrow->stash.idy = (int *) (mrow->stash.idx + mrow->stash.nmax);
  mrow->colmap    = 0;
  mrow->garray    = 0;

  /* stuff used for matrix vector multiply */
  mrow->lvec      = 0;
  mrow->Mvctx     = 0;
  mrow->assembled = 0;

  *newmat = mat;
  return 0;
}

static int MatCopy_MPIRow_Private(Mat matin,Mat *newmat)
{
  Mat        mat;
  Mat_MPIRow *mrow,*oldmat = (Mat_MPIRow *) matin->data;
  int        ierr, len;

  *newmat = 0;
  if (!oldmat->assembled) SETERR(1,"Cannot copy unassembled matrix");
  PETSCHEADERCREATE(mat,_Mat,MAT_COOKIE,MATMPIROW,matin->comm);
  PLogObjectCreate(mat);
  mat->data           = (void *) (mrow = NEW(Mat_MPIRow)); CHKPTR(mrow);
  mat->ops            = &MatOps;
  mat->destroy        = MatDestroy_MPIRow;
  mat->view           = MatView_MPIRow;
  mat->factor         = matin->factor;

  mrow->row           = 0;
  mrow->col           = 0;
  mrow->m             = oldmat->m;
  mrow->n             = oldmat->n;
  mrow->M             = oldmat->M;
  mrow->N             = oldmat->N;
  mrow->assembled     = 1;
  mrow->rstart        = oldmat->rstart;
  mrow->rend          = oldmat->rend;
  mrow->cstart        = oldmat->cstart;
  mrow->cend          = oldmat->cend;
  mrow->numtids       = oldmat->numtids;
  mrow->mytid         = oldmat->mytid;
  mrow->insertmode    = NOTSETVALUES;

  mrow->rowners       = (int *) MALLOC( (mrow->numtids+1)*sizeof(int) );
  CHKPTR(mrow->rowners);
  MEMCPY(mrow->rowners,oldmat->rowners,(mrow->numtids+1)*sizeof(int));
  mrow->stash.nmax    = 0;
  mrow->stash.n       = 0;
  mrow->stash.array   = 0;
  if (oldmat->colmap) {
    mrow->colmap      = (int *) MALLOC( (mrow->N)*sizeof(int) );
    CHKPTR(mrow->colmap);
    MEMCPY(mrow->colmap,oldmat->colmap,(mrow->N)*sizeof(int));
  } else mrow->colmap = 0;
  if (oldmat->garray && (len = ((Mat_Row *) (oldmat->B->data))->n)) {
    mrow->garray    = (int *) MALLOC(len*sizeof(int) ); CHKPTR(mrow->garray);
    MEMCPY(mrow->garray,oldmat->garray,len*sizeof(int));
  } else mrow->garray = 0;
  mat->comm           = matin->comm;

  ierr =  VecDuplicate(oldmat->lvec,&mrow->lvec); CHKERR(ierr);
  PLogObjectParent(mat,mrow->lvec);
  ierr =  VecScatterCtxCopy(oldmat->Mvctx,&mrow->Mvctx); CHKERR(ierr);
  PLogObjectParent(mat,mrow->Mvctx);
  ierr =  MatConvert(oldmat->A,MATSAME,&mrow->A); CHKERR(ierr);
  PLogObjectParent(mat,mrow->A);
  ierr =  MatConvert(oldmat->B,MATSAME,&mrow->B); CHKERR(ierr);
  PLogObjectParent(mat,mrow->B);
  *newmat = mat;
  return 0;
}
