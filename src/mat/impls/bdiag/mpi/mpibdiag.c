#ifndef lint
static char vcid[] = "$Id: mpibdiag.c,v 1.22 1995/08/17 01:31:22 curfman Exp curfman $";
#endif

#include "mpibdiag.h"
#include "vec/vecimpl.h"
#include "inline/spops.h"

static int MatSetValues_MPIBDiag(Mat mat,int m,int *idxm,int n,
                            int *idxn,Scalar *v,InsertMode addv)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  int        ierr, i, j, row, rstart = mbd->rstart, rend = mbd->rend;

  if (mbd->insertmode != NOTSETVALUES && mbd->insertmode != addv) {
    SETERRQ(1,"MatSetValues_MPIBDiag:You cannot mix inserts and adds");
  }
  mbd->insertmode = addv;
  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) SETERRQ(1,"MatSetValues_MPIBDiag:Negative row");
    if (idxm[i] >= mbd->M) 
      SETERRQ(1,"MatSetValues_MPIBDiag:Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (idxn[j] < 0) 
          SETERRQ(1,"MatSetValues_MPIBDiag:Negative column");
        if (idxn[j] >= mbd->N) 
          SETERRQ(1,"MatSetValues_MPIBDiag:Column too large");
        ierr = MatSetValues(mbd->A,1,&row,1,&idxn[j],v+i*n+j,addv);
        CHKERRQ(ierr);
      }
    } 
    else {
      ierr = StashValues_Private(&mbd->stash,idxm[i],n,idxn,v+i*n,addv);
      CHKERRQ(ierr);
    }
  }
  return 0;
}

static int MatAssemblyBegin_MPIBDiag(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIBDiag  *mbd = (Mat_MPIBDiag *) mat->data;
  MPI_Comm    comm = mat->comm;
  int         numtids = mbd->numtids, *owners = mbd->rowners;
  int         mytid = mbd->mytid;
  MPI_Request *send_waits,*recv_waits;
  int         *nprocs,i,j,idx,*procs,nsends,nreceives,nmax,*work;
  int         tag = mat->tag, *owner,*starts,count,ierr;
  InsertMode  addv;
  Scalar      *rvalues,*svalues;

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  MPI_Allreduce((void *) &mbd->insertmode,(void *) &addv,1,MPI_INT,
                MPI_BOR,comm);
  if (addv == (ADDVALUES|INSERTVALUES)) { SETERRQ(1,
    "MatAssemblyBegin_MPIBDiag:Cannot mix adds/inserts on different procs");
    }
  mbd->insertmode = addv; /* in case this processor had no cache */

  /*  first count number of contributors to each processor */
  nprocs = (int *) PETSCMALLOC( 2*numtids*sizeof(int) ); CHKPTRQ(nprocs);
  PETSCMEMSET(nprocs,0,2*numtids*sizeof(int)); procs = nprocs + numtids;
  owner = (int *) PETSCMALLOC( (mbd->stash.n+1)*sizeof(int) ); CHKPTRQ(owner);
  for ( i=0; i<mbd->stash.n; i++ ) {
    idx = mbd->stash.idx[i];
    for ( j=0; j<numtids; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; break;
      }
    }
  }
  nsends = 0;  for ( i=0; i<numtids; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PETSCMALLOC( numtids*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce((void *) procs,(void *) work,numtids,MPI_INT,MPI_SUM,comm);
  nreceives = work[mytid]; 
  MPI_Allreduce((void *) nprocs,(void *) work,numtids,MPI_INT,MPI_MAX,comm);
  nmax = work[mytid];
  PETSCFREE(work);

  /* post receives: 
       1) each message will consist of ordered pairs 
     (global index,value) we store the global index as a double 
     to simplify the message passing. 
       2) since we don't know how long each individual message is we 
     allocate the largest needed buffer for each receive. Potentially 
     this is a lot of wasted space.

       This could be done better.
  */
  rvalues = (Scalar *) PETSCMALLOC(3*(nreceives+1)*(nmax+1)*sizeof(Scalar));
  CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PETSCMALLOC((nreceives+1)*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0; i<nreceives; i++ ) {
    MPI_Irecv((void *)(rvalues+3*nmax*i),3*nmax,MPIU_SCALAR,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (Scalar *) PETSCMALLOC( 3*(mbd->stash.n+1)*sizeof(Scalar) );
  CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PETSCMALLOC( (nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  starts = (int *) PETSCMALLOC( numtids*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<numtids; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<mbd->stash.n; i++ ) {
    svalues[3*starts[owner[i]]]       = (Scalar)  mbd->stash.idx[i];
    svalues[3*starts[owner[i]]+1]     = (Scalar)  mbd->stash.idy[i];
    svalues[3*(starts[owner[i]]++)+2] =  mbd->stash.array[i];
  }
  PETSCFREE(owner);
  starts[0] = 0;
  for ( i=1; i<numtids; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<numtids; i++ ) {
    if (procs[i]) {
      MPI_Isend((void*)(svalues+3*starts[i]),3*nprocs[i],MPIU_SCALAR,i,tag,
                comm,send_waits+count++);
    }
  }
  PETSCFREE(starts); PETSCFREE(nprocs);

  /* Free cache space */
  ierr = StashDestroy_Private(&mbd->stash); CHKERRQ(ierr);

  mbd->svalues    = svalues;    mbd->rvalues = rvalues;
  mbd->nsends     = nsends;     mbd->nrecvs = nreceives;
  mbd->send_waits = send_waits; mbd->recv_waits = recv_waits;
  mbd->rmax       = nmax;

  return 0;
}
extern int MatSetUpMultiply_MPIBDiag(Mat);

static int MatAssemblyEnd_MPIBDiag(Mat mat,MatAssemblyType mode)
{ 
  int        ierr;
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;

  MPI_Status  *send_status,recv_status;
  int         imdex,nrecvs = mbd->nrecvs, count = nrecvs, i, n;
  int         row,col;
  Scalar      *values,val;
  InsertMode  addv = mbd->insertmode;

  /*  wait on receives */
  while (count) {
    MPI_Waitany(nrecvs,mbd->recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    values = mbd->rvalues + 3*imdex*mbd->rmax;
    MPI_Get_count(&recv_status,MPIU_SCALAR,&n);
    n = n/3;
    for ( i=0; i<n; i++ ) {
      row = (int) PETSCREAL(values[3*i]) - mbd->rstart;
      col = (int) PETSCREAL(values[3*i+1]);
      val = values[3*i+2];
      if (col >= 0 && col < mbd->N) {
        MatSetValues(mbd->A,1,&row,1,&col,&val,addv);
      } 
      else {SETERRQ(1,"MatAssemblyEnd_MPIBDiag:Invalid column");}
    }
    count--;
  }
  PETSCFREE(mbd->recv_waits); PETSCFREE(mbd->rvalues);
 
  /* wait on sends */
  if (mbd->nsends) {
    send_status = (MPI_Status *) PETSCMALLOC( mbd->nsends*sizeof(MPI_Status) );
    CHKPTRQ(send_status);
    MPI_Waitall(mbd->nsends,mbd->send_waits,send_status);
    PETSCFREE(send_status);
  }
  PETSCFREE(mbd->send_waits); PETSCFREE(mbd->svalues);

  mbd->insertmode = NOTSETVALUES;
  ierr = MatAssemblyBegin(mbd->A,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mbd->A,mode); CHKERRQ(ierr);

  if (!mbd->assembled && mode == FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIBDiag(mat); CHKERRQ(ierr);
  }
  mbd->assembled = 1;
  return 0;
}

static int MatZeroEntries_MPIBDiag(Mat A)
{
  Mat_MPIBDiag *l = (Mat_MPIBDiag *) A->data;
  return MatZeroEntries(l->A);
}

/* again this uses the same basic stratagy as in the assembly and 
   scatter create routines, we should try to do it systemamatically 
   if we can figure out the proper level of generality. */

/* the code does not do the diagonal entries correctly unless the 
   matrix is square and the column and row owerships are identical.
   This is a BUG. The only way to fix it seems to be to access 
   aij->A and aij->B directly and not through the MatZeroRows() 
   routine. 
*/

static int MatZeroRows_MPIBDiag(Mat A,IS is,Scalar *diag)
{
  Mat_MPIBDiag   *l = (Mat_MPIBDiag *) A->data;
  int            i,ierr,N, *rows,*owners = l->rowners,numtids = l->numtids;
  int            *procs,*nprocs,j,found,idx,nsends,*work;
  int            nmax,*svalues,*starts,*owner,nrecvs,mytid = l->mytid;
  int            *rvalues,tag = A->tag,count,base,slen,n,*source;
  int            *lens,imdex,*lrows,*values;
  MPI_Comm       comm = A->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
  IS             istmp;

  if (!l->assembled) 
    SETERRQ(1,"MatZeroRows_MPIRowBDiag:Must assemble matrix");
  ierr = ISGetLocalSize(is,&N); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows); CHKERRQ(ierr);

  /*  first count number of contributors to each processor */
  nprocs = (int *) PETSCMALLOC( 2*numtids*sizeof(int) ); CHKPTRQ(nprocs);
  PETSCMEMSET(nprocs,0,2*numtids*sizeof(int)); procs = nprocs + numtids;
  owner = (int *) PETSCMALLOC((N+1)*sizeof(int)); CHKPTRQ(owner); /* see note*/
  for ( i=0; i<N; i++ ) {
    idx = rows[i];
    found = 0;
    for ( j=0; j<numtids; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERRQ(1,"MatZeroRows_MPIRowBDiag:row out of range");
  }
  nsends = 0;  for ( i=0; i<numtids; i++ ) {nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PETSCMALLOC( numtids*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce((void *) procs,(void *) work,numtids,MPI_INT,MPI_SUM,comm);
  nrecvs = work[mytid]; 
  MPI_Allreduce((void *) nprocs,(void *) work,numtids,MPI_INT,MPI_MAX,comm);
  nmax = work[mytid];
  PETSCFREE(work);

  /* post receives:   */
  rvalues = (int *) PETSCMALLOC((nrecvs+1)*(nmax+1)*sizeof(int)); /*see note */
  CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PETSCMALLOC((nrecvs+1)*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv((void *)(rvalues+nmax*i),nmax,MPI_INT,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (int *) PETSCMALLOC( (N+1)*sizeof(int) ); CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PETSCMALLOC( (nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  starts = (int *) PETSCMALLOC( (numtids+1)*sizeof(int) ); CHKPTRQ(starts);
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
  PETSCFREE(starts);

  base = owners[mytid];

  /*  wait on receives */
  lens = (int *) PETSCMALLOC( 2*(nrecvs+1)*sizeof(int) ); CHKPTRQ(lens);
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
  PETSCFREE(recv_waits); 
  
  /* move the data into the send scatter */
  lrows = (int *) PETSCMALLOC( (slen+1)*sizeof(int) ); CHKPTRQ(lrows);
  count = 0;
  for ( i=0; i<nrecvs; i++ ) {
    values = rvalues + i*nmax;
    for ( j=0; j<lens[i]; j++ ) {
      lrows[count++] = values[j] - base;
    }
  }
  PETSCFREE(rvalues); PETSCFREE(lens);
  PETSCFREE(owner); PETSCFREE(nprocs);
    
  /* actually zap the local rows */
  ierr = ISCreateSequential(MPI_COMM_SELF,slen,lrows,&istmp); 
  PLogObjectParent(A,istmp);
  CHKERRQ(ierr);  PETSCFREE(lrows);
  ierr = MatZeroRows(l->A,istmp,diag); CHKERRQ(ierr);
  ierr = ISDestroy(istmp); CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status *) PETSCMALLOC( nsends*sizeof(MPI_Status) );
    CHKPTRQ(send_status);
    MPI_Waitall(nsends,send_waits,send_status);
    PETSCFREE(send_status);
  }
  PETSCFREE(send_waits); PETSCFREE(svalues);

  return 0;
}

static int MatMult_MPIBDiag(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  int        ierr;
  if (!mbd->assembled) 
    SETERRQ(1,"MatMult_MPIBDiag:Must assemble matrix first");
  ierr = VecScatterBegin(xx,mbd->lvec,INSERTVALUES,SCATTERALL,mbd->Mvctx);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,mbd->lvec,INSERTVALUES,SCATTERALL,mbd->Mvctx);
  CHKERRQ(ierr);
  ierr = MatMult(mbd->A,mbd->lvec,yy); CHKERRQ(ierr);
  return 0;
}

static int MatMultAdd_MPIBDiag(Mat mat,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  int        ierr;
  if (!mbd->assembled) 
    SETERRQ(1,"MatMultAdd_MPIBDiag:Must assemble matrix first");
  ierr = VecScatterBegin(xx,mbd->lvec,ADDVALUES,SCATTERALL,mbd->Mvctx);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,mbd->lvec,ADDVALUES,SCATTERALL,mbd->Mvctx);
  CHKERRQ(ierr);
  ierr = MatMultAdd(mbd->A,mbd->lvec,yy,zz); CHKERRQ(ierr);
  return 0;
}

static int MatGetInfo_MPIBDiag(Mat matin,MatInfoType flag,int *nz,
                             int *nzalloc,int *mem)
{
  Mat_MPIBDiag *mat = (Mat_MPIBDiag *) matin->data;
  int          ierr, isend[3], irecv[3];

  ierr = MatGetInfo(mat->A,MAT_LOCAL,&isend[0],&isend[1],&isend[2]); 
  CHKERRQ(ierr);
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

static int MatGetDiagonal_MPIBDiag(Mat mat,Vec v)
{
  Mat_MPIBDiag *A = (Mat_MPIBDiag *) mat->data;
  if (!A->assembled) 
    SETERRQ(1,"MatGetDiag_MPIBDiag:Must assemble matrix first");
  return MatGetDiagonal(A->A,v);
}

static int MatDestroy_MPIBDiag(PetscObject obj)
{
  Mat          mat = (Mat) obj;
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  Mat_BDiag    *ms = (Mat_BDiag *) mbd->A->data;
  int          ierr;
#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d, BSize=%d, NDiag=%d",
                  mbd->M,mbd->N,ms->nb,ms->nd);
#endif
  PETSCFREE(mbd->rowners); 
  PETSCFREE(mbd->gdiag);
  ierr = MatDestroy(mbd->A); CHKERRQ(ierr);
  if (mbd->lvec) VecDestroy(mbd->lvec);
  if (mbd->Mvctx) VecScatterCtxDestroy(mbd->Mvctx);
  PETSCFREE(mbd); 
  PLogObjectDestroy(mat);
  PETSCHEADERDESTROY(mat);
  return 0;
}

#include "draw.h"
#include "pviewer.h"
static int MatView_MPIBDiag(PetscObject obj,Viewer viewer)
{
  Mat          mat = (Mat) obj;
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  int          ierr;
  PetscObject  vobj = (PetscObject) viewer;

  if (!mbd->assembled)
    SETERRQ(1,"MatView_MPIBDiag:Must assemble matrix first");
  if (!viewer) { /* so that viewers may be used from debuggers */
    viewer = STDOUT_VIEWER; vobj = (PetscObject) viewer;
  }
  if (vobj->cookie == DRAW_COOKIE && vobj->type == NULLWINDOW) return 0;
  if (vobj->cookie == VIEWER_COOKIE && vobj->type == FILE_VIEWER) {
    FILE *fd = ViewerFileGetPointer_Private(viewer);
    MPIU_Seq_begin(mat->comm,1);
    fprintf(fd,"[%d] rows %d starts %d ends %d cols %d\n",
             mbd->mytid,mbd->m,mbd->rstart,mbd->rend,mbd->n);
    ierr = MatView(mbd->A,viewer); CHKERRQ(ierr);
    fflush(fd);
    MPIU_Seq_end(mat->comm,1);
  }
  else if ((vobj->cookie == VIEWER_COOKIE && vobj->type == FILES_VIEWER) || 
            vobj->cookie == DRAW_COOKIE) {
    int numtids = mbd->numtids, mytid = mbd->mytid; 
    if (numtids == 1) { 
      ierr = MatView(mbd->A,viewer); CHKERRQ(ierr);
    }
    else {
      /* assemble the entire matrix onto first processor. */
      Mat       A;
      int       M = mbd->M, N = mbd->N,m,row,i, nz, *cols;
      Scalar    *vals;
      Mat_BDiag *Ambd = (Mat_BDiag*) mbd->A->data;

      if (!mytid) {
        ierr = MatCreateMPIBDiag(mat->comm,M,M,N,mbd->gnd,Ambd->nb,
                                 mbd->gdiag,0,&A); CHKERRQ(ierr);
      }
      else {
        ierr = MatCreateMPIBDiag(mat->comm,0,M,N,0,1,0,0,&A); CHKERRQ(ierr);
      }
      PLogObjectParent(mat,A);

      /* Copy the matrix ... This isn't the most efficient means,
         but it's quick for now */
      row = mbd->rstart; m = Ambd->m;
      for ( i=0; i<m; i++ ) {
        ierr = MatGetRow(mat,row,&nz,&cols,&vals); CHKERRQ(ierr);
        ierr = MatSetValues(A,1,&row,nz,cols,vals,INSERTVALUES); CHKERRQ(ierr);
        ierr = MatRestoreRow(mat,row,&nz,&cols,&vals); CHKERRQ(ierr);
        row++;
      } 

      ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
      if (!mytid) {
        ierr = MatView(((Mat_MPIBDiag*)(A->data))->A,viewer); CHKERRQ(ierr);
      }
      ierr = MatDestroy(A); CHKERRQ(ierr);
    }
  }
  return 0;
}

static int MatSetOption_MPIBDiag(Mat mat,MatOption op)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;

  if      (op == NO_NEW_NONZERO_LOCATIONS)  MatSetOption(mbd->A,op);
  else if (op == YES_NEW_NONZERO_LOCATIONS) MatSetOption(mbd->A,op);
  else if (op == COLUMN_ORIENTED) 
    SETERRQ(1,"MatSetOption_MPIBDiag:Column-oriented not supported");
  return 0;
}

static int MatGetSize_MPIBDiag(Mat mat,int *m,int *n)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  *m = mbd->M; *n = mbd->N;
  return 0;
}

static int MatGetLocalSize_MPIBDiag(Mat mat,int *m,int *n)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag *) mat->data;
  *m = mbd->m; *n = mbd->N;
  return 0;
}

static int MatGetOwnershipRange_MPIBDiag(Mat matin,int *m,int *n)
{
  Mat_MPIBDiag *mat = (Mat_MPIBDiag *) matin->data;
  *m = mat->rstart; *n = mat->rend;
  return 0;
}

static int MatGetRow_MPIBDiag(Mat matin,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIBDiag *mat = (Mat_MPIBDiag *) matin->data;
  int          lrow;
  if (!mat->assembled) 
    SETERRQ(1,"MatGetRow_MPIBDiag:Must assemble matrix first");
  if (row < mat->rstart || row >= mat->rend) 
    SETERRQ(1,"MatGetRow_MPIBDiag:you can get only local rows")
  lrow = row - mat->rstart;
  return MatGetRow(mat->A,lrow,nz,idx,v);
}

static int MatRestoreRow_MPIBDiag(Mat matin,int row,int *nz,int **idx,
                                  Scalar **v)
{
  Mat_MPIBDiag *mat = (Mat_MPIBDiag *) matin->data;
  int          lrow;
  lrow = row - mat->rstart;
  return MatRestoreRow(mat->A,lrow,nz,idx,v);
}

/* -------------------------------------------------------------------*/

static struct _MatOps MatOps = {MatSetValues_MPIBDiag,
       MatGetRow_MPIBDiag,MatRestoreRow_MPIBDiag,
       MatMult_MPIBDiag,MatMultAdd_MPIBDiag, 
       0,0,
       0,0,0,0,
       0,0,
       0,
       0,
       MatGetInfo_MPIBDiag,0,
       MatGetDiagonal_MPIBDiag,0,0,
       MatAssemblyBegin_MPIBDiag,MatAssemblyEnd_MPIBDiag,
       0,
       MatSetOption_MPIBDiag,MatZeroEntries_MPIBDiag,MatZeroRows_MPIBDiag,0,
       0,0,0,0,
       MatGetSize_MPIBDiag,MatGetLocalSize_MPIBDiag,
       MatGetOwnershipRange_MPIBDiag,
       0,0,
       0,0,
       0,0,0,
       0};

/*@
   MatCreateMPIBDiag - Creates a sparse parallel matrix in MPIBDiag format.

   Input Parameters:
.  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
.  M - number of global rows (or PETSC_DECIDE to have calculated if m is given)
.  N - number of columns (local and global)
.  nd - number of block diagonals (global)
.  nb - each element of a diagonal is an nb x nb dense matrix
.  diag - array of block diagonal numbers,
$     where for a matrix element A[i,j], 
$     where i=row and j=column, the diagonal number is
$     diag = i/nb - j/nb  (integer division)
.  diagv  - pointer to actual diagonals (in same order as diag array), 
   if allocated by user. Otherwise, set diagv=0 on input for PETSc to 
   control memory allocation.

   Output Parameter:
.  newmat - the matrix 

   Notes:
   The parallel matrix is partitioned across the processors by rows, where
   each local rectangular matrix is stored in the uniprocessor block 
   diagonal format.  See the users manual for further details.

   The user MUST specify either the local or global numbers of rows
   (possibly both).

   Currently, once the diagonals have been created, no new diagonals can
   be added.  Thus, only elements that fall on the specified diagonals
   can be set or altered; trying to modify other elements results in
   an error.

   The case nb=1 (conventional diagonal storage) is implemented as
   a special case.

.keywords: matrix, block, diagonal, parallel, sparse

.seealso: MatCreate(), MatCreateSequentialBDiag(), MatSetValues()
@*/
int MatCreateMPIBDiag(MPI_Comm comm,int m,int M,int N,int nd,int nb,
                     int *diag,Scalar **diagv,Mat *newmat)
{
  Mat          mat;
  Mat_MPIBDiag *mbd;
  Mat_BDiag    *mlocal;
  int          ierr, i, k, *ldiag;
  Scalar       **ldiagv = 0;

  *newmat       = 0;
  if ((N%nb)) 
    SETERRQ(1,"MatCreateMPIBDiag:Invalid block size - bad column number");
  PETSCHEADERCREATE(mat,_Mat,MAT_COOKIE,MATMPIBDIAG,comm);
  PLogObjectCreate(mat);
  mat->data	= (void *) (mbd = PETSCNEW(Mat_MPIBDiag)); CHKPTRQ(mbd);
  mat->ops	= &MatOps;
  mat->destroy	= MatDestroy_MPIBDiag;
  mat->view	= MatView_MPIBDiag;
  mat->factor	= 0;

  mbd->insertmode = NOTSETVALUES;
  MPI_Comm_rank(comm,&mbd->mytid);
  MPI_Comm_size(comm,&mbd->numtids);

  if (M == PETSC_DECIDE) {
    if ((m%nb)) SETERRQ(1,
       "MatCreateMPIBDiag:Invalid block size - bad local row number");
    MPI_Allreduce(&m,&M,1,MPI_INT,MPI_SUM,comm);
  }
  if (m == PETSC_DECIDE) {
    if ((M%nb)) SETERRQ(1,
      "MatCreateMPIBDiag:Invalid block size - bad global row number");
    m = M/mbd->numtids + ((M % mbd->numtids) > mbd->mytid);
    if ((m%nb)) SETERRQ(1,
       "MatCreateMPIBDiag:Invalid block size - bad local row number");
  }
  mbd->N   = N;
  mbd->M   = M;
  mbd->m   = m;
  mbd->n   = mbd->N; /* each row stores all columns */
  mbd->gnd = nd;

  /* build local table of row ownerships */
  mbd->rowners = (int *) PETSCMALLOC((mbd->numtids+2)*sizeof(int)); 
  CHKPTRQ(mbd->rowners);
  MPI_Allgather(&m,1,MPI_INT,mbd->rowners+1,1,MPI_INT,comm);
  mbd->rowners[0] = 0;
  for ( i=2; i<=mbd->numtids; i++ ) {
    mbd->rowners[i] += mbd->rowners[i-1];
  }
  mbd->rstart  = mbd->rowners[mbd->mytid]; 
  mbd->rend    = mbd->rowners[mbd->mytid+1]; 

  mbd->brstart = (mbd->rstart)/nb;
  mbd->brend   = (mbd->rend)/nb;

/*  printf("[%d] m=%d, n=%d, M=%d, N=%d, ",
    mbd->mytid, mbd->m, mbd->n, mbd->M, mbd->N ); 
  printf("[%d] rstart=%d, rend=%d, brstart=%d, brend=%d\n", mbd->mytid,
           mbd->rstart,mbd->rend,mbd->brstart, mbd->brend); */

  /* Determine local diagonals; for now, assume global rows = global cols */
  /* These are sorted in MatCreateSequentialBDiag */
  ldiag = (int *) PETSCMALLOC((nd+1)*sizeof(int)); CHKPTRQ(ldiag); 
  mbd->gdiag = (int *) PETSCMALLOC((nd+1)*sizeof(int)); CHKPTRQ(mbd->gdiag);
  k = 0;
  PLogObjectMemory(mat,(nd+1)*sizeof(int) + (mbd->numtids+2)*sizeof(int)
                        + sizeof(struct _Mat) + sizeof(Mat_MPIBDiag));
  if (diagv) {
    ldiagv = (Scalar **)PETSCMALLOC((nd+1)*sizeof(Scalar*)); CHKPTRQ(ldiagv); 
  }
  for (i=0; i<nd; i++) {
    mbd->gdiag[i] = diag[i];
    if (diag[i] > 0) { /* lower triangular */
      if (diag[i] < mbd->brend) {
        ldiag[k] = diag[i] - mbd->brstart;
        if (diagv) ldiagv[k] = diagv[i];
        k++;
      }
    } else { /* upper triangular */
      if (mbd->M/nb - diag[i] > mbd->N/nb) {
        if (mbd->M/nb + diag[i] > mbd->brstart) {
          ldiag[k] = diag[i] - mbd->brstart;
          if (diagv) ldiagv[k] = diagv[i];
          k++;
        }
      } else {
        if (mbd->M/nb > mbd->brstart) {
          ldiag[k] = diag[i] - mbd->brstart;
          if (diagv) ldiagv[k] = diagv[i];
          k++;
        }
      }
    }
  }

  /* Form local matrix */
  ierr = MatCreateSequentialBDiag(MPI_COMM_SELF,mbd->m,mbd->n,k,nb,
                                  ldiag,ldiagv,&mbd->A); CHKERRQ(ierr); 

  /* Fix main diagonal location */
  mlocal = (Mat_BDiag *) mbd->A->data;
  mlocal->mainbd = -1; 
  for (i=0; i<k; i++) {
    if (ldiag[i] + mbd->brstart == 0) mlocal->mainbd = i; 
  }
/*   for (i=0; i<nd; i++)
    printf("[%d] i=%d, diag[i]=%d, ldiag[i]=%d, mainbd=%d\n", 
         mbd->mytid,i,diag[i],ldiag[i],mlocal->mainbd); */
 
  PLogObjectParent(mat,mbd->A);
  PETSCFREE(ldiag); if (ldiagv) PETSCFREE(ldiagv);

  /* build cache for off array entries formed */
  ierr = StashBuild_Private(&mbd->stash); CHKERRQ(ierr);

  /* stuff used for matrix-vector multiply */
  mbd->lvec      = 0;
  mbd->Mvctx     = 0;
  mbd->assembled = 0;

  *newmat = mat;
  return 0;
}

/*@
   MatBDiagGetData - Gets the data for the block diagonal matrix format.
   For the parallel case, this returns information for the local submatrix.

   Input Parameters:
.  mat - the matrix, stored in block diagonal format.

   Output Parameters:
.  m - number of rows
.  n - number of columns
.  nd - number of block diagonals
.  nb - each element of a diagonal is an nb x nb dense matrix
.  bdlen - array of total block lengths of block diagonals
.  diag - array of block diagonal numbers,
$     where for a matrix element A[i,j], 
$     where i=row and j=column, the diagonal number is
$     diag = i/nb - j/nb  (integer division)
.  diagv - pointer to actual diagonals (in same order as diag array), 

   Notes:
   See the users manual for further details regarding this storage format.

.keywords: matrix, block, diagonal, get, data

.seealso: MatCreateSequentialBDiag(), MatCreateMPIBDiag()
@*/
int MatBDiagGetData(Mat mat,int *nd,int *nb,int **diag,int **bdlen,
                    Scalar ***diagv)
{
  Mat_MPIBDiag *pdmat;
  Mat_BDiag *dmat;

  PETSCVALIDHEADERSPECIFIC(mat,MAT_COOKIE);
  if (mat->type == MATBDIAG) {
    dmat = (Mat_BDiag *) mat->data;
  } else if (mat->type == MATMPIBDIAG) {
    pdmat = (Mat_MPIBDiag *) mat->data;
    dmat = (Mat_BDiag *) pdmat->A->data;
  } else SETERRQ(1,
    "MatBDiagGetData: Valid only for MATBDIAG and MATMPIBDIAG formats");
  *nd    = dmat->nd;
  *nb    = dmat->nb;
  *diag  = dmat->diag;
  *bdlen = dmat->bdlen;
  *diagv = dmat->diagv;
  return 0;
}
