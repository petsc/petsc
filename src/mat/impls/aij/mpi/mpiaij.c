#ifndef lint
static char vcid[] = "$Id: mpiaij.c,v 1.51 1995/06/20 01:47:51 bsmith Exp bsmith $";
#endif

#include "mpiaij.h"
#include "vec/vecimpl.h"
#include "inline/spops.h"

/* local utility routine that creates a mapping from the global column 
number to the local number in the off-diagonal part of the local 
storage of the matrix.  This is done in a non scable way since the 
length of colmap equals the global matrix length. 
*/
static int CreateColmap_Private(Mat mat)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Mat_AIJ    *B = (Mat_AIJ*) aij->B->data;
  int        n = B->n,i;
  aij->colmap = (int *) PETSCMALLOC( aij->N*sizeof(int) ); CHKPTRQ(aij->colmap);
  PETSCMEMSET(aij->colmap,0,aij->N*sizeof(int));
  for ( i=0; i<n; i++ ) aij->colmap[aij->garray[i]] = i+1;
  return 0;
}

static int MatSetValues_MPIAIJ(Mat mat,int m,int *idxm,int n,
                            int *idxn,Scalar *v,InsertMode addv)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  int        ierr,i,j, rstart = aij->rstart, rend = aij->rend;
  int        cstart = aij->cstart, cend = aij->cend,row,col;

  if (aij->insertmode != NOTSETVALUES && aij->insertmode != addv) {
    SETERRQ(1,"You cannot mix inserts and adds");
  }
  aij->insertmode = addv;
  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) SETERRQ(1,"Negative row index");
    if (idxm[i] >= aij->M) SETERRQ(1,"Row index too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (idxn[j] < 0) SETERRQ(1,"Negative column index");
        if (idxn[j] >= aij->N) SETERRQ(1,"Column index too large");
        if (idxn[j] >= cstart && idxn[j] < cend){
          col = idxn[j] - cstart;
          ierr = MatSetValues(aij->A,1,&row,1,&col,v+i*n+j,addv);CHKERRQ(ierr);
        }
        else {
          if (aij->assembled) {
            if (!aij->colmap) {ierr = CreateColmap_Private(mat);CHKERRQ(ierr);}
            col = aij->colmap[idxn[j]] - 1;
            if (col < 0 && !((Mat_AIJ*)(aij->A->data))->nonew) {
              SETERRQ(1,"Cannot insert new off diagonal block nonzero in\
                     already\
                     assembled matrix. Contact petsc-maint@mcs.anl.gov\
                     if your need this feature");
            }
          }
          else col = idxn[j];
          ierr = MatSetValues(aij->B,1,&row,1,&col,v+i*n+j,addv);CHKERRQ(ierr);
        }
      }
    } 
    else {
      ierr = StashValues_Private(&aij->stash,idxm[i],n,idxn,v+i*n,addv);
      CHKERRQ(ierr);
    }
  }
  return 0;
}

/*
    the assembly code is a lot like the code for vectors, we should 
    sometime derive a single assembly code that can be used for 
    either case.
*/

static int MatAssemblyBegin_MPIAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIAIJ  *aij = (Mat_MPIAIJ *) mat->data;
  MPI_Comm    comm = mat->comm;
  int         numtids = aij->numtids, *owners = aij->rowners;
  int         mytid = aij->mytid;
  MPI_Request *send_waits,*recv_waits;
  int         *nprocs,i,j,idx,*procs,nsends,nreceives,nmax,*work;
  int         tag = mat->tag, *owner,*starts,count,ierr;
  InsertMode  addv;
  Scalar      *rvalues,*svalues;

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  MPI_Allreduce((void *) &aij->insertmode,(void *) &addv,1,MPI_INT,
                MPI_BOR,comm);
  if (addv == (ADDVALUES|INSERTVALUES)) {
    SETERRQ(1,"Some processors have inserted while others have added");
  }
  aij->insertmode = addv; /* in case this processor had no cache */

  /*  first count number of contributors to each processor */
  nprocs = (int *) PETSCMALLOC( 2*numtids*sizeof(int) ); CHKPTRQ(nprocs);
  PETSCMEMSET(nprocs,0,2*numtids*sizeof(int)); procs = nprocs + numtids;
  owner = (int *) PETSCMALLOC( (aij->stash.n+1)*sizeof(int) ); CHKPTRQ(owner);
  for ( i=0; i<aij->stash.n; i++ ) {
    idx = aij->stash.idx[i];
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
  svalues = (Scalar *) PETSCMALLOC( 3*(aij->stash.n+1)*sizeof(Scalar) );
  CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PETSCMALLOC( (nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  starts = (int *) PETSCMALLOC( numtids*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<numtids; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<aij->stash.n; i++ ) {
    svalues[3*starts[owner[i]]]       = (Scalar)  aij->stash.idx[i];
    svalues[3*starts[owner[i]]+1]     = (Scalar)  aij->stash.idy[i];
    svalues[3*(starts[owner[i]]++)+2] =  aij->stash.array[i];
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
  ierr = StashDestroy_Private(&aij->stash); CHKERRQ(ierr);

  aij->svalues    = svalues;    aij->rvalues = rvalues;
  aij->nsends     = nsends;     aij->nrecvs = nreceives;
  aij->send_waits = send_waits; aij->recv_waits = recv_waits;
  aij->rmax       = nmax;

  return 0;
}
extern int MatSetUpMultiply_MPIAIJ(Mat);

static int MatAssemblyEnd_MPIAIJ(Mat mat,MatAssemblyType mode)
{ 
  int        ierr;
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;

  MPI_Status  *send_status,recv_status;
  int         imdex,nrecvs = aij->nrecvs, count = nrecvs, i, n;
  int         row,col;
  Scalar      *values,val;
  InsertMode  addv = aij->insertmode;

  /*  wait on receives */
  while (count) {
    MPI_Waitany(nrecvs,aij->recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    values = aij->rvalues + 3*imdex*aij->rmax;
    MPI_Get_count(&recv_status,MPIU_SCALAR,&n);
    n = n/3;
    for ( i=0; i<n; i++ ) {
      row = (int) PETSCREAL(values[3*i]) - aij->rstart;
      col = (int) PETSCREAL(values[3*i+1]);
      val = values[3*i+2];
      if (col >= aij->cstart && col < aij->cend) {
          col -= aij->cstart;
        MatSetValues(aij->A,1,&row,1,&col,&val,addv);
      } 
      else {
        if (aij->assembled) {
          if (!aij->colmap) {ierr = CreateColmap_Private(mat);CHKERRQ(ierr);}
          col = aij->colmap[col] - 1;
          if (col < 0  && !((Mat_AIJ*)(aij->A->data))->nonew) {
            SETERRQ(1,"Cannot insert new off diagonal block nonzero in\
                     already\
                     assembled matrix. Contact petsc-maint@mcs.anl.gov\
                     if your need this feature");
          }
        }
        MatSetValues(aij->B,1,&row,1,&col,&val,addv);
      }
    }
    count--;
  }
  PETSCFREE(aij->recv_waits); PETSCFREE(aij->rvalues);
 
  /* wait on sends */
  if (aij->nsends) {
    send_status = (MPI_Status *) PETSCMALLOC( aij->nsends*sizeof(MPI_Status) );
    CHKPTRQ(send_status);
    MPI_Waitall(aij->nsends,aij->send_waits,send_status);
    PETSCFREE(send_status);
  }
  PETSCFREE(aij->send_waits); PETSCFREE(aij->svalues);

  aij->insertmode = NOTSETVALUES;
  ierr = MatAssemblyBegin(aij->A,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(aij->A,mode); CHKERRQ(ierr);

  if (!aij->assembled && mode == FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIAIJ(mat); CHKERRQ(ierr);
    aij->assembled = 1;
  }
  ierr = MatAssemblyBegin(aij->B,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(aij->B,mode); CHKERRQ(ierr);

  return 0;
}

static int MatZeroEntries_MPIAIJ(Mat A)
{
  Mat_MPIAIJ *l = (Mat_MPIAIJ *) A->data;
  int ierr;
  ierr = MatZeroEntries(l->A); CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B); CHKERRQ(ierr);
  return 0;
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
static int MatZeroRows_MPIAIJ(Mat A,IS is,Scalar *diag)
{
  Mat_MPIAIJ     *l = (Mat_MPIAIJ *) A->data;
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
    SETERRQ(1,"MatZeroRows_MPIAIJ: Must assemble matrix first");
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
    if (!found) SETERRQ(1,"Index out of range.");
  }
  nsends = 0;  for ( i=0; i<numtids; i++ ) { nsends += procs[i];} 

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
  CHKERRQ(ierr);  PETSCFREE(lrows);
  ierr = MatZeroRows(l->A,istmp,diag); CHKERRQ(ierr);
  ierr = MatZeroRows(l->B,istmp,0); CHKERRQ(ierr);
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

static int MatMult_MPIAIJ(Mat aijin,Vec xx,Vec yy)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) aijin->data;
  int        ierr;
  if (!aij->assembled) SETERRQ(1,"MatMult_MPIAIJ: must assemble matrix first");
  ierr = VecScatterBegin(xx,aij->lvec,INSERTVALUES,SCATTERALL,aij->Mvctx);
  CHKERRQ(ierr);
  ierr = MatMult(aij->A,xx,yy); CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,aij->lvec,INSERTVALUES,SCATTERALL,aij->Mvctx);
  CHKERRQ(ierr);
  ierr = MatMultAdd(aij->B,aij->lvec,yy,yy); CHKERRQ(ierr);
  return 0;
}

static int MatMultAdd_MPIAIJ(Mat aijin,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) aijin->data;
  int        ierr;
  if (!aij->assembled) SETERRQ(1,"MatMult_MPIAIJ: must assemble matrix first");
  ierr = VecScatterBegin(xx,aij->lvec,INSERTVALUES,SCATTERALL,aij->Mvctx);
  CHKERRQ(ierr);
  ierr = MatMultAdd(aij->A,xx,yy,zz); CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,aij->lvec,INSERTVALUES,SCATTERALL,aij->Mvctx);
  CHKERRQ(ierr);
  ierr = MatMultAdd(aij->B,aij->lvec,zz,zz); CHKERRQ(ierr);
  return 0;
}

static int MatMultTrans_MPIAIJ(Mat aijin,Vec xx,Vec yy)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) aijin->data;
  int        ierr;

  if (!aij->assembled) 
    SETERRQ(1,"MatMulTrans_MPIAIJ: must assemble matrix first");
  /* do nondiagonal part */
  ierr = MatMultTrans(aij->B,xx,aij->lvec); CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(aij->lvec,yy,ADDVALUES,
           (ScatterMode)(SCATTERALL|SCATTERREVERSE),aij->Mvctx); CHKERRQ(ierr);
  /* do local part */
  ierr = MatMultTrans(aij->A,xx,yy); CHKERRQ(ierr);
  /* receive remote parts: note this assumes the values are not actually */
  /* inserted in yy until the next line, which is true for my implementation*/
  /* but is not perhaps always true. */
  ierr = VecScatterEnd(aij->lvec,yy,ADDVALUES,
         (ScatterMode)(SCATTERALL|SCATTERREVERSE),aij->Mvctx); CHKERRQ(ierr);
  return 0;
}

static int MatMultTransAdd_MPIAIJ(Mat aijin,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) aijin->data;
  int        ierr;

  if (!aij->assembled) 
    SETERRQ(1,"MatMulTransAdd_MPIAIJ: must assemble matrix first");
  /* do nondiagonal part */
  ierr = MatMultTrans(aij->B,xx,aij->lvec); CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(aij->lvec,zz,ADDVALUES,
         (ScatterMode)(SCATTERALL|SCATTERREVERSE),aij->Mvctx); CHKERRQ(ierr);
  /* do local part */
  ierr = MatMultTransAdd(aij->A,xx,yy,zz); CHKERRQ(ierr);
  /* receive remote parts: note this assumes the values are not actually */
  /* inserted in yy until the next line, which is true for my implementation*/
  /* but is not perhaps always true. */
  ierr = VecScatterEnd(aij->lvec,zz,ADDVALUES,
       (ScatterMode)(SCATTERALL|SCATTERREVERSE),aij->Mvctx); CHKERRQ(ierr);
  return 0;
}

/*
  This only works correctly for square matrices where the subblock A->A is the 
   diagonal block
*/
static int MatGetDiagonal_MPIAIJ(Mat Ain,Vec v)
{
  Mat_MPIAIJ *A = (Mat_MPIAIJ *) Ain->data;
  if (!A->assembled) SETERRQ(1,"MatGetDiag_MPIAIJ: must assemble matrix first");
  return MatGetDiagonal(A->A,v);
}

static int MatDestroy_MPIAIJ(PetscObject obj)
{
  Mat        mat = (Mat) obj;
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  int        ierr;
#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows %d Cols %d",aij->M,aij->N);
#endif
  PETSCFREE(aij->rowners); 
  ierr = MatDestroy(aij->A); CHKERRQ(ierr);
  ierr = MatDestroy(aij->B); CHKERRQ(ierr);
  if (aij->colmap) PETSCFREE(aij->colmap);
  if (aij->garray) PETSCFREE(aij->garray);
  if (aij->lvec) VecDestroy(aij->lvec);
  if (aij->Mvctx) VecScatterCtxDestroy(aij->Mvctx);
  PETSCFREE(aij); 
  PLogObjectDestroy(mat);
  PETSCHEADERDESTROY(mat);
  return 0;
}
#include "draw.h"
#include "pviewer.h"

static int MatView_MPIAIJ(PetscObject obj,Viewer viewer)
{
  Mat        mat = (Mat) obj;
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  int        ierr;
  PetscObject vobj = (PetscObject) viewer;
 
  if (!aij->assembled) SETERRQ(1,"MatView_MPIAIJ: must assemble matrix first");
  if (!viewer) { /* so that viewers may be used from debuggers */
    viewer = STDOUT_VIEWER; vobj = (PetscObject) viewer;
  }
  if (vobj->cookie == DRAW_COOKIE && vobj->type == NULLWINDOW) return 0;
  if (vobj->cookie == VIEWER_COOKIE && vobj->type == FILE_VIEWER) {
    FILE *fd = ViewerFileGetPointer_Private(viewer);
    MPIU_Seq_begin(mat->comm,1);
    fprintf(fd,"[%d] rows %d starts %d ends %d cols %d starts %d ends %d\n",
             aij->mytid,aij->m,aij->rstart,aij->rend,aij->n,aij->cstart,
             aij->cend);
    ierr = MatView(aij->A,viewer); CHKERRQ(ierr);
    ierr = MatView(aij->B,viewer); CHKERRQ(ierr);
    fflush(fd);
    MPIU_Seq_end(mat->comm,1);
  }
  else if ((vobj->cookie == VIEWER_COOKIE && vobj->type == FILES_VIEWER) || 
            vobj->cookie == DRAW_COOKIE) {
    int numtids = aij->numtids, mytid = aij->mytid;
    if (numtids == 1) {
      ierr = MatView(aij->A,viewer); CHKERRQ(ierr);
    }
    else {
      /* assemble the entire matrix onto first processor. */
      Mat     A;
      Mat_AIJ *Aloc;
      int     M = aij->M, N = aij->N,m,*ai,*aj,row,*cols,i,*ct;
      Scalar  *a;

      if (!mytid) {
        ierr = MatCreateMPIAIJ(mat->comm,M,N,M,N,0,0,0,0,&A);
      }
      else {
        ierr = MatCreateMPIAIJ(mat->comm,0,0,M,N,0,0,0,0,&A);
      }
      CHKERRQ(ierr);

      /* copy over the A part */
      Aloc = (Mat_AIJ*) aij->A->data;
      m = Aloc->m; ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
      row = aij->rstart;
      for ( i=0; i<ai[m]-1; i++ ) {aj[i] += aij->cstart - 1;}
      for ( i=0; i<m; i++ ) {
        ierr = MatSetValues(A,1,&row,ai[i+1]-ai[i],aj,a,INSERTVALUES);
        CHKERRQ(ierr);
        row++; a += ai[i+1]-ai[i]; aj += ai[i+1]-ai[i];
      } 
      aj = Aloc->j;
      for ( i=0; i<ai[m]-1; i++ ) {aj[i] -= aij->cstart - 1;}

      /* copy over the B part */
      Aloc = (Mat_AIJ*) aij->B->data;
      m = Aloc->m;  ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
      row = aij->rstart;
      ct = cols = (int *) PETSCMALLOC( (ai[m]+1)*sizeof(int) ); CHKPTRQ(cols);
      for ( i=0; i<ai[m]-1; i++ ) {cols[i] = aij->garray[aj[i]-1];}
      for ( i=0; i<m; i++ ) {
        ierr = MatSetValues(A,1,&row,ai[i+1]-ai[i],cols,a,INSERTVALUES);
        CHKERRQ(ierr);
        row++; a += ai[i+1]-ai[i]; cols += ai[i+1]-ai[i];
      } 
      PETSCFREE(ct);
      ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
      if (!mytid) {
        ierr = MatView(((Mat_MPIAIJ*)(A->data))->A,viewer); CHKERRQ(ierr);
      }
      ierr = MatDestroy(A); CHKERRQ(ierr);
    }
  }
  return 0;
}

extern int MatMarkDiag_AIJ(Mat_AIJ  *);
/*
    This has to provide several versions.

     1) per sequential 
     2) a) use only local smoothing updating outer values only once.
        b) local smoothing updating outer values each inner iteration
     3) color updating out values betwen colors.
*/
static int MatRelax_MPIAIJ(Mat matin,Vec bb,double omega,MatSORType flag,
                           double shift,int its,Vec xx)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;
  Mat        AA = mat->A, BB = mat->B;
  Mat_AIJ    *A = (Mat_AIJ *) AA->data, *B = (Mat_AIJ *)BB->data;
  Scalar     zero = 0.0,*b,*x,*xs,*ls,d,*v,sum,scale,*t,*ts;
  int        ierr,*idx, *diag;
  int        n = mat->n, m = mat->m, i;
  Vec        tt;

  if (!mat->assembled) SETERRQ(1,"MatRelax_MPIAIJ: must assemble matrix first");

  VecGetArray(xx,&x); VecGetArray(bb,&b); VecGetArray(mat->lvec,&ls);
  xs = x -1; /* shift by one for index start of 1 */
  ls--;
  if (!A->diag) {if ((ierr = MatMarkDiag_AIJ(A))) return ierr;}
  diag = A->diag;
  if (flag == SOR_APPLY_UPPER || flag == SOR_APPLY_LOWER) {
    SETERRQ(1,"That option not yet support for parallel AIJ matrices");
  }
  if (flag & SOR_EISENSTAT) {
    /* Let  A = L + U + D; where L is lower trianglar,
    U is upper triangular, E is diagonal; This routine applies

            (L + E)^{-1} A (U + E)^{-1}

    to a vector efficiently using Eisenstat's trick. This is for
    the case of SSOR preconditioner, so E is D/omega where omega
    is the relaxation factor.
    */
    ierr = VecDuplicate(xx,&tt); CHKERRQ(ierr);
    VecGetArray(tt,&t);
    scale = (2.0/omega) - 1.0;
    /*  x = (E + U)^{-1} b */
    VecSet(&zero,mat->lvec);
    ierr = VecPipelineBegin(xx,mat->lvec,INSERTVALUES,PIPELINEUP,
                              mat->Mvctx); CHKERRQ(ierr);
    for ( i=m-1; i>-1; i-- ) {
      n    = A->i[i+1] - diag[i] - 1;
      idx  = A->j + diag[i];
      v    = A->a + diag[i];
      sum  = b[i];
      SPARSEDENSEMDOT(sum,xs,v,idx,n); 
      d    = shift + A->a[diag[i]-1];
      n    = B->i[i+1] - B->i[i]; 
      idx  = B->j + B->i[i] - 1;
      v    = B->a + B->i[i] - 1;
      SPARSEDENSEMDOT(sum,ls,v,idx,n); 
      x[i] = omega*(sum/d);
    }
    ierr = VecPipelineEnd(xx,mat->lvec,INSERTVALUES,PIPELINEUP,
                            mat->Mvctx); CHKERRQ(ierr);

    /*  t = b - (2*E - D)x */
    v = A->a;
    for ( i=0; i<m; i++ ) { t[i] = b[i] - scale*(v[*diag++ - 1])*x[i]; }

    /*  t = (E + L)^{-1}t */
    ts = t - 1; /* shifted by one for index start of a or mat->j*/
    diag = A->diag;
    VecSet(&zero,mat->lvec);
    ierr = VecPipelineBegin(tt,mat->lvec,INSERTVALUES,PIPELINEDOWN,
                                                 mat->Mvctx); CHKERRQ(ierr);
    for ( i=0; i<m; i++ ) {
      n    = diag[i] - A->i[i]; 
      idx  = A->j + A->i[i] - 1;
      v    = A->a + A->i[i] - 1;
      sum  = t[i];
      SPARSEDENSEMDOT(sum,ts,v,idx,n); 
      d    = shift + A->a[diag[i]-1];
      n    = B->i[i+1] - B->i[i]; 
      idx  = B->j + B->i[i] - 1;
      v    = B->a + B->i[i] - 1;
      SPARSEDENSEMDOT(sum,ls,v,idx,n); 
      t[i] = omega*(sum/d);
    }
    ierr = VecPipelineEnd(tt,mat->lvec,INSERTVALUES,PIPELINEDOWN,
                                                    mat->Mvctx); CHKERRQ(ierr);
    /*  x = x + t */
    for ( i=0; i<m; i++ ) { x[i] += t[i]; }
    VecDestroy(tt);
    return 0;
  }


  if ((flag & SOR_SYMMETRIC_SWEEP) == SOR_SYMMETRIC_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      VecSet(&zero,mat->lvec); VecSet(&zero,xx);
    }
    else {
      ierr=VecScatterBegin(xx,mat->lvec,INSERTVALUES,SCATTERUP,mat->Mvctx);
      CHKERRQ(ierr);
      ierr = VecScatterEnd(xx,mat->lvec,INSERTVALUES,SCATTERUP,mat->Mvctx);
      CHKERRQ(ierr);
    }
    while (its--) {
      /* go down through the rows */
      ierr = VecPipelineBegin(xx,mat->lvec,INSERTVALUES,PIPELINEDOWN,
                              mat->Mvctx); CHKERRQ(ierr);
      for ( i=0; i<m; i++ ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] - 1;
        v    = A->a + A->i[i] - 1;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = shift + A->a[diag[i]-1];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] - 1;
        v    = B->a + B->i[i] - 1;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
      ierr = VecPipelineEnd(xx,mat->lvec,INSERTVALUES,PIPELINEDOWN,
                            mat->Mvctx); CHKERRQ(ierr);
      /* come up through the rows */
      ierr = VecPipelineBegin(xx,mat->lvec,INSERTVALUES,PIPELINEUP,
                              mat->Mvctx); CHKERRQ(ierr);
      for ( i=m-1; i>-1; i-- ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] - 1;
        v    = A->a + A->i[i] - 1;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = shift + A->a[diag[i]-1];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] - 1;
        v    = B->a + B->i[i] - 1;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
      ierr = VecPipelineEnd(xx,mat->lvec,INSERTVALUES,PIPELINEUP,
                            mat->Mvctx); CHKERRQ(ierr);
    }    
  }
  else if (flag & SOR_FORWARD_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      VecSet(&zero,mat->lvec);
      ierr = VecPipelineBegin(xx,mat->lvec,INSERTVALUES,PIPELINEDOWN,
                              mat->Mvctx); CHKERRQ(ierr);
      for ( i=0; i<m; i++ ) {
        n    = diag[i] - A->i[i]; 
        idx  = A->j + A->i[i] - 1;
        v    = A->a + A->i[i] - 1;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = shift + A->a[diag[i]-1];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] - 1;
        v    = B->a + B->i[i] - 1;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = omega*(sum/d);
      }
      ierr = VecPipelineEnd(xx,mat->lvec,INSERTVALUES,PIPELINEDOWN,
                            mat->Mvctx); CHKERRQ(ierr);
      its--;
    }
    while (its--) {
      ierr=VecScatterBegin(xx,mat->lvec,INSERTVALUES,SCATTERUP,mat->Mvctx);
      CHKERRQ(ierr);
      ierr = VecScatterEnd(xx,mat->lvec,INSERTVALUES,SCATTERUP,mat->Mvctx);
      CHKERRQ(ierr);
      ierr = VecPipelineBegin(xx,mat->lvec,INSERTVALUES,PIPELINEDOWN,
                              mat->Mvctx); CHKERRQ(ierr);
      for ( i=0; i<m; i++ ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] - 1;
        v    = A->a + A->i[i] - 1;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = shift + A->a[diag[i]-1];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] - 1;
        v    = B->a + B->i[i] - 1;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
      ierr = VecPipelineEnd(xx,mat->lvec,INSERTVALUES,PIPELINEDOWN,
                            mat->Mvctx); CHKERRQ(ierr);
    } 
  }
  else if (flag & SOR_BACKWARD_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      VecSet(&zero,mat->lvec);
      ierr = VecPipelineBegin(xx,mat->lvec,INSERTVALUES,PIPELINEUP,
                              mat->Mvctx); CHKERRQ(ierr);
      for ( i=m-1; i>-1; i-- ) {
        n    = A->i[i+1] - diag[i] - 1; 
        idx  = A->j + diag[i];
        v    = A->a + diag[i];
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = shift + A->a[diag[i]-1];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] - 1;
        v    = B->a + B->i[i] - 1;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = omega*(sum/d);
      }
      ierr = VecPipelineEnd(xx,mat->lvec,INSERTVALUES,PIPELINEUP,
                            mat->Mvctx); CHKERRQ(ierr);
      its--;
    }
    while (its--) {
      ierr = VecScatterBegin(xx,mat->lvec,INSERTVALUES,SCATTERDOWN,
                            mat->Mvctx); CHKERRQ(ierr);
      ierr = VecScatterEnd(xx,mat->lvec,INSERTVALUES,SCATTERDOWN,
                            mat->Mvctx); CHKERRQ(ierr);
      ierr = VecPipelineBegin(xx,mat->lvec,INSERTVALUES,PIPELINEUP,
                              mat->Mvctx); CHKERRQ(ierr);
      for ( i=m-1; i>-1; i-- ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] - 1;
        v    = A->a + A->i[i] - 1;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = shift + A->a[diag[i]-1];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] - 1;
        v    = B->a + B->i[i] - 1;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
      ierr = VecPipelineEnd(xx,mat->lvec,INSERTVALUES,PIPELINEUP,
                            mat->Mvctx); CHKERRQ(ierr);
    } 
  }
  else if ((flag & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      return MatRelax(mat->A,bb,omega,flag,shift,its,xx);
    }
    ierr=VecScatterBegin(xx,mat->lvec,INSERTVALUES,SCATTERALL,mat->Mvctx);
    CHKERRQ(ierr);
    ierr = VecScatterEnd(xx,mat->lvec,INSERTVALUES,SCATTERALL,mat->Mvctx);
    CHKERRQ(ierr);
    while (its--) {
      /* go down through the rows */
      for ( i=0; i<m; i++ ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] - 1;
        v    = A->a + A->i[i] - 1;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = shift + A->a[diag[i]-1];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] - 1;
        v    = B->a + B->i[i] - 1;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
      /* come up through the rows */
      for ( i=m-1; i>-1; i-- ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] - 1;
        v    = A->a + A->i[i] - 1;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = shift + A->a[diag[i]-1];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] - 1;
        v    = B->a + B->i[i] - 1;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
    }    
  }
  else if (flag & SOR_LOCAL_FORWARD_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      return MatRelax(mat->A,bb,omega,flag,shift,its,xx);
    }
    ierr=VecScatterBegin(xx,mat->lvec,INSERTVALUES,SCATTERALL,mat->Mvctx);
    CHKERRQ(ierr);
    ierr = VecScatterEnd(xx,mat->lvec,INSERTVALUES,SCATTERALL,mat->Mvctx);
    CHKERRQ(ierr);
    while (its--) {
      for ( i=0; i<m; i++ ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] - 1;
        v    = A->a + A->i[i] - 1;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = shift + A->a[diag[i]-1];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] - 1;
        v    = B->a + B->i[i] - 1;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
    } 
  }
  else if (flag & SOR_LOCAL_BACKWARD_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      return MatRelax(mat->A,bb,omega,flag,shift,its,xx);
    }
    ierr = VecScatterBegin(xx,mat->lvec,INSERTVALUES,SCATTERALL,
                            mat->Mvctx); CHKERRQ(ierr);
    ierr = VecScatterEnd(xx,mat->lvec,INSERTVALUES,SCATTERALL,
                            mat->Mvctx); CHKERRQ(ierr);
    while (its--) {
      for ( i=m-1; i>-1; i-- ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] - 1;
        v    = A->a + A->i[i] - 1;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = shift + A->a[diag[i]-1];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] - 1;
        v    = B->a + B->i[i] - 1;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
    } 
  }
  return 0;
} 

static int MatGetInfo_MPIAIJ(Mat matin,MatInfoType flag,int *nz,
                             int *nzalloc,int *mem)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;
  Mat        A = mat->A, B = mat->B;
  int        ierr, isend[3], irecv[3], nzA, nzallocA, memA;

  ierr = MatGetInfo(A,MAT_LOCAL,&nzA,&nzallocA,&memA); CHKERRQ(ierr);
  ierr = MatGetInfo(B,MAT_LOCAL,&isend[0],&isend[1],&isend[2]); CHKERRQ(ierr);
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

static int MatSetOption_MPIAIJ(Mat aijin,MatOption op)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) aijin->data;

  if      (op == NO_NEW_NONZERO_LOCATIONS)  {
    MatSetOption(aij->A,op);
    MatSetOption(aij->B,op);
  }
  else if (op == YES_NEW_NONZERO_LOCATIONS) {
    MatSetOption(aij->A,op);
    MatSetOption(aij->B,op);
  }
  else if (op == COLUMN_ORIENTED) SETERRQ(1,"Column oriented not supported");
  return 0;
}

static int MatGetSize_MPIAIJ(Mat matin,int *m,int *n)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;
  *m = mat->M; *n = mat->N;
  return 0;
}

static int MatGetLocalSize_MPIAIJ(Mat matin,int *m,int *n)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;
  *m = mat->m; *n = mat->N;
  return 0;
}

static int MatGetOwnershipRange_MPIAIJ(Mat matin,int *m,int *n)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;
  *m = mat->rstart; *n = mat->rend;
  return 0;
}

static int MatGetRow_MPIAIJ(Mat matin,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;
  Scalar     *vworkA, *vworkB, **pvA, **pvB;
  int        i, ierr, *cworkA, *cworkB, **pcA, **pcB, cstart = mat->cstart;
  int        nztot, nzA, nzB, lrow, rstart = mat->rstart, rend = mat->rend;

  if (!mat->assembled) 
    SETERRQ(1,"MatGetRow_MPIAIJ: Must assemble matrix first.");
  if (row < rstart || row >= rend) 
    SETERRQ(1,"MatGetRow_MPIAIJ: Currently you can get only local rows.")
  lrow = row - rstart;

  pvA = &vworkA; pcA = &cworkA; pvB = &vworkB; pcB = &cworkB;
  if (!v)   {pvA = 0; pvB = 0;}
  if (!idx) {pcA = 0; if (!v) pcB = 0;}
  ierr = MatGetRow(mat->A,lrow,&nzA,pcA,pvA); CHKERRQ(ierr);
  ierr = MatGetRow(mat->B,lrow,&nzB,pcB,pvB); CHKERRQ(ierr);
  nztot = nzA + nzB;

  if (v  || idx) {
    if (nztot) {
      /* Sort by increasing column numbers, assuming A and B already sorted */
      int imark, imark2;
      for (i=0; i<nzB; i++) cworkB[i] = mat->garray[cworkB[i]];
      if (v) {
        *v = (Scalar *) PETSCMALLOC( (nztot)*sizeof(Scalar) ); CHKPTRQ(*v);
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
        *idx = (int *) PETSCMALLOC( (nztot)*sizeof(int) ); CHKPTRQ(*idx);
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
  ierr = MatRestoreRow(mat->A,lrow,&nzA,pcA,pvA); CHKERRQ(ierr);
  ierr = MatRestoreRow(mat->B,lrow,&nzB,pcB,pvB); CHKERRQ(ierr);
  return 0;
}

static int MatRestoreRow_MPIAIJ(Mat mat,int row,int *nz,int **idx,Scalar **v)
{
  if (idx) PETSCFREE(*idx);
  if (v) PETSCFREE(*v);
  return 0;
}

static int MatTranspose_MPIAIJ(Mat A,Mat *Bin)
{ 
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;
  Mat        B;
  Mat_AIJ    *Aloc;
  int        M = a->M, N = a->N,m,*ai,*aj,row,*cols,i,*ct;
  Scalar     *array;

  ierr = MatCreateMPIAIJ(A->comm,PETSC_DECIDE,PETSC_DECIDE,N,M,0,0,0,0,&B);
  CHKERRQ(ierr);

  /* copy over the A part */
  Aloc = (Mat_AIJ*) a->A->data;
  m = Aloc->m; ai = Aloc->i; aj = Aloc->j; array = Aloc->a;
  row = a->rstart;
  for ( i=0; i<ai[m]-1; i++ ) {aj[i] += a->cstart - 1;}
  for ( i=0; i<m; i++ ) {
      ierr = MatSetValues(B,ai[i+1]-ai[i],aj,1,&row,array,INSERTVALUES);
      CHKERRQ(ierr);
      row++; array += ai[i+1]-ai[i]; aj += ai[i+1]-ai[i];
  } 
  aj = Aloc->j;
  for ( i=0; i<ai[m]-1; i++ ) {aj[i] -= a->cstart - 1;}

  /* copy over the B part */
  Aloc = (Mat_AIJ*) a->B->data;
  m = Aloc->m;  ai = Aloc->i; aj = Aloc->j; array = Aloc->a;
  row = a->rstart;
  ct = cols = (int *) PETSCMALLOC( (ai[m]+1)*sizeof(int) ); CHKPTRQ(cols);
  for ( i=0; i<ai[m]-1; i++ ) {cols[i] = a->garray[aj[i]-1];}
  for ( i=0; i<m; i++ ) {
    ierr = MatSetValues(B,ai[i+1]-ai[i],cols,1,&row,array,INSERTVALUES);
    CHKERRQ(ierr);
    row++; array += ai[i+1]-ai[i]; cols += ai[i+1]-ai[i];
  } 
  PETSCFREE(ct);
  ierr = MatAssemblyBegin(B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  *Bin = B;
  return 0;
}

extern int MatConvert_MPIAIJ(Mat,MatType,Mat *);
static int MatCopyPrivate_MPIAIJ(Mat,Mat *);

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {MatSetValues_MPIAIJ,
       MatGetRow_MPIAIJ,MatRestoreRow_MPIAIJ,
       MatMult_MPIAIJ,MatMultAdd_MPIAIJ,
       MatMultTrans_MPIAIJ,MatMultTransAdd_MPIAIJ,
       0,0,0,0,
       0,0,
       MatRelax_MPIAIJ,
       MatTranspose_MPIAIJ,
       MatGetInfo_MPIAIJ,0,
       MatGetDiagonal_MPIAIJ,0,0,
       MatAssemblyBegin_MPIAIJ,MatAssemblyEnd_MPIAIJ,
       0,
       MatSetOption_MPIAIJ,MatZeroEntries_MPIAIJ,MatZeroRows_MPIAIJ,0,
       0,0,0,0,
       MatGetSize_MPIAIJ,MatGetLocalSize_MPIAIJ,MatGetOwnershipRange_MPIAIJ,
       0,0,
       0,0,MatConvert_MPIAIJ,0,0,MatCopyPrivate_MPIAIJ};

/*@
   MatCreateMPIAIJ - Creates a sparse parallel matrix in AIJ format
   (the default parallel PETSc format).

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
   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices begin at 
   one, not zero.

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   The user can set d_nz, d_nnz, o_nz, and o_nnz to zero for PETSc to
   control dynamic memory allocation.

.keywords: matrix, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSequentialAIJ(), MatSetValues()
@*/
int MatCreateMPIAIJ(MPI_Comm comm,int m,int n,int M,int N,
                 int d_nz,int *d_nnz, int o_nz,int *o_nnz,Mat *newmat)
{
  Mat          mat;
  Mat_MPIAIJ   *aij;
  int          ierr, i,sum[2],work[2];
  *newmat         = 0;
  PETSCHEADERCREATE(mat,_Mat,MAT_COOKIE,MATMPIAIJ,comm);
  PLogObjectCreate(mat);
  mat->data       = (void *) (aij = PETSCNEW(Mat_MPIAIJ)); CHKPTRQ(aij);
  mat->ops        = &MatOps;
  mat->destroy    = MatDestroy_MPIAIJ;
  mat->view       = MatView_MPIAIJ;
  mat->factor     = 0;

  aij->insertmode = NOTSETVALUES;
  MPI_Comm_rank(comm,&aij->mytid);
  MPI_Comm_size(comm,&aij->numtids);

  if (M == PETSC_DECIDE || N == PETSC_DECIDE) {
    work[0] = m; work[1] = n;
    MPI_Allreduce((void *) work,(void *) sum,2,MPI_INT,MPI_SUM,comm );
    if (M == PETSC_DECIDE) M = sum[0];
    if (N == PETSC_DECIDE) N = sum[1];
  }
  if (m == PETSC_DECIDE) 
    {m = M/aij->numtids + ((M % aij->numtids) > aij->mytid);}
  if (n == PETSC_DECIDE) 
    {n = N/aij->numtids + ((N % aij->numtids) > aij->mytid);}
  aij->m = m;
  aij->n = n;
  aij->N = N;
  aij->M = M;

  /* build local table of row and column ownerships */
  aij->rowners = (int *) PETSCMALLOC(2*(aij->numtids+2)*sizeof(int)); 
  CHKPTRQ(aij->rowners);
  aij->cowners = aij->rowners + aij->numtids + 1;
  MPI_Allgather(&m,1,MPI_INT,aij->rowners+1,1,MPI_INT,comm);
  aij->rowners[0] = 0;
  for ( i=2; i<=aij->numtids; i++ ) {
    aij->rowners[i] += aij->rowners[i-1];
  }
  aij->rstart = aij->rowners[aij->mytid]; 
  aij->rend   = aij->rowners[aij->mytid+1]; 
  MPI_Allgather(&n,1,MPI_INT,aij->cowners+1,1,MPI_INT,comm);
  aij->cowners[0] = 0;
  for ( i=2; i<=aij->numtids; i++ ) {
    aij->cowners[i] += aij->cowners[i-1];
  }
  aij->cstart = aij->cowners[aij->mytid]; 
  aij->cend   = aij->cowners[aij->mytid+1]; 


  ierr = MatCreateSequentialAIJ(MPI_COMM_SELF,m,n,d_nz,d_nnz,&aij->A); 
  CHKERRQ(ierr);
  PLogObjectParent(mat,aij->A);
  ierr = MatCreateSequentialAIJ(MPI_COMM_SELF,m,N,o_nz,o_nnz,&aij->B); 
  CHKERRQ(ierr);
  PLogObjectParent(mat,aij->B);

  /* build cache for off array entries formed */
  ierr = StashBuild_Private(&aij->stash); CHKERRQ(ierr);
  aij->colmap    = 0;
  aij->garray    = 0;

  /* stuff used for matrix vector multiply */
  aij->lvec      = 0;
  aij->Mvctx     = 0;
  aij->assembled = 0;

  *newmat = mat;
  return 0;
}

static int MatCopyPrivate_MPIAIJ(Mat matin,Mat *newmat)
{
  Mat        mat;
  Mat_MPIAIJ *aij,*oldmat = (Mat_MPIAIJ *) matin->data;
  int        ierr, len;
  *newmat      = 0;

  if (!oldmat->assembled) SETERRQ(1,"Cannot copy unassembled matrix");
  PETSCHEADERCREATE(mat,_Mat,MAT_COOKIE,MATMPIAIJ,matin->comm);
  PLogObjectCreate(mat);
  mat->data       = (void *) (aij = PETSCNEW(Mat_MPIAIJ)); CHKPTRQ(aij);
  mat->ops        = &MatOps;
  mat->destroy    = MatDestroy_MPIAIJ;
  mat->view       = MatView_MPIAIJ;
  mat->factor     = matin->factor;

  aij->m          = oldmat->m;
  aij->n          = oldmat->n;
  aij->M          = oldmat->M;
  aij->N          = oldmat->N;

  aij->assembled  = 1;
  aij->rstart     = oldmat->rstart;
  aij->rend       = oldmat->rend;
  aij->cstart     = oldmat->cstart;
  aij->cend       = oldmat->cend;
  aij->numtids    = oldmat->numtids;
  aij->mytid      = oldmat->mytid;
  aij->insertmode = NOTSETVALUES;

  aij->rowners        = (int *) PETSCMALLOC( (aij->numtids+1)*sizeof(int) );
  CHKPTRQ(aij->rowners);
  PETSCMEMCPY(aij->rowners,oldmat->rowners,(aij->numtids+1)*sizeof(int));
  ierr = StashInitialize_Private(&aij->stash); CHKERRQ(ierr);
  if (oldmat->colmap) {
    aij->colmap      = (int *) PETSCMALLOC( (aij->N)*sizeof(int) );
    CHKPTRQ(aij->colmap);
    PETSCMEMCPY(aij->colmap,oldmat->colmap,(aij->N)*sizeof(int));
  } else aij->colmap = 0;
  if (oldmat->garray && (len = ((Mat_AIJ *) (oldmat->B->data))->n)) {
    aij->garray      = (int *) PETSCMALLOC(len*sizeof(int) ); CHKPTRQ(aij->garray);
    PETSCMEMCPY(aij->garray,oldmat->garray,len*sizeof(int));
  } else aij->garray = 0;
  
  ierr =  VecDuplicate(oldmat->lvec,&aij->lvec); CHKERRQ(ierr);
  PLogObjectParent(mat,aij->lvec);
  ierr =  VecScatterCtxCopy(oldmat->Mvctx,&aij->Mvctx); CHKERRQ(ierr);
  PLogObjectParent(mat,aij->Mvctx);
  ierr =  MatConvert(oldmat->A,MATSAME,&aij->A); CHKERRQ(ierr);
  PLogObjectParent(mat,aij->A);
  ierr =  MatConvert(oldmat->B,MATSAME,&aij->B); CHKERRQ(ierr);
  PLogObjectParent(mat,aij->B);
  *newmat = mat;
  return 0;
}
