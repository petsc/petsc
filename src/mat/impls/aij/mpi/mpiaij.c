#ifndef lint
static char vcid[] = "$Id: mpiaij.c,v 1.87 1995/10/11 17:55:12 curfman Exp curfman $";
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
  Mat_SeqAIJ *B = (Mat_SeqAIJ*) aij->B->data;
  int        n = B->n,i,shift = B->indexshift;

  aij->colmap = (int *) PETSCMALLOC(aij->N*sizeof(int));CHKPTRQ(aij->colmap);
  PLogObjectMemory(mat,aij->N*sizeof(int));
  PetscZero(aij->colmap,aij->N*sizeof(int));
  for ( i=0; i<n; i++ ) aij->colmap[aij->garray[i]] = i-shift;
  return 0;
}

extern int DisAssemble_MPIAIJ(Mat);

static int MatGetReordering_MPIAIJ(Mat mat,MatOrdering type,IS *rperm,IS *cperm)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  int ierr;
  if (aij->numtids == 1) {
    ierr = MatGetReordering(aij->A,type,rperm,cperm); CHKERRQ(ierr);
  } else SETERRQ(1,"MatGetReordering_MPIAIJ:not supported in parallel");
  return 0;
}

static int MatSetValues_MPIAIJ(Mat mat,int m,int *idxm,int n,
                               int *idxn,Scalar *v,InsertMode addv)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Mat_SeqAIJ *C = (Mat_SeqAIJ*) aij->A->data;
  int        ierr,i,j, rstart = aij->rstart, rend = aij->rend;
  int        cstart = aij->cstart, cend = aij->cend,row,col;
  int        shift = C->indexshift;

  if (aij->insertmode != NOT_SET_VALUES && aij->insertmode != addv) {
    SETERRQ(1,"MatSetValues_MPIAIJ:Cannot mix inserts and adds");
  }
  aij->insertmode = addv;
  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) SETERRQ(1,"MatSetValues_MPIAIJ:Negative row");
    if (idxm[i] >= aij->M) SETERRQ(1,"MatSetValues_MPIAIJ:Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (idxn[j] < 0) SETERRQ(1,"MatSetValues_MPIAIJ:Negative column");
        if (idxn[j] >= aij->N) SETERRQ(1,"MatSetValues_MPIAIJ:Col too large");
        if (idxn[j] >= cstart && idxn[j] < cend){
          col = idxn[j] - cstart;
          ierr = MatSetValues(aij->A,1,&row,1,&col,v+i*n+j,addv);CHKERRQ(ierr);
        }
        else {
          if (aij->assembled) {
            if (!aij->colmap) {ierr = CreateColmap_Private(mat);CHKERRQ(ierr);}
            col = aij->colmap[idxn[j]] + shift;
            if (col < 0 && !((Mat_SeqAIJ*)(aij->A->data))->nonew) {
              ierr = DisAssemble_MPIAIJ(mat); CHKERRQ(ierr);
              col =  idxn[j];              
            }
          }
          else col = idxn[j];
          ierr = MatSetValues(aij->B,1,&row,1,&col,v+i*n+j,addv);CHKERRQ(ierr);
        }
      }
    } 
    else {
      ierr = StashValues_Private(&aij->stash,idxm[i],n,idxn,v+i*n,addv);CHKERRQ(ierr);
    }
  }
  return 0;
}

static int MatAssemblyBegin_MPIAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIAIJ  *aij = (Mat_MPIAIJ *) mat->data;
  MPI_Comm    comm = mat->comm;
  int         numtids = aij->numtids, *owners = aij->rowners;
  int         mytid = aij->mytid,tag = mat->tag, *owner,*starts,count,ierr;
  MPI_Request *send_waits,*recv_waits;
  int         *nprocs,i,j,idx,*procs,nsends,nreceives,nmax,*work;
  InsertMode  addv;
  Scalar      *rvalues,*svalues;

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  MPI_Allreduce(&aij->insertmode,&addv,1,MPI_INT,MPI_BOR,comm);
  if (addv == (ADD_VALUES|INSERT_VALUES)) {
    SETERRQ(1,"MatAssemblyBegin_MPIAIJ:Some processors inserted others added");
  }
  aij->insertmode = addv; /* in case this processor had no cache */

  /*  first count number of contributors to each processor */
  nprocs = (int *) PETSCMALLOC( 2*numtids*sizeof(int) ); CHKPTRQ(nprocs);
  PetscZero(nprocs,2*numtids*sizeof(int)); procs = nprocs + numtids;
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
  MPI_Allreduce(procs, work,numtids,MPI_INT,MPI_SUM,comm);
  nreceives = work[mytid]; 
  MPI_Allreduce( nprocs, work,numtids,MPI_INT,MPI_MAX,comm);
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
    MPI_Irecv(rvalues+3*nmax*i,3*nmax,MPIU_SCALAR,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (Scalar *) PETSCMALLOC(3*(aij->stash.n+1)*sizeof(Scalar));CHKPTRQ(svalues);
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
      MPI_Isend(svalues+3*starts[i],3*nprocs[i],MPIU_SCALAR,i,tag,
                comm,send_waits+count++);
    }
  }
  PETSCFREE(starts); PETSCFREE(nprocs);

  /* Free cache space */
  ierr = StashDestroy_Private(&aij->stash); CHKERRQ(ierr);

  aij->svalues    = svalues;    aij->rvalues    = rvalues;
  aij->nsends     = nsends;     aij->nrecvs     = nreceives;
  aij->send_waits = send_waits; aij->recv_waits = recv_waits;
  aij->rmax       = nmax;

  return 0;
}
extern int MatSetUpMultiply_MPIAIJ(Mat);

static int MatAssemblyEnd_MPIAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Mat_SeqAIJ *C = (Mat_SeqAIJ *) aij->A->data;
  MPI_Status  *send_status,recv_status;
  int         imdex,nrecvs = aij->nrecvs, count = nrecvs, i, n, ierr;
  int         row,col,other_disassembled,shift = C->indexshift;
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
          col = aij->colmap[col] + shift;
          if (col < 0  && !((Mat_SeqAIJ*)(aij->A->data))->nonew) {
            ierr = DisAssemble_MPIAIJ(mat); CHKERRQ(ierr);
            col = (int) PETSCREAL(values[3*i+1]);
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
    send_status = (MPI_Status *) PETSCMALLOC(aij->nsends*sizeof(MPI_Status));
    CHKPTRQ(send_status);
    MPI_Waitall(aij->nsends,aij->send_waits,send_status);
    PETSCFREE(send_status);
  }
  PETSCFREE(aij->send_waits); PETSCFREE(aij->svalues);

  aij->insertmode = NOT_SET_VALUES;
  ierr = MatAssemblyBegin(aij->A,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(aij->A,mode); CHKERRQ(ierr);

  /* determine if any processor has disassembled, if so we must 
     also disassemble ourselfs, in order that we may reassemble. */
  MPI_Allreduce(&aij->assembled,&other_disassembled,1,MPI_INT,MPI_PROD,mat->comm);
  if (aij->assembled && !other_disassembled) {
    ierr = DisAssemble_MPIAIJ(mat); CHKERRQ(ierr);
  }

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
  int        ierr;
  ierr = MatZeroEntries(l->A); CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B); CHKERRQ(ierr);
  return 0;
}

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

  if (!l->assembled) SETERRQ(1,"MatZeroRows_MPIAIJ:Must assemble matrix");
  ierr = ISGetLocalSize(is,&N); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows); CHKERRQ(ierr);

  /*  first count number of contributors to each processor */
  nprocs = (int *) PETSCMALLOC( 2*numtids*sizeof(int) ); CHKPTRQ(nprocs);
  PetscZero(nprocs,2*numtids*sizeof(int)); procs = nprocs + numtids;
  owner = (int *) PETSCMALLOC((N+1)*sizeof(int)); CHKPTRQ(owner); /* see note*/
  for ( i=0; i<N; i++ ) {
    idx = rows[i];
    found = 0;
    for ( j=0; j<numtids; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERRQ(1,"MatZeroRows_MPIAIJ:Index out of range");
  }
  nsends = 0;  for ( i=0; i<numtids; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PETSCMALLOC( numtids*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce( procs, work,numtids,MPI_INT,MPI_SUM,comm);
  nrecvs = work[mytid]; 
  MPI_Allreduce( nprocs, work,numtids,MPI_INT,MPI_MAX,comm);
  nmax = work[mytid];
  PETSCFREE(work);

  /* post receives:   */
  rvalues = (int *) PETSCMALLOC((nrecvs+1)*(nmax+1)*sizeof(int)); /*see note */
  CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PETSCMALLOC((nrecvs+1)*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv(rvalues+nmax*i,nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);
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
      MPI_Isend(svalues+starts[i],nprocs[i],MPI_INT,i,tag,comm,send_waits+count++);
    }
  }
  PETSCFREE(starts);

  base = owners[mytid];

  /*  wait on receives */
  lens   = (int *) PETSCMALLOC( 2*(nrecvs+1)*sizeof(int) ); CHKPTRQ(lens);
  source = lens + nrecvs;
  count  = nrecvs; slen = 0;
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
  ierr = ISCreateSeq(MPI_COMM_SELF,slen,lrows,&istmp);CHKERRQ(ierr);   
  PLogObjectParent(A,istmp);
  PETSCFREE(lrows);
  ierr = MatZeroRows(l->A,istmp,diag); CHKERRQ(ierr);
  ierr = MatZeroRows(l->B,istmp,0); CHKERRQ(ierr);
  ierr = ISDestroy(istmp); CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status *) PETSCMALLOC(nsends*sizeof(MPI_Status));
    CHKPTRQ(send_status);
    MPI_Waitall(nsends,send_waits,send_status);
    PETSCFREE(send_status);
  }
  PETSCFREE(send_waits); PETSCFREE(svalues);

  return 0;
}

static int MatMult_MPIAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;

  if (!a->assembled) SETERRQ(1,"MatMult_MPIAIJ:must assemble matrix");
  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_ALL,a->Mvctx);CHKERRQ(ierr);
  ierr = MatMult(a->A,xx,yy); CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_ALL,a->Mvctx);CHKERRQ(ierr);
  ierr = MatMultAdd(a->B,a->lvec,yy,yy); CHKERRQ(ierr);
  return 0;
}

static int MatMultAdd_MPIAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;
  if (!a->assembled) SETERRQ(1,"MatMult_MPIAIJ:must assemble matrix");
  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_ALL,a->Mvctx);CHKERRQ(ierr);
  ierr = MatMultAdd(a->A,xx,yy,zz); CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_ALL,a->Mvctx);CHKERRQ(ierr);
  ierr = MatMultAdd(a->B,a->lvec,zz,zz); CHKERRQ(ierr);
  return 0;
}

static int MatMultTrans_MPIAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;

  if (!a->assembled) SETERRQ(1,"MatMulTrans_MPIAIJ:must assemble matrix");
  /* do nondiagonal part */
  ierr = MatMultTrans(a->B,xx,a->lvec); CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(a->lvec,yy,ADD_VALUES,
                (ScatterMode)(SCATTER_ALL|SCATTER_REVERSE),a->Mvctx); CHKERRQ(ierr);
  /* do local part */
  ierr = MatMultTrans(a->A,xx,yy); CHKERRQ(ierr);
  /* receive remote parts: note this assumes the values are not actually */
  /* inserted in yy until the next line, which is true for my implementation*/
  /* but is not perhaps always true. */
  ierr = VecScatterEnd(a->lvec,yy,ADD_VALUES,
                  (ScatterMode)(SCATTER_ALL|SCATTER_REVERSE),a->Mvctx); CHKERRQ(ierr);
  return 0;
}

static int MatMultTransAdd_MPIAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;

  if (!a->assembled) SETERRQ(1,"MatMulTransAdd_MPIAIJ:must assemble matrix");
  /* do nondiagonal part */
  ierr = MatMultTrans(a->B,xx,a->lvec); CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(a->lvec,zz,ADD_VALUES,
                 (ScatterMode)(SCATTER_ALL|SCATTER_REVERSE),a->Mvctx); CHKERRQ(ierr);
  /* do local part */
  ierr = MatMultTransAdd(a->A,xx,yy,zz); CHKERRQ(ierr);
  /* receive remote parts: note this assumes the values are not actually */
  /* inserted in yy until the next line, which is true for my implementation*/
  /* but is not perhaps always true. */
  ierr = VecScatterEnd(a->lvec,zz,ADD_VALUES,
                  (ScatterMode)(SCATTER_ALL|SCATTER_REVERSE),a->Mvctx); CHKERRQ(ierr);
  return 0;
}

/*
  This only works correctly for square matrices where the subblock A->A is the 
   diagonal block
*/
static int MatGetDiagonal_MPIAIJ(Mat A,Vec v)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  if (!a->assembled) SETERRQ(1,"MatGetDiag_MPIAIJ:must assemble matrix");
  return MatGetDiagonal(a->A,v);
}

static int MatDestroy_MPIAIJ(PetscObject obj)
{
  Mat        mat = (Mat) obj;
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  int        ierr;
#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d",aij->M,aij->N);
#endif
  PETSCFREE(aij->rowners); 
  ierr = MatDestroy(aij->A); CHKERRQ(ierr);
  ierr = MatDestroy(aij->B); CHKERRQ(ierr);
  if (aij->colmap) PETSCFREE(aij->colmap);
  if (aij->garray) PETSCFREE(aij->garray);
  if (aij->lvec)   VecDestroy(aij->lvec);
  if (aij->Mvctx)  VecScatterCtxDestroy(aij->Mvctx);
  PETSCFREE(aij); 
  PLogObjectDestroy(mat);
  PETSCHEADERDESTROY(mat);
  return 0;
}
#include "draw.h"
#include "pinclude/pviewer.h"

static int MatView_MPIAIJ_Binary(Mat mat,Viewer viewer)
{
  Mat_MPIAIJ  *aij = (Mat_MPIAIJ *) mat->data;
  int         ierr;

  if (aij->numtids == 1) {
    ierr = MatView(aij->A,viewer); CHKERRQ(ierr);
  }
  return 0;
}

static int MatView_MPIAIJ_ASCIIorDraw(Mat mat,Viewer viewer)
{
  Mat_MPIAIJ  *aij = (Mat_MPIAIJ *) mat->data;
  Mat_SeqAIJ* C = (Mat_SeqAIJ*)aij->A->data;
  int         ierr, format,shift = C->indexshift;
  PetscObject vobj = (PetscObject) viewer;
  FILE        *fd;
 
  if (vobj->type == ASCII_FILE_VIEWER || vobj->type == ASCII_FILES_VIEWER) {
    ierr = ViewerFileGetFormat_Private(viewer,&format);
    if (format == FILE_FORMAT_INFO) {
      return 0; 
    }
  }

  if (vobj->type == ASCII_FILE_VIEWER) {
    ierr = ViewerFileGetPointer_Private(viewer,&fd); CHKERRQ(ierr);
    MPIU_Seq_begin(mat->comm,1);
    fprintf(fd,"[%d] rows %d starts %d ends %d cols %d starts %d ends %d\n",
           aij->mytid,aij->m,aij->rstart,aij->rend,aij->n,aij->cstart,
           aij->cend);
    ierr = MatView(aij->A,viewer); CHKERRQ(ierr);
    ierr = MatView(aij->B,viewer); CHKERRQ(ierr);
    fflush(fd);
    MPIU_Seq_end(mat->comm,1);
  }
  else {
    int numtids = aij->numtids, mytid = aij->mytid;
    if (numtids == 1) {
      ierr = MatView(aij->A,viewer); CHKERRQ(ierr);
    }
    else {
      /* assemble the entire matrix onto first processor. */
      Mat         A;
      Mat_SeqAIJ *Aloc;
      int         M = aij->M, N = aij->N,m,*ai,*aj,row,*cols,i,*ct;
      Scalar      *a;

      if (!mytid) {
        ierr = MatCreateMPIAIJ(mat->comm,M,N,M,N,0,0,0,0,&A);CHKERRQ(ierr);
      }
      else {
        ierr = MatCreateMPIAIJ(mat->comm,0,0,M,N,0,0,0,0,&A);CHKERRQ(ierr);
      }
      PLogObjectParent(mat,A);
      

      /* copy over the A part */
      Aloc = (Mat_SeqAIJ*) aij->A->data;
      m = Aloc->m; ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
      row = aij->rstart;
      for ( i=0; i<ai[m]+shift; i++ ) {aj[i] += aij->cstart + shift;}
      for ( i=0; i<m; i++ ) {
        ierr = MatSetValues(A,1,&row,ai[i+1]-ai[i],aj,a,INSERT_VALUES);CHKERRQ(ierr);
        row++; a += ai[i+1]-ai[i]; aj += ai[i+1]-ai[i];
      } 
      aj = Aloc->j;
      for ( i=0; i<ai[m]+shift; i++ ) {aj[i] -= aij->cstart + shift;}

      /* copy over the B part */
      Aloc = (Mat_SeqAIJ*) aij->B->data;
      m = Aloc->m;  ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
      row = aij->rstart;
      ct = cols = (int *) PETSCMALLOC( (ai[m]+1)*sizeof(int) ); CHKPTRQ(cols);
      for ( i=0; i<ai[m]+shift; i++ ) {cols[i] = aij->garray[aj[i]+shift];}
      for ( i=0; i<m; i++ ) {
        ierr = MatSetValues(A,1,&row,ai[i+1]-ai[i],cols,a,INSERT_VALUES);CHKERRQ(ierr);
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

static int MatView_MPIAIJ(PetscObject obj,Viewer viewer)
{
  Mat         mat = (Mat) obj;
  Mat_MPIAIJ  *aij = (Mat_MPIAIJ *) mat->data;
  int         ierr;
  PetscObject vobj = (PetscObject) viewer;
 
  if (!aij->assembled) SETERRQ(1,"MatView_MPIAIJ:must assemble matrix");
  if (!viewer) { 
    viewer = STDOUT_VIEWER_SELF; vobj = (PetscObject) viewer;
  }
  if (vobj->cookie == DRAW_COOKIE && vobj->type == NULLWINDOW) return 0;
  else if (vobj->cookie == VIEWER_COOKIE && vobj->type == ASCII_FILE_VIEWER) {
    ierr = MatView_MPIAIJ_ASCIIorDraw(mat,viewer); CHKERRQ(ierr);
  }
  else if (vobj->cookie == VIEWER_COOKIE && vobj->type == ASCII_FILES_VIEWER) {
    ierr = MatView_MPIAIJ_ASCIIorDraw(mat,viewer); CHKERRQ(ierr);
  }
  else if (vobj->cookie == DRAW_COOKIE) {
    ierr = MatView_MPIAIJ_ASCIIorDraw(mat,viewer); CHKERRQ(ierr);
  }
  else if (vobj->type == BINARY_FILE_VIEWER) {
    return MatView_MPIAIJ_Binary(mat,viewer);
  }
  return 0;
}

extern int MatMarkDiag_SeqAIJ(Mat);
/*
    This has to provide several versions.

     1) per sequential 
     2) a) use only local smoothing updating outer values only once.
        b) local smoothing updating outer values each inner iteration
     3) color updating out values betwen colors.
*/
static int MatRelax_MPIAIJ(Mat matin,Vec bb,double omega,MatSORType flag,
                           double fshift,int its,Vec xx)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;
  Mat        AA = mat->A, BB = mat->B;
  Mat_SeqAIJ *A = (Mat_SeqAIJ *) AA->data, *B = (Mat_SeqAIJ *)BB->data;
  Scalar     zero = 0.0,*b,*x,*xs,*ls,d,*v,sum,scale,*t,*ts;
  int        ierr,*idx, *diag;
  int        n = mat->n, m = mat->m, i,shift = A->indexshift;
  Vec        tt;

  if (!mat->assembled) SETERRQ(1,"MatRelax_MPIAIJ:must assemble matrix");

  VecGetArray(xx,&x); VecGetArray(bb,&b); VecGetArray(mat->lvec,&ls);
  xs = x + shift; /* shift by one for index start of 1 */
  ls = ls + shift;
  if (!A->diag) {if ((ierr = MatMarkDiag_SeqAIJ(AA))) return ierr;}
  diag = A->diag;
  if (flag == SOR_APPLY_UPPER || flag == SOR_APPLY_LOWER) {
    SETERRQ(1,"MatRelax_MPIAIJ:Option not supported");
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
    PLogObjectParent(matin,tt);
    VecGetArray(tt,&t);
    scale = (2.0/omega) - 1.0;
    /*  x = (E + U)^{-1} b */
    VecSet(&zero,mat->lvec);
    ierr = VecPipelineBegin(xx,mat->lvec,INSERT_VALUES,PIPELINE_UP,
                              mat->Mvctx); CHKERRQ(ierr);
    for ( i=m-1; i>-1; i-- ) {
      n    = A->i[i+1] - diag[i] - 1;
      idx  = A->j + diag[i] + !shift;
      v    = A->a + diag[i] + !shift;
      sum  = b[i];
      SPARSEDENSEMDOT(sum,xs,v,idx,n); 
      d    = fshift + A->a[diag[i]+shift];
      n    = B->i[i+1] - B->i[i]; 
      idx  = B->j + B->i[i] + shift;
      v    = B->a + B->i[i] + shift;
      SPARSEDENSEMDOT(sum,ls,v,idx,n); 
      x[i] = omega*(sum/d);
    }
    ierr = VecPipelineEnd(xx,mat->lvec,INSERT_VALUES,PIPELINE_UP,
                            mat->Mvctx); CHKERRQ(ierr);

    /*  t = b - (2*E - D)x */
    v = A->a;
    for ( i=0; i<m; i++ ) { t[i] = b[i] - scale*(v[*diag++ + shift])*x[i]; }

    /*  t = (E + L)^{-1}t */
    ts = t + shift; /* shifted by one for index start of a or mat->j*/
    diag = A->diag;
    VecSet(&zero,mat->lvec);
    ierr = VecPipelineBegin(tt,mat->lvec,INSERT_VALUES,PIPELINE_DOWN,
                                                 mat->Mvctx); CHKERRQ(ierr);
    for ( i=0; i<m; i++ ) {
      n    = diag[i] - A->i[i]; 
      idx  = A->j + A->i[i] + shift;
      v    = A->a + A->i[i] + shift;
      sum  = t[i];
      SPARSEDENSEMDOT(sum,ts,v,idx,n); 
      d    = fshift + A->a[diag[i]+shift];
      n    = B->i[i+1] - B->i[i]; 
      idx  = B->j + B->i[i] + shift;
      v    = B->a + B->i[i] + shift;
      SPARSEDENSEMDOT(sum,ls,v,idx,n); 
      t[i] = omega*(sum/d);
    }
    ierr = VecPipelineEnd(tt,mat->lvec,INSERT_VALUES,PIPELINE_DOWN,
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
      ierr=VecScatterBegin(xx,mat->lvec,INSERT_VALUES,SCATTER_UP,mat->Mvctx);
      CHKERRQ(ierr);
      ierr = VecScatterEnd(xx,mat->lvec,INSERT_VALUES,SCATTER_UP,mat->Mvctx);
      CHKERRQ(ierr);
    }
    while (its--) {
      /* go down through the rows */
      ierr = VecPipelineBegin(xx,mat->lvec,INSERT_VALUES,PIPELINE_DOWN,
                              mat->Mvctx); CHKERRQ(ierr);
      for ( i=0; i<m; i++ ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] + shift;
        v    = A->a + A->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
      ierr = VecPipelineEnd(xx,mat->lvec,INSERT_VALUES,PIPELINE_DOWN,
                            mat->Mvctx); CHKERRQ(ierr);
      /* come up through the rows */
      ierr = VecPipelineBegin(xx,mat->lvec,INSERT_VALUES,PIPELINE_UP,
                              mat->Mvctx); CHKERRQ(ierr);
      for ( i=m-1; i>-1; i-- ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] + shift;
        v    = A->a + A->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
      ierr = VecPipelineEnd(xx,mat->lvec,INSERT_VALUES,PIPELINE_UP,
                            mat->Mvctx); CHKERRQ(ierr);
    }    
  }
  else if (flag & SOR_FORWARD_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      VecSet(&zero,mat->lvec);
      ierr = VecPipelineBegin(xx,mat->lvec,INSERT_VALUES,PIPELINE_DOWN,
                              mat->Mvctx); CHKERRQ(ierr);
      for ( i=0; i<m; i++ ) {
        n    = diag[i] - A->i[i]; 
        idx  = A->j + A->i[i] + shift;
        v    = A->a + A->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = omega*(sum/d);
      }
      ierr = VecPipelineEnd(xx,mat->lvec,INSERT_VALUES,PIPELINE_DOWN,
                            mat->Mvctx); CHKERRQ(ierr);
      its--;
    }
    while (its--) {
      ierr=VecScatterBegin(xx,mat->lvec,INSERT_VALUES,SCATTER_UP,mat->Mvctx);
      CHKERRQ(ierr);
      ierr = VecScatterEnd(xx,mat->lvec,INSERT_VALUES,SCATTER_UP,mat->Mvctx);
      CHKERRQ(ierr);
      ierr = VecPipelineBegin(xx,mat->lvec,INSERT_VALUES,PIPELINE_DOWN,
                              mat->Mvctx); CHKERRQ(ierr);
      for ( i=0; i<m; i++ ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] + shift;
        v    = A->a + A->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
      ierr = VecPipelineEnd(xx,mat->lvec,INSERT_VALUES,PIPELINE_DOWN,
                            mat->Mvctx); CHKERRQ(ierr);
    } 
  }
  else if (flag & SOR_BACKWARD_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      VecSet(&zero,mat->lvec);
      ierr = VecPipelineBegin(xx,mat->lvec,INSERT_VALUES,PIPELINE_UP,
                              mat->Mvctx); CHKERRQ(ierr);
      for ( i=m-1; i>-1; i-- ) {
        n    = A->i[i+1] - diag[i] - 1; 
        idx  = A->j + diag[i] + !shift;
        v    = A->a + diag[i] + !shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = omega*(sum/d);
      }
      ierr = VecPipelineEnd(xx,mat->lvec,INSERT_VALUES,PIPELINE_UP,
                            mat->Mvctx); CHKERRQ(ierr);
      its--;
    }
    while (its--) {
      ierr = VecScatterBegin(xx,mat->lvec,INSERT_VALUES,SCATTER_DOWN,
                            mat->Mvctx); CHKERRQ(ierr);
      ierr = VecScatterEnd(xx,mat->lvec,INSERT_VALUES,SCATTER_DOWN,
                            mat->Mvctx); CHKERRQ(ierr);
      ierr = VecPipelineBegin(xx,mat->lvec,INSERT_VALUES,PIPELINE_UP,
                              mat->Mvctx); CHKERRQ(ierr);
      for ( i=m-1; i>-1; i-- ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] + shift;
        v    = A->a + A->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
      ierr = VecPipelineEnd(xx,mat->lvec,INSERT_VALUES,PIPELINE_UP,
                            mat->Mvctx); CHKERRQ(ierr);
    } 
  }
  else if ((flag & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      return MatRelax(mat->A,bb,omega,flag,fshift,its,xx);
    }
    ierr=VecScatterBegin(xx,mat->lvec,INSERT_VALUES,SCATTER_ALL,mat->Mvctx);
    CHKERRQ(ierr);
    ierr = VecScatterEnd(xx,mat->lvec,INSERT_VALUES,SCATTER_ALL,mat->Mvctx);
    CHKERRQ(ierr);
    while (its--) {
      /* go down through the rows */
      for ( i=0; i<m; i++ ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] + shift;
        v    = A->a + A->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
      /* come up through the rows */
      for ( i=m-1; i>-1; i-- ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] + shift;
        v    = A->a + A->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
    }    
  }
  else if (flag & SOR_LOCAL_FORWARD_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      return MatRelax(mat->A,bb,omega,flag,fshift,its,xx);
    }
    ierr=VecScatterBegin(xx,mat->lvec,INSERT_VALUES,SCATTER_ALL,mat->Mvctx);
    CHKERRQ(ierr);
    ierr = VecScatterEnd(xx,mat->lvec,INSERT_VALUES,SCATTER_ALL,mat->Mvctx);
    CHKERRQ(ierr);
    while (its--) {
      for ( i=0; i<m; i++ ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] + shift;
        v    = A->a + A->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
        SPARSEDENSEMDOT(sum,ls,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
    } 
  }
  else if (flag & SOR_LOCAL_BACKWARD_SWEEP){
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      return MatRelax(mat->A,bb,omega,flag,fshift,its,xx);
    }
    ierr = VecScatterBegin(xx,mat->lvec,INSERT_VALUES,SCATTER_ALL,
                            mat->Mvctx); CHKERRQ(ierr);
    ierr = VecScatterEnd(xx,mat->lvec,INSERT_VALUES,SCATTER_ALL,
                            mat->Mvctx); CHKERRQ(ierr);
    while (its--) {
      for ( i=m-1; i>-1; i-- ) {
        n    = A->i[i+1] - A->i[i]; 
        idx  = A->j + A->i[i] + shift;
        v    = A->a + A->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        d    = fshift + A->a[diag[i]+shift];
        n    = B->i[i+1] - B->i[i]; 
        idx  = B->j + B->i[i] + shift;
        v    = B->a + B->i[i] + shift;
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
    MPI_Allreduce( isend, irecv,3,MPI_INT,MPI_MAX,matin->comm);
    *nz = irecv[0]; *nzalloc = irecv[1]; *mem = irecv[2];
  } else if (flag == MAT_GLOBAL_SUM) {
    MPI_Allreduce( isend, irecv,3,MPI_INT,MPI_SUM,matin->comm);
    *nz = irecv[0]; *nzalloc = irecv[1]; *mem = irecv[2];
  }
  return 0;
}

extern int MatLUFactorSymbolic_MPIAIJ(Mat,IS,IS,double,Mat*);
extern int MatLUFactorNumeric_MPIAIJ(Mat,Mat*);
extern int MatLUFactor_MPIAIJ(Mat,IS,IS,double);
extern int MatILUFactorSymbolic_MPIAIJ(Mat,IS,IS,double,int,Mat *);
extern int MatSolve_MPIAIJ(Mat,Vec,Vec);
extern int MatSolveAdd_MPIAIJ(Mat,Vec,Vec,Vec);
extern int MatSolveTrans_MPIAIJ(Mat,Vec,Vec);
extern int MatSolveTransAdd_MPIAIJ(Mat,Vec,Vec,Vec);

static int MatSetOption_MPIAIJ(Mat A,MatOption op)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;

  if (op == NO_NEW_NONZERO_LOCATIONS ||
      op == YES_NEW_NONZERO_LOCATIONS ||
      op == COLUMNS_SORTED ||
      op == ROW_ORIENTED) {
        MatSetOption(a->A,op);
        MatSetOption(a->B,op);
  }
  else if (op == ROWS_SORTED || 
           op == SYMMETRIC_MATRIX ||
           op == STRUCTURALLY_SYMMETRIC_MATRIX ||
           op == YES_NEW_DIAGONALS)
    PLogInfo((PetscObject)A,"Info:MatSetOption_MPIAIJ:Option ignored\n");
  else if (op == COLUMN_ORIENTED)
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_MPIAIJ:COLUMN_ORIENTED not supported");}
  else if (op == NO_NEW_DIAGONALS)
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_MPIAIJ:NO_NEW_DIAGONALS not supported");}
  else 
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_MPIAIJ:Option not supported");}
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

  if (row < rstart || row >= rend) SETERRQ(1,"MatGetRow_MPIAIJ:only local rows")
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
      int imark;
      if (mat->assembled) {
        for (i=0; i<nzB; i++) cworkB[i] = mat->garray[cworkB[i]];
      }
      if (v) {
        *v = (Scalar *) PETSCMALLOC( (nztot)*sizeof(Scalar) ); CHKPTRQ(*v);
        for ( i=0; i<nzB; i++ ) {
          if (cworkB[i] < cstart)   (*v)[i] = vworkB[i];
          else break;
        }
        imark = i;
        for ( i=0; i<nzA; i++ )     (*v)[imark+i] = vworkA[i];
        for ( i=imark; i<nzB; i++ ) (*v)[nzA+i] = vworkB[i];
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
        for ( i=imark; i<nzB; i++ ) (*idx)[nzA+i] = cworkB[i];
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

static int MatNorm_MPIAIJ(Mat mat,MatNormType type,double *norm)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Mat_SeqAIJ *amat = (Mat_SeqAIJ*) aij->A->data, *bmat = (Mat_SeqAIJ*) aij->B->data;
  int        ierr, i, j, cstart = aij->cstart,shift = amat->indexshift;
  double     sum = 0.0;
  Scalar     *v;

  if (!aij->assembled) SETERRQ(1,"MatNorm_MPIAIJ:Must assemble matrix");
  if (aij->numtids == 1) {
    ierr =  MatNorm(aij->A,type,norm); CHKERRQ(ierr);
  } else {
    if (type == NORM_FROBENIUS) {
      v = amat->a;
      for (i=0; i<amat->nz; i++ ) {
#if defined(PETSC_COMPLEX)
        sum += real(conj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      v = bmat->a;
      for (i=0; i<bmat->nz; i++ ) {
#if defined(PETSC_COMPLEX)
        sum += real(conj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      MPI_Allreduce((void*)&sum,(void*)norm,1,MPI_DOUBLE,MPI_SUM,mat->comm);
      *norm = sqrt(*norm);
    }
    else if (type == NORM_1) { /* max column norm */
      double *tmp, *tmp2;
      int    *jj, *garray = aij->garray;
      tmp  = (double *) PETSCMALLOC( aij->N*sizeof(double) ); CHKPTRQ(tmp);
      tmp2 = (double *) PETSCMALLOC( aij->N*sizeof(double) ); CHKPTRQ(tmp2);
      PetscZero(tmp,aij->N*sizeof(double));
      *norm = 0.0;
      v = amat->a; jj = amat->j;
      for ( j=0; j<amat->nz; j++ ) {
#if defined(PETSC_COMPLEX)
        tmp[cstart + *jj++ + shift] += abs(*v++); 
#else
        tmp[cstart + *jj++ + shift] += fabs(*v++); 
#endif
      }
      v = bmat->a; jj = bmat->j;
      for ( j=0; j<bmat->nz; j++ ) {
#if defined(PETSC_COMPLEX)
        tmp[garray[*jj++ + shift]] += abs(*v++); 
#else
        tmp[garray[*jj++ + shift]] += fabs(*v++); 
#endif
      }
      MPI_Allreduce((void*)tmp,(void*)tmp2,aij->N,MPI_DOUBLE,MPI_SUM,mat->comm);
      for ( j=0; j<aij->N; j++ ) {
        if (tmp2[j] > *norm) *norm = tmp2[j];
      }
      PETSCFREE(tmp); PETSCFREE(tmp2);
    }
    else if (type == NORM_INFINITY) { /* max row norm */
      double normtemp = 0.0;
      for ( j=0; j<amat->m; j++ ) {
        v = amat->a + amat->i[j] + shift;
        sum = 0.0;
        for ( i=0; i<amat->i[j+1]-amat->i[j]; i++ ) {
#if defined(PETSC_COMPLEX)
          sum += abs(*v); v++;
#else
          sum += fabs(*v); v++;
#endif
        }
        v = bmat->a + bmat->i[j] + shift;
        for ( i=0; i<bmat->i[j+1]-bmat->i[j]; i++ ) {
#if defined(PETSC_COMPLEX)
          sum += abs(*v); v++;
#else
          sum += fabs(*v); v++;
#endif
        }
        if (sum > normtemp) normtemp = sum;
        MPI_Allreduce((void*)&normtemp,(void*)norm,1,MPI_DOUBLE,MPI_MAX,mat->comm);
      }
    }
    else {
      SETERRQ(1,"MatNorm_MPIRow:No support for two norm");
    }
  }
  return 0; 
}

static int MatTranspose_MPIAIJ(Mat A,Mat *matout)
{ 
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  Mat_SeqAIJ *Aloc = (Mat_SeqAIJ *) a->A->data;
  int        ierr,shift = Aloc->indexshift;
  Mat        B;
  int        M = a->M, N = a->N,m,*ai,*aj,row,*cols,i,*ct;
  Scalar     *array;

  if (!matout && M != N) SETERRQ(1,"MatTranspose_MPIAIJ:Square only in-place");
  ierr = MatCreateMPIAIJ(A->comm,PETSC_DECIDE,PETSC_DECIDE,N,M,0,0,0,0,&B);
  CHKERRQ(ierr);

  /* copy over the A part */
  Aloc = (Mat_SeqAIJ*) a->A->data;
  m = Aloc->m; ai = Aloc->i; aj = Aloc->j; array = Aloc->a;
  row = a->rstart;
  for ( i=0; i<ai[m]+shift; i++ ) {aj[i] += a->cstart + shift;}
  for ( i=0; i<m; i++ ) {
    ierr = MatSetValues(B,ai[i+1]-ai[i],aj,1,&row,array,INSERT_VALUES);CHKERRQ(ierr);
    row++; array += ai[i+1]-ai[i]; aj += ai[i+1]-ai[i];
  } 
  aj = Aloc->j;
  for ( i=0; i<ai[m]|+shift; i++ ) {aj[i] -= a->cstart + shift;}

  /* copy over the B part */
  Aloc = (Mat_SeqAIJ*) a->B->data;
  m = Aloc->m;  ai = Aloc->i; aj = Aloc->j; array = Aloc->a;
  row = a->rstart;
  ct = cols = (int *) PETSCMALLOC( (1+ai[m]-shift)*sizeof(int) ); CHKPTRQ(cols);
  for ( i=0; i<ai[m]+shift; i++ ) {cols[i] = a->garray[aj[i]+shift];}
  for ( i=0; i<m; i++ ) {
    ierr = MatSetValues(B,ai[i+1]-ai[i],cols,1,&row,array,INSERT_VALUES);CHKERRQ(ierr);
    row++; array += ai[i+1]-ai[i]; cols += ai[i+1]-ai[i];
  } 
  PETSCFREE(ct);
  ierr = MatAssemblyBegin(B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  if (matout) {
    *matout = B;
  } else {
    /* This isn't really an in-place transpose .... but free data structures from a */
    PETSCFREE(a->rowners); 
    ierr = MatDestroy(a->A); CHKERRQ(ierr);
    ierr = MatDestroy(a->B); CHKERRQ(ierr);
    if (a->colmap) PETSCFREE(a->colmap);
    if (a->garray) PETSCFREE(a->garray);
    if (a->lvec) VecDestroy(a->lvec);
    if (a->Mvctx) VecScatterCtxDestroy(a->Mvctx);
    PETSCFREE(a); 
    PetscMemcpy(A,B,sizeof(struct _Mat)); 
    PETSCHEADERDESTROY(B);
  }
  return 0;
}

extern int MatConvert_MPIAIJ(Mat,MatType,Mat *);
static int MatCopyPrivate_MPIAIJ(Mat,Mat *);

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {MatSetValues_MPIAIJ,
       MatGetRow_MPIAIJ,MatRestoreRow_MPIAIJ,
       MatMult_MPIAIJ,MatMultAdd_MPIAIJ,
       MatMultTrans_MPIAIJ,MatMultTransAdd_MPIAIJ,
       MatSolve_MPIAIJ,MatSolveAdd_MPIAIJ,
       MatSolveTrans_MPIAIJ,MatSolveTransAdd_MPIAIJ,
       MatLUFactor_MPIAIJ,0,
       MatRelax_MPIAIJ,
       MatTranspose_MPIAIJ,
       MatGetInfo_MPIAIJ,0,
       MatGetDiagonal_MPIAIJ,0,MatNorm_MPIAIJ,
       MatAssemblyBegin_MPIAIJ,MatAssemblyEnd_MPIAIJ,
       0,
       MatSetOption_MPIAIJ,MatZeroEntries_MPIAIJ,MatZeroRows_MPIAIJ,
       MatGetReordering_MPIAIJ,
       MatLUFactorSymbolic_MPIAIJ,MatLUFactorNumeric_MPIAIJ,0,0,
       MatGetSize_MPIAIJ,MatGetLocalSize_MPIAIJ,MatGetOwnershipRange_MPIAIJ,
       MatILUFactorSymbolic_MPIAIJ,0,
       0,0,MatConvert_MPIAIJ,0,0,MatCopyPrivate_MPIAIJ};

/*@C
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
.  d_nz - number of nonzeros per row in diagonal portion of local submatrix
           (same for all local rows)
.  d_nzz - number of nonzeros per row in diagonal portion of local submatrix
           or null (possibly different for each row).  You must leave room
           for the diagonal entry even if it is zero.
.  o_nz - number of nonzeros per row in off-diagonal portion of local
           submatrix (same for all local rows).
.  o_nzz - number of nonzeros per row in off-diagonal portion of local 
           submatrix or null (possibly different for each row).

   Output Parameter:
.  newmat - the matrix 

   Notes:
   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices begin at 
   one, not zero.  See the users manual for further details.

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   Storage Information:
   For a square global matrix we define each processor's diagonal portion 
   to be its local rows and the corresponding columns (a square submatrix);  
   each processor's off-diagonal portion encompasses the remainder of the
   local matrix (a rectangular submatrix). 

   The user can specify preallocated storage for the diagonal part of
   the local submatrix with either d_nz or d_nnz (not both). Set both
   d_nz and d_nnz to zero for PETSc to control dynamic memory allocation.  
   Likewise, specify preallocated storage for the off-diagonal part of 
   the local submatrix with o_nz or o_nnz (not both).

.keywords: matrix, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues()
@*/
int MatCreateMPIAIJ(MPI_Comm comm,int m,int n,int M,int N,
                    int d_nz,int *d_nnz, int o_nz,int *o_nnz,Mat *newmat)
{
  Mat          mat;
  Mat_MPIAIJ   *a;
  int          ierr, i,sum[2],work[2];

  *newmat         = 0;
  PETSCHEADERCREATE(mat,_Mat,MAT_COOKIE,MATMPIAIJ,comm);
  PLogObjectCreate(mat);
  mat->data       = (void *) (a = PETSCNEW(Mat_MPIAIJ)); CHKPTRQ(a);
  PetscMemcpy(&mat->ops,&MatOps,sizeof(struct _MatOps));
  mat->destroy    = MatDestroy_MPIAIJ;
  mat->view       = MatView_MPIAIJ;
  mat->factor     = 0;

  a->insertmode = NOT_SET_VALUES;
  MPI_Comm_rank(comm,&a->mytid);
  MPI_Comm_size(comm,&a->numtids);

  if (m == PETSC_DECIDE && (d_nnz || o_nnz)) 
    SETERRQ(1,"MatCreateMPIAIJ:Cannot have PETSc decide rows but set d_nnz or o_nnz");

  if (M == PETSC_DECIDE || N == PETSC_DECIDE) {
    work[0] = m; work[1] = n;
    MPI_Allreduce( work, sum,2,MPI_INT,MPI_SUM,comm );
    if (M == PETSC_DECIDE) M = sum[0];
    if (N == PETSC_DECIDE) N = sum[1];
  }
  if (m == PETSC_DECIDE) {m = M/a->numtids + ((M % a->numtids) > a->mytid);}
  if (n == PETSC_DECIDE) {n = N/a->numtids + ((N % a->numtids) > a->mytid);}
  a->m = m;
  a->n = n;
  a->N = N;
  a->M = M;

  /* build local table of row and column ownerships */
  a->rowners = (int *) PETSCMALLOC(2*(a->numtids+2)*sizeof(int)); CHKPTRQ(a->rowners);
  PLogObjectMemory(mat,2*(a->numtids+2)*sizeof(int)+sizeof(struct _Mat)+ 
                       sizeof(Mat_MPIAIJ));
  a->cowners = a->rowners + a->numtids + 1;
  MPI_Allgather(&m,1,MPI_INT,a->rowners+1,1,MPI_INT,comm);
  a->rowners[0] = 0;
  for ( i=2; i<=a->numtids; i++ ) {
    a->rowners[i] += a->rowners[i-1];
  }
  a->rstart = a->rowners[a->mytid]; 
  a->rend   = a->rowners[a->mytid+1]; 
  MPI_Allgather(&n,1,MPI_INT,a->cowners+1,1,MPI_INT,comm);
  a->cowners[0] = 0;
  for ( i=2; i<=a->numtids; i++ ) {
    a->cowners[i] += a->cowners[i-1];
  }
  a->cstart = a->cowners[a->mytid]; 
  a->cend   = a->cowners[a->mytid+1]; 

  ierr = MatCreateSeqAIJ(MPI_COMM_SELF,m,n,d_nz,d_nnz,&a->A); CHKERRQ(ierr);
  PLogObjectParent(mat,a->A);
  ierr = MatCreateSeqAIJ(MPI_COMM_SELF,m,N,o_nz,o_nnz,&a->B); CHKERRQ(ierr);
  PLogObjectParent(mat,a->B);

  /* build cache for off array entries formed */
  ierr = StashBuild_Private(&a->stash); CHKERRQ(ierr);
  a->colmap    = 0;
  a->garray    = 0;

  /* stuff used for matrix vector multiply */
  a->lvec      = 0;
  a->Mvctx     = 0;
  a->assembled = 0;

  *newmat = mat;
  return 0;
}

static int MatCopyPrivate_MPIAIJ(Mat matin,Mat *newmat)
{
  Mat        mat;
  Mat_MPIAIJ *a,*oldmat = (Mat_MPIAIJ *) matin->data;
  int        ierr, len;

  if (!oldmat->assembled) SETERRQ(1,"MatCopyPrivate_MPIAIJ:Must assemble matrix");
  *newmat      = 0;
  PETSCHEADERCREATE(mat,_Mat,MAT_COOKIE,MATMPIAIJ,matin->comm);
  PLogObjectCreate(mat);
  mat->data       = (void *) (a = PETSCNEW(Mat_MPIAIJ)); CHKPTRQ(a);
  PetscMemcpy(&mat->ops,&MatOps,sizeof(struct _MatOps));
  mat->destroy    = MatDestroy_MPIAIJ;
  mat->view       = MatView_MPIAIJ;
  mat->factor     = matin->factor;

  a->m          = oldmat->m;
  a->n          = oldmat->n;
  a->M          = oldmat->M;
  a->N          = oldmat->N;

  a->assembled  = 1;
  a->rstart     = oldmat->rstart;
  a->rend       = oldmat->rend;
  a->cstart     = oldmat->cstart;
  a->cend       = oldmat->cend;
  a->numtids    = oldmat->numtids;
  a->mytid      = oldmat->mytid;
  a->insertmode = NOT_SET_VALUES;

  a->rowners    = (int *) PETSCMALLOC((a->numtids+1)*sizeof(int));CHKPTRQ(a->rowners);
  PLogObjectMemory(mat,(a->numtids+1)*sizeof(int)+sizeof(struct _Mat)+sizeof(Mat_MPIAIJ));
  PetscMemcpy(a->rowners,oldmat->rowners,(a->numtids+1)*sizeof(int));
  ierr = StashInitialize_Private(&a->stash); CHKERRQ(ierr);
  if (oldmat->colmap) {
    a->colmap      = (int *) PETSCMALLOC( (a->N)*sizeof(int) );CHKPTRQ(a->colmap);
    PLogObjectMemory(mat,(a->N)*sizeof(int));
    PetscMemcpy(a->colmap,oldmat->colmap,(a->N)*sizeof(int));
  } else a->colmap = 0;
  if (oldmat->garray && (len = ((Mat_SeqAIJ *) (oldmat->B->data))->n)) {
    a->garray      = (int *) PETSCMALLOC(len*sizeof(int) ); CHKPTRQ(a->garray);
    PLogObjectMemory(mat,len*sizeof(int));
    PetscMemcpy(a->garray,oldmat->garray,len*sizeof(int));
  } else a->garray = 0;
  
  ierr =  VecDuplicate(oldmat->lvec,&a->lvec); CHKERRQ(ierr);
  PLogObjectParent(mat,a->lvec);
  ierr =  VecScatterCtxCopy(oldmat->Mvctx,&a->Mvctx); CHKERRQ(ierr);
  PLogObjectParent(mat,a->Mvctx);
  ierr =  MatConvert(oldmat->A,MATSAME,&a->A); CHKERRQ(ierr);
  PLogObjectParent(mat,a->A);
  ierr =  MatConvert(oldmat->B,MATSAME,&a->B); CHKERRQ(ierr);
  PLogObjectParent(mat,a->B);
  *newmat = mat;
  return 0;
}

#include "sysio.h"

int MatLoad_MPIAIJorMPIRow(Viewer bview,MatType type,Mat *newmat)
{
  Mat          A;
  int          i, nz, ierr, j,rstart, rend, fd;
  Scalar       *vals,*svals;
  PetscObject  vobj = (PetscObject) bview;
  MPI_Comm     comm = vobj->comm;
  MPI_Status   status;
  int          header[4],mytid,numtid,*rowlengths = 0,M,N,m,*rowners,maxnz,*cols;
  int          *ourlens,*sndcounts = 0,*procsnz = 0, *offlens,jj,*mycols,*smycols;
  int          tag = ((PetscObject)bview)->tag;

  MPI_Comm_size(comm,&numtid); MPI_Comm_rank(comm,&mytid);
  if (!mytid) {
    ierr = ViewerFileGetDescriptor_Private(bview,&fd); CHKERRQ(ierr);
    ierr = SYRead(fd,(char *)header,4,SYINT); CHKERRQ(ierr);
    if (header[0] != MAT_COOKIE) SETERRQ(1,"MatLoad_MPIAIJorMPIRow:not matrix object");
  }

  MPI_Bcast(header+1,3,MPI_INT,0,comm);
  M = header[1]; N = header[2];
  /* determine ownership of all rows */
  m = M/numtid + ((M % numtid) > mytid);
  rowners = (int *) PETSCMALLOC((numtid+2)*sizeof(int)); CHKPTRQ(rowners);
  MPI_Allgather(&m,1,MPI_INT,rowners+1,1,MPI_INT,comm);
  rowners[0] = 0;
  for ( i=2; i<=numtid; i++ ) {
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[mytid]; 
  rend   = rowners[mytid+1]; 

  /* distribute row lengths to all processors */
  ourlens = (int*) PETSCMALLOC( 2*(rend-rstart)*sizeof(int) ); CHKPTRQ(ourlens);
  offlens = ourlens + (rend-rstart);
  if (!mytid) {
    rowlengths = (int*) PETSCMALLOC( M*sizeof(int) ); CHKPTRQ(rowlengths);
    ierr = SYRead(fd,rowlengths,M,SYINT); CHKERRQ(ierr);
    sndcounts = (int*) PETSCMALLOC( numtid*sizeof(int) ); CHKPTRQ(sndcounts);
    for ( i=0; i<numtid; i++ ) sndcounts[i] = rowners[i+1] - rowners[i];
    MPI_Scatterv(rowlengths,sndcounts,rowners,MPI_INT,ourlens,rend-rstart,MPI_INT,0,comm);
    PETSCFREE(sndcounts);
  }
  else {
    MPI_Scatterv(0,0,0,MPI_INT,ourlens,rend-rstart,MPI_INT, 0,comm);
  }

  if (!mytid) {
    /* calculate the number of nonzeros on each processor */
    procsnz = (int*) PETSCMALLOC( numtid*sizeof(int) ); CHKPTRQ(procsnz);
    PetscZero(procsnz,numtid*sizeof(int));
    for ( i=0; i<numtid; i++ ) {
      for ( j=rowners[i]; j< rowners[i+1]; j++ ) {
        procsnz[i] += rowlengths[j];
      }
    }
    PETSCFREE(rowlengths);

    /* determine max buffer needed and allocate it */
    maxnz = 0;
    for ( i=0; i<numtid; i++ ) {
      maxnz = PETSCMAX(maxnz,procsnz[i]);
    }
    cols = (int *) PETSCMALLOC( maxnz*sizeof(int) ); CHKPTRQ(cols);

    /* read in my part of the matrix column indices  */
    nz = procsnz[0];
    mycols = (int *) PETSCMALLOC( nz*sizeof(int) ); CHKPTRQ(mycols);
    ierr = SYRead(fd,mycols,nz,SYINT); CHKERRQ(ierr);

    /* read in every one elses and ship off */
    for ( i=1; i<numtid; i++ ) {
      nz = procsnz[i];
      ierr = SYRead(fd,cols,nz,SYINT); CHKERRQ(ierr);
      MPI_Send(cols,nz,MPI_INT,i,tag,comm);
    }
    PETSCFREE(cols);
  }
  else {
    /* determine buffer space needed for message */
    nz = 0;
    for ( i=0; i<m; i++ ) {
      nz += ourlens[i];
    }
    mycols = (int*) PETSCMALLOC( nz*sizeof(int) ); CHKPTRQ(mycols);

    /* receive message of column indices*/
    MPI_Recv(mycols,nz,MPI_INT,0,tag,comm,&status);
    MPI_Get_count(&status,MPI_INT,&maxnz);
    if (maxnz != nz) SETERRQ(1,"MatLoad_MPIAIJorMPIRow:something is wrong with file");
  }

  /* loop over local rows, determining number of off diagonal entries */
  PetscZero(offlens,m*sizeof(int));
  jj = 0;
  for ( i=0; i<m; i++ ) {
    for ( j=0; j<ourlens[i]; j++ ) {
      if (mycols[jj] < rstart || mycols[jj] >= rend) offlens[i]++;
      jj++;
    }
  }


  /* create our matrix */
  for ( i=0; i<m; i++ ) {
    ourlens[i] -= offlens[i];
  }
  if (type == MATMPIAIJ) {
    ierr = MatCreateMPIAIJ(comm,m,PETSC_DECIDE,M,N,0,ourlens,0,offlens,newmat);CHKERRQ(ierr);
  }
  else if (type == MATMPIROW) {
    ierr = MatCreateMPIRow(comm,m,PETSC_DECIDE,M,N,0,ourlens,0,offlens,newmat);CHKERRQ(ierr);
  }
  A = *newmat;
  MatSetOption(A,COLUMNS_SORTED); 
  for ( i=0; i<m; i++ ) {
    ourlens[i] += offlens[i];
  }

  if (!mytid) {
    vals = (Scalar *) PETSCMALLOC( maxnz*sizeof(Scalar) ); CHKPTRQ(vals);

    /* read in my part of the matrix numerical values  */
    nz = procsnz[0];
    ierr = SYRead(fd,vals,nz,SYSCALAR); CHKERRQ(ierr);
    
    /* insert into matrix */
    jj      = rstart;
    smycols = mycols;
    svals   = vals;
    for ( i=0; i<m; i++ ) {
      ierr = MatSetValues(A,1,&jj,ourlens[i],smycols,svals,INSERT_VALUES);CHKERRQ(ierr);
      smycols += ourlens[i];
      svals   += ourlens[i];
      jj++;
    }

    /* read in other processors and ship out */
    for ( i=1; i<numtid; i++ ) {
      nz = procsnz[i];
      ierr = SYRead(fd,vals,nz,SYSCALAR); CHKERRQ(ierr);
      MPI_Send(vals,nz,MPIU_SCALAR,i,A->tag,comm);
    }
    PETSCFREE(procsnz);
  }
  else {
    /* receive numeric values */
    vals = (Scalar*) PETSCMALLOC( nz*sizeof(Scalar) ); CHKPTRQ(vals);

    /* receive message of values*/
    MPI_Recv(vals,nz,MPIU_SCALAR,0,A->tag,comm,&status);
    MPI_Get_count(&status,MPIU_SCALAR,&maxnz);
    if (maxnz != nz) SETERRQ(1,"MatLoad_MPIAIJorMPIRow:something is wrong with file");

    /* insert into matrix */
    jj      = rstart;
    smycols = mycols;
    svals   = vals;
    for ( i=0; i<m; i++ ) {
      ierr = MatSetValues(A,1,&jj,ourlens[i],smycols,svals,INSERT_VALUES);CHKERRQ(ierr);
      smycols += ourlens[i];
      svals   += ourlens[i];
      jj++;
    }
  }
  PETSCFREE(ourlens); PETSCFREE(vals); PETSCFREE(mycols); PETSCFREE(rowners);

  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
