#ifndef lint
static char vcid[] = "$Id: mpiaij.c,v 1.140 1996/04/11 14:07:38 curfman Exp balay $";
#endif

#include "mpiaij.h"
#include "src/vec/vecimpl.h"
#include "vec/inline/spops.h"

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

  aij->colmap = (int *) PetscMalloc(aij->N*sizeof(int));CHKPTRQ(aij->colmap);
  PLogObjectMemory(mat,aij->N*sizeof(int));
  PetscMemzero(aij->colmap,aij->N*sizeof(int));
  for ( i=0; i<n; i++ ) aij->colmap[aij->garray[i]] = i-shift;
  return 0;
}

extern int DisAssemble_MPIAIJ(Mat);

static int MatGetReordering_MPIAIJ(Mat mat,MatOrdering type,IS *rperm,IS *cperm)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  int        ierr;
  if (aij->size == 1) {
    ierr = MatGetReordering(aij->A,type,rperm,cperm); CHKERRQ(ierr);
  } else SETERRQ(1,"MatGetReordering_MPIAIJ:not supported in parallel");
  return 0;
}

static int MatSetValues_MPIAIJ(Mat mat,int m,int *im,int n,int *in,Scalar *v,InsertMode addv)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Mat_SeqAIJ *C = (Mat_SeqAIJ*) aij->A->data;
  Scalar     value;
  int        ierr,i,j, rstart = aij->rstart, rend = aij->rend;
  int        cstart = aij->cstart, cend = aij->cend,row,col;
  int        shift = C->indexshift,roworiented = aij->roworiented;

  if (aij->insertmode != NOT_SET_VALUES && aij->insertmode != addv) {
    SETERRQ(1,"MatSetValues_MPIAIJ:Cannot mix inserts and adds");
  }
  aij->insertmode = addv;
  for ( i=0; i<m; i++ ) {
    if (im[i] < 0) SETERRQ(1,"MatSetValues_MPIAIJ:Negative row");
    if (im[i] >= aij->M) SETERRQ(1,"MatSetValues_MPIAIJ:Row too large");
    if (im[i] >= rstart && im[i] < rend) {
      row = im[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (in[j] < 0) SETERRQ(1,"MatSetValues_MPIAIJ:Negative column");
        if (in[j] >= aij->N) SETERRQ(1,"MatSetValues_MPIAIJ:Col too large");
        if (in[j] >= cstart && in[j] < cend){
          col = in[j] - cstart;
          if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
          ierr = MatSetValues(aij->A,1,&row,1,&col,&value,addv);CHKERRQ(ierr);
        }
        else {
          if (mat->was_assembled) {
            if (!aij->colmap) {ierr = CreateColmap_Private(mat);CHKERRQ(ierr);}
            col = aij->colmap[in[j]] + shift;
            if (col < 0 && !((Mat_SeqAIJ*)(aij->A->data))->nonew) {
              ierr = DisAssemble_MPIAIJ(mat); CHKERRQ(ierr);
              col =  in[j];              
            }
          }
          else col = in[j];
          if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
          ierr = MatSetValues(aij->B,1,&row,1,&col,&value,addv);CHKERRQ(ierr);
        }
      }
    } 
    else {
      if (roworiented) {
        ierr = StashValues_Private(&aij->stash,im[i],n,in,v+i*n,addv);CHKERRQ(ierr);
      }
      else {
        row = im[i];
        for ( j=0; j<n; j++ ) {
          ierr = StashValues_Private(&aij->stash,row,1,in+j,v+i+j*m,addv);CHKERRQ(ierr);
        }
      }
    }
  }
  return 0;
}

static int MatGetValues_MPIAIJ(Mat mat,int m,int *idxm,int n,int *idxn,Scalar *v)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Mat_SeqAIJ *C = (Mat_SeqAIJ*) aij->A->data;
  int        ierr,i,j, rstart = aij->rstart, rend = aij->rend;
  int        cstart = aij->cstart, cend = aij->cend,row,col;
  int        shift = C->indexshift;

  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) SETERRQ(1,"MatGetValues_MPIAIJ:Negative row");
    if (idxm[i] >= aij->M) SETERRQ(1,"MatGetValues_MPIAIJ:Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (idxn[j] < 0) SETERRQ(1,"MatGetValues_MPIAIJ:Negative column");
        if (idxn[j] >= aij->N) SETERRQ(1,"MatGetValues_MPIAIJ:Col too large");
        if (idxn[j] >= cstart && idxn[j] < cend){
          col = idxn[j] - cstart;
          ierr = MatGetValues(aij->A,1,&row,1,&col,v+i*n+j); CHKERRQ(ierr);
        }
        else {
          col = aij->colmap[idxn[j]] + shift;
          ierr = MatGetValues(aij->B,1,&row,1,&col,v+i*n+j); CHKERRQ(ierr);
        }
      }
    } 
    else {
      SETERRQ(1,"MatGetValues_MPIAIJ:Only local values currently supported");
    }
  }
  return 0;
}

static int MatAssemblyBegin_MPIAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIAIJ  *aij = (Mat_MPIAIJ *) mat->data;
  MPI_Comm    comm = mat->comm;
  int         size = aij->size, *owners = aij->rowners;
  int         rank = aij->rank,tag = mat->tag, *owner,*starts,count,ierr;
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
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PetscMalloc( (aij->stash.n+1)*sizeof(int) ); CHKPTRQ(owner);
  for ( i=0; i<aij->stash.n; i++ ) {
    idx = aij->stash.idx[i];
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; break;
      }
    }
  }
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce(procs, work,size,MPI_INT,MPI_SUM,comm);
  nreceives = work[rank]; 
  MPI_Allreduce( nprocs, work,size,MPI_INT,MPI_MAX,comm);
  nmax = work[rank];
  PetscFree(work);

  /* post receives: 
       1) each message will consist of ordered pairs 
     (global index,value) we store the global index as a double 
     to simplify the message passing. 
       2) since we don't know how long each individual message is we 
     allocate the largest needed buffer for each receive. Potentially 
     this is a lot of wasted space.


       This could be done better.
  */
  rvalues = (Scalar *) PetscMalloc(3*(nreceives+1)*(nmax+1)*sizeof(Scalar));
  CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nreceives+1)*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0; i<nreceives; i++ ) {
    MPI_Irecv(rvalues+3*nmax*i,3*nmax,MPIU_SCALAR,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (Scalar *) PetscMalloc(3*(aij->stash.n+1)*sizeof(Scalar));CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PetscMalloc( (nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  starts = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<aij->stash.n; i++ ) {
    svalues[3*starts[owner[i]]]       = (Scalar)  aij->stash.idx[i];
    svalues[3*starts[owner[i]]+1]     = (Scalar)  aij->stash.idy[i];
    svalues[3*(starts[owner[i]]++)+2] =  aij->stash.array[i];
  }
  PetscFree(owner);
  starts[0] = 0;
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<size; i++ ) {
    if (procs[i]) {
      MPI_Isend(svalues+3*starts[i],3*nprocs[i],MPIU_SCALAR,i,tag,
                comm,send_waits+count++);
    }
  }
  PetscFree(starts); PetscFree(nprocs);

  /* Free cache space */
  PLogInfo(0,"[%d]MatAssemblyBegin_MPIAIJ:Number of off processor values %d\n",rank,aij->stash.n);
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
      row = (int) PetscReal(values[3*i]) - aij->rstart;
      col = (int) PetscReal(values[3*i+1]);
      val = values[3*i+2];
      if (col >= aij->cstart && col < aij->cend) {
        col -= aij->cstart;
        MatSetValues(aij->A,1,&row,1,&col,&val,addv);
      } 
      else {
        if (mat->was_assembled) {
          if (!aij->colmap) {ierr = CreateColmap_Private(mat);CHKERRQ(ierr);}
          col = aij->colmap[col] + shift;
          if (col < 0  && !((Mat_SeqAIJ*)(aij->A->data))->nonew) {
            ierr = DisAssemble_MPIAIJ(mat); CHKERRQ(ierr);
            col = (int) PetscReal(values[3*i+1]);
          }
        }
        MatSetValues(aij->B,1,&row,1,&col,&val,addv);
      }
    }
    count--;
  }
  PetscFree(aij->recv_waits); PetscFree(aij->rvalues);
 
  /* wait on sends */
  if (aij->nsends) {
    send_status = (MPI_Status *) PetscMalloc(aij->nsends*sizeof(MPI_Status));
    CHKPTRQ(send_status);
    MPI_Waitall(aij->nsends,aij->send_waits,send_status);
    PetscFree(send_status);
  }
  PetscFree(aij->send_waits); PetscFree(aij->svalues);

  aij->insertmode = NOT_SET_VALUES;
  ierr = MatAssemblyBegin(aij->A,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(aij->A,mode); CHKERRQ(ierr);

  /* determine if any processor has disassembled, if so we must 
     also disassemble ourselfs, in order that we may reassemble. */
  MPI_Allreduce(&mat->was_assembled,&other_disassembled,1,MPI_INT,MPI_PROD,mat->comm);
  if (mat->was_assembled && !other_disassembled) {
    ierr = DisAssemble_MPIAIJ(mat); CHKERRQ(ierr);
  }

  if (!mat->was_assembled && mode == FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIAIJ(mat); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(aij->B,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(aij->B,mode); CHKERRQ(ierr);

  if (aij->rowvalues) {PetscFree(aij->rowvalues); aij->rowvalues = 0;}
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
  int            i,ierr,N, *rows,*owners = l->rowners,size = l->size;
  int            *procs,*nprocs,j,found,idx,nsends,*work;
  int            nmax,*svalues,*starts,*owner,nrecvs,rank = l->rank;
  int            *rvalues,tag = A->tag,count,base,slen,n,*source;
  int            *lens,imdex,*lrows,*values;
  MPI_Comm       comm = A->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
  IS             istmp;

  ierr = ISGetSize(is,&N); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows); CHKERRQ(ierr);

  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PetscMalloc((N+1)*sizeof(int)); CHKPTRQ(owner); /* see note*/
  for ( i=0; i<N; i++ ) {
    idx = rows[i];
    found = 0;
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERRQ(1,"MatZeroRows_MPIAIJ:Index out of range");
  }
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce( procs, work,size,MPI_INT,MPI_SUM,comm);
  nrecvs = work[rank]; 
  MPI_Allreduce( nprocs, work,size,MPI_INT,MPI_MAX,comm);
  nmax = work[rank];
  PetscFree(work);

  /* post receives:   */
  rvalues = (int *) PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int)); /*see note */
  CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nrecvs+1)*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv(rvalues+nmax*i,nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (int *) PetscMalloc( (N+1)*sizeof(int) ); CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PetscMalloc( (nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  starts = (int *) PetscMalloc( (size+1)*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<N; i++ ) {
    svalues[starts[owner[i]]++] = rows[i];
  }
  ISRestoreIndices(is,&rows);

  starts[0] = 0;
  for ( i=1; i<size+1; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<size; i++ ) {
    if (procs[i]) {
      MPI_Isend(svalues+starts[i],nprocs[i],MPI_INT,i,tag,comm,send_waits+count++);
    }
  }
  PetscFree(starts);

  base = owners[rank];

  /*  wait on receives */
  lens   = (int *) PetscMalloc( 2*(nrecvs+1)*sizeof(int) ); CHKPTRQ(lens);
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
  PetscFree(recv_waits); 
  
  /* move the data into the send scatter */
  lrows = (int *) PetscMalloc( (slen+1)*sizeof(int) ); CHKPTRQ(lrows);
  count = 0;
  for ( i=0; i<nrecvs; i++ ) {
    values = rvalues + i*nmax;
    for ( j=0; j<lens[i]; j++ ) {
      lrows[count++] = values[j] - base;
    }
  }
  PetscFree(rvalues); PetscFree(lens);
  PetscFree(owner); PetscFree(nprocs);
    
  /* actually zap the local rows */
  ierr = ISCreateSeq(MPI_COMM_SELF,slen,lrows,&istmp);CHKERRQ(ierr);   
  PLogObjectParent(A,istmp);
  PetscFree(lrows);
  ierr = MatZeroRows(l->A,istmp,diag); CHKERRQ(ierr);
  ierr = MatZeroRows(l->B,istmp,0); CHKERRQ(ierr);
  ierr = ISDestroy(istmp); CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status *) PetscMalloc(nsends*sizeof(MPI_Status));
    CHKPTRQ(send_status);
    MPI_Waitall(nsends,send_waits,send_status);
    PetscFree(send_status);
  }
  PetscFree(send_waits); PetscFree(svalues);

  return 0;
}

static int MatMult_MPIAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;

  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_ALL,a->Mvctx); CHKERRQ(ierr);
  ierr = MatMult_SeqAIJ(a->A,xx,yy); CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_ALL,a->Mvctx); CHKERRQ(ierr);
  ierr = MatMultAdd_SeqAIJ(a->B,a->lvec,yy,yy); CHKERRQ(ierr);
  return 0;
}

static int MatMultAdd_MPIAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;
  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_ALL,a->Mvctx);CHKERRQ(ierr);
  ierr = MatMultAdd_SeqAIJ(a->A,xx,yy,zz); CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_ALL,a->Mvctx);CHKERRQ(ierr);
  ierr = MatMultAdd_SeqAIJ(a->B,a->lvec,zz,zz); CHKERRQ(ierr);
  return 0;
}

static int MatMultTrans_MPIAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;

  /* do nondiagonal part */
  ierr = MatMultTrans_SeqAIJ(a->B,xx,a->lvec); CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(a->lvec,yy,ADD_VALUES,
                (ScatterMode)(SCATTER_ALL|SCATTER_REVERSE),a->Mvctx); CHKERRQ(ierr);
  /* do local part */
  ierr = MatMultTrans_SeqAIJ(a->A,xx,yy); CHKERRQ(ierr);
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

  /* do nondiagonal part */
  ierr = MatMultTrans_SeqAIJ(a->B,xx,a->lvec); CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(a->lvec,zz,ADD_VALUES,
                 (ScatterMode)(SCATTER_ALL|SCATTER_REVERSE),a->Mvctx); CHKERRQ(ierr);
  /* do local part */
  ierr = MatMultTransAdd_SeqAIJ(a->A,xx,yy,zz); CHKERRQ(ierr);
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
  if (a->M != a->N) 
    SETERRQ(1,"MatGetDiagonal_MPIAIJ:Supports only square matrix where A->A is diag block");
  return MatGetDiagonal(a->A,v);
}

static int MatScale_MPIAIJ(Scalar *aa,Mat A)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *) A->data;
  int        ierr;
  ierr = MatScale(aa,a->A); CHKERRQ(ierr);
  ierr = MatScale(aa,a->B); CHKERRQ(ierr);
  return 0;
}

static int MatDestroy_MPIAIJ(PetscObject obj)
{
  Mat        mat = (Mat) obj;
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  int        ierr;
#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d",aij->M,aij->N);
#endif
  PetscFree(aij->rowners); 
  ierr = MatDestroy(aij->A); CHKERRQ(ierr);
  ierr = MatDestroy(aij->B); CHKERRQ(ierr);
  if (aij->colmap) PetscFree(aij->colmap);
  if (aij->garray) PetscFree(aij->garray);
  if (aij->lvec)   VecDestroy(aij->lvec);
  if (aij->Mvctx)  VecScatterDestroy(aij->Mvctx);
  if (aij->rowvalues) PetscFree(aij->rowvalues);
  PetscFree(aij); 
  PLogObjectDestroy(mat);
  PetscHeaderDestroy(mat);
  return 0;
}
#include "draw.h"
#include "pinclude/pviewer.h"

static int MatView_MPIAIJ_Binary(Mat mat,Viewer viewer)
{
  Mat_MPIAIJ  *aij = (Mat_MPIAIJ *) mat->data;
  int         ierr;

  if (aij->size == 1) {
    ierr = MatView(aij->A,viewer); CHKERRQ(ierr);
  }
  else SETERRQ(1,"MatView_MPIAIJ_Binary:Only uniprocessor output supported");
  return 0;
}

static int MatView_MPIAIJ_ASCIIorDraworMatlab(Mat mat,Viewer viewer)
{
  Mat_MPIAIJ  *aij = (Mat_MPIAIJ *) mat->data;
  Mat_SeqAIJ* C = (Mat_SeqAIJ*)aij->A->data;
  int         ierr, format,shift = C->indexshift,rank;
  FILE        *fd;
  ViewerType  vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILES_VIEWER || vtype == ASCII_FILE_VIEWER) { 
    ierr = ViewerGetFormat(viewer,&format);
    if (format == ASCII_FORMAT_INFO_DETAILED) {
      int nz, nzalloc, mem, flg;
      MPI_Comm_rank(mat->comm,&rank);
      ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
      ierr = MatGetInfo(mat,MAT_LOCAL,&nz,&nzalloc,&mem); 
      ierr = OptionsHasName(PETSC_NULL,"-mat_aij_no_inode",&flg); CHKERRQ(ierr);
      PetscSequentialPhaseBegin(mat->comm,1);
      if (flg) fprintf(fd,"[%d] Local rows %d nz %d nz alloced %d mem %d, not using I-node routines\n",
         rank,aij->m,nz,nzalloc,mem);       
      else fprintf(fd,"[%d] Local rows %d nz %d nz alloced %d mem %d, using I-node routines\n",
         rank,aij->m,nz,nzalloc,mem);       
      ierr = MatGetInfo(aij->A,MAT_LOCAL,&nz,&nzalloc,&mem); 
      fprintf(fd,"[%d] on-diagonal part: nz %d \n",rank,nz);
      ierr = MatGetInfo(aij->B,MAT_LOCAL,&nz,&nzalloc,&mem); 
      fprintf(fd,"[%d] off-diagonal part: nz %d \n",rank,nz); 
      fflush(fd);
      PetscSequentialPhaseEnd(mat->comm,1);
      ierr = VecScatterView(aij->Mvctx,viewer); CHKERRQ(ierr);
      return 0; 
    }
    else if (format == ASCII_FORMAT_INFO) {
      return 0;
    }
  }

  if (vtype == DRAW_VIEWER) {
    Draw       draw;
    PetscTruth isnull;
    ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRQ(ierr);
    ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) return 0;
  }

  if (vtype == ASCII_FILE_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    PetscSequentialPhaseBegin(mat->comm,1);
    fprintf(fd,"[%d] rows %d starts %d ends %d cols %d starts %d ends %d\n",
           aij->rank,aij->m,aij->rstart,aij->rend,aij->n,aij->cstart,
           aij->cend);
    ierr = MatView(aij->A,viewer); CHKERRQ(ierr);
    ierr = MatView(aij->B,viewer); CHKERRQ(ierr);
    fflush(fd);
    PetscSequentialPhaseEnd(mat->comm,1);
  }
  else {
    int size = aij->size;
    rank = aij->rank;
    if (size == 1) {
      ierr = MatView(aij->A,viewer); CHKERRQ(ierr);
    }
    else {
      /* assemble the entire matrix onto first processor. */
      Mat         A;
      Mat_SeqAIJ *Aloc;
      int         M = aij->M, N = aij->N,m,*ai,*aj,row,*cols,i,*ct;
      Scalar      *a;

      if (!rank) {
        ierr = MatCreateMPIAIJ(mat->comm,M,N,M,N,0,PETSC_NULL,0,PETSC_NULL,&A);
               CHKERRQ(ierr);
      }
      else {
        ierr = MatCreateMPIAIJ(mat->comm,0,0,M,N,0,PETSC_NULL,0,PETSC_NULL,&A);
               CHKERRQ(ierr);
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
      ct = cols = (int *) PetscMalloc( (ai[m]+1)*sizeof(int) ); CHKPTRQ(cols);
      for ( i=0; i<ai[m]+shift; i++ ) {cols[i] = aij->garray[aj[i]+shift];}
      for ( i=0; i<m; i++ ) {
        ierr = MatSetValues(A,1,&row,ai[i+1]-ai[i],cols,a,INSERT_VALUES);CHKERRQ(ierr);
        row++; a += ai[i+1]-ai[i]; cols += ai[i+1]-ai[i];
      } 
      PetscFree(ct);
      ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
      if (!rank) {
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
  int         ierr;
  ViewerType  vtype;
 
  if (!viewer) { 
    viewer = STDOUT_VIEWER_SELF; 
  }
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER ||
      vtype == DRAW_VIEWER       || vtype == MATLAB_VIEWER) { 
    ierr = MatView_MPIAIJ_ASCIIorDraworMatlab(mat,viewer); CHKERRQ(ierr);
  }
  else if (vtype == BINARY_FILE_VIEWER) {
    return MatView_MPIAIJ_Binary(mat,viewer);
  }
  return 0;
}

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
      return MatRelax_SeqAIJ(mat->A,bb,omega,flag,fshift,its,xx);
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
      return MatRelax_SeqAIJ(mat->A,bb,omega,flag,fshift,its,xx);
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
      return MatRelax_SeqAIJ(mat->A,bb,omega,flag,fshift,its,xx);
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
    if (nz)       *nz      = isend[0];
    if (nzalloc)  *nzalloc = isend[1];
    if (mem)      *mem     = isend[2];
  } else if (flag == MAT_GLOBAL_MAX) {
    MPI_Allreduce( isend, irecv,3,MPI_INT,MPI_MAX,matin->comm);
    if (nz)      *nz      = irecv[0];
    if (nzalloc) *nzalloc = irecv[1];
    if (mem)     *mem     = irecv[2];
  } else if (flag == MAT_GLOBAL_SUM) {
    MPI_Allreduce( isend, irecv,3,MPI_INT,MPI_SUM,matin->comm);
    if (nz)      *nz      = irecv[0]; 
    if (nzalloc) *nzalloc = irecv[1]; 
    if (mem)     *mem     = irecv[2];
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
    PLogInfo(A,"Info:MatSetOption_MPIAIJ:Option ignored\n");
  else if (op == COLUMN_ORIENTED) {
    a->roworiented = 0;
    MatSetOption(a->A,op);
    MatSetOption(a->B,op);
  }
  else if (op == NO_NEW_DIAGONALS)
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_MPIAIJ:NO_NEW_DIAGONALS");}
  else 
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_MPIAIJ:unknown option");}
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

extern int MatGetRow_SeqAIJ(Mat,int,int*,int**,Scalar**);
extern int MatRestoreRow_SeqAIJ(Mat,int,int*,int**,Scalar**);

int MatGetRow_MPIAIJ(Mat matin,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *) matin->data;
  Scalar     *vworkA, *vworkB, **pvA, **pvB,*v_p;
  int        i, ierr, *cworkA, *cworkB, **pcA, **pcB, cstart = mat->cstart;
  int        nztot, nzA, nzB, lrow, rstart = mat->rstart, rend = mat->rend;
  int        *cmap, *idx_p;

  if (mat->getrowactive == PETSC_TRUE) SETERRQ(1,"MatGetRow_MPIAIJ:Already active");
  mat->getrowactive = PETSC_TRUE;

  if (!mat->rowvalues && (idx || v)) {
    /*
        allocate enough space to hold information from the longest row.
    */
    Mat_SeqAIJ *Aa = (Mat_SeqAIJ *) mat->A->data,*Ba = (Mat_SeqAIJ *) mat->B->data; 
    int     max = 1,n = mat->n,tmp;
    for ( i=0; i<n; i++ ) {
      tmp = Aa->i[i+1] - Aa->i[i] + Ba->i[i+1] - Ba->i[i];
      if (max < tmp) { max = tmp; }
    }
    mat->rowvalues = (Scalar *) PetscMalloc( max*(sizeof(int)+sizeof(Scalar))); 
    CHKPTRQ(mat->rowvalues);
    mat->rowindices = (int *) (mat->rowvalues + max);
  }
       

  if (row < rstart || row >= rend) SETERRQ(1,"MatGetRow_MPIAIJ:Only local rows")
  lrow = row - rstart;

  pvA = &vworkA; pcA = &cworkA; pvB = &vworkB; pcB = &cworkB;
  if (!v)   {pvA = 0; pvB = 0;}
  if (!idx) {pcA = 0; if (!v) pcB = 0;}
  ierr = MatGetRow_SeqAIJ(mat->A,lrow,&nzA,pcA,pvA); CHKERRQ(ierr);
  ierr = MatGetRow_SeqAIJ(mat->B,lrow,&nzB,pcB,pvB); CHKERRQ(ierr);
  nztot = nzA + nzB;

  cmap  = mat->garray;
  if (v  || idx) {
    if (nztot) {
      /* Sort by increasing column numbers, assuming A and B already sorted */
      int imark = -1;
      if (v) {
        *v = v_p = mat->rowvalues;
        for ( i=0; i<nzB; i++ ) {
          if (cmap[cworkB[i]] < cstart)   v_p[i] = vworkB[i];
          else break;
        }
        imark = i;
        for ( i=0; i<nzA; i++ )     v_p[imark+i] = vworkA[i];
        for ( i=imark; i<nzB; i++ ) v_p[nzA+i]   = vworkB[i];
      }
      if (idx) {
        *idx = idx_p = mat->rowindices;
        if (imark > -1) {
          for ( i=0; i<imark; i++ ) {
            idx_p[i] = cmap[cworkB[i]];
          }
        } else {
          for ( i=0; i<nzB; i++ ) {
            if (cmap[cworkB[i]] < cstart)   idx_p[i] = cmap[cworkB[i]];
            else break;
          }
          imark = i;
        }
        for ( i=0; i<nzA; i++ )     idx_p[imark+i] = cstart + cworkA[i];
        for ( i=imark; i<nzB; i++ ) idx_p[nzA+i]   = cmap[cworkB[i]];
      } 
    } 
    else {*idx = 0; *v=0;}
  }
  *nz = nztot;
  ierr = MatRestoreRow_SeqAIJ(mat->A,lrow,&nzA,pcA,pvA); CHKERRQ(ierr);
  ierr = MatRestoreRow_SeqAIJ(mat->B,lrow,&nzB,pcB,pvB); CHKERRQ(ierr);
  return 0;
}

int MatRestoreRow_MPIAIJ(Mat mat,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  if (aij->getrowactive == PETSC_FALSE) {
    SETERRQ(1,"MatRestoreRow_MPIAIJ:MatGetRow not called");
  }
  aij->getrowactive = PETSC_FALSE;
  return 0;
}

static int MatNorm_MPIAIJ(Mat mat,NormType type,double *norm)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *) mat->data;
  Mat_SeqAIJ *amat = (Mat_SeqAIJ*) aij->A->data, *bmat = (Mat_SeqAIJ*) aij->B->data;
  int        ierr, i, j, cstart = aij->cstart,shift = amat->indexshift;
  double     sum = 0.0;
  Scalar     *v;

  if (aij->size == 1) {
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
      MPI_Allreduce(&sum,norm,1,MPI_DOUBLE,MPI_SUM,mat->comm);
      *norm = sqrt(*norm);
    }
    else if (type == NORM_1) { /* max column norm */
      double *tmp, *tmp2;
      int    *jj, *garray = aij->garray;
      tmp  = (double *) PetscMalloc( aij->N*sizeof(double) ); CHKPTRQ(tmp);
      tmp2 = (double *) PetscMalloc( aij->N*sizeof(double) ); CHKPTRQ(tmp2);
      PetscMemzero(tmp,aij->N*sizeof(double));
      *norm = 0.0;
      v = amat->a; jj = amat->j;
      for ( j=0; j<amat->nz; j++ ) {
        tmp[cstart + *jj++ + shift] += PetscAbsScalar(*v);  v++;
      }
      v = bmat->a; jj = bmat->j;
      for ( j=0; j<bmat->nz; j++ ) {
        tmp[garray[*jj++ + shift]] += PetscAbsScalar(*v); v++;
      }
      MPI_Allreduce(tmp,tmp2,aij->N,MPI_DOUBLE,MPI_SUM,mat->comm);
      for ( j=0; j<aij->N; j++ ) {
        if (tmp2[j] > *norm) *norm = tmp2[j];
      }
      PetscFree(tmp); PetscFree(tmp2);
    }
    else if (type == NORM_INFINITY) { /* max row norm */
      double ntemp = 0.0;
      for ( j=0; j<amat->m; j++ ) {
        v = amat->a + amat->i[j] + shift;
        sum = 0.0;
        for ( i=0; i<amat->i[j+1]-amat->i[j]; i++ ) {
          sum += PetscAbsScalar(*v); v++;
        }
        v = bmat->a + bmat->i[j] + shift;
        for ( i=0; i<bmat->i[j+1]-bmat->i[j]; i++ ) {
          sum += PetscAbsScalar(*v); v++;
        }
        if (sum > ntemp) ntemp = sum;
      }
      MPI_Allreduce(&ntemp,norm,1,MPI_DOUBLE,MPI_MAX,mat->comm);
    }
    else {
      SETERRQ(1,"MatNorm_MPIAIJ:No support for two norm");
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

  if (matout == PETSC_NULL && M != N) 
    SETERRQ(1,"MatTranspose_MPIAIJ:Square matrix only for in-place");
  ierr = MatCreateMPIAIJ(A->comm,PETSC_DECIDE,PETSC_DECIDE,N,M,0,PETSC_NULL,0,
         PETSC_NULL,&B); CHKERRQ(ierr);

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
  for ( i=0; i<ai[m]+shift; i++ ) {aj[i] -= a->cstart + shift;}

  /* copy over the B part */
  Aloc = (Mat_SeqAIJ*) a->B->data;
  m = Aloc->m;  ai = Aloc->i; aj = Aloc->j; array = Aloc->a;
  row = a->rstart;
  ct = cols = (int *) PetscMalloc( (1+ai[m]-shift)*sizeof(int) ); CHKPTRQ(cols);
  for ( i=0; i<ai[m]+shift; i++ ) {cols[i] = a->garray[aj[i]+shift];}
  for ( i=0; i<m; i++ ) {
    ierr = MatSetValues(B,ai[i+1]-ai[i],cols,1,&row,array,INSERT_VALUES);CHKERRQ(ierr);
    row++; array += ai[i+1]-ai[i]; cols += ai[i+1]-ai[i];
  } 
  PetscFree(ct);
  ierr = MatAssemblyBegin(B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  if (matout != PETSC_NULL) {
    *matout = B;
  } else {
    /* This isn't really an in-place transpose .... but free data structures from a */
    PetscFree(a->rowners); 
    ierr = MatDestroy(a->A); CHKERRQ(ierr);
    ierr = MatDestroy(a->B); CHKERRQ(ierr);
    if (a->colmap) PetscFree(a->colmap);
    if (a->garray) PetscFree(a->garray);
    if (a->lvec) VecDestroy(a->lvec);
    if (a->Mvctx) VecScatterDestroy(a->Mvctx);
    PetscFree(a); 
    PetscMemcpy(A,B,sizeof(struct _Mat)); 
    PetscHeaderDestroy(B);
  }
  return 0;
}

extern int MatPrintHelp_SeqAIJ(Mat);
static int MatPrintHelp_MPIAIJ(Mat A)
{
  Mat_MPIAIJ *a   = (Mat_MPIAIJ*) A->data;

  if (!a->rank) return MatPrintHelp_SeqAIJ(a->A);
  else return 0;
}

extern int MatConvert_MPIAIJ(Mat,MatType,Mat *);
static int MatConvertSameType_MPIAIJ(Mat,Mat *,int);
extern int MatIncreaseOverlap_MPIAIJ(Mat , int, IS *, int);
int MatGetSubMatrices_MPIAIJ (Mat ,int , IS *,IS *,MatGetSubMatrixCall,Mat **);
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
       0,0,MatConvert_MPIAIJ,0,0,MatConvertSameType_MPIAIJ,0,0,
       0,0,0,
       MatGetSubMatrices_MPIAIJ,MatIncreaseOverlap_MPIAIJ,MatGetValues_MPIAIJ,0,
       MatPrintHelp_MPIAIJ,
       MatScale_MPIAIJ};

/*@C
   MatCreateMPIAIJ - Creates a sparse parallel matrix in AIJ format
   (the default parallel PETSc format).  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameters 
   d_nz (or d_nnz) and o_nz (or o_nnz).  By setting these parameters accurately,
   performance can be increased by more than a factor of 50.

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
.  A - the matrix 

   Notes:
   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users manual for details.

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   By default, this format uses inodes (identical nodes) when possible.
   We search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Options Database Keys:
$    -mat_aij_no_inode  - Do not use inodes
$    -mat_aij_inode_limit <limit> - Set inode limit.
$        (max limit=5)
$    -mat_aij_oneindex - Internally use indexing starting at 1
$        rather than 0.  Note: When calling MatSetValues(),
$        the user still MUST index entries starting at 0!

   Storage Information:
   For a square global matrix we define each processor's diagonal portion 
   to be its local rows and the corresponding columns (a square submatrix);  
   each processor's off-diagonal portion encompasses the remainder of the
   local matrix (a rectangular submatrix). 

   The user can specify preallocated storage for the diagonal part of
   the local submatrix with either d_nz or d_nnz (not both).  Set 
   d_nz=PETSC_DEFAULT and d_nnz=PETSC_NULL for PETSc to control dynamic
   memory allocation.  Likewise, specify preallocated storage for the
   off-diagonal part of the local submatrix with o_nz or o_nnz (not both).

   Consider a processor that owns rows 3, 4 and 5 of a parallel matrix. In
   the figure below we depict these three local rows and all columns (0-11).

$          0 1 2 3 4 5 6 7 8 9 10 11
$         -------------------
$  row 3  |  o o o d d d o o o o o o
$  row 4  |  o o o d d d o o o o o o
$  row 5  |  o o o d d d o o o o o o
$         -------------------
$ 

   Thus, any entries in the d locations are stored in the d (diagonal) 
   submatrix, and any entries in the o locations are stored in the
   o (off-diagonal) submatrix.  Note that the d and the o submatrices are
   stored simply in the MATSEQAIJ format for compressed row storage.

   Now d_nz should indicate the number of nonzeros per row in the d matrix,
   and o_nz should indicate the number of nonzeros per row in the o matrix.
   In general, for PDE problems in which most nonzeros are near the diagonal,
   one expects d_nz >> o_nz.   For additional details, see the users manual
   chapter on matrices and the file $(PETSC_DIR)/Performance.

.keywords: matrix, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues()
@*/
int MatCreateMPIAIJ(MPI_Comm comm,int m,int n,int M,int N,
                    int d_nz,int *d_nnz,int o_nz,int *o_nnz,Mat *A)
{
  Mat          B;
  Mat_MPIAIJ   *b;
  int          ierr, i,sum[2],work[2];

  *A = 0;
  PetscHeaderCreate(B,_Mat,MAT_COOKIE,MATMPIAIJ,comm);
  PLogObjectCreate(B);
  B->data       = (void *) (b = PetscNew(Mat_MPIAIJ)); CHKPTRQ(b);
  PetscMemzero(b,sizeof(Mat_MPIAIJ));
  PetscMemcpy(&B->ops,&MatOps,sizeof(struct _MatOps));
  B->destroy    = MatDestroy_MPIAIJ;
  B->view       = MatView_MPIAIJ;
  B->factor     = 0;
  B->assembled  = PETSC_FALSE;

  b->insertmode = NOT_SET_VALUES;
  MPI_Comm_rank(comm,&b->rank);
  MPI_Comm_size(comm,&b->size);

  if (m == PETSC_DECIDE && (d_nnz != PETSC_NULL || o_nnz != PETSC_NULL)) 
    SETERRQ(1,"MatCreateMPIAIJ:Cannot have PETSC_DECIDE rows but set d_nnz or o_nnz");

  if (M == PETSC_DECIDE || N == PETSC_DECIDE) {
    work[0] = m; work[1] = n;
    MPI_Allreduce( work, sum,2,MPI_INT,MPI_SUM,comm );
    if (M == PETSC_DECIDE) M = sum[0];
    if (N == PETSC_DECIDE) N = sum[1];
  }
  if (m == PETSC_DECIDE) {m = M/b->size + ((M % b->size) > b->rank);}
  if (n == PETSC_DECIDE) {n = N/b->size + ((N % b->size) > b->rank);}
  b->m = m; B->m = m;
  b->n = n; B->n = n;
  b->N = N; B->N = N;
  b->M = M; B->M = M;

  /* build local table of row and column ownerships */
  b->rowners = (int *) PetscMalloc(2*(b->size+2)*sizeof(int)); CHKPTRQ(b->rowners);
  PLogObjectMemory(B,2*(b->size+2)*sizeof(int)+sizeof(struct _Mat)+sizeof(Mat_MPIAIJ));
  b->cowners = b->rowners + b->size + 1;
  MPI_Allgather(&m,1,MPI_INT,b->rowners+1,1,MPI_INT,comm);
  b->rowners[0] = 0;
  for ( i=2; i<=b->size; i++ ) {
    b->rowners[i] += b->rowners[i-1];
  }
  b->rstart = b->rowners[b->rank]; 
  b->rend   = b->rowners[b->rank+1]; 
  MPI_Allgather(&n,1,MPI_INT,b->cowners+1,1,MPI_INT,comm);
  b->cowners[0] = 0;
  for ( i=2; i<=b->size; i++ ) {
    b->cowners[i] += b->cowners[i-1];
  }
  b->cstart = b->cowners[b->rank]; 
  b->cend   = b->cowners[b->rank+1]; 

  if (d_nz == PETSC_DEFAULT) d_nz = 5;
  ierr = MatCreateSeqAIJ(MPI_COMM_SELF,m,n,d_nz,d_nnz,&b->A); CHKERRQ(ierr);
  PLogObjectParent(B,b->A);
  if (o_nz == PETSC_DEFAULT) o_nz = 0;
  ierr = MatCreateSeqAIJ(MPI_COMM_SELF,m,N,o_nz,o_nnz,&b->B); CHKERRQ(ierr);
  PLogObjectParent(B,b->B);

  /* build cache for off array entries formed */
  ierr = StashBuild_Private(&b->stash); CHKERRQ(ierr);
  b->colmap      = 0;
  b->garray      = 0;
  b->roworiented = 1;

  /* stuff used for matrix vector multiply */
  b->lvec      = 0;
  b->Mvctx     = 0;

  /* stuff for MatGetRow() */
  b->rowindices   = 0;
  b->rowvalues    = 0;
  b->getrowactive = PETSC_FALSE;

  *A = B;
  return 0;
}

static int MatConvertSameType_MPIAIJ(Mat matin,Mat *newmat,int cpvalues)
{
  Mat        mat;
  Mat_MPIAIJ *a,*oldmat = (Mat_MPIAIJ *) matin->data;
  int        ierr, len=0, flg;

  *newmat       = 0;
  PetscHeaderCreate(mat,_Mat,MAT_COOKIE,MATMPIAIJ,matin->comm);
  PLogObjectCreate(mat);
  mat->data       = (void *) (a = PetscNew(Mat_MPIAIJ)); CHKPTRQ(a);
  PetscMemcpy(&mat->ops,&MatOps,sizeof(struct _MatOps));
  mat->destroy    = MatDestroy_MPIAIJ;
  mat->view       = MatView_MPIAIJ;
  mat->factor     = matin->factor;
  mat->assembled  = PETSC_TRUE;

  a->m = mat->m   = oldmat->m;
  a->n = mat->n   = oldmat->n;
  a->M = mat->M   = oldmat->M;
  a->N = mat->N   = oldmat->N;

  a->rstart       = oldmat->rstart;
  a->rend         = oldmat->rend;
  a->cstart       = oldmat->cstart;
  a->cend         = oldmat->cend;
  a->size         = oldmat->size;
  a->rank         = oldmat->rank;
  a->insertmode   = NOT_SET_VALUES;
  a->rowvalues    = 0;
  a->getrowactive = PETSC_FALSE;

  a->rowners = (int *) PetscMalloc((a->size+1)*sizeof(int)); CHKPTRQ(a->rowners);
  PLogObjectMemory(mat,(a->size+1)*sizeof(int)+sizeof(struct _Mat)+sizeof(Mat_MPIAIJ));
  PetscMemcpy(a->rowners,oldmat->rowners,(a->size+1)*sizeof(int));
  ierr = StashInitialize_Private(&a->stash); CHKERRQ(ierr);
  if (oldmat->colmap) {
    a->colmap = (int *) PetscMalloc((a->N)*sizeof(int));CHKPTRQ(a->colmap);
    PLogObjectMemory(mat,(a->N)*sizeof(int));
    PetscMemcpy(a->colmap,oldmat->colmap,(a->N)*sizeof(int));
  } else a->colmap = 0;
  if (oldmat->garray && (len = ((Mat_SeqAIJ *) (oldmat->B->data))->n)) {
    a->garray = (int *) PetscMalloc(len*sizeof(int)); CHKPTRQ(a->garray);
    PLogObjectMemory(mat,len*sizeof(int));
    PetscMemcpy(a->garray,oldmat->garray,len*sizeof(int));
  } else a->garray = 0;
  
  ierr =  VecDuplicate(oldmat->lvec,&a->lvec); CHKERRQ(ierr);
  PLogObjectParent(mat,a->lvec);
  ierr =  VecScatterCopy(oldmat->Mvctx,&a->Mvctx); CHKERRQ(ierr);
  PLogObjectParent(mat,a->Mvctx);
  ierr =  MatConvert(oldmat->A,MATSAME,&a->A); CHKERRQ(ierr);
  PLogObjectParent(mat,a->A);
  ierr =  MatConvert(oldmat->B,MATSAME,&a->B); CHKERRQ(ierr);
  PLogObjectParent(mat,a->B);
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = MatPrintHelp(mat); CHKERRQ(ierr);
  }
  *newmat = mat;
  return 0;
}

#include "sys.h"

int MatLoad_MPIAIJ(Viewer viewer,MatType type,Mat *newmat)
{
  Mat          A;
  int          i, nz, ierr, j,rstart, rend, fd;
  Scalar       *vals,*svals;
  MPI_Comm     comm = ((PetscObject)viewer)->comm;
  MPI_Status   status;
  int          header[4],rank,size,*rowlengths = 0,M,N,m,*rowners,maxnz,*cols;
  int          *ourlens,*sndcounts = 0,*procsnz = 0, *offlens,jj,*mycols,*smycols;
  int          tag = ((PetscObject)viewer)->tag;

  MPI_Comm_size(comm,&size); MPI_Comm_rank(comm,&rank);
  if (!rank) {
    ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,BINARY_INT); CHKERRQ(ierr);
    if (header[0] != MAT_COOKIE) SETERRQ(1,"MatLoad_MPIAIJ:not matrix object");
  }

  MPI_Bcast(header+1,3,MPI_INT,0,comm);
  M = header[1]; N = header[2];
  /* determine ownership of all rows */
  m = M/size + ((M % size) > rank);
  rowners = (int *) PetscMalloc((size+2)*sizeof(int)); CHKPTRQ(rowners);
  MPI_Allgather(&m,1,MPI_INT,rowners+1,1,MPI_INT,comm);
  rowners[0] = 0;
  for ( i=2; i<=size; i++ ) {
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 

  /* distribute row lengths to all processors */
  ourlens = (int*) PetscMalloc( 2*(rend-rstart)*sizeof(int) ); CHKPTRQ(ourlens);
  offlens = ourlens + (rend-rstart);
  if (!rank) {
    rowlengths = (int*) PetscMalloc( M*sizeof(int) ); CHKPTRQ(rowlengths);
    ierr = PetscBinaryRead(fd,rowlengths,M,BINARY_INT); CHKERRQ(ierr);
    sndcounts = (int*) PetscMalloc( size*sizeof(int) ); CHKPTRQ(sndcounts);
    for ( i=0; i<size; i++ ) sndcounts[i] = rowners[i+1] - rowners[i];
    MPI_Scatterv(rowlengths,sndcounts,rowners,MPI_INT,ourlens,rend-rstart,MPI_INT,0,comm);
    PetscFree(sndcounts);
  }
  else {
    MPI_Scatterv(0,0,0,MPI_INT,ourlens,rend-rstart,MPI_INT, 0,comm);
  }

  if (!rank) {
    /* calculate the number of nonzeros on each processor */
    procsnz = (int*) PetscMalloc( size*sizeof(int) ); CHKPTRQ(procsnz);
    PetscMemzero(procsnz,size*sizeof(int));
    for ( i=0; i<size; i++ ) {
      for ( j=rowners[i]; j< rowners[i+1]; j++ ) {
        procsnz[i] += rowlengths[j];
      }
    }
    PetscFree(rowlengths);

    /* determine max buffer needed and allocate it */
    maxnz = 0;
    for ( i=0; i<size; i++ ) {
      maxnz = PetscMax(maxnz,procsnz[i]);
    }
    cols = (int *) PetscMalloc( maxnz*sizeof(int) ); CHKPTRQ(cols);

    /* read in my part of the matrix column indices  */
    nz = procsnz[0];
    mycols = (int *) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(mycols);
    ierr = PetscBinaryRead(fd,mycols,nz,BINARY_INT); CHKERRQ(ierr);

    /* read in every one elses and ship off */
    for ( i=1; i<size; i++ ) {
      nz = procsnz[i];
      ierr = PetscBinaryRead(fd,cols,nz,BINARY_INT); CHKERRQ(ierr);
      MPI_Send(cols,nz,MPI_INT,i,tag,comm);
    }
    PetscFree(cols);
  }
  else {
    /* determine buffer space needed for message */
    nz = 0;
    for ( i=0; i<m; i++ ) {
      nz += ourlens[i];
    }
    mycols = (int*) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(mycols);

    /* receive message of column indices*/
    MPI_Recv(mycols,nz,MPI_INT,0,tag,comm,&status);
    MPI_Get_count(&status,MPI_INT,&maxnz);
    if (maxnz != nz) SETERRQ(1,"MatLoad_MPIAIJ:something is wrong with file");
  }

  /* loop over local rows, determining number of off diagonal entries */
  PetscMemzero(offlens,m*sizeof(int));
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
  ierr = MatCreateMPIAIJ(comm,m,PETSC_DECIDE,M,N,0,ourlens,0,offlens,newmat);CHKERRQ(ierr);
  A = *newmat;
  MatSetOption(A,COLUMNS_SORTED); 
  for ( i=0; i<m; i++ ) {
    ourlens[i] += offlens[i];
  }

  if (!rank) {
    vals = (Scalar *) PetscMalloc( maxnz*sizeof(Scalar) ); CHKPTRQ(vals);

    /* read in my part of the matrix numerical values  */
    nz = procsnz[0];
    ierr = PetscBinaryRead(fd,vals,nz,BINARY_SCALAR); CHKERRQ(ierr);
    
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
    for ( i=1; i<size; i++ ) {
      nz = procsnz[i];
      ierr = PetscBinaryRead(fd,vals,nz,BINARY_SCALAR); CHKERRQ(ierr);
      MPI_Send(vals,nz,MPIU_SCALAR,i,A->tag,comm);
    }
    PetscFree(procsnz);
  }
  else {
    /* receive numeric values */
    vals = (Scalar*) PetscMalloc( nz*sizeof(Scalar) ); CHKPTRQ(vals);

    /* receive message of values*/
    MPI_Recv(vals,nz,MPIU_SCALAR,0,A->tag,comm,&status);
    MPI_Get_count(&status,MPIU_SCALAR,&maxnz);
    if (maxnz != nz) SETERRQ(1,"MatLoad_MPIAIJ:something is wrong with file");

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
  PetscFree(ourlens); PetscFree(vals); PetscFree(mycols); PetscFree(rowners);

  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
