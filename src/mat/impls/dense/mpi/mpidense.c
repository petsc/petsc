#ifndef lint
static char vcid[] = "$Id: mpidense.c,v 1.4 1995/10/22 22:17:20 curfman Exp curfman $";
#endif

#include "mpidense.h"
#include "vec/vecimpl.h"
#include "inline/spops.h"

static int MatSetValues_MPIDense(Mat mat,int m,int *idxm,int n,
                               int *idxn,Scalar *v,InsertMode addv)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr, i, j, rstart = mdn->rstart, rend = mdn->rend, row;

  if (mdn->insertmode != NOT_SET_VALUES && mdn->insertmode != addv) {
    SETERRQ(1,"MatSetValues_MPIDense:Cannot mix inserts and adds");
  }
  mdn->insertmode = addv;
  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) SETERRQ(1,"MatSetValues_MPIDense:Negative row");
    if (idxm[i] >= mdn->M) SETERRQ(1,"MatSetValues_MPIDense:Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (idxn[j] < 0) SETERRQ(1,"MatSetValues_MPIDense:Negative column");
        if (idxn[j] >= mdn->N) 
          SETERRQ(1,"MatSetValues_MPIDense:Column too large");
        ierr = MatSetValues(mdn->A,1,&row,1,&idxn[j],v+i*n+j,addv);
        CHKERRQ(ierr);
      }
    } 
    else {
      ierr = StashValues_Private(&mdn->stash,idxm[i],n,idxn,v+i*n,addv);
      CHKERRQ(ierr);
    }
  }
  return 0;
}

static int MatAssemblyBegin_MPIDense(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  MPI_Comm     comm = mat->comm;
  int          size = mdn->size, *owners = mdn->rowners, rank = mdn->rank;
  int          *nprocs,i,j,idx,*procs,nsends,nreceives,nmax,*work;
  int          tag = mat->tag, *owner,*starts,count,ierr;
  InsertMode   addv;
  MPI_Request  *send_waits,*recv_waits;
  Scalar       *rvalues,*svalues;

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  MPI_Allreduce((void *) &mdn->insertmode,(void *) &addv,1,MPI_INT,
                MPI_BOR,comm);
  if (addv == (ADD_VALUES|INSERT_VALUES)) { SETERRQ(1,
    "MatAssemblyBegin_MPIDense:Cannot mix adds/inserts on different procs");
    }
  mdn->insertmode = addv; /* in case this processor had no cache */

  /*  first count number of contributors to each processor */
  nprocs = (int *) PETSCMALLOC( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscZero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PETSCMALLOC( (mdn->stash.n+1)*sizeof(int) ); CHKPTRQ(owner);
  for ( i=0; i<mdn->stash.n; i++ ) {
    idx = mdn->stash.idx[i];
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; break;
      }
    }
  }
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PETSCMALLOC( size*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce((void *) procs,(void *) work,size,MPI_INT,MPI_SUM,comm);
  nreceives = work[rank]; 
  MPI_Allreduce((void *) nprocs,(void *) work,size,MPI_INT,MPI_MAX,comm);
  nmax = work[rank];
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
  svalues = (Scalar *) PETSCMALLOC( 3*(mdn->stash.n+1)*sizeof(Scalar) );
  CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PETSCMALLOC( (nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  starts = (int *) PETSCMALLOC( size*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<mdn->stash.n; i++ ) {
    svalues[3*starts[owner[i]]]       = (Scalar)  mdn->stash.idx[i];
    svalues[3*starts[owner[i]]+1]     = (Scalar)  mdn->stash.idy[i];
    svalues[3*(starts[owner[i]]++)+2] =  mdn->stash.array[i];
  }
  PETSCFREE(owner);
  starts[0] = 0;
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<size; i++ ) {
    if (procs[i]) {
      MPI_Isend((void*)(svalues+3*starts[i]),3*nprocs[i],MPIU_SCALAR,i,tag,
                comm,send_waits+count++);
    }
  }
  PETSCFREE(starts); PETSCFREE(nprocs);

  /* Free cache space */
  ierr = StashDestroy_Private(&mdn->stash); CHKERRQ(ierr);

  mdn->svalues    = svalues;    mdn->rvalues = rvalues;
  mdn->nsends     = nsends;     mdn->nrecvs = nreceives;
  mdn->send_waits = send_waits; mdn->recv_waits = recv_waits;
  mdn->rmax       = nmax;

  return 0;
}
extern int MatSetUpMultiply_MPIDense(Mat);

static int MatAssemblyEnd_MPIDense(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  MPI_Status   *send_status,recv_status;
  int          imdex, nrecvs=mdn->nrecvs, count=nrecvs, i, n, ierr, row, col;
  Scalar       *values,val;
  InsertMode   addv = mdn->insertmode;

  /*  wait on receives */
  while (count) {
    MPI_Waitany(nrecvs,mdn->recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    values = mdn->rvalues + 3*imdex*mdn->rmax;
    MPI_Get_count(&recv_status,MPIU_SCALAR,&n);
    n = n/3;
    for ( i=0; i<n; i++ ) {
      row = (int) PETSCREAL(values[3*i]) - mdn->rstart;
      col = (int) PETSCREAL(values[3*i+1]);
      val = values[3*i+2];
      if (col >= 0 && col < mdn->N) {
        MatSetValues(mdn->A,1,&row,1,&col,&val,addv);
      } 
      else {SETERRQ(1,"MatAssemblyEnd_MPIDense:Invalid column");}
    }
    count--;
  }
  PETSCFREE(mdn->recv_waits); PETSCFREE(mdn->rvalues);
 
  /* wait on sends */
  if (mdn->nsends) {
    send_status = (MPI_Status *) PETSCMALLOC( mdn->nsends*sizeof(MPI_Status) );
    CHKPTRQ(send_status);
    MPI_Waitall(mdn->nsends,mdn->send_waits,send_status);
    PETSCFREE(send_status);
  }
  PETSCFREE(mdn->send_waits); PETSCFREE(mdn->svalues);

  mdn->insertmode = NOT_SET_VALUES;
  ierr = MatAssemblyBegin(mdn->A,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mdn->A,mode); CHKERRQ(ierr);

  if (!mdn->assembled && mode == FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIDense(mat); CHKERRQ(ierr);
  }
  mdn->assembled = 1;
  return 0;
}

static int MatZeroEntries_MPIDense(Mat A)
{
  Mat_MPIDense *l = (Mat_MPIDense *) A->data;
  return MatZeroEntries(l->A);
}

/* the code does not do the diagonal entries correctly unless the 
   matrix is square and the column and row owerships are identical.
   This is a BUG. The only way to fix it seems to be to access 
   mdn->A and mdn->B directly and not through the MatZeroRows() 
   routine. 
*/
static int MatZeroRows_MPIDense(Mat A,IS is,Scalar *diag)
{
  Mat_MPIDense   *l = (Mat_MPIDense *) A->data;
  int            i,ierr,N, *rows,*owners = l->rowners,size = l->size;
  int            *procs,*nprocs,j,found,idx,nsends,*work;
  int            nmax,*svalues,*starts,*owner,nrecvs,rank = l->rank;
  int            *rvalues,tag = A->tag,count,base,slen,n,*source;
  int            *lens,imdex,*lrows,*values;
  MPI_Comm       comm = A->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
  IS             istmp;

  if (!l->assembled) SETERRQ(1,"MatZeroRows_MPIDense:Must assemble matrix");
  ierr = ISGetLocalSize(is,&N); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows); CHKERRQ(ierr);

  /*  first count number of contributors to each processor */
  nprocs = (int *) PETSCMALLOC( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscZero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PETSCMALLOC((N+1)*sizeof(int)); CHKPTRQ(owner); /* see note*/
  for ( i=0; i<N; i++ ) {
    idx = rows[i];
    found = 0;
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERRQ(1,"MatZeroRows_MPIDense:Index out of range");
  }
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PETSCMALLOC( size*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce( procs, work,size,MPI_INT,MPI_SUM,comm);
  nrecvs = work[rank]; 
  MPI_Allreduce( nprocs, work,size,MPI_INT,MPI_MAX,comm);
  nmax = work[rank];
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
  starts = (int *) PETSCMALLOC( (size+1)*sizeof(int) ); CHKPTRQ(starts);
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
  PETSCFREE(starts);

  base = owners[rank];

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

static int MatMult_MPIDense(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr;
  if (!mdn->assembled) 
    SETERRQ(1,"MatMult_MPIDense:Must assemble matrix first");
  ierr = VecScatterBegin(xx,mdn->lvec,INSERT_VALUES,SCATTER_ALL,mdn->Mvctx);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,mdn->lvec,INSERT_VALUES,SCATTER_ALL,mdn->Mvctx);
  CHKERRQ(ierr);
  ierr = MatMult(mdn->A,mdn->lvec,yy); CHKERRQ(ierr);
  return 0;
}

static int MatMultAdd_MPIDense(Mat mat,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr;
  if (!mdn->assembled) 
    SETERRQ(1,"MatMultAdd_MPIDense:Must assemble matrix first");
  ierr = VecScatterBegin(xx,mdn->lvec,INSERT_VALUES,SCATTER_ALL,mdn->Mvctx);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,mdn->lvec,INSERT_VALUES,SCATTER_ALL,mdn->Mvctx);
  CHKERRQ(ierr);
  ierr = MatMultAdd(mdn->A,mdn->lvec,yy,zz); CHKERRQ(ierr);
  return 0;
}

static int MatMultTrans_MPIDense(Mat A,Vec xx,Vec yy)
{
  Mat_MPIDense *a = (Mat_MPIDense *) A->data;
  int          ierr;
  Scalar       zero = 0.0;

  if (!a->assembled) SETERRQ(1,"MatMulTrans_MPIDense:must assemble matrix");
  ierr = VecSet(&zero,yy); CHKERRQ(ierr);
  ierr = MatMultTrans(a->A,xx,a->lvec); CHKERRQ(ierr);
  ierr = VecScatterBegin(a->lvec,yy,ADD_VALUES,
         (ScatterMode)(SCATTER_ALL|SCATTER_REVERSE),a->Mvctx); CHKERRQ(ierr);
  ierr = VecScatterEnd(a->lvec,yy,ADD_VALUES,
         (ScatterMode)(SCATTER_ALL|SCATTER_REVERSE),a->Mvctx); CHKERRQ(ierr);
  return 0;
}

static int MatMultTransAdd_MPIDense(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIDense *a = (Mat_MPIDense *) A->data;
  int          ierr;

  if (!a->assembled) SETERRQ(1,"MatMulTransAdd_MPIDense:must assemble matrix");
  ierr = VecCopy(yy,zz); CHKERRQ(ierr);
  ierr = MatMultTrans(a->A,xx,a->lvec); CHKERRQ(ierr);
  ierr = VecScatterBegin(a->lvec,zz,ADD_VALUES,
         (ScatterMode)(SCATTER_ALL|SCATTER_REVERSE),a->Mvctx); CHKERRQ(ierr);
  ierr = VecScatterEnd(a->lvec,zz,ADD_VALUES,
         (ScatterMode)(SCATTER_ALL|SCATTER_REVERSE),a->Mvctx); CHKERRQ(ierr);
  return 0;
}

static int MatGetDiagonal_MPIDense(Mat A,Vec v)
{
  Mat_MPIDense *a = (Mat_MPIDense *) A->data;
  Mat_SeqDense *aloc = (Mat_SeqDense *) a->A->data;
  int          ierr, i, n, m = a->m, radd;
  Scalar       *x;
  if (!a->assembled) SETERRQ(1,"MatGetDiag_MPIDense:must assemble matrix");
  ierr = VecGetArray(v,&x); CHKERRQ(ierr);
  ierr = VecGetSize(v,&n); CHKERRQ(ierr);
  if (n != a->M) SETERRQ(1,"MatGetDiagonal_SeqDense:Nonconforming mat and vec");
  radd = a->rstart*m*m;
  for ( i=0; i<m; i++ ) {
    x[i] = aloc->v[radd + i*m + i];
  }
  return 0;
}

static int MatDestroy_MPIDense(PetscObject obj)
{
  Mat          mat = (Mat) obj;
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr;
#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d",mdn->M,mdn->N);
#endif
  PETSCFREE(mdn->rowners); 
  ierr = MatDestroy(mdn->A); CHKERRQ(ierr);
  if (mdn->lvec)   VecDestroy(mdn->lvec);
  if (mdn->Mvctx)  VecScatterDestroy(mdn->Mvctx);
  PETSCFREE(mdn); 
  PLogObjectDestroy(mat);
  PETSCHEADERDESTROY(mat);
  return 0;
}

#include "pinclude/pviewer.h"

static int MatView_MPIDense_Binary(Mat mat,Viewer viewer)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr;
  if (mdn->size == 1) {
    ierr = MatView(mdn->A,viewer); CHKERRQ(ierr);
  }
  else SETERRQ(1,"MatView_MPIDense_Binary:Only uniprocessor output supported");
  return 0;
}

static int MatView_MPIDense_ASCII(Mat mat,Viewer viewer)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr, format;
  PetscObject  vobj = (PetscObject) viewer;
  FILE         *fd;

  ierr = ViewerFileGetPointer_Private(viewer,&fd); CHKERRQ(ierr);
  if (vobj->type == ASCII_FILE_VIEWER || vobj->type == ASCII_FILES_VIEWER) {
    ierr = ViewerFileGetFormat_Private(viewer,&format);
    if (format == FILE_FORMAT_INFO_DETAILED) {
      int nz, nzalloc, mem, rank;
      MPI_Comm_rank(mat->comm,&rank);
      ierr = MatGetInfo(mat,MAT_LOCAL,&nz,&nzalloc,&mem); 
      MPIU_Seq_begin(mat->comm,1);
        fprintf(fd,"  [%d] local rows %d nz %d nz alloced %d mem %d \n",
            rank,mdn->m,nz,nzalloc,mem);       
      fflush(fd);
      MPIU_Seq_end(mat->comm,1);
      ierr = VecScatterView(mdn->Mvctx,viewer); CHKERRQ(ierr);
      return 0; 
    }
    else if (format == FILE_FORMAT_INFO) {
      return 0;
    }
  }
  if (vobj->type == ASCII_FILE_VIEWER) {
    MPIU_Seq_begin(mat->comm,1);
    fprintf(fd,"[%d] rows %d starts %d ends %d cols %d\n",
             mdn->rank,mdn->m,mdn->rstart,mdn->rend,mdn->n);
    ierr = MatView(mdn->A,viewer); CHKERRQ(ierr);
    fflush(fd);
    MPIU_Seq_end(mat->comm,1);
  }
  else {
    int size = mdn->size, rank = mdn->rank; 
    if (size == 1) { 
      ierr = MatView(mdn->A,viewer); CHKERRQ(ierr);
    }
    else {
      /* assemble the entire matrix onto first processor. */
      Mat          A;
      int          M = mdn->M, N = mdn->N,m,row,i, nz, *cols;
      Scalar       *vals;
      Mat_SeqDense *Amdn = (Mat_SeqDense*) mdn->A->data;

      if (!rank) {
        ierr = MatCreateMPIDense(mat->comm,M,M,N,N,&A); CHKERRQ(ierr);
      }
      else {
        ierr = MatCreateMPIDense(mat->comm,0,M,N,N,&A); CHKERRQ(ierr);
      }
      PLogObjectParent(mat,A);

      /* Copy the matrix ... This isn't the most efficient means,
         but it's quick for now */
      row = mdn->rstart; m = Amdn->m;
      for ( i=0; i<m; i++ ) {
        ierr = MatGetRow(mat,row,&nz,&cols,&vals); CHKERRQ(ierr);
        ierr = MatSetValues(A,1,&row,nz,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
        ierr = MatRestoreRow(mat,row,&nz,&cols,&vals); CHKERRQ(ierr);
        row++;
      } 

      ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
      if (!rank) {
        ierr = MatView(((Mat_MPIDense*)(A->data))->A,viewer); CHKERRQ(ierr);
      }
      ierr = MatDestroy(A); CHKERRQ(ierr);
    }
  }
  return 0;
}

static int MatView_MPIDense(PetscObject obj,Viewer viewer)
{
  Mat          mat = (Mat) obj;
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  PetscObject  vobj = (PetscObject) viewer;
  int          ierr;
 
  if (!mdn->assembled) SETERRQ(1,"MatView_MPIDense:must assemble matrix");
  if (!viewer) { 
    viewer = STDOUT_VIEWER_SELF; vobj = (PetscObject) viewer;
  }
  if (vobj->cookie == VIEWER_COOKIE && vobj->type == ASCII_FILE_VIEWER) {
    ierr = MatView_MPIDense_ASCII(mat,viewer); CHKERRQ(ierr);
  }
  else if (vobj->cookie == VIEWER_COOKIE && vobj->type == ASCII_FILES_VIEWER) {
    ierr = MatView_MPIDense_ASCII(mat,viewer); CHKERRQ(ierr);
  }
  else if (vobj->type == BINARY_FILE_VIEWER) {
    return MatView_MPIDense_Binary(mat,viewer);
  }
  return 0;
}

static int MatGetInfo_MPIDense(Mat A,MatInfoType flag,int *nz,
                             int *nzalloc,int *mem)
{
  Mat_MPIDense *mat = (Mat_MPIDense *) A->data;
  Mat          mdn = mat->A;
  int          ierr, isend[3], irecv[3];

  ierr = MatGetInfo(mdn,MAT_LOCAL,&isend[0],&isend[1],&isend[2]); CHKERRQ(ierr);
  if (flag == MAT_LOCAL) {
    *nz = isend[0]; *nzalloc = isend[1]; *mem = isend[2];
  } else if (flag == MAT_GLOBAL_MAX) {
    MPI_Allreduce(isend,irecv,3,MPI_INT,MPI_MAX,A->comm);
    *nz = irecv[0]; *nzalloc = irecv[1]; *mem = irecv[2];
  } else if (flag == MAT_GLOBAL_SUM) {
    MPI_Allreduce(isend,irecv,3,MPI_INT,MPI_SUM,A->comm);
    *nz = irecv[0]; *nzalloc = irecv[1]; *mem = irecv[2];
  }
  return 0;
}

static int MatSetOption_MPIDense(Mat A,MatOption op)
{
  Mat_MPIDense *a = (Mat_MPIDense *) A->data;

  if (op == NO_NEW_NONZERO_LOCATIONS ||
      op == YES_NEW_NONZERO_LOCATIONS ||
      op == COLUMNS_SORTED ||
      op == ROW_ORIENTED) {
        MatSetOption(a->A,op);
  }
  else if (op == ROWS_SORTED || 
           op == SYMMETRIC_MATRIX ||
           op == STRUCTURALLY_SYMMETRIC_MATRIX ||
           op == YES_NEW_DIAGONALS)
    PLogInfo((PetscObject)A,"Info:MatSetOption_MPIDense:Option ignored\n");
  else if (op == COLUMN_ORIENTED)
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_MPIDense:COLUMN_ORIENTED");}
  else if (op == NO_NEW_DIAGONALS)
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_MPIDense:NO_NEW_DIAGONALS");}
  else 
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_MPIDense:unknown option");}
  return 0;
}

static int MatGetSize_MPIDense(Mat A,int *m,int *n)
{
  Mat_MPIDense *mat = (Mat_MPIDense *) A->data;
  *m = mat->M; *n = mat->N;
  return 0;
}

static int MatGetLocalSize_MPIDense(Mat A,int *m,int *n)
{
  Mat_MPIDense *mat = (Mat_MPIDense *) A->data;
  *m = mat->m; *n = mat->N;
  return 0;
}

static int MatGetOwnershipRange_MPIDense(Mat A,int *m,int *n)
{
  Mat_MPIDense *mat = (Mat_MPIDense *) A->data;
  *m = mat->rstart; *n = mat->rend;
  return 0;
}

static int MatGetRow_MPIDense(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIDense *mat = (Mat_MPIDense *) A->data;
  int          lrow, rstart = mat->rstart, rend = mat->rend;

  if (row < rstart || row >= rend) SETERRQ(1,"MatGetRow_MPIDense:only local rows")
  lrow = row - rstart;
  return MatGetRow(mat->A,lrow,nz,idx,v);
}

static int MatRestoreRow_MPIDense(Mat mat,int row,int *nz,int **idx,Scalar **v)
{
  if (idx) PETSCFREE(*idx);
  if (v) PETSCFREE(*v);
  return 0;
}

static int MatNorm_MPIDense(Mat A,MatNormType type,double *norm)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) A->data;
  Mat_SeqDense *mat = (Mat_SeqDense*) mdn->A->data;
  int          ierr, i, j;
  double       sum = 0.0;
  Scalar       *v = mat->v;

  if (!mdn->assembled) SETERRQ(1,"MatNorm_MPIDense:Must assemble matrix");
  if (mdn->size == 1) {
    ierr =  MatNorm(mdn->A,type,norm); CHKERRQ(ierr);
  } else {
    if (type == NORM_FROBENIUS) {
      for (i=0; i<mat->n*mat->m; i++ ) {
#if defined(PETSC_COMPLEX)
        sum += real(conj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      MPI_Allreduce((void*)&sum,(void*)norm,1,MPI_DOUBLE,MPI_SUM,A->comm);
      *norm = sqrt(*norm);
      PLogFlops(2*mat->n*mat->m);
    }
    else if (type == NORM_1) { 
      double *tmp, *tmp2;
      tmp  = (double *) PETSCMALLOC( 2*mdn->N*sizeof(double) ); CHKPTRQ(tmp);
      tmp2 = tmp + mdn->N;
      PetscZero(tmp,2*mdn->N*sizeof(double));
      *norm = 0.0;
      v = mat->v;
      for ( j=0; j<mat->n; j++ ) {
        for ( i=0; i<mat->m; i++ ) {
#if defined(PETSC_COMPLEX)
          tmp[j] += abs(*v++); 
#else
          tmp[j] += fabs(*v++); 
#endif
        }
      }
      MPI_Allreduce((void*)tmp,(void*)tmp2,mdn->N,MPI_DOUBLE,MPI_SUM,A->comm);
      for ( j=0; j<mdn->N; j++ ) {
        if (tmp2[j] > *norm) *norm = tmp2[j];
      }
      PETSCFREE(tmp);
      PLogFlops(mat->n*mat->m);
    }
    else if (type == NORM_INFINITY) { /* max row norm */
      double ntemp;
      ierr = MatNorm(mdn->A,type,&ntemp); CHKERRQ(ierr);
      MPI_Allreduce((void*)&ntemp,(void*)norm,1,MPI_DOUBLE,MPI_MAX,A->comm);
    }
    else {
      SETERRQ(1,"MatNorm_MPIDense:No support for two norm");
    }
  }
  return 0; 
}

static int MatTranspose_MPIDense(Mat A,Mat *matout)
{ 
  Mat_MPIDense *a = (Mat_MPIDense *) A->data;
  Mat_SeqDense *Aloc = (Mat_SeqDense *) a->A->data;
  Mat          B;
  int          M = a->M, N = a->N, m, n, *rwork, rstart = a->rstart;
  int          j, i, ierr;
  Scalar       *v;

  if (!matout && M != N)
    SETERRQ(1,"MatTranspose_MPIDense:Supports square matrix only in-place");
  ierr = MatCreateMPIDense(A->comm,PETSC_DECIDE,PETSC_DECIDE,N,M,&B); CHKERRQ(ierr);

  m = Aloc->m; n = Aloc->n; v = Aloc->v;
  rwork = (int *) PETSCMALLOC(n*sizeof(int)); CHKPTRQ(rwork);
  for ( j=0; j<n; j++ ) {
    for (i=0; i<m; i++) rwork[i] = rstart + i;
    ierr = MatSetValues(B,1,&j,m,rwork,v,INSERT_VALUES); CHKERRQ(ierr);
    v += m;
  } 
  PETSCFREE(rwork);
  ierr = MatAssemblyBegin(B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  if (matout) {
    *matout = B;
  } else {
    /* This isn't really an in-place transpose, but free data struct from a */
    PETSCFREE(a->rowners); 
    ierr = MatDestroy(a->A); CHKERRQ(ierr);
    if (a->lvec) VecDestroy(a->lvec);
    if (a->Mvctx) VecScatterDestroy(a->Mvctx);
    PETSCFREE(a); 
    PetscMemcpy(A,B,sizeof(struct _Mat)); 
    PETSCHEADERDESTROY(B);
  }
  return 0;
}

static int MatCopyPrivate_MPIDense(Mat,Mat *,int);

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {MatSetValues_MPIDense,
       MatGetRow_MPIDense,MatRestoreRow_MPIDense,
       MatMult_MPIDense,MatMultAdd_MPIDense,
       MatMultTrans_MPIDense,MatMultTransAdd_MPIDense,
       0,0,
       0,0,0,
       0,0,MatTranspose_MPIDense,
       MatGetInfo_MPIDense,0,
       MatGetDiagonal_MPIDense,0,MatNorm_MPIDense,
       MatAssemblyBegin_MPIDense,MatAssemblyEnd_MPIDense,
       0,
       MatSetOption_MPIDense,MatZeroEntries_MPIDense,MatZeroRows_MPIDense,
       0,
       0,0,0,0,
       MatGetSize_MPIDense,MatGetLocalSize_MPIDense,
       MatGetOwnershipRange_MPIDense,
       0,0,
       0,0,0,0,0,MatCopyPrivate_MPIDense};

/*@C
   MatCreateMPIDense - Creates a sparse parallel matrix in dense format.

   Input Parameters:
.  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
.  n - number of local columns (or PETSC_DECIDE to have calculated 
           if N is given)
.  M - number of global rows (or PETSC_DECIDE to have calculated if m is given)
.  N - number of global columns (or PETSC_DECIDE to have calculated 
           if n is given)

   Output Parameter:
.  newmat - the matrix 

   Notes:
   The dense format is fully compatible with standard Fortran 77
   storage by columns.

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   Currently, the only parallel dense matrix decomposition is by rows,
   so that n=N and each submatrix owns all of the global columns.

.keywords: matrix, dense, parallel

.seealso: MatCreate(), MatCreateSeqDense(), MatSetValues()
@*/
int MatCreateMPIDense(MPI_Comm comm,int m,int n,int M,int N,Mat *newmat)
{
  Mat          mat;
  Mat_MPIDense *a;
  int          ierr, i;

  *newmat         = 0;
  PETSCHEADERCREATE(mat,_Mat,MAT_COOKIE,MATMPIDENSE,comm);
  PLogObjectCreate(mat);
  mat->data       = (void *) (a = PETSCNEW(Mat_MPIDense)); CHKPTRQ(a);
  PetscMemcpy(&mat->ops,&MatOps,sizeof(struct _MatOps));
  mat->destroy    = MatDestroy_MPIDense;
  mat->view       = MatView_MPIDense;
  mat->factor     = 0;

  a->insertmode = NOT_SET_VALUES;
  MPI_Comm_rank(comm,&a->rank);
  MPI_Comm_size(comm,&a->size);

  if (M == PETSC_DECIDE) MPI_Allreduce(&m,&M,1,MPI_INT,MPI_SUM,comm);
  if (m == PETSC_DECIDE) {m = M/a->size + ((M % a->size) > a->rank);}

  /* each row stores all columns */
  if (N == PETSC_DECIDE) N = n;
  if (n == PETSC_DECIDE) n = N;
  if (n != N) SETERRQ(1,"MatCreateMPIDense:For now, only n=N is supported");
  a->N = N;
  a->M = M;
  a->m = m;
  a->n = n;

  /* build local table of row and column ownerships */
  a->rowners = (int *) PETSCMALLOC((a->size+2)*sizeof(int)); CHKPTRQ(a->rowners);
  PLogObjectMemory(mat,(a->size+2)*sizeof(int)+sizeof(struct _Mat)+ 
                       sizeof(Mat_MPIDense));
  MPI_Allgather(&m,1,MPI_INT,a->rowners+1,1,MPI_INT,comm);
  a->rowners[0] = 0;
  for ( i=2; i<=a->size; i++ ) {
    a->rowners[i] += a->rowners[i-1];
  }
  a->rstart = a->rowners[a->rank]; 
  a->rend   = a->rowners[a->rank+1]; 

  ierr = MatCreateSeqDense(MPI_COMM_SELF,m,N,&a->A); CHKERRQ(ierr);
  PLogObjectParent(mat,a->A);

  /* build cache for off array entries formed */
  ierr = StashBuild_Private(&a->stash); CHKERRQ(ierr);

  /* stuff used for matrix vector multiply */
  a->lvec      = 0;
  a->Mvctx     = 0;
  a->assembled = 0;

  *newmat = mat;
  return 0;
}

static int MatCopyPrivate_MPIDense(Mat A,Mat *newmat,int cpvalues)
{
  Mat          mat;
  Mat_MPIDense *a,*oldmat = (Mat_MPIDense *) A->data;
  int          ierr;

  if (!oldmat->assembled) SETERRQ(1,"MatCopyPrivate_MPIDense:Must assemble matrix");
  *newmat       = 0;
  PETSCHEADERCREATE(mat,_Mat,MAT_COOKIE,MATMPIDENSE,A->comm);
  PLogObjectCreate(mat);
  mat->data     = (void *) (a = PETSCNEW(Mat_MPIDense)); CHKPTRQ(a);
  PetscMemcpy(&mat->ops,&MatOps,sizeof(struct _MatOps));
  mat->destroy  = MatDestroy_MPIDense;
  mat->view     = MatView_MPIDense;
  mat->factor   = A->factor;

  a->m          = oldmat->m;
  a->n          = oldmat->n;
  a->M          = oldmat->M;
  a->N          = oldmat->N;

  a->assembled  = 1;
  a->rstart     = oldmat->rstart;
  a->rend       = oldmat->rend;
  a->size       = oldmat->size;
  a->rank       = oldmat->rank;
  a->insertmode = NOT_SET_VALUES;

  a->rowners = (int *) PETSCMALLOC((a->size+1)*sizeof(int)); CHKPTRQ(a->rowners);
  PLogObjectMemory(mat,(a->size+1)*sizeof(int)+sizeof(struct _Mat)+sizeof(Mat_MPIDense));
  PetscMemcpy(a->rowners,oldmat->rowners,(a->size+1)*sizeof(int));
  ierr = StashInitialize_Private(&a->stash); CHKERRQ(ierr);
  
  ierr =  VecDuplicate(oldmat->lvec,&a->lvec); CHKERRQ(ierr);
  PLogObjectParent(mat,a->lvec);
  ierr =  VecScatterCopy(oldmat->Mvctx,&a->Mvctx); CHKERRQ(ierr);
  PLogObjectParent(mat,a->Mvctx);
  ierr =  MatConvert(oldmat->A,MATSAME,&a->A); CHKERRQ(ierr);
  PLogObjectParent(mat,a->A);
  *newmat = mat;
  return 0;
}

#include "sysio.h"

int MatLoad_MPIDense(Viewer bview,MatType type,Mat *newmat)
{
  Mat          A;
  int          i, nz, ierr, j,rstart, rend, fd;
  Scalar       *vals,*svals;
  PetscObject  vobj = (PetscObject) bview;
  MPI_Comm     comm = vobj->comm;
  MPI_Status   status;
  int          header[4],rank,size,*rowlengths = 0,M,N,m,*rowners,maxnz,*cols;
  int          *ourlens,*sndcounts = 0,*procsnz = 0, *offlens,jj,*mycols,*smycols;
  int          tag = ((PetscObject)bview)->tag;

  MPI_Comm_size(comm,&size); MPI_Comm_rank(comm,&rank);
  if (!rank) {
    ierr = ViewerFileGetDescriptor_Private(bview,&fd); CHKERRQ(ierr);
    ierr = SYRead(fd,(char *)header,4,SYINT); CHKERRQ(ierr);
    if (header[0] != MAT_COOKIE) SETERRQ(1,"MatLoad_MPIDenseorMPIRow:not matrix object");
  }

  MPI_Bcast(header+1,3,MPI_INT,0,comm);
  M = header[1]; N = header[2];
  /* determine ownership of all rows */
  m = M/size + ((M % size) > rank);
  rowners = (int *) PETSCMALLOC((size+2)*sizeof(int)); CHKPTRQ(rowners);
  MPI_Allgather(&m,1,MPI_INT,rowners+1,1,MPI_INT,comm);
  rowners[0] = 0;
  for ( i=2; i<=size; i++ ) {
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 

  /* distribute row lengths to all processors */
  ourlens = (int*) PETSCMALLOC( 2*(rend-rstart)*sizeof(int) ); CHKPTRQ(ourlens);
  offlens = ourlens + (rend-rstart);
  if (!rank) {
    rowlengths = (int*) PETSCMALLOC( M*sizeof(int) ); CHKPTRQ(rowlengths);
    ierr = SYRead(fd,rowlengths,M,SYINT); CHKERRQ(ierr);
    sndcounts = (int*) PETSCMALLOC( size*sizeof(int) ); CHKPTRQ(sndcounts);
    for ( i=0; i<size; i++ ) sndcounts[i] = rowners[i+1] - rowners[i];
    MPI_Scatterv(rowlengths,sndcounts,rowners,MPI_INT,ourlens,rend-rstart,MPI_INT,0,comm);
    PETSCFREE(sndcounts);
  }
  else {
    MPI_Scatterv(0,0,0,MPI_INT,ourlens,rend-rstart,MPI_INT, 0,comm);
  }

  if (!rank) {
    /* calculate the number of nonzeros on each processor */
    procsnz = (int*) PETSCMALLOC( size*sizeof(int) ); CHKPTRQ(procsnz);
    PetscZero(procsnz,size*sizeof(int));
    for ( i=0; i<size; i++ ) {
      for ( j=rowners[i]; j< rowners[i+1]; j++ ) {
        procsnz[i] += rowlengths[j];
      }
    }
    PETSCFREE(rowlengths);

    /* determine max buffer needed and allocate it */
    maxnz = 0;
    for ( i=0; i<size; i++ ) {
      maxnz = PETSCMAX(maxnz,procsnz[i]);
    }
    cols = (int *) PETSCMALLOC( maxnz*sizeof(int) ); CHKPTRQ(cols);

    /* read in my part of the matrix column indices  */
    nz = procsnz[0];
    mycols = (int *) PETSCMALLOC( nz*sizeof(int) ); CHKPTRQ(mycols);
    ierr = SYRead(fd,mycols,nz,SYINT); CHKERRQ(ierr);

    /* read in every one elses and ship off */
    for ( i=1; i<size; i++ ) {
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
    if (maxnz != nz) SETERRQ(1,"MatLoad_MPIDenseorMPIRow:something is wrong with file");
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
  if (type == MATMPIDENSE) {
    ierr = MatCreateMPIDense(comm,m,PETSC_DECIDE,M,N,newmat);CHKERRQ(ierr);
  }
  A = *newmat;
  MatSetOption(A,COLUMNS_SORTED); 
  for ( i=0; i<m; i++ ) {
    ourlens[i] += offlens[i];
  }

  if (!rank) {
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
    for ( i=1; i<size; i++ ) {
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
    if (maxnz != nz) SETERRQ(1,"MatLoad_MPIDenseorMPIRow:something is wrong with file");

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
