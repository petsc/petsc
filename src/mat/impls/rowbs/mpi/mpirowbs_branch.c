#ifndef lint
static char vcid[] = "$Id: mpirowbs.c,v 1.42 1995/07/10 04:51:54 bsmith Exp curfman $";
#endif

#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
#include "mpirowbs.h"
#include "vec/vecimpl.h"
#include "inline/spops.h"
#include "BSprivate.h"

#define CHUNCKSIZE_LOCAL   10

/* Same as MatRow format ... should share these! */
static int MatFreeRowbs_Private(Mat matin,int n,int *i,Scalar *v)
{
  if (v) PETSCFREE(v);
  return 0;
}

/* Note:  Call MatMallocRowbs_Private only for n>0 */
static int MatMallocRowbs_Private(Mat matin,int n,int **i,Scalar **v)
{
  int len;
  if (n == 0) {
    *i = 0; *v = 0;
  } else {
    len = n*(sizeof(int) + sizeof(Scalar));
    *v = (Scalar *) PETSCMALLOC( len ); CHKPTRQ(*v);
    *i = (int *)(*v + n);
  }
  return 0;
}

/* ----------------------------------------------------------------- */
static int MatCreateMPIRowbs_local(Mat mat,int nz,int *nnz)
{
  Mat_MPIRowbs *bsif = (Mat_MPIRowbs *) mat->data;
  int          ierr, i, len, nzalloc = 0, m = bsif->m;
  BSspmat      *bsmat;
  BSsprow      *vs;

  if (!nnz) {
    if (nz <= 0) nz = 1;
    nzalloc = 1;
    nnz = (int *) PETSCMALLOC( (m+1)*sizeof(int) ); CHKPTRQ(nnz);
    for ( i=0; i<m; i++ ) nnz[i] = nz;
    nz = nz*m;
  }
  else {
    nz = 0;
    for ( i=0; i<m; i++ ) nz += nnz[i];
  }

  /* Allocate BlockSolve matrix context */
  bsif->A = bsmat = PETSCNEW(BSspmat); CHKPTRQ(bsmat);
  len = m*(sizeof(BSsprow *) + sizeof(BSsprow)) + 1;
  bsmat->rows = (BSsprow **) PETSCMALLOC( len ); CHKPTRQ(bsmat->rows);
  bsmat->num_rows = m;
  bsmat->global_num_rows = bsif->M;
  bsmat->map = bsif->bsmap;
  vs = (BSsprow *) (bsmat->rows + m);
  for (i=0; i<m; i++) {
    bsmat->rows[i] = vs;
    bsif->imax[i]   = nnz[i];
    vs->length	    = 0;
    vs->diag_ind    = -1;
    if (nnz[i] > 0) {
      ierr = MatMallocRowbs_Private(mat,nnz[i],&(vs->col),&(vs->nz)); 
      CHKERRQ(ierr);
    } else {
      vs->col = 0; vs->nz = 0;
    }
    vs++;
  }
  bsif->mem = sizeof(BSspmat) + len + nz*(sizeof(int) + sizeof(Scalar));
  bsif->nz	     = 0;
  bsif->maxnz	     = nz;
  bsif->sorted       = 0;
  bsif->roworiented  = 1;
  bsif->nonew        = 0;
  bsif->singlemalloc = 0;

  if (nzalloc) PETSCFREE(nnz);
  return 0;
}

static int MatSetValues_MPIRowbs_local(Mat matin,int m,int *idxm,int n,
                            int *idxn,Scalar *v,InsertMode addv)
{
  Mat_MPIRowbs *mat = (Mat_MPIRowbs *) matin->data;
  BSspmat      *A = mat->A;
  BSsprow      *vs;
  int          *rp,k,a,b,t,ii,row,nrow,i,col,l,rmax, ierr;
  int          *imax = mat->imax, nonew = mat->nonew, sorted = mat->sorted;
  Scalar       *ap, value;

  for ( k=0; k<m; k++ ) { /* loop over added rows */
    row = idxm[k];
    if (row < 0) SETERRQ(1,"Negative row index");
    if (row >= mat->m) SETERRQ(1,"Row index too large");
    vs = A->rows[row];
    ap = vs->nz; rp = vs->col;
    rmax = imax[row]; nrow = vs->length;
    a = 0;
    for ( l=0; l<n; l++ ) { /* loop over added columns */
      if (idxn[l] < 0) SETERRQ(1,"Negative column index");
      if (idxn[l] >= mat->N) SETERRQ(1,"Column index too large");
      col = idxn[l]; value = *v++;
      if (!sorted) a = 0; b = nrow;
      while (b-a > 5) {
        t = (b+a)/2;
        if (rp[t] > col) b = t;
        else             a = t;
      }
      for ( i=a; i<b; i++ ) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          if (addv == ADDVALUES) ap[i] += value;
          else                   ap[i] = value;
          goto noinsert;
        }
      }
      if (nonew) goto noinsert;
      if (nrow >= rmax) {
        /* there is no extra room in row, therefore enlarge */
        int      *itemp;
        register int *iout, *iin = vs->col;
        register Scalar *vout, *vin = vs->nz;
        Scalar   *vtemp;

        /* malloc new storage space */

        imax[row] += CHUNCKSIZE_LOCAL;
        ierr = MatMallocRowbs_Private(matin,imax[row],&itemp,&vtemp); 
        CHKERRQ(ierr);
        vout = vtemp; iout = itemp;
        for (ii=0; ii<i; ii++) {
          vout[ii] = vin[ii];
          iout[ii] = iin[ii];
        }
        vout[i] = value;
        iout[i] = col;
        for (ii=i+1; ii<=nrow; ii++) {
          vout[ii] = vin[ii-1];
          iout[ii] = iin[ii-1];
        }
       /* free old row storage */
        if (rmax > 0)
          {ierr = MatFreeRowbs_Private(matin,rmax,vs->col,vs->nz); 
           CHKERRQ(ierr);}
        vs->col = iout; vs->nz = vout;
        rmax = imax[row];
        mat->singlemalloc = 0;
        mat->maxnz += CHUNCKSIZE_LOCAL;
        mat->mem   += CHUNCKSIZE_LOCAL*(sizeof(int) + sizeof(Scalar));
      }
      else {
      /* this has too many shifts here; but alternative was slower*/
        for ( ii=nrow-1; ii>=i; ii-- ) {/* shift values up */
          rp[ii+1] = rp[ii];
          ap[ii+1] = ap[ii];
        }
        rp[i] = col;
        ap[i] = value;
      }
      nrow++;
      mat->nz++;
      noinsert:;
      a = i + 1;
    }
    vs->length = nrow;
  }
  return 0;
}

#include "draw.h"
#include "pviewer.h"

static int MatView_MPIRowbs_local(Mat mat,Viewer ptr)
{
  Mat_MPIRowbs *mrow = (Mat_MPIRowbs *) mat->data;
  BSspmat      *A = mrow->A;
  BSsprow      **rs = A->rows;
  int     i, j;

  for ( i=0; i<A->num_rows; i++ ) {
    printf("row %d:",i);
    for (j=0; j<rs[i]->length; j++) {
      printf(" %d %g ", rs[i]->col[j], rs[i]->nz[j]);
    }
    printf("\n");
  }
  return 0;
}

static int MatAssemblyBegin_MPIRowbs_local(Mat mat,MatAssemblyType mode)
{ 
  return 0;
}

/* Note: The local end assembly routine must be called through
   the parallel version only! */
static int MatAssemblyEnd_MPIRowbs_local(Mat mat,MatAssemblyType mode)
{
  Mat_MPIRowbs *mrow = (Mat_MPIRowbs *) mat->data;
  BSspmat      *A = mrow->A;
  BSsprow      *vs;
  int          i, j, rstart = mrow->rstart;

  if (mode == FLUSH_ASSEMBLY) return 0;

  /* No shifting needed; this is done during MatSetValues */
  /* Mark location of diagonal */
  for ( i=0; i<mrow->m; i++ ) {
    vs = A->rows[i];
    for ( j=0; j<vs->length; j++ ) {
      if (vs->col[j] == i + rstart) {
        vs->diag_ind = j;
        break;
      }
    }
    if (vs->diag_ind == -1) 
       SETERRQ(1,"No diagonal term!  Must set diagonal entry, even if zero.");
  }
  return 0;
}

static int MatZeroRows_MPIRowbs_local(Mat mat,IS is,Scalar *diag)
{
  Mat_MPIRowbs *mrow = (Mat_MPIRowbs *) mat->data;
  BSspmat      *l = mrow->A;
  int          i, ierr, N, *rz, m = mrow->m - 1;

  ierr = ISGetLocalSize(is,&N); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rz); CHKERRQ(ierr);
  if (diag) {
    for ( i=0; i<N; i++ ) {
      if (rz[i] < 0 || rz[i] > m) SETERRQ(1,"Index out of range.");
      if (l->rows[rz[i]]->length > 0) { /* in case row was completely empty */
        l->rows[rz[i]]->length = 1;
        l->rows[rz[i]]->nz[0] = *diag;
        l->rows[rz[i]]->col[0] = rz[i];
      }
      else {
        ierr = MatSetValues(mat,1,&rz[i],1,&rz[i],diag,INSERTVALUES);
        CHKERRQ(ierr);
      }
    }
  }
  else {
    for ( i=0; i<N; i++ ) {
      if (rz[i] < 0 || rz[i] > m) SETERRQ(1,"Index out of range.");
      l->rows[rz[i]]->length = 0;
    }
  }
  ISRestoreIndices(is,&rz);
  ierr = MatAssemblyBegin(mat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}

/* ----------------------------------------------------------------- */

static int MatSetValues_MPIRowbs(Mat mat,int m,int *idxm,int n,
                            int *idxn,Scalar *v,InsertMode addv)
{
  Mat_MPIRowbs *mrow = (Mat_MPIRowbs *) mat->data;
  int          ierr, i, j, row, col, rstart = mrow->rstart, rend = mrow->rend;

  if (mrow->insertmode != NOTSETVALUES && mrow->insertmode != addv) {
    SETERRQ(1,"You cannot mix inserts and adds");
  }
  mrow->insertmode = addv;
  if ((mrow->assembled) && (!mrow->reassemble_begun)) {
    /* Symmetrically unscale the matrix by the diagonal */
    BSscale_diag(mrow->pA,mrow->inv_diag,mrow->procinfo); CHKERRBS(0);
    mrow->reassemble_begun = 1;
  }
  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) SETERRQ(1,"Negative row index");
    if (idxm[i] >= mrow->M) SETERRQ(1,"Row index too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (idxn[j] < 0) SETERRQ(1,"Negative column index");
        if (idxn[j] >= mrow->N) SETERRQ(1,"Column index too large");
        if (idxn[j] >= 0 && idxn[j] < mrow->N){
          col = idxn[j];
          ierr = MatSetValues_MPIRowbs_local(mat,1,&row,1,&col,
                                             v+i*n+j,addv);CHKERRQ(ierr);
        }
        else {SETERRQ(1,"Invalid column index.");}
      }
    } 
    else {
      ierr = StashValues_Private(&mrow->stash,idxm[i],n,idxn,v+i*n,addv);
      CHKERRQ(ierr);
    }
  }
  return 0;
}

/*
    the assembly code is alot like the code for vectors, we should 
    sometime derive a single assembly code that can be used for 
    either case.
*/

static int MatAssemblyBegin_MPIRowbs(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIRowbs  *mrow = (Mat_MPIRowbs *) mat->data;
  MPI_Comm    comm = mat->comm;
  int         numtids = mrow->numtids, *owners = mrow->rowners;
  int         mytid = mrow->mytid;
  MPI_Request *send_waits,*recv_waits;
  int         *nprocs,i,j,idx,*procs,nsends,nreceives,nmax,*work;
  int         tag = mat->tag, *owner,*starts,count,ierr;
  InsertMode  addv;
  Scalar      *rvalues,*svalues;

  if ((mrow->assembled) && (!mrow->reassemble_begun)) {
    /* Symmetrically unscale the matrix by the diagonal */
    BSscale_diag(mrow->pA,mrow->inv_diag,mrow->procinfo); CHKERRBS(0);
    mrow->reassemble_begun = 1;
  }

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  MPI_Allreduce((void *) &mrow->insertmode,(void *) &addv,1,MPI_INT,
                MPI_BOR,comm);
  if (addv == (ADDVALUES|INSERTVALUES)) {
    SETERRQ(1,"Some processors have inserted while others have added");
  }
  mrow->insertmode = addv; /* in case this processor had no cache */

  /*  first count number of contributors to each processor */
  nprocs = (int *) PETSCMALLOC( 2*numtids*sizeof(int) ); CHKPTRQ(nprocs);
  PETSCMEMSET(nprocs,0,2*numtids*sizeof(int)); procs = nprocs + numtids;
  owner = (int *) PETSCMALLOC( (mrow->stash.n+1)*sizeof(int) ); CHKPTRQ(owner);
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
  svalues = (Scalar *) PETSCMALLOC( 3*(mrow->stash.n+1)*sizeof(Scalar) );
  CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PETSCMALLOC( (nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  starts = (int *) PETSCMALLOC( numtids*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<numtids; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<mrow->stash.n; i++ ) {
    svalues[3*starts[owner[i]]]       = (Scalar)  mrow->stash.idx[i];
    svalues[3*starts[owner[i]]+1]     = (Scalar)  mrow->stash.idy[i];
    svalues[3*(starts[owner[i]]++)+2] =  mrow->stash.array[i];
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
  ierr = StashDestroy_Private(&mrow->stash); CHKERRQ(ierr);

  mrow->svalues    = svalues;    mrow->rvalues = rvalues;
  mrow->nsends     = nsends;     mrow->nrecvs = nreceives;
  mrow->send_waits = send_waits; mrow->recv_waits = recv_waits;
  mrow->rmax       = nmax;

  return 0;
}

static int MatView_MPIRowbs(PetscObject obj,Viewer viewer)
{
  Mat          mat = (Mat) obj;
  Mat_MPIRowbs *mrow = (Mat_MPIRowbs *) mat->data;
  int          ierr;
  PetscObject  vobj = (PetscObject) viewer;

  if (!mrow->assembled)
    SETERRQ(1,"MatView_MPIRow: Must assemble matrix first.");
  if (!viewer) { /* so that viewers may be used from debuggers */
    viewer = STDOUT_VIEWER; vobj = (PetscObject) viewer;
  }
  if (vobj->cookie == DRAW_COOKIE && vobj->type == NULLWINDOW) return 0;
  if ((vobj->cookie == DRAW_COOKIE) || (vobj->cookie == VIEWER_COOKIE && 
     (vobj->type == FILE_VIEWER || vobj->type == FILES_VIEWER))) {
    FILE *fd = ViewerFileGetPointer_Private(viewer);
    MPIU_Seq_begin(mat->comm,1);
    fprintf(fd,"[%d] rows %d starts %d ends %d cols %d starts %d ends %d\n",
           mrow->mytid,mrow->m,mrow->rstart,mrow->rend,mrow->n,0,mrow->N);
    ierr = MatView_MPIRowbs_local(mat,viewer); CHKERRQ(ierr);
    fflush(fd);
    MPIU_Seq_end(mat->comm,1);
  }
  /* Note:  Since BlockSolve does not support matrices of dimension 0,
     we can't do the usual parallel viewer stuff */
  return 0;
}

static int MatAssemblyEnd_MPIRowbs(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIRowbs *mrow = (Mat_MPIRowbs *) mat->data;
  MPI_Status   *send_status,recv_status;
  int          imdex,nrecvs = mrow->nrecvs, count = nrecvs, i, n;
  int          ldim, low, high, loc, row, col, ierr;
  Scalar       *values, val;
  InsertMode   addv = mrow->insertmode;

  /*  wait on receives */
  while (count) {
    MPI_Waitany(nrecvs,mrow->recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    values = mrow->rvalues + 3*imdex*mrow->rmax;
    MPI_Get_count(&recv_status,MPIU_SCALAR,&n);
    n = n/3;
    for ( i=0; i<n; i++ ) {
      row = (int) PETSCREAL(values[3*i]) - mrow->rstart;
      col = (int) PETSCREAL(values[3*i+1]);
      val = values[3*i+2];
      if (col >= 0 && col < mrow->N) {
        MatSetValues_MPIRowbs_local(mat,1,&row,1,&col,&val,addv);
      } 
      else {SETERRQ(1,"Invalid column index.");}
    }
    count--;
  }
  PETSCFREE(mrow->recv_waits); PETSCFREE(mrow->rvalues);
 
  /* wait on sends */
  if (mrow->nsends) {
    send_status = (MPI_Status *) PETSCMALLOC( mrow->nsends*sizeof(MPI_Status) );
    CHKPTRQ(send_status);
    MPI_Waitall(mrow->nsends,mrow->send_waits,send_status);
    PETSCFREE(send_status);
  }
  PETSCFREE(mrow->send_waits); PETSCFREE(mrow->svalues);

  mrow->insertmode = NOTSETVALUES;
  ierr = MatAssemblyBegin_MPIRowbs_local(mat,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd_MPIRowbs_local(mat,mode); CHKERRQ(ierr);

  if (mode == FINAL_ASSEMBLY) {   /* BlockSolve stuff */
    if ((mrow->assembled) && (!mrow->nonew)) {  /* Free the old info */
      if (mrow->pA)       {BSfree_par_mat(mrow->pA); CHKERRBS(0);}
      if (mrow->comm_pA)  {BSfree_comm(mrow->comm_pA); CHKERRBS(0);}
    }
    if ((!mrow->nonew) || (!mrow->assembled)) {
      /* Form permuted matrix for efficient parallel execution */
      mrow->pA = BSmain_perm(mrow->procinfo,mrow->A); CHKERRBS(0);

      /* Set up the communication */
      mrow->comm_pA = BSsetup_forward(mrow->pA,mrow->procinfo); CHKERRBS(0);
    } else {
      /* Repermute the matrix */
      BSmain_reperm(mrow->procinfo,mrow->A,mrow->pA); CHKERRBS(0);
    }

    /* Symmetrically scale the matrix by the diagonal */
    BSscale_diag(mrow->pA,mrow->pA->diag,mrow->procinfo); CHKERRBS(0);

    /* Store inverse of square root of permuted diagonal scaling matrix */
    ierr = VecGetLocalSize( mrow->diag, &ldim ); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange( mrow->diag, &low, &high ); CHKERRQ(ierr);
    for (i=0; i<ldim; i++) {
      loc = low + i;
      val = 1.0/sqrt(mrow->pA->scale_diag[i]);
      mrow->inv_diag[i] = 1.0/(mrow->pA->scale_diag[i]);
      ierr = VecSetValues(mrow->diag,1,&loc,&val,INSERTVALUES); CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin( mrow->diag ); CHKERRQ(ierr);
    ierr = VecAssemblyEnd( mrow->diag ); CHKERRQ(ierr);
    mrow->assembled = 1;
    mrow->reassemble_begun = 0;
  }
  return 0;
}

static int MatZeroEntries_MPIRowbs(Mat mat)
{
  Mat_MPIRowbs *l = (Mat_MPIRowbs *) mat->data;
  BSspmat      *A = l->A;
  BSsprow      *vs;
  int          i, j;

  for (i=0; i < l->m; i++) {
    vs = A->rows[i];
    for (j=0; j< vs->length; j++) vs->nz[j] = 0.0;
  }
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

static int MatZeroRows_MPIRowbs(Mat A,IS is,Scalar *diag)
{
  Mat_MPIRowbs   *l = (Mat_MPIRowbs *) A->data;
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
    SETERRQ(1,"MatZeroRows_MPIRowbs: Must assemble matrix first");
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
  CHKERRQ(ierr);  PETSCFREE(lrows);
  ierr = MatZeroRows_MPIRowbs_local(A,istmp,diag); CHKERRQ(ierr);
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

static int MatMult_MPIRowbs(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIRowbs *bsif = (Mat_MPIRowbs *) mat->data;
  BSprocinfo   *bspinfo = bsif->procinfo;
  Scalar       *xxa, *xworka, *yya;
  int          ierr;

  if (!bsif->assembled) 
    SETERRQ(1,"MatMult_MPIRowbs: Must assemble matrix first.");
  ierr = VecGetArray(yy,&yya); CHKERRQ(ierr);
  ierr = VecGetArray(xx,&xxa); CHKERRQ(ierr);

  /* Permute and apply diagonal scaling:  [ xwork = D^{1/2} * x ] */
  if (!bsif->vecs_permscale) {
    ierr = VecGetArray(bsif->xwork,&xworka); CHKERRQ(ierr);
    BSperm_dvec(xxa,xworka,bsif->pA->perm); CHKERRBS(0);
    ierr = VecPDiv(bsif->xwork,bsif->diag,xx); CHKERRQ(ierr);
  } 

  /* Do lower triangular multiplication:  [ y = L * xwork ] */
#if defined(PETSC_DEBUG)
  MLOG_ELM(bspinfo->procset);
#endif
  if (bspinfo->single)
      BSforward1( bsif->pA, xxa, yya, bsif->comm_pA, bspinfo );
  else
      BSforward( bsif->pA, xxa, yya, bsif->comm_pA, bspinfo );
  CHKERRBS(0);
#if defined(PETSC_DEBUG)
  MLOG_ACC(MM_FORWARD);
  MLOG_ELM(bspinfo->procset);
#endif

  /* Do upper triangular multiplication:  [ y = y + L^{T} * xwork ] */
  if (bspinfo->single)
      BSbackward1( bsif->pA, xxa, yya, bsif->comm_pA, bspinfo );
  else
      BSbackward( bsif->pA, xxa, yya, bsif->comm_pA, bspinfo );
  CHKERRBS(0);
#if defined(PETSC_DEBUG)
  MLOG_ACC(MM_BACKWARD);
#endif

  /* Apply diagonal scaling to vector:  [  y = D^{1/2} * y ] */
  if (!bsif->vecs_permscale) {
    BSiperm_dvec(xworka,xxa,bsif->pA->perm); CHKERRBS(0);
    ierr = VecPDiv(yy,bsif->diag,bsif->xwork); CHKERRQ(ierr);
    BSiperm_dvec(xworka,yya,bsif->pA->perm); CHKERRBS(0);
  }
  return 0;
}

static int MatRelax_MPIRowbs(Mat mat,Vec bb,double omega,MatSORType flag,
                             double shift,int its,Vec xx)
{
  Mat_MPIRowbs *bsif = (Mat_MPIRowbs *) mat->data;
  Scalar *b;
  int ierr;

/* None of the relaxation code is finished now! */
  SETERRQ(1,"Not done yet");

  if (!bsif->assembled) 
    SETERRQ(1,"MatRelax_MPIRowbs: Must assemble matrix first");
  VecGetArray(bb,&b);
  if (flag & SOR_ZERO_INITIAL_GUESS) {
      /* do nothing */ 
  } else SETERRQ(1,"Not done yet.");
  if (shift) SETERRQ(1,"shift != 0 : Not yet done.");
  if (omega != 1.0) SETERRQ(1,"omega != 1.0 : Not yet done.");
  if (its != 1) SETERRQ(1,"its != 1 : Not yet done.");

  if (flag == SOR_APPLY_UPPER) {
    SETERRQ(1,"Not done yet");
  }
  if (flag == SOR_APPLY_LOWER) {
    SETERRQ(1,"Not done yet");
  }
  if (flag & SOR_EISENSTAT) {
    SETERRQ(1,"Not done yet");
  }
  if (flag & SOR_LOCAL_FORWARD_SWEEP) {
    SETERRQ(1,"Not done yet");
  }
  if (flag & SOR_LOCAL_BACKWARD_SWEEP) {
    SETERRQ(1,"Not done yet");
  }
  if (flag & SOR_LOCAL_SYMMETRIC_SWEEP) {
    SETERRQ(1,"Not done yet");
  }
  if (flag & SOR_FORWARD_SWEEP) {
    MLOG_ELM(bsif->procinfo->procset);
    if (bsif->procinfo->single) {
      BSfor_solve1(bsif->pA,b,bsif->comm_pA,bsif->procinfo); CHKERRBS(0);
    } else {
      BSfor_solve(bsif->pA,b,bsif->comm_pA,bsif->procinfo); CHKERRBS(0);
    }
    MLOG_ACC(MS_FORWARD);
  }
  if (flag & SOR_BACKWARD_SWEEP) {
    MLOG_ELM(bsif->procinfo->procset);
    if (bsif->procinfo->single) {
      BSback_solve1(bsif->pA,b,bsif->comm_pA,bsif->procinfo); CHKERRBS(0);
    } else {
      BSback_solve(bsif->pA,b,bsif->comm_pA,bsif->procinfo); CHKERRBS(0);
    }
    MLOG_ACC(MS_BACKWARD);
  }
  ierr = VecCopy(bb,xx); CHKERRQ(ierr);
  return 0;
}

static int MatGetInfo_MPIRowbs(Mat matin,MatInfoType flag,int *nz,int *nzalloc,
                               int *mem)
{
  Mat_MPIRowbs *mat = (Mat_MPIRowbs *) matin->data;
  int          isend[3], irecv[3];

  isend[0] = mat->nz; isend[1] = mat->maxnz; isend[2] = mat->mem;
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

static int MatGetDiagonal_MPIRowbs(Mat mat,Vec v)
{
  Mat_MPIRowbs *mrow = (Mat_MPIRowbs *) mat->data;
  BSsprow      **rs = mrow->A->rows;
  int          i, n;
  Scalar       *x, zero = 0.0, *scale = mrow->pA->scale_diag;

  if (!mrow->assembled) 
    SETERRQ(1,"MatGetDiag_MPIRowbs: Must assemble matrix first.");
  VecSet(&zero,v);
  VecGetArray(v,&x); VecGetLocalSize(v,&n);
  if (n != mrow->m) SETERRQ(1,"Nonconforming matrix and vector.");
  if (mrow->vecs_permscale) {
    for ( i=0; i<mrow->m; i++ ) {
      x[i] = rs[i]->nz[rs[i]->diag_ind];
    }
  } else {
    for ( i=0; i<mrow->m; i++ ) {
      x[i] = rs[i]->nz[rs[i]->diag_ind] * scale[i]; 
    }
  }
  return 0;
}

static int MatDestroy_MPIRowbs(PetscObject obj)
{
  Mat          mat = (Mat) obj;
  Mat_MPIRowbs *mrow = (Mat_MPIRowbs *) mat->data;
  BSspmat   *A = mrow->A;
  BSsprow   *vs;
  int       i, ierr;

  if (mrow->fact_clone) {
    mrow->fact_clone = 0;
    return 0;
  }
#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows %d Cols %d",mrow->M,mrow->N);
#endif
  PETSCFREE(mrow->rowners); 

    if (mrow->bsmap) {
      if (mrow->bsmap->vlocal2global) PETSCFREE(mrow->bsmap->vlocal2global);
      if (mrow->bsmap->vglobal2local) PETSCFREE(mrow->bsmap->vglobal2local);
   /* if (mrow->bsmap->vglobal2proc)  PETSCFREE(mrow->bsmap->vglobal2proc); */
      PETSCFREE(mrow->bsmap);
    } 

    if (A) {
      for (i=0; i<mrow->m; i++) {
        vs = A->rows[i];
        ierr = MatFreeRowbs_Private(mat,vs->length,vs->col,vs->nz); CHKERRQ(ierr);
      }
      /* Note: A->map = mrow->bsmap is freed above */
      PETSCFREE(A->rows);
      PETSCFREE(A);
    }
    if (mrow->procinfo) {BSfree_ctx(mrow->procinfo); CHKERRBS(0);}
    if (mrow->diag)     {ierr = VecDestroy(mrow->diag); CHKERRQ(ierr);}
    if (mrow->xwork)    {ierr = VecDestroy(mrow->xwork); CHKERRQ(ierr);}
    if (mrow->pA)       {BSfree_par_mat(mrow->pA); CHKERRBS(0);}
    if (mrow->fpA)      {BSfree_copy_par_mat(mrow->fpA); CHKERRBS(0);}
    if (mrow->comm_pA)  {BSfree_comm(mrow->comm_pA); CHKERRBS(0);}
    if (mrow->comm_fpA) {BSfree_comm(mrow->comm_fpA); CHKERRBS(0);}
    if (mrow->imax)     PETSCFREE(mrow->imax);    
    if (mrow->inv_diag) PETSCFREE(mrow->inv_diag);

  PETSCFREE(mrow);  
  PLogObjectDestroy(mat);
  PETSCHEADERDESTROY(mat);
  return 0;
}

static int MatSetOption_MPIRowbs(Mat mat,MatOption op)
{
  Mat_MPIRowbs *mrow = (Mat_MPIRowbs *) mat->data;

  if      (op == ROW_ORIENTED)              mrow->roworiented = 1;
  else if (op == COLUMN_ORIENTED)           mrow->roworiented = 0;
  else if (op == COLUMNS_SORTED)            mrow->sorted      = 1;
  if      (op == NO_NEW_NONZERO_LOCATIONS)  mrow->nonew       = 1;
  else if (op == YES_NEW_NONZERO_LOCATIONS) mrow->nonew       = 0;

  else if (op == COLUMN_ORIENTED) SETERRQ(1,"Column oriented not supported");
  return 0;
}

static int MatGetSize_MPIRowbs(Mat mat,int *m,int *n)
{
  Mat_MPIRowbs *mrow = (Mat_MPIRowbs *) mat->data;
  *m = mrow->M; *n = mrow->N;
  return 0;
}

static int MatGetLocalSize_MPIRowbs(Mat mat,int *m,int *n)
{
  Mat_MPIRowbs *mrow = (Mat_MPIRowbs *) mat->data;
  *m = mrow->m; *n = mrow->N;
  return 0;
}

static int MatGetOwnershipRange_MPIRowbs(Mat matin,int *m,int *n)
{
  Mat_MPIRowbs *mat = (Mat_MPIRowbs *) matin->data;
  *m = mat->rstart; *n = mat->rend;
  return 0;
}

static int MatGetRow_MPIRowbs(Mat matin,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIRowbs *mat = (Mat_MPIRowbs *) matin->data;
  BSspmat      *A = mat->A;
  BSsprow      *rs;
   
  if (!mat->assembled) 
    SETERRQ(1,"MatGetRow_MPIRowbs: Must assemble matrix first.");
  if (row < mat->rstart || row >= mat->rend) 
    SETERRQ(1,"MatGetRow_MPIRowbs: Currently you can get only local rows.");

  rs  = A->rows[row - mat->rstart];
  *nz = rs->length;
  if (v)   *v   = rs->nz;
  if (idx) *idx = rs->col;
  return 0;
}

static int MatRestoreRow_MPIRowbs(Mat mat,int row,int *nz,int **idx,Scalar **v)
{
  return 0;
}

/* -------------------------------------------------------------------*/
extern int MatCholeskyFactorNumeric_MPIRowbs(Mat,Mat*);
extern int MatIncompleteCholeskyFactorSymbolic_MPIRowbs(Mat,IS,double,
                                                               int,Mat *);
extern int MatSolve_MPIRowbs(Mat,Vec,Vec);
extern int MatForwardSolve_MPIRowbs(Mat,Vec,Vec);
extern int MatBackwardSolve_MPIRowbs(Mat,Vec,Vec);

static struct _MatOps MatOps = {MatSetValues_MPIRowbs,
       MatGetRow_MPIRowbs,MatRestoreRow_MPIRowbs,
       MatMult_MPIRowbs,0, 
       MatMult_MPIRowbs,0,
       MatSolve_MPIRowbs,0,0,0,
       0,0,
       MatRelax_MPIRowbs,
       0,
       MatGetInfo_MPIRowbs,0,
       MatGetDiagonal_MPIRowbs,0,0,
       MatAssemblyBegin_MPIRowbs,MatAssemblyEnd_MPIRowbs,
       0,
       MatSetOption_MPIRowbs,MatZeroEntries_MPIRowbs,MatZeroRows_MPIRowbs,0,
       0,0,0,MatCholeskyFactorNumeric_MPIRowbs,
       MatGetSize_MPIRowbs,MatGetLocalSize_MPIRowbs,
       MatGetOwnershipRange_MPIRowbs,
       0,MatIncompleteCholeskyFactorSymbolic_MPIRowbs,
       0,0,0,0,0,0,MatForwardSolve_MPIRowbs,MatBackwardSolve_MPIRowbs};

/* ------------------------------------------------------------------- */

/*@
   MatCreateMPIRowbs - Creates a symmetric, sparse parallel matrix in 
   the MPIRowbs format.  This format is currently only partially 
   supported and is intended primarily as a BlockSolve interface.

   Input Parameters:
.  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated)
.  M - number of global rows (or PETSC_DECIDE to have calculated)
.  nz - number of nonzeros per row (same for all local rows)
.  nzz - number of nonzeros per row (possibly different for each row).
.  procinfo - optional BlockSolve BSprocinfo context.  If zero, then the
   context will be created and initialized.

   Output Parameter:
.  newmat - the matrix 

   Notes:
   The MPIRowbs format is for SYMMETRIC matrices only!

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   Specify the preallocated storage with either nz or nnz (not both).
   Set both nz and nnz to zero for PETSc to control dynamic memory 
   allocation.
  
.keywords: matrix, row, symmetric, sparse, parallel, BlockSolve

.seealso: MatCreate(), MatSetValues()
@*/
int MatCreateMPIRowbs(MPI_Comm comm,int m,int M,int nz, int *nnz,
                       void *procinfo,Mat *newmat)
{
  Mat          mat;
  Mat_MPIRowbs *mrow;
  BSmapping    *bsmap;
  BSoff_map    *bsoff;
  int          i, ierr, Mtemp, *offset, low, high;
  BSprocinfo   *bspinfo = (BSprocinfo *) procinfo;
  
  PETSCHEADERCREATE(mat,_Mat,MAT_COOKIE,MATMPIROW_BS,comm);
  PLogObjectCreate(mat);
  mat->data	= (void *) (mrow = PETSCNEW(Mat_MPIRowbs)); CHKPTRQ(mrow);
  mat->ops	= &MatOps;
  mat->destroy	= MatDestroy_MPIRowbs;
  mat->view	= MatView_MPIRowbs;
  mat->factor	= 0;
  mrow->assembled        = 0;
  mrow->fact_clone       = 0;
  mrow->vecs_permscale   = 0;
  mrow->reassemble_begun = 0;
  mrow->insertmode       = NOTSETVALUES;
  MPI_Comm_rank(comm,&mrow->mytid);
  MPI_Comm_size(comm,&mrow->numtids);

  if (M != PETSC_DECIDE && m != PETSC_DECIDE) {
    /* Perhaps this should be removed for better efficiency -- but could be
       risky. */
    MPI_Allreduce(&m,&Mtemp,1,MPI_INT,MPI_SUM,comm);
    if (Mtemp != M) SETERRQ(1,"Sum of local dimensions != global dimension.");
  } else if (M == PETSC_DECIDE) {
    MPI_Allreduce(&m,&M,1,MPI_INT,MPI_SUM,comm);
  } else if (m == PETSC_DECIDE) {
    {m = M/mrow->numtids + ((M % mrow->numtids) > mrow->mytid);}
  } else {
    SETERRQ(1,"MatCreateMPIRowbs: Must set local and/or global matrix size.");
  }
  mrow->N    = M;
  mrow->M    = M;
  mrow->m    = m;
  mrow->n    = mrow->N; /* each row stores all columns */
  mrow->imax = (int *) PETSCMALLOC( (mrow->m+1)*sizeof(int) ); 
  CHKPTRQ(mrow->imax);

  /* build local table of row ownerships */
  mrow->rowners = (int *) PETSCMALLOC((mrow->numtids+2)*sizeof(int)); 
  CHKPTRQ(mrow->rowners);
  MPI_Allgather(&m,1,MPI_INT,mrow->rowners+1,1,MPI_INT,comm);
  mrow->rowners[0] = 0;
  for ( i=2; i<=mrow->numtids; i++ ) {
    mrow->rowners[i] += mrow->rowners[i-1];
  }
  mrow->rstart = mrow->rowners[mrow->mytid]; 
  mrow->rend   = mrow->rowners[mrow->mytid+1]; 

  /* build cache for off array entries formed */
  ierr = StashBuild_Private(&mrow->stash); CHKERRQ(ierr);

  /* Initialize BlockSolve information */
  mrow->A	    = 0;
  mrow->pA	    = 0;
  mrow->comm_pA	    = 0;
  mrow->fpA	    = 0;
  mrow->comm_fpA    = 0;
  mrow->alpha	    = 1.0;
  mrow->ierr	    = 0;
  mrow->failures    = 0;
  ierr = VecCreateMPI(mat->comm,mrow->m,mrow->M,&(mrow->diag)); CHKERRQ(ierr);
  ierr = VecDuplicate(mrow->diag,&(mrow->xwork));CHKERRQ(ierr);
  mrow->inv_diag = (Scalar *) PETSCMALLOC( (mrow->m+1)*sizeof(Scalar) );
  CHKPTRQ(mrow->inv_diag);
  if (!bspinfo) {bspinfo = BScreate_ctx(); CHKERRBS(0);}
  mrow->procinfo = bspinfo;
  BSctx_set_id(bspinfo,mrow->mytid); CHKERRBS(0);
  BSctx_set_np(bspinfo,mrow->numtids); CHKERRBS(0);
  BSctx_set_ps(bspinfo,comm); CHKERRBS(0);
  BSctx_set_cs(bspinfo,INT_MAX); CHKERRBS(0);
  BSctx_set_is(bspinfo,INT_MAX); CHKERRBS(0);
  BSctx_set_ct(bspinfo,IDO); CHKERRBS(0);
#if defined(PETSC_DEBUG)
  BSctx_set_err(bspinfo,1); CHKERRBS(0);  /* BS error checking */
#else
  BSctx_set_err(bspinfo,0); CHKERRBS(0);
#endif
  BSctx_set_rt(bspinfo,1); CHKERRBS(0);
  BSctx_set_pr(bspinfo,1); CHKERRBS(0);
  BSctx_set_si(bspinfo,0); CHKERRBS(0);
#if defined(PETSC_DEBUG)
  MLOG_INIT();  /* Initialize logging */
#endif

  /* Compute global offsets */
  ierr = MatGetOwnershipRange(mat,&low,&high); CHKERRQ(ierr);
  offset = &low;

  mrow->bsmap = (void *) PETSCNEW(BSmapping); CHKPTRQ(mrow->bsmap);
  bsmap = mrow->bsmap;
  bsmap->vlocal2global	= (int *) PETSCMALLOC(sizeof(int)); 
	CHKPTRQ(bsmap->vlocal2global);
	*((int *)bsmap->vlocal2global) = (*offset);
  bsmap->flocal2global	= BSloc2glob;
  bsmap->free_l2g	= 0;
  bsmap->vglobal2local	= (int *) PETSCMALLOC(sizeof(int)); 
	CHKPTRQ(bsmap->vglobal2local);
	*((int *)bsmap->vglobal2local) = (*offset);
  bsmap->fglobal2local	= BSglob2loc;
  bsmap->free_g2l	= 0;
  bsoff = BSmake_off_map( *offset, bspinfo, mrow->M );
  bsmap->vglobal2proc	= (void *)bsoff;
  bsmap->fglobal2proc	= BSglob2proc;
  bsmap->free_g2p	= 0;

  ierr = MatCreateMPIRowbs_local(mat,nz,nnz); CHKERRQ(ierr);
  *newmat = mat;
  return 0;
}
/* --------------- extra BlockSolve-specific routines -------------- */
int MatForwardSolve_MPIRowbs(Mat mat,Vec x,Vec y)
{
  Mat_MPIRowbs *mrow = (Mat_MPIRowbs *) mat->data;
  Scalar       *ya;
  int          ierr;

  /* Apply diagonal scaling to vector, where D^{-1/2} is stored */
  ierr = VecPMult(mrow->diag,x,y); CHKERRQ(ierr);
  ierr = VecGetArray(y,&ya); CHKERRQ(ierr);

#ifdef DEBUG_ALL
  MLOG_ELM(mrow->procinfo->procset);
#endif
  if (mrow->procinfo->single)
    /* Use BlockSolve routine for no cliques/inodes */
    BSfor_solve1( mrow->fpA, ya, mrow->comm_pA, mrow->procinfo );
  else
    BSfor_solve( mrow->fpA, ya, mrow->comm_pA, mrow->procinfo );
  CHKERRBS(0);
#ifdef DEBUG_ALL
  MLOG_ACC(MS_FORWARD);
#endif
  return(0);
}

int MatBackwardSolve_MPIRowbs(Mat mat,Vec x,Vec y)
{
  Mat_MPIRowbs *mrow = (Mat_MPIRowbs *) mat->data;
  Scalar       *ya;
  int          ierr;

  ierr = VecCopy(x,y); CHKERRQ(ierr);
  ierr = VecGetArray(y,&ya); CHKERRQ(ierr);
#ifdef DEBUG_ALL
  MLOG_ELM(mrow->procinfo->procset);
#endif
  if (mrow->procinfo->single)
    /* Use BlockSolve routine for no cliques/inodes */
    BSback_solve1( mrow->fpA, ya, mrow->comm_pA, mrow->procinfo );
  else
    BSback_solve( mrow->fpA, ya, mrow->comm_pA, mrow->procinfo );
  CHKERRBS(0);
#ifdef DEBUG_ALL
  MLOG_ACC(MS_BACKWARD);
#endif

  /* Apply diagonal scaling to vector, where D^{-1/2} is stored */
  ierr = VecPMult(y,mrow->diag,y); CHKERRQ(ierr);
  return 0;
}

/* @
  MatGetBSProcinfo - Gets the BlockSolve BSprocinfo context, which the
  user can then manipulate to alter the default parameters.

  Input Parameter:
  mat - matrix

  Output Parameter:
  procinfo - processor information context

  Note:
  This routine is valid only for matrices stored in the MATMPIROW_BS
  format.
@ */
int MatGetBSProcinfo(Mat mat,BSprocinfo *procinfo)
{
  Mat_MPIRowbs *mrow = (Mat_MPIRowbs *) mat->data;
  if (mat->type != MATMPIROW_BS) 
    SETERRQ(1,"Valid only for MATMPIROW_BS matrix type.");
  procinfo = mrow->procinfo;
  return 0;
}

#else
#include "petsc.h"
#include "mat.h"
int MatCreateMPIRowbs(MPI_Comm comm,int m,int M,int nz, int *nnz,
                       void *bspinfo,Mat *newmat)
{SETERRQ(1,"This matrix format requires BlockSolve.");}
#endif





