#ifndef lint
static char vcid[] = "$Id: mpirowbs.c,v 1.28 1995/05/26 19:06:31 curfman Exp bsmith $";
#endif

#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
#include "mpirowbs.h"
#include "vec/vecimpl.h"
#include "inline/spops.h"
#include "BSprivate.h"
#include "BSsparse.h"

#define CHUNCKSIZE_LOCAL   10

/* Same as MatRow format ... should share these! */
static int MatFreeRowbs_Private(Mat matin,int n,int *i,Scalar *v)
{
  if (v) PFREE(v);
  return 0;
}

static int MatMallocRowbs_Private(Mat matin,int n,int **i,Scalar **v)
{
  int len;
  if (n == 0) {
    *i = 0; *v = 0;
  } else {
    len = n*(sizeof(int) + sizeof(Scalar));
    *v = (Scalar *) PMALLOC( len ); PCHKPTR(*v);
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
    nnz = (int *) PMALLOC( m*sizeof(int) ); PCHKPTR(nnz);
    for ( i=0; i<m; i++ ) nnz[i] = nz;
    nz = nz*m;
  }
  else {
    nz = 0;
    for ( i=0; i<m; i++ ) nz += nnz[i];
  }

  /* Allocate BlockSolve matrix context */
  bsif->A = bsmat = PNEW(BSspmat); PCHKPTR(bsmat);
  len = m*(sizeof(BSsprow *) + sizeof(BSsprow));
  bsmat->rows = (BSsprow **) PETSCMALLOC( len ); PCHKPTR(bsmat->rows);
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
      PCHKERR(ierr);
    } else {
      vs->col = 0; vs->nz = 0;
    }
    vs++;
  }
  bsif->mem = sizeof(BSspmat) + len + nz*(sizeof(int) + sizeof(Scalar));
  bsif->nz	    = 0;
  bsif->maxnz	    = nz;
  bsif->sorted      = 0;
  bsif->roworiented = 1;
  bsif->nonew       = 0;
  bsif->singlemalloc = 0;

  if (nzalloc) PFREE(nnz);
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
    if (row < 0) PSETERR(1,"Negative row index");
    if (row >= mat->m) PSETERR(1,"Row index too large");
    vs = A->rows[row];
    ap = vs->nz; rp = vs->col;
    rmax = imax[row]; nrow = vs->length;
    a = 0;
    for ( l=0; l<n; l++ ) { /* loop over added columns */
      if (idxn[l] < 0) PSETERR(1,"Negative column index");
      if (idxn[l] >= mat->N) PSETERR(1,"Column index too large");
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
        ierr = MatMallocRowbs_Private(matin,imax[row],&itemp,&vtemp); PCHKERR(ierr);
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
          {ierr = MatFreeRowbs_Private(matin,rmax,vs->col,vs->nz); PCHKERR(ierr);}
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
       PSETERR(1,"No diagonal term!  Must set diagonal entry, even if zero.");
  }
  return 0;
}

/* ----------------------------------------------------------------- */

static int MatSetValues_MPIRowbs(Mat mat,int m,int *idxm,int n,
                            int *idxn,Scalar *v,InsertMode addv)
{
  Mat_MPIRowbs *mrow = (Mat_MPIRowbs *) mat->data;
  int          ierr, i, j, row, col, rstart = mrow->rstart, rend = mrow->rend;

  if (mrow->insertmode != NOTSETVALUES && mrow->insertmode != addv) {
    PSETERR(1,"You cannot mix inserts and adds");
  }
  mrow->insertmode = addv;
  if ((mrow->assembled) && (!mrow->reassemble_begun)) {
    /* Symmetrically unscale the matrix by the diagonal */
    BSscale_diag(mrow->pA,mrow->inv_diag,mrow->procinfo); CHKERRBS(0);
    mrow->reassemble_begun = 1;
  }
  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) PSETERR(1,"Negative row index");
    if (idxm[i] >= mrow->M) PSETERR(1,"Row index too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (idxn[j] < 0) PSETERR(1,"Negative column index");
        if (idxn[j] >= mrow->N) PSETERR(1,"Column index too large");
        if (idxn[j] >= 0 && idxn[j] < mrow->N){
          col = idxn[j];
          ierr = MatSetValues_MPIRowbs_local(mat,1,&row,1,&col,
                                             v+i*n+j,addv);PCHKERR(ierr);
        }
        else {PSETERR(1,"Invalid column index.");}
      }
    } 
    else {
      ierr = StashValues_Private(&mrow->stash,idxm[i],n,idxn,v+i*n,addv);
      PCHKERR(ierr);
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
  int         tag = 50, *owner,*starts,count,ierr;
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
    PSETERR(1,"Some processors have inserted while others have added");
  }
  mrow->insertmode = addv; /* in case this processor had no cache */

  /*  first count number of contributors to each processor */
  nprocs = (int *) PMALLOC( 2*numtids*sizeof(int) ); PCHKPTR(nprocs);
  MEMSET(nprocs,0,2*numtids*sizeof(int)); procs = nprocs + numtids;
  owner = (int *) PMALLOC( (mrow->stash.n+1)*sizeof(int) ); PCHKPTR(owner);
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
  work = (int *) PMALLOC( numtids*sizeof(int) ); PCHKPTR(work);
  MPI_Allreduce((void *) procs,(void *) work,numtids,MPI_INT,MPI_SUM,comm);
  nreceives = work[mytid]; 
  MPI_Allreduce((void *) nprocs,(void *) work,numtids,MPI_INT,MPI_MAX,comm);
  nmax = work[mytid];
  PFREE(work);

  /* post receives: 
       1) each message will consist of ordered pairs 
     (global index,value) we store the global index as a double 
     to simplify the message passing. 
       2) since we don't know how long each individual message is we 
     allocate the largest needed buffer for each receive. Potentially 
     this is a lot of wasted space.


       This could be done better.
  */
  rvalues = (Scalar *) PMALLOC(3*(nreceives+1)*(nmax+1)*sizeof(Scalar));
  PCHKPTR(rvalues);
  recv_waits = (MPI_Request *) PMALLOC((nreceives+1)*sizeof(MPI_Request));
  PCHKPTR(recv_waits);
  for ( i=0; i<nreceives; i++ ) {
    MPI_Irecv((void *)(rvalues+3*nmax*i),3*nmax,MPI_SCALAR,MPI_ANY_SOURCE,tag,
              comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (Scalar *) PMALLOC( 3*(mrow->stash.n+1)*sizeof(Scalar) );
  PCHKPTR(svalues);
  send_waits = (MPI_Request *) PMALLOC( (nsends+1)*sizeof(MPI_Request));
  PCHKPTR(send_waits);
  starts = (int *) PMALLOC( numtids*sizeof(int) ); PCHKPTR(starts);
  starts[0] = 0; 
  for ( i=1; i<numtids; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for ( i=0; i<mrow->stash.n; i++ ) {
    svalues[3*starts[owner[i]]]       = (Scalar)  mrow->stash.idx[i];
    svalues[3*starts[owner[i]]+1]     = (Scalar)  mrow->stash.idy[i];
    svalues[3*(starts[owner[i]]++)+2] =  mrow->stash.array[i];
  }
  PFREE(owner);
  starts[0] = 0;
  for ( i=1; i<numtids; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<numtids; i++ ) {
    if (procs[i]) {
      MPI_Isend((void*)(svalues+3*starts[i]),3*nprocs[i],MPI_SCALAR,i,tag,
                comm,send_waits+count++);
    }
  }
  PFREE(starts); PFREE(nprocs);

  /* Free cache space */
  ierr = StashDestroy_Private(&mrow->stash); PCHKERR(ierr);

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
    PSETERR(1,"MatView_MPIRow: Must assemble matrix first.");
  if (!viewer) { /* so that viewers may be used from debuggers */
    viewer = STDOUT_VIEWER; vobj = (PetscObject) viewer;
  }
  if (vobj->cookie == DRAW_COOKIE && vobj->type == NULLWINDOW) return 0;
  if (vobj->cookie == VIEWER_COOKIE && vobj->type == FILE_VIEWER) {
    FILE *fd = ViewerFileGetPointer_Private(viewer);
    MPIU_Seq_begin(mat->comm,1);
    fprintf(fd,"[%d] rows %d starts %d ends %d cols %d starts %d ends %d\n",
           mrow->mytid,mrow->m,mrow->rstart,mrow->rend,mrow->n,0,mrow->N);
    ierr = MatView_MPIRowbs_local(mat,viewer); PCHKERR(ierr);
    fflush(fd);
    MPIU_Seq_end(mat->comm,1);
  }
  else if ((vobj->cookie == VIEWER_COOKIE && vobj->type == FILES_VIEWER) || 
            vobj->cookie == DRAW_COOKIE) {
    int numtids = mrow->numtids, mytid = mrow->mytid; 
    if (numtids == 1) { 
      ierr = MatView_MPIRowbs_local(mat,viewer); PCHKERR(ierr);
    }
    else {
      /* assemble the entire matrix onto first processor. */
      Mat       A;
      int       M = mrow->M, m, row, i, nz, *cols;
      Scalar    *vals;

      if (!mytid) {
        ierr = MatCreateMPIRowbs(mat->comm,M,M,0,0,0,&A);
      }
      else {
        ierr = MatCreateMPIRowbs(mat->comm,0,M,0,0,0,&A);
      }
      PCHKERR(ierr);

      /* Copy the matrix ... This isn't the most efficient means,
         but it's quick for now */
      row = mrow->rstart; m = mrow->m;
      for ( i=0; i<m; i++ ) {
        ierr = MatGetRow(mat,row,&nz,&cols,&vals); PCHKERR(ierr);
        ierr = MatSetValues(A,1,&row,nz,cols,vals,INSERTVALUES); PCHKERR(ierr);
        ierr = MatRestoreRow(mat,row,&nz,&cols,&vals); PCHKERR(ierr);
        row++;
      } 

      ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); PCHKERR(ierr);
      ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); PCHKERR(ierr);
      if (!mytid) {
        ierr = MatView_MPIRowbs_local(A,viewer); PCHKERR(ierr);
      }
      ierr = MatDestroy(A); PCHKERR(ierr);

    }
  }
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
    MPI_Get_count(&recv_status,MPI_SCALAR,&n);
    n = n/3;
    for ( i=0; i<n; i++ ) {
      row = (int) PETSCREAL(values[3*i]) - mrow->rstart;
      col = (int) PETSCREAL(values[3*i+1]);
      val = values[3*i+2];
      if (col >= 0 && col < mrow->N) {
        MatSetValues_MPIRowbs_local(mat,1,&row,1,&col,&val,addv);
      } 
      else {PSETERR(1,"Invalid column index.");}
    }
    count--;
  }
  PFREE(mrow->recv_waits); PFREE(mrow->rvalues);
 
  /* wait on sends */
  if (mrow->nsends) {
    send_status = (MPI_Status *) PMALLOC( mrow->nsends*sizeof(MPI_Status) );
    PCHKPTR(send_status);
    MPI_Waitall(mrow->nsends,mrow->send_waits,send_status);
    PFREE(send_status);
  }
  PFREE(mrow->send_waits); PFREE(mrow->svalues);

  mrow->insertmode = NOTSETVALUES;
  ierr = MatAssemblyBegin_MPIRowbs_local(mat,mode); PCHKERR(ierr);
  ierr = MatAssemblyEnd_MPIRowbs_local(mat,mode); PCHKERR(ierr);

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
    ierr = VecGetLocalSize( mrow->diag, &ldim ); PCHKERR(ierr);
    ierr = VecGetOwnershipRange( mrow->diag, &low, &high ); PCHKERR(ierr);
    for (i=0; i<ldim; i++) {
      loc = low + i;
      val = 1.0/sqrt(mrow->pA->scale_diag[i]);
      mrow->inv_diag[i] = 1.0/(mrow->pA->scale_diag[i]);
      ierr = VecSetValues(mrow->diag,1,&loc,&val,INSERTVALUES); PCHKERR(ierr);
    }
    ierr = VecAssemblyBegin( mrow->diag ); PCHKERR(ierr);
    ierr = VecAssemblyEnd( mrow->diag ); PCHKERR(ierr);
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

static int MatMult_MPIRowbs(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIRowbs *bsif = (Mat_MPIRowbs *) mat->data;
  BSprocinfo   *bspinfo = bsif->procinfo;
  Scalar       *xxa, *xworka, *yya;
  int          ierr;

  if (!bsif->assembled) 
    PSETERR(1,"MatMult_MPIRowbs: Must assemble matrix first.");
  ierr = VecGetArray(yy,&yya); PCHKERR(ierr);
  ierr = VecGetArray(xx,&xxa); PCHKERR(ierr);

  /* Permute and apply diagonal scaling:  [ xwork = D^{1/2} * x ] */
  if (!bsif->vecs_permscale) {
    ierr = VecGetArray(bsif->xwork,&xworka); PCHKERR(ierr);
    BSperm_dvec(xxa,xworka,bsif->pA->perm); CHKERRBS(0);
    ierr = VecPDiv(bsif->xwork,bsif->diag,xx); PCHKERR(ierr);
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
    ierr = VecPDiv(yy,bsif->diag,bsif->xwork); PCHKERR(ierr);
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
  PSETERR(1,"Not done yet");

  if (!bsif->assembled) 
    PSETERR(1,"MatRelax_MPIRowbs: Must assemble matrix first");
  VecGetArray(bb,&b);
  if (flag & SOR_ZERO_INITIAL_GUESS) {
      /* do nothing */ 
  } else PSETERR(1,"Not done yet.");
  if (shift) PSETERR(1,"shift != 0 : Not yet done.")
  if (omega != 1.0) PSETERR(1,"omega != 1.0 : Not yet done.")
  if (its != 1) PSETERR(1,"its != 1 : Not yet done.")

  if (flag == SOR_APPLY_UPPER) {
    PSETERR(1,"Not done yet");
  }
  if (flag == SOR_APPLY_LOWER) {
    PSETERR(1,"Not done yet");
  }
  if (flag & SOR_EISENSTAT) {
    PSETERR(1,"Not done yet");
  }
  if (flag & SOR_LOCAL_FORWARD_SWEEP) {
    PSETERR(1,"Not done yet");
  }
  if (flag & SOR_LOCAL_BACKWARD_SWEEP) {
    PSETERR(1,"Not done yet");
  }
  if (flag & SOR_LOCAL_SYMMETRIC_SWEEP) {
    PSETERR(1,"Not done yet");
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
  ierr = VecCopy(bb,xx); PCHKERR(ierr);
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
    PSETERR(1,"MatGetDiag_MPIRowbs: Must assemble matrix first.");
  VecSet(&zero,v);
  VecGetArray(v,&x); VecGetLocalSize(v,&n);
  if (n != mrow->m) PSETERR(1,"Nonconforming matrix and vector.");
  if (mrow->vecs_permscale) {
    for ( i=0; i<mrow->m; i++ ) {
      x[i] = rs[i]->nz[rs[i]->diag_ind];
      printf("x[%d] = %g\n",i,x[i]);
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
  PFREE(mrow->rowners); 

    if (mrow->bsmap) {
      if (mrow->bsmap->vlocal2global) PFREE(mrow->bsmap->vlocal2global);
      if (mrow->bsmap->vglobal2local) PFREE(mrow->bsmap->vglobal2local);
   /* if (mrow->bsmap->vglobal2proc)  PFREE(mrow->bsmap->vglobal2proc); */
      PFREE(mrow->bsmap);
    } 

    if (A) {
      for (i=0; i<mrow->m; i++) {
        vs = A->rows[i];
        ierr = MatFreeRowbs_Private(mat,vs->length,vs->col,vs->nz); PCHKERR(ierr);
      }
      /* Note: A->map = mrow->bsmap is freed above */
      PETSCFREE(A->rows);
      PFREE(A);
    }
    if (mrow->procinfo) {BSfree_ctx(mrow->procinfo); CHKERRBS(0);}
    if (mrow->diag)     {ierr = VecDestroy(mrow->diag); PCHKERR(ierr);}
    if (mrow->xwork)    {ierr = VecDestroy(mrow->xwork); PCHKERR(ierr);}
    if (mrow->pA)       {BSfree_par_mat(mrow->pA); CHKERRBS(0);}
    if (mrow->fpA)      {BSfree_copy_par_mat(mrow->fpA); CHKERRBS(0);}
    if (mrow->comm_pA)  {BSfree_comm(mrow->comm_pA); CHKERRBS(0);}
    if (mrow->comm_fpA) {BSfree_comm(mrow->comm_fpA); CHKERRBS(0);}
    if (mrow->imax)     PFREE(mrow->imax);    
    if (mrow->inv_diag) PFREE(mrow->inv_diag);

  PFREE(mrow);  
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

  else if (op == COLUMN_ORIENTED) PSETERR(1,"Column oriented not supported");
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
  *m = mrow->m; *n = mrow->n;
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
    PSETERR(1,"MatGetRow_MPIRowbs: Must assemble matrix first.");
  if (row < mat->rstart || row >= mat->rend) 
    PSETERR(1,"MatGetRow_MPIRowbs: Currently you can get only local rows.");

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
extern int MatIncompleteCholeskyFactorSymbolic_MPIRowbs(Mat,IS,int,Mat *);
extern int MatSolve_MPIRowbs(Mat,Vec,Vec);

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
       MatSetOption_MPIRowbs,MatZeroEntries_MPIRowbs,0,0,
       0,0,0,MatCholeskyFactorNumeric_MPIRowbs,
       MatGetSize_MPIRowbs,MatGetLocalSize_MPIRowbs,
       MatGetOwnershipRange_MPIRowbs,
       0,MatIncompleteCholeskyFactorSymbolic_MPIRowbs,
       0,0,0,0,0,0};

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
  int          i, ierr, *offset, low, high;
  BSprocinfo   *bspinfo = (BSprocinfo *) procinfo;
  
  PETSCHEADERCREATE(mat,_Mat,MAT_COOKIE,MATMPIROW_BS,comm);
  PLogObjectCreate(mat);
  mat->data	= (void *) (mrow = PNEW(Mat_MPIRowbs)); PCHKPTR(mrow);
  mat->ops	= &MatOps;
  mat->destroy	= MatDestroy_MPIRowbs;
  mat->view	= MatView_MPIRowbs;
  mat->factor	= 0;
  mrow->assembled        = 0;
  mrow->fact_clone       = 0;
  mrow->reassemble_begun = 0;
  mrow->insertmode       = NOTSETVALUES;
  MPI_Comm_rank(comm,&mrow->mytid);
  MPI_Comm_size(comm,&mrow->numtids);

  if (M == PETSC_DECIDE) {MPI_Allreduce(&m,&M,1,MPI_INT,MPI_SUM,comm );}
  if (m == PETSC_DECIDE) 
    {m = M/mrow->numtids + ((M % mrow->numtids) > mrow->mytid);}
  mrow->N    = M;
  mrow->M    = M;
  mrow->m    = m;
  mrow->n    = mrow->N; /* each row stores all columns */
  mrow->imax = (int *) PMALLOC( (mrow->m)*sizeof(int) ); PCHKPTR(mrow->imax);

  /* build local table of row ownerships */
  mrow->rowners = (int *) PMALLOC((mrow->numtids+2)*sizeof(int)); 
  PCHKPTR(mrow->rowners);
  MPI_Allgather(&m,1,MPI_INT,mrow->rowners+1,1,MPI_INT,comm);
  mrow->rowners[0] = 0;
  for ( i=2; i<=mrow->numtids; i++ ) {
    mrow->rowners[i] += mrow->rowners[i-1];
  }
  mrow->rstart = mrow->rowners[mrow->mytid]; 
  mrow->rend   = mrow->rowners[mrow->mytid+1]; 

  /* build cache for off array entries formed */
  ierr = StashBuild_Private(&mrow->stash); PCHKERR(ierr);

  /* Initialize BlockSolve information */
  mrow->A	    = 0;
  mrow->pA	    = 0;
  mrow->comm_pA	    = 0;
  mrow->fpA	    = 0;
  mrow->comm_fpA    = 0;
  mrow->alpha	    = 1.0;
  mrow->ierr	    = 0;
  mrow->failures    = 0;
  if (!mrow->diag) {ierr = VecCreateMPI(mat->comm,mrow->m,mrow->M,
                            &(mrow->diag)); PCHKERR(ierr);}
  if (!mrow->xwork) {ierr = VecDuplicate(mrow->diag,&(mrow->xwork)); 
                            PCHKERR(ierr);}
  mrow->inv_diag = (Scalar *) PMALLOC( (mrow->m)*sizeof(Scalar) );
  PCHKPTR(mrow->inv_diag);
  if (!bspinfo) {bspinfo = BScreate_ctx(); CHKERRBS(0);}
  mrow->procinfo = bspinfo;
  BSctx_set_id(bspinfo,mrow->mytid); CHKERRBS(0);
  BSctx_set_np(bspinfo,mrow->numtids); CHKERRBS(0);
  BSctx_set_ps(bspinfo,NULL); CHKERRBS(0);
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
  ierr = MatGetOwnershipRange(mat,&low,&high); PCHKERR(ierr);
  offset = &low;

  if (!mrow->bsmap) {mrow->bsmap = (void *) PNEW(BSmapping); 
                     PCHKPTR(mrow->bsmap);}
  bsmap = mrow->bsmap;
  bsmap->vlocal2global	= (int *) PMALLOC(sizeof(int)); 
	PCHKPTR(bsmap->vlocal2global);
	*((int *)bsmap->vlocal2global) = (*offset);
  bsmap->flocal2global	= BSloc2glob;
  bsmap->free_l2g	= 0;
  bsmap->vglobal2local	= (int *) PMALLOC(sizeof(int)); 
	PCHKPTR(bsmap->vglobal2local);
	*((int *)bsmap->vglobal2local) = (*offset);
  bsmap->fglobal2local	= BSglob2loc;
  bsmap->free_g2l	= 0;
  bsoff = BSmake_off_map( *offset, bspinfo, mrow->M );
  bsmap->vglobal2proc	= (void *)bsoff;
  bsmap->fglobal2proc	= BSglob2proc;
  bsmap->free_g2p	= 0;

  ierr = MatCreateMPIRowbs_local(mat,nz,nnz); PCHKERR(ierr);
  /* the next line is deadly. It scribbles where it is not 
     suppose to, because mrow->A is not a PETSc object */
  /* PLogObjectParent(mat,mrow->A);*/ 
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
  ierr = VecPMult(mrow->diag,x,y); PCHKERR(ierr);
  ierr = VecGetArray(y,&ya); PCHKERR(ierr);

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

  ierr = VecCopy(x,y); PCHKERR(ierr);
  ierr = VecGetArray(y,&ya); PCHKERR(ierr);
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
  ierr = VecPMult(y,mrow->diag,y); PCHKERR(ierr);
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
    PSETERR(1,"Valid only for MATMPIROW_BS matrix type.");
  procinfo = mrow->procinfo;
  return 0;
}

#else
#include "petsc.h"
#include "mat.h"
int MatCreateMPIRowbs(MPI_Comm comm,int m,int M,int nz, int *nnz,
                       void *bspinfo,Mat *newmat)
{PSETERR(1,"This matrix format requires BlockSolve.");}
#endif





