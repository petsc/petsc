#ifndef lint
static char vcid[] = "$Id: mpirowbs.c,v 1.92 1996/02/15 00:56:34 curfman Exp bsmith $";
#endif

#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
#include "mpirowbs.h"
#include "vec/vecimpl.h"
#include "BSprivate.h"

#define CHUNCKSIZE_LOCAL   10

static int MatFreeRowbs_Private(Mat A,int n,int *i,Scalar *v)
{
  if (v) {
    int len = -n*(sizeof(int)+sizeof(Scalar));
    PetscFree(v);
    /* I don't understand why but if I simply log 
       -n*(sizeof(int)+sizeof(Scalar)) as the memory it 
      produces crazy numbers, but this works ok. */
    PLogObjectMemory(A,len);
  }
  return 0;
}

static int MatMallocRowbs_Private(Mat A,int n,int **i,Scalar **v)
{
  int len;

  if (n == 0) {
    *i = 0; *v = 0;
  } else {
    len = n*(sizeof(int) + sizeof(Scalar));
    *v = (Scalar *) PetscMalloc(len); CHKPTRQ(*v);
    PLogObjectMemory(A,len);
    *i = (int *)(*v + n);
  }
  return 0;
}

/* ----------------------------------------------------------------- */
static int MatCreateMPIRowbs_local(Mat A,int nz,int *nnz)
{
  Mat_MPIRowbs *bsif = (Mat_MPIRowbs *) A->data;
  int          ierr, i, len, nzalloc = 0, m = bsif->m;
  BSspmat      *bsmat;
  BSsprow      *vs;

  if (!nnz) {
    if (nz == PETSC_DEFAULT) nz = 5;
    if (nz <= 0)             nz = 1;
    nzalloc = 1;
    nnz     = (int *) PetscMalloc( (m+1)*sizeof(int) ); CHKPTRQ(nnz);
    for ( i=0; i<m; i++ ) nnz[i] = nz;
    nz      = nz*m;
  }
  else {
    nz = 0;
    for ( i=0; i<m; i++ ) {
      if (nnz[i] <= 0) nnz[i] = 1;
      nz += nnz[i];
    }
  }

  /* Allocate BlockSolve matrix context */
  bsif->A                = bsmat = PetscNew(BSspmat); CHKPTRQ(bsmat);
  BSset_mat_icc_storage(bsmat,PETSC_FALSE);
  BSset_mat_symmetric(bsmat,PETSC_FALSE);
  len                    = m*(sizeof(BSsprow *) + sizeof(BSsprow)) + 1;
  bsmat->rows            = (BSsprow **) PetscMalloc( len ); CHKPTRQ(bsmat->rows);
  bsmat->num_rows        = m;
  bsmat->global_num_rows = bsif->M;
  bsmat->map             = bsif->bsmap;
  vs                     = (BSsprow *) (bsmat->rows + m);
  for (i=0; i<m; i++) {
    bsmat->rows[i]  = vs;
    bsif->imax[i]   = nnz[i];
    vs->diag_ind    = -1;
    if (nnz[i] > 0) {
      ierr = MatMallocRowbs_Private(A,nnz[i],&(vs->col),&(vs->nz));CHKERRQ(ierr);
    } else {
      vs->col = 0; vs->nz = 0;
    }
    /* put zero on diagonal */
    vs->length	    = 1;
    vs->col[0]      = i + bsif->rstart;
    vs->nz[0]       = 0.0;
    vs++;
  }
  PLogObjectMemory(A,sizeof(BSspmat) + len);
  bsif->nz	     = 0;
  bsif->maxnz	     = nz;
  bsif->sorted       = 0;
  bsif->roworiented  = 1;
  bsif->nonew        = 0;

  if (nzalloc) PetscFree(nnz);
  return 0;
}

static int MatSetValues_MPIRowbs_local(Mat AA,int m,int *im,int n,int *in,Scalar *v,
                                       InsertMode addv)
{
  Mat_MPIRowbs *mat = (Mat_MPIRowbs *) AA->data;
  BSspmat      *A = mat->A;
  BSsprow      *vs;
  int          *rp,k,a,b,t,ii,row,nrow,i,col,l,rmax, ierr;
  int          *imax = mat->imax, nonew = mat->nonew, sorted = mat->sorted;
  Scalar       *ap, value;

  for ( k=0; k<m; k++ ) { /* loop over added rows */
    row = im[k];
    if (row < 0) SETERRQ(1,"MatSetValues_MPIRowbs_local:Negative row");
    if (row >= mat->m) SETERRQ(1,"MatSetValues_MPIRowbs_local:Row too large");
    vs   = A->rows[row];
    ap   = vs->nz; rp = vs->col;
    rmax = imax[row]; nrow = vs->length;
    a    = 0;
    for ( l=0; l<n; l++ ) { /* loop over added columns */
      if (in[l] < 0) SETERRQ(1,"MatSetValues_MPIRowbs_local:Negative col");
      if (in[l] >= mat->N) SETERRQ(1,"MatSetValues_MPIRowbs_local:Col too large");
      col = in[l]; value = *v++;
      if (!sorted) a = 0; b = nrow;
      while (b-a > 5) {
        t = (b+a)/2;
        if (rp[t] > col) b = t;
        else             a = t;
      }
      for ( i=a; i<b; i++ ) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          if (addv == ADD_VALUES) ap[i] += value;
          else                    ap[i] = value;
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
        ierr = MatMallocRowbs_Private(AA,imax[row],&itemp,&vtemp);CHKERRQ(ierr);
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
          {ierr = MatFreeRowbs_Private(AA,rmax,vs->col,vs->nz); CHKERRQ(ierr);}
        vs->col           =  iout; vs->nz = vout;
        rmax              =  imax[row];
        mat->maxnz        += CHUNCKSIZE_LOCAL;
      }
      else {
      /* shift higher columns over to make room for newie */
        for ( ii=nrow-1; ii>=i; ii-- ) {
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
#include "pinclude/pviewer.h"

static int MatAssemblyBegin_MPIRowbs_local(Mat A,MatAssemblyType mode)
{ 
  return 0;
}

static int MatAssemblyEnd_MPIRowbs_local(Mat AA,MatAssemblyType mode)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs *) AA->data;
  BSspmat      *A = a->A;
  BSsprow      *vs;
  int          i, j, rstart = a->rstart;

  if (mode == FLUSH_ASSEMBLY) return 0;

  /* Mark location of diagonal */
  for ( i=0; i<a->m; i++ ) {
    vs = A->rows[i];
    for ( j=0; j<vs->length; j++ ) {
      if (vs->col[j] == i + rstart) {
        vs->diag_ind = j;
        break;
      }
    }
    if (vs->diag_ind == -1) { 
      SETERRQ(1,"MatAssemblyEnd_MPIRowbs_local: no diagonal entry");
    }
  }
  return 0;
}

static int MatZeroRows_MPIRowbs_local(Mat A,IS is,Scalar *diag)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs *) A->data;
  BSspmat      *l = a->A;
  int          i, ierr, N, *rz, m = a->m - 1;

  ierr = ISGetLocalSize(is,&N); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rz); CHKERRQ(ierr);
  if (diag) {
    for ( i=0; i<N; i++ ) {
      if (rz[i] < 0 || rz[i] > m) SETERRQ(1,"MatZeroRows_MPIRowbs_local:Out of range");
      if (l->rows[rz[i]]->length > 0) { /* in case row was completely empty */
        l->rows[rz[i]]->length = 1;
        l->rows[rz[i]]->nz[0] = *diag;
        l->rows[rz[i]]->col[0] = rz[i];
      }
      else {
        ierr = MatSetValues(A,1,&rz[i],1,&rz[i],diag,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  else {
    for ( i=0; i<N; i++ ) {
      if (rz[i] < 0 || rz[i] > m) SETERRQ(1,"MatZeroRows_MPIRowbs_local:Out of range");
      l->rows[rz[i]]->length = 0;
    }
  }
  ISRestoreIndices(is,&rz);
  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}

static int MatNorm_MPIRowbs_local(Mat A,NormType type,double *norm)
{
  Mat_MPIRowbs *mat = (Mat_MPIRowbs *) A->data;
  BSsprow      *vs, **rs;
  Scalar       *xv;
  double       sum = 0.0;
  int          *xi, nz, i, j;

  rs = mat->A->rows;
  if (type == NORM_FROBENIUS) {
    for (i=0; i<mat->m; i++ ) {
      vs = *rs++;
      nz = vs->length;
      xv = vs->nz;
      while (nz--) {
#if defined(PETSC_COMPLEX)
        sum += real(conj(*xv)*(*xv)); xv++;
#else
        sum += (*xv)*(*xv); xv++;
#endif
      }
    }
    *norm = sqrt(sum);
  }
  else if (type == NORM_1) { /* max column norm */
    double *tmp;
    tmp = (double *) PetscMalloc( mat->n*sizeof(double) ); CHKPTRQ(tmp);
    PetscMemzero(tmp,mat->n*sizeof(double));
    *norm = 0.0;
    for (i=0; i<mat->m; i++) {
      vs = *rs++;
      nz = vs->length;
      xi = vs->col;
      xv = vs->nz;
      while (nz--) {
        tmp[*xi] += PetscAbsScalar(*xv); 
        xi++; xv++;
      }
    }
    for ( j=0; j<mat->n; j++ ) {
      if (tmp[j] > *norm) *norm = tmp[j];
    }
    PetscFree(tmp);
  }
  else if (type == NORM_INFINITY) { /* max row norm */
    *norm = 0.0;
    for ( i=0; i<mat->m; i++ ) {
      vs = *rs++;
      nz = vs->length;
      xv = vs->nz;
      sum = 0.0;
      while (nz--) {
        sum += PetscAbsScalar(*xv); xv++;
      }
      if (sum > *norm) *norm = sum;
    }
  }
  else {
    SETERRQ(1,"MatNorm_MPIRowbs_local:No support for the two norm");
  }
  return 0;
}

/* ----------------------------------------------------------------- */

static int MatSetValues_MPIRowbs(Mat A,int m,int *im,int n,int *in,Scalar *v,InsertMode av)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs *) A->data;
  int          ierr, i, j, row, col, rstart = a->rstart, rend = a->rend;
  int          roworiented = a->roworiented;
  Scalar       *zeros;

  if (a->insertmode != NOT_SET_VALUES && a->insertmode != av) {
    SETERRQ(1,"MatSetValues_MPIRowbs:Cannot mix inserts and adds");
  }
  a->insertmode = av;
  /* Note:  There's no need to "unscale" the matrix, since scaling is
     confined to a->pA, and we're working with a->A here */
  for ( i=0; i<m; i++ ) {
    if (im[i] < 0) SETERRQ(1,"MatSetValues_MPIRowbs:Negative row");
    if (im[i] >= a->M) SETERRQ(1,"MatSetValues_MPIRowbs:Row too large");
    if (im[i] >= rstart && im[i] < rend) {
      row = im[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (in[j] < 0) SETERRQ(1,"MatSetValues_MPIRowbs:Negative column");
        if (in[j] >= a->N) SETERRQ(1,"MatSetValues_MPIRowbs:Col too large");
        if (in[j] >= 0 && in[j] < a->N){
          col = in[j];
          if (roworiented) {
            ierr = MatSetValues_MPIRowbs_local(A,1,&row,1,&col,v+i*n+j,av);CHKERRQ(ierr);
          }
          else {
            ierr = MatSetValues_MPIRowbs_local(A,1,&row,1,&col,v+i+j*m,av);CHKERRQ(ierr);
          }
        }
        else {SETERRQ(1,"MatSetValues_MPIRowbs:Invalid column");}
      }
    } 
    else {
      if (roworiented) {
        ierr = StashValues_Private(&a->stash,im[i],n,in,v+i*n,av);CHKERRQ(ierr);
      }
      else {
        row = im[i];
        for ( j=0; j<n; j++ ) {
          ierr = StashValues_Private(&a->stash,row,1,in+j,v+i+j*m,av);CHKERRQ(ierr);
        }
      }
    }
  }

  /*
     user has indicated that they are building a symmetric matrix and will 
     insert all of the values.
  */
  if (a->mat_is_structurally_symmetric) return 0;

  if (av == INSERT_VALUES) 
    SETERRQ(1,"MatSetValues_MPIRowbs:Not currently possible to insert values\n\
                 in a MPIRowbs matrix unless options SYMMETRIC_MATRIX or\n\
                 STRUCTURALLY_SYMMETRIC_MATRIX have been used.");

  /* The following code adds zeros to the symmetric counterpart (ILU) */
  /* this is only needed to insure that the matrix is structurally symmetric */
  /* while the user creating it may not make it structurally symmetric. */
  zeros = (Scalar *) PetscMalloc ((m+1)*sizeof(Scalar));
  for ( i=0; i<m; i++ ) zeros[i] = 0.0;
  for ( i=0; i<n; i++ ) {
    if (in[i] < 0) SETERRQ(1,"MatSetValues_MPIRowbs:Negative column");
    if (in[i] >= a->M) SETERRQ(1,"MatSetValues_MPIRowbs:Col too large");
    if (in[i] >= rstart && in[i] < rend) {
      row = in[i] - rstart;
      for ( j=0; j<m; j++ ) {
        if (im[j] < 0) SETERRQ(1,"MatSetValues_MPIRowbs:Negative row");
        if (im[j] >= a->N) SETERRQ(1,"MatSetValues_MPIRowbs:Row too large");
        if (im[j] >= 0 && im[j] < a->N){
          col = im[j];
          ierr = MatSetValues_MPIRowbs_local(A,1,&row,1,&col,zeros,ADD_VALUES);CHKERRQ(ierr);
        }
        else {SETERRQ(1,"MatSetValues_MPIRowbs:Invalid row");}
      }
    } 
    else {
      ierr = StashValues_Private(&a->stash,in[i],m,im,zeros,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  PetscFree(zeros);
  return 0;
}

static int MatAssemblyBegin_MPIRowbs(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIRowbs  *a = (Mat_MPIRowbs *) mat->data;
  MPI_Comm      comm = mat->comm;
  int           size = a->size, *owners = a->rowners,st,rank = a->rank;
  int           *nprocs,i,j,idx,*procs,nsends,nreceives,nmax,*work;
  int           tag = mat->tag, *owner,*starts,count,ierr,sn;
  MPI_Request   *send_waits,*recv_waits;
  InsertMode    addv;
  Scalar        *rvalues,*svalues;

  StashInfo_Private(&a->stash);
  /* Note:  There's no need to "unscale" the matrix, since scaling is
            confined to a->pA, and we're working with a->A here */

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  MPI_Allreduce(&a->insertmode,&addv,1,MPI_INT,MPI_BOR,comm);
  if (addv == (ADD_VALUES|INSERT_VALUES)) {
    SETERRQ(1,"MatAssemblyBegin_MPIRowbs:Some procs inserted; others added");
  }
  a->insertmode = addv; /* in case this processor had no cache */

  /*  first count number of contributors to each processor */
  nprocs = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(nprocs);
  PetscMemzero(nprocs,2*size*sizeof(int)); procs = nprocs + size;
  owner = (int *) PetscMalloc( (a->stash.n+1)*sizeof(int) ); CHKPTRQ(owner);
  for ( i=0; i<a->stash.n; i++ ) {
    idx = a->stash.idx[i];
    for ( j=0; j<size; j++ ) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; break;
      }
    }
  }
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce(procs,work,size,MPI_INT,MPI_SUM,comm);
  nreceives = work[rank]; 
  MPI_Allreduce(nprocs, work,size,MPI_INT,MPI_MAX,comm);
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
  rvalues = (Scalar *) PetscMalloc(3*(nreceives+1)*(nmax+1)*sizeof(Scalar));CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nreceives+1)*sizeof(MPI_Request));
  CHKPTRQ(recv_waits);
  for ( i=0; i<nreceives; i++ ) {
    MPI_Irecv(rvalues+3*nmax*i,3*nmax,MPIU_SCALAR,MPI_ANY_SOURCE,tag,comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (Scalar *) PetscMalloc( 3*(a->stash.n+1)*sizeof(Scalar) );CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PetscMalloc( (nsends+1)*sizeof(MPI_Request));
  CHKPTRQ(send_waits);
  starts = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(starts);
  starts[0] = 0; 
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  sn = a->stash.n;
  for ( i=0; i<sn; i++ ) {
    st            = 3*starts[owner[i]]++;
    svalues[st++] = (Scalar)  a->stash.idx[i];
    svalues[st++] = (Scalar)  a->stash.idy[i];
    svalues[st]   =  a->stash.array[i];
  }
  PetscFree(owner);
  starts[0] = 0;
  for ( i=1; i<size; i++ ) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for ( i=0; i<size; i++ ) {
    if (procs[i]) {
      MPI_Isend(svalues+3*starts[i],3*nprocs[i],MPIU_SCALAR,i,tag,comm,send_waits+count++);
    }
  }
  PetscFree(starts); PetscFree(nprocs);

  /* Free cache space */
  ierr = StashDestroy_Private(&a->stash); CHKERRQ(ierr);

  a->svalues    = svalues;    a->rvalues = rvalues;
  a->nsends     = nsends;     a->nrecvs = nreceives;
  a->send_waits = send_waits; a->recv_waits = recv_waits;
  a->rmax       = nmax;

  return 0;
}

#include "viewer.h"
#include "sysio.h"

static int MatView_MPIRowbs_ASCII(Mat mat,Viewer viewer)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs *) mat->data;
  int          ierr, format, i, j;
  FILE         *fd;

  ierr = ViewerFileGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerFileGetFormat_Private(viewer,&format); CHKERRQ(ierr);

  if (format == FILE_FORMAT_INFO) {
    int ind_l, ind_g, clq_l, clq_g, color;
    ind_l = BSlocal_num_inodes(a->pA); CHKERRBS(0);
    ind_g = BSglobal_num_inodes(a->pA); CHKERRBS(0);
    clq_l = BSlocal_num_cliques(a->pA); CHKERRBS(0);
    clq_g = BSglobal_num_cliques(a->pA); CHKERRBS(0);
    color = BSnum_colors(a->pA); CHKERRBS(0);
    MPIU_fprintf(mat->comm,fd,
     "  %d global inodes, %d global cliques, %d colors\n",ind_g,clq_g,color);
    MPIU_Seq_begin(mat->comm,1);
    fprintf(fd,"    [%d] %d local inodes, %d local cliques\n",a->rank,ind_l,clq_l);
    fflush(fd);
    MPIU_Seq_end(mat->comm,1);
  }
  else {
    BSspmat *A = a->A;
    BSsprow **rs = A->rows;
    MPIU_Seq_begin(mat->comm,1);
    fprintf(fd,"[%d] rows %d starts %d ends %d cols %d starts %d ends %d\n",
            a->rank,a->m,a->rstart,a->rend,a->n,0,a->N);
    for ( i=0; i<A->num_rows; i++ ) {
      fprintf(fd,"row %d:",i+a->rstart);
      for (j=0; j<rs[i]->length; j++) {
        fprintf(fd," %d %g ", rs[i]->col[j], rs[i]->nz[j]);
      }
      fprintf(fd,"\n");
    }
    fflush(fd);
    MPIU_Seq_end(mat->comm,1);
  }
  return 0;
}

static int MatView_MPIRowbs_Binary(Mat mat,Viewer viewer)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs *) mat->data;
  int          ierr,i,M,m,rank,size,*sbuff,*rowlengths;
  int          *recvcts,*recvdisp,fd,*cols,maxnz,nz,j,totalnz,dummy;
  BSspmat      *A = a->A;
  BSsprow      **rs = A->rows;
  MPI_Comm     comm = mat->comm;
  MPI_Status   status;
  Scalar       *vals;

  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);

  M = a->M; m = a->m;
  /* First gather together on the first processor the lengths of 
     each row, and write them out to the file */
  sbuff = (int*) PetscMalloc( m*sizeof(int) ); CHKPTRQ(sbuff);
  for ( i=0; i<A->num_rows; i++ ) {
    sbuff[i] = rs[i]->length;
  }
  MatGetInfo(mat,MAT_GLOBAL_SUM,&totalnz,&dummy,&dummy);
  if (!rank) {
    ierr = ViewerFileGetDescriptor_Private(viewer,&fd); CHKERRQ(ierr);
    rowlengths = (int*) PetscMalloc( (4+M)*sizeof(int) ); CHKPTRQ(rowlengths);
    recvcts = (int*) PetscMalloc( size*sizeof(int) ); CHKPTRQ(recvcts);
    recvdisp = a->rowners;
    for ( i=0; i<size; i++ ) {
      recvcts[i] = recvdisp[i+1] - recvdisp[i];
    }
    /* first four elements of rowlength are the header */
    rowlengths[0] = mat->cookie;
    rowlengths[1] = a->M;
    rowlengths[2] = a->N;
    rowlengths[3] = totalnz;
    MPI_Gatherv(sbuff,m,MPI_INT,rowlengths+4,recvcts,recvdisp,MPI_INT,0,comm);
    PetscFree(sbuff);
    ierr = SYWrite(fd,rowlengths,4+M,SYINT,0); CHKERRQ(ierr);
    /* count the number of nonzeros on each processor */
    PetscMemzero(recvcts,size*sizeof(int));
    for ( i=0; i<size; i++ ) {
      for ( j=recvdisp[i]; j<recvdisp[i+1]; j++ ) {
        recvcts[i] += rowlengths[j+3];
      }
    }
    /* allocate buffer long enough to hold largest one */
    maxnz = 0;
    for ( i=0; i<size; i++ ) {
      maxnz = PetscMax(maxnz,recvcts[i]);
    }
    PetscFree(rowlengths); PetscFree(recvcts);
    cols = (int*) PetscMalloc( maxnz*sizeof(int) ); CHKPTRQ(cols);

    /* binary store column indices for 0th processor */
    nz = 0;
    for ( i=0; i<A->num_rows; i++ ) {
      for (j=0; j<rs[i]->length; j++) {
        cols[nz++] = rs[i]->col[j];
      }
    }
    ierr = SYWrite(fd,cols,nz,SYINT,0); CHKERRQ(ierr);

    /* receive and store column indices for all other processors */
    for ( i=1; i<size; i++ ) {
      /* should tell processor that I am now ready and to begin the send */
      MPI_Recv(cols,maxnz,MPI_INT,i,mat->tag,comm,&status);
      MPI_Get_count(&status,MPI_INT,&nz);
      ierr = SYWrite(fd,cols,nz,SYINT,0); CHKERRQ(ierr);
    }
    PetscFree(cols);
    vals = (Scalar*) PetscMalloc( maxnz*sizeof(Scalar) ); CHKPTRQ(vals);

    /* binary store values for 0th processor */
    nz = 0;
    for ( i=0; i<A->num_rows; i++ ) {
      for (j=0; j<rs[i]->length; j++) {
        vals[nz++] = rs[i]->nz[j];
      }
    }
    ierr = SYWrite(fd,vals,nz,SYSCALAR,0); CHKERRQ(ierr);

    /* receive and store nonzeros for all other processors */
    for ( i=1; i<size; i++ ) {
      /* should tell processor that I am now ready and to begin the send */
      MPI_Recv(vals,maxnz,MPIU_SCALAR,i,mat->tag,comm,&status);
      MPI_Get_count(&status,MPIU_SCALAR,&nz);
      ierr = SYWrite(fd,vals,nz,SYSCALAR,0);CHKERRQ(ierr);
    }
    PetscFree(vals);
  }
  else {
    MPI_Gatherv(sbuff,m,MPI_INT,0,0,0,MPI_INT,0,comm);
    PetscFree(sbuff);

    /* count local nonzeros */
    nz = 0;
    for ( i=0; i<A->num_rows; i++ ) {
      for (j=0; j<rs[i]->length; j++) {
        nz++;
      }
    }
    /* copy into buffer column indices */
    cols = (int*) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(cols);
    nz = 0;
    for ( i=0; i<A->num_rows; i++ ) {
      for (j=0; j<rs[i]->length; j++) {
        cols[nz++] = rs[i]->col[j];
      }
    }
    /* send */  /* should wait until processor zero tells me to go */
    MPI_Send(cols,nz,MPI_INT,0,mat->tag,comm);
    PetscFree(cols);

    /* copy into buffer column values */
    vals = (Scalar*) PetscMalloc( nz*sizeof(Scalar) ); CHKPTRQ(vals);
    nz = 0;
    for ( i=0; i<A->num_rows; i++ ) {
      for (j=0; j<rs[i]->length; j++) {
        vals[nz++] = rs[i]->nz[j];
      }
    }
    /* send */  /* should wait until processor zero tells me to go */
    MPI_Send(vals,nz,MPIU_SCALAR,0,mat->tag,comm);
    PetscFree(vals);
  }

  return 0;
}

static int MatView_MPIRowbs(PetscObject obj,Viewer viewer)
{
  Mat          mat = (Mat) obj;
  PetscObject  vobj = (PetscObject) viewer;

  if (!viewer) { 
    viewer = STDOUT_VIEWER_SELF; vobj = (PetscObject) viewer;
  }

  if (vobj->cookie == DRAW_COOKIE) {
    if (vobj->type == NULLWINDOW) return 0;
  }
  else if (vobj->cookie == VIEWER_COOKIE) {
    if (vobj->type == ASCII_FILE_VIEWER || vobj->type == ASCII_FILES_VIEWER) {
      return MatView_MPIRowbs_ASCII(mat,viewer);
    }
    else if (vobj->type == BINARY_FILE_VIEWER) {
      return MatView_MPIRowbs_Binary(mat,viewer);
    }
  }
  return 0;
}

static int MatAssemblyEnd_MPIRowbs(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIRowbs *a = (Mat_MPIRowbs *) mat->data;
  MPI_Status   *send_status,recv_status;
  int          imdex,nrecvs = a->nrecvs, count = nrecvs, i, n;
  int          ldim, low, high, row, col, ierr;
  Scalar       *values, val, *diag;
  InsertMode   addv = a->insertmode;

  /*  wait on receives */
  while (count) {
    MPI_Waitany(nrecvs,a->recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    values = a->rvalues + 3*imdex*a->rmax;
    MPI_Get_count(&recv_status,MPIU_SCALAR,&n);
    n = n/3;
    for ( i=0; i<n; i++ ) {
      row = (int) PetscReal(values[3*i]) - a->rstart;
      col = (int) PetscReal(values[3*i+1]);
      val = values[3*i+2];
      if (col >= 0 && col < a->N) {
        MatSetValues_MPIRowbs_local(mat,1,&row,1,&col,&val,addv);
      } 
      else {SETERRQ(1,"MatAssemblyEnd_MPIRowbs:Invalid column");}
    }
    count--;
  }
  PetscFree(a->recv_waits); PetscFree(a->rvalues);
 
  /* wait on sends */
  if (a->nsends) {
    send_status = (MPI_Status *) PetscMalloc( a->nsends*sizeof(MPI_Status) );
    CHKPTRQ(send_status);
    MPI_Waitall(a->nsends,a->send_waits,send_status);
    PetscFree(send_status);
  }
  PetscFree(a->send_waits); PetscFree(a->svalues);

  a->insertmode = NOT_SET_VALUES;
  ierr = MatAssemblyBegin_MPIRowbs_local(mat,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd_MPIRowbs_local(mat,mode); CHKERRQ(ierr);

  if (mode == FINAL_ASSEMBLY) {   /* BlockSolve stuff */
    if ((mat->was_assembled) && (!a->nonew)) {  /* Free the old info */
      if (a->pA)       {BSfree_par_mat(a->pA);   CHKERRBS(0);}
      if (a->comm_pA)  {BSfree_comm(a->comm_pA); CHKERRBS(0);} 
    }
    if ((!a->nonew) || (!mat->was_assembled)) {
      /* Form permuted matrix for efficient parallel execution */
      a->pA = BSmain_perm(a->procinfo,a->A); CHKERRBS(0);
      /* Set up the communication */
      a->comm_pA = BSsetup_forward(a->pA,a->procinfo); CHKERRBS(0);
    } else {
      /* Repermute the matrix */
      BSmain_reperm(a->procinfo,a->A,a->pA); CHKERRBS(0);
    }

    /* Symmetrically scale the matrix by the diagonal */
    BSscale_diag(a->pA,a->pA->diag,a->procinfo); CHKERRBS(0);

    /* Store inverse of square root of permuted diagonal scaling matrix */
    ierr = VecGetLocalSize( a->diag, &ldim ); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange( a->diag, &low, &high ); CHKERRQ(ierr);
    ierr = VecGetArray(a->diag,&diag); CHKERRQ(ierr);
    for (i=0; i<ldim; i++) {
      if (a->pA->scale_diag[i] != 0.0) {
        diag[i] = 1.0/sqrt(PetscAbsScalar(a->pA->scale_diag[i]));
      }
      else {
        diag[i] = 1.0;
      }   
    }
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

/* the code does not do the diagonal entries correctly unless the 
   matrix is square and the column and row owerships are identical.
   This is a BUG.
*/

static int MatZeroRows_MPIRowbs(Mat A,IS is,Scalar *diag)
{
  Mat_MPIRowbs   *l = (Mat_MPIRowbs *) A->data;
  int            i,ierr,N, *rows,*owners = l->rowners,size = l->size;
  int            *procs,*nprocs,j,found,idx,nsends,*work;
  int            nmax,*svalues,*starts,*owner,nrecvs,rank = l->rank;
  int            *rvalues,tag = A->tag,count,base,slen,n,*source;
  int            *lens,imdex,*lrows,*values;
  MPI_Comm       comm = A->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
  IS             istmp;

  ierr = ISGetLocalSize(is,&N); CHKERRQ(ierr);
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
    if (!found) SETERRQ(1,"MatZeroRows_MPIRowbs:Row out of range");
  }
  nsends = 0;  for ( i=0; i<size; i++ ) {nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  MPI_Allreduce(procs, work,size,MPI_INT,MPI_SUM,comm);
  nrecvs = work[rank]; 
  MPI_Allreduce( nprocs, work,size,MPI_INT,MPI_MAX,comm);
  nmax = work[rank];
  PetscFree(work);

  /* post receives:   */
  rvalues = (int *) PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int)); CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nrecvs+1)*sizeof(MPI_Request));CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    MPI_Irecv(rvalues+nmax*i,nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (int *) PetscMalloc( (N+1)*sizeof(int) ); CHKPTRQ(svalues);
  send_waits = (MPI_Request *)PetscMalloc((nsends+1)*sizeof(MPI_Request));CHKPTRQ(send_waits);
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
  lens = (int *) PetscMalloc( 2*(nrecvs+1)*sizeof(int) ); CHKPTRQ(lens);
  source = lens + nrecvs;
  count = nrecvs; slen = 0;
  while (count) {
    MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);
    /* unpack receives into our local space */
    MPI_Get_count(&recv_status,MPI_INT,&n);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]    = n;
    slen           += n;
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
  ierr = ISCreateSeq(MPI_COMM_SELF,slen,lrows,&istmp); CHKERRQ(ierr);  
  PLogObjectParent(A,istmp);
  PetscFree(lrows);
  ierr = MatZeroRows_MPIRowbs_local(A,istmp,diag); CHKERRQ(ierr);
  ierr = ISDestroy(istmp); CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status *) PetscMalloc(nsends*sizeof(MPI_Status));CHKPTRQ(send_status);
    MPI_Waitall(nsends,send_waits,send_status);
    PetscFree(send_status);
  }
  PetscFree(send_waits); PetscFree(svalues);

  return 0;
}

static int MatNorm_MPIRowbs(Mat mat,NormType type,double *norm)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs *) mat->data;
  int          ierr;
  if (a->size == 1) {
    ierr = MatNorm_MPIRowbs_local(mat,type,norm); CHKERRQ(ierr);
  } else 
    SETERRQ(1,"MatNorm_MPIRowbs:Not supported in parallel");
  return 0; 
}

static int MatMult_MPIRowbs(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIRowbs *bsif = (Mat_MPIRowbs *) mat->data;
  BSprocinfo   *bspinfo = bsif->procinfo;
  Scalar       *xxa, *xworka, *yya;
  int          ierr;

  ierr = VecGetArray(yy,&yya); CHKERRQ(ierr);
  ierr = VecGetArray(xx,&xxa); CHKERRQ(ierr);

  /* Permute and apply diagonal scaling:  [ xwork = D^{1/2} * x ] */
  if (!bsif->vecs_permscale) {
    ierr = VecGetArray(bsif->xwork,&xworka); CHKERRQ(ierr);
    BSperm_dvec(xxa,xworka,bsif->pA->perm); CHKERRBS(0);
    ierr = VecPDiv(bsif->xwork,bsif->diag,xx); CHKERRQ(ierr);
  } 

  /* Do lower triangular multiplication:  [ y = L * xwork ] */
  if (bspinfo->single)
    BSforward1( bsif->pA, xxa, yya, bsif->comm_pA, bspinfo );
  else
    BSforward( bsif->pA, xxa, yya, bsif->comm_pA, bspinfo );
  CHKERRBS(0);

  /* Do upper triangular multiplication:  [ y = y + L^{T} * xwork ] */
  if (bsif->mat_is_symmetric) {
    if (bspinfo->single)
      BSbackward1( bsif->pA, xxa, yya, bsif->comm_pA, bspinfo );
    else
      BSbackward( bsif->pA, xxa, yya, bsif->comm_pA, bspinfo );
    CHKERRBS(0);
  }
  /* not needed for ILU version since forward does it all */

  /* Apply diagonal scaling to vector:  [  y = D^{1/2} * y ] */
  if (!bsif->vecs_permscale) {
    BSiperm_dvec(xworka,xxa,bsif->pA->perm); CHKERRBS(0);
    ierr = VecPDiv(yy,bsif->diag,bsif->xwork); CHKERRQ(ierr);
    BSiperm_dvec(xworka,yya,bsif->pA->perm); CHKERRBS(0);
  }
  PLogFlops(2*bsif->nz - bsif->m);

  return 0;
}

static int MatRelax_MPIRowbs(Mat mat,Vec bb,double omega,MatSORType flag,
                             double shift,int its,Vec xx)
{
  SETERRQ(1,"MatRelax_MPIRowbs:Not done");
/* None of the relaxation code is finished now! */

/*  Mat_MPIRowbs *bsif = (Mat_MPIRowbs *) mat->data;
  Scalar *b;
  int ierr;


  if (flag & SOR_FORWARD_SWEEP) {
    if (bsif->procinfo->single) {
      BSfor_solve1(bsif->pA,b,bsif->comm_pA,bsif->procinfo); CHKERRBS(0);
    } else {
      BSfor_solve(bsif->pA,b,bsif->comm_pA,bsif->procinfo); CHKERRBS(0);
    }
  }
  if (flag & SOR_BACKWARD_SWEEP) {
    if (bsif->procinfo->single) {
      BSback_solve1(bsif->pA,b,bsif->comm_pA,bsif->procinfo); CHKERRBS(0);
    } else {
      BSback_solve(bsif->pA,b,bsif->comm_pA,bsif->procinfo); CHKERRBS(0);
    }
  }
  ierr = VecCopy(bb,xx); CHKERRQ(ierr);
  return 0; */
}

static int MatGetInfo_MPIRowbs(Mat A,MatInfoType flag,int *nz,int *nzalloc,int *mem)
{
  Mat_MPIRowbs *mat = (Mat_MPIRowbs *) A->data;
  int          isend[3], irecv[3];

  isend[0] = mat->nz; isend[1] = mat->maxnz; isend[2] = A->mem;
  if (flag == MAT_LOCAL) {
    *nz = isend[0]; *nzalloc = isend[1]; *mem = isend[2];
  } else if (flag == MAT_GLOBAL_MAX) {
    MPI_Allreduce( isend,irecv,3,MPI_INT,MPI_MAX,A->comm);
    *nz = irecv[0]; *nzalloc = irecv[1]; *mem = irecv[2];
  } else if (flag == MAT_GLOBAL_SUM) {
    MPI_Allreduce(isend,irecv,3,MPI_INT,MPI_SUM,A->comm);
    *nz = irecv[0]; *nzalloc = irecv[1]; *mem = irecv[2];
  }
  return 0;
}

static int MatGetDiagonal_MPIRowbs(Mat mat,Vec v)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs *) mat->data;
  BSsprow      **rs = a->A->rows;
  int          i, n;
  Scalar       *x, zero = 0.0, *scale = a->pA->scale_diag;

  VecSet(&zero,v);
  VecGetArray(v,&x); VecGetLocalSize(v,&n);
  if (n != a->m) SETERRQ(1,"MatGetDiag_MPIRowbs:Nonconforming mat and vec");
  if (a->vecs_permscale) {
    for ( i=0; i<a->m; i++ ) {
      x[i] = rs[i]->nz[rs[i]->diag_ind];
    }
  } else {
    for ( i=0; i<a->m; i++ ) {
      x[i] = rs[i]->nz[rs[i]->diag_ind] * scale[i]; 
    }
  }
  return 0;
}

static int MatDestroy_MPIRowbs(PetscObject obj)
{
  Mat          mat = (Mat) obj;
  Mat_MPIRowbs *a = (Mat_MPIRowbs *) mat->data;
  BSspmat      *A = a->A;
  BSsprow      *vs;
  int          i, ierr;

  if (a->fact_clone) {
    a->fact_clone = 0;
    return 0;
  }
#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d",a->M,a->N);
#endif
  PetscFree(a->rowners); 

  if (a->bsmap) {
      if (a->bsmap->vlocal2global) PetscFree(a->bsmap->vlocal2global);
      if (a->bsmap->vglobal2local) PetscFree(a->bsmap->vglobal2local);
      if (a->bsmap->vglobal2proc)  (*a->bsmap->free_g2p)(a->bsmap->vglobal2proc);
      PetscFree(a->bsmap);
  } 

  PLogObjectDestroy(mat);
  if (A) {
    for (i=0; i<a->m; i++) {
      vs = A->rows[i];
      ierr = MatFreeRowbs_Private(mat,vs->length,vs->col,vs->nz); CHKERRQ(ierr);
    }
    /* Note: A->map = a->bsmap is freed above */
    PetscFree(A->rows);
    PetscFree(A);
  }
  if (a->procinfo) {BSfree_ctx(a->procinfo); CHKERRBS(0);}
  if (a->diag)     {ierr = VecDestroy(a->diag); CHKERRQ(ierr);}
  if (a->xwork)    {ierr = VecDestroy(a->xwork); CHKERRQ(ierr);}
  if (a->pA)       {BSfree_par_mat(a->pA); CHKERRBS(0);}
  if (a->fpA)      {BSfree_copy_par_mat(a->fpA); CHKERRBS(0);}
  if (a->comm_pA)  {BSfree_comm(a->comm_pA); CHKERRBS(0);}
  if (a->comm_fpA) {BSfree_comm(a->comm_fpA); CHKERRBS(0);}
  if (a->imax)     PetscFree(a->imax);    

  PetscFree(a);  
  PetscHeaderDestroy(mat);
  return 0;
}

static int MatSetOption_MPIRowbs(Mat A,MatOption op)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs *) A->data;

  if      (op == ROW_ORIENTED)              a->roworiented = 1;
  else if (op == COLUMN_ORIENTED)           a->roworiented = 0; 
  else if (op == COLUMNS_SORTED)            a->sorted      = 1;
  else if (op == NO_NEW_NONZERO_LOCATIONS)  a->nonew       = 1;
  else if (op == YES_NEW_NONZERO_LOCATIONS) a->nonew       = 0;
  else if (op == SYMMETRIC_MATRIX) {
    BSset_mat_symmetric(a->A,PETSC_TRUE);
    BSset_mat_icc_storage(a->A,PETSC_TRUE);
    a->mat_is_symmetric = 1;
    a->mat_is_structurally_symmetric = 1;
  }
  else if (op == STRUCTURALLY_SYMMETRIC_MATRIX) {
    a->mat_is_structurally_symmetric = 1;
  }
  else if (op == YES_NEW_DIAGONALS)
    PLogInfo((PetscObject)A,"Info:MatSetOption_MPIRowbs:Option ignored\n");
  else if (op == COLUMN_ORIENTED) 
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_MPIRowbs:COLUMN_ORIENTED");}
  else if (op == NO_NEW_DIAGONALS)
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_MPIRowbs:NO_NEW_DIAGONALS");}
  else 
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_MPIRowbs:unknown option");}
  return 0;
}

static int MatGetSize_MPIRowbs(Mat mat,int *m,int *n)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs *) mat->data;
  *m = a->M; *n = a->N;
  return 0;
}

static int MatGetLocalSize_MPIRowbs(Mat mat,int *m,int *n)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs *) mat->data;
  *m = a->m; *n = a->N;
  return 0;
}

static int MatGetOwnershipRange_MPIRowbs(Mat A,int *m,int *n)
{
  Mat_MPIRowbs *mat = (Mat_MPIRowbs *) A->data;
  *m = mat->rstart; *n = mat->rend;
  return 0;
}

static int MatGetRow_MPIRowbs(Mat AA,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIRowbs *mat = (Mat_MPIRowbs *) AA->data;
  BSspmat      *A = mat->A;
  BSsprow      *rs;
   
  if (row < mat->rstart || row >= mat->rend) SETERRQ(1,"MatGetRow_MPIRowbs:Only local rows");

  rs  = A->rows[row - mat->rstart];
  *nz = rs->length;
  if (v)   *v   = rs->nz;
  if (idx) *idx = rs->col;
  return 0;
}

static int MatRestoreRow_MPIRowbs(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  return 0;
}

/* ------------------------------------------------------------------ */

int MatConvert_MPIRowbs(Mat A, MatType newtype, Mat *newmat)
{
  Mat_MPIRowbs *row = (Mat_MPIRowbs *) A->data;
  int          ierr, nz, i, ig,rstart = row->rstart, m = row->m, *cwork, size;
  Scalar       *vwork;

  switch (newtype) {
    case MATMPIAIJ:
      ierr = MatCreateMPIAIJ(A->comm,m,row->n,row->M,row->N,0,PETSC_NULL,0,
             PETSC_NULL,newmat); CHKERRQ(ierr);
      break;
    case MATSEQAIJ:
      MPI_Comm_size(A->comm,&size);
      if (size != 1) SETERRQ(1,"MatConvert_MPIRowbs: SEQAIJ requires 1 proc");
      ierr = MatCreateSeqAIJ(A->comm,row->M,row->N,0,PETSC_NULL,newmat); CHKERRQ(ierr);
      break;
    default:
      SETERRQ(1,"MatConvert_MPIRowbs:Matrix format not yet supported");
  }
  /* Each processor converts its local rows */
  for (i=0; i<m; i++) {
    ig   = i + rstart;
    ierr = MatGetRow(A,ig,&nz,&cwork,&vwork);	CHKERRQ(ierr);
    ierr = MatSetValues(*newmat,1,&ig,nz,cwork,vwork,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,ig,&nz,&cwork,&vwork); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*newmat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*newmat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}

static int MatPrintHelp_MPIRowbs(Mat A)
{
  static int called = 0; 
  MPI_Comm   comm = A->comm;

  if (called) return 0; else called = 1;
  MPIU_printf(comm," Options for MATMPIROWBS matrix format (needed for BlockSolve):\n");
  MPIU_printf(comm,"  -mat_rowbs_no_inode  - Do not use inodes\n");
  return 0;
}

/* -------------------------------------------------------------------*/
extern int MatCholeskyFactorNumeric_MPIRowbs(Mat,Mat*);
extern int MatIncompleteCholeskyFactorSymbolic_MPIRowbs(Mat,IS,double,int,Mat *);
extern int MatLUFactorNumeric_MPIRowbs(Mat,Mat*);
extern int MatILUFactorSymbolic_MPIRowbs(Mat,IS,IS,double,int,Mat *);
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
       MatGetDiagonal_MPIRowbs,0,MatNorm_MPIRowbs,
       MatAssemblyBegin_MPIRowbs,MatAssemblyEnd_MPIRowbs,
       0,
       MatSetOption_MPIRowbs,MatZeroEntries_MPIRowbs,MatZeroRows_MPIRowbs,0,
       0,MatLUFactorNumeric_MPIRowbs,0,MatCholeskyFactorNumeric_MPIRowbs,
       MatGetSize_MPIRowbs,MatGetLocalSize_MPIRowbs,
       MatGetOwnershipRange_MPIRowbs,
       MatILUFactorSymbolic_MPIRowbs,
       MatIncompleteCholeskyFactorSymbolic_MPIRowbs,
       0,0,MatConvert_MPIRowbs,
       0,0,0,MatForwardSolve_MPIRowbs,MatBackwardSolve_MPIRowbs,
       0,0,0,
       0,0,0,0,MatPrintHelp_MPIRowbs};

/* ------------------------------------------------------------------- */

/*@C
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

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   Specify the preallocated storage with either nz or nnz (not both).  Set 
   nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.

   Options Database Keys:
$    -mat_rowbs_no_inode  - Do not use inodes.
  
.keywords: matrix, row, symmetric, sparse, parallel, BlockSolve

.seealso: MatCreate(), MatSetValues()
@*/
int MatCreateMPIRowbs(MPI_Comm comm,int m,int M,int nz,int *nnz,void *procinfo,Mat *newA)
{
  Mat          A;
  Mat_MPIRowbs *a;
  BSmapping    *bsmap;
  BSoff_map    *bsoff;
  int          i, ierr, Mtemp, *offset, low, high,flg1,flg2,flg3;
  BSprocinfo   *bspinfo = (BSprocinfo *) procinfo;
  
  *newA = 0;

  if (m == PETSC_DECIDE && nnz) 
    SETERRQ(1,"MatCreateMPIRowbs:Cannot have PETSc decide rows but set nnz");

  PetscHeaderCreate(A,_Mat,MAT_COOKIE,MATMPIROWBS,comm);
  PLogObjectCreate(A);
  PLogObjectMemory(A,sizeof(struct _Mat));

  A->data = (void *) (a = PetscNew(Mat_MPIRowbs)); CHKPTRQ(a);
  PetscMemcpy(&A->ops,&MatOps,sizeof(struct _MatOps));
  A->destroy	      = MatDestroy_MPIRowbs;
  A->view	      = MatView_MPIRowbs;
  A->factor	      = 0;
  a->fact_clone       = 0;
  a->vecs_permscale   = 0;
  a->insertmode       = NOT_SET_VALUES;
  MPI_Comm_rank(comm,&a->rank);
  MPI_Comm_size(comm,&a->size);

  if (M != PETSC_DECIDE && m != PETSC_DECIDE) {
    /* Perhaps should be removed for better efficiency -- but could be risky. */
    MPI_Allreduce(&m,&Mtemp,1,MPI_INT,MPI_SUM,comm);
    if (Mtemp != M)
      SETERRQ(1,"MatCreateMPIRowbs:Sum of local dimensions!=global dimension");
  } else if (M == PETSC_DECIDE) {
    MPI_Allreduce(&m,&M,1,MPI_INT,MPI_SUM,comm);
  } else if (m == PETSC_DECIDE) {
    {m = M/a->size + ((M % a->size) > a->rank);}
  } else {
    SETERRQ(1,"MatCreateMPIRowbs:Must set local and/or global matrix size");
  }
  a->N    = M;
  a->M    = M;
  a->m    = m;
  a->n    = a->N; /* each row stores all columns */
  a->imax = (int *) PetscMalloc( (a->m+1)*sizeof(int) );CHKPTRQ(a->imax);
  a->mat_is_symmetric = 0;
  a->mat_is_structurally_symmetric = 0;

  /* build local table of row ownerships */
  a->rowners = (int *) PetscMalloc((a->size+2)*sizeof(int));CHKPTRQ(a->rowners);
  MPI_Allgather(&m,1,MPI_INT,a->rowners+1,1,MPI_INT,comm);
  a->rowners[0] = 0;
  for ( i=2; i<=a->size; i++ ) {
    a->rowners[i] += a->rowners[i-1];
  }
  a->rstart = a->rowners[a->rank]; 
  a->rend   = a->rowners[a->rank+1]; 
  PLogObjectMemory(A,(a->m+a->size+3)*sizeof(int));

  /* build cache for off array entries formed */
  ierr = StashBuild_Private(&a->stash); CHKERRQ(ierr);

  /* Initialize BlockSolve information */
  a->A	      = 0;
  a->pA	      = 0;
  a->comm_pA  = 0;
  a->fpA      = 0;
  a->comm_fpA = 0;
  a->alpha    = 1.0;
  a->ierr     = 0;
  a->failures = 0;
  ierr = VecCreateMPI(A->comm,a->m,a->M,&(a->diag)); CHKERRQ(ierr);
  ierr = VecDuplicate(a->diag,&(a->xwork));CHKERRQ(ierr);
  PLogObjectParent(A,a->diag);  PLogObjectParent(A,a->xwork);
  PLogObjectMemory(A,(a->m+1)*sizeof(Scalar));
  if (!bspinfo) {bspinfo = BScreate_ctx(); CHKERRBS(0);}
  a->procinfo = bspinfo;
  BSctx_set_id(bspinfo,a->rank); CHKERRBS(0);
  BSctx_set_np(bspinfo,a->size); CHKERRBS(0);
  BSctx_set_ps(bspinfo,(ProcSet*)comm); CHKERRBS(0);
  BSctx_set_cs(bspinfo,INT_MAX); CHKERRBS(0);
  BSctx_set_is(bspinfo,INT_MAX); CHKERRBS(0);
  BSctx_set_ct(bspinfo,IDO); CHKERRBS(0);
#if defined(PETSC_DEBUG)
  BSctx_set_err(bspinfo,1); CHKERRBS(0);  /* BS error checking */
#endif
  BSctx_set_rt(bspinfo,1); CHKERRBS(0);
  ierr = OptionsHasName(PETSC_NULL,"-info",&flg1); CHKERRQ(ierr);
  if (flg1) {
    BSctx_set_pr(bspinfo,1); CHKERRBS(0);
  }
  ierr = OptionsHasName(PETSC_NULL,"-pc_ilu_factorpointwise",&flg1); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-pc_icc_factorpointwise",&flg2); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-mat_rowbs_no_inode",&flg3); CHKERRQ(ierr);
  if (flg1 || flg2 || flg3) {
    BSctx_set_si(bspinfo,1); CHKERRBS(0);
  } else {
    BSctx_set_si(bspinfo,0); CHKERRBS(0);
  }
#if defined(PETSC_LOG)
  MLOG_INIT();  /* Initialize logging */
#endif

  /* Compute global offsets */
  ierr = MatGetOwnershipRange(A,&low,&high); CHKERRQ(ierr);
  offset = &low;

  a->bsmap = (void *) PetscNew(BSmapping); CHKPTRQ(a->bsmap);
  PLogObjectMemory(A,sizeof(BSmapping));
  bsmap = a->bsmap;
  bsmap->vlocal2global	= (int *) PetscMalloc(sizeof(int)); 
  CHKPTRQ(bsmap->vlocal2global);
  *((int *)bsmap->vlocal2global) = (*offset);
  bsmap->flocal2global	= BSloc2glob;
  bsmap->free_l2g	= 0;
  bsmap->vglobal2local	= (int *) PetscMalloc(sizeof(int)); 
  CHKPTRQ(bsmap->vglobal2local);
  *((int *)bsmap->vglobal2local) = (*offset);
  bsmap->fglobal2local	= BSglob2loc;
  bsmap->free_g2l	= 0;
  bsoff = BSmake_off_map( *offset, bspinfo, a->M );
  bsmap->vglobal2proc	= (void *)bsoff;
  bsmap->fglobal2proc	= BSglob2proc;
  bsmap->free_g2p	= BSfree_off_map;

  ierr = MatCreateMPIRowbs_local(A,nz,nnz); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg1); CHKERRQ(ierr);
  if (flg1) {
    ierr = MatPrintHelp(A); CHKERRQ(ierr);
  }
  *newA = A;
  return 0;
}
/* --------------- extra BlockSolve-specific routines -------------- */
/* @
  MatGetBSProcinfo - Gets the BlockSolve BSprocinfo context, which the
  user can then manipulate to alter the default parameters.

  Input Parameter:
  mat - matrix

  Output Parameter:
  procinfo - processor information context

  Note:
  This routine is valid only for matrices stored in the MATMPIROWBS
  format.
@ */
int MatGetBSProcinfo(Mat mat,BSprocinfo *procinfo)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs *) mat->data;
  if (mat->type != MATMPIROWBS) SETERRQ(1,"MatGetBSProcinfo:For MATMPIROWBS matrix type");
  procinfo = a->procinfo;
  return 0;
}

int MatLoad_MPIRowbs(Viewer bview,MatType type,Mat *newmat)
{
  Mat_MPIRowbs *a;
  BSspmat      *A;
  BSsprow      **rs;
  Mat          mat;
  int          i, nz, ierr, j,rstart, rend, fd, *ourlens,*sndcounts = 0,*procsnz;
  Scalar       *vals;
  PetscObject  vobj = (PetscObject) bview;
  MPI_Comm     comm = vobj->comm;
  MPI_Status   status;
  int          header[4],rank,size,*rowlengths = 0,M,N,m,*rowners,maxnz,*cols;

  MPI_Comm_size(comm,&size); MPI_Comm_rank(comm,&rank);
  if (!rank) {
    ierr = ViewerFileGetDescriptor_Private(bview,&fd); CHKERRQ(ierr);
    ierr = SYRead(fd,(char *)header,4,SYINT); CHKERRQ(ierr);
    if (header[0] != MAT_COOKIE) SETERRQ(1,"MatLoad_MPIRowbs: not matrix object");
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
  ourlens = (int*) PetscMalloc( (rend-rstart)*sizeof(int) ); CHKPTRQ(ourlens);
  if (!rank) {
    rowlengths = (int*) PetscMalloc( M*sizeof(int) ); CHKPTRQ(rowlengths);
    ierr = SYRead(fd,rowlengths,M,SYINT); CHKERRQ(ierr);
    sndcounts = (int*) PetscMalloc( size*sizeof(int) ); CHKPTRQ(sndcounts);
    for ( i=0; i<size; i++ ) sndcounts[i] = rowners[i+1] - rowners[i];
    MPI_Scatterv(rowlengths,sndcounts,rowners,MPI_INT,ourlens,rend-rstart,MPI_INT,0,comm);
    PetscFree(sndcounts);
  }
  else {
    MPI_Scatterv(0,0,0,MPI_INT,ourlens,rend-rstart,MPI_INT, 0,comm);
  }

  /* create our matrix */
  ierr = MatCreateMPIRowbs(comm,m,M,0,ourlens,PETSC_NULL,newmat); CHKERRQ(ierr);
  mat = *newmat;
  PetscFree(ourlens);

  a = (Mat_MPIRowbs *) mat->data;
  A = a->A;
  rs = A->rows;

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
    ierr = SYRead(fd,cols,nz,SYINT); CHKERRQ(ierr);
    
    /* insert it into my part of matrix */
    nz = 0;
    for ( i=0; i<A->num_rows; i++ ) {
      for (j=0; j<a->imax[i]; j++) {
        rs[i]->col[j] = cols[nz++];
      }
      rs[i]->length = a->imax[i];
    }
    /* read in parts for all other processors */
    for ( i=1; i<size; i++ ) {
      nz = procsnz[i];
      ierr = SYRead(fd,cols,nz,SYINT); CHKERRQ(ierr);
      MPI_Send(cols,nz,MPI_INT,i,mat->tag,comm);
    }
    PetscFree(cols);
    vals = (Scalar *) PetscMalloc( maxnz*sizeof(Scalar) ); CHKPTRQ(vals);

    /* read in my part of the matrix numerical values  */
    nz = procsnz[0];
    ierr = SYRead(fd,vals,nz,SYSCALAR); CHKERRQ(ierr);
    
    /* insert it into my part of matrix */
    nz = 0;
    for ( i=0; i<A->num_rows; i++ ) {
      for (j=0; j<a->imax[i]; j++) {
        rs[i]->nz[j] = vals[nz++];
      }
    }
    /* read in parts for all other processors */
    for ( i=1; i<size; i++ ) {
      nz = procsnz[i];
      ierr = SYRead(fd,vals,nz,SYSCALAR); CHKERRQ(ierr);
      MPI_Send(vals,nz,MPIU_SCALAR,i,mat->tag,comm);
    }
    PetscFree(vals); PetscFree(procsnz);
  }
  else {
    /* determine buffer space needed for message */
    nz = 0;
    for ( i=0; i<A->num_rows; i++ ) {
      nz += a->imax[i];
    }
    cols = (int*) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(cols);

    /* receive message of column indices*/
    MPI_Recv(cols,nz,MPI_INT,0,mat->tag,comm,&status);
    MPI_Get_count(&status,MPI_INT,&maxnz);
    if (maxnz != nz) SETERRQ(1,"MatLoad_MPIRowbs: something is way wrong");

    /* insert it into my part of matrix */
    nz = 0;
    for ( i=0; i<A->num_rows; i++ ) {
      for (j=0; j<a->imax[i]; j++) {
        rs[i]->col[j] = cols[nz++];
      }
      rs[i]->length = a->imax[i];
    }
    PetscFree(cols);
    vals = (Scalar*) PetscMalloc( nz*sizeof(Scalar) ); CHKPTRQ(vals);

    /* receive message of values*/
    MPI_Recv(vals,nz,MPIU_SCALAR,0,mat->tag,comm,&status);
    MPI_Get_count(&status,MPIU_SCALAR,&maxnz);
    if (maxnz != nz) SETERRQ(1,"MatLoad_MPIRowbs: something is way wrong");

    /* insert it into my part of matrix */
    nz = 0;
    for ( i=0; i<A->num_rows; i++ ) {
      for (j=0; j<a->imax[i]; j++) {
        rs[i]->nz[j] = vals[nz++];
      }
      rs[i]->length = a->imax[i];
    }
    PetscFree(vals);
 
  }
  PetscFree(rowners);
  a->nz = a->maxnz;
  ierr = MatAssemblyBegin(mat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}

#else
#include "petsc.h"
#include "mat.h"
int MatCreateMPIRowbs(MPI_Comm comm,int m,int M,int nz,int *nnz,void *info,Mat *newmat)
{
  SETERRQ(1,"MatCreateMPIRowbs:This matrix format requires BlockSolve");
}
#endif





