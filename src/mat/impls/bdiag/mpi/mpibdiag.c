/*$Id: mpibdiag.c,v 1.187 2000/05/05 22:15:57 balay Exp bsmith $*/
/*
   The basic matrix operations for the Block diagonal parallel 
  matrices.
*/
#include "src/mat/impls/bdiag/mpi/mpibdiag.h"
#include "src/vec/vecimpl.h"

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatSetValues_MPIBDiag"
int MatSetValues_MPIBDiag(Mat mat,int m,int *idxm,int n,
                            int *idxn,Scalar *v,InsertMode addv)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag*)mat->data;
  int          ierr,i,j,row,rstart = mbd->rstart,rend = mbd->rend;
  int          roworiented = mbd->roworiented;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) continue;
    if (idxm[i] >= mbd->M) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for (j=0; j<n; j++) {
        if (idxn[j] < 0) continue;
        if (idxn[j] >= mbd->N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");
        if (roworiented) {
          ierr = MatSetValues(mbd->A,1,&row,1,&idxn[j],v+i*n+j,addv);CHKERRQ(ierr);
        } else {
          ierr = MatSetValues(mbd->A,1,&row,1,&idxn[j],v+i+j*m,addv);CHKERRQ(ierr);
        }
      }
    } else { 
      if (!mbd->donotstash) {
        if (roworiented) {
          ierr = MatStashValuesRow_Private(&mat->stash,idxm[i],n,idxn,v+i*n);CHKERRQ(ierr);
        } else {
          ierr = MatStashValuesCol_Private(&mat->stash,idxm[i],n,idxn,v+i,m);CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetValues_MPIBDiag"
int MatGetValues_MPIBDiag(Mat mat,int m,int *idxm,int n,int *idxn,Scalar *v)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag*)mat->data;
  int          ierr,i,j,row,rstart = mbd->rstart,rend = mbd->rend;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative row");
    if (idxm[i] >= mbd->M) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for (j=0; j<n; j++) {
        if (idxn[j] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative column");
        if (idxn[j] >= mbd->N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");
        ierr = MatGetValues(mbd->A,1,&row,1,&idxn[j],v+i*n+j);CHKERRQ(ierr);
      }
    } else {
      SETERRQ(PETSC_ERR_SUP,0,"Only local values currently supported");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatAssemblyBegin_MPIBDiag"
int MatAssemblyBegin_MPIBDiag(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag*)mat->data;
  MPI_Comm     comm = mat->comm;
  int          ierr,nstash,reallocs;
  InsertMode   addv;

  PetscFunctionBegin;
  ierr = MPI_Allreduce(&mat->insertmode,&addv,1,MPI_INT,MPI_BOR,comm);CHKERRQ(ierr);
  if (addv == (ADD_VALUES|INSERT_VALUES)) { 
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Cannot mix adds/inserts on different procs");
  }
  mat->insertmode = addv; /* in case this processor had no cache */
  ierr = MatStashScatterBegin_Private(&mat->stash,mbd->rowners);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  PLogInfo(0,"MatAssemblyBegin_MPIBDiag:Stash has %d entries,uses %d mallocs.\n",nstash,reallocs);
  PetscFunctionReturn(0);
}
EXTERN int MatSetUpMultiply_MPIBDiag(Mat);

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatAssemblyEnd_MPIBDiag"
int MatAssemblyEnd_MPIBDiag(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag*)mat->data;
  Mat_SeqBDiag *mlocal;
  int          i,n,*row,*col;
  int          *tmp1,*tmp2,ierr,len,ict,Mblock,Nblock,flg,j,rstart,ncols;
  Scalar       *val;
  InsertMode   addv = mat->insertmode;

  PetscFunctionBegin;

  while (1) {
    ierr = MatStashScatterGetMesg_Private(&mat->stash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
    if (!flg) break;
  
    for (i=0; i<n;) {
      /* Now identify the consecutive vals belonging to the same row */
      for (j=i,rstart=row[j]; j<n; j++) { if (row[j] != rstart) break; }
      if (j < n) ncols = j-i;
      else       ncols = n-i;
      /* Now assemble all these values with a single function call */
      ierr = MatSetValues_MPIBDiag(mat,1,row+i,ncols,col+i,val+i,addv);CHKERRQ(ierr);
      i = j;
    }
  }
  ierr = MatStashScatterEnd_Private(&mat->stash);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(mbd->A,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mbd->A,mode);CHKERRQ(ierr);

  /* Fix main diagonal location and determine global diagonals */
  mlocal         = (Mat_SeqBDiag*)mbd->A->data;
  Mblock         = mbd->M/mlocal->bs; Nblock = mbd->N/mlocal->bs;
  len            = Mblock + Nblock + 1; /* add 1 to prevent 0 malloc */
  tmp1           = (int*)PetscMalloc(2*len*sizeof(int));CHKPTRQ(tmp1);
  tmp2           = tmp1 + len;
  ierr           = PetscMemzero(tmp1,2*len*sizeof(int));CHKERRQ(ierr);
  mlocal->mainbd = -1; 
  for (i=0; i<mlocal->nd; i++) {
    if (mlocal->diag[i] + mbd->brstart == 0) mlocal->mainbd = i; 
    tmp1[mlocal->diag[i] + mbd->brstart + Mblock] = 1;
  }
  ierr = MPI_Allreduce(tmp1,tmp2,len,MPI_INT,MPI_SUM,mat->comm);CHKERRQ(ierr);
  ict  = 0;
  for (i=0; i<len; i++) {
    if (tmp2[i]) {
      mbd->gdiag[ict] = i - Mblock;
      ict++;
    }
  }
  mbd->gnd = ict;
  ierr = PetscFree(tmp1);CHKERRQ(ierr);

  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIBDiag(mat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetBlockSize_MPIBDiag"
int MatGetBlockSize_MPIBDiag(Mat mat,int *bs)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag*)mat->data;
  Mat_SeqBDiag *dmat = (Mat_SeqBDiag*)mbd->A->data;

  PetscFunctionBegin;
  *bs = dmat->bs;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatZeroEntries_MPIBDiag"
int MatZeroEntries_MPIBDiag(Mat A)
{
  Mat_MPIBDiag *l = (Mat_MPIBDiag*)A->data;
  int          ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}

/* again this uses the same basic stratagy as in the assembly and 
   scatter create routines, we should try to do it systematically 
   if we can figure out the proper level of generality. */

/* the code does not do the diagonal entries correctly unless the 
   matrix is square and the column and row owerships are identical.
   This is a BUG. The only way to fix it seems to be to access 
   aij->A and aij->B directly and not through the MatZeroRows() 
   routine. 
*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatZeroRows_MPIBDiag"
int MatZeroRows_MPIBDiag(Mat A,IS is,Scalar *diag)
{
  Mat_MPIBDiag   *l = (Mat_MPIBDiag*)A->data;
  int            i,ierr,N,*rows,*owners = l->rowners,size = l->size;
  int            *procs,*nprocs,j,found,idx,nsends,*work;
  int            nmax,*svalues,*starts,*owner,nrecvs,rank = l->rank;
  int            *rvalues,tag = A->tag,count,base,slen,n,*source;
  int            *lens,imdex,*lrows,*values;
  MPI_Comm       comm = A->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
  IS             istmp;

  PetscFunctionBegin;
  ierr = ISGetSize(is,&N);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows);CHKERRQ(ierr);

  /*  first count number of contributors to each processor */
  nprocs = (int*)PetscMalloc(2*size*sizeof(int));CHKPTRQ(nprocs);
  ierr   = PetscMemzero(nprocs,2*size*sizeof(int));CHKERRQ(ierr);
  procs  = nprocs + size;
  owner  = (int*)PetscMalloc((N+1)*sizeof(int));CHKPTRQ(owner); /* see note*/
  for (i=0; i<N; i++) {
    idx = rows[i];
    found = 0;
    for (j=0; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"row out of range");
  }
  nsends = 0;  for (i=0; i<size; i++) {nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work   = (int*)PetscMalloc(2*size*sizeof(int));CHKPTRQ(work);
  ierr   = MPI_Allreduce(nprocs,work,2*size,MPI_INT,PetscMaxSum_Op,comm);CHKERRQ(ierr);
  nmax   = work[rank];
  nrecvs = work[size+rank]; 
  ierr   = PetscFree(work);CHKERRQ(ierr);

  /* post receives:   */
  rvalues = (int*)PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int));CHKPTRQ(rvalues);
  recv_waits = (MPI_Request*)PetscMalloc((nrecvs+1)*sizeof(MPI_Request));CHKPTRQ(recv_waits);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv(rvalues+nmax*i,nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues = (int*)PetscMalloc((N+1)*sizeof(int));CHKPTRQ(svalues);
  send_waits = (MPI_Request*)PetscMalloc((nsends+1)*sizeof(MPI_Request));CHKPTRQ(send_waits);
  starts = (int*)PetscMalloc((size+1)*sizeof(int));CHKPTRQ(starts);
  starts[0] = 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for (i=0; i<N; i++) {
    svalues[starts[owner[i]]++] = rows[i];
  }
  ISRestoreIndices(is,&rows);

  starts[0] = 0;
  for (i=1; i<size+1; i++) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for (i=0; i<size; i++) {
    if (procs[i]) {
      ierr = MPI_Isend(svalues+starts[i],nprocs[i],MPI_INT,i,tag,comm,send_waits+count++);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(starts);CHKERRQ(ierr);

  base = owners[rank];

  /*  wait on receives */
  lens = (int*)PetscMalloc(2*(nrecvs+1)*sizeof(int));CHKPTRQ(lens);
  source = lens + nrecvs;
  count = nrecvs; slen = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPI_INT,&n);CHKERRQ(ierr);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]  = n;
    slen += n;
    count--;
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);
  
  /* move the data into the send scatter */
  lrows = (int*)PetscMalloc((slen+1)*sizeof(int));CHKPTRQ(lrows);
  count = 0;
  for (i=0; i<nrecvs; i++) {
    values = rvalues + i*nmax;
    for (j=0; j<lens[i]; j++) {
      lrows[count++] = values[j] - base;
    }
  }
  ierr = PetscFree(rvalues);CHKERRQ(ierr);
  ierr = PetscFree(lens);CHKERRQ(ierr);
  ierr = PetscFree(owner);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);
    
  /* actually zap the local rows */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,slen,lrows,&istmp);CHKERRQ(ierr);  
  PLogObjectParent(A,istmp);
  ierr = PetscFree(lrows);CHKERRQ(ierr);
  ierr = MatZeroRows(l->A,istmp,diag);CHKERRQ(ierr);
  ierr = ISDestroy(istmp);CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status*)PetscMalloc(nsends*sizeof(MPI_Status));CHKPTRQ(send_status);
    ierr        = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }
  ierr = PetscFree(send_waits);CHKERRQ(ierr);
  ierr = PetscFree(svalues);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatMult_MPIBDiag"
int MatMult_MPIBDiag(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag*)mat->data;
  int          ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(xx,mbd->lvec,INSERT_VALUES,SCATTER_FORWARD,mbd->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,mbd->lvec,INSERT_VALUES,SCATTER_FORWARD,mbd->Mvctx);CHKERRQ(ierr);
  ierr = (*mbd->A->ops->mult)(mbd->A,mbd->lvec,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatMultAdd_MPIBDiag"
int MatMultAdd_MPIBDiag(Mat mat,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag*)mat->data;
  int          ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(xx,mbd->lvec,INSERT_VALUES,SCATTER_FORWARD,mbd->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,mbd->lvec,INSERT_VALUES,SCATTER_FORWARD,mbd->Mvctx);CHKERRQ(ierr);
  ierr = (*mbd->A->ops->multadd)(mbd->A,mbd->lvec,yy,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatMultTranspose_MPIBDiag"
int MatMultTranspose_MPIBDiag(Mat A,Vec xx,Vec yy)
{
  Mat_MPIBDiag *a = (Mat_MPIBDiag*)A->data;
  int          ierr;
  Scalar       zero = 0.0;

  PetscFunctionBegin;
  ierr = VecSet(&zero,yy);CHKERRQ(ierr);
  ierr = (*a->A->ops->multtranspose)(a->A,xx,a->lvec);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatMultTransposeAdd_MPIBDiag"
int MatMultTransposeAdd_MPIBDiag(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIBDiag *a = (Mat_MPIBDiag*)A->data;
  int          ierr;

  PetscFunctionBegin;
  ierr = VecCopy(yy,zz);CHKERRQ(ierr);
  ierr = (*a->A->ops->multtranspose)(a->A,xx,a->lvec);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetInfo_MPIBDiag"
int MatGetInfo_MPIBDiag(Mat matin,MatInfoType flag,MatInfo *info)
{
  Mat_MPIBDiag *mat = (Mat_MPIBDiag*)matin->data;
  Mat_SeqBDiag *dmat = (Mat_SeqBDiag*)mat->A->data;
  int          ierr;
  PetscReal    isend[5],irecv[5];

  PetscFunctionBegin;
  info->block_size     = (PetscReal)dmat->bs;
  ierr = MatGetInfo(mat->A,MAT_LOCAL,info);CHKERRQ(ierr);
  isend[0] = info->nz_used; isend[1] = info->nz_allocated; isend[2] = info->nz_unneeded;
  isend[3] = info->memory;  isend[4] = info->mallocs;
  if (flag == MAT_LOCAL) {
    info->nz_used      = isend[0];
    info->nz_allocated = isend[1];
    info->nz_unneeded  = isend[2];
    info->memory       = isend[3];
    info->mallocs      = isend[4];
  } else if (flag == MAT_GLOBAL_MAX) {
    ierr = MPI_Allreduce(isend,irecv,5,MPI_DOUBLE,MPI_MAX,matin->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPI_Allreduce(isend,irecv,5,MPI_DOUBLE,MPI_SUM,matin->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  }
  info->rows_global    = (double)mat->M;
  info->columns_global = (double)mat->N;
  info->rows_local     = (double)mat->m;
  info->columns_local  = (double)mat->N;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetDiagonal_MPIBDiag"
int MatGetDiagonal_MPIBDiag(Mat mat,Vec v)
{
  int          ierr;
  Mat_MPIBDiag *A = (Mat_MPIBDiag*)mat->data;

  PetscFunctionBegin;
  ierr = MatGetDiagonal(A->A,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatDestroy_MPIBDiag"
int MatDestroy_MPIBDiag(Mat mat)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag*)mat->data;
  int          ierr;
#if defined(PETSC_USE_LOG)
  Mat_SeqBDiag *ms = (Mat_SeqBDiag*)mbd->A->data;

  PetscFunctionBegin;
  PLogObjectState((PetscObject)mat,"Rows=%d, Cols=%d, BSize=%d, NDiag=%d",mbd->M,mbd->N,ms->bs,ms->nd);
#else
  PetscFunctionBegin;
  if (--mat->refct > 0) PetscFunctionReturn(0);
#endif

  if (mat->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(mat->mapping);CHKERRQ(ierr);
  }
  if (mat->bmapping) {
    ierr = ISLocalToGlobalMappingDestroy(mat->bmapping);CHKERRQ(ierr);
  }
  if (mat->rmap) {
    ierr = MapDestroy(mat->rmap);CHKERRQ(ierr);
  }
  if (mat->cmap) {
    ierr = MapDestroy(mat->cmap);CHKERRQ(ierr);
  }

  ierr = MatStashDestroy_Private(&mat->stash);CHKERRQ(ierr);
  ierr = PetscFree(mbd->rowners);CHKERRQ(ierr);
  ierr = PetscFree(mbd->gdiag);CHKERRQ(ierr);
  ierr = MatDestroy(mbd->A);CHKERRQ(ierr);
  if (mbd->lvec) {ierr = VecDestroy(mbd->lvec);CHKERRQ(ierr);}
  if (mbd->Mvctx) {ierr = VecScatterDestroy(mbd->Mvctx);CHKERRQ(ierr);}
  ierr = PetscFree(mbd);CHKERRQ(ierr);
  PLogObjectDestroy(mat);
  PetscHeaderDestroy(mat);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatView_MPIBDiag_Binary"
static int MatView_MPIBDiag_Binary(Mat mat,Viewer viewer)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag*)mat->data;
  int          ierr;

  PetscFunctionBegin;
  if (mbd->size == 1) {
    ierr = MatView(mbd->A,viewer);CHKERRQ(ierr);
  } else SETERRQ(PETSC_ERR_SUP,0,"Only uniprocessor output supported");
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatView_MPIBDiag_ASCIIorDraw"
static int MatView_MPIBDiag_ASCIIorDraw(Mat mat,Viewer viewer)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag*)mat->data;
  Mat_SeqBDiag *dmat = (Mat_SeqBDiag*)mbd->A->data;
  int          ierr,format,i,size = mbd->size,rank = mbd->rank;
  PetscTruth   isascii,isdraw;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,DRAW_VIEWER,&isdraw);CHKERRQ(ierr);
  if (isascii) {
    ierr = ViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == VIEWER_FORMAT_ASCII_INFO || format == VIEWER_FORMAT_ASCII_INFO_LONG) {
      int nline = PetscMin(10,mbd->gnd),k,nk,np;
      ierr = ViewerASCIIPrintf(viewer,"  block size=%d, total number of diagonals=%d\n",dmat->bs,mbd->gnd);CHKERRQ(ierr);
      nk = (mbd->gnd-1)/nline + 1;
      for (k=0; k<nk; k++) {
        ierr = ViewerASCIIPrintf(viewer,"  global diag numbers:");CHKERRQ(ierr);
        np = PetscMin(nline,mbd->gnd - nline*k);
        for (i=0; i<np; i++) {
          ierr = ViewerASCIIPrintf(viewer,"  %d",mbd->gdiag[i+nline*k]);CHKERRQ(ierr);
        }
        ierr = ViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);        
      }
      if (format == VIEWER_FORMAT_ASCII_INFO_LONG) {
        MatInfo info;
        ierr = MPI_Comm_rank(mat->comm,&rank);CHKERRQ(ierr);
        ierr = MatGetInfo(mat,MAT_LOCAL,&info);CHKERRQ(ierr);
        ierr = ViewerASCIISynchronizedPrintf(viewer,"[%d] local rows %d nz %d nz alloced %d mem %d \n",rank,mbd->m,
            (int)info.nz_used,(int)info.nz_allocated,(int)info.memory);CHKERRQ(ierr);
        ierr = ViewerFlush(viewer);CHKERRQ(ierr);
        ierr = VecScatterView(mbd->Mvctx,viewer);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    }
  }

  if (isdraw) {
    Draw       draw;
    PetscTruth isnull;
    ierr = ViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = DrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
  }

  if (size == 1) { 
    ierr = MatView(mbd->A,viewer);CHKERRQ(ierr);
  } else {
    /* assemble the entire matrix onto first processor. */
    Mat          A;
    int          M = mbd->M,N = mbd->N,m,row,nz,*cols;
    Scalar       *vals;
    Mat_SeqBDiag *Ambd = (Mat_SeqBDiag*)mbd->A->data;

    if (!rank) {
      ierr = MatCreateMPIBDiag(mat->comm,M,M,N,mbd->gnd,Ambd->bs,
                               mbd->gdiag,PETSC_NULL,&A);CHKERRQ(ierr);
    } else {
      ierr = MatCreateMPIBDiag(mat->comm,0,M,N,0,Ambd->bs,PETSC_NULL,PETSC_NULL,&A);CHKERRQ(ierr);
    }
    PLogObjectParent(mat,A);

    /* Copy the matrix ... This isn't the most efficient means,
       but it's quick for now */
    row = mbd->rstart; m = Ambd->m;
    for (i=0; i<m; i++) {
      ierr = MatGetRow(mat,row,&nz,&cols,&vals);CHKERRQ(ierr);
      ierr = MatSetValues(A,1,&row,nz,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(mat,row,&nz,&cols,&vals);CHKERRQ(ierr);
      row++;
    } 
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (!rank) {
      Viewer sviewer;
      ierr = ViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
      ierr = MatView(((Mat_MPIBDiag*)(A->data))->A,sviewer);CHKERRQ(ierr);
      ierr = ViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
    }
    ierr = ViewerFlush(viewer);CHKERRQ(ierr);
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatView_MPIBDiag"
int MatView_MPIBDiag(Mat mat,Viewer viewer)
{
  int        ierr;
  PetscTruth isascii,isdraw,isbinary;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,DRAW_VIEWER,&isdraw);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,BINARY_VIEWER,&isbinary);CHKERRQ(ierr);
  if (isascii || isdraw) {
    ierr = MatView_MPIBDiag_ASCIIorDraw(mat,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_MPIBDiag_Binary(mat,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported by MPIBdiag matrices",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatSetOption_MPIBDiag"
int MatSetOption_MPIBDiag(Mat A,MatOption op)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag*)A->data;

  if (op == MAT_NO_NEW_NONZERO_LOCATIONS ||
      op == MAT_YES_NEW_NONZERO_LOCATIONS ||
      op == MAT_NEW_NONZERO_LOCATION_ERR ||
      op == MAT_NEW_NONZERO_ALLOCATION_ERR ||
      op == MAT_NO_NEW_DIAGONALS ||
      op == MAT_YES_NEW_DIAGONALS) {
        MatSetOption(mbd->A,op);
  } else if (op == MAT_ROW_ORIENTED) {
    mbd->roworiented = 1;
    MatSetOption(mbd->A,op);
  } else if (op == MAT_COLUMN_ORIENTED) {
    mbd->roworiented = 0;
    MatSetOption(mbd->A,op);
  } else if (op == MAT_IGNORE_OFF_PROC_ENTRIES) {
    mbd->donotstash = 1;
  } else if (op == MAT_ROWS_SORTED || 
             op == MAT_ROWS_UNSORTED || 
             op == MAT_COLUMNS_SORTED || 
             op == MAT_COLUMNS_UNSORTED || 
             op == MAT_SYMMETRIC ||
             op == MAT_STRUCTURALLY_SYMMETRIC ||
             op == MAT_YES_NEW_DIAGONALS ||
             op == MAT_USE_HASH_TABLE) {
    PLogInfo(A,"MatSetOption_MPIBDiag:Option ignored\n");
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"unknown option");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetSize_MPIBDiag"
int MatGetSize_MPIBDiag(Mat mat,int *m,int *n)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag*)mat->data;

  PetscFunctionBegin;
  if (m) *m = mbd->M; 
  if (n) *n = mbd->N;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetLocalSize_MPIBDiag"
int MatGetLocalSize_MPIBDiag(Mat mat,int *m,int *n)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag*)mat->data;

  PetscFunctionBegin;
  *m = mbd->m; *n = mbd->n;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetOwnershipRange_MPIBDiag"
int MatGetOwnershipRange_MPIBDiag(Mat matin,int *m,int *n)
{
  Mat_MPIBDiag *mat = (Mat_MPIBDiag*)matin->data;

  PetscFunctionBegin;
  if (m) *m = mat->rstart;
  if (n) *n = mat->rend;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetRow_MPIBDiag"
int MatGetRow_MPIBDiag(Mat matin,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIBDiag *mat = (Mat_MPIBDiag*)matin->data;
  int          lrow,ierr;

  PetscFunctionBegin;
  if (row < mat->rstart || row >= mat->rend) SETERRQ(PETSC_ERR_SUP,0,"only for local rows")
  lrow = row - mat->rstart;
  ierr = MatGetRow(mat->A,lrow,nz,idx,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatRestoreRow_MPIBDiag"
int MatRestoreRow_MPIBDiag(Mat matin,int row,int *nz,int **idx,
                                  Scalar **v)
{
  Mat_MPIBDiag *mat = (Mat_MPIBDiag*)matin->data;
  int          lrow,ierr;

  PetscFunctionBegin;
  lrow = row - mat->rstart;
  ierr = MatRestoreRow(mat->A,lrow,nz,idx,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatNorm_MPIBDiag"
int MatNorm_MPIBDiag(Mat A,NormType type,PetscReal *norm)
{
  Mat_MPIBDiag *mbd = (Mat_MPIBDiag*)A->data;
  Mat_SeqBDiag *a = (Mat_SeqBDiag*)mbd->A->data;
  PetscReal    sum = 0.0;
  int          ierr,d,i,nd = a->nd,bs = a->bs,len;
  Scalar       *dv;

  PetscFunctionBegin;
  if (type == NORM_FROBENIUS) {
    for (d=0; d<nd; d++) {
      dv   = a->diagv[d];
      len  = a->bdlen[d]*bs*bs;
      for (i=0; i<len; i++) {
#if defined(PETSC_USE_COMPLEX)
        sum += PetscRealPart(PetscConj(dv[i])*dv[i]);
#else
        sum += dv[i]*dv[i];
#endif
      }
    }
    ierr = MPI_Allreduce(&sum,norm,1,MPI_DOUBLE,MPI_SUM,A->comm);CHKERRQ(ierr);
    *norm = sqrt(*norm);
    PLogFlops(2*mbd->n*mbd->m);
  } else if (type == NORM_1) { /* max column norm */
    PetscReal *tmp,*tmp2;
    int    j;
    tmp  = (PetscReal*)PetscMalloc((a->n+1)*sizeof(PetscReal));CHKPTRQ(tmp);
    tmp2 = (PetscReal*)PetscMalloc((a->n+1)*sizeof(PetscReal));CHKPTRQ(tmp2);
    ierr = MatNorm_SeqBDiag_Columns(mbd->A,tmp,a->n);CHKERRQ(ierr);
    *norm = 0.0;
    ierr = MPI_Allreduce(tmp,tmp2,a->n,MPI_DOUBLE,MPI_SUM,A->comm);CHKERRQ(ierr);
    for (j=0; j<a->n; j++) {
      if (tmp2[j] > *norm) *norm = tmp2[j];
    }
    ierr = PetscFree(tmp);CHKERRQ(ierr);
    ierr = PetscFree(tmp2);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) { /* max row norm */
    PetscReal normtemp;
    ierr = MatNorm(mbd->A,type,&normtemp);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&normtemp,norm,1,MPI_DOUBLE,MPI_MAX,A->comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN int MatPrintHelp_SeqBDiag(Mat);
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatPrintHelp_MPIBDiag"
int MatPrintHelp_MPIBDiag(Mat A)
{
  Mat_MPIBDiag *a = (Mat_MPIBDiag*)A->data;
  int          ierr;

  PetscFunctionBegin;
  if (!a->rank) {
    ierr = MatPrintHelp_SeqBDiag(a->A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN int MatScale_SeqBDiag(Scalar*,Mat);
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatScale_MPIBDiag"
int MatScale_MPIBDiag(Scalar *alpha,Mat A)
{
  int          ierr;
  Mat_MPIBDiag *a = (Mat_MPIBDiag*)A->data;

  PetscFunctionBegin;
  ierr = MatScale_SeqBDiag(alpha,a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/

static struct _MatOps MatOps_Values = {MatSetValues_MPIBDiag,
       MatGetRow_MPIBDiag,
       MatRestoreRow_MPIBDiag,
       MatMult_MPIBDiag,
       MatMultAdd_MPIBDiag,
       MatMultTranspose_MPIBDiag,
       MatMultTransposeAdd_MPIBDiag,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       MatGetInfo_MPIBDiag,0,
       MatGetDiagonal_MPIBDiag,
       0,
       MatNorm_MPIBDiag,
       MatAssemblyBegin_MPIBDiag,
       MatAssemblyEnd_MPIBDiag,
       0,
       MatSetOption_MPIBDiag,
       MatZeroEntries_MPIBDiag,
       MatZeroRows_MPIBDiag,
       0,
       0,
       0,
       0,
       MatGetSize_MPIBDiag,
       MatGetLocalSize_MPIBDiag,
       MatGetOwnershipRange_MPIBDiag,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       MatGetValues_MPIBDiag,
       0,
       MatPrintHelp_MPIBDiag,
       MatScale_MPIBDiag,
       0,
       0,
       0,
       MatGetBlockSize_MPIBDiag,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       MatGetMaps_Petsc};

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetDiagonalBlock_MPIBDiag"
int MatGetDiagonalBlock_MPIBDiag(Mat A,PetscTruth *iscopy,MatReuse reuse,Mat *a)
{
  Mat_MPIBDiag *matin = (Mat_MPIBDiag *)A->data;
  int          ierr,lrows,lcols,rstart,rend;
  IS           localc,localr;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&lrows,&lcols);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,lrows,rstart,1,&localc);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,lrows,0,1,&localr);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(matin->A,localr,localc,PETSC_DECIDE,reuse,a);CHKERRQ(ierr);
  ierr = ISDestroy(localr);CHKERRQ(ierr);
  ierr = ISDestroy(localc);CHKERRQ(ierr);

  *iscopy = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatCreateMPIBDiag"
/*@C
   MatCreateMPIBDiag - Creates a sparse parallel matrix in MPIBDiag format.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
.  M - number of global rows (or PETSC_DETERMINE to have calculated if m is given)
.  N - number of columns (local and global)
.  nd - number of block diagonals (global) (optional)
.  bs - each element of a diagonal is an bs x bs dense matrix
.  diag - optional array of block diagonal numbers (length nd).
   For a matrix element A[i,j], where i=row and j=column, the
   diagonal number is
$     diag = i/bs - j/bs  (integer division)
   Set diag=PETSC_NULL on input for PETSc to dynamically allocate memory as 
   needed (expensive).
-  diagv  - pointer to actual diagonals (in same order as diag array), 
   if allocated by user. Otherwise, set diagv=PETSC_NULL on input for PETSc
   to control memory allocation.

   Output Parameter:
.  A - the matrix 

   Options Database Keys:
.  -mat_block_size <bs> - Sets blocksize
.  -mat_bdiag_diags <s1,s2,s3,...> - Sets diagonal numbers

   Notes:
   If PETSC_DECIDE or  PETSC_DETERMINE is used for a particular argument on one processor
   than it must be used on all processors that share the object for that argument.

   The parallel matrix is partitioned across the processors by rows, where
   each local rectangular matrix is stored in the uniprocessor block 
   diagonal format.  See the users manual for further details.

   The user MUST specify either the local or global numbers of rows
   (possibly both).

   The case bs=1 (conventional diagonal storage) is implemented as
   a special case.

   Fortran Notes:
   Fortran programmers cannot set diagv; this variable is ignored.

   Level: intermediate

.keywords: matrix, block, diagonal, parallel, sparse

.seealso: MatCreate(), MatCreateSeqBDiag(), MatSetValues()
@*/
int MatCreateMPIBDiag(MPI_Comm comm,int m,int M,int N,int nd,int bs,int *diag,Scalar **diagv,Mat *A)
{
  Mat          B;
  Mat_MPIBDiag *b;
  int          ierr,i,k,*ldiag,len,dset = 0,nd2;
  PetscTruth   flg1,flg2;
  Scalar       **ldiagv = 0;

  PetscFunctionBegin;
  *A = 0;
  if (bs == PETSC_DEFAULT) bs = 1;
  if (nd == PETSC_DEFAULT) nd = 0;
  ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-mat_bdiag_ndiag",&nd,PETSC_NULL);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-mat_bdiag_diags",&flg2);CHKERRQ(ierr);
  if (nd && !diag) {
    diag = (int *)PetscMalloc(nd * sizeof(int));CHKPTRQ(diag);
    nd2 = nd; dset = 1;
    ierr = OptionsGetIntArray(PETSC_NULL,"-mat_bdiag_dvals",diag,&nd2,PETSC_NULL);CHKERRQ(ierr);
    if (nd2 != nd) {
      SETERRQ(PETSC_ERR_ARG_INCOMP,0,"Incompatible number of diags and diagonal vals");
    }
  } else if (flg2) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must specify number of diagonals with -mat_bdiag_ndiag");
  }

  if (bs <= 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Blocksize must be positive");
  if ((N%bs)) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Invalid block size - bad column number");
  PetscHeaderCreate(B,_p_Mat,struct _MatOps,MAT_COOKIE,MATMPIBDIAG,"Mat",comm,MatDestroy,MatView);
  PLogObjectCreate(B);
  B->data         = (void*)(b = PetscNew(Mat_MPIBDiag));CHKPTRQ(b);
  ierr            = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->ops->destroy = MatDestroy_MPIBDiag;
  B->ops->view    = MatView_MPIBDiag;
  B->factor       = 0;
  B->mapping      = 0;

  B->insertmode = NOT_SET_VALUES;
  ierr = MPI_Comm_rank(comm,&b->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&b->size);CHKERRQ(ierr);

  ierr = PetscSplitOwnership(comm,&m,&M);CHKERRQ(ierr);
  if ((m%bs)) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Invalid block size - bad local row number");
  if ((M%bs)) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Invalid block size - bad global row number");
  b->M = M;    B->M = M;
  b->N = N;    B->N = N;
  b->m = m;    B->m = m;
  b->n = b->N; B->n = b->N;  /* each row stores all columns */
  b->gnd = nd;

  /* the information in the maps duplicates the information computed below, eventually 
     we should remove the duplicate information that is not contained in the maps */
  ierr = MapCreateMPI(comm,m,M,&B->rmap);CHKERRQ(ierr);
  ierr = MapCreateMPI(comm,m,M,&B->cmap);CHKERRQ(ierr);

  /* build local table of row ownerships */
  b->rowners    = (int*)PetscMalloc((b->size+2)*sizeof(int));CHKPTRQ(b->rowners);
  ierr          = MPI_Allgather(&m,1,MPI_INT,b->rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  b->rowners[0] = 0;
  for (i=2; i<=b->size; i++) {
    b->rowners[i] += b->rowners[i-1];
  }
  b->rstart  = b->rowners[b->rank]; 
  b->rend    = b->rowners[b->rank+1]; 

  b->brstart = (b->rstart)/bs;
  b->brend   = (b->rend)/bs;

  /* Determine local diagonals; for now, assume global rows = global cols */
  /* These are sorted in MatCreateSeqBDiag */
  ldiag = (int*)PetscMalloc((nd+1)*sizeof(int));CHKPTRQ(ldiag); 
  len = M/bs + N/bs + 1; /* add 1 to prevent 0 malloc */
  b->gdiag = (int*)PetscMalloc(len*sizeof(int));CHKPTRQ(b->gdiag);
  k = 0;
  PLogObjectMemory(B,(nd+1)*sizeof(int) + (b->size+2)*sizeof(int)
                        + sizeof(struct _p_Mat) + sizeof(Mat_MPIBDiag));
  if (diagv) {
    ldiagv = (Scalar **)PetscMalloc((nd+1)*sizeof(Scalar*));CHKPTRQ(ldiagv); 
  }
  for (i=0; i<nd; i++) {
    b->gdiag[i] = diag[i];
    if (diag[i] > 0) { /* lower triangular */
      if (diag[i] < b->brend) {
        ldiag[k] = diag[i] - b->brstart;
        if (diagv) ldiagv[k] = diagv[i];
        k++;
      }
    } else { /* upper triangular */
      if (b->M/bs - diag[i] > b->N/bs) {
        if (b->M/bs + diag[i] > b->brstart) {
          ldiag[k] = diag[i] - b->brstart;
          if (diagv) ldiagv[k] = diagv[i];
          k++;
        }
      } else {
        if (b->M/bs > b->brstart) {
          ldiag[k] = diag[i] - b->brstart;
          if (diagv) ldiagv[k] = diagv[i];
          k++;
        }
      }
    }
  }

  /* Form local matrix */
  ierr = MatCreateSeqBDiag(PETSC_COMM_SELF,b->m,b->n,k,bs,ldiag,ldiagv,&b->A);CHKERRQ(ierr); 
  PLogObjectParent(B,b->A);
  ierr = PetscFree(ldiag);CHKERRQ(ierr);
  if (ldiagv) {ierr = PetscFree(ldiagv);CHKERRQ(ierr);}

  /* build cache for off array entries formed */
  ierr = MatStashCreate_Private(B->comm,1,&B->stash);CHKERRQ(ierr);
  b->donotstash = 0;

  /* stuff used for matrix-vector multiply */
  b->lvec        = 0;
  b->Mvctx       = 0;

  /* used for MatSetValues() input */
  b->roworiented = 1;

  ierr = OptionsHasName(PETSC_NULL,"-help",&flg1);CHKERRQ(ierr);
  if (flg1) {ierr = MatPrintHelp(B);CHKERRQ(ierr);}
  if (dset) {ierr = PetscFree(diag);CHKERRQ(ierr);}
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetDiagonalBlock_C",
                                     "MatGetDiagonalBlock_MPIBDiag",
                                      MatGetDiagonalBlock_MPIBDiag);CHKERRQ(ierr);
  *A = B;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatBDiagGetData"
/*@C
   MatBDiagGetData - Gets the data for the block diagonal matrix format.
   For the parallel case, this returns information for the local submatrix.

   Input Parameters:
.  mat - the matrix, stored in block diagonal format.

   Not Collective

   Output Parameters:
+  m - number of rows
.  n - number of columns
.  nd - number of block diagonals
.  bs - each element of a diagonal is an bs x bs dense matrix
.  bdlen - array of total block lengths of block diagonals
.  diag - optional array of block diagonal numbers (length nd).
   For a matrix element A[i,j], where i=row and j=column, the
   diagonal number is
$     diag = i/bs - j/bs  (integer division)
   Set diag=PETSC_NULL on input for PETSc to dynamically allocate memory as 
   needed (expensive).
-  diagv - pointer to actual diagonals (in same order as diag array), 

   Level: advanced

   Notes:
   See the users manual for further details regarding this storage format.

.keywords: matrix, block, diagonal, get, data

.seealso: MatCreateSeqBDiag(), MatCreateMPIBDiag()
@*/
int MatBDiagGetData(Mat mat,int *nd,int *bs,int **diag,int **bdlen,Scalar ***diagv)
{
  Mat_MPIBDiag *pdmat;
  Mat_SeqBDiag *dmat = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (mat->type == MATSEQBDIAG) {
    dmat = (Mat_SeqBDiag*)mat->data;
  } else if (mat->type == MATMPIBDIAG) {
    pdmat = (Mat_MPIBDiag*)mat->data;
    dmat = (Mat_SeqBDiag*)pdmat->A->data;
  } else SETERRQ(PETSC_ERR_SUP,0,"Valid only for MATSEQBDIAG and MATMPIBDIAG formats");
  *nd    = dmat->nd;
  *bs    = dmat->bs;
  *diag  = dmat->diag;
  *bdlen = dmat->bdlen;
  *diagv = dmat->diagv;
  PetscFunctionReturn(0);
}

#include "petscsys.h"

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatLoad_MPIBDiag"
int MatLoad_MPIBDiag(Viewer viewer,MatType type,Mat *newmat)
{
  Mat          A;
  Scalar       *vals,*svals;
  MPI_Comm     comm = ((PetscObject)viewer)->comm;
  MPI_Status   status;
  int          bs,i,nz,ierr,j,rstart,rend,fd,*rowners,maxnz,*cols;
  int          header[4],rank,size,*rowlengths = 0,M,N,m,Mbs;
  int          *ourlens,*sndcounts = 0,*procsnz = 0,jj,*mycols,*smycols;
  int          tag = ((PetscObject)viewer)->tag,extra_rows;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = ViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,PETSC_INT);CHKERRQ(ierr);
    if (header[0] != MAT_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"not matrix object");
    if (header[3] < 0) {
      SETERRQ(PETSC_ERR_FILE_UNEXPECTED,1,"Matrix stored in special format,cannot load as MPIBDiag");
    }
  }
  ierr = MPI_Bcast(header+1,3,MPI_INT,0,comm);CHKERRQ(ierr);
  M = header[1]; N = header[2];

  bs = 1;   /* uses a block size of 1 by default; */
  ierr = OptionsGetInt(PETSC_NULL,"-matload_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);

  /* 
     This code adds extra rows to make sure the number of rows is 
     divisible by the blocksize
  */
  Mbs        = M/bs;
  extra_rows = bs - M + bs*(Mbs);
  if (extra_rows == bs) extra_rows = 0;
  else                  Mbs++;
  if (extra_rows && !rank) {
    PLogInfo(0,"MatLoad_MPIBDiag:Padding loaded matrix to match blocksize\n");
  }

  /* determine ownership of all rows */
  m          = bs*(Mbs/size + ((Mbs % size) > rank));
  rowners    = (int*)PetscMalloc((size+2)*sizeof(int));CHKPTRQ(rowners);
  ierr       = MPI_Allgather(&m,1,MPI_INT,rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  rowners[0] = 0;
  for (i=2; i<=size; i++) {
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 

  /* distribute row lengths to all processors */
  ourlens = (int*)PetscMalloc((rend-rstart)*sizeof(int));CHKPTRQ(ourlens);
  if (!rank) {
    rowlengths = (int*)PetscMalloc((M+extra_rows)*sizeof(int));CHKPTRQ(rowlengths);
    ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);
    for (i=0; i<extra_rows; i++) rowlengths[M+i] = 1;
    sndcounts = (int*)PetscMalloc(size*sizeof(int));CHKPTRQ(sndcounts);
    for (i=0; i<size; i++) sndcounts[i] = rowners[i+1] - rowners[i];
    ierr = MPI_Scatterv(rowlengths,sndcounts,rowners,MPI_INT,ourlens,rend-rstart,MPI_INT,0,comm);CHKERRQ(ierr);
    ierr = PetscFree(sndcounts);CHKERRQ(ierr);
  } else {
    ierr = MPI_Scatterv(0,0,0,MPI_INT,ourlens,rend-rstart,MPI_INT,0,comm);CHKERRQ(ierr);
  }

  if (!rank) {
    /* calculate the number of nonzeros on each processor */
    procsnz = (int*)PetscMalloc(size*sizeof(int));CHKPTRQ(procsnz);
    ierr    = PetscMemzero(procsnz,size*sizeof(int));CHKERRQ(ierr);
    for (i=0; i<size; i++) {
      for (j=rowners[i]; j<rowners[i+1]; j++) {
        procsnz[i] += rowlengths[j];
      }
    }
    ierr = PetscFree(rowlengths);CHKERRQ(ierr);

    /* determine max buffer needed and allocate it */
    maxnz = 0;
    for (i=0; i<size; i++) {
      maxnz = PetscMax(maxnz,procsnz[i]);
    }
    cols = (int*)PetscMalloc(maxnz*sizeof(int));CHKPTRQ(cols);

    /* read in my part of the matrix column indices  */
    nz = procsnz[0];
    mycols = (int*)PetscMalloc(nz*sizeof(int));CHKPTRQ(mycols);
    if (size == 1)  nz -= extra_rows;
    ierr = PetscBinaryRead(fd,mycols,nz,PETSC_INT);CHKERRQ(ierr);
    if (size == 1)  for (i=0; i<extra_rows; i++) { mycols[nz+i] = M+i; }

    /* read in every one elses and ship off */
    for (i=1; i<size-1; i++) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
      ierr = MPI_Send(cols,nz,MPI_INT,i,tag,comm);CHKERRQ(ierr);
    }
    /* read in the stuff for the last proc */
    if (size != 1) {
      nz   = procsnz[size-1] - extra_rows;  /* the extra rows are not on the disk */
      ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
      for (i=0; i<extra_rows; i++) cols[nz+i] = M+i;
      ierr = MPI_Send(cols,nz+extra_rows,MPI_INT,size-1,tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(cols);CHKERRQ(ierr);
  } else {
    /* determine buffer space needed for message */
    nz = 0;
    for (i=0; i<m; i++) {
      nz += ourlens[i];
    }
    mycols = (int*)PetscMalloc(nz*sizeof(int));CHKPTRQ(mycols);

    /* receive message of column indices*/
    ierr = MPI_Recv(mycols,nz,MPI_INT,0,tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPI_INT,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"something is wrong with file");
  }

  ierr = MatCreateMPIBDiag(comm,m,M+extra_rows,N+extra_rows,PETSC_NULL,bs,PETSC_NULL,PETSC_NULL,
                           newmat);CHKERRQ(ierr);
  A = *newmat;

  if (!rank) {
    vals = (Scalar*)PetscMalloc(maxnz*sizeof(Scalar));CHKPTRQ(vals);

    /* read in my part of the matrix numerical values  */
    nz = procsnz[0];
    if (size == 1)  nz -= extra_rows;
    ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
    if (size == 1)  for (i=0; i<extra_rows; i++) { vals[nz+i] = 1.0; }   

    /* insert into matrix */
    jj      = rstart;
    smycols = mycols;
    svals   = vals;
    for (i=0; i<m; i++) {
      ierr = MatSetValues(A,1,&jj,ourlens[i],smycols,svals,INSERT_VALUES);CHKERRQ(ierr);
      smycols += ourlens[i];
      svals   += ourlens[i];
      jj++;
    }

    /* read in other processors (except the last one) and ship out */
    for (i=1; i<size-1; i++) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
      ierr = MPI_Send(vals,nz,MPIU_SCALAR,i,A->tag,comm);CHKERRQ(ierr);
    }
    /* the last proc */
    if (size != 1){
      nz   = procsnz[i] - extra_rows;
      ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
      for (i=0; i<extra_rows; i++) vals[nz+i] = 1.0;
      ierr = MPI_Send(vals,nz+extra_rows,MPIU_SCALAR,size-1,A->tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(procsnz);CHKERRQ(ierr);
  } else {
    /* receive numeric values */
    vals = (Scalar*)PetscMalloc(nz*sizeof(Scalar));CHKPTRQ(vals);

    /* receive message of values*/
    ierr = MPI_Recv(vals,nz,MPIU_SCALAR,0,A->tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPIU_SCALAR,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"something is wrong with file");

    /* insert into matrix */
    jj      = rstart;
    smycols = mycols;
    svals   = vals;
    for (i=0; i<m; i++) {
      ierr = MatSetValues(A,1,&jj,ourlens[i],smycols,svals,INSERT_VALUES);CHKERRQ(ierr);
      smycols += ourlens[i];
      svals   += ourlens[i];
      jj++;
    }
  }
  ierr = PetscFree(ourlens);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);
  ierr = PetscFree(mycols);CHKERRQ(ierr);
  ierr = PetscFree(rowners);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}






