#define PETSCMAT_DLL

/*
   The basic matrix operations for the Block diagonal parallel 
  matrices.
*/
#include "src/mat/impls/bdiag/mpi/mpibdiag.h"

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_MPIBDiag"
PetscErrorCode MatSetValues_MPIBDiag(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPIBDiag   *mbd = (Mat_MPIBDiag*)mat->data;
  PetscErrorCode ierr;
  PetscInt       i,j,row,rstart = mat->rmap.rstart,rend = mat->rmap.rend;
  PetscTruth     roworiented = mbd->roworiented;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) continue;
    if (idxm[i] >= mat->rmap.N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for (j=0; j<n; j++) {
        if (idxn[j] < 0) continue;
        if (idxn[j] >= mat->cmap.N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Column too large");
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

#undef __FUNCT__  
#define __FUNCT__ "MatGetValues_MPIBDiag"
PetscErrorCode MatGetValues_MPIBDiag(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],PetscScalar v[])
{
  Mat_MPIBDiag   *mbd = (Mat_MPIBDiag*)mat->data;
  PetscErrorCode ierr;
  PetscInt       i,j,row,rstart = mat->rmap.rstart,rend = mat->rmap.rend;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative row");
    if (idxm[i] >= mat->rmap.N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for (j=0; j<n; j++) {
        if (idxn[j] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative column");
        if (idxn[j] >= mat->cmap.N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Column too large");
        ierr = MatGetValues(mbd->A,1,&row,1,&idxn[j],v+i*n+j);CHKERRQ(ierr);
      }
    } else {
      SETERRQ(PETSC_ERR_SUP,"Only local values currently supported");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyBegin_MPIBDiag"
PetscErrorCode MatAssemblyBegin_MPIBDiag(Mat mat,MatAssemblyType mode)
{ 
  MPI_Comm       comm = mat->comm;
  PetscErrorCode ierr;
  PetscInt       nstash,reallocs;
  InsertMode     addv;

  PetscFunctionBegin;
  ierr = MPI_Allreduce(&mat->insertmode,&addv,1,MPI_INT,MPI_BOR,comm);CHKERRQ(ierr);
  if (addv == (ADD_VALUES|INSERT_VALUES)) { 
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot mix adds/inserts on different procs");
  }
  mat->insertmode = addv; /* in case this processor had no cache */
  ierr = MatStashScatterBegin_Private(&mat->stash,mat->rmap.range);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscInfo2(0,"Stash has %D entries,uses %D mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_MPIBDiag"
PetscErrorCode MatAssemblyEnd_MPIBDiag(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIBDiag   *mbd = (Mat_MPIBDiag*)mat->data;
  Mat_SeqBDiag   *mlocal;
  PetscErrorCode ierr;
  PetscMPIInt    n;
  PetscInt       i,*row,*col;
  PetscInt       *tmp1,*tmp2,len,ict,Mblock,Nblock,flg,j,rstart,ncols;
  PetscScalar    *val;
  InsertMode     addv = mat->insertmode;

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
  Mblock         = mat->rmap.N/mat->rmap.bs; Nblock = mat->cmap.N/mat->rmap.bs;
  len            = Mblock + Nblock + 1; /* add 1 to prevent 0 malloc */
  ierr           = PetscMalloc(2*len*sizeof(PetscInt),&tmp1);CHKERRQ(ierr);
  tmp2           = tmp1 + len;
  ierr           = PetscMemzero(tmp1,2*len*sizeof(PetscInt));CHKERRQ(ierr);
  mlocal->mainbd = -1; 
  for (i=0; i<mlocal->nd; i++) {
    if (mlocal->diag[i] + mbd->brstart == 0) mlocal->mainbd = i; 
    tmp1[mlocal->diag[i] + mbd->brstart + Mblock] = 1;
  }
  ierr = MPI_Allreduce(tmp1,tmp2,len,MPIU_INT,MPI_SUM,mat->comm);CHKERRQ(ierr);
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

#undef __FUNCT__  
#define __FUNCT__ "MatZeroEntries_MPIBDiag"
PetscErrorCode MatZeroEntries_MPIBDiag(Mat A)
{
  Mat_MPIBDiag   *l = (Mat_MPIBDiag*)A->data;
  PetscErrorCode ierr;

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

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_MPIBDiag"
PetscErrorCode MatZeroRows_MPIBDiag(Mat A,PetscInt N,const PetscInt rows[],PetscScalar diag)
{
  Mat_MPIBDiag   *l = (Mat_MPIBDiag*)A->data;
  PetscErrorCode ierr;
  PetscMPIInt    n,imdex,size = l->size,rank = l->rank,tag = A->tag;
  PetscInt       i,*owners = A->rmap.range;
  PetscInt       *nprocs,j,idx,nsends;
  PetscInt       nmax,*svalues,*starts,*owner,nrecvs;
  PetscInt       *rvalues,count,base,slen,*source;
  PetscInt       *lens,*lrows,*values;
  MPI_Comm       comm = A->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
  PetscTruth     found;

  PetscFunctionBegin;
  /*  first count number of contributors to each processor */
  ierr   = PetscMalloc(2*size*sizeof(PetscInt),&nprocs);CHKERRQ(ierr);
  ierr   = PetscMemzero(nprocs,2*size*sizeof(PetscInt));CHKERRQ(ierr);
  ierr   = PetscMalloc((N+1)*sizeof(PetscInt),&owner);CHKERRQ(ierr); /* see note*/
  for (i=0; i<N; i++) {
    idx = rows[i];
    found = PETSC_FALSE;
    for (j=0; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[2*j]++; nprocs[2*j+1] = 1; owner[i] = j; found = PETSC_TRUE; break;
      }
    }
    if (!found) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"row out of range");
  }
  nsends = 0;  for (i=0; i<size; i++) {nsends += nprocs[2*i+1];} 

  /* inform other processors of number of messages and max length*/
  ierr = PetscMaxSum(comm,nprocs,&nmax,&nrecvs);CHKERRQ(ierr);

  /* post receives:   */
  ierr = PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(PetscInt),&rvalues);CHKERRQ(ierr);
  ierr = PetscMalloc((nrecvs+1)*sizeof(MPI_Request),&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv(rvalues+nmax*i,nmax,MPIU_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  ierr = PetscMalloc((N+1)*sizeof(PetscInt),&svalues);CHKERRQ(ierr);
  ierr = PetscMalloc((nsends+1)*sizeof(MPI_Request),&send_waits);CHKERRQ(ierr);
  ierr = PetscMalloc((size+1)*sizeof(PetscInt),&starts);CHKERRQ(ierr);
  starts[0] = 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + nprocs[2*i-2];} 
  for (i=0; i<N; i++) {
    svalues[starts[owner[i]]++] = rows[i];
  }

  starts[0] = 0;
  for (i=1; i<size+1; i++) { starts[i] = starts[i-1] + nprocs[2*i-2];} 
  count = 0;
  for (i=0; i<size; i++) {
    if (nprocs[2*i+1]) {
      ierr = MPI_Isend(svalues+starts[i],nprocs[2*i],MPIU_INT,i,tag,comm,send_waits+count++);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(starts);CHKERRQ(ierr);

  base = owners[rank];

  /*  wait on receives */
  ierr   = PetscMalloc(2*(nrecvs+1)*sizeof(PetscInt),&lens);CHKERRQ(ierr);
  source = lens + nrecvs;
  count  = nrecvs;
  slen   = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPIU_INT,&n);CHKERRQ(ierr);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]  = n;
    slen += n;
    count--;
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);
  
  /* move the data into the send scatter */
  ierr  = PetscMalloc((slen+1)*sizeof(PetscInt),&lrows);CHKERRQ(ierr);
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
  ierr = MatZeroRows(l->A,slen,lrows,diag);CHKERRQ(ierr);
  ierr = PetscFree(lrows);CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {
    ierr = PetscMalloc(nsends*sizeof(MPI_Status),&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }
  ierr = PetscFree(send_waits);CHKERRQ(ierr);
  ierr = PetscFree(svalues);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_MPIBDiag"
PetscErrorCode MatMult_MPIBDiag(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIBDiag   *mbd = (Mat_MPIBDiag*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(xx,mbd->lvec,INSERT_VALUES,SCATTER_FORWARD,mbd->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,mbd->lvec,INSERT_VALUES,SCATTER_FORWARD,mbd->Mvctx);CHKERRQ(ierr);
  ierr = (*mbd->A->ops->mult)(mbd->A,mbd->lvec,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_MPIBDiag"
PetscErrorCode MatMultAdd_MPIBDiag(Mat mat,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIBDiag   *mbd = (Mat_MPIBDiag*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(xx,mbd->lvec,INSERT_VALUES,SCATTER_FORWARD,mbd->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,mbd->lvec,INSERT_VALUES,SCATTER_FORWARD,mbd->Mvctx);CHKERRQ(ierr);
  ierr = (*mbd->A->ops->multadd)(mbd->A,mbd->lvec,yy,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_MPIBDiag"
PetscErrorCode MatMultTranspose_MPIBDiag(Mat A,Vec xx,Vec yy)
{
  Mat_MPIBDiag   *a = (Mat_MPIBDiag*)A->data;
  PetscErrorCode ierr;
  PetscScalar    zero = 0.0;

  PetscFunctionBegin;
  ierr = VecSet(yy,zero);CHKERRQ(ierr);
  ierr = (*a->A->ops->multtranspose)(a->A,xx,a->lvec);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_MPIBDiag"
PetscErrorCode MatMultTransposeAdd_MPIBDiag(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIBDiag   *a = (Mat_MPIBDiag*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(yy,zz);CHKERRQ(ierr);
  ierr = (*a->A->ops->multtranspose)(a->A,xx,a->lvec);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_MPIBDiag"
PetscErrorCode MatGetInfo_MPIBDiag(Mat matin,MatInfoType flag,MatInfo *info)
{
  Mat_MPIBDiag   *mat = (Mat_MPIBDiag*)matin->data;
  PetscErrorCode ierr;
  PetscReal      isend[5],irecv[5];

  PetscFunctionBegin;
  info->block_size     = (PetscReal)mat->A->rmap.bs;
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
    ierr = MPI_Allreduce(isend,irecv,5,MPIU_REAL,MPI_MAX,matin->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPI_Allreduce(isend,irecv,5,MPIU_REAL,MPI_SUM,matin->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  }
  info->rows_global    = (double)matin->rmap.N;
  info->columns_global = (double)matin->cmap.N;
  info->rows_local     = (double)matin->rmap.n;
  info->columns_local  = (double)matin->cmap.N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_MPIBDiag"
PetscErrorCode MatGetDiagonal_MPIBDiag(Mat mat,Vec v)
{
  PetscErrorCode ierr;
  Mat_MPIBDiag   *A = (Mat_MPIBDiag*)mat->data;

  PetscFunctionBegin;
  ierr = MatGetDiagonal(A->A,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIBDiag"
PetscErrorCode MatDestroy_MPIBDiag(Mat mat)
{
  Mat_MPIBDiag   *mbd = (Mat_MPIBDiag*)mat->data;
  PetscErrorCode ierr;
#if defined(PETSC_USE_LOG)
  Mat_SeqBDiag   *ms = (Mat_SeqBDiag*)mbd->A->data;

  PetscFunctionBegin;
  PetscLogObjectState((PetscObject)mat,"Rows=%D, Cols=%D, BSize=%D, NDiag=%D",mat->rmap.N,mat->cmap.N,mat->rmap.bs,ms->nd);
#else
  PetscFunctionBegin;
#endif
  ierr = MatStashDestroy_Private(&mat->stash);CHKERRQ(ierr);
  ierr = PetscFree(mbd->gdiag);CHKERRQ(ierr);
  ierr = MatDestroy(mbd->A);CHKERRQ(ierr);
  if (mbd->lvec) {ierr = VecDestroy(mbd->lvec);CHKERRQ(ierr);}
  if (mbd->Mvctx) {ierr = VecScatterDestroy(mbd->Mvctx);CHKERRQ(ierr);}
  ierr = PetscFree(mbd);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatGetDiagonalBlock_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPIBDiagSetPreallocation_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIBDiag_Binary"
static PetscErrorCode MatView_MPIBDiag_Binary(Mat mat,PetscViewer viewer)
{
  Mat_MPIBDiag   *mbd = (Mat_MPIBDiag*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mbd->size == 1) {
    ierr = MatView(mbd->A,viewer);CHKERRQ(ierr);
  } else SETERRQ(PETSC_ERR_SUP,"Only uniprocessor output supported");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIBDiag_ASCIIorDraw"
static PetscErrorCode MatView_MPIBDiag_ASCIIorDraw(Mat mat,PetscViewer viewer)
{
  Mat_MPIBDiag      *mbd = (Mat_MPIBDiag*)mat->data;
  PetscErrorCode    ierr;
  PetscMPIInt       size = mbd->size,rank = mbd->rank;
  PetscInt          i;
  PetscTruth        iascii,isdraw;
  PetscViewer       sviewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      PetscInt nline = PetscMin(10,mbd->gnd),k,nk,np;
      ierr = PetscViewerASCIIPrintf(viewer,"  block size=%D, total number of diagonals=%D\n",mat->rmap.bs,mbd->gnd);CHKERRQ(ierr);
      nk = (mbd->gnd-1)/nline + 1;
      for (k=0; k<nk; k++) {
        ierr = PetscViewerASCIIPrintf(viewer,"  global diag numbers:");CHKERRQ(ierr);
        np = PetscMin(nline,mbd->gnd - nline*k);
        for (i=0; i<np; i++) {
          ierr = PetscViewerASCIIPrintf(viewer,"  %D",mbd->gdiag[i+nline*k]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);        
      }
      if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
        MatInfo info;
        ierr = MPI_Comm_rank(mat->comm,&rank);CHKERRQ(ierr);
        ierr = MatGetInfo(mat,MAT_LOCAL,&info);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] local rows %D nz %D nz alloced %D mem %D \n",rank,mat->rmap.N,
            (PetscInt)info.nz_used,(PetscInt)info.nz_allocated,(PetscInt)info.memory);CHKERRQ(ierr);
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
        ierr = VecScatterView(mbd->Mvctx,viewer);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    }
  }

  if (isdraw) {
    PetscDraw       draw;
    PetscTruth isnull;
    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
  }

  if (size == 1) { 
    ierr = MatView(mbd->A,viewer);CHKERRQ(ierr);
  } else {
    /* assemble the entire matrix onto first processor. */
    Mat          A;
    PetscInt     M = mat->rmap.N,N = mat->cmap.N,m,row,nz,*cols;
    PetscScalar  *vals;

    /* Here we are constructing a temporary matrix, so we will explicitly set the type to MPIBDiag */
    ierr = MatCreate(mat->comm,&A);CHKERRQ(ierr);
    if (!rank) {
      ierr = MatSetSizes(A,M,N,M,N);CHKERRQ(ierr);
      ierr = MatSetType(A,MATMPIBDIAG);CHKERRQ(ierr);
      ierr = MatMPIBDiagSetPreallocation(A,mbd->gnd,mbd->A->rmap.bs,mbd->gdiag,PETSC_NULL);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(A,0,0,M,N);CHKERRQ(ierr);
      ierr = MatSetType(A,MATMPIBDIAG);CHKERRQ(ierr);
      ierr = MatMPIBDiagSetPreallocation(A,0,mbd->A->rmap.bs,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = PetscLogObjectParent(mat,A);CHKERRQ(ierr);

    /* Copy the matrix ... This isn't the most efficient means,
       but it's quick for now */
    row = mat->rmap.rstart;
    m = mbd->A->rmap.N;
    for (i=0; i<m; i++) {
      ierr = MatGetRow_MPIBDiag(mat,row,&nz,&cols,&vals);CHKERRQ(ierr);
      ierr = MatSetValues(A,1,&row,nz,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow_MPIBDiag(mat,row,&nz,&cols,&vals);CHKERRQ(ierr);
      row++;
    } 
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
    if (!rank) {
      ierr = MatView(((Mat_MPIBDiag*)(A->data))->A,sviewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIBDiag"
PetscErrorCode MatView_MPIBDiag(Mat mat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscTruth     iascii,isdraw,isbinary;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  if (iascii || isdraw) {
    ierr = MatView_MPIBDiag_ASCIIorDraw(mat,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_MPIBDiag_Binary(mat,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by MPIBdiag matrices",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_MPIBDiag"
PetscErrorCode MatSetOption_MPIBDiag(Mat A,MatOption op)
{
  Mat_MPIBDiag   *mbd = (Mat_MPIBDiag*)A->data;
  PetscErrorCode ierr;

  switch (op) {
  case MAT_NO_NEW_NONZERO_LOCATIONS:
  case MAT_YES_NEW_NONZERO_LOCATIONS:
  case MAT_NEW_NONZERO_LOCATION_ERR:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
  case MAT_NO_NEW_DIAGONALS:
  case MAT_YES_NEW_DIAGONALS:
    ierr = MatSetOption(mbd->A,op);CHKERRQ(ierr);
    break;
  case MAT_ROW_ORIENTED:
    mbd->roworiented = PETSC_TRUE;
    ierr = MatSetOption(mbd->A,op);CHKERRQ(ierr);
    break;
  case MAT_COLUMN_ORIENTED:
    mbd->roworiented = PETSC_FALSE;
    ierr = MatSetOption(mbd->A,op);CHKERRQ(ierr);
    break;
  case MAT_IGNORE_OFF_PROC_ENTRIES:
    mbd->donotstash = PETSC_TRUE;
    break;
  case MAT_ROWS_SORTED:
  case MAT_ROWS_UNSORTED:
  case MAT_COLUMNS_SORTED:
  case MAT_COLUMNS_UNSORTED:
    ierr = PetscInfo1(A,"Option %d ignored\n",op);CHKERRQ(ierr);
    break;
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_NOT_SYMMETRIC:
  case MAT_NOT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
  case MAT_NOT_HERMITIAN:
  case MAT_SYMMETRY_ETERNAL:
  case MAT_NOT_SYMMETRY_ETERNAL:
    break;
  default:
    SETERRQ1(PETSC_ERR_SUP,"unknown option %d",op);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatGetRow_MPIBDiag"
PetscErrorCode MatGetRow_MPIBDiag(Mat matin,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPIBDiag   *mat = (Mat_MPIBDiag*)matin->data;
  PetscErrorCode ierr;
  PetscInt       lrow;

  PetscFunctionBegin;
  if (row < matin->rmap.rstart || row >= matin->rmap.rend) SETERRQ(PETSC_ERR_SUP,"only for local rows")
  lrow = row - matin->rmap.rstart;
  ierr = MatGetRow_SeqBDiag(mat->A,lrow,nz,idx,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRow_MPIBDiag"
PetscErrorCode MatRestoreRow_MPIBDiag(Mat matin,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPIBDiag   *mat = (Mat_MPIBDiag*)matin->data;
  PetscErrorCode ierr;
  PetscInt       lrow;

  PetscFunctionBegin;
  lrow = row - matin->rmap.rstart;
  ierr = MatRestoreRow_SeqBDiag(mat->A,lrow,nz,idx,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatNorm_MPIBDiag"
PetscErrorCode MatNorm_MPIBDiag(Mat A,NormType type,PetscReal *nrm)
{
  Mat_MPIBDiag   *mbd = (Mat_MPIBDiag*)A->data;
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)mbd->A->data;
  PetscReal      sum = 0.0;
  PetscErrorCode ierr;
  PetscInt       d,i,nd = a->nd,bs = A->rmap.bs,len;
  PetscScalar    *dv;

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
    ierr = MPI_Allreduce(&sum,nrm,1,MPIU_REAL,MPI_SUM,A->comm);CHKERRQ(ierr);
    *nrm = sqrt(*nrm);
    ierr = PetscLogFlops(2*A->rmap.n*A->rmap.N);CHKERRQ(ierr);
  } else if (type == NORM_1) { /* max column norm */
    PetscReal *tmp,*tmp2;
    PetscInt    j;
    ierr = PetscMalloc((mbd->A->cmap.n+1)*sizeof(PetscReal),&tmp);CHKERRQ(ierr);
    ierr = PetscMalloc((mbd->A->cmap.n+1)*sizeof(PetscReal),&tmp2);CHKERRQ(ierr);
    ierr = MatNorm_SeqBDiag_Columns(mbd->A,tmp,mbd->A->cmap.n);CHKERRQ(ierr);
    *nrm = 0.0;
    ierr = MPI_Allreduce(tmp,tmp2,mbd->A->cmap.n,MPIU_REAL,MPI_SUM,A->comm);CHKERRQ(ierr);
    for (j=0; j<mbd->A->cmap.n; j++) {
      if (tmp2[j] > *nrm) *nrm = tmp2[j];
    }
    ierr = PetscFree(tmp);CHKERRQ(ierr);
    ierr = PetscFree(tmp2);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) { /* max row norm */
    PetscReal normtemp;
    ierr = MatNorm(mbd->A,type,&normtemp);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&normtemp,nrm,1,MPIU_REAL,MPI_MAX,A->comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_MPIBDiag"
PetscErrorCode MatScale_MPIBDiag(Mat A,PetscScalar alpha)
{
  PetscErrorCode ierr;
  Mat_MPIBDiag   *a = (Mat_MPIBDiag*)A->data;

  PetscFunctionBegin;
  ierr = MatScale_SeqBDiag(a->A,alpha);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpPreallocation_MPIBDiag"
PetscErrorCode MatSetUpPreallocation_MPIBDiag(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr =  MatMPIBDiagSetPreallocation(A,PETSC_DEFAULT,PETSC_DEFAULT,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/

static struct _MatOps MatOps_Values = {MatSetValues_MPIBDiag,
       MatGetRow_MPIBDiag,
       MatRestoreRow_MPIBDiag,
       MatMult_MPIBDiag,
/* 4*/ MatMultAdd_MPIBDiag,
       MatMultTranspose_MPIBDiag,
       MatMultTransposeAdd_MPIBDiag,
       0,
       0,
       0,
/*10*/ 0,
       0,
       0,
       0,
       0,
/*15*/ MatGetInfo_MPIBDiag,
       0,
       MatGetDiagonal_MPIBDiag,
       0,
       MatNorm_MPIBDiag,
/*20*/ MatAssemblyBegin_MPIBDiag,
       MatAssemblyEnd_MPIBDiag,
       0,
       MatSetOption_MPIBDiag,
       MatZeroEntries_MPIBDiag,
/*25*/ MatZeroRows_MPIBDiag,
       0,
       0,
       0,
       0,
/*30*/ MatSetUpPreallocation_MPIBDiag,
       0,
       0,
       0,
       0,
/*35*/ 0,
       0,
       0,
       0,
       0,
/*40*/ 0,
       0,
       0,
       MatGetValues_MPIBDiag,
       0,
/*45*/ 0,
       MatScale_MPIBDiag,
       0,
       0,
       0,
/*50*/ 0,
       0,
       0,
       0,
       0,
/*55*/ 0,
       0,
       0,
       0,
       0,
/*60*/ 0,
       MatDestroy_MPIBDiag,
       MatView_MPIBDiag,
       0,
       0,
/*65*/ 0,
       0,
       0,
       0,
       0,
/*70*/ 0,
       0,
       0,
       0,
       0,
/*75*/ 0,
       0,
       0,
       0,
       0,
/*80*/ 0,
       0,
       0,
       0,
       MatLoad_MPIBDiag,
/*85*/ 0,
       0,
       0,
       0,
       0,
/*90*/ 0,
       0,
       0,
       0,
       0,
/*95*/ 0,
       0,
       0,
       0};

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonalBlock_MPIBDiag"
PetscErrorCode PETSCMAT_DLLEXPORT MatGetDiagonalBlock_MPIBDiag(Mat A,PetscTruth *iscopy,MatReuse reuse,Mat *a)
{
  Mat_MPIBDiag   *matin = (Mat_MPIBDiag *)A->data;
  PetscErrorCode ierr;
  PetscInt       lrows,lcols,rstart,rend;
  IS             localc,localr;

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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMPIBDiagSetPreallocation_MPIBDiag"
PetscErrorCode PETSCMAT_DLLEXPORT MatMPIBDiagSetPreallocation_MPIBDiag(Mat B,PetscInt nd,PetscInt bs,PetscInt *diag,PetscScalar **diagv)
{
  Mat_MPIBDiag   *b;
  PetscErrorCode ierr;
  PetscInt       i,k,*ldiag,len,nd2;
  PetscScalar    **ldiagv = 0;
  PetscTruth     flg2;

  PetscFunctionBegin;
  B->preallocated = PETSC_TRUE;
  if (bs == PETSC_DEFAULT) bs = 1;
  if (nd == PETSC_DEFAULT) nd = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mat_bdiag_ndiag",&nd,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-mat_bdiag_diags",&flg2);CHKERRQ(ierr);
  if (nd && !diag) {
    ierr = PetscMalloc(nd*sizeof(PetscInt),&diag);CHKERRQ(ierr);
    nd2  = nd;
    ierr = PetscOptionsGetIntArray(PETSC_NULL,"-mat_bdiag_dvals",diag,&nd2,PETSC_NULL);CHKERRQ(ierr);
    if (nd2 != nd) {
      SETERRQ(PETSC_ERR_ARG_INCOMP,"Incompatible number of diags and diagonal vals");
    }
  } else if (flg2) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Must specify number of diagonals with -mat_bdiag_ndiag");
  }

  if (bs <= 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Blocksize must be positive");

  B->rmap.bs = B->cmap.bs = bs;

  ierr = PetscMapInitialize(B->comm,&B->rmap);CHKERRQ(ierr);
  ierr = PetscMapInitialize(B->comm,&B->cmap);CHKERRQ(ierr);

  if ((B->cmap.N%bs)) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size - bad column number");
  if ((B->rmap.N%bs)) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size - bad local row number");
  if ((B->rmap.N%bs)) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size - bad global row number");


  b          = (Mat_MPIBDiag*)B->data;CHKERRQ(ierr);
  b->gnd     = nd;
  b->brstart = (B->rmap.rstart)/bs;
  b->brend   = (B->rmap.rend)/bs;


  /* Determine local diagonals; for now, assume global rows = global cols */
  /* These are sorted in MatCreateSeqBDiag */
  ierr = PetscMalloc((nd+1)*sizeof(PetscInt),&ldiag);CHKERRQ(ierr); 
  len  = B->rmap.N/bs + B->cmap.N/bs + 1;
  ierr = PetscMalloc(len*sizeof(PetscInt),&b->gdiag);CHKERRQ(ierr);
  k    = 0;
  if (diagv) {
    ierr = PetscMalloc((nd+1)*sizeof(PetscScalar*),&ldiagv);CHKERRQ(ierr); 
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
      if (B->rmap.N/bs - diag[i] > B->cmap.N/bs) {
        if (B->rmap.N/bs + diag[i] > b->brstart) {
          ldiag[k] = diag[i] - b->brstart;
          if (diagv) ldiagv[k] = diagv[i];
          k++;
        }
      } else {
        if (B->rmap.N/bs > b->brstart) {
          ldiag[k] = diag[i] - b->brstart;
          if (diagv) ldiagv[k] = diagv[i];
          k++;
        }
      }
    }
  }

  /* Form local matrix */
  ierr = MatCreate(PETSC_COMM_SELF,&b->A);CHKERRQ(ierr);
  ierr = MatSetSizes(b->A,B->rmap.n,B->cmap.N,B->rmap.n,B->cmap.N);CHKERRQ(ierr);
  ierr = MatSetType(b->A,MATSEQBDIAG);CHKERRQ(ierr);
  ierr = MatSeqBDiagSetPreallocation(b->A,k,bs,ldiag,ldiagv);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(B,b->A);CHKERRQ(ierr);
  ierr = PetscFree(ldiag);CHKERRQ(ierr);
  ierr = PetscFree(ldiagv);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
   MATMPIBDIAG - MATMPIBDIAG = "mpibdiag" - A matrix type to be used for distributed block diagonal matrices.

   Options Database Keys:
. -mat_type mpibdiag - sets the matrix type to "mpibdiag" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateMPIBDiag
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_MPIBDiag"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPIBDiag(Mat B)
{
  Mat_MPIBDiag   *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr            = PetscNew(Mat_MPIBDiag,&b);CHKERRQ(ierr);
  B->data         = (void*)b;
  ierr            = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->factor       = 0;
  B->mapping      = 0;

  B->insertmode = NOT_SET_VALUES;
  ierr = MPI_Comm_rank(B->comm,&b->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(B->comm,&b->size);CHKERRQ(ierr);

  /* build cache for off array entries formed */
  ierr = MatStashCreate_Private(B->comm,1,&B->stash);CHKERRQ(ierr);
  b->donotstash = PETSC_FALSE;

  /* stuff used for matrix-vector multiply */
  b->lvec        = 0;
  b->Mvctx       = 0;

  /* used for MatSetValues() input */
  b->roworiented = PETSC_TRUE;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetDiagonalBlock_C",
                                     "MatGetDiagonalBlock_MPIBDiag",
                                      MatGetDiagonalBlock_MPIBDiag);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMPIBDiagSetPreallocation_C",
                                     "MatMPIBDiagSetPreallocation_MPIBDiag",
                                      MatMPIBDiagSetPreallocation_MPIBDiag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
   MATBDIAG - MATBDIAG = "bdiag" - A matrix type to be used for block diagonal matrices.

   This matrix type is identical to MATSEQBDIAG when constructed with a single process communicator,
   and MATMPIBDIAG otherwise.

   Options Database Keys:
. -mat_type bdiag - sets the matrix type to "bdiag" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateMPIBDiag,MATSEQBDIAG,MATMPIBDIAG
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_BDiag"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_BDiag(Mat A) 
{
  PetscErrorCode ierr;
  PetscMPIInt   size;

  PetscFunctionBegin;
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATBDIAG);CHKERRQ(ierr);
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(A,MATSEQBDIAG);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(A,MATMPIBDIAG);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatMPIBDiagSetPreallocation"
/*@C
   MatMPIBDiagSetPreallocation - 

   Collective on Mat

   Input Parameters:
+  A - the matrix 
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
PetscErrorCode PETSCMAT_DLLEXPORT MatMPIBDiagSetPreallocation(Mat B,PetscInt nd,PetscInt bs,const PetscInt diag[],PetscScalar *diagv[])
{
  PetscErrorCode ierr,(*f)(Mat,PetscInt,PetscInt,const PetscInt[],PetscScalar*[]);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)B,"MatMPIBDiagSetPreallocation_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(B,nd,bs,diag,diagv);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateMPIBDiag"
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
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMPIBDiag(MPI_Comm comm,PetscInt m,PetscInt M,PetscInt N,PetscInt nd,PetscInt bs,const PetscInt diag[],PetscScalar *diagv[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,m,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIBDIAG);CHKERRQ(ierr);
    ierr = MatMPIBDiagSetPreallocation(*A,nd,bs,diag,diagv);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQBDIAG);CHKERRQ(ierr);
    ierr = MatSeqBDiagSetPreallocation(*A,nd,bs,diag,diagv);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatBDiagGetData"
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
PetscErrorCode PETSCMAT_DLLEXPORT MatBDiagGetData(Mat mat,PetscInt *nd,PetscInt *bs,PetscInt *diag[],PetscInt *bdlen[],PetscScalar ***diagv)
{
  Mat_MPIBDiag   *pdmat;
  Mat_SeqBDiag   *dmat = 0;
  PetscTruth     isseq,ismpi;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  ierr = PetscTypeCompare((PetscObject)mat,MATSEQBDIAG,&isseq);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)mat,MATMPIBDIAG,&ismpi);CHKERRQ(ierr);
  if (isseq) {
    dmat = (Mat_SeqBDiag*)mat->data;
  } else if (ismpi) {
    pdmat = (Mat_MPIBDiag*)mat->data;
    dmat = (Mat_SeqBDiag*)pdmat->A->data;
  } else SETERRQ(PETSC_ERR_SUP,"Valid only for MATSEQBDIAG and MATMPIBDIAG formats");
  *nd    = dmat->nd;
  *bs    = mat->rmap.bs;
  *diag  = dmat->diag;
  *bdlen = dmat->bdlen;
  *diagv = dmat->diagv;
  PetscFunctionReturn(0);
}

#include "petscsys.h"

#undef __FUNCT__  
#define __FUNCT__ "MatLoad_MPIBDiag"
PetscErrorCode MatLoad_MPIBDiag(PetscViewer viewer, MatType type,Mat *newmat)
{
  Mat            A;
  PetscScalar    *vals,*svals;
  MPI_Comm       comm = ((PetscObject)viewer)->comm;
  MPI_Status     status;
  PetscErrorCode ierr;
  int            fd;
  PetscMPIInt    tag = ((PetscObject)viewer)->tag,rank,size,*sndcounts = 0,*rowners,maxnz,mm;
  PetscInt       bs,i,nz,j,rstart,rend,*cols;
  PetscInt       header[4],*rowlengths = 0,M,N,m,Mbs;
  PetscInt       *ourlens,*procsnz = 0,jj,*mycols,*smycols;
  PetscInt       extra_rows;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,PETSC_INT);CHKERRQ(ierr);
    if (header[0] != MAT_FILE_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"not matrix object");
    if (header[3] < 0) {
      SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"Matrix stored in special format,cannot load as MPIBDiag");
    }
  }
  ierr = MPI_Bcast(header+1,3,MPIU_INT,0,comm);CHKERRQ(ierr);
  M = header[1]; N = header[2];

  bs = 1;   /* uses a block size of 1 by default; */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-matload_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);

  /* 
     This code adds extra rows to make sure the number of rows is 
     divisible by the blocksize
  */
  Mbs        = M/bs;
  extra_rows = bs - M + bs*(Mbs);
  if (extra_rows == bs) extra_rows = 0;
  else                  Mbs++;
  if (extra_rows && !rank) {
    ierr = PetscInfo(0,"Padding loaded matrix to match blocksize\n");CHKERRQ(ierr);
  }

  /* determine ownership of all rows */
  m          = bs*(Mbs/size + ((Mbs % size) > rank));
  ierr       = PetscMalloc((size+2)*sizeof(PetscInt),&rowners);CHKERRQ(ierr);
  mm         = (PetscMPIInt)m;
  ierr       = MPI_Allgather(&mm,1,MPI_INT,rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  rowners[0] = 0;
  for (i=2; i<=size; i++) {
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 

  /* distribute row lengths to all processors */
  ierr = PetscMalloc((rend-rstart)*sizeof(PetscInt),&ourlens);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscMalloc((M+extra_rows)*sizeof(PetscInt),&rowlengths);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);
    for (i=0; i<extra_rows; i++) rowlengths[M+i] = 1;
    ierr = PetscMalloc(size*sizeof(PetscMPIInt),&sndcounts);CHKERRQ(ierr);
    for (i=0; i<size; i++) sndcounts[i] = rowners[i+1] - rowners[i];
    ierr = MPI_Scatterv(rowlengths,sndcounts,rowners,MPIU_INT,ourlens,rend-rstart,MPIU_INT,0,comm);CHKERRQ(ierr);
    ierr = PetscFree(sndcounts);CHKERRQ(ierr);
  } else {
    ierr = MPI_Scatterv(0,0,0,MPIU_INT,ourlens,rend-rstart,MPIU_INT,0,comm);CHKERRQ(ierr);
  }

  if (!rank) {
    /* calculate the number of nonzeros on each processor */
    ierr = PetscMalloc(size*sizeof(PetscInt),&procsnz);CHKERRQ(ierr);
    ierr = PetscMemzero(procsnz,size*sizeof(PetscInt));CHKERRQ(ierr);
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
    ierr = PetscMalloc(maxnz*sizeof(PetscInt),&cols);CHKERRQ(ierr);

    /* read in my part of the matrix column indices  */
    nz   = procsnz[0];
    ierr = PetscMalloc(nz*sizeof(PetscInt),&mycols);CHKERRQ(ierr);
    if (size == 1)  nz -= extra_rows;
    ierr = PetscBinaryRead(fd,mycols,nz,PETSC_INT);CHKERRQ(ierr);
    if (size == 1)  for (i=0; i<extra_rows; i++) { mycols[nz+i] = M+i; }

    /* read in every one elses and ship off */
    for (i=1; i<size-1; i++) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
      ierr = MPI_Send(cols,nz,MPIU_INT,i,tag,comm);CHKERRQ(ierr);
    }
    /* read in the stuff for the last proc */
    if (size != 1) {
      nz   = procsnz[size-1] - extra_rows;  /* the extra rows are not on the disk */
      ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
      for (i=0; i<extra_rows; i++) cols[nz+i] = M+i;
      ierr = MPI_Send(cols,nz+extra_rows,MPIU_INT,size-1,tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(cols);CHKERRQ(ierr);
  } else {
    /* determine buffer space needed for message */
    nz = 0;
    for (i=0; i<m; i++) {
      nz += ourlens[i];
    }
    ierr = PetscMalloc(nz*sizeof(PetscInt),&mycols);CHKERRQ(ierr);

    /* receive message of column indices*/
    ierr = MPI_Recv(mycols,nz,MPIU_INT,0,tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPIU_INT,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"something is wrong with file");
  }

  ierr = MatCreate(comm,newmat);CHKERRQ(ierr);
  ierr = MatSetSizes(*newmat,m,m,M+extra_rows,N+extra_rows);CHKERRQ(ierr);
  ierr = MatSetType(*newmat,type);CHKERRQ(ierr);
  ierr = MatMPIBDiagSetPreallocation(*newmat,0,bs,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  A = *newmat;

  if (!rank) {
    ierr = PetscMalloc(maxnz*sizeof(PetscScalar),&vals);CHKERRQ(ierr);

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
    ierr = PetscMalloc(nz*sizeof(PetscScalar),&vals);CHKERRQ(ierr);

    /* receive message of values*/
    ierr = MPI_Recv(vals,nz,MPIU_SCALAR,0,A->tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPIU_SCALAR,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"something is wrong with file");

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






