#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mpidense.c,v 1.109 1999/03/16 21:10:18 balay Exp balay $";
#endif

/*
   Basic functions for basic parallel dense matrices.
*/
    
#include "src/mat/impls/dense/mpi/mpidense.h"
#include "src/vec/vecimpl.h"

extern int MatSetUpMultiply_MPIDense(Mat);

#undef __FUNC__  
#define __FUNC__ "MatSetValues_MPIDense"
int MatSetValues_MPIDense(Mat mat,int m,int *idxm,int n,int *idxn,Scalar *v,InsertMode addv)
{
  Mat_MPIDense *A = (Mat_MPIDense *) mat->data;
  int          ierr, i, j, rstart = A->rstart, rend = A->rend, row;
  int          roworiented = A->roworiented;

  PetscFunctionBegin;
  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) continue;
    if (idxm[i] >= A->M) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      if (roworiented) {
        ierr = MatSetValues(A->A,1,&row,n,idxn,v+i*n,addv); CHKERRQ(ierr);
      } else {
        for ( j=0; j<n; j++ ) {
          if (idxn[j] < 0) continue;
          if (idxn[j] >= A->N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");
          ierr = MatSetValues(A->A,1,&row,1,&idxn[j],v+i+j*m,addv); CHKERRQ(ierr);
        }
      }
    } else {
      if (roworiented) {
        ierr = MatStashValuesRow_Private(&mat->stash,idxm[i],n,idxn,v+i*n); CHKERRQ(ierr);
      } else {
        ierr = MatStashValuesCol_Private(&mat->stash,idxm[i],n,idxn,v+i,m);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetValues_MPIDense"
int MatGetValues_MPIDense(Mat mat,int m,int *idxm,int n,int *idxn,Scalar *v)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr, i, j, rstart = mdn->rstart, rend = mdn->rend, row;

  PetscFunctionBegin;
  for ( i=0; i<m; i++ ) {
    if (idxm[i] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative row");
    if (idxm[i] >= mdn->M) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for ( j=0; j<n; j++ ) {
        if (idxn[j] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative column");
        if (idxn[j] >= mdn->N) {
          SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");
        }
        ierr = MatGetValues(mdn->A,1,&row,1,&idxn[j],v+i*n+j); CHKERRQ(ierr);
      }
    } else {
      SETERRQ(PETSC_ERR_SUP,0,"Only local values currently supported");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetArray_MPIDense"
int MatGetArray_MPIDense(Mat A,Scalar **array)
{
  Mat_MPIDense *a = (Mat_MPIDense *) A->data;
  int          ierr;

  PetscFunctionBegin;
  ierr = MatGetArray(a->A,array); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreArray_MPIDense"
int MatRestoreArray_MPIDense(Mat A,Scalar **array)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatAssemblyBegin_MPIDense"
int MatAssemblyBegin_MPIDense(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  MPI_Comm     comm = mat->comm;
  int          ierr,nstash,reallocs;
  InsertMode   addv;

  PetscFunctionBegin;
  /* make sure all processors are either in INSERTMODE or ADDMODE */
  ierr = MPI_Allreduce(&mat->insertmode,&addv,1,MPI_INT,MPI_BOR,comm);CHKERRQ(ierr);
  if (addv == (ADD_VALUES|INSERT_VALUES)) { 
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Cannot mix adds/inserts on different procs");
  }
  mat->insertmode = addv; /* in case this processor had no cache */

  ierr = MatStashScatterBegin_Private(&mat->stash,mdn->rowners); CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs); CHKERRQ(ierr);
  PLogInfo(mdn->A,"MatAssemblyBegin_MPIDense:Stash has %d entries, uses %d mallocs.\n",
           nstash,reallocs);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatAssemblyEnd_MPIDense"
int MatAssemblyEnd_MPIDense(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIDense *mdn=(Mat_MPIDense*)mat->data;
  int          i,n,ierr,*row,*col,flg,j,rstart,ncols;
  Scalar       *val;
  InsertMode   addv=mat->insertmode;

  PetscFunctionBegin;
  /*  wait on receives */
  while (1) {
    ierr = MatStashScatterGetMesg_Private(&mat->stash,&n,&row,&col,&val,&flg); CHKERRQ(ierr);
    if (!flg) break;
    
    for ( i=0; i<n; ) {
      /* Now identify the consecutive vals belonging to the same row */
      for ( j=i,rstart=row[j]; j<n; j++ ) { if (row[j] != rstart) break; }
      if (j < n) ncols = j-i;
      else       ncols = n-i;
      /* Now assemble all these values with a single function call */
      ierr = MatSetValues_MPIDense(mat,1,row+i,ncols,col+i,val+i,addv); CHKERRQ(ierr);
      i = j;
    }
  }
  ierr = MatStashScatterEnd_Private(&mat->stash); CHKERRQ(ierr);
  
  ierr = MatAssemblyBegin(mdn->A,mode); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mdn->A,mode); CHKERRQ(ierr);

  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIDense(mat); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatZeroEntries_MPIDense"
int MatZeroEntries_MPIDense(Mat A)
{
  int          ierr;
  Mat_MPIDense *l = (Mat_MPIDense *) A->data;

  PetscFunctionBegin;
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetBlockSize_MPIDense"
int MatGetBlockSize_MPIDense(Mat A,int *bs)
{
  PetscFunctionBegin;
  *bs = 1;
  PetscFunctionReturn(0);
}

/* the code does not do the diagonal entries correctly unless the 
   matrix is square and the column and row owerships are identical.
   This is a BUG. The only way to fix it seems to be to access 
   mdn->A and mdn->B directly and not through the MatZeroRows() 
   routine. 
*/
#undef __FUNC__  
#define __FUNC__ "MatZeroRows_MPIDense"
int MatZeroRows_MPIDense(Mat A,IS is,Scalar *diag)
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

  PetscFunctionBegin;
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
    if (!found) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Index out of range");
  }
  nsends = 0;  for ( i=0; i<size; i++ ) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  work   = (int *) PetscMalloc( size*sizeof(int) ); CHKPTRQ(work);
  ierr   = MPI_Allreduce( procs, work,size,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  nrecvs = work[rank]; 
  ierr   = MPI_Allreduce( nprocs, work,size,MPI_INT,MPI_MAX,comm);CHKERRQ(ierr);
  nmax   = work[rank];
  PetscFree(work);

  /* post receives:   */
  rvalues    = (int *) PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int));CHKPTRQ(rvalues);
  recv_waits = (MPI_Request *) PetscMalloc((nrecvs+1)*sizeof(MPI_Request));CHKPTRQ(recv_waits);
  for ( i=0; i<nrecvs; i++ ) {
    ierr = MPI_Irecv(rvalues+nmax*i,nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  svalues    = (int *) PetscMalloc( (N+1)*sizeof(int) ); CHKPTRQ(svalues);
  send_waits = (MPI_Request *) PetscMalloc((nsends+1)*sizeof(MPI_Request));CHKPTRQ(send_waits);
  starts     = (int *) PetscMalloc( (size+1)*sizeof(int) ); CHKPTRQ(starts);
  starts[0]  = 0; 
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
      ierr = MPI_Isend(svalues+starts[i],nprocs[i],MPI_INT,i,tag,comm,send_waits+count++);CHKERRQ(ierr);
    }
  }
  PetscFree(starts);

  base = owners[rank];

  /*  wait on receives */
  lens   = (int *) PetscMalloc( 2*(nrecvs+1)*sizeof(int) ); CHKPTRQ(lens);
  source = lens + nrecvs;
  count  = nrecvs; slen = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPI_INT,&n);CHKERRQ(ierr);
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
  ierr = ISCreateGeneral(PETSC_COMM_SELF,slen,lrows,&istmp);CHKERRQ(ierr);   
  PLogObjectParent(A,istmp);
  PetscFree(lrows);
  ierr = MatZeroRows(l->A,istmp,diag); CHKERRQ(ierr);
  ierr = ISDestroy(istmp); CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status *) PetscMalloc(nsends*sizeof(MPI_Status));CHKPTRQ(send_status);
    ierr        = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    PetscFree(send_status);
  }
  PetscFree(send_waits); PetscFree(svalues);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMult_MPIDense"
int MatMult_MPIDense(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(xx,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD,mdn->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD,mdn->Mvctx);CHKERRQ(ierr);
  ierr = MatMult_SeqDense(mdn->A,mdn->lvec,yy); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMultAdd_MPIDense"
int MatMultAdd_MPIDense(Mat mat,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(xx,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD,mdn->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD,mdn->Mvctx);CHKERRQ(ierr);
  ierr = MatMultAdd_SeqDense(mdn->A,mdn->lvec,yy,zz); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMultTrans_MPIDense"
int MatMultTrans_MPIDense(Mat A,Vec xx,Vec yy)
{
  Mat_MPIDense *a = (Mat_MPIDense *) A->data;
  int          ierr;
  Scalar       zero = 0.0;

  PetscFunctionBegin;
  ierr = VecSet(&zero,yy); CHKERRQ(ierr);
  ierr = MatMultTrans_SeqDense(a->A,xx,a->lvec); CHKERRQ(ierr);
  ierr = VecScatterBegin(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx); CHKERRQ(ierr);
  ierr = VecScatterEnd(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMultTransAdd_MPIDense"
int MatMultTransAdd_MPIDense(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIDense *a = (Mat_MPIDense *) A->data;
  int          ierr;

  PetscFunctionBegin;
  ierr = VecCopy(yy,zz); CHKERRQ(ierr);
  ierr = MatMultTrans_SeqDense(a->A,xx,a->lvec); CHKERRQ(ierr);
  ierr = VecScatterBegin(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx); CHKERRQ(ierr);
  ierr = VecScatterEnd(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetDiagonal_MPIDense"
int MatGetDiagonal_MPIDense(Mat A,Vec v)
{
  Mat_MPIDense *a = (Mat_MPIDense *) A->data;
  Mat_SeqDense *aloc = (Mat_SeqDense *) a->A->data;
  int          ierr, len, i, n, m = a->m, radd;
  Scalar       *x, zero = 0.0;
  
  PetscFunctionBegin;
  VecSet(&zero,v);
  ierr = VecGetArray(v,&x); CHKERRQ(ierr);
  ierr = VecGetSize(v,&n); CHKERRQ(ierr);
  if (n != a->M) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Nonconforming mat and vec");
  len = PetscMin(aloc->m,aloc->n);
  radd = a->rstart*m;
  for ( i=0; i<len; i++ ) {
    x[i] = aloc->v[radd + i*m + i];
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDestroy_MPIDense"
int MatDestroy_MPIDense(Mat mat)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr;

  PetscFunctionBegin;
  if (--mat->refct > 0) PetscFunctionReturn(0);

  if (mat->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(mat->mapping); CHKERRQ(ierr);
  }
  if (mat->bmapping) {
    ierr = ISLocalToGlobalMappingDestroy(mat->bmapping); CHKERRQ(ierr);
  }
#if defined(USE_PETSC_LOG)
  PLogObjectState((PetscObject)mat,"Rows=%d, Cols=%d",mdn->M,mdn->N);
#endif
  ierr = MatStashDestroy_Private(&mat->stash); CHKERRQ(ierr);
  PetscFree(mdn->rowners); 
  ierr = MatDestroy(mdn->A); CHKERRQ(ierr);
  if (mdn->lvec)   VecDestroy(mdn->lvec);
  if (mdn->Mvctx)  VecScatterDestroy(mdn->Mvctx);
  if (mdn->factor) {
    if (mdn->factor->temp)   PetscFree(mdn->factor->temp);
    if (mdn->factor->tag)    PetscFree(mdn->factor->tag);
    if (mdn->factor->pivots) PetscFree(mdn->factor->pivots);
    PetscFree(mdn->factor);
  }
  PetscFree(mdn); 
  if (mat->rmap) {
    ierr = MapDestroy(mat->rmap);CHKERRQ(ierr);
  }
  if (mat->cmap) {
    ierr = MapDestroy(mat->cmap);CHKERRQ(ierr);
  }
  PLogObjectDestroy(mat);
  PetscHeaderDestroy(mat);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_MPIDense_Binary"
static int MatView_MPIDense_Binary(Mat mat,Viewer viewer)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr;

  PetscFunctionBegin;
  if (mdn->size == 1) {
    ierr = MatView(mdn->A,viewer); CHKERRQ(ierr);
  }
  else SETERRQ(PETSC_ERR_SUP,0,"Only uniprocessor output supported");
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_MPIDense_ASCII"
static int MatView_MPIDense_ASCII(Mat mat,Viewer viewer)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) mat->data;
  int          ierr, format, size = mdn->size, rank = mdn->rank; 
  FILE         *fd;
  ViewerType   vtype;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype);CHKERRQ(ierr);
  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);
  if (format == VIEWER_FORMAT_ASCII_INFO_LONG) {
    MatInfo info;
    ierr = MatGetInfo(mat,MAT_LOCAL,&info);
    PetscSequentialPhaseBegin(mat->comm,1);
      fprintf(fd,"  [%d] local rows %d nz %d nz alloced %d mem %d \n",rank,mdn->m,
         (int)info.nz_used,(int)info.nz_allocated,(int)info.memory);       
      fflush(fd);
    PetscSequentialPhaseEnd(mat->comm,1);
    ierr = VecScatterView(mdn->Mvctx,viewer); CHKERRQ(ierr);
    PetscFunctionReturn(0); 
  } else if (format == VIEWER_FORMAT_ASCII_INFO) {
    PetscFunctionReturn(0);
  }

  if (size == 1) { 
    ierr = MatView(mdn->A,viewer); CHKERRQ(ierr);
  } else {
    /* assemble the entire matrix onto first processor. */
    Mat          A;
    int          M = mdn->M, N = mdn->N,m,row,i, nz, *cols;
    Scalar       *vals;
    Mat_SeqDense *Amdn = (Mat_SeqDense*) mdn->A->data;

    if (!rank) {
      ierr = MatCreateMPIDense(mat->comm,M,N,M,N,PETSC_NULL,&A); CHKERRQ(ierr);
    } else {
      ierr = MatCreateMPIDense(mat->comm,0,N,M,N,PETSC_NULL,&A); CHKERRQ(ierr);
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

    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if (!rank) {
      ierr = MatView(((Mat_MPIDense*)(A->data))->A,viewer); CHKERRQ(ierr);
    }
    ierr = MatDestroy(A); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_MPIDense"
int MatView_MPIDense(Mat mat,Viewer viewer)
{
  int          ierr;
  ViewerType   vtype;
 
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
    ierr = MatView_MPIDense_ASCII(mat,viewer); CHKERRQ(ierr);
  } else if (PetscTypeCompare(vtype,BINARY_VIEWER)) {
    ierr = MatView_MPIDense_Binary(mat,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Viewer type not supported by PETSc object");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetInfo_MPIDense"
int MatGetInfo_MPIDense(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_MPIDense *mat = (Mat_MPIDense *) A->data;
  Mat          mdn = mat->A;
  int          ierr;
  double       isend[5], irecv[5];

  PetscFunctionBegin;
  info->rows_global    = (double)mat->M;
  info->columns_global = (double)mat->N;
  info->rows_local     = (double)mat->m;
  info->columns_local  = (double)mat->N;
  info->block_size     = 1.0;
  ierr = MatGetInfo(mdn,MAT_LOCAL,info); CHKERRQ(ierr);
  isend[0] = info->nz_used; isend[1] = info->nz_allocated; isend[2] = info->nz_unneeded;
  isend[3] = info->memory;  isend[4] = info->mallocs;
  if (flag == MAT_LOCAL) {
    info->nz_used      = isend[0];
    info->nz_allocated = isend[1];
    info->nz_unneeded  = isend[2];
    info->memory       = isend[3];
    info->mallocs      = isend[4];
  } else if (flag == MAT_GLOBAL_MAX) {
    ierr = MPI_Allreduce(isend,irecv,5,MPI_DOUBLE,MPI_MAX,A->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPI_Allreduce(isend,irecv,5,MPI_DOUBLE,MPI_SUM,A->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  }
  info->fill_ratio_given  = 0; /* no parallel LU/ILU/Cholesky */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

/* extern int MatLUFactorSymbolic_MPIDense(Mat,IS,IS,double,Mat*);
   extern int MatLUFactorNumeric_MPIDense(Mat,Mat*);
   extern int MatLUFactor_MPIDense(Mat,IS,IS,double);
   extern int MatSolve_MPIDense(Mat,Vec,Vec);
   extern int MatSolveAdd_MPIDense(Mat,Vec,Vec,Vec);
   extern int MatSolveTrans_MPIDense(Mat,Vec,Vec);
   extern int MatSolveTransAdd_MPIDense(Mat,Vec,Vec,Vec); */

#undef __FUNC__  
#define __FUNC__ "MatSetOption_MPIDense"
int MatSetOption_MPIDense(Mat A,MatOption op)
{
  Mat_MPIDense *a = (Mat_MPIDense *) A->data;

  PetscFunctionBegin;
  if (op == MAT_NO_NEW_NONZERO_LOCATIONS ||
      op == MAT_YES_NEW_NONZERO_LOCATIONS ||
      op == MAT_NEW_NONZERO_LOCATION_ERR ||
      op == MAT_NEW_NONZERO_ALLOCATION_ERR ||
      op == MAT_COLUMNS_SORTED ||
      op == MAT_COLUMNS_UNSORTED) {
        MatSetOption(a->A,op);
  } else if (op == MAT_ROW_ORIENTED) {
        a->roworiented = 1;
        MatSetOption(a->A,op);
  } else if (op == MAT_ROWS_SORTED || 
             op == MAT_ROWS_UNSORTED ||
             op == MAT_SYMMETRIC ||
             op == MAT_STRUCTURALLY_SYMMETRIC ||
             op == MAT_YES_NEW_DIAGONALS ||
             op == MAT_USE_HASH_TABLE) {
    PLogInfo(A,"MatSetOption_MPIDense:Option ignored\n");
  } else if (op == MAT_COLUMN_ORIENTED) {
    a->roworiented = 0; MatSetOption(a->A,op);
  } else if (op == MAT_NO_NEW_DIAGONALS) {
    SETERRQ(PETSC_ERR_SUP,0,"MAT_NO_NEW_DIAGONALS");
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"unknown option");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSize_MPIDense"
int MatGetSize_MPIDense(Mat A,int *m,int *n)
{
  Mat_MPIDense *mat = (Mat_MPIDense *) A->data;

  PetscFunctionBegin;
  *m = mat->M; *n = mat->N;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetLocalSize_MPIDense"
int MatGetLocalSize_MPIDense(Mat A,int *m,int *n)
{
  Mat_MPIDense *mat = (Mat_MPIDense *) A->data;

  PetscFunctionBegin;
  *m = mat->m; *n = mat->N;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetOwnershipRange_MPIDense"
int MatGetOwnershipRange_MPIDense(Mat A,int *m,int *n)
{
  Mat_MPIDense *mat = (Mat_MPIDense *) A->data;

  PetscFunctionBegin;
  *m = mat->rstart; *n = mat->rend;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetRow_MPIDense"
int MatGetRow_MPIDense(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPIDense *mat = (Mat_MPIDense *) A->data;
  int          lrow, rstart = mat->rstart, rend = mat->rend,ierr;

  PetscFunctionBegin;
  if (row < rstart || row >= rend) SETERRQ(PETSC_ERR_SUP,0,"only local rows")
  lrow = row - rstart;
  ierr = MatGetRow(mat->A,lrow,nz,idx,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRow_MPIDense"
int MatRestoreRow_MPIDense(Mat mat,int row,int *nz,int **idx,Scalar **v)
{
  PetscFunctionBegin;
  if (idx) PetscFree(*idx);
  if (v) PetscFree(*v);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatNorm_MPIDense"
int MatNorm_MPIDense(Mat A,NormType type,double *norm)
{
  Mat_MPIDense *mdn = (Mat_MPIDense *) A->data;
  Mat_SeqDense *mat = (Mat_SeqDense*) mdn->A->data;
  int          ierr, i, j;
  double       sum = 0.0;
  Scalar       *v = mat->v;

  PetscFunctionBegin;
  if (mdn->size == 1) {
    ierr =  MatNorm(mdn->A,type,norm); CHKERRQ(ierr);
  } else {
    if (type == NORM_FROBENIUS) {
      for (i=0; i<mat->n*mat->m; i++ ) {
#if defined(USE_PETSC_COMPLEX)
        sum += PetscReal(PetscConj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      ierr = MPI_Allreduce(&sum,norm,1,MPI_DOUBLE,MPI_SUM,A->comm);CHKERRQ(ierr);
      *norm = sqrt(*norm);
      PLogFlops(2*mat->n*mat->m);
    } else if (type == NORM_1) { 
      double *tmp, *tmp2;
      tmp  = (double *) PetscMalloc( 2*mdn->N*sizeof(double) ); CHKPTRQ(tmp);
      tmp2 = tmp + mdn->N;
      PetscMemzero(tmp,2*mdn->N*sizeof(double));
      *norm = 0.0;
      v = mat->v;
      for ( j=0; j<mat->n; j++ ) {
        for ( i=0; i<mat->m; i++ ) {
          tmp[j] += PetscAbsScalar(*v);  v++;
        }
      }
      ierr = MPI_Allreduce(tmp,tmp2,mdn->N,MPI_DOUBLE,MPI_SUM,A->comm);CHKERRQ(ierr);
      for ( j=0; j<mdn->N; j++ ) {
        if (tmp2[j] > *norm) *norm = tmp2[j];
      }
      PetscFree(tmp);
      PLogFlops(mat->n*mat->m);
    } else if (type == NORM_INFINITY) { /* max row norm */
      double ntemp;
      ierr = MatNorm(mdn->A,type,&ntemp); CHKERRQ(ierr);
      ierr = MPI_Allreduce(&ntemp,norm,1,MPI_DOUBLE,MPI_MAX,A->comm);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,0,"No support for two norm");
    }
  }
  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ "MatTranspose_MPIDense"
int MatTranspose_MPIDense(Mat A,Mat *matout)
{ 
  Mat_MPIDense *a = (Mat_MPIDense *) A->data;
  Mat_SeqDense *Aloc = (Mat_SeqDense *) a->A->data;
  Mat          B;
  int          M = a->M, N = a->N, m, n, *rwork, rstart = a->rstart;
  int          j, i, ierr;
  Scalar       *v;

  PetscFunctionBegin;
  if (matout == PETSC_NULL && M != N) {
    SETERRQ(PETSC_ERR_SUP,0,"Supports square matrix only in-place");
  }
  ierr = MatCreateMPIDense(A->comm,PETSC_DECIDE,PETSC_DECIDE,N,M,PETSC_NULL,&B);CHKERRQ(ierr);

  m = Aloc->m; n = Aloc->n; v = Aloc->v;
  rwork = (int *) PetscMalloc(n*sizeof(int)); CHKPTRQ(rwork);
  for ( j=0; j<n; j++ ) {
    for (i=0; i<m; i++) rwork[i] = rstart + i;
    ierr = MatSetValues(B,1,&j,m,rwork,v,INSERT_VALUES); CHKERRQ(ierr);
    v   += m;
  } 
  PetscFree(rwork);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  if (matout != PETSC_NULL) {
    *matout = B;
  } else {
    PetscOps *Abops;
    MatOps   Aops;

    /* This isn't really an in-place transpose, but free data struct from a */
    PetscFree(a->rowners); 
    ierr = MatDestroy(a->A); CHKERRQ(ierr);
    if (a->lvec) VecDestroy(a->lvec);
    if (a->Mvctx) VecScatterDestroy(a->Mvctx);
    PetscFree(a);
 
    /*
         This is horrible, horrible code. We need to keep the 
      A pointers for the bops and ops but copy everything 
      else from C.
    */
    Abops = A->bops;
    Aops  = A->ops;
    PetscMemcpy(A,B,sizeof(struct _p_Mat));
    A->bops = Abops;
    A->ops  = Aops;

    PetscHeaderDestroy(B);
  }
  PetscFunctionReturn(0);
}

#include "pinclude/blaslapack.h"
#undef __FUNC__  
#define __FUNC__ "MatScale_MPIDense"
int MatScale_MPIDense(Scalar *alpha,Mat inA)
{
  Mat_MPIDense *A = (Mat_MPIDense *) inA->data;
  Mat_SeqDense *a = (Mat_SeqDense *) A->A->data;
  int          one = 1, nz;

  PetscFunctionBegin;
  nz = a->m*a->n;
  BLscal_( &nz, alpha, a->v, &one );
  PLogFlops(nz);
  PetscFunctionReturn(0);
}

static int MatDuplicate_MPIDense(Mat,MatDuplicateOption,Mat *);
extern int MatGetSubMatrices_MPIDense(Mat,int,IS *,IS *,MatReuse,Mat **);

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_MPIDense,
       MatGetRow_MPIDense,
       MatRestoreRow_MPIDense,
       MatMult_MPIDense,
       MatMultAdd_MPIDense,
       MatMultTrans_MPIDense,
       MatMultTransAdd_MPIDense,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       MatTranspose_MPIDense,
       MatGetInfo_MPIDense,0,
       MatGetDiagonal_MPIDense,
       0,
       MatNorm_MPIDense,
       MatAssemblyBegin_MPIDense,
       MatAssemblyEnd_MPIDense,
       0,
       MatSetOption_MPIDense,
       MatZeroEntries_MPIDense,
       MatZeroRows_MPIDense,
       0,
       0,
       0,
       0,
       MatGetSize_MPIDense,
       MatGetLocalSize_MPIDense,
       MatGetOwnershipRange_MPIDense,
       0,
       0, 
       MatGetArray_MPIDense, 
       MatRestoreArray_MPIDense,
       MatDuplicate_MPIDense,
       0,
       0,
       0,
       0,
       0,
       MatGetSubMatrices_MPIDense,
       0,
       MatGetValues_MPIDense,
       0,
       0,
       MatScale_MPIDense,
       0,
       0,
       0,
       MatGetBlockSize_MPIDense,
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

#undef __FUNC__  
#define __FUNC__ "MatCreateMPIDense"
/*@C
   MatCreateMPIDense - Creates a sparse parallel matrix in dense format.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
.  n - number of local columns (or PETSC_DECIDE to have calculated if N is given)
.  M - number of global rows (or PETSC_DECIDE to have calculated if m is given)
.  N - number of global columns (or PETSC_DECIDE to have calculated if n is given)
-  data - optional location of matrix data.  Set data=PETSC_NULL for PETSc
   to control all matrix memory allocation.

   Output Parameter:
.  A - the matrix

   Notes:
   The dense format is fully compatible with standard Fortran 77
   storage by columns.

   The data input variable is intended primarily for Fortran programmers
   who wish to allocate their own matrix memory space.  Most users should
   set data=PETSC_NULL.

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   Currently, the only parallel dense matrix decomposition is by rows,
   so that n=N and each submatrix owns all of the global columns.

   Level: intermediate

.keywords: matrix, dense, parallel

.seealso: MatCreate(), MatCreateSeqDense(), MatSetValues()
@*/
int MatCreateMPIDense(MPI_Comm comm,int m,int n,int M,int N,Scalar *data,Mat *A)
{
  Mat          mat;
  Mat_MPIDense *a;
  int          ierr, i,flg;

  PetscFunctionBegin;
  /* Note:  For now, when data is specified above, this assumes the user correctly
   allocates the local dense storage space.  We should add error checking. */

  *A = 0;
  PetscHeaderCreate(mat,_p_Mat,struct _MatOps,MAT_COOKIE,MATMPIDENSE,"Mat",comm,MatDestroy,MatView);
  PLogObjectCreate(mat);
  mat->data       = (void *) (a = PetscNew(Mat_MPIDense)); CHKPTRQ(a);
  PetscMemcpy(mat->ops,&MatOps_Values,sizeof(struct _MatOps));
  mat->ops->destroy    = MatDestroy_MPIDense;
  mat->ops->view       = MatView_MPIDense;
  mat->factor          = 0;
  mat->mapping         = 0;

  a->factor       = 0;
  mat->insertmode = NOT_SET_VALUES;
  MPI_Comm_rank(comm,&a->rank);
  MPI_Comm_size(comm,&a->size);

  ierr = PetscSplitOwnership(comm,&m,&M);CHKERRQ(ierr);

  /*
     The computation of n is wrong below, n should represent the number of local 
     rows in the right (column vector) 
  */

  /* each row stores all columns */
  if (N == PETSC_DECIDE) N = n;
  if (n == PETSC_DECIDE) {n = N/a->size + ((N % a->size) > a->rank);}
  /*  if (n != N) SETERRQ(PETSC_ERR_SUP,0,"For now, only n=N is supported"); */
  a->N = mat->N = N;
  a->M = mat->M = M;
  a->m = mat->m = m;
  a->n = mat->n = n;

  /* the information in the maps duplicates the information computed below, eventually 
     we should remove the duplicate information that is not contained in the maps */
  ierr = MapCreateMPI(comm,m,M,&mat->rmap);CHKERRQ(ierr);
  ierr = MapCreateMPI(comm,PETSC_DECIDE,N,&mat->cmap);CHKERRQ(ierr);

  /* build local table of row and column ownerships */
  a->rowners = (int *) PetscMalloc(2*(a->size+2)*sizeof(int)); CHKPTRQ(a->rowners);
  a->cowners = a->rowners + a->size + 1;
  PLogObjectMemory(mat,2*(a->size+2)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_MPIDense));
  ierr = MPI_Allgather(&m,1,MPI_INT,a->rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  a->rowners[0] = 0;
  for ( i=2; i<=a->size; i++ ) {
    a->rowners[i] += a->rowners[i-1];
  }
  a->rstart = a->rowners[a->rank]; 
  a->rend   = a->rowners[a->rank+1]; 
  ierr      = MPI_Allgather(&n,1,MPI_INT,a->cowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  a->cowners[0] = 0;
  for ( i=2; i<=a->size; i++ ) {
    a->cowners[i] += a->cowners[i-1];
  }

  ierr = MatCreateSeqDense(PETSC_COMM_SELF,m,N,data,&a->A); CHKERRQ(ierr);
  PLogObjectParent(mat,a->A);

  /* build cache for off array entries formed */
  ierr = MatStashCreate_Private(comm,1,&mat->stash); CHKERRQ(ierr);

  /* stuff used for matrix vector multiply */
  a->lvec        = 0;
  a->Mvctx       = 0;
  a->roworiented = 1;

  *A = mat;
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = MatPrintHelp(mat); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDuplicate_MPIDense"
static int MatDuplicate_MPIDense(Mat A,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat          mat;
  Mat_MPIDense *a,*oldmat = (Mat_MPIDense *) A->data;
  int          ierr;
  FactorCtx    *factor;

  PetscFunctionBegin;
  *newmat       = 0;
  PetscHeaderCreate(mat,_p_Mat,struct _MatOps,MAT_COOKIE,MATMPIDENSE,"Mat",A->comm,MatDestroy,MatView);
  PLogObjectCreate(mat);
  mat->data      = (void *) (a = PetscNew(Mat_MPIDense)); CHKPTRQ(a);
  PetscMemcpy(mat->ops,&MatOps_Values,sizeof(struct _MatOps));
  mat->ops->destroy   = MatDestroy_MPIDense;
  mat->ops->view      = MatView_MPIDense;
  mat->factor         = A->factor;
  mat->assembled      = PETSC_TRUE;

  a->m = mat->m = oldmat->m;
  a->n = mat->n = oldmat->n;
  a->M = mat->M = oldmat->M;
  a->N = mat->N = oldmat->N;
  if (oldmat->factor) {
    a->factor = (FactorCtx *) (factor = PetscNew(FactorCtx)); CHKPTRQ(factor);
    /* copy factor contents ... add this code! */
  } else a->factor = 0;

  a->rstart       = oldmat->rstart;
  a->rend         = oldmat->rend;
  a->size         = oldmat->size;
  a->rank         = oldmat->rank;
  mat->insertmode = NOT_SET_VALUES;

  a->rowners = (int *) PetscMalloc((a->size+1)*sizeof(int)); CHKPTRQ(a->rowners);
  PLogObjectMemory(mat,(a->size+1)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_MPIDense));
  PetscMemcpy(a->rowners,oldmat->rowners,(a->size+1)*sizeof(int));
  ierr = MatStashCreate_Private(A->comm,1,&mat->stash); CHKERRQ(ierr);

  ierr =  VecDuplicate(oldmat->lvec,&a->lvec); CHKERRQ(ierr);
  PLogObjectParent(mat,a->lvec);
  ierr =  VecScatterCopy(oldmat->Mvctx,&a->Mvctx); CHKERRQ(ierr);
  PLogObjectParent(mat,a->Mvctx);
  ierr =  MatDuplicate(oldmat->A,cpvalues,&a->A); CHKERRQ(ierr);
  PLogObjectParent(mat,a->A);
  *newmat = mat;
  PetscFunctionReturn(0);
}

#include "sys.h"

#undef __FUNC__  
#define __FUNC__ "MatLoad_MPIDense_DenseInFile"
int MatLoad_MPIDense_DenseInFile(MPI_Comm comm,int fd,int M, int N, Mat *newmat)
{
  int        *rowners, i,size,rank,m,ierr,nz,j;
  Scalar     *array,*vals,*vals_ptr;
  MPI_Status status;

  PetscFunctionBegin;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);

  /* determine ownership of all rows */
  m          = M/size + ((M % size) > rank);
  rowners    = (int *) PetscMalloc((size+2)*sizeof(int)); CHKPTRQ(rowners);
  ierr       = MPI_Allgather(&m,1,MPI_INT,rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  rowners[0] = 0;
  for ( i=2; i<=size; i++ ) {
    rowners[i] += rowners[i-1];
  }

  ierr = MatCreateMPIDense(comm,m,PETSC_DECIDE,M,N,PETSC_NULL,newmat);CHKERRQ(ierr);
  ierr = MatGetArray(*newmat,&array); CHKERRQ(ierr);

  if (!rank) {
    vals = (Scalar *) PetscMalloc( m*N*sizeof(Scalar) ); CHKPTRQ(vals);

    /* read in my part of the matrix numerical values  */
    ierr = PetscBinaryRead(fd,vals,m*N,PETSC_SCALAR); CHKERRQ(ierr);
    
    /* insert into matrix-by row (this is why cannot directly read into array */
    vals_ptr = vals;
    for ( i=0; i<m; i++ ) {
      for ( j=0; j<N; j++ ) {
        array[i + j*m] = *vals_ptr++;
      }
    }

    /* read in other processors and ship out */
    for ( i=1; i<size; i++ ) {
      nz   = (rowners[i+1] - rowners[i])*N;
      ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR); CHKERRQ(ierr);
      ierr = MPI_Send(vals,nz,MPIU_SCALAR,i,(*newmat)->tag,comm);CHKERRQ(ierr);
    }
  } else {
    /* receive numeric values */
    vals = (Scalar*) PetscMalloc( m*N*sizeof(Scalar) ); CHKPTRQ(vals);

    /* receive message of values*/
    ierr = MPI_Recv(vals,m*N,MPIU_SCALAR,0,(*newmat)->tag,comm,&status);CHKERRQ(ierr);

    /* insert into matrix-by row (this is why cannot directly read into array */
    vals_ptr = vals;
    for ( i=0; i<m; i++ ) {
      for ( j=0; j<N; j++ ) {
        array[i + j*m] = *vals_ptr++;
      }
    }
  }
  PetscFree(rowners);
  PetscFree(vals);
  ierr = MatAssemblyBegin(*newmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*newmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatLoad_MPIDense"
int MatLoad_MPIDense(Viewer viewer,MatType type,Mat *newmat)
{
  Mat          A;
  Scalar       *vals,*svals;
  MPI_Comm     comm = ((PetscObject)viewer)->comm;
  MPI_Status   status;
  int          header[4],rank,size,*rowlengths = 0,M,N,m,*rowners,maxnz,*cols;
  int          *ourlens,*sndcounts = 0,*procsnz = 0, *offlens,jj,*mycols,*smycols;
  int          tag = ((PetscObject)viewer)->tag;
  int          i, nz, ierr, j,rstart, rend, fd;

  PetscFunctionBegin;
  MPI_Comm_size(comm,&size); MPI_Comm_rank(comm,&rank);
  if (!rank) {
    ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,PETSC_INT); CHKERRQ(ierr);
    if (header[0] != MAT_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"not matrix object");
  }

  ierr = MPI_Bcast(header+1,3,MPI_INT,0,comm);CHKERRQ(ierr);
  M = header[1]; N = header[2]; nz = header[3];

  /*
       Handle case where matrix is stored on disk as a dense matrix 
  */
  if (nz == MATRIX_BINARY_FORMAT_DENSE) {
    ierr = MatLoad_MPIDense_DenseInFile(comm,fd,M,N,newmat);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* determine ownership of all rows */
  m          = M/size + ((M % size) > rank);
  rowners    = (int *) PetscMalloc((size+2)*sizeof(int)); CHKPTRQ(rowners);
  ierr       = MPI_Allgather(&m,1,MPI_INT,rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
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
    ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT); CHKERRQ(ierr);
    sndcounts = (int*) PetscMalloc( size*sizeof(int) ); CHKPTRQ(sndcounts);
    for ( i=0; i<size; i++ ) sndcounts[i] = rowners[i+1] - rowners[i];
    ierr = MPI_Scatterv(rowlengths,sndcounts,rowners,MPI_INT,ourlens,rend-rstart,MPI_INT,0,comm);CHKERRQ(ierr);
    PetscFree(sndcounts);
  } else {
    ierr = MPI_Scatterv(0,0,0,MPI_INT,ourlens,rend-rstart,MPI_INT, 0,comm);CHKERRQ(ierr);
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
    ierr = PetscBinaryRead(fd,mycols,nz,PETSC_INT); CHKERRQ(ierr);

    /* read in every one elses and ship off */
    for ( i=1; i<size; i++ ) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT); CHKERRQ(ierr);
      ierr = MPI_Send(cols,nz,MPI_INT,i,tag,comm);CHKERRQ(ierr);
    }
    PetscFree(cols);
  } else {
    /* determine buffer space needed for message */
    nz = 0;
    for ( i=0; i<m; i++ ) {
      nz += ourlens[i];
    }
    mycols = (int*) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(mycols);

    /* receive message of column indices*/
    ierr = MPI_Recv(mycols,nz,MPI_INT,0,tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPI_INT,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"something is wrong with file");
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
  ierr = MatCreateMPIDense(comm,m,PETSC_DECIDE,M,N,PETSC_NULL,newmat);CHKERRQ(ierr);
  A = *newmat;
  for ( i=0; i<m; i++ ) {
    ourlens[i] += offlens[i];
  }

  if (!rank) {
    vals = (Scalar *) PetscMalloc( maxnz*sizeof(Scalar) ); CHKPTRQ(vals);

    /* read in my part of the matrix numerical values  */
    nz = procsnz[0];
    ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR); CHKERRQ(ierr);
    
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
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR); CHKERRQ(ierr);
      ierr = MPI_Send(vals,nz,MPIU_SCALAR,i,A->tag,comm);CHKERRQ(ierr);
    }
    PetscFree(procsnz);
  } else {
    /* receive numeric values */
    vals = (Scalar*) PetscMalloc( nz*sizeof(Scalar) ); CHKPTRQ(vals);

    /* receive message of values*/
    ierr = MPI_Recv(vals,nz,MPIU_SCALAR,0,A->tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPIU_SCALAR,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"something is wrong with file");

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

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}





