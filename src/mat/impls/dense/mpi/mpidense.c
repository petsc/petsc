#define PETSCMAT_DLL

/*
   Basic functions for basic parallel dense matrices.
*/
    
#include "src/mat/impls/dense/mpi/mpidense.h"    /*I   "petscmat.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_MPIDense"
PetscErrorCode MatLUFactorSymbolic_MPIDense(Mat A,IS row,IS col,MatFactorInfo *info,Mat *fact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,fact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_MPIDense"
PetscErrorCode MatCholeskyFactorSymbolic_MPIDense(Mat A,IS perm,MatFactorInfo *info,Mat *fact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,fact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDenseGetLocalMatrix"
/*@

      MatDenseGetLocalMatrix - For a MATMPIDENSE or MATSEQDENSE matrix returns the sequential
              matrix that represents the operator. For sequential matrices it returns itself.

    Input Parameter:
.      A - the Seq or MPI dense matrix

    Output Parameter:
.      B - the inner matrix

    Level: intermediate

@*/
PetscErrorCode MatDenseGetLocalMatrix(Mat A,Mat *B)
{
  Mat_MPIDense   *mat = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;
  PetscTruth     flg;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)A,MATMPIDENSE,&flg);CHKERRQ(ierr);
  if (!flg) {  /* this check sucks! */
    ierr = PetscTypeCompare((PetscObject)A,MATDENSE,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
      if (size == 1) flg = PETSC_FALSE;
    }
  }
  if (flg) {
    *B = mat->A;
  } else {
    *B = A;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRow_MPIDense"
PetscErrorCode MatGetRow_MPIDense(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPIDense   *mat = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       lrow,rstart = A->rmap.rstart,rend = A->rmap.rend;

  PetscFunctionBegin;
  if (row < rstart || row >= rend) SETERRQ(PETSC_ERR_SUP,"only local rows")
  lrow = row - rstart;
  ierr = MatGetRow(mat->A,lrow,nz,(const PetscInt **)idx,(const PetscScalar **)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRow_MPIDense"
PetscErrorCode MatRestoreRow_MPIDense(Mat mat,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (idx) {ierr = PetscFree(*idx);CHKERRQ(ierr);}
  if (v) {ierr = PetscFree(*v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonalBlock_MPIDense"
PetscErrorCode PETSCMAT_DLLEXPORT MatGetDiagonalBlock_MPIDense(Mat A,PetscTruth *iscopy,MatReuse reuse,Mat *B)
{
  Mat_MPIDense   *mdn = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       m = A->rmap.n,rstart = A->rmap.rstart;
  PetscScalar    *array;
  MPI_Comm       comm;

  PetscFunctionBegin;
  if (A->rmap.N != A->cmap.N) SETERRQ(PETSC_ERR_SUP,"Only square matrices supported.");

  /* The reuse aspect is not implemented efficiently */
  if (reuse) { ierr = MatDestroy(*B);CHKERRQ(ierr);}

  ierr = PetscObjectGetComm((PetscObject)(mdn->A),&comm);CHKERRQ(ierr);
  ierr = MatGetArray(mdn->A,&array);CHKERRQ(ierr);
  ierr = MatCreate(comm,B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,m,m,m,m);CHKERRQ(ierr);
  ierr = MatSetType(*B,mdn->A->type_name);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(*B,array+m*rstart);CHKERRQ(ierr);
  ierr = MatRestoreArray(mdn->A,&array);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    
  *iscopy = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_MPIDense"
PetscErrorCode MatSetValues_MPIDense(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPIDense   *A = (Mat_MPIDense*)mat->data;
  PetscErrorCode ierr;
  PetscInt       i,j,rstart = mat->rmap.rstart,rend = mat->rmap.rend,row;
  PetscTruth     roworiented = A->roworiented;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) continue;
    if (idxm[i] >= mat->rmap.N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      if (roworiented) {
        ierr = MatSetValues(A->A,1,&row,n,idxn,v+i*n,addv);CHKERRQ(ierr);
      } else {
        for (j=0; j<n; j++) {
          if (idxn[j] < 0) continue;
          if (idxn[j] >= mat->cmap.N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Column too large");
          ierr = MatSetValues(A->A,1,&row,1,&idxn[j],v+i+j*m,addv);CHKERRQ(ierr);
        }
      }
    } else {
      if (!A->donotstash) {
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
#define __FUNCT__ "MatGetValues_MPIDense"
PetscErrorCode MatGetValues_MPIDense(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],PetscScalar v[])
{
  Mat_MPIDense   *mdn = (Mat_MPIDense*)mat->data;
  PetscErrorCode ierr;
  PetscInt       i,j,rstart = mat->rmap.rstart,rend = mat->rmap.rend,row;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative row");
    if (idxm[i] >= mat->rmap.N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for (j=0; j<n; j++) {
        if (idxn[j] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative column");
        if (idxn[j] >= mat->cmap.N) {
          SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Column too large");
        }
        ierr = MatGetValues(mdn->A,1,&row,1,&idxn[j],v+i*n+j);CHKERRQ(ierr);
      }
    } else {
      SETERRQ(PETSC_ERR_SUP,"Only local values currently supported");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetArray_MPIDense"
PetscErrorCode MatGetArray_MPIDense(Mat A,PetscScalar *array[])
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetArray(a->A,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrix_MPIDense"
static PetscErrorCode MatGetSubMatrix_MPIDense(Mat A,IS isrow,IS iscol,PetscInt cs,MatReuse scall,Mat *B)
{
  Mat_MPIDense   *mat = (Mat_MPIDense*)A->data,*newmatd;
  Mat_SeqDense   *lmat = (Mat_SeqDense*)mat->A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,*irow,*icol,rstart,rend,nrows,ncols,nlrows,nlcols;
  PetscScalar    *av,*bv,*v = lmat->v;
  Mat            newmat;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrow,&nrows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscol,&ncols);CHKERRQ(ierr);

  /* No parallel redistribution currently supported! Should really check each index set
     to comfirm that it is OK.  ... Currently supports only submatrix same partitioning as
     original matrix! */

  ierr = MatGetLocalSize(A,&nlrows,&nlcols);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  
  /* Check submatrix call */
  if (scall == MAT_REUSE_MATRIX) {
    /* SETERRQ(PETSC_ERR_ARG_SIZ,"Reused submatrix wrong size"); */
    /* Really need to test rows and column sizes! */
    newmat = *B;
  } else {
    /* Create and fill new matrix */
    ierr = MatCreate(A->comm,&newmat);CHKERRQ(ierr);
    ierr = MatSetSizes(newmat,nrows,cs,PETSC_DECIDE,ncols);CHKERRQ(ierr);
    ierr = MatSetType(newmat,A->type_name);CHKERRQ(ierr);
    ierr = MatMPIDenseSetPreallocation(newmat,PETSC_NULL);CHKERRQ(ierr);
  }

  /* Now extract the data pointers and do the copy, column at a time */
  newmatd = (Mat_MPIDense*)newmat->data;
  bv      = ((Mat_SeqDense *)newmatd->A->data)->v;
  
  for (i=0; i<ncols; i++) {
    av = v + nlrows*icol[i];
    for (j=0; j<nrows; j++) {
      *bv++ = av[irow[j] - rstart];
    }
  }

  /* Assemble the matrices so that the correct flags are set */
  ierr = MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Free work space */
  ierr = ISRestoreIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&icol);CHKERRQ(ierr);
  *B = newmat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreArray_MPIDense"
PetscErrorCode MatRestoreArray_MPIDense(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyBegin_MPIDense"
PetscErrorCode MatAssemblyBegin_MPIDense(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIDense   *mdn = (Mat_MPIDense*)mat->data;
  MPI_Comm       comm = mat->comm;
  PetscErrorCode ierr;
  PetscInt       nstash,reallocs;
  InsertMode     addv;

  PetscFunctionBegin;
  /* make sure all processors are either in INSERTMODE or ADDMODE */
  ierr = MPI_Allreduce(&mat->insertmode,&addv,1,MPI_INT,MPI_BOR,comm);CHKERRQ(ierr);
  if (addv == (ADD_VALUES|INSERT_VALUES)) { 
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot mix adds/inserts on different procs");
  }
  mat->insertmode = addv; /* in case this processor had no cache */

  ierr = MatStashScatterBegin_Private(&mat->stash,mat->rmap.range);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscInfo2(mdn->A,"Stash has %D entries, uses %D mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_MPIDense"
PetscErrorCode MatAssemblyEnd_MPIDense(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIDense    *mdn=(Mat_MPIDense*)mat->data;
  PetscErrorCode  ierr;
  PetscInt        i,*row,*col,flg,j,rstart,ncols;
  PetscMPIInt     n;
  PetscScalar     *val;
  InsertMode      addv=mat->insertmode;

  PetscFunctionBegin;
  /*  wait on receives */
  while (1) {
    ierr = MatStashScatterGetMesg_Private(&mat->stash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
    if (!flg) break;
    
    for (i=0; i<n;) {
      /* Now identify the consecutive vals belonging to the same row */
      for (j=i,rstart=row[j]; j<n; j++) { if (row[j] != rstart) break; }
      if (j < n) ncols = j-i;
      else       ncols = n-i;
      /* Now assemble all these values with a single function call */
      ierr = MatSetValues_MPIDense(mat,1,row+i,ncols,col+i,val+i,addv);CHKERRQ(ierr);
      i = j;
    }
  }
  ierr = MatStashScatterEnd_Private(&mat->stash);CHKERRQ(ierr);
  
  ierr = MatAssemblyBegin(mdn->A,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mdn->A,mode);CHKERRQ(ierr);

  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIDense(mat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroEntries_MPIDense"
PetscErrorCode MatZeroEntries_MPIDense(Mat A)
{
  PetscErrorCode ierr;
  Mat_MPIDense   *l = (Mat_MPIDense*)A->data;

  PetscFunctionBegin;
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* the code does not do the diagonal entries correctly unless the 
   matrix is square and the column and row owerships are identical.
   This is a BUG. The only way to fix it seems to be to access 
   mdn->A and mdn->B directly and not through the MatZeroRows() 
   routine. 
*/
#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_MPIDense"
PetscErrorCode MatZeroRows_MPIDense(Mat A,PetscInt N,const PetscInt rows[],PetscScalar diag)
{
  Mat_MPIDense   *l = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,*owners = A->rmap.range;
  PetscInt       *nprocs,j,idx,nsends;
  PetscInt       nmax,*svalues,*starts,*owner,nrecvs;
  PetscInt       *rvalues,tag = A->tag,count,base,slen,*source;
  PetscInt       *lens,*lrows,*values;
  PetscMPIInt    n,imdex,rank = l->rank,size = l->size;
  MPI_Comm       comm = A->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
  PetscTruth     found;

  PetscFunctionBegin;
  /*  first count number of contributors to each processor */
  ierr  = PetscMalloc(2*size*sizeof(PetscInt),&nprocs);CHKERRQ(ierr);
  ierr  = PetscMemzero(nprocs,2*size*sizeof(PetscInt));CHKERRQ(ierr);
  ierr  = PetscMalloc((N+1)*sizeof(PetscInt),&owner);CHKERRQ(ierr); /* see note*/
  for (i=0; i<N; i++) {
    idx = rows[i];
    found = PETSC_FALSE;
    for (j=0; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[2*j]++; nprocs[2*j+1] = 1; owner[i] = j; found = PETSC_TRUE; break;
      }
    }
    if (!found) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Index out of range");
  }
  nsends = 0;  for (i=0; i<size; i++) { nsends += nprocs[2*i+1];} 

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
  starts[0]  = 0; 
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
  count  = nrecvs; slen = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPIU_INT,&n);CHKERRQ(ierr);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]    = n;
    slen += n;
    count--;
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);
  
  /* move the data into the send scatter */
  ierr = PetscMalloc((slen+1)*sizeof(PetscInt),&lrows);CHKERRQ(ierr);
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
#define __FUNCT__ "MatMult_MPIDense"
PetscErrorCode MatMult_MPIDense(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIDense   *mdn = (Mat_MPIDense*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(xx,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD,mdn->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD,mdn->Mvctx);CHKERRQ(ierr);
  ierr = MatMult_SeqDense(mdn->A,mdn->lvec,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_MPIDense"
PetscErrorCode MatMultAdd_MPIDense(Mat mat,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIDense   *mdn = (Mat_MPIDense*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(xx,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD,mdn->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(xx,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD,mdn->Mvctx);CHKERRQ(ierr);
  ierr = MatMultAdd_SeqDense(mdn->A,mdn->lvec,yy,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_MPIDense"
PetscErrorCode MatMultTranspose_MPIDense(Mat A,Vec xx,Vec yy)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;
  PetscScalar    zero = 0.0;

  PetscFunctionBegin;
  ierr = VecSet(yy,zero);CHKERRQ(ierr);
  ierr = MatMultTranspose_SeqDense(a->A,xx,a->lvec);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_MPIDense"
PetscErrorCode MatMultTransposeAdd_MPIDense(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(yy,zz);CHKERRQ(ierr);
  ierr = MatMultTranspose_SeqDense(a->A,xx,a->lvec);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_MPIDense"
PetscErrorCode MatGetDiagonal_MPIDense(Mat A,Vec v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  Mat_SeqDense   *aloc = (Mat_SeqDense*)a->A->data;
  PetscErrorCode ierr;
  PetscInt       len,i,n,m = A->rmap.n,radd;
  PetscScalar    *x,zero = 0.0;
  
  PetscFunctionBegin;
  ierr = VecSet(v,zero);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetSize(v,&n);CHKERRQ(ierr);
  if (n != A->rmap.N) SETERRQ(PETSC_ERR_ARG_SIZ,"Nonconforming mat and vec");
  len  = PetscMin(a->A->rmap.n,a->A->cmap.n);
  radd = A->rmap.rstart*m;
  for (i=0; i<len; i++) {
    x[i] = aloc->v[radd + i*m + i];
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIDense"
PetscErrorCode MatDestroy_MPIDense(Mat mat)
{
  Mat_MPIDense   *mdn = (Mat_MPIDense*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;

#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows=%D, Cols=%D",mat->rmap.N,mat->cmap.N);
#endif
  ierr = MatStashDestroy_Private(&mat->stash);CHKERRQ(ierr);
  ierr = MatDestroy(mdn->A);CHKERRQ(ierr);
  if (mdn->lvec)   {ierr = VecDestroy(mdn->lvec);CHKERRQ(ierr);}
  if (mdn->Mvctx)  {ierr = VecScatterDestroy(mdn->Mvctx);CHKERRQ(ierr);}
  if (mdn->factor) {
    ierr = PetscFree(mdn->factor->temp);CHKERRQ(ierr);
    ierr = PetscFree(mdn->factor->tag);CHKERRQ(ierr);
    ierr = PetscFree(mdn->factor->pivots);CHKERRQ(ierr);
    ierr = PetscFree(mdn->factor);CHKERRQ(ierr);
  }
  ierr = PetscFree(mdn);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)mat,"MatGetDiagonalBlock_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)mat,"MatMPIDenseSetPreallocation_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMatMult_mpiaij_mpidense_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMatMultSymbolic_mpiaij_mpidense_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMatMultNumeric_mpiaij_mpidense_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIDense_Binary"
static PetscErrorCode MatView_MPIDense_Binary(Mat mat,PetscViewer viewer)
{
  Mat_MPIDense   *mdn = (Mat_MPIDense*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mdn->size == 1) {
    ierr = MatView(mdn->A,viewer);CHKERRQ(ierr);
  }
  else SETERRQ(PETSC_ERR_SUP,"Only uniprocessor output supported");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIDense_ASCIIorDraworSocket"
static PetscErrorCode MatView_MPIDense_ASCIIorDraworSocket(Mat mat,PetscViewer viewer)
{
  Mat_MPIDense      *mdn = (Mat_MPIDense*)mat->data;
  PetscErrorCode    ierr;
  PetscMPIInt       size = mdn->size,rank = mdn->rank; 
  PetscViewerType   vtype;
  PetscTruth        iascii,isdraw;
  PetscViewer       sviewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetType(viewer,&vtype);CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      MatInfo info;
      ierr = MatGetInfo(mat,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] local rows %D nz %D nz alloced %D mem %D \n",rank,mat->rmap.n,
                   (PetscInt)info.nz_used,(PetscInt)info.nz_allocated,(PetscInt)info.memory);CHKERRQ(ierr);       
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = VecScatterView(mdn->Mvctx,viewer);CHKERRQ(ierr);
      PetscFunctionReturn(0); 
    } else if (format == PETSC_VIEWER_ASCII_INFO) {
      PetscFunctionReturn(0);
    }
  } else if (isdraw) {
    PetscDraw  draw;
    PetscTruth isnull;

    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
    if (isnull) PetscFunctionReturn(0);
  }

  if (size == 1) { 
    ierr = MatView(mdn->A,viewer);CHKERRQ(ierr);
  } else {
    /* assemble the entire matrix onto first processor. */
    Mat         A;
    PetscInt    M = mat->rmap.N,N = mat->cmap.N,m,row,i,nz;
    PetscInt    *cols;
    PetscScalar *vals;

    ierr = MatCreate(mat->comm,&A);CHKERRQ(ierr);
    if (!rank) {
      ierr = MatSetSizes(A,M,N,M,N);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(A,0,0,M,N);CHKERRQ(ierr);
    }
    /* Since this is a temporary matrix, MATMPIDENSE instead of A->type_name here is probably acceptable. */
    ierr = MatSetType(A,MATMPIDENSE);CHKERRQ(ierr);
    ierr = MatMPIDenseSetPreallocation(A,PETSC_NULL);
    ierr = PetscLogObjectParent(mat,A);CHKERRQ(ierr);

    /* Copy the matrix ... This isn't the most efficient means,
       but it's quick for now */
    A->insertmode = INSERT_VALUES;
    row = mat->rmap.rstart; m = mdn->A->rmap.n;
    for (i=0; i<m; i++) {
      ierr = MatGetRow_MPIDense(mat,row,&nz,&cols,&vals);CHKERRQ(ierr);
      ierr = MatSetValues_MPIDense(A,1,&row,nz,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow_MPIDense(mat,row,&nz,&cols,&vals);CHKERRQ(ierr);
      row++;
    } 

    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
    if (!rank) {
      ierr = MatView(((Mat_MPIDense*)(A->data))->A,sviewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIDense"
PetscErrorCode MatView_MPIDense(Mat mat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscTruth     iascii,isbinary,isdraw,issocket;
 
  PetscFunctionBegin;
  
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_SOCKET,&issocket);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);

  if (iascii || issocket || isdraw) {
    ierr = MatView_MPIDense_ASCIIorDraworSocket(mat,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_MPIDense_Binary(mat,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by MPI dense matrix",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_MPIDense"
PetscErrorCode MatGetInfo_MPIDense(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_MPIDense   *mat = (Mat_MPIDense*)A->data;
  Mat            mdn = mat->A;
  PetscErrorCode ierr;
  PetscReal      isend[5],irecv[5];

  PetscFunctionBegin;
  info->rows_global    = (double)A->rmap.N;
  info->columns_global = (double)A->cmap.N;
  info->rows_local     = (double)A->rmap.n;
  info->columns_local  = (double)A->cmap.N;
  info->block_size     = 1.0;
  ierr = MatGetInfo(mdn,MAT_LOCAL,info);CHKERRQ(ierr);
  isend[0] = info->nz_used; isend[1] = info->nz_allocated; isend[2] = info->nz_unneeded;
  isend[3] = info->memory;  isend[4] = info->mallocs;
  if (flag == MAT_LOCAL) {
    info->nz_used      = isend[0];
    info->nz_allocated = isend[1];
    info->nz_unneeded  = isend[2];
    info->memory       = isend[3];
    info->mallocs      = isend[4];
  } else if (flag == MAT_GLOBAL_MAX) {
    ierr = MPI_Allreduce(isend,irecv,5,MPIU_REAL,MPI_MAX,A->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPI_Allreduce(isend,irecv,5,MPIU_REAL,MPI_SUM,A->comm);CHKERRQ(ierr);
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

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_MPIDense"
PetscErrorCode MatSetOption_MPIDense(Mat A,MatOption op)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_NO_NEW_NONZERO_LOCATIONS:
  case MAT_YES_NEW_NONZERO_LOCATIONS:
  case MAT_NEW_NONZERO_LOCATION_ERR:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
  case MAT_COLUMNS_SORTED:
  case MAT_COLUMNS_UNSORTED:
    ierr = MatSetOption(a->A,op);CHKERRQ(ierr);
    break;
  case MAT_ROW_ORIENTED:
    a->roworiented = PETSC_TRUE;
    ierr = MatSetOption(a->A,op);CHKERRQ(ierr);
    break;
  case MAT_ROWS_SORTED: 
  case MAT_ROWS_UNSORTED:
  case MAT_YES_NEW_DIAGONALS:
  case MAT_USE_HASH_TABLE:
    ierr = PetscInfo1(A,"Option %d ignored\n",op);CHKERRQ(ierr);
    break;
  case MAT_COLUMN_ORIENTED:
    a->roworiented = PETSC_FALSE;
    ierr = MatSetOption(a->A,op);CHKERRQ(ierr);
    break;
  case MAT_IGNORE_OFF_PROC_ENTRIES:
    a->donotstash = PETSC_TRUE;
    break;
  case MAT_NO_NEW_DIAGONALS:
    SETERRQ(PETSC_ERR_SUP,"MAT_NO_NEW_DIAGONALS");
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
#define __FUNCT__ "MatDiagonalScale_MPIDense"
PetscErrorCode MatDiagonalScale_MPIDense(Mat A,Vec ll,Vec rr)
{
  Mat_MPIDense   *mdn = (Mat_MPIDense*)A->data;
  Mat_SeqDense   *mat = (Mat_SeqDense*)mdn->A->data;
  PetscScalar    *l,*r,x,*v;
  PetscErrorCode ierr;
  PetscInt       i,j,s2a,s3a,s2,s3,m=mdn->A->rmap.n,n=mdn->A->cmap.n;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&s2,&s3);CHKERRQ(ierr);
  if (ll) {
    ierr = VecGetLocalSize(ll,&s2a);CHKERRQ(ierr);
    if (s2a != s2) SETERRQ2(PETSC_ERR_ARG_SIZ,"Left scaling vector non-conforming local size, %d != %d.", s2a, s2);
    ierr = VecGetArray(ll,&l);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      x = l[i];
      v = mat->v + i;
      for (j=0; j<n; j++) { (*v) *= x; v+= m;} 
    }
    ierr = VecRestoreArray(ll,&l);CHKERRQ(ierr);
    ierr = PetscLogFlops(n*m);CHKERRQ(ierr);
  }
  if (rr) {
    ierr = VecGetLocalSize(rr,&s3a);CHKERRQ(ierr);
    if (s3a != s3) SETERRQ2(PETSC_ERR_ARG_SIZ,"Right scaling vec non-conforming local size, %d != %d.", s3a, s3);
    ierr = VecScatterBegin(rr,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD,mdn->Mvctx);CHKERRQ(ierr);
    ierr = VecScatterEnd(rr,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD,mdn->Mvctx);CHKERRQ(ierr);
    ierr = VecGetArray(mdn->lvec,&r);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      x = r[i];
      v = mat->v + i*m;
      for (j=0; j<m; j++) { (*v++) *= x;} 
    }
    ierr = VecRestoreArray(mdn->lvec,&r);CHKERRQ(ierr);
    ierr = PetscLogFlops(n*m);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNorm_MPIDense"
PetscErrorCode MatNorm_MPIDense(Mat A,NormType type,PetscReal *nrm)
{
  Mat_MPIDense   *mdn = (Mat_MPIDense*)A->data;
  Mat_SeqDense   *mat = (Mat_SeqDense*)mdn->A->data;
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscReal      sum = 0.0;
  PetscScalar    *v = mat->v;

  PetscFunctionBegin;
  if (mdn->size == 1) {
    ierr =  MatNorm(mdn->A,type,nrm);CHKERRQ(ierr);
  } else {
    if (type == NORM_FROBENIUS) {
      for (i=0; i<mdn->A->cmap.n*mdn->A->rmap.n; i++) {
#if defined(PETSC_USE_COMPLEX)
        sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      ierr = MPI_Allreduce(&sum,nrm,1,MPIU_REAL,MPI_SUM,A->comm);CHKERRQ(ierr);
      *nrm = sqrt(*nrm);
      ierr = PetscLogFlops(2*mdn->A->cmap.n*mdn->A->rmap.n);CHKERRQ(ierr);
    } else if (type == NORM_1) { 
      PetscReal *tmp,*tmp2;
      ierr = PetscMalloc(2*A->cmap.N*sizeof(PetscReal),&tmp);CHKERRQ(ierr);
      tmp2 = tmp + A->cmap.N;
      ierr = PetscMemzero(tmp,2*A->cmap.N*sizeof(PetscReal));CHKERRQ(ierr);
      *nrm = 0.0;
      v = mat->v;
      for (j=0; j<mdn->A->cmap.n; j++) {
        for (i=0; i<mdn->A->rmap.n; i++) {
          tmp[j] += PetscAbsScalar(*v);  v++;
        }
      }
      ierr = MPI_Allreduce(tmp,tmp2,A->cmap.N,MPIU_REAL,MPI_SUM,A->comm);CHKERRQ(ierr);
      for (j=0; j<A->cmap.N; j++) {
        if (tmp2[j] > *nrm) *nrm = tmp2[j];
      }
      ierr = PetscFree(tmp);CHKERRQ(ierr);
      ierr = PetscLogFlops(A->cmap.n*A->rmap.n);CHKERRQ(ierr);
    } else if (type == NORM_INFINITY) { /* max row norm */
      PetscReal ntemp;
      ierr = MatNorm(mdn->A,type,&ntemp);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&ntemp,nrm,1,MPIU_REAL,MPI_MAX,A->comm);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"No support for two norm");
    }
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatTranspose_MPIDense"
PetscErrorCode MatTranspose_MPIDense(Mat A,Mat *matout)
{ 
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  Mat_SeqDense   *Aloc = (Mat_SeqDense*)a->A->data;
  Mat            B;
  PetscInt       M = A->rmap.N,N = A->cmap.N,m,n,*rwork,rstart = A->rmap.rstart;
  PetscErrorCode ierr;
  PetscInt       j,i;
  PetscScalar    *v;

  PetscFunctionBegin;
  if (!matout && M != N) {
    SETERRQ(PETSC_ERR_SUP,"Supports square matrix only in-place");
  }
  ierr = MatCreate(A->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,N,M);CHKERRQ(ierr);
  ierr = MatSetType(B,A->type_name);CHKERRQ(ierr);
  ierr = MatMPIDenseSetPreallocation(B,PETSC_NULL);CHKERRQ(ierr);

  m = a->A->rmap.n; n = a->A->cmap.n; v = Aloc->v;
  ierr = PetscMalloc(n*sizeof(PetscInt),&rwork);CHKERRQ(ierr);
  for (j=0; j<n; j++) {
    for (i=0; i<m; i++) rwork[i] = rstart + i;
    ierr = MatSetValues(B,1,&j,m,rwork,v,INSERT_VALUES);CHKERRQ(ierr);
    v   += m;
  } 
  ierr = PetscFree(rwork);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (matout) {
    *matout = B;
  } else {
    ierr = MatHeaderCopy(A,B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include "petscblaslapack.h"
#undef __FUNCT__  
#define __FUNCT__ "MatScale_MPIDense"
PetscErrorCode MatScale_MPIDense(Mat inA,PetscScalar alpha)
{
  Mat_MPIDense   *A = (Mat_MPIDense*)inA->data;
  Mat_SeqDense   *a = (Mat_SeqDense*)A->A->data;
  PetscScalar    oalpha = alpha;
  PetscBLASInt   one = 1,nz = (PetscBLASInt)inA->rmap.n*inA->cmap.N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  BLASscal_(&nz,&oalpha,a->v,&one);
  ierr = PetscLogFlops(nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_MPIDense(Mat,MatDuplicateOption,Mat *);

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpPreallocation_MPIDense"
PetscErrorCode MatSetUpPreallocation_MPIDense(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr =  MatMPIDenseSetPreallocation(A,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultSymbolic_MPIDense_MPIDense"
PetscErrorCode MatMatMultSymbolic_MPIDense_MPIDense(Mat A,Mat B,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;
  PetscInt       m=A->rmap.n,n=B->cmap.n;
  Mat            Cmat;

  PetscFunctionBegin;
  if (A->cmap.n != B->rmap.n) SETERRQ2(PETSC_ERR_ARG_SIZ,"A->cmap.n %d != B->rmap.n %d\n",A->cmap.n,B->rmap.n);
  ierr = MatCreate(B->comm,&Cmat);CHKERRQ(ierr);
  ierr = MatSetSizes(Cmat,m,n,A->rmap.N,B->cmap.N);CHKERRQ(ierr);
  ierr = MatSetType(Cmat,MATMPIDENSE);CHKERRQ(ierr);
  ierr = MatMPIDenseSetPreallocation(Cmat,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Cmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Cmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *C = Cmat;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_MPIDense,
       MatGetRow_MPIDense,
       MatRestoreRow_MPIDense,
       MatMult_MPIDense,
/* 4*/ MatMultAdd_MPIDense,
       MatMultTranspose_MPIDense,
       MatMultTransposeAdd_MPIDense,
       0,
       0,
       0,
/*10*/ 0,
       0,
       0,
       0,
       MatTranspose_MPIDense,
/*15*/ MatGetInfo_MPIDense,
       MatEqual_MPIDense,
       MatGetDiagonal_MPIDense,
       MatDiagonalScale_MPIDense,
       MatNorm_MPIDense,
/*20*/ MatAssemblyBegin_MPIDense,
       MatAssemblyEnd_MPIDense,
       0,
       MatSetOption_MPIDense,
       MatZeroEntries_MPIDense,
/*25*/ MatZeroRows_MPIDense,
       MatLUFactorSymbolic_MPIDense,
       0,
       MatCholeskyFactorSymbolic_MPIDense,
       0,
/*30*/ MatSetUpPreallocation_MPIDense,
       0,
       0,
       MatGetArray_MPIDense,
       MatRestoreArray_MPIDense,
/*35*/ MatDuplicate_MPIDense,
       0,
       0,
       0,
       0,
/*40*/ 0,
       MatGetSubMatrices_MPIDense,
       0,
       MatGetValues_MPIDense,
       0,
/*45*/ 0,
       MatScale_MPIDense,
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
/*60*/ MatGetSubMatrix_MPIDense,
       MatDestroy_MPIDense,
       MatView_MPIDense,
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
/*84*/ MatLoad_MPIDense,
       0,
       0,
       0,
       0,
       0,
/*90*/ 0,
       MatMatMultSymbolic_MPIDense_MPIDense,
       0,
       0,
       0,
/*95*/ 0,
       0,
       0,
       0};

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatMPIDenseSetPreallocation_MPIDense"
PetscErrorCode PETSCMAT_DLLEXPORT MatMPIDenseSetPreallocation_MPIDense(Mat mat,PetscScalar *data)
{
  Mat_MPIDense   *a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mat->preallocated = PETSC_TRUE;
  /* Note:  For now, when data is specified above, this assumes the user correctly
   allocates the local dense storage space.  We should add error checking. */

  a    = (Mat_MPIDense*)mat->data;
  ierr = MatCreate(PETSC_COMM_SELF,&a->A);CHKERRQ(ierr);
  ierr = MatSetSizes(a->A,mat->rmap.n,mat->cmap.N,mat->rmap.n,mat->cmap.N);CHKERRQ(ierr);
  ierr = MatSetType(a->A,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(a->A,data);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
   MATMPIDENSE - MATMPIDENSE = "mpidense" - A matrix type to be used for distributed dense matrices.

   Options Database Keys:
. -mat_type mpidense - sets the matrix type to "mpidense" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateMPIDense
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_MPIDense"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPIDense(Mat mat)
{
  Mat_MPIDense   *a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr              = PetscNew(Mat_MPIDense,&a);CHKERRQ(ierr);
  mat->data         = (void*)a;
  ierr              = PetscMemcpy(mat->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  mat->factor       = 0;
  mat->mapping      = 0;

  a->factor       = 0;
  mat->insertmode = NOT_SET_VALUES;
  ierr = MPI_Comm_rank(mat->comm,&a->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(mat->comm,&a->size);CHKERRQ(ierr);

  mat->rmap.bs = mat->cmap.bs = 1;
  ierr = PetscMapInitialize(mat->comm,&mat->rmap);CHKERRQ(ierr);
  ierr = PetscMapInitialize(mat->comm,&mat->cmap);CHKERRQ(ierr);
  a->nvec = mat->cmap.n;

  /* build cache for off array entries formed */
  a->donotstash = PETSC_FALSE;
  ierr = MatStashCreate_Private(mat->comm,1,&mat->stash);CHKERRQ(ierr);

  /* stuff used for matrix vector multiply */
  a->lvec        = 0;
  a->Mvctx       = 0;
  a->roworiented = PETSC_TRUE;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)mat,"MatGetDiagonalBlock_C",
                                     "MatGetDiagonalBlock_MPIDense",
                                     MatGetDiagonalBlock_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)mat,"MatMPIDenseSetPreallocation_C",
                                     "MatMPIDenseSetPreallocation_MPIDense",
                                     MatMPIDenseSetPreallocation_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)mat,"MatMatMult_mpiaij_mpidense_C",
                                     "MatMatMult_MPIAIJ_MPIDense",
                                      MatMatMult_MPIAIJ_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)mat,"MatMatMultSymbolic_mpiaij_mpidense_C",
                                     "MatMatMultSymbolic_MPIAIJ_MPIDense",
                                      MatMatMultSymbolic_MPIAIJ_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)mat,"MatMatMultNumeric_mpiaij_mpidense_C",
                                     "MatMatMultNumeric_MPIAIJ_MPIDense",
                                      MatMatMultNumeric_MPIAIJ_MPIDense);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)mat,MATMPIDENSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
   MATDENSE - MATDENSE = "dense" - A matrix type to be used for dense matrices.

   This matrix type is identical to MATSEQDENSE when constructed with a single process communicator,
   and MATMPIDENSE otherwise.

   Options Database Keys:
. -mat_type dense - sets the matrix type to "dense" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateMPIDense,MATSEQDENSE,MATMPIDENSE
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_Dense"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Dense(Mat A) 
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(A,MATSEQDENSE);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(A,MATMPIDENSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatMPIDenseSetPreallocation"
/*@C
   MatMPIDenseSetPreallocation - Sets the array used to store the matrix entries

   Not collective

   Input Parameters:
.  A - the matrix
-  data - optional location of matrix data.  Set data=PETSC_NULL for PETSc
   to control all matrix memory allocation.

   Notes:
   The dense format is fully compatible with standard Fortran 77
   storage by columns.

   The data input variable is intended primarily for Fortran programmers
   who wish to allocate their own matrix memory space.  Most users should
   set data=PETSC_NULL.

   Level: intermediate

.keywords: matrix,dense, parallel

.seealso: MatCreate(), MatCreateSeqDense(), MatSetValues()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMPIDenseSetPreallocation(Mat mat,PetscScalar *data)
{
  PetscErrorCode ierr,(*f)(Mat,PetscScalar *);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatMPIDenseSetPreallocation_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat,data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateMPIDense"
/*@C
   MatCreateMPIDense - Creates a sparse parallel matrix in dense format.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
.  n - number of local columns (or PETSC_DECIDE to have calculated if N is given)
.  M - number of global rows (or PETSC_DECIDE to have calculated if m is given)
.  N - number of global columns (or PETSC_DECIDE to have calculated if n is given)
-  data - optional location of matrix data.  Set data=PETSC_NULL (PETSC_NULL_SCALAR for Fortran users) for PETSc
   to control all matrix memory allocation.

   Output Parameter:
.  A - the matrix

   Notes:
   The dense format is fully compatible with standard Fortran 77
   storage by columns.

   The data input variable is intended primarily for Fortran programmers
   who wish to allocate their own matrix memory space.  Most users should
   set data=PETSC_NULL (PETSC_NULL_SCALAR for Fortran users).

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   Level: intermediate

.keywords: matrix,dense, parallel

.seealso: MatCreate(), MatCreateSeqDense(), MatSetValues()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMPIDense(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscScalar *data,Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIDENSE);CHKERRQ(ierr);
    ierr = MatMPIDenseSetPreallocation(*A,data);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(*A,data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate_MPIDense"
static PetscErrorCode MatDuplicate_MPIDense(Mat A,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat            mat;
  Mat_MPIDense   *a,*oldmat = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *newmat       = 0;
  ierr = MatCreate(A->comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,A->rmap.n,A->cmap.n,A->rmap.N,A->cmap.N);CHKERRQ(ierr);
  ierr = MatSetType(mat,A->type_name);CHKERRQ(ierr);
  a                 = (Mat_MPIDense*)mat->data;
  ierr              = PetscMemcpy(mat->ops,A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
  mat->factor       = A->factor;
  mat->assembled    = PETSC_TRUE;
  mat->preallocated = PETSC_TRUE;

  mat->rmap.rstart     = A->rmap.rstart;
  mat->rmap.rend       = A->rmap.rend;
  a->size              = oldmat->size;
  a->rank              = oldmat->rank;
  mat->insertmode      = NOT_SET_VALUES;
  a->nvec              = oldmat->nvec;
  a->donotstash        = oldmat->donotstash;
 
  ierr = PetscMemcpy(mat->rmap.range,A->rmap.range,(a->size+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(mat->cmap.range,A->cmap.range,(a->size+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = MatStashCreate_Private(A->comm,1,&mat->stash);CHKERRQ(ierr);

  ierr = MatSetUpMultiply_MPIDense(mat);CHKERRQ(ierr);
  ierr = MatDuplicate(oldmat->A,cpvalues,&a->A);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->A);CHKERRQ(ierr);
  *newmat = mat;
  PetscFunctionReturn(0);
}

#include "petscsys.h"

#undef __FUNCT__  
#define __FUNCT__ "MatLoad_MPIDense_DenseInFile"
PetscErrorCode MatLoad_MPIDense_DenseInFile(MPI_Comm comm,PetscInt fd,PetscInt M,PetscInt N, MatType type,Mat *newmat)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       *rowners,i,m,nz,j;
  PetscScalar    *array,*vals,*vals_ptr;
  MPI_Status     status;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* determine ownership of all rows */
  m          = M/size + ((M % size) > rank);
  ierr       = PetscMalloc((size+2)*sizeof(PetscInt),&rowners);CHKERRQ(ierr);
  ierr       = MPI_Allgather(&m,1,MPIU_INT,rowners+1,1,MPIU_INT,comm);CHKERRQ(ierr);
  rowners[0] = 0;
  for (i=2; i<=size; i++) {
    rowners[i] += rowners[i-1];
  }

  ierr = MatCreate(comm,newmat);CHKERRQ(ierr);
  ierr = MatSetSizes(*newmat,m,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*newmat,type);CHKERRQ(ierr);
  ierr = MatMPIDenseSetPreallocation(*newmat,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetArray(*newmat,&array);CHKERRQ(ierr);

  if (!rank) {
    ierr = PetscMalloc(m*N*sizeof(PetscScalar),&vals);CHKERRQ(ierr);

    /* read in my part of the matrix numerical values  */
    ierr = PetscBinaryRead(fd,vals,m*N,PETSC_SCALAR);CHKERRQ(ierr);
    
    /* insert into matrix-by row (this is why cannot directly read into array */
    vals_ptr = vals;
    for (i=0; i<m; i++) {
      for (j=0; j<N; j++) {
        array[i + j*m] = *vals_ptr++;
      }
    }

    /* read in other processors and ship out */
    for (i=1; i<size; i++) {
      nz   = (rowners[i+1] - rowners[i])*N;
      ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
      ierr = MPI_Send(vals,nz,MPIU_SCALAR,i,(*newmat)->tag,comm);CHKERRQ(ierr);
    }
  } else {
    /* receive numeric values */
    ierr = PetscMalloc(m*N*sizeof(PetscScalar),&vals);CHKERRQ(ierr);

    /* receive message of values*/
    ierr = MPI_Recv(vals,m*N,MPIU_SCALAR,0,(*newmat)->tag,comm,&status);CHKERRQ(ierr);

    /* insert into matrix-by row (this is why cannot directly read into array */
    vals_ptr = vals;
    for (i=0; i<m; i++) {
      for (j=0; j<N; j++) {
        array[i + j*m] = *vals_ptr++;
      }
    }
  }
  ierr = PetscFree(rowners);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLoad_MPIDense"
PetscErrorCode MatLoad_MPIDense(PetscViewer viewer, MatType type,Mat *newmat)
{
  Mat            A;
  PetscScalar    *vals,*svals;
  MPI_Comm       comm = ((PetscObject)viewer)->comm;
  MPI_Status     status;
  PetscMPIInt    rank,size,tag = ((PetscObject)viewer)->tag,*rowners,*sndcounts,m,maxnz;
  PetscInt       header[4],*rowlengths = 0,M,N,*cols;
  PetscInt       *ourlens,*procsnz = 0,*offlens,jj,*mycols,*smycols;
  PetscInt       i,nz,j,rstart,rend;
  int            fd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,PETSC_INT);CHKERRQ(ierr);
    if (header[0] != MAT_FILE_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"not matrix object");
  }

  ierr = MPI_Bcast(header+1,3,MPIU_INT,0,comm);CHKERRQ(ierr);
  M = header[1]; N = header[2]; nz = header[3];

  /*
       Handle case where matrix is stored on disk as a dense matrix 
  */
  if (nz == MATRIX_BINARY_FORMAT_DENSE) {
    ierr = MatLoad_MPIDense_DenseInFile(comm,fd,M,N,type,newmat);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* determine ownership of all rows */
  m          = M/size + ((M % size) > rank);
  ierr       = PetscMalloc((size+2)*sizeof(PetscInt),&rowners);CHKERRQ(ierr);
  ierr       = MPI_Allgather(&m,1,MPI_INT,rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  rowners[0] = 0;
  for (i=2; i<=size; i++) {
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 

  /* distribute row lengths to all processors */
  ierr    = PetscMalloc(2*(rend-rstart+1)*sizeof(PetscInt),&ourlens);CHKERRQ(ierr);
  offlens = ourlens + (rend-rstart);
  if (!rank) {
    ierr = PetscMalloc(M*sizeof(PetscInt),&rowlengths);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);
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
      for (j=rowners[i]; j< rowners[i+1]; j++) {
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
    nz = procsnz[0];
    ierr = PetscMalloc(nz*sizeof(PetscInt),&mycols);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,mycols,nz,PETSC_INT);CHKERRQ(ierr);

    /* read in every one elses and ship off */
    for (i=1; i<size; i++) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
      ierr = MPI_Send(cols,nz,MPIU_INT,i,tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(cols);CHKERRQ(ierr);
  } else {
    /* determine buffer space needed for message */
    nz = 0;
    for (i=0; i<m; i++) {
      nz += ourlens[i];
    }
    ierr = PetscMalloc((nz+1)*sizeof(PetscInt),&mycols);CHKERRQ(ierr);

    /* receive message of column indices*/
    ierr = MPI_Recv(mycols,nz,MPIU_INT,0,tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPIU_INT,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"something is wrong with file");
  }

  /* loop over local rows, determining number of off diagonal entries */
  ierr = PetscMemzero(offlens,m*sizeof(PetscInt));CHKERRQ(ierr);
  jj = 0;
  for (i=0; i<m; i++) {
    for (j=0; j<ourlens[i]; j++) {
      if (mycols[jj] < rstart || mycols[jj] >= rend) offlens[i]++;
      jj++;
    }
  }

  /* create our matrix */
  for (i=0; i<m; i++) {
    ourlens[i] -= offlens[i];
  }
  ierr = MatCreate(comm,newmat);CHKERRQ(ierr);
  ierr = MatSetSizes(*newmat,m,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*newmat,type);CHKERRQ(ierr);
  ierr = MatMPIDenseSetPreallocation(*newmat,PETSC_NULL);CHKERRQ(ierr);
  A = *newmat;
  for (i=0; i<m; i++) {
    ourlens[i] += offlens[i];
  }

  if (!rank) {
    ierr = PetscMalloc(maxnz*sizeof(PetscScalar),&vals);CHKERRQ(ierr);

    /* read in my part of the matrix numerical values  */
    nz = procsnz[0];
    ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
    
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

    /* read in other processors and ship out */
    for (i=1; i<size; i++) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
      ierr = MPI_Send(vals,nz,MPIU_SCALAR,i,A->tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(procsnz);CHKERRQ(ierr);
  } else {
    /* receive numeric values */
    ierr = PetscMalloc((nz+1)*sizeof(PetscScalar),&vals);CHKERRQ(ierr);

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

#undef __FUNCT__  
#define __FUNCT__ "MatEqual_MPIDense"
PetscErrorCode MatEqual_MPIDense(Mat A,Mat B,PetscTruth *flag)
{
  Mat_MPIDense   *matB = (Mat_MPIDense*)B->data,*matA = (Mat_MPIDense*)A->data;
  Mat            a,b;
  PetscTruth     flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  a = matA->A; 
  b = matB->A;
  ierr = MatEqual(a,b,&flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



