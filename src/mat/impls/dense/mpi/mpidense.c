
/*
   Basic functions for basic parallel dense matrices.
*/

    
#include <../src/mat/impls/dense/mpi/mpidense.h>    /*I   "petscmat.h"  I*/
#if defined(PETSC_HAVE_PLAPACK)
static PetscMPIInt Plapack_nprows,Plapack_npcols,Plapack_ierror,Plapack_nb_alg;
static MPI_Comm Plapack_comm_2d;
EXTERN_C_BEGIN 
#include <PLA.h>
EXTERN_C_END 

typedef struct {
  PLA_Obj        A,pivots;
  PLA_Template   templ;
  MPI_Datatype   datatype;
  PetscInt       nb,rstart;
  VecScatter     ctx;
  IS             is_pla,is_petsc;
  PetscBool      pla_solved;
  MatStructure   mstruct;
} Mat_Plapack;
#endif

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
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIDENSE,&flg);CHKERRQ(ierr);
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
  PetscInt       lrow,rstart = A->rmap->rstart,rend = A->rmap->rend;

  PetscFunctionBegin;
  if (row < rstart || row >= rend) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"only local rows");
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
PetscErrorCode  MatGetDiagonalBlock_MPIDense(Mat A,Mat *a)
{
  Mat_MPIDense   *mdn = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       m = A->rmap->n,rstart = A->rmap->rstart;
  PetscScalar    *array;
  MPI_Comm       comm;
  Mat            B;

  PetscFunctionBegin;
  if (A->rmap->N != A->cmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only square matrices supported.");

  ierr = PetscObjectQuery((PetscObject)A,"DiagonalBlock",(PetscObject*)&B);CHKERRQ(ierr);
  if (!B) {
    ierr = PetscObjectGetComm((PetscObject)(mdn->A),&comm);CHKERRQ(ierr);
    ierr = MatCreate(comm,&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,m,m,m,m);CHKERRQ(ierr);
    ierr = MatSetType(B,((PetscObject)mdn->A)->type_name);CHKERRQ(ierr);
    ierr = MatGetArray(mdn->A,&array);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(B,array+m*rstart);CHKERRQ(ierr);
    ierr = MatRestoreArray(mdn->A,&array);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)A,"DiagonalBlock",(PetscObject)B);CHKERRQ(ierr);
    *a = B;
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  } else {
    *a = B;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_MPIDense"
PetscErrorCode MatSetValues_MPIDense(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPIDense   *A = (Mat_MPIDense*)mat->data;
  PetscErrorCode ierr;
  PetscInt       i,j,rstart = mat->rmap->rstart,rend = mat->rmap->rend,row;
  PetscBool      roworiented = A->roworiented;

  PetscFunctionBegin;
  if (v) PetscValidScalarPointer(v,6);
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) continue;
    if (idxm[i] >= mat->rmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      if (roworiented) {
        ierr = MatSetValues(A->A,1,&row,n,idxn,v+i*n,addv);CHKERRQ(ierr);
      } else {
        for (j=0; j<n; j++) {
          if (idxn[j] < 0) continue;
          if (idxn[j] >= mat->cmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large");
          ierr = MatSetValues(A->A,1,&row,1,&idxn[j],v+i+j*m,addv);CHKERRQ(ierr);
        }
      }
    } else {
      if (!A->donotstash) {
        mat->assembled = PETSC_FALSE;
        if (roworiented) {
          ierr = MatStashValuesRow_Private(&mat->stash,idxm[i],n,idxn,v+i*n,PETSC_FALSE);CHKERRQ(ierr);
        } else {
          ierr = MatStashValuesCol_Private(&mat->stash,idxm[i],n,idxn,v+i,m,PETSC_FALSE);CHKERRQ(ierr);
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
  PetscInt       i,j,rstart = mat->rmap->rstart,rend = mat->rmap->rend,row;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) continue; /* SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row"); */
    if (idxm[i] >= mat->rmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
    if (idxm[i] >= rstart && idxm[i] < rend) {
      row = idxm[i] - rstart;
      for (j=0; j<n; j++) {
        if (idxn[j] < 0) continue; /* SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative column"); */
        if (idxn[j] >= mat->cmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large");
        ierr = MatGetValues(mdn->A,1,&row,1,&idxn[j],v+i*n+j);CHKERRQ(ierr);
      }
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only local values currently supported");
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
static PetscErrorCode MatGetSubMatrix_MPIDense(Mat A,IS isrow,IS iscol,MatReuse scall,Mat *B)
{
  Mat_MPIDense   *mat = (Mat_MPIDense*)A->data,*newmatd;
  Mat_SeqDense   *lmat = (Mat_SeqDense*)mat->A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,rstart,rend,nrows,ncols,Ncols,nlrows,nlcols;
  const PetscInt *irow,*icol;
  PetscScalar    *av,*bv,*v = lmat->v;
  Mat            newmat;
  IS             iscol_local;

  PetscFunctionBegin;
  ierr = ISAllGather(iscol,&iscol_local);CHKERRQ(ierr);
  ierr = ISGetIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISGetIndices(iscol_local,&icol);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrow,&nrows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscol,&ncols);CHKERRQ(ierr);
  ierr = ISGetSize(iscol,&Ncols);CHKERRQ(ierr); /* global number of columns, size of iscol_local */

  /* No parallel redistribution currently supported! Should really check each index set
     to comfirm that it is OK.  ... Currently supports only submatrix same partitioning as
     original matrix! */

  ierr = MatGetLocalSize(A,&nlrows,&nlcols);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  
  /* Check submatrix call */
  if (scall == MAT_REUSE_MATRIX) {
    /* SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Reused submatrix wrong size"); */
    /* Really need to test rows and column sizes! */
    newmat = *B;
  } else {
    /* Create and fill new matrix */
    ierr = MatCreate(((PetscObject)A)->comm,&newmat);CHKERRQ(ierr);
    ierr = MatSetSizes(newmat,nrows,ncols,PETSC_DECIDE,Ncols);CHKERRQ(ierr);
    ierr = MatSetType(newmat,((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = MatMPIDenseSetPreallocation(newmat,PETSC_NULL);CHKERRQ(ierr);
  }

  /* Now extract the data pointers and do the copy, column at a time */
  newmatd = (Mat_MPIDense*)newmat->data;
  bv      = ((Mat_SeqDense *)newmatd->A->data)->v;
  
  for (i=0; i<Ncols; i++) {
    av = v + ((Mat_SeqDense *)mat->A->data)->lda*icol[i];
    for (j=0; j<nrows; j++) {
      *bv++ = av[irow[j] - rstart];
    }
  }

  /* Assemble the matrices so that the correct flags are set */
  ierr = MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Free work space */
  ierr = ISRestoreIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol_local,&icol);CHKERRQ(ierr);
  ierr = ISDestroy(&iscol_local);CHKERRQ(ierr);
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
  MPI_Comm       comm = ((PetscObject)mat)->comm;
  PetscErrorCode ierr;
  PetscInt       nstash,reallocs;
  InsertMode     addv;

  PetscFunctionBegin;
  /* make sure all processors are either in INSERTMODE or ADDMODE */
  ierr = MPI_Allreduce(&mat->insertmode,&addv,1,MPI_INT,MPI_BOR,comm);CHKERRQ(ierr);
  if (addv == (ADD_VALUES|INSERT_VALUES)) SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot mix adds/inserts on different procs");
  mat->insertmode = addv; /* in case this processor had no cache */

  ierr = MatStashScatterBegin_Private(mat,&mat->stash,mat->rmap->range);CHKERRQ(ierr);
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
PetscErrorCode MatZeroRows_MPIDense(Mat A,PetscInt N,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  Mat_MPIDense      *l = (Mat_MPIDense*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,*owners = A->rmap->range;
  PetscInt          *nprocs,j,idx,nsends;
  PetscInt          nmax,*svalues,*starts,*owner,nrecvs;
  PetscInt          *rvalues,tag = ((PetscObject)A)->tag,count,base,slen,*source;
  PetscInt          *lens,*lrows,*values;
  PetscMPIInt       n,imdex,rank = l->rank,size = l->size;
  MPI_Comm          comm = ((PetscObject)A)->comm;
  MPI_Request       *send_waits,*recv_waits;
  MPI_Status        recv_status,*send_status;
  PetscBool         found;
  const PetscScalar *xx;
  PetscScalar       *bb;

  PetscFunctionBegin;
  if (A->rmap->N != A->cmap->N) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"Only handles square matrices");
  if (A->rmap->n != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only handles matrices with identical column and row ownership");
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
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index out of range");
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
  ierr   = PetscMalloc2(nrecvs,PetscInt,&lens,nrecvs,PetscInt,&source);CHKERRQ(ierr);
  count  = nrecvs;
  slen   = 0;
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
  ierr = PetscFree2(lens,source);CHKERRQ(ierr);
  ierr = PetscFree(owner);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);
    
  /* fix right hand side if needed */
  if (x && b) {
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
    for (i=0; i<slen; i++) {
      bb[lrows[i]] = diag*xx[lrows[i]];
    }
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(b,&bb);CHKERRQ(ierr);
  }

  /* actually zap the local rows */
  ierr = MatZeroRows(l->A,slen,lrows,0.0,0,0);CHKERRQ(ierr);
  if (diag != 0.0) {
    Mat_SeqDense *ll = (Mat_SeqDense*)l->A->data;
    PetscInt      m = ll->lda, i;
 
    for (i=0; i<slen; i++) {
      ll->v[lrows[i] + m*(A->cmap->rstart + lrows[i])] = diag;
    }
  }
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
  ierr = VecScatterBegin(mdn->Mvctx,xx,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(mdn->Mvctx,xx,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
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
  ierr = VecScatterBegin(mdn->Mvctx,xx,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(mdn->Mvctx,xx,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
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
  ierr = VecScatterBegin(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
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
  ierr = VecScatterBegin(a->Mvctx,a->lvec,zz,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,a->lvec,zz,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_MPIDense"
PetscErrorCode MatGetDiagonal_MPIDense(Mat A,Vec v)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  Mat_SeqDense   *aloc = (Mat_SeqDense*)a->A->data;
  PetscErrorCode ierr;
  PetscInt       len,i,n,m = A->rmap->n,radd;
  PetscScalar    *x,zero = 0.0;
  
  PetscFunctionBegin;
  ierr = VecSet(v,zero);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetSize(v,&n);CHKERRQ(ierr);
  if (n != A->rmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming mat and vec");
  len  = PetscMin(a->A->rmap->n,a->A->cmap->n);
  radd = A->rmap->rstart*m;
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
#if defined(PETSC_HAVE_PLAPACK)
  Mat_Plapack   *lu=(Mat_Plapack*)mat->spptr;
#endif

  PetscFunctionBegin;

#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows=%D, Cols=%D",mat->rmap->N,mat->cmap->N);
#endif
  ierr = MatStashDestroy_Private(&mat->stash);CHKERRQ(ierr);
  ierr = MatDestroy(&mdn->A);CHKERRQ(ierr);
  ierr = VecDestroy(&mdn->lvec);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&mdn->Mvctx);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PLAPACK)
  if (lu) {
    ierr = PLA_Obj_free(&lu->A);CHKERRQ(ierr);
    ierr = PLA_Obj_free (&lu->pivots);CHKERRQ(ierr);
    ierr = PLA_Temp_free(&lu->templ);CHKERRQ(ierr);
    ierr = ISDestroy(&lu->is_pla);CHKERRQ(ierr);
    ierr = ISDestroy(&lu->is_petsc);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&lu->ctx);CHKERRQ(ierr);
  }
  ierr = PetscFree(mat->spptr);CHKERRQ(ierr);
#endif

  ierr = PetscFree(mat->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)mat,0);CHKERRQ(ierr);
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
  Mat_MPIDense      *mdn = (Mat_MPIDense*)mat->data;
  PetscErrorCode    ierr;
  PetscViewerFormat format;
  int               fd;
  PetscInt          header[4],mmax,N = mat->cmap->N,i,j,m,k;
  PetscMPIInt       rank,tag  = ((PetscObject)viewer)->tag,size;
  PetscScalar       *work,*v,*vv;
  Mat_SeqDense      *a = (Mat_SeqDense*)mdn->A->data;

  PetscFunctionBegin;
  if (mdn->size == 1) {
    ierr = MatView(mdn->A,viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(((PetscObject)mat)->comm,&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(((PetscObject)mat)->comm,&size);CHKERRQ(ierr);

    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_NATIVE) {

      if (!rank) {
        /* store the matrix as a dense matrix */
        header[0] = MAT_FILE_CLASSID;
        header[1] = mat->rmap->N;
        header[2] = N;
        header[3] = MATRIX_BINARY_FORMAT_DENSE;
        ierr = PetscBinaryWrite(fd,header,4,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);

        /* get largest work array needed for transposing array */
        mmax = mat->rmap->n;
        for (i=1; i<size; i++) {
          mmax = PetscMax(mmax,mat->rmap->range[i+1] - mat->rmap->range[i]);
        }
        ierr = PetscMalloc(mmax*N*sizeof(PetscScalar),&work);CHKERRQ(ierr);

        /* write out local array, by rows */
        m    = mat->rmap->n;
        v    = a->v;
        for (j=0; j<N; j++) {
          for (i=0; i<m; i++) {
            work[j + i*N] = *v++;
          }
        }
        ierr = PetscBinaryWrite(fd,work,m*N,PETSC_SCALAR,PETSC_FALSE);CHKERRQ(ierr);
        /* get largest work array to receive messages from other processes, excludes process zero */
        mmax = 0;
        for (i=1; i<size; i++) {
          mmax = PetscMax(mmax,mat->rmap->range[i+1] - mat->rmap->range[i]);
        }
        ierr = PetscMalloc(mmax*N*sizeof(PetscScalar),&vv);CHKERRQ(ierr);
        for(k = 1; k < size; k++) {
          v    = vv;
          m    = mat->rmap->range[k+1] - mat->rmap->range[k];
          ierr = MPIULong_Recv(v,m*N,MPIU_SCALAR,k,tag,((PetscObject)mat)->comm);CHKERRQ(ierr);

          for(j = 0; j < N; j++) {
            for(i = 0; i < m; i++) {
              work[j + i*N] = *v++;
            }
          }
          ierr = PetscBinaryWrite(fd,work,m*N,PETSC_SCALAR,PETSC_FALSE);CHKERRQ(ierr);
        }
        ierr = PetscFree(work);CHKERRQ(ierr);
        ierr = PetscFree(vv);CHKERRQ(ierr);
      } else {
        ierr = MPIULong_Send(a->v,mat->rmap->n*mat->cmap->N,MPIU_SCALAR,0,tag,((PetscObject)mat)->comm);CHKERRQ(ierr);
      }
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"To store a parallel dense matrix you must first call PetscViewerSetFormat(viewer,PETSC_VIEWER_NATIVE)");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIDense_ASCIIorDraworSocket"
static PetscErrorCode MatView_MPIDense_ASCIIorDraworSocket(Mat mat,PetscViewer viewer)
{
  Mat_MPIDense          *mdn = (Mat_MPIDense*)mat->data;
  PetscErrorCode        ierr;
  PetscMPIInt           size = mdn->size,rank = mdn->rank; 
  const PetscViewerType vtype;
  PetscBool             iascii,isdraw;
  PetscViewer           sviewer;
  PetscViewerFormat     format;
#if defined(PETSC_HAVE_PLAPACK)
  Mat_Plapack           *lu; 
#endif

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetType(viewer,&vtype);CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      MatInfo info;
      ierr = MatGetInfo(mat,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] local rows %D nz %D nz alloced %D mem %D \n",rank,mat->rmap->n,(PetscInt)info.nz_used,(PetscInt)info.nz_allocated,(PetscInt)info.memory);CHKERRQ(ierr);       
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PLAPACK)
      ierr = PetscViewerASCIIPrintf(viewer,"PLAPACK run parameters:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Processor mesh: nprows %d, npcols %d\n",Plapack_nprows, Plapack_npcols);CHKERRQ(ierr);      
      ierr = PetscViewerASCIIPrintf(viewer,"  Error checking: %d\n",Plapack_ierror);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Algorithmic block size: %d\n",Plapack_nb_alg);CHKERRQ(ierr);
      if (mat->factortype){
        lu=(Mat_Plapack*)(mat->spptr);
        ierr = PetscViewerASCIIPrintf(viewer,"  Distr. block size nb: %d \n",lu->nb);CHKERRQ(ierr); 
      }
#else
      ierr = VecScatterView(mdn->Mvctx,viewer);CHKERRQ(ierr);
#endif
      PetscFunctionReturn(0); 
    } else if (format == PETSC_VIEWER_ASCII_INFO) {
      PetscFunctionReturn(0);
    }
  } else if (isdraw) {
    PetscDraw  draw;
    PetscBool  isnull;

    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
    if (isnull) PetscFunctionReturn(0);
  }

  if (size == 1) { 
    ierr = MatView(mdn->A,viewer);CHKERRQ(ierr);
  } else {
    /* assemble the entire matrix onto first processor. */
    Mat         A;
    PetscInt    M = mat->rmap->N,N = mat->cmap->N,m,row,i,nz;
    PetscInt    *cols;
    PetscScalar *vals;

    ierr = MatCreate(((PetscObject)mat)->comm,&A);CHKERRQ(ierr);
    if (!rank) {
      ierr = MatSetSizes(A,M,N,M,N);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(A,0,0,M,N);CHKERRQ(ierr);
    }
    /* Since this is a temporary matrix, MATMPIDENSE instead of ((PetscObject)A)->type_name here is probably acceptable. */
    ierr = MatSetType(A,MATMPIDENSE);CHKERRQ(ierr);
    ierr = MatMPIDenseSetPreallocation(A,PETSC_NULL);
    ierr = PetscLogObjectParent(mat,A);CHKERRQ(ierr);

    /* Copy the matrix ... This isn't the most efficient means,
       but it's quick for now */
    A->insertmode = INSERT_VALUES;
    row = mat->rmap->rstart; m = mdn->A->rmap->n;
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
      ierr = PetscObjectSetName((PetscObject)((Mat_MPIDense*)(A->data))->A,((PetscObject)mat)->name);CHKERRQ(ierr);
      /* Set the type name to MATMPIDense so that the correct type can be printed out by PetscObjectPrintClassNamePrefixType() in MatView_SeqDense_ASCII()*/
      PetscStrcpy(((PetscObject)((Mat_MPIDense*)(A->data))->A)->type_name,MATMPIDENSE);
      ierr = MatView(((Mat_MPIDense*)(A->data))->A,sviewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIDense"
PetscErrorCode MatView_MPIDense(Mat mat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isbinary,isdraw,issocket;
 
  PetscFunctionBegin;
  
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSOCKET,&issocket);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);

  if (iascii || issocket || isdraw) {
    ierr = MatView_MPIDense_ASCIIorDraworSocket(mat,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_MPIDense_Binary(mat,viewer);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported by MPI dense matrix",((PetscObject)viewer)->type_name);
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
    ierr = MPI_Allreduce(isend,irecv,5,MPIU_REAL,MPIU_MAX,((PetscObject)A)->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPI_Allreduce(isend,irecv,5,MPIU_REAL,MPIU_SUM,((PetscObject)A)->comm);CHKERRQ(ierr);
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
PetscErrorCode MatSetOption_MPIDense(Mat A,MatOption op,PetscBool  flg)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_NEW_NONZERO_LOCATIONS:
  case MAT_NEW_NONZERO_LOCATION_ERR:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    break;
  case MAT_ROW_ORIENTED:
    a->roworiented = flg;
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    break;
  case MAT_NEW_DIAGONALS:
  case MAT_KEEP_NONZERO_PATTERN:
  case MAT_USE_HASH_TABLE:
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
    break;
  case MAT_IGNORE_OFF_PROC_ENTRIES:
    a->donotstash = flg;
    break;
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
  case MAT_SYMMETRY_ETERNAL:
  case MAT_IGNORE_LOWER_TRIANGULAR:
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %s",MatOptions[op]);
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
  PetscInt       i,j,s2a,s3a,s2,s3,m=mdn->A->rmap->n,n=mdn->A->cmap->n;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&s2,&s3);CHKERRQ(ierr);
  if (ll) {
    ierr = VecGetLocalSize(ll,&s2a);CHKERRQ(ierr);
    if (s2a != s2) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Left scaling vector non-conforming local size, %d != %d.", s2a, s2);
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
    if (s3a != s3) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Right scaling vec non-conforming local size, %d != %d.", s3a, s3);
    ierr = VecScatterBegin(mdn->Mvctx,rr,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(mdn->Mvctx,rr,mdn->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
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
      for (i=0; i<mdn->A->cmap->n*mdn->A->rmap->n; i++) {
#if defined(PETSC_USE_COMPLEX)
        sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      ierr = MPI_Allreduce(&sum,nrm,1,MPIU_REAL,MPIU_SUM,((PetscObject)A)->comm);CHKERRQ(ierr);
      *nrm = PetscSqrtReal(*nrm);
      ierr = PetscLogFlops(2.0*mdn->A->cmap->n*mdn->A->rmap->n);CHKERRQ(ierr);
    } else if (type == NORM_1) { 
      PetscReal *tmp,*tmp2;
      ierr = PetscMalloc2(A->cmap->N,PetscReal,&tmp,A->cmap->N,PetscReal,&tmp2);CHKERRQ(ierr);
      ierr = PetscMemzero(tmp,A->cmap->N*sizeof(PetscReal));CHKERRQ(ierr);
      ierr = PetscMemzero(tmp2,A->cmap->N*sizeof(PetscReal));CHKERRQ(ierr);
      *nrm = 0.0;
      v = mat->v;
      for (j=0; j<mdn->A->cmap->n; j++) {
        for (i=0; i<mdn->A->rmap->n; i++) {
          tmp[j] += PetscAbsScalar(*v);  v++;
        }
      }
      ierr = MPI_Allreduce(tmp,tmp2,A->cmap->N,MPIU_REAL,MPIU_SUM,((PetscObject)A)->comm);CHKERRQ(ierr);
      for (j=0; j<A->cmap->N; j++) {
        if (tmp2[j] > *nrm) *nrm = tmp2[j];
      }
      ierr = PetscFree2(tmp,tmp);CHKERRQ(ierr);
      ierr = PetscLogFlops(A->cmap->n*A->rmap->n);CHKERRQ(ierr);
    } else if (type == NORM_INFINITY) { /* max row norm */
      PetscReal ntemp;
      ierr = MatNorm(mdn->A,type,&ntemp);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&ntemp,nrm,1,MPIU_REAL,MPIU_MAX,((PetscObject)A)->comm);CHKERRQ(ierr);
    } else SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"No support for two norm");
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatTranspose_MPIDense"
PetscErrorCode MatTranspose_MPIDense(Mat A,MatReuse reuse,Mat *matout)
{ 
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  Mat_SeqDense   *Aloc = (Mat_SeqDense*)a->A->data;
  Mat            B;
  PetscInt       M = A->rmap->N,N = A->cmap->N,m,n,*rwork,rstart = A->rmap->rstart;
  PetscErrorCode ierr;
  PetscInt       j,i;
  PetscScalar    *v;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX && A == *matout && M != N) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"Supports square matrix only in-place");
  if (reuse == MAT_INITIAL_MATRIX || A == *matout) {
    ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,A->cmap->n,A->rmap->n,N,M);CHKERRQ(ierr);
    ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = MatMPIDenseSetPreallocation(B,PETSC_NULL);CHKERRQ(ierr);
  } else {
    B = *matout;
  }

  m = a->A->rmap->n; n = a->A->cmap->n; v = Aloc->v;
  ierr = PetscMalloc(m*sizeof(PetscInt),&rwork);CHKERRQ(ierr);
  for (i=0; i<m; i++) rwork[i] = rstart + i;
  for (j=0; j<n; j++) {
    ierr = MatSetValues(B,1,&j,m,rwork,v,INSERT_VALUES);CHKERRQ(ierr);
    v   += m;
  } 
  ierr = PetscFree(rwork);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX || *matout != A) {
    *matout = B;
  } else {
    ierr = MatHeaderMerge(A,B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


static PetscErrorCode MatDuplicate_MPIDense(Mat,MatDuplicateOption,Mat *);
extern PetscErrorCode MatScale_MPIDense(Mat,PetscScalar);

#undef __FUNCT__  
#define __FUNCT__ "MatSetUp_MPIDense"
PetscErrorCode MatSetUp_MPIDense(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr =  MatMPIDenseSetPreallocation(A,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_PLAPACK)

#undef __FUNCT__   
#define __FUNCT__ "MatMPIDenseCopyToPlapack"
PetscErrorCode MatMPIDenseCopyToPlapack(Mat A,Mat F)
{
  Mat_Plapack    *lu = (Mat_Plapack*)(F)->spptr;
  PetscErrorCode ierr;
  PetscInt       M=A->cmap->N,m=A->rmap->n,rstart;
  PetscScalar    *array;
  PetscReal      one = 1.0;

  PetscFunctionBegin;
  /* Copy A into F->lu->A */
  ierr = PLA_Obj_set_to_zero(lu->A);CHKERRQ(ierr);
  ierr = PLA_API_begin();CHKERRQ(ierr);
  ierr = PLA_Obj_API_open(lu->A);CHKERRQ(ierr);  
  ierr = MatGetOwnershipRange(A,&rstart,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetArray(A,&array);CHKERRQ(ierr);
  ierr = PLA_API_axpy_matrix_to_global(m,M, &one,(void *)array,m,lu->A,rstart,0);CHKERRQ(ierr);
  ierr = MatRestoreArray(A,&array);CHKERRQ(ierr);
  ierr = PLA_Obj_API_close(lu->A);CHKERRQ(ierr); 
  ierr = PLA_API_end();CHKERRQ(ierr); 
  lu->rstart = rstart;
  PetscFunctionReturn(0);
}

#undef __FUNCT__   
#define __FUNCT__ "MatMPIDenseCopyFromPlapack"
PetscErrorCode MatMPIDenseCopyFromPlapack(Mat F,Mat A)
{
  Mat_Plapack    *lu = (Mat_Plapack*)(F)->spptr;
  PetscErrorCode ierr;
  PetscInt       M=A->cmap->N,m=A->rmap->n,rstart;
  PetscScalar    *array;
  PetscReal      one = 1.0;

  PetscFunctionBegin;
  /* Copy F into A->lu->A */
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = PLA_API_begin();CHKERRQ(ierr);
  ierr = PLA_Obj_API_open(lu->A);CHKERRQ(ierr);  
  ierr = MatGetOwnershipRange(A,&rstart,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetArray(A,&array);CHKERRQ(ierr);
  ierr = PLA_API_axpy_global_to_matrix(m,M, &one,lu->A,rstart,0,(void *)array,m);CHKERRQ(ierr);
  ierr = MatRestoreArray(A,&array);CHKERRQ(ierr);
  ierr = PLA_Obj_API_close(lu->A);CHKERRQ(ierr); 
  ierr = PLA_API_end();CHKERRQ(ierr); 
  lu->rstart = rstart;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultNumeric_MPIDense_MPIDense"
PetscErrorCode MatMatMultNumeric_MPIDense_MPIDense(Mat A,Mat B,Mat C) 
{
  PetscErrorCode ierr;
  Mat_Plapack    *luA = (Mat_Plapack*)A->spptr;
  Mat_Plapack    *luB = (Mat_Plapack*)B->spptr;
  Mat_Plapack    *luC = (Mat_Plapack*)C->spptr;
  PLA_Obj        alpha = NULL,beta = NULL;

  PetscFunctionBegin;
  ierr = MatMPIDenseCopyToPlapack(A,A);CHKERRQ(ierr);
  ierr = MatMPIDenseCopyToPlapack(B,B);CHKERRQ(ierr);

  /* 
  ierr = PLA_Global_show("A = ",luA->A,"%g ","");CHKERRQ(ierr);
  ierr = PLA_Global_show("B = ",luB->A,"%g ","");CHKERRQ(ierr);
  */

  /* do the multiply in PLA  */
  ierr = PLA_Create_constants_conf_to(luA->A,NULL,NULL,&alpha);CHKERRQ(ierr);
  ierr = PLA_Create_constants_conf_to(luC->A,NULL,&beta,NULL);CHKERRQ(ierr);
  CHKMEMQ;

  ierr = PLA_Gemm(PLA_NO_TRANSPOSE,PLA_NO_TRANSPOSE,alpha,luA->A,luB->A,beta,luC->A); /* CHKERRQ(ierr); */
  CHKMEMQ;
  ierr = PLA_Obj_free(&alpha);CHKERRQ(ierr);
  ierr = PLA_Obj_free(&beta);CHKERRQ(ierr);

  /*
  ierr = PLA_Global_show("C = ",luC->A,"%g ","");CHKERRQ(ierr);
  */
  ierr = MatMPIDenseCopyFromPlapack(C,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatMatMult_MPIDense_MPIDense(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C);

#undef __FUNCT__
#define __FUNCT__ "MatMatMultSymbolic_MPIDense_MPIDense"
PetscErrorCode MatMatMultSymbolic_MPIDense_MPIDense(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  PetscInt       m=A->rmap->n,n=B->cmap->n;
  Mat            Cmat;

  PetscFunctionBegin;
  if (A->cmap->n != B->rmap->n) SETERRQ2(((PetscObject)A)->comm,PETSC_ERR_ARG_SIZ,"A->cmap->n %d != B->rmap->n %d\n",A->cmap->n,B->rmap->n);
  SETERRQ(((PetscObject)A)->comm,PETSC_ERR_LIB,"Due to apparent bugs in PLAPACK,this is not currently supported");
  ierr = MatCreate(((PetscObject)B)->comm,&Cmat);CHKERRQ(ierr);
  ierr = MatSetSizes(Cmat,m,n,A->rmap->N,B->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(Cmat,MATMPIDENSE);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Cmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Cmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  Cmat->ops->matmult = MatMatMult_MPIDense_MPIDense;
  *C = Cmat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_MPIDense_MPIDense"
PetscErrorCode MatMatMult_MPIDense_MPIDense(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = MatMatMultSymbolic_MPIDense_MPIDense(A,B,fill,C);CHKERRQ(ierr);
  }
  ierr = MatMatMultNumeric_MPIDense_MPIDense(A,B,*C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_MPIDense"
PetscErrorCode MatSolve_MPIDense(Mat A,Vec b,Vec x)
{
  MPI_Comm       comm = ((PetscObject)A)->comm;
  Mat_Plapack    *lu = (Mat_Plapack*)A->spptr;
  PetscErrorCode ierr;
  PetscInt       M=A->rmap->N,m=A->rmap->n,rstart,i,j,*idx_pla,*idx_petsc,loc_m,loc_stride;
  PetscScalar    *array;
  PetscReal      one = 1.0;
  PetscMPIInt    size,rank,r_rank,r_nproc,c_rank,c_nproc;;
  PLA_Obj        v_pla = NULL;
  PetscScalar    *loc_buf;
  Vec            loc_x;
   
  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* Create PLAPACK vector objects, then copy b into PLAPACK b */
  PLA_Mvector_create(lu->datatype,M,1,lu->templ,PLA_ALIGN_FIRST,&v_pla);  
  PLA_Obj_set_to_zero(v_pla);

  /* Copy b into rhs_pla */
  PLA_API_begin();   
  PLA_Obj_API_open(v_pla);
  ierr = VecGetArray(b,&array);CHKERRQ(ierr);
  PLA_API_axpy_vector_to_global(m,&one,(void *)array,1,v_pla,lu->rstart);
  ierr = VecRestoreArray(b,&array);CHKERRQ(ierr);
  PLA_Obj_API_close(v_pla);
  PLA_API_end(); 

  if (A->factortype == MAT_FACTOR_LU){
    /* Apply the permutations to the right hand sides */
    PLA_Apply_pivots_to_rows (v_pla,lu->pivots);

    /* Solve L y = b, overwriting b with y */
    PLA_Trsv( PLA_LOWER_TRIANGULAR,PLA_NO_TRANSPOSE,PLA_UNIT_DIAG,lu->A,v_pla );

    /* Solve U x = y (=b), overwriting b with x */
    PLA_Trsv( PLA_UPPER_TRIANGULAR,PLA_NO_TRANSPOSE,PLA_NONUNIT_DIAG,lu->A,v_pla );
  } else { /* MAT_FACTOR_CHOLESKY */
    PLA_Trsv( PLA_LOWER_TRIANGULAR,PLA_NO_TRANSPOSE,PLA_NONUNIT_DIAG,lu->A,v_pla);
    PLA_Trsv( PLA_LOWER_TRIANGULAR,(lu->datatype == MPI_DOUBLE ? PLA_TRANSPOSE : PLA_CONJUGATE_TRANSPOSE),
                                    PLA_NONUNIT_DIAG,lu->A,v_pla);
  }

  /* Copy PLAPACK x into Petsc vector x  */   
  PLA_Obj_local_length(v_pla, &loc_m);
  PLA_Obj_local_buffer(v_pla, (void**)&loc_buf);
  PLA_Obj_local_stride(v_pla, &loc_stride);
  /*
    PetscPrintf(PETSC_COMM_SELF," [%d] b - local_m %d local_stride %d, loc_buf: %g %g, nb: %d\n",rank,loc_m,loc_stride,loc_buf[0],loc_buf[(loc_m-1)*loc_stride],lu->nb); 
  */
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,loc_m*loc_stride,loc_buf,&loc_x);CHKERRQ(ierr);
  if (!lu->pla_solved){
    
    PLA_Temp_comm_row_info(lu->templ,&Plapack_comm_2d,&r_rank,&r_nproc);
    PLA_Temp_comm_col_info(lu->templ,&Plapack_comm_2d,&c_rank,&c_nproc);

    /* Create IS and cts for VecScatterring */
    PLA_Obj_local_length(v_pla, &loc_m);
    PLA_Obj_local_stride(v_pla, &loc_stride);
    ierr = PetscMalloc2(loc_m,PetscInt,&idx_pla,loc_m,PetscInt,&idx_petsc);CHKERRQ(ierr);

    rstart = (r_rank*c_nproc+c_rank)*lu->nb;
    for (i=0; i<loc_m; i+=lu->nb){
      j = 0; 
      while (j < lu->nb && i+j < loc_m){
        idx_petsc[i+j] = rstart + j; j++;
      }
      rstart += size*lu->nb;
    }

    for (i=0; i<loc_m; i++) idx_pla[i] = i*loc_stride;

    ierr = ISCreateGeneral(PETSC_COMM_SELF,loc_m,idx_pla,PETSC_COPY_VALUES,&lu->is_pla);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,loc_m,idx_petsc,PETSC_COPY_VALUES,&lu->is_petsc);CHKERRQ(ierr);  
    ierr = PetscFree2(idx_pla,idx_petsc);CHKERRQ(ierr);
    ierr = VecScatterCreate(loc_x,lu->is_pla,x,lu->is_petsc,&lu->ctx);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(lu->ctx,loc_x,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(lu->ctx,loc_x,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  
  /* Free data */
  ierr = VecDestroy(&loc_x);CHKERRQ(ierr);
  PLA_Obj_free(&v_pla); 

  lu->pla_solved = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__   
#define __FUNCT__ "MatLUFactorNumeric_MPIDense"
PetscErrorCode MatLUFactorNumeric_MPIDense(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_Plapack    *lu = (Mat_Plapack*)(F)->spptr;
  PetscErrorCode ierr;
  PetscInt       M=A->rmap->N,m=A->rmap->n,rstart,rend;
  PetscInt       info_pla=0;
  PetscScalar    *array,one = 1.0;

  PetscFunctionBegin;
  if (lu->mstruct == SAME_NONZERO_PATTERN){
    PLA_Obj_free(&lu->A);
    PLA_Obj_free (&lu->pivots);
  }
  /* Create PLAPACK matrix object */
  lu->A = NULL; lu->pivots = NULL;
  PLA_Matrix_create(lu->datatype,M,M,lu->templ,PLA_ALIGN_FIRST,PLA_ALIGN_FIRST,&lu->A);  
  PLA_Obj_set_to_zero(lu->A);
  PLA_Mvector_create(MPI_INT,M,1,lu->templ,PLA_ALIGN_FIRST,&lu->pivots);

  /* Copy A into lu->A */
  PLA_API_begin();
  PLA_Obj_API_open(lu->A);  
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = MatGetArray(A,&array);CHKERRQ(ierr);
  PLA_API_axpy_matrix_to_global(m,M, &one,(void *)array,m,lu->A,rstart,0); 
  ierr = MatRestoreArray(A,&array);CHKERRQ(ierr);
  PLA_Obj_API_close(lu->A); 
  PLA_API_end(); 

  /* Factor P A -> L U overwriting lower triangular portion of A with L, upper, U */
  info_pla = PLA_LU(lu->A,lu->pivots);
  if (info_pla != 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot encountered at row %d from PLA_LU()",info_pla);

  lu->rstart         = rstart;
  lu->mstruct        = SAME_NONZERO_PATTERN;
  F->ops->solve      = MatSolve_MPIDense;
  F->assembled       = PETSC_TRUE;  /* required by -ksp_view */
  PetscFunctionReturn(0);
}

#undef __FUNCT__   
#define __FUNCT__ "MatCholeskyFactorNumeric_MPIDense"
PetscErrorCode MatCholeskyFactorNumeric_MPIDense(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_Plapack    *lu = (Mat_Plapack*)F->spptr;
  PetscErrorCode ierr;
  PetscInt       M=A->rmap->N,m=A->rmap->n,rstart,rend;
  PetscInt       info_pla=0;
  PetscScalar    *array,one = 1.0;

  PetscFunctionBegin;
  if (lu->mstruct == SAME_NONZERO_PATTERN){
    PLA_Obj_free(&lu->A);
  }
  /* Create PLAPACK matrix object */
  lu->A      = NULL; 
  lu->pivots = NULL; 
  PLA_Matrix_create(lu->datatype,M,M,lu->templ,PLA_ALIGN_FIRST,PLA_ALIGN_FIRST,&lu->A);  

  /* Copy A into lu->A */
  PLA_API_begin();
  PLA_Obj_API_open(lu->A);  
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = MatGetArray(A,&array);CHKERRQ(ierr);
  PLA_API_axpy_matrix_to_global(m,M, &one,(void *)array,m,lu->A,rstart,0); 
  ierr = MatRestoreArray(A,&array);CHKERRQ(ierr);
  PLA_Obj_API_close(lu->A); 
  PLA_API_end(); 

  /* Factor P A -> Chol */
  info_pla = PLA_Chol(PLA_LOWER_TRIANGULAR,lu->A);
  if (info_pla != 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_MAT_CH_ZRPVT,"Nonpositive definite matrix detected at row %d from PLA_Chol()",info_pla);

  lu->rstart         = rstart;
  lu->mstruct        = SAME_NONZERO_PATTERN;
  F->ops->solve      = MatSolve_MPIDense;  
  F->assembled       = PETSC_TRUE;  /* required by -ksp_view */
  PetscFunctionReturn(0);
}

/* Note the Petsc perm permutation is ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_MPIDense"
PetscErrorCode MatCholeskyFactorSymbolic_MPIDense(Mat F,Mat A,IS perm,const MatFactorInfo *info)
{ 
  PetscErrorCode ierr;
  PetscBool      issymmetric,set;

  PetscFunctionBegin;
  ierr = MatIsSymmetricKnown(A,&set,&issymmetric);CHKERRQ(ierr);
  if (!set || !issymmetric) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_USER,"Matrix must be set as MAT_SYMMETRIC for CholeskyFactor()");
  F->ops->choleskyfactornumeric  = MatCholeskyFactorNumeric_MPIDense;
  F->preallocated                = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_MPIDense"
PetscErrorCode MatLUFactorSymbolic_MPIDense(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{  
  PetscErrorCode ierr;
  PetscInt       M = A->rmap->N;
  Mat_Plapack    *lu;

  PetscFunctionBegin;
  lu = (Mat_Plapack*)F->spptr;
  ierr = PLA_Mvector_create(MPI_INT,M,1,lu->templ,PLA_ALIGN_FIRST,&lu->pivots);CHKERRQ(ierr);
  F->ops->lufactornumeric  = MatLUFactorNumeric_MPIDense;
  F->preallocated          = PETSC_TRUE;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatFactorGetSolverPackage_mpidense_plapack"
PetscErrorCode MatFactorGetSolverPackage_mpidense_plapack(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERPLAPACK;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_mpidense_plapack"
PetscErrorCode MatGetFactor_mpidense_plapack(Mat A,MatFactorType ftype,Mat *F)
{
  PetscErrorCode ierr;
  Mat_Plapack    *lu;
  PetscMPIInt    size;
  PetscInt       M=A->rmap->N;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  ierr = MatCreate(((PetscObject)A)->comm,F);CHKERRQ(ierr);
  ierr = MatSetSizes(*F,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(*F,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(*F);CHKERRQ(ierr);
  ierr = PetscNewLog(*F,Mat_Plapack,&lu);CHKERRQ(ierr);
  (*F)->spptr = (void*)lu;

  /* Set default Plapack parameters */
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  lu->nb = M/size;
  if (M - lu->nb*size) lu->nb++; /* without cyclic distribution */
 
  /* Set runtime options */
  ierr = PetscOptionsBegin(((PetscObject)A)->comm,((PetscObject)A)->prefix,"PLAPACK Options","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_plapack_nb","block size of template vector","None",lu->nb,&lu->nb,PETSC_NULL);CHKERRQ(ierr); 
  PetscOptionsEnd(); 

  /* Create object distribution template */
  lu->templ = NULL;
  ierr = PLA_Temp_create(lu->nb, 0, &lu->templ);CHKERRQ(ierr);

  /* Set the datatype */
#if defined(PETSC_USE_COMPLEX)
  lu->datatype = MPI_DOUBLE_COMPLEX;
#else
  lu->datatype = MPI_DOUBLE;
#endif

  ierr = PLA_Matrix_create(lu->datatype,M,A->cmap->N,lu->templ,PLA_ALIGN_FIRST,PLA_ALIGN_FIRST,&lu->A);CHKERRQ(ierr);


  lu->pla_solved     = PETSC_FALSE; /* MatSolve_Plapack() is called yet */
  lu->mstruct        = DIFFERENT_NONZERO_PATTERN;

  if (ftype == MAT_FACTOR_LU) {
    (*F)->ops->lufactorsymbolic = MatLUFactorSymbolic_MPIDense;
  } else if (ftype == MAT_FACTOR_CHOLESKY) {
    (*F)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_MPIDense;
  } else SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"No incomplete factorizations for dense matrices");
  (*F)->factortype = ftype;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)(*F),"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_mpidense_plapack",MatFactorGetSolverPackage_mpidense_plapack);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif

#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_mpidense_petsc"
PetscErrorCode MatGetFactor_mpidense_petsc(Mat A,MatFactorType ftype,Mat *F)
{
#if defined(PETSC_HAVE_PLAPACK)
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_PLAPACK)
  ierr = MatGetFactor_mpidense_plapack(A,ftype,F);CHKERRQ(ierr);
#else
  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrix format %s uses PLAPACK direct solver. Install PLAPACK",((PetscObject)A)->type_name);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAXPY_MPIDense"
PetscErrorCode MatAXPY_MPIDense(Mat Y,PetscScalar alpha,Mat X,MatStructure str)
{
  PetscErrorCode ierr;
  Mat_MPIDense   *A = (Mat_MPIDense*)Y->data, *B = (Mat_MPIDense*)X->data;

  PetscFunctionBegin;
  ierr = MatAXPY(A->A,alpha,B->A,str);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatConjugate_MPIDense"
PetscErrorCode  MatConjugate_MPIDense(Mat mat)
{
  Mat_MPIDense   *a = (Mat_MPIDense *)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatConjugate(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRealPart_MPIDense"
PetscErrorCode MatRealPart_MPIDense(Mat A)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatRealPart(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatImaginaryPart_MPIDense"
PetscErrorCode MatImaginaryPart_MPIDense(Mat A)
{
  Mat_MPIDense   *a = (Mat_MPIDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatImaginaryPart(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatGetColumnNorms_SeqDense(Mat,NormType,PetscReal*);
#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnNorms_MPIDense"
PetscErrorCode MatGetColumnNorms_MPIDense(Mat A,NormType type,PetscReal *norms)
{
  PetscErrorCode ierr;
  PetscInt       i,n;
  Mat_MPIDense   *a = (Mat_MPIDense*) A->data;
  PetscReal      *work;

  PetscFunctionBegin;
  ierr = MatGetSize(A,PETSC_NULL,&n);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscReal),&work);CHKERRQ(ierr);
  ierr = MatGetColumnNorms_SeqDense(a->A,type,work);CHKERRQ(ierr);
  if (type == NORM_2) {
    for (i=0; i<n; i++) work[i] *= work[i];
  }
  if (type == NORM_INFINITY) {
    ierr = MPI_Allreduce(work,norms,n,MPIU_REAL,MPIU_MAX,A->hdr.comm);CHKERRQ(ierr);
  } else {
    ierr = MPI_Allreduce(work,norms,n,MPIU_REAL,MPIU_SUM,A->hdr.comm);CHKERRQ(ierr);
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  if (type == NORM_2) {
    for (i=0; i<n; i++) norms[i] = PetscSqrtReal(norms[i]);
  }
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
       MatSetOption_MPIDense,
       MatZeroEntries_MPIDense,
/*24*/ MatZeroRows_MPIDense,
       0,
       0,
       0,
       0,
/*29*/ MatSetUp_MPIDense,
       0,
       0,
       MatGetArray_MPIDense,
       MatRestoreArray_MPIDense,
/*34*/ MatDuplicate_MPIDense,
       0,
       0,
       0,
       0,
/*39*/ MatAXPY_MPIDense,
       MatGetSubMatrices_MPIDense,
       0,
       MatGetValues_MPIDense,
       0,
/*44*/ 0,
       MatScale_MPIDense,
       0,
       0,
       0,
/*49*/ 0,
       0,
       0,
       0,
       0,
/*54*/ 0,
       0,
       0,
       0,
       0,
/*59*/ MatGetSubMatrix_MPIDense,
       MatDestroy_MPIDense,
       MatView_MPIDense,
       0,
       0,
/*64*/ 0,
       0,
       0,
       0,
       0,
/*69*/ 0,
       0,
       0,
       0,
       0,
/*74*/ 0,
       0,
       0,
       0,
       0,
/*79*/ 0,
       0,
       0,
       0,
/*83*/ MatLoad_MPIDense,
       0,
       0,
       0,
       0,
       0,
/*89*/ 
#if defined(PETSC_HAVE_PLAPACK)
       MatMatMult_MPIDense_MPIDense,
       MatMatMultSymbolic_MPIDense_MPIDense,
       MatMatMultNumeric_MPIDense_MPIDense,
#else
       0,
       0,
       0,
#endif
       0,
       0,
/*94*/ 0,
       0,
       0,
       0,
       0,
/*99*/ 0,
       0,
       0,
       MatConjugate_MPIDense,
       0,
/*104*/0,
       MatRealPart_MPIDense,
       MatImaginaryPart_MPIDense,
       0,
       0,
/*109*/0,
       0,
       0,
       0,
       0,
/*114*/0,
       0,
       0,
       0,
       0,
/*119*/0,
       0,
       0,
       0,
       0,
/*124*/0,
       MatGetColumnNorms_MPIDense
};

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatMPIDenseSetPreallocation_MPIDense"
PetscErrorCode  MatMPIDenseSetPreallocation_MPIDense(Mat mat,PetscScalar *data)
{
  Mat_MPIDense   *a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mat->preallocated = PETSC_TRUE;
  /* Note:  For now, when data is specified above, this assumes the user correctly
   allocates the local dense storage space.  We should add error checking. */

  a    = (Mat_MPIDense*)mat->data;
  ierr = PetscLayoutSetUp(mat->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(mat->cmap);CHKERRQ(ierr);
  a->nvec = mat->cmap->n;

  ierr = MatCreate(PETSC_COMM_SELF,&a->A);CHKERRQ(ierr);
  ierr = MatSetSizes(a->A,mat->rmap->n,mat->cmap->N,mat->rmap->n,mat->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(a->A,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(a->A,data);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
   MATSOLVERPLAPACK = "mpidense" - Parallel LU and Cholesky factorization for MATMPIDENSE matrices

  run ./configure with the option --download-plapack


  Options Database Keys:
. -mat_plapack_nprows <n> - number of rows in processor partition
. -mat_plapack_npcols <n> - number of columns in processor partition
. -mat_plapack_nb <n> - block size of template vector
. -mat_plapack_nb_alg <n> - algorithmic block size
- -mat_plapack_ckerror <n> - error checking flag

   Level: intermediate

.seealso: MatCreateDense(), MATDENSE, MATSEQDENSE, PCFactorSetSolverPackage(), MatSolverPackage

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_MPIDense"
PetscErrorCode  MatCreate_MPIDense(Mat mat)
{
  Mat_MPIDense   *a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr              = PetscNewLog(mat,Mat_MPIDense,&a);CHKERRQ(ierr);
  mat->data         = (void*)a;
  ierr              = PetscMemcpy(mat->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);

  mat->insertmode = NOT_SET_VALUES;
  ierr = MPI_Comm_rank(((PetscObject)mat)->comm,&a->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)mat)->comm,&a->size);CHKERRQ(ierr);

  /* build cache for off array entries formed */
  a->donotstash = PETSC_FALSE;
  ierr = MatStashCreate_Private(((PetscObject)mat)->comm,1,&mat->stash);CHKERRQ(ierr);

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
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)mat,"MatGetFactor_petsc_C",
                                     "MatGetFactor_mpidense_petsc",
                                      MatGetFactor_mpidense_petsc);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PLAPACK)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)mat,"MatGetFactor_plapack_C",
                                     "MatGetFactor_mpidense_plapack",
                                      MatGetFactor_mpidense_plapack);CHKERRQ(ierr);
  ierr = PetscPLAPACKInitializePackage(((PetscObject)mat)->comm);CHKERRQ(ierr);
#endif
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
PetscErrorCode  MatMPIDenseSetPreallocation(Mat mat,PetscScalar *data)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(mat,"MatMPIDenseSetPreallocation_C",(Mat,PetscScalar *),(mat,data));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateDense"
/*@C
   MatCreateDense - Creates a parallel matrix in dense format.

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
PetscErrorCode  MatCreateDense(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscScalar *data,Mat *A)
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
  ierr = MatCreate(((PetscObject)A)->comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(mat,((PetscObject)A)->type_name);CHKERRQ(ierr);
  a                 = (Mat_MPIDense*)mat->data;
  ierr              = PetscMemcpy(mat->ops,A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);

  mat->factortype   = A->factortype;
  mat->assembled    = PETSC_TRUE;
  mat->preallocated = PETSC_TRUE;

  a->size           = oldmat->size;
  a->rank           = oldmat->rank;
  mat->insertmode   = NOT_SET_VALUES;
  a->nvec           = oldmat->nvec;
  a->donotstash     = oldmat->donotstash;

  ierr = PetscLayoutReference(A->rmap,&mat->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutReference(A->cmap,&mat->cmap);CHKERRQ(ierr);

  ierr = MatSetUpMultiply_MPIDense(mat);CHKERRQ(ierr);
  ierr = MatDuplicate(oldmat->A,cpvalues,&a->A);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->A);CHKERRQ(ierr);

  *newmat = mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLoad_MPIDense_DenseInFile"
PetscErrorCode MatLoad_MPIDense_DenseInFile(MPI_Comm comm,PetscInt fd,PetscInt M,PetscInt N,Mat newmat,PetscInt sizesset)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       *rowners,i,m,nz,j;
  PetscScalar    *array,*vals,*vals_ptr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* determine ownership of all rows */
  if (newmat->rmap->n < 0) m          = M/size + ((M % size) > rank);
  else m = newmat->rmap->n;
  ierr       = PetscMalloc((size+2)*sizeof(PetscInt),&rowners);CHKERRQ(ierr);
  ierr       = MPI_Allgather(&m,1,MPIU_INT,rowners+1,1,MPIU_INT,comm);CHKERRQ(ierr);
  rowners[0] = 0;
  for (i=2; i<=size; i++) {
    rowners[i] += rowners[i-1];
  }

  if (!sizesset) {
    ierr = MatSetSizes(newmat,m,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  }
  ierr = MatMPIDenseSetPreallocation(newmat,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetArray(newmat,&array);CHKERRQ(ierr);

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
      ierr = MPIULong_Send(vals,nz,MPIU_SCALAR,i,((PetscObject)(newmat))->tag,comm);CHKERRQ(ierr);
    }
  } else {
    /* receive numeric values */
    ierr = PetscMalloc(m*N*sizeof(PetscScalar),&vals);CHKERRQ(ierr);

    /* receive message of values*/
    ierr = MPIULong_Recv(vals,m*N,MPIU_SCALAR,0,((PetscObject)(newmat))->tag,comm);CHKERRQ(ierr);

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
  ierr = MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLoad_MPIDense"
PetscErrorCode MatLoad_MPIDense(Mat newmat,PetscViewer viewer)
{
  PetscScalar    *vals,*svals;
  MPI_Comm       comm = ((PetscObject)viewer)->comm;
  MPI_Status     status;
  PetscMPIInt    rank,size,tag = ((PetscObject)viewer)->tag,*rowners,*sndcounts,m,maxnz;
  PetscInt       header[4],*rowlengths = 0,M,N,*cols;
  PetscInt       *ourlens,*procsnz = 0,*offlens,jj,*mycols,*smycols;
  PetscInt       i,nz,j,rstart,rend,sizesset=1,grows,gcols;
  int            fd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,PETSC_INT);CHKERRQ(ierr);
    if (header[0] != MAT_FILE_CLASSID) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"not matrix object");
  }
  if (newmat->rmap->n < 0 && newmat->rmap->N < 0 && newmat->cmap->n < 0 && newmat->cmap->N < 0) sizesset = 0;

  ierr = MPI_Bcast(header+1,3,MPIU_INT,0,comm);CHKERRQ(ierr);
  M = header[1]; N = header[2]; nz = header[3];

  /* If global rows/cols are set to PETSC_DECIDE, set it to the sizes given in the file */
  if (sizesset && newmat->rmap->N < 0) newmat->rmap->N = M;
  if (sizesset && newmat->cmap->N < 0) newmat->cmap->N = N;
  
  /* If global sizes are set, check if they are consistent with that given in the file */
  if (sizesset) {
    ierr = MatGetSize(newmat,&grows,&gcols);CHKERRQ(ierr);
  } 
  if (sizesset && newmat->rmap->N != grows) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Inconsistent # of rows:Matrix in file has (%d) and input matrix has (%d)",M,grows);
  if (sizesset && newmat->cmap->N != gcols) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Inconsistent # of cols:Matrix in file has (%d) and input matrix has (%d)",N,gcols);

  /*
       Handle case where matrix is stored on disk as a dense matrix 
  */
  if (nz == MATRIX_BINARY_FORMAT_DENSE) {
    ierr = MatLoad_MPIDense_DenseInFile(comm,fd,M,N,newmat,sizesset);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* determine ownership of all rows */
  if (newmat->rmap->n < 0) m          = PetscMPIIntCast(M/size + ((M % size) > rank));
  else m = PetscMPIIntCast(newmat->rmap->n);
  ierr       = PetscMalloc((size+2)*sizeof(PetscMPIInt),&rowners);CHKERRQ(ierr);
  ierr       = MPI_Allgather(&m,1,MPI_INT,rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  rowners[0] = 0;
  for (i=2; i<=size; i++) {
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 

  /* distribute row lengths to all processors */
  ierr    = PetscMalloc2(rend-rstart,PetscInt,&ourlens,rend-rstart,PetscInt,&offlens);CHKERRQ(ierr);
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
    if (maxnz != nz) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"something is wrong with file");
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

  if (!sizesset) {
    ierr = MatSetSizes(newmat,m,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  }
  ierr = MatMPIDenseSetPreallocation(newmat,PETSC_NULL);CHKERRQ(ierr);
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
      ierr = MatSetValues(newmat,1,&jj,ourlens[i],smycols,svals,INSERT_VALUES);CHKERRQ(ierr);
      smycols += ourlens[i];
      svals   += ourlens[i];
      jj++;
    }

    /* read in other processors and ship out */
    for (i=1; i<size; i++) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
      ierr = MPI_Send(vals,nz,MPIU_SCALAR,i,((PetscObject)newmat)->tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(procsnz);CHKERRQ(ierr);
  } else {
    /* receive numeric values */
    ierr = PetscMalloc((nz+1)*sizeof(PetscScalar),&vals);CHKERRQ(ierr);

    /* receive message of values*/
    ierr = MPI_Recv(vals,nz,MPIU_SCALAR,0,((PetscObject)newmat)->tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPIU_SCALAR,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"something is wrong with file");

    /* insert into matrix */
    jj      = rstart;
    smycols = mycols;
    svals   = vals;
    for (i=0; i<m; i++) {
      ierr = MatSetValues(newmat,1,&jj,ourlens[i],smycols,svals,INSERT_VALUES);CHKERRQ(ierr);
      smycols += ourlens[i];
      svals   += ourlens[i];
      jj++;
    }
  }
  ierr = PetscFree2(ourlens,offlens);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);
  ierr = PetscFree(mycols);CHKERRQ(ierr);
  ierr = PetscFree(rowners);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatEqual_MPIDense"
PetscErrorCode MatEqual_MPIDense(Mat A,Mat B,PetscBool  *flag)
{
  Mat_MPIDense   *matB = (Mat_MPIDense*)B->data,*matA = (Mat_MPIDense*)A->data;
  Mat            a,b;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  a = matA->A; 
  b = matB->A;
  ierr = MatEqual(a,b,&flg);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&flg,flag,1,MPI_INT,MPI_LAND,((PetscObject)A)->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_PLAPACK)

#undef __FUNCT__  
#define __FUNCT__ "PetscPLAPACKFinalizePackage" 
/*@C
  PetscPLAPACKFinalizePackage - This function destroys everything in the Petsc interface to PLAPACK.
  Level: developer

.keywords: Petsc, destroy, package, PLAPACK
.seealso: PetscFinalize()
@*/
PetscErrorCode  PetscPLAPACKFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PLA_Finalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscPLAPACKInitializePackage" 
/*@C
  PetscPLAPACKInitializePackage - This function initializes everything in the Petsc interface to PLAPACK. It is
  called from MatCreate_MPIDense() the first time an MPI dense matrix is called.

  Input Parameter:
.   comm - the communicator the matrix lives on

  Level: developer

   Notes: PLAPACK does not have a good fit with MPI communicators; all (parallel) PLAPACK objects have to live in the
  same communicator (because there is some global state that is initialized and used for all matrices). In addition if 
  PLAPACK is initialized (that is the initial matrices created) are on subcommunicators of MPI_COMM_WORLD, these subcommunicators
  cannot overlap.

.keywords: Petsc, initialize, package, PLAPACK
.seealso: PetscSysInitializePackage(), PetscInitialize()
@*/
PetscErrorCode  PetscPLAPACKInitializePackage(MPI_Comm comm)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!PLA_Initialized(PETSC_NULL)) {

    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    Plapack_nprows = 1;
    Plapack_npcols = size; 
    
    ierr = PetscOptionsBegin(comm,PETSC_NULL,"PLAPACK Options","Mat");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-mat_plapack_nprows","row dimension of 2D processor mesh","None",Plapack_nprows,&Plapack_nprows,PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-mat_plapack_npcols","column dimension of 2D processor mesh","None",Plapack_npcols,&Plapack_npcols,PETSC_NULL);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
      Plapack_ierror = 3;
#else
      Plapack_ierror = 0;
#endif
      ierr = PetscOptionsInt("-mat_plapack_ckerror","error checking flag","None",Plapack_ierror,&Plapack_ierror,PETSC_NULL);CHKERRQ(ierr);  
      if (Plapack_ierror){
	ierr = PLA_Set_error_checking(Plapack_ierror,PETSC_TRUE,PETSC_TRUE,PETSC_FALSE );CHKERRQ(ierr);
      } else {
	ierr = PLA_Set_error_checking(Plapack_ierror,PETSC_FALSE,PETSC_FALSE,PETSC_FALSE );CHKERRQ(ierr);
      }
  
      Plapack_nb_alg = 0;
      ierr = PetscOptionsInt("-mat_plapack_nb_alg","algorithmic block size","None",Plapack_nb_alg,&Plapack_nb_alg,PETSC_NULL);CHKERRQ(ierr);
      if (Plapack_nb_alg) {
        ierr = pla_Environ_set_nb_alg (PLA_OP_ALL_ALG,Plapack_nb_alg);CHKERRQ(ierr);
      }
    PetscOptionsEnd(); 

    ierr = PLA_Comm_1D_to_2D(comm,Plapack_nprows,Plapack_npcols,&Plapack_comm_2d);CHKERRQ(ierr);
    ierr = PLA_Init(Plapack_comm_2d);CHKERRQ(ierr);
    ierr = PetscRegisterFinalize(PetscPLAPACKFinalizePackage);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#endif
