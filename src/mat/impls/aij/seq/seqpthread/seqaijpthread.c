
#include <petsc-private/vecimpl.h>
#include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/
#include <../src/sys/objects/pthread/pthreadimpl.h>
#include <../src/mat/impls/aij/seq/seqpthread/seqaijpthread.h>

static PetscInt    mats_created=0;

PetscErrorCode MatRealPart_Kernel(void* arg)
{
  Mat_KernelData    *data=(Mat_KernelData*)arg;
  const PetscInt    *ai = (const PetscInt*)data->ai;
  MatScalar         *aabase = data->aa,*aa;
  PetscInt          nrows=data->nrows;
  PetscInt          nz,i;

  
  nz = ai[nrows] - ai[0];
  aa = aabase + ai[0];
  for(i=0;i<nz;i++) aa[i] = PetscRealPart(aa[i]);

  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRealPart_SeqAIJPThread"
PetscErrorCode MatRealPart_SeqAIJPThread(Mat A)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  PetscThreadsLayout tmap=A->rmap->tmap;
  PetscInt           *trstarts = tmap->trstarts;
  PetscInt           i;
  MatScalar          *aa=a->a;
  PetscInt           *ai=a->i;

  PetscFunctionBegin;

  for(i=0; i < tmap->nthreads; i++) {
    mat_kerneldatap[i].ai = ai + trstarts[i];
    mat_kerneldatap[i].aa = aa;
    mat_kerneldatap[i].nrows = trstarts[i+1] - trstarts[i];
    mat_pdata[i] = &mat_kerneldatap[i];
  }
  ierr = PetscThreadsRunKernel(MatRealPart_Kernel,(void**)mat_pdata,tmap->nthreads,tmap->affinity);CHKERRQ(ierr);

  a->idiagvalid = PETSC_FALSE;
  a->ibdiagvalid = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatImaginaryPart_Kernel(void* arg)
{
  Mat_KernelData    *data=(Mat_KernelData*)arg;
  const PetscInt    *ai = (const PetscInt*)data->ai;
  MatScalar         *aabase = data->aa,*aa;
  PetscInt          nrows=data->nrows;
  PetscInt          nz,i;

  
  nz = ai[nrows] - ai[0];
  aa = aabase + ai[0];
  for(i=0;i<nz;i++) aa[i] = PetscImaginaryPart(aa[i]);

  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatImaginaryPart_SeqAIJPThread"
PetscErrorCode MatImaginaryPart_SeqAIJPThread(Mat A)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  PetscThreadsLayout tmap=A->rmap->tmap;
  PetscInt           *trstarts = tmap->trstarts;
  PetscInt           i;
  MatScalar          *aa=a->a;
  PetscInt           *ai=a->i;

  PetscFunctionBegin;

  for(i=0; i < tmap->nthreads; i++) {
    mat_kerneldatap[i].ai = ai + trstarts[i];
    mat_kerneldatap[i].aa = aa;
    mat_kerneldatap[i].nrows = trstarts[i+1] - trstarts[i];
    mat_pdata[i] = &mat_kerneldatap[i];
  }
  ierr = PetscThreadsRunKernel(MatImaginaryPart_Kernel,(void**)mat_pdata,tmap->nthreads,tmap->affinity);CHKERRQ(ierr);

  a->idiagvalid = PETSC_FALSE;
  a->ibdiagvalid = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorGetDiagonal_Kernel(void* arg)
{
  Mat_KernelData    *data=(Mat_KernelData*)arg;
  const PetscInt    *adiag = (const PetscInt*)data->adiag;
  const MatScalar   *aa = (const MatScalar*)data->aa;
  PetscScalar       *x = (PetscScalar*)data->x; 
  PetscInt           nrows=(PetscInt)data->nrows,i;

  
  for(i=0;i < nrows;i++) {
    x[i] = 1.0/aa[adiag[i]];
  }
  return(0);
}

PetscErrorCode MatGetDiagonal_Kernel(void* arg)
{
  Mat_KernelData    *data=(Mat_KernelData*)arg;
  const PetscInt    *ai = (const PetscInt*)data->ai,*aj = (const PetscInt*)data->aj;
  const MatScalar   *aa = (const MatScalar*)data->aa;
  PetscScalar       *x = (PetscScalar*)data->x; 
  PetscInt           nrows=(PetscInt)data->nrows;
  PetscInt           i,j,row;
  PetscInt           rstart=(PetscInt)data->rstart;

  
  for(i=0;i < nrows;i++) {
    row = rstart+i;
    for(j=ai[i]; j < ai[i+1];j++) {
      if(aj[j] == row) { 
	x[i] = aa[j];
	break;
      }
    }
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_SeqAIJPThread"
PetscErrorCode MatGetDiagonal_SeqAIJPThread(Mat A,Vec v)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  PetscThreadsLayout tmap=A->rmap->tmap;
  PetscInt           *trstarts=tmap->trstarts;
  PetscScalar        *x;
  MatScalar          *aa=a->a;
  PetscInt           *ai=a->i,*aj=a->j;
  PetscInt           i,n;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");

  if(A->factortype == MAT_FACTOR_ILU || A->factortype == MAT_FACTOR_LU) {
    PetscInt *diag=a->diag;
    ierr = VecGetArray(v,&x);CHKERRQ(ierr);
    for(i=0;i < tmap->nthreads; i++) {
      mat_kerneldatap[i].nrows = trstarts[i+1] - trstarts[i];
      mat_kerneldatap[i].aa    = aa;
      mat_kerneldatap[i].adiag = diag + trstarts[i];
      mat_kerneldatap[i].x     = x + trstarts[i];
      mat_pdata[i]             = &mat_kerneldatap[i];
    }
    ierr = PetscThreadsRunKernel(MatFactorGetDiagonal_Kernel,(void**)mat_pdata,tmap->nthreads,tmap->affinity);CHKERRQ(ierr);
    ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = VecSet(v,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  for(i=0;i < tmap->nthreads; i++) {
    mat_kerneldatap[i].ai     = ai + trstarts[i];
    mat_kerneldatap[i].rstart = trstarts[i];
    mat_kerneldatap[i].aj     = aj;
    mat_kerneldatap[i].aa     = aa;
    mat_kerneldatap[i].nrows  = trstarts[i+1]-trstarts[i];
    mat_kerneldatap[i].x      = x + trstarts[i];
    mat_pdata[i]              = &mat_kerneldatap[i];
  }
  ierr = PetscThreadsRunKernel(MatGetDiagonal_Kernel,(void**)mat_pdata,tmap->nthreads,tmap->affinity);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_Kernel(void* arg)
{
  Mat_KernelData    *data=(Mat_KernelData*)arg;
  const PetscInt    *ai = (const PetscInt*)data->ai;
  MatScalar         *aabase = data->aa;
  MatScalar         *aa;
  PetscInt          nrows=data->nrows;
  PetscInt          nz;

  
  nz = ai[nrows] - ai[0];
  aa = aabase + ai[0];
  PetscMemzero(aa,nz*sizeof(PetscScalar));

  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatZeroEntries_SeqAIJPThread"
PetscErrorCode MatZeroEntries_SeqAIJPThread(Mat A)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  PetscThreadsLayout tmap=A->rmap->tmap;
  PetscInt           *trstarts = tmap->trstarts;
  PetscInt           i;
  MatScalar          *aa=a->a;
  PetscInt           *ai=a->i;

  PetscFunctionBegin;

  for(i=0; i < tmap->nthreads; i++) {
    mat_kerneldatap[i].ai = ai + trstarts[i];
    mat_kerneldatap[i].aa = aa;
    mat_kerneldatap[i].nrows = trstarts[i+1] - trstarts[i];
    mat_pdata[i] = &mat_kerneldatap[i];
  }
  
  ierr = PetscThreadsRunKernel(MatZeroEntries_Kernel,(void**)mat_pdata,tmap->nthreads,tmap->affinity);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_Kernel(void* arg)
{
  Mat_KernelData    *data=(Mat_KernelData*)arg;
  const PetscInt    *ai = (const PetscInt*)data->ai,*ajbase = (const PetscInt*)data->aj,*aj;
  const MatScalar   *aabase = (const MatScalar*)data->aa,*aa;
  const PetscScalar *x = (const PetscScalar*)data->x; 
  PetscScalar       *y = data->y;
  PetscInt           nrows = data->nrows;
  PetscInt           nz,i;
  PetscScalar        sum;

  
  data->nonzerorow = 0;
  for(i=0;i<nrows;i++) {
    nz = ai[i+1] - ai[i];
    aj = ajbase + ai[i];
    aa = aabase + ai[i];
    sum = 0.0;
    data->nonzerorow += (nz>0);
    PetscSparseDensePlusDot(sum,x,aa,aj,nz);
    y[i] = sum;
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_SeqAIJPThread"
PetscErrorCode MatMult_SeqAIJPThread(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscThreadsLayout tmap=A->rmap->tmap;
  PetscInt          *trstarts=tmap->trstarts;
  PetscScalar       *x,*y;
  PetscErrorCode    ierr;
  MatScalar         *aa = a->a;
  PetscInt          *ai = a->i, *aj = a->j;
  PetscInt          i,nonzerorow=0;;

  PetscFunctionBegin;

  if(a->compressedrow.use) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Compressed row format not supported yet");
  }
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  for(i=0;i < tmap->nthreads;i++) {
    mat_kerneldatap[i].ai    = ai + trstarts[i];
    mat_kerneldatap[i].aa    = aa;
    mat_kerneldatap[i].aj    = aj;
    mat_kerneldatap[i].nrows = trstarts[i+1] - trstarts[i];
    mat_kerneldatap[i].x     = x;
    mat_kerneldatap[i].y     = y + trstarts[i];
    mat_pdata[i]             = &mat_kerneldatap[i];
  }
  ierr = PetscThreadsRunKernel(MatMult_Kernel,(void**)mat_pdata,tmap->nthreads,tmap->affinity);

  for(i=0;i< tmap->nthreads;i++) nonzerorow += mat_kerneldatap[i].nonzerorow;

  ierr = PetscLogFlops(2.0*a->nz - nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_Kernel(void* arg)
{
  Mat_KernelData    *data=(Mat_KernelData*)arg;
  const PetscInt    *ai = (const PetscInt*)data->ai,*aj = (const PetscInt*)data->aj;
  const MatScalar   *aa = (const MatScalar*)data->aa;
  const PetscScalar *x = (const PetscScalar*)data->x; 
  PetscScalar       *y = data->y;
  PetscScalar       *z = data->z;
  PetscInt           nrows = data->nrows;
  PetscInt           nz,i;
  PetscScalar        sum;

  
  data->nonzerorow = 0;
  for(i=0;i<nrows;i++) {
    nz = ai[i+1] - ai[i];
    aj = data->aj + ai[i];
    aa = data->aa + ai[i];
    sum = y[i];
    data->nonzerorow += (nz>0);
    PetscSparseDensePlusDot(sum,x,aa,aj,nz);
    z[i] = sum;
  }
  return(0);
}
  
#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_SeqAIJPThread"
PetscErrorCode MatMultAdd_SeqAIJPThread(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscThreadsLayout tmap=A->rmap->tmap;
  PetscInt          *trstarts=tmap->trstarts;
  PetscScalar       *x,*y,*z;
  PetscErrorCode    ierr;
  MatScalar         *aa = a->a;
  PetscInt          *ai = a->i, *aj = a->j;
  PetscInt          i,nonzerorow=0;;

  PetscFunctionBegin;

  if(a->compressedrow.use) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Compressed row format not supported yet");
  }
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if(zz != yy) {
    ierr = VecGetArray(zz,&z);CHKERRQ(ierr);
  } else z = y;

  for(i=0;i < tmap->nthreads;i++) {
    mat_kerneldatap[i].ai    = ai + trstarts[i];
    mat_kerneldatap[i].aa    = aa;
    mat_kerneldatap[i].aj    = aj;
    mat_kerneldatap[i].nrows = trstarts[i+1]-trstarts[i];
    mat_kerneldatap[i].x     = x;
    mat_kerneldatap[i].y     = y + trstarts[i];
    mat_kerneldatap[i].z     = z + trstarts[i];
    mat_pdata[i]             = &mat_kerneldatap[i];
  }
  ierr = PetscThreadsRunKernel(MatMultAdd_Kernel,(void**)mat_pdata,tmap->nthreads,tmap->affinity);

  for(i=0;i< tmap->nthreads;i++) nonzerorow += mat_kerneldatap[i].nonzerorow;

  ierr = PetscLogFlops(2.0*a->nz - nonzerorow);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  if(zz != yy) {
    ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MatMarkDiagonal_Kernel(void* arg)
{
  Mat_KernelData    *data=(Mat_KernelData*)arg;
  const PetscInt    *ai = (const PetscInt*)data->ai,*aj = (const PetscInt*)data->aj;
  PetscInt          *adiag=(PetscInt*)data->adiag;
  PetscInt           nrows=(PetscInt)data->nrows;
  PetscInt           i,j,row;
  PetscInt           rstart=(PetscInt)data->rstart;

  
  for(i=0;i < nrows;i++) {
    row = rstart+i;
    for(j=ai[i]; j < ai[i+1];j++) {
      if(aj[j] == row) { 
	adiag[i] = j;
	break;
      }
    }
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMarkDiagonal_SeqAIJPThread"
PetscErrorCode MatMarkDiagonal_SeqAIJPThread(Mat A)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  PetscThreadsLayout tmap=A->rmap->tmap;
  PetscInt           *trstarts=tmap->trstarts;
  PetscInt           *ai=a->i,*aj=a->j;
  PetscInt           i,m=A->rmap->n;

  PetscFunctionBegin;

  if(!a->diag) {
    ierr = PetscMalloc(m*sizeof(PetscInt),&a->diag);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(A,m*sizeof(PetscInt));CHKERRQ(ierr);
  } 

  for(i=0;i < tmap->nthreads; i++) {
    mat_kerneldatap[i].nrows  = trstarts[i+1] - trstarts[i];
    mat_kerneldatap[i].adiag  = a->diag + trstarts[i];
    mat_kerneldatap[i].ai     = ai + trstarts[i];
    mat_kerneldatap[i].aj     = aj;
    mat_kerneldatap[i].rstart = trstarts[i];
    mat_pdata[i]              = &mat_kerneldatap[i];
  }
  ierr = PetscThreadsRunKernel(MatMarkDiagonal_Kernel,(void**)mat_pdata,tmap->nthreads,tmap->affinity);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatFindZeroDiagonalCount_Kernel(void* arg)
{
  Mat_KernelData     *data=(Mat_KernelData*)arg;
  const PetscInt     *aj = (const PetscInt*)data->aj;
  const MatScalar    *aa = (const MatScalar*)data->aa;
  const PetscInt     *adiag = (const PetscInt*)data->adiag;
  PetscInt           nrows = (PetscInt)data->nrows;
  PetscInt           i,row;
  PetscInt           rstart = (PetscInt)data->rstart;

  
  for(i=0;i < nrows; i++) {
    row = rstart+i;
    if((aj[adiag[i]] != row) || (aa[adiag[i]] == 0.0)) data->nzerodiags++;
  }
  return(0);
}

PetscErrorCode MatFindZeroDiagonals_Kernel(void* arg)
{
  Mat_KernelData     *data=(Mat_KernelData*)arg;
  const PetscInt     *aj = (const PetscInt*)data->aj;
  const MatScalar    *aa = (const MatScalar*)data->aa;
  const PetscInt     *adiag = (const PetscInt*)data->adiag;
  PetscInt           nrows = (PetscInt)data->nrows;
  PetscInt           i,row;
  PetscInt           rstart = (PetscInt)data->rstart,cnt=0;;

  
  for(i=0;i < nrows; i++) {
    row = rstart+i;
    if((aj[adiag[i]] != row) || (aa[adiag[i]] == 0.0)) data->zerodiags[cnt++] = row;
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFindZeroDiagonals_SeqAIJPThread"
PetscErrorCode MatFindZeroDiagonals_SeqAIJPThread(Mat A,IS *zrows)
{
  Mat_SeqAIJ          *a = (Mat_SeqAIJ*)A->data;
  PetscThreadsLayout  tmap=A->rmap->tmap;
  PetscInt            *trstarts = tmap->trstarts;
  PetscInt            i,cnt=0,*rows,ctr=0;
  PetscInt            *aj = a->j,*diag;
  MatScalar           *aa = a->a;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MatMarkDiagonal_SeqAIJPThread(A);CHKERRQ(ierr);
  diag = a->diag;

  /* Find zero diagonals count */
  for(i=0;i < tmap->nthreads;i++) {
    mat_kerneldatap[i].aj      = aj;
    mat_kerneldatap[i].nrows   = trstarts[i+1]-trstarts[i];
    mat_kerneldatap[i].aa      = aa;
    mat_kerneldatap[i].rstart  = trstarts[i];
    mat_kerneldatap[i].nzerodiags = 0;
    mat_kerneldatap[i].adiag   = diag + trstarts[i];
    mat_pdata[i]               = &mat_kerneldatap[i];
  }
  ierr = PetscThreadsRunKernel(MatFindZeroDiagonalCount_Kernel,(void**)mat_pdata,tmap->nthreads,tmap->affinity);CHKERRQ(ierr);

  for(i=0;i < tmap->nthreads;i++) cnt += mat_kerneldatap[i].nzerodiags;
  ierr = PetscMalloc(cnt*sizeof(PetscInt),&rows);CHKERRQ(ierr);

  /* Get zero diagonals */
  for(i=0;i < tmap->nthreads;i++) {
    mat_kerneldatap[i].aj      = aj;
    mat_kerneldatap[i].nrows   = trstarts[i+1]-trstarts[i];
    mat_kerneldatap[i].aa      = aa;
    mat_kerneldatap[i].rstart  = trstarts[i];
    mat_kerneldatap[i].zerodiags = rows + ctr;
    mat_kerneldatap[i].adiag   = diag + trstarts[i];
    mat_pdata[i]               = &mat_kerneldatap[i];
    ctr += mat_kerneldatap[i].nzerodiags;
  }
  ierr = PetscThreadsRunKernel(MatFindZeroDiagonals_Kernel,(void**)mat_pdata,tmap->nthreads,tmap->affinity);CHKERRQ(ierr);

  ierr = ISCreateGeneral(((PetscObject)A)->comm,cnt,rows,PETSC_OWN_POINTER,zrows);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  
PetscErrorCode MatMissingDiagonal_Kernel(void* arg)
{
  Mat_KernelData    *data=(Mat_KernelData*)arg;
  const PetscInt    *aj = (const PetscInt*)data->aj;
  PetscInt          *adiag=(PetscInt*)data->adiag;
  PetscInt           nrows=(PetscInt)data->nrows;
  PetscInt           i,row;
  PetscInt           rstart=(PetscInt)data->rstart;

  
  data->missing_diag = PETSC_FALSE;
  for(i=0; i < nrows; i++) {
    row = rstart + i;
    if(aj[adiag[i]] != row) {
      data->missing_diag = PETSC_TRUE;
      if(data->find_d) data->d =  row;
      break;
    }
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMissingDiagonal_SeqAIJPThread"
PetscErrorCode MatMissingDiagonal_SeqAIJPThread(Mat A, PetscBool *missing,PetscInt *d)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode     ierr;
  PetscThreadsLayout tmap=A->rmap->tmap;
  PetscInt           *trstarts=tmap->trstarts;
  PetscInt           *aj=a->j,*diag,i;

  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  if(A->rmap->n > 0 && !aj) {
    *missing = PETSC_TRUE;
    if (d) *d = 0;
    PetscInfo(A,"Matrix has no entries therefore is missing diagonal");
  } else {
    diag = a->diag;
    for(i=0; i < tmap->nthreads; i++) {
      mat_kerneldatap[i].nrows = trstarts[i+1] - trstarts[i];
      mat_kerneldatap[i].adiag = diag + trstarts[i];
      mat_kerneldatap[i].aj    = aj;
      mat_kerneldatap[i].rstart = trstarts[i];
      mat_kerneldatap[i].missing_diag = PETSC_FALSE;
      mat_kerneldatap[i].find_d = PETSC_FALSE;;
      if(d) mat_kerneldatap[i].find_d = PETSC_TRUE;
      mat_pdata[i]  = &mat_kerneldatap[i];
    }
    ierr = PetscThreadsRunKernel(MatMissingDiagonal_Kernel,(void**)mat_pdata,tmap->nthreads,tmap->affinity);CHKERRQ(ierr);

    for(i=0;i < tmap->nthreads;i++) {
      if(mat_kerneldatap[i].missing_diag) {
	*missing = PETSC_TRUE;
	if(d) *d = mat_kerneldatap[i].d;
	PetscInfo1(A,"Matrix is missing diagonal number %D",mat_kerneldatap[i].d);
	break;
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDiagonalScale_Kernel(void* arg)
{
  Mat_KernelData    *data=(Mat_KernelData*)arg;
  const PetscInt    *ai = (const PetscInt*)data->ai,*aj = (const PetscInt*)data->aj;
  MatScalar         *aa = (MatScalar*)data->aa;
  const PetscScalar *ll = (const PetscScalar*)data->x; 
  const PetscScalar *rr = (const PetscScalar*)data->y;
  PetscInt           nrows = data->nrows;
  PetscInt           i,j;

  if(ll) {
    for(i=0; i < nrows; i++) {
      for(j=ai[i]; j < ai[i+1]; j++) aa[j] *= ll[i];
    }
  }

  if(rr) {
    for(i=0; i < nrows; i++) {
      for(j=ai[i]; j < ai[i+1]; j++) aa[j] *= rr[aj[j]];
    }
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDiagonalScale_SeqAIJPThread"
PetscErrorCode MatDiagonalScale_SeqAIJPThread(Mat A,Vec ll,Vec rr)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  PetscThreadsLayout tmap=A->rmap->tmap;
  PetscInt           *trstarts=tmap->trstarts;
  PetscScalar        *l,*r;
  MatScalar          *aa=a->a;
  PetscInt           *ai=a->i,*aj=a->j;
  PetscInt           i,m = A->rmap->n, n = A->cmap->n,nz = a->nz;

  PetscFunctionBegin;
  if(ll) {
    ierr = VecGetLocalSize(ll,&m);CHKERRQ(ierr);
    if (m != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Left scaling vector wrong length");
    ierr = VecGetArray(ll,&l);CHKERRQ(ierr);
  }
  if(rr) {
    ierr = VecGetLocalSize(rr,&n);CHKERRQ(ierr);
    if (n != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Right scaling vector wrong length");
    ierr = VecGetArray(rr,&r);CHKERRQ(ierr);
  }

  for(i=0;i< tmap->nthreads;i++) {
    mat_kerneldatap[i].nrows = trstarts[i+1] - trstarts[i];
    mat_kerneldatap[i].ai    = ai + trstarts[i];
    mat_kerneldatap[i].aj    = aj;
    mat_kerneldatap[i].aa    = aa;
    if(ll) mat_kerneldatap[i].x = l + trstarts[i];
    else mat_kerneldatap[i].x = PETSC_NULL;
    if(rr) mat_kerneldatap[i].y = r;
    else mat_kerneldatap[i].y = PETSC_NULL;
    mat_pdata[i] = &mat_kerneldatap[i];
  }
  ierr = PetscThreadsRunKernel(MatDiagonalScale_Kernel,(void**)mat_pdata,tmap->nthreads,tmap->affinity);CHKERRQ(ierr);

  if(ll) {
    ierr = VecRestoreArray(ll,&l);CHKERRQ(ierr);
    ierr = PetscLogFlops(nz);CHKERRQ(ierr);
  }
  if(rr) {
    ierr = VecRestoreArray(rr,&r);CHKERRQ(ierr);
    ierr = PetscLogFlops(nz);CHKERRQ(ierr);
  }
  a->idiagvalid = PETSC_FALSE;
  a->ibdiagvalid = PETSC_FALSE;
  PetscFunctionReturn(0);
}
 
PetscErrorCode MatDiagonalSet_Kernel(void* arg)
{
  Mat_KernelData    *data=(Mat_KernelData*)arg;
  MatScalar         *aa = (MatScalar*)data->aa;
  const PetscInt    *adiag = (const PetscInt*)data->adiag;
  PetscInt          nrows = data->nrows;
  PetscInt          i;
  PetscScalar       *x = (PetscScalar*)data->x;
  InsertMode        is = data->is;
  
  if(is == INSERT_VALUES) {
    for(i=0; i < nrows; i++) {
      aa[adiag[i]] = x[i];
    }
  } else {
    for(i=0;i < nrows; i++) {
      aa[adiag[i]] += x[i];
    }
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDiagonalSet_SeqAIJPThread"
PetscErrorCode MatDiagonalSet_SeqAIJPThread(Mat Y,Vec D,InsertMode is)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *aij = (Mat_SeqAIJ*)Y->data;
  PetscThreadsLayout tmap=Y->rmap->tmap;
  PetscInt           *trstarts=tmap->trstarts;
  MatScalar          *aa = aij->a;
  PetscInt           *diag;
  PetscInt           i;
  PetscBool          missing;
  PetscScalar        *v;

  PetscFunctionBegin;
  if(Y->assembled) {
    ierr = MatMissingDiagonal_SeqAIJPThread(Y,&missing,PETSC_NULL);CHKERRQ(ierr);
    if(!missing) {
      diag = aij->diag;
      ierr = VecGetArray(D,&v);CHKERRQ(ierr);
      for(i=0; i < tmap->nthreads;i++) {
	mat_kerneldatap[i].nrows = trstarts[i+1] - trstarts[i];
	mat_kerneldatap[i].aa    = aa;
	mat_kerneldatap[i].adiag = diag + trstarts[i];
	mat_kerneldatap[i].x     = v + trstarts[i];
	mat_kerneldatap[i].is    = is;
	mat_pdata[i]             = &mat_kerneldatap[i];
      }
      ierr = PetscThreadsRunKernel(MatDiagonalSet_Kernel,(void**)mat_pdata,tmap->nthreads,tmap->affinity);CHKERRQ(ierr);

      ierr = VecRestoreArray(D,&v);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    aij->idiagvalid = PETSC_FALSE;
    aij->ibdiagvalid = PETSC_FALSE;
  }
  ierr = MatDiagonalSet_Default(Y,D,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetUp_SeqAIJPThread"
PetscErrorCode MatSetUp_SeqAIJPThread(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJPThreadSetPreallocation_SeqAIJPThread(A,PETSC_DEFAULT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_SeqAIJPThread"
static PetscErrorCode MatView_SeqAIJPThread(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (iascii && (format == PETSC_VIEWER_ASCII_FACTOR_INFO || format == PETSC_VIEWER_ASCII_INFO)) {
    PetscInt nthreads;
    ierr = MatGetNThreads(A,&nthreads);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"nthreads=%D\n",nthreads);CHKERRQ(ierr);
  }
  ierr = MatView_SeqAIJ(A,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJPThread"
PetscErrorCode MatDestroy_SeqAIJPThread(Mat A)
{
  PetscErrorCode      ierr;

  PetscFunctionBegin;

  if(!A->rmap->refcnt) {
    ierr = PetscThreadsLayoutDestroy(&A->rmap->tmap);CHKERRQ(ierr);
  }

  mats_created--;
  /* Free the kernel data structure on the destruction of the last matrix */
  if(!mats_created) {
    ierr = PetscFree(mat_kerneldatap);CHKERRQ(ierr);
    ierr = PetscFree(mat_pdata);CHKERRQ(ierr);
  }
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetNThreads"
/*@
   MatSetNThreads - Set the number of threads to be used for matrix operations.

   Not Collective, but it is usually desirable to use the same number of threads per process

   Input Parameters
+  A - the matrix
-  nthreads - number of threads

   Level: intermediate

   Concepts: matrix^setting number of threads

.seealso: MatCreateSeqAIJPThread()
@*/
PetscErrorCode MatSetNThreads(Mat A,PetscInt nthreads)
{
  PetscErrorCode     ierr;
  PetscThreadsLayout tmap=A->rmap->tmap;
  PetscInt           nworkThreads=PetscMaxThreads+PetscMainThreadShareWork;

  PetscFunctionBegin;
  
  if(!tmap) {
    ierr = PetscThreadsLayoutCreate(&tmap);CHKERRQ(ierr);
    A->rmap->tmap = tmap;
  }

  if(nthreads == PETSC_DECIDE) {
    tmap->nthreads = nworkThreads;
    ierr = PetscOptionsInt("-mat_threads","Set number of threads to be used for matrix operations","MatSetNThreads",nworkThreads,&tmap->nthreads,PETSC_NULL);CHKERRQ(ierr);
    if(tmap->nthreads > nworkThreads) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Mat A: threads requested %D, Max. threads initialized %D",tmap->nthreads,nworkThreads);
    }
  } else {
    if(nthreads > nworkThreads) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Mat A: threads requested %D, Max. threads initialized %D",nthreads,nworkThreads);
    }
    tmap->nthreads = nthreads;
  }
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetNThreads"
/*@
   MatGetNThreads - Get the number of threads used for matrix operations.

   Not Collective

   Input Parameter
.  A - the matrix

   Output Parameter:
.  nthreads - number of threads

   Level: intermediate

   Concepts: matrix^getting number of threads

.seealso: MatCreateSeqAIJPThread(), MatSetNThreads()
@*/
PetscErrorCode MatGetNThreads(Mat A,PetscInt *nthreads)
{
  PetscThreadsLayout tmap=A->rmap->tmap;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  if (tmap) *nthreads = tmap->nthreads - PetscMainThreadShareWork;
  else *nthreads = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetThreadAffinities"
/*
   MatSetThreadAffinities - Sets the CPU affinities of matrix threads.

   Input Parameters
+  A - the matrix
-  affinities - list of cpu affinities for threads.

   Notes:
   Must set affinities for all the threads used with the matrix.
   size(affinities[]) = nthreads
   Use affinities[] = PETSC_NULL for PETSc to set the thread affinities.

   Options Database Keys:
+  -mat_thread_affinities - Comma seperated list of thread affinities

   Level: intermediate

   Concepts: matrices^thread cpu affinity

.seealso: MatPThreadGetThreadAffinities()
*/
PetscErrorCode MatSetThreadAffinities(Mat A,const PetscInt affinities[])
{
  PetscErrorCode    ierr;
  PetscThreadsLayout tmap=A->rmap->tmap;
  PetscInt          nmax=PetscMaxThreads+PetscMainThreadShareWork;
  PetscBool         flg;

  PetscFunctionBegin;

  if(!tmap) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must set the number of threads before setting thread affinities");
  }

  ierr = PetscMalloc(tmap->nthreads*sizeof(PetscInt),&tmap->affinity);CHKERRQ(ierr);

  if(affinities == PETSC_NULL) {
    PetscInt *thread_affinities;
    ierr = PetscMalloc(nmax*sizeof(PetscInt),&thread_affinities);CHKERRQ(ierr);
    ierr = PetscMemzero(thread_affinities,nmax*sizeof(PetscInt));CHKERRQ(ierr);
    /* Check if run-time option is set */
    ierr = PetscOptionsIntArray("-mat_thread_affinities","Set CPU affinity for each thread","MatSetThreadAffinities",thread_affinities,&nmax,&flg);CHKERRQ(ierr);
    if(flg) {
      if(nmax != tmap->nthreads) {
	SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Must set affinities for all threads, matrix A Threads = %D, CPU affinities set = %D",tmap->nthreads,nmax);
      }
      ierr = PetscMemcpy(tmap->affinity,thread_affinities,tmap->nthreads*sizeof(PetscInt));
    } else {
      /* Reuse the core affinities set for first s->nthreads */
      ierr = PetscMemcpy(tmap->affinity,PetscThreadsCoreAffinities,tmap->nthreads*sizeof(PetscInt));
    }
    ierr = PetscFree(thread_affinities);CHKERRQ(ierr);
  } else {
    /* Set user provided affinities */
    ierr = PetscMemcpy(tmap->affinity,affinities,tmap->nthreads*sizeof(PetscInt));
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatSeqAIJPThreadSetPreallocation_SeqAIJPThread"
PetscErrorCode MatSeqAIJPThreadSetPreallocation_SeqAIJPThread(Mat A, PetscInt nz,const PetscInt nnz[])
{
  PetscErrorCode ierr;
  PetscThreadsLayout tmap=A->rmap->tmap;

  PetscFunctionBegin;
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(A, nz, nnz);CHKERRQ(ierr);

  tmap->N = A->rmap->n;
  ierr = PetscThreadsLayoutSetUp(tmap);CHKERRQ(ierr);

  ierr = MatZeroEntries_SeqAIJPThread(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqAIJPThread"
PetscErrorCode MatCreate_SeqAIJPThread(Mat B)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *b;
  PetscThreadsLayout tmap=B->rmap->tmap;

  PetscFunctionBegin;
  ierr = PetscThreadsInitialize(PetscMaxThreads);CHKERRQ(ierr);
  ierr = MatCreate_SeqAIJ(B);
  b = (Mat_SeqAIJ*)B->data;
  /* Set inodes off */
  b->inode.use = PETSC_FALSE;

  if(!B->rmap->tmap) {
    ierr = PetscThreadsLayoutCreate(&B->rmap->tmap);CHKERRQ(ierr);
    tmap = B->rmap->tmap;
  }

  /* Set the number of threads */
  if(tmap->nthreads == PETSC_DECIDE) {
    ierr = MatSetNThreads(B,PETSC_DECIDE);CHKERRQ(ierr);
  }
  /* Set thread affinities */
  if(!tmap->affinity) {
    ierr = MatSetThreadAffinities(B,PETSC_NULL);CHKERRQ(ierr);
  }

  B->ops->destroy         = MatDestroy_SeqAIJPThread;
  B->ops->mult            = MatMult_SeqAIJPThread;
  B->ops->multadd         = MatMultAdd_SeqAIJPThread;
  B->ops->setup           = MatSetUp_SeqAIJPThread;
  B->ops->zeroentries     = MatZeroEntries_SeqAIJPThread;
  B->ops->realpart        = MatRealPart_SeqAIJPThread;
  B->ops->imaginarypart   = MatImaginaryPart_SeqAIJPThread;
  B->ops->getdiagonal     = MatGetDiagonal_SeqAIJPThread;
  B->ops->missingdiagonal = MatMissingDiagonal_SeqAIJPThread;
  B->ops->findzerodiagonals = MatFindZeroDiagonals_SeqAIJPThread;
  B->ops->diagonalscale     = MatDiagonalScale_SeqAIJPThread;
  B->ops->diagonalset       = MatDiagonalSet_SeqAIJPThread;
  B->ops->view = MatView_SeqAIJPThread;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSeqAIJSetPreallocation_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSeqAIJSetPreallocation_C","MatSeqAIJPThreadSetPreallocation_SeqAIJPThread",MatSeqAIJPThreadSetPreallocation_SeqAIJPThread);CHKERRQ(ierr);

  if(mats_created == 0) {
    ierr = PetscMalloc((PetscMaxThreads+PetscMainThreadShareWork)*sizeof(Mat_KernelData),&mat_kerneldatap);CHKERRQ(ierr);
    ierr = PetscMalloc((PetscMaxThreads+PetscMainThreadShareWork)*sizeof(Mat_KernelData*),&mat_pdata);CHKERRQ(ierr);
  }
  mats_created++;

  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJPTHREAD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatCreateSeqAIJPThread"
/*
   MatCreateSeqAIJPThread - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format) using posix threads.  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nthreads - number of threads for matrix operations
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows 
         (possibly different for each row) or PETSC_NULL

   Output Parameter:
.  A - the matrix 

   It is recommended that one use the MatCreate(), MatSetSizes(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradgm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   If nnz is given then nz is ignored

   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.  For large problems you MUST preallocate memory or you 
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   By default, this format uses inodes (identical nodes) when possible, to 
   improve numerical efficiency of matrix-vector products and solves. We 
   search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Options Database Keys:
+  -mat_no_inode  - Do not use inodes
-  -mat_inode_limit <limit> - Sets inode limit (max limit=5)

   Level: intermediate

.seealso: MatCreate(), MatCreateAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays()

*/
PetscErrorCode  MatCreateSeqAIJPThread(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nthreads,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSetNThreads(*A,nthreads);CHKERRQ(ierr);
  ierr = MatSeqAIJPThreadSetPreallocation_SeqAIJPThread(*A,nz,nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
