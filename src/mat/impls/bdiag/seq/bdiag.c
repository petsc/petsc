#define PETSCMAT_DLL

/* Block diagonal matrix format */

#include "src/mat/impls/bdiag/seq/bdiag.h"
#include "src/inline/ilu.h"

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqBDiag"
PetscErrorCode MatDestroy_SeqBDiag(Mat A)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,bs = A->bs;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)A,"Rows=%D, Cols=%D, NZ=%D, BSize=%D, NDiag=%D",A->m,A->n,a->nz,A->bs,a->nd);
#endif
  if (!a->user_alloc) { /* Free the actual diagonals */
    for (i=0; i<a->nd; i++) {
      if (a->diag[i] > 0) {
        PetscScalar *dummy = a->diagv[i] + bs*bs*a->diag[i]; 
        ierr = PetscFree(dummy);CHKERRQ(ierr);
      } else {
        ierr = PetscFree(a->diagv[i]);CHKERRQ(ierr);
      }
    }
  }
  if (a->pivot) {ierr = PetscFree(a->pivot);CHKERRQ(ierr);}
  ierr = PetscFree(a->diagv);CHKERRQ(ierr);
  ierr = PetscFree(a->diag);CHKERRQ(ierr);
  ierr = PetscFree(a->colloc);CHKERRQ(ierr);
  ierr = PetscFree(a->dvalue);CHKERRQ(ierr);
  if (a->solvework) {ierr = PetscFree(a->solvework);CHKERRQ(ierr);}
  ierr = PetscFree(a);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqBDiagSetPreallocation_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_SeqBDiag"
PetscErrorCode MatAssemblyEnd_SeqBDiag(Mat A,MatAssemblyType mode)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscInt       i,k,temp,*diag = a->diag,*bdlen = a->bdlen;
  PetscScalar    *dtemp,**dv = a->diagv;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  /* Sort diagonals */
  for (i=0; i<a->nd; i++) {
    for (k=i+1; k<a->nd; k++) {
      if (diag[i] < diag[k]) {
        temp     = diag[i];   
        diag[i]  = diag[k];
        diag[k]  = temp;
        temp     = bdlen[i];   
        bdlen[i] = bdlen[k];
        bdlen[k] = temp;
        dtemp    = dv[i];
        dv[i]    = dv[k];
        dv[k]    = dtemp;
      }
    }
  }

  /* Set location of main diagonal */
  for (i=0; i<a->nd; i++) {
    if (!a->diag[i]) {a->mainbd = i; break;}
  }
  ierr = PetscLogInfo((A,"MatAssemblyEnd_SeqBDiag:Number diagonals %D,memory used %D, block size %D\n",a->nd,a->maxnz,A->bs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_SeqBDiag"
PetscErrorCode MatSetOption_SeqBDiag(Mat A,MatOption op)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_NO_NEW_NONZERO_LOCATIONS:
    a->nonew       = 1;
    break;
  case MAT_YES_NEW_NONZERO_LOCATIONS:
    a->nonew       = 0;
    break;
  case MAT_NO_NEW_DIAGONALS:
    a->nonew_diag  = 1;
    break;
  case MAT_YES_NEW_DIAGONALS:
    a->nonew_diag  = 0;
    break;
  case MAT_COLUMN_ORIENTED:
    a->roworiented = PETSC_FALSE;
    break;
  case MAT_ROW_ORIENTED:
    a->roworiented = PETSC_TRUE;
    break;
  case MAT_ROWS_SORTED:
  case MAT_ROWS_UNSORTED:
  case MAT_COLUMNS_SORTED:
  case MAT_COLUMNS_UNSORTED:
  case MAT_IGNORE_OFF_PROC_ENTRIES:
  case MAT_NEW_NONZERO_LOCATION_ERR:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
  case MAT_USE_HASH_TABLE:
    ierr = PetscLogInfo((A,"MatSetOption_SeqBDiag:Option ignored\n"));CHKERRQ(ierr);
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
    SETERRQ(PETSC_ERR_SUP,"unknown option");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPrintHelp_SeqBDiag"
PetscErrorCode MatPrintHelp_SeqBDiag(Mat A)
{
  static PetscTruth called = PETSC_FALSE; 
  MPI_Comm          comm = A->comm;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (called) {PetscFunctionReturn(0);} else called = PETSC_TRUE;
  ierr = (*PetscHelpPrintf)(comm," Options for MATSEQBDIAG and MATMPIBDIAG matrix formats:\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -mat_block_size <block_size>\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -mat_bdiag_diags <d1,d2,d3,...> (diagonal numbers)\n");CHKERRQ(ierr); 
  ierr = (*PetscHelpPrintf)(comm,"   (for example) -mat_bdiag_diags -5,-1,0,1,5\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_SeqBDiag_N"
static PetscErrorCode MatGetDiagonal_SeqBDiag_N(Mat A,Vec v)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,n,len,ibase,bs = A->bs,iloc;
  PetscScalar    *x,*dd,zero = 0.0;

  PetscFunctionBegin;
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");  
  ierr = VecSet(v,zero);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != A->m) SETERRQ(PETSC_ERR_ARG_SIZ,"Nonconforming mat and vec");
  if (a->mainbd == -1) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Main diagonal not set");
  len = PetscMin(a->mblock,a->nblock);
  dd = a->diagv[a->mainbd];
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  for (i=0; i<len; i++) {
    ibase = i*bs*bs;  iloc = i*bs;
    for (j=0; j<bs; j++) x[j + iloc] = dd[ibase + j*(bs+1)];
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_SeqBDiag_1"
static PetscErrorCode MatGetDiagonal_SeqBDiag_1(Mat A,Vec v)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,n,len;
  PetscScalar    *x,*dd,zero = 0.0;

  PetscFunctionBegin;
  ierr = VecSet(v,zero);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != A->m) SETERRQ(PETSC_ERR_ARG_SIZ,"Nonconforming mat and vec");
  if (a->mainbd == -1) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Main diagonal not set");
  dd = a->diagv[a->mainbd];
  len = PetscMin(A->m,A->n);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  for (i=0; i<len; i++) x[i] = dd[i];
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroEntries_SeqBDiag"
PetscErrorCode MatZeroEntries_SeqBDiag(Mat A)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag*)A->data;
  PetscInt     d,i,len,bs = A->bs;
  PetscScalar  *dv;

  PetscFunctionBegin;
  for (d=0; d<a->nd; d++) {
    dv  = a->diagv[d];
    if (a->diag[d] > 0) {
      dv += bs*bs*a->diag[d];
    }
    len = a->bdlen[d]*bs*bs;
    for (i=0; i<len; i++) dv[i] = 0.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_SeqBDiag"
PetscErrorCode MatZeroRows_SeqBDiag(Mat A,PetscInt N,const PetscInt rows[],PetscScalar diag)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,m = A->m - 1,nz;
  PetscScalar    *dd;
  PetscScalar    *val;

  PetscFunctionBegin;
  for (i=0; i<N; i++) {
    if (rows[i]<0 || rows[i]>m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"row out of range");
    ierr = MatGetRow_SeqBDiag(A,rows[i],&nz,PETSC_NULL,&val);CHKERRQ(ierr);
    ierr = PetscMemzero((void*)val,nz*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = MatRestoreRow_SeqBDiag(A,rows[i],&nz,PETSC_NULL,&val);CHKERRQ(ierr);
  }
  if (diag != 0.0) {
    if (a->mainbd == -1) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Main diagonal does not exist");
    dd = a->diagv[a->mainbd];
    for (i=0; i<N; i++) dd[rows[i]] = diag;
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrix_SeqBDiag"
PetscErrorCode MatGetSubMatrix_SeqBDiag(Mat A,IS isrow,IS iscol,MatReuse scall,Mat *submat)
{
  PetscErrorCode ierr;
  PetscInt       nznew,*smap,i,j,oldcols = A->n;
  PetscInt       *irow,*icol,newr,newc,*cwork,nz,bs;
  PetscInt       *col;
  PetscScalar    *vwork;
  PetscScalar    *val;
  Mat            newmat;

  PetscFunctionBegin;
  if (scall == MAT_REUSE_MATRIX) { /* no support for reuse so simply destroy all */
    ierr = MatDestroy(*submat);CHKERRQ(ierr);
  }

  ierr = ISGetIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrow,&newr);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscol,&newc);CHKERRQ(ierr);

  ierr = PetscMalloc((oldcols+1)*sizeof(PetscInt),&smap);CHKERRQ(ierr);
  ierr = PetscMalloc((newc+1)*sizeof(PetscInt),&cwork);CHKERRQ(ierr);
  ierr = PetscMalloc((newc+1)*sizeof(PetscScalar),&vwork);CHKERRQ(ierr);
  ierr  = PetscMemzero((char*)smap,oldcols*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<newc; i++) smap[icol[i]] = i+1;

  /* Determine diagonals; then create submatrix */
  bs = A->bs; /* Default block size remains the same */
  ierr = MatCreate(A->comm,&newmat);CHKERRQ(ierr);
  ierr = MatSetSizes(newmat,newr,newc,newr,newc);CHKERRQ(ierr);
  ierr = MatSetType(newmat,A->type_name);CHKERRQ(ierr);
  ierr = MatSeqBDiagSetPreallocation(newmat,0,bs,PETSC_NULL,PETSC_NULL);

  /* Fill new matrix */
  for (i=0; i<newr; i++) {
    ierr = MatGetRow_SeqBDiag(A,irow[i],&nz,&col,&val);CHKERRQ(ierr);
    nznew = 0;
    for (j=0; j<nz; j++) {
      if (smap[col[j]]) {
        cwork[nznew]   = smap[col[j]] - 1;
        vwork[nznew++] = val[j];
      }
    }
    ierr = MatSetValues(newmat,1,&i,nznew,cwork,vwork,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow_SeqBDiag(A,i,&nz,&col,&val);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Free work space */
  ierr = PetscFree(smap);CHKERRQ(ierr);
  ierr = PetscFree(cwork);CHKERRQ(ierr);
  ierr = PetscFree(vwork);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&icol);CHKERRQ(ierr);
  *submat = newmat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrices_SeqBDiag"
PetscErrorCode MatGetSubMatrices_SeqBDiag(Mat A,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *B[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscMalloc((n+1)*sizeof(Mat),B);CHKERRQ(ierr);
  }

  for (i=0; i<n; i++) {
    ierr = MatGetSubMatrix_SeqBDiag(A,irow[i],icol[i],scall,&(*B)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_SeqBDiag"
PetscErrorCode MatScale_SeqBDiag(Mat inA,PetscScalar alpha)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag*)inA->data;
  PetscInt          i,bs = inA->bs;
  PetscScalar  oalpha = alpha;
  PetscBLASInt one = 1,len;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<a->nd; i++) {
    len = (PetscBLASInt)bs*bs*a->bdlen[i];
    if (a->diag[i] > 0) {
      BLASscal_(&len,&oalpha,a->diagv[i] + bs*bs*a->diag[i],&one);
    } else {
      BLASscal_(&len,&oalpha,a->diagv[i],&one);
    }
  }
  ierr = PetscLogFlops(a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalScale_SeqBDiag"
PetscErrorCode MatDiagonalScale_SeqBDiag(Mat A,Vec ll,Vec rr)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscScalar    *l,*r,*dv;
  PetscErrorCode ierr;
  PetscInt       d,j,len;
  PetscInt       nd = a->nd,bs = A->bs,diag,m,n;

  PetscFunctionBegin;
  if (ll) {
    ierr = VecGetSize(ll,&m);CHKERRQ(ierr);
    if (m != A->m) SETERRQ(PETSC_ERR_ARG_SIZ,"Left scaling vector wrong length");
    if (bs == 1) {
      ierr = VecGetArray(ll,&l);CHKERRQ(ierr); 
      for (d=0; d<nd; d++) {
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];
        if (diag > 0) for (j=0; j<len; j++) dv[j+diag] *= l[j+diag];
        else          for (j=0; j<len; j++) dv[j]      *= l[j];
      }
      ierr = VecRestoreArray(ll,&l);CHKERRQ(ierr); 
      ierr = PetscLogFlops(a->nz);CHKERRQ(ierr);
    } else SETERRQ(PETSC_ERR_SUP,"Not yet done for bs>1");
  }
  if (rr) {
    ierr = VecGetSize(rr,&n);CHKERRQ(ierr);
    if (n != A->n) SETERRQ(PETSC_ERR_ARG_SIZ,"Right scaling vector wrong length");
    if (bs == 1) {
      ierr = VecGetArray(rr,&r);CHKERRQ(ierr);  
      for (d=0; d<nd; d++) {
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];
        if (diag > 0) for (j=0; j<len; j++) dv[j+diag] *= r[j];
        else          for (j=0; j<len; j++) dv[j]      *= r[j-diag];
      }
      ierr = VecRestoreArray(rr,&r);CHKERRQ(ierr);  
      ierr = PetscLogFlops(a->nz);CHKERRQ(ierr);
    } else SETERRQ(PETSC_ERR_SUP,"Not yet done for bs>1");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_SeqBDiag(Mat,MatDuplicateOption,Mat *);

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpPreallocation_SeqBDiag"
PetscErrorCode MatSetUpPreallocation_SeqBDiag(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr =  MatSeqBDiagSetPreallocation(A,PETSC_DEFAULT,PETSC_DEFAULT,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_SeqBDiag_N,
       MatGetRow_SeqBDiag,
       MatRestoreRow_SeqBDiag,
       MatMult_SeqBDiag_N,
/* 4*/ MatMultAdd_SeqBDiag_N,
       MatMultTranspose_SeqBDiag_N,
       MatMultTransposeAdd_SeqBDiag_N,
       MatSolve_SeqBDiag_N,
       0,
       0,
/*10*/ 0,
       0,
       0,
       MatRelax_SeqBDiag_N,
       MatTranspose_SeqBDiag,
/*15*/ MatGetInfo_SeqBDiag,
       0,
       MatGetDiagonal_SeqBDiag_N,
       MatDiagonalScale_SeqBDiag,
       MatNorm_SeqBDiag,
/*20*/ 0,
       MatAssemblyEnd_SeqBDiag,
       0,
       MatSetOption_SeqBDiag,
       MatZeroEntries_SeqBDiag,
/*25*/ MatZeroRows_SeqBDiag,
       0,
       MatLUFactorNumeric_SeqBDiag_N,
       0,
       0,
/*30*/ MatSetUpPreallocation_SeqBDiag,
       MatILUFactorSymbolic_SeqBDiag,
       0,
       0,
       0,
/*35*/ MatDuplicate_SeqBDiag,
       0,
       0,
       MatILUFactor_SeqBDiag,
       0,
/*40*/ 0,
       MatGetSubMatrices_SeqBDiag,
       0,
       MatGetValues_SeqBDiag_N,
       0,
/*45*/ MatPrintHelp_SeqBDiag,
       MatScale_SeqBDiag,
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
       MatDestroy_SeqBDiag,
       MatView_SeqBDiag,
       MatGetPetscMaps_Petsc,
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
       MatLoad_SeqBDiag,
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

#undef __FUNCT__  
#define __FUNCT__ "MatSeqBDiagSetPreallocation"
/*@C
   MatSeqBDiagSetPreallocation - Sets the nonzero structure and (optionally) arrays.

   Collective on MPI_Comm

   Input Parameters:
+  B - the matrix
.  nd - number of block diagonals (optional)
.  bs - each element of a diagonal is an bs x bs dense matrix
.  diag - optional array of block diagonal numbers (length nd).
   For a matrix element A[i,j], where i=row and j=column, the
   diagonal number is
$     diag = i/bs - j/bs  (integer division)
   Set diag=PETSC_NULL on input for PETSc to dynamically allocate memory as 
   needed (expensive).
-  diagv - pointer to actual diagonals (in same order as diag array), 
   if allocated by user.  Otherwise, set diagv=PETSC_NULL on input for PETSc
   to control memory allocation.

   Options Database Keys:
.  -mat_block_size <bs> - Sets blocksize
.  -mat_bdiag_diags <s1,s2,s3,...> - Sets diagonal numbers

   Notes:
   See the users manual for further details regarding this storage format.

   Fortran Note:
   Fortran programmers cannot set diagv; this value is ignored.

   Level: intermediate

.keywords: matrix, block, diagonal, sparse

.seealso: MatCreate(), MatCreateMPIBDiag(), MatSetValues()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSeqBDiagSetPreallocation(Mat B,PetscInt nd,PetscInt bs,const PetscInt diag[],PetscScalar *diagv[])
{
  PetscErrorCode ierr,(*f)(Mat,PetscInt,PetscInt,const PetscInt[],PetscScalar*[]);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)B,"MatSeqBDiagSetPreallocation_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(B,nd,bs,diag,diagv);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSeqBDiagSetPreallocation_SeqBDiag"
PetscErrorCode PETSCMAT_DLLEXPORT MatSeqBDiagSetPreallocation_SeqBDiag(Mat B,PetscInt nd,PetscInt bs,PetscInt *diag,PetscScalar **diagv)
{
  Mat_SeqBDiag   *b;
  PetscErrorCode ierr;
  PetscInt       i,nda,sizetot, nd2 = 128,idiag[128];
  PetscTruth     flg1;

  PetscFunctionBegin;

  B->preallocated = PETSC_TRUE;
  if (bs == PETSC_DEFAULT) bs = 1;
  if (!bs) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Blocksize cannot be zero");
  if (nd == PETSC_DEFAULT) nd = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetIntArray(PETSC_NULL,"-mat_bdiag_diags",idiag,&nd2,&flg1);CHKERRQ(ierr);
  if (flg1) {
    diag = idiag;
    nd   = nd2;
  }

  if ((B->n%bs) || (B->m%bs)) SETERRQ(PETSC_ERR_ARG_SIZ,"Invalid block size");
  if (!nd) nda = nd + 1;
  else     nda = nd;
  b            = (Mat_SeqBDiag*)B->data;

  ierr = PetscOptionsHasName(PETSC_NULL,"-mat_no_unroll",&flg1);CHKERRQ(ierr);
  if (!flg1) {
    switch (bs) {
      case 1:
        B->ops->setvalues       = MatSetValues_SeqBDiag_1;
        B->ops->getvalues       = MatGetValues_SeqBDiag_1;
        B->ops->getdiagonal     = MatGetDiagonal_SeqBDiag_1;
        B->ops->mult            = MatMult_SeqBDiag_1;
        B->ops->multadd         = MatMultAdd_SeqBDiag_1;
        B->ops->multtranspose   = MatMultTranspose_SeqBDiag_1;
        B->ops->multtransposeadd= MatMultTransposeAdd_SeqBDiag_1;
        B->ops->relax           = MatRelax_SeqBDiag_1;
        B->ops->solve           = MatSolve_SeqBDiag_1;
        B->ops->lufactornumeric = MatLUFactorNumeric_SeqBDiag_1;
        break;
      case 2:
	B->ops->mult            = MatMult_SeqBDiag_2; 
        B->ops->multadd         = MatMultAdd_SeqBDiag_2;
        B->ops->solve           = MatSolve_SeqBDiag_2;
        break;
      case 3:
	B->ops->mult            = MatMult_SeqBDiag_3; 
        B->ops->multadd         = MatMultAdd_SeqBDiag_3;
	B->ops->solve           = MatSolve_SeqBDiag_3; 
        break;
      case 4:
	B->ops->mult            = MatMult_SeqBDiag_4; 
        B->ops->multadd         = MatMultAdd_SeqBDiag_4;
	B->ops->solve           = MatSolve_SeqBDiag_4; 
        break;
      case 5:
	B->ops->mult            = MatMult_SeqBDiag_5; 
        B->ops->multadd         = MatMultAdd_SeqBDiag_5;
	B->ops->solve           = MatSolve_SeqBDiag_5; 
        break;
   }
  }

  b->mblock = B->m/bs;
  b->nblock = B->n/bs;
  b->nd     = nd;
  B->bs     = bs;
  b->ndim   = 0;
  b->mainbd = -1;
  b->pivot  = 0;

  ierr      = PetscMalloc(2*nda*sizeof(PetscInt),&b->diag);CHKERRQ(ierr);
  b->bdlen  = b->diag + nda;
  ierr      = PetscMalloc((B->n+1)*sizeof(PetscInt),&b->colloc);CHKERRQ(ierr);
  ierr      = PetscMalloc(nda*sizeof(PetscScalar*),&b->diagv);CHKERRQ(ierr);
  sizetot   = 0;

  if (diagv) { /* user allocated space */
    b->user_alloc = PETSC_TRUE;
    for (i=0; i<nd; i++) b->diagv[i] = diagv[i];
  } else b->user_alloc = PETSC_FALSE;

  for (i=0; i<nd; i++) {
    b->diag[i] = diag[i];
    if (diag[i] > 0) { /* lower triangular */
      b->bdlen[i] = PetscMin(b->nblock,b->mblock - diag[i]);
    } else {           /* upper triangular */
      b->bdlen[i] = PetscMin(b->mblock,b->nblock + diag[i]);
    }
    sizetot += b->bdlen[i];
  }
  sizetot   *= bs*bs;
  b->maxnz  =  sizetot;
  ierr      = PetscMalloc((B->n+1)*sizeof(PetscScalar),&b->dvalue);CHKERRQ(ierr);
  ierr      = PetscLogObjectMemory(B,(nda*(bs+2))*sizeof(PetscInt) + bs*nda*sizeof(PetscScalar)
                    + nda*sizeof(PetscScalar*) + sizeof(Mat_SeqBDiag)
                    + sizeof(struct _p_Mat) + sizetot*sizeof(PetscScalar));CHKERRQ(ierr);

  if (!b->user_alloc) {
    for (i=0; i<nd; i++) {
      ierr = PetscMalloc(bs*bs*b->bdlen[i]*sizeof(PetscScalar),&b->diagv[i]);CHKERRQ(ierr);
      ierr = PetscMemzero(b->diagv[i],bs*bs*b->bdlen[i]*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    b->nonew = 0; b->nonew_diag = 0;
  } else { /* diagonals are set on input; don't allow dynamic allocation */
    b->nonew = 1; b->nonew_diag = 1;
  }

  /* adjust diagv so one may access rows with diagv[diag][row] for all rows */
  for (i=0; i<nd; i++) {
    if (diag[i] > 0) {
      b->diagv[i] -= bs*bs*diag[i];
    }
  }

  b->nz          = b->maxnz; /* Currently not keeping track of exact count */
  b->roworiented = PETSC_TRUE;
  B->info.nz_unneeded = (double)b->maxnz;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate_SeqBDiag"
static PetscErrorCode MatDuplicate_SeqBDiag(Mat A,MatDuplicateOption cpvalues,Mat *matout)
{ 
  Mat_SeqBDiag   *newmat,*a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,len,diag,bs = A->bs;
  Mat            mat;

  PetscFunctionBegin;
  ierr = MatCreate(A->comm,matout);CHKERRQ(ierr);
  ierr = MatSetSizes(*matout,A->m,A->n,A->m,A->n);CHKERRQ(ierr);
  ierr = MatSetType(*matout,A->type_name);CHKERRQ(ierr);
  ierr = MatSeqBDiagSetPreallocation(*matout,a->nd,bs,a->diag,PETSC_NULL);CHKERRQ(ierr);

  /* Copy contents of diagonals */
  mat = *matout;
  newmat = (Mat_SeqBDiag*)mat->data;
  if (cpvalues == MAT_COPY_VALUES) {
    for (i=0; i<a->nd; i++) {
      len = a->bdlen[i] * bs * bs * sizeof(PetscScalar);
      diag = a->diag[i];
      if (diag > 0) {
        ierr = PetscMemcpy(newmat->diagv[i]+bs*bs*diag,a->diagv[i]+bs*bs*diag,len);CHKERRQ(ierr);
      } else {
        ierr = PetscMemcpy(newmat->diagv[i],a->diagv[i],len);CHKERRQ(ierr);
      }
    }
  } 
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLoad_SeqBDiag"
PetscErrorCode MatLoad_SeqBDiag(PetscViewer viewer, MatType type,Mat *A)
{
  Mat            B;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  int            fd;
  PetscInt       *scols,i,nz,header[4],nd = 128;
  PetscInt       bs,*rowlengths = 0,M,N,*cols,extra_rows,*diag = 0;
  PetscInt       idiag[128];
  PetscScalar    *vals,*svals;
  MPI_Comm       comm;
  PetscTruth     flg;
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_SIZ,"view must have one processor");
  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,PETSC_INT);CHKERRQ(ierr);
  if (header[0] != MAT_FILE_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"Not matrix object");
  M = header[1]; N = header[2]; nz = header[3];
  if (M != N) SETERRQ(PETSC_ERR_SUP,"Can only load square matrices");
  if (header[3] < 0) {
    SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"Matrix stored in special format, cannot load as SeqBDiag");
  }

  /* 
     This code adds extra rows to make sure the number of rows is 
    divisible by the blocksize
  */
  bs = 1;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-matload_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);
  extra_rows = bs - M + bs*(M/bs);
  if (extra_rows == bs) extra_rows = 0;
  if (extra_rows) {
    ierr = PetscLogInfo((0,"MatLoad_SeqBDiag:Padding loaded matrix to match blocksize\n"));CHKERRQ(ierr);
  }

  /* read row lengths */
  ierr = PetscMalloc((M+extra_rows)*sizeof(PetscInt),&rowlengths);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);
  for (i=0; i<extra_rows; i++) rowlengths[M+i] = 1;

  /* load information about diagonals */
  ierr = PetscOptionsGetIntArray(PETSC_NULL,"-matload_bdiag_diags",idiag,&nd,&flg);CHKERRQ(ierr);
  if (flg) {
    diag = idiag;
  }

  /* create our matrix */
  ierr = MatCreate(comm,A);
  ierr = MatSetSizes(*A,M+extra_rows,M+extra_rows,M+extra_rows,M+extra_rows);
  ierr = MatSetType(*A,type);CHKERRQ(ierr);
  ierr = MatSeqBDiagSetPreallocation(*A,nd,bs,diag,PETSC_NULL);CHKERRQ(ierr);
  B = *A;

  /* read column indices and nonzeros */
  ierr = PetscMalloc(nz*sizeof(PetscInt),&scols);CHKERRQ(ierr);
  cols = scols;
  ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscMalloc(nz*sizeof(PetscScalar),&svals);CHKERRQ(ierr);
  vals = svals;
  ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
  /* insert into matrix */

  for (i=0; i<M; i++) {
    ierr = MatSetValues(B,1,&i,rowlengths[i],scols,svals,INSERT_VALUES);CHKERRQ(ierr);
    scols += rowlengths[i]; svals += rowlengths[i];
  }
  vals[0] = 1.0;
  for (i=M; i<M+extra_rows; i++) {
    ierr = MatSetValues(B,1,&i,1,&i,vals,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);
  ierr = PetscFree(rowlengths);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATSEQBDIAG - MATSEQBDIAG = "seqbdiag" - A matrix type to be used for sequential block diagonal matrices.

   Options Database Keys:
. -mat_type seqbdiag - sets the matrix type to "seqbdiag" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateSeqBDiag
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_SeqBDiag"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqBDiag(Mat B)
{
  Mat_SeqBDiag   *b;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(B->comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_WRONG,"Comm must be of size 1");

  B->m = B->M = PetscMax(B->m,B->M);
  B->n = B->N = PetscMax(B->n,B->N);

  ierr            = PetscNew(Mat_SeqBDiag,&b);CHKERRQ(ierr);
  B->data         = (void*)b;
  ierr            = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->factor       = 0;
  B->mapping      = 0;

  ierr = PetscMapCreateMPI(B->comm,B->m,B->m,&B->rmap);CHKERRQ(ierr);
  ierr = PetscMapCreateMPI(B->comm,B->n,B->n,&B->cmap);CHKERRQ(ierr);

  b->ndim   = 0;
  b->mainbd = -1;
  b->pivot  = 0;

  b->roworiented = PETSC_TRUE;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSeqBDiagSetPreallocation_C",
                                    "MatSeqBDiagSetPreallocation_SeqBDiag",
                                     MatSeqBDiagSetPreallocation_SeqBDiag);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatCreateSeqBDiag"
/*@C
   MatCreateSeqBDiag - Creates a sequential block diagonal matrix.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nd - number of block diagonals (optional)
.  bs - each element of a diagonal is an bs x bs dense matrix
.  diag - optional array of block diagonal numbers (length nd).
   For a matrix element A[i,j], where i=row and j=column, the
   diagonal number is
$     diag = i/bs - j/bs  (integer division)
   Set diag=PETSC_NULL on input for PETSc to dynamically allocate memory as 
   needed (expensive).
-  diagv - pointer to actual diagonals (in same order as diag array), 
   if allocated by user.  Otherwise, set diagv=PETSC_NULL on input for PETSc
   to control memory allocation.

   Output Parameters:
.  A - the matrix

   Options Database Keys:
.  -mat_block_size <bs> - Sets blocksize
.  -mat_bdiag_diags <s1,s2,s3,...> - Sets diagonal numbers

   Notes:
   See the users manual for further details regarding this storage format.

   Fortran Note:
   Fortran programmers cannot set diagv; this value is ignored.

   Level: intermediate

.keywords: matrix, block, diagonal, sparse

.seealso: MatCreate(), MatCreateMPIBDiag(), MatSetValues()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqBDiag(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nd,PetscInt bs,const PetscInt diag[],PetscScalar *diagv[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQBDIAG);CHKERRQ(ierr);
  ierr = MatSeqBDiagSetPreallocation(*A,nd,bs,diag,diagv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
