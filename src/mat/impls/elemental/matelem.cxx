#include <../src/mat/impls/elemental/matelemimpl.h> /*I "petscmat.h" I*/

/*
    The variable Petsc_Elemental_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Mat_Elemental_Grid
*/
static PetscMPIInt Petsc_Elemental_keyval = MPI_KEYVAL_INVALID;

#undef __FUNCT__
#define __FUNCT__ "PetscElementalInitializePackage"
/*@C
   PetscElementalInitializePackage - Initialize Elemental package

   Logically Collective

   Input Arguments:
.  path - the dynamic library path or PETSC_NULL

   Level: developer

.seealso: MATELEMENTAL, PetscElementalFinalizePackage()
@*/
PetscErrorCode PetscElementalInitializePackage(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (elem::Initialized()) PetscFunctionReturn(0);
  { /* We have already initialized MPI, so this song and dance is just to pass these variables (which won't be used by Elemental) through the interface that needs references */
    int zero = 0;
    char **nothing = 0;
    elem::Initialize(zero,nothing);
  }
  ierr = PetscRegisterFinalize(PetscElementalFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscElementalFinalizePackage"
/*@C
   PetscElementalFinalizePackage - Finalize Elemental package

   Logically Collective

   Level: developer

.seealso: MATELEMENTAL, PetscElementalInitializePackage()
@*/
PetscErrorCode PetscElementalFinalizePackage(void)
{

  PetscFunctionBegin;
  elem::Finalize();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_Elemental"
static PetscErrorCode MatView_Elemental(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscViewerFormat format;
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Info viewer not implemented yet");
    } else if (format == PETSC_VIEWER_DEFAULT) {
      Mat Aaij;
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscObjectPrintClassNamePrefixType((PetscObject)A,viewer,"Matrix Object");CHKERRQ(ierr);
      a->emat->Print("Elemental matrix (cyclic ordering)");
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscPrintf(((PetscObject)viewer)->comm,"Elemental matrix (explicit ordering)\n");CHKERRQ(ierr);
      ierr = MatComputeExplicitOperator(A,&Aaij);CHKERRQ(ierr);
      ierr = MatView(Aaij,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = MatDestroy(&Aaij);CHKERRQ(ierr);     
    } else SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Format");
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported by Elemental matrices",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetValues_Elemental"
static PetscErrorCode MatSetValues_Elemental(Mat A,PetscInt nr,const PetscInt *rows,PetscInt nc,const PetscInt *cols,const PetscScalar *vals,InsertMode imode)
{
  PetscErrorCode ierr;
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  PetscMPIInt    rank;
  PetscInt       i,j,rrank,ridx,crank,cidx;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)A)->comm,&rank);CHKERRQ(ierr);
  
  const elem::Grid &grid = a->emat->Grid();
  for (i=0; i<nr; i++) {
    PetscInt erow,ecol,elrow,elcol;
    if (rows[i] < 0) continue;
    P2RO(A,0,rows[i],&rrank,&ridx);
    RO2E(A,0,rrank,ridx,&erow);
    if (rrank < 0 || ridx < 0 || erow < 0) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_PLIB,"Incorrect row translation");
    for (j=0; j<nc; j++) {
      if (cols[j] < 0) continue;
      P2RO(A,1,cols[j],&crank,&cidx);
      RO2E(A,1,crank,cidx,&ecol);
      if (crank < 0 || cidx < 0 || ecol < 0) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_PLIB,"Incorrect col translation");
      if (erow % grid.MCSize() != grid.MCRank() || ecol % grid.MRSize() != grid.MRRank()){ /* off-proc entry */
        if (imode != ADD_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only ADD_VALUES to off-processor entry is supported");
        /* PetscPrintf(PETSC_COMM_SELF,"[%D] add off-proc entry (%D,%D, %g) (%D %D)\n",rank,rows[i],cols[j],*(vals+i*nc),erow,ecol); */
        a->esubmat->Set(0,0, vals[i*nc+j]);
        a->interface->Axpy(1.0,*(a->esubmat),erow,ecol); 
        continue;
      }
      elrow = erow / grid.MCSize();
      elcol = ecol / grid.MRSize();
      switch (imode) {
      case INSERT_VALUES: a->emat->SetLocal(elrow,elcol,vals[i*nc+j]); break;
      case ADD_VALUES: a->emat->UpdateLocal(elrow,elcol,vals[i*nc+j]); break;
      default: SETERRQ1(((PetscObject)A)->comm,PETSC_ERR_SUP,"No support for InsertMode %d",(int)imode);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Elemental"
static PetscErrorCode MatMult_Elemental(Mat A,Vec X,Vec Y)
{
  Mat_Elemental     *a = (Mat_Elemental*)A->data;
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscScalar       *y;
  PetscScalar       one = 1,zero = 0;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  { /* Scoping so that constructor is called before pointer is returned */
    elem::DistMatrix<PetscScalar,elem::VC,elem::STAR> xe(A->cmap->N,1,0,x,A->cmap->n,*a->grid);
    elem::DistMatrix<PetscScalar,elem::VC,elem::STAR> ye(A->rmap->N,1,0,y,A->rmap->n,*a->grid);
    elem::Gemv(elem::NORMAL,one,*a->emat,xe,zero,ye);
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Elemental"
static PetscErrorCode MatMultAdd_Elemental(Mat A,Vec X,Vec Y,Vec Z)
{
  Mat_Elemental     *a = (Mat_Elemental*)A->data;
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscScalar       *z;
  PetscScalar       one = 1.0;

  PetscFunctionBegin;
  if (Y != Z) {ierr = VecCopy(Y,Z);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Z,&z);CHKERRQ(ierr);
  { /* Scoping so that constructor is called before pointer is returned */
    elem::DistMatrix<PetscScalar,elem::VC,elem::STAR> xe(A->cmap->N,1,0,x,A->cmap->n,*a->grid);
    elem::DistMatrix<PetscScalar,elem::VC,elem::STAR> ze(A->rmap->N,1,0,z,A->rmap->n,*a->grid);
    elem::Gemv(elem::NORMAL,one,*a->emat,xe,one,ze);
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Z,&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultNumeric_Elemental"
static PetscErrorCode MatMatMultNumeric_Elemental(Mat A,Mat B,Mat C)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  Mat_Elemental  *b = (Mat_Elemental*)B->data;
  Mat_Elemental  *c = (Mat_Elemental*)C->data;
  PetscScalar    one = 1.0,zero = 0.0;

  PetscFunctionBegin;
  { /* Scoping so that constructor is called before pointer is returned */
    elem::Gemm(elem::NORMAL,elem::NORMAL,one,*a->emat,*b->emat,zero,*c->emat);
  }
  C->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultSymbolic_Elemental"
static PetscErrorCode MatMatMultSymbolic_Elemental(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  Mat            Ce;
  MPI_Comm       comm=((PetscObject)A)->comm;

  PetscFunctionBegin;
  ierr = MatCreate(comm,&Ce);CHKERRQ(ierr);
  ierr = MatSetSizes(Ce,A->rmap->n,B->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Ce,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetUp(Ce);CHKERRQ(ierr);
  *C = Ce;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_Elemental"
static PetscErrorCode MatMatMult_Elemental(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = PetscLogEventBegin(MAT_MatMultSymbolic,A,B,0,0);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_Elemental(A,B,1.0,C);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_MatMultSymbolic,A,B,0,0);CHKERRQ(ierr);   
  }
  ierr = PetscLogEventBegin(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr); 
  ierr = MatMatMultNumeric_Elemental(A,B,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatScale_Elemental"
static PetscErrorCode MatScale_Elemental(Mat X,PetscScalar a)
{
  Mat_Elemental  *x = (Mat_Elemental*)X->data;

  PetscFunctionBegin;
  elem::Scal(a,*x->emat);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAXPY_Elemental"
static PetscErrorCode MatAXPY_Elemental(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_Elemental  *x = (Mat_Elemental*)X->data;
  Mat_Elemental  *y = (Mat_Elemental*)Y->data;

  PetscFunctionBegin;
  elem::Axpy(a,*x->emat,*y->emat);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCopy_Elemental"
static PetscErrorCode MatCopy_Elemental(Mat A,Mat B,MatStructure str)
{
  Mat_Elemental *a=(Mat_Elemental*)A->data;
  Mat_Elemental *b=(Mat_Elemental*)B->data;

  PetscFunctionBegin;
  elem::Copy(*a->emat,*b->emat);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTranspose_Elemental"
static PetscErrorCode MatTranspose_Elemental(Mat A,MatReuse reuse,Mat *B)
{
  Mat            Be;
  PetscErrorCode ierr;
  MPI_Comm       comm=((PetscObject)A)->comm;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX){
    ierr = MatCreate(comm,&Be);CHKERRQ(ierr);
    ierr = MatSetSizes(Be,A->cmap->n,A->rmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetType(Be,MATELEMENTAL);CHKERRQ(ierr);
    ierr = MatSetUp(Be);CHKERRQ(ierr);
    *B = Be;
  }
  Mat_Elemental     *a = (Mat_Elemental*)A->data;
  Mat_Elemental     *b = (Mat_Elemental*)Be->data;
  elem::Transpose(*a->emat,*b->emat);
  Be->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_Elemental"
static PetscErrorCode MatSolve_Elemental(Mat A,Vec B,Vec X)
{
  Mat_Elemental     *a = (Mat_Elemental*)A->data;
  PetscErrorCode    ierr;
  PetscScalar       *x;

  PetscFunctionBegin;
  ierr = VecCopy(B,X);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  elem::DistMatrix<PetscScalar,elem::VC,elem::STAR> xe(A->rmap->N,1,0,x,A->rmap->n,*a->grid);
  elem::DistMatrix<PetscScalar,elem::MC,elem::MR> xer = xe;
  switch (A->factortype) {
  case MAT_FACTOR_LU:
    if ((*a->pivot).AllocatedMemory()) {
      elem::SolveAfterLU(elem::NORMAL,*a->emat,*a->pivot,xer);
      elem::Copy(xer,xe);
    } else {
      elem::SolveAfterLU(elem::NORMAL,*a->emat,xer);
      elem::Copy(xer,xe);
    }
    break;
  case MAT_FACTOR_CHOLESKY:
    elem::SolveAfterCholesky(elem::UPPER,elem::NORMAL,*a->emat,xer);
    elem::Copy(xer,xe);
    break;
  default:
    printf("Error: Unfactored Matrix or Unsupported MatFactorType!\n");
    break;
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatSolve_Elemental"
static PetscErrorCode MatMatSolve_Elemental(Mat A,Mat B,Mat X)
{
  Mat_Elemental *a=(Mat_Elemental*)A->data;
  Mat_Elemental *b=(Mat_Elemental*)B->data;
  Mat_Elemental *x=(Mat_Elemental*)X->data;

  PetscFunctionBegin;
  elem::Copy(*b->emat,*x->emat);
  switch (A->factortype) {
  case MAT_FACTOR_LU:
    if ((*a->pivot).AllocatedMemory()) {
      elem::SolveAfterLU(elem::NORMAL,*a->emat,*a->pivot,*x->emat);
    } else {
      elem::SolveAfterLU(elem::NORMAL,*a->emat,*x->emat);
    }
    break;
  case MAT_FACTOR_CHOLESKY:
    elem::SolveAfterCholesky(elem::UPPER,elem::NORMAL,*a->emat,*x->emat);
    break;
  default:
    printf("Error: Unfactored Matrix or Unsupported MatFactorType!\n");
    break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactor_Elemental"
static PetscErrorCode MatLUFactor_Elemental(Mat A,IS row,IS col,const MatFactorInfo *info)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;

  PetscFunctionBegin;
  if (info->dtcol){
    elem::LU(*a->emat,*a->pivot);
  } else {
    elem::LU(*a->emat);
  }
  A->factortype = MAT_FACTOR_LU; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorNumeric_Elemental"
static PetscErrorCode  MatLUFactorNumeric_Elemental(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCopy(A,F,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatLUFactor_Elemental(F,0,0,info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorSymbolic_Elemental"
static PetscErrorCode  MatLUFactorSymbolic_Elemental(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  /* F is create and allocated by MatGetFactor_elemental_petsc(), skip this routine. */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactor_Elemental"
static PetscErrorCode MatCholeskyFactor_Elemental(Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;

  PetscFunctionBegin;
  printf("MatCholeskyFactor_Elemental is called...\n");
  if (info->dtcol){
    /* A = U^T * U for SPD Matrix A */
    printf("Cholesky Factorization for SPD Matrices...\n");
    elem::Cholesky(elem::UPPER,*a->emat);
  } else {
    /* A = U^T * D * U * for Symmetric Matrix A */ 
    printf("LDL^H Factorization for Symmetric Matrices\n");
    //elem::LU(*a->emat);
  }
  A->factortype = MAT_FACTOR_CHOLESKY; 
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN 
#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_elemental_petsc"
static PetscErrorCode MatGetFactor_elemental_petsc(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(B,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  B->factortype = ftype;
  *F            = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatNorm_Elemental"
static PetscErrorCode MatNorm_Elemental(Mat A,NormType type,PetscReal *nrm)
{
  Mat_Elemental *a=(Mat_Elemental*)A->data;  

  PetscFunctionBegin;
  switch (type){
  case NORM_1:
    *nrm = elem::Norm(*a->emat,elem::ONE_NORM);
    break;
  case NORM_FROBENIUS:
    *nrm = elem::Norm(*a->emat,elem::FROBENIUS_NORM);
    break;
  case NORM_INFINITY:
    *nrm = elem::Norm(*a->emat,elem::INFINITY_NORM);
    break;
  default:
    printf("Error: unsupported norm type!\n");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatZeroEntries_Elemental"
static PetscErrorCode MatZeroEntries_Elemental(Mat A)
{
  Mat_Elemental *a=(Mat_Elemental*)A->data;  

  PetscFunctionBegin;
  elem::Zero(*a->emat);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN 
#undef __FUNCT__
#define __FUNCT__ "MatGetOwnershipIS_Elemental"
static PetscErrorCode MatGetOwnershipIS_Elemental(Mat A,IS *rows,IS *cols)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,m,shift,stride,*idx;

  PetscFunctionBegin;
  if (rows) {
    m = a->emat->LocalHeight();
    shift = a->emat->ColShift();
    stride = a->emat->ColStride();
    ierr = PetscMalloc(m*sizeof(PetscInt),&idx);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      PetscInt rank,offset;
      E2RO(A,0,shift+i*stride,&rank,&offset);
      RO2P(A,0,rank,offset,&idx[i]);
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,m,idx,PETSC_OWN_POINTER,rows);CHKERRQ(ierr);
  }
  if (cols) {
    m = a->emat->LocalWidth();
    shift = a->emat->RowShift();
    stride = a->emat->RowStride();
    ierr = PetscMalloc(m*sizeof(PetscInt),&idx);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      PetscInt rank,offset;
      E2RO(A,1,shift+i*stride,&rank,&offset);
      RO2P(A,1,rank,offset,&idx[i]);
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,m,idx,PETSC_OWN_POINTER,cols);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Elemental"
static PetscErrorCode MatDestroy_Elemental(Mat A)
{
  Mat_Elemental      *a = (Mat_Elemental*)A->data;
  PetscErrorCode     ierr;
  Mat_Elemental_Grid *commgrid;
  PetscBool          flg;
  MPI_Comm           icomm;

  PetscFunctionBegin;
  delete a->interface;
  delete a->esubmat;
  delete a->emat;
  elem::mpi::Comm cxxcomm(((PetscObject)A)->comm);
  
  ierr = PetscCommDuplicate(cxxcomm,&icomm,PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Attr_get(icomm,Petsc_Elemental_keyval,(void**)&commgrid,(int*)&flg);CHKERRQ(ierr);
  if (--commgrid->grid_refct == 0) {
    delete commgrid->grid;
    ierr = PetscFree(commgrid);CHKERRQ(ierr);
  }
  ierr = PetscCommDestroy(&icomm);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatGetOwnershipIS_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatGetFactor_petsc_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetUp_Elemental"
PetscErrorCode MatSetUp_Elemental(Mat A)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  PetscErrorCode ierr;
  PetscMPIInt    rsize,csize;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  a->emat->ResizeTo(A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  elem::Zero(*a->emat);

  ierr = MPI_Comm_size(A->rmap->comm,&rsize);CHKERRQ(ierr);
  ierr = MPI_Comm_size(A->cmap->comm,&csize);CHKERRQ(ierr);
  if (csize != rsize) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_INCOMP,"Cannot use row and column communicators of different sizes");
  a->commsize = rsize;
  a->mr[0] = A->rmap->N % rsize; if (!a->mr[0]) a->mr[0] = rsize;
  a->mr[1] = A->cmap->N % csize; if (!a->mr[1]) a->mr[1] = csize;
  a->m[0] = A->rmap->N / rsize + (a->mr[0] != rsize);
  a->m[1] = A->cmap->N / csize + (a->mr[1] != csize);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_Elemental"
PetscErrorCode MatAssemblyBegin_Elemental(Mat A, MatAssemblyType type)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;

  PetscFunctionBegin;
  a->interface->Detach();
  a->interface->Attach(elem::LOCAL_TO_GLOBAL,*(a->emat));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_Elemental"
PetscErrorCode MatAssemblyEnd_Elemental(Mat A, MatAssemblyType type)
{
  PetscFunctionBegin;
  /* Currently does nothing */
  PetscFunctionReturn(0);
}

/*MC
   MATELEMENTAL = "elemental" - A matrix type for dense matrices using the Elemental package

   Options Database Keys:
. -mat_type elemental - sets the matrix type to "elemental" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MATDENSE,MatCreateElemental()
M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_Elemental"
PETSC_EXTERN_C PetscErrorCode MatCreate_Elemental(Mat A)
{
  Mat_Elemental      *a;
  PetscErrorCode     ierr;
  PetscBool          flg;
  Mat_Elemental_Grid *commgrid;
  MPI_Comm           icomm;

  PetscFunctionBegin;
  ierr = PetscElementalInitializePackage(PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscNewLog(A,Mat_Elemental,&a);CHKERRQ(ierr);
  A->data = (void*)a;

  A->ops->view            = MatView_Elemental;
  A->ops->destroy         = MatDestroy_Elemental;
  A->ops->setup           = MatSetUp_Elemental;
  A->ops->setvalues       = MatSetValues_Elemental;
  A->ops->mult            = MatMult_Elemental;
  A->ops->multadd         = MatMultAdd_Elemental;
  A->ops->matmult         = MatMatMult_Elemental;
  A->ops->matmultsymbolic = MatMatMultSymbolic_Elemental;
  A->ops->matmultnumeric  = MatMatMultNumeric_Elemental;
  A->ops->assemblybegin   = MatAssemblyBegin_Elemental;
  A->ops->assemblyend     = MatAssemblyEnd_Elemental;
  A->ops->scale           = MatScale_Elemental;
  A->ops->axpy            = MatAXPY_Elemental;
  A->ops->lufactor        = MatLUFactor_Elemental;
  A->ops->lufactorsymbolic = MatLUFactorSymbolic_Elemental;
  A->ops->lufactornumeric = MatLUFactorNumeric_Elemental;
  A->ops->matsolve        = MatMatSolve_Elemental;
  A->ops->copy            = MatCopy_Elemental;
  A->ops->transpose       = MatTranspose_Elemental;
  A->ops->norm            = MatNorm_Elemental;
  A->ops->solve           = MatSolve_Elemental;
  A->ops->zeroentries     = MatZeroEntries_Elemental;
  A->ops->choleskyfactor  = MatCholeskyFactor_Elemental;

  A->insertmode = NOT_SET_VALUES;

  /* Set up the elemental matrix */
  elem::mpi::Comm cxxcomm(((PetscObject)A)->comm); 

  /* Grid needs to be shared between multiple Mats on the same communicator, implement by attribute caching on the MPI_Comm */
  if (Petsc_Elemental_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Elemental_keyval,(void*)0); // MPI_Keyval_free()?
  }
  ierr = PetscCommDuplicate(cxxcomm,&icomm,PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Attr_get(icomm,Petsc_Elemental_keyval,(void**)&commgrid,(int*)&flg);CHKERRQ(ierr);
  if (!flg) { 
    ierr = PetscNewLog(A,Mat_Elemental_Grid,&commgrid);CHKERRQ(ierr);
    commgrid->grid       = new elem::Grid(cxxcomm);
    commgrid->grid_refct = 1;
    ierr = MPI_Attr_put(icomm,Petsc_Elemental_keyval,(void*)commgrid);CHKERRQ(ierr);
  } else {
    commgrid->grid_refct++;
  }
  ierr = PetscCommDestroy(&icomm);CHKERRQ(ierr);
  a->grid      = commgrid->grid;
  a->emat      = new elem::DistMatrix<PetscScalar>(*a->grid);
  a->esubmat   = new elem::Matrix<PetscScalar>(1,1);
  a->interface = new elem::AxpyInterface<PetscScalar>;
  a->pivot     = new elem::DistMatrix<PetscInt,elem::VC,elem::STAR>;
 
  /* build cache for off array entries formed */
  a->interface->Attach(elem::LOCAL_TO_GLOBAL,*(a->emat));

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatGetOwnershipIS_C","MatGetOwnershipIS_Elemental",MatGetOwnershipIS_Elemental);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatGetFactor_petsc_C","MatGetFactor_elemental_petsc",MatGetFactor_elemental_petsc);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATELEMENTAL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
