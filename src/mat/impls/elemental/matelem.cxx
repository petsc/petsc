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
      /* call elemental viewing function */
      ierr = PetscViewerASCIIPrintf(viewer,"Elemental run parameters:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  allocated entries=%d\n",(*a->emat).AllocatedMemory());CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  grid height=%d, grid width=%d\n",(*a->emat).Grid().Height(),(*a->emat).Grid().Width());CHKERRQ(ierr);
      if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
        /* call elemental viewing function */
        ierr = PetscPrintf(((PetscObject)viewer)->comm,"test matview_elemental 2\n");CHKERRQ(ierr);
      }
      
    } else if (format == PETSC_VIEWER_DEFAULT) {
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscObjectPrintClassNamePrefixType((PetscObject)A,viewer,"Matrix Object");CHKERRQ(ierr);
      a->emat->Print("Elemental matrix (cyclic ordering)");
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      if (A->factortype == MAT_FACTOR_NONE){
        Mat Adense;
        ierr = PetscPrintf(((PetscObject)viewer)->comm,"Elemental matrix (explicit ordering)\n");CHKERRQ(ierr);
        ierr = MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&Adense);CHKERRQ(ierr);
        ierr = MatView(Adense,viewer);CHKERRQ(ierr);
        ierr = MatDestroy(&Adense);CHKERRQ(ierr);     
      }
    } else SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Format");
  } else {
    /* convert to dense format and call MatView() */
    Mat Adense;
    ierr = PetscPrintf(((PetscObject)viewer)->comm,"Elemental matrix (explicit ordering)\n");CHKERRQ(ierr);
    ierr = MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&Adense);CHKERRQ(ierr);
    ierr = MatView(Adense,viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&Adense);CHKERRQ(ierr);     
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetInfo_Elemental"
static PetscErrorCode MatGetInfo_Elemental(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  MPI_Comm_rank(((PetscObject)A)->comm,&rank);
  
  /* if (!rank) printf("          .........MatGetInfo_Elemental ...\n"); */
  info->block_size     = 1.0; 

  if (flag == MAT_LOCAL) {
    info->nz_allocated   = (double)(*a->emat).AllocatedMemory(); /* locally allocated */
    info->nz_used        = info->nz_allocated; 
  } else if (flag == MAT_GLOBAL_MAX) {
    //ierr = MPI_Allreduce(isend,irecv,5,MPIU_REAL,MPIU_MAX,((PetscObject)matin)->comm);CHKERRQ(ierr);
    /* see MatGetInfo_MPIAIJ() for getting global info->nz_allocated! */
    //SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP," MAT_GLOBAL_MAX not written yet");
  } else if (flag == MAT_GLOBAL_SUM) {
    //SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP," MAT_GLOBAL_SUM not written yet");
    info->nz_allocated   = (double)(*a->emat).AllocatedMemory(); /* locally allocated */
    info->nz_used        = info->nz_allocated; /* assume Elemental does accurate allocation */
    //ierr = MPI_Allreduce(isend,irecv,1,MPIU_REAL,MPIU_SUM,((PetscObject)A)->comm);CHKERRQ(ierr);
    //PetscPrintf(PETSC_COMM_SELF,"    ... [%d] locally allocated %g\n",rank,info->nz_allocated);
  }
  
  info->nz_unneeded       = 0.0;
  info->assemblies        = (double)A->num_ass;
  info->mallocs           = 0;
  info->memory            = ((PetscObject)A)->mem;
  info->fill_ratio_given  = 0; /* determined by Elemental */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
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
        a->esubmat->Set(0,0, (PetscElemScalar)vals[i*nc+j]);
        a->interface->Axpy(1.0,*(a->esubmat),erow,ecol); 
        continue;
      }
      elrow = erow / grid.MCSize();
      elcol = ecol / grid.MRSize();
      switch (imode) {
      case INSERT_VALUES: a->emat->SetLocal(elrow,elcol,(PetscElemScalar)vals[i*nc+j]); break;
      case ADD_VALUES: a->emat->UpdateLocal(elrow,elcol,(PetscElemScalar)vals[i*nc+j]); break;
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
  Mat_Elemental         *a = (Mat_Elemental*)A->data;
  PetscErrorCode        ierr;
  const PetscElemScalar *x;
  PetscElemScalar       *y;
  PetscElemScalar       one = 1,zero = 0;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecGetArray(Y,(PetscScalar **)&y);CHKERRQ(ierr);
  { /* Scoping so that constructor is called before pointer is returned */
    elem::DistMatrix<PetscElemScalar,elem::VC,elem::STAR> xe(A->cmap->N,1,0,x,A->cmap->n,*a->grid);
    elem::DistMatrix<PetscElemScalar,elem::VC,elem::STAR> ye(A->rmap->N,1,0,y,A->rmap->n,*a->grid);
    elem::Gemv(elem::NORMAL,one,*a->emat,xe,zero,ye);
  }
  ierr = VecRestoreArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,(PetscScalar **)&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_Elemental"
static PetscErrorCode MatMultTranspose_Elemental(Mat A,Vec X,Vec Y)
{
  Mat_Elemental         *a = (Mat_Elemental*)A->data;
  PetscErrorCode        ierr;
  const PetscElemScalar *x;
  PetscElemScalar       *y;
  PetscElemScalar       one = 1,zero = 0;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecGetArray(Y,(PetscScalar **)&y);CHKERRQ(ierr);
  { /* Scoping so that constructor is called before pointer is returned */
    elem::DistMatrix<PetscElemScalar,elem::VC,elem::STAR> xe(A->rmap->N,1,0,x,A->rmap->n,*a->grid);
    elem::DistMatrix<PetscElemScalar,elem::VC,elem::STAR> ye(A->cmap->N,1,0,y,A->cmap->n,*a->grid);
    elem::Gemv(elem::TRANSPOSE,one,*a->emat,xe,zero,ye);
  }
  ierr = VecRestoreArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,(PetscScalar **)&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Elemental"
static PetscErrorCode MatMultAdd_Elemental(Mat A,Vec X,Vec Y,Vec Z)
{
  Mat_Elemental         *a = (Mat_Elemental*)A->data;
  PetscErrorCode        ierr;
  const PetscElemScalar *x;
  PetscElemScalar       *z;
  PetscElemScalar       one = 1;

  PetscFunctionBegin;
  if (Y != Z) {ierr = VecCopy(Y,Z);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecGetArray(Z,(PetscScalar **)&z);CHKERRQ(ierr);
  { /* Scoping so that constructor is called before pointer is returned */
    elem::DistMatrix<PetscElemScalar,elem::VC,elem::STAR> xe(A->cmap->N,1,0,x,A->cmap->n,*a->grid);
    elem::DistMatrix<PetscElemScalar,elem::VC,elem::STAR> ze(A->rmap->N,1,0,z,A->rmap->n,*a->grid);
    elem::Gemv(elem::NORMAL,one,*a->emat,xe,one,ze);
  }
  ierr = VecRestoreArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Z,(PetscScalar **)&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_Elemental"
static PetscErrorCode MatMultTransposeAdd_Elemental(Mat A,Vec X,Vec Y,Vec Z)
{
  Mat_Elemental         *a = (Mat_Elemental*)A->data;
  PetscErrorCode        ierr;
  const PetscElemScalar *x;
  PetscElemScalar       *z;
  PetscElemScalar       one = 1;

  PetscFunctionBegin;
  if (Y != Z) {ierr = VecCopy(Y,Z);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecGetArray(Z,(PetscScalar **)&z);CHKERRQ(ierr);
  { /* Scoping so that constructor is called before pointer is returned */
    elem::DistMatrix<PetscElemScalar,elem::VC,elem::STAR> xe(A->rmap->N,1,0,x,A->rmap->n,*a->grid);
    elem::DistMatrix<PetscElemScalar,elem::VC,elem::STAR> ze(A->cmap->N,1,0,z,A->cmap->n,*a->grid);
    elem::Gemv(elem::TRANSPOSE,one,*a->emat,xe,one,ze);
  }
  ierr = VecRestoreArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Z,(PetscScalar **)&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultNumeric_Elemental"
static PetscErrorCode MatMatMultNumeric_Elemental(Mat A,Mat B,Mat C)
{
  Mat_Elemental    *a = (Mat_Elemental*)A->data;
  Mat_Elemental    *b = (Mat_Elemental*)B->data;
  Mat_Elemental    *c = (Mat_Elemental*)C->data;
  PetscElemScalar  one = 1,zero = 0;

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
#define __FUNCT__ "MatMatTransposeMultNumeric_Elemental"
static PetscErrorCode MatMatTransposeMultNumeric_Elemental(Mat A,Mat B,Mat C)
{
  Mat_Elemental      *a = (Mat_Elemental*)A->data;
  Mat_Elemental      *b = (Mat_Elemental*)B->data;
  Mat_Elemental      *c = (Mat_Elemental*)C->data;
  PetscElemScalar    one = 1,zero = 0;

  PetscFunctionBegin;
  { /* Scoping so that constructor is called before pointer is returned */
    elem::Gemm(elem::NORMAL,elem::TRANSPOSE,one,*a->emat,*b->emat,zero,*c->emat);
  }
  C->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatTransposeMultSymbolic_Elemental"
static PetscErrorCode MatMatTransposeMultSymbolic_Elemental(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  Mat            Ce;
  MPI_Comm       comm=((PetscObject)A)->comm;

  PetscFunctionBegin;
  ierr = MatCreate(comm,&Ce);CHKERRQ(ierr);
  ierr = MatSetSizes(Ce,A->rmap->n,B->rmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Ce,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetUp(Ce);CHKERRQ(ierr);
  *C = Ce;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatTransposeMult_Elemental"
static PetscErrorCode MatMatTransposeMult_Elemental(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = PetscLogEventBegin(MAT_MatTransposeMultSymbolic,A,B,0,0);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_Elemental(A,B,1.0,C);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_MatTransposeMultSymbolic,A,B,0,0);CHKERRQ(ierr);   
  }
  ierr = PetscLogEventBegin(MAT_MatTransposeMultNumeric,A,B,0,0);CHKERRQ(ierr); 
  ierr = MatMatTransposeMultNumeric_Elemental(A,B,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatTransposeMultNumeric,A,B,0,0);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Elemental"
static PetscErrorCode MatGetDiagonal_Elemental(Mat A,Vec D)
{
  PetscInt        i,nrows,ncols,nD,rrank,ridx,crank,cidx;
  Mat_Elemental   *a = (Mat_Elemental*)A->data;
  PetscErrorCode  ierr;
  PetscElemScalar v;
  MPI_Comm        comm=((PetscObject)A)->comm;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&nrows,&ncols);CHKERRQ(ierr);
  nD = nrows>ncols ? ncols : nrows;
  for (i=0; i<nD; i++) {
    PetscInt erow,ecol;
    P2RO(A,0,i,&rrank,&ridx);
    RO2E(A,0,rrank,ridx,&erow);
    if (rrank < 0 || ridx < 0 || erow < 0) SETERRQ(comm,PETSC_ERR_PLIB,"Incorrect row translation");
    P2RO(A,1,i,&crank,&cidx);
    RO2E(A,1,crank,cidx,&ecol);
    if (crank < 0 || cidx < 0 || ecol < 0) SETERRQ(comm,PETSC_ERR_PLIB,"Incorrect col translation");
    v = a->emat->Get(erow,ecol);
    ierr = VecSetValue(D,i,(PetscScalar)v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(D);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDiagonalScale_Elemental"
static PetscErrorCode MatDiagonalScale_Elemental(Mat X,Vec L,Vec R)
{
  Mat_Elemental         *x = (Mat_Elemental*)X->data;
  const PetscElemScalar *d;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (L == PETSC_NULL) {
    ierr = VecGetArrayRead(R,(const PetscScalar **)&d);CHKERRQ(ierr);
    elem::DistMatrix<PetscElemScalar,elem::VC,elem::STAR> de(X->cmap->N,1,0,d,X->cmap->n,*x->grid);
    elem::DiagonalScale(elem::RIGHT,elem::NORMAL,de,*x->emat);
    ierr = VecRestoreArrayRead(R,(const PetscScalar **)&d);CHKERRQ(ierr);
  } else {
    ierr = VecGetArrayRead(L,(const PetscScalar **)&d);CHKERRQ(ierr);
    elem::DistMatrix<PetscElemScalar,elem::VC,elem::STAR> de(X->rmap->N,1,0,d,X->rmap->n,*x->grid);
    elem::DiagonalScale(elem::LEFT,elem::NORMAL,de,*x->emat);
    ierr = VecRestoreArrayRead(L,(const PetscScalar **)&d);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatScale_Elemental"
static PetscErrorCode MatScale_Elemental(Mat X,PetscScalar a)
{
  Mat_Elemental  *x = (Mat_Elemental*)X->data;

  PetscFunctionBegin;
  elem::Scal((PetscElemScalar)a,*x->emat);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAXPY_Elemental"
static PetscErrorCode MatAXPY_Elemental(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_Elemental  *x = (Mat_Elemental*)X->data;
  Mat_Elemental  *y = (Mat_Elemental*)Y->data;

  PetscFunctionBegin;
  elem::Axpy((PetscElemScalar)a,*x->emat,*y->emat);
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
#define __FUNCT__ "MatDuplicate_Elemental"
static PetscErrorCode MatDuplicate_Elemental(Mat A,MatDuplicateOption op,Mat *B)
{
  Mat            Be;
  MPI_Comm       comm=((PetscObject)A)->comm;
  Mat_Elemental  *a=(Mat_Elemental*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,&Be);CHKERRQ(ierr);
  ierr = MatSetSizes(Be,A->rmap->n,A->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Be,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetUp(Be);CHKERRQ(ierr);
  *B = Be;
  if (op == MAT_COPY_VALUES) {
    Mat_Elemental *b=(Mat_Elemental*)Be->data;
    elem::Copy(*a->emat,*b->emat);
  }
  Be->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTranspose_Elemental"
static PetscErrorCode MatTranspose_Elemental(Mat A,MatReuse reuse,Mat *B)
{
  Mat            Be;
  PetscErrorCode ierr;
  MPI_Comm       comm=((PetscObject)A)->comm;
  Mat_Elemental  *a = (Mat_Elemental*)A->data, *b;

  PetscFunctionBegin;
  /* Only out-of-place supported */
  if (reuse == MAT_INITIAL_MATRIX){
    ierr = MatCreate(comm,&Be);CHKERRQ(ierr);
    ierr = MatSetSizes(Be,A->cmap->n,A->rmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetType(Be,MATELEMENTAL);CHKERRQ(ierr);
    ierr = MatSetUp(Be);CHKERRQ(ierr);
    *B = Be;
  }
  b = (Mat_Elemental*)Be->data;
  elem::Transpose(*a->emat,*b->emat);
  Be->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConjugate_Elemental"
static PetscErrorCode MatConjugate_Elemental(Mat A)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;

  PetscFunctionBegin;
  elem::Conjugate(*a->emat);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatHermitianTranspose_Elemental"
static PetscErrorCode MatHermitianTranspose_Elemental(Mat A,MatReuse reuse,Mat *B)
{
  Mat            Be;
  PetscErrorCode ierr;
  MPI_Comm       comm=((PetscObject)A)->comm;
  Mat_Elemental  *a = (Mat_Elemental*)A->data, *b;

  PetscFunctionBegin;
  /* Only out-of-place supported */
  if (reuse == MAT_INITIAL_MATRIX){
    ierr = MatCreate(comm,&Be);CHKERRQ(ierr);
    ierr = MatSetSizes(Be,A->cmap->n,A->rmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetType(Be,MATELEMENTAL);CHKERRQ(ierr);
    ierr = MatSetUp(Be);CHKERRQ(ierr);
    *B = Be;
  }
  b = (Mat_Elemental*)Be->data;
  elem::Adjoint(*a->emat,*b->emat);
  Be->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_Elemental"
static PetscErrorCode MatSolve_Elemental(Mat A,Vec B,Vec X)
{
  Mat_Elemental     *a = (Mat_Elemental*)A->data;
  PetscErrorCode    ierr;
  PetscElemScalar   *x;

  PetscFunctionBegin;
  ierr = VecCopy(B,X);CHKERRQ(ierr);
  ierr = VecGetArray(X,(PetscScalar **)&x);CHKERRQ(ierr);
  elem::DistMatrix<PetscElemScalar,elem::VC,elem::STAR> xe(A->rmap->N,1,0,x,A->rmap->n,*a->grid);
  elem::DistMatrix<PetscElemScalar,elem::MC,elem::MR> xer = xe;
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
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unfactored Matrix or Unsupported MatFactorType"); 
    break;
  }
  ierr = VecRestoreArray(X,(PetscScalar **)&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolveAdd_Elemental"
static PetscErrorCode MatSolveAdd_Elemental(Mat A,Vec B,Vec Y,Vec X)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatSolve_Elemental(A,B,X);CHKERRQ(ierr);
  ierr = VecAXPY(X,1,Y);CHKERRQ(ierr);
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
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unfactored Matrix or Unsupported MatFactorType"); 
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
  A->assembled  = PETSC_TRUE;
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
  elem::DistMatrix<PetscElemScalar,elem::MC,elem::STAR> d;

  PetscFunctionBegin;
  elem::Cholesky(elem::UPPER,*a->emat);
  A->factortype = MAT_FACTOR_CHOLESKY; 
  A->assembled  = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorNumeric_Elemental"
static PetscErrorCode MatCholeskyFactorNumeric_Elemental(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCopy(A,F,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatCholeskyFactor_Elemental(F,0,info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorSymbolic_Elemental"
static PetscErrorCode MatCholeskyFactorSymbolic_Elemental(Mat F,Mat A,IS perm,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  /* F is create and allocated by MatGetFactor_elemental_petsc(), skip this routine. */
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "MatFactorGetSolverPackage_elemental_elemental"
PetscErrorCode MatFactorGetSolverPackage_elemental_elemental(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERELEMENTAL;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN 
#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_elemental_elemental"
static PetscErrorCode MatGetFactor_elemental_elemental(Mat A,MatFactorType ftype,Mat *F)
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
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_elemental_elemental",MatFactorGetSolverPackage_elemental_elemental);CHKERRQ(ierr);
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
#define __FUNCT__ "MatConvert_Elemental_Dense"
static PetscErrorCode MatConvert_Elemental_Dense(Mat A,MatType newtype,MatReuse reuse,Mat *B)
{
  Mat                Bmpi;
  Mat_Elemental      *a = (Mat_Elemental*)A->data;
  MPI_Comm           comm=((PetscObject)A)->comm;
  PetscErrorCode     ierr;
  PetscInt           rrank,ridx,crank,cidx,nrows,ncols,i,j;
  PetscElemScalar    v;

  PetscFunctionBegin;
  if (strcmp(newtype,MATDENSE) && strcmp(newtype,MATSEQDENSE) && strcmp(newtype,MATMPIDENSE)) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported New MatType: must be MATDENSE, MATSEQDENSE or MATMPIDENSE");
  }
  ierr = MatCreate(comm,&Bmpi);CHKERRQ(ierr);
  ierr = MatSetSizes(Bmpi,A->rmap->n,A->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Bmpi,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(Bmpi);CHKERRQ(ierr);
  ierr = MatGetSize(A,&nrows,&ncols);CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    PetscInt erow,ecol;
    P2RO(A,0,i,&rrank,&ridx);
    RO2E(A,0,rrank,ridx,&erow);
    if (rrank < 0 || ridx < 0 || erow < 0) SETERRQ(comm,PETSC_ERR_PLIB,"Incorrect row translation");
    for (j=0; j<ncols; j++) {
      P2RO(A,1,j,&crank,&cidx);
      RO2E(A,1,crank,cidx,&ecol);
      if (crank < 0 || cidx < 0 || ecol < 0) SETERRQ(comm,PETSC_ERR_PLIB,"Incorrect col translation");
      v = a->emat->Get(erow,ecol);
      ierr = MatSetValues(Bmpi,1,&i,1,&j,(PetscScalar *)&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Bmpi,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Bmpi,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,Bmpi);CHKERRQ(ierr);
  } else {
    *B = Bmpi;
  }
  PetscFunctionReturn(0);
}

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
  a->interface->Detach();
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
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatFactorGetSolverPackage_C","",PETSC_NULL);CHKERRQ(ierr);
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
  a->m[0]  = A->rmap->N / rsize + (a->mr[0] != rsize);
  a->m[1]  = A->cmap->N / csize + (a->mr[1] != csize);
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

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {
       MatSetValues_Elemental,
       0,
       0,
       MatMult_Elemental,
/* 4*/ MatMultAdd_Elemental,
       MatMultTranspose_Elemental,
       MatMultTransposeAdd_Elemental,
       MatSolve_Elemental,
       MatSolveAdd_Elemental,
       0, //MatSolveTranspose_Elemental,
/*10*/ 0, //MatSolveTransposeAdd_Elemental,
       MatLUFactor_Elemental,
       MatCholeskyFactor_Elemental,
       0,
       MatTranspose_Elemental,
/*15*/ MatGetInfo_Elemental,
       0, 
       MatGetDiagonal_Elemental,
       MatDiagonalScale_Elemental,
       MatNorm_Elemental,
/*20*/ MatAssemblyBegin_Elemental,
       MatAssemblyEnd_Elemental,
       0, //MatSetOption_Elemental,
       MatZeroEntries_Elemental,
/*24*/ 0,
       MatLUFactorSymbolic_Elemental,
       MatLUFactorNumeric_Elemental,
       MatCholeskyFactorSymbolic_Elemental,
       MatCholeskyFactorNumeric_Elemental,
/*29*/ MatSetUp_Elemental,
       0,
       0,
       0,
       0,
/*34*/ MatDuplicate_Elemental,
       0,
       0,
       0,
       0,
/*39*/ MatAXPY_Elemental,
       0,
       0,
       0,
       MatCopy_Elemental,
/*44*/ 0,
       MatScale_Elemental,
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
/*59*/ 0,
       MatDestroy_Elemental,
       MatView_Elemental,
       0, 
       0,
/*64*/ 0,
       0,
       0,
       0,
       0,
/*69*/ 0,
       0,
       MatConvert_Elemental_Dense,
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
       0,
/*84*/ 0,
       0,
       0,
       0,
       0,
/*89*/ MatMatMult_Elemental,  
       MatMatMultSymbolic_Elemental,  
       MatMatMultNumeric_Elemental,   
       0,
       0,
/*94*/ 0,
       MatMatTransposeMult_Elemental,  
       MatMatTransposeMultSymbolic_Elemental,  
       MatMatTransposeMultNumeric_Elemental,  
       0,
/*99*/ 0,
       0,
       0,
       MatConjugate_Elemental,
       0,
/*104*/0,
       0,
       0,
       0,
       0,
/*109*/MatMatSolve_Elemental,
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
       MatHermitianTranspose_Elemental,
       0,
       0,
       0,
/*124*/0,
       0,
       0,
       0,
       0,
/*129*/0,
       0,
       0,
       0,
       0,
/*134*/0,
       0,
       0,
       0,
       0
};

/*MC
   MATELEMENTAL = "elemental" - A matrix type for dense matrices using the Elemental package

   Options Database Keys:
. -mat_type elemental - sets the matrix type to "elemental" during a call to MatSetFromOptions()
. -mat_elemental_grid_height - sets Grid Height 
. -mat_elemental_grid_width - sets Grid Width

  Level: beginner

.seealso: MATDENSE
M*/

#undef __FUNCT__
#define __FUNCT__ "MatCreate_Elemental"
PETSC_EXTERN_C PetscErrorCode MatCreate_Elemental(Mat A)
{
  Mat_Elemental      *a;
  PetscErrorCode     ierr;
  PetscBool          flg,flg1,flg2;
  Mat_Elemental_Grid *commgrid;
  MPI_Comm           icomm;
  PetscInt           optv1,optv2;

  PetscFunctionBegin;
  ierr = PetscElementalInitializePackage(PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  A->insertmode = NOT_SET_VALUES;

  ierr = PetscNewLog(A,Mat_Elemental,&a);CHKERRQ(ierr);
  A->data = (void*)a;
  
  /* Set up the elemental matrix */
  elem::mpi::Comm cxxcomm(((PetscObject)A)->comm); 
 
  /* Grid needs to be shared between multiple Mats on the same communicator, implement by attribute caching on the MPI_Comm */
  if (Petsc_Elemental_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Elemental_keyval,(void*)0); 
  }
  ierr = PetscCommDuplicate(cxxcomm,&icomm,PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Attr_get(icomm,Petsc_Elemental_keyval,(void**)&commgrid,(int*)&flg);CHKERRQ(ierr);
  if (!flg) { 
    ierr = PetscNewLog(A,Mat_Elemental_Grid,&commgrid);CHKERRQ(ierr);

    ierr = PetscOptionsBegin(((PetscObject)A)->comm,((PetscObject)A)->prefix,"Elemental Options","Mat");CHKERRQ(ierr);
    /* displayed default grid sizes (CommSize,1) are set by us arbitrarily until elem::Grid() is called */
    ierr = PetscOptionsInt("-mat_elemental_grid_height","Grid Height","None",elem::mpi::CommSize(cxxcomm),&optv1,&flg1);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_elemental_grid_width","Grid Width","None",1,&optv2,&flg2);CHKERRQ(ierr);
    if (flg1 || flg2) {
      if (optv1*optv2 != elem::mpi::CommSize(cxxcomm)) {
        SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_INCOMP,"Grid Height times Grid Width must equal CommSize");
      }
      commgrid->grid = new elem::Grid(cxxcomm,optv1,optv2); /* use user-provided grid sizes */
    } else {
      commgrid->grid = new elem::Grid(cxxcomm); /* use Elemental default grid sizes */
    }
    commgrid->grid_refct = 1;
    ierr = MPI_Attr_put(icomm,Petsc_Elemental_keyval,(void*)commgrid);CHKERRQ(ierr);
    PetscOptionsEnd();
  } else {
    commgrid->grid_refct++;
  }
  ierr = PetscCommDestroy(&icomm);CHKERRQ(ierr);
  a->grid      = commgrid->grid;
  a->emat      = new elem::DistMatrix<PetscElemScalar>(*a->grid);
  a->esubmat   = new elem::Matrix<PetscElemScalar>(1,1);
  a->interface = new elem::AxpyInterface<PetscElemScalar>;
  a->pivot     = new elem::DistMatrix<PetscInt,elem::VC,elem::STAR>;
 
  /* build cache for off array entries formed */
  a->interface->Attach(elem::LOCAL_TO_GLOBAL,*(a->emat));

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatGetOwnershipIS_C","MatGetOwnershipIS_Elemental",MatGetOwnershipIS_Elemental);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatGetFactor_elemental_C","MatGetFactor_elemental_elemental",MatGetFactor_elemental_elemental);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATELEMENTAL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

