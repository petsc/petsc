#include <petsc/private/petscelemental.h>

const char ElementalCitation[] = "@Article{Elemental2012,\n"
"  author  = {Jack Poulson and Bryan Marker and Jeff R. Hammond and Nichols A. Romero and Robert {v}an~{d}e~{G}eijn},\n"
"  title   = {Elemental: A New Framework for Distributed Memory Dense Matrix Computations},\n"
"  journal = {{ACM} Transactions on Mathematical Software},\n"
"  volume  = {39},\n"
"  number  = {2},\n"
"  year    = {2013}\n"
"}\n";
static PetscBool ElementalCite = PETSC_FALSE;

/*
    The variable Petsc_Elemental_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Mat_Elemental_Grid
*/
static PetscMPIInt Petsc_Elemental_keyval = MPI_KEYVAL_INVALID;

static PetscErrorCode MatView_Elemental(Mat A,PetscViewer viewer)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscViewerFormat format;
    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_INFO) {
      /* call elemental viewing function */
      PetscCall(PetscViewerASCIIPrintf(viewer,"Elemental run parameters:\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  allocated entries=%zu\n",(*a->emat).AllocatedMemory()));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  grid height=%d, grid width=%d\n",(*a->emat).Grid().Height(),(*a->emat).Grid().Width()));
      if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
        /* call elemental viewing function */
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)viewer),"test matview_elemental 2\n"));
      }

    } else if (format == PETSC_VIEWER_DEFAULT) {
      PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
      El::Print( *a->emat, "Elemental matrix (cyclic ordering)");
      PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
      if (A->factortype == MAT_FACTOR_NONE) {
        Mat Adense;
        PetscCall(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&Adense));
        PetscCall(MatView(Adense,viewer));
        PetscCall(MatDestroy(&Adense));
      }
    } else SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Format");
  } else {
    /* convert to dense format and call MatView() */
    Mat Adense;
    PetscCall(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&Adense));
    PetscCall(MatView(Adense,viewer));
    PetscCall(MatDestroy(&Adense));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetInfo_Elemental(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;

  PetscFunctionBegin;
  info->block_size = 1.0;

  if (flag == MAT_LOCAL) {
    info->nz_allocated   = (*a->emat).AllocatedMemory(); /* locally allocated */
    info->nz_used        = info->nz_allocated;
  } else if (flag == MAT_GLOBAL_MAX) {
    //PetscCall(MPIU_Allreduce(isend,irecv,5,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)matin)));
    /* see MatGetInfo_MPIAIJ() for getting global info->nz_allocated! */
    //SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP," MAT_GLOBAL_MAX not written yet");
  } else if (flag == MAT_GLOBAL_SUM) {
    //SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP," MAT_GLOBAL_SUM not written yet");
    info->nz_allocated   = (*a->emat).AllocatedMemory(); /* locally allocated */
    info->nz_used        = info->nz_allocated; /* assume Elemental does accurate allocation */
    //PetscCall(MPIU_Allreduce(isend,irecv,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)A)));
    //PetscPrintf(PETSC_COMM_SELF,"    ... [%d] locally allocated %g\n",rank,info->nz_allocated);
  }

  info->nz_unneeded       = 0.0;
  info->assemblies        = A->num_ass;
  info->mallocs           = 0;
  info->memory            = ((PetscObject)A)->mem;
  info->fill_ratio_given  = 0; /* determined by Elemental */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_Elemental(Mat A,MatOption op,PetscBool flg)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;

  PetscFunctionBegin;
  switch (op) {
  case MAT_NEW_NONZERO_LOCATIONS:
  case MAT_NEW_NONZERO_LOCATION_ERR:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
  case MAT_SYMMETRIC:
  case MAT_SORTED_FULL:
  case MAT_HERMITIAN:
    break;
  case MAT_ROW_ORIENTED:
    a->roworiented = flg;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %s",MatOptions[op]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValues_Elemental(Mat A,PetscInt nr,const PetscInt *rows,PetscInt nc,const PetscInt *cols,const PetscScalar *vals,InsertMode imode)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  PetscInt       i,j,rrank,ridx,crank,cidx,erow,ecol,numQueues=0;

  PetscFunctionBegin;
  // TODO: Initialize matrix to all zeros?

  // Count the number of queues from this process
  if (a->roworiented) {
    for (i=0; i<nr; i++) {
      if (rows[i] < 0) continue;
      P2RO(A,0,rows[i],&rrank,&ridx);
      RO2E(A,0,rrank,ridx,&erow);
      PetscCheckFalse(rrank < 0 || ridx < 0 || erow < 0,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Incorrect row translation");
      for (j=0; j<nc; j++) {
        if (cols[j] < 0) continue;
        P2RO(A,1,cols[j],&crank,&cidx);
        RO2E(A,1,crank,cidx,&ecol);
        PetscCheckFalse(crank < 0 || cidx < 0 || ecol < 0,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Incorrect col translation");
        if (!a->emat->IsLocal(erow,ecol)) { /* off-proc entry */
          /* printf("Will later remotely update (%d,%d)\n",erow,ecol); */
          PetscCheck(imode == ADD_VALUES,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only ADD_VALUES to off-processor entry is supported");
          ++numQueues;
          continue;
        }
        /* printf("Locally updating (%d,%d)\n",erow,ecol); */
        switch (imode) {
        case INSERT_VALUES: a->emat->Set(erow,ecol,(PetscElemScalar)vals[i*nc+j]); break;
        case ADD_VALUES: a->emat->Update(erow,ecol,(PetscElemScalar)vals[i*nc+j]); break;
        default: SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support for InsertMode %d",(int)imode);
        }
      }
    }

    /* printf("numQueues=%d\n",numQueues); */
    a->emat->Reserve( numQueues);
    for (i=0; i<nr; i++) {
      if (rows[i] < 0) continue;
      P2RO(A,0,rows[i],&rrank,&ridx);
      RO2E(A,0,rrank,ridx,&erow);
      for (j=0; j<nc; j++) {
        if (cols[j] < 0) continue;
        P2RO(A,1,cols[j],&crank,&cidx);
        RO2E(A,1,crank,cidx,&ecol);
        if (!a->emat->IsLocal(erow,ecol)) { /*off-proc entry*/
          /* printf("Queueing remotely update of (%d,%d)\n",erow,ecol); */
          a->emat->QueueUpdate( erow, ecol, vals[i*nc+j]);
        }
      }
    }
  } else { /* columnoriented */
    for (j=0; j<nc; j++) {
      if (cols[j] < 0) continue;
      P2RO(A,1,cols[j],&crank,&cidx);
      RO2E(A,1,crank,cidx,&ecol);
      PetscCheckFalse(crank < 0 || cidx < 0 || ecol < 0,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Incorrect col translation");
      for (i=0; i<nr; i++) {
        if (rows[i] < 0) continue;
        P2RO(A,0,rows[i],&rrank,&ridx);
        RO2E(A,0,rrank,ridx,&erow);
        PetscCheckFalse(rrank < 0 || ridx < 0 || erow < 0,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Incorrect row translation");
        if (!a->emat->IsLocal(erow,ecol)) { /* off-proc entry */
          /* printf("Will later remotely update (%d,%d)\n",erow,ecol); */
          PetscCheck(imode == ADD_VALUES,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only ADD_VALUES to off-processor entry is supported");
          ++numQueues;
          continue;
        }
        /* printf("Locally updating (%d,%d)\n",erow,ecol); */
        switch (imode) {
        case INSERT_VALUES: a->emat->Set(erow,ecol,(PetscElemScalar)vals[i+j*nr]); break;
        case ADD_VALUES: a->emat->Update(erow,ecol,(PetscElemScalar)vals[i+j*nr]); break;
        default: SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support for InsertMode %d",(int)imode);
        }
      }
    }

    /* printf("numQueues=%d\n",numQueues); */
    a->emat->Reserve( numQueues);
    for (j=0; j<nc; j++) {
      if (cols[j] < 0) continue;
      P2RO(A,1,cols[j],&crank,&cidx);
      RO2E(A,1,crank,cidx,&ecol);

      for (i=0; i<nr; i++) {
        if (rows[i] < 0) continue;
        P2RO(A,0,rows[i],&rrank,&ridx);
        RO2E(A,0,rrank,ridx,&erow);
        if (!a->emat->IsLocal(erow,ecol)) { /*off-proc entry*/
          /* printf("Queueing remotely update of (%d,%d)\n",erow,ecol); */
          a->emat->QueueUpdate( erow, ecol, vals[i+j*nr]);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_Elemental(Mat A,Vec X,Vec Y)
{
  Mat_Elemental         *a = (Mat_Elemental*)A->data;
  const PetscElemScalar *x;
  PetscElemScalar       *y;
  PetscElemScalar       one = 1,zero = 0;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(X,(const PetscScalar **)&x));
  PetscCall(VecGetArray(Y,(PetscScalar **)&y));
  { /* Scoping so that constructor is called before pointer is returned */
    El::DistMatrix<PetscElemScalar,El::VC,El::STAR> xe, ye;
    xe.LockedAttach(A->cmap->N,1,*a->grid,0,0,x,A->cmap->n);
    ye.Attach(A->rmap->N,1,*a->grid,0,0,y,A->rmap->n);
    El::Gemv(El::NORMAL,one,*a->emat,xe,zero,ye);
  }
  PetscCall(VecRestoreArrayRead(X,(const PetscScalar **)&x));
  PetscCall(VecRestoreArray(Y,(PetscScalar **)&y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_Elemental(Mat A,Vec X,Vec Y)
{
  Mat_Elemental         *a = (Mat_Elemental*)A->data;
  const PetscElemScalar *x;
  PetscElemScalar       *y;
  PetscElemScalar       one = 1,zero = 0;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(X,(const PetscScalar **)&x));
  PetscCall(VecGetArray(Y,(PetscScalar **)&y));
  { /* Scoping so that constructor is called before pointer is returned */
    El::DistMatrix<PetscElemScalar,El::VC,El::STAR> xe, ye;
    xe.LockedAttach(A->rmap->N,1,*a->grid,0,0,x,A->rmap->n);
    ye.Attach(A->cmap->N,1,*a->grid,0,0,y,A->cmap->n);
    El::Gemv(El::TRANSPOSE,one,*a->emat,xe,zero,ye);
  }
  PetscCall(VecRestoreArrayRead(X,(const PetscScalar **)&x));
  PetscCall(VecRestoreArray(Y,(PetscScalar **)&y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_Elemental(Mat A,Vec X,Vec Y,Vec Z)
{
  Mat_Elemental         *a = (Mat_Elemental*)A->data;
  const PetscElemScalar *x;
  PetscElemScalar       *z;
  PetscElemScalar       one = 1;

  PetscFunctionBegin;
  if (Y != Z) PetscCall(VecCopy(Y,Z));
  PetscCall(VecGetArrayRead(X,(const PetscScalar **)&x));
  PetscCall(VecGetArray(Z,(PetscScalar **)&z));
  { /* Scoping so that constructor is called before pointer is returned */
    El::DistMatrix<PetscElemScalar,El::VC,El::STAR> xe, ze;
    xe.LockedAttach(A->cmap->N,1,*a->grid,0,0,x,A->cmap->n);
    ze.Attach(A->rmap->N,1,*a->grid,0,0,z,A->rmap->n);
    El::Gemv(El::NORMAL,one,*a->emat,xe,one,ze);
  }
  PetscCall(VecRestoreArrayRead(X,(const PetscScalar **)&x));
  PetscCall(VecRestoreArray(Z,(PetscScalar **)&z));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_Elemental(Mat A,Vec X,Vec Y,Vec Z)
{
  Mat_Elemental         *a = (Mat_Elemental*)A->data;
  const PetscElemScalar *x;
  PetscElemScalar       *z;
  PetscElemScalar       one = 1;

  PetscFunctionBegin;
  if (Y != Z) PetscCall(VecCopy(Y,Z));
  PetscCall(VecGetArrayRead(X,(const PetscScalar **)&x));
  PetscCall(VecGetArray(Z,(PetscScalar **)&z));
  { /* Scoping so that constructor is called before pointer is returned */
    El::DistMatrix<PetscElemScalar,El::VC,El::STAR> xe, ze;
    xe.LockedAttach(A->rmap->N,1,*a->grid,0,0,x,A->rmap->n);
    ze.Attach(A->cmap->N,1,*a->grid,0,0,z,A->cmap->n);
    El::Gemv(El::TRANSPOSE,one,*a->emat,xe,one,ze);
  }
  PetscCall(VecRestoreArrayRead(X,(const PetscScalar **)&x));
  PetscCall(VecRestoreArray(Z,(PetscScalar **)&z));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_Elemental(Mat A,Mat B,Mat C)
{
  Mat_Elemental    *a = (Mat_Elemental*)A->data;
  Mat_Elemental    *b = (Mat_Elemental*)B->data;
  Mat_Elemental    *c = (Mat_Elemental*)C->data;
  PetscElemScalar  one = 1,zero = 0;

  PetscFunctionBegin;
  { /* Scoping so that constructor is called before pointer is returned */
    El::Gemm(El::NORMAL,El::NORMAL,one,*a->emat,*b->emat,zero,*c->emat);
  }
  C->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_Elemental(Mat A,Mat B,PetscReal fill,Mat Ce)
{
  PetscFunctionBegin;
  PetscCall(MatSetSizes(Ce,A->rmap->n,B->cmap->n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(Ce,MATELEMENTAL));
  PetscCall(MatSetUp(Ce));
  Ce->ops->matmultnumeric = MatMatMultNumeric_Elemental;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatTransposeMultNumeric_Elemental(Mat A,Mat B,Mat C)
{
  Mat_Elemental      *a = (Mat_Elemental*)A->data;
  Mat_Elemental      *b = (Mat_Elemental*)B->data;
  Mat_Elemental      *c = (Mat_Elemental*)C->data;
  PetscElemScalar    one = 1,zero = 0;

  PetscFunctionBegin;
  { /* Scoping so that constructor is called before pointer is returned */
    El::Gemm(El::NORMAL,El::TRANSPOSE,one,*a->emat,*b->emat,zero,*c->emat);
  }
  C->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatTransposeMultSymbolic_Elemental(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatSetSizes(C,A->rmap->n,B->rmap->n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(C,MATELEMENTAL));
  PetscCall(MatSetUp(C));
  PetscFunctionReturn(0);
}

/* --------------------------------------- */
static PetscErrorCode MatProductSetFromOptions_Elemental_AB(Mat C)
{
  PetscFunctionBegin;
  C->ops->matmultsymbolic = MatMatMultSymbolic_Elemental;
  C->ops->productsymbolic = MatProductSymbolic_AB;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_Elemental_ABt(Mat C)
{
  PetscFunctionBegin;
  C->ops->mattransposemultsymbolic = MatMatTransposeMultSymbolic_Elemental;
  C->ops->productsymbolic          = MatProductSymbolic_ABt;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_Elemental(Mat C)
{
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
    PetscCall(MatProductSetFromOptions_Elemental_AB(C));
    break;
  case MATPRODUCT_ABt:
    PetscCall(MatProductSetFromOptions_Elemental_ABt(C));
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_Elemental_MPIDense(Mat A,Mat B,Mat C)
{
  Mat            Be,Ce;

  PetscFunctionBegin;
  PetscCall(MatConvert(B,MATELEMENTAL,MAT_INITIAL_MATRIX,&Be));
  PetscCall(MatMatMult(A,Be,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Ce));
  PetscCall(MatConvert(Ce,MATMPIDENSE,MAT_REUSE_MATRIX,&C));
  PetscCall(MatDestroy(&Be));
  PetscCall(MatDestroy(&Ce));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_Elemental_MPIDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatSetSizes(C,A->rmap->n,B->cmap->n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(C,MATMPIDENSE));
  PetscCall(MatSetUp(C));
  C->ops->matmultnumeric = MatMatMultNumeric_Elemental_MPIDense;
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSetFromOptions_Elemental_MPIDense_AB(Mat C)
{
  PetscFunctionBegin;
  C->ops->matmultsymbolic = MatMatMultSymbolic_Elemental_MPIDense;
  C->ops->productsymbolic = MatProductSymbolic_AB;
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSetFromOptions_Elemental_MPIDense(Mat C)
{
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  if (product->type == MATPRODUCT_AB) {
    PetscCall(MatProductSetFromOptions_Elemental_MPIDense_AB(C));
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------- */

static PetscErrorCode MatGetDiagonal_Elemental(Mat A,Vec D)
{
  PetscInt        i,nrows,ncols,nD,rrank,ridx,crank,cidx;
  Mat_Elemental   *a = (Mat_Elemental*)A->data;
  PetscElemScalar v;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCall(MatGetSize(A,&nrows,&ncols));
  nD = nrows>ncols ? ncols : nrows;
  for (i=0; i<nD; i++) {
    PetscInt erow,ecol;
    P2RO(A,0,i,&rrank,&ridx);
    RO2E(A,0,rrank,ridx,&erow);
    PetscCheckFalse(rrank < 0 || ridx < 0 || erow < 0,comm,PETSC_ERR_PLIB,"Incorrect row translation");
    P2RO(A,1,i,&crank,&cidx);
    RO2E(A,1,crank,cidx,&ecol);
    PetscCheckFalse(crank < 0 || cidx < 0 || ecol < 0,comm,PETSC_ERR_PLIB,"Incorrect col translation");
    v = a->emat->Get(erow,ecol);
    PetscCall(VecSetValues(D,1,&i,(PetscScalar*)&v,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(D));
  PetscCall(VecAssemblyEnd(D));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalScale_Elemental(Mat X,Vec L,Vec R)
{
  Mat_Elemental         *x = (Mat_Elemental*)X->data;
  const PetscElemScalar *d;

  PetscFunctionBegin;
  if (R) {
    PetscCall(VecGetArrayRead(R,(const PetscScalar **)&d));
    El::DistMatrix<PetscElemScalar,El::VC,El::STAR> de;
    de.LockedAttach(X->cmap->N,1,*x->grid,0,0,d,X->cmap->n);
    El::DiagonalScale(El::RIGHT,El::NORMAL,de,*x->emat);
    PetscCall(VecRestoreArrayRead(R,(const PetscScalar **)&d));
  }
  if (L) {
    PetscCall(VecGetArrayRead(L,(const PetscScalar **)&d));
    El::DistMatrix<PetscElemScalar,El::VC,El::STAR> de;
    de.LockedAttach(X->rmap->N,1,*x->grid,0,0,d,X->rmap->n);
    El::DiagonalScale(El::LEFT,El::NORMAL,de,*x->emat);
    PetscCall(VecRestoreArrayRead(L,(const PetscScalar **)&d));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMissingDiagonal_Elemental(Mat A,PetscBool *missing,PetscInt *d)
{
  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_Elemental(Mat X,PetscScalar a)
{
  Mat_Elemental  *x = (Mat_Elemental*)X->data;

  PetscFunctionBegin;
  El::Scale((PetscElemScalar)a,*x->emat);
  PetscFunctionReturn(0);
}

/*
  MatAXPY - Computes Y = a*X + Y.
*/
static PetscErrorCode MatAXPY_Elemental(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_Elemental  *x = (Mat_Elemental*)X->data;
  Mat_Elemental  *y = (Mat_Elemental*)Y->data;

  PetscFunctionBegin;
  El::Axpy((PetscElemScalar)a,*x->emat,*y->emat);
  PetscCall(PetscObjectStateIncrease((PetscObject)Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_Elemental(Mat A,Mat B,MatStructure str)
{
  Mat_Elemental *a=(Mat_Elemental*)A->data;
  Mat_Elemental *b=(Mat_Elemental*)B->data;

  PetscFunctionBegin;
  El::Copy(*a->emat,*b->emat);
  PetscCall(PetscObjectStateIncrease((PetscObject)B));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_Elemental(Mat A,MatDuplicateOption op,Mat *B)
{
  Mat            Be;
  MPI_Comm       comm;
  Mat_Elemental  *a=(Mat_Elemental*)A->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCall(MatCreate(comm,&Be));
  PetscCall(MatSetSizes(Be,A->rmap->n,A->cmap->n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(Be,MATELEMENTAL));
  PetscCall(MatSetUp(Be));
  *B = Be;
  if (op == MAT_COPY_VALUES) {
    Mat_Elemental *b=(Mat_Elemental*)Be->data;
    El::Copy(*a->emat,*b->emat);
  }
  Be->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTranspose_Elemental(Mat A,MatReuse reuse,Mat *B)
{
  Mat            Be = *B;
  MPI_Comm       comm;
  Mat_Elemental  *a = (Mat_Elemental*)A->data, *b;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  /* Only out-of-place supported */
  PetscCheck(reuse != MAT_INPLACE_MATRIX,comm,PETSC_ERR_SUP,"Only out-of-place supported");
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatCreate(comm,&Be));
    PetscCall(MatSetSizes(Be,A->cmap->n,A->rmap->n,PETSC_DECIDE,PETSC_DECIDE));
    PetscCall(MatSetType(Be,MATELEMENTAL));
    PetscCall(MatSetUp(Be));
    *B = Be;
  }
  b = (Mat_Elemental*)Be->data;
  El::Transpose(*a->emat,*b->emat);
  Be->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConjugate_Elemental(Mat A)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;

  PetscFunctionBegin;
  El::Conjugate(*a->emat);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHermitianTranspose_Elemental(Mat A,MatReuse reuse,Mat *B)
{
  Mat            Be = *B;
  MPI_Comm       comm;
  Mat_Elemental  *a = (Mat_Elemental*)A->data, *b;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  /* Only out-of-place supported */
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatCreate(comm,&Be));
    PetscCall(MatSetSizes(Be,A->cmap->n,A->rmap->n,PETSC_DECIDE,PETSC_DECIDE));
    PetscCall(MatSetType(Be,MATELEMENTAL));
    PetscCall(MatSetUp(Be));
    *B = Be;
  }
  b = (Mat_Elemental*)Be->data;
  El::Adjoint(*a->emat,*b->emat);
  Be->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_Elemental(Mat A,Vec B,Vec X)
{
  Mat_Elemental     *a = (Mat_Elemental*)A->data;
  PetscElemScalar   *x;
  PetscInt          pivoting = a->pivoting;

  PetscFunctionBegin;
  PetscCall(VecCopy(B,X));
  PetscCall(VecGetArray(X,(PetscScalar **)&x));

  El::DistMatrix<PetscElemScalar,El::VC,El::STAR> xe;
  xe.Attach(A->rmap->N,1,*a->grid,0,0,x,A->rmap->n);
  El::DistMatrix<PetscElemScalar,El::MC,El::MR> xer(xe);
  switch (A->factortype) {
  case MAT_FACTOR_LU:
    if (pivoting == 0) {
      El::lu::SolveAfter(El::NORMAL,*a->emat,xer);
    } else if (pivoting == 1) {
      El::lu::SolveAfter(El::NORMAL,*a->emat,*a->P,xer);
    } else { /* pivoting == 2 */
      El::lu::SolveAfter(El::NORMAL,*a->emat,*a->P,*a->Q,xer);
    }
    break;
  case MAT_FACTOR_CHOLESKY:
    El::cholesky::SolveAfter(El::UPPER,El::NORMAL,*a->emat,xer);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unfactored Matrix or Unsupported MatFactorType");
    break;
  }
  El::Copy(xer,xe);

  PetscCall(VecRestoreArray(X,(PetscScalar **)&x));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveAdd_Elemental(Mat A,Vec B,Vec Y,Vec X)
{
  PetscFunctionBegin;
  PetscCall(MatSolve_Elemental(A,B,X));
  PetscCall(VecAXPY(X,1,Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_Elemental(Mat A,Mat B,Mat X)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  Mat_Elemental  *x;
  Mat            C;
  PetscInt       pivoting = a->pivoting;
  PetscBool      flg;
  MatType        type;

  PetscFunctionBegin;
  PetscCall(MatGetType(X,&type));
  PetscCall(PetscStrcmp(type,MATELEMENTAL,&flg));
  if (!flg) {
    PetscCall(MatConvert(B,MATELEMENTAL,MAT_INITIAL_MATRIX,&C));
    x = (Mat_Elemental*)C->data;
  } else {
    x = (Mat_Elemental*)X->data;
    El::Copy(*((Mat_Elemental*)B->data)->emat,*x->emat);
  }
  switch (A->factortype) {
  case MAT_FACTOR_LU:
    if (pivoting == 0) {
      El::lu::SolveAfter(El::NORMAL,*a->emat,*x->emat);
    } else if (pivoting == 1) {
      El::lu::SolveAfter(El::NORMAL,*a->emat,*a->P,*x->emat);
    } else {
      El::lu::SolveAfter(El::NORMAL,*a->emat,*a->P,*a->Q,*x->emat);
    }
    break;
  case MAT_FACTOR_CHOLESKY:
    El::cholesky::SolveAfter(El::UPPER,El::NORMAL,*a->emat,*x->emat);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unfactored Matrix or Unsupported MatFactorType");
    break;
  }
  if (!flg) {
    PetscCall(MatConvert(C,type,MAT_REUSE_MATRIX,&X));
    PetscCall(MatDestroy(&C));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactor_Elemental(Mat A,IS row,IS col,const MatFactorInfo *info)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  PetscInt       pivoting = a->pivoting;

  PetscFunctionBegin;
  if (pivoting == 0) {
    El::LU(*a->emat);
  } else if (pivoting == 1) {
    El::LU(*a->emat,*a->P);
  } else {
    El::LU(*a->emat,*a->P,*a->Q);
  }
  A->factortype = MAT_FACTOR_LU;
  A->assembled  = PETSC_TRUE;

  PetscCall(PetscFree(A->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERELEMENTAL,&A->solvertype));
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatLUFactorNumeric_Elemental(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatCopy(A,F,SAME_NONZERO_PATTERN));
  PetscCall(MatLUFactor_Elemental(F,0,0,info));
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatLUFactorSymbolic_Elemental(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  /* F is created and allocated by MatGetFactor_elemental_petsc(), skip this routine. */
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactor_Elemental(Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  El::DistMatrix<PetscElemScalar,El::MC,El::STAR> d;

  PetscFunctionBegin;
  El::Cholesky(El::UPPER,*a->emat);
  A->factortype = MAT_FACTOR_CHOLESKY;
  A->assembled  = PETSC_TRUE;

  PetscCall(PetscFree(A->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERELEMENTAL,&A->solvertype));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorNumeric_Elemental(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatCopy(A,F,SAME_NONZERO_PATTERN));
  PetscCall(MatCholeskyFactor_Elemental(F,0,info));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorSymbolic_Elemental(Mat F,Mat A,IS perm,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  /* F is created and allocated by MatGetFactor_elemental_petsc(), skip this routine. */
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorGetSolverType_elemental_elemental(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERELEMENTAL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetFactor_elemental_elemental(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
  PetscCall(MatSetSizes(B,A->rmap->n,A->cmap->n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(B,MATELEMENTAL));
  PetscCall(MatSetUp(B));
  B->factortype = ftype;
  B->trivialsymbolic = PETSC_TRUE;
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERELEMENTAL,&B->solvertype));

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_elemental_elemental));
  *F            = B;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_Elemental(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERELEMENTAL,MATELEMENTAL,        MAT_FACTOR_LU,MatGetFactor_elemental_elemental));
  PetscCall(MatSolverTypeRegister(MATSOLVERELEMENTAL,MATELEMENTAL,        MAT_FACTOR_CHOLESKY,MatGetFactor_elemental_elemental));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNorm_Elemental(Mat A,NormType type,PetscReal *nrm)
{
  Mat_Elemental *a=(Mat_Elemental*)A->data;

  PetscFunctionBegin;
  switch (type) {
  case NORM_1:
    *nrm = El::OneNorm(*a->emat);
    break;
  case NORM_FROBENIUS:
    *nrm = El::FrobeniusNorm(*a->emat);
    break;
  case NORM_INFINITY:
    *nrm = El::InfinityNorm(*a->emat);
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Unsupported norm type");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_Elemental(Mat A)
{
  Mat_Elemental *a=(Mat_Elemental*)A->data;

  PetscFunctionBegin;
  El::Zero(*a->emat);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetOwnershipIS_Elemental(Mat A,IS *rows,IS *cols)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  PetscInt       i,m,shift,stride,*idx;

  PetscFunctionBegin;
  if (rows) {
    m = a->emat->LocalHeight();
    shift = a->emat->ColShift();
    stride = a->emat->ColStride();
    PetscCall(PetscMalloc1(m,&idx));
    for (i=0; i<m; i++) {
      PetscInt rank,offset;
      E2RO(A,0,shift+i*stride,&rank,&offset);
      RO2P(A,0,rank,offset,&idx[i]);
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,m,idx,PETSC_OWN_POINTER,rows));
  }
  if (cols) {
    m = a->emat->LocalWidth();
    shift = a->emat->RowShift();
    stride = a->emat->RowStride();
    PetscCall(PetscMalloc1(m,&idx));
    for (i=0; i<m; i++) {
      PetscInt rank,offset;
      E2RO(A,1,shift+i*stride,&rank,&offset);
      RO2P(A,1,rank,offset,&idx[i]);
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,m,idx,PETSC_OWN_POINTER,cols));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConvert_Elemental_Dense(Mat A,MatType newtype,MatReuse reuse,Mat *B)
{
  Mat                Bmpi;
  Mat_Elemental      *a = (Mat_Elemental*)A->data;
  MPI_Comm           comm;
  IS                 isrows,iscols;
  PetscInt           rrank,ridx,crank,cidx,nrows,ncols,i,j,erow,ecol,elrow,elcol;
  const PetscInt     *rows,*cols;
  PetscElemScalar    v;
  const El::Grid     &grid = a->emat->Grid();

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));

  if (reuse == MAT_REUSE_MATRIX) {
    Bmpi = *B;
  } else {
    PetscCall(MatCreate(comm,&Bmpi));
    PetscCall(MatSetSizes(Bmpi,A->rmap->n,A->cmap->n,PETSC_DECIDE,PETSC_DECIDE));
    PetscCall(MatSetType(Bmpi,MATDENSE));
    PetscCall(MatSetUp(Bmpi));
  }

  /* Get local entries of A */
  PetscCall(MatGetOwnershipIS(A,&isrows,&iscols));
  PetscCall(ISGetLocalSize(isrows,&nrows));
  PetscCall(ISGetIndices(isrows,&rows));
  PetscCall(ISGetLocalSize(iscols,&ncols));
  PetscCall(ISGetIndices(iscols,&cols));

  if (a->roworiented) {
    for (i=0; i<nrows; i++) {
      P2RO(A,0,rows[i],&rrank,&ridx); /* convert indices between PETSc <-> (Rank,Offset) <-> Elemental */
      RO2E(A,0,rrank,ridx,&erow);
      PetscCheckFalse(rrank < 0 || ridx < 0 || erow < 0,comm,PETSC_ERR_PLIB,"Incorrect row translation");
      for (j=0; j<ncols; j++) {
        P2RO(A,1,cols[j],&crank,&cidx);
        RO2E(A,1,crank,cidx,&ecol);
        PetscCheckFalse(crank < 0 || cidx < 0 || ecol < 0,comm,PETSC_ERR_PLIB,"Incorrect col translation");

        elrow = erow / grid.MCSize(); /* Elemental local row index */
        elcol = ecol / grid.MRSize(); /* Elemental local column index */
        v = a->emat->GetLocal(elrow,elcol);
        PetscCall(MatSetValues(Bmpi,1,&rows[i],1,&cols[j],(PetscScalar *)&v,INSERT_VALUES));
      }
    }
  } else { /* column-oriented */
    for (j=0; j<ncols; j++) {
      P2RO(A,1,cols[j],&crank,&cidx);
      RO2E(A,1,crank,cidx,&ecol);
      PetscCheckFalse(crank < 0 || cidx < 0 || ecol < 0,comm,PETSC_ERR_PLIB,"Incorrect col translation");
      for (i=0; i<nrows; i++) {
        P2RO(A,0,rows[i],&rrank,&ridx); /* convert indices between PETSc <-> (Rank,Offset) <-> Elemental */
        RO2E(A,0,rrank,ridx,&erow);
        PetscCheckFalse(rrank < 0 || ridx < 0 || erow < 0,comm,PETSC_ERR_PLIB,"Incorrect row translation");

        elrow = erow / grid.MCSize(); /* Elemental local row index */
        elcol = ecol / grid.MRSize(); /* Elemental local column index */
        v = a->emat->GetLocal(elrow,elcol);
        PetscCall(MatSetValues(Bmpi,1,&rows[i],1,&cols[j],(PetscScalar *)&v,INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(Bmpi,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Bmpi,MAT_FINAL_ASSEMBLY));
  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&Bmpi));
  } else {
    *B = Bmpi;
  }
  PetscCall(ISDestroy(&isrows));
  PetscCall(ISDestroy(&iscols));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_Elemental(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat               mat_elemental;
  PetscInt          M=A->rmap->N,N=A->cmap->N,row,ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    mat_elemental = *newmat;
    PetscCall(MatZeroEntries(mat_elemental));
  } else {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &mat_elemental));
    PetscCall(MatSetSizes(mat_elemental,PETSC_DECIDE,PETSC_DECIDE,M,N));
    PetscCall(MatSetType(mat_elemental,MATELEMENTAL));
    PetscCall(MatSetUp(mat_elemental));
  }
  for (row=0; row<M; row++) {
    PetscCall(MatGetRow(A,row,&ncols,&cols,&vals));
    /* PETSc-Elemental interface uses axpy for setting off-processor entries, only ADD_VALUES is allowed */
    PetscCall(MatSetValues(mat_elemental,1,&row,ncols,cols,vals,ADD_VALUES));
    PetscCall(MatRestoreRow(A,row,&ncols,&cols,&vals));
  }
  PetscCall(MatAssemblyBegin(mat_elemental, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat_elemental, MAT_FINAL_ASSEMBLY));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&mat_elemental));
  } else {
    *newmat = mat_elemental;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_Elemental(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat               mat_elemental;
  PetscInt          row,ncols,rstart=A->rmap->rstart,rend=A->rmap->rend,j;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    mat_elemental = *newmat;
    PetscCall(MatZeroEntries(mat_elemental));
  } else {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &mat_elemental));
    PetscCall(MatSetSizes(mat_elemental,PETSC_DECIDE,PETSC_DECIDE,A->rmap->N,A->cmap->N));
    PetscCall(MatSetType(mat_elemental,MATELEMENTAL));
    PetscCall(MatSetUp(mat_elemental));
  }
  for (row=rstart; row<rend; row++) {
    PetscCall(MatGetRow(A,row,&ncols,&cols,&vals));
    for (j=0; j<ncols; j++) {
      /* PETSc-Elemental interface uses axpy for setting off-processor entries, only ADD_VALUES is allowed */
      PetscCall(MatSetValues(mat_elemental,1,&row,1,&cols[j],&vals[j],ADD_VALUES));
    }
    PetscCall(MatRestoreRow(A,row,&ncols,&cols,&vals));
  }
  PetscCall(MatAssemblyBegin(mat_elemental, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat_elemental, MAT_FINAL_ASSEMBLY));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&mat_elemental));
  } else {
    *newmat = mat_elemental;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqSBAIJ_Elemental(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat               mat_elemental;
  PetscInt          M=A->rmap->N,N=A->cmap->N,row,ncols,j;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    mat_elemental = *newmat;
    PetscCall(MatZeroEntries(mat_elemental));
  } else {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &mat_elemental));
    PetscCall(MatSetSizes(mat_elemental,PETSC_DECIDE,PETSC_DECIDE,M,N));
    PetscCall(MatSetType(mat_elemental,MATELEMENTAL));
    PetscCall(MatSetUp(mat_elemental));
  }
  PetscCall(MatGetRowUpperTriangular(A));
  for (row=0; row<M; row++) {
    PetscCall(MatGetRow(A,row,&ncols,&cols,&vals));
    /* PETSc-Elemental interface uses axpy for setting off-processor entries, only ADD_VALUES is allowed */
    PetscCall(MatSetValues(mat_elemental,1,&row,ncols,cols,vals,ADD_VALUES));
    for (j=0; j<ncols; j++) { /* lower triangular part */
      PetscScalar v;
      if (cols[j] == row) continue;
      v    = A->hermitian ? PetscConj(vals[j]) : vals[j];
      PetscCall(MatSetValues(mat_elemental,1,&cols[j],1,&row,&v,ADD_VALUES));
    }
    PetscCall(MatRestoreRow(A,row,&ncols,&cols,&vals));
  }
  PetscCall(MatRestoreRowUpperTriangular(A));
  PetscCall(MatAssemblyBegin(mat_elemental, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat_elemental, MAT_FINAL_ASSEMBLY));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&mat_elemental));
  } else {
    *newmat = mat_elemental;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_MPISBAIJ_Elemental(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat               mat_elemental;
  PetscInt          M=A->rmap->N,N=A->cmap->N,row,ncols,j,rstart=A->rmap->rstart,rend=A->rmap->rend;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    mat_elemental = *newmat;
    PetscCall(MatZeroEntries(mat_elemental));
  } else {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &mat_elemental));
    PetscCall(MatSetSizes(mat_elemental,PETSC_DECIDE,PETSC_DECIDE,M,N));
    PetscCall(MatSetType(mat_elemental,MATELEMENTAL));
    PetscCall(MatSetUp(mat_elemental));
  }
  PetscCall(MatGetRowUpperTriangular(A));
  for (row=rstart; row<rend; row++) {
    PetscCall(MatGetRow(A,row,&ncols,&cols,&vals));
    /* PETSc-Elemental interface uses axpy for setting off-processor entries, only ADD_VALUES is allowed */
    PetscCall(MatSetValues(mat_elemental,1,&row,ncols,cols,vals,ADD_VALUES));
    for (j=0; j<ncols; j++) { /* lower triangular part */
      PetscScalar v;
      if (cols[j] == row) continue;
      v    = A->hermitian ? PetscConj(vals[j]) : vals[j];
      PetscCall(MatSetValues(mat_elemental,1,&cols[j],1,&row,&v,ADD_VALUES));
    }
    PetscCall(MatRestoreRow(A,row,&ncols,&cols,&vals));
  }
  PetscCall(MatRestoreRowUpperTriangular(A));
  PetscCall(MatAssemblyBegin(mat_elemental, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat_elemental, MAT_FINAL_ASSEMBLY));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&mat_elemental));
  } else {
    *newmat = mat_elemental;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_Elemental(Mat A)
{
  Mat_Elemental      *a = (Mat_Elemental*)A->data;
  Mat_Elemental_Grid *commgrid;
  PetscBool          flg;
  MPI_Comm           icomm;

  PetscFunctionBegin;
  delete a->emat;
  delete a->P;
  delete a->Q;

  El::mpi::Comm cxxcomm(PetscObjectComm((PetscObject)A));
  PetscCall(PetscCommDuplicate(cxxcomm.comm,&icomm,NULL));
  PetscCallMPI(MPI_Comm_get_attr(icomm,Petsc_Elemental_keyval,(void**)&commgrid,(int*)&flg));
  if (--commgrid->grid_refct == 0) {
    delete commgrid->grid;
    PetscCall(PetscFree(commgrid));
    PetscCallMPI(MPI_Comm_free_keyval(&Petsc_Elemental_keyval));
  }
  PetscCall(PetscCommDestroy(&icomm));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatGetOwnershipIS_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_elemental_mpidense_C",NULL));
  PetscCall(PetscFree(A->data));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_Elemental(Mat A)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  MPI_Comm       comm;
  PetscMPIInt    rsize,csize;
  PetscInt       n;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));

  /* Check if local row and column sizes are equally distributed.
     Jed: Elemental uses "element" cyclic ordering so the sizes need to match that
     exactly.  The strategy in MatElemental is for PETSc to implicitly permute to block ordering (like would be returned by
     PetscSplitOwnership(comm,&n,&N), at which point Elemental matrices can act on PETSc vectors without redistributing the vectors. */
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  n = PETSC_DECIDE;
  PetscCall(PetscSplitOwnership(comm,&n,&A->rmap->N));
  PetscCheck(n == A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local row size %" PetscInt_FMT " of ELEMENTAL matrix must be equally distributed",A->rmap->n);

  n = PETSC_DECIDE;
  PetscCall(PetscSplitOwnership(comm,&n,&A->cmap->N));
  PetscCheck(n == A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local column size %" PetscInt_FMT " of ELEMENTAL matrix must be equally distributed",A->cmap->n);

  a->emat->Resize(A->rmap->N,A->cmap->N);
  El::Zero(*a->emat);

  PetscCallMPI(MPI_Comm_size(A->rmap->comm,&rsize));
  PetscCallMPI(MPI_Comm_size(A->cmap->comm,&csize));
  PetscCheck(csize == rsize,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Cannot use row and column communicators of different sizes");
  a->commsize = rsize;
  a->mr[0] = A->rmap->N % rsize; if (!a->mr[0]) a->mr[0] = rsize;
  a->mr[1] = A->cmap->N % csize; if (!a->mr[1]) a->mr[1] = csize;
  a->m[0]  = A->rmap->N / rsize + (a->mr[0] != rsize);
  a->m[1]  = A->cmap->N / csize + (a->mr[1] != csize);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyBegin_Elemental(Mat A, MatAssemblyType type)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;

  PetscFunctionBegin;
  /* printf("Calling ProcessQueues\n"); */
  a->emat->ProcessQueues();
  /* printf("Finished ProcessQueues\n"); */
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_Elemental(Mat A, MatAssemblyType type)
{
  PetscFunctionBegin;
  /* Currently does nothing */
  PetscFunctionReturn(0);
}

PetscErrorCode MatLoad_Elemental(Mat newMat, PetscViewer viewer)
{
  Mat            Adense,Ae;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)newMat,&comm));
  PetscCall(MatCreate(comm,&Adense));
  PetscCall(MatSetType(Adense,MATDENSE));
  PetscCall(MatLoad(Adense,viewer));
  PetscCall(MatConvert(Adense, MATELEMENTAL, MAT_INITIAL_MATRIX,&Ae));
  PetscCall(MatDestroy(&Adense));
  PetscCall(MatHeaderReplace(newMat,&Ae));
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
       0,
/*10*/ 0,
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
       MatSetOption_Elemental,
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
       MatShift_Basic,
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
       MatLoad_Elemental,
/*84*/ 0,
       0,
       0,
       0,
       0,
/*89*/ 0,
       0,
       MatMatMultNumeric_Elemental,
       0,
       0,
/*94*/ 0,
       0,
       0,
       MatMatTransposeMultNumeric_Elemental,
       0,
/*99*/ MatProductSetFromOptions_Elemental,
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
       MatMissingDiagonal_Elemental,
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
       0,
       0,
/*140*/0,
       0,
       0,
       0,
       0,
/*145*/0,
       0,
       0
};

/*MC
   MATELEMENTAL = "elemental" - A matrix type for dense matrices using the Elemental package

  Use ./configure --download-elemental to install PETSc to use Elemental

  Use -pc_type lu -pc_factor_mat_solver_type elemental to use this direct solver

   Options Database Keys:
+ -mat_type elemental - sets the matrix type to "elemental" during a call to MatSetFromOptions()
- -mat_elemental_grid_height - sets Grid Height for 2D cyclic ordering of internal matrix

  Level: beginner

.seealso: `MATDENSE`
M*/

PETSC_EXTERN PetscErrorCode MatCreate_Elemental(Mat A)
{
  Mat_Elemental      *a;
  PetscBool          flg,flg1;
  Mat_Elemental_Grid *commgrid;
  MPI_Comm           icomm;
  PetscInt           optv1;

  PetscFunctionBegin;
  PetscCall(PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps)));
  A->insertmode = NOT_SET_VALUES;

  PetscCall(PetscNewLog(A,&a));
  A->data = (void*)a;

  /* Set up the elemental matrix */
  El::mpi::Comm cxxcomm(PetscObjectComm((PetscObject)A));

  /* Grid needs to be shared between multiple Mats on the same communicator, implement by attribute caching on the MPI_Comm */
  if (Petsc_Elemental_keyval == MPI_KEYVAL_INVALID) {
    PetscCallMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,MPI_COMM_NULL_DELETE_FN,&Petsc_Elemental_keyval,(void*)0));
    PetscCall(PetscCitationsRegister(ElementalCitation,&ElementalCite));
  }
  PetscCall(PetscCommDuplicate(cxxcomm.comm,&icomm,NULL));
  PetscCallMPI(MPI_Comm_get_attr(icomm,Petsc_Elemental_keyval,(void**)&commgrid,(int*)&flg));
  if (!flg) {
    PetscCall(PetscNewLog(A,&commgrid));

    PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"Elemental Options","Mat");
    /* displayed default grid sizes (CommSize,1) are set by us arbitrarily until El::Grid() is called */
    PetscCall(PetscOptionsInt("-mat_elemental_grid_height","Grid Height","None",El::mpi::Size(cxxcomm),&optv1,&flg1));
    if (flg1) {
      PetscCheckFalse(El::mpi::Size(cxxcomm) % optv1,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Grid Height %" PetscInt_FMT " must evenly divide CommSize %" PetscInt_FMT,optv1,(PetscInt)El::mpi::Size(cxxcomm));
      commgrid->grid = new El::Grid(cxxcomm,optv1); /* use user-provided grid height */
    } else {
      commgrid->grid = new El::Grid(cxxcomm); /* use Elemental default grid sizes */
      /* printf("new commgrid->grid = %p\n",commgrid->grid);  -- memory leak revealed by valgrind? */
    }
    commgrid->grid_refct = 1;
    PetscCallMPI(MPI_Comm_set_attr(icomm,Petsc_Elemental_keyval,(void*)commgrid));

    a->pivoting    = 1;
    PetscCall(PetscOptionsInt("-mat_elemental_pivoting","Pivoting","None",a->pivoting,&a->pivoting,NULL));

    PetscOptionsEnd();
  } else {
    commgrid->grid_refct++;
  }
  PetscCall(PetscCommDestroy(&icomm));
  a->grid        = commgrid->grid;
  a->emat        = new El::DistMatrix<PetscElemScalar>(*a->grid);
  a->roworiented = PETSC_TRUE;

  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatGetOwnershipIS_C",MatGetOwnershipIS_Elemental));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_elemental_mpidense_C",MatProductSetFromOptions_Elemental_MPIDense));
  PetscCall(PetscObjectChangeTypeName((PetscObject)A,MATELEMENTAL));
  PetscFunctionReturn(0);
}
