#include <petsc/private/petscscalapack.h>  /*I "petscmat.h" I*/

const char ScaLAPACKCitation[] = "@BOOK{scalapack-user-guide,\n"
"       AUTHOR = {L. S. Blackford and J. Choi and A. Cleary and E. D'Azevedo and\n"
"                 J. Demmel and I. Dhillon and J. Dongarra and S. Hammarling and\n"
"                 G. Henry and A. Petitet and K. Stanley and D. Walker and R. C. Whaley},\n"
"       TITLE = {Sca{LAPACK} Users' Guide},\n"
"       PUBLISHER = {SIAM},\n"
"       ADDRESS = {Philadelphia, PA},\n"
"       YEAR = 1997\n"
"}\n";
static PetscBool ScaLAPACKCite = PETSC_FALSE;

#define DEFAULT_BLOCKSIZE 64

/*
    The variable Petsc_ScaLAPACK_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Mat_ScaLAPACK_Grid
*/
static PetscMPIInt Petsc_ScaLAPACK_keyval = MPI_KEYVAL_INVALID;

static PetscErrorCode Petsc_ScaLAPACK_keyval_free(void)
{
  PetscFunctionBegin;
  PetscCall(PetscInfo(NULL,"Freeing Petsc_ScaLAPACK_keyval\n"));
  PetscCallMPI(MPI_Comm_free_keyval(&Petsc_ScaLAPACK_keyval));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_ScaLAPACK(Mat A,PetscViewer viewer)
{
  Mat_ScaLAPACK     *a = (Mat_ScaLAPACK*)A->data;
  PetscBool         iascii;
  PetscViewerFormat format;
  Mat               Adense;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"block sizes: %d,%d\n",(int)a->mb,(int)a->nb));
      PetscCall(PetscViewerASCIIPrintf(viewer,"grid height=%d, grid width=%d\n",(int)a->grid->nprow,(int)a->grid->npcol));
      PetscCall(PetscViewerASCIIPrintf(viewer,"coordinates of process owning first row and column: (%d,%d)\n",(int)a->rsrc,(int)a->csrc));
      PetscCall(PetscViewerASCIIPrintf(viewer,"dimension of largest local matrix: %d x %d\n",(int)a->locr,(int)a->locc));
      PetscFunctionReturn(0);
    } else if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      PetscFunctionReturn(0);
    }
  }
  /* convert to dense format and call MatView() */
  PetscCall(MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&Adense));
  PetscCall(MatView(Adense,viewer));
  PetscCall(MatDestroy(&Adense));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetInfo_ScaLAPACK(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data;
  PetscLogDouble isend[2],irecv[2];

  PetscFunctionBegin;
  info->block_size = 1.0;

  isend[0] = a->lld*a->locc;     /* locally allocated */
  isend[1] = a->locr*a->locc;    /* used submatrix */
  if (flag == MAT_LOCAL || flag == MAT_GLOBAL_MAX) {
    info->nz_allocated   = isend[0];
    info->nz_used        = isend[1];
  } else if (flag == MAT_GLOBAL_MAX) {
    PetscCall(MPIU_Allreduce(isend,irecv,2,MPIU_PETSCLOGDOUBLE,MPIU_MAX,PetscObjectComm((PetscObject)A)));
    info->nz_allocated   = irecv[0];
    info->nz_used        = irecv[1];
  } else if (flag == MAT_GLOBAL_SUM) {
    PetscCall(MPIU_Allreduce(isend,irecv,2,MPIU_PETSCLOGDOUBLE,MPIU_SUM,PetscObjectComm((PetscObject)A)));
    info->nz_allocated   = irecv[0];
    info->nz_used        = irecv[1];
  }

  info->nz_unneeded       = 0;
  info->assemblies        = A->num_ass;
  info->mallocs           = 0;
  info->memory            = ((PetscObject)A)->mem;
  info->fill_ratio_given  = 0;
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_ScaLAPACK(Mat A,MatOption op,PetscBool flg)
{
  PetscFunctionBegin;
  switch (op) {
    case MAT_NEW_NONZERO_LOCATIONS:
    case MAT_NEW_NONZERO_LOCATION_ERR:
    case MAT_NEW_NONZERO_ALLOCATION_ERR:
    case MAT_SYMMETRIC:
    case MAT_SORTED_FULL:
    case MAT_HERMITIAN:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported option %s",MatOptions[op]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValues_ScaLAPACK(Mat A,PetscInt nr,const PetscInt *rows,PetscInt nc,const PetscInt *cols,const PetscScalar *vals,InsertMode imode)
{
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data;
  PetscInt       i,j;
  PetscBLASInt   gridx,gcidx,lridx,lcidx,rsrc,csrc;

  PetscFunctionBegin;
  for (i=0;i<nr;i++) {
    if (rows[i] < 0) continue;
    PetscCall(PetscBLASIntCast(rows[i]+1,&gridx));
    for (j=0;j<nc;j++) {
      if (cols[j] < 0) continue;
      PetscCall(PetscBLASIntCast(cols[j]+1,&gcidx));
      PetscStackCallBLAS("SCALAPACKinfog2l",SCALAPACKinfog2l_(&gridx,&gcidx,a->desc,&a->grid->nprow,&a->grid->npcol,&a->grid->myrow,&a->grid->mycol,&lridx,&lcidx,&rsrc,&csrc));
      if (rsrc==a->grid->myrow && csrc==a->grid->mycol) {
        switch (imode) {
          case INSERT_VALUES: a->loc[lridx-1+(lcidx-1)*a->lld] = vals[i*nc+j]; break;
          case ADD_VALUES: a->loc[lridx-1+(lcidx-1)*a->lld] += vals[i*nc+j]; break;
          default: SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support for InsertMode %d",(int)imode);
        }
      } else {
        PetscCheck(!A->nooffprocentries,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Setting off process entry even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set");
        A->assembled = PETSC_FALSE;
        PetscCall(MatStashValuesRow_Private(&A->stash,rows[i],1,cols+j,vals+i*nc+j,(PetscBool)(imode==ADD_VALUES)));
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultXXXYYY_ScaLAPACK(Mat A,PetscBool transpose,PetscScalar beta,const PetscScalar *x,PetscScalar *y)
{
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data;
  PetscScalar    *x2d,*y2d,alpha=1.0;
  const PetscInt *ranges;
  PetscBLASInt   xdesc[9],ydesc[9],x2desc[9],y2desc[9],mb,nb,lszx,lszy,zero=0,one=1,xlld,ylld,info;

  PetscFunctionBegin;
  if (transpose) {

    /* create ScaLAPACK descriptors for vectors (1d block distribution) */
    PetscCall(PetscLayoutGetRanges(A->rmap,&ranges));
    PetscCall(PetscBLASIntCast(ranges[1],&mb));  /* x block size */
    xlld = PetscMax(1,A->rmap->n);
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(xdesc,&a->M,&one,&mb,&one,&zero,&zero,&a->grid->ictxcol,&xlld,&info));
    PetscCheckScaLapackInfo("descinit",info);
    PetscCall(PetscLayoutGetRanges(A->cmap,&ranges));
    PetscCall(PetscBLASIntCast(ranges[1],&nb));  /* y block size */
    ylld = 1;
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(ydesc,&one,&a->N,&one,&nb,&zero,&zero,&a->grid->ictxrow,&ylld,&info));
    PetscCheckScaLapackInfo("descinit",info);

    /* allocate 2d vectors */
    lszx = SCALAPACKnumroc_(&a->M,&a->mb,&a->grid->myrow,&a->rsrc,&a->grid->nprow);
    lszy = SCALAPACKnumroc_(&a->N,&a->nb,&a->grid->mycol,&a->csrc,&a->grid->npcol);
    PetscCall(PetscMalloc2(lszx,&x2d,lszy,&y2d));
    xlld = PetscMax(1,lszx);

    /* create ScaLAPACK descriptors for vectors (2d block distribution) */
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(x2desc,&a->M,&one,&a->mb,&one,&zero,&zero,&a->grid->ictxt,&xlld,&info));
    PetscCheckScaLapackInfo("descinit",info);
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(y2desc,&one,&a->N,&one,&a->nb,&zero,&zero,&a->grid->ictxt,&ylld,&info));
    PetscCheckScaLapackInfo("descinit",info);

    /* redistribute x as a column of a 2d matrix */
    PetscStackCallBLAS("SCALAPACKgemr2d",SCALAPACKgemr2d_(&a->M,&one,(PetscScalar*)x,&one,&one,xdesc,x2d,&one,&one,x2desc,&a->grid->ictxcol));

    /* redistribute y as a row of a 2d matrix */
    if (beta!=0.0) PetscStackCallBLAS("SCALAPACKgemr2d",SCALAPACKgemr2d_(&one,&a->N,y,&one,&one,ydesc,y2d,&one,&one,y2desc,&a->grid->ictxrow));

    /* call PBLAS subroutine */
    PetscStackCallBLAS("PBLASgemv",PBLASgemv_("T",&a->M,&a->N,&alpha,a->loc,&one,&one,a->desc,x2d,&one,&one,x2desc,&one,&beta,y2d,&one,&one,y2desc,&one));

    /* redistribute y from a row of a 2d matrix */
    PetscStackCallBLAS("SCALAPACKgemr2d",SCALAPACKgemr2d_(&one,&a->N,y2d,&one,&one,y2desc,y,&one,&one,ydesc,&a->grid->ictxrow));

  } else {   /* non-transpose */

    /* create ScaLAPACK descriptors for vectors (1d block distribution) */
    PetscCall(PetscLayoutGetRanges(A->cmap,&ranges));
    PetscCall(PetscBLASIntCast(ranges[1],&nb));  /* x block size */
    xlld = 1;
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(xdesc,&one,&a->N,&one,&nb,&zero,&zero,&a->grid->ictxrow,&xlld,&info));
    PetscCheckScaLapackInfo("descinit",info);
    PetscCall(PetscLayoutGetRanges(A->rmap,&ranges));
    PetscCall(PetscBLASIntCast(ranges[1],&mb));  /* y block size */
    ylld = PetscMax(1,A->rmap->n);
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(ydesc,&a->M,&one,&mb,&one,&zero,&zero,&a->grid->ictxcol,&ylld,&info));
    PetscCheckScaLapackInfo("descinit",info);

    /* allocate 2d vectors */
    lszy = SCALAPACKnumroc_(&a->M,&a->mb,&a->grid->myrow,&a->rsrc,&a->grid->nprow);
    lszx = SCALAPACKnumroc_(&a->N,&a->nb,&a->grid->mycol,&a->csrc,&a->grid->npcol);
    PetscCall(PetscMalloc2(lszx,&x2d,lszy,&y2d));
    ylld = PetscMax(1,lszy);

    /* create ScaLAPACK descriptors for vectors (2d block distribution) */
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(x2desc,&one,&a->N,&one,&a->nb,&zero,&zero,&a->grid->ictxt,&xlld,&info));
    PetscCheckScaLapackInfo("descinit",info);
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(y2desc,&a->M,&one,&a->mb,&one,&zero,&zero,&a->grid->ictxt,&ylld,&info));
    PetscCheckScaLapackInfo("descinit",info);

    /* redistribute x as a row of a 2d matrix */
    PetscStackCallBLAS("SCALAPACKgemr2d",SCALAPACKgemr2d_(&one,&a->N,(PetscScalar*)x,&one,&one,xdesc,x2d,&one,&one,x2desc,&a->grid->ictxrow));

    /* redistribute y as a column of a 2d matrix */
    if (beta!=0.0) PetscStackCallBLAS("SCALAPACKgemr2d",SCALAPACKgemr2d_(&a->M,&one,y,&one,&one,ydesc,y2d,&one,&one,y2desc,&a->grid->ictxcol));

    /* call PBLAS subroutine */
    PetscStackCallBLAS("PBLASgemv",PBLASgemv_("N",&a->M,&a->N,&alpha,a->loc,&one,&one,a->desc,x2d,&one,&one,x2desc,&one,&beta,y2d,&one,&one,y2desc,&one));

    /* redistribute y from a column of a 2d matrix */
    PetscStackCallBLAS("SCALAPACKgemr2d",SCALAPACKgemr2d_(&a->M,&one,y2d,&one,&one,y2desc,y,&one,&one,ydesc,&a->grid->ictxcol));

  }
  PetscCall(PetscFree2(x2d,y2d));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_ScaLAPACK(Mat A,Vec x,Vec y)
{
  const PetscScalar *xarray;
  PetscScalar       *yarray;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x,&xarray));
  PetscCall(VecGetArray(y,&yarray));
  PetscCall(MatMultXXXYYY_ScaLAPACK(A,PETSC_FALSE,0.0,xarray,yarray));
  PetscCall(VecRestoreArrayRead(x,&xarray));
  PetscCall(VecRestoreArray(y,&yarray));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_ScaLAPACK(Mat A,Vec x,Vec y)
{
  const PetscScalar *xarray;
  PetscScalar       *yarray;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x,&xarray));
  PetscCall(VecGetArray(y,&yarray));
  PetscCall(MatMultXXXYYY_ScaLAPACK(A,PETSC_TRUE,0.0,xarray,yarray));
  PetscCall(VecRestoreArrayRead(x,&xarray));
  PetscCall(VecRestoreArray(y,&yarray));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_ScaLAPACK(Mat A,Vec x,Vec y,Vec z)
{
  const PetscScalar *xarray;
  PetscScalar       *zarray;

  PetscFunctionBegin;
  if (y != z) PetscCall(VecCopy(y,z));
  PetscCall(VecGetArrayRead(x,&xarray));
  PetscCall(VecGetArray(z,&zarray));
  PetscCall(MatMultXXXYYY_ScaLAPACK(A,PETSC_FALSE,1.0,xarray,zarray));
  PetscCall(VecRestoreArrayRead(x,&xarray));
  PetscCall(VecRestoreArray(z,&zarray));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_ScaLAPACK(Mat A,Vec x,Vec y,Vec z)
{
  const PetscScalar *xarray;
  PetscScalar       *zarray;

  PetscFunctionBegin;
  if (y != z) PetscCall(VecCopy(y,z));
  PetscCall(VecGetArrayRead(x,&xarray));
  PetscCall(VecGetArray(z,&zarray));
  PetscCall(MatMultXXXYYY_ScaLAPACK(A,PETSC_TRUE,1.0,xarray,zarray));
  PetscCall(VecRestoreArrayRead(x,&xarray));
  PetscCall(VecRestoreArray(z,&zarray));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_ScaLAPACK(Mat A,Mat B,Mat C)
{
  Mat_ScaLAPACK *a = (Mat_ScaLAPACK*)A->data;
  Mat_ScaLAPACK *b = (Mat_ScaLAPACK*)B->data;
  Mat_ScaLAPACK *c = (Mat_ScaLAPACK*)C->data;
  PetscScalar   sone=1.0,zero=0.0;
  PetscBLASInt  one=1;

  PetscFunctionBegin;
  PetscStackCallBLAS("PBLASgemm",PBLASgemm_("N","N",&a->M,&b->N,&a->N,&sone,a->loc,&one,&one,a->desc,b->loc,&one,&one,b->desc,&zero,c->loc,&one,&one,c->desc));
  C->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_ScaLAPACK(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatSetSizes(C,A->rmap->n,B->cmap->n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(C,MATSCALAPACK));
  PetscCall(MatSetUp(C));
  C->ops->matmultnumeric = MatMatMultNumeric_ScaLAPACK;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatTransposeMultNumeric_ScaLAPACK(Mat A,Mat B,Mat C)
{
  Mat_ScaLAPACK *a = (Mat_ScaLAPACK*)A->data;
  Mat_ScaLAPACK *b = (Mat_ScaLAPACK*)B->data;
  Mat_ScaLAPACK *c = (Mat_ScaLAPACK*)C->data;
  PetscScalar   sone=1.0,zero=0.0;
  PetscBLASInt  one=1;

  PetscFunctionBegin;
  PetscStackCallBLAS("PBLASgemm",PBLASgemm_("N","T",&a->M,&b->M,&a->N,&sone,a->loc,&one,&one,a->desc,b->loc,&one,&one,b->desc,&zero,c->loc,&one,&one,c->desc));
  C->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatTransposeMultSymbolic_ScaLAPACK(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatSetSizes(C,A->rmap->n,B->rmap->n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(C,MATSCALAPACK));
  PetscCall(MatSetUp(C));
  PetscFunctionReturn(0);
}

/* --------------------------------------- */
static PetscErrorCode MatProductSetFromOptions_ScaLAPACK_AB(Mat C)
{
  PetscFunctionBegin;
  C->ops->matmultsymbolic = MatMatMultSymbolic_ScaLAPACK;
  C->ops->productsymbolic = MatProductSymbolic_AB;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_ScaLAPACK_ABt(Mat C)
{
  PetscFunctionBegin;
  C->ops->mattransposemultsymbolic = MatMatTransposeMultSymbolic_ScaLAPACK;
  C->ops->productsymbolic          = MatProductSymbolic_ABt;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_ScaLAPACK(Mat C)
{
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  switch (product->type) {
    case MATPRODUCT_AB:
      PetscCall(MatProductSetFromOptions_ScaLAPACK_AB(C));
      break;
    case MATPRODUCT_ABt:
      PetscCall(MatProductSetFromOptions_ScaLAPACK_ABt(C));
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"MatProduct type %s is not supported for ScaLAPACK and ScaLAPACK matrices",MatProductTypes[product->type]);
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------- */

static PetscErrorCode MatGetDiagonal_ScaLAPACK(Mat A,Vec D)
{
  Mat_ScaLAPACK     *a = (Mat_ScaLAPACK*)A->data;
  PetscScalar       *darray,*d2d,v;
  const PetscInt    *ranges;
  PetscBLASInt      j,ddesc[9],d2desc[9],mb,nb,lszd,zero=0,one=1,dlld,info;

  PetscFunctionBegin;
  PetscCall(VecGetArray(D,&darray));

  if (A->rmap->N<=A->cmap->N) {   /* row version */

    /* create ScaLAPACK descriptor for vector (1d block distribution) */
    PetscCall(PetscLayoutGetRanges(A->rmap,&ranges));
    PetscCall(PetscBLASIntCast(ranges[1],&mb));  /* D block size */
    dlld = PetscMax(1,A->rmap->n);
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(ddesc,&a->M,&one,&mb,&one,&zero,&zero,&a->grid->ictxcol,&dlld,&info));
    PetscCheckScaLapackInfo("descinit",info);

    /* allocate 2d vector */
    lszd = SCALAPACKnumroc_(&a->M,&a->mb,&a->grid->myrow,&a->rsrc,&a->grid->nprow);
    PetscCall(PetscCalloc1(lszd,&d2d));
    dlld = PetscMax(1,lszd);

    /* create ScaLAPACK descriptor for vector (2d block distribution) */
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(d2desc,&a->M,&one,&a->mb,&one,&zero,&zero,&a->grid->ictxt,&dlld,&info));
    PetscCheckScaLapackInfo("descinit",info);

    /* collect diagonal */
    for (j=1;j<=a->M;j++) {
      PetscStackCallBLAS("SCALAPACKelget",SCALAPACKelget_("R"," ",&v,a->loc,&j,&j,a->desc));
      PetscStackCallBLAS("SCALAPACKelset",SCALAPACKelset_(d2d,&j,&one,d2desc,&v));
    }

    /* redistribute d from a column of a 2d matrix */
    PetscStackCallBLAS("SCALAPACKgemr2d",SCALAPACKgemr2d_(&a->M,&one,d2d,&one,&one,d2desc,darray,&one,&one,ddesc,&a->grid->ictxcol));
    PetscCall(PetscFree(d2d));

  } else {   /* column version */

    /* create ScaLAPACK descriptor for vector (1d block distribution) */
    PetscCall(PetscLayoutGetRanges(A->cmap,&ranges));
    PetscCall(PetscBLASIntCast(ranges[1],&nb));  /* D block size */
    dlld = 1;
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(ddesc,&one,&a->N,&one,&nb,&zero,&zero,&a->grid->ictxrow,&dlld,&info));
    PetscCheckScaLapackInfo("descinit",info);

    /* allocate 2d vector */
    lszd = SCALAPACKnumroc_(&a->N,&a->nb,&a->grid->mycol,&a->csrc,&a->grid->npcol);
    PetscCall(PetscCalloc1(lszd,&d2d));

    /* create ScaLAPACK descriptor for vector (2d block distribution) */
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(d2desc,&one,&a->N,&one,&a->nb,&zero,&zero,&a->grid->ictxt,&dlld,&info));
    PetscCheckScaLapackInfo("descinit",info);

    /* collect diagonal */
    for (j=1;j<=a->N;j++) {
      PetscStackCallBLAS("SCALAPACKelget",SCALAPACKelget_("C"," ",&v,a->loc,&j,&j,a->desc));
      PetscStackCallBLAS("SCALAPACKelset",SCALAPACKelset_(d2d,&one,&j,d2desc,&v));
    }

    /* redistribute d from a row of a 2d matrix */
    PetscStackCallBLAS("SCALAPACKgemr2d",SCALAPACKgemr2d_(&one,&a->N,d2d,&one,&one,d2desc,darray,&one,&one,ddesc,&a->grid->ictxrow));
    PetscCall(PetscFree(d2d));
  }

  PetscCall(VecRestoreArray(D,&darray));
  PetscCall(VecAssemblyBegin(D));
  PetscCall(VecAssemblyEnd(D));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalScale_ScaLAPACK(Mat A,Vec L,Vec R)
{
  Mat_ScaLAPACK     *a = (Mat_ScaLAPACK*)A->data;
  const PetscScalar *d;
  const PetscInt    *ranges;
  PetscScalar       *d2d;
  PetscBLASInt      i,j,ddesc[9],d2desc[9],mb,nb,lszd,zero=0,one=1,dlld,info;

  PetscFunctionBegin;
  if (R) {
    PetscCall(VecGetArrayRead(R,(const PetscScalar **)&d));
    /* create ScaLAPACK descriptor for vector (1d block distribution) */
    PetscCall(PetscLayoutGetRanges(A->cmap,&ranges));
    PetscCall(PetscBLASIntCast(ranges[1],&nb));  /* D block size */
    dlld = 1;
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(ddesc,&one,&a->N,&one,&nb,&zero,&zero,&a->grid->ictxrow,&dlld,&info));
    PetscCheckScaLapackInfo("descinit",info);

    /* allocate 2d vector */
    lszd = SCALAPACKnumroc_(&a->N,&a->nb,&a->grid->mycol,&a->csrc,&a->grid->npcol);
    PetscCall(PetscCalloc1(lszd,&d2d));

    /* create ScaLAPACK descriptor for vector (2d block distribution) */
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(d2desc,&one,&a->N,&one,&a->nb,&zero,&zero,&a->grid->ictxt,&dlld,&info));
    PetscCheckScaLapackInfo("descinit",info);

    /* redistribute d to a row of a 2d matrix */
    PetscStackCallBLAS("SCALAPACKgemr2d",SCALAPACKgemr2d_(&one,&a->N,(PetscScalar*)d,&one,&one,ddesc,d2d,&one,&one,d2desc,&a->grid->ictxrow));

    /* broadcast along process columns */
    if (!a->grid->myrow) Cdgebs2d(a->grid->ictxt,"C"," ",1,lszd,d2d,dlld);
    else Cdgebr2d(a->grid->ictxt,"C"," ",1,lszd,d2d,dlld,0,a->grid->mycol);

    /* local scaling */
    for (j=0;j<a->locc;j++) for (i=0;i<a->locr;i++) a->loc[i+j*a->lld] *= d2d[j];

    PetscCall(PetscFree(d2d));
    PetscCall(VecRestoreArrayRead(R,(const PetscScalar **)&d));
  }
  if (L) {
    PetscCall(VecGetArrayRead(L,(const PetscScalar **)&d));
    /* create ScaLAPACK descriptor for vector (1d block distribution) */
    PetscCall(PetscLayoutGetRanges(A->rmap,&ranges));
    PetscCall(PetscBLASIntCast(ranges[1],&mb));  /* D block size */
    dlld = PetscMax(1,A->rmap->n);
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(ddesc,&a->M,&one,&mb,&one,&zero,&zero,&a->grid->ictxcol,&dlld,&info));
    PetscCheckScaLapackInfo("descinit",info);

    /* allocate 2d vector */
    lszd = SCALAPACKnumroc_(&a->M,&a->mb,&a->grid->myrow,&a->rsrc,&a->grid->nprow);
    PetscCall(PetscCalloc1(lszd,&d2d));
    dlld = PetscMax(1,lszd);

    /* create ScaLAPACK descriptor for vector (2d block distribution) */
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(d2desc,&a->M,&one,&a->mb,&one,&zero,&zero,&a->grid->ictxt,&dlld,&info));
    PetscCheckScaLapackInfo("descinit",info);

    /* redistribute d to a column of a 2d matrix */
    PetscStackCallBLAS("SCALAPACKgemr2d",SCALAPACKgemr2d_(&a->M,&one,(PetscScalar*)d,&one,&one,ddesc,d2d,&one,&one,d2desc,&a->grid->ictxcol));

    /* broadcast along process rows */
    if (!a->grid->mycol) Cdgebs2d(a->grid->ictxt,"R"," ",lszd,1,d2d,dlld);
    else Cdgebr2d(a->grid->ictxt,"R"," ",lszd,1,d2d,dlld,a->grid->myrow,0);

    /* local scaling */
    for (i=0;i<a->locr;i++) for (j=0;j<a->locc;j++) a->loc[i+j*a->lld] *= d2d[i];

    PetscCall(PetscFree(d2d));
    PetscCall(VecRestoreArrayRead(L,(const PetscScalar **)&d));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMissingDiagonal_ScaLAPACK(Mat A,PetscBool *missing,PetscInt *d)
{
  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_ScaLAPACK(Mat X,PetscScalar a)
{
  Mat_ScaLAPACK *x = (Mat_ScaLAPACK*)X->data;
  PetscBLASInt  n,one=1;

  PetscFunctionBegin;
  n = x->lld*x->locc;
  PetscStackCallBLAS("BLASscal",BLASscal_(&n,&a,x->loc,&one));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatShift_ScaLAPACK(Mat X,PetscScalar alpha)
{
  Mat_ScaLAPACK *x = (Mat_ScaLAPACK*)X->data;
  PetscBLASInt  i,n;
  PetscScalar   v;

  PetscFunctionBegin;
  n = PetscMin(x->M,x->N);
  for (i=1;i<=n;i++) {
    PetscStackCallBLAS("SCALAPACKelget",SCALAPACKelget_("-"," ",&v,x->loc,&i,&i,x->desc));
    v += alpha;
    PetscStackCallBLAS("SCALAPACKelset",SCALAPACKelset_(x->loc,&i,&i,x->desc,&v));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAXPY_ScaLAPACK(Mat Y,PetscScalar alpha,Mat X,MatStructure str)
{
  Mat_ScaLAPACK  *x = (Mat_ScaLAPACK*)X->data;
  Mat_ScaLAPACK  *y = (Mat_ScaLAPACK*)Y->data;
  PetscBLASInt   one=1;
  PetscScalar    beta=1.0;

  PetscFunctionBegin;
  MatScaLAPACKCheckDistribution(Y,1,X,3);
  PetscStackCallBLAS("SCALAPACKmatadd",SCALAPACKmatadd_(&x->M,&x->N,&alpha,x->loc,&one,&one,x->desc,&beta,y->loc,&one,&one,y->desc));
  PetscCall(PetscObjectStateIncrease((PetscObject)Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_ScaLAPACK(Mat A,Mat B,MatStructure str)
{
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data;
  Mat_ScaLAPACK  *b = (Mat_ScaLAPACK*)B->data;

  PetscFunctionBegin;
  PetscCall(PetscArraycpy(b->loc,a->loc,a->lld*a->locc));
  PetscCall(PetscObjectStateIncrease((PetscObject)B));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_ScaLAPACK(Mat A,MatDuplicateOption op,Mat *B)
{
  Mat            Bs;
  MPI_Comm       comm;
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data,*b;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCall(MatCreate(comm,&Bs));
  PetscCall(MatSetSizes(Bs,A->rmap->n,A->cmap->n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(Bs,MATSCALAPACK));
  b = (Mat_ScaLAPACK*)Bs->data;
  b->M    = a->M;
  b->N    = a->N;
  b->mb   = a->mb;
  b->nb   = a->nb;
  b->rsrc = a->rsrc;
  b->csrc = a->csrc;
  PetscCall(MatSetUp(Bs));
  *B = Bs;
  if (op == MAT_COPY_VALUES) {
    PetscCall(PetscArraycpy(b->loc,a->loc,a->lld*a->locc));
  }
  Bs->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTranspose_ScaLAPACK(Mat A,MatReuse reuse,Mat *B)
{
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data, *b;
  Mat            Bs = *B;
  PetscBLASInt   one=1;
  PetscScalar    sone=1.0,zero=0.0;
#if defined(PETSC_USE_COMPLEX)
  PetscInt       i,j;
#endif

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatCreateScaLAPACK(PetscObjectComm((PetscObject)A),a->nb,a->mb,a->N,a->M,a->csrc,a->rsrc,&Bs));
    *B = Bs;
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only MAT_INITIAL_MATRIX supported");
  b = (Mat_ScaLAPACK*)Bs->data;
  PetscStackCallBLAS("PBLAStran",PBLAStran_(&a->N,&a->M,&sone,a->loc,&one,&one,a->desc,&zero,b->loc,&one,&one,b->desc));
#if defined(PETSC_USE_COMPLEX)
  /* undo conjugation */
  for (i=0;i<b->locr;i++) for (j=0;j<b->locc;j++) b->loc[i+j*b->lld] = PetscConj(b->loc[i+j*b->lld]);
#endif
  Bs->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConjugate_ScaLAPACK(Mat A)
{
  Mat_ScaLAPACK *a = (Mat_ScaLAPACK*)A->data;
  PetscInt      i,j;

  PetscFunctionBegin;
  for (i=0;i<a->locr;i++) for (j=0;j<a->locc;j++) a->loc[i+j*a->lld] = PetscConj(a->loc[i+j*a->lld]);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHermitianTranspose_ScaLAPACK(Mat A,MatReuse reuse,Mat *B)
{
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data, *b;
  Mat            Bs = *B;
  PetscBLASInt   one=1;
  PetscScalar    sone=1.0,zero=0.0;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatCreateScaLAPACK(PetscObjectComm((PetscObject)A),a->nb,a->mb,a->N,a->M,a->csrc,a->rsrc,&Bs));
    *B = Bs;
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only MAT_INITIAL_MATRIX supported");
  b = (Mat_ScaLAPACK*)Bs->data;
  PetscStackCallBLAS("PBLAStran",PBLAStran_(&a->N,&a->M,&sone,a->loc,&one,&one,a->desc,&zero,b->loc,&one,&one,b->desc));
  Bs->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_ScaLAPACK(Mat A,Vec B,Vec X)
{
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data;
  PetscScalar    *x,*x2d;
  const PetscInt *ranges;
  PetscBLASInt   xdesc[9],x2desc[9],mb,lszx,zero=0,one=1,xlld,nrhs=1,info;

  PetscFunctionBegin;
  PetscCall(VecCopy(B,X));
  PetscCall(VecGetArray(X,&x));

  /* create ScaLAPACK descriptor for a vector (1d block distribution) */
  PetscCall(PetscLayoutGetRanges(A->rmap,&ranges));
  PetscCall(PetscBLASIntCast(ranges[1],&mb));  /* x block size */
  xlld = PetscMax(1,A->rmap->n);
  PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(xdesc,&a->M,&one,&mb,&one,&zero,&zero,&a->grid->ictxcol,&xlld,&info));
  PetscCheckScaLapackInfo("descinit",info);

  /* allocate 2d vector */
  lszx = SCALAPACKnumroc_(&a->M,&a->mb,&a->grid->myrow,&a->rsrc,&a->grid->nprow);
  PetscCall(PetscMalloc1(lszx,&x2d));
  xlld = PetscMax(1,lszx);

  /* create ScaLAPACK descriptor for a vector (2d block distribution) */
  PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(x2desc,&a->M,&one,&a->mb,&one,&zero,&zero,&a->grid->ictxt,&xlld,&info));
  PetscCheckScaLapackInfo("descinit",info);

  /* redistribute x as a column of a 2d matrix */
  PetscStackCallBLAS("SCALAPACKgemr2d",SCALAPACKgemr2d_(&a->M,&one,x,&one,&one,xdesc,x2d,&one,&one,x2desc,&a->grid->ictxcol));

  /* call ScaLAPACK subroutine */
  switch (A->factortype) {
    case MAT_FACTOR_LU:
      PetscStackCallBLAS("SCALAPACKgetrs",SCALAPACKgetrs_("N",&a->M,&nrhs,a->loc,&one,&one,a->desc,a->pivots,x2d,&one,&one,x2desc,&info));
      PetscCheckScaLapackInfo("getrs",info);
      break;
    case MAT_FACTOR_CHOLESKY:
      PetscStackCallBLAS("SCALAPACKpotrs",SCALAPACKpotrs_("L",&a->M,&nrhs,a->loc,&one,&one,a->desc,x2d,&one,&one,x2desc,&info));
      PetscCheckScaLapackInfo("potrs",info);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unfactored Matrix or Unsupported MatFactorType");
  }

  /* redistribute x from a column of a 2d matrix */
  PetscStackCallBLAS("SCALAPACKgemr2d",SCALAPACKgemr2d_(&a->M,&one,x2d,&one,&one,x2desc,x,&one,&one,xdesc,&a->grid->ictxcol));

  PetscCall(PetscFree(x2d));
  PetscCall(VecRestoreArray(X,&x));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveAdd_ScaLAPACK(Mat A,Vec B,Vec Y,Vec X)
{
  PetscFunctionBegin;
  PetscCall(MatSolve_ScaLAPACK(A,B,X));
  PetscCall(VecAXPY(X,1,Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_ScaLAPACK(Mat A,Mat B,Mat X)
{
  Mat_ScaLAPACK *a = (Mat_ScaLAPACK*)A->data,*b,*x;
  PetscBool      flg1,flg2;
  PetscBLASInt   one=1,info;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)B,MATSCALAPACK,&flg1));
  PetscCall(PetscObjectTypeCompare((PetscObject)X,MATSCALAPACK,&flg2));
  PetscCheckFalse(!(flg1 && flg2),PETSC_COMM_SELF,PETSC_ERR_SUP,"Both B and X must be of type MATSCALAPACK");
  MatScaLAPACKCheckDistribution(B,1,X,2);
  b = (Mat_ScaLAPACK*)B->data;
  x = (Mat_ScaLAPACK*)X->data;
  PetscCall(PetscArraycpy(x->loc,b->loc,b->lld*b->locc));

  switch (A->factortype) {
    case MAT_FACTOR_LU:
      PetscStackCallBLAS("SCALAPACKgetrs",SCALAPACKgetrs_("N",&a->M,&x->N,a->loc,&one,&one,a->desc,a->pivots,x->loc,&one,&one,x->desc,&info));
      PetscCheckScaLapackInfo("getrs",info);
      break;
    case MAT_FACTOR_CHOLESKY:
      PetscStackCallBLAS("SCALAPACKpotrs",SCALAPACKpotrs_("L",&a->M,&x->N,a->loc,&one,&one,a->desc,x->loc,&one,&one,x->desc,&info));
      PetscCheckScaLapackInfo("potrs",info);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unfactored Matrix or Unsupported MatFactorType");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactor_ScaLAPACK(Mat A,IS row,IS col,const MatFactorInfo *factorinfo)
{
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data;
  PetscBLASInt   one=1,info;

  PetscFunctionBegin;
  if (!a->pivots) {
    PetscCall(PetscMalloc1(a->locr+a->mb,&a->pivots));
    PetscCall(PetscLogObjectMemory((PetscObject)A,a->locr*sizeof(PetscBLASInt)));
  }
  PetscStackCallBLAS("SCALAPACKgetrf",SCALAPACKgetrf_(&a->M,&a->N,a->loc,&one,&one,a->desc,a->pivots,&info));
  PetscCheckScaLapackInfo("getrf",info);
  A->factortype = MAT_FACTOR_LU;
  A->assembled  = PETSC_TRUE;

  PetscCall(PetscFree(A->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERSCALAPACK,&A->solvertype));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorNumeric_ScaLAPACK(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatCopy(A,F,SAME_NONZERO_PATTERN));
  PetscCall(MatLUFactor_ScaLAPACK(F,0,0,info));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorSymbolic_ScaLAPACK(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  /* F is created and allocated by MatGetFactor_scalapack_petsc(), skip this routine. */
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactor_ScaLAPACK(Mat A,IS perm,const MatFactorInfo *factorinfo)
{
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data;
  PetscBLASInt   one=1,info;

  PetscFunctionBegin;
  PetscStackCallBLAS("SCALAPACKpotrf",SCALAPACKpotrf_("L",&a->M,a->loc,&one,&one,a->desc,&info));
  PetscCheckScaLapackInfo("potrf",info);
  A->factortype = MAT_FACTOR_CHOLESKY;
  A->assembled  = PETSC_TRUE;

  PetscCall(PetscFree(A->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERSCALAPACK,&A->solvertype));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorNumeric_ScaLAPACK(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatCopy(A,F,SAME_NONZERO_PATTERN));
  PetscCall(MatCholeskyFactor_ScaLAPACK(F,0,info));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorSymbolic_ScaLAPACK(Mat F,Mat A,IS perm,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  /* F is created and allocated by MatGetFactor_scalapack_petsc(), skip this routine. */
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorGetSolverType_scalapack_scalapack(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSCALAPACK;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetFactor_scalapack_scalapack(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  PetscCall(MatCreateScaLAPACK(PetscObjectComm((PetscObject)A),a->mb,a->nb,a->M,a->N,a->rsrc,a->csrc,&B));
  B->trivialsymbolic = PETSC_TRUE;
  B->factortype = ftype;
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERSCALAPACK,&B->solvertype));

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_scalapack_scalapack));
  *F = B;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_ScaLAPACK(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERSCALAPACK,MATSCALAPACK,MAT_FACTOR_LU,MatGetFactor_scalapack_scalapack));
  PetscCall(MatSolverTypeRegister(MATSOLVERSCALAPACK,MATSCALAPACK,MAT_FACTOR_CHOLESKY,MatGetFactor_scalapack_scalapack));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNorm_ScaLAPACK(Mat A,NormType type,PetscReal *nrm)
{
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data;
  PetscBLASInt   one=1,lwork=0;
  const char     *ntype;
  PetscScalar    *work=NULL,dummy;

  PetscFunctionBegin;
  switch (type) {
    case NORM_1:
      ntype = "1";
      lwork = PetscMax(a->locr,a->locc);
      break;
    case NORM_FROBENIUS:
      ntype = "F";
      work  = &dummy;
      break;
    case NORM_INFINITY:
      ntype = "I";
      lwork = PetscMax(a->locr,a->locc);
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Unsupported norm type");
  }
  if (lwork) PetscCall(PetscMalloc1(lwork,&work));
  *nrm = SCALAPACKlange_(ntype,&a->M,&a->N,a->loc,&one,&one,a->desc,work);
  if (lwork) PetscCall(PetscFree(work));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_ScaLAPACK(Mat A)
{
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data;

  PetscFunctionBegin;
  PetscCall(PetscArrayzero(a->loc,a->lld*a->locc));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetOwnershipIS_ScaLAPACK(Mat A,IS *rows,IS *cols)
{
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data;
  PetscInt       i,n,nb,isrc,nproc,iproc,*idx;

  PetscFunctionBegin;
  if (rows) {
    n     = a->locr;
    nb    = a->mb;
    isrc  = a->rsrc;
    nproc = a->grid->nprow;
    iproc = a->grid->myrow;
    PetscCall(PetscMalloc1(n,&idx));
    for (i=0;i<n;i++) idx[i] = nproc*nb*(i/nb) + i%nb + ((nproc+iproc-isrc)%nproc)*nb;
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,n,idx,PETSC_OWN_POINTER,rows));
  }
  if (cols) {
    n     = a->locc;
    nb    = a->nb;
    isrc  = a->csrc;
    nproc = a->grid->npcol;
    iproc = a->grid->mycol;
    PetscCall(PetscMalloc1(n,&idx));
    for (i=0;i<n;i++) idx[i] = nproc*nb*(i/nb) + i%nb + ((nproc+iproc-isrc)%nproc)*nb;
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,n,idx,PETSC_OWN_POINTER,cols));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConvert_ScaLAPACK_Dense(Mat A,MatType newtype,MatReuse reuse,Mat *B)
{
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data;
  Mat            Bmpi;
  MPI_Comm       comm;
  PetscInt       i,M=A->rmap->N,N=A->cmap->N,m,n,rstart,rend,nz;
  const PetscInt *ranges,*branges,*cwork;
  const PetscScalar *vwork;
  PetscBLASInt   bdesc[9],bmb,zero=0,one=1,lld,info;
  PetscScalar    *barray;
  PetscBool      differ=PETSC_FALSE;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCall(PetscLayoutGetRanges(A->rmap,&ranges));

  if (reuse == MAT_REUSE_MATRIX) { /* check if local sizes differ in A and B */
    PetscCallMPI(MPI_Comm_size(comm,&size));
    PetscCall(PetscLayoutGetRanges((*B)->rmap,&branges));
    for (i=0;i<size;i++) if (ranges[i+1]!=branges[i+1]) { differ=PETSC_TRUE; break; }
  }

  if (reuse == MAT_REUSE_MATRIX && differ) { /* special case, use auxiliary dense matrix */
    PetscCall(MatCreate(comm,&Bmpi));
    m = PETSC_DECIDE;
    PetscCall(PetscSplitOwnershipEqual(comm,&m,&M));
    n = PETSC_DECIDE;
    PetscCall(PetscSplitOwnershipEqual(comm,&n,&N));
    PetscCall(MatSetSizes(Bmpi,m,n,M,N));
    PetscCall(MatSetType(Bmpi,MATDENSE));
    PetscCall(MatSetUp(Bmpi));

    /* create ScaLAPACK descriptor for B (1d block distribution) */
    PetscCall(PetscBLASIntCast(ranges[1],&bmb));  /* row block size */
    lld = PetscMax(A->rmap->n,1);  /* local leading dimension */
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(bdesc,&a->M,&a->N,&bmb,&a->N,&zero,&zero,&a->grid->ictxcol,&lld,&info));
    PetscCheckScaLapackInfo("descinit",info);

    /* redistribute matrix */
    PetscCall(MatDenseGetArray(Bmpi,&barray));
    PetscStackCallBLAS("SCALAPACKgemr2d",SCALAPACKgemr2d_(&a->M,&a->N,a->loc,&one,&one,a->desc,barray,&one,&one,bdesc,&a->grid->ictxcol));
    PetscCall(MatDenseRestoreArray(Bmpi,&barray));
    PetscCall(MatAssemblyBegin(Bmpi,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Bmpi,MAT_FINAL_ASSEMBLY));

    /* transfer rows of auxiliary matrix to the final matrix B */
    PetscCall(MatGetOwnershipRange(Bmpi,&rstart,&rend));
    for (i=rstart;i<rend;i++) {
      PetscCall(MatGetRow(Bmpi,i,&nz,&cwork,&vwork));
      PetscCall(MatSetValues(*B,1,&i,nz,cwork,vwork,INSERT_VALUES));
      PetscCall(MatRestoreRow(Bmpi,i,&nz,&cwork,&vwork));
    }
    PetscCall(MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatDestroy(&Bmpi));

  } else {  /* normal cases */

    if (reuse == MAT_REUSE_MATRIX) Bmpi = *B;
    else {
      PetscCall(MatCreate(comm,&Bmpi));
      m = PETSC_DECIDE;
      PetscCall(PetscSplitOwnershipEqual(comm,&m,&M));
      n = PETSC_DECIDE;
      PetscCall(PetscSplitOwnershipEqual(comm,&n,&N));
      PetscCall(MatSetSizes(Bmpi,m,n,M,N));
      PetscCall(MatSetType(Bmpi,MATDENSE));
      PetscCall(MatSetUp(Bmpi));
    }

    /* create ScaLAPACK descriptor for B (1d block distribution) */
    PetscCall(PetscBLASIntCast(ranges[1],&bmb));  /* row block size */
    lld = PetscMax(A->rmap->n,1);  /* local leading dimension */
    PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(bdesc,&a->M,&a->N,&bmb,&a->N,&zero,&zero,&a->grid->ictxcol,&lld,&info));
    PetscCheckScaLapackInfo("descinit",info);

    /* redistribute matrix */
    PetscCall(MatDenseGetArray(Bmpi,&barray));
    PetscStackCallBLAS("SCALAPACKgemr2d",SCALAPACKgemr2d_(&a->M,&a->N,a->loc,&one,&one,a->desc,barray,&one,&one,bdesc,&a->grid->ictxcol));
    PetscCall(MatDenseRestoreArray(Bmpi,&barray));

    PetscCall(MatAssemblyBegin(Bmpi,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Bmpi,MAT_FINAL_ASSEMBLY));
    if (reuse == MAT_INPLACE_MATRIX) {
      PetscCall(MatHeaderReplace(A,&Bmpi));
    } else *B = Bmpi;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_Dense_ScaLAPACK(Mat A,MatType newtype,MatReuse reuse,Mat *B)
{
  Mat_ScaLAPACK  *b;
  Mat            Bmpi;
  MPI_Comm       comm;
  PetscInt       M=A->rmap->N,N=A->cmap->N,m,n;
  const PetscInt *ranges;
  PetscBLASInt   adesc[9],amb,zero=0,one=1,lld,info;
  PetscScalar    *aarray;
  PetscInt       lda;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));

  if (reuse == MAT_REUSE_MATRIX) Bmpi = *B;
  else {
    PetscCall(MatCreate(comm,&Bmpi));
    m = PETSC_DECIDE;
    PetscCall(PetscSplitOwnershipEqual(comm,&m,&M));
    n = PETSC_DECIDE;
    PetscCall(PetscSplitOwnershipEqual(comm,&n,&N));
    PetscCall(MatSetSizes(Bmpi,m,n,M,N));
    PetscCall(MatSetType(Bmpi,MATSCALAPACK));
    PetscCall(MatSetUp(Bmpi));
  }
  b = (Mat_ScaLAPACK*)Bmpi->data;

  /* create ScaLAPACK descriptor for A (1d block distribution) */
  PetscCall(PetscLayoutGetRanges(A->rmap,&ranges));
  PetscCall(PetscBLASIntCast(ranges[1],&amb));  /* row block size */
  PetscCall(MatDenseGetLDA(A,&lda));
  lld = PetscMax(lda,1);  /* local leading dimension */
  PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(adesc,&b->M,&b->N,&amb,&b->N,&zero,&zero,&b->grid->ictxcol,&lld,&info));
  PetscCheckScaLapackInfo("descinit",info);

  /* redistribute matrix */
  PetscCall(MatDenseGetArray(A,&aarray));
  PetscStackCallBLAS("SCALAPACKgemr2d",SCALAPACKgemr2d_(&b->M,&b->N,aarray,&one,&one,adesc,b->loc,&one,&one,b->desc,&b->grid->ictxcol));
  PetscCall(MatDenseRestoreArray(A,&aarray));

  PetscCall(MatAssemblyBegin(Bmpi,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Bmpi,MAT_FINAL_ASSEMBLY));
  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&Bmpi));
  } else *B = Bmpi;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_AIJ_ScaLAPACK(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat               mat_scal;
  PetscInt          M=A->rmap->N,N=A->cmap->N,rstart=A->rmap->rstart,rend=A->rmap->rend,m,n,row,ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    mat_scal = *newmat;
    PetscCall(MatZeroEntries(mat_scal));
  } else {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&mat_scal));
    m = PETSC_DECIDE;
    PetscCall(PetscSplitOwnershipEqual(PetscObjectComm((PetscObject)A),&m,&M));
    n = PETSC_DECIDE;
    PetscCall(PetscSplitOwnershipEqual(PetscObjectComm((PetscObject)A),&n,&N));
    PetscCall(MatSetSizes(mat_scal,m,n,M,N));
    PetscCall(MatSetType(mat_scal,MATSCALAPACK));
    PetscCall(MatSetUp(mat_scal));
  }
  for (row=rstart;row<rend;row++) {
    PetscCall(MatGetRow(A,row,&ncols,&cols,&vals));
    PetscCall(MatSetValues(mat_scal,1,&row,ncols,cols,vals,INSERT_VALUES));
    PetscCall(MatRestoreRow(A,row,&ncols,&cols,&vals));
  }
  PetscCall(MatAssemblyBegin(mat_scal,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat_scal,MAT_FINAL_ASSEMBLY));

  if (reuse == MAT_INPLACE_MATRIX) PetscCall(MatHeaderReplace(A,&mat_scal));
  else *newmat = mat_scal;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SBAIJ_ScaLAPACK(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat               mat_scal;
  PetscInt          M=A->rmap->N,N=A->cmap->N,m,n,row,ncols,j,rstart=A->rmap->rstart,rend=A->rmap->rend;
  const PetscInt    *cols;
  const PetscScalar *vals;
  PetscScalar       v;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    mat_scal = *newmat;
    PetscCall(MatZeroEntries(mat_scal));
  } else {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&mat_scal));
    m = PETSC_DECIDE;
    PetscCall(PetscSplitOwnershipEqual(PetscObjectComm((PetscObject)A),&m,&M));
    n = PETSC_DECIDE;
    PetscCall(PetscSplitOwnershipEqual(PetscObjectComm((PetscObject)A),&n,&N));
    PetscCall(MatSetSizes(mat_scal,m,n,M,N));
    PetscCall(MatSetType(mat_scal,MATSCALAPACK));
    PetscCall(MatSetUp(mat_scal));
  }
  PetscCall(MatGetRowUpperTriangular(A));
  for (row=rstart;row<rend;row++) {
    PetscCall(MatGetRow(A,row,&ncols,&cols,&vals));
    PetscCall(MatSetValues(mat_scal,1,&row,ncols,cols,vals,ADD_VALUES));
    for (j=0;j<ncols;j++) { /* lower triangular part */
      if (cols[j] == row) continue;
      v    = A->hermitian ? PetscConj(vals[j]) : vals[j];
      PetscCall(MatSetValues(mat_scal,1,&cols[j],1,&row,&v,ADD_VALUES));
    }
    PetscCall(MatRestoreRow(A,row,&ncols,&cols,&vals));
  }
  PetscCall(MatRestoreRowUpperTriangular(A));
  PetscCall(MatAssemblyBegin(mat_scal,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat_scal,MAT_FINAL_ASSEMBLY));

  if (reuse == MAT_INPLACE_MATRIX) PetscCall(MatHeaderReplace(A,&mat_scal));
  else *newmat = mat_scal;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScaLAPACKSetPreallocation(Mat A)
{
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data;
  PetscInt       sz=0;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  if (!a->lld) a->lld = a->locr;

  PetscCall(PetscFree(a->loc));
  PetscCall(PetscIntMultError(a->lld,a->locc,&sz));
  PetscCall(PetscCalloc1(sz,&a->loc));
  PetscCall(PetscLogObjectMemory((PetscObject)A,sz*sizeof(PetscScalar)));

  A->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_ScaLAPACK(Mat A)
{
  Mat_ScaLAPACK      *a = (Mat_ScaLAPACK*)A->data;
  Mat_ScaLAPACK_Grid *grid;
  PetscBool          flg;
  MPI_Comm           icomm;

  PetscFunctionBegin;
  PetscCall(MatStashDestroy_Private(&A->stash));
  PetscCall(PetscFree(a->loc));
  PetscCall(PetscFree(a->pivots));
  PetscCall(PetscCommDuplicate(PetscObjectComm((PetscObject)A),&icomm,NULL));
  PetscCallMPI(MPI_Comm_get_attr(icomm,Petsc_ScaLAPACK_keyval,(void**)&grid,(int*)&flg));
  if (--grid->grid_refct == 0) {
    Cblacs_gridexit(grid->ictxt);
    Cblacs_gridexit(grid->ictxrow);
    Cblacs_gridexit(grid->ictxcol);
    PetscCall(PetscFree(grid));
    PetscCallMPI(MPI_Comm_delete_attr(icomm,Petsc_ScaLAPACK_keyval));
  }
  PetscCall(PetscCommDestroy(&icomm));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatGetOwnershipIS_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatScaLAPACKSetBlockSizes_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatScaLAPACKGetBlockSizes_C",NULL));
  PetscCall(PetscFree(A->data));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode MatScaLAPACKCheckLayout(PetscLayout map)
{
  const PetscInt *ranges;
  PetscMPIInt    size;
  PetscInt       i,n;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(map->comm,&size));
  if (size>2) {
    PetscCall(PetscLayoutGetRanges(map,&ranges));
    n = ranges[1]-ranges[0];
    for (i=1;i<size-1;i++) if (ranges[i+1]-ranges[i]!=n) break;
    PetscCheckFalse(i<size-1 && ranges[i+1]-ranges[i]!=0 && ranges[i+2]-ranges[i+1]!=0,map->comm,PETSC_ERR_SUP,"MATSCALAPACK must have equal local sizes in all processes (except possibly the last one), consider using MatCreateScaLAPACK");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_ScaLAPACK(Mat A)
{
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data;
  PetscBLASInt   info=0;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));

  /* check that the layout is as enforced by MatCreateScaLAPACK */
  PetscCall(MatScaLAPACKCheckLayout(A->rmap));
  PetscCall(MatScaLAPACKCheckLayout(A->cmap));

  /* compute local sizes */
  PetscCall(PetscBLASIntCast(A->rmap->N,&a->M));
  PetscCall(PetscBLASIntCast(A->cmap->N,&a->N));
  a->locr = SCALAPACKnumroc_(&a->M,&a->mb,&a->grid->myrow,&a->rsrc,&a->grid->nprow);
  a->locc = SCALAPACKnumroc_(&a->N,&a->nb,&a->grid->mycol,&a->csrc,&a->grid->npcol);
  a->lld  = PetscMax(1,a->locr);

  /* allocate local array */
  PetscCall(MatScaLAPACKSetPreallocation(A));

  /* set up ScaLAPACK descriptor */
  PetscStackCallBLAS("SCALAPACKdescinit",SCALAPACKdescinit_(a->desc,&a->M,&a->N,&a->mb,&a->nb,&a->rsrc,&a->csrc,&a->grid->ictxt,&a->lld,&info));
  PetscCheckScaLapackInfo("descinit",info);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyBegin_ScaLAPACK(Mat A,MatAssemblyType type)
{
  PetscInt       nstash,reallocs;

  PetscFunctionBegin;
  if (A->nooffprocentries) PetscFunctionReturn(0);
  PetscCall(MatStashScatterBegin_Private(A,&A->stash,NULL));
  PetscCall(MatStashGetInfo_Private(&A->stash,&nstash,&reallocs));
  PetscCall(PetscInfo(A,"Stash has %" PetscInt_FMT " entries, uses %" PetscInt_FMT " mallocs.\n",nstash,reallocs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_ScaLAPACK(Mat A,MatAssemblyType type)
{
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data;
  PetscMPIInt    n;
  PetscInt       i,flg,*row,*col;
  PetscScalar    *val;
  PetscBLASInt   gridx,gcidx,lridx,lcidx,rsrc,csrc;

  PetscFunctionBegin;
  if (A->nooffprocentries) PetscFunctionReturn(0);
  while (1) {
    PetscCall(MatStashScatterGetMesg_Private(&A->stash,&n,&row,&col,&val,&flg));
    if (!flg) break;
    for (i=0;i<n;i++) {
      PetscCall(PetscBLASIntCast(row[i]+1,&gridx));
      PetscCall(PetscBLASIntCast(col[i]+1,&gcidx));
      PetscStackCallBLAS("SCALAPACKinfog2l",SCALAPACKinfog2l_(&gridx,&gcidx,a->desc,&a->grid->nprow,&a->grid->npcol,&a->grid->myrow,&a->grid->mycol,&lridx,&lcidx,&rsrc,&csrc));
      PetscCheckFalse(rsrc!=a->grid->myrow || csrc!=a->grid->mycol,PetscObjectComm((PetscObject)A),PETSC_ERR_LIB,"Something went wrong, received value does not belong to this process");
      switch (A->insertmode) {
        case INSERT_VALUES: a->loc[lridx-1+(lcidx-1)*a->lld] = val[i]; break;
        case ADD_VALUES: a->loc[lridx-1+(lcidx-1)*a->lld] += val[i]; break;
        default: SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support for InsertMode %d",(int)A->insertmode);
      }
    }
  }
  PetscCall(MatStashScatterEnd_Private(&A->stash));
  PetscFunctionReturn(0);
}

PetscErrorCode MatLoad_ScaLAPACK(Mat newMat,PetscViewer viewer)
{
  Mat            Adense,As;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)newMat,&comm));
  PetscCall(MatCreate(comm,&Adense));
  PetscCall(MatSetType(Adense,MATDENSE));
  PetscCall(MatLoad(Adense,viewer));
  PetscCall(MatConvert(Adense,MATSCALAPACK,MAT_INITIAL_MATRIX,&As));
  PetscCall(MatDestroy(&Adense));
  PetscCall(MatHeaderReplace(newMat,&As));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {
       MatSetValues_ScaLAPACK,
       0,
       0,
       MatMult_ScaLAPACK,
/* 4*/ MatMultAdd_ScaLAPACK,
       MatMultTranspose_ScaLAPACK,
       MatMultTransposeAdd_ScaLAPACK,
       MatSolve_ScaLAPACK,
       MatSolveAdd_ScaLAPACK,
       0,
/*10*/ 0,
       MatLUFactor_ScaLAPACK,
       MatCholeskyFactor_ScaLAPACK,
       0,
       MatTranspose_ScaLAPACK,
/*15*/ MatGetInfo_ScaLAPACK,
       0,
       MatGetDiagonal_ScaLAPACK,
       MatDiagonalScale_ScaLAPACK,
       MatNorm_ScaLAPACK,
/*20*/ MatAssemblyBegin_ScaLAPACK,
       MatAssemblyEnd_ScaLAPACK,
       MatSetOption_ScaLAPACK,
       MatZeroEntries_ScaLAPACK,
/*24*/ 0,
       MatLUFactorSymbolic_ScaLAPACK,
       MatLUFactorNumeric_ScaLAPACK,
       MatCholeskyFactorSymbolic_ScaLAPACK,
       MatCholeskyFactorNumeric_ScaLAPACK,
/*29*/ MatSetUp_ScaLAPACK,
       0,
       0,
       0,
       0,
/*34*/ MatDuplicate_ScaLAPACK,
       0,
       0,
       0,
       0,
/*39*/ MatAXPY_ScaLAPACK,
       0,
       0,
       0,
       MatCopy_ScaLAPACK,
/*44*/ 0,
       MatScale_ScaLAPACK,
       MatShift_ScaLAPACK,
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
       MatDestroy_ScaLAPACK,
       MatView_ScaLAPACK,
       0,
       0,
/*64*/ 0,
       0,
       0,
       0,
       0,
/*69*/ 0,
       0,
       MatConvert_ScaLAPACK_Dense,
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
       MatLoad_ScaLAPACK,
/*84*/ 0,
       0,
       0,
       0,
       0,
/*89*/ 0,
       0,
       MatMatMultNumeric_ScaLAPACK,
       0,
       0,
/*94*/ 0,
       0,
       0,
       MatMatTransposeMultNumeric_ScaLAPACK,
       0,
/*99*/ MatProductSetFromOptions_ScaLAPACK,
       0,
       0,
       MatConjugate_ScaLAPACK,
       0,
/*104*/0,
       0,
       0,
       0,
       0,
/*109*/MatMatSolve_ScaLAPACK,
       0,
       0,
       0,
       MatMissingDiagonal_ScaLAPACK,
/*114*/0,
       0,
       0,
       0,
       0,
/*119*/0,
       MatHermitianTranspose_ScaLAPACK,
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

static PetscErrorCode MatStashScatterBegin_ScaLAPACK(Mat mat,MatStash *stash,PetscInt *owners)
{
  PetscInt           *owner,*startv,*starti,tag1=stash->tag1,tag2=stash->tag2,bs2;
  PetscInt           size=stash->size,nsends;
  PetscInt           count,*sindices,**rindices,i,j,l;
  PetscScalar        **rvalues,*svalues;
  MPI_Comm           comm = stash->comm;
  MPI_Request        *send_waits,*recv_waits,*recv_waits1,*recv_waits2;
  PetscMPIInt        *sizes,*nlengths,nreceives;
  PetscInt           *sp_idx,*sp_idy;
  PetscScalar        *sp_val;
  PetscMatStashSpace space,space_next;
  PetscBLASInt       gridx,gcidx,lridx,lcidx,rsrc,csrc;
  Mat_ScaLAPACK      *a = (Mat_ScaLAPACK*)mat->data;

  PetscFunctionBegin;
  {                             /* make sure all processors are either in INSERTMODE or ADDMODE */
    InsertMode addv;
    PetscCall(MPIU_Allreduce((PetscEnum*)&mat->insertmode,(PetscEnum*)&addv,1,MPIU_ENUM,MPI_BOR,PetscObjectComm((PetscObject)mat)));
    PetscCheckFalse(addv == (ADD_VALUES|INSERT_VALUES),PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Some processors inserted others added");
    mat->insertmode = addv; /* in case this processor had no cache */
  }

  bs2 = stash->bs*stash->bs;

  /*  first count number of contributors to each processor */
  PetscCall(PetscCalloc1(size,&nlengths));
  PetscCall(PetscMalloc1(stash->n+1,&owner));

  i     = j    = 0;
  space = stash->space_head;
  while (space) {
    space_next = space->next;
    for (l=0; l<space->local_used; l++) {
      PetscCall(PetscBLASIntCast(space->idx[l]+1,&gridx));
      PetscCall(PetscBLASIntCast(space->idy[l]+1,&gcidx));
      PetscStackCallBLAS("SCALAPACKinfog2l",SCALAPACKinfog2l_(&gridx,&gcidx,a->desc,&a->grid->nprow,&a->grid->npcol,&a->grid->myrow,&a->grid->mycol,&lridx,&lcidx,&rsrc,&csrc));
      j = Cblacs_pnum(a->grid->ictxt,rsrc,csrc);
      nlengths[j]++; owner[i] = j;
      i++;
    }
    space = space_next;
  }

  /* Now check what procs get messages - and compute nsends. */
  PetscCall(PetscCalloc1(size,&sizes));
  for (i=0, nsends=0; i<size; i++) {
    if (nlengths[i]) {
      sizes[i] = 1; nsends++;
    }
  }

  {PetscMPIInt *onodes,*olengths;
   /* Determine the number of messages to expect, their lengths, from from-ids */
   PetscCall(PetscGatherNumberOfMessages(comm,sizes,nlengths,&nreceives));
   PetscCall(PetscGatherMessageLengths(comm,nsends,nreceives,nlengths,&onodes,&olengths));
   /* since clubbing row,col - lengths are multiplied by 2 */
   for (i=0; i<nreceives; i++) olengths[i] *=2;
   PetscCall(PetscPostIrecvInt(comm,tag1,nreceives,onodes,olengths,&rindices,&recv_waits1));
   /* values are size 'bs2' lengths (and remove earlier factor 2 */
   for (i=0; i<nreceives; i++) olengths[i] = olengths[i]*bs2/2;
   PetscCall(PetscPostIrecvScalar(comm,tag2,nreceives,onodes,olengths,&rvalues,&recv_waits2));
   PetscCall(PetscFree(onodes));
   PetscCall(PetscFree(olengths));}

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to
         the ith processor
  */
  PetscCall(PetscMalloc2(bs2*stash->n,&svalues,2*(stash->n+1),&sindices));
  PetscCall(PetscMalloc1(2*nsends,&send_waits));
  PetscCall(PetscMalloc2(size,&startv,size,&starti));
  /* use 2 sends the first with all_a, the next with all_i and all_j */
  startv[0] = 0; starti[0] = 0;
  for (i=1; i<size; i++) {
    startv[i] = startv[i-1] + nlengths[i-1];
    starti[i] = starti[i-1] + 2*nlengths[i-1];
  }

  i     = 0;
  space = stash->space_head;
  while (space) {
    space_next = space->next;
    sp_idx     = space->idx;
    sp_idy     = space->idy;
    sp_val     = space->val;
    for (l=0; l<space->local_used; l++) {
      j = owner[i];
      if (bs2 == 1) {
        svalues[startv[j]] = sp_val[l];
      } else {
        PetscInt    k;
        PetscScalar *buf1,*buf2;
        buf1 = svalues+bs2*startv[j];
        buf2 = space->val + bs2*l;
        for (k=0; k<bs2; k++) buf1[k] = buf2[k];
      }
      sindices[starti[j]]             = sp_idx[l];
      sindices[starti[j]+nlengths[j]] = sp_idy[l];
      startv[j]++;
      starti[j]++;
      i++;
    }
    space = space_next;
  }
  startv[0] = 0;
  for (i=1; i<size; i++) startv[i] = startv[i-1] + nlengths[i-1];

  for (i=0,count=0; i<size; i++) {
    if (sizes[i]) {
      PetscCallMPI(MPI_Isend(sindices+2*startv[i],2*nlengths[i],MPIU_INT,i,tag1,comm,send_waits+count++));
      PetscCallMPI(MPI_Isend(svalues+bs2*startv[i],bs2*nlengths[i],MPIU_SCALAR,i,tag2,comm,send_waits+count++));
    }
  }
#if defined(PETSC_USE_INFO)
  PetscCall(PetscInfo(NULL,"No of messages: %" PetscInt_FMT "\n",nsends));
  for (i=0; i<size; i++) {
    if (sizes[i]) {
      PetscCall(PetscInfo(NULL,"Mesg_to: %" PetscInt_FMT ": size: %zu bytes\n",i,(size_t)(nlengths[i]*(bs2*sizeof(PetscScalar)+2*sizeof(PetscInt)))));
    }
  }
#endif
  PetscCall(PetscFree(nlengths));
  PetscCall(PetscFree(owner));
  PetscCall(PetscFree2(startv,starti));
  PetscCall(PetscFree(sizes));

  /* recv_waits need to be contiguous for MatStashScatterGetMesg_Private() */
  PetscCall(PetscMalloc1(2*nreceives,&recv_waits));

  for (i=0; i<nreceives; i++) {
    recv_waits[2*i]   = recv_waits1[i];
    recv_waits[2*i+1] = recv_waits2[i];
  }
  stash->recv_waits = recv_waits;

  PetscCall(PetscFree(recv_waits1));
  PetscCall(PetscFree(recv_waits2));

  stash->svalues         = svalues;
  stash->sindices        = sindices;
  stash->rvalues         = rvalues;
  stash->rindices        = rindices;
  stash->send_waits      = send_waits;
  stash->nsends          = nsends;
  stash->nrecvs          = nreceives;
  stash->reproduce_count = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScaLAPACKSetBlockSizes_ScaLAPACK(Mat A,PetscInt mb,PetscInt nb)
{
  Mat_ScaLAPACK  *a = (Mat_ScaLAPACK*)A->data;

  PetscFunctionBegin;
  PetscCheck(!A->preallocated,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Cannot change block sizes after MatSetUp");
  PetscCheckFalse(mb<1 && mb!=PETSC_DECIDE,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"mb %" PetscInt_FMT " must be at least 1",mb);
  PetscCheckFalse(nb<1 && nb!=PETSC_DECIDE,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"nb %" PetscInt_FMT " must be at least 1",nb);
  PetscCall(PetscBLASIntCast((mb==PETSC_DECIDE)?DEFAULT_BLOCKSIZE:mb,&a->mb));
  PetscCall(PetscBLASIntCast((nb==PETSC_DECIDE)?a->mb:nb,&a->nb));
  PetscFunctionReturn(0);
}

/*@
   MatScaLAPACKSetBlockSizes - Sets the block sizes to be used for the distribution of
   the ScaLAPACK matrix

   Logically Collective on A

   Input Parameters:
+  A  - a MATSCALAPACK matrix
.  mb - the row block size
-  nb - the column block size

   Level: intermediate

.seealso: MatCreateScaLAPACK(), MatScaLAPACKGetBlockSizes()
@*/
PetscErrorCode MatScaLAPACKSetBlockSizes(Mat A,PetscInt mb,PetscInt nb)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(A,mb,2);
  PetscValidLogicalCollectiveInt(A,nb,3);
  PetscTryMethod(A,"MatScaLAPACKSetBlockSizes_C",(Mat,PetscInt,PetscInt),(A,mb,nb));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScaLAPACKGetBlockSizes_ScaLAPACK(Mat A,PetscInt *mb,PetscInt *nb)
{
  Mat_ScaLAPACK *a = (Mat_ScaLAPACK*)A->data;

  PetscFunctionBegin;
  if (mb) *mb = a->mb;
  if (nb) *nb = a->nb;
  PetscFunctionReturn(0);
}

/*@
   MatScaLAPACKGetBlockSizes - Gets the block sizes used in the distribution of
   the ScaLAPACK matrix

   Not collective

   Input Parameter:
.  A  - a MATSCALAPACK matrix

   Output Parameters:
+  mb - the row block size
-  nb - the column block size

   Level: intermediate

.seealso: MatCreateScaLAPACK(), MatScaLAPACKSetBlockSizes()
@*/
PetscErrorCode MatScaLAPACKGetBlockSizes(Mat A,PetscInt *mb,PetscInt *nb)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscUseMethod(A,"MatScaLAPACKGetBlockSizes_C",(Mat,PetscInt*,PetscInt*),(A,mb,nb));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatStashScatterGetMesg_Ref(MatStash*,PetscMPIInt*,PetscInt**,PetscInt**,PetscScalar**,PetscInt*);
PETSC_INTERN PetscErrorCode MatStashScatterEnd_Ref(MatStash*);

/*MC
   MATSCALAPACK = "scalapack" - A matrix type for dense matrices using the ScaLAPACK package

   Use ./configure --download-scalapack to install PETSc to use ScaLAPACK

   Use -pc_type lu -pc_factor_mat_solver_type scalapack to use this direct solver

   Options Database Keys:
+  -mat_type scalapack - sets the matrix type to "scalapack" during a call to MatSetFromOptions()
.  -mat_scalapack_grid_height - sets Grid Height for 2D cyclic ordering of internal matrix
-  -mat_scalapack_block_sizes - size of the blocks to use (one or two integers separated by comma)

   Level: beginner

.seealso: MATDENSE, MATELEMENTAL
M*/

PETSC_EXTERN PetscErrorCode MatCreate_ScaLAPACK(Mat A)
{
  Mat_ScaLAPACK      *a;
  PetscErrorCode     ierr;
  PetscBool          flg,flg1;
  Mat_ScaLAPACK_Grid *grid;
  MPI_Comm           icomm;
  PetscBLASInt       nprow,npcol,myrow,mycol;
  PetscInt           optv1,k=2,array[2]={0,0};
  PetscMPIInt        size;

  PetscFunctionBegin;
  PetscCall(PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps)));
  A->insertmode = NOT_SET_VALUES;

  PetscCall(MatStashCreate_Private(PetscObjectComm((PetscObject)A),1,&A->stash));
  A->stash.ScatterBegin   = MatStashScatterBegin_ScaLAPACK;
  A->stash.ScatterGetMesg = MatStashScatterGetMesg_Ref;
  A->stash.ScatterEnd     = MatStashScatterEnd_Ref;
  A->stash.ScatterDestroy = NULL;

  PetscCall(PetscNewLog(A,&a));
  A->data = (void*)a;

  /* Grid needs to be shared between multiple Mats on the same communicator, implement by attribute caching on the MPI_Comm */
  if (Petsc_ScaLAPACK_keyval == MPI_KEYVAL_INVALID) {
    PetscCallMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,MPI_COMM_NULL_DELETE_FN,&Petsc_ScaLAPACK_keyval,(void*)0));
    PetscCall(PetscRegisterFinalize(Petsc_ScaLAPACK_keyval_free));
    PetscCall(PetscCitationsRegister(ScaLAPACKCitation,&ScaLAPACKCite));
  }
  PetscCall(PetscCommDuplicate(PetscObjectComm((PetscObject)A),&icomm,NULL));
  PetscCallMPI(MPI_Comm_get_attr(icomm,Petsc_ScaLAPACK_keyval,(void**)&grid,(int*)&flg));
  if (!flg) {
    PetscCall(PetscNewLog(A,&grid));

    PetscCallMPI(MPI_Comm_size(icomm,&size));
    grid->nprow = (PetscInt) (PetscSqrtReal((PetscReal)size) + 0.001);

    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"ScaLAPACK Grid Options","Mat");PetscCall(ierr);
    PetscCall(PetscOptionsInt("-mat_scalapack_grid_height","Grid Height","None",grid->nprow,&optv1,&flg1));
    if (flg1) {
      PetscCheckFalse(size % optv1,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Grid Height %" PetscInt_FMT " must evenly divide CommSize %d",optv1,size);
      grid->nprow = optv1;
    }
    ierr = PetscOptionsEnd();PetscCall(ierr);

    if (size % grid->nprow) grid->nprow = 1;  /* cannot use a squarish grid, use a 1d grid */
    grid->npcol = size/grid->nprow;
    PetscCall(PetscBLASIntCast(grid->nprow,&nprow));
    PetscCall(PetscBLASIntCast(grid->npcol,&npcol));
    grid->ictxt = Csys2blacs_handle(icomm);
    Cblacs_gridinit(&grid->ictxt,"R",nprow,npcol);
    Cblacs_gridinfo(grid->ictxt,&nprow,&npcol,&myrow,&mycol);
    grid->grid_refct = 1;
    grid->nprow      = nprow;
    grid->npcol      = npcol;
    grid->myrow      = myrow;
    grid->mycol      = mycol;
    /* auxiliary 1d BLACS contexts for 1xsize and sizex1 grids */
    grid->ictxrow = Csys2blacs_handle(icomm);
    Cblacs_gridinit(&grid->ictxrow,"R",1,size);
    grid->ictxcol = Csys2blacs_handle(icomm);
    Cblacs_gridinit(&grid->ictxcol,"R",size,1);
    PetscCallMPI(MPI_Comm_set_attr(icomm,Petsc_ScaLAPACK_keyval,(void*)grid));

  } else grid->grid_refct++;
  PetscCall(PetscCommDestroy(&icomm));
  a->grid = grid;
  a->mb   = DEFAULT_BLOCKSIZE;
  a->nb   = DEFAULT_BLOCKSIZE;

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),NULL,"ScaLAPACK Options","Mat");PetscCall(ierr);
  PetscCall(PetscOptionsIntArray("-mat_scalapack_block_sizes","Size of the blocks to use (one or two comma-separated integers)","MatCreateScaLAPACK",array,&k,&flg));
  if (flg) {
    a->mb = array[0];
    a->nb = (k>1)? array[1]: a->mb;
  }
  ierr = PetscOptionsEnd();PetscCall(ierr);

  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatGetOwnershipIS_C",MatGetOwnershipIS_ScaLAPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatScaLAPACKSetBlockSizes_C",MatScaLAPACKSetBlockSizes_ScaLAPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatScaLAPACKGetBlockSizes_C",MatScaLAPACKGetBlockSizes_ScaLAPACK));
  PetscCall(PetscObjectChangeTypeName((PetscObject)A,MATSCALAPACK));
  PetscFunctionReturn(0);
}

/*@C
   MatCreateScaLAPACK - Creates a dense parallel matrix in ScaLAPACK format
   (2D block cyclic distribution).

   Collective

   Input Parameters:
+  comm - MPI communicator
.  mb   - row block size (or PETSC_DECIDE to have it set)
.  nb   - column block size (or PETSC_DECIDE to have it set)
.  M    - number of global rows
.  N    - number of global columns
.  rsrc - coordinate of process that owns the first row of the distributed matrix
-  csrc - coordinate of process that owns the first column of the distributed matrix

   Output Parameter:
.  A - the matrix

   Options Database Keys:
.  -mat_scalapack_block_sizes - size of the blocks to use (one or two integers separated by comma)

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   If PETSC_DECIDE is used for the block sizes, then an appropriate value
   is chosen.

   Storage Information:
   Storate is completely managed by ScaLAPACK, so this requires PETSc to be
   configured with ScaLAPACK. In particular, PETSc's local sizes lose
   significance and are thus ignored. The block sizes refer to the values
   used for the distributed matrix, not the same meaning as in BAIJ.

   Level: intermediate

.seealso: MatCreate(), MatCreateDense(), MatSetValues()
@*/
PetscErrorCode MatCreateScaLAPACK(MPI_Comm comm,PetscInt mb,PetscInt nb,PetscInt M,PetscInt N,PetscInt rsrc,PetscInt csrc,Mat *A)
{
  Mat_ScaLAPACK  *a;
  PetscInt       m,n;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetType(*A,MATSCALAPACK));
  PetscCheckFalse(M==PETSC_DECIDE || N==PETSC_DECIDE,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot use PETSC_DECIDE for matrix dimensions");
  /* rows and columns are NOT distributed according to PetscSplitOwnership */
  m = PETSC_DECIDE;
  PetscCall(PetscSplitOwnershipEqual(comm,&m,&M));
  n = PETSC_DECIDE;
  PetscCall(PetscSplitOwnershipEqual(comm,&n,&N));
  PetscCall(MatSetSizes(*A,m,n,M,N));
  a = (Mat_ScaLAPACK*)(*A)->data;
  PetscCall(PetscBLASIntCast(M,&a->M));
  PetscCall(PetscBLASIntCast(N,&a->N));
  PetscCall(PetscBLASIntCast((mb==PETSC_DECIDE)?DEFAULT_BLOCKSIZE:mb,&a->mb));
  PetscCall(PetscBLASIntCast((nb==PETSC_DECIDE)?a->mb:nb,&a->nb));
  PetscCall(PetscBLASIntCast(rsrc,&a->rsrc));
  PetscCall(PetscBLASIntCast(csrc,&a->csrc));
  PetscCall(MatSetUp(*A));
  PetscFunctionReturn(0);
}
