#include <../src/mat/impls/elemental/matelemimpl.h> /*I "petscmat.h" I*/

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
  PetscBool iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscViewerFormat format;
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Info viewer not implemented yet");
    } else if (format == PETSC_VIEWER_DEFAULT) {
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscObjectPrintClassNamePrefixType((PetscObject)A,viewer,"Matrix Object");CHKERRQ(ierr);
      a->emat->Print("Elemental matrix (cyclic ordering)");
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
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
  PetscMPIInt rank;
  PetscInt i,j,rrank,ridx,crank,cidx;

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
      if (erow % grid.MCSize() != grid.MCRank() || ecol % grid.MRSize() != grid.MRRank()){
        SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for setting off-process entry (%D,%D), Elemental (%D,%D)",rows[i],cols[j],erow,ecol);
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
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  PetscErrorCode ierr;
  const PetscScalar *x;
  PetscScalar *y;
  PetscScalar one = 1,zero = 0;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  {                             /* Scoping so that constructor is called before pointer is returned */
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
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  PetscErrorCode ierr;
  const PetscScalar *x;
  PetscScalar *z;
  PetscScalar one = 1;

  PetscFunctionBegin;
  if (Y != Z) {ierr = VecCopy(Y,Z);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Z,&z);CHKERRQ(ierr);
  {                             /* Scoping so that constructor is called before pointer is returned */
    elem::DistMatrix<PetscScalar,elem::VC,elem::STAR> xe(A->cmap->N,1,0,x,A->cmap->n,*a->grid);
    elem::DistMatrix<PetscScalar,elem::VC,elem::STAR> ze(A->rmap->N,1,0,z,A->rmap->n,*a->grid);
    elem::Gemv(elem::NORMAL,one,*a->emat,xe,one,ze);
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Z,&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetOwnershipIS_Elemental"
static PetscErrorCode MatGetOwnershipIS_Elemental(Mat A,IS *rows,IS *cols)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  PetscErrorCode ierr;
  PetscInt i,m,shift,stride,*idx;

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

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Elemental"
static PetscErrorCode MatDestroy_Elemental(Mat A)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  delete a->emat;
  delete a->grid;
  ierr = MatStashDestroy_Private(&A->stash);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatGetOwnershipIS_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetUp_Elemental"
PetscErrorCode MatSetUp_Elemental(Mat A)
{
  Mat_Elemental  *a = (Mat_Elemental*)A->data;
  PetscErrorCode ierr;
  PetscMPIInt rsize,csize;

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
  Mat_Elemental  *a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscElementalInitializePackage(PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscNewLog(A,Mat_Elemental,&a);CHKERRQ(ierr);
  A->data = (void*)a;

  A->ops->view      = MatView_Elemental;
  A->ops->destroy   = MatDestroy_Elemental;
  A->ops->setup     = MatSetUp_Elemental;
  A->ops->setvalues = MatSetValues_Elemental;
  A->ops->mult      = MatMult_Elemental;
  A->ops->multadd   = MatMultAdd_Elemental;

  A->insertmode = NOT_SET_VALUES;

  /* Set up the elemental matrix */
  elem::mpi::Comm cxxcomm(((PetscObject)A)->comm);
  a->grid = new elem::Grid(cxxcomm);
  a->emat = new elem::DistMatrix<PetscScalar>(*a->grid);

  /* build cache for off array entries formed */
  ierr = MatStashCreate_Private(((PetscObject)A)->comm,1,&A->stash);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatGetOwnershipIS_C","MatGetOwnershipIS_Elemental",MatGetOwnershipIS_Elemental);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATELEMENTAL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
