
#include <private/pcimpl.h>   /*I "petscpc.h" I*/
#include <petscblaslapack.h>

/* 
   Private context (data structure) for the SVD preconditioner.  
*/
typedef struct {
  Vec        diag,work;    
  Mat        A,U,V;
  PetscInt   nzero;
} PC_SVD;


/* -------------------------------------------------------------------------- */
/*
   PCSetUp_SVD - Prepares for the use of the SVD preconditioner
                    by setting data structures and options.   

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_SVD"
static PetscErrorCode PCSetUp_SVD(PC pc)
{
#if defined(PETSC_MISSING_LAPACK_GESVD)
  SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"GESVD - Lapack routine is unavailable\nNot able to provide singular value estimates.");
#else
  PC_SVD         *jac = (PC_SVD*)pc->data;
  PetscErrorCode ierr;
  PetscScalar    *a,*u,*v,*d,*work;
  PetscBLASInt   nb,lwork,lierr;
  PetscInt       i,n;

  PetscFunctionBegin;
  if (!jac->diag) {
    /* assume square matrices */
    ierr = MatGetVecs(pc->mat,&jac->diag,&jac->work);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&jac->A);CHKERRQ(ierr);
  ierr = MatConvert(pc->mat,MATSEQDENSE,MAT_INITIAL_MATRIX,&jac->A);CHKERRQ(ierr);
  if (!jac->U) {
    ierr = MatDuplicate(jac->A,MAT_DO_NOT_COPY_VALUES,&jac->U);CHKERRQ(ierr);
    ierr = MatDuplicate(jac->A,MAT_DO_NOT_COPY_VALUES,&jac->V);CHKERRQ(ierr);
  }
  ierr = MatGetSize(pc->mat,&n,PETSC_NULL);CHKERRQ(ierr);
  nb    = PetscBLASIntCast(n);
  lwork = 5*nb;
  ierr  = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr); 
  ierr  = MatGetArray(jac->A,&a);CHKERRQ(ierr); 
  ierr  = MatGetArray(jac->U,&u);CHKERRQ(ierr); 
  ierr  = MatGetArray(jac->V,&v);CHKERRQ(ierr); 
  ierr  = VecGetArray(jac->diag,&d);CHKERRQ(ierr); 
#if !defined(PETSC_USE_COMPLEX)
  LAPACKgesvd_("A","A",&nb,&nb,a,&nb,d,u,&nb,v,&nb,work,&lwork,&lierr);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not coded for complex");
#endif
  if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"gesv() error %d",lierr);
  ierr  = MatRestoreArray(jac->A,&a);CHKERRQ(ierr); 
  ierr  = MatRestoreArray(jac->U,&u);CHKERRQ(ierr); 
  ierr  = MatRestoreArray(jac->V,&v);CHKERRQ(ierr); 
  jac->nzero = 0;
  for (i=0; i<n; i++) {
    if (PetscRealPart(d[i]) < 1.e-12) {jac->nzero = n - i;break;}
    d[i] = 1.0/d[i];
  }
  ierr = PetscInfo1(pc,"Number of zero or nearly singular values %D\n",jac->nzero);
  ierr = VecRestoreArray(jac->diag,&d);CHKERRQ(ierr);
#if defined(foo)
{
  PetscViewer viewer;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,"joe",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(jac->A,viewer);CHKERRQ(ierr);
  ierr = MatView(jac->U,viewer);CHKERRQ(ierr);
  ierr = MatView(jac->V,viewer);CHKERRQ(ierr);
  ierr = VecView(jac->diag,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
 }
#endif
  ierr = PetscFree(work);
  PetscFunctionReturn(0);
#endif
}

/* -------------------------------------------------------------------------- */
/*
   PCApply_SVD - Applies the SVD preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
#undef __FUNCT__  
#define __FUNCT__ "PCApply_SVD"
static PetscErrorCode PCApply_SVD(PC pc,Vec x,Vec y)
{
  PC_SVD         *jac = (PC_SVD*)pc->data;
  Vec            work = jac->work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultTranspose(jac->U,x,work);CHKERRQ(ierr);
  ierr = VecPointwiseMult(work,work,jac->diag);CHKERRQ(ierr);
  ierr = MatMultTranspose(jac->V,work,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCReset_SVD"
static PetscErrorCode PCReset_SVD(PC pc)
{
  PC_SVD         *jac = (PC_SVD*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&jac->A);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->U);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->V);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->diag);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCDestroy_SVD - Destroys the private context for the SVD preconditioner
   that was created with PCCreate_SVD().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_SVD"
static PetscErrorCode PCDestroy_SVD(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_SVD(pc);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_SVD"
static PetscErrorCode PCSetFromOptions_SVD(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SVD options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreate_SVD - Creates a SVD preconditioner context, PC_SVD, 
   and sets this as the private data within the generic preconditioning 
   context, PC, that was created within PCCreate().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCCreate()
*/

/*MC
     PCSVD - Use pseudo inverse defined by SVD of operator

   Level: advanced

  Concepts: SVD

         Zero entries along the diagonal are replaced with the value 0.0

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_SVD"
PetscErrorCode PCCreate_SVD(PC pc)
{
  PC_SVD         *jac;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  ierr      = PetscNewLog(pc,PC_SVD,&jac);CHKERRQ(ierr);
  pc->data  = (void*)jac;

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_SVD;
  pc->ops->applytranspose      = PCApply_SVD;
  pc->ops->setup               = PCSetUp_SVD;
  pc->ops->reset               = PCReset_SVD;
  pc->ops->destroy             = PCDestroy_SVD;
  pc->ops->setfromoptions      = PCSetFromOptions_SVD;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

