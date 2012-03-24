
#include <petsc-private/pcimpl.h>   /*I "petscpc.h" I*/
#include <petscblaslapack.h>

/* 
   Private context (data structure) for the SVD preconditioner.  
*/
typedef struct {
  Vec         diag,work;
  Mat         A,U,V;
  PetscInt    nzero;
  PetscReal   zerosing;         /* measure of smallest singular value treated as nonzero */
  PetscViewer monitor;
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
  PetscBLASInt   nb,lwork;
  PetscInt       i,n;

  PetscFunctionBegin;
  if (!jac->diag) {
    /* assume square matrices */
    ierr = MatGetVecs(pc->pmat,&jac->diag,&jac->work);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&jac->A);CHKERRQ(ierr);
  ierr = MatConvert(pc->pmat,MATSEQDENSE,MAT_INITIAL_MATRIX,&jac->A);CHKERRQ(ierr);
  if (!jac->U) {
    ierr = MatDuplicate(jac->A,MAT_DO_NOT_COPY_VALUES,&jac->U);CHKERRQ(ierr);
    ierr = MatDuplicate(jac->A,MAT_DO_NOT_COPY_VALUES,&jac->V);CHKERRQ(ierr);
  }
  ierr = MatGetSize(pc->pmat,&n,PETSC_NULL);CHKERRQ(ierr);
  nb    = PetscBLASIntCast(n);
  lwork = 5*nb;
  ierr  = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr); 
  ierr  = MatGetArray(jac->A,&a);CHKERRQ(ierr); 
  ierr  = MatGetArray(jac->U,&u);CHKERRQ(ierr); 
  ierr  = MatGetArray(jac->V,&v);CHKERRQ(ierr); 
  ierr  = VecGetArray(jac->diag,&d);CHKERRQ(ierr); 
#if !defined(PETSC_USE_COMPLEX)
  {
    PetscBLASInt lierr;
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    LAPACKgesvd_("A","A",&nb,&nb,a,&nb,d,u,&nb,v,&nb,work,&lwork,&lierr);
    if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"gesv() error %d",lierr);
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
  }
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not coded for complex");
#endif
  ierr  = MatRestoreArray(jac->A,&a);CHKERRQ(ierr); 
  ierr  = MatRestoreArray(jac->U,&u);CHKERRQ(ierr); 
  ierr  = MatRestoreArray(jac->V,&v);CHKERRQ(ierr);
  for (i=n-1; i>=0; i--) if (PetscRealPart(d[i]) > jac->zerosing) break;
  jac->nzero = n-1-i;
  if (jac->monitor) {
    ierr = PetscViewerASCIIAddTab(jac->monitor,((PetscObject)pc)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(jac->monitor,"    SVD: condition number %14.12e, %D of %D singular values are (nearly) zero\n",(double)PetscRealPart(d[0]/d[n-1]),jac->nzero,n);CHKERRQ(ierr);
    if (n >= 10) {              /* print 5 smallest and 5 largest */
      ierr = PetscViewerASCIIPrintf(jac->monitor,"    SVD: smallest singular values: %14.12e %14.12e %14.12e %14.12e %14.12e\n",(double)PetscRealPart(d[n-1]),(double)PetscRealPart(d[n-2]),(double)PetscRealPart(d[n-3]),(double)PetscRealPart(d[n-4]),(double)PetscRealPart(d[n-5]));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(jac->monitor,"    SVD: largest singular values : %14.12e %14.12e %14.12e %14.12e %14.12e\n",(double)PetscRealPart(d[4]),(double)PetscRealPart(d[3]),(double)PetscRealPart(d[2]),(double)PetscRealPart(d[1]),(double)PetscRealPart(d[0]));CHKERRQ(ierr);
    } else {                    /* print all singular values */
      char buf[256],*p;
      size_t left = sizeof buf,used;
      PetscInt thisline;
      for (p=buf,i=n-1,thisline=1; i>=0; i--,thisline++) {
        ierr = PetscSNPrintfCount(p,left," %14.12e",&used,(double)PetscRealPart(d[i]));CHKERRQ(ierr);
        left -= used;
        p += used;
        if (thisline > 4 || i==0) {
          ierr = PetscViewerASCIIPrintf(jac->monitor,"    SVD: singular values:%s\n",buf);CHKERRQ(ierr);
          p = buf;
          thisline = 0;
        }
      }
    }
    ierr = PetscViewerASCIISubtractTab(jac->monitor,((PetscObject)pc)->tablevel);CHKERRQ(ierr);
  }
  ierr = PetscInfo2(pc,"Largest and smallest singular values %14.12e %14.12e\n",(double)PetscRealPart(d[0]),(double)PetscRealPart(d[n-1]));
  for (i=0; i<n-jac->nzero; i++) d[i] = 1.0/d[i];
  for (; i<n; i++) d[i] = 0.0;
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
  PC_SVD         *jac = (PC_SVD*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_SVD(pc);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&jac->monitor);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_SVD"
static PetscErrorCode PCSetFromOptions_SVD(PC pc)
{
  PetscErrorCode ierr;
  PC_SVD         *jac = (PC_SVD*)pc->data;
  PetscBool      flg,set;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SVD options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_svd_zero_sing","Singular values smaller than this treated as zero","None",jac->zerosing,&jac->zerosing,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_svd_monitor","Monitor the conditioning, and extremal singular values","None",jac->monitor?PETSC_TRUE:PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set) {                    /* Should make PCSVDSetMonitor() */
    if (flg && !jac->monitor) {
      ierr = PetscViewerASCIIOpen(((PetscObject)pc)->comm,"stdout",&jac->monitor);CHKERRQ(ierr);
    } else if (!flg) {
      ierr = PetscViewerDestroy(&jac->monitor);CHKERRQ(ierr);
    }
  }
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

  Options Database:
-  -pc_svd_zero_sing <rtol> Singular values smaller than this are treated as zero
+  -pc_svd_monitor  Print information on the extreme singular values of the operator

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
  ierr          = PetscNewLog(pc,PC_SVD,&jac);CHKERRQ(ierr);
  jac->zerosing = 1.e-12;
  pc->data      = (void*)jac;

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

