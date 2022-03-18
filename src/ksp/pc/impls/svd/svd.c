
#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/
#include <petscblaslapack.h>

/*
   Private context (data structure) for the SVD preconditioner.
*/
typedef struct {
  Vec         diag,work;
  Mat         A,U,Vt;
  PetscInt    nzero;
  PetscReal   zerosing;         /* measure of smallest singular value treated as nonzero */
  PetscInt    essrank;          /* essential rank of operator */
  VecScatter  left2red,right2red;
  Vec         leftred,rightred;
  PetscViewer monitor;
} PC_SVD;

typedef enum {READ=1, WRITE=2, READ_WRITE=3} AccessMode;

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
static PetscErrorCode PCSetUp_SVD(PC pc)
{
  PC_SVD         *jac = (PC_SVD*)pc->data;
  PetscErrorCode ierr;
  PetscScalar    *a,*u,*v,*d,*work;
  PetscBLASInt   nb,lwork;
  PetscInt       i,n;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatDestroy(&jac->A);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)pc->pmat)->comm,&size);CHKERRMPI(ierr);
  if (size > 1) {
    Mat redmat;

    ierr = MatCreateRedundantMatrix(pc->pmat,size,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&redmat);CHKERRQ(ierr);
    ierr = MatConvert(redmat,MATSEQDENSE,MAT_INITIAL_MATRIX,&jac->A);CHKERRQ(ierr);
    ierr = MatDestroy(&redmat);CHKERRQ(ierr);
  } else {
    ierr = MatConvert(pc->pmat,MATSEQDENSE,MAT_INITIAL_MATRIX,&jac->A);CHKERRQ(ierr);
  }
  if (!jac->diag) {    /* assume square matrices */
    ierr = MatCreateVecs(jac->A,&jac->diag,&jac->work);CHKERRQ(ierr);
  }
  if (!jac->U) {
    ierr = MatDuplicate(jac->A,MAT_DO_NOT_COPY_VALUES,&jac->U);CHKERRQ(ierr);
    ierr = MatDuplicate(jac->A,MAT_DO_NOT_COPY_VALUES,&jac->Vt);CHKERRQ(ierr);
  }
  ierr  = MatGetSize(jac->A,&n,NULL);CHKERRQ(ierr);
  if (!n) {
    ierr = PetscInfo(pc,"Matrix has zero rows, skipping svd\n");CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr  = PetscBLASIntCast(n,&nb);CHKERRQ(ierr);
  lwork = 5*nb;
  ierr  = PetscMalloc1(lwork,&work);CHKERRQ(ierr);
  ierr  = MatDenseGetArray(jac->A,&a);CHKERRQ(ierr);
  ierr  = MatDenseGetArray(jac->U,&u);CHKERRQ(ierr);
  ierr  = MatDenseGetArray(jac->Vt,&v);CHKERRQ(ierr);
  ierr  = VecGetArray(jac->diag,&d);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  {
    PetscBLASInt lierr;
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("A","A",&nb,&nb,a,&nb,d,u,&nb,v,&nb,work,&lwork,&lierr));
    PetscCheckFalse(lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"gesv() error %d",lierr);
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
  }
#else
  {
    PetscBLASInt lierr;
    PetscReal    *rwork,*dd;
    ierr = PetscMalloc1(5*nb,&rwork);CHKERRQ(ierr);
    ierr = PetscMalloc1(nb,&dd);CHKERRQ(ierr);
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("A","A",&nb,&nb,a,&nb,dd,u,&nb,v,&nb,work,&lwork,rwork,&lierr));
    PetscCheckFalse(lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"gesv() error %d",lierr);
    ierr = PetscFree(rwork);CHKERRQ(ierr);
    for (i=0; i<n; i++) d[i] = dd[i];
    ierr = PetscFree(dd);CHKERRQ(ierr);
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
  }
#endif
  ierr = MatDenseRestoreArray(jac->A,&a);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(jac->U,&u);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(jac->Vt,&v);CHKERRQ(ierr);
  for (i=n-1; i>=0; i--) if (PetscRealPart(d[i]) > jac->zerosing) break;
  jac->nzero = n-1-i;
  if (jac->monitor) {
    ierr = PetscViewerASCIIAddTab(jac->monitor,((PetscObject)pc)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(jac->monitor,"    SVD: condition number %14.12e, %D of %D singular values are (nearly) zero\n",(double)PetscRealPart(d[0]/d[n-1]),jac->nzero,n);CHKERRQ(ierr);
    if (n >= 10) {              /* print 5 smallest and 5 largest */
      ierr = PetscViewerASCIIPrintf(jac->monitor,"    SVD: smallest singular values: %14.12e %14.12e %14.12e %14.12e %14.12e\n",(double)PetscRealPart(d[n-1]),(double)PetscRealPart(d[n-2]),(double)PetscRealPart(d[n-3]),(double)PetscRealPart(d[n-4]),(double)PetscRealPart(d[n-5]));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(jac->monitor,"    SVD: largest singular values : %14.12e %14.12e %14.12e %14.12e %14.12e\n",(double)PetscRealPart(d[4]),(double)PetscRealPart(d[3]),(double)PetscRealPart(d[2]),(double)PetscRealPart(d[1]),(double)PetscRealPart(d[0]));CHKERRQ(ierr);
    } else {                    /* print all singular values */
      char     buf[256],*p;
      size_t   left = sizeof(buf),used;
      PetscInt thisline;
      for (p=buf,i=n-1,thisline=1; i>=0; i--,thisline++) {
        ierr  = PetscSNPrintfCount(p,left," %14.12e",&used,(double)PetscRealPart(d[i]));CHKERRQ(ierr);
        left -= used;
        p    += used;
        if (thisline > 4 || i==0) {
          ierr     = PetscViewerASCIIPrintf(jac->monitor,"    SVD: singular values:%s\n",buf);CHKERRQ(ierr);
          p        = buf;
          thisline = 0;
        }
      }
    }
    ierr = PetscViewerASCIISubtractTab(jac->monitor,((PetscObject)pc)->tablevel);CHKERRQ(ierr);
  }
  ierr = PetscInfo(pc,"Largest and smallest singular values %14.12e %14.12e\n",(double)PetscRealPart(d[0]),(double)PetscRealPart(d[n-1]));CHKERRQ(ierr);
  for (i=0; i<n-jac->nzero; i++) d[i] = 1.0/d[i];
  for (; i<n; i++) d[i] = 0.0;
  if (jac->essrank > 0) for (i=0; i<n-jac->nzero-jac->essrank; i++) d[i] = 0.0; /* Skip all but essrank eigenvalues */
  ierr = PetscInfo(pc,"Number of zero or nearly singular values %D\n",jac->nzero);CHKERRQ(ierr);
  ierr = VecRestoreArray(jac->diag,&d);CHKERRQ(ierr);
#if defined(foo)
  {
    PetscViewer viewer;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,"joe",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = MatView(jac->A,viewer);CHKERRQ(ierr);
    ierr = MatView(jac->U,viewer);CHKERRQ(ierr);
    ierr = MatView(jac->Vt,viewer);CHKERRQ(ierr);
    ierr = VecView(jac->diag,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  }
#endif
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSVDGetVec(PC pc,PCSide side,AccessMode amode,Vec x,Vec *xred)
{
  PC_SVD         *jac = (PC_SVD*)pc->data;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr  = MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size);CHKERRMPI(ierr);
  *xred = NULL;
  switch (side) {
  case PC_LEFT:
    if (size == 1) *xred = x;
    else {
      if (!jac->left2red) {ierr = VecScatterCreateToAll(x,&jac->left2red,&jac->leftred);CHKERRQ(ierr);}
      if (amode & READ) {
        ierr = VecScatterBegin(jac->left2red,x,jac->leftred,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(jac->left2red,x,jac->leftred,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      }
      *xred = jac->leftred;
    }
    break;
  case PC_RIGHT:
    if (size == 1) *xred = x;
    else {
      if (!jac->right2red) {ierr = VecScatterCreateToAll(x,&jac->right2red,&jac->rightred);CHKERRQ(ierr);}
      if (amode & READ) {
        ierr = VecScatterBegin(jac->right2red,x,jac->rightred,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(jac->right2red,x,jac->rightred,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      }
      *xred = jac->rightred;
    }
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Side must be LEFT or RIGHT");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSVDRestoreVec(PC pc,PCSide side,AccessMode amode,Vec x,Vec *xred)
{
  PC_SVD         *jac = (PC_SVD*)pc->data;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size);CHKERRMPI(ierr);
  switch (side) {
  case PC_LEFT:
    if (size != 1 && amode & WRITE) {
      ierr = VecScatterBegin(jac->left2red,jac->leftred,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(jac->left2red,jac->leftred,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    }
    break;
  case PC_RIGHT:
    if (size != 1 && amode & WRITE) {
      ierr = VecScatterBegin(jac->right2red,jac->rightred,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(jac->right2red,jac->rightred,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    }
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Side must be LEFT or RIGHT");
  }
  *xred = NULL;
  PetscFunctionReturn(0);
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
static PetscErrorCode PCApply_SVD(PC pc,Vec x,Vec y)
{
  PC_SVD         *jac = (PC_SVD*)pc->data;
  Vec            work = jac->work,xred,yred;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCSVDGetVec(pc,PC_RIGHT,READ,x,&xred);CHKERRQ(ierr);
  ierr = PCSVDGetVec(pc,PC_LEFT,WRITE,y,&yred);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = MatMultTranspose(jac->U,xred,work);CHKERRQ(ierr);
#else
  ierr = MatMultHermitianTranspose(jac->U,xred,work);CHKERRQ(ierr);
#endif
  ierr = VecPointwiseMult(work,work,jac->diag);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = MatMultTranspose(jac->Vt,work,yred);CHKERRQ(ierr);
#else
  ierr = MatMultHermitianTranspose(jac->Vt,work,yred);CHKERRQ(ierr);
#endif
  ierr = PCSVDRestoreVec(pc,PC_RIGHT,READ,x,&xred);CHKERRQ(ierr);
  ierr = PCSVDRestoreVec(pc,PC_LEFT,WRITE,y,&yred);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_SVD(PC pc,Vec x,Vec y)
{
  PC_SVD         *jac = (PC_SVD*)pc->data;
  Vec            work = jac->work,xred,yred;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCSVDGetVec(pc,PC_LEFT,READ,x,&xred);CHKERRQ(ierr);
  ierr = PCSVDGetVec(pc,PC_RIGHT,WRITE,y,&yred);CHKERRQ(ierr);
  ierr = MatMult(jac->Vt,xred,work);CHKERRQ(ierr);
  ierr = VecPointwiseMult(work,work,jac->diag);CHKERRQ(ierr);
  ierr = MatMult(jac->U,work,yred);CHKERRQ(ierr);
  ierr = PCSVDRestoreVec(pc,PC_LEFT,READ,x,&xred);CHKERRQ(ierr);
  ierr = PCSVDRestoreVec(pc,PC_RIGHT,WRITE,y,&yred);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_SVD(PC pc)
{
  PC_SVD         *jac = (PC_SVD*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&jac->A);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->U);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->Vt);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->diag);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->work);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&jac->right2red);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&jac->left2red);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->rightred);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->leftred);CHKERRQ(ierr);
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

static PetscErrorCode PCSetFromOptions_SVD(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscErrorCode ierr;
  PC_SVD         *jac = (PC_SVD*)pc->data;
  PetscBool      flg,set;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SVD options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_svd_zero_sing","Singular values smaller than this treated as zero","None",jac->zerosing,&jac->zerosing,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_svd_ess_rank","Essential rank of operator (0 to use entire operator)","None",jac->essrank,&jac->essrank,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_svd_monitor","Monitor the conditioning, and extremal singular values","None",jac->monitor ? PETSC_TRUE : PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set) {                    /* Should make PCSVDSetMonitor() */
    if (flg && !jac->monitor) {
      ierr = PetscViewerASCIIOpen(PetscObjectComm((PetscObject)pc),"stdout",&jac->monitor);CHKERRQ(ierr);
    } else if (!flg) {
      ierr = PetscViewerDestroy(&jac->monitor);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_SVD(PC pc,PetscViewer viewer)
{
  PC_SVD         *svd = (PC_SVD*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  All singular values smaller than %g treated as zero\n",(double)svd->zerosing);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Provided essential rank of the matrix %D (all other eigenvalues are zeroed)\n",svd->essrank);CHKERRQ(ierr);
  }
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

  Options Database:
+  -pc_svd_zero_sing <rtol> - Singular values smaller than this are treated as zero
-  -pc_svd_monitor - Print information on the extreme singular values of the operator

  Developer Note:
  This implementation automatically creates a redundant copy of the
   matrix on each process and uses a sequential SVD solve. Why does it do this instead
   of using the composable PCREDUNDANT object?

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC
M*/

PETSC_EXTERN PetscErrorCode PCCreate_SVD(PC pc)
{
  PC_SVD         *jac;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  ierr          = PetscNewLog(pc,&jac);CHKERRQ(ierr);
  jac->zerosing = 1.e-12;
  pc->data      = (void*)jac;

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply           = PCApply_SVD;
  pc->ops->applytranspose  = PCApplyTranspose_SVD;
  pc->ops->setup           = PCSetUp_SVD;
  pc->ops->reset           = PCReset_SVD;
  pc->ops->destroy         = PCDestroy_SVD;
  pc->ops->setfromoptions  = PCSetFromOptions_SVD;
  pc->ops->view            = PCView_SVD;
  pc->ops->applyrichardson = NULL;
  PetscFunctionReturn(0);
}

