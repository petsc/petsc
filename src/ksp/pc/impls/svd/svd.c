
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
  PetscScalar    *a,*u,*v,*d,*work;
  PetscBLASInt   nb,lwork;
  PetscInt       i,n;
  PetscMPIInt    size;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&jac->A));
  CHKERRMPI(MPI_Comm_size(((PetscObject)pc->pmat)->comm,&size));
  if (size > 1) {
    Mat redmat;

    CHKERRQ(MatCreateRedundantMatrix(pc->pmat,size,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&redmat));
    CHKERRQ(MatConvert(redmat,MATSEQDENSE,MAT_INITIAL_MATRIX,&jac->A));
    CHKERRQ(MatDestroy(&redmat));
  } else {
    CHKERRQ(MatConvert(pc->pmat,MATSEQDENSE,MAT_INITIAL_MATRIX,&jac->A));
  }
  if (!jac->diag) {    /* assume square matrices */
    CHKERRQ(MatCreateVecs(jac->A,&jac->diag,&jac->work));
  }
  if (!jac->U) {
    CHKERRQ(MatDuplicate(jac->A,MAT_DO_NOT_COPY_VALUES,&jac->U));
    CHKERRQ(MatDuplicate(jac->A,MAT_DO_NOT_COPY_VALUES,&jac->Vt));
  }
  CHKERRQ(MatGetSize(jac->A,&n,NULL));
  if (!n) {
    CHKERRQ(PetscInfo(pc,"Matrix has zero rows, skipping svd\n"));
    PetscFunctionReturn(0);
  }
  CHKERRQ(PetscBLASIntCast(n,&nb));
  lwork = 5*nb;
  CHKERRQ(PetscMalloc1(lwork,&work));
  CHKERRQ(MatDenseGetArray(jac->A,&a));
  CHKERRQ(MatDenseGetArray(jac->U,&u));
  CHKERRQ(MatDenseGetArray(jac->Vt,&v));
  CHKERRQ(VecGetArray(jac->diag,&d));
#if !defined(PETSC_USE_COMPLEX)
  {
    PetscBLASInt lierr;
    CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("A","A",&nb,&nb,a,&nb,d,u,&nb,v,&nb,work,&lwork,&lierr));
    PetscCheckFalse(lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"gesv() error %d",lierr);
    CHKERRQ(PetscFPTrapPop());
  }
#else
  {
    PetscBLASInt lierr;
    PetscReal    *rwork,*dd;
    CHKERRQ(PetscMalloc1(5*nb,&rwork));
    CHKERRQ(PetscMalloc1(nb,&dd));
    CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("A","A",&nb,&nb,a,&nb,dd,u,&nb,v,&nb,work,&lwork,rwork,&lierr));
    PetscCheckFalse(lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"gesv() error %d",lierr);
    CHKERRQ(PetscFree(rwork));
    for (i=0; i<n; i++) d[i] = dd[i];
    CHKERRQ(PetscFree(dd));
    CHKERRQ(PetscFPTrapPop());
  }
#endif
  CHKERRQ(MatDenseRestoreArray(jac->A,&a));
  CHKERRQ(MatDenseRestoreArray(jac->U,&u));
  CHKERRQ(MatDenseRestoreArray(jac->Vt,&v));
  for (i=n-1; i>=0; i--) if (PetscRealPart(d[i]) > jac->zerosing) break;
  jac->nzero = n-1-i;
  if (jac->monitor) {
    CHKERRQ(PetscViewerASCIIAddTab(jac->monitor,((PetscObject)pc)->tablevel));
    CHKERRQ(PetscViewerASCIIPrintf(jac->monitor,"    SVD: condition number %14.12e, %D of %D singular values are (nearly) zero\n",(double)PetscRealPart(d[0]/d[n-1]),jac->nzero,n));
    if (n >= 10) {              /* print 5 smallest and 5 largest */
      CHKERRQ(PetscViewerASCIIPrintf(jac->monitor,"    SVD: smallest singular values: %14.12e %14.12e %14.12e %14.12e %14.12e\n",(double)PetscRealPart(d[n-1]),(double)PetscRealPart(d[n-2]),(double)PetscRealPart(d[n-3]),(double)PetscRealPart(d[n-4]),(double)PetscRealPart(d[n-5])));
      CHKERRQ(PetscViewerASCIIPrintf(jac->monitor,"    SVD: largest singular values : %14.12e %14.12e %14.12e %14.12e %14.12e\n",(double)PetscRealPart(d[4]),(double)PetscRealPart(d[3]),(double)PetscRealPart(d[2]),(double)PetscRealPart(d[1]),(double)PetscRealPart(d[0])));
    } else {                    /* print all singular values */
      char     buf[256],*p;
      size_t   left = sizeof(buf),used;
      PetscInt thisline;
      for (p=buf,i=n-1,thisline=1; i>=0; i--,thisline++) {
        CHKERRQ(PetscSNPrintfCount(p,left," %14.12e",&used,(double)PetscRealPart(d[i])));
        left -= used;
        p    += used;
        if (thisline > 4 || i==0) {
          CHKERRQ(PetscViewerASCIIPrintf(jac->monitor,"    SVD: singular values:%s\n",buf));
          p        = buf;
          thisline = 0;
        }
      }
    }
    CHKERRQ(PetscViewerASCIISubtractTab(jac->monitor,((PetscObject)pc)->tablevel));
  }
  CHKERRQ(PetscInfo(pc,"Largest and smallest singular values %14.12e %14.12e\n",(double)PetscRealPart(d[0]),(double)PetscRealPart(d[n-1])));
  for (i=0; i<n-jac->nzero; i++) d[i] = 1.0/d[i];
  for (; i<n; i++) d[i] = 0.0;
  if (jac->essrank > 0) for (i=0; i<n-jac->nzero-jac->essrank; i++) d[i] = 0.0; /* Skip all but essrank eigenvalues */
  CHKERRQ(PetscInfo(pc,"Number of zero or nearly singular values %D\n",jac->nzero));
  CHKERRQ(VecRestoreArray(jac->diag,&d));
#if defined(foo)
  {
    PetscViewer viewer;
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_SELF,"joe",FILE_MODE_WRITE,&viewer));
    CHKERRQ(MatView(jac->A,viewer));
    CHKERRQ(MatView(jac->U,viewer));
    CHKERRQ(MatView(jac->Vt,viewer));
    CHKERRQ(VecView(jac->diag,viewer));
    CHKERRQ(PetscViewerDestroy(viewer));
  }
#endif
  CHKERRQ(PetscFree(work));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSVDGetVec(PC pc,PCSide side,AccessMode amode,Vec x,Vec *xred)
{
  PC_SVD         *jac = (PC_SVD*)pc->data;
  PetscMPIInt    size;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));
  *xred = NULL;
  switch (side) {
  case PC_LEFT:
    if (size == 1) *xred = x;
    else {
      if (!jac->left2red) CHKERRQ(VecScatterCreateToAll(x,&jac->left2red,&jac->leftred));
      if (amode & READ) {
        CHKERRQ(VecScatterBegin(jac->left2red,x,jac->leftred,INSERT_VALUES,SCATTER_FORWARD));
        CHKERRQ(VecScatterEnd(jac->left2red,x,jac->leftred,INSERT_VALUES,SCATTER_FORWARD));
      }
      *xred = jac->leftred;
    }
    break;
  case PC_RIGHT:
    if (size == 1) *xred = x;
    else {
      if (!jac->right2red) CHKERRQ(VecScatterCreateToAll(x,&jac->right2red,&jac->rightred));
      if (amode & READ) {
        CHKERRQ(VecScatterBegin(jac->right2red,x,jac->rightred,INSERT_VALUES,SCATTER_FORWARD));
        CHKERRQ(VecScatterEnd(jac->right2red,x,jac->rightred,INSERT_VALUES,SCATTER_FORWARD));
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
  PetscMPIInt    size;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));
  switch (side) {
  case PC_LEFT:
    if (size != 1 && amode & WRITE) {
      CHKERRQ(VecScatterBegin(jac->left2red,jac->leftred,x,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(jac->left2red,jac->leftred,x,INSERT_VALUES,SCATTER_REVERSE));
    }
    break;
  case PC_RIGHT:
    if (size != 1 && amode & WRITE) {
      CHKERRQ(VecScatterBegin(jac->right2red,jac->rightred,x,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(jac->right2red,jac->rightred,x,INSERT_VALUES,SCATTER_REVERSE));
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

  PetscFunctionBegin;
  CHKERRQ(PCSVDGetVec(pc,PC_RIGHT,READ,x,&xred));
  CHKERRQ(PCSVDGetVec(pc,PC_LEFT,WRITE,y,&yred));
#if !defined(PETSC_USE_COMPLEX)
  CHKERRQ(MatMultTranspose(jac->U,xred,work));
#else
  CHKERRQ(MatMultHermitianTranspose(jac->U,xred,work));
#endif
  CHKERRQ(VecPointwiseMult(work,work,jac->diag));
#if !defined(PETSC_USE_COMPLEX)
  CHKERRQ(MatMultTranspose(jac->Vt,work,yred));
#else
  CHKERRQ(MatMultHermitianTranspose(jac->Vt,work,yred));
#endif
  CHKERRQ(PCSVDRestoreVec(pc,PC_RIGHT,READ,x,&xred));
  CHKERRQ(PCSVDRestoreVec(pc,PC_LEFT,WRITE,y,&yred));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_SVD(PC pc,Vec x,Vec y)
{
  PC_SVD         *jac = (PC_SVD*)pc->data;
  Vec            work = jac->work,xred,yred;

  PetscFunctionBegin;
  CHKERRQ(PCSVDGetVec(pc,PC_LEFT,READ,x,&xred));
  CHKERRQ(PCSVDGetVec(pc,PC_RIGHT,WRITE,y,&yred));
  CHKERRQ(MatMult(jac->Vt,xred,work));
  CHKERRQ(VecPointwiseMult(work,work,jac->diag));
  CHKERRQ(MatMult(jac->U,work,yred));
  CHKERRQ(PCSVDRestoreVec(pc,PC_LEFT,READ,x,&xred));
  CHKERRQ(PCSVDRestoreVec(pc,PC_RIGHT,WRITE,y,&yred));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_SVD(PC pc)
{
  PC_SVD         *jac = (PC_SVD*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&jac->A));
  CHKERRQ(MatDestroy(&jac->U));
  CHKERRQ(MatDestroy(&jac->Vt));
  CHKERRQ(VecDestroy(&jac->diag));
  CHKERRQ(VecDestroy(&jac->work));
  CHKERRQ(VecScatterDestroy(&jac->right2red));
  CHKERRQ(VecScatterDestroy(&jac->left2red));
  CHKERRQ(VecDestroy(&jac->rightred));
  CHKERRQ(VecDestroy(&jac->leftred));
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

  PetscFunctionBegin;
  CHKERRQ(PCReset_SVD(pc));
  CHKERRQ(PetscViewerDestroy(&jac->monitor));
  CHKERRQ(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_SVD(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_SVD         *jac = (PC_SVD*)pc->data;
  PetscBool      flg,set;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"SVD options"));
  CHKERRQ(PetscOptionsReal("-pc_svd_zero_sing","Singular values smaller than this treated as zero","None",jac->zerosing,&jac->zerosing,NULL));
  CHKERRQ(PetscOptionsInt("-pc_svd_ess_rank","Essential rank of operator (0 to use entire operator)","None",jac->essrank,&jac->essrank,NULL));
  CHKERRQ(PetscOptionsBool("-pc_svd_monitor","Monitor the conditioning, and extremal singular values","None",jac->monitor ? PETSC_TRUE : PETSC_FALSE,&flg,&set));
  if (set) {                    /* Should make PCSVDSetMonitor() */
    if (flg && !jac->monitor) {
      CHKERRQ(PetscViewerASCIIOpen(PetscObjectComm((PetscObject)pc),"stdout",&jac->monitor));
    } else if (!flg) {
      CHKERRQ(PetscViewerDestroy(&jac->monitor));
    }
  }
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_SVD(PC pc,PetscViewer viewer)
{
  PC_SVD         *svd = (PC_SVD*)pc->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  All singular values smaller than %g treated as zero\n",(double)svd->zerosing));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Provided essential rank of the matrix %D (all other eigenvalues are zeroed)\n",svd->essrank));
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

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  CHKERRQ(PetscNewLog(pc,&jac));
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
