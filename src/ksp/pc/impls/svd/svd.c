
#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/
#include <petscblaslapack.h>

/*
   Private context (data structure) for the SVD preconditioner.
*/
typedef struct {
  Vec               diag,work;
  Mat               A,U,Vt;
  PetscInt          nzero;
  PetscReal         zerosing;         /* measure of smallest singular value treated as nonzero */
  PetscInt          essrank;          /* essential rank of operator */
  VecScatter        left2red,right2red;
  Vec               leftred,rightred;
  PetscViewer       monitor;
  PetscViewerFormat monitorformat;
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
  PetscCall(MatDestroy(&jac->A));
  PetscCallMPI(MPI_Comm_size(((PetscObject)pc->pmat)->comm,&size));
  if (size > 1) {
    Mat redmat;

    PetscCall(MatCreateRedundantMatrix(pc->pmat,size,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&redmat));
    PetscCall(MatConvert(redmat,MATSEQDENSE,MAT_INITIAL_MATRIX,&jac->A));
    PetscCall(MatDestroy(&redmat));
  } else {
    PetscCall(MatConvert(pc->pmat,MATSEQDENSE,MAT_INITIAL_MATRIX,&jac->A));
  }
  if (!jac->diag) {    /* assume square matrices */
    PetscCall(MatCreateVecs(jac->A,&jac->diag,&jac->work));
  }
  if (!jac->U) {
    PetscCall(MatDuplicate(jac->A,MAT_DO_NOT_COPY_VALUES,&jac->U));
    PetscCall(MatDuplicate(jac->A,MAT_DO_NOT_COPY_VALUES,&jac->Vt));
  }
  PetscCall(MatGetSize(jac->A,&n,NULL));
  if (!n) {
    PetscCall(PetscInfo(pc,"Matrix has zero rows, skipping svd\n"));
    PetscFunctionReturn(0);
  }
  PetscCall(PetscBLASIntCast(n,&nb));
  lwork = 5*nb;
  PetscCall(PetscMalloc1(lwork,&work));
  PetscCall(MatDenseGetArray(jac->A,&a));
  PetscCall(MatDenseGetArray(jac->U,&u));
  PetscCall(MatDenseGetArray(jac->Vt,&v));
  PetscCall(VecGetArray(jac->diag,&d));
#if !defined(PETSC_USE_COMPLEX)
  {
    PetscBLASInt lierr;
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("A","A",&nb,&nb,a,&nb,d,u,&nb,v,&nb,work,&lwork,&lierr));
    PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"gesv() error %" PetscBLASInt_FMT,lierr);
    PetscCall(PetscFPTrapPop());
  }
#else
  {
    PetscBLASInt lierr;
    PetscReal    *rwork,*dd;
    PetscCall(PetscMalloc1(5*nb,&rwork));
    PetscCall(PetscMalloc1(nb,&dd));
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("A","A",&nb,&nb,a,&nb,dd,u,&nb,v,&nb,work,&lwork,rwork,&lierr));
    PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"gesv() error %" PetscBLASInt_FMT,lierr);
    PetscCall(PetscFree(rwork));
    for (i=0; i<n; i++) d[i] = dd[i];
    PetscCall(PetscFree(dd));
    PetscCall(PetscFPTrapPop());
  }
#endif
  PetscCall(MatDenseRestoreArray(jac->A,&a));
  PetscCall(MatDenseRestoreArray(jac->U,&u));
  PetscCall(MatDenseRestoreArray(jac->Vt,&v));
  for (i=n-1; i>=0; i--) if (PetscRealPart(d[i]) > jac->zerosing) break;
  jac->nzero = n-1-i;
  if (jac->monitor) {
    PetscCall(PetscViewerASCIIAddTab(jac->monitor,((PetscObject)pc)->tablevel));
    PetscCall(PetscViewerASCIIPrintf(jac->monitor,"    SVD: condition number %14.12e, %" PetscInt_FMT " of %" PetscInt_FMT " singular values are (nearly) zero\n",(double)PetscRealPart(d[0]/d[n-1]),jac->nzero,n));
    if (n < 10 || jac->monitorformat == PETSC_VIEWER_ALL) {
      PetscCall(PetscViewerASCIIPrintf(jac->monitor,"    SVD: singular values:\n"));
      for (i=0; i<n; i++) {
        if (i%5 == 0) {
            if (i != 0) {
              PetscCall(PetscViewerASCIIPrintf(jac->monitor,"\n"));
            }
            PetscCall(PetscViewerASCIIPrintf(jac->monitor,"        "));
          }
        PetscCall(PetscViewerASCIIPrintf(jac->monitor," %14.12e",(double)PetscRealPart(d[i])));
      }
      PetscCall(PetscViewerASCIIPrintf(jac->monitor,"\n"));
    } else {              /* print 5 smallest and 5 largest */
      PetscCall(PetscViewerASCIIPrintf(jac->monitor,"    SVD: smallest singular values: %14.12e %14.12e %14.12e %14.12e %14.12e\n",(double)PetscRealPart(d[n-1]),(double)PetscRealPart(d[n-2]),(double)PetscRealPart(d[n-3]),(double)PetscRealPart(d[n-4]),(double)PetscRealPart(d[n-5])));
      PetscCall(PetscViewerASCIIPrintf(jac->monitor,"    SVD: largest singular values : %14.12e %14.12e %14.12e %14.12e %14.12e\n",(double)PetscRealPart(d[4]),(double)PetscRealPart(d[3]),(double)PetscRealPart(d[2]),(double)PetscRealPart(d[1]),(double)PetscRealPart(d[0])));
    }
    PetscCall(PetscViewerASCIISubtractTab(jac->monitor,((PetscObject)pc)->tablevel));
  }
  PetscCall(PetscInfo(pc,"Largest and smallest singular values %14.12e %14.12e\n",(double)PetscRealPart(d[0]),(double)PetscRealPart(d[n-1])));
  for (i=0; i<n-jac->nzero; i++) d[i] = 1.0/d[i];
  for (; i<n; i++) d[i] = 0.0;
  if (jac->essrank > 0) for (i=0; i<n-jac->nzero-jac->essrank; i++) d[i] = 0.0; /* Skip all but essrank eigenvalues */
  PetscCall(PetscInfo(pc,"Number of zero or nearly singular values %" PetscInt_FMT "\n",jac->nzero));
  PetscCall(VecRestoreArray(jac->diag,&d));
  PetscCall(PetscFree(work));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSVDGetVec(PC pc,PCSide side,AccessMode amode,Vec x,Vec *xred)
{
  PC_SVD         *jac = (PC_SVD*)pc->data;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));
  *xred = NULL;
  switch (side) {
  case PC_LEFT:
    if (size == 1) *xred = x;
    else {
      if (!jac->left2red) PetscCall(VecScatterCreateToAll(x,&jac->left2red,&jac->leftred));
      if (amode & READ) {
        PetscCall(VecScatterBegin(jac->left2red,x,jac->leftred,INSERT_VALUES,SCATTER_FORWARD));
        PetscCall(VecScatterEnd(jac->left2red,x,jac->leftred,INSERT_VALUES,SCATTER_FORWARD));
      }
      *xred = jac->leftred;
    }
    break;
  case PC_RIGHT:
    if (size == 1) *xred = x;
    else {
      if (!jac->right2red) PetscCall(VecScatterCreateToAll(x,&jac->right2red,&jac->rightred));
      if (amode & READ) {
        PetscCall(VecScatterBegin(jac->right2red,x,jac->rightred,INSERT_VALUES,SCATTER_FORWARD));
        PetscCall(VecScatterEnd(jac->right2red,x,jac->rightred,INSERT_VALUES,SCATTER_FORWARD));
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
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));
  switch (side) {
  case PC_LEFT:
    if (size != 1 && amode & WRITE) {
      PetscCall(VecScatterBegin(jac->left2red,jac->leftred,x,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(jac->left2red,jac->leftred,x,INSERT_VALUES,SCATTER_REVERSE));
    }
    break;
  case PC_RIGHT:
    if (size != 1 && amode & WRITE) {
      PetscCall(VecScatterBegin(jac->right2red,jac->rightred,x,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(jac->right2red,jac->rightred,x,INSERT_VALUES,SCATTER_REVERSE));
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
  PetscCall(PCSVDGetVec(pc,PC_RIGHT,READ,x,&xred));
  PetscCall(PCSVDGetVec(pc,PC_LEFT,WRITE,y,&yred));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(MatMultTranspose(jac->U,xred,work));
#else
  PetscCall(MatMultHermitianTranspose(jac->U,xred,work));
#endif
  PetscCall(VecPointwiseMult(work,work,jac->diag));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(MatMultTranspose(jac->Vt,work,yred));
#else
  PetscCall(MatMultHermitianTranspose(jac->Vt,work,yred));
#endif
  PetscCall(PCSVDRestoreVec(pc,PC_RIGHT,READ,x,&xred));
  PetscCall(PCSVDRestoreVec(pc,PC_LEFT,WRITE,y,&yred));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_SVD(PC pc,Mat X,Mat Y)
{
  PC_SVD *jac = (PC_SVD*)pc->data;
  Mat    W;

  PetscFunctionBegin;
  PetscCall(MatTransposeMatMult(jac->U,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&W));
  PetscCall(MatDiagonalScale(W,jac->diag,NULL));
  PetscCall(MatTransposeMatMult(jac->Vt,W,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y));
  PetscCall(MatDestroy(&W));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_SVD(PC pc,Vec x,Vec y)
{
  PC_SVD         *jac = (PC_SVD*)pc->data;
  Vec            work = jac->work,xred,yred;

  PetscFunctionBegin;
  PetscCall(PCSVDGetVec(pc,PC_LEFT,READ,x,&xred));
  PetscCall(PCSVDGetVec(pc,PC_RIGHT,WRITE,y,&yred));
  PetscCall(MatMult(jac->Vt,xred,work));
  PetscCall(VecPointwiseMult(work,work,jac->diag));
  PetscCall(MatMult(jac->U,work,yred));
  PetscCall(PCSVDRestoreVec(pc,PC_LEFT,READ,x,&xred));
  PetscCall(PCSVDRestoreVec(pc,PC_RIGHT,WRITE,y,&yred));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_SVD(PC pc)
{
  PC_SVD         *jac = (PC_SVD*)pc->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&jac->A));
  PetscCall(MatDestroy(&jac->U));
  PetscCall(MatDestroy(&jac->Vt));
  PetscCall(VecDestroy(&jac->diag));
  PetscCall(VecDestroy(&jac->work));
  PetscCall(VecScatterDestroy(&jac->right2red));
  PetscCall(VecScatterDestroy(&jac->left2red));
  PetscCall(VecDestroy(&jac->rightred));
  PetscCall(VecDestroy(&jac->leftred));
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
  PetscCall(PCReset_SVD(pc));
  PetscCall(PetscViewerDestroy(&jac->monitor));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_SVD(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_SVD         *jac = (PC_SVD*)pc->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"SVD options");
  PetscCall(PetscOptionsReal("-pc_svd_zero_sing","Singular values smaller than this treated as zero","None",jac->zerosing,&jac->zerosing,NULL));
  PetscCall(PetscOptionsInt("-pc_svd_ess_rank","Essential rank of operator (0 to use entire operator)","None",jac->essrank,&jac->essrank,NULL));
  PetscCall(PetscOptionsViewer("-pc_svd_monitor","Monitor the conditioning, and extremal singular values","None",&jac->monitor,&jac->monitorformat,&flg));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_SVD(PC pc,PetscViewer viewer)
{
  PC_SVD         *svd = (PC_SVD*)pc->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  All singular values smaller than %g treated as zero\n",(double)svd->zerosing));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Provided essential rank of the matrix %" PetscInt_FMT " (all other eigenvalues are zeroed)\n",svd->essrank));
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

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_SVD(PC pc)
{
  PC_SVD      *jac;
  PetscMPIInt size = 0;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  PetscCall(PetscNewLog(pc,&jac));
  jac->zerosing = 1.e-12;
  pc->data      = (void*)jac;

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */

#if defined(PETSC_HAVE_COMPLEX)
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));
#endif
  if (size == 1) pc->ops->matapply = PCMatApply_SVD;
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
