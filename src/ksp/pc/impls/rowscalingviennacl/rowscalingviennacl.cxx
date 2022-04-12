
/*  -------------------------------------------------------------------- */

/*
   Include files needed for the ViennaCL row-scaling preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/
#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1

#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/mat/impls/aij/seq/seqviennacl/viennaclmatimpl.h>
#include <../src/vec/vec/impls/seq/seqviennacl/viennaclvecimpl.h>
#include <viennacl/linalg/row_scaling.hpp>

/*
   Private context (data structure) for the ROWSCALINGVIENNACL preconditioner.
*/
typedef struct {
  viennacl::linalg::row_scaling< viennacl::compressed_matrix<PetscScalar> > *ROWSCALINGVIENNACL;
} PC_ROWSCALINGVIENNACL;

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_ROWSCALINGVIENNACL - Prepares for the use of the ROWSCALINGVIENNACL preconditioner
                                by setting data structures and options.

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
static PetscErrorCode PCSetUp_ROWSCALINGVIENNACL(PC pc)
{
  PC_ROWSCALINGVIENNACL  *rowscaling = (PC_ROWSCALINGVIENNACL*)pc->data;
  PetscBool              flg = PETSC_FALSE;
  Mat_SeqAIJViennaCL     *gpustruct;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pc->pmat,MATSEQAIJVIENNACL,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Currently only handles ViennaCL matrices");
  if (pc->setupcalled != 0) {
    try {
      delete rowscaling->ROWSCALINGVIENNACL;
    } catch(char *ex) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex);
    }
  }
  try {
#if defined(PETSC_USE_COMPLEX)
    gpustruct = NULL;
    SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"No support for complex arithmetic in ROWSCALINGVIENNACL preconditioner");
#else
    PetscCall(MatViennaCLCopyToGPU(pc->pmat));
    gpustruct = (Mat_SeqAIJViennaCL*)(pc->pmat->spptr);

    viennacl::linalg::row_scaling_tag pc_tag(1);
    ViennaCLAIJMatrix *mat = (ViennaCLAIJMatrix*)gpustruct->mat;
    rowscaling->ROWSCALINGVIENNACL = new viennacl::linalg::row_scaling<viennacl::compressed_matrix<PetscScalar> >(*mat, pc_tag);
#endif
  } catch(char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApply_ROWSCALINGVIENNACL - Applies the ROWSCALINGVIENNACL preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
static PetscErrorCode PCApply_ROWSCALINGVIENNACL(PC pc,Vec x,Vec y)
{
  PC_ROWSCALINGVIENNACL         *ilu = (PC_ROWSCALINGVIENNACL*)pc->data;
  PetscBool                     flg1,flg2;
  viennacl::vector<PetscScalar> const *xarray=NULL;
  viennacl::vector<PetscScalar> *yarray=NULL;

  PetscFunctionBegin;
  /*how to apply a certain fixed number of iterations?*/
  PetscCall(PetscObjectTypeCompare((PetscObject)x,VECSEQVIENNACL,&flg1));
  PetscCall(PetscObjectTypeCompare((PetscObject)y,VECSEQVIENNACL,&flg2));
  PetscCheck((flg1 && flg2),PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP, "Currently only handles ViennaCL vectors");
  if (!ilu->ROWSCALINGVIENNACL) {
    PetscCall(PCSetUp_ROWSCALINGVIENNACL(pc));
  }
  PetscCall(VecSet(y,0.0));
  PetscCall(VecViennaCLGetArrayRead(x,&xarray));
  PetscCall(VecViennaCLGetArrayWrite(y,&yarray));
  try {
#if defined(PETSC_USE_COMPLEX)

#else
    *yarray = *xarray;
    ilu->ROWSCALINGVIENNACL->apply(*yarray);
#endif
  } catch(char * ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex);
  }
  PetscCall(VecViennaCLRestoreArrayRead(x,&xarray));
  PetscCall(VecViennaCLRestoreArrayWrite(y,&yarray));
  PetscCall(PetscObjectStateIncrease((PetscObject)y));
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCDestroy_ROWSCALINGVIENNACL - Destroys the private context for the ROWSCALINGVIENNACL preconditioner
   that was created with PCCreate_ROWSCALINGVIENNACL().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
static PetscErrorCode PCDestroy_ROWSCALINGVIENNACL(PC pc)
{
  PC_ROWSCALINGVIENNACL  *rowscaling = (PC_ROWSCALINGVIENNACL*)pc->data;

  PetscFunctionBegin;
  if (rowscaling->ROWSCALINGVIENNACL) {
    try {
      delete rowscaling->ROWSCALINGVIENNACL;
    } catch(char *ex) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex);
    }
  }

  /*
      Free the private data structure that was hanging off the PC
  */
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_ROWSCALINGVIENNACL(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"ROWSCALINGVIENNACL options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

/*MC
     PCRowScalingViennaCL  - A diagonal preconditioner (scaling rows of matrices by their norm) that can be used via the CUDA, OpenCL, and OpenMP backends of ViennaCL

   Level: advanced

.seealso:  PCCreate(), PCSetType(), PCType, PC

M*/

PETSC_EXTERN PetscErrorCode PCCreate_ROWSCALINGVIENNACL(PC pc)
{
  PC_ROWSCALINGVIENNACL  *rowscaling;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  PetscCall(PetscNewLog(pc,&rowscaling));
  pc->data = (void*)rowscaling;

  /*
     Initialize the pointer to zero
     Initialize number of v-cycles to default (1)
  */
  rowscaling->ROWSCALINGVIENNACL = 0;

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_ROWSCALINGVIENNACL;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_ROWSCALINGVIENNACL;
  pc->ops->destroy             = PCDestroy_ROWSCALINGVIENNACL;
  pc->ops->setfromoptions      = PCSetFromOptions_ROWSCALINGVIENNACL;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  PetscFunctionReturn(0);
}
