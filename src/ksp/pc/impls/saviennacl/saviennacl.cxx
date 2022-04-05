
/*  -------------------------------------------------------------------- */

/*
   Include files needed for the ViennaCL Smoothed Aggregation preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/
#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1
#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/mat/impls/aij/seq/seqviennacl/viennaclmatimpl.h>
#include <../src/vec/vec/impls/seq/seqviennacl/viennaclvecimpl.h>
#include <viennacl/linalg/amg.hpp>

/*
   Private context (data structure) for the SAVIENNACL preconditioner.
*/
typedef struct {
  viennacl::linalg::amg_precond<viennacl::compressed_matrix<PetscScalar> > *SAVIENNACL;
} PC_SAVIENNACL;

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_SAVIENNACL - Prepares for the use of the SAVIENNACL preconditioner
                        by setting data structures and options.

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
static PetscErrorCode PCSetUp_SAVIENNACL(PC pc)
{
  PC_SAVIENNACL      *sa = (PC_SAVIENNACL*)pc->data;
  PetscBool          flg = PETSC_FALSE;
  Mat_SeqAIJViennaCL *gpustruct;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pc->pmat,MATSEQAIJVIENNACL,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Currently only handles ViennaCL matrices");
  if (pc->setupcalled != 0) {
    try {
      delete sa->SAVIENNACL;
    } catch(char *ex) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex);
    }
  }
  try {
#if defined(PETSC_USE_COMPLEX)
    gpustruct = NULL;
    SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"No support for complex arithmetic in SAVIENNACL preconditioner");
#else
    PetscCall(MatViennaCLCopyToGPU(pc->pmat));
    gpustruct = (Mat_SeqAIJViennaCL*)(pc->pmat->spptr);

    viennacl::linalg::amg_tag amg_tag_sa_pmis;
    amg_tag_sa_pmis.set_coarsening_method(viennacl::linalg::AMG_COARSENING_METHOD_MIS2_AGGREGATION);
    amg_tag_sa_pmis.set_interpolation_method(viennacl::linalg::AMG_INTERPOLATION_METHOD_SMOOTHED_AGGREGATION);
    ViennaCLAIJMatrix *mat = (ViennaCLAIJMatrix*)gpustruct->mat;
    sa->SAVIENNACL = new viennacl::linalg::amg_precond<viennacl::compressed_matrix<PetscScalar> >(*mat, amg_tag_sa_pmis);
    sa->SAVIENNACL->setup();
#endif
  } catch(char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApply_SAVIENNACL - Applies the SAVIENNACL preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
static PetscErrorCode PCApply_SAVIENNACL(PC pc,Vec x,Vec y)
{
  PC_SAVIENNACL                 *sac = (PC_SAVIENNACL*)pc->data;
  PetscBool                     flg1,flg2;
  viennacl::vector<PetscScalar> const *xarray=NULL;
  viennacl::vector<PetscScalar> *yarray=NULL;

  PetscFunctionBegin;
  /*how to apply a certain fixed number of iterations?*/
  PetscCall(PetscObjectTypeCompare((PetscObject)x,VECSEQVIENNACL,&flg1));
  PetscCall(PetscObjectTypeCompare((PetscObject)y,VECSEQVIENNACL,&flg2));
  PetscCheck((flg1 && flg2),PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP, "Currently only handles ViennaCL vectors");
  if (!sac->SAVIENNACL) {
    PetscCall(PCSetUp_SAVIENNACL(pc));
  }
  PetscCall(VecViennaCLGetArrayRead(x,&xarray));
  PetscCall(VecViennaCLGetArrayWrite(y,&yarray));
  try {
#if !defined(PETSC_USE_COMPLEX)
    *yarray = *xarray;
    sac->SAVIENNACL->apply(*yarray);
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
   PCDestroy_SAVIENNACL - Destroys the private context for the SAVIENNACL preconditioner
   that was created with PCCreate_SAVIENNACL().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
static PetscErrorCode PCDestroy_SAVIENNACL(PC pc)
{
  PC_SAVIENNACL  *sac = (PC_SAVIENNACL*)pc->data;

  PetscFunctionBegin;
  if (sac->SAVIENNACL) {
    try {
      delete sac->SAVIENNACL;
    } catch(char * ex) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex);
    }
  }

  /*
      Free the private data structure that was hanging off the PC
  */
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_SAVIENNACL(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"SAVIENNACL options"));
  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

/*MC
     PCSAViennaCL  - A smoothed agglomeration algorithm that can be used via the CUDA, OpenCL, and OpenMP backends of ViennaCL

   Level: advanced

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC

M*/

PETSC_EXTERN PetscErrorCode PCCreate_SAVIENNACL(PC pc)
{
  PC_SAVIENNACL  *sac;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  PetscCall(PetscNewLog(pc,&sac));
  pc->data = (void*)sac;

  /*
     Initialize the pointer to zero
     Initialize number of v-cycles to default (1)
  */
  sac->SAVIENNACL = 0;

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_SAVIENNACL;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_SAVIENNACL;
  pc->ops->destroy             = PCDestroy_SAVIENNACL;
  pc->ops->setfromoptions      = PCSetFromOptions_SAVIENNACL;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  PetscFunctionReturn(0);
}
