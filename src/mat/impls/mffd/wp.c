
/*MC
     MATMFFD_WP - Implements an alternative approach for computing the differencing parameter
        h used with the finite difference based matrix-free Jacobian.  This code
        implements the strategy of M. Pernice and H. Walker:

      h = error_rel * sqrt(1 + ||U||) / ||a||

      Notes:
        1) || U || does not change between linear iterations so is reused
        2) In GMRES || a || == 1 and so does not need to ever be computed except at restart
           when it is recomputed.

      Reference:  M. Pernice and H. F. Walker, "NITSOL: A Newton Iterative 
      Solver for Nonlinear Systems", SIAM J. Sci. Stat. Comput.", 1998, 
      vol 19, pp. 302--318.

   Options Database Keys:
.   -mat_mffd_compute_normu -Compute the norm of u everytime see MatMFFDWPSetComputeNormU()


   Level: intermediate

   Notes: Requires no global collectives when used with GMRES

   Formula used:
     F'(u)*a = [F(u+h*a) - F(u)]/h where

.seealso: MATMFFD, MatCreateMFFD(), MatCreateSNESMF(), MATMFFD_DS

M*/

/*
    This include file defines the data structure  MatMFFD that 
   includes information about the computation of h. It is shared by 
   all implementations that people provide.

   See snesmfjdef.c for  a full set of comments on the routines below.
*/
#include <petsc-private/matimpl.h>
#include <../src/mat/impls/mffd/mffdimpl.h>   /*I  "petscmat.h"   I*/

typedef struct {
  PetscReal  normUfact;                   /* previous sqrt(1.0 + || U ||) */
  PetscBool  computenormU;   
} MatMFFD_WP;

#undef __FUNCT__  
#define __FUNCT__ "MatMFFDCompute_WP"
/*
     MatMFFDCompute_WP - Standard PETSc code for 
   computing h with matrix-free finite differences.

  Input Parameters:
+   ctx - the matrix free context
.   U - the location at which you want the Jacobian
-   a - the direction you want the derivative

  Output Parameter:
.   h - the scale computed

*/
static PetscErrorCode MatMFFDCompute_WP(MatMFFD ctx,Vec U,Vec a,PetscScalar *h,PetscBool  *zeroa)
{
  MatMFFD_WP    *hctx = (MatMFFD_WP*)ctx->hctx;
  PetscReal      normU,norma;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!(ctx->count % ctx->recomputeperiod)) {
    if (hctx->computenormU || !ctx->ncurrenth) {
      ierr = VecNorm(U,NORM_2,&normU);CHKERRQ(ierr);
      hctx->normUfact = PetscSqrtReal(1.0+normU);
    }
    ierr = VecNorm(a,NORM_2,&norma);CHKERRQ(ierr);
    if (norma == 0.0) {
      *zeroa = PETSC_TRUE;
      PetscFunctionReturn(0);
    }
    *zeroa = PETSC_FALSE;
    *h = ctx->error_rel*hctx->normUfact/norma;
  } else {
    *h = ctx->currenth;
  }
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatMFFDView_WP"
/*
   MatMFFDView_WP - Prints information about this particular 
     method for computing h. Note that this does not print the general
     information about the matrix free, that is printed by the calling
     routine.

  Input Parameters:
+   ctx - the matrix free context
-   viewer - the PETSc viewer

*/   
static PetscErrorCode MatMFFDView_WP(MatMFFD ctx,PetscViewer viewer)
{
  MatMFFD_WP     *hctx = (MatMFFD_WP *)ctx->hctx;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (hctx->computenormU){ierr =  PetscViewerASCIIPrintf(viewer,"    Computes normU\n");CHKERRQ(ierr);}  
    else                   {ierr =  PetscViewerASCIIPrintf(viewer,"    Does not compute normU\n");CHKERRQ(ierr);}  
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported for SNES matrix-free WP",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMFFDSetFromOptions_WP"
/*
   MatMFFDSetFromOptions_WP - Looks in the options database for 
     any options appropriate for this method

  Input Parameter:
.  ctx - the matrix free context

*/
static PetscErrorCode MatMFFDSetFromOptions_WP(MatMFFD ctx)
{
  PetscErrorCode ierr;
  MatMFFD_WP     *hctx = (MatMFFD_WP*)ctx->hctx;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Walker-Pernice options");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-mat_mffd_compute_normu","Compute the norm of u","MatMFFDWPSetComputeNormU", hctx->computenormU,&hctx->computenormU,0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMFFDDestroy_WP"
/*
   MatMFFDDestroy_WP - Frees the space allocated by 
       MatCreateMFFD_WP(). 

  Input Parameter:
.  ctx - the matrix free context

   Notes: does not free the ctx, that is handled by the calling routine

*/
static PetscErrorCode MatMFFDDestroy_WP(MatMFFD ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree(ctx->hctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMFFDWPSetComputeNormU_P"
PetscErrorCode  MatMFFDWPSetComputeNormU_P(Mat mat,PetscBool  flag)
{
  MatMFFD     ctx = (MatMFFD)mat->data;
  MatMFFD_WP  *hctx = (MatMFFD_WP*)ctx->hctx;

  PetscFunctionBegin;
  hctx->computenormU = flag;
  PetscFunctionReturn(0);
} 
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatMFFDWPSetComputeNormU"
/*@
    MatMFFDWPSetComputeNormU - Sets whether it computes the ||U|| used by the WP
             PETSc routine for computing h. With any Krylov solver this need only 
             be computed during the first iteration and kept for later.

  Input Parameters:
+   A - the matrix created with MatCreateSNESMF()
-   flag - PETSC_TRUE causes it to compute ||U||, PETSC_FALSE uses the previous value

  Options Database Key:
.   -mat_mffd_compute_normu <true,false> - true by default, false can save calculations but you 
              must be sure that ||U|| has not changed in the mean time.

  Level: advanced

  Notes:
   See the manual page for MATMFFD_WP for a complete description of the
   algorithm used to compute h.

.seealso: MatMFFDSetFunctionError(), MatCreateSNESMF()

@*/
PetscErrorCode  MatMFFDWPSetComputeNormU(Mat A,PetscBool  flag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscTryMethod(A,"MatMFFDWPSetComputeNormU_C",(Mat,PetscBool),(A,flag));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreateMFFD_WP"
/*
     MatCreateMFFD_WP - Standard PETSc code for 
   computing h with matrix-free finite differences.

   Input Parameter:
.  ctx - the matrix free context created by MatCreateMFFD()

*/
PetscErrorCode  MatCreateMFFD_WP(MatMFFD ctx)
{
  PetscErrorCode ierr;
  MatMFFD_WP     *hctx;

  PetscFunctionBegin;

  /* allocate my own private data structure */
  ierr               = PetscNewLog(ctx,MatMFFD_WP,&hctx);CHKERRQ(ierr);
  ctx->hctx          = (void*)hctx;
  hctx->computenormU = PETSC_FALSE;

  /* set the functions I am providing */
  ctx->ops->compute        = MatMFFDCompute_WP;
  ctx->ops->destroy        = MatMFFDDestroy_WP;
  ctx->ops->view           = MatMFFDView_WP;  
  ctx->ops->setfromoptions = MatMFFDSetFromOptions_WP;  

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ctx->mat,"MatMFFDWPSetComputeNormU_C","MatMFFDWPSetComputeNormU_P",MatMFFDWPSetComputeNormU_P);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END



