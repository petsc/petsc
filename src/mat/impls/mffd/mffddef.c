
/*
  Implements the DS PETSc approach for computing the h
  parameter used with the finite difference based matrix-free
  Jacobian-vector products.

  To make your own: clone this file and modify for your needs.

  Mandatory functions:
  -------------------
      MatMFFDCompute_  - for a given point and direction computes h

      MatCreateMFFD _ - fills in the MatMFFD data structure
                           for this particular implementation


   Optional functions:
   -------------------
      MatMFFDView_ - prints information about the parameters being used.
                       This is called when SNESView() or -snes_view is used.

      MatMFFDSetFromOptions_ - checks the options database for options that
                               apply to this method.

      MatMFFDDestroy_ - frees any space allocated by the routines above

*/

/*
    This include file defines the data structure  MatMFFD that
   includes information about the computation of h. It is shared by
   all implementations that people provide
*/
#include <petsc/private/matimpl.h>
#include <../src/mat/impls/mffd/mffdimpl.h>   /*I  "petscmat.h"   I*/

/*
      The  method has one parameter that is used to
   "cutoff" very small values. This is stored in a data structure
   that is only visible to this file. If your method has no parameters
   it can omit this, if it has several simply reorganize the data structure.
   The data structure is "hung-off" the MatMFFD data structure in
   the void *hctx; field.
*/
typedef struct {
  PetscReal umin;          /* minimum allowable u'a value relative to |u|_1 */
} MatMFFD_DS;

/*
   MatMFFDCompute_DS - Standard PETSc code for computing the
   differencing parameter (h) for use with matrix-free finite differences.

   Input Parameters:
+  ctx - the matrix free context
.  U - the location at which you want the Jacobian
-  a - the direction you want the derivative


   Output Parameter:
.  h - the scale computed

*/
static PetscErrorCode MatMFFDCompute_DS(MatMFFD ctx,Vec U,Vec a,PetscScalar *h,PetscBool  *zeroa)
{
  MatMFFD_DS     *hctx = (MatMFFD_DS*)ctx->hctx;
  PetscReal      nrm,sum,umin = hctx->umin;
  PetscScalar    dot;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!(ctx->count % ctx->recomputeperiod)) {
    /*
     This algorithm requires 2 norms and 1 inner product. Rather than
     use directly the VecNorm() and VecDot() routines (and thus have
     three separate collective operations, we use the VecxxxBegin/End() routines
    */
    ierr = VecDotBegin(U,a,&dot);CHKERRQ(ierr);
    ierr = VecNormBegin(a,NORM_1,&sum);CHKERRQ(ierr);
    ierr = VecNormBegin(a,NORM_2,&nrm);CHKERRQ(ierr);
    ierr = VecDotEnd(U,a,&dot);CHKERRQ(ierr);
    ierr = VecNormEnd(a,NORM_1,&sum);CHKERRQ(ierr);
    ierr = VecNormEnd(a,NORM_2,&nrm);CHKERRQ(ierr);

    if (nrm == 0.0) {
      *zeroa = PETSC_TRUE;
      PetscFunctionReturn(0);
    }
    *zeroa = PETSC_FALSE;

    /*
      Safeguard for step sizes that are "too small"
    */
    if (PetscAbsScalar(dot) < umin*sum && PetscRealPart(dot) >= 0.0) dot = umin*sum;
    else if (PetscAbsScalar(dot) < 0.0 && PetscRealPart(dot) > -umin*sum) dot = -umin*sum;
    *h = ctx->error_rel*dot/(nrm*nrm);
    if (PetscIsInfOrNanScalar(*h)) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Differencing parameter is not a number sum = %g dot = %g norm = %g",(double)sum,(double)PetscRealPart(dot),(double)nrm);
  } else {
    *h = ctx->currenth;
  }
  ctx->count++;
  PetscFunctionReturn(0);
}

/*
   MatMFFDView_DS - Prints information about this particular
   method for computing h. Note that this does not print the general
   information about the matrix-free method, as such info is printed
   by the calling routine.

   Input Parameters:
+  ctx - the matrix free context
-  viewer - the PETSc viewer
*/
static PetscErrorCode MatMFFDView_DS(MatMFFD ctx,PetscViewer viewer)
{
  MatMFFD_DS     *hctx = (MatMFFD_DS*)ctx->hctx;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  /*
     Currently this only handles the ascii file viewers, others
     could be added, but for this type of object other viewers
     make less sense
  */
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"    umin=%g (minimum iterate parameter)\n",(double)hctx->umin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   MatMFFDSetFromOptions_DS - Looks in the options database for
   any options appropriate for this method.

   Input Parameter:
.  ctx - the matrix free context

*/
static PetscErrorCode MatMFFDSetFromOptions_DS(PetscOptionItems *PetscOptionsObject,MatMFFD ctx)
{
  PetscErrorCode ierr;
  MatMFFD_DS     *hctx = (MatMFFD_DS*)ctx->hctx;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Finite difference matrix free parameters");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_mffd_umin","umin","MatMFFDDSSetUmin",hctx->umin,&hctx->umin,0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   MatMFFDDestroy_DS - Frees the space allocated by
   MatCreateMFFD_DS().

   Input Parameter:
.  ctx - the matrix free context

   Notes:
   Does not free the ctx, that is handled by the calling routine
*/
static PetscErrorCode MatMFFDDestroy_DS(MatMFFD ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(ctx->hctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   The following two routines use the PetscObjectCompose() and PetscObjectQuery()
   mechanism to allow the user to change the Umin parameter used in this method.
*/
PetscErrorCode MatMFFDDSSetUmin_DS(Mat mat,PetscReal umin)
{
  MatMFFD        ctx=NULL;
  MatMFFD_DS     *hctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,&ctx);CHKERRQ(ierr);
  if (!ctx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"MatMFFDDSSetUmin() attached to non-shell matrix");
  hctx       = (MatMFFD_DS*)ctx->hctx;
  hctx->umin = umin;
  PetscFunctionReturn(0);
}

/*@
    MatMFFDDSSetUmin - Sets the "umin" parameter used by the
    PETSc routine for computing the differencing parameter, h, which is used
    for matrix-free Jacobian-vector products.

   Input Parameters:
+  A - the matrix created with MatCreateSNESMF()
-  umin - the parameter

   Level: advanced

   Notes:
   See the manual page for MatCreateSNESMF() for a complete description of the
   algorithm used to compute h.

.seealso: MatMFFDSetFunctionError(), MatCreateSNESMF()

@*/
PetscErrorCode  MatMFFDDSSetUmin(Mat A,PetscReal umin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscTryMethod(A,"MatMFFDDSSetUmin_C",(Mat,PetscReal),(A,umin));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     MATMFFD_DS - the code for compute the "h" used in the finite difference
            matrix-free matrix vector product.  This code
        implements the strategy in Dennis and Schnabel, "Numerical Methods for Unconstrained
        Optimization and Nonlinear Equations".

   Options Database Keys:
.  -mat_mffd_umin <umin> see MatMFFDDSSetUmin()

   Level: intermediate

   Notes:
    Requires 2 norms and 1 inner product, but they are computed together
       so only one parallel collective operation is needed. See MATMFFD_WP for a method
       (with GMRES) that requires NO collective operations.

   Formula used:
     F'(u)*a = [F(u+h*a) - F(u)]/h where
     h = error_rel*u'a/||a||^2                        if  |u'a| > umin*||a||_{1}
       = error_rel*umin*sign(u'a)*||a||_{1}/||a||^2   otherwise
 where
     error_rel = square root of relative error in function evaluation
     umin = minimum iterate parameter

.seealso: MATMFFD, MatCreateMFFD(), MatCreateSNESMF(), MATMFFD_WP, MatMFFDDSSetUmin()

M*/
PETSC_EXTERN PetscErrorCode MatCreateMFFD_DS(MatMFFD ctx)
{
  MatMFFD_DS     *hctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* allocate my own private data structure */
  ierr      = PetscNewLog(ctx,&hctx);CHKERRQ(ierr);
  ctx->hctx = (void*)hctx;
  /* set a default for my parameter */
  hctx->umin = 1.e-6;

  /* set the functions I am providing */
  ctx->ops->compute        = MatMFFDCompute_DS;
  ctx->ops->destroy        = MatMFFDDestroy_DS;
  ctx->ops->view           = MatMFFDView_DS;
  ctx->ops->setfromoptions = MatMFFDSetFromOptions_DS;

  ierr = PetscObjectComposeFunction((PetscObject)ctx->mat,"MatMFFDDSSetUmin_C",MatMFFDDSSetUmin_DS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}







