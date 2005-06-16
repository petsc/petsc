#define PETSCSNES_DLL

/*
  Implements the DS PETSc approach for computing the h 
  parameter used with the finite difference based matrix-free 
  Jacobian-vector products.

  To make your own: clone this file and modify for your needs.

  Mandatory functions:
  -------------------
      MatSNESMFCompute_  - for a given point and direction computes h

      MatSNESMFCreate_ - fills in the MatSNESMF data structure
                           for this particular implementation

      
   Optional functions:
   -------------------
      MatSNESMFView_ - prints information about the parameters being used.
                       This is called when SNESView() or -snes_view is used.

      MatSNESMFSetFromOptions_ - checks the options database for options that 
                               apply to this method.

      MatSNESMFDestroy_ - frees any space allocated by the routines above

*/

/*
    This include file defines the data structure  MatSNESMF that 
   includes information about the computation of h. It is shared by 
   all implementations that people provide
*/
#include "src/mat/matimpl.h"
#include "src/snes/mf/snesmfj.h"   /*I  "petscsnes.h"   I*/

/*
      The  method has one parameter that is used to 
   "cutoff" very small values. This is stored in a data structure
   that is only visible to this file. If your method has no parameters
   it can omit this, if it has several simply reorganize the data structure.
   The data structure is "hung-off" the MatSNESMF data structure in
   the void *hctx; field.
*/
typedef struct {
  PetscReal umin;          /* minimum allowable u'a value relative to |u|_1 */
} MatSNESMF_DS;

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFCompute_DS"
/*
   MatSNESMFCompute_DS - Standard PETSc code for computing the
   differencing paramter (h) for use with matrix-free finite differences.

   Input Parameters:
+  ctx - the matrix free context
.  U - the location at which you want the Jacobian
-  a - the direction you want the derivative

  
   Output Parameter:
.  h - the scale computed

*/
static PetscErrorCode MatSNESMFCompute_DS(MatSNESMFCtx ctx,Vec U,Vec a,PetscScalar *h,PetscTruth *zeroa)
{
  MatSNESMF_DS     *hctx = (MatSNESMF_DS*)ctx->hctx;
  PetscReal        nrm,sum,umin = hctx->umin;
  PetscScalar      dot;
  PetscErrorCode   ierr;

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
#if defined(PETSC_USE_COMPLEX)
    if (PetscAbsScalar(dot) < umin*sum && PetscRealPart(dot) >= 0.0) dot = umin*sum;
    else if (PetscAbsScalar(dot) < 0.0 && PetscRealPart(dot) > -umin*sum) dot = -umin*sum;
#else
    if (dot < umin*sum && dot >= 0.0) dot = umin*sum;
    else if (dot < 0.0 && dot > -umin*sum) dot = -umin*sum;
#endif
    *h = ctx->error_rel*dot/(nrm*nrm);
  } else {
    *h = ctx->currenth;
  }
  if (*h != *h) SETERRQ3(PETSC_ERR_PLIB,"Differencing parameter is not a number sum = %g dot = %g norm = %g",sum,PetscRealPart(dot),nrm);
  ctx->count++;
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFView_DS"
/*
   MatSNESMFView_DS - Prints information about this particular 
   method for computing h. Note that this does not print the general
   information about the matrix-free method, as such info is printed
   by the calling routine.

   Input Parameters:
+  ctx - the matrix free context
-  viewer - the PETSc viewer
*/   
static PetscErrorCode MatSNESMFView_DS(MatSNESMFCtx ctx,PetscViewer viewer)
{
  MatSNESMF_DS     *hctx = (MatSNESMF_DS *)ctx->hctx;
  PetscErrorCode   ierr;
  PetscTruth       iascii;

  PetscFunctionBegin;
  /*
     Currently this only handles the ascii file viewers, others
     could be added, but for this type of object other viewers
     make less sense
  */
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"    umin=%g (minimum iterate parameter)\n",hctx->umin);CHKERRQ(ierr); 
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for this SNES matrix free matrix",((PetscObject)viewer)->type_name);
  }    
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFSetFromOptions_DS"
/*
   MatSNESMFSetFromOptions_DS - Looks in the options database for 
   any options appropriate for this method.

   Input Parameter:
.  ctx - the matrix free context

*/
static PetscErrorCode MatSNESMFSetFromOptions_DS(MatSNESMFCtx ctx)
{
  PetscErrorCode   ierr;
  MatSNESMF_DS     *hctx = (MatSNESMF_DS*)ctx->hctx;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Finite difference matrix free parameters");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_mf_umin","umin","MatSNESMFDSSetUmin",hctx->umin,&hctx->umin,0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFDestroy_DS"
/*
   MatSNESMFDestroy_DS - Frees the space allocated by 
   MatSNESMFCreate_DS(). 

   Input Parameter:
.  ctx - the matrix free context

   Notes: 
   Does not free the ctx, that is handled by the calling routine
*/
static PetscErrorCode MatSNESMFDestroy_DS(MatSNESMFCtx ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(ctx->hctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFDSSetUmin_Private"
/*
   The following two routines use the PetscObjectCompose() and PetscObjectQuery()
   mechanism to allow the user to change the Umin parameter used in this method.
*/
PetscErrorCode MatSNESMFDSSetUmin_Private(Mat mat,PetscReal umin)
{
  MatSNESMFCtx ctx = (MatSNESMFCtx)mat->data;
  MatSNESMF_DS *hctx;

  PetscFunctionBegin;
  if (!ctx) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"MatSNESMFDSSetUmin() attached to non-shell matrix");
  }
  hctx = (MatSNESMF_DS*)ctx->hctx;
  hctx->umin = umin;
  PetscFunctionReturn(0);
} 
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFDSSetUmin"
/*@
    MatSNESMFDSSetUmin - Sets the "umin" parameter used by the 
    PETSc routine for computing the differencing parameter, h, which is used
    for matrix-free Jacobian-vector products.

   Input Parameters:
+  A - the matrix created with MatCreateSNESMF()
-  umin - the parameter

   Level: advanced

   Notes:
   See the manual page for MatCreateSNESMF() for a complete description of the
   algorithm used to compute h.

.seealso: MatSNESMFSetFunctionError(), MatCreateSNESMF()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFDSSetUmin(Mat A,PetscReal umin)
{
  PetscErrorCode ierr,(*f)(Mat,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatSNESMFDSSetUmin_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(A,umin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
     MATSNESMF_DS - the code for compute the "h" used in the finite difference
            matrix-free matrix vector product.  This code
        implements the strategy in Dennis and Schnabel, "Numerical Methods for Unconstrained
        Optimization and Nonlinear Equations".

   Options Database Keys:
.  -snes_mf_umin <umin> see MatSNESMFDSSetUmin()

   Level: intermediate

   Notes: Requires 2 norms and 1 inner product, but they are computed together
       so only one parallel collective operation is needed. See MATSNESMF_WP for a method
       (with GMRES) that requires NO collective operations.

   Formula used:
     F'(u)*a = [F(u+h*a) - F(u)]/h where
     h = error_rel*u'a/||a||^2                        if  |u'a| > umin*||a||_{1}
       = error_rel*umin*sign(u'a)*||a||_{1}/||a||^2   otherwise
 where
     error_rel = square root of relative error in function evaluation
     umin = minimum iterate parameter

.seealso: MATMFFD, MatCreateMF(), MatCreateSNESMF(), MATSNESMF_WP, MatSNESMFDSSetUmin()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFCreate_DS"
PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFCreate_DS(MatSNESMFCtx ctx)
{
  MatSNESMF_DS     *hctx;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  /* allocate my own private data structure */
  ierr       = PetscNew(MatSNESMF_DS,&hctx);CHKERRQ(ierr);
  ctx->hctx  = (void*)hctx;
  /* set a default for my parameter */
  hctx->umin = 1.e-6;

  /* set the functions I am providing */
  ctx->ops->compute        = MatSNESMFCompute_DS;
  ctx->ops->destroy        = MatSNESMFDestroy_DS;
  ctx->ops->view           = MatSNESMFView_DS;  
  ctx->ops->setfromoptions = MatSNESMFSetFromOptions_DS;  

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ctx->mat,"MatSNESMFDSSetUmin_C",
                            "MatSNESMFDSSetUmin_Private",
                             MatSNESMFDSSetUmin_Private);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END







