#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: snesmfjdef.c,v 1.7 1999/10/01 21:22:26 bsmith Exp bsmith $";
#endif
/*
  Implements the default PETSc approach for computing the h 
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

      MatSNESMFPrintHelp_ - prints a help message on what options are
                          available for this implementation

      MatSNESMFSetFromOptions_ - checks the options database for options that 
                               apply to this method.

      MatSNESMFDestroy_ - frees any space allocated by the routines above

*/

/*
    This include file defines the data structure  MatSNESMF that 
   includes information about the computation of h. It is shared by 
   all implementations that people provide
*/
#include "src/snes/mf/snesmfj.h"   /*I  "snes.h"   I*/

/*
      The default method has one parameter that is used to 
   "cutoff" very small values. This is stored in a data structure
   that is only visible to this file. If your method has no parameters
   it can omit this, if it has several simply reorganize the data structure.
   The data structure is "hung-off" the MatSNESMF data structure in
   the void *hctx; field.
*/
typedef struct {
  double umin;          /* minimum allowable u'a value relative to |u|_1 */
} MatSNESMFDefault;

#undef __FUNC__  
#define __FUNC__ "MatSNESMFCompute_Default"
/*
   MatSNESMFCompute_Default - Standard PETSc code for computing the
   differencing paramter (h) for use with matrix-free finite differences.

   Input Parameters:
+  ctx - the matrix free context
.  U - the location at which you want the Jacobian
-  a - the direction you want the derivative

   Output Parameter:
.  h - the scale computed

*/
static int MatSNESMFCompute_Default(MatSNESMFCtx ctx,Vec U,Vec a,Scalar *h)
{
  MatSNESMFDefault *hctx = (MatSNESMFDefault *) ctx->hctx;
  double           norm, sum, umin = hctx->umin;
  Scalar           dot;
  int              ierr;

  PetscFunctionBegin;
  /*
     This algorithm requires 2 norms and 1 inner product. Rather than
     use directly the VecNorm() and VecDot() routines (and thus have 
     three separate collective operations, we use the VecxxxBegin/End() routines
     and manually call MPI for the collective phase.

  */
  ierr = VecDotBegin(U,a,&dot);CHKERRQ(ierr);
  ierr = VecNormBegin(a,NORM_1,&sum);CHKERRQ(ierr);
  ierr = VecNormBegin(a,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecDotEnd(U,a,&dot);CHKERRQ(ierr);
  ierr = VecNormEnd(a,NORM_1,&sum);CHKERRQ(ierr);
  ierr = VecNormEnd(a,NORM_2,&norm);CHKERRQ(ierr);

  /* 
     Safeguard for step sizes that are "too small"
  */
  if (sum == 0.0) {dot = 1.0; norm = 1.0;}
#if defined(PETSC_USE_COMPLEX)
  else if (PetscAbsScalar(dot) < umin*sum && PetscReal(dot) >= 0.0) dot = umin*sum;
  else if (PetscAbsScalar(dot) < 0.0 && PetscReal(dot) > -umin*sum) dot = -umin*sum;
#else
  else if (dot < umin*sum && dot >= 0.0) dot = umin*sum;
  else if (dot < 0.0 && dot > -umin*sum) dot = -umin*sum;
#endif
  *h = ctx->error_rel*dot/(norm*norm);
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "MatSNESMFView_Default"
/*
   MatSNESMFView_Default - Prints information about this particular 
   method for computing h. Note that this does not print the general
   information about the matrix-free method, as such info is printed
   by the calling routine.

   Input Parameters:
+  ctx - the matrix free context
-  viewer - the PETSc viewer
*/   
static int MatSNESMFView_Default(MatSNESMFCtx ctx,Viewer viewer)
{
  FILE             *fd;
  MatSNESMFDefault *hctx = (MatSNESMFDefault *)ctx->hctx;
  int              ierr;
  int              isascii;

  PetscFunctionBegin;
  ierr = ViewerASCIIGetPointer(viewer,&fd);CHKERRQ(ierr);
  
  /*
     Currently this only handles the ascii file viewers, others
     could be added, but for this type of object other viewers
     make less sense
  */
  isascii = PetscTypeCompare(viewer,ASCII_VIEWER);
  if (isascii) {
    ierr = PetscFPrintf(ctx->comm,fd,"    umin=%g (minimum iterate parameter)\n",hctx->umin);CHKERRQ(ierr); 
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for this SNES matrix free matrix",((PetscObject)viewer)->type_name);
  }    
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESMFPrintHelp_Default"
/*
   MatSNESMFPrintHelp_Default - Prints a list of all the options 
   this particular method supports.

   Input Parameter:
.  ctx - the matrix free context

*/
static int MatSNESMFPrintHelp_Default(MatSNESMFCtx ctx)
{
  char*            p;
  MatSNESMFDefault *hctx = (MatSNESMFDefault *)ctx->hctx;
  int              ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetOptionsPrefix((PetscObject)ctx->snes,&p);CHKERRQ(ierr);
  if (!p) p = "";
  ierr = (*PetscHelpPrintf)(ctx->comm,"   -%ssnes_mf_umin <umin> see users manual (default %g)\n",p,hctx->umin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESMFSetFromOptions_Default"
/*
   MatSNESMFSetFromOptions_Default - Looks in the options database for 
   any options appropriate for this method.

   Input Parameter:
.  ctx - the matrix free context

*/
static int MatSNESMFSetFromOptions_Default(MatSNESMFCtx ctx)
{
  char*              p;
  int                ierr,flag;
  double             umin;

  PetscFunctionBegin;
  ierr = PetscObjectGetOptionsPrefix((PetscObject)ctx->snes,&p);CHKERRQ(ierr);
  ierr = OptionsGetDouble(p,"-snes_mf_umin",&umin,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = MatSNESMFDefaultSetUmin(ctx->mat,umin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESMFDestroy_Default"
/*
   MatSNESMFDestroy_Default - Frees the space allocated by 
   MatSNESMFCreate_Default(). 

   Input Parameter:
.  ctx - the matrix free context

   Notes: 
   Does not free the ctx, that is handled by the calling routine
*/
static int MatSNESMFDestroy_Default(MatSNESMFCtx ctx)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscFree(ctx->hctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "MatSNESMFDefaultSetUmin_Private"
/*
   The following two routines use the PetscObjectCompose() and PetscObjectQuery()
   mechanism to allow the user to change the Umin parameter used in this method.
*/
int MatSNESMFDefaultSetUmin_Private(Mat mat,double umin)
{
  MatSNESMFCtx     ctx;
  MatSNESMFDefault *hctx;
  int              ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  if (!ctx) {
    SETERRQ(1,1,"MatSNESMFDefaultSetUmin() attached to non-shell matrix");
  }
  hctx = (MatSNESMFDefault *) ctx->hctx;
  hctx->umin = umin;

 PetscFunctionReturn(0);
} 
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "MatSNESMFDefaultSetUmin"
/*@
    MatSNESMFDefaultSetUmin - Sets the "umin" parameter used by the default
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
int MatSNESMFDefaultSetUmin(Mat A,double umin)
{
  int ierr, (*f)(Mat,double);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatSNESMFDefaultSetUmin_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(A,umin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "MatSNESMFCreate_Default"
/*
   MatSNESMFCreate_Default - Standard PETSc code for 
   computing h with matrix-free finite differences.

   Input Parameter:
.  ctx - the matrix free context created by MatSNESMFCreate()

*/
int MatSNESMFCreate_Default(MatSNESMFCtx ctx)
{
  MatSNESMFDefault *hctx;
  int              ierr;

  PetscFunctionBegin;

  /* allocate my own private data structure */
  hctx                     = (MatSNESMFDefault *)PetscMalloc(sizeof(MatSNESMFDefault));CHKPTRQ(hctx);
  ctx->hctx                = (void *) hctx;
  /* set a default for my parameter */
  hctx->umin               = 1.e-6;

  /* set the functions I am providing */
  ctx->ops->compute        = MatSNESMFCompute_Default;
  ctx->ops->destroy        = MatSNESMFDestroy_Default;
  ctx->ops->view           = MatSNESMFView_Default;  
  ctx->ops->printhelp      = MatSNESMFPrintHelp_Default;  
  ctx->ops->setfromoptions = MatSNESMFSetFromOptions_Default;  

  ierr = PetscObjectComposeFunction((PetscObject)ctx->mat,"MatSNESMFDefaultSetUmin_C",
                            "MatSNESMFDefaultSetUmin_Private",
                            (void *) MatSNESMFDefaultSetUmin_Private);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END







