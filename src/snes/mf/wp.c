#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: snesmfjdef.c,v 1.3 1998/11/09 03:32:04 bsmith Exp bsmith $";
#endif
/*
      Implements the default PETSc approach for computing the h 
  parameter used with the finite difference based matrix free Jacobian.

      To make your own: clone this file and modify for your needs.

   Mandatory functions:
   -------------------
      MatSNESFDMFCompute_  - for a given point and direction computes h

      MatSNESFDMFCreate_ - files in the MatSNESFDMF data structure
                           for this particular implementation

      
   Option functions:
   ----------------
      MatSNESFDMFView_ - prints information about the parameters being 
                         used. This is called when SNESView() or -snes_view
                         is used 

      MatSNESFDMFPrintHelp_ - prints a help message on what options are
                          available for this implementation

      MatSNESFDMFSetFromOptions_ - checks the options database for options that 
                               apply to this method.

      MatSNESFDMFDestroy_ - frees any space allocated by the routines above

   Function particular to this method:
   ----------------------------------      
*/

/*
    This include file defines the data structure  MatSNESFDMF that 
   includes information about the computation of h. It is shared by 
   all implementations that people provide
*/
#include "src/snes/mf/snesmfj.h"   /*I  "snes.h"   I*/

/*
      The default method has one parameter that is used to 
   "cutoff" very small values. This is stored in a data structure
   that is only visible to this file. If your method has no parameters
   it can omit this, if it has several simply reorganize the data structure.
   The data structure is "hung-off" the MatSNESFDMF data structure in
   the void *hctx; field.
*/
typedef struct {
  double umin;          /* minimum allowable u'a value relative to |u|_1 */
} MatSNESFDMFDefault;
  
extern int VecDot_Seq(Vec,Vec,Scalar *);
extern int VecNorm_Seq(Vec,NormType,double *);

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFCompute_Default"
/*
     MatSNESFDMFCompute_Default - Standard PETSc code for 
   computing h with matrix-free finite differences.

  Input Parameters:
+   ctx - the matrix free context
.   U - the location at which you want the Jacobian
-   a - the direction you want the derivative

  Output Parameter:
.   h - the scale computed

*/
static int MatSNESFDMFCompute_Default(MatSNESFDMFCtx ctx,Vec U,Vec a,Scalar *h)
{
  MatSNESFDMFDefault *hctx = (MatSNESFDMFDefault *) ctx->hctx;
  MPI_Comm           comm = ctx->comm;
  double             norm, sum, umin = hctx->umin;
  Scalar             dot;
  int                ierr;
  Scalar             ovalues[3],values[3];

  /*
     This algorithm requires 2 norms and 1 inner product. Rather than
     use directly the VecNorm() and VecDot() routines (and thus have 
     three separate collective operations, we use the sequential routines
     and manually call MPI for the collective phase.

     We sparately log the VecDot() and VecNorm() stages to get 
     accurate profiling. Note that the collective time for the VecDot()
     is "free", because it piggy-backs on the collective time for the norm
  */


  PLogEventBegin(VEC_Dot,U,a,0,0);
  ierr = VecDot_Seq(U,a,ovalues); CHKERRQ(ierr);
  PLogEventEnd(VEC_Dot,U,a,0,0);
  PLogEventBegin(VEC_Norm,a,0,0,0);
  ierr = VecNorm_Seq(a,NORM_1,(double *)(ovalues+1)); CHKERRQ(ierr);
  ierr = VecNorm_Seq(a,NORM_2,(double *)(ovalues+2)); CHKERRQ(ierr);
  ovalues[2] = ovalues[2]*ovalues[2];
  PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,comm);
#if !defined(USE_PETSC_COMPLEX)
  ierr = MPI_Allreduce(ovalues,values,3,MPI_DOUBLE,MPI_SUM,comm );CHKERRQ(ierr);
#else
#endif
  PLogEventBarrierEnd(VEC_NormBarrier,0,0,0,0,comm);
  dot = values[0]; sum = PetscReal(values[1]); norm = sqrt(PetscReal(values[2]));
  PLogEventEnd(VEC_Norm,a,0,0,0);

  /* 
     Safeguard for step sizes that are "too small"
  */
  if (sum == 0.0) {dot = 1.0; norm = 1.0;}
#if defined(USE_PETSC_COMPLEX)
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
#define __FUNC__ "MatSNESFDMFView_Default"
/*
   MatSNESFDMFView_Default - Prints information about this particular 
     method for computing h. Note that this does not print the general
     information about the matrix free, that is printed by the calling
     routine.

  Input Parameters:
+   ctx - the matrix free context
-   viewer - the PETSc viewer

*/   
static int MatSNESFDMFView_Default(MatSNESFDMFCtx ctx,Viewer viewer)
{
  FILE               *fd;
  ViewerType         vtype;
  MatSNESFDMFDefault *hctx = (MatSNESFDMFDefault *)ctx->hctx;
  int                ierr;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  
  /*
     Currently this only handles the ascii file viewers, others
     could be added, but for this type of object other viewers
     make less sense
  */
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    PetscFPrintf(ctx->comm,fd,"    umin=%g (minimum iterate parameter)\n",hctx->umin);  
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }    
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFPrintHelp_Default"
/*
   MatSNESFDMFPrintHelp_Default - Prints a list of all the options 
      this particular method supports.

  Input Parameter:
.  ctx - the matrix free context

*/
static int MatSNESFDMFPrintHelp_Default(MatSNESFDMFCtx ctx)
{
  char*              p;
  MatSNESFDMFDefault *hctx = (MatSNESFDMFDefault *)ctx->hctx;
  int                ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetOptionsPrefix((PetscObject)ctx->snes,&p); CHKERRQ(ierr);
  if (!p) p = "";
  (*PetscHelpPrintf)(ctx->comm,"   -%ssnes_mf_umin <umin> see users manual (default %g)\n",p,hctx->umin);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFSetFromOptions_Default"
/*
   MatSNESFDMFSetFromOptions_Default - Looks in the options database for 
     any options appropriate for this method

  Input Parameter:
.  ctx - the matrix free context

*/
static int MatSNESFDMFSetFromOptions_Default(MatSNESFDMFCtx ctx)
{
  char*              p;
  int                ierr,flag;
  double             umin;

  PetscFunctionBegin;
  ierr = PetscObjectGetOptionsPrefix((PetscObject)ctx->snes,&p); CHKERRQ(ierr);
  ierr = OptionsGetDouble(p,"-snes_mf_umin",&umin,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = MatSNESFDMFDefaultSetUmin(ctx->mat,umin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFDestroy_Default"
/*
   MatSNESFDMFDestroy_Default - Frees the space allocated by 
       MatSNESFDMFCreate_Default(). 

  Input Parameter:
.  ctx - the matrix free context

   Notes: does not free the ctx, that is handled by the calling routine

*/
static int MatSNESFDMFDestroy_Default(MatSNESFDMFCtx ctx)
{
  PetscFunctionBegin;
  PetscFree(ctx->hctx);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFDefaultSetUmin_Private"
/*
      The following two routines use the PetscObjectCompose() and PetscObjectQuery()
   mechanism to allow the user to change the Umin parameter used in this method.
*/
int MatSNESFDMFDefaultSetUmin_Private(Mat mat,double umin)
{
  MatSNESFDMFCtx     ctx;
  MatSNESFDMFDefault *hctx;
  int                ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  if (!ctx) {
    SETERRQ(1,1,"MatSNESFDMFDefaultSetUmin() attached to non-shell matrix");
  }
  hctx = (MatSNESFDMFDefault *) ctx->hctx;
  hctx->umin = umin;

 PetscFunctionReturn(0);
} 
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFDefaultSetUmin"
/*@
    MatSNESFDMFDefaultSetUmin - Sets the "umin" parameter used by the default
             PETSc routine for computing h

  Input Parameters:
+   A - the matrix created with MatCreateSNESFDMF()
-   umin - the parameter

  Notes:
   See the manual page for MatCreateSNESFDMF() for a complete description of the
   algorithm used to compute h.

.seealso: MatSNESFDMFSetFunctionError(), MatCreateSNESFDMF()

@*/
int MatSNESFDMFDefaultSetUmin(Mat A,double umin)
{
  int ierr, (*f)(Mat,double);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatSNESFDMFDefaultSetUmin_C",(void **)&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(A,umin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFCreate_Default"
/*
     MatSNESFDMFCreate_Default - Standard PETSc code for 
   computing h with matrix-free finite differences.

   Input Parameter:
.  ctx - the matrix free context created by MatSNESFDMFCreate()

*/
int MatSNESFDMFCreate_Default(MatSNESFDMFCtx ctx)
{
  MatSNESFDMFDefault *hctx;
  int                ierr;

  PetscFunctionBegin;

  /* allocate my own private data structure */
  hctx                     = (MatSNESFDMFDefault *)PetscMalloc(sizeof(MatSNESFDMFDefault));CHKPTRQ(hctx);
  ctx->hctx                = (void *) hctx;
  /* set a default for my parameter */
  hctx->umin               = 1.e-6;

  /* set the functions I am providing */
  ctx->ops->compute        = MatSNESFDMFCompute_Default;
  ctx->ops->destroy        = MatSNESFDMFDestroy_Default;
  ctx->ops->view           = MatSNESFDMFView_Default;  
  ctx->ops->printhelp      = MatSNESFDMFPrintHelp_Default;  
  ctx->ops->setfromoptions = MatSNESFDMFSetFromOptions_Default;  

  ierr = PetscObjectComposeFunction((PetscObject)ctx->mat,"MatSNESFDMFDefaultSetUmin_C",
                            "MatSNESFDMFDefaultSetUmin_Private",
                            (void *) MatSNESFDMFDefaultSetUmin_Private);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END







