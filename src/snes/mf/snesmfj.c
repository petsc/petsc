/*$Id: snesmfj.c,v 1.102 2000/04/01 04:21:11 bsmith Exp bsmith $*/

#include "src/snes/snesimpl.h"
#include "src/snes/mf/snesmfj.h"   /*I  "snes.h"   I*/

FList      MatSNESMFList              = 0;
PetscTruth MatSNESMFRegisterAllCalled = PETSC_FALSE;

#undef __FUNC__  
#define __FUNC__ "MatSNESMFSetType"
/*@C
    MatSNESMFSetType - Sets the method that is used to compute the 
    differencing parameter for finite difference matrix-free formulations. 

    Input Parameters:
+   mat - the "matrix-free" matrix created via MatCreateSNESMF()
-   ftype - the type requested

    Level: advanced

    Notes:
    For example, such routines can compute h for use in
    Jacobian-vector products of the form

                        F(x+ha) - F(x)
          F'(u)a  ~=  ----------------
                              h

.seealso: MatCreateSNESMF(), MatSNESMFRegisterDynamic)
@*/
int MatSNESMFSetType(Mat mat,MatSNESMFType ftype)
{
  int          ierr,(*r)(MatSNESMFCtx);
  MatSNESMFCtx ctx;
  PetscTruth   match;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidCharPointer(ftype);

  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);

  /* already set, so just return */
  ierr = PetscTypeCompare((PetscObject)ctx,ftype,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  /* destroy the old one if it exists */
  if (ctx->ops->destroy) {
    ierr = (*ctx->ops->destroy)(ctx);CHKERRQ(ierr);
  }

  /* Get the function pointers for the requrested method */
  if (!MatSNESMFRegisterAllCalled) {ierr = MatSNESMFRegisterAll(PETSC_NULL);CHKERRQ(ierr);}

  ierr =  FListFind(ctx->comm,MatSNESMFList,ftype,(int (**)(void *)) &r);CHKERRQ(ierr);

  if (!r) SETERRQ(1,1,"Unknown MatSNESMF type given");

  ierr = (*r)(ctx);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)ctx,ftype);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*MC
   MatSNESMFRegisterDynamic - Adds a method to the MatSNESMF registry.

   Synopsis:
   MatSNESMFRegisterDynamicchar *name_solver,char *path,char *name_create,int (*routine_create)(MatSNESMF))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined compute-h module
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Level: developer

   Notes:
   MatSNESMFRegisterDynamic) may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   MatSNESMFRegisterDynamic"my_h",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyHCreate",MyHCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     MatSNESMFSetType(mfctx,"my_h")
   or at runtime via the option
$     -snes_mf_type my_h

.keywords: MatSNESMF, register

.seealso: MatSNESMFRegisterAll(), MatSNESMFRegisterDestroy()
M*/

#undef __FUNC__  
#define __FUNC__ "MatSNESMFRegister"
int MatSNESMFRegister(char *sname,char *path,char *name,int (*function)(MatSNESMFCtx))
{
  int ierr;
  char fullname[256];

  PetscFunctionBegin;
  ierr = FListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = FListAdd(&MatSNESMFList,sname,fullname,(int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatSNESMFRegisterDestroy"
/*@C
   MatSNESMFRegisterDestroy - Frees the list of MatSNESMF methods that were
   registered by MatSNESMFRegisterDynamic).

   Not Collective

   Level: developer

.keywords: MatSNESMF, register, destroy

.seealso: MatSNESMFRegisterDynamic), MatSNESMFRegisterAll()
@*/
int MatSNESMFRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (MatSNESMFList) {
    ierr = FListDestroy(MatSNESMFList);CHKERRQ(ierr);
    MatSNESMFList = 0;
  }
  MatSNESMFRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "MatSNESMFDestroy_Private"
int MatSNESMFDestroy_Private(Mat mat)
{
  int          ierr;
  MatSNESMFCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(ctx->w);CHKERRQ(ierr);
  if (ctx->ops->destroy) {ierr = (*ctx->ops->destroy)(ctx);CHKERRQ(ierr);}
  if (ctx->sp) {ierr = PCNullSpaceDestroy(ctx->sp);CHKERRQ(ierr);}
  PetscHeaderDestroy(ctx);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESMFView_Private"
/*
   MatSNESMFView_Private - Views matrix-free parameters.

*/
int MatSNESMFView_Private(Mat J,Viewer viewer)
{
  int          ierr;
  MatSNESMFCtx ctx;
  PetscTruth   isascii;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J,(void **)&ctx);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
     ierr = ViewerASCIIPrintf(viewer,"  SNES matrix-free approximation:\n");CHKERRQ(ierr);
     ierr = ViewerASCIIPrintf(viewer,"    err=%g (relative error in function evaluation)\n",ctx->error_rel);CHKERRQ(ierr);
     ierr = ViewerASCIIPrintf(viewer,"    Using %s compute h routine\n",ctx->type_name);CHKERRQ(ierr);
     if (ctx->ops->view) {
       ierr = (*ctx->ops->view)(ctx,viewer);CHKERRQ(ierr);
     }
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for SNES matrix free matrix",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESMFAssemblyEnd_Private"
/*
   MatSNESMFAssemblyEnd_Private - Resets the ctx->ncurrenth to zero. This 
   allows the user to indicate the beginning of a new linear solve by calling
   MatAssemblyXXX() on the matrix free matrix. This then allows the 
   MatSNESMFCreate_WP() to properly compute ||U|| only the first time
   in the linear solver rather than every time.
*/
int MatSNESMFAssemblyEnd_Private(Mat J)
{
  int            ierr;

  PetscFunctionBegin;
  ierr = MatSNESMFResetHHistory(J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatSNESMFMult_Private"
/*
  MatSNESMFMult_Private - Default matrix-free form for Jacobian-vector
  product, y = F'(u)*a:

        y ~= (F(u + ha) - F(u))/h, 
  where F = nonlinear function, as set by SNESSetFunction()
        u = current iterate
        h = difference interval
*/
int MatSNESMFMult_Private(Mat mat,Vec a,Vec y)
{
  MatSNESMFCtx ctx;
  SNES           snes;
  Scalar         h,mone = -1.0;
  Vec            w,U,F;
  int            ierr,(*eval_fct)(SNES,Vec,Vec)=0;

  PetscFunctionBegin;
  /* We log matrix-free matrix-vector products separately, so that we can
     separate the performance monitoring from the cases that use conventional
     storage.  We may eventually modify event logging to associate events
     with particular objects, hence alleviating the more general problem. */
  PLogEventBegin(MAT_MatrixFreeMult,a,y,0,0);

  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  snes = ctx->snes;
  w    = ctx->w;
  ierr = SNESGetSolution(snes,&U);CHKERRQ(ierr);

  /* 
      Compute differencing parameter 
  */
  if (!ctx->ops->compute) {
    ierr = MatSNESMFSetType(mat,MATSNESMF_DEFAULT);CHKERRQ(ierr);
    ierr = MatSNESMFSetFromOptions(mat);CHKERRQ(ierr);
  }
  ierr = (*ctx->ops->compute)(ctx,U,a,&h);CHKERRQ(ierr);

  /* keep a record of the current differencing parameter h */  
  ctx->currenth = h;
#if defined(PETSC_USE_COMPLEX)
  PLogInfo(mat,"Current differencing parameter: %g + %g i\n",PetscRealPart(h),PetscImaginaryPart(h));
#else
  PLogInfo(mat,"Current differencing parameter: %g\n",h);
#endif
  if (ctx->historyh && ctx->ncurrenth < ctx->maxcurrenth) {
    ctx->historyh[ctx->ncurrenth] = h;
  }
  ctx->ncurrenth++;

  /* w = u + ha */
  ierr = VecWAXPY(&h,a,U,w);CHKERRQ(ierr);


  if (!ctx->func) {
    if (snes->method_class == SNES_NONLINEAR_EQUATIONS) {
      eval_fct = SNESComputeFunction;
      ierr     = SNESGetFunction(snes,&F,PETSC_NULL);CHKERRQ(ierr);
    } else if (snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION) {
      eval_fct = SNESComputeGradient;
      ierr     = SNESGetGradient(snes,&F,PETSC_NULL);CHKERRQ(ierr);
    } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Invalid method class");
    ierr = eval_fct(snes,w,y);CHKERRQ(ierr);
  } else {
    F = ctx->funcvec;
    /* compute func(U) as base for differencing */
    if (ctx->ncurrenth == 1) {
      ierr = (*ctx->func)(snes,U,F,ctx->funcctx);CHKERRQ(ierr);
    }
    ierr = (*ctx->func)(snes,w,y,ctx->funcctx);CHKERRQ(ierr);
  }

  ierr = VecAXPY(&mone,F,y);CHKERRQ(ierr);
  h    = 1.0/h;
  ierr = VecScale(&h,y);CHKERRQ(ierr);
  if (ctx->sp) {ierr = PCNullSpaceRemove(ctx->sp,y);CHKERRQ(ierr);}

  PLogEventEnd(MAT_MatrixFreeMult,a,y,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCreateSNESMF"
/*@C
   MatCreateSNESMF - Creates a matrix-free matrix context for use with
   a SNES solver.  This matrix can be used as the Jacobian argument for
   the routine SNESSetJacobian().

   Collective on SNES and Vec

   Input Parameters:
+  snes - the SNES context
-  x - vector where SNES solution is to be stored.

   Output Parameter:
.  J - the matrix-free matrix

   Level: advanced

   Notes:
   The matrix-free matrix context merely contains the function pointers
   and work space for performing finite difference approximations of
   Jacobian-vector products, F'(u)*a, 

   The default code uses the following approach to compute h

.vb
     F'(u)*a = [F(u+h*a) - F(u)]/h where
     h = error_rel*u'a/||a||^2                        if  |u'a| > umin*||a||_{1}
       = error_rel*umin*sign(u'a)*||a||_{1}/||a||^2   otherwise
 where
     error_rel = square root of relative error in function evaluation
     umin = minimum iterate parameter
.ve

   The user can set the error_rel via MatSNESMFSetFunctionError() and 
   umin via MatSNESMFDefaultSetUmin(); see the nonlinear solvers chapter
   of the users manual for details.

   The user should call MatDestroy() when finished with the matrix-free
   matrix context.

   Options Database Keys:
+  -snes_mf_err <error_rel> - Sets error_rel
.  -snes_mf_unim <umin> - Sets umin (for default PETSc routine that computes h only)
-  -snes_mf_ksp_monitor - KSP monitor routine that prints differencing h

.keywords: SNES, default, matrix-free, create, matrix

.seealso: MatDestroy(), MatSNESMFSetFunctionError(), MatSNESMFDefaultSetUmin()
          MatSNESMFSetHHistory(), MatSNESMFResetHHistory(),
          MatSNESMFGetH(),MatSNESMFKSPMonitor(), MatSNESMFRegisterDynamic)
 
@*/
int MatCreateSNESMF(SNES snes,Vec x,Mat *J)
{
  MPI_Comm     comm;
  MatSNESMFCtx mfctx;
  int          n,nloc,ierr;

  PetscFunctionBegin;
  PetscHeaderCreate(mfctx,_p_MatSNESMFCtx,struct _MFOps,MATSNESMFCTX_COOKIE,0,"SNESMF",snes->comm,MatSNESMFDestroy_Private,MatSNESMFView_Private);
  PLogObjectCreate(mfctx);
  mfctx->sp              = 0;
  mfctx->snes            = snes;
  mfctx->error_rel       = 1.e-8; /* assumes PetscReal precision */
  mfctx->recomputeperiod = 1;
  mfctx->count           = 0;
  mfctx->currenth        = 0.0;
  mfctx->historyh        = PETSC_NULL;
  mfctx->ncurrenth       = 0;
  mfctx->maxcurrenth     = 0;
  mfctx->type_name       = 0;

  /* 
     Create the empty data structure to contain compute-h routines.
     These will be filled in below from the command line options or 
     a later call with MatSNESMFSetType() or if that is not called 
     then it will default in the first use of MatSNESMFMult_private()
  */
  mfctx->ops->compute        = 0;
  mfctx->ops->destroy        = 0;
  mfctx->ops->view           = 0;
  mfctx->ops->printhelp      = 0;
  mfctx->ops->setfromoptions = 0;
  mfctx->hctx                = 0;

  mfctx->func                = 0;
  mfctx->funcctx             = 0;
  mfctx->funcvec             = 0;

  ierr = VecDuplicate(x,&mfctx->w);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&nloc);CHKERRQ(ierr);
  ierr = MatCreateShell(comm,nloc,nloc,n,n,mfctx,J);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_MULT,(void*)MatSNESMFMult_Private);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_DESTROY,(void *)MatSNESMFDestroy_Private);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_VIEW,(void *)MatSNESMFView_Private);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_ASSEMBLY_END,(void *)MatSNESMFAssemblyEnd_Private);CHKERRQ(ierr);
  PLogObjectParent(*J,mfctx->w);
  PLogObjectParent(snes,*J);

  mfctx->mat = *J;


  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESMFSetFromOptions"
/*@
   MatSNESMFSetFromOptions - Sets the MatSNESMF options from the command line
   parameter.

   Collective on Mat

   Input Parameters:
.  mat - the matrix obtained with MatCreateSNESMF()

   Options Database Keys:
+  -snes_mf_type - <default,wp>
-  -snes_mf_err - square root of estimated relative error in function evaluation
-  -snes_mf_period - how often h is recomputed, defaults to 1, everytime

   Level: advanced

.keywords: SNES, matrix-free, parameters

.seealso: MatCreateSNESMF(),MatSNESMFSetHHistory(), 
          MatSNESMFResetHHistory(), MatSNESMFKSPMonitor()
@*/
int MatSNESMFSetFromOptions(Mat mat)
{
  MatSNESMFCtx mfctx;
  int          ierr;
  PetscTruth   flg;
  char         ftype[256],p[64];

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&mfctx);CHKERRQ(ierr);
  if (mfctx) {
    /* allow user to set the type */
    ierr = OptionsGetString(mfctx->snes->prefix,"-snes_mf_type",ftype,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatSNESMFSetType(mat,ftype);CHKERRQ(ierr);
    }

    ierr = OptionsGetDouble(mfctx->snes->prefix,"-snes_mf_err",&mfctx->error_rel,PETSC_NULL);CHKERRQ(ierr);
    ierr = OptionsGetInt(mfctx->snes->prefix,"-snes_mf_period",&mfctx->recomputeperiod,PETSC_NULL);CHKERRQ(ierr);
    if (mfctx->ops->setfromoptions) {
      ierr = (*mfctx->ops->setfromoptions)(mfctx);CHKERRQ(ierr);
    }

    ierr = OptionsHasName(PETSC_NULL,"-help",&flg);CHKERRQ(ierr);
    ierr = PetscStrcpy(p,"-");CHKERRQ(ierr);
    if (mfctx->snes->prefix) {ierr = PetscStrcat(p,mfctx->snes->prefix);CHKERRQ(ierr);}
    if (flg) {
      ierr = (*PetscHelpPrintf)(mfctx->snes->comm,"   %ssnes_mf_err <err>: set sqrt rel error in function (default %g)\n",p,mfctx->error_rel);CHKERRQ(ierr);
      ierr = (*PetscHelpPrintf)(mfctx->snes->comm,"   %ssnes_mf_period <p>: how often h is recomputed (default 1, everytime)\n",p);CHKERRQ(ierr);
      if (mfctx->ops->printhelp) {
        ierr = (*mfctx->ops->printhelp)(mfctx);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESMFGetH"
/*@
   MatSNESMFGetH - Gets the last value that was used as the differencing 
   parameter.

   Not Collective

   Input Parameters:
.  mat - the matrix obtained with MatCreateSNESMF()

   Output Paramter:
.  h - the differencing step size

   Level: advanced

.keywords: SNES, matrix-free, parameters

.seealso: MatCreateSNESMF(),MatSNESMFSetHHistory(), 
          MatSNESMFResetHHistory(),MatSNESMFKSPMonitor()
@*/
int MatSNESMFGetH(Mat mat,Scalar *h)
{
  MatSNESMFCtx ctx;
  int          ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  if (ctx) {
    *h = ctx->currenth;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESMFKSPMonitor"
/*
   MatSNESMFKSPMonitor - A KSP monitor for use with the default PETSc
   SNES matrix free routines. Prints the differencing parameter used at 
   each step.
*/
int MatSNESMFKSPMonitor(KSP ksp,int n,PetscReal rnorm,void *dummy)
{
  PC             pc;
  MatSNESMFCtx   ctx;
  int            ierr;
  Mat            mat;
  MPI_Comm       comm;
  PetscTruth     nonzeroinitialguess;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = KSPGetInitialGuessNonzero(ksp,&nonzeroinitialguess);CHKERRQ(ierr);
  ierr = PCGetOperators(pc,&mat,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  if (!ctx) {
    SETERRQ(1,1,"Matrix is not a matrix free shell matrix");
  }
  if (n > 0 || nonzeroinitialguess) {
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscPrintf(comm,"%d KSP Residual norm %14.12e h %g + %g i\n",n,rnorm,
                PetscRealPart(ctx->currenth),PetscImaginaryPart(ctx->currenth));CHKERRQ(ierr);
#else
    ierr = PetscPrintf(comm,"%d KSP Residual norm %14.12e h %g \n",n,rnorm,ctx->currenth);CHKERRQ(ierr); 
#endif
  } else {
    ierr = PetscPrintf(comm,"%d KSP Residual norm %14.12e\n",n,rnorm);CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESMFSetFunction"
/*@C
   MatSNESMFSetFunction - Sets the function used in applying the matrix free.

   Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via MatCreateSNESMF()
.  v   - workspace vector
.  func - the function to use
-  funcctx - optional function context passed to function

   Level: advanced

   Notes:
    If you use this you MUST call MatAssemblyBegin()/MatAssemblyEnd() on the matrix free
    matrix inside your compute Jacobian routine

    If this is not set then it will use the function set with SNESSetFunction()

.keywords: SNES, matrix-free, function

.seealso: MatCreateSNESMF(),MatSNESMFGetH(),
          MatSNESMFSetHHistory(), MatSNESMFResetHHistory(),
          MatSNESMFKSPMonitor(), SNESetFunction()
@*/
int MatSNESMFSetFunction(Mat mat,Vec v,int (*func)(SNES,Vec,Vec,void *),void *funcctx)
{
  MatSNESMFCtx ctx;
  int          ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  if (ctx) {
    ctx->func    = func;
    ctx->funcctx = funcctx;
    ctx->funcvec = v;
  }
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatSNESMFSetPeriod"
/*@
   MatSNESMFSetPeriod - Sets how often h is recomputed, by default it is everytime

   Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via MatCreateSNESMF()
-  period - 1 for everytime, 2 for every second etc

   Options Database Keys:
+  -snes_mf_period <period>

   Level: advanced


.keywords: SNES, matrix-free, parameters

.seealso: MatCreateSNESMF(),MatSNESMFGetH(),
          MatSNESMFSetHHistory(), MatSNESMFResetHHistory(),
          MatSNESMFKSPMonitor()
@*/
int MatSNESMFSetPeriod(Mat mat,int period)
{
  MatSNESMFCtx ctx;
  int          ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  if (ctx) {
    ctx->recomputeperiod = period;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESMFSetFunctionError"
/*@
   MatSNESMFSetFunctionError - Sets the error_rel for the approximation of
   matrix-vector products using finite differences.

   Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via MatCreateSNESMF()
-  error_rel - relative error (should be set to the square root of
               the relative error in the function evaluations)

   Options Database Keys:
+  -snes_mf_err <error_rel> - Sets error_rel

   Level: advanced

   Notes:
   The default matrix-free matrix-vector product routine computes
.vb
     F'(u)*a = [F(u+h*a) - F(u)]/h where
     h = error_rel*u'a/||a||^2                        if  |u'a| > umin*||a||_{1}
       = error_rel*umin*sign(u'a)*||a||_{1}/||a||^2   else
.ve

.keywords: SNES, matrix-free, parameters

.seealso: MatCreateSNESMF(),MatSNESMFGetH(),
          MatSNESMFSetHHistory(), MatSNESMFResetHHistory(),
          MatSNESMFKSPMonitor()
@*/
int MatSNESMFSetFunctionError(Mat mat,PetscReal error)
{
  MatSNESMFCtx ctx;
  int          ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  if (ctx) {
    if (error != PETSC_DEFAULT) ctx->error_rel = error;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESMFAddNullSpace"
/*@
   MatSNESMFAddNullSpace - Provides a null space that an operator is
   supposed to have.  Since roundoff will create a small component in
   the null space, if you know the null space you may have it
   automatically removed.

   Collective on Mat 

   Input Parameters:
+  J - the matrix-free matrix context
-  nullsp - object created with PCNullSpaceCreate()

   Level: advanced

.keywords: SNES, matrix-free, null space

.seealso: PCNullSpaceCreate(), MatSNESMFGetH(), MatCreateSNESMF(),
          MatSNESMFSetHHistory(), MatSNESMFResetHHistory(),
          MatSNESMFKSPMonitor(), MatSNESMFErrorRel()
@*/
int MatSNESMFAddNullSpace(Mat J,PCNullSpace nullsp)
{
  int          ierr;
  MatSNESMFCtx ctx;
  MPI_Comm     comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)J,&comm);CHKERRQ(ierr);

  ierr = MatShellGetContext(J,(void **)&ctx);CHKERRQ(ierr);
  /* no context indicates that it is not the "matrix free" matrix type */
  if (!ctx) PetscFunctionReturn(0);
  ctx->sp = nullsp;
  ierr = PetscObjectReference((PetscObject)nullsp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESMFSetHHistory"
/*@
   MatSNESMFSetHHistory - Sets an array to collect a history of the
   differencing values (h) computed for the matrix-free product.

   Collective on Mat 

   Input Parameters:
+  J - the matrix-free matrix context
.  histroy - space to hold the history
-  nhistory - number of entries in history, if more entries are generated than
              nhistory, then the later ones are discarded

   Level: advanced

   Notes:
   Use MatSNESMFResetHHistory() to reset the history counter and collect
   a new batch of differencing parameters, h.

.keywords: SNES, matrix-free, h history, differencing history

.seealso: MatSNESMFGetH(), MatCreateSNESMF(),
          MatSNESMFResetHHistory(),
          MatSNESMFKSPMonitor(), MatSNESMFSetFunctionError()

@*/
int MatSNESMFSetHHistory(Mat J,Scalar *history,int nhistory)
{
  int          ierr;
  MatSNESMFCtx ctx;

  PetscFunctionBegin;

  ierr = MatShellGetContext(J,(void **)&ctx);CHKERRQ(ierr);
  /* no context indicates that it is not the "matrix free" matrix type */
  if (!ctx) PetscFunctionReturn(0);
  ctx->historyh    = history;
  ctx->maxcurrenth = nhistory;
  ctx->currenth    = 0;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESMFResetHHistory"
/*@
   MatSNESMFResetHHistory - Resets the counter to zero to begin 
   collecting a new set of differencing histories.

   Collective on Mat 

   Input Parameters:
.  J - the matrix-free matrix context

   Level: advanced

   Notes:
   Use MatSNESMFSetHHistory() to create the original history counter.

.keywords: SNES, matrix-free, h history, differencing history

.seealso: MatSNESMFGetH(), MatCreateSNESMF(),
          MatSNESMFSetHHistory(),
          MatSNESMFKSPMonitor(), MatSNESMFSetFunctionError()

@*/
int MatSNESMFResetHHistory(Mat J)
{
  int          ierr;
  MatSNESMFCtx ctx;

  PetscFunctionBegin;

  ierr = MatShellGetContext(J,(void **)&ctx);CHKERRQ(ierr);
  /* no context indicates that it is not the "matrix free" matrix type */
  if (!ctx) PetscFunctionReturn(0);
  ctx->ncurrenth    = 0;

  PetscFunctionReturn(0);
}

