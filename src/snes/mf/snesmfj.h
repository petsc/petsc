#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: snesmfj.c,v 1.72 1998/11/05 04:28:55 bsmith Exp bsmith $";
#endif

#include "src/snes/snesimpl.h"   /*I  "snes.h"   I*/
#include "pinclude/pviewer.h"

typedef struct _p_MatSNESFDMFCtx   *MatSNESFDMFCtx;

typedef struct {
  int (*compute)(MatSNESFDMFCtx,Vec,Vec,Scalar *);
  int (*view)(MatSNESFDMFCtx,Viewer);
  int (*destroy)(MatSNESFDMFCtx);
  int (*printhelp)(MatSNESFDMFCtx);
  int (*setfromoptions)(MatSNESFDMFCtx);
} MFOps;

struct _p_MatSNESFDMFCtx {    /* context for default matrix-free SNES */
  MFOps            *ops;
  MPI_Comm         comm;
  SNES             snes;      /* SNES context */
  Vec              w;         /* work vector */
  PCNullSpace      sp;        /* null space context */
  double           error_rel; /* square root of relative error in computing function */
  Scalar           currenth;  /* last differencing parameter used */
  Scalar           *historyh; /* history of h */
  int              ncurrenth,maxcurrenth;
  void             *hctx;
  char             type_name[256];
};

/* ----------------------------------------------------------------------------------------*/
/*
   Contains the list of registered compute-h routines
*/
FList MatSNESFDMFList              = 0;
int   MatSNESFDMFRegisterAllCalled = 0;

#if defined(USE_DYNAMIC_LIBRARIES)
#define MatSNESFDMFRegister(a,b,c,d) MatSNESFDMFRegister_Private(a,b,c,0)
#else
#define MatSNESFDMFRegister(a,b,c,d) MatSNESFDMFRegister_Private(a,b,c,d)
#endif


extern int MatFDMFCreate_Default(MatSNESFDMFCtx);

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFRegisterAll"
/*@C
  MatSNESFDMFRegisterAll - Registers all of the Krylov subspace methods in the MatSNESFDMF package.

  Not Collective

.keywords: MatSNESFDMF, register, all

.seealso:  MatSNESFDMFRegisterDestroy()
@*/
int MatSNESFDMFRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  MatSNESFDMFRegisterAllCalled = 1;

  ierr = MatSNESFDMFRegister("default",         path,"MatFDMFCreate_Default",MatFDMFCreate_Default);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFSetType"
/*@
      MatSNESFDMFSetType - Sets the method that is used to compute the h in the 
            finite difference matrix free formulation. 

   Input Parameters:
+     mat - the matrix free matrix created via MatCreateSNESFDMF()
-     ftype - the type requested

.seealso: MatCreateSNESFDMF(), MatSNESFDMFRegister()
@*/
int MatSNESFDMFSetType(Mat mat,char *ftype)
{
  int            ierr, (*r)(MatSNESFDMFCtx);
  MatSNESFDMFCtx ctx;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);

  /* already set, so just return */
  if (!PetscStrcmp(ctx->type_name,ftype)) PetscFunctionReturn(0);

  /* destroy the old one if it exists */
  if (ctx->ops->destroy) {
    ierr = (*ctx->ops->destroy)(ctx);CHKERRQ(ierr);
  }

  /* Get the function pointers for the iterative method requested */
  if (!MatSNESFDMFRegisterAllCalled) {ierr = MatSNESFDMFRegisterAll(PETSC_NULL); CHKERRQ(ierr);}

  ierr =  FListFind(ctx->comm, MatSNESFDMFList, ftype,(int (**)(void *)) &r );CHKERRQ(ierr);

  if (!r) SETERRQ(1,1,"Unknown MatSNESFDMF type given");

  ierr = (*r)(ctx); CHKERRQ(ierr);
  ierr = PetscStrncpy(ctx->type_name,ftype,256);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*MC
   MatSNESFDMFRegister - Adds a method to the MatSNESFDMF registry

   Synopsis:
   MatSNESFDMFRegister(char *name_solver,char *path,char *name_create,int (*routine_create)(MatSNESFDMF))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined compute-h module
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   MatSNESFDMFRegister() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   MatSNESFDMFRegister("my_h",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyHCreate",MyHCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     MatSNESFDMFSetType(mfctx,"my_h")
   or at runtime via the option
$     -snes_mf_type my_h

.keywords: MatSNESFDMF, register

.seealso: MatSNESFDMFRegisterAll(), MatSNESFDMFRegisterDestroy()
M*/

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFRegister_Private"
int MatSNESFDMFRegister_Private(char *sname,char *path,char *name,int (*function)(MatSNESFDMFCtx))
{
  int ierr;
  char fullname[256];

  PetscFunctionBegin;
  PetscStrcpy(fullname,path); PetscStrcat(fullname,":");PetscStrcat(fullname,name);
  ierr = FListAdd_Private(&MatSNESFDMFList,sname,fullname,(int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* ----------------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFDestroy_Private"
int MatSNESFDMFDestroy_Private(Mat mat)
{
  int            ierr;
  MatSNESFDMFCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(ctx->w); CHKERRQ(ierr);
  if (ctx->ops->destroy) {ierr = (*ctx->ops->destroy)(ctx);CHKERRQ(ierr);}
  if (ctx->sp) {ierr = PCNullSpaceDestroy(ctx->sp);CHKERRQ(ierr);}
  PetscFree(ctx->ops);
  PetscFree(ctx);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFView_Private"
/*
   MatSNESFDMFView_Private - Views matrix-free parameters.

*/
int MatSNESFDMFView_Private(Mat J,Viewer viewer)
{
  int            ierr;
  MatSNESFDMFCtx ctx;
  MPI_Comm       comm;
  FILE           *fd;
  ViewerType     vtype;

  PetscFunctionBegin;
  PetscObjectGetComm((PetscObject)J,&comm);
  ierr = MatShellGetContext(J,(void **)&ctx); CHKERRQ(ierr);
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
     PetscFPrintf(comm,fd,"  SNES matrix-free approximation:\n");
     PetscFPrintf(comm,fd,"    err=%g (relative error in function evaluation)\n",ctx->error_rel);
     if (ctx->ops->view) {
       ierr = (*ctx->ops->view)(ctx,viewer);CHKERRQ(ierr);
     }
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFMult_Private"
/*
  MatSNESFDMFMult_Private - Default matrix-free form for Jacobian-vector
  product, y = F'(u)*a:

        y ~= ( F(u + ha) - F(u) )/h, 
  where F = nonlinear function, as set by SNESSetFunction()
        u = current iterate
        h = difference interval
*/
int MatSNESFDMFMult_Private(Mat mat,Vec a,Vec y)
{
  MatSNESFDMFCtx ctx;
  SNES           snes;
  Scalar         h, mone = -1.0;
  Vec            w,U,F;
  int            ierr, (*eval_fct)(SNES,Vec,Vec);

  PetscFunctionBegin;
  /* We log matrix-free matrix-vector products separately, so that we can
     separate the performance monitoring from the cases that use conventional
     storage.  We may eventually modify event logging to associate events
     with particular objects, hence alleviating the more general problem. */
  PLogEventBegin(MAT_MatrixFreeMult,a,y,0,0);

  ierr = MatShellGetContext(mat,(void **)&ctx); CHKERRQ(ierr);
  snes = ctx->snes;
  w    = ctx->w;

  ierr = SNESGetSolution(snes,&U); CHKERRQ(ierr);
  if (snes->method_class == SNES_NONLINEAR_EQUATIONS) {
    eval_fct = SNESComputeFunction;
    ierr = SNESGetFunction(snes,&F); CHKERRQ(ierr);
  } else if (snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION) {
    eval_fct = SNESComputeGradient;
    ierr = SNESGetGradient(snes,&F); CHKERRQ(ierr);
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Invalid method class");

  if (!ctx->ops->compute) {
    ierr = MatSNESFDMFSetType(mat,"default");CHKERRQ(ierr);
    ierr = MatSNESFDMFSetFromOptions(mat);CHKERRQ(ierr);
  }
  ierr = (*ctx->ops->compute)(ctx,U,a,&h); CHKERRQ(ierr);

  /* keep a record of the current differencing parameter h */  
  ctx->currenth = h;
#if defined(USE_PETSC_COMPLEX)
  PLogInfo(mat,"Current differencing parameter: %g + %g i\n",PetscReal(h),PetscImaginary(h));
#else
  PLogInfo(mat,"Current differencing parameter: %g\n",h);
#endif
  if (ctx->historyh && ctx->ncurrenth < ctx->maxcurrenth) {
    ctx->historyh[ctx->ncurrenth++] = h;
  }

  /* Evaluate function at F(u + ha) */
  ierr = VecWAXPY(&h,a,U,w); CHKERRQ(ierr);
  ierr = eval_fct(snes,w,y); CHKERRQ(ierr);

  ierr = VecAXPY(&mone,F,y); CHKERRQ(ierr);
  h = 1.0/h;
  ierr = VecScale(&h,y); CHKERRQ(ierr);
  if (ctx->sp) {ierr = PCNullSpaceRemove(ctx->sp,y); CHKERRQ(ierr);}

  PLogEventEnd(MAT_MatrixFreeMult,a,y,0,0);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCreateSNESFDMF"
/*@C
   MatCreateSNESFDMF - Creates a matrix-free matrix
   context for use with a SNES solver.  This matrix can be used as
   the Jacobian argument for the routine SNESSetJacobian().

   Collective on SNES and Vec

   Input Parameters:
+  snes - the SNES context
-  x - vector where SNES solution is to be stored.

   Output Parameter:
.  J - the matrix-free matrix

   Notes:
   The matrix-free matrix context merely contains the function pointers
   and work space for performing finite difference approximations of
   Jacobian-vector products, J(u)*a, via

.vb
     J(u)*a = [J(u+h*a) - J(u)]/h where
     h = error_rel*u'a/||a||^2                        if  |u'a| > umin*||a||_{1}
       = error_rel*umin*sign(u'a)*||a||_{1}/||a||^2   otherwise
 where
     error_rel = square root of relative error in function evaluation
     umin = minimum iterate parameter
.ve

   The user can set the error_rel via MatSNESFDMFSetFunctionError() and 
   umin via MatSNESFDMFDefaultSetUmin()
   See the nonlinear solvers chapter of the users manual for details.

   The user should call MatDestroy() when finished with the matrix-free
   matrix context.

   Options Database Keys:
+  -snes_mf_err <error_rel> - Sets error_rel
.  -snes_mf_unim <umin> - Sets umin (for default routines that compute h only)
-  -snes_mf_ksp_monitor - KSP monitor routine that prints differencing h

.keywords: SNES, default, matrix-free, create, matrix

.seealso: MatDestroy(), MatSNESFDMFSetFunctionError(), MatSNESFDMFDefaultSetUmin()
          MatSNESFDMFSetHHistory(), MatSNESFDMFResetHHistory(),
          MatSNESFDMFGetH(),MatSNESFDMFKSPMonitor()
 
@*/
int MatCreateSNESFDMF(SNES snes,Vec x, Mat *J)
{
  MPI_Comm       comm;
  MatSNESFDMFCtx mfctx;
  int            n, nloc, ierr;

  PetscFunctionBegin;
  mfctx = (MatSNESFDMFCtx) PetscMalloc(sizeof(struct _p_MatSNESFDMFCtx)); CHKPTRQ(mfctx);
  PLogObjectMemory(snes,sizeof(MatSNESFDMFCtx));
  mfctx->comm         = snes->comm;
  mfctx->sp           = 0;
  mfctx->snes         = snes;
  mfctx->error_rel    = 1.e-8; /* assumes double precision */
  mfctx->currenth     = 0.0;
  mfctx->historyh     = PETSC_NULL;
  mfctx->ncurrenth    = 0;
  mfctx->maxcurrenth  = 0;

  /* 
     Create the empty data structure to contain compute-h routines.
     These will be filled in below from the command line options or 
     a later call with MatSNESFDMFSetType() or if that is not called 
     then it will default in the first use of MatSNESFDMFMult_private()
  */
  mfctx->ops                 = (MFOps *)PetscMalloc(sizeof(MFOps)); CHKPTRQ(mfctx->ops); 
  mfctx->ops->compute        = 0;
  mfctx->ops->destroy        = 0;
  mfctx->ops->view           = 0;
  mfctx->ops->printhelp      = 0;
  mfctx->ops->setfromoptions = 0;
  mfctx->hctx                = 0;

  ierr = VecDuplicate(x,&mfctx->w); CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)x,&comm); CHKERRQ(ierr);
  ierr = VecGetSize(x,&n); CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&nloc); CHKERRQ(ierr);
  ierr = MatCreateShell(comm,nloc,n,n,n,mfctx,J); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_MULT,(void*)MatSNESFDMFMult_Private);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_DESTROY,(void *)MatSNESFDMFDestroy_Private);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_VIEW,(void *)MatSNESFDMFView_Private); CHKERRQ(ierr);
  PLogObjectParent(*J,mfctx->w);
  PLogObjectParent(snes,*J);


  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFGetH"
/*@
   MatSNESFDMFSetFromOptions - Sets the MatSNESFDMF options from the command line
     parameter.

   Collective on Mat

   Input Parameters:
.   mat - the matrix obtained with MatCreateSNESFDMF()

.keywords: SNES, matrix-free, parameters

.seealso: MatCreateSNESFDMF(),MatSNESFDMFSetHHistory(), 
          MatSNESFDMFResetHHistory(),MatSNESFDMFKSPMonitor()
@*/
int MatSNESFDMFSetFromOptions(Mat mat)
{
  MatSNESFDMFCtx mfctx;
  int            ierr,flg;
  char           ftype[256],p[64];

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&mfctx); CHKERRQ(ierr);
  if (mfctx) {
    /* allow user to set the type */
    ierr = OptionsGetString(mfctx->snes->prefix,"-snes_mf_type",ftype,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatSNESFDMFSetType(mat,ftype);CHKERRQ(ierr);
    }

    ierr = OptionsGetDouble(mfctx->snes->prefix,"-snes_mf_err",&mfctx->error_rel,&flg);CHKERRQ(ierr);
    if (mfctx->ops->setfromoptions) {
      ierr = (*mfctx->ops->setfromoptions)(mfctx);CHKERRQ(ierr);
    }

    ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
    PetscStrcpy(p,"-");
    if (mfctx->snes->prefix) PetscStrcat(p,mfctx->snes->prefix);
    if (flg) {
      (*PetscHelpPrintf)(mfctx->snes->comm,"   %ssnes_mf_err <err>: set sqrt rel error in function (default %g)\n",p,mfctx->error_rel);
      if (mfctx->ops->printhelp) {
        (*mfctx->ops->printhelp)(mfctx);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFGetH"
/*@
   MatSNESFDMFGetH - Gets the last h that was used as the differencing 
     parameter.

   Not Collective

   Input Parameters:
.   mat - the matrix obtained with MatCreateSNESFDMF()

   Output Paramter:
.  h - the differencing step size

.keywords: SNES, matrix-free, parameters

.seealso: MatCreateSNESFDMF(),MatSNESFDMFSetHHistory(), 
          MatSNESFDMFResetHHistory(),MatSNESFDMFKSPMonitor()
@*/
int MatSNESFDMFGetH(Mat mat,Scalar *h)
{
  MatSNESFDMFCtx ctx;
  int            ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx); CHKERRQ(ierr);
  if (ctx) {
    *h = ctx->currenth;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFKSPMonitor"
/*
   MatSNESFDMFKSPMonitor - A KSP monitor for use with the default PETSc
      SNES matrix free routines. Prints the h differencing parameter used at each
      timestep.

*/
int MatSNESFDMFKSPMonitor(KSP ksp,int n,double rnorm,void *dummy)
{
  PC             pc;
  MatSNESFDMFCtx ctx;
  int            ierr;
  Mat            mat;
  MPI_Comm       comm;
  PetscTruth     nonzeroinitialguess;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc); CHKERRQ(ierr);
  ierr = KSPGetInitialGuessNonzero(ksp,&nonzeroinitialguess);CHKERRQ(ierr);
  ierr = PCGetOperators(pc,&mat,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
  ierr = MatShellGetContext(mat,(void **)&ctx); CHKERRQ(ierr);
  if (!ctx) {
    SETERRQ(1,1,"Matrix is not a matrix free shell matrix");
  }
  if (n > 0 || nonzeroinitialguess) {
#if defined(USE_PETSC_COMPLEX)
    PetscPrintf(comm,"%d KSP Residual norm %14.12e h %g + %g i\n",n,rnorm,
                PetscReal(ctx->currenth),PetscImaginary(ctx->currenth)); 
#else
    PetscPrintf(comm,"%d KSP Residual norm %14.12e h %g \n",n,rnorm,ctx->currenth); 
#endif
  } else {
    PetscPrintf(comm,"%d KSP Residual norm %14.12e\n",n,rnorm); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFSetFunctionError"
/*@
   MatSNESFDMFSetFunctionError - Sets the error_rel for the approximation of
   matrix-vector products using finite differences.

   Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via MatCreateSNESFDMF()
-  error_rel - relative error (should be set to the square root of
               the relative error in the function evaluations)

   Notes:
   The default matrix-free matrix-vector product routine computes
.vb
     J(u)*a = [J(u+h*a) - J(u)]/h where
     h = error_rel*u'a/||a||^2                        if  |u'a| > umin*||a||_{1}
       = error_rel*umin*sign(u'a)*||a||_{1}/||a||^2   else
.ve

   Options Database Keys:
+  -snes_mf_err <error_rel> - Sets error_rel

.keywords: SNES, matrix-free, parameters

.seealso: MatCreateSNESFDMF(),MatSNESFDMFGetH(),
          MatSNESFDMFSetHHistory(), MatSNESFDMFResetHHistory(),
          MatSNESFDMFKSPMonitor()
@*/
int MatSNESFDMFSetFunctionError(Mat mat,double error)
{
  MatSNESFDMFCtx ctx;
  int            ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx); CHKERRQ(ierr);
  if (ctx) {
    if (error != PETSC_DEFAULT) ctx->error_rel = error;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFAddNullSpace"
/*@
   MatSNESFDMFAddNullSpace - Provides a null space that 
   an operator is supposed to have.  Since roundoff will create a 
   small component in the null space, if you know the null space 
   you may have it automatically removed.

   Collective on Mat 

   Input Parameters:
+  J - the matrix-free matrix context
.  has_cnst - PETSC_TRUE or PETSC_FALSE, indicating if null space has constants
.  n - number of vectors (excluding constant vector) in null space
-  vecs - the vectors that span the null space (excluding the constant vector);
          these vectors must be orthonormal

.keywords: SNES, matrix-free, null space

.seealso: MatSNESFDMFGetH(), MatCreateSNESFDMF(),
          MatSNESFDMFSetHHistory(), MatSNESFDMFResetHHistory(),
          MatSNESFDMFKSPMonitor(), MatSNESFDMFErrorRel()

@*/
int MatSNESFDMFAddNullSpace(Mat J,int has_cnst,int n,Vec *vecs)
{
  int            ierr;
  MatSNESFDMFCtx ctx;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscObjectGetComm((PetscObject)J,&comm);

  ierr = MatShellGetContext(J,(void **)&ctx); CHKERRQ(ierr);
  /* no context indicates that it is not the "matrix free" matrix type */
  if (!ctx) PetscFunctionReturn(0);
  ierr = PCNullSpaceCreate(comm,has_cnst,n,vecs,&ctx->sp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFSetHHistory"
/*@
   MatSNESFDMFSetHHistory - Sets an array to collect a history
      of the differencing values h computed for the matrix free product

   Collective on Mat 

   Input Parameters:
+  J - the matrix-free matrix context
.  histroy - space to hold the h history
-  nhistory - number of entries in history, if more h are generated than
              nhistory the later ones are discarded

   Notes:
    Use MatSNESFDMFResetHHistory() to reset the history counter
    and collect a new batch of h.

.keywords: SNES, matrix-free, h history, differencing history

.seealso: MatSNESFDMFGetH(), MatCreateSNESFDMF(),
          MatSNESFDMFResetHHistory(),
          MatSNESFDMFKSPMonitor(), MatSNESFDMFSetFunctionError()

@*/
int MatSNESFDMFSetHHistory(Mat J,Scalar *history,int nhistory)
{
  int            ierr;
  MatSNESFDMFCtx ctx;

  PetscFunctionBegin;

  ierr = MatShellGetContext(J,(void **)&ctx); CHKERRQ(ierr);
  /* no context indicates that it is not the "matrix free" matrix type */
  if (!ctx) PetscFunctionReturn(0);
  ctx->historyh    = history;
  ctx->maxcurrenth = nhistory;
  ctx->currenth    = 0;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFResetHHistory"
/*@
   MatSNESFDMFResetHHistory - Resets the counter to zero to begin 
      collecting a new set of differencing histories.

   Collective on Mat 

   Input Parameters:
.  J - the matrix-free matrix context

   Notes:
    Use MatSNESFDMFSetHHistory() to create the original history counter

.keywords: SNES, matrix-free, h history, differencing history

.seealso: MatSNESFDMFGetH(), MatCreateSNESFDMF(),
          MatSNESFDMFSetHHistory(),
          MatSNESFDMFKSPMonitor(), MatSNESFDMFSetFunctionError()

@*/
int MatSNESFDMFResetHHistory(Mat J)
{
  int            ierr;
  MatSNESFDMFCtx ctx;

  PetscFunctionBegin;

  ierr = MatShellGetContext(J,(void **)&ctx); CHKERRQ(ierr);
  /* no context indicates that it is not the "matrix free" matrix type */
  if (!ctx) PetscFunctionReturn(0);
  ctx->currenth    = 0;

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------- */
extern int VecDot_Seq(Vec,Vec,Scalar *);
extern int VecNorm_Seq(Vec,NormType,double *);

typedef struct {
  double umin;          /* minimum allowable u'a value relative to |u|_1 */
} MatSNESFDMFDefault;
  
#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFCompute_Default"
/*
     MatSNESFDMFCompute_Default - Standard PETSc code for 
   computing h with matrix-free finite differences.

*/
int MatFDMFCompute_Default(MatSNESFDMFCtx ctx,Vec U,Vec a,Scalar *h)
{
  MatSNESFDMFDefault *hctx = (MatSNESFDMFDefault *) ctx->hctx;
  MPI_Comm           comm = ctx->comm;
  double             ovalues[3],norm, sum, umin = hctx->umin;
  Scalar             dot;
  int                ierr;
#if !defined(USE_PETSC_COMPLEX)
  double             values[3];
#endif

  /*
    ierr = VecDot(U,a,&dot); CHKERRQ(ierr);
    ierr = VecNorm(a,NORM_1,&sum); CHKERRQ(ierr);
    ierr = VecNorm(a,NORM_2,&norm); CHKERRQ(ierr);
  */

  /*
     Call the Seq Vector routines and then do a single reduction 
     to reduce the number of communications required
  */

#if !defined(USE_PETSC_COMPLEX)
  PLogEventBegin(VEC_Dot,U,a,0,0);
  ierr = VecDot_Seq(U,a,ovalues); CHKERRQ(ierr);
  PLogEventEnd(VEC_Dot,U,a,0,0);
  PLogEventBegin(VEC_Norm,a,0,0,0);
  ierr = VecNorm_Seq(a,NORM_1,ovalues+1); CHKERRQ(ierr);
  ierr = VecNorm_Seq(a,NORM_2,ovalues+2); CHKERRQ(ierr);
  ovalues[2] = ovalues[2]*ovalues[2];
  PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,comm);
  ierr = MPI_Allreduce(ovalues,values,3,MPI_DOUBLE,MPI_SUM,comm );CHKERRQ(ierr);
  PLogEventBarrierEnd(VEC_NormBarrier,0,0,0,0,comm);
  dot = values[0]; sum = values[1]; norm = sqrt(values[2]);
  PLogEventEnd(VEC_Norm,a,0,0,0);
#else
  {
    Scalar cvalues[3],covalues[3];

    PLogEventBegin(VEC_Dot,U,a,0,0);
    ierr = VecDot_Seq(U,a,covalues); CHKERRQ(ierr);
    PLogEventEnd(VEC_Dot,U,a,0,0);
    PLogEventBegin(VEC_Norm,a,0,0,0);
    ierr = VecNorm_Seq(a,NORM_1,ovalues+1); CHKERRQ(ierr);
    ierr = VecNorm_Seq(a,NORM_2,ovalues+2); CHKERRQ(ierr);
    covalues[1] = ovalues[1];
    covalues[2] = ovalues[2]*ovalues[2];
    PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,comm);
    ierr = MPI_Allreduce(covalues,cvalues,6,MPI_DOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
    PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,comm);
    dot = cvalues[0]; sum = PetscReal(cvalues[1]); norm = sqrt(PetscReal(cvalues[2]));
    PLogEventEnd(VEC_Norm,a,0,0,0);
  }
#endif

  /* Safeguard for step sizes too small */
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
int MatFDMFView_Default(MatSNESFDMFCtx ctx,Viewer viewer)
{
  FILE               *fd;
  ViewerType         vtype;
  MatSNESFDMFDefault *hctx = (MatSNESFDMFDefault *)ctx->hctx;
  int                ierr;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    PetscFPrintf(ctx->comm,fd,"    Using default PETSc compute h routine \n");  
    PetscFPrintf(ctx->comm,fd,"    umin=%g (minimum iterate parameter)\n",hctx->umin);  
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }    
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFPrintHelp_Default"
int MatFDMFPrintHelp_Default(MatSNESFDMFCtx ctx)
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
int MatFDMFSetFromOptions_Default(MatSNESFDMFCtx ctx)
{
  char*              p;
  MatSNESFDMFDefault *hctx = (MatSNESFDMFDefault *)ctx->hctx;
  int                ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetOptionsPrefix((PetscObject)ctx->snes,&p); CHKERRQ(ierr);
  ierr = OptionsGetDouble(p,"-snes_mf_umin",&hctx->umin,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFDestroy_Default"
int MatFDMFDestroy_Default(MatSNESFDMFCtx ctx)
{
  PetscFunctionBegin;
  PetscFree(ctx->hctx);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFCreate_Default"
/*
     MatSNESFDMFCreate_Default - Standard PETSc code for 
   computing h with matrix-free finite differences.

*/
int MatFDMFCreate_Default(MatSNESFDMFCtx ctx)
{
  MatSNESFDMFDefault *hctx;
  PetscFunctionBegin;

  hctx                     = (MatSNESFDMFDefault *) PetscMalloc(sizeof(MatSNESFDMFDefault));CHKPTRQ(hctx);
  ctx->hctx                = (void *) hctx;
  hctx->umin               = 1.e-6;

  ctx->ops->compute        = MatFDMFCompute_Default;
  ctx->ops->destroy        = MatFDMFDestroy_Default;
  ctx->ops->view           = MatFDMFView_Default;  
  ctx->ops->printhelp      = MatFDMFPrintHelp_Default;  
  ctx->ops->setfromoptions = MatFDMFSetFromOptions_Default;  
  PetscFunctionReturn(0);
}




