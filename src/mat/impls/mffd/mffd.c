
#include <petsc/private/matimpl.h>
#include <../src/mat/impls/mffd/mffdimpl.h>   /*I  "petscmat.h"   I*/

PetscFunctionList MatMFFDList              = NULL;
PetscBool         MatMFFDRegisterAllCalled = PETSC_FALSE;

PetscClassId  MATMFFD_CLASSID;
PetscLogEvent MATMFFD_Mult;

static PetscBool MatMFFDPackageInitialized = PETSC_FALSE;
/*@C
  MatMFFDFinalizePackage - This function destroys everything in the MatMFFD package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize(), MatCreateMFFD(), MatCreateSNESMF()
@*/
PetscErrorCode  MatMFFDFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&MatMFFDList));
  MatMFFDPackageInitialized = PETSC_FALSE;
  MatMFFDRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  MatMFFDInitializePackage - This function initializes everything in the MatMFFD package. It is called
  from MatInitializePackage().

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode  MatMFFDInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (MatMFFDPackageInitialized) PetscFunctionReturn(0);
  MatMFFDPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("MatMFFD",&MATMFFD_CLASSID));
  /* Register Constructors */
  PetscCall(MatMFFDRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("MatMult MF",MATMFFD_CLASSID,&MATMFFD_Mult));
 /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = MATMFFD_CLASSID;
    PetscCall(PetscInfoProcessClass("matmffd", 1, classids));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    PetscCall(PetscStrInList("matmffd",logList,',',&pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(MATMFFD_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(MatMFFDFinalizePackage));
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatMFFDSetType_MFFD(Mat mat,MatMFFDType ftype)
{
  MatMFFD        ctx;
  PetscBool      match;
  PetscErrorCode (*r)(MatMFFD);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidCharPointer(ftype,2);
  PetscCall(MatShellGetContext(mat,&ctx));

  /* already set, so just return */
  PetscCall(PetscObjectTypeCompare((PetscObject)ctx,ftype,&match));
  if (match) PetscFunctionReturn(0);

  /* destroy the old one if it exists */
  if (ctx->ops->destroy) PetscCall((*ctx->ops->destroy)(ctx));

  PetscCall(PetscFunctionListFind(MatMFFDList,ftype,&r));
  PetscCheck(r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown MatMFFD type %s given",ftype);
  PetscCall((*r)(ctx));
  PetscCall(PetscObjectChangeTypeName((PetscObject)ctx,ftype));
  PetscFunctionReturn(0);
}

/*@C
    MatMFFDSetType - Sets the method that is used to compute the
    differencing parameter for finite differene matrix-free formulations.

    Input Parameters:
+   mat - the "matrix-free" matrix created via MatCreateSNESMF(), or MatCreateMFFD()
          or MatSetType(mat,MATMFFD);
-   ftype - the type requested, either MATMFFD_WP or MATMFFD_DS

    Level: advanced

    Notes:
    For example, such routines can compute h for use in
    Jacobian-vector products of the form

                        F(x+ha) - F(x)
          F'(u)a  ~=  ----------------
                              h

.seealso: MatCreateSNESMF(), MatMFFDRegister(), MatMFFDSetFunction(), MatCreateMFFD()
@*/
PetscErrorCode  MatMFFDSetType(Mat mat,MatMFFDType ftype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidCharPointer(ftype,2);
  PetscTryMethod(mat,"MatMFFDSetType_C",(Mat,MatMFFDType),(mat,ftype));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_MFFD(Mat,Vec);

typedef PetscErrorCode (*FCN1)(void*,Vec); /* force argument to next function to not be extern C*/
static PetscErrorCode  MatMFFDSetFunctioniBase_MFFD(Mat mat,FCN1 func)
{
  MatMFFD        ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  ctx->funcisetbase = func;
  PetscFunctionReturn(0);
}

typedef PetscErrorCode (*FCN2)(void*,PetscInt,Vec,PetscScalar*); /* force argument to next function to not be extern C*/
static PetscErrorCode  MatMFFDSetFunctioni_MFFD(Mat mat,FCN2 funci)
{
  MatMFFD        ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  ctx->funci = funci;
  PetscCall(MatShellSetOperation(mat,MATOP_GET_DIAGONAL,(void (*)(void))MatGetDiagonal_MFFD));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMFFDGetH_MFFD(Mat mat,PetscScalar *h)
{
  MatMFFD        ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  *h = ctx->currenth;
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatMFFDResetHHistory_MFFD(Mat J)
{
  MatMFFD        ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(J,&ctx));
  ctx->ncurrenth = 0;
  PetscFunctionReturn(0);
}

/*@C
   MatMFFDRegister - Adds a method to the MatMFFD registry.

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined compute-h module
-  routine_create - routine to create method context

   Level: developer

   Notes:
   MatMFFDRegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
   MatMFFDRegister("my_h",MyHCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     MatMFFDSetType(mfctx,"my_h")
   or at runtime via the option
$     -mat_mffd_type my_h

.seealso: MatMFFDRegisterAll(), MatMFFDRegisterDestroy()
 @*/
PetscErrorCode  MatMFFDRegister(const char sname[],PetscErrorCode (*function)(MatMFFD))
{
  PetscFunctionBegin;
  PetscCall(MatInitializePackage());
  PetscCall(PetscFunctionListAdd(&MatMFFDList,sname,function));
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------*/
static PetscErrorCode MatDestroy_MFFD(Mat mat)
{
  MatMFFD        ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  PetscCall(VecDestroy(&ctx->w));
  PetscCall(VecDestroy(&ctx->current_u));
  if (ctx->current_f_allocated) {
    PetscCall(VecDestroy(&ctx->current_f));
  }
  if (ctx->ops->destroy) PetscCall((*ctx->ops->destroy)(ctx));
  PetscCall(PetscHeaderDestroy(&ctx));

  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetBase_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetFunctioniBase_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetFunctioni_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetFunction_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetFunctionError_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetCheckh_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetPeriod_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatMFFDResetHHistory_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetHHistory_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatMFFDGetH_C",NULL));
  PetscFunctionReturn(0);
}

/*
   MatMFFDView_MFFD - Views matrix-free parameters.

*/
static PetscErrorCode MatView_MFFD(Mat J,PetscViewer viewer)
{
  MatMFFD        ctx;
  PetscBool      iascii, viewbase, viewfunction;
  const char     *prefix;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(J,&ctx));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"Matrix-free approximation:\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer,"err=%g (relative error in function evaluation)\n",(double)ctx->error_rel));
    if (!((PetscObject)ctx)->type_name) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"The compute h routine has not yet been set\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer,"Using %s compute h routine\n",((PetscObject)ctx)->type_name));
    }
#if defined(PETSC_USE_COMPLEX)
    if (ctx->usecomplex) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"Using Lyness complex number trick to compute the matrix-vector product\n"));
    }
#endif
    if (ctx->ops->view) {
      PetscCall((*ctx->ops->view)(ctx,viewer));
    }
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)J, &prefix));

    PetscCall(PetscOptionsHasName(((PetscObject)J)->options,prefix, "-mat_mffd_view_base", &viewbase));
    if (viewbase) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Base:\n"));
      PetscCall(VecView(ctx->current_u, viewer));
    }
    PetscCall(PetscOptionsHasName(((PetscObject)J)->options,prefix, "-mat_mffd_view_function", &viewfunction));
    if (viewfunction) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Function:\n"));
      PetscCall(VecView(ctx->current_f, viewer));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

/*
   MatAssemblyEnd_MFFD - Resets the ctx->ncurrenth to zero. This
   allows the user to indicate the beginning of a new linear solve by calling
   MatAssemblyXXX() on the matrix free matrix. This then allows the
   MatCreateMFFD_WP() to properly compute ||U|| only the first time
   in the linear solver rather than every time.

   This function is referenced directly from MatAssemblyEnd_SNESMF(), which may be in a different shared library hence
   it must be labeled as PETSC_EXTERN
*/
PETSC_EXTERN PetscErrorCode MatAssemblyEnd_MFFD(Mat J,MatAssemblyType mt)
{
  MatMFFD        j;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(J,&j));
  PetscCall(MatMFFDResetHHistory(J));
  PetscFunctionReturn(0);
}

/*
  MatMult_MFFD - Default matrix-free form for Jacobian-vector product, y = F'(u)*a:

        y ~= (F(u + ha) - F(u))/h,
  where F = nonlinear function, as set by SNESSetFunction()
        u = current iterate
        h = difference interval
*/
static PetscErrorCode MatMult_MFFD(Mat mat,Vec a,Vec y)
{
  MatMFFD        ctx;
  PetscScalar    h;
  Vec            w,U,F;
  PetscBool      zeroa;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  PetscCheck(ctx->current_u,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"MatMFFDSetBase() has not been called, this is often caused by forgetting to call \n\t\tMatAssemblyBegin/End on the first Mat in the SNES compute function");
  /* We log matrix-free matrix-vector products separately, so that we can
     separate the performance monitoring from the cases that use conventional
     storage.  We may eventually modify event logging to associate events
     with particular objects, hence alleviating the more general problem. */
  PetscCall(PetscLogEventBegin(MATMFFD_Mult,a,y,0,0));

  w = ctx->w;
  U = ctx->current_u;
  F = ctx->current_f;
  /*
      Compute differencing parameter
  */
  if (!((PetscObject)ctx)->type_name) {
    PetscCall(MatMFFDSetType(mat,MATMFFD_WP));
    PetscCall(MatSetFromOptions(mat));
  }
  PetscCall((*ctx->ops->compute)(ctx,U,a,&h,&zeroa));
  if (zeroa) {
    PetscCall(VecSet(y,0.0));
    PetscFunctionReturn(0);
  }

  PetscCheckFalse(mat->erroriffailure && PetscIsInfOrNanScalar(h),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Computed Nan differencing parameter h");
  if (ctx->checkh) {
    PetscCall((*ctx->checkh)(ctx->checkhctx,U,a,&h));
  }

  /* keep a record of the current differencing parameter h */
  ctx->currenth = h;
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscInfo(mat,"Current differencing parameter: %g + %g i\n",(double)PetscRealPart(h),(double)PetscImaginaryPart(h)));
#else
  PetscCall(PetscInfo(mat,"Current differencing parameter: %15.12e\n",(double)PetscRealPart(h)));
#endif
  if (ctx->historyh && ctx->ncurrenth < ctx->maxcurrenth) {
    ctx->historyh[ctx->ncurrenth] = h;
  }
  ctx->ncurrenth++;

#if defined(PETSC_USE_COMPLEX)
  if (ctx->usecomplex) h = PETSC_i*h;
#endif

  /* w = u + ha */
  PetscCall(VecWAXPY(w,h,a,U));

  /* compute func(U) as base for differencing; only needed first time in and not when provided by user */
  if (ctx->ncurrenth == 1 && ctx->current_f_allocated) {
    PetscCall((*ctx->func)(ctx->funcctx,U,F));
  }
  PetscCall((*ctx->func)(ctx->funcctx,w,y));

#if defined(PETSC_USE_COMPLEX)
  if (ctx->usecomplex) {
    PetscCall(VecImaginaryPart(y));
    h    = PetscImaginaryPart(h);
  } else {
    PetscCall(VecAXPY(y,-1.0,F));
  }
#else
  PetscCall(VecAXPY(y,-1.0,F));
#endif
  PetscCall(VecScale(y,1.0/h));
  if (mat->nullsp) PetscCall(MatNullSpaceRemove(mat->nullsp,y));

  PetscCall(PetscLogEventEnd(MATMFFD_Mult,a,y,0,0));
  PetscFunctionReturn(0);
}

/*
  MatGetDiagonal_MFFD - Gets the diagonal for a matrix free matrix

        y ~= (F(u + ha) - F(u))/h,
  where F = nonlinear function, as set by SNESSetFunction()
        u = current iterate
        h = difference interval
*/
PetscErrorCode MatGetDiagonal_MFFD(Mat mat,Vec a)
{
  MatMFFD        ctx;
  PetscScalar    h,*aa,*ww,v;
  PetscReal      epsilon = PETSC_SQRT_MACHINE_EPSILON,umin = 100.0*PETSC_SQRT_MACHINE_EPSILON;
  Vec            w,U;
  PetscInt       i,rstart,rend;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  PetscCheck(ctx->func,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Requires calling MatMFFDSetFunction() first");
  PetscCheck(ctx->funci,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Requires calling MatMFFDSetFunctioni() first");
  w    = ctx->w;
  U    = ctx->current_u;
  PetscCall((*ctx->func)(ctx->funcctx,U,a));
  if (ctx->funcisetbase) {
    PetscCall((*ctx->funcisetbase)(ctx->funcctx,U));
  }
  PetscCall(VecCopy(U,w));

  PetscCall(VecGetOwnershipRange(a,&rstart,&rend));
  PetscCall(VecGetArray(a,&aa));
  for (i=rstart; i<rend; i++) {
    PetscCall(VecGetArray(w,&ww));
    h    = ww[i-rstart];
    if (h == 0.0) h = 1.0;
    if (PetscAbsScalar(h) < umin && PetscRealPart(h) >= 0.0)     h = umin;
    else if (PetscRealPart(h) < 0.0 && PetscAbsScalar(h) < umin) h = -umin;
    h *= epsilon;

    ww[i-rstart] += h;
    PetscCall(VecRestoreArray(w,&ww));
    PetscCall((*ctx->funci)(ctx->funcctx,i,w,&v));
    aa[i-rstart]  = (v - aa[i-rstart])/h;

    PetscCall(VecGetArray(w,&ww));
    ww[i-rstart] -= h;
    PetscCall(VecRestoreArray(w,&ww));
  }
  PetscCall(VecRestoreArray(a,&aa));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatMFFDSetBase_MFFD(Mat J,Vec U,Vec F)
{
  MatMFFD        ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(J,&ctx));
  PetscCall(MatMFFDResetHHistory(J));
  if (!ctx->current_u) {
    PetscCall(VecDuplicate(U,&ctx->current_u));
    PetscCall(VecLockReadPush(ctx->current_u));
  }
  PetscCall(VecLockReadPop(ctx->current_u));
  PetscCall(VecCopy(U,ctx->current_u));
  PetscCall(VecLockReadPush(ctx->current_u));
  if (F) {
    if (ctx->current_f_allocated) PetscCall(VecDestroy(&ctx->current_f));
    ctx->current_f           = F;
    ctx->current_f_allocated = PETSC_FALSE;
  } else if (!ctx->current_f_allocated) {
    PetscCall(MatCreateVecs(J,NULL,&ctx->current_f));
    ctx->current_f_allocated = PETSC_TRUE;
  }
  if (!ctx->w) {
    PetscCall(VecDuplicate(ctx->current_u,&ctx->w));
  }
  J->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

typedef PetscErrorCode (*FCN3)(void*,Vec,Vec,PetscScalar*); /* force argument to next function to not be extern C*/

static PetscErrorCode  MatMFFDSetCheckh_MFFD(Mat J,FCN3 fun,void *ectx)
{
  MatMFFD        ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(J,&ctx));
  ctx->checkh    = fun;
  ctx->checkhctx = ectx;
  PetscFunctionReturn(0);
}

/*@C
   MatMFFDSetOptionsPrefix - Sets the prefix used for searching for all
   MatMFFD options in the database.

   Collective on Mat

   Input Parameters:
+  A - the Mat context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: MatSetFromOptions(), MatCreateSNESMF(), MatCreateMFFD()
@*/
PetscErrorCode  MatMFFDSetOptionsPrefix(Mat mat,const char prefix[])
{
  MatMFFD mfctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscCall(MatShellGetContext(mat,&mfctx));
  PetscValidHeaderSpecific(mfctx,MATMFFD_CLASSID,1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)mfctx,prefix));
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatSetFromOptions_MFFD(PetscOptionItems *PetscOptionsObject,Mat mat)
{
  MatMFFD        mfctx;
  PetscErrorCode ierr;
  PetscBool      flg;
  char           ftype[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  PetscCall(MatShellGetContext(mat,&mfctx));
  PetscValidHeaderSpecific(mfctx,MATMFFD_CLASSID,2);
  ierr = PetscObjectOptionsBegin((PetscObject)mfctx);PetscCall(ierr);
  PetscCall(PetscOptionsFList("-mat_mffd_type","Matrix free type","MatMFFDSetType",MatMFFDList,((PetscObject)mfctx)->type_name,ftype,256,&flg));
  if (flg) {
    PetscCall(MatMFFDSetType(mat,ftype));
  }

  PetscCall(PetscOptionsReal("-mat_mffd_err","set sqrt relative error in function","MatMFFDSetFunctionError",mfctx->error_rel,&mfctx->error_rel,NULL));
  PetscCall(PetscOptionsInt("-mat_mffd_period","how often h is recomputed","MatMFFDSetPeriod",mfctx->recomputeperiod,&mfctx->recomputeperiod,NULL));

  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-mat_mffd_check_positivity","Insure that U + h*a is nonnegative","MatMFFDSetCheckh",flg,&flg,NULL));
  if (flg) {
    PetscCall(MatMFFDSetCheckh(mat,MatMFFDCheckPositivity,NULL));
  }
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscOptionsBool("-mat_mffd_complex","Use Lyness complex number trick to compute the matrix-vector product","None",mfctx->usecomplex,&mfctx->usecomplex,NULL));
#endif
  if (mfctx->ops->setfromoptions) {
    PetscCall((*mfctx->ops->setfromoptions)(PetscOptionsObject,mfctx));
  }
  ierr = PetscOptionsEnd();PetscCall(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatMFFDSetPeriod_MFFD(Mat mat,PetscInt period)
{
  MatMFFD        ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  ctx->recomputeperiod = period;
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatMFFDSetFunction_MFFD(Mat mat,PetscErrorCode (*func)(void*,Vec,Vec),void *funcctx)
{
  MatMFFD        ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  ctx->func    = func;
  ctx->funcctx = funcctx;
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatMFFDSetFunctionError_MFFD(Mat mat,PetscReal error)
{
  MatMFFD        ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  if (error != PETSC_DEFAULT) ctx->error_rel = error;
  PetscFunctionReturn(0);
}

PetscErrorCode  MatMFFDSetHHistory_MFFD(Mat J,PetscScalar history[],PetscInt nhistory)
{
  MatMFFD        ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(J,&ctx));
  ctx->historyh    = history;
  ctx->maxcurrenth = nhistory;
  ctx->currenth    = 0.;
  PetscFunctionReturn(0);
}

/*MC
  MATMFFD - MATMFFD = "mffd" - A matrix free matrix type.

  Level: advanced

  Developers Note: This is implemented on top of MATSHELL to get support for scaling and shifting without requiring duplicate code

.seealso: MatCreateMFFD(), MatCreateSNESMF(), MatMFFDSetFunction(), MatMFFDSetType(),
          MatMFFDSetFunctionError(), MatMFFDDSSetUmin(), MatMFFDSetFunction()
          MatMFFDSetHHistory(), MatMFFDResetHHistory(), MatCreateSNESMF(),
          MatMFFDGetH(),
M*/
PETSC_EXTERN PetscErrorCode MatCreate_MFFD(Mat A)
{
  MatMFFD        mfctx;

  PetscFunctionBegin;
  PetscCall(MatMFFDInitializePackage());

  PetscCall(PetscHeaderCreate(mfctx,MATMFFD_CLASSID,"MatMFFD","Matrix-free Finite Differencing","Mat",PetscObjectComm((PetscObject)A),NULL,NULL));

  mfctx->error_rel                = PETSC_SQRT_MACHINE_EPSILON;
  mfctx->recomputeperiod          = 1;
  mfctx->count                    = 0;
  mfctx->currenth                 = 0.0;
  mfctx->historyh                 = NULL;
  mfctx->ncurrenth                = 0;
  mfctx->maxcurrenth              = 0;
  ((PetscObject)mfctx)->type_name = NULL;

  /*
     Create the empty data structure to contain compute-h routines.
     These will be filled in below from the command line options or
     a later call with MatMFFDSetType() or if that is not called
     then it will default in the first use of MatMult_MFFD()
  */
  mfctx->ops->compute        = NULL;
  mfctx->ops->destroy        = NULL;
  mfctx->ops->view           = NULL;
  mfctx->ops->setfromoptions = NULL;
  mfctx->hctx                = NULL;

  mfctx->func    = NULL;
  mfctx->funcctx = NULL;
  mfctx->w       = NULL;
  mfctx->mat     = A;

  PetscCall(MatSetType(A,MATSHELL));
  PetscCall(MatShellSetContext(A,mfctx));
  PetscCall(MatShellSetOperation(A,MATOP_MULT,(void (*)(void))MatMult_MFFD));
  PetscCall(MatShellSetOperation(A,MATOP_DESTROY,(void (*)(void))MatDestroy_MFFD));
  PetscCall(MatShellSetOperation(A,MATOP_VIEW,(void (*)(void))MatView_MFFD));
  PetscCall(MatShellSetOperation(A,MATOP_ASSEMBLY_END,(void (*)(void))MatAssemblyEnd_MFFD));
  PetscCall(MatShellSetOperation(A,MATOP_SET_FROM_OPTIONS,(void (*)(void))MatSetFromOptions_MFFD));

  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetBase_C",MatMFFDSetBase_MFFD));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetFunctioniBase_C",MatMFFDSetFunctioniBase_MFFD));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetFunctioni_C",MatMFFDSetFunctioni_MFFD));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetFunction_C",MatMFFDSetFunction_MFFD));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetCheckh_C",MatMFFDSetCheckh_MFFD));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetPeriod_C",MatMFFDSetPeriod_MFFD));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetFunctionError_C",MatMFFDSetFunctionError_MFFD));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMFFDResetHHistory_C",MatMFFDResetHHistory_MFFD));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetHHistory_C",MatMFFDSetHHistory_MFFD));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetType_C",MatMFFDSetType_MFFD));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMFFDGetH_C",MatMFFDGetH_MFFD));
  PetscCall(PetscObjectChangeTypeName((PetscObject)A,MATMFFD));
  PetscFunctionReturn(0);
}

/*@
   MatCreateMFFD - Creates a matrix-free matrix. See also MatCreateSNESMF()

   Collective on Vec

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
           This value should be the same as the local size used in creating the
           y vector for the matrix-vector product y = Ax.
.  n - This value should be the same as the local size used in creating the
       x vector for the matrix-vector product y = Ax. (or PETSC_DECIDE to have
       calculated if N is given) For square matrices n is almost always m.
.  M - number of global rows (or PETSC_DETERMINE to have calculated if m is given)
-  N - number of global columns (or PETSC_DETERMINE to have calculated if n is given)

   Output Parameter:
.  J - the matrix-free matrix

   Options Database Keys: call MatSetFromOptions() to trigger these
+  -mat_mffd_type - wp or ds (see MATMFFD_WP or MATMFFD_DS)
.  -mat_mffd_err - square root of estimated relative error in function evaluation
.  -mat_mffd_period - how often h is recomputed, defaults to 1, everytime
.  -mat_mffd_check_positivity - possibly decrease h until U + h*a has only positive values
-  -mat_mffd_complex - use the Lyness trick with complex numbers to compute the matrix-vector product instead of differencing
                       (requires real valued functions but that PETSc be configured for complex numbers)

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

   You can call SNESSetJacobian() with MatMFFDComputeJacobian() if you are using matrix and not a different
   preconditioner matrix

   The user can set the error_rel via MatMFFDSetFunctionError() and
   umin via MatMFFDDSSetUmin(); see Users-Manual: ch_snes for details.

   The user should call MatDestroy() when finished with the matrix-free
   matrix context.

   Options Database Keys:
+  -mat_mffd_err <error_rel> - Sets error_rel
.  -mat_mffd_umin <umin> - Sets umin (for default PETSc routine that computes h only)
-  -mat_mffd_check_positivity - check positivity

.seealso: MatDestroy(), MatMFFDSetFunctionError(), MatMFFDDSSetUmin(), MatMFFDSetFunction()
          MatMFFDSetHHistory(), MatMFFDResetHHistory(), MatCreateSNESMF(),
          MatMFFDGetH(), MatMFFDRegister(), MatMFFDComputeJacobian()

@*/
PetscErrorCode  MatCreateMFFD(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,Mat *J)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm,J));
  PetscCall(MatSetSizes(*J,m,n,M,N));
  PetscCall(MatSetType(*J,MATMFFD));
  PetscCall(MatSetUp(*J));
  PetscFunctionReturn(0);
}

/*@
   MatMFFDGetH - Gets the last value that was used as the differencing
   parameter.

   Not Collective

   Input Parameters:
.  mat - the matrix obtained with MatCreateSNESMF()

   Output Parameter:
.  h - the differencing step size

   Level: advanced

.seealso: MatCreateSNESMF(),MatMFFDSetHHistory(), MatCreateMFFD(), MATMFFD, MatMFFDResetHHistory()
@*/
PetscErrorCode  MatMFFDGetH(Mat mat,PetscScalar *h)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidScalarPointer(h,2);
  PetscUseMethod(mat,"MatMFFDGetH_C",(Mat,PetscScalar*),(mat,h));
  PetscFunctionReturn(0);
}

/*@C
   MatMFFDSetFunction - Sets the function used in applying the matrix free.

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via MatCreateSNESMF() or MatCreateMFFD()
.  func - the function to use
-  funcctx - optional function context passed to function

   Calling Sequence of func:
$     func (void *funcctx, Vec x, Vec f)

+  funcctx - user provided context
.  x - input vector
-  f - computed output function

   Level: advanced

   Notes:
    If you use this you MUST call MatAssemblyBegin()/MatAssemblyEnd() on the matrix free
    matrix inside your compute Jacobian routine

    If this is not set then it will use the function set with SNESSetFunction() if MatCreateSNESMF() was used.

.seealso: MatCreateSNESMF(),MatMFFDGetH(), MatCreateMFFD(), MATMFFD,
          MatMFFDSetHHistory(), MatMFFDResetHHistory(), SNESetFunction()
@*/
PetscErrorCode  MatMFFDSetFunction(Mat mat,PetscErrorCode (*func)(void*,Vec,Vec),void *funcctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscTryMethod(mat,"MatMFFDSetFunction_C",(Mat,PetscErrorCode (*)(void*,Vec,Vec),void*),(mat,func,funcctx));
  PetscFunctionReturn(0);
}

/*@C
   MatMFFDSetFunctioni - Sets the function for a single component

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via MatCreateSNESMF()
-  funci - the function to use

   Level: advanced

   Notes:
    If you use this you MUST call MatAssemblyBegin()/MatAssemblyEnd() on the matrix free
    matrix inside your compute Jacobian routine.
    This function is necessary to compute the diagonal of the matrix.
    funci must not contain any MPI call as it is called inside a loop on the local portion of the vector.

.seealso: MatCreateSNESMF(),MatMFFDGetH(), MatMFFDSetHHistory(), MatMFFDResetHHistory(), SNESetFunction(), MatGetDiagonal()

@*/
PetscErrorCode  MatMFFDSetFunctioni(Mat mat,PetscErrorCode (*funci)(void*,PetscInt,Vec,PetscScalar*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscTryMethod(mat,"MatMFFDSetFunctioni_C",(Mat,PetscErrorCode (*)(void*,PetscInt,Vec,PetscScalar*)),(mat,funci));
  PetscFunctionReturn(0);
}

/*@C
   MatMFFDSetFunctioniBase - Sets the base vector for a single component function evaluation

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via MatCreateSNESMF()
-  func - the function to use

   Level: advanced

   Notes:
    If you use this you MUST call MatAssemblyBegin()/MatAssemblyEnd() on the matrix free
    matrix inside your compute Jacobian routine.
    This function is necessary to compute the diagonal of the matrix.

.seealso: MatCreateSNESMF(),MatMFFDGetH(), MatCreateMFFD(), MATMFFD
          MatMFFDSetHHistory(), MatMFFDResetHHistory(), SNESetFunction(), MatGetDiagonal()
@*/
PetscErrorCode  MatMFFDSetFunctioniBase(Mat mat,PetscErrorCode (*func)(void*,Vec))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscTryMethod(mat,"MatMFFDSetFunctioniBase_C",(Mat,PetscErrorCode (*)(void*,Vec)),(mat,func));
  PetscFunctionReturn(0);
}

/*@
   MatMFFDSetPeriod - Sets how often h is recomputed, by default it is everytime

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via MatCreateSNESMF()
-  period - 1 for everytime, 2 for every second etc

   Options Database Keys:
.  -mat_mffd_period <period> - Sets how often h is recomputed

   Level: advanced

.seealso: MatCreateSNESMF(),MatMFFDGetH(),
          MatMFFDSetHHistory(), MatMFFDResetHHistory()
@*/
PetscErrorCode  MatMFFDSetPeriod(Mat mat,PetscInt period)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(mat,period,2);
  PetscTryMethod(mat,"MatMFFDSetPeriod_C",(Mat,PetscInt),(mat,period));
  PetscFunctionReturn(0);
}

/*@
   MatMFFDSetFunctionError - Sets the error_rel for the approximation of
   matrix-vector products using finite differences.

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via MatCreateMFFD() or MatCreateSNESMF()
-  error_rel - relative error (should be set to the square root of
               the relative error in the function evaluations)

   Options Database Keys:
.  -mat_mffd_err <error_rel> - Sets error_rel

   Level: advanced

   Notes:
   The default matrix-free matrix-vector product routine computes
.vb
     F'(u)*a = [F(u+h*a) - F(u)]/h where
     h = error_rel*u'a/||a||^2                        if  |u'a| > umin*||a||_{1}
       = error_rel*umin*sign(u'a)*||a||_{1}/||a||^2   else
.ve

.seealso: MatCreateSNESMF(),MatMFFDGetH(), MatCreateMFFD(), MATMFFD
          MatMFFDSetHHistory(), MatMFFDResetHHistory()
@*/
PetscErrorCode  MatMFFDSetFunctionError(Mat mat,PetscReal error)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveReal(mat,error,2);
  PetscTryMethod(mat,"MatMFFDSetFunctionError_C",(Mat,PetscReal),(mat,error));
  PetscFunctionReturn(0);
}

/*@
   MatMFFDSetHHistory - Sets an array to collect a history of the
   differencing values (h) computed for the matrix-free product.

   Logically Collective on Mat

   Input Parameters:
+  J - the matrix-free matrix context
.  history - space to hold the history
-  nhistory - number of entries in history, if more entries are generated than
              nhistory, then the later ones are discarded

   Level: advanced

   Notes:
   Use MatMFFDResetHHistory() to reset the history counter and collect
   a new batch of differencing parameters, h.

.seealso: MatMFFDGetH(), MatCreateSNESMF(),
          MatMFFDResetHHistory(), MatMFFDSetFunctionError()

@*/
PetscErrorCode  MatMFFDSetHHistory(Mat J,PetscScalar history[],PetscInt nhistory)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(J,MAT_CLASSID,1);
  if (history) PetscValidScalarPointer(history,2);
  PetscValidLogicalCollectiveInt(J,nhistory,3);
  PetscUseMethod(J,"MatMFFDSetHHistory_C",(Mat,PetscScalar[],PetscInt),(J,history,nhistory));
  PetscFunctionReturn(0);
}

/*@
   MatMFFDResetHHistory - Resets the counter to zero to begin
   collecting a new set of differencing histories.

   Logically Collective on Mat

   Input Parameters:
.  J - the matrix-free matrix context

   Level: advanced

   Notes:
   Use MatMFFDSetHHistory() to create the original history counter.

.seealso: MatMFFDGetH(), MatCreateSNESMF(),
          MatMFFDSetHHistory(), MatMFFDSetFunctionError()

@*/
PetscErrorCode  MatMFFDResetHHistory(Mat J)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(J,MAT_CLASSID,1);
  PetscTryMethod(J,"MatMFFDResetHHistory_C",(Mat),(J));
  PetscFunctionReturn(0);
}

/*@
    MatMFFDSetBase - Sets the vector U at which matrix vector products of the
        Jacobian are computed

    Logically Collective on Mat

    Input Parameters:
+   J - the MatMFFD matrix
.   U - the vector
-   F - (optional) vector that contains F(u) if it has been already computed

    Notes:
    This is rarely used directly

    If F is provided then it is not recomputed. Otherwise the function is evaluated at the base
    point during the first MatMult() after each call to MatMFFDSetBase().

    Level: advanced

@*/
PetscErrorCode  MatMFFDSetBase(Mat J,Vec U,Vec F)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(J,MAT_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,2);
  if (F) PetscValidHeaderSpecific(F,VEC_CLASSID,3);
  PetscTryMethod(J,"MatMFFDSetBase_C",(Mat,Vec,Vec),(J,U,F));
  PetscFunctionReturn(0);
}

/*@C
    MatMFFDSetCheckh - Sets a function that checks the computed h and adjusts
        it to satisfy some criteria

    Logically Collective on Mat

    Input Parameters:
+   J - the MatMFFD matrix
.   fun - the function that checks h
-   ctx - any context needed by the function

    Options Database Keys:
.   -mat_mffd_check_positivity <bool> - Insure that U + h*a is non-negative

    Level: advanced

    Notes:
    For example, MatMFFDCheckPositivity() insures that all entries
       of U + h*a are non-negative

     The function you provide is called after the default h has been computed and allows you to
     modify it.

.seealso:  MatMFFDCheckPositivity()
@*/
PetscErrorCode  MatMFFDSetCheckh(Mat J,PetscErrorCode (*fun)(void*,Vec,Vec,PetscScalar*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(J,MAT_CLASSID,1);
  PetscTryMethod(J,"MatMFFDSetCheckh_C",(Mat,PetscErrorCode (*)(void*,Vec,Vec,PetscScalar*),void*),(J,fun,ctx));
  PetscFunctionReturn(0);
}

/*@
    MatMFFDCheckPositivity - Checks that all entries in U + h*a are positive or
        zero, decreases h until this is satisfied.

    Logically Collective on Vec

    Input Parameters:
+   U - base vector that is added to
.   a - vector that is added
.   h - scaling factor on a
-   dummy - context variable (unused)

    Options Database Keys:
.   -mat_mffd_check_positivity <bool> - Insure that U + h*a is nonnegative

    Level: advanced

    Notes:
    This is rarely used directly, rather it is passed as an argument to
           MatMFFDSetCheckh()

.seealso:  MatMFFDSetCheckh()
@*/
PetscErrorCode  MatMFFDCheckPositivity(void *dummy,Vec U,Vec a,PetscScalar *h)
{
  PetscReal      val, minval;
  PetscScalar    *u_vec, *a_vec;
  PetscInt       i,n;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(U,VEC_CLASSID,2);
  PetscValidHeaderSpecific(a,VEC_CLASSID,3);
  PetscValidScalarPointer(h,4);
  PetscCall(PetscObjectGetComm((PetscObject)U,&comm));
  PetscCall(VecGetArray(U,&u_vec));
  PetscCall(VecGetArray(a,&a_vec));
  PetscCall(VecGetLocalSize(U,&n));
  minval = PetscAbsScalar(*h)*PetscRealConstant(1.01);
  for (i=0; i<n; i++) {
    if (PetscRealPart(u_vec[i] + *h*a_vec[i]) <= 0.0) {
      val = PetscAbsScalar(u_vec[i]/a_vec[i]);
      if (val < minval) minval = val;
    }
  }
  PetscCall(VecRestoreArray(U,&u_vec));
  PetscCall(VecRestoreArray(a,&a_vec));
  PetscCall(MPIU_Allreduce(&minval,&val,1,MPIU_REAL,MPIU_MIN,comm));
  if (val <= PetscAbsScalar(*h)) {
    PetscCall(PetscInfo(U,"Scaling back h from %g to %g\n",(double)PetscRealPart(*h),(double)(.99*val)));
    if (PetscRealPart(*h) > 0.0) *h =  0.99*val;
    else                         *h = -0.99*val;
  }
  PetscFunctionReturn(0);
}
