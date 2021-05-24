
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&MatMFFDList);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MatMFFDPackageInitialized) PetscFunctionReturn(0);
  MatMFFDPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("MatMFFD",&MATMFFD_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = MatMFFDRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("MatMult MF",MATMFFD_CLASSID,&MATMFFD_Mult);CHKERRQ(ierr);
 /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = MATMFFD_CLASSID;
    ierr = PetscInfoProcessClass("matmffd", 1, classids);CHKERRQ(ierr);
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("matmffd",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(MATMFFD_CLASSID);CHKERRQ(ierr);}
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(MatMFFDFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatMFFDSetType_MFFD(Mat mat,MatMFFDType ftype)
{
  PetscErrorCode ierr,(*r)(MatMFFD);
  MatMFFD        ctx;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidCharPointer(ftype,2);
  ierr = MatShellGetContext(mat,&ctx);CHKERRQ(ierr);

  /* already set, so just return */
  ierr = PetscObjectTypeCompare((PetscObject)ctx,ftype,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  /* destroy the old one if it exists */
  if (ctx->ops->destroy) {
    ierr = (*ctx->ops->destroy)(ctx);CHKERRQ(ierr);
  }

  ierr =  PetscFunctionListFind(MatMFFDList,ftype,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown MatMFFD type %s given",ftype);
  ierr = (*r)(ctx);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)ctx,ftype);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidCharPointer(ftype,2);
  ierr = PetscTryMethod(mat,"MatMFFDSetType_C",(Mat,MatMFFDType),(mat,ftype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_MFFD(Mat,Vec);

typedef PetscErrorCode (*FCN1)(void*,Vec); /* force argument to next function to not be extern C*/
static PetscErrorCode  MatMFFDSetFunctioniBase_MFFD(Mat mat,FCN1 func)
{
  MatMFFD        ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,&ctx);CHKERRQ(ierr);
  ctx->funcisetbase = func;
  PetscFunctionReturn(0);
}

typedef PetscErrorCode (*FCN2)(void*,PetscInt,Vec,PetscScalar*); /* force argument to next function to not be extern C*/
static PetscErrorCode  MatMFFDSetFunctioni_MFFD(Mat mat,FCN2 funci)
{
  MatMFFD        ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,&ctx);CHKERRQ(ierr);
  ctx->funci = funci;
  ierr = MatShellSetOperation(mat,MATOP_GET_DIAGONAL,(void (*)(void))MatGetDiagonal_MFFD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMFFDGetH_MFFD(Mat mat,PetscScalar *h)
{
  MatMFFD        ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,&ctx);CHKERRQ(ierr);
  *h = ctx->currenth;
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatMFFDResetHHistory_MFFD(Mat J)
{
  MatMFFD        ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J,&ctx);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&MatMFFDList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------*/
static PetscErrorCode MatDestroy_MFFD(Mat mat)
{
  PetscErrorCode ierr;
  MatMFFD        ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->w);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->current_u);CHKERRQ(ierr);
  if (ctx->current_f_allocated) {
    ierr = VecDestroy(&ctx->current_f);CHKERRQ(ierr);
  }
  if (ctx->ops->destroy) {ierr = (*ctx->ops->destroy)(ctx);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(&ctx);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetBase_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetFunctioniBase_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetFunctioni_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetFunction_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetFunctionError_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetCheckh_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetPeriod_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMFFDResetHHistory_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetHHistory_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMFFDGetH_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   MatMFFDView_MFFD - Views matrix-free parameters.

*/
static PetscErrorCode MatView_MFFD(Mat J,PetscViewer viewer)
{
  PetscErrorCode ierr;
  MatMFFD        ctx;
  PetscBool      iascii, viewbase, viewfunction;
  const char     *prefix;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J,&ctx);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Matrix-free approximation:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"err=%g (relative error in function evaluation)\n",(double)ctx->error_rel);CHKERRQ(ierr);
    if (!((PetscObject)ctx)->type_name) {
      ierr = PetscViewerASCIIPrintf(viewer,"The compute h routine has not yet been set\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"Using %s compute h routine\n",((PetscObject)ctx)->type_name);CHKERRQ(ierr);
    }
#if defined(PETSC_USE_COMPLEX)
    if (ctx->usecomplex) {
      ierr = PetscViewerASCIIPrintf(viewer,"Using Lyness complex number trick to compute the matrix-vector product\n");CHKERRQ(ierr);
    }
#endif
    if (ctx->ops->view) {
      ierr = (*ctx->ops->view)(ctx,viewer);CHKERRQ(ierr);
    }
    ierr = PetscObjectGetOptionsPrefix((PetscObject)J, &prefix);CHKERRQ(ierr);

    ierr = PetscOptionsHasName(((PetscObject)J)->options,prefix, "-mat_mffd_view_base", &viewbase);CHKERRQ(ierr);
    if (viewbase) {
      ierr = PetscViewerASCIIPrintf(viewer, "Base:\n");CHKERRQ(ierr);
      ierr = VecView(ctx->current_u, viewer);CHKERRQ(ierr);
    }
    ierr = PetscOptionsHasName(((PetscObject)J)->options,prefix, "-mat_mffd_view_function", &viewfunction);CHKERRQ(ierr);
    if (viewfunction) {
      ierr = PetscViewerASCIIPrintf(viewer, "Function:\n");CHKERRQ(ierr);
      ierr = VecView(ctx->current_f, viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  MatMFFD        j;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J,&j);CHKERRQ(ierr);
  ierr = MatMFFDResetHHistory(J);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscBool      zeroa;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,&ctx);CHKERRQ(ierr);
  if (!ctx->current_u) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"MatMFFDSetBase() has not been called, this is often caused by forgetting to call \n\t\tMatAssemblyBegin/End on the first Mat in the SNES compute function");
  /* We log matrix-free matrix-vector products separately, so that we can
     separate the performance monitoring from the cases that use conventional
     storage.  We may eventually modify event logging to associate events
     with particular objects, hence alleviating the more general problem. */
  ierr = PetscLogEventBegin(MATMFFD_Mult,a,y,0,0);CHKERRQ(ierr);

  w = ctx->w;
  U = ctx->current_u;
  F = ctx->current_f;
  /*
      Compute differencing parameter
  */
  if (!((PetscObject)ctx)->type_name) {
    ierr = MatMFFDSetType(mat,MATMFFD_WP);CHKERRQ(ierr);
    ierr = MatSetFromOptions(mat);CHKERRQ(ierr);
  }
  ierr = (*ctx->ops->compute)(ctx,U,a,&h,&zeroa);CHKERRQ(ierr);
  if (zeroa) {
    ierr = VecSet(y,0.0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (mat->erroriffailure && PetscIsInfOrNanScalar(h)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Computed Nan differencing parameter h");
  if (ctx->checkh) {
    ierr = (*ctx->checkh)(ctx->checkhctx,U,a,&h);CHKERRQ(ierr);
  }

  /* keep a record of the current differencing parameter h */
  ctx->currenth = h;
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscInfo2(mat,"Current differencing parameter: %g + %g i\n",(double)PetscRealPart(h),(double)PetscImaginaryPart(h));CHKERRQ(ierr);
#else
  ierr = PetscInfo1(mat,"Current differencing parameter: %15.12e\n",h);CHKERRQ(ierr);
#endif
  if (ctx->historyh && ctx->ncurrenth < ctx->maxcurrenth) {
    ctx->historyh[ctx->ncurrenth] = h;
  }
  ctx->ncurrenth++;

#if defined(PETSC_USE_COMPLEX)
  if (ctx->usecomplex) h = PETSC_i*h;
#endif

  /* w = u + ha */
  ierr = VecWAXPY(w,h,a,U);CHKERRQ(ierr);

  /* compute func(U) as base for differencing; only needed first time in and not when provided by user */
  if (ctx->ncurrenth == 1 && ctx->current_f_allocated) {
    ierr = (*ctx->func)(ctx->funcctx,U,F);CHKERRQ(ierr);
  }
  ierr = (*ctx->func)(ctx->funcctx,w,y);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
  if (ctx->usecomplex) {
    ierr = VecImaginaryPart(y);CHKERRQ(ierr);
    h    = PetscImaginaryPart(h);
  } else {
    ierr = VecAXPY(y,-1.0,F);CHKERRQ(ierr);
  }
#else
  ierr = VecAXPY(y,-1.0,F);CHKERRQ(ierr);
#endif
  ierr = VecScale(y,1.0/h);CHKERRQ(ierr);
  if (mat->nullsp) {ierr = MatNullSpaceRemove(mat->nullsp,y);CHKERRQ(ierr);}

  ierr = PetscLogEventEnd(MATMFFD_Mult,a,y,0,0);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i,rstart,rend;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,&ctx);CHKERRQ(ierr);
  if (!ctx->func) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Requires calling MatMFFDSetFunction() first");
  if (!ctx->funci) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Requires calling MatMFFDSetFunctioni() first");
  w    = ctx->w;
  U    = ctx->current_u;
  ierr = (*ctx->func)(ctx->funcctx,U,a);CHKERRQ(ierr);
  if (ctx->funcisetbase) {
    ierr = (*ctx->funcisetbase)(ctx->funcctx,U);CHKERRQ(ierr);
  }
  ierr = VecCopy(U,w);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(a,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetArray(a,&aa);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = VecGetArray(w,&ww);CHKERRQ(ierr);
    h    = ww[i-rstart];
    if (h == 0.0) h = 1.0;
    if (PetscAbsScalar(h) < umin && PetscRealPart(h) >= 0.0)     h = umin;
    else if (PetscRealPart(h) < 0.0 && PetscAbsScalar(h) < umin) h = -umin;
    h *= epsilon;

    ww[i-rstart] += h;
    ierr          = VecRestoreArray(w,&ww);CHKERRQ(ierr);
    ierr          = (*ctx->funci)(ctx->funcctx,i,w,&v);CHKERRQ(ierr);
    aa[i-rstart]  = (v - aa[i-rstart])/h;

    ierr          = VecGetArray(w,&ww);CHKERRQ(ierr);
    ww[i-rstart] -= h;
    ierr          = VecRestoreArray(w,&ww);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(a,&aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatMFFDSetBase_MFFD(Mat J,Vec U,Vec F)
{
  PetscErrorCode ierr;
  MatMFFD        ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J,&ctx);CHKERRQ(ierr);
  ierr = MatMFFDResetHHistory(J);CHKERRQ(ierr);
  if (!ctx->current_u) {
    ierr = VecDuplicate(U,&ctx->current_u);CHKERRQ(ierr);
    ierr = VecLockReadPush(ctx->current_u);CHKERRQ(ierr);
  }
  ierr = VecLockReadPop(ctx->current_u);CHKERRQ(ierr);
  ierr = VecCopy(U,ctx->current_u);CHKERRQ(ierr);
  ierr = VecLockReadPush(ctx->current_u);CHKERRQ(ierr);
  if (F) {
    if (ctx->current_f_allocated) {ierr = VecDestroy(&ctx->current_f);CHKERRQ(ierr);}
    ctx->current_f           = F;
    ctx->current_f_allocated = PETSC_FALSE;
  } else if (!ctx->current_f_allocated) {
    ierr = MatCreateVecs(J,NULL,&ctx->current_f);CHKERRQ(ierr);
    ctx->current_f_allocated = PETSC_TRUE;
  }
  if (!ctx->w) {
    ierr = VecDuplicate(ctx->current_u,&ctx->w);CHKERRQ(ierr);
  }
  J->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

typedef PetscErrorCode (*FCN3)(void*,Vec,Vec,PetscScalar*); /* force argument to next function to not be extern C*/

static PetscErrorCode  MatMFFDSetCheckh_MFFD(Mat J,FCN3 fun,void *ectx)
{
  MatMFFD        ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J,&ctx);CHKERRQ(ierr);
  ctx->checkh    = fun;
  ctx->checkhctx = ectx;
  PetscFunctionReturn(0);
}

/*@C
   MatMFFDSetOptionsPrefix - Sets the prefix used for searching for all
   MatMFFD options in the database.

   Collective on Mat

   Input Parameter:
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
  MatMFFD        mfctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = MatShellGetContext(mat,&mfctx);CHKERRQ(ierr);
  PetscValidHeaderSpecific(mfctx,MATMFFD_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)mfctx,prefix);CHKERRQ(ierr);
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
  ierr = MatShellGetContext(mat,&mfctx);CHKERRQ(ierr);
  PetscValidHeaderSpecific(mfctx,MATMFFD_CLASSID,2);
  ierr = PetscObjectOptionsBegin((PetscObject)mfctx);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-mat_mffd_type","Matrix free type","MatMFFDSetType",MatMFFDList,((PetscObject)mfctx)->type_name,ftype,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatMFFDSetType(mat,ftype);CHKERRQ(ierr);
  }

  ierr = PetscOptionsReal("-mat_mffd_err","set sqrt relative error in function","MatMFFDSetFunctionError",mfctx->error_rel,&mfctx->error_rel,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mffd_period","how often h is recomputed","MatMFFDSetPeriod",mfctx->recomputeperiod,&mfctx->recomputeperiod,NULL);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsBool("-mat_mffd_check_positivity","Insure that U + h*a is nonnegative","MatMFFDSetCheckh",flg,&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = MatMFFDSetCheckh(mat,MatMFFDCheckPositivity,NULL);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscOptionsBool("-mat_mffd_complex","Use Lyness complex number trick to compute the matrix-vector product","None",mfctx->usecomplex,&mfctx->usecomplex,NULL);CHKERRQ(ierr);
#endif
  if (mfctx->ops->setfromoptions) {
    ierr = (*mfctx->ops->setfromoptions)(PetscOptionsObject,mfctx);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatMFFDSetPeriod_MFFD(Mat mat,PetscInt period)
{
  MatMFFD        ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,&ctx);CHKERRQ(ierr);
  ctx->recomputeperiod = period;
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatMFFDSetFunction_MFFD(Mat mat,PetscErrorCode (*func)(void*,Vec,Vec),void *funcctx)
{
  MatMFFD        ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,&ctx);CHKERRQ(ierr);
  ctx->func    = func;
  ctx->funcctx = funcctx;
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatMFFDSetFunctionError_MFFD(Mat mat,PetscReal error)
{
  MatMFFD        ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,&ctx);CHKERRQ(ierr);
  if (error != PETSC_DEFAULT) ctx->error_rel = error;
  PetscFunctionReturn(0);
}

PetscErrorCode  MatMFFDSetHHistory_MFFD(Mat J,PetscScalar history[],PetscInt nhistory)
{
  MatMFFD        ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J,&ctx);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMFFDInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(mfctx,MATMFFD_CLASSID,"MatMFFD","Matrix-free Finite Differencing","Mat",PetscObjectComm((PetscObject)A),NULL,NULL);CHKERRQ(ierr);

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

  ierr = MatSetType(A,MATSHELL);CHKERRQ(ierr);
  ierr = MatShellSetContext(A,mfctx);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A,MATOP_MULT,(void (*)(void))MatMult_MFFD);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A,MATOP_DESTROY,(void (*)(void))MatDestroy_MFFD);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A,MATOP_VIEW,(void (*)(void))MatView_MFFD);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A,MATOP_ASSEMBLY_END,(void (*)(void))MatAssemblyEnd_MFFD);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A,MATOP_SET_FROM_OPTIONS,(void (*)(void))MatSetFromOptions_MFFD);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetBase_C",MatMFFDSetBase_MFFD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetFunctioniBase_C",MatMFFDSetFunctioniBase_MFFD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetFunctioni_C",MatMFFDSetFunctioni_MFFD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetFunction_C",MatMFFDSetFunction_MFFD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetCheckh_C",MatMFFDSetCheckh_MFFD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetPeriod_C",MatMFFDSetPeriod_MFFD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetFunctionError_C",MatMFFDSetFunctionError_MFFD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMFFDResetHHistory_C",MatMFFDResetHHistory_MFFD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetHHistory_C",MatMFFDSetHHistory_MFFD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMFFDSetType_C",MatMFFDSetType_MFFD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMFFDGetH_C",MatMFFDGetH_MFFD);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATMFFD);CHKERRQ(ierr);
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
.  -mat_mffd_unim <umin> - Sets umin (for default PETSc routine that computes h only)
-  -mat_mffd_check_positivity

.seealso: MatDestroy(), MatMFFDSetFunctionError(), MatMFFDDSSetUmin(), MatMFFDSetFunction()
          MatMFFDSetHHistory(), MatMFFDResetHHistory(), MatCreateSNESMF(),
          MatMFFDGetH(), MatMFFDRegister(), MatMFFDComputeJacobian()

@*/
PetscErrorCode  MatCreateMFFD(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,Mat *J)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*J,MATMFFD);CHKERRQ(ierr);
  ierr = MatSetUp(*J);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(h,2);
  ierr = PetscUseMethod(mat,"MatMFFDGetH_C",(Mat,PetscScalar*),(mat,h));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscTryMethod(mat,"MatMFFDSetFunction_C",(Mat,PetscErrorCode (*)(void*,Vec,Vec),void*),(mat,func,funcctx));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscTryMethod(mat,"MatMFFDSetFunctioni_C",(Mat,PetscErrorCode (*)(void*,PetscInt,Vec,PetscScalar*)),(mat,funci));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscTryMethod(mat,"MatMFFDSetFunctioniBase_C",(Mat,PetscErrorCode (*)(void*,Vec)),(mat,func));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatMFFDSetPeriod - Sets how often h is recomputed, by default it is everytime

   Logically Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via MatCreateSNESMF()
-  period - 1 for everytime, 2 for every second etc

   Options Database Keys:
.  -mat_mffd_period <period>

   Level: advanced

.seealso: MatCreateSNESMF(),MatMFFDGetH(),
          MatMFFDSetHHistory(), MatMFFDResetHHistory()
@*/
PetscErrorCode  MatMFFDSetPeriod(Mat mat,PetscInt period)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(mat,period,2);
  ierr = PetscTryMethod(mat,"MatMFFDSetPeriod_C",(Mat,PetscInt),(mat,period));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveReal(mat,error,2);
  ierr = PetscTryMethod(mat,"MatMFFDSetFunctionError_C",(Mat,PetscReal),(mat,error));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatMFFDSetHHistory - Sets an array to collect a history of the
   differencing values (h) computed for the matrix-free product.

   Logically Collective on Mat

   Input Parameters:
+  J - the matrix-free matrix context
.  histroy - space to hold the history
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(J,MAT_CLASSID,1);
  if (history) PetscValidPointer(history,2);
  PetscValidLogicalCollectiveInt(J,nhistory,3);
  ierr = PetscUseMethod(J,"MatMFFDSetHHistory_C",(Mat,PetscScalar[],PetscInt),(J,history,nhistory));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(J,MAT_CLASSID,1);
  ierr = PetscTryMethod(J,"MatMFFDResetHHistory_C",(Mat),(J));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(J,MAT_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,2);
  if (F) PetscValidHeaderSpecific(F,VEC_CLASSID,3);
  ierr = PetscTryMethod(J,"MatMFFDSetBase_C",(Mat,Vec,Vec),(J,U,F));CHKERRQ(ierr);
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
.   -mat_mffd_check_positivity

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(J,MAT_CLASSID,1);
  ierr = PetscTryMethod(J,"MatMFFDSetCheckh_C",(Mat,PetscErrorCode (*)(void*,Vec,Vec,PetscScalar*),void*),(J,fun,ctx));CHKERRQ(ierr);
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
.   -mat_mffd_check_positivity

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
  PetscErrorCode ierr;
  PetscInt       i,n;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(U,VEC_CLASSID,2);
  PetscValidHeaderSpecific(a,VEC_CLASSID,3);
  PetscValidPointer(h,4);
  ierr   = PetscObjectGetComm((PetscObject)U,&comm);CHKERRQ(ierr);
  ierr   = VecGetArray(U,&u_vec);CHKERRQ(ierr);
  ierr   = VecGetArray(a,&a_vec);CHKERRQ(ierr);
  ierr   = VecGetLocalSize(U,&n);CHKERRQ(ierr);
  minval = PetscAbsScalar(*h)*PetscRealConstant(1.01);
  for (i=0; i<n; i++) {
    if (PetscRealPart(u_vec[i] + *h*a_vec[i]) <= 0.0) {
      val = PetscAbsScalar(u_vec[i]/a_vec[i]);
      if (val < minval) minval = val;
    }
  }
  ierr = VecRestoreArray(U,&u_vec);CHKERRQ(ierr);
  ierr = VecRestoreArray(a,&a_vec);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&minval,&val,1,MPIU_REAL,MPIU_MIN,comm);CHKERRMPI(ierr);
  if (val <= PetscAbsScalar(*h)) {
    ierr = PetscInfo2(U,"Scaling back h from %g to %g\n",(double)PetscRealPart(*h),(double)(.99*val));CHKERRQ(ierr);
    if (PetscRealPart(*h) > 0.0) *h =  0.99*val;
    else                         *h = -0.99*val;
  }
  PetscFunctionReturn(0);
}
