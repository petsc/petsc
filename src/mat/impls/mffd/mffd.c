#define PETSCMAT_DLL

#include "private/matimpl.h"
#include "../src/mat/impls/mffd/mffdimpl.h"   /*I  "petscmat.h"   I*/

PetscFList MatMFFDList        = 0;
PetscTruth MatMFFDRegisterAllCalled = PETSC_FALSE;

PetscCookie PETSCMAT_DLLEXPORT MATMFFD_COOKIE;
PetscLogEvent  MATMFFD_Mult;

static PetscTruth MatMFFDPackageInitialized = PETSC_FALSE;
#undef __FUNCT__  
#define __FUNCT__ "MatMFFDFinalizePackage"
/*@C
  MatMFFDFinalizePackage - This function destroys everything in the MatMFFD package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode PETSC_DLLEXPORT MatMFFDFinalizePackage(void) 
{
  PetscFunctionBegin;
  MatMFFDPackageInitialized = PETSC_FALSE;
  MatMFFDRegisterAllCalled  = PETSC_FALSE;
  MatMFFDList               = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMFFDInitializePackage"
/*@C
  MatMFFDInitializePackage - This function initializes everything in the MatMFFD package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to MatCreate_MFFD()
  when using static libraries.

  Input Parameter:
. path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: Vec, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT MatMFFDInitializePackage(const char path[]) 
{
  char              logList[256];
  char              *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (MatMFFDPackageInitialized) PetscFunctionReturn(0);
  MatMFFDPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscCookieRegister("MatMFFD",&MATMFFD_COOKIE);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = MatMFFDRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("MatMult MF",          MATMFFD_COOKIE,&MATMFFD_Mult);CHKERRQ(ierr);

  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "matmffd", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(MATMFFD_COOKIE);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "matmffd", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(MATMFFD_COOKIE);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(MatMFFDFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMFFDSetType"
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

.seealso: MatCreateSNESMF(), MatMFFDRegisterDynamic(), MatMFFDSetFunction()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetType(Mat mat,const MatMFFDType ftype)
{
  PetscErrorCode ierr,(*r)(MatMFFD);
  MatMFFD        ctx = (MatMFFD)mat->data;
  PetscTruth     match;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidCharPointer(ftype,2);

  /* already set, so just return */
  ierr = PetscTypeCompare((PetscObject)ctx,ftype,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  /* destroy the old one if it exists */
  if (ctx->ops->destroy) {
    ierr = (*ctx->ops->destroy)(ctx);CHKERRQ(ierr);
  }

  ierr =  PetscFListFind(MatMFFDList,((PetscObject)ctx)->comm,ftype,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown MatMFFD type %s given",ftype);
  ierr = (*r)(ctx);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)ctx,ftype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef PetscErrorCode (*FCN1)(void*,Vec); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMFFDSetFunctioniBase_MFFD"
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetFunctioniBase_MFFD(Mat mat,FCN1 func)
{
  MatMFFD ctx = (MatMFFD)mat->data;

  PetscFunctionBegin;
  ctx->funcisetbase = func;
  PetscFunctionReturn(0);
}
EXTERN_C_END

typedef PetscErrorCode (*FCN2)(void*,PetscInt,Vec,PetscScalar*); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMFFDSetFunctioni_MFFD"
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetFunctioni_MFFD(Mat mat,FCN2 funci)
{
  MatMFFD ctx = (MatMFFD)mat->data;

  PetscFunctionBegin;
  ctx->funci = funci;
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "MatMFFDRegister"
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(MatMFFD))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&MatMFFDList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMFFDRegisterDestroy"
/*@C
   MatMFFDRegisterDestroy - Frees the list of MatMFFD methods that were
   registered by MatMFFDRegisterDynamic).

   Not Collective

   Level: developer

.keywords: MatMFFD, register, destroy

.seealso: MatMFFDRegisterDynamic), MatMFFDRegisterAll()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&MatMFFDList);CHKERRQ(ierr);
  MatMFFDRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MFFD"
PetscErrorCode MatDestroy_MFFD(Mat mat)
{
  PetscErrorCode ierr;
  MatMFFD        ctx = (MatMFFD)mat->data;

  PetscFunctionBegin;
  if (ctx->w) {
    ierr = VecDestroy(ctx->w);CHKERRQ(ierr);
  }
  if (ctx->current_f_allocated) {
    ierr = VecDestroy(ctx->current_f);
  }
  if (ctx->ops->destroy) {ierr = (*ctx->ops->destroy)(ctx);CHKERRQ(ierr);}
  if (ctx->sp) {ierr = MatNullSpaceDestroy(ctx->sp);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(ctx);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetBase_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetFunctioniBase_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetFunctioni_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMFFDSetCheckh_C","",PETSC_NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MFFD"
/*
   MatMFFDView_MFFD - Views matrix-free parameters.

*/
PetscErrorCode MatView_MFFD(Mat J,PetscViewer viewer)
{
  PetscErrorCode ierr;
  MatMFFD        ctx = (MatMFFD)J->data;
  PetscTruth     iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
     ierr = PetscViewerASCIIPrintf(viewer,"  matrix-free approximation:\n");CHKERRQ(ierr);
     ierr = PetscViewerASCIIPrintf(viewer,"    err=%G (relative error in function evaluation)\n",ctx->error_rel);CHKERRQ(ierr);
     if (!((PetscObject)ctx)->type_name) {
       ierr = PetscViewerASCIIPrintf(viewer,"    The compute h routine has not yet been set\n");CHKERRQ(ierr);
     } else {
       ierr = PetscViewerASCIIPrintf(viewer,"    Using %s compute h routine\n",((PetscObject)ctx)->type_name);CHKERRQ(ierr);
     }
     if (ctx->ops->view) {
       ierr = (*ctx->ops->view)(ctx,viewer);CHKERRQ(ierr);
     }
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for matrix-free matrix",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_MFFD"
/*
   MatAssemblyEnd_MFFD - Resets the ctx->ncurrenth to zero. This 
   allows the user to indicate the beginning of a new linear solve by calling
   MatAssemblyXXX() on the matrix free matrix. This then allows the 
   MatCreateMFFD_WP() to properly compute ||U|| only the first time
   in the linear solver rather than every time.
*/
PetscErrorCode MatAssemblyEnd_MFFD(Mat J,MatAssemblyType mt)
{
  PetscErrorCode ierr;
  MatMFFD        j = (MatMFFD)J->data;

  PetscFunctionBegin;
  ierr      = MatMFFDResetHHistory(J);CHKERRQ(ierr);
  j->vshift = 0.0;
  j->vscale = 1.0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_MFFD"
/*
  MatMult_MFFD - Default matrix-free form for Jacobian-vector product, y = F'(u)*a:

        y ~= (F(u + ha) - F(u))/h, 
  where F = nonlinear function, as set by SNESSetFunction()
        u = current iterate
        h = difference interval
*/
PetscErrorCode MatMult_MFFD(Mat mat,Vec a,Vec y)
{
  MatMFFD        ctx = (MatMFFD)mat->data;
  PetscScalar    h;
  Vec            w,U,F;
  PetscErrorCode ierr;
  PetscTruth     zeroa;

  PetscFunctionBegin;
  /* We log matrix-free matrix-vector products separately, so that we can
     separate the performance monitoring from the cases that use conventional
     storage.  We may eventually modify event logging to associate events
     with particular objects, hence alleviating the more general problem. */
  ierr = PetscLogEventBegin(MATMFFD_Mult,a,y,0,0);CHKERRQ(ierr);

  w    = ctx->w;
  U    = ctx->current_u;
  F    = ctx->current_f;
  /* 
      Compute differencing parameter 
  */
  if (!ctx->ops->compute) {
    ierr = MatMFFDSetType(mat,MATMFFD_WP);CHKERRQ(ierr);
    ierr = MatMFFDSetFromOptions(mat);CHKERRQ(ierr);
  }
  ierr = (*ctx->ops->compute)(ctx,U,a,&h,&zeroa);CHKERRQ(ierr);
  if (zeroa) {
    ierr = VecSet(y,0.0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (h != h) SETERRQ(PETSC_ERR_PLIB,"Computed Nan differencing parameter h");
  if (ctx->checkh) {
    ierr = (*ctx->checkh)(ctx->checkhctx,U,a,&h);CHKERRQ(ierr);
  }

  /* keep a record of the current differencing parameter h */  
  ctx->currenth = h;
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscInfo2(mat,"Current differencing parameter: %G + %G i\n",PetscRealPart(h),PetscImaginaryPart(h));CHKERRQ(ierr);
#else
  ierr = PetscInfo1(mat,"Current differencing parameter: %15.12e\n",h);CHKERRQ(ierr);
#endif
  if (ctx->historyh && ctx->ncurrenth < ctx->maxcurrenth) {
    ctx->historyh[ctx->ncurrenth] = h;
  }
  ctx->ncurrenth++;

  /* w = u + ha */
  ierr = VecWAXPY(w,h,a,U);CHKERRQ(ierr);

  /* compute func(U) as base for differencing; only needed first time in and not when provided by user */
  if (ctx->ncurrenth == 1 && ctx->current_f_allocated) {
    ierr = (*ctx->func)(ctx->funcctx,U,F);CHKERRQ(ierr);
  }
  ierr = (*ctx->func)(ctx->funcctx,w,y);CHKERRQ(ierr);

  ierr = VecAXPY(y,-1.0,F);CHKERRQ(ierr);
  ierr = VecScale(y,1.0/h);CHKERRQ(ierr);

  ierr = VecAXPBY(y,ctx->vshift,ctx->vscale,a);CHKERRQ(ierr);

  if (ctx->sp) {ierr = MatNullSpaceRemove(ctx->sp,y,PETSC_NULL);CHKERRQ(ierr);}

  ierr = PetscLogEventEnd(MATMFFD_Mult,a,y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_MFFD"
/*
  MatGetDiagonal_MFFD - Gets the diagonal for a matrix free matrix

        y ~= (F(u + ha) - F(u))/h, 
  where F = nonlinear function, as set by SNESSetFunction()
        u = current iterate
        h = difference interval
*/
PetscErrorCode MatGetDiagonal_MFFD(Mat mat,Vec a)
{
  MatMFFD        ctx = (MatMFFD)mat->data;
  PetscScalar    h,*aa,*ww,v;
  PetscReal      epsilon = PETSC_SQRT_MACHINE_EPSILON,umin = 100.0*PETSC_SQRT_MACHINE_EPSILON;
  Vec            w,U;
  PetscErrorCode ierr;
  PetscInt       i,rstart,rend;

  PetscFunctionBegin;
  if (!ctx->funci) {
    SETERRQ(PETSC_ERR_ORDER,"Requires calling MatMFFDSetFunctioni() first");
  }

  w    = ctx->w;
  U    = ctx->current_u;
  ierr = (*ctx->func)(ctx->funcctx,U,a);CHKERRQ(ierr);
  ierr = (*ctx->funcisetbase)(ctx->funcctx,U);CHKERRQ(ierr);
  ierr = VecCopy(U,w);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(a,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetArray(a,&aa);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = VecGetArray(w,&ww);CHKERRQ(ierr);
    h  = ww[i-rstart];
    if (h == 0.0) h = 1.0;
#if !defined(PETSC_USE_COMPLEX)
    if (h < umin && h >= 0.0)      h = umin;
    else if (h < 0.0 && h > -umin) h = -umin;
#else
    if (PetscAbsScalar(h) < umin && PetscRealPart(h) >= 0.0)     h = umin;
    else if (PetscRealPart(h) < 0.0 && PetscAbsScalar(h) < umin) h = -umin;
#endif
    h     *= epsilon;
    
    ww[i-rstart] += h;
    ierr = VecRestoreArray(w,&ww);CHKERRQ(ierr);
    ierr          = (*ctx->funci)(ctx->funcctx,i,w,&v);CHKERRQ(ierr);
    aa[i-rstart]  = (v - aa[i-rstart])/h;

    /* possibly shift and scale result */
    aa[i - rstart] = ctx->vshift + ctx->vscale*aa[i-rstart];

    ierr = VecGetArray(w,&ww);CHKERRQ(ierr);
    ww[i-rstart] -= h;
    ierr = VecRestoreArray(w,&ww);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(a,&aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatShift_MFFD"
PetscErrorCode MatShift_MFFD(Mat Y,PetscScalar a)
{
  MatMFFD shell = (MatMFFD)Y->data;  
  PetscFunctionBegin;
  shell->vshift += a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_MFFD"
PetscErrorCode MatScale_MFFD(Mat Y,PetscScalar a)
{
  MatMFFD shell = (MatMFFD)Y->data;  
  PetscFunctionBegin;
  shell->vscale *= a;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMFFDSetBase_MFFD"
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetBase_MFFD(Mat J,Vec U,Vec F)
{
  PetscErrorCode ierr;
  MatMFFD        ctx = (MatMFFD)J->data;

  PetscFunctionBegin;
  ierr = MatMFFDResetHHistory(J);CHKERRQ(ierr);
  ctx->current_u = U;
  if (F) {
    if (ctx->current_f_allocated) {ierr = VecDestroy(ctx->current_f);CHKERRQ(ierr);}
    ctx->current_f           = F;
    ctx->current_f_allocated = PETSC_FALSE;
  } else if (!ctx->current_f_allocated) {
    ierr = VecDuplicate(ctx->current_u, &ctx->current_f);CHKERRQ(ierr);
    ctx->current_f_allocated = PETSC_TRUE;
  }
  if (!ctx->w) {
    ierr = VecDuplicate(ctx->current_u, &ctx->w);CHKERRQ(ierr);
  }
  J->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END
typedef PetscErrorCode (*FCN3)(void*,Vec,Vec,PetscScalar*); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMFFDSetCheckh_MFFD"
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetCheckh_MFFD(Mat J,FCN3 fun,void*ectx)
{
  MatMFFD ctx = (MatMFFD)J->data;

  PetscFunctionBegin;
  ctx->checkh    = fun;
  ctx->checkhctx = ectx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatMFFDSetOptionsPrefix"
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

.keywords: SNES, matrix-free, parameters

.seealso: MatMFFDSetFromOptions(), MatCreateSNESMF()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetOptionsPrefix(Mat mat,const char prefix[])

{
  MatMFFD        mfctx = mat ? (MatMFFD)mat->data : PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(mfctx,MATMFFD_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)mfctx,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMFFDSetFromOptions"
/*@
   MatMFFDSetFromOptions - Sets the MatMFFD options from the command line
   parameter.

   Collective on Mat

   Input Parameters:
.  mat - the matrix obtained with MatCreateMFFD() or MatCreateSNESMF()

   Options Database Keys:
+  -mat_mffd_type - wp or ds (see MATMFFD_WP or MATMFFD_DS)
-  -mat_mffd_err - square root of estimated relative error in function evaluation
-  -mat_mffd_period - how often h is recomputed, defaults to 1, everytime

   Level: advanced

.keywords: SNES, matrix-free, parameters

.seealso: MatCreateSNESMF(),MatMFFDSetHHistory(), MatMFFDResetHHistory()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetFromOptions(Mat mat)
{
  MatMFFD        mfctx = mat ? (MatMFFD)mat->data : PETSC_NULL;
  PetscErrorCode ierr;
  PetscTruth     flg;
  char           ftype[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(mfctx,MATMFFD_COOKIE,1);
  ierr = PetscOptionsBegin(((PetscObject)mfctx)->comm,((PetscObject)mfctx)->prefix,"Set matrix free computation parameters","MatMFFD");CHKERRQ(ierr);
  ierr = PetscOptionsList("-mat_mffd_type","Matrix free type","MatMFFDSetType",MatMFFDList,((PetscObject)mfctx)->type_name,ftype,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatMFFDSetType(mat,ftype);CHKERRQ(ierr);
  }

  ierr = PetscOptionsReal("-mat_mffd_err","set sqrt relative error in function","MatMFFDSetFunctionError",mfctx->error_rel,&mfctx->error_rel,0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mffd_period","how often h is recomputed","MatMFFDSetPeriod",mfctx->recomputeperiod,&mfctx->recomputeperiod,0);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsTruth("-mat_mffd_check_positivity","Insure that U + h*a is nonnegative","MatMFFDSetCheckh",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = MatMFFDSetCheckh(mat,MatMFFDCheckPositivity,0);CHKERRQ(ierr);
  }
  if (mfctx->ops->setfromoptions) {
    ierr = (*mfctx->ops->setfromoptions)(mfctx);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  MATMFFD - MATMFFD = "mffd" - A matrix free matrix type.

  Level: advanced

.seealso: MatCreateMFFD(), MatCreateSNESMF(), MatMFFDSetFunction()
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_MFFD"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MFFD(Mat A)
{
  MatMFFD         mfctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = MatMFFDInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(mfctx,_p_MatMFFD,struct _MFOps,MATMFFD_COOKIE,0,"MatMFFD",((PetscObject)A)->comm,MatDestroy_MFFD,MatView_MFFD);CHKERRQ(ierr);
  mfctx->sp              = 0;
  mfctx->error_rel       = PETSC_SQRT_MACHINE_EPSILON;
  mfctx->recomputeperiod = 1;
  mfctx->count           = 0;
  mfctx->currenth        = 0.0;
  mfctx->historyh        = PETSC_NULL;
  mfctx->ncurrenth       = 0;
  mfctx->maxcurrenth     = 0;
  ((PetscObject)mfctx)->type_name       = 0;

  mfctx->vshift          = 0.0;
  mfctx->vscale          = 1.0;

  /* 
     Create the empty data structure to contain compute-h routines.
     These will be filled in below from the command line options or 
     a later call with MatMFFDSetType() or if that is not called 
     then it will default in the first use of MatMult_MFFD()
  */
  mfctx->ops->compute        = 0;
  mfctx->ops->destroy        = 0;
  mfctx->ops->view           = 0;
  mfctx->ops->setfromoptions = 0;
  mfctx->hctx                = 0;

  mfctx->func                = 0;
  mfctx->funcctx             = 0;
  mfctx->w                   = PETSC_NULL;

  A->data                = mfctx;

  A->ops->mult           = MatMult_MFFD;
  A->ops->destroy        = MatDestroy_MFFD;
  A->ops->view           = MatView_MFFD;
  A->ops->assemblyend    = MatAssemblyEnd_MFFD;
  A->ops->getdiagonal    = MatGetDiagonal_MFFD;
  A->ops->scale          = MatScale_MFFD;
  A->ops->shift          = MatShift_MFFD;
  A->ops->setfromoptions = MatMFFDSetFromOptions;
  A->assembled = PETSC_TRUE;

  ierr = PetscLayoutSetBlockSize(A->rmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(A->cmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatMFFDSetBase_C","MatMFFDSetBase_MFFD",MatMFFDSetBase_MFFD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatMFFDSetFunctioniBase_C","MatMFFDSetFunctioniBase_MFFD",MatMFFDSetFunctioniBase_MFFD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatMFFDSetFunctioni_C","MatMFFDSetFunctioni_MFFD",MatMFFDSetFunctioni_MFFD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatMFFDSetCheckh_C","MatMFFDSetCheckh_MFFD",MatMFFDSetCheckh_MFFD);CHKERRQ(ierr);
  mfctx->mat = A;
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATMFFD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatCreateMFFD"
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

   The user can set the error_rel via MatMFFDSetFunctionError() and 
   umin via MatMFFDDSSetUmin(); see the nonlinear solvers chapter
   of the users manual for details.

   The user should call MatDestroy() when finished with the matrix-free
   matrix context.

   Options Database Keys:
+  -mat_mffd_err <error_rel> - Sets error_rel
.  -mat_mffd_unim <umin> - Sets umin (for default PETSc routine that computes h only)
-  -mat_mffd_check_positivity

.keywords: default, matrix-free, create, matrix

.seealso: MatDestroy(), MatMFFDSetFunctionError(), MatMFFDDSSetUmin(), MatMFFDSetFunction()
          MatMFFDSetHHistory(), MatMFFDResetHHistory(), MatCreateSNESMF(), 
          MatMFFDGetH(), MatMFFDRegisterDynamic), MatMFFDComputeJacobian()
 
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMFFD(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,Mat *J)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*J,MATMFFD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMFFDGetH"
/*@
   MatMFFDGetH - Gets the last value that was used as the differencing 
   parameter.

   Not Collective

   Input Parameters:
.  mat - the matrix obtained with MatCreateSNESMF()

   Output Paramter:
.  h - the differencing step size

   Level: advanced

.keywords: SNES, matrix-free, parameters

.seealso: MatCreateSNESMF(),MatMFFDSetHHistory(), MatCreateMFFD(), MATMFFD, MatMFFDResetHHistory()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDGetH(Mat mat,PetscScalar *h)
{
  MatMFFD ctx = (MatMFFD)mat->data;

  PetscFunctionBegin;
  *h = ctx->currenth;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMFFDSetFunction"
/*@C
   MatMFFDSetFunction - Sets the function used in applying the matrix free.

   Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via MatCreateSNESMF()
.  func - the function to use
-  funcctx - optional function context passed to function

   Level: advanced

   Notes:
    If you use this you MUST call MatAssemblyBegin()/MatAssemblyEnd() on the matrix free
    matrix inside your compute Jacobian routine

    If this is not set then it will use the function set with SNESSetFunction() if MatCreateSNESMF() was used.

.keywords: SNES, matrix-free, function

.seealso: MatCreateSNESMF(),MatMFFDGetH(), MatCreateMFFD(), MATMFFD,
          MatMFFDSetHHistory(), MatMFFDResetHHistory(), SNESetFunction()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetFunction(Mat mat,PetscErrorCode (*func)(void*,Vec,Vec),void *funcctx)
{
  MatMFFD ctx = (MatMFFD)mat->data;

  PetscFunctionBegin;
  ctx->func    = func;
  ctx->funcctx = funcctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMFFDSetFunctioni"
/*@C
   MatMFFDSetFunctioni - Sets the function for a single component

   Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via MatCreateSNESMF()
-  funci - the function to use

   Level: advanced

   Notes:
    If you use this you MUST call MatAssemblyBegin()/MatAssemblyEnd() on the matrix free
    matrix inside your compute Jacobian routine


.keywords: SNES, matrix-free, function

.seealso: MatCreateSNESMF(),MatMFFDGetH(), MatMFFDSetHHistory(), MatMFFDResetHHistory(), SNESetFunction()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetFunctioni(Mat mat,PetscErrorCode (*funci)(void*,PetscInt,Vec,PetscScalar*))
{
  PetscErrorCode ierr,(*f)(Mat,PetscErrorCode (*)(void*,PetscInt,Vec,PetscScalar*));

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatMFFDSetFunctioni_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat,funci);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMFFDSetFunctioniBase"
/*@C
   MatMFFDSetFunctioniBase - Sets the base vector for a single component function evaluation

   Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via MatCreateSNESMF()
-  func - the function to use

   Level: advanced

   Notes:
    If you use this you MUST call MatAssemblyBegin()/MatAssemblyEnd() on the matrix free
    matrix inside your compute Jacobian routine


.keywords: SNES, matrix-free, function

.seealso: MatCreateSNESMF(),MatMFFDGetH(), MatCreateMFFD(), MATMFFD
          MatMFFDSetHHistory(), MatMFFDResetHHistory(), SNESetFunction()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetFunctioniBase(Mat mat,PetscErrorCode (*func)(void*,Vec))
{
  PetscErrorCode ierr,(*f)(Mat,PetscErrorCode (*)(void*,Vec));

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatMFFDSetFunctioniBase_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat,func);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMFFDSetPeriod"
/*@
   MatMFFDSetPeriod - Sets how often h is recomputed, by default it is everytime

   Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via MatCreateSNESMF()
-  period - 1 for everytime, 2 for every second etc

   Options Database Keys:
+  -mat_mffd_period <period>

   Level: advanced


.keywords: SNES, matrix-free, parameters

.seealso: MatCreateSNESMF(),MatMFFDGetH(),
          MatMFFDSetHHistory(), MatMFFDResetHHistory()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetPeriod(Mat mat,PetscInt period)
{
  MatMFFD ctx = (MatMFFD)mat->data;

  PetscFunctionBegin;
  ctx->recomputeperiod = period;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMFFDSetFunctionError"
/*@
   MatMFFDSetFunctionError - Sets the error_rel for the approximation of
   matrix-vector products using finite differences.

   Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via MatCreateMFFD() or MatCreateSNESMF()
-  error_rel - relative error (should be set to the square root of
               the relative error in the function evaluations)

   Options Database Keys:
+  -mat_mffd_err <error_rel> - Sets error_rel

   Level: advanced

   Notes:
   The default matrix-free matrix-vector product routine computes
.vb
     F'(u)*a = [F(u+h*a) - F(u)]/h where
     h = error_rel*u'a/||a||^2                        if  |u'a| > umin*||a||_{1}
       = error_rel*umin*sign(u'a)*||a||_{1}/||a||^2   else
.ve

.keywords: SNES, matrix-free, parameters

.seealso: MatCreateSNESMF(),MatMFFDGetH(), MatCreateMFFD(), MATMFFD
          MatMFFDSetHHistory(), MatMFFDResetHHistory()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetFunctionError(Mat mat,PetscReal error)
{
  MatMFFD ctx = (MatMFFD)mat->data;

  PetscFunctionBegin;
  if (error != PETSC_DEFAULT) ctx->error_rel = error;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMFFDAddNullSpace"
/*@
   MatMFFDAddNullSpace - Provides a null space that an operator is
   supposed to have.  Since roundoff will create a small component in
   the null space, if you know the null space you may have it
   automatically removed.

   Collective on Mat 

   Input Parameters:
+  J - the matrix-free matrix context
-  nullsp - object created with MatNullSpaceCreate()

   Level: advanced

.keywords: SNES, matrix-free, null space

.seealso: MatNullSpaceCreate(), MatMFFDGetH(), MatCreateSNESMF(), MatCreateMFFD(), MATMFFD
          MatMFFDSetHHistory(), MatMFFDResetHHistory()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDAddNullSpace(Mat J,MatNullSpace nullsp)
{
  PetscErrorCode ierr;
  MatMFFD      ctx = (MatMFFD)J->data;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)nullsp);CHKERRQ(ierr);
  if (ctx->sp) { ierr = MatNullSpaceDestroy(ctx->sp);CHKERRQ(ierr); }
  ctx->sp = nullsp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMFFDSetHHistory"
/*@
   MatMFFDSetHHistory - Sets an array to collect a history of the
   differencing values (h) computed for the matrix-free product.

   Collective on Mat 

   Input Parameters:
+  J - the matrix-free matrix context
.  histroy - space to hold the history
-  nhistory - number of entries in history, if more entries are generated than
              nhistory, then the later ones are discarded

   Level: advanced

   Notes:
   Use MatMFFDResetHHistory() to reset the history counter and collect
   a new batch of differencing parameters, h.

.keywords: SNES, matrix-free, h history, differencing history

.seealso: MatMFFDGetH(), MatCreateSNESMF(),
          MatMFFDResetHHistory(), MatMFFDSetFunctionError()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetHHistory(Mat J,PetscScalar history[],PetscInt nhistory)
{
  MatMFFD ctx = (MatMFFD)J->data;

  PetscFunctionBegin;
  ctx->historyh    = history;
  ctx->maxcurrenth = nhistory;
  ctx->currenth    = 0.;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMFFDResetHHistory"
/*@
   MatMFFDResetHHistory - Resets the counter to zero to begin 
   collecting a new set of differencing histories.

   Collective on Mat 

   Input Parameters:
.  J - the matrix-free matrix context

   Level: advanced

   Notes:
   Use MatMFFDSetHHistory() to create the original history counter.

.keywords: SNES, matrix-free, h history, differencing history

.seealso: MatMFFDGetH(), MatCreateSNESMF(),
          MatMFFDSetHHistory(), MatMFFDSetFunctionError()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDResetHHistory(Mat J)
{
  MatMFFD ctx = (MatMFFD)J->data;

  PetscFunctionBegin;
  ctx->ncurrenth    = 0;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMFFDSetBase"
/*@
    MatMFFDSetBase - Sets the vector U at which matrix vector products of the 
        Jacobian are computed

    Collective on Mat

    Input Parameters:
+   J - the MatMFFD matrix
.   U - the vector
-   F - (optional) vector that contains F(u) if it has been already computed

    Notes: This is rarely used directly

    If F is provided then it is not recomputed. Otherwise the function is evaluated at the base
    point during the first MatMult() after each call to MatMFFDSetBase().

    Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetBase(Mat J,Vec U,Vec F)
{
  PetscErrorCode ierr,(*f)(Mat,Vec,Vec);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(J,MAT_COOKIE,1);
  PetscValidHeaderSpecific(U,VEC_COOKIE,2);
  if (F) PetscValidHeaderSpecific(F,VEC_COOKIE,3);
  ierr = PetscObjectQueryFunction((PetscObject)J,"MatMFFDSetBase_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(J,U,F);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMFFDSetCheckh"
/*@C
    MatMFFDSetCheckh - Sets a function that checks the computed h and adjusts
        it to satisfy some criteria

    Collective on Mat

    Input Parameters:
+   J - the MatMFFD matrix
.   fun - the function that checks h
-   ctx - any context needed by the function

    Options Database Keys:
.   -mat_mffd_check_positivity

    Level: advanced

    Notes: For example, MatMFFDSetCheckPositivity() insures that all entries
       of U + h*a are non-negative

.seealso:  MatMFFDSetCheckPositivity()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDSetCheckh(Mat J,PetscErrorCode (*fun)(void*,Vec,Vec,PetscScalar*),void* ctx)
{
  PetscErrorCode ierr,(*f)(Mat,PetscErrorCode (*)(void*,Vec,Vec,PetscScalar*),void*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(J,MAT_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)J,"MatMFFDSetCheckh_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(J,fun,ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMFFDSetCheckPositivity"
/*@
    MatMFFDCheckPositivity - Checks that all entries in U + h*a are positive or
        zero, decreases h until this is satisfied.

    Collective on Vec

    Input Parameters:
+   U - base vector that is added to
.   a - vector that is added
.   h - scaling factor on a
-   dummy - context variable (unused)

    Options Database Keys:
.   -mat_mffd_check_positivity

    Level: advanced

    Notes: This is rarely used directly, rather it is passed as an argument to 
           MatMFFDSetCheckh()

.seealso:  MatMFFDSetCheckh()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMFFDCheckPositivity(void* dummy,Vec U,Vec a,PetscScalar *h)
{
  PetscReal      val, minval;
  PetscScalar    *u_vec, *a_vec;
  PetscErrorCode ierr;
  PetscInt       i,n;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)U,&comm);CHKERRQ(ierr);
  ierr = VecGetArray(U,&u_vec);CHKERRQ(ierr);  
  ierr = VecGetArray(a,&a_vec);CHKERRQ(ierr);  
  ierr = VecGetLocalSize(U,&n);CHKERRQ(ierr);
  minval = PetscAbsScalar(*h*1.01);
  for(i=0;i<n;i++) {
    if (PetscRealPart(u_vec[i] + *h*a_vec[i]) <= 0.0) {
      val = PetscAbsScalar(u_vec[i]/a_vec[i]);
      if (val < minval) minval = val;
    }
  }
  ierr = VecRestoreArray(U,&u_vec);CHKERRQ(ierr);  
  ierr = VecRestoreArray(a,&a_vec);CHKERRQ(ierr);
  ierr = PetscGlobalMin(&minval,&val,comm);CHKERRQ(ierr);
  if (val <= PetscAbsScalar(*h)) {
    ierr = PetscInfo2(U,"Scaling back h from %G to %G\n",PetscRealPart(*h),.99*val);CHKERRQ(ierr);
    if (PetscRealPart(*h) > 0.0) *h =  0.99*val;
    else                         *h = -0.99*val;
  }
  PetscFunctionReturn(0);
}







