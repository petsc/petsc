/*$Id: snesmfj.c,v 1.131 2001/09/05 18:45:40 bsmith Exp $*/

#include "src/mat/matimpl.h"
#include "src/snes/mf/snesmfj.h"   /*I  "petscsnes.h"   I*/

PetscFList      MatSNESMPetscFList              = 0;
PetscTruth MatSNESMFRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFSetType"
/*@C
    MatSNESMFSetType - Sets the method that is used to compute the 
    differencing parameter for finite differene matrix-free formulations. 

    Input Parameters:
+   mat - the "matrix-free" matrix created via MatCreateSNESMF(), or MatCreateMF()
          or MatSetType(mat,MATMFFD);
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
int MatSNESMFSetType(Mat mat,const MatSNESMFType ftype)
{
  int          ierr,(*r)(MatSNESMFCtx);
  MatSNESMFCtx ctx = (MatSNESMFCtx)mat->data;
  PetscTruth   match;
  
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

  /* Get the function pointers for the requrested method */
  if (!MatSNESMFRegisterAllCalled) {ierr = MatSNESMFRegisterAll(PETSC_NULL);CHKERRQ(ierr);}

  ierr =  PetscFListFind(ctx->comm,MatSNESMPetscFList,ftype,(void (**)(void)) &r);CHKERRQ(ierr);

  if (!r) SETERRQ(1,"Unknown MatSNESMF type given");

  ierr = (*r)(ctx);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)ctx,ftype);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

typedef int (*FCN1)(Vec,void*); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFSetFunctioniBase_FD"
int MatSNESMFSetFunctioniBase_FD(Mat mat,FCN1 func)
{
  MatSNESMFCtx ctx = (MatSNESMFCtx)mat->data;

  PetscFunctionBegin;
  ctx->funcisetbase = func;
  PetscFunctionReturn(0);
}
EXTERN_C_END

typedef int (*FCN2)(int,Vec,PetscScalar*,void*); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFSetFunctioni_FD"
int MatSNESMFSetFunctioni_FD(Mat mat,FCN2 funci)
{
  MatSNESMFCtx ctx = (MatSNESMFCtx)mat->data;

  PetscFunctionBegin;
  ctx->funci = funci;
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFRegister"
int MatSNESMFRegister(const char sname[],const char path[],const char name[],int (*function)(MatSNESMFCtx))
{
  int ierr;
  char fullname[256];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&MatSNESMPetscFList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFRegisterDestroy"
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
  if (MatSNESMPetscFList) {
    ierr = PetscFListDestroy(&MatSNESMPetscFList);CHKERRQ(ierr);
    MatSNESMPetscFList = 0;
  }
  MatSNESMFRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MFFD"
int MatDestroy_MFFD(Mat mat)
{
  int          ierr;
  MatSNESMFCtx ctx = (MatSNESMFCtx)mat->data;

  PetscFunctionBegin;
  if (ctx->w != PETSC_NULL) {
    ierr = VecDestroy(ctx->w);CHKERRQ(ierr);
  }
  if (ctx->ops->destroy) {ierr = (*ctx->ops->destroy)(ctx);CHKERRQ(ierr);}
  if (ctx->sp) {ierr = MatNullSpaceDestroy(ctx->sp);CHKERRQ(ierr);}
  PetscHeaderDestroy(ctx);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MFFD"
/*
   MatSNESMFView_MFFD - Views matrix-free parameters.

*/
int MatView_MFFD(Mat J,PetscViewer viewer)
{
  int          ierr;
  MatSNESMFCtx ctx = (MatSNESMFCtx)J->data;
  PetscTruth   isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
     ierr = PetscViewerASCIIPrintf(viewer,"  SNES matrix-free approximation:\n");CHKERRQ(ierr);
     ierr = PetscViewerASCIIPrintf(viewer,"    err=%g (relative error in function evaluation)\n",ctx->error_rel);CHKERRQ(ierr);
     if (!ctx->type_name) {
       ierr = PetscViewerASCIIPrintf(viewer,"    The compute h routine has not yet been set\n");CHKERRQ(ierr);
     } else {
       ierr = PetscViewerASCIIPrintf(viewer,"    Using %s compute h routine\n",ctx->type_name);CHKERRQ(ierr);
     }
     if (ctx->ops->view) {
       ierr = (*ctx->ops->view)(ctx,viewer);CHKERRQ(ierr);
     }
  } else {
    SETERRQ1(1,"Viewer type %s not supported for SNES matrix free matrix",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_MFFD"
/*
   MatSNESMFAssemblyEnd_Private - Resets the ctx->ncurrenth to zero. This 
   allows the user to indicate the beginning of a new linear solve by calling
   MatAssemblyXXX() on the matrix free matrix. This then allows the 
   MatSNESMFCreate_WP() to properly compute ||U|| only the first time
   in the linear solver rather than every time.
*/
int MatAssemblyEnd_MFFD(Mat J,MatAssemblyType mt)
{
  int             ierr;
  MatSNESMFCtx    j = (MatSNESMFCtx)J->data;

  PetscFunctionBegin;
  ierr = MatSNESMFResetHHistory(J);CHKERRQ(ierr);
  if (j->usesnes) {
    ierr = SNESGetSolution(j->snes,&j->current_u);CHKERRQ(ierr);
    ierr = SNESGetFunction(j->snes,&j->current_f,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    if (j->w == PETSC_NULL) {
      ierr = VecDuplicate(j->current_u, &j->w);CHKERRQ(ierr);
    }
  }
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
int MatMult_MFFD(Mat mat,Vec a,Vec y)
{
  MatSNESMFCtx    ctx = (MatSNESMFCtx)mat->data;
  SNES            snes;
  PetscScalar     h,mone = -1.0;
  Vec             w,U,F;
  int             ierr,(*eval_fct)(SNES,Vec,Vec)=0;

  PetscFunctionBegin;
  /* We log matrix-free matrix-vector products separately, so that we can
     separate the performance monitoring from the cases that use conventional
     storage.  We may eventually modify event logging to associate events
     with particular objects, hence alleviating the more general problem. */
  ierr = PetscLogEventBegin(MAT_MultMatrixFree,a,y,0,0);CHKERRQ(ierr);

  snes = ctx->snes;
  w    = ctx->w;
  U    = ctx->current_u;

  /* 
      Compute differencing parameter 
  */
  if (!ctx->ops->compute) {
    ierr = MatSNESMFSetType(mat,MATSNESMF_WP);CHKERRQ(ierr);
    ierr = MatSNESMFSetFromOptions(mat);CHKERRQ(ierr);
  }
  ierr = (*ctx->ops->compute)(ctx,U,a,&h);CHKERRQ(ierr);

  if (ctx->checkh) {
    ierr = (*ctx->checkh)(U,a,&h,ctx->checkhctx);CHKERRQ(ierr);
  }

  /* keep a record of the current differencing parameter h */  
  ctx->currenth = h;
#if defined(PETSC_USE_COMPLEX)
  PetscLogInfo(mat,"MatMult_MFFD:Current differencing parameter: %g + %g i\n",PetscRealPart(h),PetscImaginaryPart(h));
#else
  PetscLogInfo(mat,"MatMult_MFFD:Current differencing parameter: %15.12e\n",h);
#endif
  if (ctx->historyh && ctx->ncurrenth < ctx->maxcurrenth) {
    ctx->historyh[ctx->ncurrenth] = h;
  }
  ctx->ncurrenth++;

  /* w = u + ha */
  ierr = VecWAXPY(&h,a,U,w);CHKERRQ(ierr);

  if (ctx->usesnes) {
    eval_fct = SNESComputeFunction;
    F    = ctx->current_f;
    if (!F) SETERRQ(1,"You must call MatAssembly() even on matrix-free matrices");
    ierr = (*eval_fct)(snes,w,y);CHKERRQ(ierr);
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


  if (ctx->vshift != 0.0 && ctx->vscale != 1.0) {
    ierr = VecAXPBY(&ctx->vshift,&ctx->vscale,a,y);CHKERRQ(ierr);
  } else if (ctx->vscale != 1.0) {
    ierr = VecScale(&ctx->vscale,y);CHKERRQ(ierr);
  } else if (ctx->vshift != 0.0) {
    ierr = VecAXPY(&ctx->vshift,a,y);CHKERRQ(ierr);
  }

  if (ctx->sp) {ierr = MatNullSpaceRemove(ctx->sp,y,PETSC_NULL);CHKERRQ(ierr);}

  ierr = PetscLogEventEnd(MAT_MultMatrixFree,a,y,0,0);CHKERRQ(ierr);
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
int MatGetDiagonal_MFFD(Mat mat,Vec a)
{
  MatSNESMFCtx ctx = (MatSNESMFCtx)mat->data;
  PetscScalar  h,*aa,*ww,v;
  PetscReal    epsilon = PETSC_SQRT_MACHINE_EPSILON,umin = 100.0*PETSC_SQRT_MACHINE_EPSILON;
  Vec          w,U;
  int          i,ierr,rstart,rend;

  PetscFunctionBegin;
  if (!ctx->funci) {
    SETERRQ(1,"Requirers calling MatSNESMFSetFunctioni() first");
  }

  w    = ctx->w;
  U    = ctx->current_u;
  ierr = (*ctx->func)(0,U,a,ctx->funcctx);CHKERRQ(ierr);
  ierr = (*ctx->funcisetbase)(U,ctx->funcctx);CHKERRQ(ierr);
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
    ierr          = (*ctx->funci)(i,w,&v,ctx->funcctx);CHKERRQ(ierr);
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
int MatShift_MFFD(const PetscScalar *a,Mat Y)
{
  MatSNESMFCtx shell = (MatSNESMFCtx)Y->data;  
  PetscFunctionBegin;
  shell->vshift += *a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_MFFD"
int MatScale_MFFD(const PetscScalar *a,Mat Y)
{
  MatSNESMFCtx shell = (MatSNESMFCtx)Y->data;  
  PetscFunctionBegin;
  shell->vscale *= *a;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatCreateSNESMF"
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
          MatSNESMFSetHHistory(), MatSNESMFResetHHistory(), MatCreateMF(),
          MatSNESMFGetH(),MatSNESMFKSPMonitor(), MatSNESMFRegisterDynamic), MatSNESMFComputeJacobian()
 
@*/
int MatCreateSNESMF(SNES snes,Vec x,Mat *J)
{
  MatSNESMFCtx mfctx;
  int          ierr;

  PetscFunctionBegin;
  ierr = MatCreateMF(x,J);CHKERRQ(ierr);

  mfctx          = (MatSNESMFCtx)(*J)->data;
  mfctx->snes    = snes;
  mfctx->usesnes = PETSC_TRUE;
  PetscLogObjectParent(snes,*J);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFSetBase_FD"
int MatSNESMFSetBase_FD(Mat J,Vec U)
{
  int          ierr;
  MatSNESMFCtx ctx = (MatSNESMFCtx)J->data;

  PetscFunctionBegin;
  ierr = MatSNESMFResetHHistory(J);CHKERRQ(ierr);
  ctx->current_u = U;
  ctx->usesnes   = PETSC_FALSE;
  if (ctx->w == PETSC_NULL) {
    ierr = VecDuplicate(ctx->current_u, &ctx->w);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

typedef int (*FCN3)(Vec,Vec,PetscScalar*,void*); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFSetCheckh_FD"
int MatSNESMFSetCheckh_FD(Mat J,FCN3 fun,void*ectx)
{
  MatSNESMFCtx ctx = (MatSNESMFCtx)J->data;

  PetscFunctionBegin;
  ctx->checkh    = fun;
  ctx->checkhctx = ectx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFSetFromOptions"
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
  MatSNESMFCtx mfctx = (MatSNESMFCtx)mat->data;
  int          ierr;
  PetscTruth   flg;
  char         ftype[256];

  PetscFunctionBegin;
  if (!MatSNESMFRegisterAllCalled) {ierr = MatSNESMFRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  
  ierr = PetscOptionsBegin(mfctx->comm,mfctx->prefix,"Set matrix free computation parameters","MatSNESMF");CHKERRQ(ierr);
  ierr = PetscOptionsList("-snes_mf_type","Matrix free type","MatSNESMFSetType",MatSNESMPetscFList,mfctx->type_name,ftype,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatSNESMFSetType(mat,ftype);CHKERRQ(ierr);
  }

  ierr = PetscOptionsReal("-snes_mf_err","set sqrt relative error in function","MatSNESMFSetFunctionError",mfctx->error_rel,&mfctx->error_rel,0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_mf_period","how often h is recomputed","MatSNESMFSetPeriod",mfctx->recomputeperiod,&mfctx->recomputeperiod,0);CHKERRQ(ierr);
  if (mfctx->snes) {
    ierr = PetscOptionsName("-snes_mf_ksp_monitor","Monitor matrix-free parameters","MatSNESMFKSPMonitor",&flg);CHKERRQ(ierr);
    if (flg) {
      KSP ksp;
      ierr = SNESGetKSP(mfctx->snes,&ksp);CHKERRQ(ierr);
      ierr = KSPSetMonitor(ksp,MatSNESMFKSPMonitor,PETSC_NULL,0);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsName("-snes_mf_check_positivity","Insure that U + h*a is nonnegative","MatSNESMFSetCheckh",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatSNESMFSetCheckh(mat,MatSNESMFCheckPositivity,0);CHKERRQ(ierr);
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

.seealso: MatCreateMF, MatCreateSNESMF
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_MFFD"
int MatCreate_MFFD(Mat A)
{
  MatSNESMFCtx mfctx;
  int          ierr;

  PetscFunctionBegin;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = SNESInitializePackage(PETSC_NULL);                                                               CHKERRQ(ierr);
#endif

  PetscHeaderCreate(mfctx,_p_MatSNESMFCtx,struct _MFOps,MATSNESMFCTX_COOKIE,0,"SNESMF",A->comm,MatDestroy_MFFD,MatView_MFFD);
  PetscLogObjectCreate(mfctx);
  mfctx->sp              = 0;
  mfctx->snes            = 0;
  mfctx->error_rel       = PETSC_SQRT_MACHINE_EPSILON;
  mfctx->recomputeperiod = 1;
  mfctx->count           = 0;
  mfctx->currenth        = 0.0;
  mfctx->historyh        = PETSC_NULL;
  mfctx->ncurrenth       = 0;
  mfctx->maxcurrenth     = 0;
  mfctx->type_name       = 0;
  mfctx->usesnes         = PETSC_FALSE;

  mfctx->vshift          = 0.0;
  mfctx->vscale          = 1.0;

  /* 
     Create the empty data structure to contain compute-h routines.
     These will be filled in below from the command line options or 
     a later call with MatSNESMFSetType() or if that is not called 
     then it will default in the first use of MatMult_MFFD()
  */
  mfctx->ops->compute        = 0;
  mfctx->ops->destroy        = 0;
  mfctx->ops->view           = 0;
  mfctx->ops->setfromoptions = 0;
  mfctx->hctx                = 0;

  mfctx->func                = 0;
  mfctx->funcctx             = 0;
  mfctx->funcvec             = 0;
  mfctx->w                   = PETSC_NULL;

  A->data                = mfctx;

  A->ops->mult           = MatMult_MFFD;
  A->ops->destroy        = MatDestroy_MFFD;
  A->ops->view           = MatView_MFFD;
  A->ops->assemblyend    = MatAssemblyEnd_MFFD;
  A->ops->getdiagonal    = MatGetDiagonal_MFFD;
  A->ops->scale          = MatScale_MFFD;
  A->ops->shift          = MatShift_MFFD;
  A->ops->setfromoptions = MatSNESMFSetFromOptions;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatSNESMFSetBase_C","MatSNESMFSetBase_FD",MatSNESMFSetBase_FD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatSNESMFSetFunctioniBase_C","MatSNESMFSetFunctioniBase_FD",MatSNESMFSetFunctioniBase_FD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatSNESMFSetFunctioni_C","MatSNESMFSetFunctioni_FD",MatSNESMFSetFunctioni_FD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatSNESMFSetCheckh_C","MatSNESMFSetCheckh_FD",MatSNESMFSetCheckh_FD);CHKERRQ(ierr);
  mfctx->mat = A;

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatCreateMF"
/*@C
   MatCreateMF - Creates a matrix-free matrix. See also MatCreateSNESMF() 

   Collective on Vec

   Input Parameters:
.  x - vector that defines layout of the vectors and matrices

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
.  -snes_mf_ksp_monitor - KSP monitor routine that prints differencing h
-  -snes_mf_check_positivity

.keywords: default, matrix-free, create, matrix

.seealso: MatDestroy(), MatSNESMFSetFunctionError(), MatSNESMFDefaultSetUmin()
          MatSNESMFSetHHistory(), MatSNESMFResetHHistory(), MatCreateSNESMF(),
          MatSNESMFGetH(),MatSNESMFKSPMonitor(), MatSNESMFRegisterDynamic),, MatSNESMFComputeJacobian()
 
@*/
int MatCreateMF(Vec x,Mat *J)
{
  MPI_Comm     comm;
  int          n,nloc,ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&nloc);CHKERRQ(ierr);
  ierr = MatCreate(comm,nloc,nloc,n,n,J);CHKERRQ(ierr);
  ierr = MatRegisterDynamic(MATMFFD,0,"MatCreate_MFFD",MatCreate_MFFD);CHKERRQ(ierr);
  ierr = MatSetType(*J,MATMFFD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFGetH"
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
int MatSNESMFGetH(Mat mat,PetscScalar *h)
{
  MatSNESMFCtx ctx = (MatSNESMFCtx)mat->data;

  PetscFunctionBegin;
  *h = ctx->currenth;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFKSPMonitor"
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
  ctx  = (MatSNESMFCtx)mat->data;

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

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFSetFunction"
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
  MatSNESMFCtx ctx = (MatSNESMFCtx)mat->data;

  PetscFunctionBegin;
  ctx->func    = func;
  ctx->funcctx = funcctx;
  ctx->funcvec = v;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFSetFunctioni"
/*@C
   MatSNESMFSetFunctioni - Sets the function for a single component

   Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via MatCreateSNESMF()
-  funci - the function to use

   Level: advanced

   Notes:
    If you use this you MUST call MatAssemblyBegin()/MatAssemblyEnd() on the matrix free
    matrix inside your compute Jacobian routine


.keywords: SNES, matrix-free, function

.seealso: MatCreateSNESMF(),MatSNESMFGetH(),
          MatSNESMFSetHHistory(), MatSNESMFResetHHistory(),
          MatSNESMFKSPMonitor(), SNESetFunction()
@*/
int MatSNESMFSetFunctioni(Mat mat,int (*funci)(int,Vec,PetscScalar*,void *))
{
  int  ierr,(*f)(Mat,int (*)(int,Vec,PetscScalar*,void *));

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatSNESMFSetFunctioni_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat,funci);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFSetFunctioniBase"
/*@C
   MatSNESMFSetFunctioniBase - Sets the base vector for a single component function evaluation

   Collective on Mat

   Input Parameters:
+  mat - the matrix free matrix created via MatCreateSNESMF()
-  func - the function to use

   Level: advanced

   Notes:
    If you use this you MUST call MatAssemblyBegin()/MatAssemblyEnd() on the matrix free
    matrix inside your compute Jacobian routine


.keywords: SNES, matrix-free, function

.seealso: MatCreateSNESMF(),MatSNESMFGetH(),
          MatSNESMFSetHHistory(), MatSNESMFResetHHistory(),
          MatSNESMFKSPMonitor(), SNESetFunction()
@*/
int MatSNESMFSetFunctioniBase(Mat mat,int (*func)(Vec,void *))
{
  int  ierr,(*f)(Mat,int (*)(Vec,void *));

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatSNESMFSetFunctioniBase_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat,func);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFSetPeriod"
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
  MatSNESMFCtx ctx = (MatSNESMFCtx)mat->data;

  PetscFunctionBegin;
  ctx->recomputeperiod = period;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFSetFunctionError"
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
  MatSNESMFCtx ctx = (MatSNESMFCtx)mat->data;

  PetscFunctionBegin;
  if (error != PETSC_DEFAULT) ctx->error_rel = error;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFAddNullSpace"
/*@
   MatSNESMFAddNullSpace - Provides a null space that an operator is
   supposed to have.  Since roundoff will create a small component in
   the null space, if you know the null space you may have it
   automatically removed.

   Collective on Mat 

   Input Parameters:
+  J - the matrix-free matrix context
-  nullsp - object created with MatNullSpaceCreate()

   Level: advanced

.keywords: SNES, matrix-free, null space

.seealso: MatNullSpaceCreate(), MatSNESMFGetH(), MatCreateSNESMF(),
          MatSNESMFSetHHistory(), MatSNESMFResetHHistory(),
          MatSNESMFKSPMonitor(), MatSNESMFErrorRel()
@*/
int MatSNESMFAddNullSpace(Mat J,MatNullSpace nullsp)
{
  int          ierr;
  MatSNESMFCtx ctx = (MatSNESMFCtx)J->data;
  MPI_Comm     comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)J,&comm);CHKERRQ(ierr);

  ctx->sp = nullsp;
  ierr    = PetscObjectReference((PetscObject)nullsp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFSetHHistory"
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
int MatSNESMFSetHHistory(Mat J,PetscScalar history[],int nhistory)
{
  MatSNESMFCtx ctx = (MatSNESMFCtx)J->data;

  PetscFunctionBegin;
  ctx->historyh    = history;
  ctx->maxcurrenth = nhistory;
  ctx->currenth    = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFResetHHistory"
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
  MatSNESMFCtx ctx = (MatSNESMFCtx)J->data;

  PetscFunctionBegin;
  ctx->ncurrenth    = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFComputeJacobian"
int MatSNESMFComputeJacobian(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure *flag,void *dummy)
{
  int ierr;
  PetscFunctionBegin;
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFSetBase"
/*@
    MatSNESMFSetBase - Sets the vector U at which matrix vector products of the 
        Jacobian are computed

    Collective on Mat

    Input Parameters:
+   J - the MatSNESMF matrix
-   U - the vector

    Notes: This is rarely used directly

    Level: advanced

@*/
int MatSNESMFSetBase(Mat J,Vec U)
{
  int  ierr,(*f)(Mat,Vec);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(J,MAT_COOKIE,1);
  PetscValidHeaderSpecific(U,VEC_COOKIE,2);
  ierr = PetscObjectQueryFunction((PetscObject)J,"MatSNESMFSetBase_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(J,U);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFSetCheckh"
/*@C
    MatSNESMFSetCheckh - Sets a function that checks the computed h and adjusts
        it to satisfy some criteria

    Collective on Mat

    Input Parameters:
+   J - the MatSNESMF matrix
.   fun - the function that checks h
-   ctx - any context needed by the function

    Options Database Keys:
.   -snes_mf_check_positivity

    Level: advanced

    Notes: For example, MatSNESMFSetCheckPositivity() insures that all entries
       of U + h*a are non-negative

.seealso:  MatSNESMFSetCheckPositivity()
@*/
int MatSNESMFSetCheckh(Mat J,int (*fun)(Vec,Vec,PetscScalar*,void*),void* ctx)
{
  int  ierr,(*f)(Mat,int (*)(Vec,Vec,PetscScalar*,void*),void*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(J,MAT_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)J,"MatSNESMFSetCheckh_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(J,fun,ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSNESMFSetCheckPositivity"
/*@
    MatSNESMFCheckPositivity - Checks that all entries in U + h*a are positive or
        zero, decreases h until this is satisfied.

    Collective on Vec

    Input Parameters:
+   U - base vector that is added to
.   a - vector that is added
.   h - scaling factor on a
-   dummy - context variable (unused)

    Options Database Keys:
.   -snes_mf_check_positivity

    Level: advanced

    Notes: This is rarely used directly, rather it is passed as an argument to 
           MatSNESMFSetCheckh()

.seealso:  MatSNESMFSetCheckh()
@*/
int MatSNESMFCheckPositivity(Vec U,Vec a,PetscScalar *h,void *dummy)
{
  PetscReal     val, minval;
  PetscScalar   *u_vec, *a_vec;
  int           ierr, i, size;
  MPI_Comm      comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)U,&comm);CHKERRQ(ierr);
  ierr = VecGetArray(U,&u_vec);CHKERRQ(ierr);  
  ierr = VecGetArray(a,&a_vec);CHKERRQ(ierr);  
  ierr = VecGetLocalSize(U,&size);CHKERRQ(ierr);
  minval = PetscAbsScalar(*h*1.01);
  for(i=0;i<size;i++) {
    if (PetscRealPart(u_vec[i] + *h*a_vec[i]) <= 0.0) {
      val = PetscAbsScalar(u_vec[i]/a_vec[i]);
      if (val < minval) minval = val;
    }
  }
  ierr = VecRestoreArray(U,&u_vec);CHKERRQ(ierr);  
  ierr = VecRestoreArray(a,&a_vec);CHKERRQ(ierr);
  ierr = PetscGlobalMin(&minval,&val,comm);CHKERRQ(ierr);
  if (val <= PetscAbsScalar(*h)) {
    PetscLogInfo(U,"MatSNESMFCheckPositivity: Scaling back h from %g to %g\n",PetscRealPart(*h),.99*val);
    if (PetscRealPart(*h) > 0.0) *h =  0.99*val;
    else                         *h = -0.99*val;
  }
  PetscFunctionReturn(0);
}







