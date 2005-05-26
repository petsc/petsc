

#include "ad_deriv.h"
#if !defined(AD_INCLUDE_mat_h)
#define AD_INCLUDE_mat_h
#include "mat.ad.h"
#endif

#include "src/adic/src/adpetsc.h"
typedef struct _n_PetscADICFunction   *PetscADICFunction;

#define SETERRQ(n,p,s) {return ad_PetscError(__LINE__,__FUNC__,__FILE__,__SDIR__,n,p,s);}
#define CHKERRQ(n)     {if (n) SETERRQ(n,0,(char *)0);}

#undef __FUNC__  
#define __FUNC__ "ad_PetscADICFunctionCreate"
int ad_PetscADICFunctionCreate(PetscADICFunction ctx)
{
  int ierr;

  /* Create active vectors of approriate size */
  ierr = ad_VecCreate(MPI_COMM_SELF, ctx->m, &(ctx->din)); CHKERRQ(ierr);
  ierr = ad_VecCreate(MPI_COMM_SELF, ctx->n, &(ctx->dout));CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ad_PetscADICFunctionEvaluate"
int ad_PetscADICFunctionEvaluateGradient(PetscADICFunction ctx,double *inx,double *outx,double *grad)
{
  int         ierr, i, m = ctx->m, n = ctx->n,j;
  DERIV_TYPE  *ina, *outa;

  ad_AD_ResetIndep();

  /*
    Copy inactive array into active PETSc vector
  */
  ierr = ad_VecGetArray(ctx->din, &(ina)); CHKERRQ(ierr);
  for (i = 0; i < m; i++) {
    DERIV_VAL(ina[i]) = inx[i];
    ad_AD_SetIndep(ina[i]);
  }
  ad_AD_SetIndepDone();
  ierr = ad_VecRestoreArray(ctx->din, &(ina)); CHKERRQ(ierr);

  ierr = (*ctx->ad_Function)(ctx->din, ctx->dout); CHKERRQ(ierr);

  /*
     Copy active result vector into inactive array
     and gradient information into grad array
  */
  ierr = ad_VecGetArray(ctx->dout, &(outa)); CHKERRQ(ierr);
  for (i = 0; i < n; i++) {
    ad_AD_ExtractGrad(outx,outa[i]);
    for ( j=0; j<m; j++ ) {
      grad[i+j*n] = outx[j];
    }
  }
  for (i = 0; i < n; i++) {
    outx[i] = DERIV_VAL(outa[i]);
  }
  ierr = ad_VecRestoreArray(ctx->dout, &(outa)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ad_PetscADICApplyGradientInitialize"
int ad_PetscADICFunctionApplyGradientInitialize(PetscADICFunction ctx,double *inx)
{
  int         ierr, i, m = ctx->m;
  DERIV_TYPE  *ina;

  ad_AD_ResetIndep();

  /*
    Copy inactive array into active PETSc vector
  */
  ierr = ad_VecGetArray(ctx->din, &(ina)); CHKERRQ(ierr);
  ad_AD_SetIndep(ina[0]);
  ad_AD_SetIndepDone();
  for (i = 0; i < m; i++) {
    DERIV_VAL(ina[i])     = inx[i];
  }
  ierr = ad_VecRestoreArray(ctx->din, &(ina)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ad_PetscADICApplyGradient"
int ad_PetscADICFunctionApplyGradient(PetscADICFunction ctx,double *inx,double *outx)
{
  int         ierr, i, m = ctx->m, n = ctx->n;
  DERIV_TYPE  *ina, *outa;

  /*
    Copy inactive array into active PETSc vector
  */
  ierr = ad_VecGetArray(ctx->din, &(ina)); CHKERRQ(ierr);
  for (i = 0; i < m; i++) {
    DERIV_grad(ina[i])[0] = inx[i];
  }
  ierr = ad_VecRestoreArray(ctx->din, &(ina)); CHKERRQ(ierr);

  ierr = (*ctx->ad_Function)(ctx->din, ctx->dout); CHKERRQ(ierr);

  /*
     Copy active result vector into inactive array
     and gradient information into grad array
  */
  ierr = ad_VecGetArray(ctx->dout, &(outa)); CHKERRQ(ierr);
  for (i = 0; i < n; i++) {
    ad_AD_ExtractGrad(outx+i,outa[i]);
  }
  ierr = ad_VecRestoreArray(ctx->dout, &(outa)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ad_PetscADICFunctionInitialize"
int ad_PetscADICFunctionInitialize(PetscADICFunction ctx)
{
  int ierr;

  if (!ctx->ad_FunctionInitialize) PetscFunctionReturn(0);

  ierr = (*ctx->ad_FunctionInitialize)(&ctx->ad_ctx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if !defined(AD_INCLUDE_viewer_h)
#define AD_INCLUDE_viewer_h
#include "viewer.ad.h"
#endif
#if !defined(AD_INCLUDE_mat_h)
#define AD_INCLUDE_mat_h
#include "vec.ad.h"
#endif
#if !defined(AD_INCLUDE_src_vec_vecimpl_h)
#define AD_INCLUDE_src_vec_vecimpl_h
#include "src/vec/vecimpl.ad.h"
#endif
#if !defined(AD_INCLUDE_src_vec_impls_dvecimpl_h)
#define AD_INCLUDE_src_vec_impls_dvecimpl_h
#include "src/vec/impls/dvecimpl.ad.h"
#endif
#if !defined(AD_INCLUDE_sys_h)
#define AD_INCLUDE_sys_h
#include "sys.ad.h"
#endif
int ad_VecView_Seq_File(Vec xin, Viewer viewer) 
{
  Vec_Seq  *x = (Vec_Seq  *)(xin->data);
  int      i, n = x->n, ierr, format,j;
  FILE     *fd;

  ierr = ad_ViewerASCIIGetPointer(viewer, &(fd)); CHKERRQ(ierr);
  ierr = ad_ViewerGetFormat(viewer, &(format));
  for (i = (0); i < n; i++) {
    fprintf(fd, "%20.18e\n", DERIV_VAL(x->array[i]));
  }
  fprintf(fd,"Gradient\n");
  for (i = 0; i < n; i++) {
    for ( j=0; j<n; j++ ) {
      fprintf(fd, "%20.18e ", x->array[i].grad[j]);
    }
    fprintf(fd,"\n");
  }
  fflush(fd);
  PetscFunctionReturn(0);
}
