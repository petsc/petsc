

#include "ad_deriv.h"
#if !defined(AD_INCLUDE_mat_h)
#define AD_INCLUDE_mat_h
#include "mat.ad.h"
#endif

#define SETERRQ(n,p,s) {return ad_PetscError(__LINE__,__FUNC__,__FILE__,__SDIR__,n,p,s);}
#define CHKERRQ(n)     {if (n) SETERRQ(n,0,(char *)0);}

typedef struct _PetscADICFunction   *PetscADICFunction;
struct _PetscADICFunction{
  Vec din, dout;
  int ( *Function)(Vec , Vec );
};

#undef __FUNC__  
#define __FUNC__ "ad_PetscADICFunctionCreate"
int ad_PetscADICFunctionCreate(PetscADICFunction ctx, int m, int n) 
{
  int ierr;

  /* Create active vectors of approriate size */
  ierr = ad_VecCreate(MPI_COMM_SELF, m, &(ctx->din)); CHKERRQ(ierr);
  ierr = ad_VecCreate(MPI_COMM_SELF, n, &(ctx->dout));CHKERRQ(ierr);

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscADICFunctionEvaluate"
int ad_PetscADICFunctionEvaluate(PetscADICFunction ctx,double  *inx,double  *outx, int m, int n)
{
  int         ierr, i;
  DERIV_TYPE  *ina,  *outa;

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




  ierr = (*ctx->Function)(ctx->din, ctx->dout); CHKERRQ(ierr);

  /*
     Copy active result vector into inactive array
  */
  ierr = ad_VecGetArray(ctx->dout, &(outa)); CHKERRQ(ierr);
  for (i = 0; i < n; i++) {
    outx[i] = DERIV_VAL(outa[i]);
  }
  ierr = ad_VecRestoreArray(ctx->dout, &(outa)); CHKERRQ(ierr);
  return 0;
}
