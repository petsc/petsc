
#include "mp1.h"
/*
         Defines the u, v, omega physics for a given tempature
*/
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuessLocal1"
PetscErrorCode FormInitialGuessLocal1(DALocalInfo *info,Field1 **x)
{
  PetscInt       i,j;

  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      x[j][i].u     = 0.0;
      x[j][i].v     = 0.0;
      x[j][i].omega = 0.0;
    }
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal1"
/* 
      x2 contains given tempature field
*/
PetscErrorCode FormFunctionLocal1(DALocalInfo *info,Field1 **x,Field2 **x2,Field1 **f,void *ptr)
 {
  AppCtx         *user = (AppCtx*)ptr;
  PetscInt       xints,xinte,yints,yinte,i,j;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx;
  PetscReal      grashof,prandtl,lid;
  PetscScalar    u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;

  PetscFunctionBegin;
  grashof = user->grashof;  
  prandtl = user->prandtl;
  lid     = user->lidvelocity;

  dhx = (PetscReal)(info->mx-1);  dhy = (PetscReal)(info->my-1);
  hx = 1.0/dhx;                   hy = 1.0/dhy;
  hxdhy = hx*dhy;                 hydhx = hy*dhx;

  xints = info->xs; xinte = info->xs+info->xm; yints = info->ys; yinte = info->ys+info->ym;

  /* Test whether we are on the bottom edge of the global array */
  if (yints == 0) {
    j = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega + (x[j+1][i].u - x[j][i].u)*dhy;  
    }
  }

  /* Test whether we are on the top edge of the global array */
  if (yinte == info->my) {
    j = info->my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
        f[j][i].u     = x[j][i].u - lid;
        f[j][i].v     = x[j][i].v;
        f[j][i].omega = x[j][i].omega + (x[j][i].u - x[j-1][i].u)*dhy; 
    }
  }

  /* Test whether we are on the left edge of the global array */
  if (xints == 0) {
    i = 0;
    xints = xints + 1;
    /* left edge */
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega - (x[j][i+1].v - x[j][i].v)*dhx; 
    }
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte == info->mx) {
    i = info->mx - 1;
    xinte = xinte - 1;
    /* right edge */ 
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega - (x[j][i].v - x[j][i-1].v)*dhx; 
    }
  }

  /* Compute over the interior points */
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {      
      /* convective coefficients for upwinding */
	vx = x[j][i].u; avx = PetscAbsScalar(vx);
        vxp = .5*(vx+avx); vxm = .5*(vx-avx);
	vy = x[j][i].v; avy = PetscAbsScalar(vy);
        vyp = .5*(vy+avy); vym = .5*(vy-avy);

	/* U velocity */
        u          = x[j][i].u;
        uxx        = (2.0*u - x[j][i-1].u - x[j][i+1].u)*hydhx;
        uyy        = (2.0*u - x[j-1][i].u - x[j+1][i].u)*hxdhy;
        f[j][i].u  = uxx + uyy - .5*(x[j+1][i].omega-x[j-1][i].omega)*hx;

	/* V velocity */
        u          = x[j][i].v;
        uxx        = (2.0*u - x[j][i-1].v - x[j][i+1].v)*hydhx;
        uyy        = (2.0*u - x[j-1][i].v - x[j+1][i].v)*hxdhy;
        f[j][i].v  = uxx + uyy + .5*(x[j][i+1].omega-x[j][i-1].omega)*hy;

	/* Omega */
        u          = x[j][i].omega;
        uxx        = (2.0*u - x[j][i-1].omega - x[j][i+1].omega)*hydhx;
        uyy        = (2.0*u - x[j-1][i].omega - x[j+1][i].omega)*hxdhy;   
	f[j][i].omega = uxx + uyy + (vxp*(u - x[j][i-1].omega) + vxm*(x[j][i+1].omega - u))*hy + (vyp*(u - x[j-1][i].omega) + vym*(x[j+1][i].omega - u))*hx;
        if (x2) {
          f[j][i].omega += - .5 * grashof * (x2[j][i+1].temp - x2[j][i-1].temp) * hy;
        }
    }
  }
  PetscFunctionReturn(0);
} 
