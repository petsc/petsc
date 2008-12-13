#include "mp.h"
/*
         Defines the tempature physics for a given u, v, omega
*/

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuessLocal2"
PetscErrorCode FormInitialGuessLocal2(DALocalInfo *info,Field2 **x,AppCtx *user)
{
  PetscInt       i,j;
  PetscScalar    dx;

  dx  = 1.0/(info->mx-1);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      x[j][i].temp  = .1 + ((PetscReal)(user->grashof>0))*i*dx;  
    }
  }
  return 0;
}

PetscLogEvent EVENT_FORMFUNCTIONLOCAL2;

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal2"
/* 
      
     x1 contains given velocity field

*/
PetscErrorCode FormFunctionLocal2(DALocalInfo *info,Field1**x1,Field2 **x,Field2 **f,void *ptr)
 {
  AppCtx         *user = (AppCtx*)ptr;
  PetscInt       xints,xinte,yints,yinte,i,j;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx;
  PetscReal      grashof,prandtl,lid;
  PetscScalar    u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(EVENT_FORMFUNCTIONLOCAL2,0,0,0,0);CHKERRQ(ierr);
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
      f[j][i].temp  = x[j][i].temp-x[j+1][i].temp;
    }
  }

  /* Test whether we are on the top edge of the global array */
  if (yinte == info->my) {
    j = info->my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
      f[j][i].temp  = x[j][i].temp-x[j-1][i].temp;
    }
  }

  /* Test whether we are on the left edge of the global array */
  if (xints == 0) {
    i = 0;
    xints = xints + 1;
    /* left edge */
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].temp  = x[j][i].temp;
    }
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte == info->mx) {
    i = info->mx - 1;
    xinte = xinte - 1;
    /* right edge */ 
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].temp  = x[j][i].temp - (PetscReal)(grashof>0);
    }
  }

  /* Compute over the interior points */
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      /* convective coefficients for upwinding */
      if (x1) {
        vx = x1[j][i].u; vy = x1[j][i].v; 
      } else {
        vx = vy = 0;
      }
      avx = PetscAbsScalar(vx); 
      vxp = .5*(vx+avx); vxm = .5*(vx-avx);
      avy = PetscAbsScalar(vy); 
      vyp = .5*(vy+avy); 
      vym = .5*(vy-avy);

      /* Temperature */
      u             = x[j][i].temp;
      uxx           = (2.0*u - x[j][i-1].temp - x[j][i+1].temp)*hydhx;
      uyy           = (2.0*u - x[j-1][i].temp - x[j+1][i].temp)*hxdhy;
      f[j][i].temp  =  uxx + uyy + prandtl * ((vxp*(u - x[j][i-1].temp) + vxm*(x[j][i+1].temp - u))*hy + (vyp*(u - x[j-1][i].temp) + vym*(x[j+1][i].temp - u))*hx);
    }
  }
  ierr = PetscLogEventEnd(EVENT_FORMFUNCTIONLOCAL2,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 
