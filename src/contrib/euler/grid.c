/*
   This file contains the routines that read the grid from a file and
   set various grid parameters.
 */
#include "user.h"

#undef __FUNC__
#define __FUNC__ "UserSetGridParameters"
/*
   UserSetGridParameters - Sets various grid parameters within the application
   context.

   Input Parameter:
   u - user-defined application context
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 */
int UserSetGridParameters(Euler *u)
{
  /* 
     Define Fortran grid points. Shifts between Fortran/C for the grid:
       - explicit boundary condition formulation:
         PETSc code works with the interior grid points only
           C:       i=0,i<ni-1; j=0,j<nj-1; k=0,k<nk-1
           Fortran: i=2,ni;     j=2,nj;     k=2,nk
       - implicit boundary condition formulation:
         PETSc code works with the interior grid and boundary points
           C:       i=0,i<ni+1; j=0,j<nj+1; k=0,k<nk+1
           Fortran: i=1,ni+1;   j=1,nj+1;   k=1,nk+1
  */
  if (u->bctype == EXPLICIT) {
    u->xsf  = u->xs+2; 
    u->ysf  = u->ys+2;
    u->zsf  = u->zs+2;
    u->gxsf = u->gxs+2;
    u->gysf = u->gys+2;
    u->gzsf = u->gzs+2;
    u->xef  = u->xe+1;
    u->yef  = u->ye+1;
    u->zef  = u->ze+1;
    u->gxef = u->gxe+1;
    u->gyef = u->gye+1;
    u->gzef = u->gze+1;
  } else {
    u->xsf  = u->xs+1; 
    u->ysf  = u->ys+1;
    u->zsf  = u->zs+1;
    u->gxsf = u->gxs+1;
    u->gysf = u->gys+1;
    u->gzsf = u->gzs+1;
    u->xef  = u->xe;
    u->yef  = u->ye;
    u->zef  = u->ze;
    u->gxef = u->gxe;
    u->gyef = u->gye;
    u->gzef = u->gze;
  }

  /* Use in Fortran code to get specific points */
  if (u->xe == u->mx) { 
    if (u->bctype == EXPLICIT) {
      u->xefm1  = u->xef-1;    /* points mx-1, my-1, mz-1 */
      u->xef01  = u->xef;      /* points mx, my, mz */
      u->xefp1  = u->xef+1;    /* points mx+1, my+1, mz+1 */
      u->gxef01 = u->gxef;     /* ghost points mx, my, mz */ 
      u->gxefp1 = u->gxef+1;   /* ghost points mx+1, my+1, mz+1 */ 
      u->gxefw  = u->gxef;     /* ending ghost point - 1 */ 
      u->xei    = u->xe;
      u->gxei   = u->gxe;
    } else {
      u->xefm1  = u->xef-2;    /* points mx-1, my-1, mz-1 */
      u->xef01  = u->xef-1;    /* points mx, my, mz */
      u->xefp1  = u->xef;      /* points mx+1, my+1, mz+1 */
      u->gxef01 = u->gxef-1;   /* ghost points mx, my, mz */ 
      u->gxefp1 = u->gxef;     /* ghost points mx+1, my+1, mz+1 */ 
      u->gxefw  = u->gxef01;   /* ending ghost point - 1 */ 
      u->xei    = u->xe-1;
      u->gxei   = u->gxe-1;
    }
  } else {
      u->xefm1  = u->xef;
      u->xef01  = u->xef;   
      u->xefp1  = u->xef;   
      u->gxef01 = u->gxef;
      u->gxefp1 = u->gxef;
      u->gxefw  = u->gxef-1;   
      u->xei    = u->xe;
      u->gxei   = u->gxe;
  }
  if (u->ye == u->my) {
    if (u->bctype == EXPLICIT) {
      u->yefm1  = u->yef-1;
      u->yef01  = u->yef;
      u->yefp1  = u->yef+1;
      u->gyef01 = u->gyef;
      u->gyefp1 = u->gyef+1;
      u->gyefw  = u->gyef;
      u->yei    = u->ye;
      u->gyei   = u->gye;
    } else {
      u->yefm1  = u->yef-2;
      u->yef01  = u->yef-1;
      u->yefp1  = u->yef;
      u->gyef01 = u->gyef-1;
      u->gyefp1 = u->gyef;
      u->gyefw  = u->gyef01;
      u->yei    = u->ye-1;
      u->gyei   = u->gye-1;
    }
  } else {
    u->yefm1  = u->yef;
    u->yef01  = u->yef;
    u->yefp1  = u->yef; 
    u->gyef01 = u->gyef;  
    u->gyefp1 = u->gyef;
    u->gyefw  = u->gyef-1;   
    u->yei    = u->ye;
    u->gyei   = u->gye;
  }
  if (u->ze == u->mz) {
    if (u->bctype == EXPLICIT) {
      u->zefm1  = u->zef-1;
      u->zef01  = u->zef;
      u->zefp1  = u->zef+1;
      u->gzef01 = u->gzef;
      u->gzefp1 = u->gzef+1;
      u->gzefw  = u->gzef;
      u->zei    = u->ze;
      u->gzei   = u->gze;
    } else {
      u->zefm1  = u->zef-2;
      u->zef01  = u->zef-1;
      u->zefp1  = u->zef;
      u->gzef01 = u->gzef-1;
      u->gzefp1 = u->gzef;
      u->gzefw  = u->gzef01;
      u->zei    = u->ze-1;
      u->gzei   = u->gze-1;
    }
  } else {
    u->zefm1  = u->zef;
    u->zef01  = u->zef;
    u->zefp1  = u->zef;   
    u->gzef01 = u->gzef;
    u->gzefp1 = u->gzef;
    u->gzefw  = u->gzef-1;   
    u->zei    = u->ze;
    u->gzei   = u->gze;
  }

  if (u->xs == 0) { 
    u->xsf1  = 1;         /* grid points:  x=1, y=1, z=1 */
    u->xsf2  = 2;         /* grid points:  x=2, y=2, z=2 */
    u->gxsf1 = 1;         /* ghost points: x=1, y=1, z=1 */ 
    u->gxsf2 = 2;         /* ghost points: x=2, y=2, z=2 */ 
    u->gxsfw = u->gxsf;   /* starting ghost point + 1 */
    if (u->bctype == EXPLICIT) {
      u->xsi  = u->xs;
      u->gxsi = u->gxs;
    } else {
      u->xsi  = u->xs+1;
      u->gxsi = u->gxs+1;
    }
  } else {
    u->xsf1  = u->xsf;
    u->xsf2  = u->xsf;
    u->gxsf1 = u->gxsf;
    u->gxsf2 = u->gxsf;
    u->gxsfw = u->gxsf+1;
    u->xsi   = u->xs;
    u->gxsi  = u->gxs;
  }
  if (u->ys == 0) {
    u->ysf1  = 1;
    u->ysf2  = 2;
    u->gysf1 = 1;
    u->gysf2 = 2;
    u->gysfw = u->gysf;
    if (u->bctype == EXPLICIT) {
      u->ysi  = u->ys;
      u->gysi = u->gys;
    } else {
      u->ysi  = u->ys+1;
      u->gysi = u->gys+1;
    }
  } else {
    u->ysf1  = u->ysf;
    u->ysf2  = u->ysf;
    u->gysf1 = u->gysf;
    u->gysf2 = u->gysf;
    u->gysfw = u->gysf+1;
    u->ysi   = u->ys;
    u->gysi  = u->gys;
  }
  if (u->zs == 0) {
    u->zsf1  = 1;
    u->zsf2  = 2;
    u->gzsf1 = 1;
    u->gzsf2 = 2;
    u->gzsfw = u->gzsf;
    if (u->bctype == EXPLICIT) {
      u->zsi  = u->zs;
      u->gzsi = u->gzs;
    } else {
      u->zsi  = u->zs+1;
      u->gzsi = u->gzs+1;
    }
  } else {
    u->zsf1  = u->zsf;
    u->zsf2  = u->zsf;
    u->gzsf1 = u->gzsf;
    u->gzsf2 = u->gzsf;
    u->gzsfw = u->gzsf+1;
    u->zsi   = u->zs;
    u->gzsi  = u->gzs;
  }

  u->xmfp1 = u->xefp1 - u->xsf1 + 1; /* widths for Fortran */
  u->ymfp1 = u->yefp1 - u->ysf1 + 1;
  u->zmfp1 = u->zefp1 - u->zsf1 + 1;
  u->gxmfp1 = u->gxefp1 - u->gxsf1 + 1; /* ghost widths for Fortran */
  u->gymfp1 = u->gyefp1 - u->gysf1 + 1;
  u->gzmfp1 = u->gzefp1 - u->gzsf1 + 1;

  if (u->print_grid) {
    PetscSequentialPhaseBegin(u->comm,1);
    fprintf(stdout,"[%d] Grid points:\n\
     xs=%d, xsi=%d, xe=%d, xei=%d, xm=%d, xmfp1=%d\n\
     ys=%d, ysi=%d, ye=%d, yei=%d, ym=%d, ymfp1=%d\n\
     zs=%d, zsi=%d, ze=%d, zei=%d, zm=%d, zmfp1=%d\n\
   Ghost points:\n\
     gxs=%d, gxsi=%d, gxe=%d, gxei=%d, gxm=%d, gxmfp1=%d\n\
     gys=%d, gysi=%d, gye=%d, gyei=%d, gym=%d, gymfp1=%d\n\
     gzs=%d, gzsi=%d, gze=%d, gzei=%d, gzm=%d, gzmfp1=%d\n",
     u->rank,u->xs,u->xsi,u->xe,u->xei,u->xm,u->xmfp1,
     u->ys,u->ysi,u->ye,u->yei,u->ym,u->ymfp1,
     u->zs,u->zsi,u->ze,u->zei,u->zm,u->zmfp1,
     u->gxs,u->gxsi,u->gxe,u->gxei,u->gxm,u->gxmfp1,
     u->gys,u->gysi,u->gye,u->gyei,u->gym,u->gymfp1,
     u->gzs,u->gzsi,u->gze,u->gzei,u->gzm,u->gzmfp1);
    fflush(stdout);
    PetscSequentialPhaseEnd(u->comm,1);
  }
  return 0;
}
/* ------------------------------------------------------------------------ */
#undef __FUNC__
#define __FUNC__ "UserSetGrid"
/* 
   UserSetGrid - Reads mesh and optionally retains only the local portion of grid.

   Input Parameter:
   app - application-defined context

   Notes:
   The -local_grid option saves considerable space for parallel runs.  However, 
   the post-processing for viewing physical quantities is NOT currently 
   compatible with this mode; we may eventually upgrade the post-processing
   phase.
 */
int UserSetGrid(Euler *app)
{
  Scalar *xt, *yt, *zt;
  int    llen, i, j ,k, gxs1, gxe01, gys1, gye01, gzs1, gze01;
  int    mx_l, my_l, mz_l, mx_g, my_g, mz_g, ict_g, ict_l, ierr;
  int    itl, itu, ile, ktip;

  ierr = readmesh_(&itl,&itu,&ile,&ktip,app->xc,app->yc,app->zc); CHKERRQ(ierr);
  if ((app->bctype == EXPLICIT 
          && (app->ktip+2 != ktip || app->itl+2 != itl 
             || app->itu+2 != itu || app->ile+2 != ile)) ||
      ((app->bctype == IMPLICIT || app->bctype == IMPLICIT_SIZE)
          && (app->ktip+1 != ktip || app->itl+1 != itl 
            || app->itu+1 != itu || app->ile+1 != ile)))
     SETERRQ(1,1,"Conflicting wing parameters");

  /* Create local mesh and free global mesh (optional if using > 1 processor) */
  ierr = OptionsHasName(PETSC_NULL,"-local_grid",&flg); CHKERRA(ierr);
  if (if flg && app->size > 1) {
    mx_l = (app->gxef01 - app->gxsf1 + 1); mx_g = app->ni1-1;
    my_l = (app->gyef01 - app->gysf1 + 1); my_g = app->nj1-1;
    mz_l = (app->gzef01 - app->gzsf1 + 1); mz_g = app->nk1-1;

    llen = mx_l * my_l * mz_l;
    xt = (Scalar *)PetscMalloc(llen * 3 * sizeof(Scalar)); CHKPTRQ(xt);
    yt = xt + llen;
    zt = yt + llen;

    gxs1 = app->gxsf1-1; gxe01 = app->gxef01;
    gys1 = app->gysf1-1; gye01 = app->gyef01;
    gzs1 = app->gzsf1-1; gze01 = app->gzef01;

    for (k=gzs1; k<gze01; k++) {
      for (j=gys1; j<gye01; j++) {
        for (i=gxs1; i<gxe01; i++) {
          ict_l = (k-gzs1)*mx_l*my_l + (j-gys1)*mx_l + i-gxs1;
          ict_g = k*mx_g*my_g + j*mx_g + i;
          xt[ict_l] = app->xc[ict_g];
          yt[ict_l] = app->yc[ict_g];
          zt[ict_l] = app->zc[ict_g];
        }
      }
    }
    PetscFree(app->xc);
    app->xc = xt;
    app->yc = yt;
    app->zc = zt;
  }
  return 0;
}
