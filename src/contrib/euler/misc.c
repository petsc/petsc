/*
   This file contains various tools used by the SNES/Julianne code.
 */
#include "user.h"

/***************************************************************************/
/*
   These dummy routines are used because some machines REQUIRE linking with
   the Fortran linker if any Fortran IO is performed.  The Fortran linker 
   requires one of these routines, even though it will never be called.
*/

int MAIN__()
{
  return 0;
}
#if !defined(PARCH_alpha)
int __main()
{
  return 0;
}
#endif

#undef __FUNC__
#define __FUNC__ "UnpackWork"
/***************************************************************************/
/*
   UnpackWork - Unpacks Fortran work arrays, converting from 5 
   component arrays to a single vector.  This routine converts 
   locally owned grid points only -- no ghost points!

   Input Parameters:
   app            - user-defined application context
   v0,v1,v2,v3,v4 - work arrays (dimensions include ghost points)
   X              - vector (dimension does not include ghost points)

   Output Parameter:
   X   - fully assembled vector

   Note: 
   This routine is called by ComputeFunction() and other routines
   that require the transfer of data from Fortran work arrays to a
   PETSc vector.
 */
int UnpackWork(Euler *app,Scalar *v0,Scalar *v1,Scalar *v2,Scalar *v3,
               Scalar *v4,Vec X)
{
  int    i, j, k, jkx, jkx_ng, jkv, ijkx, ijkv, ierr, ijkxt;
  int    gxs = app->gxs, gys = app->gys, gzs = app->gzs;
  int    xs = app->xs, ys = app->ys, zs = app->zs, xm = app->xm;
  int    xe = app->xe, ye = app->ye, ze = app->ze, ym = app->ym;
  int    gxm = app->gxm, gxmfp1 = app->gxmfp1, gym = app->gym, gymfp1 = app->gymfp1;
  int    gxsf1 = app->gxsf1, gysf1 = app->gysf1, gzsf1 = app->gzsf1;
  Scalar val[5], *x;
  int    pos[5], *ltog, nloc, nc = app->nc;

  /* Note: Due to grid point reordering with DAs, we must always work
     with the local grid points for the vector X, then transform them to
     the new global numbering via DAGetGlobalIndices().  We cannot work
     directly with the global numbers for the original uniprocessor grid! */

  if (app->use_vecsetvalues) {
    /* Either use VecSetValues() to set vector elements ... */
    ierr = DAGetGlobalIndices(app->da,&nloc,&ltog); CHKERRQ(ierr);
    if (app->bctype == IMPLICIT) {
      for (k=zs; k<ze; k++) {
        for (j=ys; j<ye; j++) {
          jkx = (j-gys)*gxm + (k-gzs)*gxm*gym;
          for (i=xs; i<xe; i++) {
            ijkx   = jkx + i-gxs;
            ijkxt  = nc * ijkx;
            pos[0] = ltog[ijkxt];
            pos[1] = ltog[ijkxt+1];
            pos[2] = ltog[ijkxt+2];
            pos[3] = ltog[ijkxt+3];
            pos[4] = ltog[ijkxt+4];
            val[0] = v0[ijkx];
            val[1] = v1[ijkx];
            val[2] = v2[ijkx];
            val[3] = v3[ijkx];
            val[4] = v4[ijkx];
            ierr = VecSetValues(X,5,pos,val,INSERT_VALUES); CHKERRQ(ierr);
          }
        }
      }
    } else if (app->bctype == IMPLICIT_SIZE) {
      int xsi = app->xsi, ysi = app->ysi, zsi = app->zsi;
      int xei = app->xei, yei = app->yei, zei = app->zei;
      for (k=zsi; k<zei; k++) {
        for (j=ysi; j<yei; j++) {
          jkx = (j-gys)*gxm + (k-gzs)*gxm*gym;
          for (i=xsi; i<xei; i++) {
            ijkx   = jkx + i-gxs;
            ijkxt  = nc * ijkx;
            pos[0] = ltog[ijkxt];
            pos[1] = ltog[ijkxt+1];
            pos[2] = ltog[ijkxt+2];
            pos[3] = ltog[ijkxt+3];
            pos[4] = ltog[ijkxt+4];
            val[0] = v0[ijkx];
            val[1] = v1[ijkx];
            val[2] = v2[ijkx];
            val[3] = v3[ijkx];
            val[4] = v4[ijkx];
            ierr = VecSetValues(X,5,pos,val,INSERT_VALUES); CHKERRQ(ierr);
          }
        }
      }
    } else if (app->bctype == EXPLICIT) {  /* Set interior grid points only */
      for (k=zs; k<ze; k++) {
        for (j=ys; j<ye; j++) {
  	jkv = (j+2-gysf1)*gxmfp1 + (k+2-gzsf1)*gxmfp1*gymfp1;
          jkx = (j-gys)*gxm + (k-gzs)*gxm*gym;
          for (i=xs; i<xe; i++) {
            ijkv   = jkv + i+2-gxsf1;
            ijkx   = nc * (jkx + i-gxs);
            pos[0] = ltog[ijkx];
            pos[1] = ltog[ijkx+1];
            pos[2] = ltog[ijkx+2];
            pos[3] = ltog[ijkx+3];
            pos[4] = ltog[ijkx+4];
            val[0] = v0[ijkv];
            val[1] = v1[ijkv];
            val[2] = v2[ijkv];
            val[3] = v3[ijkv];
            val[4] = v4[ijkv];
            ierr = VecSetValues(X,5,pos,val,INSERT_VALUES); CHKERRQ(ierr);
          }
        }
      }
    } else {
      SETERRQ(1,1,"Unsupported bctype");
    }
    ierr = VecAssemblyBegin(X); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(X); CHKERRQ(ierr);
  } else {
    /* ... Or alternatively place values directly in the local vector array. 
       Since all vector elements are local, this direct assembly is
       simple (requiring no communication) and saves the overhead of
       calling VecSetValues().  Note: The work arrays (v0,v1,v2,v3,v4)
       include ghost points, while x does not! */
    ierr = VecGetArray(X,&x); CHKERRQ(ierr);
    if (app->bctype == IMPLICIT) {
      for (k=zs; k<ze; k++) {
        for (j=ys; j<ye; j++) {
          jkx    = (j-gys)*gxm + (k-gzs)*gxm*gym;
          jkx_ng = (j-ys)*xm + (k-zs)*xm*ym;
          for (i=xs; i<xe; i++) {
            ijkx  = jkx + i-gxs;
            ijkxt = nc * (jkx_ng + i-xs);
            x[ijkxt]   = v0[ijkx];
            x[ijkxt+1] = v1[ijkx];
            x[ijkxt+2] = v2[ijkx];
            x[ijkxt+3] = v3[ijkx];
            x[ijkxt+4] = v4[ijkx];
          }
        }
      }
    } else if (app->bctype == IMPLICIT_SIZE) { /* Set interior grid points only */
      int gxsi = app->gxsi, gysi = app->gysi, gzsi = app->gzsi;
      int gxei = app->gxei, gyei = app->gyei, gzei = app->gzei;
      for (k=gzsi; k<gzei; k++) {
        for (j=gysi; j<gyei; j++) {
          jkx    = (j-gys)*gxm + (k-gzs)*gxm*gym;
          jkx_ng = (j-ys)*xm + (k-zs)*xm*ym;
          for (i=gxsi; i<gxei; i++) {
            ijkx  = jkx + i-gxs;
            ijkxt = nc * (jkx_ng + i-xs);
            x[ijkxt]   = v0[ijkx];
            x[ijkxt+1] = v1[ijkx];
            x[ijkxt+2] = v2[ijkx];
            x[ijkxt+3] = v3[ijkx];
            x[ijkxt+4] = v4[ijkx];
          }
        }
      }
    } else if (app->bctype == EXPLICIT) {  /* Set interior grid points only */
      for (k=zs; k<ze; k++) {
        for (j=ys; j<ye; j++) {
          jkv    = (j+2-gysf1)*gxmfp1 + (k+2-gzsf1)*gxmfp1*gymfp1;
          jkx_ng = (j-ys)*xm + (k-zs)*xm*ym;
          for (i=xs; i<xe; i++) {
            ijkv = jkv + i+2-gxsf1;
            ijkx = nc * (jkx_ng + i-xs);
            x[ijkx]   = v0[ijkv];
            x[ijkx+1] = v1[ijkv];
            x[ijkx+2] = v2[ijkv];
            x[ijkx+3] = v3[ijkv];
            x[ijkx+4] = v4[ijkv];
          }
        }
      }
    } else SETERRQ(1,1,"Unsupported bctype");
  }
  return 0;
}
#undef __FUNC__
#define __FUNC__ "PackWork"
/***************************************************************************/
/*
   PackWork - Packs Fortran work arrays, including all ghost points.
   Converts from a single PETSc vector to 5 component arrays.

   Input Parameters:
   app            - user-defined application context
   v0,v1,v2,v3,v4 - work arrays (dimensions include ghost points)
   X              - parallel vector
   localX         - local work vector (dimension includes ghost points)

   Output Parameter:
   v0,v1,v2,v3,v4 - newly filled filled work arrays

   Note: 
   This routine is called by ComputeFunction() and other routines
   that require the transfer of data from a PETSc vector to Fortran
   work arrays.
 */
int PackWork(Euler *app,Vec X,Vec localX,
             Scalar *v0,Scalar *v1,Scalar *v2,Scalar *v3,Scalar *v4)
{
  int    ierr, i, j, k, jkx, jkv, ijkx, ijkv, ijkxt, nc = app->nc;
  int    gxm = app->gxm, gxmfp1 = app->gxmfp1, gym = app->gym, gymfp1 = app->gymfp1;
  int    gxs = app->gxs, gys = app->gys, gzs = app->gzs;
  int    gxe = app->gxe, gye = app->gye, gze = app->gze;
  int    gxsf1 = app->gxsf1, gysf1 = app->gysf1, gzsf1 = app->gzsf1;
  Scalar *x;
  
  /* Get ghost points */
  ierr = DAGlobalToLocalBegin(app->da,X,INSERT_VALUES,localX); CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(app->da,X,INSERT_VALUES,localX); CHKERRQ(ierr);
  ierr = VecGetArray(localX,&x); CHKERRQ(ierr);

  /* Set Fortran work arrays, including ghost points */
  if (app->bctype == IMPLICIT) {  /* Set interior and boundary grid points */
    for (k=gzs; k<gze; k++) {
      for (j=gys; j<gye; j++) {
        jkx = (j-gys)*gxm + (k-gzs)*gxm*gym;
        for (i=gxs; i<gxe; i++) {
          ijkx  = jkx + i-gxs;
          ijkxt = nc * ijkx;
          v0[ijkx] = x[ijkxt];
          v1[ijkx] = x[ijkxt+1];
          v2[ijkx] = x[ijkxt+2];
          v3[ijkx] = x[ijkxt+3];
          v4[ijkx] = x[ijkxt+4];
        }
      }
    }
  } else if (app->bctype == IMPLICIT_SIZE) { /* Set interior grid points only */
    int gxsi = app->gxsi, gysi = app->gysi, gzsi = app->gzsi;
    int gxei = app->gxei, gyei = app->gyei, gzei = app->gzei;
    for (k=gzsi; k<gzei; k++) {
      for (j=gysi; j<gyei; j++) {
        jkx = (j-gys)*gxm + (k-gzs)*gxm*gym;
        for (i=gxsi; i<gxei; i++) {
          ijkx  = jkx + i-gxs;
          ijkxt = nc * ijkx;
          v0[ijkx] = x[ijkxt];
          v1[ijkx] = x[ijkxt+1];
          v2[ijkx] = x[ijkxt+2];
          v3[ijkx] = x[ijkxt+3];
          v4[ijkx] = x[ijkxt+4];
        }
      }
    }
  } else if (app->bctype == EXPLICIT) {  /* Set interior grid points only */
    for (k=gzs; k<gze; k++) {
      for (j=gys; j<gye; j++) {
	jkv = (j+2-gysf1)*gxmfp1 + (k+2-gzsf1)*gxmfp1*gymfp1;
        jkx = (j-gys)*gxm + (k-gzs)*gxm*gym;
        for (i=gxs; i<gxe; i++) {
          ijkv = jkv + i+2-gxsf1;
          ijkx = nc * (jkx + i-gxs);
          v0[ijkv] = x[ijkx];
          v1[ijkv] = x[ijkx + 1];
          v2[ijkv] = x[ijkx + 2];
          v3[ijkv] = x[ijkx + 3];
          v4[ijkv] = x[ijkx + 4];
        }
      }
    }
  } else SETERRQ(1,1,"Unsupported bctype");

  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);
  return 0;
}
#undef __FUNC__
#define __FUNC__ "PackWorkComponent"
/***************************************************************************/
/*
   PackWorkComponent - Packs Fortran work array, including all ghost points
   and boundary points.  Converts from a vector to 1 component array.  

.  v0     - work array (dimension includes ghost points)
.  X      - parallel vector
.  localX - local work vector (dimension includes ghost points)

   Output Parameter:
.  v0      - newly filled filled work array

   Note:
   This routine is used only for the parallel pressure BCs.
 */
int PackWorkComponent(Euler *app,Vec X,Vec localX,Scalar *v0)
{
  int    ierr, i, j, k, jkx, jkv, ijkx, ijkv;
  int    gxe = app->gxe, gye = app->gye, gze = app->gze;
  int    gxm = app->gxm, gxmfp1 = app->gxmfp1, gym = app->gym, gymfp1 = app->gymfp1;
  int    gxs = app->gxs, gys = app->gys, gzs = app->gzs;
  int    gxsf1 = app->gxsf1, gysf1 = app->gysf1, gzsf1 = app->gzsf1;
  Scalar *x;

  /* Get ghost points */
  ierr = DAGlobalToLocalBegin(app->da1,X,INSERT_VALUES,localX); CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(app->da1,X,INSERT_VALUES,localX); CHKERRQ(ierr);
  ierr = VecGetArray(localX,&x); CHKERRQ(ierr);

  /* Set Fortran work arrays, including ghost points */
  if (app->bctype == IMPLICIT) {   /* Set interior and boundary grid points */
    for (k=gzs; k<gze; k++) {
      for (j=gys; j<gye; j++) {
        jkx = (j-gys)*gxm + (k-gzs)*gxm*gym;
        for (i=gxs; i<gxe; i++) {
          ijkx = jkx + i-gxs;
          v0[ijkx] = x[ijkx];
        }
      }
    }
  } else if (app->bctype == IMPLICIT_SIZE) {  /* Set interior grid points only */
    int gxsi = app->gxsi, gysi = app->gysi, gzsi = app->gzsi;
    int gxei = app->gxei, gyei = app->gyei, gzei = app->gzei;
    for (k=gzsi; k<gzei; k++) {
      for (j=gysi; j<gyei; j++) {
        jkx = (j-gys)*gxm + (k-gzs)*gxm*gym;
        for (i=gxsi; i<gxei; i++) {
          ijkx = jkx + i-gxs;
          v0[ijkx] = x[ijkx];
        }
      }
    }
  } else if (app->bctype == EXPLICIT) {  /* Set interior grid points only */
    for (k=gzs; k<gze; k++) {
      for (j=gys; j<gye; j++) {
	jkv = (j+2-gysf1)*gxmfp1 + (k+2-gzsf1)*gxmfp1*gymfp1;
        jkx = (j-gys)*gxm + (k-gzs)*gxm*gym;
        for (i=gxs; i<gxe; i++) {
          ijkv = jkv + i+2-gxsf1;
          ijkx = jkx + i-gxs;
          v0[ijkv] = x[ijkx];
        }
      }
    }
  } else SETERRQ(1,1,"Unsupported bctype");

  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);
  return 0;
}
#undef __FUNC__
#define __FUNC__ "UnpackWorkComponent"
/***************************************************************************/
/*
   UnpackWorkComponent - Unpacks Fortran work array, converting from 1
   component array a vector.  This routine converts locally owned grid
   points only -- no ghost points!  

   Input Parameters:
   app - user-defined application context
   v0  - work array (dimension includes ghost points)
   X   - vector (dimension does not include ghost points)

   Output Parameter:
   X   - fully assembled vector

   Note: 
   This routine is used for applying parallel pressure BCs and output of
   data for external viewing; it is called by BoundaryConditionsExplicit()
   and MonitorDumpVRML().
 */
int UnpackWorkComponent(Euler *app,Scalar *v0,Vec X)
{
  int    i, j, k, jkx_ng, jkx, jkv, ijkx, ijkv, ierr, *ltog, nloc;
  int    xs = app->xs, ys = app->ys, zs = app->zs, xm = app->xm;
  int    xe = app->xe, ye = app->ye, ze = app->ze, ym = app->ym;
  int    gxs = app->gxs, gys = app->gys, gzs = app->gzs;
  int    gxsf1 = app->gxsf1, gysf1 = app->gysf1, gzsf1 = app->gzsf1;
  int    gxm = app->gxm, gxmfp1 = app->gxmfp1, gym = app->gym, gymfp1 = app->gymfp1;
  Scalar *x;
 
  /* Note: Due to grid point reordering with DAs, we must always work
     with the local grid points for the vector X, then transform them to
     the new global numbering via DAGetGlobalIndices().  We cannot work
     directly with the global numbers for the original uniprocessor grid! */

  if (app->use_vecsetvalues) {
    /* Either use VecSetValues() to set vector elements ... */
    ierr = DAGetGlobalIndices(app->da1,&nloc,&ltog); CHKERRQ(ierr);
    if (app->bctype == IMPLICIT) {
      for (k=zs; k<ze; k++) {
        for (j=ys; j<ye; j++) {
          jkx = (j-gys)*gxm + (k-gzs)*gxm*gym;
          for (i=xs; i<xe; i++) {
            ijkx = jkx + i-gxs;
            ierr = VecSetValues(X,1,&ltog[ijkx],&v0[ijkx],INSERT_VALUES); CHKERRQ(ierr);
          }
        }
      }
    } else if (app->bctype == IMPLICIT_SIZE) {
      int xsi = app->xsi, ysi = app->ysi, zsi = app->zsi;
      int xei = app->xei, yei = app->yei, zei = app->zei;
      for (k=zsi; k<zei; k++) {
        for (j=ysi; j<yei; j++) {
          jkx = (j-gys)*gxm + (k-gzs)*gxm*gym;
          for (i=xsi; i<xei; i++) {
            ijkx = jkx + i-gxs;
            ierr = VecSetValues(X,1,&ltog[ijkx],&v0[ijkx],INSERT_VALUES); CHKERRQ(ierr);
          }
        }
      }
    } else if (app->bctype == EXPLICIT) {  /* Set interior grid points only */
      for (k=zs; k<ze; k++) {
        for (j=ys; j<ye; j++) {
  	jkv = (j+2-gysf1)*gxmfp1 + (k+2-gzsf1)*gxmfp1*gymfp1;
          jkx = (j-gys)*gxm + (k-gzs)*gxm*gym;
          for (i=xs; i<xe; i++) {
            ijkv = jkv + i+2-gxsf1;
            ijkx = jkx + i-gxs;
            ierr = VecSetValues(X,1,&ltog[ijkx],&v0[ijkv],INSERT_VALUES); CHKERRQ(ierr);
          }
        }
      }
    } else SETERRQ(1,1,"Unsupported bctype");
    ierr = VecAssemblyBegin(X); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(X); CHKERRQ(ierr);
  } else {
    /* ... Or alternatively place values directly in the local vector array. 
       Since all vector elements are local, this direct assembly is
       simple (requiring no communication) and saves the overhead of
       calling VecSetValues().  Note: The work array v0 includes ghost
       points, while x does not! */
    ierr = VecGetArray(X,&x); CHKERRQ(ierr);
    if (app->bctype == IMPLICIT) {
      for (k=zs; k<ze; k++) {
        for (j=ys; j<ye; j++) {
          jkx    = (j-gys)*gxm + (k-gzs)*gxm*gym;
          jkx_ng = (j-ys)*xm + (k-zs)*xm*ym;
          for (i=xs; i<xe; i++) {
            x[jkx_ng + i-xs] = v0[jkx + i-gxs];
          }
        }
      }
    } else if (app->bctype == IMPLICIT_SIZE) {
      int xsi = app->xsi, ysi = app->ysi, zsi = app->zsi;
      int xei = app->xei, yei = app->yei, zei = app->zei;
      for (k=zsi; k<zei; k++) {
        for (j=ysi; j<yei; j++) {
          jkx    = (j-gys)*gxm + (k-gzs)*gxm*gym;
          jkx_ng = (j-ys)*xm + (k-zs)*xm*ym;
          for (i=xsi; i<xei; i++) {
            x[jkx_ng + i-xs] = v0[jkx + i-gxs];
          }
        }
      }
    } else if (app->bctype == EXPLICIT) {  /* Set interior grid points only */
      for (k=zs; k<ze; k++) {
        for (j=ys; j<ye; j++) {
          jkv = (j+2-gysf1)*gxmfp1 + (k+2-gzsf1)*gxmfp1*gymfp1;
          jkx_ng = (j-ys)*xm + (k-zs)*xm*ym;
          for (i=xs; i<xe; i++) {
            x[jkx_ng + i-xs] = v0[jkv + i+2-gxsf1];
          }
        }
      }
    } else SETERRQ(1,1,"Unsupported bctype");
  }
  return 0;
}
#undef __FUNC__
#define __FUNC__ "GridTest"
/***************************************************************************/
/* 
   GridTest - Tests parallel grid operations (not used when running actual
   application code).

   Note:
   This routine enables testing of small size grids separately from the
   application, the Euler input files required by the application code are
   restricted to only 2 problem sizes, both of which are annoyingly large
   for simple debugging tests.

   This routine also sets a vector that enables testing of the routines
   PackWork(), UnpackWork(), and BCScatterSetUp().
 */
int GridTest(Euler *app)
{
  Viewer view1;
  int    i,j,k,ierr,jkx,ijkx1,ijkx;
  int    gxs = app->gxs, gys = app->gys, gzs = app->gzs;
  int    xs = app->xs, ys = app->ys, zs = app->zs;
  int    xe = app->xe, ye = app->ye, ze = app->ze;
  int    gxm = app->gxm, gym = app->gym;
  Scalar val[5];
  int    pos[5], *ltog, nloc;
  Vec    X = app->X;

  ierr = DAGetGlobalIndices(app->da,&nloc,&ltog); CHKERRQ(ierr);
  for (k=zs; k<ze; k++) {
    for (j=ys; j<ye; j++) {
      jkx = (j-gys)*gxm + (k-gzs)*gxm*gym;  
      for (i=xs; i<xe; i++) {
        ijkx1  = jkx + i-gxs;
        ijkx   = ijkx1 * 5;
	pos[0] = ltog[ijkx];
        pos[1] = ltog[ijkx+1];
        pos[2] = ltog[ijkx+2];
        pos[3] = ltog[ijkx+3];
        pos[4] = ltog[ijkx+4];
        val[0] = 10000*(k+1) + 1000*(j+1) + 10*(i+1);
	val[1] = 10000*(k+1) + 1000*(j+1) + 10*(i+1) + 1;
        val[2] = 10000*(k+1) + 1000*(j+1) + 10*(i+1) + 2;
        val[3] = 10000*(k+1) + 1000*(j+1) + 10*(i+1) + 3;
        val[4] = 10000*(k+1) + 1000*(j+1) + 10*(i+1) + 4;
        ierr = VecSetValues(X,5,pos,val,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
  }
  ierr = VecAssemblyBegin(X); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X); CHKERRQ(ierr);

  /* Note: DFVecView() provides vector output in the natural ordering 
     that would be used for 1 processor, regardless of the ordering used
     internally by the DA */
  if (app->print_debug) {
    ierr = ViewerFileOpenASCII(app->comm,"gtot.out",&view1); CHKERRQ(ierr);
    ierr = ViewerSetFormat(view1,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
    ierr = DFVecView(X,view1); CHKERRQ(ierr);
    ierr = ViewerDestroy(view1); CHKERRQ(ierr);
  }

  /* To test boundary condition scatters, set P vector as well */
  ierr = DAGetGlobalIndices(app->da1,&nloc,&ltog); CHKERRQ(ierr);
  for (k=zs; k<ze; k++) {
    for (j=ys; j<ye; j++) {
      jkx = (j-gys)*gxm + (k-gzs)*gxm*gym;  
      for (i=xs; i<xe; i++) {
        ijkx   = jkx + i-gxs;
	pos[0] = ltog[ijkx];
        val[0] = 10000*(k+1) + 1000*(j+1) + 10*(i+1);
        ierr = VecSetValues(app->P,1,pos,val,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
  }
  ierr = VecAssemblyBegin(app->P); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(app->P); CHKERRQ(ierr);

  ierr = ViewerFileOpenASCII(app->comm,"p.out",&view1); CHKERRQ(ierr);
  ierr = ViewerSetFormat(view1,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
  ierr = DFVecView(app->P,view1); CHKERRQ(ierr);

  return 0;
}
#undef __FUNC__
#define __FUNC__ "UnpackWorkComponent"
/***************************************************************************/
/*
  CheckSolution - Prints max/min of each component for interior of grid.
  Uniprocessor only for now.
 */
int CheckSolution(Euler *app,Vec X)
{
  int    ierr, i, j, k, jkx, ijkx, nc = app->nc;
  int    xm = app->xm, ym = app->ym, interior;
  int    xs = app->xs, ys = app->ys, zs = app->zs;
  int    xsi = app->xsi, ysi = app->ysi, zsi = app->zsi;
  int    xei = app->xei, yei = app->yei, zei = app->zei;
  int    imin[5], imax[5], jmin[5], jmax[5], kmin[5], kmax[5];
  Scalar *x, xmax[5], xmin[5], tmin, tmax;

  if (app->size > 1) {
    PetscPrintf(app->comm,"CheckSolution: uniprocessor only!\n"); return 0;
  }
  if (app->bctype != IMPLICIT) SETERRQ(1,1,"Only implicit bctype supported");

  ierr = VecGetArray(X,&x); CHKERRQ(ierr);
  interior = nc * ((ysi-ys)*xm + (zsi-zs)*xm*ym + xsi-xs);
  xmin[0] = xmax[0] = x[interior];
  xmin[1] = xmax[1] = x[interior + 1];
  xmin[2] = xmax[2] = x[interior + 2];
  xmin[3] = xmax[3] = x[interior + 3];
  xmin[4] = xmax[4] = x[interior + 4];
  for (i=0; i<nc; i++) {
    imin[i] = imax[i] = xsi;
    jmin[i] = jmax[i] = ysi;
    kmin[i] = kmax[i] = zsi;
  }
  for (k=zsi; k<zei; k++) {
    for (j=ysi; j<yei; j++) {
      jkx = (j-ys)*xm + (k-zs)*xm*ym;
      for (i=xsi; i<xei; i++) {
        ijkx  = nc * (jkx + i-xs);
        if (PetscAbsScalar(x[ijkx])   > PetscAbsScalar(xmax[0])) 
           {xmax[0] = x[ijkx];  imax[0] = i; jmax[0] = j; kmax[0] = k;}
        if (PetscAbsScalar(x[ijkx+1]) > PetscAbsScalar(xmax[1]))
           {xmax[1] = x[ijkx+1];  imax[1] = i; jmax[1] = j; kmax[1] = k;}
        if (PetscAbsScalar(x[ijkx+2]) > PetscAbsScalar(xmax[2]))
           {xmax[2] = x[ijkx+2];  imax[2] = i; jmax[2] = j; kmax[2] = k;}
        if (PetscAbsScalar(x[ijkx+3]) > PetscAbsScalar(xmax[3]))
           {xmax[3] = x[ijkx+3];  imax[3] = i; jmax[3] = j; kmax[3] = k;}
        if (PetscAbsScalar(x[ijkx+4]) > PetscAbsScalar(xmax[4]))
           {xmax[4] = x[ijkx+4];  imax[4] = i; jmax[4] = j; kmax[4] = k;}

        if (PetscAbsScalar(x[ijkx])   < PetscAbsScalar(xmin[0])) 
           {xmin[0] = x[ijkx];  imin[0] = i; jmin[0] = j; kmin[0] = k;}
        if (PetscAbsScalar(x[ijkx+1]) < PetscAbsScalar(xmin[1])) 
           {xmin[1] = x[ijkx+1]; imin[1] = i; jmin[1] = j; kmin[1] = k;}
        if (PetscAbsScalar(x[ijkx+2]) < PetscAbsScalar(xmin[2]))
           {xmin[2] = x[ijkx+2]; imin[2] = i; jmin[2] = j; kmin[2] = k;}
        if (PetscAbsScalar(x[ijkx+3]) < PetscAbsScalar(xmin[3]))
           {xmin[3] = x[ijkx+3]; imin[3] = i; jmin[3] = j; kmin[3] = k;}
        if (PetscAbsScalar(x[ijkx+4]) < PetscAbsScalar(xmin[4]))
           {xmin[4] = x[ijkx+4]; imin[4] = i; jmin[4] = j; kmin[4] = k;}
      }
    }
  }

  /* Need to do communication to get global min and max */
  tmin = PetscAbsScalar(xmin[0]);
  tmax = PetscAbsScalar(xmax[0]);
  for (i=1; i<nc; i++) {
    if (PetscAbsScalar(xmax[i]) > PetscAbsScalar(tmax)) tmax = xmax[i];
    if (PetscAbsScalar(xmin[i]) < PetscAbsScalar(tmin)) tmin = xmin[i];
  }
  PetscPrintf(app->comm,"  min=%g, max=%g, ratio=%g\n",tmin,tmax,tmin/tmax);
  PetscPrintf(app->comm,"    density: min=%g [%d,%d,%d], max=%g [%d,%d,%d], ratio=%g\n",
      xmin[0],imin[0],jmin[0],kmin[0],xmax[0],imax[0],jmax[0],kmax[0],xmin[0]/xmax[0]);
  PetscPrintf(app->comm,"    vel-u  : min=%g [%d,%d,%d], max=%g [%d,%d,%d], ratio=%g\n",
      xmin[1],imin[1],jmin[1],kmin[1],xmax[1],imax[1],jmax[1],kmax[1],xmin[1]/xmax[1]);
  PetscPrintf(app->comm,"    vel-v  : min=%g [%d,%d,%d], max=%g [%d,%d,%d], ratio=%g\n",
      xmin[2],imin[2],jmin[2],kmin[2],xmax[2],imax[2],jmax[2],kmax[2],xmin[2]/xmax[2]);
  PetscPrintf(app->comm,"    vel-w  : min=%g [%d,%d,%d], max=%g [%d,%d,%d], ratio=%g\n",
      xmin[3],imin[3],jmin[3],kmin[3],xmax[3],imax[3],jmax[3],kmax[3],xmin[3]/xmax[3]);
  PetscPrintf(app->comm,"    energy : min=%g [%d,%d,%d], max=%g [%d,%d,%d], ratio=%g\n",
      xmin[4],imin[4],jmin[4],kmin[4],xmax[4],imax[4],jmax[4],kmax[4],xmin[4]/xmax[4]);
  return 0;
}
