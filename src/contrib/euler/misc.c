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

int __main()
{
  return 0;
}
/***************************************************************************/
/*
   UnpackWork - Unpacks Fortran work arrays, converting from 5 
   component arrays to a single vector.  This routine converts 
   local interior grid points only -- no ghost points and no
   boundary points!
 */
int UnpackWork(Euler *app,Scalar *v0,Scalar *v1,Scalar *v2,Scalar *v3,
               Scalar *v4,Vec X)
{
  int    i, j, k, jkx, jkv, ijkx, ijkv, ierr, ijkxt;
  int    gxs = app->gxs, gys = app->gys, gzs = app->gzs;
  int    xs = app->xs, ys = app->ys, zs = app->zs;
  int    xe = app->xe, ye = app->ye, ze = app->ze;
  int    gxm = app->gxm, gxmfp1 = app->gxmfp1, gym = app->gym, gymfp1 = app->gymfp1;
  int    gxsf1 = app->gxsf1, gysf1 = app->gysf1, gzsf1 = app->gzsf1;
  Scalar val[5];
  int    pos[5], *ltog, nloc, nc = app->nc;

  /* Note: Due to grid point reordering with DAs, we must always work
     with the local grid points for the vector X, then transform them to
     the new global numbering via DAGetGlobalIndices().  We cannot work
     directly with the global numbers for the original uniprocessor grid! */

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
  } else SETERRQ(1,1,"UnpackWork: Unsupported bctype");

  ierr = VecAssemblyBegin(X); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X); CHKERRQ(ierr);
  return 0;
}
/***************************************************************************/
/*
   PackWork - Packs Fortran work arrays, including all ghost points
   Converts from a single vector to 5 component arrays.
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
  } else SETERRQ(1,1,"PackWork: Unsupported bctype");

  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);
  return 0;
}
/***************************************************************************/
/*
   PackWorkComponent - Packs Fortran work array, including all ghost points
   and boundary points.  Converts from a vector to 1 component array.  

   Note:  This routine is used only for the parallel pressure BCs.
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
  } else SETERRQ(1,1,"PackWorkComponent: Unsupported bctype");

  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);
  return 0;
}
/***************************************************************************/
/*
   UnpackWorkComponent - Unpacks Fortran work array, converting from 1
   component array a vector.  This routine converts standard grid points
   only -- no ghost points!  

   Note: This routine is used only for the parallel pressure BCs.
 */
int UnpackWorkComponent(Euler *app,Scalar *v0,Vec X)
{
  int    i, j, k, jkx, jkv, ijkx, ijkv, ierr, *ltog, nloc;
  int    xs = app->xs, ys = app->ys, zs = app->zs;
  int    xe = app->xe, ye = app->ye, ze = app->ze;
  int    gxs = app->gxs, gys = app->gys, gzs = app->gzs;
  int    gxsf1 = app->gxsf1, gysf1 = app->gysf1, gzsf1 = app->gzsf1;
  int    gxm = app->gxm, gxmfp1 = app->gxmfp1, gym = app->gym, gymfp1 = app->gymfp1;
 
  /* Note: Due to grid point reordering with DAs, we must always work
     with the local grid points for the vector X, then transform them to
     the new global numbering via DAGetGlobalIndices().  We cannot work
     directly with the global numbers for the original uniprocessor grid! */

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
  } else SETERRQ(1,1,"UnpackWorkComponent: Unsupported bctype");
  ierr = VecAssemblyBegin(X); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X); CHKERRQ(ierr);
  return 0;
}
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
