#define PETSCDM_DLL

/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "private/daimpl.h"    /*I   "petscda.h"   I*/
#if defined(PETSC_HAVE_PNETCDF)
EXTERN_C_BEGIN
#include "pnetcdf.h"
EXTERN_C_END
#endif


#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include "mat.h"   /* Matlab include file */

#undef __FUNCT__  
#define __FUNCT__ "DAView_Matlab"
PetscErrorCode DAView_Matlab(DA da,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       dim,m,n,p,dof,swidth;
  DAStencilType  stencil;
  DAPeriodicType periodic;
  mxArray        *mx;
  const char     *fnames[] = {"dimension","m","n","p","dof","stencil_width","periodicity","stencil_type"};

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)da)->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = DAGetInfo(da,&dim,&m,&n,&p,0,0,0,&dof,&swidth,&periodic,&stencil);CHKERRQ(ierr);
    mx = mxCreateStructMatrix(1,1,8,(const char **)fnames);
    if (!mx) SETERRQ(PETSC_ERR_LIB,"Unable to generate Matlab struct array to hold DA informations");
    mxSetFieldByNumber(mx,0,0,mxCreateDoubleScalar((double)dim));
    mxSetFieldByNumber(mx,0,1,mxCreateDoubleScalar((double)m));
    mxSetFieldByNumber(mx,0,2,mxCreateDoubleScalar((double)n));
    mxSetFieldByNumber(mx,0,3,mxCreateDoubleScalar((double)p));
    mxSetFieldByNumber(mx,0,4,mxCreateDoubleScalar((double)dof));
    mxSetFieldByNumber(mx,0,5,mxCreateDoubleScalar((double)swidth));
    mxSetFieldByNumber(mx,0,6,mxCreateDoubleScalar((double)periodic));
    mxSetFieldByNumber(mx,0,7,mxCreateDoubleScalar((double)stencil));
    ierr = PetscObjectName((PetscObject)da);CHKERRQ(ierr);
    ierr = PetscViewerMatlabPutVariable(viewer,((PetscObject)da)->name,mx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "DAView_Binary"
PetscErrorCode DAView_Binary(DA da,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i,dim,m,n,p,dof,swidth,M,N,P;
  size_t         j,len;
  DAStencilType  stencil;
  DAPeriodicType periodic;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = DAGetInfo(da,&dim,&m,&n,&p,&M,&N,&P,&dof,&swidth,&periodic,&stencil);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    FILE *file;

    ierr = PetscViewerBinaryGetInfoPointer(viewer,&file);CHKERRQ(ierr);
    if (file) {
      char fieldname[PETSC_MAX_PATH_LEN];

      ierr = PetscFPrintf(PETSC_COMM_SELF,file,"-daload_info %D,%D,%D,%D,%D,%D,%D,%D\n",dim,m,n,p,dof,swidth,stencil,periodic);CHKERRQ(ierr);
      for (i=0; i<dof; i++) {
        if (da->fieldname[i]) {
          ierr = PetscStrncpy(fieldname,da->fieldname[i],PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
          ierr = PetscStrlen(fieldname,&len);CHKERRQ(ierr);
          len  = PetscMin(PETSC_MAX_PATH_LEN,len);CHKERRQ(ierr);
          for (j=0; j<len; j++) {
            if (fieldname[j] == ' ') fieldname[j] = '_';
          }
          ierr = PetscFPrintf(PETSC_COMM_SELF,file,"-daload_fieldname_%D %s\n",i,fieldname);CHKERRQ(ierr);
        }
      }
      if (da->coordinates) { /* save the DA's coordinates */
        ierr = PetscFPrintf(PETSC_COMM_SELF,file,"-daload_coordinates\n");CHKERRQ(ierr);
      }
    }
  } 

  /* save the coordinates if they exist to disk (in the natural ordering) */
  if (da->coordinates) {
    DA             dac;
    const PetscInt *lx,*ly,*lz;
    Vec            natural;

    /* create the appropriate DA to map to natural ordering */
    ierr = DAGetOwnershipRanges(da,&lx,&ly,&lz);CHKERRQ(ierr);
    if (dim == 1) {
      ierr = DACreate1d(comm,DA_NONPERIODIC,m,dim,0,lx,&dac);CHKERRQ(ierr); 
    } else if (dim == 2) {
      ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,m,n,M,N,dim,0,lx,ly,&dac);CHKERRQ(ierr); 
    } else if (dim == 3) {
      ierr = DACreate3d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,m,n,p,M,N,P,dim,0,lx,ly,lz,&dac);CHKERRQ(ierr); 
    } else {
      SETERRQ1(PETSC_ERR_ARG_CORRUPT,"Dimension is not 1 2 or 3: %D\n",dim);
    }
    ierr = DACreateNaturalVector(dac,&natural);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)natural,"coor_");CHKERRQ(ierr);
    ierr = DAGlobalToNaturalBegin(dac,da->coordinates,INSERT_VALUES,natural);CHKERRQ(ierr);
    ierr = DAGlobalToNaturalEnd(dac,da->coordinates,INSERT_VALUES,natural);CHKERRQ(ierr);
    ierr = VecView(natural,viewer);CHKERRQ(ierr);
    ierr = VecDestroy(natural);CHKERRQ(ierr);
    ierr = DADestroy(dac);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAView_VTK"
PetscErrorCode DAView_VTK(DA da, PetscViewer viewer)
{
  PetscInt       dim, dof, M = 0, N = 0, P = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetInfo(da, &dim, &M, &N, &P, PETSC_NULL, PETSC_NULL, PETSC_NULL, &dof, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  /* if (dim != 3) {SETERRQ(PETSC_ERR_SUP, "VTK output only works for three dimensional DAs.");} */
  if (!da->coordinates) {SETERRQ(PETSC_ERR_SUP, "VTK output requires DA coordinates.");}
  /* Write Header */
  ierr = PetscViewerASCIIPrintf(viewer,"# vtk DataFile Version 2.0\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Structured Mesh Example\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"ASCII\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"DATASET STRUCTURED_GRID\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"DIMENSIONS %d %d %d\n", M, N, P);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"POINTS %d double\n", M*N*P);CHKERRQ(ierr);
  if (da->coordinates) {
    DA  dac;
    Vec natural;

    ierr = DAGetCoordinateDA(da, &dac);CHKERRQ(ierr);
    ierr = DACreateNaturalVector(dac, &natural);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) natural, "coor_");CHKERRQ(ierr);
    ierr = DAGlobalToNaturalBegin(dac, da->coordinates, INSERT_VALUES, natural);CHKERRQ(ierr);
    ierr = DAGlobalToNaturalEnd(dac, da->coordinates, INSERT_VALUES, natural);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_COORDS);CHKERRQ(ierr);
    ierr = VecView(natural, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = VecDestroy(natural);CHKERRQ(ierr);
    ierr = DADestroy(dac);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAView"
/*@C
   DAView - Visualizes a distributed array object.

   Collective on DA

   Input Parameters:
+  da - the distributed array
-  ptr - an optional visualization context

   Notes:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
.     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 
-     PETSC_VIEWER_DRAW_WORLD - to default window

   The user can open alternative visualization contexts with
+    PetscViewerASCIIOpen() - Outputs vector to a specified file
-    PetscViewerDrawOpen() - Outputs vector to an X window display

   Default Output Format:
  (for 3d arrays)
.vb
   Processor [proc] M  N  P  m  n  p  w  s
   X range: xs xe, Y range: ys, ye, Z range: zs, ze

   where
      M,N,P - global dimension in each direction of the array
      m,n,p - corresponding number of procs in each dimension 
      w - number of degrees of freedom per node
      s - stencil width
      xs, xe - internal local starting/ending grid points
               in x-direction, (augmented to handle multiple 
               degrees of freedom per node)
      ys, ye - local starting/ending grid points in y-direction
      zs, ze - local starting/ending grid points in z-direction
.ve

   Options Database Key:
.  -da_view - Calls DAView() at the conclusion of DACreate1d(),
              DACreate2d(), and DACreate3d()

   Level: beginner

   Notes:
   Use DAGetCorners() and DAGetGhostCorners() to get the starting
   and ending grid points (ghost points) in each direction.

   When drawing the DA grid it only draws the logical grid and does not
   respect the grid coordinates set with DASetCoordinates()

.keywords: distributed array, view, visualize

.seealso: PetscViewerASCIIOpen(), PetscViewerDrawOpen(), DAGetInfo(), DAGetCorners(),
          DAGetGhostCorners(), DAGetOwnershipRanges(), DACreate(), DACreate1d(), DACreate2d(), DACreate3d()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAView(DA da,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscInt       i,dof = da->w;
  PetscTruth     iascii,fieldsnamed = PETSC_FALSE,isbinary;
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  PetscTruth     ismatlab;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)da)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_MATLAB,&ismatlab);CHKERRQ(ierr);
#endif
  if (iascii) {
    PetscViewerFormat format;

    ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_VTK || format == PETSC_VIEWER_ASCII_VTK_CELL) {
      ierr = DAView_VTK(da, viewer);CHKERRQ(ierr);
    } else {
      for (i=0; i<dof; i++) {
        if (da->fieldname[i]) {
          fieldsnamed = PETSC_TRUE;
          break;
        }
      }
      if (fieldsnamed) {
        ierr = PetscViewerASCIIPrintf(viewer,"FieldNames: ");CHKERRQ(ierr);
        for (i=0; i<dof; i++) {
          if (da->fieldname[i]) {
            ierr = PetscViewerASCIIPrintf(viewer,"%s ",da->fieldname[i]);CHKERRQ(ierr);
          } else {
            ierr = PetscViewerASCIIPrintf(viewer,"(not named) ");CHKERRQ(ierr);
          }
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    }
  }
  if (isbinary){
    ierr = DAView_Binary(da,viewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  } else if (ismatlab) {
    ierr = DAView_Matlab(da,viewer);CHKERRQ(ierr);
#endif
  } else {
    ierr = (*da->ops->view)(da,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}  

#undef __FUNCT__  
#define __FUNCT__ "DAGetInfo"
/*@C
   DAGetInfo - Gets information about a given distributed array.

   Not Collective

   Input Parameter:
.  da - the distributed array

   Output Parameters:
+  dim     - dimension of the distributed array (1, 2, or 3)
.  M, N, P - global dimension in each direction of the array
.  m, n, p - corresponding number of procs in each dimension
.  dof     - number of degrees of freedom per node
.  s       - stencil width
.  wrap    - type of periodicity, one of DA_NONPERIODIC, DA_XPERIODIC, DA_YPERIODIC, 
             DA_XYPERIODIC, DA_XYZPERIODIC, DA_XZPERIODIC, DA_YZPERIODIC,DA_ZPERIODIC
-  st      - stencil type, either DA_STENCIL_STAR or DA_STENCIL_BOX

   Level: beginner
  
   Note:
   Use PETSC_NULL (PETSC_NULL_INTEGER in Fortran) in place of any output parameter that is not of interest.

.keywords: distributed array, get, information

.seealso: DAView(), DAGetCorners(), DAGetLocalInfo()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetInfo(DA da,PetscInt *dim,PetscInt *M,PetscInt *N,PetscInt *P,PetscInt *m,PetscInt *n,PetscInt *p,PetscInt *dof,PetscInt *s,DAPeriodicType *wrap,DAStencilType *st)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (dim)  *dim  = da->dim;
  if (M)    *M    = da->M;
  if (N)    *N    = da->N;
  if (P)    *P    = da->P;
  if (m)    *m    = da->m;
  if (n)    *n    = da->n;
  if (p)    *p    = da->p;
  if (dof)  *dof  = da->w;
  if (s)    *s    = da->s;
  if (wrap) *wrap = da->wrap;
  if (st)   *st   = da->stencil_type;
  PetscFunctionReturn(0);
}  

#undef __FUNCT__  
#define __FUNCT__ "DAGetLocalInfo"
/*@C
   DAGetLocalInfo - Gets information about a given distributed array and this processors location in it

   Not Collective

   Input Parameter:
.  da - the distributed array

   Output Parameters:
.  dainfo - structure containing the information

   Level: beginner
  
.keywords: distributed array, get, information

.seealso: DAGetInfo(), DAGetCorners()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetLocalInfo(DA da,DALocalInfo *info)
{
  PetscInt w;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidPointer(info,2);
  info->da   = da;
  info->dim  = da->dim;
  info->mx   = da->M;
  info->my   = da->N;
  info->mz   = da->P;
  info->dof  = da->w;
  info->sw   = da->s;
  info->pt   = da->wrap;
  info->st   = da->stencil_type;

  /* since the xs, xe ... have all been multiplied by the number of degrees 
     of freedom per cell, w = da->w, we divide that out before returning.*/
  w = da->w;  
  info->xs = da->xs/w; 
  info->xm = (da->xe - da->xs)/w;
  /* the y and z have NOT been multiplied by w */
  info->ys = da->ys;
  info->ym = (da->ye - da->ys);
  info->zs = da->zs;
  info->zm = (da->ze - da->zs); 

  info->gxs = da->Xs/w; 
  info->gxm = (da->Xe - da->Xs)/w;
  /* the y and z have NOT been multiplied by w */
  info->gys = da->Ys;
  info->gym = (da->Ye - da->Ys);
  info->gzs = da->Zs;
  info->gzm = (da->Ze - da->Zs); 
  PetscFunctionReturn(0);
}  

