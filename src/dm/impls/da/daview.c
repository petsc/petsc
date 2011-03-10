#define PETSCDM_DLL

/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "private/daimpl.h"    /*I   "petscdm.h"   I*/

#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include "mat.h"   /* MATLAB include file */

#undef __FUNCT__  
#define __FUNCT__ "DMView_DA_Matlab"
PetscErrorCode DMView_DA_Matlab(DM da,PetscViewer viewer)
{
  PetscErrorCode   ierr;
  PetscMPIInt      rank;
  PetscInt         dim,m,n,p,dof,swidth;
  DMDAStencilType  stencil;
  DMDABoundaryType periodic;
  mxArray          *mx;
  const char       *fnames[] = {"dimension","m","n","p","dof","stencil_width","periodicity","stencil_type"};

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)da)->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = DMDAGetInfo(da,&dim,&m,&n,&p,0,0,0,&dof,&swidth,&periodic,&stencil);CHKERRQ(ierr);
    mx = mxCreateStructMatrix(1,1,8,(const char **)fnames);
    if (!mx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Unable to generate MATLAB struct array to hold DMDA informations");
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
#define __FUNCT__ "DMView_DA_Binary"
PetscErrorCode DMView_DA_Binary(DM da,PetscViewer viewer)
{
  PetscErrorCode   ierr;
  PetscMPIInt      rank;
  PetscInt         i,dim,m,n,p,dof,swidth,M,N,P;
  size_t           j,len;
  DMDAStencilType  stencil;
  DMDABoundaryType periodic;
  MPI_Comm         comm;
  DM_DA            *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da,&dim,&m,&n,&p,&M,&N,&P,&dof,&swidth,&periodic,&stencil);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    FILE *file;

    ierr = PetscViewerBinaryGetInfoPointer(viewer,&file);CHKERRQ(ierr);
    if (file) {
      char fieldname[PETSC_MAX_PATH_LEN];

      ierr = PetscFPrintf(PETSC_COMM_SELF,file,"-daload_info %D,%D,%D,%D,%D,%D,%D,%D\n",dim,m,n,p,dof,swidth,stencil,periodic);CHKERRQ(ierr);
      for (i=0; i<dof; i++) {
        if (dd->fieldname[i]) {
          ierr = PetscStrncpy(fieldname,dd->fieldname[i],PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
          ierr = PetscStrlen(fieldname,&len);CHKERRQ(ierr);
          len  = PetscMin(PETSC_MAX_PATH_LEN,len);CHKERRQ(ierr);
          for (j=0; j<len; j++) {
            if (fieldname[j] == ' ') fieldname[j] = '_';
          }
          ierr = PetscFPrintf(PETSC_COMM_SELF,file,"-daload_fieldname_%D %s\n",i,fieldname);CHKERRQ(ierr);
        }
      }
      if (dd->coordinates) { /* save the DMDA's coordinates */
        ierr = PetscFPrintf(PETSC_COMM_SELF,file,"-daload_coordinates\n");CHKERRQ(ierr);
      }
    }
  } 

  /* save the coordinates if they exist to disk (in the natural ordering) */
  if (dd->coordinates) {
    DM             dac;
    const PetscInt *lx,*ly,*lz;
    Vec            natural;

    /* create the appropriate DMDA to map to natural ordering */
    ierr = DMDAGetOwnershipRanges(da,&lx,&ly,&lz);CHKERRQ(ierr);
    if (dim == 1) {
      ierr = DMDACreate1d(comm,DMDA_NONPERIODIC,m,dim,0,lx,&dac);CHKERRQ(ierr); 
    } else if (dim == 2) {
      ierr = DMDACreate2d(comm,DMDA_NONPERIODIC,DMDA_STENCIL_BOX,m,n,M,N,dim,0,lx,ly,&dac);CHKERRQ(ierr); 
    } else if (dim == 3) {
      ierr = DMDACreate3d(comm,DMDA_NONPERIODIC,DMDA_STENCIL_BOX,m,n,p,M,N,P,dim,0,lx,ly,lz,&dac);CHKERRQ(ierr); 
    } else SETERRQ1(comm,PETSC_ERR_ARG_CORRUPT,"Dimension is not 1 2 or 3: %D\n",dim);
    ierr = DMDACreateNaturalVector(dac,&natural);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)natural,"coor_");CHKERRQ(ierr);
    ierr = DMDAGlobalToNaturalBegin(dac,dd->coordinates,INSERT_VALUES,natural);CHKERRQ(ierr);
    ierr = DMDAGlobalToNaturalEnd(dac,dd->coordinates,INSERT_VALUES,natural);CHKERRQ(ierr);
    ierr = VecView(natural,viewer);CHKERRQ(ierr);
    ierr = VecDestroy(natural);CHKERRQ(ierr);
    ierr = DMDestroy(dac);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMView_DA_VTK"
PetscErrorCode DMView_DA_VTK(DM da, PetscViewer viewer)
{
  PetscInt       dim, dof, M = 0, N = 0, P = 0;
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da, &dim, &M, &N, &P, PETSC_NULL, PETSC_NULL, PETSC_NULL, &dof, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  /* if (dim != 3) {SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "VTK output only works for three dimensional DMDAs.");} */
  if (!dd->coordinates) SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP, "VTK output requires DMDA coordinates.");
  /* Write Header */
  ierr = PetscViewerASCIIPrintf(viewer,"# vtk DataFile Version 2.0\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Structured Mesh Example\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"ASCII\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"DATASET STRUCTURED_GRID\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"DIMENSIONS %d %d %d\n", M, N, P);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"POINTS %d double\n", M*N*P);CHKERRQ(ierr);
  if (dd->coordinates) {
    DM  dac;
    Vec natural;

    ierr = DMDAGetCoordinateDA(da, &dac);CHKERRQ(ierr);
    ierr = DMDACreateNaturalVector(dac, &natural);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) natural, "coor_");CHKERRQ(ierr);
    ierr = DMDAGlobalToNaturalBegin(dac, dd->coordinates, INSERT_VALUES, natural);CHKERRQ(ierr);
    ierr = DMDAGlobalToNaturalEnd(dac, dd->coordinates, INSERT_VALUES, natural);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_COORDS);CHKERRQ(ierr);
    ierr = VecView(natural, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = VecDestroy(natural);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMDAGetInfo"
/*@C
   DMDAGetInfo - Gets information about a given distributed array.

   Not Collective

   Input Parameter:
.  da - the distributed array

   Output Parameters:
+  dim     - dimension of the distributed array (1, 2, or 3)
.  M, N, P - global dimension in each direction of the array
.  m, n, p - corresponding number of procs in each dimension
.  dof     - number of degrees of freedom per node
.  s       - stencil width
.  wrap    - type of periodicity, one of DMDA_NONPERIODIC, DMDA_XPERIODIC, DMDA_YPERIODIC, 
             DMDA_XYPERIODIC, DMDA_XYZPERIODIC, DMDA_XZPERIODIC, DMDA_YZPERIODIC,DMDA_ZPERIODIC
-  st      - stencil type, either DMDA_STENCIL_STAR or DMDA_STENCIL_BOX

   Level: beginner
  
   Note:
   Use PETSC_NULL (PETSC_NULL_INTEGER in Fortran) in place of any output parameter that is not of interest.

.keywords: distributed array, get, information

.seealso: DMView(), DMDAGetCorners(), DMDAGetLocalInfo()
@*/
PetscErrorCode  DMDAGetInfo(DM da,PetscInt *dim,PetscInt *M,PetscInt *N,PetscInt *P,PetscInt *m,PetscInt *n,PetscInt *p,PetscInt *dof,PetscInt *s,DMDABoundaryType *wrap,DMDAStencilType *st)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (dim)  *dim  = dd->dim;
  if (M)    *M    = dd->M;
  if (N)    *N    = dd->N;
  if (P)    *P    = dd->P;
  if (m)    *m    = dd->m;
  if (n)    *n    = dd->n;
  if (p)    *p    = dd->p;
  if (dof)  *dof  = dd->w;
  if (s)    *s    = dd->s;
  if (wrap) *wrap = dd->wrap;
  if (st)   *st   = dd->stencil_type;
  PetscFunctionReturn(0);
}  

#undef __FUNCT__  
#define __FUNCT__ "DMDAGetLocalInfo"
/*@C
   DMDAGetLocalInfo - Gets information about a given distributed array and this processors location in it

   Not Collective

   Input Parameter:
.  da - the distributed array

   Output Parameters:
.  dainfo - structure containing the information

   Level: beginner
  
.keywords: distributed array, get, information

.seealso: DMDAGetInfo(), DMDAGetCorners()
@*/
PetscErrorCode  DMDAGetLocalInfo(DM da,DMDALocalInfo *info)
{
  PetscInt w;
  DM_DA    *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(info,2);
  info->da   = da;
  info->dim  = dd->dim;
  info->mx   = dd->M;
  info->my   = dd->N;
  info->mz   = dd->P;
  info->dof  = dd->w;
  info->sw   = dd->s;
  info->pt   = dd->wrap;
  info->st   = dd->stencil_type;

  /* since the xs, xe ... have all been multiplied by the number of degrees 
     of freedom per cell, w = dd->w, we divide that out before returning.*/
  w = dd->w;  
  info->xs = dd->xs/w; 
  info->xm = (dd->xe - dd->xs)/w;
  /* the y and z have NOT been multiplied by w */
  info->ys = dd->ys;
  info->ym = (dd->ye - dd->ys);
  info->zs = dd->zs;
  info->zm = (dd->ze - dd->zs); 

  info->gxs = dd->Xs/w; 
  info->gxm = (dd->Xe - dd->Xs)/w;
  /* the y and z have NOT been multiplied by w */
  info->gys = dd->Ys;
  info->gym = (dd->Ye - dd->Ys);
  info->gzs = dd->Zs;
  info->gzm = (dd->Ze - dd->Zs); 
  PetscFunctionReturn(0);
}  

