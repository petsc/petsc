#define PETSCDM_DLL

/* 
   Plots vectors obtained with DACreate2d()
*/

#include "src/dm/da/daimpl.h"      /*I  "petscda.h"   I*/
#include "vecimpl.h" 

#if defined(PETSC_HAVE_PNETCDF)
EXTERN_C_BEGIN
#include "pnetcdf.h"
EXTERN_C_END
#endif


/*
        The data that is passed into the graphics callback
*/
typedef struct {
  PetscInt     m,n,step,k;
  PetscReal    min,max,scale;
  PetscScalar  *xy,*v;
  PetscTruth   showgrid;
} ZoomCtx;

/*
       This does the drawing for one particular field 
    in one particular set of coordinates. It is a callback
    called from PetscDrawZoom()
*/
#undef __FUNCT__  
#define __FUNCT__ "VecView_MPI_Draw_DA2d_Zoom"
PetscErrorCode VecView_MPI_Draw_DA2d_Zoom(PetscDraw draw,void *ctx)
{
  ZoomCtx        *zctx = (ZoomCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       m,n,i,j,k,step,id,c1,c2,c3,c4;
  PetscReal      s,min,x1,x2,x3,x4,y_1,y2,y3,y4;
  PetscScalar   *v,*xy;

  PetscFunctionBegin; 
  m    = zctx->m;
  n    = zctx->n;
  step = zctx->step;
  k    = zctx->k;
  v    = zctx->v;
  xy   = zctx->xy;
  s    = zctx->scale;
  min  = zctx->min;
   
  /* PetscDraw the contour plot patch */
  for (j=0; j<n-1; j++) {
    for (i=0; i<m-1; i++) {
#if !defined(PETSC_USE_COMPLEX)
      id = i+j*m;    x1 = xy[2*id];y_1 = xy[2*id+1];c1 = (int)(PETSC_DRAW_BASIC_COLORS+s*(v[k+step*id]-min));
      id = i+j*m+1;  x2 = xy[2*id];y2  = y_1;       c2 = (int)(PETSC_DRAW_BASIC_COLORS+s*(v[k+step*id]-min));
      id = i+j*m+1+m;x3 = x2;      y3  = xy[2*id+1];c3 = (int)(PETSC_DRAW_BASIC_COLORS+s*(v[k+step*id]-min));
      id = i+j*m+m;  x4 = x1;      y4  = y3;        c4 = (int)(PETSC_DRAW_BASIC_COLORS+s*(v[k+step*id]-min));
#else
      id = i+j*m;    x1 = PetscRealPart(xy[2*id]);y_1 = PetscRealPart(xy[2*id+1]);c1 = (int)(PETSC_DRAW_BASIC_COLORS+s*(PetscRealPart(v[k+step*id])-min));
      id = i+j*m+1;  x2 = PetscRealPart(xy[2*id]);y2  = y_1;       c2 = (int)(PETSC_DRAW_BASIC_COLORS+s*(PetscRealPart(v[k+step*id])-min));
      id = i+j*m+1+m;x3 = x2;      y3  = PetscRealPart(xy[2*id+1]);c3 = (int)(PETSC_DRAW_BASIC_COLORS+s*(PetscRealPart(v[k+step*id])-min));
      id = i+j*m+m;  x4 = x1;      y4  = y3;        c4 = (int)(PETSC_DRAW_BASIC_COLORS+s*(PetscRealPart(v[k+step*id])-min));
#endif
      ierr = PetscDrawTriangle(draw,x1,y_1,x2,y2,x3,y3,c1,c2,c3);CHKERRQ(ierr);
      ierr = PetscDrawTriangle(draw,x1,y_1,x3,y3,x4,y4,c1,c3,c4);CHKERRQ(ierr);
      if (zctx->showgrid) {
        ierr = PetscDrawLine(draw,x1,y_1,x2,y2,PETSC_DRAW_BLACK);CHKERRQ(ierr);
        ierr = PetscDrawLine(draw,x2,y2,x3,y3,PETSC_DRAW_BLACK);CHKERRQ(ierr);
        ierr = PetscDrawLine(draw,x3,y3,x4,y4,PETSC_DRAW_BLACK);CHKERRQ(ierr);
        ierr = PetscDrawLine(draw,x4,y4,x1,y_1,PETSC_DRAW_BLACK);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecView_MPI_Draw_DA2d"
PetscErrorCode VecView_MPI_Draw_DA2d(Vec xin,PetscViewer viewer)
{
  DA                 da,dac,dag;
  PetscErrorCode     ierr;
  PetscMPIInt        rank;
  PetscInt           igstart,N,s,M,istart,isize,jgstart,*lx,*ly,w;
  PetscReal          coors[4],ymin,ymax,xmin,xmax;
  PetscDraw          draw,popup;
  PetscTruth         isnull,useports;
  MPI_Comm           comm;
  Vec                xlocal,xcoor,xcoorl;
  DAPeriodicType     periodic;
  DAStencilType      st;
  ZoomCtx            zctx;
  PetscDrawViewPorts *ports;
  PetscViewerFormat  format;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);

  ierr = PetscObjectQuery((PetscObject)xin,"DA",(PetscObject*)&da);CHKERRQ(ierr);
  if (!da) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector not generated from a DA");

  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = DAGetInfo(da,0,&M,&N,0,&zctx.m,&zctx.n,0,&w,&s,&periodic,&st);CHKERRQ(ierr);
  ierr = DAGetOwnershipRange(da,&lx,&ly,PETSC_NULL);CHKERRQ(ierr);

  /* 
        Obtain a sequential vector that is going to contain the local values plus ONE layer of 
     ghosted values to draw the graphics from. We also need its corresponding DA (dac) that will
     update the local values pluse ONE layer of ghost values. 
  */
  ierr = PetscObjectQuery((PetscObject)da,"GraphicsGhosted",(PetscObject*)&xlocal);CHKERRQ(ierr);
  if (!xlocal) {
    if (periodic != DA_NONPERIODIC || s != 1 || st != DA_STENCIL_BOX) {
      /* 
         if original da is not of stencil width one, or periodic or not a box stencil then
         create a special DA to handle one level of ghost points for graphics
      */
      ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,M,N,zctx.m,zctx.n,w,1,lx,ly,&dac);CHKERRQ(ierr); 
      ierr = PetscLogInfo((da,"VecView_MPI_Draw_DA2d:Creating auxilary DA for managing graphics ghost points\n"));CHKERRQ(ierr);
    } else {
      /* otherwise we can use the da we already have */
      dac = da;
    }
    /* create local vector for holding ghosted values used in graphics */
    ierr = DACreateLocalVector(dac,&xlocal);CHKERRQ(ierr);
    if (dac != da) {
      /* don't keep any public reference of this DA, is is only available through xlocal */
      ierr = DADestroy(dac);CHKERRQ(ierr);
    } else {
      /* remove association between xlocal and da, because below we compose in the opposite
         direction and if we left this connect we'd get a loop, so the objects could 
         never be destroyed */
      ierr = PetscObjectCompose((PetscObject)xlocal,"DA",0);CHKERRQ(ierr);
    }
    ierr = PetscObjectCompose((PetscObject)da,"GraphicsGhosted",(PetscObject)xlocal);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)xlocal);CHKERRQ(ierr);
  } else {
    if (periodic == DA_NONPERIODIC && s == 1 && st == DA_STENCIL_BOX) {
      dac = da;
    } else {
      ierr = PetscObjectQuery((PetscObject)xlocal,"DA",(PetscObject*)&dac);CHKERRQ(ierr);
    }
  }

  /*
      Get local (ghosted) values of vector
  */
  ierr = DAGlobalToLocalBegin(dac,xin,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(dac,xin,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = VecGetArray(xlocal,&zctx.v);CHKERRQ(ierr);

  /* get coordinates of nodes */
  ierr = DAGetCoordinates(da,&xcoor);CHKERRQ(ierr);
  if (!xcoor) {
    ierr = DASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
    ierr = DAGetCoordinates(da,&xcoor);CHKERRQ(ierr);
  }

  /*
      Determine the min and max  coordinates in plot 
  */
  ierr = VecStrideMin(xcoor,0,PETSC_NULL,&xmin);CHKERRQ(ierr);
  ierr = VecStrideMax(xcoor,0,PETSC_NULL,&xmax);CHKERRQ(ierr);
  ierr = VecStrideMin(xcoor,1,PETSC_NULL,&ymin);CHKERRQ(ierr);
  ierr = VecStrideMax(xcoor,1,PETSC_NULL,&ymax);CHKERRQ(ierr);
  coors[0] = xmin - .05*(xmax- xmin); coors[2] = xmax + .05*(xmax - xmin);
  coors[1] = ymin - .05*(ymax- ymin); coors[3] = ymax + .05*(ymax - ymin);
  ierr = PetscLogInfo((da,"VecView_MPI_Draw_DA2d:Preparing DA 2d contour plot coordinates %g %g %g %g\n",coors[0],coors[1],coors[2],coors[3]));CHKERRQ(ierr);

  /*
       get local ghosted version of coordinates 
  */
  ierr = PetscObjectQuery((PetscObject)da,"GraphicsCoordinateGhosted",(PetscObject*)&xcoorl);CHKERRQ(ierr);
  if (!xcoorl) {
    /* create DA to get local version of graphics */
    ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,M,N,zctx.m,zctx.n,2,1,lx,ly,&dag);CHKERRQ(ierr); 
    ierr = PetscLogInfo((dag,"VecView_MPI_Draw_DA2d:Creating auxilary DA for managing graphics coordinates ghost points\n"));CHKERRQ(ierr);
    ierr = DACreateLocalVector(dag,&xcoorl);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)da,"GraphicsCoordinateGhosted",(PetscObject)xcoorl);CHKERRQ(ierr);
    ierr = DADestroy(dag);CHKERRQ(ierr);/* dereference dag */
    ierr = PetscObjectDereference((PetscObject)xcoorl);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectQuery((PetscObject)xcoorl,"DA",(PetscObject*)&dag);CHKERRQ(ierr);
  }
  ierr = DAGlobalToLocalBegin(dag,xcoor,INSERT_VALUES,xcoorl);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(dag,xcoor,INSERT_VALUES,xcoorl);CHKERRQ(ierr);
  ierr = VecGetArray(xcoorl,&zctx.xy);CHKERRQ(ierr);
  
  /*
        Get information about size of area each processor must do graphics for
  */
  ierr = DAGetInfo(dac,0,&M,&N,0,0,0,0,&zctx.step,0,&periodic,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&igstart,&jgstart,0,&zctx.m,&zctx.n,0);CHKERRQ(ierr);
  ierr = DAGetCorners(dac,&istart,0,0,&isize,0,0);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-draw_contour_grid",&zctx.showgrid);CHKERRQ(ierr);

  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-draw_ports",&useports);CHKERRQ(ierr);
  if (useports || format == PETSC_VIEWER_DRAW_PORTS){
    ierr = PetscDrawSynchronizedClear(draw);CHKERRQ(ierr);
    ierr = PetscDrawViewPortsCreate(draw,zctx.step,&ports);CHKERRQ(ierr);
  }
  /*
     Loop over each field; drawing each in a different window
  */
  for (zctx.k=0; zctx.k<zctx.step; zctx.k++) {
    if (useports) {
      ierr = PetscDrawViewPortsSet(ports,zctx.k);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerDrawGetDraw(viewer,zctx.k,&draw);CHKERRQ(ierr);
      ierr = PetscDrawSynchronizedClear(draw);CHKERRQ(ierr);
    }

    /*
        Determine the min and max color in plot 
    */
    ierr = VecStrideMin(xin,zctx.k,PETSC_NULL,&zctx.min);CHKERRQ(ierr);
    ierr = VecStrideMax(xin,zctx.k,PETSC_NULL,&zctx.max);CHKERRQ(ierr);
    if (zctx.min == zctx.max) {
      zctx.min -= 1.e-12;
      zctx.max += 1.e-12;
    }

    if (!rank) {
      char *title;

      ierr = DAGetFieldName(da,zctx.k,&title);CHKERRQ(ierr);
      if (title) {
        ierr = PetscDrawSetTitle(draw,title);CHKERRQ(ierr);
      }
    }
    ierr = PetscDrawSetCoordinates(draw,coors[0],coors[1],coors[2],coors[3]);CHKERRQ(ierr);
    ierr = PetscLogInfo((da,"VecView_MPI_Draw_DA2d:DA 2d contour plot min %g max %g\n",zctx.min,zctx.max));CHKERRQ(ierr);

    ierr = PetscDrawGetPopup(draw,&popup);CHKERRQ(ierr);
    if (popup) {ierr = PetscDrawScalePopup(popup,zctx.min,zctx.max);CHKERRQ(ierr);}

    zctx.scale = (245.0 - PETSC_DRAW_BASIC_COLORS)/(zctx.max - zctx.min);

    ierr = PetscDrawZoom(draw,VecView_MPI_Draw_DA2d_Zoom,&zctx);CHKERRQ(ierr);
  }
  if (useports){
    ierr = PetscDrawViewPortsDestroy(ports);CHKERRQ(ierr);
  }

  ierr = VecRestoreArray(xcoorl,&zctx.xy);CHKERRQ(ierr);
  ierr = VecRestoreArray(xlocal,&zctx.v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode VecView_MPI_HDF4_Ex(Vec X, PetscViewer viewer, PetscInt d, PetscInt *dims);

#if defined(PETSC_HAVE_HDF4)
#undef __FUNCT__  
#define __FUNCT__ "VecView_MPI_HDF4_DA2d"
PetscErrorCode VecView_MPI_HDF4_DA2d(Vec xin,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscInt       dims[2];
  DA             da;
  Vec            natural;

  PetscFunctionBegin;

  ierr = PetscObjectQuery((PetscObject)xin,"DA",(PetscObject*)&da);CHKERRQ(ierr);
  if (!da) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector not generated from a DA");

  dims[0] = da->M;
  dims[1] = da->N;

  ierr = DACreateNaturalVector(da,&natural);CHKERRQ(ierr);
  ierr = DAGlobalToNaturalBegin(da,xin,INSERT_VALUES,natural);CHKERRQ(ierr);
  ierr = DAGlobalToNaturalEnd(da,xin,INSERT_VALUES,natural);CHKERRQ(ierr);
  ierr = VecView_MPI_HDF4_Ex(natural, viewer, 2, dims);CHKERRQ(ierr);
  ierr = VecDestroy(natural);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_PNETCDF)
#undef __FUNCT__  
#define __FUNCT__ "VecView_MPI_Netcdf_DA"
PetscErrorCode VecView_MPI_Netcdf_DA(Vec xin,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscInt       ncid,xstart,xdim_num=1;
  PetscInt       i,j,len,dim,m,n,p,dof,swidth,M,N,P;
  PetscInt       xin_dim,xin_id,xin_n,xin_N,xyz_dim,xyz_id,xyz_n,xyz_N;
  PetscInt       *lx,*ly,*lz;
  PetscScalar    *xarray;
  DA             da,dac;
  Vec            natural,xyz;
  DAStencilType  stencil;
  DAPeriodicType periodic;
  MPI_Comm       comm;  

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)xin,"DA",(PetscObject*)&da);CHKERRQ(ierr);
  if (!da) SETERRQ(PETSC_ERR_ARG_WRONG,,"Vector not generated from a DA");
  ierr = DAGetInfo(da,&dim,&m,&n,&p,&M,&N,&P,&dof,&swidth,&periodic,&stencil);CHKERRQ(ierr);

  /* create the appropriate DA to map the coordinates to natural ordering */
  ierr = DAGetOwnershipRange(da,&lx,&ly,&lz);CHKERRQ(ierr);
  if (dim == 1) {
    ierr = DACreate1d(comm,DA_NONPERIODIC,m,dim,0,lx,&dac);CHKERRQ(ierr); 
  } else if (dim == 2) {
    ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,m,n,M,N,dim,0,lx,ly,&dac);CHKERRQ(ierr); 
  } else if (dim == 3) {
    ierr = DACreate3d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,m,n,p,M,N,P,dim,0,lx,ly,lz,&dac);CHKERRQ(ierr); 
  } else {
    SETERRQ1(PETSC_ERR_ARG_CORRUPT,"Dimension is not 1 2 or 3: %D\n",dim);
  }
  ierr = DACreateNaturalVector(dac,&xyz);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)xyz,"coor_");CHKERRQ(ierr);
  ierr = DAGlobalToNaturalBegin(dac,da->coordinates,INSERT_VALUES,xyz);CHKERRQ(ierr);
  ierr = DAGlobalToNaturalEnd(dac,da->coordinates,INSERT_VALUES,xyz);CHKERRQ(ierr);
  /* Create the DA vector in natural ordering */
  ierr = DACreateNaturalVector(da,&natural);CHKERRQ(ierr);
  ierr = DAGlobalToNaturalBegin(da,xin,INSERT_VALUES,natural);CHKERRQ(ierr);
  ierr = DAGlobalToNaturalEnd(da,xin,INSERT_VALUES,natural);CHKERRQ(ierr);
  /* Write the netCDF dataset */
  ierr = PetscViewerNetcdfGetID(viewer,&ncid);CHKERRQ(ierr);
  if (ncid < 0) SETERRQ(PETSC_ERR_ORDER,"First call PetscViewerNetcdfOpen to create NetCDF dataset");
  /* define dimensions */
  ierr = VecGetSize(xin,&xin_N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(xin,&xin_n);CHKERRQ(ierr);
  ierr = ncmpi_def_dim(ncid,"PETSc_DA_Vector_Global_Size",xin_N,&xin_dim);CHKERRQ(ierr);
  ierr = VecGetSize(xyz,&xyz_N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(xyz,&xyz_n);CHKERRQ(ierr);
  ierr = ncmpi_def_dim(ncid,"PETSc_DA_Coordinate_Vector_Global_Size",xyz_N,&xyz_dim);CHKERRQ(ierr);
  /* define variables */
  ierr = ncmpi_def_var(ncid,"PETSc_DA_Vector",NC_DOUBLE,xdim_num,&xin_dim,&xin_id);CHKERRQ(ierr);
  ierr = ncmpi_def_var(ncid,"PETSc_DA_Coordinate_Vector",NC_DOUBLE,xdim_num,&xyz_dim,&xyz_id);CHKERRQ(ierr);
  /* leave define mode */
  ierr = ncmpi_enddef(ncid);CHKERRQ(ierr);
  /* store the vector */
  ierr = VecGetArray(xin,&xarray);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(xin,&xstart,PETSC_NULL);CHKERRQ(ierr);
  ierr = ncmpi_put_vara_double_all(ncid,xin_id,(const size_t*)&xstart,(const size_t*)&xin_n,xarray);CHKERRQ(ierr);
  ierr = VecRestoreArray(xin,&xarray);CHKERRQ(ierr);
  /* store the coordinate vector */
  ierr = VecGetArray(xyz,&xarray);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(xyz,&xstart,PETSC_NULL);CHKERRQ(ierr);
  ierr = ncmpi_put_vara_double_all(ncid,xyz_id,(const size_t*)&xstart,(const size_t*)&xyz_n,xarray);CHKERRQ(ierr);
  ierr = VecRestoreArray(xyz,&xarray);CHKERRQ(ierr);
  /* destroy the vectors and da */
  ierr = VecDestroy(natural);CHKERRQ(ierr);
  ierr = VecDestroy(xyz);CHKERRQ(ierr);
  ierr = DADestroy(dac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

EXTERN PetscErrorCode VecView_MPI_Draw_DA1d(Vec,PetscViewer);

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecView_MPI_DA"
PetscErrorCode PETSCDM_DLLEXPORT VecView_MPI_DA(Vec xin,PetscViewer viewer)
{
  DA             da;
  PetscErrorCode ierr;
  PetscInt       dim;
  Vec            natural;
  PetscTruth     isdraw;
#if defined(PETSC_HAVE_HDF4)
  PetscTruth     ishdf4;
#endif
#if defined(PETSC_HAVE_PNETCDF)
  PetscTruth     isnetcdf;
#endif
  const char     *prefix;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)xin,"DA",(PetscObject*)&da);CHKERRQ(ierr);
  if (!da) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector not generated from a DA");
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF4)
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_HDF4,&ishdf4);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PNETCDF)
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_NETCDF,&isnetcdf);CHKERRQ(ierr);
#endif
  if (isdraw) {
    ierr = DAGetInfo(da,&dim,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
    if (dim == 1) {
      ierr = VecView_MPI_Draw_DA1d(xin,viewer);CHKERRQ(ierr);
    } else if (dim == 2) {
      ierr = VecView_MPI_Draw_DA2d(xin,viewer);CHKERRQ(ierr);
    } else {
      SETERRQ1(PETSC_ERR_SUP,"Cannot graphically view vector associated with this dimensional DA %D",dim);
    }
#if defined(PETSC_HAVE_HDF4)
  } else if (ishdf4) {
    ierr = DAGetInfo(da,&dim,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
    switch (dim) {
    case 2:
      ierr = VecView_MPI_HDF4_DA2d(xin,viewer);CHKERRQ(ierr);
      break;
    default:
      SETERRQ1(PETSC_ERR_SUP,"Cannot view HDF4 vector associated with this dimensional DA %D",dim);
    }
#endif
#if defined(PETSC_HAVE_PNETCDF)
  } else if (isnetcdf) {
    ierr = VecView_MPI_Netcdf_DA(xin,viewer);CHKERRQ(ierr);
#endif
  } else {
    /* call viewer on natural ordering */
    ierr = PetscObjectGetOptionsPrefix((PetscObject)xin,&prefix);CHKERRQ(ierr);
    ierr = DACreateNaturalVector(da,&natural);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)natural,prefix);CHKERRQ(ierr);
    ierr = DAGlobalToNaturalBegin(da,xin,INSERT_VALUES,natural);CHKERRQ(ierr);
    ierr = DAGlobalToNaturalEnd(da,xin,INSERT_VALUES,natural);CHKERRQ(ierr);
    ierr = PetscObjectName((PetscObject)xin);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)natural,xin->name);CHKERRQ(ierr);
    ierr = VecView(natural,viewer);CHKERRQ(ierr);
    ierr = VecDestroy(natural);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecLoadIntoVector_Binary_DA"
PetscErrorCode PETSCDM_DLLEXPORT VecLoadIntoVector_Binary_DA(PetscViewer viewer,Vec xin)
{
  DA             da;
  PetscErrorCode ierr;
  Vec            natural;
  const char     *prefix;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)xin,"DA",(PetscObject*)&da);CHKERRQ(ierr);
  if (!da) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector not generated from a DA");
  ierr = PetscObjectGetOptionsPrefix((PetscObject)xin,&prefix);CHKERRQ(ierr);
  ierr = DACreateNaturalVector(da,&natural);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)natural,prefix);CHKERRQ(ierr);
  ierr = VecLoadIntoVector(viewer,natural);CHKERRQ(ierr);
  ierr = DANaturalToGlobalBegin(da,natural,INSERT_VALUES,xin);CHKERRQ(ierr);
  ierr = DANaturalToGlobalEnd(da,natural,INSERT_VALUES,xin);CHKERRQ(ierr);
  ierr = VecDestroy(natural);CHKERRQ(ierr);
  ierr = PetscLogInfo((xin,"VecLoadIntoVector_Binary_DA:Loading vector from natural ordering into DA\n"));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
