
/*
   Plots vectors obtained with DMDACreate1d()
*/

#include <petsc/private/dmdaimpl.h>      /*I  "petscdmda.h"   I*/

/*@
    DMDASetUniformCoordinates - Sets a DMDA coordinates to be a uniform grid

  Collective on da

  Input Parameters:
+  da - the distributed array object
.  xmin,xmax - extremes in the x direction
.  ymin,ymax - extremes in the y direction (value ignored for 1 dimensional problems)
-  zmin,zmax - extremes in the z direction (value ignored for 1 or 2 dimensional problems)

  Level: beginner

.seealso: DMSetCoordinates(), DMGetCoordinates(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMStagSetUniformCoordinates()

@*/
PetscErrorCode  DMDASetUniformCoordinates(DM da,PetscReal xmin,PetscReal xmax,PetscReal ymin,PetscReal ymax,PetscReal zmin,PetscReal zmax)
{
  MPI_Comm         comm;
  DM               cda;
  DM_DA            *dd = (DM_DA*)da->data;
  DMBoundaryType   bx,by,bz;
  Vec              xcoor;
  PetscScalar      *coors;
  PetscReal        hx,hy,hz_;
  PetscInt         i,j,k,M,N,P,istart,isize,jstart,jsize,kstart,ksize,dim,cnt;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscCheck(dd->gtol,PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_WRONGSTATE,"Cannot set coordinates until after DMDA has been setup");
  PetscCall(DMDAGetInfo(da,&dim,&M,&N,&P,NULL,NULL,NULL,NULL,NULL,&bx,&by,&bz,NULL));
  PetscCheckFalse(xmax < xmin,PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_INCOMP,"xmax must be larger than xmin %g %g",(double)xmin,(double)xmax);
  PetscCheckFalse((dim > 1) && (ymax < ymin),PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_INCOMP,"ymax must be larger than ymin %g %g",(double)ymin,(double)ymax);
  PetscCheckFalse((dim > 2) && (zmax < zmin),PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_INCOMP,"zmax must be larger than zmin %g %g",(double)zmin,(double)zmax);
  PetscCall(PetscObjectGetComm((PetscObject)da,&comm));
  PetscCall(DMDAGetCorners(da,&istart,&jstart,&kstart,&isize,&jsize,&ksize));
  PetscCall(DMGetCoordinateDM(da, &cda));
  PetscCall(DMCreateGlobalVector(cda, &xcoor));
  if (dim == 1) {
    if (bx == DM_BOUNDARY_PERIODIC) hx = (xmax-xmin)/M;
    else hx = (xmax-xmin)/(M-1);
    PetscCall(VecGetArray(xcoor,&coors));
    for (i=0; i<isize; i++) {
      coors[i] = xmin + hx*(i+istart);
    }
    PetscCall(VecRestoreArray(xcoor,&coors));
  } else if (dim == 2) {
    if (bx == DM_BOUNDARY_PERIODIC) hx = (xmax-xmin)/(M);
    else hx = (xmax-xmin)/(M-1);
    if (by == DM_BOUNDARY_PERIODIC) hy = (ymax-ymin)/(N);
    else hy = (ymax-ymin)/(N-1);
    PetscCall(VecGetArray(xcoor,&coors));
    cnt  = 0;
    for (j=0; j<jsize; j++) {
      for (i=0; i<isize; i++) {
        coors[cnt++] = xmin + hx*(i+istart);
        coors[cnt++] = ymin + hy*(j+jstart);
      }
    }
    PetscCall(VecRestoreArray(xcoor,&coors));
  } else if (dim == 3) {
    if (bx == DM_BOUNDARY_PERIODIC) hx = (xmax-xmin)/(M);
    else hx = (xmax-xmin)/(M-1);
    if (by == DM_BOUNDARY_PERIODIC) hy = (ymax-ymin)/(N);
    else hy = (ymax-ymin)/(N-1);
    if (bz == DM_BOUNDARY_PERIODIC) hz_ = (zmax-zmin)/(P);
    else hz_ = (zmax-zmin)/(P-1);
    PetscCall(VecGetArray(xcoor,&coors));
    cnt  = 0;
    for (k=0; k<ksize; k++) {
      for (j=0; j<jsize; j++) {
        for (i=0; i<isize; i++) {
          coors[cnt++] = xmin + hx*(i+istart);
          coors[cnt++] = ymin + hy*(j+jstart);
          coors[cnt++] = zmin + hz_*(k+kstart);
        }
      }
    }
    PetscCall(VecRestoreArray(xcoor,&coors));
  } else SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Cannot create uniform coordinates for this dimension %D",dim);
  PetscCall(DMSetCoordinates(da,xcoor));
  PetscCall(PetscLogObjectParent((PetscObject)da,(PetscObject)xcoor));
  PetscCall(VecDestroy(&xcoor));
  PetscFunctionReturn(0);
}

/*
    Allows a user to select a subset of the fields to be drawn by VecView() when the vector comes from a DMDA
*/
PetscErrorCode DMDASelectFields(DM da,PetscInt *outfields,PetscInt **fields)
{
  PetscInt       step,ndisplayfields,*displayfields,k,j;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(DMDAGetInfo(da,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&step,NULL,NULL,NULL,NULL,NULL));
  PetscCall(PetscMalloc1(step,&displayfields));
  for (k=0; k<step; k++) displayfields[k] = k;
  ndisplayfields = step;
  PetscCall(PetscOptionsGetIntArray(NULL,NULL,"-draw_fields",displayfields,&ndisplayfields,&flg));
  if (!ndisplayfields) ndisplayfields = step;
  if (!flg) {
    char       **fields;
    const char *fieldname;
    PetscInt   nfields = step;
    PetscCall(PetscMalloc1(step,&fields));
    PetscCall(PetscOptionsGetStringArray(NULL,NULL,"-draw_fields_by_name",fields,&nfields,&flg));
    if (flg) {
      ndisplayfields = 0;
      for (k=0; k<nfields;k++) {
        for (j=0; j<step; j++) {
          PetscCall(DMDAGetFieldName(da,j,&fieldname));
          PetscCall(PetscStrcmp(fieldname,fields[k],&flg));
          if (flg) {
            goto found;
          }
        }
        SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_USER,"Unknown fieldname %s",fields[k]);
found:  displayfields[ndisplayfields++] = j;
      }
    }
    for (k=0; k<nfields; k++) {
      PetscCall(PetscFree(fields[k]));
    }
    PetscCall(PetscFree(fields));
  }
  *fields    = displayfields;
  *outfields = ndisplayfields;
  PetscFunctionReturn(0);
}

#include <petscdraw.h>

PetscErrorCode VecView_MPI_Draw_DA1d(Vec xin,PetscViewer v)
{
  DM                da;
  PetscMPIInt       rank,size,tag;
  PetscInt          i,n,N,dof,istart,isize,j,nbounds;
  MPI_Status        status;
  PetscReal         min,max,xmin = 0.0,xmax = 0.0,tmp = 0.0,xgtmp = 0.0;
  const PetscScalar *array,*xg;
  PetscDraw         draw;
  PetscBool         isnull,useports = PETSC_FALSE,showmarkers = PETSC_FALSE;
  MPI_Comm          comm;
  PetscDrawAxis     axis;
  Vec               xcoor;
  DMBoundaryType    bx;
  const char        *tlabel = NULL,*xlabel = NULL;
  const PetscReal   *bounds;
  PetscInt          *displayfields;
  PetscInt          k,ndisplayfields;
  PetscBool         hold;
  PetscDrawViewPorts *ports = NULL;
  PetscViewerFormat  format;

  PetscFunctionBegin;
  PetscCall(PetscViewerDrawGetDraw(v,0,&draw));
  PetscCall(PetscDrawIsNull(draw,&isnull));
  if (isnull) PetscFunctionReturn(0);
  PetscCall(PetscViewerDrawGetBounds(v,&nbounds,&bounds));

  PetscCall(VecGetDM(xin,&da));
  PetscCheck(da,PetscObjectComm((PetscObject)xin),PETSC_ERR_ARG_WRONG,"Vector not generated from a DMDA");
  PetscCall(PetscObjectGetComm((PetscObject)xin,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-draw_vec_use_markers",&showmarkers,NULL));

  PetscCall(DMDAGetInfo(da,NULL,&N,NULL,NULL,NULL,NULL,NULL,&dof,NULL,&bx,NULL,NULL,NULL));
  PetscCall(DMDAGetCorners(da,&istart,NULL,NULL,&isize,NULL,NULL));
  PetscCall(VecGetArrayRead(xin,&array));
  PetscCall(VecGetLocalSize(xin,&n));
  n    = n/dof;

  /* Get coordinates of nodes */
  PetscCall(DMGetCoordinates(da,&xcoor));
  if (!xcoor) {
    PetscCall(DMDASetUniformCoordinates(da,0.0,1.0,0.0,0.0,0.0,0.0));
    PetscCall(DMGetCoordinates(da,&xcoor));
  }
  PetscCall(VecGetArrayRead(xcoor,&xg));
  PetscCall(DMDAGetCoordinateName(da,0,&xlabel));

  /* Determine the min and max coordinate in plot */
  if (rank == 0) xmin = PetscRealPart(xg[0]);
  if (rank == size-1) xmax = PetscRealPart(xg[n-1]);
  PetscCallMPI(MPI_Bcast(&xmin,1,MPIU_REAL,0,comm));
  PetscCallMPI(MPI_Bcast(&xmax,1,MPIU_REAL,size-1,comm));

  PetscCall(DMDASelectFields(da,&ndisplayfields,&displayfields));
  PetscCall(PetscViewerGetFormat(v,&format));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-draw_ports",&useports,NULL));
  if (format == PETSC_VIEWER_DRAW_PORTS) useports = PETSC_TRUE;
  if (useports) {
    PetscCall(PetscViewerDrawGetDraw(v,0,&draw));
    PetscCall(PetscViewerDrawGetDrawAxis(v,0,&axis));
    PetscCall(PetscDrawCheckResizedWindow(draw));
    PetscCall(PetscDrawClear(draw));
    PetscCall(PetscDrawViewPortsCreate(draw,ndisplayfields,&ports));
  }

  /* Loop over each field; drawing each in a different window */
  for (k=0; k<ndisplayfields; k++) {
    PetscErrorCode ierr;
    j = displayfields[k];

    /* determine the min and max value in plot */
    PetscCall(VecStrideMin(xin,j,NULL,&min));
    PetscCall(VecStrideMax(xin,j,NULL,&max));
    if (j < nbounds) {
      min = PetscMin(min,bounds[2*j]);
      max = PetscMax(max,bounds[2*j+1]);
    }
    if (min == max) {
      min -= 1.e-5;
      max += 1.e-5;
    }

    if (useports) {
      PetscCall(PetscDrawViewPortsSet(ports,k));
      PetscCall(DMDAGetFieldName(da,j,&tlabel));
    } else {
      const char *title;
      PetscCall(PetscViewerDrawGetHold(v,&hold));
      PetscCall(PetscViewerDrawGetDraw(v,k,&draw));
      PetscCall(PetscViewerDrawGetDrawAxis(v,k,&axis));
      PetscCall(DMDAGetFieldName(da,j,&title));
      if (title) PetscCall(PetscDrawSetTitle(draw,title));
      PetscCall(PetscDrawCheckResizedWindow(draw));
      if (!hold) PetscCall(PetscDrawClear(draw));
    }
    PetscCall(PetscDrawAxisSetLabels(axis,tlabel,xlabel,NULL));
    PetscCall(PetscDrawAxisSetLimits(axis,xmin,xmax,min,max));
    PetscCall(PetscDrawAxisDraw(axis));

    /* draw local part of vector */
    PetscCall(PetscObjectGetNewTag((PetscObject)xin,&tag));
    if (rank < size-1) { /*send value to right */
      PetscCallMPI(MPI_Send((void*)&xg[n-1],1,MPIU_REAL,rank+1,tag,comm));
      PetscCallMPI(MPI_Send((void*)&array[j+(n-1)*dof],1,MPIU_REAL,rank+1,tag,comm));
    }
    if (rank) { /* receive value from left */
      PetscCallMPI(MPI_Recv(&xgtmp,1,MPIU_REAL,rank-1,tag,comm,&status));
      PetscCallMPI(MPI_Recv(&tmp,1,MPIU_REAL,rank-1,tag,comm,&status));
    }
    ierr = PetscDrawCollectiveBegin(draw);PetscCall(ierr);
    if (rank) {
      PetscCall(PetscDrawLine(draw,xgtmp,tmp,PetscRealPart(xg[0]),PetscRealPart(array[j]),PETSC_DRAW_RED));
      if (showmarkers) PetscCall(PetscDrawPoint(draw,xgtmp,tmp,PETSC_DRAW_BLACK));
    }
    for (i=1; i<n; i++) {
      PetscCall(PetscDrawLine(draw,PetscRealPart(xg[i-1]),PetscRealPart(array[j+dof*(i-1)]),PetscRealPart(xg[i]),PetscRealPart(array[j+dof*i]),PETSC_DRAW_RED));
      if (showmarkers) PetscCall(PetscDrawMarker(draw,PetscRealPart(xg[i-1]),PetscRealPart(array[j+dof*(i-1)]),PETSC_DRAW_BLACK));
    }
    if (rank == size-1) {
      if (showmarkers) PetscCall(PetscDrawMarker(draw,PetscRealPart(xg[n-1]),PetscRealPart(array[j+dof*(n-1)]),PETSC_DRAW_BLACK));
    }
    ierr = PetscDrawCollectiveEnd(draw);PetscCall(ierr);
    PetscCall(PetscDrawFlush(draw));
    PetscCall(PetscDrawPause(draw));
    if (!useports) PetscCall(PetscDrawSave(draw));
  }
  if (useports) {
    PetscCall(PetscViewerDrawGetDraw(v,0,&draw));
    PetscCall(PetscDrawSave(draw));
  }

  PetscCall(PetscDrawViewPortsDestroy(ports));
  PetscCall(PetscFree(displayfields));
  PetscCall(VecRestoreArrayRead(xcoor,&xg));
  PetscCall(VecRestoreArrayRead(xin,&array));
  PetscFunctionReturn(0);
}
