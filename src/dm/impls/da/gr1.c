
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
  CHKERRQ(DMDAGetInfo(da,&dim,&M,&N,&P,NULL,NULL,NULL,NULL,NULL,&bx,&by,&bz,NULL));
  PetscCheckFalse(xmax < xmin,PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_INCOMP,"xmax must be larger than xmin %g %g",(double)xmin,(double)xmax);
  PetscCheckFalse((dim > 1) && (ymax < ymin),PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_INCOMP,"ymax must be larger than ymin %g %g",(double)ymin,(double)ymax);
  PetscCheckFalse((dim > 2) && (zmax < zmin),PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_INCOMP,"zmax must be larger than zmin %g %g",(double)zmin,(double)zmax);
  CHKERRQ(PetscObjectGetComm((PetscObject)da,&comm));
  CHKERRQ(DMDAGetCorners(da,&istart,&jstart,&kstart,&isize,&jsize,&ksize));
  CHKERRQ(DMGetCoordinateDM(da, &cda));
  CHKERRQ(DMCreateGlobalVector(cda, &xcoor));
  if (dim == 1) {
    if (bx == DM_BOUNDARY_PERIODIC) hx = (xmax-xmin)/M;
    else hx = (xmax-xmin)/(M-1);
    CHKERRQ(VecGetArray(xcoor,&coors));
    for (i=0; i<isize; i++) {
      coors[i] = xmin + hx*(i+istart);
    }
    CHKERRQ(VecRestoreArray(xcoor,&coors));
  } else if (dim == 2) {
    if (bx == DM_BOUNDARY_PERIODIC) hx = (xmax-xmin)/(M);
    else hx = (xmax-xmin)/(M-1);
    if (by == DM_BOUNDARY_PERIODIC) hy = (ymax-ymin)/(N);
    else hy = (ymax-ymin)/(N-1);
    CHKERRQ(VecGetArray(xcoor,&coors));
    cnt  = 0;
    for (j=0; j<jsize; j++) {
      for (i=0; i<isize; i++) {
        coors[cnt++] = xmin + hx*(i+istart);
        coors[cnt++] = ymin + hy*(j+jstart);
      }
    }
    CHKERRQ(VecRestoreArray(xcoor,&coors));
  } else if (dim == 3) {
    if (bx == DM_BOUNDARY_PERIODIC) hx = (xmax-xmin)/(M);
    else hx = (xmax-xmin)/(M-1);
    if (by == DM_BOUNDARY_PERIODIC) hy = (ymax-ymin)/(N);
    else hy = (ymax-ymin)/(N-1);
    if (bz == DM_BOUNDARY_PERIODIC) hz_ = (zmax-zmin)/(P);
    else hz_ = (zmax-zmin)/(P-1);
    CHKERRQ(VecGetArray(xcoor,&coors));
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
    CHKERRQ(VecRestoreArray(xcoor,&coors));
  } else SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Cannot create uniform coordinates for this dimension %D",dim);
  CHKERRQ(DMSetCoordinates(da,xcoor));
  CHKERRQ(PetscLogObjectParent((PetscObject)da,(PetscObject)xcoor));
  CHKERRQ(VecDestroy(&xcoor));
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
  CHKERRQ(DMDAGetInfo(da,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&step,NULL,NULL,NULL,NULL,NULL));
  CHKERRQ(PetscMalloc1(step,&displayfields));
  for (k=0; k<step; k++) displayfields[k] = k;
  ndisplayfields = step;
  CHKERRQ(PetscOptionsGetIntArray(NULL,NULL,"-draw_fields",displayfields,&ndisplayfields,&flg));
  if (!ndisplayfields) ndisplayfields = step;
  if (!flg) {
    char       **fields;
    const char *fieldname;
    PetscInt   nfields = step;
    CHKERRQ(PetscMalloc1(step,&fields));
    CHKERRQ(PetscOptionsGetStringArray(NULL,NULL,"-draw_fields_by_name",fields,&nfields,&flg));
    if (flg) {
      ndisplayfields = 0;
      for (k=0; k<nfields;k++) {
        for (j=0; j<step; j++) {
          CHKERRQ(DMDAGetFieldName(da,j,&fieldname));
          CHKERRQ(PetscStrcmp(fieldname,fields[k],&flg));
          if (flg) {
            goto found;
          }
        }
        SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_USER,"Unknown fieldname %s",fields[k]);
found:  displayfields[ndisplayfields++] = j;
      }
    }
    for (k=0; k<nfields; k++) {
      CHKERRQ(PetscFree(fields[k]));
    }
    CHKERRQ(PetscFree(fields));
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
  CHKERRQ(PetscViewerDrawGetDraw(v,0,&draw));
  CHKERRQ(PetscDrawIsNull(draw,&isnull));
  if (isnull) PetscFunctionReturn(0);
  CHKERRQ(PetscViewerDrawGetBounds(v,&nbounds,&bounds));

  CHKERRQ(VecGetDM(xin,&da));
  PetscCheck(da,PetscObjectComm((PetscObject)xin),PETSC_ERR_ARG_WRONG,"Vector not generated from a DMDA");
  CHKERRQ(PetscObjectGetComm((PetscObject)xin,&comm));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-draw_vec_use_markers",&showmarkers,NULL));

  CHKERRQ(DMDAGetInfo(da,NULL,&N,NULL,NULL,NULL,NULL,NULL,&dof,NULL,&bx,NULL,NULL,NULL));
  CHKERRQ(DMDAGetCorners(da,&istart,NULL,NULL,&isize,NULL,NULL));
  CHKERRQ(VecGetArrayRead(xin,&array));
  CHKERRQ(VecGetLocalSize(xin,&n));
  n    = n/dof;

  /* Get coordinates of nodes */
  CHKERRQ(DMGetCoordinates(da,&xcoor));
  if (!xcoor) {
    CHKERRQ(DMDASetUniformCoordinates(da,0.0,1.0,0.0,0.0,0.0,0.0));
    CHKERRQ(DMGetCoordinates(da,&xcoor));
  }
  CHKERRQ(VecGetArrayRead(xcoor,&xg));
  CHKERRQ(DMDAGetCoordinateName(da,0,&xlabel));

  /* Determine the min and max coordinate in plot */
  if (rank == 0) xmin = PetscRealPart(xg[0]);
  if (rank == size-1) xmax = PetscRealPart(xg[n-1]);
  CHKERRMPI(MPI_Bcast(&xmin,1,MPIU_REAL,0,comm));
  CHKERRMPI(MPI_Bcast(&xmax,1,MPIU_REAL,size-1,comm));

  CHKERRQ(DMDASelectFields(da,&ndisplayfields,&displayfields));
  CHKERRQ(PetscViewerGetFormat(v,&format));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-draw_ports",&useports,NULL));
  if (format == PETSC_VIEWER_DRAW_PORTS) useports = PETSC_TRUE;
  if (useports) {
    CHKERRQ(PetscViewerDrawGetDraw(v,0,&draw));
    CHKERRQ(PetscViewerDrawGetDrawAxis(v,0,&axis));
    CHKERRQ(PetscDrawCheckResizedWindow(draw));
    CHKERRQ(PetscDrawClear(draw));
    CHKERRQ(PetscDrawViewPortsCreate(draw,ndisplayfields,&ports));
  }

  /* Loop over each field; drawing each in a different window */
  for (k=0; k<ndisplayfields; k++) {
    PetscErrorCode ierr;
    j = displayfields[k];

    /* determine the min and max value in plot */
    CHKERRQ(VecStrideMin(xin,j,NULL,&min));
    CHKERRQ(VecStrideMax(xin,j,NULL,&max));
    if (j < nbounds) {
      min = PetscMin(min,bounds[2*j]);
      max = PetscMax(max,bounds[2*j+1]);
    }
    if (min == max) {
      min -= 1.e-5;
      max += 1.e-5;
    }

    if (useports) {
      CHKERRQ(PetscDrawViewPortsSet(ports,k));
      CHKERRQ(DMDAGetFieldName(da,j,&tlabel));
    } else {
      const char *title;
      CHKERRQ(PetscViewerDrawGetHold(v,&hold));
      CHKERRQ(PetscViewerDrawGetDraw(v,k,&draw));
      CHKERRQ(PetscViewerDrawGetDrawAxis(v,k,&axis));
      CHKERRQ(DMDAGetFieldName(da,j,&title));
      if (title) CHKERRQ(PetscDrawSetTitle(draw,title));
      CHKERRQ(PetscDrawCheckResizedWindow(draw));
      if (!hold) CHKERRQ(PetscDrawClear(draw));
    }
    CHKERRQ(PetscDrawAxisSetLabels(axis,tlabel,xlabel,NULL));
    CHKERRQ(PetscDrawAxisSetLimits(axis,xmin,xmax,min,max));
    CHKERRQ(PetscDrawAxisDraw(axis));

    /* draw local part of vector */
    CHKERRQ(PetscObjectGetNewTag((PetscObject)xin,&tag));
    if (rank < size-1) { /*send value to right */
      CHKERRMPI(MPI_Send((void*)&xg[n-1],1,MPIU_REAL,rank+1,tag,comm));
      CHKERRMPI(MPI_Send((void*)&array[j+(n-1)*dof],1,MPIU_REAL,rank+1,tag,comm));
    }
    if (rank) { /* receive value from left */
      CHKERRMPI(MPI_Recv(&xgtmp,1,MPIU_REAL,rank-1,tag,comm,&status));
      CHKERRMPI(MPI_Recv(&tmp,1,MPIU_REAL,rank-1,tag,comm,&status));
    }
    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    if (rank) {
      CHKERRQ(PetscDrawLine(draw,xgtmp,tmp,PetscRealPart(xg[0]),PetscRealPart(array[j]),PETSC_DRAW_RED));
      if (showmarkers) CHKERRQ(PetscDrawPoint(draw,xgtmp,tmp,PETSC_DRAW_BLACK));
    }
    for (i=1; i<n; i++) {
      CHKERRQ(PetscDrawLine(draw,PetscRealPart(xg[i-1]),PetscRealPart(array[j+dof*(i-1)]),PetscRealPart(xg[i]),PetscRealPart(array[j+dof*i]),PETSC_DRAW_RED));
      if (showmarkers) CHKERRQ(PetscDrawMarker(draw,PetscRealPart(xg[i-1]),PetscRealPart(array[j+dof*(i-1)]),PETSC_DRAW_BLACK));
    }
    if (rank == size-1) {
      if (showmarkers) CHKERRQ(PetscDrawMarker(draw,PetscRealPart(xg[n-1]),PetscRealPart(array[j+dof*(n-1)]),PETSC_DRAW_BLACK));
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
    CHKERRQ(PetscDrawFlush(draw));
    CHKERRQ(PetscDrawPause(draw));
    if (!useports) CHKERRQ(PetscDrawSave(draw));
  }
  if (useports) {
    CHKERRQ(PetscViewerDrawGetDraw(v,0,&draw));
    CHKERRQ(PetscDrawSave(draw));
  }

  CHKERRQ(PetscDrawViewPortsDestroy(ports));
  CHKERRQ(PetscFree(displayfields));
  CHKERRQ(VecRestoreArrayRead(xcoor,&xg));
  CHKERRQ(VecRestoreArrayRead(xin,&array));
  PetscFunctionReturn(0);
}
