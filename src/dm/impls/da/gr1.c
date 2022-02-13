
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscCheckFalse(!dd->gtol,PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_WRONGSTATE,"Cannot set coordinates until after DMDA has been setup");
  ierr = DMDAGetInfo(da,&dim,&M,&N,&P,NULL,NULL,NULL,NULL,NULL,&bx,&by,&bz,NULL);CHKERRQ(ierr);
  PetscCheckFalse(xmax < xmin,PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_INCOMP,"xmax must be larger than xmin %g %g",(double)xmin,(double)xmax);
  PetscCheckFalse((dim > 1) && (ymax < ymin),PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_INCOMP,"ymax must be larger than ymin %g %g",(double)ymin,(double)ymax);
  PetscCheckFalse((dim > 2) && (zmax < zmin),PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_INCOMP,"zmax must be larger than zmin %g %g",(double)zmin,(double)zmax);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&istart,&jstart,&kstart,&isize,&jsize,&ksize);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(da, &cda);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(cda, &xcoor);CHKERRQ(ierr);
  if (dim == 1) {
    if (bx == DM_BOUNDARY_PERIODIC) hx = (xmax-xmin)/M;
    else hx = (xmax-xmin)/(M-1);
    ierr = VecGetArray(xcoor,&coors);CHKERRQ(ierr);
    for (i=0; i<isize; i++) {
      coors[i] = xmin + hx*(i+istart);
    }
    ierr = VecRestoreArray(xcoor,&coors);CHKERRQ(ierr);
  } else if (dim == 2) {
    if (bx == DM_BOUNDARY_PERIODIC) hx = (xmax-xmin)/(M);
    else hx = (xmax-xmin)/(M-1);
    if (by == DM_BOUNDARY_PERIODIC) hy = (ymax-ymin)/(N);
    else hy = (ymax-ymin)/(N-1);
    ierr = VecGetArray(xcoor,&coors);CHKERRQ(ierr);
    cnt  = 0;
    for (j=0; j<jsize; j++) {
      for (i=0; i<isize; i++) {
        coors[cnt++] = xmin + hx*(i+istart);
        coors[cnt++] = ymin + hy*(j+jstart);
      }
    }
    ierr = VecRestoreArray(xcoor,&coors);CHKERRQ(ierr);
  } else if (dim == 3) {
    if (bx == DM_BOUNDARY_PERIODIC) hx = (xmax-xmin)/(M);
    else hx = (xmax-xmin)/(M-1);
    if (by == DM_BOUNDARY_PERIODIC) hy = (ymax-ymin)/(N);
    else hy = (ymax-ymin)/(N-1);
    if (bz == DM_BOUNDARY_PERIODIC) hz_ = (zmax-zmin)/(P);
    else hz_ = (zmax-zmin)/(P-1);
    ierr = VecGetArray(xcoor,&coors);CHKERRQ(ierr);
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
    ierr = VecRestoreArray(xcoor,&coors);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Cannot create uniform coordinates for this dimension %D",dim);
  ierr = DMSetCoordinates(da,xcoor);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)da,(PetscObject)xcoor);CHKERRQ(ierr);
  ierr = VecDestroy(&xcoor);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Allows a user to select a subset of the fields to be drawn by VecView() when the vector comes from a DMDA
*/
PetscErrorCode DMDASelectFields(DM da,PetscInt *outfields,PetscInt **fields)
{
  PetscErrorCode ierr;
  PetscInt       step,ndisplayfields,*displayfields,k,j;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&step,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(step,&displayfields);CHKERRQ(ierr);
  for (k=0; k<step; k++) displayfields[k] = k;
  ndisplayfields = step;
  ierr           = PetscOptionsGetIntArray(NULL,NULL,"-draw_fields",displayfields,&ndisplayfields,&flg);CHKERRQ(ierr);
  if (!ndisplayfields) ndisplayfields = step;
  if (!flg) {
    char       **fields;
    const char *fieldname;
    PetscInt   nfields = step;
    ierr = PetscMalloc1(step,&fields);CHKERRQ(ierr);
    ierr = PetscOptionsGetStringArray(NULL,NULL,"-draw_fields_by_name",fields,&nfields,&flg);CHKERRQ(ierr);
    if (flg) {
      ndisplayfields = 0;
      for (k=0; k<nfields;k++) {
        for (j=0; j<step; j++) {
          ierr = DMDAGetFieldName(da,j,&fieldname);CHKERRQ(ierr);
          ierr = PetscStrcmp(fieldname,fields[k],&flg);CHKERRQ(ierr);
          if (flg) {
            goto found;
          }
        }
        SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_USER,"Unknown fieldname %s",fields[k]);
found:  displayfields[ndisplayfields++] = j;
      }
    }
    for (k=0; k<nfields; k++) {
      ierr = PetscFree(fields[k]);CHKERRQ(ierr);
    }
    ierr = PetscFree(fields);CHKERRQ(ierr);
  }
  *fields    = displayfields;
  *outfields = ndisplayfields;
  PetscFunctionReturn(0);
}

#include <petscdraw.h>

PetscErrorCode VecView_MPI_Draw_DA1d(Vec xin,PetscViewer v)
{
  DM                da;
  PetscErrorCode    ierr;
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
  ierr = PetscViewerDrawGetDraw(v,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = PetscViewerDrawGetBounds(v,&nbounds,&bounds);CHKERRQ(ierr);

  ierr = VecGetDM(xin,&da);CHKERRQ(ierr);
  PetscCheckFalse(!da,PetscObjectComm((PetscObject)xin),PETSC_ERR_ARG_WRONG,"Vector not generated from a DMDA");
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  ierr = PetscOptionsGetBool(NULL,NULL,"-draw_vec_use_markers",&showmarkers,NULL);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da,NULL,&N,NULL,NULL,NULL,NULL,NULL,&dof,NULL,&bx,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&istart,NULL,NULL,&isize,NULL,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xin,&array);CHKERRQ(ierr);
  ierr = VecGetLocalSize(xin,&n);CHKERRQ(ierr);
  n    = n/dof;

  /* Get coordinates of nodes */
  ierr = DMGetCoordinates(da,&xcoor);CHKERRQ(ierr);
  if (!xcoor) {
    ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,0.0,0.0,0.0);CHKERRQ(ierr);
    ierr = DMGetCoordinates(da,&xcoor);CHKERRQ(ierr);
  }
  ierr = VecGetArrayRead(xcoor,&xg);CHKERRQ(ierr);
  ierr = DMDAGetCoordinateName(da,0,&xlabel);CHKERRQ(ierr);

  /* Determine the min and max coordinate in plot */
  if (rank == 0) xmin = PetscRealPart(xg[0]);
  if (rank == size-1) xmax = PetscRealPart(xg[n-1]);
  ierr = MPI_Bcast(&xmin,1,MPIU_REAL,0,comm);CHKERRMPI(ierr);
  ierr = MPI_Bcast(&xmax,1,MPIU_REAL,size-1,comm);CHKERRMPI(ierr);

  ierr = DMDASelectFields(da,&ndisplayfields,&displayfields);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(v,&format);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-draw_ports",&useports,NULL);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_DRAW_PORTS) useports = PETSC_TRUE;
  if (useports) {
    ierr = PetscViewerDrawGetDraw(v,0,&draw);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDrawAxis(v,0,&axis);CHKERRQ(ierr);
    ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
    ierr = PetscDrawClear(draw);CHKERRQ(ierr);
    ierr = PetscDrawViewPortsCreate(draw,ndisplayfields,&ports);CHKERRQ(ierr);
  }

  /* Loop over each field; drawing each in a different window */
  for (k=0; k<ndisplayfields; k++) {
    j = displayfields[k];

    /* determine the min and max value in plot */
    ierr = VecStrideMin(xin,j,NULL,&min);CHKERRQ(ierr);
    ierr = VecStrideMax(xin,j,NULL,&max);CHKERRQ(ierr);
    if (j < nbounds) {
      min = PetscMin(min,bounds[2*j]);
      max = PetscMax(max,bounds[2*j+1]);
    }
    if (min == max) {
      min -= 1.e-5;
      max += 1.e-5;
    }

    if (useports) {
      ierr = PetscDrawViewPortsSet(ports,k);CHKERRQ(ierr);
      ierr = DMDAGetFieldName(da,j,&tlabel);CHKERRQ(ierr);
    } else {
      const char *title;
      ierr = PetscViewerDrawGetHold(v,&hold);CHKERRQ(ierr);
      ierr = PetscViewerDrawGetDraw(v,k,&draw);CHKERRQ(ierr);
      ierr = PetscViewerDrawGetDrawAxis(v,k,&axis);CHKERRQ(ierr);
      ierr = DMDAGetFieldName(da,j,&title);CHKERRQ(ierr);
      if (title) {ierr = PetscDrawSetTitle(draw,title);CHKERRQ(ierr);}
      ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
      if (!hold) {ierr = PetscDrawClear(draw);CHKERRQ(ierr);}
    }
    ierr = PetscDrawAxisSetLabels(axis,tlabel,xlabel,NULL);CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLimits(axis,xmin,xmax,min,max);CHKERRQ(ierr);
    ierr = PetscDrawAxisDraw(axis);CHKERRQ(ierr);

    /* draw local part of vector */
    ierr = PetscObjectGetNewTag((PetscObject)xin,&tag);CHKERRQ(ierr);
    if (rank < size-1) { /*send value to right */
      ierr = MPI_Send((void*)&xg[n-1],1,MPIU_REAL,rank+1,tag,comm);CHKERRMPI(ierr);
      ierr = MPI_Send((void*)&array[j+(n-1)*dof],1,MPIU_REAL,rank+1,tag,comm);CHKERRMPI(ierr);
    }
    if (rank) { /* receive value from left */
      ierr = MPI_Recv(&xgtmp,1,MPIU_REAL,rank-1,tag,comm,&status);CHKERRMPI(ierr);
      ierr = MPI_Recv(&tmp,1,MPIU_REAL,rank-1,tag,comm,&status);CHKERRMPI(ierr);
    }
    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    if (rank) {
      ierr = PetscDrawLine(draw,xgtmp,tmp,PetscRealPart(xg[0]),PetscRealPart(array[j]),PETSC_DRAW_RED);CHKERRQ(ierr);
      if (showmarkers) {ierr = PetscDrawPoint(draw,xgtmp,tmp,PETSC_DRAW_BLACK);CHKERRQ(ierr);}
    }
    for (i=1; i<n; i++) {
      ierr = PetscDrawLine(draw,PetscRealPart(xg[i-1]),PetscRealPart(array[j+dof*(i-1)]),PetscRealPart(xg[i]),PetscRealPart(array[j+dof*i]),PETSC_DRAW_RED);CHKERRQ(ierr);
      if (showmarkers) {ierr = PetscDrawMarker(draw,PetscRealPart(xg[i-1]),PetscRealPart(array[j+dof*(i-1)]),PETSC_DRAW_BLACK);CHKERRQ(ierr);}
    }
    if (rank == size-1) {
      if (showmarkers) {ierr = PetscDrawMarker(draw,PetscRealPart(xg[n-1]),PetscRealPart(array[j+dof*(n-1)]),PETSC_DRAW_BLACK);CHKERRQ(ierr);}
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
    ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawPause(draw);CHKERRQ(ierr);
    if (!useports) {ierr = PetscDrawSave(draw);CHKERRQ(ierr);}
  }
  if (useports) {
    ierr = PetscViewerDrawGetDraw(v,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  }

  ierr = PetscDrawViewPortsDestroy(ports);CHKERRQ(ierr);
  ierr = PetscFree(displayfields);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xcoor,&xg);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xin,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

