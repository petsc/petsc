
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc-private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMCreateCoordinateDM_DA"
PetscErrorCode DMCreateCoordinateDM_DA(DM dm, DM *cdm)
{
  PetscErrorCode ierr;
  DM_DA          *da = (DM_DA*) dm->data;
  PetscFunctionBegin;
  ierr = DMDAGetReducedDMDA(dm,da->dim,cdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASetFieldName"
/*@C
   DMDASetFieldName - Sets the names of individual field components in multicomponent
   vectors associated with a DMDA.

   Not Collective

   Input Parameters:
+  da - the distributed array
.  nf - field number for the DMDA (0, 1, ... dof-1), where dof indicates the
        number of degrees of freedom per node within the DMDA
-  names - the name of the field (component)

  Level: intermediate

.keywords: distributed array, get, component name

.seealso: DMDAGetFieldName(), DMDASetCoordinateName(), DMDAGetCoordinateName()
@*/
PetscErrorCode  DMDASetFieldName(DM da,PetscInt nf,const char name[])
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (nf < 0 || nf >= dd->w) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid field number: %D",nf);
  ierr = PetscFree(dd->fieldname[nf]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name,&dd->fieldname[nf]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetFieldName"
/*@C
   DMDAGetFieldName - Gets the names of individual field components in multicomponent
   vectors associated with a DMDA.

   Not Collective

   Input Parameter:
+  da - the distributed array
-  nf - field number for the DMDA (0, 1, ... dof-1), where dof indicates the
        number of degrees of freedom per node within the DMDA

   Output Parameter:
.  names - the name of the field (component)

  Level: intermediate

.keywords: distributed array, get, component name

.seealso: DMDASetFieldName(), DMDASetCoordinateName(), DMDAGetCoordinateName()
@*/
PetscErrorCode  DMDAGetFieldName(DM da,PetscInt nf,const char **name)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(name,3);
  if (nf < 0 || nf >= dd->w) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid field number: %D",nf);
  *name = dd->fieldname[nf];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASetCoordinateName"
/*@C
   DMDASetCoordinateName - Sets the name of the coordinate directions associated with a DMDA, for example "x" or "y"

   Not Collective

   Input Parameters:
+  da - the distributed array
.  nf - coordinate number for the DMDA (0, 1, ... dim-1),
-  name - the name of the coordinate

  Level: intermediate

.keywords: distributed array, get, component name

.seealso: DMDAGetCoordinateName(), DMDASetFieldName(), DMDAGetFieldName()
@*/
PetscErrorCode  DMDASetCoordinateName(DM da,PetscInt nf,const char name[])
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (nf < 0 || nf >= dd->dim) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid coordinate number: %D",nf);
  ierr = PetscFree(dd->coordinatename[nf]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name,&dd->coordinatename[nf]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetCoordinateName"
/*@C
   DMDAGetCoordinateName - Gets the name of a coodinate direction associated with a DMDA.

   Not Collective

   Input Parameter:
+  da - the distributed array
-  nf -  number for the DMDA (0, 1, ... dim-1)

   Output Parameter:
.  names - the name of the coordinate direction

  Level: intermediate

.keywords: distributed array, get, component name

.seealso: DMDASetCoordinateName(), DMDASetFieldName(), DMDAGetFieldName()
@*/
PetscErrorCode  DMDAGetCoordinateName(DM da,PetscInt nf,const char **name)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(name,3);
  if (nf < 0 || nf >= dd->dim) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid coordinate number: %D",nf);
  *name = dd->coordinatename[nf];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetCorners"
/*@
   DMDAGetCorners - Returns the global (x,y,z) indices of the lower left
   corner of the local region, excluding ghost points.

   Not Collective

   Input Parameter:
.  da - the distributed array

   Output Parameters:
+  x,y,z - the corner indices (where y and z are optional; these are used
           for 2D and 3D problems)
-  m,n,p - widths in the corresponding directions (where n and p are optional;
           these are used for 2D and 3D problems)

   Note:
   The corner information is independent of the number of degrees of
   freedom per node set with the DMDACreateXX() routine. Thus the x, y, z, and
   m, n, p can be thought of as coordinates on a logical grid, where each
   grid point has (potentially) several degrees of freedom.
   Any of y, z, n, and p can be passed in as NULL if not needed.

  Level: beginner

.keywords: distributed array, get, corners, nodes, local indices

.seealso: DMDAGetGhostCorners(), DMDAGetOwnershipRanges()
@*/
PetscErrorCode  DMDAGetCorners(DM da,PetscInt *x,PetscInt *y,PetscInt *z,PetscInt *m,PetscInt *n,PetscInt *p)
{
  PetscInt w;
  DM_DA    *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  /* since the xs, xe ... have all been multiplied by the number of degrees
     of freedom per cell, w = dd->w, we divide that out before returning.*/
  w = dd->w;
  if (x) *x = dd->xs/w + dd->xo; if (m) *m = (dd->xe - dd->xs)/w;
  /* the y and z have NOT been multiplied by w */
  if (y) *y = dd->ys + dd->yo; if (n) *n = (dd->ye - dd->ys);
  if (z) *z = dd->zs + dd->zo; if (p) *p = (dd->ze - dd->zs);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetLocalBoundingBox"
/*@
   DMDAGetLocalBoundingBox - Returns the local bounding box for the DMDA.

   Not Collective

   Input Parameter:
.  da - the distributed array

   Output Parameters:
+  lmin - local minimum coordinates (length dim, optional)
-  lmax - local maximim coordinates (length dim, optional)

  Level: beginner

.keywords: distributed array, get, coordinates

.seealso: DMDAGetCoordinateDA(), DMGetCoordinates(), DMDAGetBoundingBox()
@*/
PetscErrorCode  DMDAGetLocalBoundingBox(DM da,PetscReal lmin[],PetscReal lmax[])
{
  PetscErrorCode    ierr;
  Vec               coords = NULL;
  PetscInt          dim,i,j;
  const PetscScalar *local_coords;
  PetscReal         min[3]={PETSC_MAX_REAL,PETSC_MAX_REAL,PETSC_MAX_REAL},max[3]={PETSC_MIN_REAL,PETSC_MIN_REAL,PETSC_MIN_REAL};
  PetscInt          N,Ni;
  DM_DA             *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  dim  = dd->dim;
  ierr = DMGetCoordinates(da,&coords);CHKERRQ(ierr);
  if (coords) {
    ierr = VecGetArrayRead(coords,&local_coords);CHKERRQ(ierr);
    ierr = VecGetLocalSize(coords,&N);CHKERRQ(ierr);
    Ni   = N/dim;
    for (i=0; i<Ni; i++) {
      for (j=0; j<3; j++) {
        min[j] = j < dim ? PetscMin(min[j],PetscRealPart(local_coords[i*dim+j])) : 0;
        max[j] = j < dim ? PetscMax(max[j],PetscRealPart(local_coords[i*dim+j])) : 0;
      }
    }
    ierr = VecRestoreArrayRead(coords,&local_coords);CHKERRQ(ierr);
  } else {                      /* Just use grid indices */
    DMDALocalInfo info;
    ierr   = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
    min[0] = info.xs;
    min[1] = info.ys;
    min[2] = info.zs;
    max[0] = info.xs + info.xm-1;
    max[1] = info.ys + info.ym-1;
    max[2] = info.zs + info.zm-1;
  }
  if (lmin) {ierr = PetscMemcpy(lmin,min,dim*sizeof(PetscReal));CHKERRQ(ierr);}
  if (lmax) {ierr = PetscMemcpy(lmax,max,dim*sizeof(PetscReal));CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetBoundingBox"
/*@
   DMDAGetBoundingBox - Returns the global bounding box for the DMDA.

   Collective on DMDA

   Input Parameter:
.  da - the distributed array

   Output Parameters:
+  gmin - global minimum coordinates (length dim, optional)
-  gmax - global maximim coordinates (length dim, optional)

  Level: beginner

.keywords: distributed array, get, coordinates

.seealso: DMDAGetCoordinateDA(), DMGetCoordinates(), DMDAGetLocalBoundingBox()
@*/
PetscErrorCode  DMDAGetBoundingBox(DM da,PetscReal gmin[],PetscReal gmax[])
{
  PetscErrorCode ierr;
  PetscMPIInt    count;
  PetscReal      lmin[3],lmax[3];
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  ierr = PetscMPIIntCast(dd->dim,&count);CHKERRQ(ierr);
  ierr = DMDAGetLocalBoundingBox(da,lmin,lmax);CHKERRQ(ierr);
  if (gmin) {ierr = MPI_Allreduce(lmin,gmin,count,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)da));CHKERRQ(ierr);}
  if (gmax) {ierr = MPI_Allreduce(lmax,gmax,count,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)da));CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetReducedDMDA"
/*@
   DMDAGetReducedDMDA - Gets the DMDA with the same layout but with fewer or more fields

   Collective on DMDA

   Input Parameter:
+  da - the distributed array
.  nfields - number of fields in new DMDA

   Output Parameter:
.  nda - the new DMDA

  Level: intermediate

.keywords: distributed array, get, corners, nodes, local indices, coordinates

.seealso: DMDAGetGhostCorners(), DMSetCoordinates(), DMDASetUniformCoordinates(), DMGetCoordinates(), DMDAGetGhostedCoordinates()
@*/
PetscErrorCode  DMDAGetReducedDMDA(DM da,PetscInt nfields,DM *nda)
{
  PetscErrorCode   ierr;
  DM_DA            *dd = (DM_DA*)da->data;
  PetscInt         s,m,n,p,M,N,P,dim,Mo,No,Po;
  const PetscInt   *lx,*ly,*lz;
  DMDABoundaryType bx,by,bz;
  DMDAStencilType  stencil_type;
  PetscInt         ox,oy,oz;
  PetscInt         cl,rl;

  PetscFunctionBegin;
  dim = dd->dim;
  M   = dd->M;
  N   = dd->N;
  P   = dd->P;
  m   = dd->m;
  n   = dd->n;
  p   = dd->p;
  s   = dd->s;
  bx  = dd->bx;
  by  = dd->by;
  bz  = dd->bz;

  stencil_type = dd->stencil_type;

  ierr = DMDAGetOwnershipRanges(da,&lx,&ly,&lz);CHKERRQ(ierr);
  if (dim == 1) {
    ierr = DMDACreate1d(PetscObjectComm((PetscObject)da),bx,M,nfields,s,dd->lx,nda);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = DMDACreate2d(PetscObjectComm((PetscObject)da),bx,by,stencil_type,M,N,m,n,nfields,s,lx,ly,nda);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = DMDACreate3d(PetscObjectComm((PetscObject)da),bx,by,bz,stencil_type,M,N,P,m,n,p,nfields,s,lx,ly,lz,nda);CHKERRQ(ierr);
  }
  if (da->coordinates) {
    ierr = PetscObjectReference((PetscObject)da->coordinates);CHKERRQ(ierr);

    (*nda)->coordinates = da->coordinates;
  }

  /* allow for getting a reduced DA corresponding to a domain decomposition */
  ierr = DMDAGetOffset(da,&ox,&oy,&oz,&Mo,&No,&Po);CHKERRQ(ierr);
  ierr = DMDASetOffset(*nda,ox,oy,oz,Mo,No,Po);CHKERRQ(ierr);

  /* allow for getting a reduced DA corresponding to a coarsened DA */
  ierr = DMGetCoarsenLevel(da,&cl);CHKERRQ(ierr);
  ierr = DMGetRefineLevel(da,&rl);CHKERRQ(ierr);

  (*nda)->levelup   = rl;
  (*nda)->leveldown = cl;
  PetscFunctionReturn(0);
}

