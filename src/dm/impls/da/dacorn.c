
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc-private/daimpl.h>    /*I   "petscdmda.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMCreateCoordinateDM_DA"
PetscErrorCode DMCreateCoordinateDM_DA(DM dm, DM *cdm)
{
  DM_DA         *da = (DM_DA *) dm->data;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject) dm)->comm, &size);CHKERRQ(ierr);
  if (da->dim == 1) {
    PetscInt         s,m,*lc,l;
    DMDABoundaryType bx;

    ierr = DMDAGetInfo(dm,0,&m,0,0,0,0,0,0,&s,&bx,0,0,0);CHKERRQ(ierr);
    ierr = DMDAGetCorners(dm,0,0,0,&l,0,0);CHKERRQ(ierr);
    ierr = PetscMalloc(size * sizeof(PetscInt), &lc);CHKERRQ(ierr);
    ierr = MPI_Allgather(&l,1,MPIU_INT,lc,1,MPIU_INT,((PetscObject)dm)->comm);CHKERRQ(ierr);
    ierr = DMDACreate1d(((PetscObject)dm)->comm,bx,m,1,s,lc,cdm);CHKERRQ(ierr);
    ierr = PetscFree(lc);CHKERRQ(ierr);
  } else if (da->dim == 2) {
    PetscInt         i,s,m,*lc,*ld,l,k,n,M,N;
    DMDABoundaryType bx,by;

    ierr = DMDAGetInfo(dm,0,&m,&n,0,&M,&N,0,0,&s,&bx,&by,0,0);CHKERRQ(ierr);
    ierr = DMDAGetCorners(dm,0,0,0,&l,&k,0);CHKERRQ(ierr);
    ierr = PetscMalloc2(size,PetscInt,&lc,size,PetscInt,&ld);CHKERRQ(ierr);
    /* only first M values in lc matter */
    ierr = MPI_Allgather(&l,1,MPIU_INT,lc,1,MPIU_INT,((PetscObject)dm)->comm);CHKERRQ(ierr);
    /* every Mth value in ld matters */
      ierr = MPI_Allgather(&k,1,MPIU_INT,ld,1,MPIU_INT,((PetscObject)dm)->comm);CHKERRQ(ierr);
      for(i = 0; i < N; ++i) {
        ld[i] = ld[M*i];
      }
      ierr = DMDACreate2d(((PetscObject)dm)->comm,bx,by,DMDA_STENCIL_BOX,m,n,M,N,2,s,lc,ld,cdm);CHKERRQ(ierr);
      ierr = PetscFree2(lc,ld);CHKERRQ(ierr);
  } else if (da->dim == 3) {
    PetscInt         i,s,m,*lc,*ld,*le,l,k,q,n,M,N,P,p;
    DMDABoundaryType bx,by,bz;

    ierr = DMDAGetInfo(dm,0,&m,&n,&p,&M,&N,&P,0,&s,&bx,&by,&bz,0);CHKERRQ(ierr);
    ierr = DMDAGetCorners(dm,0,0,0,&l,&k,&q);CHKERRQ(ierr);
    ierr = PetscMalloc3(size,PetscInt,&lc,size,PetscInt,&ld,size,PetscInt,&le);CHKERRQ(ierr);
    /* only first M values in lc matter */
    ierr = MPI_Allgather(&l,1,MPIU_INT,lc,1,MPIU_INT,((PetscObject)dm)->comm);CHKERRQ(ierr);
    /* every Mth value in ld matters */
    ierr = MPI_Allgather(&k,1,MPIU_INT,ld,1,MPIU_INT,((PetscObject)dm)->comm);CHKERRQ(ierr);
    for(i = 0; i < N; ++i) {
      ld[i] = ld[M*i];
    }
    ierr = MPI_Allgather(&q,1,MPIU_INT,le,1,MPIU_INT,((PetscObject)dm)->comm);CHKERRQ(ierr);
    for(i = 0; i < P; ++i) {
      le[i] = le[M*N*i];
    }
    ierr = DMDACreate3d(((PetscObject)dm)->comm,bx,by,bz,DMDA_STENCIL_BOX,m,n,p,M,N,P,3,s,lc,ld,le,cdm);CHKERRQ(ierr);
    ierr = PetscFree3(lc,ld,le);CHKERRQ(ierr);
  }
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

.seealso: DMDAGetFieldName()
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

.seealso: DMDASetFieldName()
@*/
PetscErrorCode  DMDAGetFieldName(DM da,PetscInt nf,const char **name)
{
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(name,3);
  if (nf < 0 || nf >= dd->w) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid field number: %D",nf);
  *name = dd->fieldname[nf];
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
   Any of y, z, n, and p can be passed in as PETSC_NULL if not needed.

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
  if (x) *x = dd->xs/w; if (m) *m = (dd->xe - dd->xs)/w;
  /* the y and z have NOT been multiplied by w */
  if (y) *y = dd->ys;   if (n) *n = (dd->ye - dd->ys);
  if (z) *z = dd->zs;   if (p) *p = (dd->ze - dd->zs);
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
  Vec               coords  = PETSC_NULL;
  PetscInt          dim,i,j;
  const PetscScalar *local_coords;
  PetscReal         min[3]={PETSC_MAX_REAL,PETSC_MAX_REAL,PETSC_MAX_REAL},max[3]={PETSC_MIN_REAL,PETSC_MIN_REAL,PETSC_MIN_REAL};
  PetscInt          N,Ni;
  DM_DA             *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  dim = dd->dim;
  ierr = DMGetCoordinates(da,&coords);CHKERRQ(ierr);
  if (coords) {
    ierr = VecGetArrayRead(coords,&local_coords);CHKERRQ(ierr);
    ierr = VecGetLocalSize(coords,&N);CHKERRQ(ierr);
    Ni = N/dim;
    for (i=0; i<Ni; i++) {
      for (j=0; j<3; j++) {
        min[j] = j < dim ? PetscMin(min[j],PetscRealPart(local_coords[i*dim+j])) : 0;
        max[j] = j < dim ? PetscMax(min[j],PetscRealPart(local_coords[i*dim+j])) : 0;
      }
    }
    ierr = VecRestoreArrayRead(coords,&local_coords);CHKERRQ(ierr);
  } else {                      /* Just use grid indices */
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
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
  count = PetscMPIIntCast(dd->dim);
  ierr = DMDAGetLocalBoundingBox(da,lmin,lmax);CHKERRQ(ierr);
  if (gmin) {ierr = MPI_Allreduce(lmin,gmin,count,MPIU_REAL,MPIU_MIN,((PetscObject)da)->comm);CHKERRQ(ierr);}
  if (gmax) {ierr = MPI_Allreduce(lmax,gmax,count,MPIU_REAL,MPIU_MAX,((PetscObject)da)->comm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetReducedDA"
/*@
   DMDAGetReducedDA - Gets the DMDA with the same layout but with fewer or more fields

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
PetscErrorCode  DMDAGetReducedDA(DM da,PetscInt nfields,DM *nda)
{
  PetscErrorCode   ierr;
  DM_DA            *dd = (DM_DA*)da->data;
  PetscInt         s,m,n,p,M,N,P,dim;
  const PetscInt   *lx,*ly,*lz;
  DMDABoundaryType bx,by,bz;
  DMDAStencilType  stencil_type;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,&dim,&M,&N,&P,&m,&n,&p,0,&s,&bx,&by,&bz,&stencil_type);CHKERRQ(ierr);
  ierr = DMDAGetOwnershipRanges(da,&lx,&ly,&lz);CHKERRQ(ierr);
  if (dim == 1) {
    ierr = DMDACreate1d(((PetscObject)da)->comm,bx,M,nfields,s,dd->lx,nda);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = DMDACreate2d(((PetscObject)da)->comm,bx,by,stencil_type,M,N,m,n,nfields,s,lx,ly,nda);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = DMDACreate3d(((PetscObject)da)->comm,bx,by,bz,stencil_type,M,N,P,m,n,p,nfields,s,lx,ly,lz,nda);CHKERRQ(ierr);
  }
  if (da->coordinates) {
    ierr        = PetscObjectReference((PetscObject)da->coordinates);CHKERRQ(ierr);
    (*nda)->coordinates = da->coordinates;
  }
  PetscFunctionReturn(0);
}

