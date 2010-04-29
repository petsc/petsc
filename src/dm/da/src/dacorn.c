#define PETSCDM_DLL

/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "private/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DASetCoordinates"
/*@
   DASetCoordinates - Sets into the DA a vector that indicates the 
      coordinates of the local nodes (NOT including ghost nodes).

   Collective on DA

   Input Parameter:
+  da - the distributed array
-  c - coordinate vector

   Note:
    The coordinates should NOT include those for all ghost points

  Level: intermediate

.keywords: distributed array, get, corners, nodes, local indices, coordinates

.seealso: DAGetGhostCorners(), DAGetCoordinates(), DASetUniformCoordinates(). DAGetGhostCoordinates(), DAGetCoordinateDA()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetCoordinates(DA da,Vec c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidHeaderSpecific(c,VEC_COOKIE,2);
  ierr = PetscObjectReference((PetscObject)c);CHKERRQ(ierr);
  if (da->coordinates) {ierr = VecDestroy(da->coordinates);CHKERRQ(ierr);}
  da->coordinates = c;
  ierr = VecSetBlockSize(c,da->dim);CHKERRQ(ierr);
  if (da->ghosted_coordinates) { /* The ghosted coordinates are no longer valid */
    ierr = VecDestroy(da->ghosted_coordinates);CHKERRQ(ierr);
    da->ghosted_coordinates = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetCoordinates"
/*@
   DAGetCoordinates - Gets the node coordinates associated with a DA.

   Collective on DA

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  c - coordinate vector

   Note:
    Each process has only the coordinates for its local nodes (does NOT have the
  coordinates for the ghost nodes).

    For two and three dimensions coordinates are interlaced (x_0,y_0,x_1,y_1,...)
    and (x_0,y_0,z_0,x_1,y_1,z_1...)

    The user is responsible for destroying this vector.

  Level: intermediate

.keywords: distributed array, get, corners, nodes, local indices, coordinates

.seealso: DAGetGhostCorners(), DASetCoordinates(), DASetUniformCoordinates(), DAGetGhostedCoordinates(), DAGetCoordinateDA()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetCoordinates(DA da,Vec *c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidPointer(c,2);
  if (da->coordinates) {ierr = PetscObjectReference((PetscObject) da->coordinates);CHKERRQ(ierr);}
  *c = da->coordinates;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetCoordinateDA"
/*@
   DAGetCoordinateDA - Gets the DA that scatters between global and local DA coordinates

   Collective on DA

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  dac - coordinate DA

  Note: The user is responsible for destroying this DA when finished

  Level: intermediate

.keywords: distributed array, get, corners, nodes, local indices, coordinates

.seealso: DAGetGhostCorners(), DASetCoordinates(), DASetUniformCoordinates(), DAGetCoordinates(), DAGetGhostedCoordinates()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetCoordinateDA(DA da,DA *cda)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!da->da_coordinates) {
    ierr = MPI_Comm_size(((PetscObject)da)->comm,&size);CHKERRQ(ierr);
    if (da->dim == 1) {
      PetscInt            s,m,*lc,l;
      DAPeriodicType pt;
      ierr = DAGetInfo(da,0,&m,0,0,0,0,0,0,&s,&pt,0);CHKERRQ(ierr);
      ierr = DAGetCorners(da,0,0,0,&l,0,0);CHKERRQ(ierr);
      ierr = PetscMalloc(size*sizeof(PetscInt),&lc);CHKERRQ(ierr);
      ierr = MPI_Allgather(&l,1,MPIU_INT,lc,1,MPIU_INT,((PetscObject)da)->comm);CHKERRQ(ierr);
      ierr = DACreate1d(((PetscObject)da)->comm,pt,m,1,s,lc,&da->da_coordinates);CHKERRQ(ierr);
      ierr = PetscFree(lc);CHKERRQ(ierr);
    } else if (da->dim == 2) {
      PetscInt            i,s,m,*lc,*ld,l,k,n,M,N;
      DAPeriodicType pt;
      ierr = DAGetInfo(da,0,&m,&n,0,&M,&N,0,0,&s,&pt,0);CHKERRQ(ierr);
      ierr = DAGetCorners(da,0,0,0,&l,&k,0);CHKERRQ(ierr);
      ierr = PetscMalloc2(size,PetscInt,&lc,size,PetscInt,&ld);CHKERRQ(ierr);
      /* only first M values in lc matter */
      ierr = MPI_Allgather(&l,1,MPIU_INT,lc,1,MPIU_INT,((PetscObject)da)->comm);CHKERRQ(ierr);
      /* every Mth value in ld matters */
      ierr = MPI_Allgather(&k,1,MPIU_INT,ld,1,MPIU_INT,((PetscObject)da)->comm);CHKERRQ(ierr);
      for ( i=0; i<N; i++) {
        ld[i] = ld[M*i];
      }
      ierr = DACreate2d(((PetscObject)da)->comm,pt,DA_STENCIL_BOX,m,n,M,N,2,s,lc,ld,&da->da_coordinates);CHKERRQ(ierr);
      ierr = PetscFree2(lc,ld);CHKERRQ(ierr);
    } else if (da->dim == 3) {
      PetscInt            i,s,m,*lc,*ld,*le,l,k,q,n,M,N,P,p;
      DAPeriodicType pt;
      ierr = DAGetInfo(da,0,&m,&n,&p,&M,&N,&P,0,&s,&pt,0);CHKERRQ(ierr);
      ierr = DAGetCorners(da,0,0,0,&l,&k,&q);CHKERRQ(ierr);
      ierr = PetscMalloc3(size,PetscInt,&lc,size,PetscInt,&ld,size,PetscInt,&le);CHKERRQ(ierr);
      /* only first M values in lc matter */
      ierr = MPI_Allgather(&l,1,MPIU_INT,lc,1,MPIU_INT,((PetscObject)da)->comm);CHKERRQ(ierr);
      /* every Mth value in ld matters */
      ierr = MPI_Allgather(&k,1,MPIU_INT,ld,1,MPIU_INT,((PetscObject)da)->comm);CHKERRQ(ierr);
      for ( i=0; i<N; i++) {
        ld[i] = ld[M*i];
      }
      ierr = MPI_Allgather(&q,1,MPIU_INT,le,1,MPIU_INT,((PetscObject)da)->comm);CHKERRQ(ierr);
      for ( i=0; i<P; i++) {
        le[i] = le[M*N*i];
      }
      ierr = DACreate3d(((PetscObject)da)->comm,pt,DA_STENCIL_BOX,m,n,p,M,N,P,3,s,lc,ld,le,&da->da_coordinates);CHKERRQ(ierr);
      ierr = PetscFree3(lc,ld,le);CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectReference((PetscObject) da->da_coordinates);CHKERRQ(ierr);
  *cda = da->da_coordinates;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DAGetGhostedCoordinates"
/*@
   DAGetGhostedCoordinates - Gets the node coordinates associated with a DA.

   Collective on DA

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  c - coordinate vector

   Note:
    Each process has only the coordinates for its local AND ghost nodes

    For two and three dimensions coordinates are interlaced (x_0,y_0,x_1,y_1,...)
    and (x_0,y_0,z_0,x_1,y_1,z_1...)

    The user is responsible for destroying this vector.

  Level: intermediate

.keywords: distributed array, get, corners, nodes, local indices, coordinates

.seealso: DAGetGhostCorners(), DASetCoordinates(), DASetUniformCoordinates(), DAGetCoordinates(), DAGetCoordinateDA()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetGhostedCoordinates(DA da,Vec *c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidPointer(c,2);
  if (!da->coordinates) SETERRQ(PETSC_ERR_ORDER,"You must call DASetCoordinates() before this call");
  if (!da->ghosted_coordinates) {
    DA dac;
    ierr = DAGetCoordinateDA(da,&dac);CHKERRQ(ierr);
    ierr = DACreateLocalVector(dac,&da->ghosted_coordinates);CHKERRQ(ierr);
    if (dac == da) {ierr = PetscObjectDereference((PetscObject)dac);CHKERRQ(ierr);}
    ierr = DAGlobalToLocalBegin(dac,da->coordinates,INSERT_VALUES,da->ghosted_coordinates);CHKERRQ(ierr);
    ierr = DAGlobalToLocalEnd(dac,da->coordinates,INSERT_VALUES,da->ghosted_coordinates);CHKERRQ(ierr);
    ierr = DADestroy(dac);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject) da->ghosted_coordinates);CHKERRQ(ierr);
  *c = da->ghosted_coordinates;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetFieldName"
/*@C
   DASetFieldName - Sets the names of individual field components in multicomponent
   vectors associated with a DA.

   Not Collective

   Input Parameters:
+  da - the distributed array
.  nf - field number for the DA (0, 1, ... dof-1), where dof indicates the 
        number of degrees of freedom per node within the DA
-  names - the name of the field (component)

  Level: intermediate

.keywords: distributed array, get, component name

.seealso: DAGetFieldName()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetFieldName(DA da,PetscInt nf,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
 
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (nf < 0 || nf >= da->w) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Invalid field number: %D",nf);
  if (da->fieldname[nf]) {ierr = PetscFree(da->fieldname[nf]);CHKERRQ(ierr);}
  
  ierr = PetscStrallocpy(name,&da->fieldname[nf]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetFieldName"
/*@C
   DAGetFieldName - Gets the names of individual field components in multicomponent
   vectors associated with a DA.

   Not Collective

   Input Parameter:
+  da - the distributed array
-  nf - field number for the DA (0, 1, ... dof-1), where dof indicates the 
        number of degrees of freedom per node within the DA

   Output Parameter:
.  names - the name of the field (component)

  Level: intermediate

.keywords: distributed array, get, component name

.seealso: DASetFieldName()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetFieldName(DA da,PetscInt nf,char **name)
{
  PetscFunctionBegin;
 
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidPointer(name,3);
  if (nf < 0 || nf >= da->w) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Invalid field number: %D",nf);
  *name = da->fieldname[nf];
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetCorners"
/*@
   DAGetCorners - Returns the global (x,y,z) indices of the lower left
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
   freedom per node set with the DACreateXX() routine. Thus the x, y, z, and
   m, n, p can be thought of as coordinates on a logical grid, where each
   grid point has (potentially) several degrees of freedom.
   Any of y, z, n, and p can be passed in as PETSC_NULL if not needed.

  Level: beginner

.keywords: distributed array, get, corners, nodes, local indices

.seealso: DAGetGhostCorners(), DAGetOwnershipRanges()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetCorners(DA da,PetscInt *x,PetscInt *y,PetscInt *z,PetscInt *m,PetscInt *n,PetscInt *p)
{
  PetscInt w;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  /* since the xs, xe ... have all been multiplied by the number of degrees 
     of freedom per cell, w = da->w, we divide that out before returning.*/
  w = da->w;  
  if (x) *x = da->xs/w; if(m) *m = (da->xe - da->xs)/w;
  /* the y and z have NOT been multiplied by w */
  if (y) *y = da->ys;   if (n) *n = (da->ye - da->ys);
  if (z) *z = da->zs;   if (p) *p = (da->ze - da->zs); 
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "DAGetLocalBoundingBox"
/*@
   DAGetLocalBoundingBox - Returns the local bounding box for the DA.

   Not Collective

   Input Parameter:
.  da - the distributed array

   Output Parameters:
+  lmin - local minimum coordinates (length dim, optional)
-  lmax - local maximim coordinates (length dim, optional)

  Level: beginner

.keywords: distributed array, get, coordinates

.seealso: DAGetCoordinateDA(), DAGetCoordinates(), DAGetBoundingBox()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetLocalBoundingBox(DA da,PetscReal lmin[],PetscReal lmax[])
{
  PetscErrorCode ierr;
  Vec            coords  = PETSC_NULL;
  PetscInt       dim,i,j;
  PetscScalar    *local_coords;
  PetscReal      min[3]={PETSC_MAX,PETSC_MAX,PETSC_MAX},max[3]={PETSC_MIN,PETSC_MIN,PETSC_MIN};
  PetscInt       N,Ni;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  dim = da->dim;
  ierr = DAGetCoordinates(da,&coords);CHKERRQ(ierr);
  ierr = VecGetArray(coords,&local_coords);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coords,&N);CHKERRQ(ierr);
  Ni = N/dim;
  for (i=0; i<Ni; i++) {
    for (j=0; j<dim; j++) {
      min[j] = PetscMin(min[j],PetscRealPart(local_coords[i*dim+j]));CHKERRQ(ierr);
      max[j] = PetscMax(min[j],PetscRealPart(local_coords[i*dim+j]));CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(coords,&local_coords);CHKERRQ(ierr);
  ierr = VecDestroy(coords);CHKERRQ(ierr);
  if (lmin) {ierr = PetscMemcpy(lmin,min,dim*sizeof(PetscReal));CHKERRQ(ierr);}
  if (lmax) {ierr = PetscMemcpy(lmax,max,dim*sizeof(PetscReal));CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetBoundingBox"
/*@
   DAGetBoundingBox - Returns the global bounding box for the DA.

   Collective on DA

   Input Parameter:
.  da - the distributed array

   Output Parameters:
+  gmin - global minimum coordinates (length dim, optional)
-  gmax - global maximim coordinates (length dim, optional)

  Level: beginner

.keywords: distributed array, get, coordinates

.seealso: DAGetCoordinateDA(), DAGetCoordinates(), DAGetLocalBoundingBox()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetBoundingBox(DA da,PetscReal gmin[],PetscReal gmax[])
{
  PetscErrorCode ierr;
  PetscMPIInt count;
  PetscReal lmin[3],lmax[3];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  count = PetscMPIIntCast(da->dim);
  ierr = DAGetLocalBoundingBox(da,lmin,lmax);CHKERRQ(ierr);
  if (gmin) {ierr = MPI_Allreduce(lmin,gmin,count,MPIU_REAL,MPI_MIN,((PetscObject)da)->comm);CHKERRQ(ierr);}
  if (gmax) {ierr = MPI_Allreduce(lmax,gmax,count,MPIU_REAL,MPI_MAX,((PetscObject)da)->comm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
