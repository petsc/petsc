
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/
#include <petscdmfield.h>

PetscErrorCode DMCreateCoordinateDM_DA(DM dm, DM *cdm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMDACreateCompatibleDMDA(dm,dm->dim,cdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateCoordinateField_DA(DM dm, DMField *field)
{
  PetscReal      gmin[3], gmax[3];
  PetscScalar    corners[24];
  PetscInt       dim;
  PetscInt       i, j;
  DM             cdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  /* TODO: this is wrong if coordinates are not rectilinear */
  ierr = DMGetBoundingBox(dm,gmin,gmax);CHKERRQ(ierr);
  for (i = 0; i < (1 << dim); i++) {
    for (j = 0; j < dim; j++) {
      corners[i*dim + j] = (i & (1 << j)) ? gmax[j] : gmin[j];
    }
  }
  ierr = DMClone(dm,&cdm);CHKERRQ(ierr);
  ierr = DMFieldCreateDA(cdm,dim,corners,field);CHKERRQ(ierr);
  ierr = DMDestroy(&cdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   DMDASetFieldName - Sets the names of individual field components in multicomponent
   vectors associated with a DMDA.

   Logically collective; name must contain a common value

   Input Parameters:
+  da - the distributed array
.  nf - field number for the DMDA (0, 1, ... dof-1), where dof indicates the
        number of degrees of freedom per node within the DMDA
-  names - the name of the field (component)

  Notes:
    It must be called after having called DMSetUp().

  Level: intermediate

.seealso: DMDAGetFieldName(), DMDASetCoordinateName(), DMDAGetCoordinateName(), DMDASetFieldNames(), DMSetUp()
@*/
PetscErrorCode  DMDASetFieldName(DM da,PetscInt nf,const char name[])
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  if (nf < 0 || nf >= dd->w) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid field number: %D",nf);
  if (!dd->fieldname) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_ORDER,"You should call DMSetUp() first");
  ierr = PetscFree(dd->fieldname[nf]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name,&dd->fieldname[nf]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   DMDAGetFieldNames - Gets the name of each component in the vector associated with the DMDA

   Not collective; names will contain a common value

   Input Parameter:
.  dm - the DMDA object

   Output Parameter:
.  names - the names of the components, final string is NULL, will have the same number of entries as the dof used in creating the DMDA

   Level: intermediate

   Not supported from Fortran, use DMDAGetFieldName()

.seealso: DMDAGetFieldName(), DMDASetCoordinateName(), DMDAGetCoordinateName(), DMDASetFieldName(), DMDASetFieldNames()
@*/
PetscErrorCode  DMDAGetFieldNames(DM da,const char * const **names)
{
  DM_DA             *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  *names = (const char * const *) dd->fieldname;
  PetscFunctionReturn(0);
}

/*@C
   DMDASetFieldNames - Sets the name of each component in the vector associated with the DMDA

   Logically collective; names must contain a common value

   Input Parameters:
+  dm - the DMDA object
-  names - the names of the components, final string must be NULL, must have the same number of entries as the dof used in creating the DMDA

   Notes:
    It must be called after having called DMSetUp().

   Level: intermediate

   Not supported from Fortran, use DMDASetFieldName()

.seealso: DMDAGetFieldName(), DMDASetCoordinateName(), DMDAGetCoordinateName(), DMDASetFieldName(), DMSetUp()
@*/
PetscErrorCode  DMDASetFieldNames(DM da,const char * const *names)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;
  char           **fieldname;
  PetscInt       nf = 0;

  PetscFunctionBegin;
  if (!dd->fieldname) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_ORDER,"You should call DMSetUp() first");
  while (names[nf++]) {};
  if (nf != dd->w+1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid number of fields %D",nf-1);
  ierr = PetscStrArrayallocpy(names,&fieldname);CHKERRQ(ierr);
  ierr = PetscStrArrayDestroy(&dd->fieldname);CHKERRQ(ierr);
  dd->fieldname = fieldname;
  PetscFunctionReturn(0);
}

/*@C
   DMDAGetFieldName - Gets the names of individual field components in multicomponent
   vectors associated with a DMDA.

   Not collective; name will contain a common value

   Input Parameter:
+  da - the distributed array
-  nf - field number for the DMDA (0, 1, ... dof-1), where dof indicates the
        number of degrees of freedom per node within the DMDA

   Output Parameter:
.  names - the name of the field (component)

  Notes:
    It must be called after having called DMSetUp().

  Level: intermediate

.seealso: DMDASetFieldName(), DMDASetCoordinateName(), DMDAGetCoordinateName(), DMSetUp()
@*/
PetscErrorCode  DMDAGetFieldName(DM da,PetscInt nf,const char **name)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidPointer(name,3);
  if (nf < 0 || nf >= dd->w) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid field number: %D",nf);
  if (!dd->fieldname) SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_ORDER,"You should call DMSetUp() first");
  *name = dd->fieldname[nf];
  PetscFunctionReturn(0);
}

/*@C
   DMDASetCoordinateName - Sets the name of the coordinate directions associated with a DMDA, for example "x" or "y"

   Logically collective; name must contain a common value

   Input Parameters:
+  dm - the DM
.  nf - coordinate number for the DMDA (0, 1, ... dim-1),
-  name - the name of the coordinate

  Notes:
    It must be called after having called DMSetUp().


  Level: intermediate

  Not supported from Fortran

.seealso: DMDAGetCoordinateName(), DMDASetFieldName(), DMDAGetFieldName(), DMSetUp()
@*/
PetscErrorCode DMDASetCoordinateName(DM dm,PetscInt nf,const char name[])
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMDA);
  if (nf < 0 || nf >= dm->dim) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid coordinate number: %D",nf);
  if (!dd->coordinatename) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ORDER,"You should call DMSetUp() first");
  ierr = PetscFree(dd->coordinatename[nf]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name,&dd->coordinatename[nf]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   DMDAGetCoordinateName - Gets the name of a coodinate direction associated with a DMDA.

   Not collective; name will contain a common value

   Input Parameter:
+  dm - the DM
-  nf -  number for the DMDA (0, 1, ... dim-1)

   Output Parameter:
.  names - the name of the coordinate direction

  Notes:
    It must be called after having called DMSetUp().


  Level: intermediate

  Not supported from Fortran

.seealso: DMDASetCoordinateName(), DMDASetFieldName(), DMDAGetFieldName(), DMSetUp()
@*/
PetscErrorCode DMDAGetCoordinateName(DM dm,PetscInt nf,const char **name)
{
  DM_DA *dd = (DM_DA*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMDA);
  PetscValidPointer(name,3);
  if (nf < 0 || nf >= dm->dim) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid coordinate number: %D",nf);
  if (!dd->coordinatename) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ORDER,"You should call DMSetUp() first");
  *name = dd->coordinatename[nf];
  PetscFunctionReturn(0);
}

/*@C
   DMDAGetCorners - Returns the global (x,y,z) indices of the lower left
   corner and size of the local region, excluding ghost points.

   Not collective

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

.seealso: DMDAGetGhostCorners(), DMDAGetOwnershipRanges()
@*/
PetscErrorCode  DMDAGetCorners(DM da,PetscInt *x,PetscInt *y,PetscInt *z,PetscInt *m,PetscInt *n,PetscInt *p)
{
  PetscInt w;
  DM_DA    *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  /* since the xs, xe ... have all been multiplied by the number of degrees
     of freedom per cell, w = dd->w, we divide that out before returning.*/
  w = dd->w;
  if (x) *x = dd->xs/w + dd->xo;
  /* the y and z have NOT been multiplied by w */
  if (y) *y = dd->ys + dd->yo;
  if (z) *z = dd->zs + dd->zo;
  if (m) *m = (dd->xe - dd->xs)/w;
  if (n) *n = (dd->ye - dd->ys);
  if (p) *p = (dd->ze - dd->zs);
  PetscFunctionReturn(0);
}

PetscErrorCode DMGetLocalBoundingIndices_DMDA(DM dm, PetscReal lmin[], PetscReal lmax[])
{
  DMDALocalInfo  info;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr   = DMDAGetLocalInfo(dm, &info);CHKERRQ(ierr);
  lmin[0] = info.xs;
  lmin[1] = info.ys;
  lmin[2] = info.zs;
  lmax[0] = info.xs + info.xm-1;
  lmax[1] = info.ys + info.ym-1;
  lmax[2] = info.zs + info.zm-1;
  PetscFunctionReturn(0);
}

/*@
   DMDAGetReducedDMDA - Deprecated; use DMDACreateCompatibleDMDA()

   Level: deprecated
@*/
PetscErrorCode DMDAGetReducedDMDA(DM da,PetscInt nfields,DM *nda)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDACreateCompatibleDMDA(da,nfields,nda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   DMDACreateCompatibleDMDA - Creates a DMDA with the same layout but with fewer or more fields

   Collective

   Input Parameters:
+  da - the distributed array
-  nfields - number of fields in new DMDA

   Output Parameter:
.  nda - the new DMDA

  Level: intermediate

.seealso: DMDAGetGhostCorners(), DMSetCoordinates(), DMDASetUniformCoordinates(), DMGetCoordinates(), DMDAGetGhostedCoordinates()
@*/
PetscErrorCode  DMDACreateCompatibleDMDA(DM da,PetscInt nfields,DM *nda)
{
  PetscErrorCode   ierr;
  DM_DA            *dd = (DM_DA*)da->data;
  PetscInt         s,m,n,p,M,N,P,dim,Mo,No,Po;
  const PetscInt   *lx,*ly,*lz;
  DMBoundaryType   bx,by,bz;
  DMDAStencilType  stencil_type;
  PetscInt         ox,oy,oz;
  PetscInt         cl,rl;

  PetscFunctionBegin;
  dim = da->dim;
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
  ierr = DMSetUp(*nda);CHKERRQ(ierr);
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

/*@C
   DMDAGetCoordinateArray - Gets an array containing the coordinates of the DMDA

   Not collective

   Input Parameter:
.  dm - the DM

   Output Parameter:
.  xc - the coordinates

  Level: intermediate

  Not supported from Fortran

.seealso: DMDASetCoordinateName(), DMDASetFieldName(), DMDAGetFieldName(), DMDARestoreCoordinateArray()
@*/
PetscErrorCode DMDAGetCoordinateArray(DM dm,void *xc)
{
  PetscErrorCode ierr;
  DM             cdm;
  Vec            x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetCoordinates(dm,&x);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm,&cdm);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cdm,x,xc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   DMDARestoreCoordinateArray - Sets an array containing the coordinates of the DMDA

   Not collective

   Input Parameter:
+  dm - the DM
-  xc - the coordinates

  Level: intermediate

  Not supported from Fortran

.seealso: DMDASetCoordinateName(), DMDASetFieldName(), DMDAGetFieldName(), DMDAGetCoordinateArray()
@*/
PetscErrorCode DMDARestoreCoordinateArray(DM dm,void *xc)
{
  PetscErrorCode ierr;
  DM             cdm;
  Vec            x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetCoordinates(dm,&x);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm,&cdm);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(cdm,x,xc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
