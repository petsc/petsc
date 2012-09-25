#include <petsc-private/igaimpl.h>    /*I   "petscdmiga.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_IGA"
PetscErrorCode DMDestroy_IGA(DM dm)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(iga->Ux);CHKERRQ(ierr);
  ierr = PetscFree(iga->Uy);CHKERRQ(ierr);
  ierr = PetscFree(iga->Uz);CHKERRQ(ierr);
  ierr = BDDestroy(&iga->bdX);CHKERRQ(ierr);
  ierr = BDDestroy(&iga->bdY);CHKERRQ(ierr);
  ierr = BDDestroy(&iga->bdZ);CHKERRQ(ierr);
  ierr = DMDestroy(&iga->da_dof);CHKERRQ(ierr);
  ierr = DMDestroy(&iga->da_geometry);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMView_IGA"
PetscErrorCode DMView_IGA(DM dm, PetscViewer viewer)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);

  if (iascii){
    ierr = PetscViewerASCIIPrintf(viewer, "IGA:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  number of elements: %d %d %d\n", iga->Nx, iga->Ny, iga->Nz);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  polynomial order: %d %d %d\n", iga->px, iga->py, iga->pz);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  Data DM:\n");CHKERRQ(ierr);
    ierr = DMView(iga->da_dof, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  Geometry DM:\n");CHKERRQ(ierr);
    ierr = DMView(iga->da_geometry, viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Viewer type %s not supported by this mesh object", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_IGA"
PetscErrorCode DMCreateGlobalVector_IGA(DM dm, Vec *gvec)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateGlobalVector(iga->da_dof, gvec);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*gvec,"DM",(PetscObject)dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateLocalVector_IGA"
PetscErrorCode DMCreateLocalVector_IGA(DM dm, Vec *lvec)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateLocalVector(iga->da_dof, lvec);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*lvec,"DM",(PetscObject)dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateMatrix_IGA"
PetscErrorCode DMCreateMatrix_IGA(DM dm, const MatType mtype, Mat *J)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateMatrix(iga->da_dof, mtype, J);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*J,"DM",(PetscObject)dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGlobalToLocalBegin_IGA"
PetscErrorCode DMGlobalToLocalBegin_IGA(DM dm, Vec gv, InsertMode mode, Vec lv)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGlobalToLocalBegin(iga->da_dof, gv, mode, lv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGlobalToLocalEnd_IGA"
PetscErrorCode DMGlobalToLocalEnd_IGA(DM dm, Vec gv, InsertMode mode, Vec lv)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGlobalToLocalEnd(iga->da_dof, gv, mode, lv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLocalToGlobalBegin_IGA"
PetscErrorCode DMLocalToGlobalBegin_IGA(DM dm, Vec lv, InsertMode mode, Vec gv)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMLocalToGlobalBegin(iga->da_dof, lv, mode, gv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMLocalToGlobalEnd_IGA"
PetscErrorCode DMLocalToGlobalEnd_IGA(DM dm, Vec lv, InsertMode mode, Vec gv)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMLocalToGlobalEnd(iga->da_dof, lv, mode, gv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMIGAGetPolynomialOrder"
/*@C
  DMIGAGetPolynomialOrder - Gets the polynomial order for each direction

  Not Collective

  Input Parameter:
. dm - the IGA

  Output Parameters:
+ px - polynomial order in X
. py - polynomial order in Y
- pz - polynomial order in Z

  Level: beginner

.keywords: distributed array, get, information
.seealso: DMCreate()
@*/
PetscErrorCode DMIGAGetPolynomialOrder(DM dm, PetscInt *px, PetscInt *py, PetscInt *pz)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (px) {
    PetscValidPointer(px, 2);
    *px = iga->px;
  }
  if (py) {
    PetscValidPointer(py, 3);
    *py = iga->py;
  }
  if (pz) {
    PetscValidPointer(pz, 4);
    *pz = iga->pz;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMIGAGetNumQuadraturePoints"
/*@C
  DMIGAGetNumQuadraturePoints - Gets the number of quadrature points for each direction

  Not Collective

  Input Parameter:
. dm - the IGA

  Output Parameters:
+ nx - number of quadrature points in X
. ny - number of quadrature points in Y
- nz - number of quadrature points in Z

  Level: beginner

.keywords: distributed array, get, information
.seealso: DMCreate()
@*/
PetscErrorCode DMIGAGetNumQuadraturePoints(DM dm, PetscInt *nx, PetscInt *ny, PetscInt *nz)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (nx) {
    PetscValidPointer(nx, 2);
    *nx = iga->ngx;
  }
  if (ny) {
    PetscValidPointer(ny, 3);
    *ny = iga->ngy;
  }
  if (nz) {
    PetscValidPointer(nz, 4);
    *nz = iga->ngz;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMIGAGetBasisData"
/*@C
  DMIGAGetBasisData - Gets the basis data at quadrature points for each direction

  Not Collective

  Input Parameter:
. dm - the IGA

  Output Parameters:
+ bdX - basis data in X
. bdY - basis data in Y
- bdZ - basis data in Z

  Level: beginner

.keywords: distributed array, get, information
.seealso: DMCreate()
@*/
PetscErrorCode DMIGAGetBasisData(DM dm, BD *bdX, BD *bdY, BD *bdZ)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (bdX) {
    PetscValidPointer(bdX, 2);
    *bdX = iga->bdX;
  }
  if (bdY) {
    PetscValidPointer(bdY, 3);
    *bdY = iga->bdY;
  }
  if (bdZ) {
    PetscValidPointer(bdZ, 4);
    *bdZ = iga->bdZ;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMIGASetFieldName"
/*@C
  DMIGASetFieldName - Sets the names of individual field components in multicomponent vectors associated with a IGA.

  Not Collective

  Input Parameters:
+ dm - the IGA
. nf - field number for the IGA (0, 1, ... dof-1), where dof indicates the number of degrees of freedom per node within the IGA
- names - the name of the field (component)

  Level: intermediate

.keywords: distributed array, get, component name
.seealso: DMIGAGetFieldName()
@*/
PetscErrorCode DMIGASetFieldName(DM dm, PetscInt nf, const char name[])
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMDASetFieldName(iga->da_dof, nf, name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMIGAGetFieldName"
/*@C
  DMIGAGetFieldName - Gets the names of individual field components in multicomponent vectors associated with a IGA.

  Not Collective

  Input Parameters:
+ dm - the IGA
- nf - field number for the IGA (0, 1, ... dof-1), where dof indicates the number of degrees of freedom per node within the IGA

  Output Parameter:
. names - the name of the field (component)

  Level: intermediate

.keywords: distributed array, get, component name
.seealso: DMIGASetFieldName()
@*/
PetscErrorCode DMIGAGetFieldName(DM dm, PetscInt nf, const char **name)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(name, 3);
  ierr = DMDAGetFieldName(iga->da_dof, nf, name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMIGAVecGetArray"
/*@C
  DMIGAVecGetArray - Returns a multiple dimension array that shares data with the underlying vector and is indexed using the global dimensions.

  Not Collective

  Input Parameters:
+ dm - the IGA
- vec - the vector, either a vector the same size as one obtained with DMCreateGlobalVector() or DMCreateLocalVector()

  Output Parameter:
. array - the array

  Notes:
    Call DMIGAVecRestoreArray() once you have finished accessing the vector entries.

    In C, the indexing is "backwards" from what expects: array[k][j][i] NOT array[i][j][k]!

    If vec is a local vector (obtained with DMCreateLocalVector() etc) then they ghost point locations are accessable. If it is
    a global vector then the ghost points are not accessable. Of course with the local vector you will have had to do the
    appropriate DMLocalToGlobalBegin() and DMLocalToGlobalEnd() to have correct values in the ghost locations.

  Level: intermediate

.keywords: distributed array, get, corners, nodes, local indices, coordinates
.seealso: DMIGAVecRestoreArray(), VecGetArray(), VecRestoreArray(), DMDAVecRestoreArray(), DMDAVecGetArray()
@*/
PetscErrorCode DMIGAVecGetArray(DM dm, Vec vec, void *array)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 2);
  PetscValidPointer(array, 3);
  ierr = DMDAVecGetArray(iga->da_dof, vec, array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMIGAVecRestoreArray"
/*@
  DMIGAVecRestoreArray - Restores a multiple dimension array obtained with DMIGAVecGetArray()

  Not Collective

  Input Parameters:
+ dm - the IGA
. vec - the vector, either a vector the same size as one obtained with DMCreateGlobalVector() or DMCreateLocalVector()
- array - the array

  Level: intermediate

.keywords: distributed array, get, corners, nodes, local indices, coordinates
.seealso: DMIGAVecGetArray(), VecGetArray(), VecRestoreArray(), DMDAVecRestoreArray(), DMDAVecGetArray()
@*/
PetscErrorCode DMIGAVecRestoreArray(DM dm, Vec vec, void *array)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 2);
  PetscValidPointer(array, 3);
  ierr = DMDAVecRestoreArray(iga->da_dof, vec, array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMIGAGetLocalInfo"
/*@C
  DMIGAGetLocalInfo - Gets information about a given IGA and this processors location in it

  Not Collective

  Input Parameter:
. dm - the IGA

  Output Parameter:
. dainfo - structure containing the information

  Level: beginner

.keywords: distributed array, get, information
.seealso: DMDAGetLocalInfo()
@*/
PetscErrorCode DMIGAGetLocalInfo(DM dm, DMDALocalInfo *info)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(info, 2);
  ierr = DMDAGetLocalInfo(iga->da_dof, info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMIGAInitializeUniform3d"
PetscErrorCode DMIGAInitializeUniform3d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                        PetscInt px,PetscInt Nx,PetscInt Cx,PetscReal Ux0, PetscReal Uxf,PetscBool IsPeriodicX,PetscInt ngx,
                                        PetscInt py,PetscInt Ny,PetscInt Cy,PetscReal Uy0, PetscReal Uyf,PetscBool IsPeriodicY,PetscInt ngy,
                                        PetscInt pz,PetscInt Nz,PetscInt Cz,PetscReal Uz0, PetscReal Uzf,PetscBool IsPeriodicZ,PetscInt ngz)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;
  DMDABoundaryType xptype = DMDA_BOUNDARY_NONE;
  DMDABoundaryType yptype = DMDA_BOUNDARY_NONE;
  DMDABoundaryType zptype = DMDA_BOUNDARY_NONE;
  PetscInt   sw;
  DMDALocalInfo       info_dof;

  PetscFunctionBegin;
  /* Test C < p */
  if(px <= Cx || py <= Cy || pz <= Cz){
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Discretization inconsistent: polynomial order must be greater than degree of continuity");
  }

  /* Load constants */
  iga->px = px; iga->py = py; iga->pz = pz;
  iga->Nx = Nx; iga->Ny = Nx; iga->Nz = Nz;
  iga->Cx = Cx; iga->Cy = Cx; iga->Cz = Cx;
  iga->ngx = ngx; iga->ngy = ngy; iga->ngz = ngz;
  iga->IsPeriodicX = IsPeriodicX; iga->IsPeriodicY = IsPeriodicY; iga->IsPeriodicZ = IsPeriodicZ;
  iga->numD = NumDerivatives;

  /* Knot vector size */
  iga->mx = 2*(iga->px+1);
  iga->my = 2*(iga->py+1);
  iga->mz = 2*(iga->pz+1);
  iga->mx += (iga->px-iga->Cx)*(iga->Nx-1);
  iga->my += (iga->py-iga->Cy)*(iga->Ny-1);
  iga->mz += (iga->pz-iga->Cz)*(iga->Nz-1);

  /* number of basis functions */
  iga->nbx = iga->mx-iga->px-1;
  iga->nby = iga->my-iga->py-1;
  iga->nbz = iga->mz-iga->pz-1;

  /* compute knot vectors */
  ierr = PetscMalloc(iga->mx*sizeof(PetscReal), &iga->Ux);CHKERRQ(ierr);
  ierr = PetscMalloc(iga->my*sizeof(PetscReal), &iga->Uy);CHKERRQ(ierr);
  ierr = PetscMalloc(iga->mz*sizeof(PetscReal), &iga->Uz);CHKERRQ(ierr);

  if(IsPeriodicX){
    ierr = CreatePeriodicKnotVector(iga->Nx,iga->px,iga->Cx,iga->mx,iga->Ux,Ux0,Uxf);CHKERRQ(ierr);
    iga->nbx -= iga->px;
  }else{
    ierr = CreateKnotVector(iga->Nx,iga->px,iga->Cx,iga->mx,iga->Ux,Ux0,Uxf);CHKERRQ(ierr);
  }
  if(IsPeriodicY){
    ierr = CreatePeriodicKnotVector(iga->Ny,iga->py,iga->Cy,iga->my,iga->Uy,Uy0,Uyf);CHKERRQ(ierr);
    iga->nby -= iga->py;
  }else{
    ierr = CreateKnotVector(iga->Ny,iga->py,iga->Cy,iga->my,iga->Uy,Uy0,Uyf);CHKERRQ(ierr);
  }
  if(IsPeriodicZ){
    ierr = CreatePeriodicKnotVector(iga->Nz,iga->pz,iga->Cz,iga->mz,iga->Uz,Uz0,Uzf);CHKERRQ(ierr);
    iga->nbz -= iga->pz;
  }else{
    ierr = CreateKnotVector(iga->Nz,iga->pz,iga->Cz,iga->mz,iga->Uz,Uz0,Uzf);CHKERRQ(ierr);
  }

  /* compute and store 1d basis functions at gauss points */
  ierr = Compute1DBasisFunctions(iga->ngx, iga->numD, iga->Ux, iga->mx, iga->px, &iga->bdX);CHKERRQ(ierr);
  ierr = Compute1DBasisFunctions(iga->ngy, iga->numD, iga->Uy, iga->my, iga->py, &iga->bdY);CHKERRQ(ierr);
  ierr = Compute1DBasisFunctions(iga->ngz, iga->numD, iga->Uz, iga->mz, iga->pz, &iga->bdZ);CHKERRQ(ierr);

  if (IsPeriodicX) xptype = DMDA_BOUNDARY_PERIODIC;
  if (IsPeriodicY) yptype = DMDA_BOUNDARY_PERIODIC;
  if (IsPeriodicZ) zptype = DMDA_BOUNDARY_PERIODIC;

  sw = (iga->px>iga->py) ? iga->px : iga->py ; sw = (sw>iga->pz) ? sw : iga->pz ;
  ierr = DMDACreate(((PetscObject) dm)->comm,&iga->da_dof); CHKERRQ(ierr);
  ierr = DMDASetDim(iga->da_dof, 3); CHKERRQ(ierr);
  ierr = DMDASetSizes(iga->da_dof,iga->nbx,iga->nby,iga->nbz); CHKERRQ(ierr);
  ierr = DMDASetDof(iga->da_dof, ndof); CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(iga->da_dof, xptype, yptype, zptype); CHKERRQ(ierr);
  ierr = DMDASetStencilType(iga->da_dof,DMDA_STENCIL_BOX); CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(iga->da_dof,sw); CHKERRQ(ierr);
  ierr = DMSetFromOptions(iga->da_dof); CHKERRQ(ierr);
  ierr = DMSetUp(iga->da_dof);CHKERRQ(ierr);

  /* Determine how the elements map to processors */

  ierr = DMDAGetLocalInfo(iga->da_dof,&info_dof);CHKERRQ(ierr);
  ierr = BDSetElementOwnership(iga->bdX,iga->Nx,info_dof.xs,info_dof.xs+info_dof.xm-1,iga->px);CHKERRQ(ierr);
  ierr = BDSetElementOwnership(iga->bdY,iga->Ny,info_dof.ys,info_dof.ys+info_dof.ym-1,iga->py);CHKERRQ(ierr);
  ierr = BDSetElementOwnership(iga->bdZ,iga->Nz,info_dof.zs,info_dof.zs+info_dof.zm-1,iga->pz);CHKERRQ(ierr);


  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMIGAInitializeGeometry3d"
PetscErrorCode DMIGAInitializeGeometry3d(DM dm,PetscInt ndof,PetscInt NumDerivatives,char *FunctionSpaceFile,char *GeomFile)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  FILE          *fp;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscInt       spatial_dim,i;
  PetscReal      Umax;
  PetscInt       numEl;
  DMDABoundaryType ptype;
  PetscInt       sw;
  DMDALocalInfo  info_dof;
  int            ival;
  double         dval;

  PetscFunctionBegin;
  fp = fopen(FunctionSpaceFile, "r");
  if (fp == NULL ){
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_FILE_OPEN, "Cannot find geometry file");
  }

  if (fscanf(fp, "%d", &ival) != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SYS,"Failed to read spatial dimension from %s",FunctionSpaceFile);
  spatial_dim = ival;
  if(spatial_dim != 3){
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Geometry dimension != problem dimension");
  }

  /* Read in polynomial orders and number of basis functions */
  {
    int a,b,c;
    if (fscanf(fp, "%d %d %d", &a, &b, &c) != 3) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SYS,"Failed to read polynomial orders from %s",FunctionSpaceFile);
    iga->px = a; iga->py = b; iga->pz= c;
    if (fscanf(fp, "%d %d %d", &a, &b, &c) != 3) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SYS,"Failed to read number of basis functions from %s",FunctionSpaceFile);
    iga->nbx = a; iga->nby = b; iga->nbz= c;
  }

  /* Knot vector size */
  iga->mx = iga->nbx + iga->px + 1;
  iga->my = iga->nby + iga->py + 1;
  iga->mz = iga->nbz + iga->pz + 1;

  /* Read in my knot vectors */
  ierr = PetscMalloc(iga->mx*sizeof(PetscReal), &iga->Ux);CHKERRQ(ierr);
  ierr = PetscMalloc(iga->my*sizeof(PetscReal), &iga->Uy);CHKERRQ(ierr);
  ierr = PetscMalloc(iga->mz*sizeof(PetscReal), &iga->Uz);CHKERRQ(ierr);

  Umax = 0.0;
  for(i=0;i<iga->mx;i++) {
    if (fscanf(fp, "%lf ", &dval) != 1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SYS,"Failed to read X coordinate at %D from %s",i,FunctionSpaceFile);
    iga->Ux[i] = dval;
    if(iga->Ux[i] > Umax) Umax = iga->Ux[i];
  }
  for(i=0;i<iga->mx;i++) iga->Ux[i] /= Umax;

  Umax = 0.0;
  for(i=0;i<iga->my;i++) {
    if (fscanf(fp, "%lf ", &dval) != 1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SYS,"Failed to read Y coordinate at %D from %s",i,FunctionSpaceFile);
    iga->Uy[i] = dval;
    if(iga->Uy[i] > Umax) Umax = iga->Uy[i];
  }
  for(i=0;i<iga->my;i++) iga->Uy[i] /= Umax;

  Umax = 0.0;
  for(i=0;i<iga->mz;i++) {
    if (fscanf(fp, "%lf ", &dval) != 1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SYS,"Failed to read Z coordinate at %D from %s",i,FunctionSpaceFile);
    iga->Uz[i] = dval;
    if(iga->Uz[i] > Umax) Umax = iga->Uz[i];
  }
  for(i=0;i<iga->mz;i++) iga->Uz[i] /= Umax;

  fclose(fp);

  /* count the number of elements */
  numEl = 0;
  for(i = 0; i < iga->mx-1; ++i) {
    PetscReal du = (iga->Ux[i+1]-iga->Ux[i]);
    if(du > 1.0e-13) numEl++;
  }
  iga->Nx = numEl;

  numEl = 0;
  for(i = 0; i < iga->my-1; ++i) {
    PetscReal du = (iga->Uy[i+1]-iga->Uy[i]);
    if(du > 1.0e-13) numEl++;
  }
  iga->Ny = numEl;

  numEl = 0;
  for(i = 0; i < iga->mz-1; ++i) {
    PetscReal du = (iga->Uz[i+1]-iga->Uz[i]);
    if(du > 1.0e-13) numEl++;
  }
  iga->Nz = numEl;

  /* Load constants */
  iga->ngx = iga->px+1; iga->ngy = iga->py+1; iga->ngz = iga->pz+1;
  iga->numD = NumDerivatives;
  iga->IsRational = PETSC_TRUE;
  iga->IsMapped = PETSC_TRUE;

  /* compute and store 1d basis functions at gauss points */
  ierr = Compute1DBasisFunctions(iga->ngx, iga->numD, iga->Ux, iga->mx, iga->px, &iga->bdX);CHKERRQ(ierr);
  ierr = Compute1DBasisFunctions(iga->ngy, iga->numD, iga->Uy, iga->my, iga->py, &iga->bdY);CHKERRQ(ierr);
  ierr = Compute1DBasisFunctions(iga->ngz, iga->numD, iga->Uz, iga->mz, iga->pz, &iga->bdZ);CHKERRQ(ierr);

  ptype = DMDA_BOUNDARY_NONE ;
  sw = (iga->px>iga->py) ? iga->px : iga->py ; sw = (sw>iga->pz) ? sw : iga->pz ;

  /* DOF DA */
  ierr = DMDACreate(((PetscObject) dm)->comm,&iga->da_dof); CHKERRQ(ierr);
  ierr = DMDASetDim(iga->da_dof, 3); CHKERRQ(ierr);
  ierr = DMDASetSizes(iga->da_dof,iga->nbx,iga->nby,iga->nbz); CHKERRQ(ierr);
  ierr = DMDASetDof(iga->da_dof, ndof); CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(iga->da_dof, ptype, ptype, ptype); CHKERRQ(ierr);
  ierr = DMDASetStencilType(iga->da_dof,DMDA_STENCIL_BOX); CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(iga->da_dof,sw); CHKERRQ(ierr);
  ierr = DMSetFromOptions(iga->da_dof); CHKERRQ(ierr);
  ierr = DMSetUp(iga->da_dof);CHKERRQ(ierr);

  /* Determine how the elements map to processors */
  ierr = DMDAGetLocalInfo(iga->da_dof,&info_dof);CHKERRQ(ierr);
  ierr = BDSetElementOwnership(iga->bdX,iga->Nx,info_dof.xs,info_dof.xs+info_dof.xm-1,iga->px);CHKERRQ(ierr);
  ierr = BDSetElementOwnership(iga->bdY,iga->Ny,info_dof.ys,info_dof.ys+info_dof.ym-1,iga->py);CHKERRQ(ierr);
  ierr = BDSetElementOwnership(iga->bdZ,iga->Nz,info_dof.zs,info_dof.zs+info_dof.zm-1,iga->pz);CHKERRQ(ierr);

  /* Geometry DA */
  ierr = DMDACreate(((PetscObject) dm)->comm,&iga->da_geometry); CHKERRQ(ierr);
  ierr = DMDASetDim(iga->da_geometry, 3); CHKERRQ(ierr);
  ierr = DMDASetSizes(iga->da_geometry,iga->nbx,iga->nby,iga->nbz); CHKERRQ(ierr);
  ierr = DMDASetDof(iga->da_geometry, 4); CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(iga->da_geometry, ptype, ptype, ptype); CHKERRQ(ierr);
  ierr = DMDASetStencilType(iga->da_geometry,DMDA_STENCIL_BOX); CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(iga->da_geometry,sw); CHKERRQ(ierr);
  ierr = DMSetFromOptions(iga->da_geometry); CHKERRQ(ierr);
  ierr = DMSetUp(iga->da_geometry);CHKERRQ(ierr);

  /* Read in the geometry */
  ierr = DMCreateGlobalVector(iga->da_geometry,&iga->G);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)(iga->G),&comm);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,GeomFile,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(iga->G,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMIGAInitializeSymmetricTaper2d"
PetscErrorCode DMIGAInitializeSymmetricTaper2d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                               PetscInt px,PetscInt Nx,PetscInt Cx,PetscReal fx,
                                               PetscReal Ux0, PetscReal Uxf,PetscBool IsPeriodicX,PetscInt ngx,
                                               PetscInt py,PetscInt Ny,PetscInt Cy,PetscReal fy,
                                               PetscReal Uy0, PetscReal Uyf,PetscBool IsPeriodicY,PetscInt ngy)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;
  DMDABoundaryType xptype = DMDA_BOUNDARY_NONE ;
  DMDABoundaryType yptype = DMDA_BOUNDARY_NONE ;
  PetscInt   sw;
  DMDALocalInfo       info_dof;

  PetscFunctionBegin;
  /* Test C < p */
  if(px <= Cx || py <= Cy){
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Discretization inconsistent: polynomial order must be greater than degree of continuity");
  }

  /* Load constants */
  iga->px = px; iga->py = py;
  iga->Nx = Nx; iga->Ny = Ny;
  iga->Cx = Cx; iga->Cy = Cy;
  iga->ngx = ngx; iga->ngy = ngy;
  iga->IsPeriodicX = IsPeriodicX; iga->IsPeriodicY = IsPeriodicY;
  iga->numD = NumDerivatives;

  /* Knot vector size*/
  iga->mx = 2*(iga->px+1);
  iga->my = 2*(iga->py+1);
  iga->mx += (iga->px-iga->Cx)*(iga->Nx-1);
  iga->my += (iga->py-iga->Cy)*(iga->Ny-1);

  /* number of basis functions */
  iga->nbx = iga->mx-iga->px-1;
  iga->nby = iga->my-iga->py-1;

  /* compute knot vectors */
  ierr = PetscMalloc(iga->mx*sizeof(PetscReal), &iga->Ux);CHKERRQ(ierr);
  ierr = PetscMalloc(iga->my*sizeof(PetscReal), &iga->Uy);CHKERRQ(ierr);

  if(IsPeriodicX){

    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Initialization routine for tapered meshes does not yet support periodicity");
    ierr = CreatePeriodicKnotVector(iga->Nx,iga->px,iga->Cx,iga->mx,iga->Ux,Ux0,Uxf);CHKERRQ(ierr);
    iga->nbx -= iga->px;

  }else{

    PetscReal *X1;
    PetscReal *X2;
    PetscInt i;
    PetscReal *X;

    ierr = PetscMalloc((Nx/2+1)*sizeof(PetscReal),&X1);CHKERRQ(ierr);
    ierr = PetscMalloc((Nx/2+1)*sizeof(PetscReal),&X2);CHKERRQ(ierr);

    CreateTaperSetOfPoints(Ux0,0.5*(Uxf+Ux0),fx,Nx/2+1,X1);
    CreateTaperSetOfPoints(Uxf,0.5*(Uxf+Ux0),fx,Nx/2+1,X2);

    ierr = PetscMalloc((Nx+1)*sizeof(PetscReal),&X);CHKERRQ(ierr);

    if( Nx % 2 == 0){

      for(i=0;i<Nx/2+1;i++) {
	X[i]=X1[i];
	X[Nx/2+i]=X2[Nx/2-i];
      }

    }else{

      for(i=0;i<Nx/2+1;i++) {
	X[i]=X1[i];
	X[Nx/2+1+i]=X2[Nx/2-i];
      }

      X[Nx/2]   =  2.0/3.0*X[Nx/2 - 1] + 1.0/3.0*X[Nx/2 + 2];
      X[Nx/2+1] =  1.0/3.0*X[Nx/2 - 1] + 2.0/3.0*X[Nx/2 + 2];

    }

    ierr = CreateKnotVectorFromMesh(iga->Nx,iga->px,iga->Cx,iga->mx,iga->Ux,X,Nx+1);

    PetscFree(X1);
    PetscFree(X2);
    PetscFree(X);
  }
  if (IsPeriodicY) {
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Initialization routine for tapered meshes does not yet support periodicity");
    ierr = CreatePeriodicKnotVector(iga->Ny,iga->py,iga->Cy,iga->my,iga->Uy,Uy0,Uyf);CHKERRQ(ierr);
    iga->nby -= iga->py;
  } else {


    PetscReal *X1;
    PetscReal *X2;
    PetscInt i;
    PetscReal *X;

    ierr = PetscMalloc((Ny/2+1)*sizeof(PetscReal),&X1);CHKERRQ(ierr);
    ierr = PetscMalloc((Ny/2+1)*sizeof(PetscReal),&X2);CHKERRQ(ierr);

    CreateTaperSetOfPoints(Uy0,0.5*(Uyf+Uy0),fy,Ny/2+1,X1);
    CreateTaperSetOfPoints(Uyf,0.5*(Uyf+Uy0),fy,Ny/2+1,X2);

    ierr = PetscMalloc((Ny+1)*sizeof(PetscReal),&X);CHKERRQ(ierr);

    if( Ny % 2 == 0){

      for(i=0;i<Ny/2+1;i++) {
	X[i]=X1[i];
	X[Ny/2+i]=X2[Ny/2-i];
      }

    }else{

      for(i=0;i<Ny/2+1;i++) {
	X[i]=X1[i];
	X[Ny/2+1+i]=X2[Ny/2-i];
      }

      X[Ny/2]   =  2.0/3.0*X[Ny/2 - 1] + 1.0/3.0*X[Ny/2 + 2];
      X[Ny/2+1] =  1.0/3.0*X[Ny/2 - 1] + 2.0/3.0*X[Ny/2 + 2];

    }

    ierr = CreateKnotVectorFromMesh(iga->Ny,iga->py,iga->Cy,iga->my,iga->Uy,X,Ny+1);

    PetscFree(X1);
    PetscFree(X2);
    PetscFree(X);

  }

  /* compute and store 1d basis functions at gauss points */
  ierr = Compute1DBasisFunctions(iga->ngx, iga->numD, iga->Ux, iga->mx, iga->px, &iga->bdX);CHKERRQ(ierr);
  ierr = Compute1DBasisFunctions(iga->ngy, iga->numD, iga->Uy, iga->my, iga->py, &iga->bdY);CHKERRQ(ierr);

  if (IsPeriodicX) xptype = DMDA_BOUNDARY_PERIODIC;
  if (IsPeriodicY) yptype = DMDA_BOUNDARY_PERIODIC;

  sw = (iga->px>iga->py) ? iga->px : iga->py ;
  ierr = DMDACreate(((PetscObject) dm)->comm,&iga->da_dof); CHKERRQ(ierr);
  ierr = DMDASetDim(iga->da_dof, 2); CHKERRQ(ierr);
  ierr = DMDASetSizes(iga->da_dof,iga->nbx,iga->nby,1); CHKERRQ(ierr);
  ierr = DMDASetDof(iga->da_dof, ndof); CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(iga->da_dof, xptype, yptype, DMDA_BOUNDARY_NONE); CHKERRQ(ierr);
  ierr = DMDASetStencilType(iga->da_dof,DMDA_STENCIL_BOX); CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(iga->da_dof,sw); CHKERRQ(ierr);
  ierr = DMSetFromOptions(iga->da_dof); CHKERRQ(ierr);
  ierr = DMSetUp(iga->da_dof);CHKERRQ(ierr);

  /* Determine how the elements map to processors */

  ierr = DMDAGetLocalInfo(iga->da_dof,&info_dof);CHKERRQ(ierr);
  ierr = BDSetElementOwnership(iga->bdX,iga->Nx,info_dof.xs,info_dof.xs+info_dof.xm-1,iga->px);CHKERRQ(ierr);
  ierr = BDSetElementOwnership(iga->bdY,iga->Ny,info_dof.ys,info_dof.ys+info_dof.ym-1,iga->py);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMIGAInitializeUniform2d"
PetscErrorCode DMIGAInitializeUniform2d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                        PetscInt px,PetscInt Nx,PetscInt Cx,PetscReal Ux0, PetscReal Uxf,PetscBool IsPeriodicX,PetscInt ngx,
                                        PetscInt py,PetscInt Ny,PetscInt Cy,PetscReal Uy0, PetscReal Uyf,PetscBool IsPeriodicY,PetscInt ngy)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;
  DMDABoundaryType xptype = DMDA_BOUNDARY_NONE ;
  DMDABoundaryType yptype = DMDA_BOUNDARY_NONE ;
  PetscInt   sw;
  DMDALocalInfo       info_dof;

  PetscFunctionBegin;
  /* Test C < p */
  if(px <= Cx || py <= Cy){
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Discretization inconsistent: polynomial order must be greater than degree of continuity");
  }

  /* Load constants */
  iga->px = px; iga->py = py;
  iga->Nx = Nx; iga->Ny = Ny;
  iga->Cx = Cx; iga->Cy = Cy;
  iga->ngx = ngx; iga->ngy = ngy;
  iga->IsPeriodicX = IsPeriodicX; iga->IsPeriodicY = IsPeriodicY;
  iga->numD = NumDerivatives;

  /* Knot vector size */
  iga->mx = 2*(iga->px+1);
  iga->my = 2*(iga->py+1);
  iga->mx += (iga->px-iga->Cx)*(iga->Nx-1);
  iga->my += (iga->py-iga->Cy)*(iga->Ny-1);

  /* number of basis functions */
  iga->nbx = iga->mx-iga->px-1;
  iga->nby = iga->my-iga->py-1;

  /* compute knot vectors */
  ierr = PetscMalloc(iga->mx*sizeof(PetscReal), &iga->Ux);CHKERRQ(ierr);
  ierr = PetscMalloc(iga->my*sizeof(PetscReal), &iga->Uy);CHKERRQ(ierr);

  if(IsPeriodicX){
    ierr = CreatePeriodicKnotVector(iga->Nx,iga->px,iga->Cx,iga->mx,iga->Ux,Ux0,Uxf);CHKERRQ(ierr);
    iga->nbx -= iga->px;
  }else{
    ierr = CreateKnotVector(iga->Nx,iga->px,iga->Cx,iga->mx,iga->Ux,Ux0,Uxf);CHKERRQ(ierr);
  }
  if(IsPeriodicY){
    ierr = CreatePeriodicKnotVector(iga->Ny,iga->py,iga->Cy,iga->my,iga->Uy,Uy0,Uyf);CHKERRQ(ierr);
    iga->nby -= iga->py;
  }else{
    ierr = CreateKnotVector(iga->Ny,iga->py,iga->Cy,iga->my,iga->Uy,Uy0,Uyf);CHKERRQ(ierr);
  }

  /* compute and store 1d basis functions at gauss points */
  ierr = Compute1DBasisFunctions(iga->ngx, iga->numD, iga->Ux, iga->mx, iga->px, &iga->bdX);CHKERRQ(ierr);
  ierr = Compute1DBasisFunctions(iga->ngy, iga->numD, iga->Uy, iga->my, iga->py, &iga->bdY);CHKERRQ(ierr);

  if (IsPeriodicX) xptype = DMDA_BOUNDARY_PERIODIC;
  if (IsPeriodicY) yptype = DMDA_BOUNDARY_PERIODIC;

  sw = (iga->px>iga->py) ? iga->px : iga->py ;
  ierr = DMDACreate(((PetscObject) dm)->comm,&iga->da_dof); CHKERRQ(ierr);
  ierr = DMDASetDim(iga->da_dof, 2); CHKERRQ(ierr);
  ierr = DMDASetSizes(iga->da_dof,iga->nbx,iga->nby,1); CHKERRQ(ierr);
  ierr = DMDASetDof(iga->da_dof, ndof); CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(iga->da_dof, xptype, yptype, DMDA_BOUNDARY_NONE); CHKERRQ(ierr);
  ierr = DMDASetStencilType(iga->da_dof,DMDA_STENCIL_BOX); CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(iga->da_dof,sw); CHKERRQ(ierr);
  ierr = DMSetFromOptions(iga->da_dof); CHKERRQ(ierr);
  ierr = DMSetUp(iga->da_dof);CHKERRQ(ierr);

  /* Determine how the elements map to processors */

  ierr = DMDAGetLocalInfo(iga->da_dof,&info_dof);CHKERRQ(ierr);
  ierr = BDSetElementOwnership(iga->bdX,iga->Nx,info_dof.xs,info_dof.xs+info_dof.xm-1,iga->px);CHKERRQ(ierr);
  ierr = BDSetElementOwnership(iga->bdY,iga->Ny,info_dof.ys,info_dof.ys+info_dof.ym-1,iga->py);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMIGAInitializeUniform1d"
PetscErrorCode DMIGAInitializeUniform1d(DM dm,PetscBool IsRational,PetscInt NumDerivatives,PetscInt ndof,
                                        PetscInt px,PetscInt Nx,PetscInt Cx,PetscReal Ux0, PetscReal Uxf,PetscBool IsPeriodicX,PetscInt ngx)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;
  DMDABoundaryType ptype = DMDA_BOUNDARY_NONE ;
  PetscInt   sw;
  DMDALocalInfo       info_dof;

  PetscFunctionBegin;
  /* Test C < p */
  if(px <= Cx){
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Discretization inconsistent: polynomial order must be greater than degree of continuity");
  }

  /* Load constants */
  iga->px = px;
  iga->Nx = Nx;
  iga->Cx = Cx;
  iga->ngx = ngx;
  iga->IsPeriodicX = IsPeriodicX;
  iga->numD = NumDerivatives;

  /* Knot vector size */
  iga->mx = 2*(iga->px+1);
  iga->mx += (iga->px-iga->Cx)*(iga->Nx-1);

  /* number of basis functions */
  iga->nbx = iga->mx-iga->px-1;

  /* compute knot vectors */
  ierr = PetscMalloc(iga->mx*sizeof(PetscReal), &iga->Ux);CHKERRQ(ierr);

  if(IsPeriodicX){
    ierr = CreatePeriodicKnotVector(iga->Nx,iga->px,iga->Cx,iga->mx,iga->Ux,Ux0,Uxf);CHKERRQ(ierr);
    iga->nbx -= iga->px;
  }else{
    ierr = CreateKnotVector(iga->Nx,iga->px,iga->Cx,iga->mx,iga->Ux,Ux0,Uxf);CHKERRQ(ierr);
  }

  /* compute and store 1d basis functions at gauss points */
  ierr = Compute1DBasisFunctions(iga->ngx, iga->numD, iga->Ux, iga->mx, iga->px, &iga->bdX);CHKERRQ(ierr);

  if (IsPeriodicX) ptype = DMDA_BOUNDARY_PERIODIC;

  sw = iga->px;
  ierr = DMDACreate(((PetscObject) dm)->comm,&iga->da_dof); CHKERRQ(ierr);
  ierr = DMDASetDim(iga->da_dof, 1); CHKERRQ(ierr);
  ierr = DMDASetSizes(iga->da_dof,iga->nbx,1,1); CHKERRQ(ierr);
  ierr = DMDASetDof(iga->da_dof, ndof); CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(iga->da_dof, ptype, ptype, ptype); CHKERRQ(ierr);
  ierr = DMDASetStencilType(iga->da_dof,DMDA_STENCIL_BOX); CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(iga->da_dof,sw); CHKERRQ(ierr);
  ierr = DMSetFromOptions(iga->da_dof); CHKERRQ(ierr);
  ierr = DMSetUp(iga->da_dof);CHKERRQ(ierr);

  /* Determine how the elements map to processors */

  ierr = DMDAGetLocalInfo(iga->da_dof,&info_dof);CHKERRQ(ierr);
  ierr = BDSetElementOwnership(iga->bdX,iga->Nx,info_dof.xs,info_dof.xs+info_dof.xm-1,iga->px);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMIGAKnotRefine3d"
PetscErrorCode DMIGAKnotRefine3d(DM dm,PetscInt kx,PetscReal *Ux,PetscInt ky,PetscReal *Uy,PetscInt kz,PetscReal *Uz,DM iga_new)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* are the knots you are trying to insert feasible? */
  ierr = CheckKnots(iga->mx,iga->Ux,kx,Ux);CHKERRQ(ierr);
  ierr = CheckKnots(iga->my,iga->Uy,ky,Uy);CHKERRQ(ierr);
  ierr = CheckKnots(iga->mz,iga->Uz,kz,Uz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMIGAKnotRefine2d"
PetscErrorCode DMIGAKnotRefine2d(DM dm,PetscInt kx,PetscReal *Ux,PetscInt ky,PetscReal *Uy,DM iga_new)
{
  DM_IGA        *iga = (DM_IGA *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* are the knots you are trying to insert feasible? */
  ierr = CheckKnots(iga->mx,iga->Ux,kx,Ux);CHKERRQ(ierr);
  ierr = CheckKnots(iga->my,iga->Uy,ky,Uy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BDCreate"
PetscErrorCode BDCreate(BD *bd,PetscInt numD,PetscInt p,PetscInt numGP,PetscInt numEl)
{
  BD bdd = PETSC_NULL;
  PetscInt i,j;

  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(BasisData1D),bd); CHKERRQ(ierr);
  bdd = *bd;

  ierr = PetscMalloc(numEl*numGP*sizeof(GP),&(bdd->data));CHKERRQ(ierr);
  for(i=0;i<numEl;i++){
    for(j=0;j<numGP;j++){
      ierr = PetscMalloc((p+1)*(numD+1)*sizeof(PetscReal),&(bdd->data[i*numGP+j].basis));CHKERRQ(ierr);
    }
  }

  bdd->numD = numD;
  bdd->p = p;
  bdd->numGP = numGP;
  bdd->numEl = numEl;

  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "BDDestroy"
PetscErrorCode BDDestroy(BD *bd)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscFunctionBegin;
  PetscValidPointer(bd,1);
  if (!(*bd)) PetscFunctionReturn(0);
  for(i=0;i<(*bd)->numEl;i++) {
    for(j=0;j<(*bd)->numGP;j++) {
      ierr = PetscFree((*bd)->data[i*(*bd)->numGP+j].basis);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree((*bd)->data); CHKERRQ(ierr);
  ierr = PetscFree(*bd); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BDGetBasis"
PetscErrorCode BDGetBasis(BD bd, PetscInt iel, PetscInt igp, PetscInt ib, PetscInt ider, PetscReal *basis)
{
  PetscFunctionBegin;

  *basis = bd->data[iel*bd->numGP+igp].basis[ider*(bd->p+1)+ib];
  /*  *basis = bd->arrayX[iel][igp*(bd->numD+1)*(bd->p+1) + ider*(bd->p+1) + ib]; */

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BDSetBasis"
PetscErrorCode BDSetBasis(BD bd, PetscInt iel, PetscInt igp, PetscInt ib, PetscInt ider, PetscReal basis)
{
  PetscFunctionBegin;

  bd->data[iel*bd->numGP+igp].basis[ider*(bd->p+1)+ib] = basis;
  /* bd->arrayX[iel][igp*(bd->numD+1)*(bd->p+1) + ider*(bd->p+1) + ib] = basis; */

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BDSetGaussPt"
PetscErrorCode BDSetGaussPt(BD bd, PetscInt iel, PetscInt igp, PetscReal gp)
{
  PetscFunctionBegin;

  bd->data[iel*bd->numGP+igp].gx = gp;
  /* bd->arrayX[iel][(bd->numD+1)*(bd->p+1)*(bd->numGP) + igp*(2)] = gp; */

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BDSetGaussWt"
PetscErrorCode BDSetGaussWt(BD bd, PetscInt iel, PetscInt igp, PetscReal gw)
{
  PetscFunctionBegin;

  bd->data[iel*bd->numGP+igp].gw = gw;
  /* bd->arrayX[iel][(bd->numD+1)*(bd->p+1)*(bd->numGP) + igp*(2) + 1] = gw; */

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BDSetBasisOffset"
PetscErrorCode BDSetBasisOffset(BD bd, PetscInt iel, PetscInt bOffset)
{
  PetscInt igp = 0;

  PetscFunctionBegin;
  bd->data[iel*bd->numGP+igp].offset = bOffset;
  /* (bd->numD+1)*(bd->p+1)*(bd->numGP) + (bd->numGP)*(2) */
  /* bd->arrayX[iel][((bd->numD+1)*(bd->p+1)+2)*(bd->numGP)] = (PetscReal)bOffset; */

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BDGetGaussPt"
PetscErrorCode BDGetGaussPt(BD bd, PetscInt iel, PetscInt igp, PetscReal *gp)
{
  PetscFunctionBegin;
  *gp = bd->data[iel*bd->numGP+igp].gx;
  /* *gp = bd->arrayX[iel][(bd->numD+1)*(bd->p+1)*(bd->numGP) + igp*(2)]; */

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BDGetGaussWt"
PetscErrorCode BDGetGaussWt(BD bd, PetscInt iel, PetscInt igp, PetscReal *gw)
{
  PetscFunctionBegin;

  *gw = bd->data[iel*bd->numGP+igp].gw;
  /* *gw = bd->arrayX[iel][(bd->numD+1)*(bd->p+1)*(bd->numGP) + igp*(2) + 1]; */

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BDGetBasisOffset"
PetscErrorCode BDGetBasisOffset(BD bd, PetscInt iel, PetscInt *bOffset)
{
  PetscInt igp = 0;

  PetscFunctionBegin;
  *bOffset = bd->data[iel*bd->numGP+igp].offset;
  /* (bd->numD+1)*(bd->p+1)*(bd->numGP) + (bd->numGP)*(2) */
  /* *bOffset = (int)bd->arrayX[iel][((bd->numD+1)*(bd->p+1)+2)*(bd->numGP)]; */

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BDSetElementOwnership"
PetscErrorCode BDSetElementOwnership(BD bd,PetscInt nel,PetscInt dof_b,PetscInt dof_e,PetscInt p)
{
  PetscInt i,boffset;

  PetscFunctionBegin;
  bd->cont_b = nel; bd->cont_e = -1;
  bd->own_b = nel; bd->own_e = -1;
  for(i=0;i<nel;i++){ /* loop thru elements */

    BDGetBasisOffset(bd,i,&boffset); /* left-most dof */

    if(boffset >= dof_b && boffset <= dof_e){ /* I own this element */
      if(i < bd->own_b) bd->own_b = i;
      if(i > bd->own_e) bd->own_e = i;
    }

    if((boffset >= dof_b && boffset <= dof_e)||(boffset+p >= dof_b && boffset+p <= dof_e)){ /* This element contributes to me */
      if(i < bd->cont_b) bd->cont_b = i;
      if(i > bd->cont_e) bd->cont_e = i;
    }

  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Compute1DBasisFunctions"
PetscErrorCode Compute1DBasisFunctions(PetscInt numGP, PetscInt numD, PetscReal *U, PetscInt m, PetscInt porder, BD *bd1D)
{
  /* The idea of the 1d DA is to hold all information for a particular
     non-zero span of the knot vector. Here I will need to pass in the
     knot vector of which all processors will have a copy. Then I will
     need to determine which span in the knot vector corresponds to
     the nonzero spans represented in the DA */

  PetscErrorCode ierr;
  PetscReal *X, *W;
  PetscReal **Nu;
  PetscInt    i;
  PetscInt numEl = 0;
  BD bd;
  PetscInt k,j,l;

  PetscFunctionBegin;
  /* Setup gauss points */
  ierr = PetscMalloc2(numGP,PetscReal,&X,numGP,PetscReal,&W);CHKERRQ(ierr);
  ierr = SetupGauss1D(numGP,X,W);CHKERRQ(ierr);

  /* create space to get basis functions */
  ierr = PetscMalloc((numD+1)*sizeof(PetscReal *), &Nu);CHKERRQ(ierr);
  for(i = 0; i <= numD; ++i) {
    ierr = PetscMalloc((porder+1)*sizeof(PetscReal), &Nu[i]);CHKERRQ(ierr);
  }

  /* count the number of elements */
  for(i = 0; i < m-1; ++i) {
    PetscReal du = (U[i+1]-U[i]);
    if(du > 1.0e-13) numEl++;
  }

  /* initialize the bd */
  ierr = BDCreate(&bd,numD,porder,numGP,numEl);CHKERRQ(ierr);

  /* precompute the basis */
  for(i = 0; i < numEl; ++i) {
    PetscInt uspan = FindSpan(U,m,i,porder);
    PetscReal du = (U[uspan+1]-U[uspan]);

    /* here I am storing the first global basis number in 1d */
    ierr = BDSetBasisOffset(bd,i,uspan-porder);CHKERRQ(ierr);

    for(k = 0; k < numGP; ++k) {
      PetscReal u = (X[k]+1.0)*0.5*du + U[uspan];

      /* and also storing the gauss point and its weight (sneaky and flagrant abuse of DAs) */
      ierr = BDSetGaussPt(bd,i,k,u);CHKERRQ(ierr);
      ierr = BDSetGaussWt(bd,i,k,W[k]*0.5*du);CHKERRQ(ierr); /* note includes detJ for [-1:1]->[U[span]:U[span+1]] */
      ierr = GetDersBasisFuns(uspan,u,porder,U,Nu,numD);CHKERRQ(ierr);

      /* load values */
      for(j = 0; j < porder+1; ++j) {
	for(l = 0; l <= numD; ++l) {
          ierr = BDSetBasis(bd,i,k,j,l,Nu[l][j]);CHKERRQ(ierr);
	}
      }

    }
  }

  *bd1D = bd;

  /* Cleanup */
  for(i = 0; i <= numD; ++i) {
    ierr = PetscFree(Nu[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(Nu);CHKERRQ(ierr);
  ierr = PetscFree2(X, W);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscInt FindSpan(PetscReal *U,PetscInt m,PetscInt j,PetscInt p)
{
  /* i is the span not counting zero spans, return the span including */

  PetscInt i,id = -1;
  for(i=p;i<m-p-1;i++)
    if(fabs(U[i+1]-U[i]) > 1.0e-14)
      {
	id += 1;
	if(id == j) return i;
      }

  return -1;
}

#undef __FUNCT__
#define __FUNCT__ "GetDersBasisFuns"
PetscErrorCode GetDersBasisFuns(PetscInt i,PetscReal u,PetscInt p,PetscReal *U, PetscReal **N,PetscInt nd)
{
  /*  This function calculates the non-vanishing spline basis functions
      and their derivatives. See the notes from the above function.
      i <-- knot span index
      u <-- parameter value for evaluation
      p <-- 1D polynomial order
      U <-- knot vector
      N --> (nd+1,p+1) matrix of function and derivative
      evaluations. The first row are the function evaluations,
      equivalent to the above function. The second row are the
      derivatives. */
  PetscInt j,k,r,s1,s2,rk,pk,j1,j2;
  PetscReal saved,temp,d;
  PetscErrorCode ierr;

  PetscReal **a, **ndu;
  PetscReal *left, *right;
  ierr = PetscMalloc(2*sizeof(PetscReal *), &a);CHKERRQ(ierr);
  ierr = PetscMalloc((p+1)*sizeof(PetscReal *), &ndu);CHKERRQ(ierr);
  ierr = PetscMalloc2((p+1),PetscReal,&left,(p+1),PetscReal,&right);CHKERRQ(ierr);

  PetscFunctionBegin;
  for(j=0;j<(p+1);j++)
  {
    ierr = PetscMalloc((p+1)*sizeof(PetscReal), &ndu[j]);CHKERRQ(ierr);
    for(k=0;k<p+1;k++) {
      ndu[j][k] = 0.0;
    }
  }
  for(j=0;j<2;j++)
  {
    ierr = PetscMalloc((p+1)*sizeof(PetscReal), &a[j]);CHKERRQ(ierr);
    for(k=0;k<(p+1);k++) {
      a[j][k] = 0.0;
    }
  }
  ndu[0][0] = 1.0;
  for (j = 1; j <= p; j++)
  {
    left[j] = u - U[i + 1 - j];
    right[j] = U[i + j] - u;
    saved = 0.0;
    for (r = 0; r < j; r++)
    {
      /* Lower triangle */
      ndu[j][r] = right[r+1] + left[j-r];
      temp = ndu[r][j-1] / ndu[j][r];

      /* Upper triangle */
      ndu[r][j] = saved + right[r+1]*temp;
      saved = left[j-r]*temp;
    }
    ndu[j][j] = saved;
  }

  for (j = 0; j <= p; j++) N[0][j] = ndu[j][p];
  for (r = 0; r <= p; r++)
  {
    s1 = 0; s2 = 1; a[0][0] = 1.0;
    for (k = 1; k <= nd; k++)
    {
      d = 0.0;
      rk = r - k; pk = p - k;
      if (r >= k)
      {
        a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
        d = a[s2][0] * ndu[rk][pk];
      }
      if (rk >= -1) {
        j1 = 1;
      } else {
        j1 = -rk;
      }
      if (r - 1 <= pk) {
        j2 = k - 1;
      } else {
        j2 = p - r;
      }
      for (j = j1; j <= j2; j++)
      {
        a[s2][j] = ( a[s1][j] - a[s1][j - 1] )/ndu[pk + 1][rk + j];
        d += a[s2][j]*ndu[rk + j][pk];
      }
      if (r <= pk)
      {
        a[s2][k] = - a[s1][k - 1]/ndu[pk + 1][r];
        d += a[s2][k]*ndu[r][pk];
      }
      N[k][r] = d;
      j = s1;
      s1 = s2;
      s2 = j;
    }
  }

  /* Multiply through by correct factors */
  r = p;
  for(k = 1; k <= nd; k++)
  {
    for(j = 0; j <= p; j++) N[k][j] *=r;
    r *= (p-k);
  }

  for(j=0;j<(p+1);j++)
    ierr = PetscFree(ndu[j]);CHKERRQ(ierr);
  for(j=0;j<2;j++)
    ierr = PetscFree(a[j]);CHKERRQ(ierr);
  ierr = PetscFree(a);CHKERRQ(ierr);
  ierr = PetscFree(ndu);CHKERRQ(ierr);
  ierr = PetscFree2(left, right);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupGauss1D"
PetscErrorCode SetupGauss1D(PetscInt n,PetscReal *X,PetscReal *W)
{
  PetscFunctionBegin;
  switch(n)
  {
  case 1: /* porder = 1 */
    X[0] = 0.0;
    W[0] = 2.0;
    break;
  case 2: /* porder = 3 */
    X[0] = -0.577350269189626;
    X[1] = -X[0];
    W[0] = 1.0;
    W[1] = 1.0;
    break;
  case 3: /* porder = 5 */
    X[0] = -0.774596669241483;
    X[1] = 0.0;
    X[2] = -X[0];
    W[0] = 0.555555555555556;
    W[1] = 0.888888888888889;
    W[2] = W[0];
    break;
  case 4: /* porder = 7 */
    X[0] = -.86113631159405257524;
    X[1] = -.33998104358485626481;
    X[2] = -X[1];
    X[3] = -X[0];
    W[0] = .34785484513745385736;
    W[1] = .65214515486254614264;
    W[2] = W[1];
    W[3] = W[0];
    break;
  case 5: /* porder = 9 */
    X[0] = -.90617984593866399282;
    X[1] = -.53846931010568309105;
    X[2] = 0.0;
    X[3] = .53846931010568309105;
    X[4] = .90617984593866399282;
    W[0] = .23692688505618908749;
    W[1] = .47862867049936646808;
    W[2] = .56888888888888888888;
    W[3] = .47862867049936646808;
    W[4] = .23692688505618908749;
    break;
  case 6:
    X[0] = -.9324695142031520 ;
    X[1] = -.6612093864662645 ;
    X[2] = -.2386191860831969 ;
    X[3] = .2386191860831969 ;
    X[4] = .6612093864662645 ;
    X[5] = .9324695142031520 ;
    W[0] = .1713244923791703 ;
    W[1] = .3607615730481386 ;
    W[2] = .4679139345726911 ;
    W[3] = .4679139345726911 ;
    W[4] = .3607615730481386 ;
    W[5] = .1713244923791703 ;
    break ;
  case 7:
    X[0] = -.9491079123427585 ;
    X[1] = -.7415311855993944 ;
    X[2] = -.4058451513773972 ;
    X[3] = 0.0 ;
    X[4] = .4058451513773972 ;
    X[5] = .7415311855993944 ;
    X[6] = .9491079123427585 ;
    W[0] = .1294849661688697 ;
    W[1] = .2797053914892767 ;
    W[2] = .3818300505051189 ;
    W[3] = .4179591836734694 ;
    W[4] = .3818300505051189 ;
    W[5] = .2797053914892767 ;
    W[6] = .1294849661688697 ;
    break ;
  case 8:
    X[0] = -.9602898564975362 ;
    X[1] = -.7966664774136267 ;
    X[2] = -.5255324099163290 ;
    X[3] = -.1834346424956498 ;
    X[4] = .1834346424956498 ;
    X[5] = .5255324099163290 ;
    X[6] = .7966664774136267 ;
    X[7] = .9602898564975362 ;
    W[0] = .1012285362903763 ;
    W[1] = .2223810344533745 ;
    W[2] = .3137066458778873 ;
    W[3] = .3626837833783620 ;
    W[4] = .3626837833783620 ;
    W[5] = .3137066458778873 ;
    W[6] = .2223810344533745 ;
    W[7] = .1012285362903763 ;
    break ;
  case 9:
    X[0] = -.9681602395076261 ;
    X[1] = -.8360311073266358 ;
    X[2] = -.6133714327005904 ;
    X[3] = -.3242534234038089 ;
    X[4] = 0.0 ;
    X[5] = .3242534234038089 ;
    X[6] = .6133714327005904 ;
    X[7] = .8360311073266358 ;
    X[8] = .9681602395076261 ;
    W[0] = .0812743883615744 ;
    W[1] = .1806481606948574 ;
    W[2] = .2606106964029354 ;
    W[3] = .3123470770400029 ;
    W[4] = .3302393550012598 ;
    W[5] = .3123470770400028 ;
    W[6] = .2606106964029355 ;
    W[7] = .1806481606948574 ;
    W[8] = .0812743883615744 ;
    break ;
  case 15:
    /* Generated in Mathematica:
       << NumericalDifferentialEquationAnalysis`
       GaussianQuadratureWeights[15, -1, 1, 25] here I overkilled
       the precision to make sure we get as much as possible. */
    X[0] = -0.9879925180204854284896 ; W[0] = 0.0307532419961172683546284 ;
    X[1] = -0.9372733924007059043078 ; W[1] = 0.0703660474881081247092674 ;
    X[2] = -0.8482065834104272162006 ; W[2] = 0.1071592204671719350118695 ;
    X[3] = -0.7244177313601700474162 ; W[3] = 0.139570677926154314447805 ;
    X[4] = -0.5709721726085388475372 ; W[4] = 0.166269205816993933553201 ;
    X[5] = -0.3941513470775633698972 ; W[5] = 0.186161000015562211026801 ;
    X[6] = -0.2011940939974345223006 ; W[6] = 0.198431485327111576456118 ;
    X[7] = 0.0000000000000000000000  ; W[7] = 0.202578241925561272880620 ;
    X[8] = 0.201194093997434522301 ; W[8] = 0.198431485327111576456118 ;
    X[9] = 0.394151347077563369897 ; W[9] = 0.186161000015562211026801 ;
    X[10] = 0.570972172608538847537 ; W[10] = 0.166269205816993933553201 ;
    X[11] = 0.724417731360170047416 ; W[11] = 0.139570677926154314447805 ;
    X[12] = 0.848206583410427216201 ; W[12] = 0.1071592204671719350118695 ;
    X[13] = 0.937273392400705904308 ; W[13] = 0.0703660474881081247092674 ;
    X[14] = 0.987992518020485428490 ; W[14] = 0.0307532419961172683546284 ;
    break ;
  case 20:
    /* Generated in Mathematica:
       << NumericalDifferentialEquationAnalysis`
       GaussianQuadratureWeights[20, -1, 1, 25] here I overkilled
       the precision to make sure we get as much as possible. */
    X[0] = -0.9931285991850949247861 ; W[0] = 0.0176140071391521183118620 ;
    X[1] = -0.9639719272779137912677 ; W[1] = 0.0406014298003869413310400 ;
    X[2] = -0.9122344282513259058678 ; W[2] = 0.0626720483341090635695065 ;
    X[3] = -0.839116971822218823395 ; W[3] = 0.083276741576704748724758 ;
    X[4] = -0.746331906460150792614 ; W[4] = 0.101930119817240435036750 ;
    X[5] = -0.636053680726515025453 ; W[5] = 0.118194531961518417312377 ;
    X[6] = -0.510867001950827098004 ; W[6] = 0.131688638449176626898494 ;
    X[7] = -0.373706088715419560673 ; W[7] = 0.142096109318382051329298 ;
    X[8] = -0.227785851141645078080 ; W[8] = 0.149172986472603746787829 ;
    X[9] = -0.076526521133497333755 ; W[9] = 0.152753387130725850698084 ;
    X[10] = 0.076526521133497333755 ; W[10] = 0.152753387130725850698084 ;
    X[11] = 0.227785851141645078080 ; W[11] = 0.149172986472603746787829 ;
    X[12] = 0.373706088715419560673 ; W[12] = 0.142096109318382051329298 ;
    X[13] = 0.510867001950827098004 ; W[13] = 0.131688638449176626898494 ;
    X[14] = 0.636053680726515025453 ; W[14] = 0.118194531961518417312377 ;
    X[15] = 0.746331906460150792614 ; W[15] = 0.101930119817240435036750 ;
    X[16] = 0.839116971822218823395 ; W[16] = 0.083276741576704748724758 ;
    X[17] = 0.912234428251325905868 ; W[17] = 0.0626720483341090635695065 ;
    X[18] = 0.963971927277913791268 ; W[18] = 0.0406014298003869413310400 ;
    X[19] = 0.993128599185094924786 ; W[19] = 0.0176140071391521183118620 ;
    break ;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unimplemented number of gauss points %d!", n);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateKnotVector"
PetscErrorCode CreateKnotVector(PetscInt N,PetscInt p,PetscInt C,PetscInt m, PetscReal *U,PetscReal U0,PetscReal Uf)
{
  PetscInt  i,j;
  PetscReal dU;

  PetscFunctionBegin;

  dU = (Uf-U0)/N;
  for(i=0;i<(N-1);i++) /* insert N-1 knots */
    for(j=0;j<(p-C);j++) /* p-C times */
      U[(p+1) + i*(p-C) + j] = U0 + (i+1)*dU;

  for(i=0;i<(p+1);i++) /* open part */
  {
    U[i] = U0;
    U[m-i-1] = Uf;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePeriodicKnotVector"
PetscErrorCode CreatePeriodicKnotVector(PetscInt N,PetscInt p,PetscInt C,PetscInt m, PetscReal *U,PetscReal U0,PetscReal Uf)
{
  PetscInt  i,j;
  PetscReal dU;

  PetscFunctionBegin;

  dU = (Uf-U0)/N;
  for(i=0;i<(N+1);i++) /* insert N+1 knots */
    for(j=0;j<(p-C);j++) /* p-C times */
      U[(C+1) + i*(p-C) + j] = U0 + i*dU;

  for(i=0;i<(C+1);i++) /* periodic part */
  {
    U[i] = U0 - (Uf - U[(m-1-p)-(C+1)+i]);
    U[m-(C+1)+i] = Uf + (U[p+1+i] - U0);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateKnotVectorFromMesh"
PetscErrorCode CreateKnotVectorFromMesh(PetscInt N,PetscInt p,PetscInt C,PetscInt m, PetscReal *U,PetscReal *X,PetscInt nX)
{
  PetscInt i,j,countU=0;

  PetscFunctionBegin;
  for(i=0;i<nX;i++){

    if(i==0 || i==nX-1){
      for(j=0;j<p+1;j++) {
	U[countU] = X[i];
	countU += 1;
      }
    }else{
      for(j=0;j<p-C;j++) {
	U[countU] = X[i];
	countU += 1;
      }
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateTaperSetOfPoints"
PetscErrorCode CreateTaperSetOfPoints(PetscReal Xbegin,PetscReal Xend,PetscReal f,PetscInt N,PetscReal *X)
{
  /* N is the number of points, we will need number of spans, Ns */
  PetscInt Ns=N-1;
  PetscReal sum=0.0;
  PetscInt i;
  PetscReal dX;

  PetscFunctionBegin;
  for(i=0;i<Ns;i++){
    sum += pow(f,(PetscReal)i);
  }

  dX = (Xend-Xbegin)/sum;
  X[0] = Xbegin;
  for(i=1;i<N;i++){
    X[i] = X[i-1] + dX*pow(f,(PetscReal)(i-1));
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckKnots"
PetscErrorCode CheckKnots(PetscInt m,PetscReal *U,PetscInt k,PetscReal *Uadd)
{
  /* Check the knots we are trying to insert into the vector U */

  /* 1) check that they are U[0] < Uadd[i] < U[m-1] */
  PetscInt j;

  PetscFunctionBegin;
  for(j=0;j<k;j++)
    if(Uadd[j] < U[0] || Uadd[j] > U[m-1])
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Inserted knots beyond original knot vector limits");

  /* 2) I am lazy so I am not thinking about more that could go wrong */

  PetscFunctionReturn(0);
}
