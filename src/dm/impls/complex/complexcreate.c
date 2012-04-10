#define PETSCDM_DLL
#include <petsc-private/compleximpl.h>    /*I   "petscdmcomplex.h"   I*/
#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "DMSetFromOptions_Complex"
PetscErrorCode  DMSetFromOptions_Complex(DM dm)
{
  DM_Complex    *mesh = (DM_Complex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscOptionsHead("DMComplex Options");CHKERRQ(ierr);
    /* Handle DMComplex refinement */
    /* Handle associated vectors */
    /* Handle viewing */
    ierr = PetscOptionsBool("-dm_complex_print_set_values", "Output all set values info", "DMView", PETSC_FALSE, &mesh->printSetValues, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateSquareBoundary"
/*
 Simple square boundary:

 18--5-17--4--16
  |     |     |
  6    10     3
  |     |     |
 19-11-20--9--15
  |     |     |
  7     8     2
  |     |     |
 12--0-13--1--14
*/
PetscErrorCode DMComplexCreateSquareBoundary(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt edges[])
{
  DM_Complex    *mesh        = (DM_Complex *) dm->data;
  PetscInt       numVertices = (edges[0]+1)*(edges[1]+1);
  PetscInt       numEdges    = edges[0]*(edges[1]+1) + (edges[0]+1)*edges[1];
  PetscScalar   *coords;
  PetscInt       coordSize;
  PetscMPIInt    rank;
  PetscInt       v, vx, vy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject) dm)->comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    PetscInt e, ex, ey;

    ierr = DMComplexSetChart(dm, 0, numEdges+numVertices);CHKERRQ(ierr);
    for(e = 0; e < numEdges; ++e) {
      ierr = DMComplexSetConeSize(dm, e, 2);CHKERRQ(ierr);
    }
    ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
    for(vy = 0; vy <= edges[1]; vy++) {
      for(ex = 0; ex < edges[0]; ex++) {
        PetscInt edge    = vy*edges[0]     + ex;
        PetscInt vertex  = vy*(edges[0]+1) + ex + numEdges;
        PetscInt cone[2] = {vertex, vertex+1};

        ierr = DMComplexSetCone(dm, edge, cone);CHKERRQ(ierr);
        if ((vy == 0) || (vy == edges[1])) {
          ierr = DMComplexSetLabelValue(dm, "marker", edge,    1);CHKERRQ(ierr);
          ierr = DMComplexSetLabelValue(dm, "marker", cone[0], 1);CHKERRQ(ierr);
          if (ex == edges[0]-1) {
            ierr = DMComplexSetLabelValue(dm, "marker", cone[1], 1);CHKERRQ(ierr);
          }
        }
      }
    }
    for(vx = 0; vx <= edges[0]; vx++) {
      for(ey = 0; ey < edges[1]; ey++) {
        PetscInt edge    = vx*edges[1] + ey + edges[0]*(edges[1]+1);
        PetscInt vertex  = ey*(edges[0]+1) + vx + numEdges;
        PetscInt cone[2] = {vertex, vertex+edges[0]+1};

        ierr = DMComplexSetCone(dm, edge, cone);CHKERRQ(ierr);
        if ((vx == 0) || (vx == edges[0])) {
          ierr = DMComplexSetLabelValue(dm, "marker", edge,    1);CHKERRQ(ierr);
          ierr = DMComplexSetLabelValue(dm, "marker", cone[0], 1);CHKERRQ(ierr);
          if (ey == edges[1]-1) {
            ierr = DMComplexSetLabelValue(dm, "marker", cone[1], 1);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = DMComplexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMComplexStratify(dm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = PetscSectionSetChart(mesh->coordSection, numEdges, numEdges + numVertices);CHKERRQ(ierr);
  for(v = numEdges; v < numEdges+numVertices; ++v) {
    ierr = PetscSectionSetDof(mesh->coordSection, v, 2);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(mesh->coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(mesh->coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecSetSizes(mesh->coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(mesh->coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(mesh->coordinates, &coords);CHKERRQ(ierr);
  for(vy = 0; vy <= edges[1]; ++vy) {
    for(vx = 0; vx <= edges[0]; ++vx) {
      coords[(vy*(edges[0]+1)+vx)*2+0] = lower[0] + ((upper[0] - lower[0])/edges[0])*vx;
      coords[(vy*(edges[0]+1)+vx)*2+1] = lower[1] + ((upper[1] - lower[1])/edges[1])*vy;
    }
  }
  ierr = VecRestoreArray(mesh->coordinates, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateCubeBoundary"
/*
 Simple cubic boundary:

     2-------3
    /|      /|
   6-------7 |
   | |     | |
   | 0-----|-1
   |/      |/
   4-------5
*/
PetscErrorCode DMComplexCreateCubeBoundary(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt faces[])
{
  DM_Complex    *mesh        = (DM_Complex *) dm->data;
  PetscInt       numVertices = (faces[0]+1)*(faces[1]+1)*(faces[2]+1);
  PetscInt       numFaces    = 6;
  PetscScalar   *coords;
  PetscInt       coordSize;
  PetscMPIInt    rank;
  PetscInt       v, vx, vy, vz;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((faces[0] < 1) || (faces[1] < 1) || (faces[2] < 1)) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Must have at least 1 face per side");
  if ((faces[0] > 1) || (faces[1] > 1) || (faces[2] > 1)) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Currently can't handle more than 1 face per side");
  ierr = PetscMalloc(numVertices*2 * sizeof(PetscReal), &coords);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject) dm)->comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    PetscInt f;

    ierr = DMComplexSetChart(dm, 0, numFaces+numVertices);CHKERRQ(ierr);
    for(f = 0; f < numFaces; ++f) {
      ierr = DMComplexSetConeSize(dm, f, 4);CHKERRQ(ierr);
    }
    ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
    for(v = 0; v < numFaces+numVertices; ++v) {
      ierr = DMComplexSetLabelValue(dm, "marker", v, 1);CHKERRQ(ierr);
    }
    { /* Side 0 (Front) */
      PetscInt cone[4] = {numFaces+4, numFaces+5, numFaces+7, numFaces+6};
      ierr = DMComplexSetCone(dm, 0, cone);CHKERRQ(ierr);
    }
    { /* Side 1 (Back) */
      PetscInt cone[4] = {numFaces+1, numFaces+0, numFaces+2, numFaces+3};
      ierr = DMComplexSetCone(dm, 1, cone);CHKERRQ(ierr);
    }
    { /* Side 2 (Bottom) */
      PetscInt cone[4] = {numFaces+0, numFaces+1, numFaces+5, numFaces+4};
      ierr = DMComplexSetCone(dm, 2, cone);CHKERRQ(ierr);
    }
    { /* Side 3 (Top) */
      PetscInt cone[4] = {numFaces+6, numFaces+7, numFaces+3, numFaces+2};
      ierr = DMComplexSetCone(dm, 3, cone);CHKERRQ(ierr);
    }
    { /* Side 4 (Left) */
      PetscInt cone[4] = {numFaces+0, numFaces+4, numFaces+6, numFaces+2};
      ierr = DMComplexSetCone(dm, 4, cone);CHKERRQ(ierr);
    }
    { /* Side 5 (Right) */
      PetscInt cone[4] = {numFaces+5, numFaces+1, numFaces+3, numFaces+7};
      ierr = DMComplexSetCone(dm, 5, cone);CHKERRQ(ierr);
    }
  }
  ierr = DMComplexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMComplexStratify(dm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = PetscSectionSetChart(mesh->coordSection, numFaces, numFaces + numVertices);CHKERRQ(ierr);
  for(v = numFaces; v < numFaces+numVertices; ++v) {
    ierr = PetscSectionSetDof(mesh->coordSection, v, 3);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(mesh->coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(mesh->coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecSetSizes(mesh->coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(mesh->coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(mesh->coordinates, &coords);CHKERRQ(ierr);
  for(vz = 0; vz <= faces[2]; ++vz) {
    for(vy = 0; vy <= faces[1]; ++vy) {
      for(vx = 0; vx <= faces[0]; ++vx) {
        coords[((vz*(faces[1]+1)+vy)*(faces[0]+1)+vx)*3+0] = lower[0] + ((upper[0] - lower[0])/faces[0])*vx;
        coords[((vz*(faces[1]+1)+vy)*(faces[0]+1)+vx)*3+1] = lower[1] + ((upper[1] - lower[1])/faces[1])*vy;
        coords[((vz*(faces[1]+1)+vy)*(faces[0]+1)+vx)*3+2] = lower[2] + ((upper[2] - lower[2])/faces[2])*vz;
      }
    }
  }
  ierr = VecRestoreArray(mesh->coordinates, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreateBoxMesh"
PetscErrorCode DMComplexCreateBoxMesh(MPI_Comm comm, PetscInt dim, PetscBool interpolate, DM *dm) {
  DM             boundary;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm, 4);
  ierr = DMCreate(comm, &boundary);CHKERRQ(ierr);
  PetscValidLogicalCollectiveInt(boundary,dim,2);
  ierr = DMSetType(boundary, DMCOMPLEX);CHKERRQ(ierr);
  ierr = DMComplexSetDimension(boundary, dim-1);CHKERRQ(ierr);
  switch(dim) {
  case 2:
  {
    PetscReal lower[2] = {0.0, 0.0};
    PetscReal upper[2] = {1.0, 1.0};
    PetscInt  edges[2] = {2, 2};

    ierr = DMComplexCreateSquareBoundary(boundary, lower, upper, edges);CHKERRQ(ierr);
    break;
  }
  case 3:
  {
    PetscReal lower[3] = {0.0, 0.0, 0.0};
    PetscReal upper[3] = {1.0, 1.0, 1.0};
    PetscInt  faces[3] = {1, 1, 1};

    ierr = DMComplexCreateCubeBoundary(boundary, lower, upper, faces);CHKERRQ(ierr);
    break;
  }
  default:
    SETERRQ1(comm, PETSC_ERR_SUP, "Dimension not supported: %d", dim);
  }
  ierr = DMComplexGenerate(boundary, PETSC_NULL, interpolate, dm);CHKERRQ(ierr);
  ierr = DMDestroy(&boundary);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* External function declarations here */
extern PetscErrorCode DMGlobalToLocalBegin_Complex(DM dm, Vec g, InsertMode mode, Vec l);
extern PetscErrorCode DMGlobalToLocalEnd_Complex(DM dm, Vec g, InsertMode mode, Vec l);
extern PetscErrorCode DMLocalToGlobalBegin_Complex(DM dm, Vec l, InsertMode mode, Vec g);
extern PetscErrorCode DMLocalToGlobalEnd_Complex(DM dm, Vec l, InsertMode mode, Vec g);
extern PetscErrorCode DMCreateGlobalVector_Complex(DM dm, Vec *gvec);
extern PetscErrorCode DMCreateLocalVector_Complex(DM dm, Vec *lvec);
extern PetscErrorCode DMCreateLocalToGlobalMapping_Complex(DM dm);
extern PetscErrorCode DMCreateFieldIS_Complex(DM dm, PetscInt *numFields, char ***names, IS **fields);
extern PetscErrorCode DMCreateInterpolation_Complex(DM dmCoarse, DM dmFine, Mat *interpolation, Vec *scaling);
extern PetscErrorCode DMCreateMatrix_Complex(DM dm, const MatType mtype, Mat *J);
extern PetscErrorCode DMRefine_Complex(DM dm, MPI_Comm comm, DM *dmRefined);
extern PetscErrorCode DMSetUp_Complex(DM dm);
extern PetscErrorCode DMDestroy_Complex(DM dm);
extern PetscErrorCode DMView_Complex(DM dm, PetscViewer viewer);

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "DMCreate_Complex"
PetscErrorCode DMCreate_Complex(DM dm)
{
  DM_Complex    *mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscNewLog(dm, DM_Complex, &mesh);CHKERRQ(ierr);
  dm->data = mesh;

  mesh->dim              = 0;
  ierr = PetscSFCreate(((PetscObject) dm)->comm, &mesh->sf);CHKERRQ(ierr);
  ierr = PetscSFCreate(((PetscObject) dm)->comm, &mesh->sfDefault);CHKERRQ(ierr);
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, &mesh->coneSection);CHKERRQ(ierr);
  mesh->maxConeSize      = 0;
  mesh->cones            = PETSC_NULL;
  mesh->coneOrientations = PETSC_NULL;
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, &mesh->supportSection);CHKERRQ(ierr);
  mesh->maxSupportSize   = 0;
  mesh->supports         = PETSC_NULL;
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, &mesh->coordSection);CHKERRQ(ierr);
  ierr = VecCreate(((PetscObject) dm)->comm, &mesh->coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) mesh->coordinates, "coordinates");CHKERRQ(ierr);
  mesh->refinementLimit  = -1.0;

  mesh->meetTmpA         = PETSC_NULL;
  mesh->meetTmpB         = PETSC_NULL;
  mesh->joinTmpA         = PETSC_NULL;
  mesh->joinTmpB         = PETSC_NULL;
  mesh->closureTmpA      = PETSC_NULL;
  mesh->closureTmpB      = PETSC_NULL;
  mesh->facesTmp         = PETSC_NULL;

  mesh->labels               = PETSC_NULL;
  mesh->defaultSection       = PETSC_NULL;
  mesh->defaultGlobalSection = PETSC_NULL;
  mesh->lf                   = PETSC_NULL;
  mesh->lj                   = PETSC_NULL;
  mesh->printSetValues       = PETSC_FALSE;

  ierr = PetscStrallocpy(VECSTANDARD, &dm->vectype);CHKERRQ(ierr);
  dm->ops->view               = DMView_Complex;
  dm->ops->setfromoptions     = DMSetFromOptions_Complex;
  dm->ops->setup              = DMSetUp_Complex;
  dm->ops->createglobalvector = DMCreateGlobalVector_Complex;
  dm->ops->createlocalvector  = DMCreateLocalVector_Complex;
  dm->ops->createlocaltoglobalmapping      = DMCreateLocalToGlobalMapping_Complex;
  dm->ops->createlocaltoglobalmappingblock = 0;
  dm->ops->createfieldis      = DMCreateFieldIS_Complex;
  dm->ops->getcoloring        = 0;
  dm->ops->creatematrix       = DMCreateMatrix_Complex;
  dm->ops->createinterpolation= 0;
  dm->ops->getaggregates      = 0;
  dm->ops->getinjection       = 0;
  dm->ops->refine             = DMRefine_Complex;
  dm->ops->coarsen            = 0;
  dm->ops->refinehierarchy    = 0;
  dm->ops->coarsenhierarchy   = 0;
  dm->ops->globaltolocalbegin = DMGlobalToLocalBegin_Complex;
  dm->ops->globaltolocalend   = DMGlobalToLocalEnd_Complex;
  dm->ops->localtoglobalbegin = DMLocalToGlobalBegin_Complex;
  dm->ops->localtoglobalend   = DMLocalToGlobalEnd_Complex;
  dm->ops->destroy            = DMDestroy_Complex;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "DMComplexCreate"
/*@
  DMComplexCreate - Creates a DMComplex object, which encapsulates an unstructured mesh, or CW complex, which can be expressed using a Hasse Diagram.

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DMComplex object

  Output Parameter:
. mesh  - The DMComplex object

  Level: beginner

.keywords: DMComplex, create
@*/
PetscErrorCode  DMComplexCreate(MPI_Comm comm, DM *mesh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(mesh,2);
  ierr = DMCreate(comm, mesh);CHKERRQ(ierr);
  ierr = DMSetType(*mesh, DMCOMPLEX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
