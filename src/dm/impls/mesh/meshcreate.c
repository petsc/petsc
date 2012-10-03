#define PETSCDM_DLL
#include <petsc-private/meshimpl.h>    /*I   "petscdmmesh.h"   I*/
#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "DMSetFromOptions_Mesh"
PetscErrorCode  DMSetFromOptions_Mesh(DM dm)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscOptionsHead("DMMesh Options");CHKERRQ(ierr);
    /* Handle DMMesh refinement */
    /* Handle associated vectors */
    /* Handle viewing */
    ierr = PetscOptionsBool("-dm_mesh_view_vtk", "Output mesh in VTK format", "DMView", PETSC_FALSE, &flg, PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      PetscViewer viewer;

      ierr = PetscViewerCreate(((PetscObject) dm)->comm, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, "mesh.vtk");CHKERRQ(ierr);
      ierr = DMView(dm, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    ierr = PetscOptionsBool("-dm_mesh_view", "Exhaustive mesh description", "DMView", PETSC_FALSE, &flg, PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      PetscViewer viewer;

      ierr = PetscViewerCreate(((PetscObject) dm)->comm, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
      ierr = DMView(dm, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <sieve/DMBuilder.hh>

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateSquareBoundary"
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
PetscErrorCode DMMeshCreateSquareBoundary(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt edges[])
{
  DM_Mesh       *mesh        = (DM_Mesh *) dm->data;
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

    ierr = DMMeshSetChart(dm, 0, numEdges+numVertices);CHKERRQ(ierr);
    for(e = 0; e < numEdges; ++e) {
      ierr = DMMeshSetConeSize(dm, e, 2);CHKERRQ(ierr);
    }
    ierr = DMMeshSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
    for(vy = 0; vy <= edges[1]; vy++) {
      for(ex = 0; ex < edges[0]; ex++) {
        PetscInt edge    = vy*edges[0]     + ex;
        PetscInt vertex  = vy*(edges[0]+1) + ex + numEdges;
        PetscInt cone[2] = {vertex, vertex+1};

        ierr = DMMeshSetCone(dm, edge, cone);CHKERRQ(ierr);
        if ((vy == 0) || (vy == edges[1])) {
          ierr = DMMeshSetLabelValue(dm, "marker", edge,    1);CHKERRQ(ierr);
          ierr = DMMeshSetLabelValue(dm, "marker", cone[0], 1);CHKERRQ(ierr);
          if (ex == edges[0]-1) {
            ierr = DMMeshSetLabelValue(dm, "marker", cone[1], 1);CHKERRQ(ierr);
          }
        }
      }
    }
    for(vx = 0; vx <= edges[0]; vx++) {
      for(ey = 0; ey < edges[1]; ey++) {
        PetscInt edge    = vx*edges[1] + ey + edges[0]*(edges[1]+1);
        PetscInt vertex  = ey*(edges[0]+1) + vx + numEdges;
        PetscInt cone[2] = {vertex, vertex+edges[0]+1};

        ierr = DMMeshSetCone(dm, edge, cone);CHKERRQ(ierr);
        if ((vx == 0) || (vx == edges[0])) {
          ierr = DMMeshSetLabelValue(dm, "marker", edge,    1);CHKERRQ(ierr);
          ierr = DMMeshSetLabelValue(dm, "marker", cone[0], 1);CHKERRQ(ierr);
          if (ey == edges[1]-1) {
            ierr = DMMeshSetLabelValue(dm, "marker", cone[1], 1);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = DMMeshSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMMeshStratify(dm);CHKERRQ(ierr);
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
#define __FUNCT__ "DMMeshCreateCubeBoundary"
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
PetscErrorCode DMMeshCreateCubeBoundary(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt faces[])
{
  DM_Mesh       *mesh        = (DM_Mesh *) dm->data;
  PetscInt       numVertices = (faces[0]+1)*(faces[1]+1)*(faces[2]+1);
  PetscInt       numFaces    = 6;
  PetscScalar   *coords;
  PetscInt       coordSize;
  PetscMPIInt    rank;
  PetscInt       v, vx, vy, vz;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((faces[0] < 1) || (faces[1] < 1) || (faces[2] < 1)) {SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Must have at least 1 face per side");}
  if ((faces[0] > 1) || (faces[1] > 1) || (faces[2] > 1)) {SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Currently can't handle more than 1 face per side");}
  ierr = PetscMalloc(numVertices*2 * sizeof(PetscReal), &coords);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject) dm)->comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    PetscInt f;

    ierr = DMMeshSetChart(dm, 0, numFaces+numVertices);CHKERRQ(ierr);
    for(f = 0; f < numFaces; ++f) {
      ierr = DMMeshSetConeSize(dm, f, 4);CHKERRQ(ierr);
    }
    ierr = DMMeshSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
    for(v = 0; v < numFaces+numVertices; ++v) {
      ierr = DMMeshSetLabelValue(dm, "marker", v, 1);CHKERRQ(ierr);
    }
    { // Side 0 (Front)
      PetscInt cone[4] = {numFaces+4, numFaces+5, numFaces+7, numFaces+6};
      ierr = DMMeshSetCone(dm, 0, cone);CHKERRQ(ierr);
    }
    { // Side 1 (Back)
      PetscInt cone[4] = {numFaces+1, numFaces+0, numFaces+2, numFaces+3};
      ierr = DMMeshSetCone(dm, 0, cone);CHKERRQ(ierr);
    }
    { // Side 0 (Bottom)
      PetscInt cone[4] = {numFaces+0, numFaces+1, numFaces+5, numFaces+4};
      ierr = DMMeshSetCone(dm, 0, cone);CHKERRQ(ierr);
    }
    { // Side 0 (Top)
      PetscInt cone[4] = {numFaces+6, numFaces+7, numFaces+3, numFaces+2};
      ierr = DMMeshSetCone(dm, 0, cone);CHKERRQ(ierr);
    }
    { // Side 0 (Left)
      PetscInt cone[4] = {numFaces+0, numFaces+4, numFaces+6, numFaces+2};
      ierr = DMMeshSetCone(dm, 0, cone);CHKERRQ(ierr);
    }
    { // Side 0 (Right)
      PetscInt cone[4] = {numFaces+5, numFaces+1, numFaces+3, numFaces+7};
      ierr = DMMeshSetCone(dm, 0, cone);CHKERRQ(ierr);
    }
  }
  ierr = DMMeshSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMMeshStratify(dm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = PetscSectionSetChart(mesh->coordSection, numFaces, numFaces + numVertices);CHKERRQ(ierr);
  for(v = numFaces; v < numFaces+numVertices; ++v) {
    ierr = PetscSectionSetDof(mesh->coordSection, v, 3);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(mesh->coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(mesh->coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecSetSizes(mesh->coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
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
#define __FUNCT__ "DMMeshCreateBoxMesh"
PetscErrorCode DMMeshCreateBoxMesh(MPI_Comm comm, PetscInt dim, PetscBool interpolate, DM *dm) {
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm, 4);
  ierr = PetscOptionsBool("-dm_mesh_new_impl", "Use the new C unstructured mesh implementation", "DMView", PETSC_FALSE, &flg, PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    DM boundary;

    if (interpolate) {SETERRQ(comm, PETSC_ERR_SUP, "Interpolation (creation of faces and edges) is not yet supported.");}
    ierr = DMCreate(comm, &boundary);CHKERRQ(ierr);
    PetscValidLogicalCollectiveInt(boundary,dim,2);
    ierr = DMSetType(boundary, DMMESH);CHKERRQ(ierr);
    ierr = DMMeshSetDimension(boundary, dim-1);CHKERRQ(ierr);
    switch(dim) {
    case 2:
    {
      PetscReal lower[2] = {0.0, 0.0};
      PetscReal upper[2] = {1.0, 1.0};
      PetscInt  edges[2] = {2, 2};

      ierr = DMMeshCreateSquareBoundary(boundary, lower, upper, edges);CHKERRQ(ierr);
      break;
    }
    case 3:
    {
      PetscReal lower[3] = {0.0, 0.0, 0.0};
      PetscReal upper[3] = {1.0, 1.0, 1.0};
      PetscInt  faces[3] = {1, 1, 1};

      ierr = DMMeshCreateCubeBoundary(boundary, lower, upper, faces);CHKERRQ(ierr);
      break;
    }
    default:
      SETERRQ1(comm, PETSC_ERR_SUP, "Dimension not supported: %d", dim);
    }
    ierr = DMMeshGenerate(boundary, interpolate, dm);CHKERRQ(ierr);
    ierr = DMDestroy(&boundary);CHKERRQ(ierr);
  } else {
    PetscInt debug = 0;
    try {
      ierr = ALE::DMBuilder::createBoxMesh(comm, dim, false, interpolate, debug, dm);CHKERRQ(ierr);
    } catch(ALE::Exception e) {
      SETERRQ1(comm, PETSC_ERR_PLIB, "Unable to create mesh: %s", e.message());
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateMeshFromAdjacency"
/*@
  DMMeshCreateMeshFromAdjacency - Create an unstrctured mesh from a list of the vertices for each cell, and the coordinates for each vertex.

 Collective on comm

  Input Parameters:
+ comm - An MPI communicator
. dim - The dimension of the cells, e.g. triangles have dimension 2
. numCells - The number of cells in the mesh
. numCorners - The number of vertices in each cell
. cellVertices - An array of the vertices for each cell, numbered 0 to numVertices-1
. spatialDim - The dimension for coordinates, e.g. for a triangle in 3D this would be 3
. numVertices - The number of mesh vertices
. coordinates - An array of the coordinates for each vertex
- interpolate - Flag to create faces and edges

  Output Parameter:
. dm - The DMMesh object

  Level: beginner

.seealso DMMESH, DMMeshCreateMeshFromAdjacencyHybrid(), DMMeshCreateBoxMesh()
@*/
PetscErrorCode DMMeshCreateMeshFromAdjacency(MPI_Comm comm, PetscInt dim, PetscInt numCells, PetscInt numCorners, PetscInt cellVertices[], PetscInt spatialDim, PetscInt numVertices, const PetscReal coordinates[], PetscBool interpolate, DM *dm) {
  PetscInt      *cone;
  PetscInt      *coneO;
  PetscInt       debug = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(cellVertices, 5);
  /* PetscValidLogicalCollectiveBool(comm,interpolate,6); */
  PetscValidPointer(dm, 7);
  if (interpolate) {SETERRQ(comm, PETSC_ERR_SUP, "Interpolation (creation of faces and edges) is not yet supported.");}
  ierr = PetscOptionsGetInt(PETSC_NULL, "-dm_mesh_debug", &debug, PETSC_NULL);CHKERRQ(ierr);
  Obj<PETSC_MESH_TYPE>             mesh  = new PETSC_MESH_TYPE(comm, dim, debug);
  Obj<PETSC_MESH_TYPE::sieve_type> sieve = new PETSC_MESH_TYPE::sieve_type(comm, 0, numCells+numVertices, debug);

  mesh->setSieve(sieve);
  for(PetscInt c = 0; c < numCells; ++c) {
    sieve->setConeSize(c, numCorners);
  }
  sieve->symmetrizeSizes(numCells, numCorners, cellVertices, numCells);
  sieve->allocate();
  ierr = PetscMalloc2(numCorners,PetscInt,&cone,numCorners,PetscInt,&coneO);CHKERRQ(ierr);
  for(PetscInt v = 0; v < numCorners; ++v) {
    coneO[v] = 1;
  }
  for(PetscInt c = 0; c < numCells; ++c) {
    for(PetscInt v = 0; v < numCorners; ++v) {
      cone[v] = cellVertices[c*numCorners+v]+numCells;
    }
    sieve->setCone(cone, c);
    sieve->setConeOrientation(coneO, c);
  }
  ierr = PetscFree2(cone,coneO);CHKERRQ(ierr);
  sieve->symmetrize();
  mesh->stratify();
  ALE::SieveBuilder<PETSC_MESH_TYPE>::buildCoordinates(mesh, spatialDim, coordinates, numCells);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMMESH);CHKERRQ(ierr);
  ierr = DMMeshSetMesh(*dm, mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateMeshFromAdjacencyHybrid"
PetscErrorCode DMMeshCreateMeshFromAdjacencyHybrid(MPI_Comm comm, PetscInt dim, PetscInt numCells, PetscInt numCorners[], PetscInt cellVertices[], PetscInt spatialDim, PetscInt numVertices, const PetscReal coordinates[], PetscBool interpolate, DM *dm) {
  PetscInt       debug = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(numCorners, 4);
  PetscValidPointer(cellVertices, 5);
  /* PetscValidLogicalCollectiveBool(comm,interpolate,6); */
  PetscValidPointer(dm, 7);
  if (interpolate) {SETERRQ(comm, PETSC_ERR_SUP, "Interpolation (creation of faces and edges) is not yet supported.");}
  ierr = PetscOptionsGetInt(PETSC_NULL, "-dmmesh_debug", &debug, PETSC_NULL);CHKERRQ(ierr);
  Obj<PETSC_MESH_TYPE>             mesh  = new PETSC_MESH_TYPE(comm, dim, debug);
  Obj<PETSC_MESH_TYPE::sieve_type> sieve = new PETSC_MESH_TYPE::sieve_type(comm, 0, numCells+numVertices, debug);

  mesh->setSieve(sieve);
  for(PetscInt c = 0; c < numCells; ++c) {
    sieve->setConeSize(c, numCorners[c]);
  }
  //sieve->symmetrizeSizes(numCells, numCorners, cellVertices);
  sieve->allocate();
  for(PetscInt c = 0, offset = 0; c < numCells; offset += numCorners[c], ++c) {
    sieve->setCone(&cellVertices[offset], c);
  }
  sieve->symmetrize();
  PetscFunctionReturn(0);
}

/* External function declarations here */
extern PetscErrorCode DMGlobalToLocalBegin_Mesh(DM dm, Vec g, InsertMode mode, Vec l);
extern PetscErrorCode DMGlobalToLocalEnd_Mesh(DM dm, Vec g, InsertMode mode, Vec l);
extern PetscErrorCode DMLocalToGlobalBegin_Mesh(DM dm, Vec l, InsertMode mode, Vec g);
extern PetscErrorCode DMLocalToGlobalEnd_Mesh(DM dm, Vec l, InsertMode mode, Vec g);
extern PetscErrorCode DMCreateGlobalVector_Mesh(DM dm, Vec *gvec);
extern PetscErrorCode DMCreateLocalVector_Mesh(DM dm, Vec *lvec);
extern PetscErrorCode DMCreateLocalToGlobalMapping_Mesh(DM dm);
extern PetscErrorCode DMCreateInterpolation_Mesh(DM dmCoarse, DM dmFine, Mat *interpolation, Vec *scaling);
extern PetscErrorCode DMCreateMatrix_Mesh(DM dm, MatType mtype, Mat *J);
extern PetscErrorCode DMRefine_Mesh(DM dm, MPI_Comm comm, DM *dmRefined);
extern PetscErrorCode DMCoarsenHierarchy_Mesh(DM dm, int numLevels, DM *coarseHierarchy);
extern PetscErrorCode DMDestroy_Mesh(DM dm);
extern PetscErrorCode DMView_Mesh(DM dm, PetscViewer viewer);

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "DMConvert_DA_Mesh"
PetscErrorCode DMConvert_DA_Mesh(DM dm, DMType newtype, DM *dmNew)
{
  PetscSection   section;
  DM             cda;
  DMDALocalInfo  info;
  Vec            coordinates;
  PetscInt      *cone, *coneO;
  PetscInt       dim, M, N, P, numCells, numGlobalCells, numCorners, numVertices, c = 0, v = 0;
  PetscInt       ye, ze;
  PetscInt       debug = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-dm_mesh_debug", &debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dm, &dim, &M, &N, &P, 0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm, &info);CHKERRQ(ierr);
  if (info.sw  > 1) SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Currently, only DMDAs with unti stencil width can be converted to DMMeshes.");
  /* In order to get a partition of cells, rather than vertices, we give each process the cells between vertices it owns
     and also higher numbered ghost vertices (vertices to the right and up) */
  numCorners  = 1 << dim;
  numCells    = ((info.gxm+info.gxs - info.xs) - 1);
  if (dim > 1) {numCells *= ((info.gym+info.gys - info.ys) - 1);}
  if (dim > 2) {numCells *= ((info.gzm+info.gzs - info.zs) - 1);}
  numVertices = (info.gxm+info.gxs - info.xs);
  if (dim > 1) {numVertices *= (info.gym+info.gys - info.ys);}
  if (dim > 2) {numVertices *= (info.gzm+info.gzs - info.zs);}
  numGlobalCells = M-1;
  if (dim > 1) {numGlobalCells *= N-1;}
  if (dim > 2) {numGlobalCells *= P-1;}

  ALE::Obj<PETSC_MESH_TYPE>             mesh  = new PETSC_MESH_TYPE(((PetscObject) dm)->comm, info.dim, debug);
  ALE::Obj<PETSC_MESH_TYPE::sieve_type> sieve = new PETSC_MESH_TYPE::sieve_type(((PetscObject) dm)->comm, 0, numCells+numVertices, debug);
  PETSC_MESH_TYPE::renumbering_type     renumbering;

  mesh->setSieve(sieve);
  /* Number each cell for the vertex in the lower left corner */
  if (dim < 3) {ze = 1; P = 1;} else {ze = info.gzs+info.gzm-1;}
  if (dim < 2) {ye = 1; N = 1;} else {ye = info.gys+info.gym-1;}
  for(PetscInt k = info.zs; k < ze; ++k) {
    for(PetscInt j = info.ys; j < ye; ++j) {
      for(PetscInt i = info.xs; i < info.gxs+info.gxm-1; ++i, ++c) {
        PetscInt globalC = (k*(N-1) + j)*(M-1) + i;

        renumbering[globalC] = c;
        sieve->setConeSize(c, numCorners);
      }
    }
  }
  if (c != numCells) {SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Error in generated cell numbering, %d should be %d", c, numCells);}
  /* Get vertex renumbering */
  for(PetscInt k = info.zs; k < info.gzs+info.gzm; ++k) {
    for(PetscInt j = info.ys; j < info.gys+info.gym; ++j) {
      for(PetscInt i = info.xs; i < info.gxs+info.gxm; ++i, ++v) {
        PetscInt globalV = (k*N + j)*M + i + numGlobalCells;

        renumbering[globalV] = v+numCells;
      }
    }
  }
  if (v != numVertices) {SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Error in generated vertex numbering, %d should be %d", v, numVertices);}
  /* Calculate support sizes */
  for(PetscInt k = info.zs; k < ze; ++k, ++c) {
    for(PetscInt j = info.ys; j < ye; ++j) {
      for(PetscInt i = info.xs; i < info.gxs+info.gxm-1; ++i) {
        for(PetscInt kp = k; kp <= k+(dim>2); ++kp) {
          for(PetscInt jp = j; jp <= j+(dim>1); ++jp) {
            for(PetscInt ip = i; ip <= i+1; ++ip) {
              PetscInt globalV = (kp*N + jp)*M + ip + numGlobalCells;

              sieve->addSupportSize(renumbering[globalV], 1);
            }
          }
        }
      }
    }
  }
  sieve->allocate();
  ierr = PetscMalloc2(numCorners,PetscInt,&cone,numCorners,PetscInt,&coneO);CHKERRQ(ierr);
  for(PetscInt v = 0; v < numCorners; ++v) {
    coneO[v] = 1;
  }
  for(PetscInt k = info.zs; k < ze; ++k) {
    for(PetscInt j = info.ys; j < ye; ++j) {
      for(PetscInt i = info.xs; i < info.gxs+info.gxm-1; ++i) {
        PetscInt globalC = (k*(N-1) + j)*(M-1) + i;
        PetscInt v       = 0;

        cone[v++] = renumbering[(k*N + j)*M + i+0 + numGlobalCells];
        cone[v++] = renumbering[(k*N + j)*M + i+1 + numGlobalCells];
        if (dim > 1) {
          cone[v++] = renumbering[(k*N + j+1)*M + i+0 + numGlobalCells];
          cone[v++] = renumbering[(k*N + j+1)*M + i+1 + numGlobalCells];
        }
        if (dim > 2) {
          cone[v++] = renumbering[((k+1)*N + j+0)*M + i+0 + numGlobalCells];
          cone[v++] = renumbering[((k+1)*N + j+0)*M + i+1 + numGlobalCells];
          cone[v++] = renumbering[((k+1)*N + j+1)*M + i+0 + numGlobalCells];
          cone[v++] = renumbering[((k+1)*N + j+1)*M + i+1 + numGlobalCells];
        }
        sieve->setCone(cone, renumbering[globalC]);
        sieve->setConeOrientation(coneO, renumbering[globalC]);
      }
    }
  }
  ierr = PetscFree2(cone,coneO);CHKERRQ(ierr);
  sieve->symmetrize();
  mesh->stratify();
  /* Create boundary marker */
  {
    const Obj<PETSC_MESH_TYPE::label_type>& boundary = mesh->createLabel("marker");

    for(PetscInt k = info.zs; k < info.gzs+info.gzm; ++k) {
      for(PetscInt j = info.ys; j < info.gys+info.gym; ++j) {
        if (info.xs == 0) {
          PetscInt globalV = (k*N + j)*M + info.xs + numGlobalCells;

          mesh->setValue(boundary, renumbering[globalV], 1);
        }
        if (info.gxs+info.gxm-1 == M-1) {
          PetscInt globalV = (k*N + j)*M + info.gxs+info.gxm-1 + numGlobalCells;

          mesh->setValue(boundary, renumbering[globalV], 1);
        }
      }
    }
    if (dim > 1) {
      for(PetscInt k = info.zs; k < info.gzs+info.gzm; ++k) {
        for(PetscInt i = info.xs; i < info.gxs+info.gxm; ++i) {
          if (info.ys == 0) {
            PetscInt globalV = (k*N + info.ys)*M + i + numGlobalCells;

            mesh->setValue(boundary, renumbering[globalV], 1);
          }
          if (info.gys+info.gym-1 == N-1) {
            PetscInt globalV = (k*N + info.gys+info.gym-1)*M + i + numGlobalCells;

            mesh->setValue(boundary, renumbering[globalV], 1);
          }
        }
      }
    }
    if (dim > 2) {
      for(PetscInt j = info.ys; j < info.gys+info.gym; ++j) {
        for(PetscInt i = info.xs; i < info.gxs+info.gxm; ++i) {
          if (info.zs == 0) {
            PetscInt globalV = (info.zs*N + j)*M + i + numGlobalCells;

            mesh->setValue(boundary, renumbering[globalV], 1);
          }
          if (info.gzs+info.gzm-1 == P-1) {
            PetscInt globalV = ((info.gzs+info.gzm-1)*N + j)*M + i + numGlobalCells;

            mesh->setValue(boundary, renumbering[globalV], 1);
          }
        }
      }
    }
  }
  /* Create new DM */
  ierr = DMMeshCreate(((PetscObject) dm)->comm, dmNew);CHKERRQ(ierr);
  ierr = DMMeshSetMesh(*dmNew, mesh);CHKERRQ(ierr);
  /* Set coordinates */
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, &section);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(section, numCells, numCells+numVertices);CHKERRQ(ierr);
  for(PetscInt v = numCells; v < numCells+numVertices; ++v) {
    ierr = PetscSectionSetDof(section, v, dim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
  ierr = DMMeshSetCoordinateSection(*dmNew, section);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  {
    Obj<PETSC_MESH_TYPE::real_section_type> coordSection = mesh->getRealSection("coordinates");

    switch(dim) {
    case 1:
    {
      PetscScalar **coords;

      ierr = DMDAVecGetArrayDOF(cda, coordinates, &coords);CHKERRQ(ierr);
      for(PetscInt i = info.xs; i < info.gxs+info.gxm; ++i) {
        PetscInt globalV = i + numGlobalCells;

        coordSection->updatePoint(renumbering[globalV], coords[i]);
      }
      ierr = DMDAVecRestoreArrayDOF(cda, coordinates, &coords);CHKERRQ(ierr);
      break;
    }
    case 2:
    {
      PetscScalar ***coords;

      ierr = DMDAVecGetArrayDOF(cda, coordinates, &coords);CHKERRQ(ierr);
      for(PetscInt j = info.ys; j < info.gys+info.gym; ++j) {
        for(PetscInt i = info.xs; i < info.gxs+info.gxm; ++i) {
          PetscInt globalV = j*M + i + numGlobalCells;

          coordSection->updatePoint(renumbering[globalV], coords[j][i]);
        }
      }
      ierr = DMDAVecRestoreArrayDOF(cda, coordinates, &coords);CHKERRQ(ierr);
      break;
    }
    case 3:
    {
      PetscScalar ****coords;

      ierr = DMDAVecGetArrayDOF(cda, coordinates, &coords);CHKERRQ(ierr);
      for(PetscInt k = info.zs; k < info.gzs+info.gzm; ++k, ++v) {
        for(PetscInt j = info.ys; j < info.gys+info.gym; ++j) {
          for(PetscInt i = info.xs; i < info.gxs+info.gxm; ++i) {
            PetscInt globalV = (k*N + j)*M + i + numGlobalCells;

            coordSection->updatePoint(renumbering[globalV], coords[k][j][i]);
          }
        }
      }
      ierr = DMDAVecRestoreArrayDOF(cda, coordinates, &coords);CHKERRQ(ierr);
      break;
    }
    default:
      SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid DMDA dimension %d", dim);
    }
  }
  /* Get overlap for interdomain communication */
  {
    typedef PETSC_MESH_TYPE::point_type point_type;
    PETSc::Log::Event("CreateOverlap").begin();
    ALE::Obj<PETSC_MESH_TYPE::send_overlap_type> sendParallelMeshOverlap = mesh->getSendOverlap();
    ALE::Obj<PETSC_MESH_TYPE::recv_overlap_type> recvParallelMeshOverlap = mesh->getRecvOverlap();
    //   Can I figure this out in a nicer way?
    ALE::SetFromMap<std::map<point_type,point_type> > globalPoints(renumbering);

    ALE::OverlapBuilder<>::constructOverlap(globalPoints, renumbering, sendParallelMeshOverlap, recvParallelMeshOverlap);
    if (debug) {
      sendParallelMeshOverlap->view("Send Overlap");
      recvParallelMeshOverlap->view("Recieve Overlap");
    }
    mesh->setCalculatedOverlap(true);
    PETSc::Log::Event("CreateOverlap").end();
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreate_Mesh"
PetscErrorCode DMCreate_Mesh(DM dm)
{
  DM_Mesh       *mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscNewLog(dm, DM_Mesh, &mesh);CHKERRQ(ierr);
  dm->data = mesh;

  new(&mesh->m) ALE::Obj<PETSC_MESH_TYPE>(PETSC_NULL);

  mesh->globalScatter  = PETSC_NULL;
  mesh->defaultSection = PETSC_NULL;
  mesh->lf             = PETSC_NULL;
  mesh->lj             = PETSC_NULL;

  mesh->useNewImpl     = PETSC_FALSE;
  mesh->dim            = 0;
  mesh->sf             = PETSC_NULL;
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, &mesh->coneSection);CHKERRQ(ierr);
  mesh->maxConeSize    = 0;
  mesh->cones          = PETSC_NULL;
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, &mesh->supportSection);CHKERRQ(ierr);
  mesh->maxSupportSize = 0;
  mesh->supports       = PETSC_NULL;
  ierr = PetscSectionCreate(((PetscObject) dm)->comm, &mesh->coordSection);CHKERRQ(ierr);
  ierr = VecCreate(((PetscObject) dm)->comm, &mesh->coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) mesh->coordinates, "coordinates");CHKERRQ(ierr);

  mesh->meetTmpA       = PETSC_NULL;
  mesh->meetTmpB       = PETSC_NULL;
  mesh->joinTmpA       = PETSC_NULL;
  mesh->joinTmpB       = PETSC_NULL;
  mesh->closureTmpA    = PETSC_NULL;
  mesh->closureTmpB    = PETSC_NULL;

  ierr = DMSetVecType(dm,VECSTANDARD);CHKERRQ(ierr);
  dm->ops->view               = DMView_Mesh;
  dm->ops->setfromoptions     = DMSetFromOptions_Mesh;
  dm->ops->setup              = 0;
  dm->ops->createglobalvector = DMCreateGlobalVector_Mesh;
  dm->ops->createlocalvector  = DMCreateLocalVector_Mesh;
  dm->ops->createlocaltoglobalmapping      = DMCreateLocalToGlobalMapping_Mesh;
  dm->ops->createlocaltoglobalmappingblock = 0;

  dm->ops->getcoloring        = 0;
  dm->ops->creatematrix          = DMCreateMatrix_Mesh;
  dm->ops->createinterpolation   = DMCreateInterpolation_Mesh;
  dm->ops->getaggregates      = 0;
  dm->ops->getinjection       = 0;

  dm->ops->refine             = DMRefine_Mesh;
  dm->ops->coarsen            = 0;
  dm->ops->refinehierarchy    = 0;
  dm->ops->coarsenhierarchy   = DMCoarsenHierarchy_Mesh;

  dm->ops->globaltolocalbegin = DMGlobalToLocalBegin_Mesh;
  dm->ops->globaltolocalend   = DMGlobalToLocalEnd_Mesh;
  dm->ops->localtoglobalbegin = DMLocalToGlobalBegin_Mesh;
  dm->ops->localtoglobalend   = DMLocalToGlobalEnd_Mesh;

  dm->ops->destroy            = DMDestroy_Mesh;

  ierr = PetscObjectComposeFunction((PetscObject) dm, "DMConvert_da_mesh_C", "DMConvert_DA_Mesh", (void (*)(void)) DMConvert_DA_Mesh);CHKERRQ(ierr);

  /* NEW_MESH_IMPL */
  ierr = PetscOptionsBool("-dm_mesh_new_impl", "Use the new C unstructured mesh implementation", "DMCreate", PETSC_FALSE, &mesh->useNewImpl, PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreate"
/*@
  DMMeshCreate - Creates a DMMesh object.

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DMMesh object

  Output Parameter:
. mesh  - The DMMesh object

  Level: beginner

.keywords: DMMesh, create
@*/
PetscErrorCode  DMMeshCreate(MPI_Comm comm, DM *mesh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(mesh,2);
  ierr = DMCreate(comm, mesh);CHKERRQ(ierr);
  ierr = DMSetType(*mesh, DMMESH);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
