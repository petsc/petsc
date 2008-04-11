static char help[] =
"subduction: Steady-state 2D subduction flow, pressure and temperature solver.\n\
       The flow is driven by the subducting slab.\n\
-------------------------------subduction help---------------------------------\n\
  -OPTION <DEFAULT> = (UNITS) DESCRIPTION.\n\n\
  -width <320> = (km) width of domain.\n\
  -depth <300> = (km) depth of domain.\n\
  -slab_dip <45> = (degrees) dip angle of the slab (determines the grid aspect ratio).\n\
  -lid_depth <35> = (km) depth of the static conductive lid.\n\
  -fault_depth <35> = (km) depth of slab-wedge mechanical coupling\n\
     (fault dept >= lid depth).\n\
\n\
  -ni <82> = grid cells in x-direction. (nj adjusts to accommodate\n\
      the slab dip & depth). DO NOT USE -da_grid_x option!!!\n\
  -ivisc <3> = rheology option.\n\
      0 --- constant viscosity.\n\
      1 --- olivine diffusion creep rheology (T&P-dependent, newtonian).\n\
      2 --- olivine dislocation creep rheology (T&P-dependent, non-newtonian).\n\
      3 --- Full mantle rheology, combination of 1 & 2.\n\
\n\
  -slab_velocity <5> = (cm/year) convergence rate of slab into subduction zone.\n\
  -slab_age <50> = (million yrs) age of slab for thermal profile boundary condition.\n\
  -lid_age <50> = (million yrs) age of lid for thermal profile boundary condition.\n\
\n\
  FOR OTHER PARAMETER OPTIONS AND THEIR DEFAULT VALUES, see SetParams() in ex30.c.\n\
-------------------------------subduction help---------------------------------\n";

#include <petscdmmg.h>
#include <petscmesh.h>
#include <Distribution.hh>

using ALE::Obj;

typedef enum {SLAB_LID, SLAB, BOTTOM, RIGHT, RIGHT_LID, TOP, LID, NUM_BC} BCType;

typedef enum {MANTLE_MAT, LID_MAT} MaterialType;
 
typedef struct {
  PetscInt   debug;             // The debugging level
  char       partitioner[2048]; // The graph partitioner
  PetscTruth interpolate;       // Create intermediate mesh elements
  PetscReal  refinementLimit;   // The maximum volume of any cell

  PetscReal  width;             // The width of the top of the wedge (m)
  PetscReal  depth;             // The depth of the wedge (m)
  PetscReal  lidDepth;          // The depth of the lid (m)
  PetscReal  slabDip;           // The dip of the wedge (radians)
  PetscReal  featureSize;       // An initial discretization size (m)

  double   (*funcs[4])(const double []); // The function to project
} Options;

#include "subduction_quadrature.h"

double zero(const double x[]) {
  return 0.0;
}

double one(const double x[]) {
  return 1.0;
}

double two(const double x[]) {
  return 2.0;
}

double three(const double x[]) {
  return 3.0;
}

double four(const double x[]) {
  return 4.0;
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->width           = 320.0;
  options->depth           = 300.0;
  options->lidDepth        = 35.0;
  options->slabDip         = 45;
  options->featureSize     = 100000.0;
  options->interpolate     = PETSC_TRUE;
  options->refinementLimit = 0.0;
  ierr = PetscStrcpy(options->partitioner, "parmetis");CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, "", "PFLOTRAN Options", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "subduction.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-partitioner", "The graph partitioner", "subduction.cxx", options->partitioner, options->partitioner, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-width", "The width of the top of the slab (km)", "subduction.cxx", options->width, &options->width, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-depth", "The depth of the slab (km)", "subduction.cxx", options->depth, &options->depth, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-lid_depth", "The depth of the lid (km)", "subduction.cxx", options->lidDepth, &options->lidDepth, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-slab_dip", "The slab dip angle (degrees)", "subduction.cxx", options->slabDip, &options->slabDip, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-feature_size", "An initial discretzation size (m)", "subduction.cxx", options->featureSize, &options->featureSize, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Create intermediate mesh elements", "subduction.cxx", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-refinement_limit", "The maximum size of any cell", "subduction.cxx", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  // Fix units
  options->width    *= 1000.0;
  options->depth    *= 1000.0;
  options->lidDepth *= 1000.0;
  options->slabDip  *= PETSC_PI/180.0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMeshBoundary"
/*
  Subduction zone boundary:

        width
  ----------------
  \_______________| lid depth
   \slab dip      |
    \             |
     \            | depth
      \           |
       \          |
        \         |
         \________|

*/
PetscErrorCode CreateMeshBoundary(MPI_Comm comm, Mesh *meshBoundary, Options *options)
{
  const int      dim = 1;
  Obj<ALE::Mesh> m   = new ALE::Mesh(comm, dim, options->debug);
  PetscErrorCode ierr;

  ierr = MeshCreate(comm, meshBoundary);CHKERRQ(ierr);
  ierr = MeshSetMesh(*meshBoundary, m);CHKERRQ(ierr);
  const Obj<ALE::Mesh::sieve_type> sieve = new ALE::Mesh::sieve_type(m->comm(), m->debug());
  const int       embedDim   = 2;
  const PetscReal h          = options->featureSize;
  const PetscReal depth      = options->depth;
  const PetscReal lidDepth   = options->lidDepth;
  const PetscReal topWidth   = options->width;
  const PetscReal botWidth   = options->width - options->depth/tan(options->slabDip);
  const PetscReal slabDip    = options->slabDip;
  const PetscReal slabLen    = options->depth/sin(slabDip);
  const PetscReal slabLidLen = options->lidDepth/sin(slabDip);
  const PetscReal lidStartX  = options->lidDepth/tan(slabDip);
  ALE::Mesh::real_section_type::value_type *coords = PETSC_NULL;
  PetscInt numVertices = 0;

  PetscFunctionBegin;
  if (botWidth <= 0.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Slab dip is too shallow for slab depth.");
  if (m->commRank() == 0) {
    // Determine sizes
    numVertices += (PetscInt) (slabLidLen/h) + 1;
    numVertices += (PetscInt) (slabLen/h)    + 1;
    numVertices += (PetscInt) (botWidth/h)   + 1;
    numVertices += (PetscInt) (lidDepth/h)   + 1;
    numVertices += (PetscInt) ((depth - lidDepth)/h) + 1;
    numVertices += (PetscInt) (topWidth/h)   + 1;
    numVertices += (PetscInt) ((topWidth - lidStartX)/h) + 1;
    // Create vertices and edges
    coords = new ALE::Mesh::real_section_type::value_type[numVertices*embedDim];
    int v = 0, e = -1, tmpV, lidStart, lidEnd;
    // Slab lid boundary
    coords[v*embedDim+0] = 0.0;
    coords[v*embedDim+1] = 0.0;
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) slabLidLen/h; ++v, ++e, ++tmpV) {
      coords[v*embedDim+0] =  tmpV*h*cos(slabDip);
      coords[v*embedDim+1] = -tmpV*h*sin(slabDip);
      sieve->addArrow(v+numVertices-1, e, 0);
      sieve->addArrow(v+numVertices,   e, 1);
    }
    lidStart             = v+numVertices;
    coords[v*embedDim+0] = lidStartX;
    coords[v*embedDim+1] = -lidDepth;
    sieve->addArrow(v+numVertices-1, e, 0);
    sieve->addArrow(v+numVertices,   e, 1);
    // Slab boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) slabLen/h; ++v, ++e, ++tmpV) {
      coords[v*embedDim+0] =  tmpV*h*cos(slabDip);
      coords[v*embedDim+1] = -tmpV*h*sin(slabDip);
      sieve->addArrow(v+numVertices-1, e, 0);
      sieve->addArrow(v+numVertices,   e, 1);
    }
    coords[v*embedDim+0] = topWidth - botWidth;
    coords[v*embedDim+1] = -depth;
    sieve->addArrow(v+numVertices-1, e, 0);
    sieve->addArrow(v+numVertices,   e, 1);
    // Bottom boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) botWidth/h; ++v, ++e, ++tmpV) {
      coords[v*embedDim+0] = topWidth - botWidth + tmpV*h;
      coords[v*embedDim+1] = -depth;
      sieve->addArrow(v+numVertices-1, e, 0);
      sieve->addArrow(v+numVertices,   e, 1);
    }
    coords[v*embedDim+0] = topWidth;
    coords[v*embedDim+1] = -depth;
    sieve->addArrow(v+numVertices-1, e, 0);
    sieve->addArrow(v+numVertices,   e, 1);
    // Right boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) (depth - lidDepth)/h; ++v, ++e, ++tmpV) {
      coords[v*embedDim+0] = topWidth;
      coords[v*embedDim+1] = tmpV*h - depth;
      sieve->addArrow(v+numVertices-1, e, 0);
      sieve->addArrow(v+numVertices,   e, 1);
    }
    lidEnd               = v+numVertices;
    coords[v*embedDim+0] = topWidth;
    coords[v*embedDim+1] = -lidDepth;
    sieve->addArrow(v+numVertices-1, e, 0);
    sieve->addArrow(v+numVertices,   e, 1);
    // Right lid boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) lidDepth/h; ++v, ++e, ++tmpV) {
      coords[v*embedDim+0] = topWidth;
      coords[v*embedDim+1] = tmpV*h - lidDepth;
      sieve->addArrow(v+numVertices-1, e, 0);
      sieve->addArrow(v+numVertices,   e, 1);
    }
    coords[v*embedDim+0] = topWidth;
    coords[v*embedDim+1] = 0.0;
    sieve->addArrow(v+numVertices-1, e, 0);
    sieve->addArrow(v+numVertices,   e, 1);
    // Top boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) topWidth/h; ++v, ++e, ++tmpV) {
      coords[v*embedDim+0] = topWidth - tmpV*h;
      coords[v*embedDim+1] = 0.0;
      sieve->addArrow(v+numVertices-1, e, 0);
      sieve->addArrow(v+numVertices,   e, 1);
    }
    sieve->addArrow(v+numVertices-1, e, 0);
    sieve->addArrow(0+numVertices,   e, 1);
    // Lid boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) (topWidth - lidStartX)/h; ++v, ++e, ++tmpV) {
      coords[v*embedDim+0] = lidStartX + tmpV*h;
      coords[v*embedDim+1] = -lidDepth;
      if (tmpV == 1) {
        sieve->addArrow(lidStart,        e, 0);
        sieve->addArrow(v+numVertices,   e, 1);
      } else {
        sieve->addArrow(v+numVertices-1, e, 0);
        sieve->addArrow(v+numVertices,   e, 1);
      }
    }
    sieve->addArrow(v+numVertices-1, e, 0);
    sieve->addArrow(lidEnd,          e, 1);
    if (numVertices != v) SETERRQ2(PETSC_ERR_PLIB, "Mismatch in number of vertices %d should be %d", v, numVertices);
  }
  m->setSieve(sieve);
  m->stratify();
  ALE::SieveBuilder<ALE::Mesh>::buildCoordinates(m, dim+1, coords);
  // Create boundary conditions
  const Obj<ALE::Mesh::label_type>& markers = m->createLabel("marker");

  if (m->commRank() == 0) {
    int v = 0, e = -1, tmpV, lidStart, lidEnd;
    // Slab lid boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) slabLidLen/h; ++v, ++e, ++tmpV) {
      m->setValue(markers, v+numVertices-1, SLAB_LID);
      m->setValue(markers, v+numVertices,   SLAB_LID);
      m->setValue(markers, e,               SLAB_LID);
    }
    lidStart = v+numVertices;
    m->setValue(markers, v+numVertices-1, SLAB_LID);
    m->setValue(markers, v+numVertices,   SLAB_LID);
    m->setValue(markers, e,               SLAB_LID);
    // Slab boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) slabLen/h; ++v, ++e, ++tmpV) {
      m->setValue(markers, v+numVertices-1, SLAB);
      m->setValue(markers, v+numVertices,   SLAB);
      m->setValue(markers, e,               SLAB);
    }
    m->setValue(markers, v+numVertices-1, SLAB);
    m->setValue(markers, v+numVertices,   SLAB);
    m->setValue(markers, e,               SLAB);
    // Bottom boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) botWidth/h; ++v, ++e, ++tmpV) {
      m->setValue(markers, v+numVertices-1, BOTTOM);
      m->setValue(markers, v+numVertices,   BOTTOM);
      m->setValue(markers, e,               BOTTOM);
    }
    m->setValue(markers, v+numVertices-1, BOTTOM);
    m->setValue(markers, v+numVertices,   BOTTOM);
    m->setValue(markers, e,               BOTTOM);
    // Right boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) (depth - lidDepth)/h; ++v, ++e, ++tmpV) {
      m->setValue(markers, v+numVertices-1, RIGHT);
      m->setValue(markers, v+numVertices,   RIGHT);
      m->setValue(markers, e,               RIGHT);
    }
    lidEnd = v+numVertices;
    m->setValue(markers, v+numVertices-1, RIGHT);
    m->setValue(markers, v+numVertices,   RIGHT);
    m->setValue(markers, e,               RIGHT);
    // Right lid boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) lidDepth/h; ++v, ++e, ++tmpV) {
      m->setValue(markers, v+numVertices-1, RIGHT_LID);
      m->setValue(markers, v+numVertices,   RIGHT_LID);
      m->setValue(markers, e,               RIGHT_LID);
    }
    m->setValue(markers, v+numVertices-1, RIGHT_LID);
    m->setValue(markers, v+numVertices,   RIGHT_LID);
    m->setValue(markers, e,               RIGHT_LID);
    // Top boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) topWidth/h; ++v, ++e, ++tmpV) {
      m->setValue(markers, v+numVertices-1, TOP);
      m->setValue(markers, v+numVertices,   TOP);
      m->setValue(markers, e,               TOP);
    }
    m->setValue(markers, v+numVertices-1, TOP);
    m->setValue(markers, 0+numVertices,   TOP);
    m->setValue(markers, e,               TOP);
    // Lid boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) (topWidth - lidStartX)/h; ++v, ++e, ++tmpV) {
      if (tmpV == 1) {
        m->setValue(markers, lidStart,        LID);
        m->setValue(markers, v+numVertices,   LID);
        m->setValue(markers, e,               LID);
      } else {
        m->setValue(markers, v+numVertices-1, LID);
        m->setValue(markers, v+numVertices,   LID);
        m->setValue(markers, e,               LID);
      }
    }
    m->setValue(markers, v+numVertices-1, LID);
    m->setValue(markers, lidEnd,          LID);
    m->setValue(markers, e,               LID);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePartition"
// Creates a field whose value is the processor rank on each element
PetscErrorCode CreatePartition(Mesh mesh, SectionInt *partition)
{
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MeshGetCellSectionInt(mesh, 1, partition);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *partition, "partition");CHKERRQ(ierr);
  const Obj<ALE::Mesh::label_sequence>&     cells = m->heightStratum(0);
  const ALE::Mesh::label_sequence::iterator end   = cells->end();
  const int                                 rank  = m->commRank();

  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
    ierr = SectionIntUpdate(*partition, *c_iter, &rank);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMaterialField"
// Creates a field whose value is the material on each element
PetscErrorCode CreateMaterialField(Mesh mesh, Options *options, SectionInt *material)
{
  Obj<ALE::Mesh> m;
  SectionReal    coordinates;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MeshGetCellSectionInt(mesh, 1, material);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *material, "material");CHKERRQ(ierr);
  ierr = MeshGetSectionReal(mesh, "coordinates", &coordinates);CHKERRQ(ierr);
  const Obj<ALE::Mesh::label_sequence>&     cells = m->heightStratum(0);
  const ALE::Mesh::label_sequence::iterator end   = cells->end();
  const int dim      = m->getDimension();
  const int corners  = m->getSieve()->nCone(*cells->begin(), m->depth())->size();
  double   *centroid = new double[dim];

  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
    double *coords;
    int     mat;

    ierr = SectionRealRestrict(coordinates, *c_iter, &coords);CHKERRQ(ierr);
    for(int d = 0; d < dim; ++d) {
      centroid[d] = 0.0;
      for(int c = 0; c < corners; ++c) {
        centroid[d] += coords[c*dim+d];
      }
      centroid[d] /= corners;
    }
    if (centroid[1] >= -options->lidDepth) {
      mat = LID_MAT;
    } else {
      mat = MANTLE_MAT;
    }
    ierr = SectionIntUpdate(*material, *c_iter, &mat);CHKERRQ(ierr);
  }
  delete [] centroid;
  ierr = SectionRealDestroy(coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ViewMesh"
PetscErrorCode ViewMesh(Mesh mesh, const char filename[], Options *options)
{
  MPI_Comm       comm;
  SectionInt     partition, material;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
  ierr = MeshView(mesh, viewer);CHKERRQ(ierr);
  ierr = CreatePartition(mesh, &partition);CHKERRQ(ierr);
  ierr = CreateMaterialField(mesh, options, &material);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
  ierr = SectionIntView(partition, viewer);CHKERRQ(ierr);
  ierr = SectionIntView(material, viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = SectionIntDestroy(partition);CHKERRQ(ierr);
  ierr = SectionIntDestroy(material);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ViewSection"
PetscErrorCode ViewSection(Mesh mesh, SectionReal section, const char filename[])
{
  MPI_Comm       comm;
  SectionInt     partition;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
  ierr = MeshView(mesh, viewer);CHKERRQ(ierr);
  ierr = SectionRealView(section, viewer);CHKERRQ(ierr);
  ierr = CreatePartition(mesh, &partition);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
  ierr = SectionIntView(partition, viewer);CHKERRQ(ierr);
  ierr = SectionIntDestroy(partition);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, Options *options)
{
  Mesh           mesh, meshBoundary;
  PetscMPIInt    size;
  PetscTruth     view;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = CreateMeshBoundary(comm, &meshBoundary, options);CHKERRQ(ierr);
  ierr = MeshGenerate(meshBoundary, options->interpolate, &mesh);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (size > 1) {
    Mesh parallelMesh;

    ierr = MeshDistributeByFace(mesh, options->partitioner, &parallelMesh);CHKERRQ(ierr);
    ierr = MeshDestroy(mesh);CHKERRQ(ierr);
    mesh = parallelMesh;
  }
  if (options->refinementLimit > 0.0) {
    Mesh refinedMesh;

    ierr = MeshRefine(mesh, options->refinementLimit, options->interpolate, &refinedMesh);CHKERRQ(ierr);
    ierr = MeshDestroy(mesh);CHKERRQ(ierr);
    mesh = refinedMesh;
  }
  ALE::Obj<ALE::Mesh> m;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  for(int bc = 0; bc < NUM_BC; ++bc) {
    m->markBoundaryCells("marker", bc, NUM_BC);
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view", &view);CHKERRQ(ierr);
  if (view) {m->view("Mesh");}
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view_vtk", &view);CHKERRQ(ierr);
  if (view) {ierr = ViewMesh(mesh, "subduction.vtk", options);CHKERRQ(ierr);}
  *dm = (DM) mesh;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DestroyMesh"
PetscErrorCode DestroyMesh(DM dm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshDestroy((Mesh) dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateSlabLabel"
PetscErrorCode CreateSlabLabel(Mesh mesh, Options *options)
{
  Obj<ALE::Mesh> m;
  SectionReal    coordinates;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MeshGetSectionReal(mesh, "coordinates", &coordinates);CHKERRQ(ierr);
  const Obj<ALE::Mesh::label_type>&         slabLabel = m->createLabel("exclude-u");
  const Obj<ALE::Mesh::label_sequence>&     cells     = m->heightStratum(0);
  const ALE::Mesh::label_sequence::iterator end       = cells->end();
  const int dim      = m->getDimension();
  const int corners  = m->getSieve()->nCone(*cells->begin(), m->depth())->size();
  double   *centroid = new double[dim];

  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
    double *coords;

    ierr = SectionRealRestrict(coordinates, *c_iter, &coords);CHKERRQ(ierr);
    for(int d = 0; d < dim; ++d) {
      centroid[d] = 0.0;
      for(int c = 0; c < corners; ++c) {
        centroid[d] += coords[c*dim+d];
      }
      centroid[d] /= corners;
    }
    if (centroid[1] > -options->lidDepth) {
      m->setValue(slabLabel, *c_iter, 1);
    }
  }
  delete [] centroid;
  ierr = SectionRealDestroy(coordinates);CHKERRQ(ierr);
  m->setLabel("exclude-v", slabLabel);
  const Obj<ALE::Mesh::label_sequence>&     exclusion = m->getLabelStratum("exclude-u", 1);
  const ALE::Mesh::label_sequence::iterator eEnd      = exclusion->end();

  for(ALE::Mesh::label_sequence::iterator e_iter = exclusion->begin(); e_iter != eEnd; ++e_iter) {
    const Obj<ALE::Mesh::coneArray>      closure = ALE::SieveAlg<ALE::Mesh>::closure(m, *e_iter);
    const ALE::Mesh::coneArray::iterator cEnd    = closure->end();

    for(ALE::Mesh::coneArray::iterator c_iter = closure->begin(); c_iter != cEnd; ++c_iter) {
      if (*c_iter == *e_iter) continue;
      const Obj<ALE::Mesh::supportArray>&     support  = m->getSieve()->nSupport(*c_iter, m->height(*c_iter));
      const ALE::Mesh::supportArray::iterator sEnd     = support->end();
      int                                     preserve = 1;

      for(ALE::Mesh::supportArray::iterator s_iter = support->begin(); s_iter != sEnd; ++s_iter) {
        if (!m->getValue(slabLabel, *s_iter)) {
          preserve = 0;
          break;
        }
      }
      m->setValue(slabLabel, *c_iter, preserve);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateProblem"
PetscErrorCode CreateProblem(DM dm, Options *options)
{
  Mesh           mesh = (Mesh) dm;
  Obj<ALE::Mesh> m;
  int            velMarkers[2]  = {LID, SLAB};
  int            tempMarkers[1] = {TOP};
  double       (*velFuncs[2])(const double *coords)  = {three, four};
  double       (*tempFuncs[1])(const double *coords) = {one};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  options->funcs[0]  = one;
  options->funcs[1]  = two;
  options->funcs[2]  = three;
  options->funcs[3]  = four;
  ierr = CreateProblem_gen_0(dm, "p", 0, PETSC_NULL,  PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  ierr = CreateProblem_gen_1(dm, "u", 2, velMarkers,  velFuncs,   PETSC_NULL);CHKERRQ(ierr);
  ierr = CreateProblem_gen_1(dm, "v", 2, velMarkers,  velFuncs,   PETSC_NULL);CHKERRQ(ierr);
  ierr = CreateProblem_gen_1(dm, "T", 1, tempMarkers, tempFuncs,  PETSC_NULL);CHKERRQ(ierr);
  ierr = CreateSlabLabel(mesh, options);CHKERRQ(ierr);

  const ALE::Obj<ALE::Mesh::real_section_type> s = m->getRealSection("default");
  s->setDebug(options->debug);
  m->setupField(s, NUM_BC);
  if (options->debug) {s->view("Default field");}
  PetscFunctionReturn(0);
}

// Isoviscous analytic corner-flow solution
PetscErrorCode BatchelorSolution(double coords[], double values[], Options *options) {
  const PetscReal slabDip   = options->slabDip;
  const PetscReal lidStartX = options->lidDepth/tan(slabDip);
  const PetscReal sb = sin(slabDip); 
  const PetscReal cb = cos(slabDip);
  const PetscReal c  =  slabDip*sb/(slabDip*slabDip - sb*sb);
  const PetscReal d  = (slabDip*cb - sb)/(slabDip*slabDip - sb*sb); 
  // First shift to the corner
  const PetscReal x  = coords[0] - lidStartX;
  const PetscReal y  = coords[1] - options->lidDepth;
  const PetscReal r  = sqrt(x*x+y*y);
  const PetscReal st = y/r;
  const PetscReal ct = x/r;
  const PetscReal theta = atan(y/x); 

  PetscFunctionBegin;
  values[0] = -2.0*(c*ct-d*st)/r;
  values[1] = ct*(c*theta*st + d*(st+theta*ct)) + st*(c*(st-theta*ct) + d*theta*st);
  values[2] = st*(c*theta*st + d*(st+theta*ct)) - ct*(c*(st-theta*ct) + d*theta*st);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Rhs_Unstructured"
PetscErrorCode Rhs_Unstructured(Mesh mesh, SectionReal X, SectionReal section, void *ctx)
{
  Options       *options = (Options *) ctx;
  double      (**funcs)(const double *) = options->funcs;
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::label_sequence>&    cells       = m->heightStratum(0);
  const int                                dim         = m->getDimension();
  const Obj<std::set<std::string> >&       discs       = m->getDiscretizations();
  int          totBasisFuncs = 0;
  double      *t_der, *b_der, *coords, *v0, *J, *invJ, detJ;
  PetscScalar *elemVec, *elemMat;

  ierr = SectionRealZero(section);CHKERRQ(ierr);
  for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
    totBasisFuncs += m->getDiscretization(*f_iter)->getBasisSize();
  }
  ierr = PetscMalloc2(totBasisFuncs,PetscScalar,&elemVec,totBasisFuncs*totBasisFuncs,PetscScalar,&elemMat);CHKERRQ(ierr);
  ierr = PetscMalloc6(dim,double,&t_der,dim,double,&b_der,dim,double,&coords,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    PetscScalar *x;
    int          field = 0;

    m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
    //ierr = SectionRealRestrict(X, *c_iter, &x);CHKERRQ(ierr);
    {
      Obj<ALE::Mesh::real_section_type> sX;

      ierr = SectionRealGetSection(X, sX);CHKERRQ(ierr);
      x = (PetscScalar *) m->restrictNew(sX, *c_iter);
    }
    if (detJ < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, *c_iter);
    ierr = PetscMemzero(elemVec, totBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
    for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++field) {
      const Obj<ALE::Discretization>& disc          = m->getDiscretization(*f_iter);
      const int                       numQuadPoints = disc->getQuadratureSize();
      const double                   *quadPoints    = disc->getQuadraturePoints();
      const double                   *quadWeights   = disc->getQuadratureWeights();
      const int                       numBasisFuncs = disc->getBasisSize();
      const double                   *basis         = disc->getBasis();
      const double                   *basisDer      = disc->getBasisDerivatives();
      const int                      *indices       = disc->getIndices();

      ierr = PetscMemzero(elemMat, numBasisFuncs*totBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
      // Loop over quadrature points
      for(int q = 0; q < numQuadPoints; ++q) {
        for(int d = 0; d < dim; d++) {
          coords[d] = v0[d];
          for(int e = 0; e < dim; e++) {
            coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
          }
        }
        PetscScalar funcVal = (*funcs[field])(coords);

        // Loop over trial functions
        for(int f = 0; f < numBasisFuncs; ++f) {
          // Constant part
          elemVec[indices[f]] -= basis[q*numBasisFuncs+f]*funcVal*quadWeights[q]*detJ;
          // Linear part
          //if (*f_iter == "T") {
          if (field == 0) {
            // Laplacian of T
            for(int d = 0; d < dim; ++d) {
              t_der[d] = 0.0;
              for(int e = 0; e < dim; ++e) t_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+f)*dim+e];
            }
            for(int g = 0; g < numBasisFuncs; ++g) {
              for(int d = 0; d < dim; ++d) {
                b_der[d] = 0.0;
                for(int e = 0; e < dim; ++e) b_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+g)*dim+e];
              }
              PetscScalar product = 0.0;

              for(int d = 0; d < dim; ++d) product += t_der[d]*b_der[d];
              elemMat[f*totBasisFuncs+indices[g]] += product*quadWeights[q]*detJ;
            }
            // Nonlinear advection term
            const Obj<ALE::Discretization>& u         = m->getDiscretization("u");
            const int                       numUFuncs = u->getBasisSize();
            const double                   *uBasis    = u->getBasis();
            const int                      *uIndices  = u->getIndices();
            const Obj<ALE::Discretization>& v         = m->getDiscretization("v");
            const int                       numVFuncs = v->getBasisSize();
            const double                   *vBasis    = v->getBasis();
            const int                      *vIndices  = v->getIndices();
            PetscScalar u_q = 0.0, v_q = 0.0;
            PetscScalar gradT[2] = {0.0, 0.0};

            for(int d = 0; d < dim; ++d) {
              for(int g = 0; g < numUFuncs; ++g) {
                u_q += x[uIndices[g]]*uBasis[q*numBasisFuncs+g];
              }
              for(int g = 0; g < numVFuncs; ++g) {
                v_q += x[vIndices[g]]*vBasis[q*numBasisFuncs+g];
              }
              for(int e = 0; e < dim; ++e) {
                b_der[e] = 0.0;
                for(int g = 0; g < numBasisFuncs; ++g) {
                  b_der[e] += x[indices[g]]*basisDer[(q*numBasisFuncs+g)*dim+e];
                }
                gradT[d] += invJ[e*dim+d]*b_der[e];
              }
            }
            elemVec[indices[f]] += basis[q*numBasisFuncs+f]*(u_q*gradT[0] + v_q*gradT[1])*quadWeights[q]*detJ;
          //} else if (*f_iter == "p") {
          } else if (field == 1) {
            // Divergence of u
            const Obj<ALE::Discretization>& u         = m->getDiscretization("u");
            const int                       numUFuncs = u->getBasisSize();
            const double                   *uBasisDer = u->getBasisDerivatives();
            const int                      *uIndices  = u->getIndices();

            for(int g = 0; g < numUFuncs; ++g) {
              PetscScalar uDiv = 0.0;

              for(int e = 0; e < dim; ++e) uDiv += invJ[e*dim+0]*uBasisDer[(q*numUFuncs+g)*dim+e];
              elemMat[f*totBasisFuncs+uIndices[g]] += basis[q*numBasisFuncs+f]*uDiv*quadWeights[q]*detJ;
            }
            // Divergence of v
            const Obj<ALE::Discretization>& v         = m->getDiscretization("v");
            const int                       numVFuncs = v->getBasisSize();
            const double                   *vBasisDer = v->getBasisDerivatives();
            const int                      *vIndices  = v->getIndices();

            for(int g = 0; g < numVFuncs; ++g) {
              PetscScalar vDiv = 0.0;

              for(int e = 0; e < dim; ++e) vDiv += invJ[e*dim+1]*vBasisDer[(q*numVFuncs+g)*dim+e];
              elemMat[f*totBasisFuncs+vIndices[g]] += basis[q*numBasisFuncs+f]*vDiv*quadWeights[q]*detJ;
            }
          } else {
            // Laplacian of u or v
            for(int d = 0; d < dim; ++d) {
              t_der[d] = 0.0;
              for(int e = 0; e < dim; ++e) t_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+f)*dim+e];
            }
            for(int g = 0; g < numBasisFuncs; ++g) {
              for(int d = 0; d < dim; ++d) {
                b_der[d] = 0.0;
                for(int e = 0; e < dim; ++e) b_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+g)*dim+e];
              }
              PetscScalar product = 0.0;

              for(int d = 0; d < dim; ++d) product += t_der[d]*b_der[d];
              elemMat[f*totBasisFuncs+indices[g]] += product*quadWeights[q]*detJ;
            }
            // Gradient of pressure
            const Obj<ALE::Discretization>& pres         = m->getDiscretization("p");
            const int                       numPresFuncs = pres->getBasisSize();
            const double                   *presBasisDer = pres->getBasisDerivatives();
            const int                      *presIndices  = pres->getIndices();

            for(int g = 0; g < numPresFuncs; ++g) {
              PetscScalar presGrad = 0.0;
              const int   d        = field-1;

              for(int e = 0; e < dim; ++e) presGrad -= invJ[e*dim+d]*presBasisDer[(q*numPresFuncs+g)*dim+e];
              elemMat[f*totBasisFuncs+presIndices[g]] += basis[q*numBasisFuncs+f]*presGrad*quadWeights[q]*detJ;
            }
          }
        }
      }
      if (options->debug) {
        std::cout << "Constant element vector for field " << *f_iter << ":" << std::endl;
        for(int f = 0; f < numBasisFuncs; ++f) {
          std::cout << "  " << elemVec[indices[f]] << std::endl;
        }
      }
      // Add linear contribution
      for(int f = 0; f < numBasisFuncs; ++f) {
        for(int g = 0; g < totBasisFuncs; ++g) {
          elemVec[indices[f]] += elemMat[f*totBasisFuncs+g]*x[g];
        }
      }
      if (options->debug) {
        ostringstream label; label << "Element Matrix for field " << *f_iter;
        std::cout << ALE::Mesh::printMatrix(label.str(), numBasisFuncs, totBasisFuncs, elemMat, m->commRank()) << std::endl;
        std::cout << "Linear element vector for field " << *f_iter << ":" << std::endl;
        for(int f = 0; f < numBasisFuncs; ++f) {
          std::cout << "  " << elemVec[indices[f]] << std::endl;
        }
      }
    }
    if (options->debug) {
      std::cout << "Element vector:" << std::endl;
      for(int f = 0; f < totBasisFuncs; ++f) {
        std::cout << "  " << elemVec[f] << std::endl;
      }
    }
    ierr = SectionRealUpdateAdd(section, *c_iter, elemVec);CHKERRQ(ierr);
    if (options->debug) {
      ierr = SectionRealView(section, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(elemVec,elemMat);CHKERRQ(ierr);
  ierr = PetscFree6(t_der,b_der,coords,v0,J,invJ);CHKERRQ(ierr);
  // Exchange neighbors
  ierr = SectionRealComplete(section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Jac_Unstructured"
PetscErrorCode Jac_Unstructured(Mesh mesh, SectionReal section, Mat A, void *ctx)
{
  //Options       *options = (Options *) ctx;
  Obj<ALE::Mesh::real_section_type> s;
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  const Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::label_sequence>&    cells       = m->heightStratum(0);
  const Obj<ALE::Mesh::order_type>&        order       = m->getFactory()->getGlobalOrder(m, "default", s);
  const int                                dim         = m->getDimension();
  const Obj<std::set<std::string> >&       discs       = m->getDiscretizations();
  int          totBasisFuncs = 0;
  double      *t_der, *b_der, *v0, *J, *invJ, detJ;
  PetscScalar *elemMat;

  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
    totBasisFuncs += m->getDiscretization(*f_iter)->getBasisSize();
  }
  ierr = PetscMalloc(totBasisFuncs*totBasisFuncs * sizeof(PetscScalar), &elemMat);CHKERRQ(ierr);
  ierr = PetscMalloc5(dim,double,&t_der,dim,double,&b_der,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    PetscScalar *x;
    int          field = 0;

    x = (PetscScalar *) m->restrictNew(s, *c_iter);
    m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
    if (detJ < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, *c_iter);
    ierr = PetscMemzero(elemMat, totBasisFuncs*totBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
    for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++field) {
      const Obj<ALE::Discretization>& disc          = m->getDiscretization(*f_iter);
      const int                       numQuadPoints = disc->getQuadratureSize();
      //const double                   *quadPoints    = disc->getQuadraturePoints();
      const double                   *quadWeights   = disc->getQuadratureWeights();
      const int                       numBasisFuncs = disc->getBasisSize();
      const double                   *basis         = disc->getBasis();
      const double                   *basisDer      = disc->getBasisDerivatives();
      const int                      *indices       = disc->getIndices();

      // Loop over quadrature points
      for(int q = 0; q < numQuadPoints; ++q) {
        // Loop over trial functions
        for(int f = 0; f < numBasisFuncs; ++f) {
          //if (*f_iter == "pressure") {
          if (field == 0) {
            // Divergence of u
            const Obj<ALE::Discretization>& u         = m->getDiscretization("u");
            const int                       numUFuncs = u->getBasisSize();
            const double                   *uBasisDer = u->getBasisDerivatives();
            const int                      *uIndices  = u->getIndices();

            for(int g = 0; g < numUFuncs; ++g) {
              PetscScalar uDiv = 0.0;

              for(int e = 0; e < dim; ++e) uDiv += invJ[e*dim+0]*uBasisDer[(q*numUFuncs+g)*dim+e];
              elemMat[indices[f]*totBasisFuncs+uIndices[g]] += basis[q*numBasisFuncs+f]*uDiv*quadWeights[q]*detJ;
            }
            // Divergence of v
            const Obj<ALE::Discretization>& v         = m->getDiscretization("v");
            const int                       numVFuncs = v->getBasisSize();
            const double                   *vBasisDer = v->getBasisDerivatives();
            const int                      *vIndices  = v->getIndices();

            for(int g = 0; g < numVFuncs; ++g) {
              PetscScalar vDiv = 0.0;

              for(int e = 0; e < dim; ++e) vDiv += invJ[e*dim+1]*vBasisDer[(q*numVFuncs+g)*dim+e];
              elemMat[indices[f]*totBasisFuncs+vIndices[g]] += basis[q*numBasisFuncs+f]*vDiv*quadWeights[q]*detJ;
            }
          } else {
            // Laplacian of u or v
            for(int d = 0; d < dim; ++d) {
              t_der[d] = 0.0;
              for(int e = 0; e < dim; ++e) t_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+f)*dim+e];
            }
            for(int g = 0; g < numBasisFuncs; ++g) {
              for(int d = 0; d < dim; ++d) {
                b_der[d] = 0.0;
                for(int e = 0; e < dim; ++e) b_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+g)*dim+e];
              }
              PetscScalar product = 0.0;

              for(int d = 0; d < dim; ++d) product += t_der[d]*b_der[d];
              elemMat[indices[f]*totBasisFuncs+indices[g]] += product*quadWeights[q]*detJ;
            }
            // Gradient of pressure
            const Obj<ALE::Discretization>& pres         = m->getDiscretization("p");
            const int                       numPresFuncs = pres->getBasisSize();
            const double                   *presBasisDer = pres->getBasisDerivatives();
            const int                      *presIndices  = pres->getIndices();

            for(int g = 0; g < numPresFuncs; ++g) {
              PetscScalar presGrad = 0.0;
              const int   d        = field-1;

              for(int e = 0; e < dim; ++e) presGrad -= invJ[e*dim+d]*presBasisDer[(q*numPresFuncs+g)*dim+e];
              elemMat[indices[f]*totBasisFuncs+presIndices[g]] += basis[q*numBasisFuncs+f]*presGrad*quadWeights[q]*detJ;
            }
          }
        }
      }
    }
    ierr = updateOperator(A, m, s, order, *c_iter, elemMat, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree(elemMat);CHKERRQ(ierr);
  ierr = PetscFree5(t_der,b_der,v0,J,invJ);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateSolver"
PetscErrorCode CreateSolver(DM dm, DMMG **dmmg, Options *options)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMMGCreate(comm, 1, options, dmmg);CHKERRQ(ierr);
  ierr = DMMGSetDM(*dmmg, dm);CHKERRQ(ierr);
  ierr = DMMGSetSNESLocal(*dmmg, Rhs_Unstructured, Jac_Unstructured, 0, 0);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(*dmmg);CHKERRQ(ierr);
  // With Neumann conditions, we tell DMMG that constants are in the null space of the operator
  //ierr = DMMGSetNullSpace(*dmmg, PETSC_TRUE, 0, PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Solve"
PetscErrorCode Solve(DMMG *dmmg, Options *options)
{
  Mesh                mesh = (Mesh) DMMGGetDM(dmmg);
  SNES                snes;
  MPI_Comm            comm;
  PetscInt            its;
  PetscTruth          flag;
  SNESConvergedReason reason;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMGSolve(dmmg);CHKERRQ(ierr);
  snes = DMMGGetSNES(dmmg);
  ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes, &reason);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) snes, &comm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Number of Newton iterations = %D\n", its);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Reason for solver termination: %s\n", SNESConvergedReasons[reason]);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
  if (flag) {ierr = VecView(DMMGGetx(dmmg), PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_draw", &flag);CHKERRQ(ierr);
  if (flag) {ierr = VecView(DMMGGetx(dmmg), PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  SectionReal solution;

  ierr = MeshGetSectionReal(mesh, "default", &solution);CHKERRQ(ierr);
  ierr = SectionRealToVec(solution, mesh, SCATTER_REVERSE, DMMGGetx(dmmg));CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_vtk", &flag);CHKERRQ(ierr);
  if (flag) {ierr = ViewSection(mesh, solution, "sol.vtk");CHKERRQ(ierr);}
  ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
  if (flag) {ierr = SectionRealView(solution, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_fibrated", &flag);CHKERRQ(ierr);
  if (flag) {
    Obj<ALE::Mesh::real_section_type> sol;
    ierr = SectionRealGetSection(solution, sol);CHKERRQ(ierr);
    Obj<ALE::Mesh::real_section_type> pressure    = sol->getFibration(0);
    Obj<ALE::Mesh::real_section_type> velocityX   = sol->getFibration(1);
    Obj<ALE::Mesh::real_section_type> velocityY   = sol->getFibration(2);
    Obj<ALE::Mesh::real_section_type> temperature = sol->getFibration(3);

    pressure->view("Pressure Solution");
    velocityX->view("X-Velocity Solution");
    velocityY->view("Y-Velocity Solution");
    temperature->view("Temperature Solution");
  }
  ierr = SectionRealDestroy(solution);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Options        options;
  DM             dm;
  DMMG          *dmmg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  try {
    comm = PETSC_COMM_WORLD;
    ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
    ierr = CreateMesh(comm, &dm, &options);CHKERRQ(ierr);
    ierr = CreateProblem(dm, &options);CHKERRQ(ierr);
    ierr = CreateSolver(dm, &dmmg, &options);CHKERRQ(ierr);
    ierr = Solve(dmmg, &options);CHKERRQ(ierr);
    ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
    ierr = DestroyMesh(dm, &options);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    std::cout << "ERROR: " << e.msg() << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
