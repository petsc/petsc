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

#include <petscda.h>
#include <petscmesh.h>
#include <Distribution.hh>

typedef enum {SLAB_LID, SLAB, BOTTOM, RIGHT, RIGHT_LID, TOP, LID} BCType;

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
} Options;

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
  const int           dim = 1;
  ALE::Obj<ALE::Mesh> m   = new ALE::Mesh(comm, dim, options->debug);
  PetscErrorCode      ierr;

  ierr = MeshCreate(comm, meshBoundary);CHKERRQ(ierr);
  ierr = MeshSetMesh(*meshBoundary, m);CHKERRQ(ierr);
  const ALE::Obj<ALE::Mesh::sieve_type>      sieve    = new ALE::Mesh::sieve_type(m->comm(), m->debug());
  const ALE::Obj<ALE::Mesh::topology_type>   topology = new ALE::Mesh::topology_type(m->comm(), m->debug());
  const ALE::Mesh::topology_type::patch_type patch    = 0;
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
  topology->setPatch(patch, sieve);
  topology->stratify();
  m->setTopology(topology);
  ALE::New::SieveBuilder<ALE::Mesh::sieve_type>::buildCoordinates(m->getRealSection("coordinates"), dim+1, coords);
  // Create boundary conditions
  const ALE::Obj<ALE::Mesh::topology_type::patch_label_type>& markers = topology->createLabel(patch, "marker");

  if (m->commRank() == 0) {
    for(int v = 12; v < 20; v++) {
      topology->setValue(markers, v, 1);
    }
    for(int e = 0; e < 8; e++) {
      topology->setValue(markers, e, 1);
    }
    int v = 0, e = -1, tmpV, lidStart, lidEnd;
    // Slab lid boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) slabLidLen/h; ++v, ++e, ++tmpV) {
      topology->setValue(markers, v+numVertices-1, SLAB_LID);
      topology->setValue(markers, v+numVertices,   SLAB_LID);
      topology->setValue(markers, e,               SLAB_LID);
    }
    lidStart = v+numVertices;
    topology->setValue(markers, v+numVertices-1, SLAB_LID);
    topology->setValue(markers, v+numVertices,   SLAB_LID);
    topology->setValue(markers, e,               SLAB_LID);
    // Slab boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) slabLen/h; ++v, ++e, ++tmpV) {
      topology->setValue(markers, v+numVertices-1, SLAB);
      topology->setValue(markers, v+numVertices,   SLAB);
      topology->setValue(markers, e,               SLAB);
    }
    topology->setValue(markers, v+numVertices-1, SLAB);
    topology->setValue(markers, v+numVertices,   SLAB);
    topology->setValue(markers, e,               SLAB);
    // Bottom boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) botWidth/h; ++v, ++e, ++tmpV) {
      topology->setValue(markers, v+numVertices-1, BOTTOM);
      topology->setValue(markers, v+numVertices,   BOTTOM);
      topology->setValue(markers, e,               BOTTOM);
    }
    topology->setValue(markers, v+numVertices-1, BOTTOM);
    topology->setValue(markers, v+numVertices,   BOTTOM);
    topology->setValue(markers, e,               BOTTOM);
    // Right boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) (depth - lidDepth)/h; ++v, ++e, ++tmpV) {
      topology->setValue(markers, v+numVertices-1, RIGHT);
      topology->setValue(markers, v+numVertices,   RIGHT);
      topology->setValue(markers, e,               RIGHT);
    }
    lidEnd = v+numVertices;
    topology->setValue(markers, v+numVertices-1, RIGHT);
    topology->setValue(markers, v+numVertices,   RIGHT);
    topology->setValue(markers, e,               RIGHT);
    // Right lid boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) lidDepth/h; ++v, ++e, ++tmpV) {
      topology->setValue(markers, v+numVertices-1, RIGHT_LID);
      topology->setValue(markers, v+numVertices,   RIGHT_LID);
      topology->setValue(markers, e,               RIGHT_LID);
    }
    topology->setValue(markers, v+numVertices-1, RIGHT_LID);
    topology->setValue(markers, v+numVertices,   RIGHT_LID);
    topology->setValue(markers, e,               RIGHT_LID);
    // Top boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) topWidth/h; ++v, ++e, ++tmpV) {
      topology->setValue(markers, v+numVertices-1, TOP);
      topology->setValue(markers, v+numVertices,   TOP);
      topology->setValue(markers, e,               TOP);
    }
    topology->setValue(markers, v+numVertices-1, TOP);
    topology->setValue(markers, 0+numVertices,   TOP);
    topology->setValue(markers, e,               TOP);
    // Lid boundary
    for(++v, ++e, tmpV = 1; tmpV < (PetscInt) (topWidth - lidStartX)/h; ++v, ++e, ++tmpV) {
      if (tmpV == 1) {
        topology->setValue(markers, lidStart,        LID);
        topology->setValue(markers, v+numVertices,   LID);
        topology->setValue(markers, e,               LID);
      } else {
        topology->setValue(markers, v+numVertices-1, LID);
        topology->setValue(markers, v+numVertices,   LID);
        topology->setValue(markers, e,               LID);
      }
    }
    topology->setValue(markers, v+numVertices-1, LID);
    topology->setValue(markers, lidEnd,          LID);
    topology->setValue(markers, e,               LID);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePartition"
// Creates a field whose value is the processor rank on each element
PetscErrorCode CreatePartition(Mesh mesh, SectionInt *partition)
{
  ALE::Obj<ALE::Mesh::int_section_type> section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetCellSectionInt(mesh, 1, partition);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *partition, "partition");CHKERRQ(ierr);
  ierr = SectionIntGetSection(*partition, section);CHKERRQ(ierr);
  const ALE::Mesh::int_section_type::patch_type             patch    = 0;
  const ALE::Obj<ALE::Mesh::topology_type>&                 topology = section->getTopology();
  const ALE::Obj<ALE::Mesh::topology_type::label_sequence>& cells    = topology->heightStratum(patch, 0);
  const ALE::Mesh::topology_type::label_sequence::iterator  end      = cells->end();
  const ALE::Mesh::int_section_type::value_type             rank     = section->commRank();

  for(ALE::Mesh::topology_type::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
    section->updatePoint(patch, *c_iter, &rank);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMaterialField"
// Creates a field whose value is the material on each element
PetscErrorCode CreateMaterialField(Mesh mesh, Options *options, SectionInt *material)
{
  ALE::Obj<ALE::Mesh> m;
  SectionReal    coordinates;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetCellSectionInt(mesh, 1, material);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *material, "material");CHKERRQ(ierr);
  ierr = MeshGetSectionReal(mesh, "coordinates", &coordinates);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const ALE::Mesh::int_section_type::patch_type             patch    = 0;
  const ALE::Obj<ALE::Mesh::topology_type>&                 topology = m->getTopology();
  const ALE::Obj<ALE::Mesh::topology_type::label_sequence>& cells    = topology->heightStratum(patch, 0);
  const ALE::Mesh::topology_type::label_sequence::iterator  end      = cells->end();
  const int dim     = m->getDimension();
  const int corners = topology->getPatch(patch)->nCone(*cells->begin(), topology->depth())->size();
  double   *centroid = new double[dim];

  for(ALE::Mesh::topology_type::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
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
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view", &view);CHKERRQ(ierr);
  if (view) {
    ALE::Obj<ALE::Mesh> m;

    ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
    m->view("Mesh");
  }
  *dm = (DM) mesh;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DestroyMesh"
PetscErrorCode DestroyMesh(DM dm, Options *options)
{
  PetscTruth     view;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view_vtk", &view);CHKERRQ(ierr);
  if (view) {ierr = ViewMesh((Mesh) dm, "subduction.vtk", options);CHKERRQ(ierr);}
  ierr = MeshDestroy((Mesh) dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Options        options;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  try {
    comm = PETSC_COMM_WORLD;
    ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
    ierr = CreateMesh(comm, &dm, &options);CHKERRQ(ierr);
    ierr = DestroyMesh(dm, &options);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    std::cout << "ERROR: " << e.msg() << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
