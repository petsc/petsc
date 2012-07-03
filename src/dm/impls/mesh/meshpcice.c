#include <petscdmmesh_formats.hh>   /*I      "petscmesh.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "WritePCICEVertices"
PetscErrorCode WritePCICEVertices(DM dm, PetscViewer viewer)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  return ALE::PCICE::Viewer::writeVertices(m, viewer);
}

#undef __FUNCT__
#define __FUNCT__ "WritePCICEElements"
PetscErrorCode WritePCICEElements(DM dm, PetscViewer viewer)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  return ALE::PCICE::Viewer::writeElements(m, viewer);
}

#undef __FUNCT__
#define __FUNCT__ "WritePCICERestart"
PetscErrorCode WritePCICERestart(DM dm, PetscViewer viewer)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  return ALE::PCICE::Viewer::writeRestart(m, viewer);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreatePCICE"
/*@C
  DMMeshCreatePCICE - Create a DMMesh from PCICE files.

  Not Collective

  Input Parameters:
+ dim - The topological mesh dimension
. coordFilename - The file containing vertex coordinates
. adjFilename - The file containing the vertices for each element
. interpolate - The flag for construction of intermediate elements
. bcFilename - The file containing the boundary topology and conditions
. numBdFaces - The number of boundary faces (or edges)
- numBdVertices - The number of boundary vertices

  Output Parameter:
. mesh - The DMMesh object

  Level: beginner

.keywords: mesh, PCICE
.seealso: DMMeshCreate()
@*/
PetscErrorCode DMMeshCreatePCICE(MPI_Comm comm, const int dim, const char coordFilename[], const char adjFilename[], PetscBool  interpolate, const char bcFilename[], DM *mesh)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscInt            debug = 0;
  PetscBool           flag;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMeshCreate(comm, mesh);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-debug", &debug, &flag);CHKERRQ(ierr);
  try {
    m  = ALE::PCICE::Builder::readMesh(comm, dim, std::string(coordFilename), std::string(adjFilename), true, interpolate, debug);
    if (debug) {m->view("Mesh");}
  } catch(ALE::Exception e) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN, e.message());
  }
  if (bcFilename) {
    ALE::PCICE::Builder::readBoundary(m, std::string(bcFilename));
  }
  ierr = DMMeshSetMesh(*mesh, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCICERenumberBoundary"
/*@C
  PCICERenumberBoundary - Change global element names into offsets

  Collective on DMMesh

  Input Parameters:
. mesh - the mesh

  Level: advanced

  .seealso: DMMeshCreate()
@*/
PetscErrorCode  PCICERenumberBoundary(DM dm)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  try {
    ALE::PCICE::fuseBoundary(m);
  } catch(ALE::Exception e) {
    SETERRQ(PETSC_COMM_SELF,100, e.msg().c_str());
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BCSectionGetArray"
/*@C
  BCSectionGetArray - Returns the array underlying the BCSection.

  Not Collective

  Input Parameters:
+ mesh - The DMMesh object
- name - The section name

  Output Parameters:
+ numElements - The number of mesh element with values
. fiberDim - The number of values per element
- array - The array

  Level: intermediate

.keywords: mesh, elements
.seealso: DMMeshCreate()
@*/
PetscErrorCode BCSectionGetArray(DM dm, const char name[], PetscInt *numElements, PetscInt *fiberDim, PetscInt *array[])
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::int_section_type>& section = m->getIntSection(std::string(name));
  if (!section->size()) {
    *numElements = 0;
    *fiberDim    = 0;
    *array       = NULL;
    PetscFunctionReturn(0);
  }
  const PETSC_MESH_TYPE::int_section_type::chart_type& chart = section->getChart();
  int fiberDimMin = section->getFiberDimension(*chart.begin());
  int numElem     = 0;

  for(PETSC_MESH_TYPE::int_section_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const int fiberDim = section->getFiberDimension(*c_iter);

    if (fiberDim < fiberDimMin) fiberDimMin = fiberDim;
  }
  for(PETSC_MESH_TYPE::int_section_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const int fiberDim = section->getFiberDimension(*c_iter);

    numElem += fiberDim/fiberDimMin;
  }
  *numElements = numElem;
  *fiberDim    = fiberDimMin;
  *array       = (PetscInt *) section->restrictSpace();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BCSectionRealCreate"
/*@C
  BCSectionRealCreate - Creates a BCSection.

  Not Collective

  Input Parameters:
+ mesh - The DMMesh object
. name - The section name
- fiberDim - The number of values per element

  Level: intermediate

.keywords: mesh, elements
.seealso: DMMeshCreate()
@*/
PetscErrorCode BCSectionRealCreate(DM dm, const char name[], PetscInt fiberDim)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&  section = m->getRealSection(std::string(name));
  const ALE::Obj<PETSC_MESH_TYPE::int_section_type>&   ibc     = m->getIntSection("IBC");
  const PETSC_MESH_TYPE::int_section_type::chart_type& chart   = ibc->getChart();

  for(PETSC_MESH_TYPE::int_section_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
    section->setFiberDimension(*p_iter, ibc->getFiberDimension(*p_iter));
  }
  m->allocate(section);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BCSectionRealGetArray"
/*@C
  BCSectionRealGetArray - Returns the array underlying the BCSection.

  Not Collective

  Input Parameters:
+ mesh - The DMMesh object
- name - The section name

  Output Parameters:
+ numElements - The number of mesh element with values
. fiberDim - The number of values per element
- array - The array

  Level: intermediate

.keywords: mesh, elements
.seealso: DMMeshCreate()
@*/
PetscErrorCode BCSectionRealGetArray(DM dm, const char name[], PetscInt *numElements, PetscInt *fiberDim, PetscReal *array[])
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& section = m->getRealSection(std::string(name));
  if (!section->size()) {
    *numElements = 0;
    *fiberDim    = 0;
    *array       = NULL;
    PetscFunctionReturn(0);
  }
  const PETSC_MESH_TYPE::real_section_type::chart_type& chart = section->getChart();
  int fiberDimMin = section->getFiberDimension(*chart.begin());
  int numElem     = 0;

  for(PETSC_MESH_TYPE::real_section_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const int fiberDim = section->getFiberDimension(*c_iter);

    if (fiberDim < fiberDimMin) fiberDimMin = fiberDim;
  }
  for(PETSC_MESH_TYPE::real_section_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const int fiberDim = section->getFiberDimension(*c_iter);

    numElem += fiberDim/fiberDimMin;
  }
  *numElements = numElem;
  *fiberDim    = fiberDimMin;
  *array       = (PetscReal *) section->restrictSpace();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BCFUNCGetArray"
PetscErrorCode BCFUNCGetArray(DM dm, PetscInt *numElements, PetscInt *fiberDim, PetscScalar *array[])
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
#if 0
  PETSC_MESH_TYPE::bc_values_type& bcValues = m->getBCValues();
  *numElements = bcValues.size();
  *fiberDim    = 4;
  *array       = new PetscScalar[(*numElements)*(*fiberDim)];
  for(int bcf = 1; bcf <= (int) bcValues.size(); ++bcf) {
    (*array)[(bcf-1)*4+0] = bcValues[bcf].rho;
    (*array)[(bcf-1)*4+1] = bcValues[bcf].u;
    (*array)[(bcf-1)*4+2] = bcValues[bcf].v;
    (*array)[(bcf-1)*4+3] = bcValues[bcf].p;
  }
#else
  *numElements = 0;
  *fiberDim    = 0;
  *array       = NULL;
#endif
  PetscFunctionReturn(0);
}

namespace ALE {
  namespace PCICE {
    //
    // Builder methods
    //
    void Builder::readConnectivity(MPI_Comm comm, const std::string& filename, int& corners, const bool useZeroBase, int& numElements, int *vertices[]) {
      PetscViewer    viewer;
      FILE          *f;
      PetscInt       numCells, cellCount = 0;
      PetscInt      *verts;
      char           buf[2048];
      PetscInt       c;
      PetscInt       commRank;
      PetscErrorCode ierr;

      ierr = MPI_Comm_rank(comm, &commRank);

      if (commRank != 0) return;
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
      ierr = PetscViewerFileSetName(viewer, filename.c_str());
      if (ierr) {
        ostringstream txt;
        txt << "Could not open PCICE connectivity file: " << filename;
        throw ALE::Exception(txt.str().c_str());
      }
      ierr = PetscViewerASCIIGetPointer(viewer, &f);
      if (fgets(buf, 2048, f) == NULL) {
        throw ALE::Exception("Invalid connectivity file: Missing number of elements");
      }
      const char *sizes = strtok(buf, " ");
      numCells = atoi(sizes);
      sizes = strtok(NULL, " ");
      if (sizes != NULL) {
        corners = atoi(sizes);
        std::cout << "Reset corners to " << corners << std::endl;
      }
      ierr = PetscMalloc(numCells*corners * sizeof(PetscInt), &verts);
      while(fgets(buf, 2048, f) != NULL) {
        const char *v = strtok(buf, " ");

        /* Ignore cell number */
        v = strtok(NULL, " ");
        for(c = 0; c < corners; c++) {
          int vertex = atoi(v);

          if (!useZeroBase) vertex -= 1;
          verts[cellCount*corners+c] = vertex;
          v = strtok(NULL, " ");
        }
        cellCount++;
      }
      ierr = PetscViewerDestroy(&viewer);
      numElements = numCells;
      *vertices = verts;
    };
    void Builder::readCoordinates(MPI_Comm comm, const std::string& filename, const int dim, int& numVertices, PetscReal *coordinates[]) {
      PetscViewer    viewer;
      FILE          *f;
      PetscInt       numVerts, vertexCount = 0;
      PetscReal     *coords;
      char           buf[2048];
      PetscInt       c;
      PetscInt       commRank;
      PetscErrorCode ierr;

      ierr = MPI_Comm_rank(comm, &commRank);

      if (commRank != 0) return;
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
      ierr = PetscViewerFileSetName(viewer, filename.c_str());
      if (ierr) {
        ostringstream txt;
        txt << "Could not open PCICE coordinate file: " << filename;
        throw ALE::Exception(txt.str().c_str());
      }
      ierr = PetscViewerASCIIGetPointer(viewer, &f);
      numVerts = atoi(fgets(buf, 2048, f));
      ierr = PetscMalloc(numVerts*dim * sizeof(PetscReal), &coords);
      while(fgets(buf, 2048, f) != NULL) {
        const char *x = strtok(buf, " ");

        /* Ignore vertex number */
        x = strtok(NULL, " ");
        for(c = 0; c < dim; c++) {
          coords[vertexCount*dim+c] = atof(x);
          x = strtok(NULL, " ");
        }
        vertexCount++;
      }
      ierr = PetscViewerDestroy(&viewer);
      numVertices = numVerts;
      *coordinates = coords;
    };
    Obj<PETSC_MESH_TYPE> Builder::readMesh(MPI_Comm comm, const int dim, const std::string& basename, const bool useZeroBase = true, const bool interpolate = true, const int debug = 0) {
      return readMesh(comm, dim, basename+".nodes", basename+".lcon", useZeroBase, interpolate, debug);
    };
#ifdef PETSC_OPT_SIEVE
    Obj<PETSC_MESH_TYPE> Builder::readMesh(MPI_Comm comm, const int dim, const std::string& coordFilename, const std::string& adjFilename, const bool useZeroBase = true, const bool interpolate = true, const int debug = 0) {
      typedef ALE::Mesh<PetscInt,PetscScalar> FlexMesh;
      Obj<Mesh>          mesh  = new Mesh(comm, dim, debug);
      Obj<sieve_type>    sieve = new sieve_type(comm, debug);
      const Obj<FlexMesh>             m = new FlexMesh(comm, dim, debug);
      const Obj<FlexMesh::sieve_type> s = new FlexMesh::sieve_type(comm, debug);
      int       *cells            = NULL;
      PetscReal *coordinates      = NULL;
      int        numCells = 0, numVertices = 0, numCorners = dim+1;
      PetscErrorCode ierr;

      ALE::PCICE::Builder::readConnectivity(comm, adjFilename, numCorners, useZeroBase, numCells, &cells);
      ALE::PCICE::Builder::readCoordinates(comm, coordFilename, dim, numVertices, &coordinates);
      ALE::SieveBuilder<FlexMesh>::buildTopology(s, dim, numCells, cells, numVertices, interpolate, numCorners, -1, m->getArrowSection("orientation"));
      m->setSieve(s);
      m->stratify();
      mesh->setSieve(sieve);
      std::map<Mesh::point_type,Mesh::point_type> renumbering;
      ALE::ISieveConverter::convertSieve(*s, *sieve, renumbering, false);
      mesh->stratify();
      ALE::ISieveConverter::convertOrientation(*s, *sieve, renumbering, m->getArrowSection("orientation").ptr());
      ALE::SieveBuilder<PETSC_MESH_TYPE>::buildCoordinates(mesh, dim, coordinates);
      if (cells) {ierr = PetscFree(cells);CHKERRXX(ierr);}
      if (coordinates) {ierr = PetscFree(coordinates);CHKERRXX(ierr);}
      return mesh;
    };
    void Builder::readBoundary(const Obj<Mesh>& mesh, const std::string& bcFilename) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
    void Builder::outputVerticesLocal(const Obj<Mesh>& mesh, int *numVertices, int *dim, PetscReal *coordinates[], const bool columnMajor) {
      const Obj<Mesh::real_section_type>& coordSec = mesh->getRealSection("coordinates");
      if (!coordSec->size()) {
        *numVertices = 0;
        *dim         = 0;
        *coordinates = NULL;
        return;
      }
      const Obj<Mesh::label_sequence>& vertices   = mesh->depthStratum(0);
      const Obj<Mesh::numbering_type>& vNumbering = mesh->getFactory()->getLocalNumbering(mesh, 0);
      int            size     = vertices->size();
      int            embedDim = coordSec->getFiberDimension(*vertices->begin());
      PetscReal     *coords;
      PetscErrorCode ierr;

      ierr = PetscMalloc(vertices->size()*embedDim * sizeof(double), &coords);CHKERRXX(ierr);
      for(Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
        const Mesh::real_section_type::value_type *array = coordSec->restrictPoint(*v_iter);
        const int                                  row   = vNumbering->getIndex(*v_iter);

        if (columnMajor) {
          for(int d = 0; d < embedDim; d++) {
            coords[d*size + row] = array[d];
          }
        } else {
          for(int d = 0; d < embedDim; d++) {
            coords[row*embedDim + d] = array[d];
          }
        }
      }
      *numVertices = size;
      *dim         = embedDim;
      *coordinates = coords;
    };
    void Builder::outputElementsLocal(const Obj<Mesh>& mesh, int *numElements, int *numCorners, int *vertices[], const bool columnMajor) {
      if (!mesh->heightStratum(0)->size()) {
        *numElements = 0;
        *numCorners  = 0;
        *vertices    = NULL;
        return;
      }
      const Obj<Mesh::sieve_type>&     sieve      = mesh->getSieve();
      const Obj<Mesh::label_sequence>& elements   = mesh->heightStratum(0);
      const Obj<Mesh::numbering_type>& eNumbering = mesh->getFactory()->getLocalNumbering(mesh, mesh->depth());
      const Obj<Mesh::numbering_type>& vNumbering = mesh->getFactory()->getLocalNumbering(mesh, 0);
      int            size         = elements->size();
      //int            corners      = sieve->nCone(*elements->begin(), topology->depth())->size();
      int            corners      = sieve->getConeSize(*elements->begin());
      int           *v;
      PetscErrorCode ierr;

      ierr = PetscMalloc(size*corners * sizeof(int), &v);CHKERRXX(ierr);
      for(Mesh::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
        const Obj<Mesh::sieve_type::coneSequence>      cone  = sieve->cone(*e_iter);
        Mesh::sieve_type::coneSequence::const_iterator begin = cone->begin();
        Mesh::sieve_type::coneSequence::const_iterator end   = cone->end();

        const int row = eNumbering->getIndex(*e_iter);
        int       c   = -1;
        if (columnMajor) {
          for(Mesh::sieve_type::coneSequence::iterator c_iter = begin; c_iter != end; ++c_iter) {
            v[(++c)*size + row] = vNumbering->getIndex(*c_iter)+1;
          }
        } else {
          for(Mesh::sieve_type::coneSequence::iterator c_iter = begin; c_iter != end; ++c_iter) {
            v[row*corners + ++c] = vNumbering->getIndex(*c_iter)+1;
          }
        }
      }
      *numElements = size;
      *numCorners  = corners;
      *vertices    = v;
    };
    PetscErrorCode Viewer::writeVertices(const ALE::Obj<Mesh>& mesh, PetscViewer viewer) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
    PetscErrorCode Viewer::writeElements(const ALE::Obj<Mesh>& mesh, PetscViewer viewer) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
    PetscErrorCode Viewer::writeVerticesLocal(const Obj<Mesh>& mesh, PetscViewer viewer) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
    PetscErrorCode Viewer::writeRestart(const Obj<Mesh>& mesh, PetscViewer viewer) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
    void fuseBoundary(const ALE::Obj<PETSC_MESH_TYPE>& mesh) {
      throw ALE::Exception("Not implemented for optimized sieves");
    };
#else
    Obj<PETSC_MESH_TYPE> Builder::readMesh(MPI_Comm comm, const int dim, const std::string& coordFilename, const std::string& adjFilename, const bool useZeroBase = true, const bool interpolate = true, const int debug = 0) {
      Obj<Mesh>          mesh     = new DMMesh(comm, dim, debug);
      Obj<sieve_type>    sieve    = new sieve_type(comm, debug);
      int    *cells = NULL;
      double *coordinates = NULL;
      int     numCells = 0, numVertices = 0, numCorners = dim+1;
      PetscErrorCode ierr;

      ALE::PCICE::Builder::readConnectivity(comm, adjFilename, numCorners, useZeroBase, numCells, &cells);
      ALE::PCICE::Builder::readCoordinates(comm, coordFilename, dim, numVertices, &coordinates);
      ALE::SieveBuilder<PETSC_MESH_TYPE>::buildTopology(sieve, dim, numCells, cells, numVertices, interpolate, numCorners, -1, mesh->getArrowSection("orientation"));
      mesh->setSieve(sieve);
      mesh->stratify();
      ALE::SieveBuilder<PETSC_MESH_TYPE>::buildCoordinates(mesh, dim, coordinates);
      if (cells) {ierr = PetscFree(cells);CHKERRXX(ierr);}
      if (coordinates) {ierr = PetscFree(coordinates);CHKERRXX(ierr);}
      return mesh;
    };
    // Creates boundary sections:
    //   IBC[NBFS,2]:     ALL
    //     BL[NBFS,1]:
    //     BNVEC[NBFS,2]:
    //   BCFUNC[NBCF,NV]: ALL
    //   IBNDFS[NBN,2]:   STILL NEED 4-5
    //     BNNV[NBN,2]
    void Builder::readBoundary(const Obj<Mesh>& mesh, const std::string& bcFilename) {
      PetscViewer    viewer;
      FILE          *f;
      char           buf[2048];
      PetscErrorCode ierr;

      const Obj<Mesh::int_section_type>&  ibc    = mesh->getIntSection("IBC");
      const Obj<Mesh::int_section_type>&  ibndfs = mesh->getIntSection("IBNDFS");
      const Obj<Mesh::int_section_type>&  ibcnum = mesh->getIntSection("IBCNUM");
      const Obj<Mesh::int_section_type>&  ibfcon = mesh->getIntSection("IBFCON");
      const Obj<Mesh::real_section_type>& bl     = mesh->getRealSection("BL");
      const Obj<Mesh::real_section_type>& bnvec  = mesh->getRealSection("BNVEC");
      const Obj<Mesh::real_section_type>& bnnv   = mesh->getRealSection("BNNV");
      if (mesh->commRank() != 0) {
#if 0
        mesh->distributeBCValues();
#endif
        return;
      }
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);CHKERRXX(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRXX(ierr);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRXX(ierr);
      ierr = PetscViewerFileSetName(viewer, bcFilename.c_str());CHKERRXX(ierr);
      ierr = PetscViewerASCIIGetPointer(viewer, &f);CHKERRXX(ierr);
      // Create IBC section
      int  numBdFaces = atoi(strtok(fgets(buf, 2048, f), " "));
      int *tmpIBC     = new int[numBdFaces*4];
      std::map<int,std::set<int> > elem2Idx;
      std::map<int,int> bfReorder;
      for(int bf = 0; bf < numBdFaces; bf++) {
        const char *x = strtok(fgets(buf, 2048, f), " ");

        // Ignore boundary face number
        x = strtok(NULL, " ");
        tmpIBC[bf*4+0] = atoi(x);
        x = strtok(NULL, " ");
        tmpIBC[bf*4+1] = atoi(x);
        x = strtok(NULL, " ");
        tmpIBC[bf*4+2] = atoi(x);
        x = strtok(NULL, " ");
        tmpIBC[bf*4+3] = atoi(x);
        const int elem = tmpIBC[bf*4+0]-1;

        ibc->addFiberDimension(elem, 4);
        ibcnum->addFiberDimension(elem, 1);
        ibfcon->addFiberDimension(elem, 2);
        bl->addFiberDimension(elem, 1);
        bnvec->addFiberDimension(elem, 2);
        elem2Idx[elem].insert(bf);
      }
      mesh->allocate(ibc);
      mesh->allocate(ibcnum);
      mesh->allocate(ibfcon);
      mesh->allocate(bl);
      mesh->allocate(bnvec);
      const DMMesh::int_section_type::chart_type& chart = ibc->getChart();
      int num = 1;

      for(Mesh::int_section_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const int elem = *p_iter;
        int bfNum[2];
        int k = 0;

        for(std::set<int>::const_iterator i_iter = elem2Idx[elem].begin(); i_iter != elem2Idx[elem].end(); ++i_iter) {
          bfReorder[(*i_iter)+1] = num;
          bfNum[k++] = num;
          num++;
        }
        ibcnum->updatePoint(elem, bfNum);
      }
      for(int bf = 0; bf < numBdFaces; bf++) {
        const int elem = tmpIBC[bf*4]-1;

        if (elem2Idx[elem].size() > 1) {
          if (*elem2Idx[elem].begin() == bf) {
            int values[8];
            int k = 0;

            for(std::set<int>::const_iterator i_iter = elem2Idx[elem].begin(); i_iter != elem2Idx[elem].end(); ++i_iter) {
              for(int v = 0; v < 4; ++v) {
                values[k*4+v] = tmpIBC[*i_iter*4+v];
              }
              k++;
            }
            ibc->updatePoint(elem, values);
          }
        } else {
          ibc->updatePoint(elem, &tmpIBC[bf*4]);
        }
      }
      delete [] tmpIBC;
      // Create BCFUNC section
      int numBcFunc = atoi(strtok(fgets(buf, 2048, f), " "));
      if (numBcFunc != 0) {throw ALE::Exception("Cannot handle BCFUNCS after rewrite");}
      for(int bc = 0; bc < numBcFunc; bc++) {
#if 0
        const char *x = strtok(fgets(buf, 2048, f), " ");
        DMMesh::bc_value_type value;

        // Ignore function number
        x = strtok(NULL, " ");
        value.rho = atof(x);
        x = strtok(NULL, " ");
        value.u   = atof(x);
        x = strtok(NULL, " ");
        value.v   = atof(x);
        x = strtok(NULL, " ");
        value.p   = atof(x);
        mesh->setBCValue(bc+1, value);
#endif
      }
#if 0
      mesh->distributeBCValues();
#endif
      // Create IBNDFS section
      int       numBdVertices = atoi(strtok(fgets(buf, 2048, f), " "));
      const int numElements   = mesh->heightStratum(0)->size();
      int      *tmpIBNDFS     = new int[numBdVertices*3];

      for(int bv = 0; bv < numBdVertices; bv++) {
        const char *x = strtok(fgets(buf, 2048, f), " ");

        // Ignore boundary node number
        x = strtok(NULL, " ");
        tmpIBNDFS[bv*3+0] = atoi(x);
        x = strtok(NULL, " ");
        tmpIBNDFS[bv*3+1] = atoi(x);
        x = strtok(NULL, " ");
        tmpIBNDFS[bv*3+2] = atoi(x);
        ibndfs->setFiberDimension(tmpIBNDFS[bv*3+0]-1+numElements, 6);
      }
      mesh->allocate(ibndfs);
      for(int bv = 0; bv < numBdVertices; bv++) {
        int values[5];

        values[0] = tmpIBNDFS[bv*3+0];
        // Covert to new boundary face numbers
        values[1] = bfReorder[tmpIBNDFS[bv*3+1]];
        values[2] = bfReorder[tmpIBNDFS[bv*3+2]];
        values[3] = 0;
        values[4] = 0;
        ibndfs->updatePoint(values[0]-1+numElements, values);
      }
      ierr = PetscViewerDestroy(&viewer);CHKERRXX(ierr);
      // Create BNNV[NBN,2]
      const int dim = mesh->getDimension();

      for(int bv = 0; bv < numBdVertices; bv++) {
        bnnv->setFiberDimension(tmpIBNDFS[bv*3+0]-1+numElements, dim);
      }
      mesh->allocate(bnnv);
      delete [] tmpIBNDFS;
    };
    void Builder::outputVerticesLocal(const Obj<Mesh>& mesh, int *numVertices, int *dim, double *coordinates[], const bool columnMajor) {
      const Obj<Mesh::real_section_type>& coordSec = mesh->getRealSection("coordinates");
      if (!coordSec->size()) {
        *numVertices = 0;
        *dim         = 0;
        *coordinates = NULL;
        return;
      }
      const Obj<Mesh::label_sequence>& vertices   = mesh->depthStratum(0);
      const Obj<Mesh::numbering_type>& vNumbering = mesh->getFactory()->getLocalNumbering(mesh, 0);
      int            size     = vertices->size();
      int            embedDim = coordSec->getFiberDimension(*vertices->begin());
      double        *coords;
      PetscErrorCode ierr;

      ierr = PetscMalloc(vertices->size()*embedDim * sizeof(double), &coords);CHKERRXX(ierr);
      for(Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
        const DMMesh::real_section_type::value_type *array = coordSec->restrictPoint(*v_iter);
        const int                                  row   = vNumbering->getIndex(*v_iter);

        if (columnMajor) {
          for(int d = 0; d < embedDim; d++) {
            coords[d*size + row] = array[d];
          }
        } else {
          for(int d = 0; d < embedDim; d++) {
            coords[row*embedDim + d] = array[d];
          }
        }
      }
      *numVertices = size;
      *dim         = embedDim;
      *coordinates = coords;
    };
    void Builder::outputElementsLocal(const Obj<Mesh>& mesh, int *numElements, int *numCorners, int *vertices[], const bool columnMajor) {
      if (!mesh->heightStratum(0)->size()) {
        *numElements = 0;
        *numCorners  = 0;
        *vertices    = NULL;
        return;
      }
      const Obj<Mesh::sieve_type>&     sieve      = mesh->getSieve();
      const Obj<Mesh::label_sequence>& elements   = mesh->heightStratum(0);
      const Obj<Mesh::numbering_type>& eNumbering = mesh->getFactory()->getLocalNumbering(mesh, mesh->depth());
      const Obj<Mesh::numbering_type>& vNumbering = mesh->getFactory()->getLocalNumbering(mesh, 0);
      int            size         = elements->size();
      //int            corners      = sieve->nCone(*elements->begin(), topology->depth())->size();
      int            corners      = sieve->getConeSize(*elements->begin());
      int           *v;
      PetscErrorCode ierr;

      ierr = PetscMalloc(elements->size()*corners * sizeof(int), &v);CHKERRXX(ierr);
      for(Mesh::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
        const Obj<Mesh::sieve_type::traits::coneSequence> cone  = sieve->cone(*e_iter);
        DMMesh::sieve_type::traits::coneSequence::iterator  begin = cone->begin();
        DMMesh::sieve_type::traits::coneSequence::iterator  end   = cone->end();

        const int row = eNumbering->getIndex(*e_iter);
        int       c   = -1;
        if (columnMajor) {
          for(Mesh::sieve_type::traits::coneSequence::iterator c_iter = begin; c_iter != end; ++c_iter) {
            v[(++c)*size + row] = vNumbering->getIndex(*c_iter)+1;
          }
        } else {
          for(Mesh::sieve_type::traits::coneSequence::iterator c_iter = begin; c_iter != end; ++c_iter) {
            v[row*corners + ++c] = vNumbering->getIndex(*c_iter)+1;
          }
        }
      }
      *numElements = size;
      *numCorners  = corners;
      *vertices    = v;
    };
    #undef __FUNCT__  
    #define __FUNCT__ "PCICEWriteVertices"
    PetscErrorCode Viewer::writeVertices(const ALE::Obj<Mesh>& mesh, PetscViewer viewer) {
      ALE::Obj<Mesh::real_section_type> coordinates = mesh->getRealSection("coordinates");
#if 0
      DMMesh::field_type::patch_type patch;
      const double  *array = coordinates->restrict(patch);
      int            numVertices;
      PetscErrorCode ierr;

      PetscFunctionBegin;
      //FIX:
      if (vertexBundle->getGlobalOffsets()) {
        numVertices = vertexBundle->getGlobalOffsets()[mesh->commSize()];
      } else {
        numVertices = mesh->getTopology()->depthStratum(0)->size();
      }
      ierr = PetscViewerASCIIPrintf(viewer, "%D\n", numVertices);CHKERRQ(ierr);
      if (mesh->commRank() == 0) {
        int numLocalVertices = mesh->getTopology()->depthStratum(0)->size();
        int embedDim = coordinates->getFiberDimension(patch, *mesh->getTopology()->depthStratum(0)->begin());
        int vertexCount = 1;

        for(int v = 0; v < numLocalVertices; v++) {
          ierr = PetscViewerASCIIPrintf(viewer, "%7D   ", vertexCount++);CHKERRQ(ierr);
          for(int d = 0; d < embedDim; d++) {
            if (d > 0) {
              ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "% 12.5E", array[v*embedDim+d]);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
        for(int p = 1; p < mesh->commSize(); p++) {
          double    *remoteCoords;
          MPI_Status status;

          ierr = MPI_Recv(&numLocalVertices, 1, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          ierr = PetscMalloc(numLocalVertices*embedDim * sizeof(double), &remoteCoords);CHKERRQ(ierr);
          ierr = MPI_Recv(remoteCoords, numLocalVertices*embedDim, MPI_DOUBLE, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          for(int v = 0; v < numLocalVertices; v++) {
            ierr = PetscViewerASCIIPrintf(viewer,"%7D   ", vertexCount++);CHKERRQ(ierr);
            for(int d = 0; d < embedDim; d++) {
              if (d > 0) {
                ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
              }
              ierr = PetscViewerASCIIPrintf(viewer, "% 12.5E", remoteCoords[v*embedDim+d]);CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
          }
        }
      } else {
        ALE::Obj<Mesh::bundle_type>                           globalOrder = coordinates->getGlobalOrder();
        ALE::Obj<Mesh::bundle_type::order_type::coneSequence> cone        = globalOrder->getPatch(patch);
        const int *offsets = coordinates->getGlobalOffsets();
        int        embedDim = coordinates->getFiberDimension(patch, *mesh->getTopology()->depthStratum(0)->begin());
        int        numLocalVertices = (offsets[mesh->commRank()+1] - offsets[mesh->commRank()])/embedDim;
        double    *localCoords;
        int        k = 0;

        ierr = PetscMalloc(numLocalVertices*embedDim * sizeof(double), &localCoords);CHKERRQ(ierr);
        for(Mesh::bundle_type::order_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          int dim = globalOrder->getFiberDimension(patch, *p_iter);

          if (dim > 0) {
            int offset = coordinates->getFiberOffset(patch, *p_iter);

            for(int i = offset; i < offset+dim; ++i) {
              localCoords[k++] = array[i];
            }
          }
        }
        if (k != numLocalVertices*embedDim) {
          SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of coordinates to send %d should be %d", k, numLocalVertices*embedDim);
        }
        ierr = MPI_Send(&numLocalVertices, 1, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = MPI_Send(localCoords, numLocalVertices*embedDim, MPI_DOUBLE, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = PetscFree(localCoords);CHKERRQ(ierr);
      }
#endif
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__  
    #define __FUNCT__ "PCICEWriteElements"
    PetscErrorCode Viewer::writeElements(const ALE::Obj<Mesh>& mesh, PetscViewer viewer) {
#if 0
      ALE::Obj<Mesh::sieve_type::traits::heightSequence> elements = topology->heightStratum(0);
      ALE::Obj<Mesh::bundle_type> elementBundle = mesh->getBundle(topology->depth());
      ALE::Obj<Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
      ALE::Obj<Mesh::bundle_type> globalVertex = vertexBundle->getGlobalOrder();
      ALE::Obj<Mesh::bundle_type> globalElement = elementBundle->getGlobalOrder();
      DMMesh::bundle_type::patch_type patch;
      std::string    orderName("element");
      int            dim  = mesh->getDimension();
      int            corners = topology->nCone(*elements->begin(), topology->depth())->size();
      int            numElements;
      PetscErrorCode ierr;

      PetscFunctionBegin;
      if (corners != dim+1) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "PCICE only supports simplicies");
      }
      if (!globalVertex) {
        globalVertex = vertexBundle;
      }
      if (elementBundle->getGlobalOffsets()) {
        numElements = elementBundle->getGlobalOffsets()[mesh->commSize()];
      } else {
        numElements = mesh->getTopology()->heightStratum(0)->size();
      }
      if (mesh->commRank() == 0) {
        int elementCount = 1;

        ierr = PetscViewerASCIIPrintf(viewer, "%d\n", numElements);CHKERRQ(ierr);
        for(Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
          ALE::Obj<Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *e_itor);

          ierr = PetscViewerASCIIPrintf(viewer, "%7d", elementCount++);CHKERRQ(ierr);
          for(Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
            ierr = PetscViewerASCIIPrintf(viewer, " %7d", globalVertex->getIndex(patch, *c_itor).prefix);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
        for(int p = 1; p < mesh->commSize(); p++) {
          int        numLocalElements;
          int       *remoteVertices;
          MPI_Status status;

          ierr = MPI_Recv(&numLocalElements, 1, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          ierr = PetscMalloc(numLocalElements*corners * sizeof(int), &remoteVertices);CHKERRQ(ierr);
          ierr = MPI_Recv(remoteVertices, numLocalElements*corners, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          for(int e = 0; e < numLocalElements; e++) {
            ierr = PetscViewerASCIIPrintf(viewer, "%7d", elementCount++);CHKERRQ(ierr);
            for(int c = 0; c < corners; c++) {
              ierr = PetscViewerASCIIPrintf(viewer, " %7d", remoteVertices[e*corners+c]);CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
          }
          ierr = PetscFree(remoteVertices);CHKERRQ(ierr);
        }
      } else {
        const int *offsets = elementBundle->getGlobalOffsets();
        int        numLocalElements = offsets[mesh->commRank()+1] - offsets[mesh->commRank()];
        int       *localVertices;
        int        k = 0;

        ierr = PetscMalloc(numLocalElements*corners * sizeof(int), &localVertices);CHKERRQ(ierr);
        for(Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
          ALE::Obj<Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *e_itor);

          if (globalElement->getFiberDimension(patch, *e_itor) > 0) {
            for(Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
              localVertices[k++] = globalVertex->getIndex(patch, *c_itor).prefix;
            }
          }
        }
        if (k != numLocalElements*corners) {
          SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of vertices to send %d should be %d", k, numLocalElements*corners);
        }
        ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = MPI_Send(localVertices, numLocalElements*corners, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = PetscFree(localVertices);CHKERRQ(ierr);
      }
#endif
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__  
    #define __FUNCT__ "PCICEWriteVerticesLocal"
    PetscErrorCode Viewer::writeVerticesLocal(const Obj<Mesh>& mesh, PetscViewer viewer) {
      Obj<Mesh::real_section_type>     coordinates = mesh->getRealSection("coordinates");
      const Obj<Mesh::label_sequence>& vertices    = mesh->depthStratum(0);
      const Obj<Mesh::numbering_type>& vNumbering  = mesh->getFactory()->getLocalNumbering(mesh, 0);
      int            embedDim = coordinates->getFiberDimension(*vertices->begin());
      PetscErrorCode ierr;

      PetscFunctionBegin;
      ierr = PetscViewerASCIIPrintf(viewer, "%D\n", vertices->size());CHKERRQ(ierr);
      for(Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
        const DMMesh::real_section_type::value_type *array = coordinates->restrictPoint(*v_iter);

        PetscViewerASCIIPrintf(viewer, "%7D   ", vNumbering->getIndex(*v_iter)+1);
        for(int d = 0; d < embedDim; d++) {
          if (d > 0) {
            PetscViewerASCIIPrintf(viewer, " ");
          }
          PetscViewerASCIIPrintf(viewer, "% 12.5E", array[d]);
        }
        PetscViewerASCIIPrintf(viewer, "\n");
      }
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__  
    #define __FUNCT__ "PCICEWriteRestart"
    PetscErrorCode Viewer::writeRestart(const Obj<Mesh>& mesh, PetscViewer viewer) {
      const Obj<Mesh::real_section_type>&   velocity    = mesh->getRealSection("VELN");
      const Obj<Mesh::real_section_type>&   pressure    = mesh->getRealSection("PN");
      const Obj<Mesh::real_section_type>&   temperature = mesh->getRealSection("TN");
      const Obj<Mesh::numbering_type>& cNumbering  = mesh->getFactory()->getNumbering(mesh, mesh->depth());
      const Obj<Mesh::numbering_type>& vNumbering  = mesh->getFactory()->getNumbering(mesh, 0);
      const int                        numCells    = cNumbering->getGlobalSize();
      PetscErrorCode ierr;

      PetscFunctionBegin;
      int          blen[2];
      MPI_Aint     indices[2];
      MPI_Datatype oldtypes[2], newtype;
      blen[0] = 1; indices[0] = 0;           oldtypes[0] = MPI_INT;
      blen[1] = 4; indices[1] = sizeof(int); oldtypes[1] = MPI_DOUBLE;
      ierr = MPI_Type_struct(2, blen, indices, oldtypes, &newtype);CHKERRQ(ierr);
      ierr = MPI_Type_commit(&newtype);CHKERRQ(ierr);

      if (mesh->commRank() == 0) {
        const Obj<Mesh::label_sequence>& vertices = mesh->depthStratum(0);

        for(Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
          if (vNumbering->isLocal(*v_iter)) {
            const DMMesh::real_section_type::value_type *veln = velocity->restrictPoint(*v_iter);
            const DMMesh::real_section_type::value_type *pn   = pressure->restrictPoint(*v_iter);
            const DMMesh::real_section_type::value_type *tn   = temperature->restrictPoint(*v_iter);

            ierr = PetscViewerASCIIPrintf(viewer, "%6d% 16.8E% 16.8E% 16.8E% 16.8E\n", *v_iter-numCells+1, veln[0], veln[1], pn[0], tn[0]);CHKERRQ(ierr);
          }
        }
        for(int p = 1; p < mesh->commSize(); p++) {
          RestartType *remoteValues;
          int          numLocalElements;
          MPI_Status   status;

          ierr = MPI_Recv(&numLocalElements, 1, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          ierr = PetscMalloc(numLocalElements * sizeof(RestartType), &remoteValues);CHKERRQ(ierr);
          ierr = MPI_Recv(remoteValues, numLocalElements, newtype, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          for(int e = 0; e < numLocalElements; e++) {
            ierr = PetscViewerASCIIPrintf(viewer, "%6d% 16.8E% 16.8E% 16.8E% 16.8E\n", remoteValues[e].vertex-numCells+1, remoteValues[e].veln_x, remoteValues[e].veln_y, remoteValues[e].pn, remoteValues[e].tn);CHKERRQ(ierr);
          }
        }
      } else {
        const Obj<Mesh::label_sequence>& vertices = mesh->depthStratum(0);
        RestartType *localValues;
        int numLocalElements = vNumbering->getLocalSize();
        int k = 0;

        ierr = PetscMalloc(numLocalElements * sizeof(RestartType), &localValues);CHKERRQ(ierr);
        for(Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
          if (vNumbering->isLocal(*v_iter)) {
            const DMMesh::real_section_type::value_type *veln = velocity->restrictPoint(*v_iter);
            const DMMesh::real_section_type::value_type *pn   = pressure->restrictPoint(*v_iter);
            const DMMesh::real_section_type::value_type *tn   = temperature->restrictPoint(*v_iter);

            localValues[k].vertex = *v_iter;
            localValues[k].veln_x = veln[0];
            localValues[k].veln_y = veln[1];
            localValues[k].pn     = pn[0];
            localValues[k].tn     = tn[0];
            k++;
          }
        }
        if (k != numLocalElements) {
          SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of values to send for field, %d should be %d", k, numLocalElements);
        }
        ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = MPI_Send(localValues, numLocalElements, newtype, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = PetscFree(localValues);CHKERRQ(ierr);
      }
      ierr = MPI_Type_free(&newtype);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    };

    //   This class reconstructs the local pieces of the boundary that distributed PCICE needs.
    // The boundary along with the boundary conditions is encoded in a collection of sections
    // over the PCICE mesh.  These sections contain a description of the boundary topology 
    // using elements' global names.  This is unacceptable for PCICE, since it interprets 
    // elements of the connectivity data arrays as local offsets into (some other of) these arrays.
    //   This subroutine performs the renumbering based on the local numbering on the distributed
    // mesh.  Essentially, we replace each global element id by its local number.
    //
    // Note: Any vertex or element number from PCICE is 1-based, but in Sieve we are 0-based. Thus
    //       we add and subtract 1 during conversion. Also, Sieve vertices are numbered after cells.
    void fuseBoundary(const ALE::Obj<PETSC_MESH_TYPE>& mesh) {
      // Extract PCICE boundary sections
      ALE::Obj<PETSC_MESH_TYPE::int_section_type> IBCsec    = mesh->getIntSection("IBC");
      ALE::Obj<PETSC_MESH_TYPE::int_section_type> IBNDFSsec = mesh->getIntSection("IBNDFS");
      ALE::Obj<PETSC_MESH_TYPE::int_section_type> IBCNUMsec = mesh->getIntSection("IBCNUM");

      // Look at the sections, if debugging
      if (mesh->debug()) {
        IBCsec->view("IBCsec", mesh->comm());
        IBNDFSsec->view("IBNDFSsec", mesh->comm());
      }

      // Extract the local numberings
      ALE::Obj<PETSC_MESH_TYPE::numbering_type> vertexNum  = mesh->getFactory()->getLocalNumbering(mesh, 0);
      ALE::Obj<PETSC_MESH_TYPE::numbering_type> elementNum = mesh->getFactory()->getLocalNumbering(mesh, mesh->depth());
      const int numElements = mesh->getFactory()->getNumbering(mesh, mesh->depth())->getGlobalSize();
      std::map<int,int> bfMap;
      // Declare points to the extracted numbering data
      const PETSC_MESH_TYPE::numbering_type::value_type *vNum, *eNum;

      // Create map from serial bdFace numbers to local bdFace numbers
      {
        const PETSC_MESH_TYPE::int_section_type::chart_type           chart = IBCNUMsec->getChart();
        PETSC_MESH_TYPE::int_section_type::chart_type::const_iterator begin = chart.begin();
        PETSC_MESH_TYPE::int_section_type::chart_type::const_iterator end   = chart.end();
        int num = 0;

        for(PETSC_MESH_TYPE::int_section_type::chart_type::const_iterator p_iter = begin; p_iter != end; ++p_iter) {
          const int  fiberDim  = IBCNUMsec->getFiberDimension(*p_iter);
          const int *globalNum = IBCNUMsec->restrictPoint(*p_iter);

          for(int n = 0; n < fiberDim; ++n) {
            bfMap[globalNum[n]] = ++num;
          }
        }
      }
      // Renumber vertex section IBC
      {
        const PETSC_MESH_TYPE::int_section_type::chart_type           IBCchart = IBCsec->getChart();
        PETSC_MESH_TYPE::int_section_type::chart_type::const_iterator begin    = IBCchart.begin();
        PETSC_MESH_TYPE::int_section_type::chart_type::const_iterator end      = IBCchart.end();
        for(PETSC_MESH_TYPE::int_section_type::chart_type::const_iterator p_iter = begin; p_iter != end; ++p_iter) {
          PETSC_MESH_TYPE::point_type p = *p_iter;
          const PETSC_MESH_TYPE::int_section_type::value_type *ibc_in = IBCsec->restrictPoint(p);
          int fiberDimension = IBCsec->getFiberDimension(p);
          PETSC_MESH_TYPE::int_section_type::value_type ibc_out[8];
          // k controls the update of edge connectivity for each containing element;
          // if fiberDimension is 4, only one boundary face is connected to the element, and that edge's data
          // are contained in entries 0 - 3 of the section over the element p;
          // if fiberDimension is 8, two boundary faces are connected to the element, so the second edge's data
          // are contained in entries 4 - 7
          for(int k = 0; k < fiberDimension/4; k++) {
            // Extract IBC entry 1 (entry kk*4) for edge kk connected to element p.
            // This is the entry that needs renumbering for renumbering (2,3 & 4 are invariant under distribution), 
            // see IBC's description.
            // Here we assume that elementNum's domain contains all boundary elements found in IBC, 
            // so we can restrict to the extracted entry.
            eNum = elementNum->restrictPoint((PETSC_MESH_TYPE::numbering_type::point_type) ibc_in[k*4]-1);
            ibc_out[k*4+0] = eNum[0]+1;
            // Copy the other entries right over
            ibc_out[k*4+1] = ibc_in[k*4+1];
            ibc_out[k*4+2] = ibc_in[k*4+2];
            ibc_out[k*4+3] = ibc_in[k*4+3];
          }
          // Update IBC
          IBCsec->updatePoint(p, ibc_out);
        }
      }
      {
        // Renumber vertex section IBNDFS
        const PETSC_MESH_TYPE::int_section_type::chart_type           IBNDFSchart = IBNDFSsec->getChart();
        PETSC_MESH_TYPE::int_section_type::chart_type::const_iterator begin       = IBNDFSchart.begin();
        PETSC_MESH_TYPE::int_section_type::chart_type::const_iterator end         = IBNDFSchart.end();
        for(PETSC_MESH_TYPE::int_section_type::chart_type::const_iterator p_iter = begin; p_iter != end; ++p_iter) {
          PETSC_MESH_TYPE::point_type p = *p_iter;
          const PETSC_MESH_TYPE::int_section_type::value_type *ibndfs_in = IBNDFSsec->restrictPoint(p);
          // Here we assume the fiber dimension is 5
          PETSC_MESH_TYPE::int_section_type::value_type ibndfs_out[5];
          // Renumber entries 1,2 & 3 (4 & 5 are invariant under distribution), see IBNDFS's description
          // Here we assume that vertexNum's domain contains all boundary verticies found in IBNDFS, so we can restrict to the first extracted entry
          vNum= vertexNum->restrictPoint((PETSC_MESH_TYPE::numbering_type::point_type)ibndfs_in[0]-1+numElements);
          ibndfs_out[0] = vNum[0]+1;
          // Map serial bdFace numbers to local bdFace numbers
          ibndfs_out[1] = bfMap[ibndfs_in[1]];
          ibndfs_out[2] = bfMap[ibndfs_in[2]];
          // Copy the other entries right over
          ibndfs_out[3] = ibndfs_in[3];
          ibndfs_out[4] = ibndfs_in[4];
          // Update IBNDFS
          IBNDFSsec->updatePoint(p,ibndfs_out);
        }
      }
      if (mesh->debug()) {
        IBCsec->view("Renumbered IBCsec", mesh->comm());
        IBNDFSsec->view("Renumbered IBNDFSsec", mesh->comm());
      }
      // It's not clear whether IBFCON needs to be renumbered (what does it mean that its entries are not "GLOBAL NODE NUMBER(s)" -- see IBFCON's descriptions
    };
#endif
  };
};
