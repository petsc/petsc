#ifndef included_ALE_Partitioner_hh
#define included_ALE_Partitioner_hh

#ifndef  included_ALE_Completion_hh
#include <Completion.hh>
#endif

#ifdef PETSC_HAVE_ZOLTAN
#include <zoltan.h>

extern "C" {
  // Inputs
  extern int  nvtxs_Zoltan;   // The number of vertices
  extern int  nhedges_Zoltan; // The number of hyperedges
  extern int *eptr_Zoltan;    // The offsets of each hyperedge
  extern int *eind_Zoltan;    // The vertices in each hyperedge, indexed by eptr

  int getNumVertices_Zoltan(void *, int *);

  void getLocalElements_Zoltan(void *, int, int, ZOLTAN_ID_PTR, ZOLTAN_ID_PTR, int, float *, int *);

  void getHgSizes_Zoltan(void *, int *, int *, int *, int *);

  void getHg_Zoltan(void *, int, int, int, int, ZOLTAN_ID_PTR, int *, ZOLTAN_ID_PTR, int *);
}

#endif

#ifdef PETSC_HAVE_CHACO
#ifdef PETSC_HAVE_UNISTD_H
#include <unistd.h>
#endif
/* Chaco does not have an include file */
extern "C" {
  extern int interface(int nvtxs, int *start, int *adjacency, int *vwgts,
                       float *ewgts, float *x, float *y, float *z, char *outassignname,
                       char *outfilename, short *assignment, int architecture, int ndims_tot,
                       int mesh_dims[3], double *goal, int global_method, int local_method,
                       int rqi_flag, int vmax, int ndims, double eigtol, long seed);

  extern int FREE_GRAPH;
}
#endif
#ifdef PETSC_HAVE_PARMETIS
extern "C" {
  #include <parmetis.h>
  extern void METIS_PartGraphKway(int *, int *, int *, int *, int *, int *, int *, int *, int *, int *, int *);
}
#endif
#ifdef PETSC_HAVE_HMETIS
extern "C" {
  extern void HMETIS_PartKway(int nvtxs, int nhedges, int *vwgts, int *eptr, int *eind, int *hewgts, int nparts, int ubfactor, int *options, int *part, int *edgeCut);
}
#endif

namespace ALE {
#if 1
#ifdef PETSC_HAVE_CHACO
  namespace Chaco {
    template<typename Alloc_ = malloc_allocator<short int> >
    class Partitioner {
    public:
      typedef short int part_type;
      typedef Alloc_    alloc_type;
      enum {DEFAULT_METHOD = 1, INERTIAL_METHOD = 3};
    protected:
      const int  logSize;
      char      *msgLog;
      int        fd_stdout, fd_pipe[2];
      alloc_type _allocator;
    public:
      Partitioner(): logSize(10000) {};
      ~Partitioner() {};
    protected:
      // Chaco outputs to stdout. We redirect this to a buffer.
      // TODO: check error codes for UNIX calls
      void startStdoutRedirect() {
#ifdef PETSC_HAVE_UNISTD_H
        this->fd_stdout = dup(1);
        pipe(this->fd_pipe);
        close(1);
        dup2(this->fd_pipe[1], 1);
#endif
      };
      void stopStdoutRedirect() {
#ifdef PETSC_HAVE_UNISTD_H
        int count;

        fflush(stdout);
        this->msgLog = new char[this->logSize];
        count = read(this->fd_pipe[0], this->msgLog, (this->logSize-1)*sizeof(char));
        if (count < 0) count = 0;
        this->msgLog[count] = 0;
        close(1);
        dup2(this->fd_stdout, 1);
        close(this->fd_stdout);
        close(this->fd_pipe[0]);
        close(this->fd_pipe[1]);
        //std::cout << this->msgLog << std::endl;
        delete [] this->msgLog;
#endif
      };
    public:
      static bool zeroBase() {return false;}
      // This method returns the partition section mapping sieve points (here cells) to partitions
      //   start:     start of edge list for each vertex
      //   adjacency: adj[start[v]] is edge list data for vertex v
      //   partition: this section is over the partitions and takes points as values
      // TODO: Read global and local methods from options
      template<typename Section, typename MeshManager>
      void partition(const int numVertices, const int start[], const int adjacency[], const Obj<Section>& partition, const MeshManager& manager) {
        FREE_GRAPH = 0;                         /* Do not let Chaco free my memory */
        int nvtxs = numVertices;                /* number of vertices in full graph */
        int *vwgts = NULL;                      /* weights for all vertices */
        float *ewgts = NULL;                    /* weights for all edges */
        float *x = NULL, *y = NULL, *z = NULL;  /* coordinates for inertial method */
        char *outassignname = NULL;             /*  name of assignment output file */
        char *outfilename = NULL;               /* output file name */
        int architecture = 1;                   /* 0 => hypercube, d => d-dimensional mesh */
        int ndims_tot = 0;                      /* total number of cube dimensions to divide */
        int mesh_dims[3];                       /* dimensions of mesh of processors */
        double *goal = NULL;                    /* desired set sizes for each set */
        int global_method = 1;                  /* global partitioning algorithm */
        int local_method = 1;                   /* local partitioning algorithm */
        int rqi_flag = 0;                       /* should I use RQI/Symmlq eigensolver? */
        int vmax = 200;                         /* how many vertices to coarsen down to? */
        int ndims = 1;                          /* number of eigenvectors (2^d sets) */
        double eigtol = 0.001;                  /* tolerance on eigenvectors */
        long seed = 123636512;                  /* for random graph mutations */
        int maxSize = 0;

	    if (global_method == INERTIAL_METHOD) {manager.createCellCoordinates(nvtxs, &x, &y, &z);}
        mesh_dims[0] = partition->commSize(); mesh_dims[1] = 1; mesh_dims[2] = 1;
        part_type *assignment = this->_allocator.allocate(nvtxs);
        for(int i = 0; i < nvtxs; ++i) {this->_allocator.construct(assignment+i, 0);}

        this->startStdoutRedirect();
        interface(nvtxs, (int *) start, (int *) adjacency, vwgts, ewgts, x, y, z, outassignname, outfilename,
                  assignment, architecture, ndims_tot, mesh_dims, goal, global_method, local_method, rqi_flag,
                  vmax, ndims, eigtol, seed);
        this->stopStdoutRedirect();

        for(int v = 0; v < nvtxs; ++v) {partition->addFiberDimension(assignment[v], 1);}
        partition->allocatePoint();
        for(int p = 0; p < partition->commSize(); ++p) {
          maxSize = std::max(maxSize, partition->getFiberDimension(p));
        }
        typename Section::value_type *values = new typename Section::value_type[maxSize];

        for(int p = 0; p < partition->commSize(); ++p) {
          int k = 0;

          for(int v = 0; v < nvtxs; ++v) {
            if (assignment[v] == p) values[k++] = manager.getCell(v);
          }
          if (k != partition->getFiberDimension(p)) throw ALE::Exception("Invalid partition");
          partition->updatePoint(p, values);
        }
        delete [] values;

	    if (global_method == INERTIAL_METHOD) {manager.destroyCellCoordinates(nvtxs, &x, &y, &z);}
        for(int i = 0; i < nvtxs; ++i) {this->_allocator.destroy(assignment+i);}
        this->_allocator.deallocate(assignment, nvtxs);
      };
    };
  }
#endif
#ifdef PETSC_HAVE_PARMETIS
  namespace ParMetis {
    template<typename Alloc_ = malloc_allocator<int> >
    class Partitioner {
    public:
      typedef int    part_type;
      typedef Alloc_ alloc_type;
    protected:
      alloc_type _allocator;
    public:
      static bool zeroBase() {return true;}
      // This method returns the partition section mapping sieve points (here cells) to partitions
      //   start:     start of edge list for each vertex
      //   adjacency: adj[start[v]] is edge list data for vertex v
      //   partition: this section is over the partitions and takes points as values
      // TODO: Read parameters from options
      template<typename Section, typename MeshManager>
      void partition(const int numVertices, const int start[], const int adjacency[], const Obj<Section>& partition, const MeshManager& manager) {
        //static part_type *partitionSieve(const Obj<bundle_type>& bundle, const int dim) {
        int    nvtxs      = numVertices; // The number of vertices in full graph
        int   *vtxdist;                  // Distribution of vertices across processes
        int   *xadj       = const_cast<int*>(start);       // Start of edge list for each vertex
        int   *adjncy     = const_cast<int*>(adjacency);   // Edge lists for all vertices
        int   *vwgt       = NULL;        // Vertex weights
        int   *adjwgt     = NULL;        // Edge weights
        int    wgtflag    = 0;           // Indicates which weights are present
        int    numflag    = 0;           // Indicates initial offset (0 or 1)
        int    ncon       = 1;           // The number of weights per vertex
        int    nparts     = partition->commSize(); // The number of partitions
        float *tpwgts;                   // The fraction of vertex weights assigned to each partition
        float *ubvec;                    // The balance intolerance for vertex weights
        int    options[5];               // Options
        int    maxSize    = 0;
        // Outputs
        int        edgeCut;              // The number of edges cut by the partition
        part_type *assignment;

        options[0] = 0; // Use all defaults
        // Calculate vertex distribution
        //   Not sure this still works in parallel
        vtxdist    = new int[nparts+1];
        vtxdist[0] = 0;
        MPI_Allgather(&nvtxs, 1, MPI_INT, &vtxdist[1], 1, MPI_INT, partition->comm());
        for(int p = 2; p <= nparts; ++p) {
          vtxdist[p] += vtxdist[p-1];
        }
        // Calculate weights
        tpwgts     = new float[ncon*nparts];
        for(int p = 0; p < nparts; ++p) {
          tpwgts[p] = 1.0/nparts;
        }
        ubvec      = new float[ncon];
        ubvec[0]   = 1.05;

        assignment = this->_allocator.allocate(nvtxs);
        for(int i = 0; i < nvtxs; ++i) {this->_allocator.construct(assignment+i, 0);}

        if (partition->commSize() == 1) {
          PetscMemzero(assignment, nvtxs * sizeof(part_type));
        } else {
          if (partition->debug() && nvtxs) {
            for(int p = 0; p <= nvtxs; ++p) {
              std::cout << "["<<partition->commRank()<<"]xadj["<<p<<"] = " << xadj[p] << std::endl;
            }
            for(int i = 0; i < xadj[nvtxs]; ++i) {
              std::cout << "["<<partition->commRank()<<"]adjncy["<<i<<"] = " << adjncy[i] << std::endl;
            }
          }
          if (vtxdist[1] == vtxdist[nparts]) {
            if (partition->commRank() == 0) {
              METIS_PartGraphKway(&nvtxs, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &nparts, options, &edgeCut, assignment);
              if (partition->debug()) {std::cout << "Metis: edgecut is " << edgeCut << std::endl;}
            }
          } else {
            MPI_Comm comm = partition->comm();

            ParMETIS_V3_PartKway(vtxdist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgeCut, assignment, &comm);
            if (partition->debug()) {std::cout << "ParMetis: edgecut is " << edgeCut << std::endl;}
          }
        }
        delete [] vtxdist;
        delete [] tpwgts;
        delete [] ubvec;

        for(int v = 0; v < nvtxs; ++v) {partition->addFiberDimension(assignment[v], 1);}
        partition->allocatePoint();
        for(int p = 0; p < partition->commSize(); ++p) {
          maxSize = std::max(maxSize, partition->getFiberDimension(p));
        }
        typename Section::value_type *values = new typename Section::value_type[maxSize];

        for(int p = 0; p < partition->commSize(); ++p) {
          int k = 0;

          for(int v = 0; v < nvtxs; ++v) {
            if (assignment[v] == p) values[k++] = manager.getCell(v);
          }
          if (k != partition->getFiberDimension(p)) throw ALE::Exception("Invalid partition");
          partition->updatePoint(p, values);
        }
        delete [] values;

        for(int i = 0; i < nvtxs; ++i) {this->_allocator.destroy(assignment+i);}
        this->_allocator.deallocate(assignment, nvtxs);
      };
    };
  };
#endif
  namespace Simple {
    template<typename Alloc_ = malloc_allocator<short int> >
    class Partitioner {
    public:
      typedef int    part_type;
      typedef Alloc_ alloc_type;
    protected:
      alloc_type _allocator;
    public:
      Partitioner() {};
      ~Partitioner() {};
    public:
      static bool zeroBase() {return true;}
      template<typename Section, typename MeshManager>
      void partition(const int numVertices, const int start[], const int adjacency[], const Obj<Section>& partition, const MeshManager& manager) {
        const int numProcs = partition->commSize();
        const int rank     = partition->commRank();
        int       maxSize  = 0;

        for(int p = 0; p < numProcs; ++p) {
          partition->setFiberDimension(p, numVertices/numProcs + ((numVertices % numProcs) > rank));
          maxSize = std::max(maxSize, partition->getFiberDimension(p));
        }
        partition->allocatePoint();
        typename Section::value_type *values = new typename Section::value_type[maxSize];

        for(int p = 0; p < partition->commSize(); ++p) {
          const int start = p*(numVertices/numProcs)     + p*((numVertices % numProcs) > p+1);
          const int end   = (p+1)*(numVertices/numProcs) + (p+1)*((numVertices % numProcs) > p+2);
          int       k     = 0;

          for(int v = start; v < end; ++v, ++k) {
            values[k] = manager.getCell(v);
          }
          if (k != partition->getFiberDimension(p)) throw ALE::Exception("Invalid partition");
          partition->updatePoint(p, values);
        }
        delete [] values;
      };
    };
  }
#ifdef PETSC_HAVE_CHACO
  template<typename GraphPartitioner = ALE::Chaco::Partitioner<>, typename Alloc_ = malloc_allocator<int> >
#elif defined(PETSC_HAVE_PARMETIS)
  template<typename GraphPartitioner = ALE::ParMetis::Partitioner<>, typename Alloc_ = malloc_allocator<int> >
#else
  template<typename GraphPartitioner = ALE::Simple::Partitioner<>, typename Alloc_ = malloc_allocator<int> >
#endif
  class Partitioner {
  public:
    typedef Alloc_                               alloc_type;
    typedef GraphPartitioner                     graph_partitioner_type;
    typedef typename GraphPartitioner::part_type part_type;
    template<typename Mesh>
    class MeshManager {
    public:
      typedef typename Mesh::point_type point_type;
    protected:
      const Obj<Mesh>& mesh;
      bool             simpleCellNumbering;
      point_type      *cells;
    protected:
      void createCells(const int height) {
        const Obj<typename Mesh::label_sequence>&     mcells   = mesh->heightStratum(height);
        const typename Mesh::label_sequence::iterator cEnd     = mcells->end();
        const int                                     numCells = mcells->size();
        int                                           c        = 0;

        this->cells               = NULL;
        this->simpleCellNumbering = true;
        for(typename Mesh::label_sequence::iterator c_iter = mcells->begin(); c_iter != cEnd; ++c_iter, ++c) {
          if (*c_iter != c) {
            this->simpleCellNumbering = false;
            break;
          }
        }
        if (!this->simpleCellNumbering) {
          this->cells = new point_type[numCells];
          c           = 0;
          for(typename Mesh::label_sequence::iterator c_iter = mcells->begin(); c_iter != cEnd; ++c_iter, ++c) {
            this->cells[c] = *c_iter;
          }
        }
      };
    public:
      MeshManager(const Obj<Mesh>& mesh, const int height = 0): mesh(mesh) {
        this->createCells(height);
      };
      ~MeshManager() {
        if (this->cells) {delete [] this->cells;}
      };
    public:
      template<typename Float>
      void createCellCoordinates(const int numVertices, Float *X[], Float *Y[], Float *Z[]) const {
        const Obj<typename Mesh::real_section_type>& coordinates = mesh->getRealSection("coordinates");
        const int dim = mesh->getDimension();
        typedef typename alloc_type::template rebind<Float>::other float_alloc_type;
        Float *x = float_alloc_type().allocate(numVertices*3);
        for(int i = 0; i < numVertices*3; ++i) {float_alloc_type().construct(x+i, 0.0);}
        Float *y = x+numVertices;
        Float *z = y+numVertices;
        Float *vCoords[3];

        vCoords[0] = x; vCoords[1] = y; vCoords[2] = z;
        const Obj<typename Mesh::label_sequence>& cells = mesh->heightStratum(0);
        const int corners = mesh->size(coordinates, *(cells->begin()))/dim;
        int       c       = 0;

        for(typename Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter !=cells->end(); ++c_iter, ++c) {
          const double *coords = mesh->restrictClosure(coordinates, *c_iter);

          for(int d = 0; d < dim; ++d) {
            vCoords[d][c] = 0.0;
          }
          for(int v = 0; v < corners; ++v) {
            for(int d = 0; d < dim; ++d) {
              vCoords[d][c] += coords[v*dim+d];
            }
          }
          for(int d = 0; d < dim; ++d) {
            vCoords[d][c] /= corners;
          }
        }
        *X = x;
        *Y = y;
        *Z = z;
      };
      template<typename Float>
      void destroyCellCoordinates(const int numVertices, Float *X[], Float *Y[], Float *Z[]) const {
        typedef typename alloc_type::template rebind<Float>::other float_alloc_type;
        Float *x = *X;

        for(int i = 0; i < numVertices*3; ++i) {float_alloc_type().destroy(x+i);}
        float_alloc_type().deallocate(x, numVertices*3);
      };
      point_type getCell(const int cellNumber) const {
        if (this->simpleCellNumbering) {
          return cellNumber;
        }
        return this->cells[cellNumber];
      };
    };
    template<typename Sieve>
    class OffsetVisitor {
      const Sieve& sieve;
      const Sieve& overlapSieve;
      int         *offsets;
    public:
      OffsetVisitor(const Sieve& s, const Sieve& ovS, int off[]) : sieve(s), overlapSieve(ovS), offsets(off) {};
      void visitPoint(const typename Sieve::point_type& point) {};
      void visitArrow(const typename Sieve::arrow_type& arrow) {
        const typename Sieve::point_type cell   = arrow.target;
        const typename Sieve::point_type face   = arrow.source;
        const int                        size   = this->sieve.getSupportSize(face);
        const int                        ovSize = this->overlapSieve.getSupportSize(face);

        if (size == 2) {
          offsets[cell+1]++;
        } else if ((size == 1) && (ovSize == 1)) {
          offsets[cell+1]++;
        }
      };
    };
    template<typename Sieve>
    class AdjVisitor {
    protected:
      typename Sieve::point_type cell;
      int                       *adjacency;
      const int                  cellOffset;
      int                        offset;
    public:
      AdjVisitor(int adj[], const bool zeroBase) : adjacency(adj), cellOffset(zeroBase ? 0 : 1), offset(0) {};
      void visitPoint(const typename Sieve::point_type& point) {};
      void visitArrow(const typename Sieve::arrow_type& arrow) {
        const int neighbor = arrow.target;

        if (neighbor != this->cell) {
          //std::cout << "Adding dual edge from " << cell << " to " << neighbor << std::endl;
          this->adjacency[this->offset++] = neighbor + this->cellOffset;
        }
      };
    public:
      void setCell(const typename Sieve::point_type cell) {this->cell = cell;};
      int  getOffset() {return this->offset;}
    };
    template<typename Sieve>
    class MeetVisitor {
    public:
      typedef std::set<typename Sieve::point_type> neighbors_type;
    protected:
      const Sieve& sieve;
      const int numCells;
      const int faceVertices;
      typename Sieve::point_type newCell;
      neighbors_type *neighborCells;
      typename ISieveVisitor::PointRetriever<Sieve> *pR;
      typename Sieve::point_type cell;
      std::map<typename Sieve::point_type, typename Sieve::point_type> newCells;
    public:
      MeetVisitor(const Sieve& s, const int n, const int fV) : sieve(s), numCells(n), faceVertices(fV), newCell(n) {
        this->neighborCells = new std::set<typename Sieve::point_type>[numCells];
        this->pR            = new typename ISieveVisitor::PointRetriever<Sieve>(this->sieve.getMaxConeSize());
      };
      ~MeetVisitor() {delete [] this->neighborCells; delete this->pR;};
      void visitArrow(const typename Sieve::arrow_type& arrow) {};
      void visitPoint(const typename Sieve::point_type& point) {
        const typename Sieve::point_type& neighbor = point;

        if (this->cell == neighbor) return;
        this->pR->clear();
        this->sieve.meet(this->cell, neighbor, *this->pR);
        if (this->pR->getSize() == (size_t) this->faceVertices) {
          if ((this->cell < numCells) && (neighbor < numCells)) {
            this->neighborCells[this->cell].insert(neighbor);
          } else {
            typename Sieve::point_type e = this->cell, n = neighbor;

            if (this->cell >= numCells) {
              if (this->newCells.find(cell) == this->newCells.end()) this->newCells[cell] = --newCell;
              e = this->newCells[cell];
            }
            if (neighbor >= numCells) {
              if (this->newCells.find(neighbor) == this->newCells.end()) this->newCells[neighbor] = --newCell;
              n = this->newCells[neighbor];
            }
            this->neighborCells[e].insert(n);
          }
        }
      };
    public:
      void setCell(const typename Sieve::point_type& c) {this->cell = c;};
      const neighbors_type *getNeighbors() {return this->neighborCells;};
    };
  public: // Creating overlaps
    // Create a partition point overlap for distribution
    //   This is the default overlap which comes from distributing a serial mesh on process 0
    template<typename SendOverlap, typename RecvOverlap>
    static void createDistributionPartOverlap(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap) {
      const int rank = sendOverlap->commRank();
      const int size = sendOverlap->commSize();

      if (rank == 0) {
        for(int p = 1; p < size; p++) {
          // The arrow is from local partition point p (source) to remote partition point p (color) on rank p (target)
          sendOverlap->addCone(p, p, p);
        }
      }
      if (rank != 0) {
        // The arrow is from remote partition point rank (color) on rank 0 (source) to local partition point rank (target)
        recvOverlap->addCone(0, rank, rank);
      }
    };
    // Create a mesh point overlap for distribution
    //   A local numbering is created for the remote points
    //   This is the default overlap which comes from distributing a serial mesh on process 0
    template<typename Section, typename RecvPartOverlap, typename Renumbering, typename SendOverlap, typename RecvOverlap>
    static void createDistributionMeshOverlap(const Obj<Section>& partition, const Obj<RecvPartOverlap>& recvPartOverlap, Renumbering& renumbering, const Obj<Section>& overlapPartition, const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap) {
      const typename Section::chart_type& chart = partition->getChart();

      for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        if (*p_iter == sendOverlap->commRank()) continue;
        const typename Section::value_type *points     = partition->restrictPoint(*p_iter);
        const int                           numPoints  = partition->getFiberDimension(*p_iter);
        
        for(int i = 0; i < numPoints; ++i) {
          // Notice here that we do not know the local renumbering (but we do not use it)
          sendOverlap->addArrow(points[i], *p_iter, points[i]);
        }
      }
      if (sendOverlap->debug()) {sendOverlap->view("Send mesh overlap");}
      const Obj<typename RecvPartOverlap::traits::baseSequence> rPoints    = recvPartOverlap->base();

      for(typename RecvPartOverlap::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rPoints->end(); ++p_iter) {
        const Obj<typename RecvPartOverlap::coneSequence>& ranks           = recvPartOverlap->cone(*p_iter);
        //const typename Section::point_type&                localPartPoint  = *p_iter;
        const typename Section::point_type                 rank            = *ranks->begin();
        const typename Section::point_type&                remotePartPoint = ranks->begin().color();
        const typename Section::value_type                *points          = overlapPartition->restrictPoint(remotePartPoint);
        const int                                          numPoints       = overlapPartition->getFiberDimension(remotePartPoint);

        for(int i = 0; i < numPoints; ++i) {
          recvOverlap->addArrow(rank, renumbering[points[i]], points[i]);
        }
      }
      if (recvOverlap->debug()) {recvOverlap->view("Receive mesh overlap");}
    };
    // Create a partition point overlap from a partition
    //   The intention is to create an overlap which enables exchange of redistribution information
    template<typename Section, typename SendOverlap, typename RecvOverlap>
    static void createPartitionPartOverlap(const Obj<Section>& partition, const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap) {
      const typename Section::chart_type& chart      = partition->getChart();
      const int                           rank       = partition->commRank();
      const int                           size       = partition->commSize();
      int                                *adj        = new int[size];
      int                                *recvCounts = new int[size];
      int                                 numNeighbors;

      for(int p = 0; p < size; ++p) {
        adj[p]        = 0;
        recvCounts[p] = 1;
      }
      for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart->end(); ++p_iter) {
        const typename Section::value_type& p = partition->restrictPoint(*p_iter)[0];
        // The arrow is from local partition point p (source) to remote partition point p (color) on rank p (target)
        sendOverlap->addCone(p, p, p);
        adj[p] = 1;
      }
      MPI_Reduce_Scatter(adj, &numNeighbors, recvCounts, size, MPI_INT, MPI_SUM, partition->comm());
      MPI_Request *recvRequests = new MPI_Request[numNeighbors];
      int          dummy        = 0;

      // TODO: Get a unique tag
      for(int n = 0; n < numNeighbors; ++n) {
        MPI_Irecv(&dummy, 1, MPI_INT, MPI_ANY_SOURCE, 1, partition->comm(), &recvRequests[n]);
      }
      const Obj<typename SendOverlap::traits::baseSequence>      ranks        = sendOverlap->base();
      const typename SendOverlap::traits::baseSequence::iterator rEnd         = ranks->end();
      MPI_Request                                               *sendRequests = new MPI_Request[ranks->size()];
      int                                                        s            = 0;

      for(typename SendOverlap::traits::baseSequence::iterator r_iter = ranks->begin(); r_iter != rEnd; ++r_iter, ++s) {
        MPI_Isend(&dummy, 1, MPI_INT, *r_iter, 1, partition->comm(), &sendRequests[s]);
      }
      for(int n = 0; n < numNeighbors; ++n) {
        MPI_Status status;
        int        idx;

        MPI_Waitany(numNeighbors, recvRequests, &idx, &status);
        // The arrow is from remote partition point rank (color) on rank p (source) to local partition point rank (target)
        recvOverlap->addCone(status.MPI_SOURCE, rank, rank);
      }
      MPI_Waitall(ranks->size(), sendRequests, MPI_STATUSES_IGNORE);
      delete [] sendRequests;
      delete [] recvRequests;
      delete [] adj;
      delete [] recvCounts;
    };
  public: // Building CSR meshes
    // This produces the dual graph (each cell is a vertex and each face is an edge)
    //   numbering:   A contiguous numbering of the cells (not yet included)
    //   numVertices: The number of vertices in the graph (cells in the mesh)
    //   adjacency:   The vertices adjacent to each vertex (cells adjacent to each mesh cell)
    // - We allow an exception to contiguous numbering.
    //   If the cell id > numElements, we assign a new number starting at
    //     the top and going downward. I know these might not match up with
    //     the iterator order, but we can fix it later.
    template<typename Mesh>
    static void buildDualCSR(const Obj<Mesh>& mesh, int *numVertices, int **offsets, int **adjacency, const bool zeroBase = true) {
      const Obj<typename Mesh::sieve_type>&         sieve        = mesh->getSieve();
      const Obj<typename Mesh::label_sequence>&     cells        = mesh->heightStratum(0);
      const typename Mesh::label_sequence::iterator cEnd         = cells->end();
      const int                                     numCells     = cells->size();
      int                                           newCell      = numCells;
      Obj<typename Mesh::sieve_type>                overlapSieve = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
      int                                           offset       = 0;
      const int                                     cellOffset   = zeroBase ? 0 : 1;
      const int                                     dim          = mesh->getDimension();
      std::map<typename Mesh::point_type, typename Mesh::point_type> newCells;

      // TODO: This is necessary for parallel partitioning
      //completion::scatterSupports(sieve, overlapSieve, mesh->getSendOverlap(), mesh->getRecvOverlap(), mesh);
      if (numCells == 0) {
        *numVertices = 0;
        *offsets     = NULL;
        *adjacency   = NULL;
        return;
      }
      int *off = alloc_type().allocate(numCells+1);
      int *adj;
      for(int i = 0; i < numCells+1; ++i) {alloc_type().construct(off+i, 0);}
      if (mesh->depth() == dim) {
        int c = 1;

        for(typename Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cEnd; ++c_iter, ++c) {
          const Obj<typename Mesh::sieve_type::traits::coneSequence>&     faces = sieve->cone(*c_iter);
          const typename Mesh::sieve_type::traits::coneSequence::iterator fEnd  = faces->end();

          off[c] = off[c-1];
          for(typename Mesh::sieve_type::traits::coneSequence::iterator f_iter = faces->begin(); f_iter != fEnd; ++f_iter) {
            if (sieve->support(*f_iter)->size() == 2) {
              off[c]++;
            } else if ((sieve->support(*f_iter)->size() == 1) && (overlapSieve->support(*f_iter)->size() == 1)) {
              off[c]++;
            }
          }
        }
        adj = alloc_type().allocate(off[numCells]);
        for(int i = 0; i < off[numCells]; ++i) {alloc_type().construct(adj+i, 0);}
        for(typename Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cEnd; ++c_iter) {
          const Obj<typename Mesh::sieve_type::traits::coneSequence>&     faces = sieve->cone(*c_iter);
          const typename Mesh::sieve_type::traits::coneSequence::iterator fEnd  = faces->end();

          for(typename Mesh::sieve_type::traits::coneSequence::iterator f_iter = faces->begin(); f_iter != fEnd; ++f_iter) {
            const Obj<typename Mesh::sieve_type::traits::supportSequence>&     neighbors = sieve->support(*f_iter);
            const typename Mesh::sieve_type::traits::supportSequence::iterator nEnd      = neighbors->end();

            for(typename Mesh::sieve_type::traits::supportSequence::iterator n_iter = neighbors->begin(); n_iter != nEnd; ++n_iter) {
              if (*n_iter != *c_iter) adj[offset++] = *n_iter + cellOffset;
            }
            const Obj<typename Mesh::sieve_type::traits::supportSequence>&     oNeighbors = overlapSieve->support(*f_iter);
            const typename Mesh::sieve_type::traits::supportSequence::iterator onEnd      = oNeighbors->end();

            for(typename Mesh::sieve_type::traits::supportSequence::iterator n_iter = oNeighbors->begin(); n_iter != onEnd; ++n_iter) {
              adj[offset++] = *n_iter + cellOffset;
            }
          }
        }
      } else if (mesh->depth() == 1) {
        std::set<typename Mesh::point_type> *neighborCells = new std::set<typename Mesh::point_type>[numCells];
        const int                            corners       = sieve->cone(*cells->begin())->size();
        int                                  faceVertices;

        if (corners == dim+1) {
          faceVertices = dim;
        } else if ((dim == 2) && (corners == 4)) {
          faceVertices = 2;
        } else if ((dim == 3) && (corners == 8)) {
          faceVertices = 4;
        } else {
          throw ALE::Exception("Could not determine number of face vertices");
        }
        for(typename Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
            const Obj<typename Mesh::sieve_type::traits::coneSequence>&     vertices = sieve->cone(*c_iter);
            const typename Mesh::sieve_type::traits::coneSequence::iterator vEnd     = vertices->end();

            for(typename Mesh::sieve_type::traits::coneSequence::iterator v_iter = vertices->begin(); v_iter != vEnd; ++v_iter) {
              const Obj<typename Mesh::sieve_type::traits::supportSequence>&     neighbors = sieve->support(*v_iter);
              const typename Mesh::sieve_type::traits::supportSequence::iterator nEnd      = neighbors->end();

              for(typename Mesh::sieve_type::traits::supportSequence::iterator n_iter = neighbors->begin(); n_iter != nEnd; ++n_iter) {
                if (*c_iter == *n_iter) continue;
                if ((int) sieve->nMeet(*c_iter, *n_iter, 1)->size() == faceVertices) {
                  if ((*c_iter < numCells) && (*n_iter < numCells)) {
                    neighborCells[*c_iter].insert(*n_iter);
                  } else {
                    typename Mesh::point_type e = *c_iter, n = *n_iter;

                    if (*c_iter >= numCells) {
                      if (newCells.find(*c_iter) == newCells.end()) newCells[*c_iter] = --newCell;
                      e = newCells[*c_iter];
                    }
                    if (*n_iter >= numCells) {
                      if (newCells.find(*n_iter) == newCells.end()) newCells[*n_iter] = --newCell;
                      n = newCells[*n_iter];
                    }
                    neighborCells[e].insert(n);
                  }
                }
              }
            }
          }
          off[0] = 0;
          for(int c = 1; c <= numCells; c++) {
            off[c] = neighborCells[c-1].size() + off[c-1];
          }
          adj = alloc_type().allocate(off[numCells]);
          for(int i = 0; i < off[numCells]; ++i) {alloc_type().construct(adj+i, 0);}
          for(int c = 0; c < numCells; c++) {
            for(typename std::set<typename Mesh::point_type>::iterator n_iter = neighborCells[c].begin(); n_iter != neighborCells[c].end(); ++n_iter) {
              adj[offset++] = *n_iter + cellOffset;
            }
          }
          delete [] neighborCells;
      } else {
        throw ALE::Exception("Dual creation not defined for partially interpolated meshes");
      }
      if (offset != off[numCells]) {
        ostringstream msg;
        msg << "ERROR: Total number of neighbors " << offset << " does not match the offset array " << off[numCells];
        throw ALE::Exception(msg.str().c_str());
      }
      *numVertices = numCells;
      *offsets     = off;
      *adjacency   = adj;
    };
    template<typename Mesh>
    static void buildDualCSRV(const Obj<Mesh>& mesh, int *numVertices, int **offsets, int **adjacency, const bool zeroBase = true) {
      const Obj<typename Mesh::sieve_type>&         sieve        = mesh->getSieve();
      const Obj<typename Mesh::label_sequence>&     cells        = mesh->heightStratum(0);
      const typename Mesh::label_sequence::iterator cEnd         = cells->end();
      const int                                     numCells     = cells->size();
      Obj<typename Mesh::sieve_type>                overlapSieve = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
      int                                           offset       = 0;
      const int                                     cellOffset   = zeroBase ? 0 : 1;
      const int                                     dim          = mesh->getDimension();
      std::map<typename Mesh::point_type, typename Mesh::point_type> newCells;

      // TODO: This is necessary for parallel partitioning
      //completion::scatterSupports(sieve, overlapSieve, mesh->getSendOverlap(), mesh->getRecvOverlap(), mesh);
      overlapSieve->setChart(sieve->getChart());
      overlapSieve->allocate();
      if (numCells == 0) {
        *numVertices = 0;
        *offsets     = NULL;
        *adjacency   = NULL;
        return;
      }
      int *off = alloc_type().allocate(numCells+1);
      int *adj;
      for(int i = 0; i < numCells+1; ++i) {alloc_type().construct(off+i, 0);}
      if (mesh->depth() == dim) {
        OffsetVisitor<typename Mesh::sieve_type> oV(*sieve, *overlapSieve, off);

        for(typename Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cEnd; ++c_iter) {
          sieve->cone(*c_iter, oV);
        }
        for(int p = 1; p <= numCells; ++p) {
          off[p] = off[p] + off[p-1];
        }
        adj = alloc_type().allocate(off[numCells]);
        for(int i = 0; i < off[numCells]; ++i) {alloc_type().construct(adj+i, 0);}
        AdjVisitor<typename Mesh::sieve_type> aV(adj, zeroBase);
        ISieveVisitor::SupportVisitor<typename Mesh::sieve_type, AdjVisitor<typename Mesh::sieve_type> > sV(*sieve, aV);
        ISieveVisitor::SupportVisitor<typename Mesh::sieve_type, AdjVisitor<typename Mesh::sieve_type> > ovSV(*overlapSieve, aV);

        for(typename Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cEnd; ++c_iter) {
          aV.setCell(*c_iter);
          sieve->cone(*c_iter, sV);
          sieve->cone(*c_iter, ovSV);
        }
        offset = aV.getOffset();
      } else if (mesh->depth() == 1) {
        typedef MeetVisitor<typename Mesh::sieve_type>    mv_type;
        typedef typename ISieveVisitor::SupportVisitor<typename Mesh::sieve_type, mv_type> sv_type;
        const int corners = sieve->getConeSize(*cells->begin());
        int       faceVertices;

        if (corners == dim+1) {
          faceVertices = dim;
        } else if ((dim == 2) && (corners == 4)) {
          faceVertices = 2;
        } else if ((dim == 3) && (corners == 8)) {
          faceVertices = 4;
        } else {
          throw ALE::Exception("Could not determine number of face vertices");
        }
        mv_type mV(*sieve, numCells, faceVertices);
        sv_type sV(*sieve, mV);

        for(typename Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
          mV.setCell(*c_iter);
          sieve->cone(*c_iter, sV);
        }
        const typename mv_type::neighbors_type *neighborCells = mV.getNeighbors();

        off[0] = 0;
        for(int c = 1; c <= numCells; c++) {
          off[c] = neighborCells[c-1].size() + off[c-1];
        }
        adj = alloc_type().allocate(off[numCells]);
        for(int i = 0; i < off[numCells]; ++i) {alloc_type().construct(adj+i, 0);}
        for(int c = 0; c < numCells; c++) {
          for(typename mv_type::neighbors_type::const_iterator n_iter = neighborCells[c].begin(); n_iter != neighborCells[c].end(); ++n_iter) {
            //std::cout << "Adding dual edge from " << c << " to " << *n_iter << std::endl;
            adj[offset++] = *n_iter + cellOffset;
          }
        }
      } else {
        throw ALE::Exception("Dual creation not defined for partially interpolated meshes");
      }
      if (offset != off[numCells]) {
        ostringstream msg;
        msg << "ERROR: Total number of neighbors " << offset << " does not match the offset array " << off[numCells];
        throw ALE::Exception(msg.str().c_str());
      }
      *numVertices = numCells;
      *offsets     = off;
      *adjacency   = adj;
    };
    // This produces a hypergraph (each face is a vertex and each cell is a hyperedge)
    //   numbering: A contiguous numbering of the faces
    //   numEdges:  The number of edges in the hypergraph
    //   adjacency: The vertices in each edge
    template<typename Mesh>
    static void buildFaceDualCSR(const Obj<Mesh>& mesh, const Obj<typename Mesh::numbering_type>& numbering, int *numEdges, int **offsets, int **adjacency, const bool zeroBase = true) {
      const Obj<typename Mesh::sieve_type>&         sieve = mesh->getSieve();
      const Obj<typename Mesh::label_sequence>&     cells = mesh->heightStratum(0);
      const typename Mesh::label_sequence::iterator cEnd  = cells->end();
      const int faceOffset = zeroBase ? 0 : 1;
      int       numCells = cells->size();
      int       c        = 1;

      if (mesh->depth() != mesh->getDimension()) {throw ALE::Exception("Not yet implemented for non-interpolated meshes");}
      int *off = alloc_type().allocate(numCells+1);
      for(int i = 0; i < numCells+1; ++i) {alloc_type().construct(off+i, 0);}
      for(typename Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cEnd; ++c_iter, ++c) {
        off[c] = sieve->cone(*c_iter)->size() + off[c-1];
      }
      int *adj = alloc_type().allocate(off[numCells]);
      for(int i = 0; i < off[numCells]; ++i) {alloc_type().construct(adj+i, 0);}
      int  offset = 0;
      for(typename Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cEnd; ++c_iter) {
        const Obj<typename Mesh::sieve_type::traits::coneSequence>&     faces = sieve->cone(*c_iter);
        const typename Mesh::sieve_type::traits::coneSequence::iterator fEnd  = faces->end();

        for(typename Mesh::sieve_type::traits::coneSequence::iterator f_iter = faces->begin(); f_iter != fEnd; ++f_iter, ++offset) {
          adj[offset] = numbering->getIndex(*f_iter) + faceOffset;
        }
      }
      if (offset != off[numCells]) {
        ostringstream msg;
        msg << "ERROR: Total number of neighbors " << offset << " does not match the offset array " << off[numCells];
        throw ALE::Exception(msg.str().c_str());
      }
      *numEdges  = numCells;
      *offsets   = off;
      *adjacency = adj;
    };
    template<typename Mesh>
    static void buildFaceDualCSRV(const Obj<Mesh>& mesh, const Obj<typename Mesh::numbering_type>& numbering, int *numEdges, int **offsets, int **adjacency, const bool zeroBase = true) {
      throw ALE::Exception("Not implemented");
    };
    static void destroyCSR(int numPoints, int *offsets, int *adjacency) {
      if (adjacency) {
        for(int i = 0; i < offsets[numPoints]; ++i) {alloc_type().destroy(adjacency+i);}
        alloc_type().deallocate(adjacency, offsets[numPoints]);
      }
      if (offsets) {
        for(int i = 0; i < numPoints+1; ++i) {alloc_type().destroy(offsets+i);}
        alloc_type().deallocate(offsets, numPoints+1);
      }
    };
    template<typename OldSection, typename Partition, typename Renumbering, typename NewSection>
    static void createLocalSection(const Obj<OldSection>& oldSection, const Obj<Partition>& partition, Renumbering& renumbering, const Obj<NewSection>& newSection) {
      const typename Partition::value_type *points    = partition->restrictPoint(oldSection->commRank());
      const int                             numPoints = partition->getFiberDimension(oldSection->commRank());

      for(int p = 0; p < numPoints; ++p) {
        if (oldSection->hasPoint(points[p])) {
          newSection->setFiberDimension(renumbering[points[p]], oldSection->getFiberDimension(points[p]));
        }
      }
      if (numPoints) {newSection->allocatePoint();}
      for(int p = 0; p < numPoints; ++p) {
        if (oldSection->hasPoint(points[p])) {
          newSection->updatePointAll(renumbering[points[p]], oldSection->restrictPoint(points[p]));
        }
      }
    };
    // Specialize to ArrowSection
    template<typename OldSection, typename Partition, typename Renumbering>
    static void createLocalSection(const Obj<OldSection>& oldSection, const Obj<Partition>& partition, Renumbering& renumbering, const Obj<UniformSection<MinimalArrow<int,int>,int> >& newSection) {
      typedef UniformSection<MinimalArrow<int,int>,int> NewSection;
      const typename Partition::value_type    *points    = partition->restrictPoint(oldSection->commRank());
      const int                                numPoints = partition->getFiberDimension(oldSection->commRank());
      const typename OldSection::chart_type&   oldChart  = oldSection->getChart();
      std::set<typename Partition::value_type> myPoints;

      for(int p = 0; p < numPoints; ++p) {
        myPoints.insert(points[p]);
      }
      for(typename OldSection::chart_type::const_iterator c_iter = oldChart.begin(); c_iter != oldChart.end(); ++c_iter) {
        if (myPoints.count(c_iter->source) && myPoints.count(c_iter->target)) {
          newSection->setFiberDimension(typename OldSection::point_type(renumbering[c_iter->source], renumbering[c_iter->target]), oldSection->getFiberDimension(*c_iter));
        }
      }
      if (oldChart.size()) {newSection->allocatePoint();}
      for(typename OldSection::chart_type::const_iterator c_iter = oldChart.begin(); c_iter != oldChart.end(); ++c_iter) {
        if (myPoints.count(c_iter->source) && myPoints.count(c_iter->target)) {
          const typename OldSection::value_type *values = oldSection->restrictPoint(*c_iter);

          newSection->updatePointAll(typename OldSection::point_type(renumbering[c_iter->source], renumbering[c_iter->target]), values);
        }
      }
    };
    template<typename Sifter, typename Section, typename Renumbering>
    static void createLocalSifter(const Obj<Sifter>& sifter, const Obj<Section>& partition, Renumbering& renumbering, const Obj<Sifter>& localSifter) {
      const typename Section::value_type *points    = partition->restrictPoint(sifter->commRank());
      const int                           numPoints = partition->getFiberDimension(sifter->commRank());

      for(int p = 0; p < numPoints; ++p) {
        const Obj<typename Sifter::traits::coneSequence>&     cone = sifter->cone(points[p]);
        const typename Sifter::traits::coneSequence::iterator cEnd = cone->end();

        for(typename Sifter::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cEnd; ++c_iter) {
          localSifter->addArrow(*c_iter, renumbering[points[p]]);
        }
      }
    };
    template<typename Sieve, typename Section, typename Renumbering>
    static void createLocalSieve(const Obj<Sieve>& sieve, const Obj<Section>& partition, Renumbering& renumbering, const Obj<Sieve>& localSieve, const int height = 0) {
      const typename Section::value_type *points    = partition->restrictPoint(sieve->commRank());
      const int                           numPoints = partition->getFiberDimension(sieve->commRank());

      for(int p = 0; p < numPoints; ++p) {
        Obj<typename Sieve::coneSet> current = new typename Sieve::coneSet();
        Obj<typename Sieve::coneSet> next    = new typename Sieve::coneSet();
        Obj<typename Sieve::coneSet> tmp;

        current->insert(points[p]);
        while(current->size()) {
          for(typename Sieve::coneSet::const_iterator p_iter = current->begin(); p_iter != current->end(); ++p_iter) {
            const Obj<typename Sieve::traits::coneSequence>& cone = sieve->cone(*p_iter);
            
            for(typename Sieve::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
              localSieve->addArrow(renumbering[*c_iter], renumbering[*p_iter], c_iter.color());
              next->insert(*c_iter);
            }
          }
          tmp = current; current = next; next = tmp;
          next->clear();
        }
        if (height) {
          current->insert(points[p]);
          while(current->size()) {
            for(typename Sieve::coneSet::const_iterator p_iter = current->begin(); p_iter != current->end(); ++p_iter) {
              const Obj<typename Sieve::traits::supportSequence>& support = sieve->support(*p_iter);
            
              for(typename Sieve::traits::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter) {
                localSieve->addArrow(renumbering[*p_iter], renumbering[*s_iter], s_iter.color());
                next->insert(*s_iter);
              }
            }
            tmp = current; current = next; next = tmp;
            next->clear();
          }
        }
      }
    };
    template<typename Mesh, typename Section, typename Renumbering>
    static void createLocalMesh(const Obj<Mesh>& mesh, const Obj<Section>& partition, Renumbering& renumbering, const Obj<Mesh>& localMesh, const int height = 0) {
      const Obj<typename Mesh::sieve_type>& sieve      = mesh->getSieve();
      const Obj<typename Mesh::sieve_type>& localSieve = localMesh->getSieve();

      createLocalSieve(sieve, partition, renumbering, localSieve, height);
    };
    template<typename Sieve, typename Section, typename Renumbering>
    static void sizeLocalSieveV(const Obj<Sieve>& sieve, const Obj<Section>& partition, Renumbering& renumbering, const Obj<Sieve>& localSieve, const int height = 0) {
      typedef std::set<typename Sieve::point_type> pointSet;
      const typename Section::value_type *points    = partition->restrictPoint(sieve->commRank());
      const int                           numPoints = partition->getFiberDimension(sieve->commRank());
      int                                 maxSize   = std::max(0, std::max(sieve->getMaxConeSize(), sieve->getMaxSupportSize()));
      const pointSet                      pSet(points, &points[numPoints]);
      ISieveVisitor::FilteredPointRetriever<Sieve,pointSet,Renumbering> fV(pSet, renumbering, maxSize);

      for(int p = 0; p < numPoints; ++p) {
        sieve->cone(points[p], fV);
        localSieve->setConeSize(renumbering[points[p]], fV.getSize());
        fV.clear();
        sieve->support(points[p], fV);
        localSieve->setSupportSize(renumbering[points[p]], fV.getSize());
        fV.clear();
      }
    };
    template<typename Mesh, typename Section, typename Renumbering>
    static void sizeLocalMeshV(const Obj<Mesh>& mesh, const Obj<Section>& partition, Renumbering& renumbering, const Obj<Mesh>& localMesh, const int height = 0) {
      const Obj<typename Mesh::sieve_type>& sieve      = mesh->getSieve();
      const Obj<typename Mesh::sieve_type>& localSieve = localMesh->getSieve();

      sizeLocalSieveV(sieve, partition, renumbering, localSieve, height);
    };
    template<typename Sieve, typename Section, typename Renumbering>
    static void createLocalLabelV(const Obj<Sieve>& sieve, const Obj<Section>& partition, Renumbering& renumbering, const Obj<Sieve>& localSieve, const int height = 0) {
      typedef std::set<typename Sieve::point_type> pointSet;
      typedef ISieveVisitor::FilteredPointRetriever<Sieve,pointSet,Renumbering> visitor_type;
      const typename Section::value_type *points    = partition->restrictPoint(sieve->commRank());
      const int                           numPoints = partition->getFiberDimension(sieve->commRank());
      int                                 maxSize   = std::max(0, std::max(sieve->getMaxConeSize(), sieve->getMaxSupportSize()));
      typename Sieve::point_type         *oPoints   = new typename Sieve::point_type[std::max(1, sieve->getMaxConeSize())];
      int                                *oOrients  = new int[std::max(1, sieve->getMaxConeSize())];
      const pointSet                      pSet(points, &points[numPoints]);
      visitor_type fV(pSet, renumbering, maxSize);

      for(int p = 0; p < numPoints; ++p) {
        fV.useRenumbering(false);
        sieve->orientedCone(points[p], fV);
        const typename visitor_type::oriented_point_type *q = fV.getOrientedPoints();
        const int                                         n = fV.getOrientedSize();
        for(int i = 0; i < n; ++i) {
          oPoints[i]  = q[i].first;
          oOrients[i] = q[i].second;
        }
        localSieve->setCone(oPoints, renumbering[points[p]]);
        localSieve->setConeOrientation(oOrients, renumbering[points[p]]);
        fV.clear();
        fV.useRenumbering(true);
        sieve->support(points[p], fV);
        if (fV.getSize()) {localSieve->setSupport(points[p], fV.getPoints());}
        fV.clear();
      }
      delete [] oPoints;
      delete [] oOrients;
    };
    template<typename Sieve, typename Section, typename Renumbering>
    static void createLocalSieveV(const Obj<Sieve>& sieve, const Obj<Section>& partition, Renumbering& renumbering, const Obj<Sieve>& localSieve, const int height = 0) {
      typedef std::set<typename Sieve::point_type> pointSet;
      typedef ISieveVisitor::FilteredPointRetriever<Sieve,pointSet,Renumbering> visitor_type;
      const typename Section::value_type *points    = partition->restrictPoint(sieve->commRank());
      const int                           numPoints = partition->getFiberDimension(sieve->commRank());
      int                                 maxSize   = std::max(0, std::max(sieve->getMaxConeSize(), sieve->getMaxSupportSize()));
      typename Sieve::point_type         *oPoints   = new typename Sieve::point_type[std::max(1, sieve->getMaxConeSize())];
      int                                *oOrients  = new int[std::max(1, sieve->getMaxConeSize())];
      const pointSet                      pSet(points, &points[numPoints]);
      visitor_type fV(pSet, renumbering, maxSize);

      for(int p = 0; p < numPoints; ++p) {
        ///sieve->cone(points[p], fV);
        ///localSiaeve->setCone(fV.getPoints(), renumbering[points[p]]);
        sieve->orientedCone(points[p], fV);
        const typename visitor_type::oriented_point_type *q = fV.getOrientedPoints();
        const int                                         n = fV.getOrientedSize();
        for(int i = 0; i < n; ++i) {
          oPoints[i]  = q[i].first;
          oOrients[i] = q[i].second;
        }
        localSieve->setCone(oPoints, renumbering[points[p]]);
        localSieve->setConeOrientation(oOrients, renumbering[points[p]]);
        fV.clear();
        sieve->support(points[p], fV);
        localSieve->setSupport(renumbering[points[p]], fV.getPoints());
        fV.clear();
      }
      delete [] oPoints;
      delete [] oOrients;
    };
    template<typename Mesh, typename Section, typename Renumbering>
    static void createLocalMeshV(const Obj<Mesh>& mesh, const Obj<Section>& partition, Renumbering& renumbering, const Obj<Mesh>& localMesh, const int height = 0) {
      const Obj<typename Mesh::sieve_type>& sieve      = mesh->getSieve();
      const Obj<typename Mesh::sieve_type>& localSieve = localMesh->getSieve();

      createLocalSieveV(sieve, partition, renumbering, localSieve, height);
    };
  public: // Partitioning
    //   partition:    Should be properly allocated on input
    //   height:       Height of the point set to uniquely partition
    // TODO: Could rebind assignment section to the type of the output
    template<typename Mesh, typename Section>
    static void createPartition(const Obj<Mesh>& mesh, const Obj<Section>& partition, const int height = 0) {
      MeshManager<Mesh> manager(mesh, height);
      int              *start     = NULL;
      int              *adjacency = NULL;

      if (height == 0) {
        int numVertices;

        buildDualCSR(mesh, &numVertices, &start, &adjacency, GraphPartitioner::zeroBase());
        GraphPartitioner().partition(numVertices, start, adjacency, partition, manager);
        destroyCSR(numVertices, start, adjacency);
      } else if (height == 1) {
        int numEdges;

        buildFaceDualCSR(mesh, mesh->getFactory()->getNumbering(mesh, mesh->depth()-1), &numEdges, &start, &adjacency, GraphPartitioner::zeroBase());
        GraphPartitioner().partition(numEdges, start, adjacency, partition, manager);
        destroyCSR(numEdges, start, adjacency);
      } else {
        throw ALE::Exception("Invalid partition height");
      }
    };
    template<typename Mesh, typename Section>
    static void createPartitionV(const Obj<Mesh>& mesh, const Obj<Section>& partition, const int height = 0) {
      MeshManager<Mesh> manager(mesh, height);
      int              *start     = NULL;
      int              *adjacency = NULL;

      PETSc::Log::Event("PartitionCreate").begin();
      if (height == 0) {
        int numVertices;

        buildDualCSRV(mesh, &numVertices, &start, &adjacency, GraphPartitioner::zeroBase());
        GraphPartitioner().partition(numVertices, start, adjacency, partition, manager);
        destroyCSR(numVertices, start, adjacency);
      } else if (height == 1) {
        int numEdges;

        throw ALE::Exception("Not yet implemented");
#if 0
        buildFaceDualCSRV(mesh, mesh->getFactory()->getNumbering(mesh, mesh->depth()-1), &numEdges, &start, &adjacency, GraphPartitioner::zeroBase());
#endif
        GraphPartitioner().partition(numEdges, start, adjacency, partition, manager);
        destroyCSR(numEdges, start, adjacency);
      } else {
        throw ALE::Exception("Invalid partition height");
      }
      PETSc::Log::Event("PartitionCreate").end();
    };
    // Add in the points in the closure (and star) of the partitioned points
    template<typename Mesh, typename Section>
    static void createPartitionClosure(const Obj<Mesh>& mesh, const Obj<Section>& pointPartition, const Obj<Section>& partition, const int height = 0) {
      const Obj<typename Mesh::sieve_type>& sieve = mesh->getSieve();
      const typename Section::chart_type&   chart = pointPartition->getChart();
      size_t                                size  = 0;

      for(typename Section::chart_type::const_iterator r_iter = chart.begin(); r_iter != chart.end(); ++r_iter) {
        const typename Section::value_type    *points    = pointPartition->restrictPoint(*r_iter);
        const int                              numPoints = pointPartition->getFiberDimension(*r_iter);
        std::set<typename Section::value_type> closure;

        // TODO: Use Quiver's closure() here instead
        for(int p = 0; p < numPoints; ++p) {
          Obj<typename Mesh::sieve_type::coneSet> current = new typename Mesh::sieve_type::coneSet();
          Obj<typename Mesh::sieve_type::coneSet> next    = new typename Mesh::sieve_type::coneSet();
          Obj<typename Mesh::sieve_type::coneSet> tmp;

          current->insert(points[p]);
          closure.insert(points[p]);
          while(current->size()) {
            for(typename Mesh::sieve_type::coneSet::const_iterator p_iter = current->begin(); p_iter != current->end(); ++p_iter) {
              const Obj<typename Mesh::sieve_type::traits::coneSequence>& cone = sieve->cone(*p_iter);
            
              for(typename Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
                closure.insert(*c_iter);
                next->insert(*c_iter);
              }
            }
            tmp = current; current = next; next = tmp;
            next->clear();
          }
          if (height) {
            current->insert(points[p]);
            while(current->size()) {
              for(typename Mesh::sieve_type::coneSet::const_iterator p_iter = current->begin(); p_iter != current->end(); ++p_iter) {
                const Obj<typename Mesh::sieve_type::traits::supportSequence>& support = sieve->support(*p_iter);
            
                for(typename Mesh::sieve_type::traits::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter) {
                  closure.insert(*s_iter);
                  next->insert(*s_iter);
                }
              }
              tmp = current; current = next; next = tmp;
              next->clear();
            }
          }
        }
        partition->setFiberDimension(*r_iter, closure.size());
        size = std::max(size, closure.size());
      }
      partition->allocatePoint();
      typename Section::value_type *values = new typename Section::value_type[size];

      for(typename Section::chart_type::const_iterator r_iter = chart.begin(); r_iter != chart.end(); ++r_iter) {
        const typename Section::value_type    *points    = pointPartition->restrictPoint(*r_iter);
        const int                              numPoints = pointPartition->getFiberDimension(*r_iter);
        std::set<typename Section::value_type> closure;

        // TODO: Use Quiver's closure() here instead
        for(int p = 0; p < numPoints; ++p) {
          Obj<typename Mesh::sieve_type::coneSet> current = new typename Mesh::sieve_type::coneSet();
          Obj<typename Mesh::sieve_type::coneSet> next    = new typename Mesh::sieve_type::coneSet();
          Obj<typename Mesh::sieve_type::coneSet> tmp;

          current->insert(points[p]);
          closure.insert(points[p]);
          while(current->size()) {
            for(typename Mesh::sieve_type::coneSet::const_iterator p_iter = current->begin(); p_iter != current->end(); ++p_iter) {
              const Obj<typename Mesh::sieve_type::traits::coneSequence>& cone = sieve->cone(*p_iter);
            
              for(typename Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
                closure.insert(*c_iter);
                next->insert(*c_iter);
              }
            }
            tmp = current; current = next; next = tmp;
            next->clear();
          }
          if (height) {
            current->insert(points[p]);
            while(current->size()) {
              for(typename Mesh::sieve_type::coneSet::const_iterator p_iter = current->begin(); p_iter != current->end(); ++p_iter) {
                const Obj<typename Mesh::sieve_type::traits::supportSequence>& support = sieve->support(*p_iter);
            
                for(typename Mesh::sieve_type::traits::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter) {
                  closure.insert(*s_iter);
                  next->insert(*s_iter);
                }
              }
              tmp = current; current = next; next = tmp;
              next->clear();
            }
          }
        }
        int i = 0;

        for(typename std::set<typename Section::value_type>::const_iterator p_iter = closure.begin(); p_iter != closure.end(); ++p_iter, ++i) {
          values[i] = *p_iter;
        }
        partition->updatePoint(*r_iter, values);
      }
      delete [] values;
    };
    template<typename Mesh, typename Section>
    static void createPartitionClosureV(const Obj<Mesh>& mesh, const Obj<Section>& pointPartition, const Obj<Section>& partition, const int height = 0) {
      typedef ISieveVisitor::TransitiveClosureVisitor<typename Mesh::sieve_type> visitor_type;
      const Obj<typename Mesh::sieve_type>& sieve = mesh->getSieve();
      const typename Section::chart_type&   chart = pointPartition->getChart();
      size_t                                size  = 0;

      PETSc::Log::Event("PartitionClosure").begin();
      for(typename Section::chart_type::const_iterator r_iter = chart.begin(); r_iter != chart.end(); ++r_iter) {
        const typename Section::value_type *points    = pointPartition->restrictPoint(*r_iter);
        const int                           numPoints = pointPartition->getFiberDimension(*r_iter);
        typename visitor_type::visitor_type nV;
        visitor_type                        cV(*sieve, nV);

        for(int p = 0; p < numPoints; ++p) {
          sieve->cone(points[p], cV);
          if (height) {
            cV.setIsCone(false);
            sieve->support(points[p], cV);
          }
        }
        partition->setFiberDimension(*r_iter, cV.getPoints().size());
        size = std::max(size, cV.getPoints().size());
      }
      partition->allocatePoint();
      typename Section::value_type *values = new typename Section::value_type[size];

      for(typename Section::chart_type::const_iterator r_iter = chart.begin(); r_iter != chart.end(); ++r_iter) {
        const typename Section::value_type *points    = pointPartition->restrictPoint(*r_iter);
        const int                           numPoints = pointPartition->getFiberDimension(*r_iter);
        typename visitor_type::visitor_type nV;
        visitor_type                        cV(*sieve, nV);

        for(int p = 0; p < numPoints; ++p) {
          sieve->cone(points[p], cV);
          if (height) {
            cV.setIsCone(false);
            sieve->support(points[p], cV);
          }
        }
        int i = 0;

        for(typename std::set<typename Mesh::point_type>::const_iterator p_iter = cV.getPoints().begin(); p_iter != cV.getPoints().end(); ++p_iter, ++i) {
          values[i] = *p_iter;
        }
        partition->updatePoint(*r_iter, values);
      }
      delete [] values;
      PETSc::Log::Event("PartitionClosure").end();
    };
    // Create a section mapping points to partitions
    template<typename Section, typename MapSection>
    static void createPartitionMap(const Obj<Section>& partition, const Obj<MapSection>& partitionMap) {
      const typename Section::chart_type& chart = partition->getChart();

      for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        partitionMap->setFiberDimension(*p_iter, 1);
      }
      partitionMap->allocatePoint();
      for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const typename Section::value_type *points = partition->restrictPoint(*p_iter);
        const int                           size   = partition->getFiberDimension(*p_iter);
        const typename Section::point_type  part   = *p_iter;

        for(int i = 0; i < size; ++i) {
          partitionMap->updatePoint(points[i], &part);
        }
      }
    };
  };
#endif

  namespace New {
    template<typename Bundle_, typename Alloc_ = typename Bundle_::alloc_type>
    class Partitioner {
    public:
      typedef Bundle_                          bundle_type;
      typedef Alloc_                           alloc_type;
      typedef typename bundle_type::sieve_type sieve_type;
      typedef typename bundle_type::point_type point_type;
    public:
      #undef __FUNCT__
      #define __FUNCT__ "buildDualCSR"
      // This creates a CSR representation of the adjacency matrix for cells
      // - We allow an exception to contiguous numbering.
      //   If the cell id > numElements, we assign a new number starting at
      //     the top and going downward. I know these might not match up with
      //     the iterator order, but we can fix it later.
      static void buildDualCSR(const Obj<bundle_type>& bundle, const int dim, int **offsets, int **adjacency) {
        ALE_LOG_EVENT_BEGIN;
        typedef typename ALE::New::Completion<bundle_type, point_type, alloc_type> completion;
        const Obj<sieve_type>&                           sieve        = bundle->getSieve();
        const Obj<typename bundle_type::label_sequence>& elements     = bundle->heightStratum(0);
        Obj<sieve_type>                                  overlapSieve = new sieve_type(bundle->comm(), bundle->debug());
        std::map<point_type, point_type>                 newCells;
        int  numElements = elements->size();
        int  newCell     = numElements;
        int *off         = new int[numElements+1];
        int  offset      = 0;
        int *adj;

        completion::scatterSupports(sieve, overlapSieve, bundle->getSendOverlap(), bundle->getRecvOverlap(), bundle);
        if (numElements == 0) {
          *offsets   = NULL;
          *adjacency = NULL;
          ALE_LOG_EVENT_END;
          return;
        }
        if (bundle->depth() == dim) {
          int e = 1;

          off[0] = 0;
          for(typename bundle_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
            const Obj<typename sieve_type::traits::coneSequence>& faces  = sieve->cone(*e_iter);
            typename sieve_type::traits::coneSequence::iterator   fBegin = faces->begin();
            typename sieve_type::traits::coneSequence::iterator   fEnd   = faces->end();

            off[e] = off[e-1];
            for(typename sieve_type::traits::coneSequence::iterator f_iter = fBegin; f_iter != fEnd; ++f_iter) {
              if (sieve->support(*f_iter)->size() == 2) {
                off[e]++;
              } else if ((sieve->support(*f_iter)->size() == 1) && (overlapSieve->support(*f_iter)->size() == 1)) {
                off[e]++;
              }
            }
            e++;
          }
          adj = new int[off[numElements]];
          for(typename bundle_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
            const Obj<typename sieve_type::traits::coneSequence>& faces  = sieve->cone(*e_iter);
            typename sieve_type::traits::coneSequence::iterator   fBegin = faces->begin();
            typename sieve_type::traits::coneSequence::iterator   fEnd   = faces->end();

            for(typename sieve_type::traits::coneSequence::iterator f_iter = fBegin; f_iter != fEnd; ++f_iter) {
              const Obj<typename sieve_type::traits::supportSequence>& neighbors = sieve->support(*f_iter);
              typename sieve_type::traits::supportSequence::iterator   nBegin    = neighbors->begin();
              typename sieve_type::traits::supportSequence::iterator   nEnd      = neighbors->end();

              for(typename sieve_type::traits::supportSequence::iterator n_iter = nBegin; n_iter != nEnd; ++n_iter) {
                if (*n_iter != *e_iter) adj[offset++] = *n_iter;
              }
              const Obj<typename sieve_type::traits::supportSequence>& oNeighbors = overlapSieve->support(*f_iter);
              typename sieve_type::traits::supportSequence::iterator   onBegin    = oNeighbors->begin();
              typename sieve_type::traits::supportSequence::iterator   onEnd      = oNeighbors->end();

              for(typename sieve_type::traits::supportSequence::iterator n_iter = onBegin; n_iter != onEnd; ++n_iter) {
                adj[offset++] = *n_iter;
              }
            }
          }
        } else if (bundle->depth() == 1) {
          std::set<point_type> *neighborCells = new std::set<point_type>[numElements];
          int corners      = sieve->cone(*elements->begin())->size();
          int faceVertices = -1;

          if (corners == dim+1) {
            faceVertices = dim;
          } else if ((dim == 2) && (corners == 4)) {
            faceVertices = 2;
          } else if ((dim == 3) && (corners == 8)) {
            faceVertices = 4;
          } else {
            throw ALE::Exception("Could not determine number of face vertices");
          }
          for(typename bundle_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
            const Obj<typename sieve_type::traits::coneSequence>& vertices  = sieve->cone(*e_iter);
            typename sieve_type::traits::coneSequence::iterator vEnd = vertices->end();

            for(typename sieve_type::traits::coneSequence::iterator v_iter = vertices->begin(); v_iter != vEnd; ++v_iter) {
              const Obj<typename sieve_type::traits::supportSequence>& neighbors = sieve->support(*v_iter);
              typename sieve_type::traits::supportSequence::iterator nEnd = neighbors->end();

              for(typename sieve_type::traits::supportSequence::iterator n_iter = neighbors->begin(); n_iter != nEnd; ++n_iter) {
                if (*e_iter == *n_iter) continue;
                if ((int) sieve->nMeet(*e_iter, *n_iter, 1)->size() == faceVertices) {
                  if ((*e_iter < numElements) && (*n_iter < numElements)) {
                    neighborCells[*e_iter].insert(*n_iter);
                  } else {
                    point_type e = *e_iter, n = *n_iter;

                    if (*e_iter >= numElements) {
                      if (newCells.find(*e_iter) == newCells.end()) newCells[*e_iter] = --newCell;
                      e = newCells[*e_iter];
                    }
                    if (*n_iter >= numElements) {
                      if (newCells.find(*n_iter) == newCells.end()) newCells[*n_iter] = --newCell;
                      n = newCells[*n_iter];
                    }
                    neighborCells[e].insert(n);
                  }
                }
              }
            }
          }
          off[0] = 0;
          for(int e = 1; e <= numElements; e++) {
            off[e] = neighborCells[e-1].size() + off[e-1];
          }
          adj = new int[off[numElements]];
          for(int e = 0; e < numElements; e++) {
            for(typename std::set<point_type>::iterator n_iter = neighborCells[e].begin(); n_iter != neighborCells[e].end(); ++n_iter) {
              adj[offset++] = *n_iter;
            }
          }
          delete [] neighborCells;
        } else {
          throw ALE::Exception("Dual creation not defined for partially interpolated meshes");
        }
        if (offset != off[numElements]) {
          ostringstream msg;
          msg << "ERROR: Total number of neighbors " << offset << " does not match the offset array " << off[numElements];
          throw ALE::Exception(msg.str().c_str());
        }
        //std::cout << "numElements: " << numElements << " newCell: " << newCell << std::endl;
        *offsets   = off;
        *adjacency = adj;
        ALE_LOG_EVENT_END;
      };
      #undef __FUNCT__
      #define __FUNCT__ "buildFaceCSR"
      // This creates a CSR representation of the adjacency hypergraph for faces
      static void buildFaceCSR(const Obj<bundle_type>& bundle, const int dim, const Obj<typename bundle_type::numbering_type>& fNumbering, int *numEdges, int **offsets, int **adjacency) {
        ALE_LOG_EVENT_BEGIN;
        const Obj<sieve_type>&                           sieve    = bundle->getSieve();
        const Obj<typename bundle_type::label_sequence>& elements = bundle->heightStratum(0);
        int  numElements = elements->size();
        int *off         = new int[numElements+1];
        int  e;

        if (bundle->depth() != dim) {
          throw ALE::Exception("Not yet implemented for non-interpolated meshes");
        }
        off[0] = 0;
        e      = 1;
        for(typename bundle_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
          off[e] = sieve->cone(*e_iter)->size() + off[e-1];
          e++;
        }
        int *adj    = new int[off[numElements]];
        int  offset = 0;
        for(typename bundle_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
          const Obj<typename sieve_type::traits::coneSequence>& faces = sieve->cone(*e_iter);
          typename sieve_type::traits::coneSequence::iterator   fEnd  = faces->end();

          for(typename sieve_type::traits::coneSequence::iterator f_iter = faces->begin(); f_iter != fEnd; ++f_iter) {
            adj[offset++] = fNumbering->getIndex(*f_iter);
          }
        }
        if (offset != off[numElements]) {
          ostringstream msg;
          msg << "ERROR: Total number of neighbors " << offset << " does not match the offset array " << off[numElements];
          throw ALE::Exception(msg.str().c_str());
        }
        *numEdges  = numElements;
        *offsets   = off;
        *adjacency = adj;
        ALE_LOG_EVENT_END;
      };
      template<typename PartitionType>
      static PartitionType *subordinatePartition(const Obj<bundle_type>& bundle, int levels, const Obj<bundle_type>& subBundle, const PartitionType assignment[]) {
        const Obj<typename bundle_type::numbering_type>& cNumbering = bundle->getFactory()->getLocalNumbering(bundle, bundle->depth());
        const Obj<typename bundle_type::label_sequence>& cells      = subBundle->heightStratum(0);
        const Obj<typename bundle_type::numbering_type>& sNumbering = bundle->getFactory()->getLocalNumbering(subBundle, subBundle->depth());
        const int        numCells      = cells->size();
        PartitionType   *subAssignment = new PartitionType[numCells];

        if (levels != 1) {
          throw ALE::Exception("Cannot calculate subordinate partition for any level separation other than 1");
        } else {
          const Obj<typename bundle_type::sieve_type>&   sieve    = bundle->getSieve();
          const Obj<typename bundle_type::sieve_type>&   subSieve = subBundle->getSieve();
          Obj<typename bundle_type::sieve_type::coneSet> tmpSet   = new typename bundle_type::sieve_type::coneSet();
          Obj<typename bundle_type::sieve_type::coneSet> tmpSet2  = new typename bundle_type::sieve_type::coneSet();

          for(typename bundle_type::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
            const Obj<typename bundle_type::sieve_type::coneSequence>& cone = subSieve->cone(*c_iter);

            Obj<typename bundle_type::sieve_type::supportSet> cell = sieve->nJoin1(cone);
            if (cell->size() != 1) {
              std::cout << "Indeterminate subordinate partition for face " << *c_iter << std::endl;
              for(typename bundle_type::sieve_type::supportSet::iterator s_iter = cell->begin(); s_iter != cell->end(); ++s_iter) {
                std::cout << "  cell " << *s_iter << std::endl;
              }
              // Could relax this to choosing the first one
              throw ALE::Exception("Indeterminate subordinate partition");
            }
            subAssignment[sNumbering->getIndex(*c_iter)] = assignment[cNumbering->getIndex(*cell->begin())];
            tmpSet->clear();
            tmpSet2->clear();
          }
        }
        return subAssignment;
      };
    };
#ifdef PETSC_HAVE_CHACO
    namespace Chaco {
      template<typename Bundle_>
      class Partitioner {
      public:
        typedef Bundle_                          bundle_type;
        typedef typename bundle_type::sieve_type sieve_type;
        typedef typename bundle_type::point_type point_type;
        typedef short int                        part_type;
      public:
        #undef __FUNCT__
        #define __FUNCT__ "ChacoPartitionSieve"
        static part_type *partitionSieve(const Obj<bundle_type>& bundle, const int dim) {
          part_type *assignment = NULL; /* set number of each vtx (length n) */
          int       *start;             /* start of edge list for each vertex */
          int       *adjacency;         /* = adj -> j; edge list data  */

          ALE_LOG_EVENT_BEGIN;
          ALE::New::Partitioner<bundle_type>::buildDualCSR(bundle, dim, &start, &adjacency);
          if (bundle->commRank() == 0) {
            /* arguments for Chaco library */
            FREE_GRAPH = 0;                         /* Do not let Chaco free my memory */
            int nvtxs;                              /* number of vertices in full graph */
            int *vwgts = NULL;                      /* weights for all vertices */
            float *ewgts = NULL;                    /* weights for all edges */
            float *x = NULL, *y = NULL, *z = NULL;  /* coordinates for inertial method */
            char *outassignname = NULL;             /*  name of assignment output file */
            char *outfilename = NULL;               /* output file name */
            int architecture = dim;                 /* 0 => hypercube, d => d-dimensional mesh */
            int ndims_tot = 0;                      /* total number of cube dimensions to divide */
            int mesh_dims[3];                       /* dimensions of mesh of processors */
            double *goal = NULL;                    /* desired set sizes for each set */
            int global_method = 1;                  /* global partitioning algorithm */
            int local_method = 1;                   /* local partitioning algorithm */
            int rqi_flag = 0;                       /* should I use RQI/Symmlq eigensolver? */
            int vmax = 200;                         /* how many vertices to coarsen down to? */
            int ndims = 1;                          /* number of eigenvectors (2^d sets) */
            double eigtol = 0.001;                  /* tolerance on eigenvectors */
            long seed = 123636512;                  /* for random graph mutations */
	    float *vCoords[3];
            PetscErrorCode ierr;

	    ierr = PetscOptionsGetInt(PETSC_NULL, "-partitioner_chaco_global_method", &global_method, PETSC_NULL);CHKERROR(ierr, "Error in PetscOptionsGetInt");
	    ierr = PetscOptionsGetInt(PETSC_NULL, "-partitioner_chaco_local_method",  &local_method,  PETSC_NULL);CHKERROR(ierr, "Error in PetscOptionsGetInt");
	    if (global_method == 3) {
	      // Inertial Partitioning
	      ierr = PetscMalloc3(nvtxs,float,&x,nvtxs,float,&y,nvtxs,float,&z);CHKERROR(ierr, "Error in PetscMalloc");
	      vCoords[0] = x; vCoords[1] = y; vCoords[2] = z;
	      const Obj<typename bundle_type::label_sequence>&    cells       = bundle->heightStratum(0);
	      const Obj<typename bundle_type::real_section_type>& coordinates = bundle->getRealSection("coordinates");
	      const int corners = bundle->size(coordinates, *(cells->begin()))/dim;
	      int       c       = 0;

	      for(typename bundle_type::label_sequence::iterator c_iter = cells->begin(); c_iter !=cells->end(); ++c_iter, ++c) {
		const double *coords = bundle->restrictClosure(coordinates, *c_iter);

		for(int d = 0; d < dim; ++d) {
		  vCoords[d][c] = 0.0;
		}
		for(int v = 0; v < corners; ++v) {
		  for(int d = 0; d < dim; ++d) {
		    vCoords[d][c] += coords[v*dim+d];
		  }
		}
		for(int d = 0; d < dim; ++d) {
		  vCoords[d][c] /= corners;
		}
	      }
	    }

            nvtxs = bundle->heightStratum(0)->size();
            mesh_dims[0] = bundle->commSize(); mesh_dims[1] = 1; mesh_dims[2] = 1;
            for(int e = 0; e < start[nvtxs]; e++) {
              adjacency[e]++;
            }
            assignment = new part_type[nvtxs];
            ierr = PetscMemzero(assignment, nvtxs * sizeof(part_type));CHKERROR(ierr, "Error in PetscMemzero");

            /* redirect output to buffer: chaco -> msgLog */
#ifdef PETSC_HAVE_UNISTD_H
            char *msgLog;
            int fd_stdout, fd_pipe[2], count;

            fd_stdout = dup(1);
            pipe(fd_pipe);
            close(1);
            dup2(fd_pipe[1], 1);
            msgLog = new char[16284];
#endif

            ierr = interface(nvtxs, start, adjacency, vwgts, ewgts, x, y, z,
                             outassignname, outfilename, assignment, architecture, ndims_tot,
                             mesh_dims, goal, global_method, local_method, rqi_flag, vmax, ndims,
                             eigtol, seed);

#ifdef PETSC_HAVE_UNISTD_H
            int SIZE_LOG  = 10000;

            fflush(stdout);
            count = read(fd_pipe[0], msgLog, (SIZE_LOG - 1) * sizeof(char));
            if (count < 0) count = 0;
            msgLog[count] = 0;
            close(1);
            dup2(fd_stdout, 1);
            close(fd_stdout);
            close(fd_pipe[0]);
            close(fd_pipe[1]);
            if (bundle->debug()) {
              std::cout << msgLog << std::endl;
            }
            delete [] msgLog;
#endif
	    if (global_method == 3) {
	      // Inertial Partitioning
	      ierr = PetscFree3(x, y, z);CHKERROR(ierr, "Error in PetscFree");
	    }
          }
          if (adjacency) delete [] adjacency;
          if (start)     delete [] start;
          ALE_LOG_EVENT_END;
          return assignment;
        };
        static part_type *partitionSieveByFace(const Obj<bundle_type>& bundle, const int dim) {
          throw ALE::Exception("Chaco cannot partition a mesh by faces");
        };
      };
    };
#endif
#ifdef PETSC_HAVE_PARMETIS
    namespace ParMetis {
      template<typename Bundle_>
      class Partitioner {
      public:
        typedef Bundle_                          bundle_type;
        typedef typename bundle_type::sieve_type sieve_type;
        typedef typename bundle_type::point_type point_type;
        typedef int                              part_type;
      public:
        #undef __FUNCT__
        #define __FUNCT__ "ParMetisPartitionSieve"
        static part_type *partitionSieve(const Obj<bundle_type>& bundle, const int dim) {
          int    nvtxs      = 0;    // The number of vertices in full graph
          int   *vtxdist;           // Distribution of vertices across processes
          int   *xadj;              // Start of edge list for each vertex
          int   *adjncy;            // Edge lists for all vertices
          int   *vwgt       = NULL; // Vertex weights
          int   *adjwgt     = NULL; // Edge weights
          int    wgtflag    = 0;    // Indicates which weights are present
          int    numflag    = 0;    // Indicates initial offset (0 or 1)
          int    ncon       = 1;    // The number of weights per vertex
          int    nparts     = bundle->commSize(); // The number of partitions
          float *tpwgts;            // The fraction of vertex weights assigned to each partition
          float *ubvec;             // The balance intolerance for vertex weights
          int    options[5];        // Options
          // Outputs
          int    edgeCut;           // The number of edges cut by the partition
          int   *assignment = NULL; // The vertex partition

          options[0] = 0; // Use all defaults
          vtxdist    = new int[nparts+1];
          vtxdist[0] = 0;
          tpwgts     = new float[ncon*nparts];
          for(int p = 0; p < nparts; ++p) {
            tpwgts[p] = 1.0/nparts;
          }
          ubvec      = new float[ncon];
          ubvec[0]   = 1.05;
          nvtxs      = bundle->heightStratum(0)->size();
          assignment = new part_type[nvtxs];
          MPI_Allgather(&nvtxs, 1, MPI_INT, &vtxdist[1], 1, MPI_INT, bundle->comm());
          for(int p = 2; p <= nparts; ++p) {
            vtxdist[p] += vtxdist[p-1];
          }
          if (bundle->commSize() == 1) {
            PetscMemzero(assignment, nvtxs * sizeof(part_type));
          } else {
            ALE::New::Partitioner<bundle_type>::buildDualCSR(bundle, dim, &xadj, &adjncy);

            if (bundle->debug() && nvtxs) {
              for(int p = 0; p <= nvtxs; ++p) {
                std::cout << "["<<bundle->commRank()<<"]xadj["<<p<<"] = " << xadj[p] << std::endl;
              }
              for(int i = 0; i < xadj[nvtxs]; ++i) {
                std::cout << "["<<bundle->commRank()<<"]adjncy["<<i<<"] = " << adjncy[i] << std::endl;
              }
            }
            if (vtxdist[1] == vtxdist[nparts]) {
              if (bundle->commRank() == 0) {
                METIS_PartGraphKway(&nvtxs, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &nparts, options, &edgeCut, assignment);
                if (bundle->debug()) {std::cout << "Metis: edgecut is " << edgeCut << std::endl;}
              }
            } else {
              MPI_Comm comm = bundle->comm();

              ParMETIS_V3_PartKway(vtxdist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgeCut, assignment, &comm);
              if (bundle->debug()) {std::cout << "ParMetis: edgecut is " << edgeCut << std::endl;}
            }
            if (xadj   != NULL) delete [] xadj;
            if (adjncy != NULL) delete [] adjncy;
          }
          delete [] vtxdist;
          delete [] tpwgts;
          delete [] ubvec;
          return assignment;
        };
        #undef __FUNCT__
        #define __FUNCT__ "ParMetisPartitionSieveByFace"
        static part_type *partitionSieveByFace(const Obj<bundle_type>& bundle, const int dim) {
#ifdef PETSC_HAVE_HMETIS
          int   *assignment = NULL; // The vertex partition
          int    nvtxs;      // The number of vertices
          int    nhedges;    // The number of hyperedges
          int   *vwgts;      // The vertex weights
          int   *eptr;       // The offsets of each hyperedge
          int   *eind;       // The vertices in each hyperedge, indexed by eptr
          int   *hewgts;     // The hyperedge weights
          int    nparts;     // The number of partitions
          int    ubfactor;   // The allowed load imbalance (1-50)
          int    options[9]; // Options
          // Outputs
          int    edgeCut;    // The number of edges cut by the partition
          const Obj<ALE::Mesh::numbering_type>& fNumbering = bundle->getFactory()->getNumbering(bundle, bundle->depth()-1);

          if (topology->commRank() == 0) {
            nvtxs      = bundle->heightStratum(1)->size();
            vwgts      = NULL;
            hewgts     = NULL;
            nparts     = bundle->commSize();
            ubfactor   = 5;
            options[0] = 1;  // Use all defaults
            options[1] = 10; // Number of bisections tested
            options[2] = 1;  // Vertex grouping scheme
            options[3] = 1;  // Objective function
            options[4] = 1;  // V-cycle refinement
            options[5] = 0;
            options[6] = 0;
            options[7] = 1; // Random seed
            options[8] = 24; // Debugging level
            assignment = new part_type[nvtxs];

            if (bundle->commSize() == 1) {
              PetscMemzero(assignment, nvtxs * sizeof(part_type));
            } else {
              ALE::New::Partitioner<bundle_type>::buildFaceCSR(bundle, dim, fNumbering, &nhedges, &eptr, &eind);
              HMETIS_PartKway(nvtxs, nhedges, vwgts, eptr, eind, hewgts, nparts, ubfactor, options, assignment, &edgeCut);

              delete [] eptr;
              delete [] eind;
            }
            if (bundle->debug()) {for (int i = 0; i<nvtxs; i++) printf("[%d] %d\n", PetscGlobalRank, assignment[i]);}
          } else {
            assignment = NULL;
          }
          return assignment;
#else
          throw ALE::Exception("hmetis partitioner is not available.");
#endif
        };
      };
    };
#endif
#ifdef PETSC_HAVE_ZOLTAN
    namespace Zoltan {
      template<typename Bundle_>
      class Partitioner {
      public:
        typedef Bundle_                          bundle_type;
        typedef typename bundle_type::sieve_type sieve_type;
        typedef typename bundle_type::point_type point_type;
        typedef int                              part_type;
      public:
        static part_type *partitionSieve(const Obj<bundle_type>& bundle, const int dim) {
          throw ALE::Exception("Zoltan partition by cells not implemented");
        };
        #undef __FUNCT__
        #define __FUNCT__ "ZoltanPartitionSieveByFace"
        static part_type *partitionSieveByFace(const Obj<bundle_type>& bundle, const int dim) {
          // Outputs
          float         version;           // The library version
          int           changed;           // Did the partition change?
          int           numGidEntries;     // Number of array entries for a single global ID (1)
          int           numLidEntries;     // Number of array entries for a single local ID (1)
          int           numImport;         // The number of imported points
          ZOLTAN_ID_PTR import_global_ids; // The imported points
          ZOLTAN_ID_PTR import_local_ids;  // The imported points
          int          *import_procs;      // The proc each point was imported from
          int          *import_to_part;    // The partition of each imported point
          int           numExport;         // The number of exported points
          ZOLTAN_ID_PTR export_global_ids; // The exported points
          ZOLTAN_ID_PTR export_local_ids;  // The exported points
          int          *export_procs;      // The proc each point was exported to
          int          *export_to_part;    // The partition assignment of all local points
          int          *assignment;        // The partition assignment of all local points
          const Obj<typename bundle_type::numbering_type>& fNumbering = bundle->getFactory()->getNumbering(bundle, bundle->depth()-1);

          if (bundle->commSize() == 1) {
            PetscMemzero(assignment, bundle->heightStratum(1)->size() * sizeof(part_type));
          } else {
            if (bundle->commRank() == 0) {
              nvtxs_Zoltan = bundle->heightStratum(1)->size();
              ALE::New::Partitioner<bundle_type>::buildFaceCSR(bundle, dim, fNumbering, &nhedges_Zoltan, &eptr_Zoltan, &eind_Zoltan);
              assignment = new int[nvtxs_Zoltan];
            } else {
              nvtxs_Zoltan   = bundle->heightStratum(1)->size();
              nhedges_Zoltan = 0;
              eptr_Zoltan    = new int[1];
              eind_Zoltan    = new int[1];
              eptr_Zoltan[0] = 0;
              assignment     = NULL;
            }

            int ierr = Zoltan_Initialize(0, NULL, &version);
            struct Zoltan_Struct *zz = Zoltan_Create(bundle->comm());
            // General parameters
            Zoltan_Set_Param(zz, "DEBUG_LEVEL", "2");
            Zoltan_Set_Param(zz, "LB_METHOD", "PHG");
            Zoltan_Set_Param(zz, "RETURN_LISTS", "PARTITION");
            // PHG parameters
            Zoltan_Set_Param(zz, "PHG_OUTPUT_LEVEL", "2");
            Zoltan_Set_Param(zz, "PHG_EDGE_SIZE_THRESHOLD", "1.0"); // Do not throw out dense edges
            // Call backs
            Zoltan_Set_Num_Obj_Fn(zz, getNumVertices_Zoltan, NULL);
            Zoltan_Set_Obj_List_Fn(zz, getLocalElements_Zoltan, NULL);
            Zoltan_Set_HG_Size_CS_Fn(zz, getHgSizes_Zoltan, NULL);
            Zoltan_Set_HG_CS_Fn(zz, getHg_Zoltan, NULL);
            // Debugging
            //Zoltan_Generate_Files(zz, "zoltan.debug", 1, 0, 0, 1); // if using hypergraph callbacks

            ierr = Zoltan_LB_Partition(zz, &changed, &numGidEntries, &numLidEntries,
                                       &numImport, &import_global_ids, &import_local_ids, &import_procs, &import_to_part,
                                       &numExport, &export_global_ids, &export_local_ids, &export_procs, &export_to_part);
            for(int v = 0; v < nvtxs_Zoltan; ++v) {
              assignment[v] = export_to_part[v];
            }
            Zoltan_LB_Free_Part(&import_global_ids, &import_local_ids, &import_procs, &import_to_part);
            Zoltan_LB_Free_Part(&export_global_ids, &export_local_ids, &export_procs, &export_to_part);
            Zoltan_Destroy(&zz);

            delete [] eptr_Zoltan;
            delete [] eind_Zoltan;
          }
          if (assignment) {for (int i=0; i<nvtxs_Zoltan; i++) printf("[%d] %d\n",PetscGlobalRank,assignment[i]);}
          return assignment;
        };
      };
    };
#endif
  }
}
#endif
