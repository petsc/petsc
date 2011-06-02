static char help[] = "Sieve Package Parallel Correctness Tests.\n\n";

#define ALE_HAVE_CXX_ABI
#define ALE_MEM_LOGGING
#include <petscdmmesh.hh>
#include <petscdmmesh_viewers.hh>

#include <IField.hh>
#include <ParallelMapping.hh>

using ALE::Obj;

template<typename Point, typename Rank>
class PetscNewOverlap : public ALE::ParallelObject {
public:
  typedef PetscInt index_type;
  typedef Point    point_type;
  typedef Rank     rank_type;
  class sort_indices {
   private:
    const std::vector<point_type>& values;
   public:
    sort_indices(const std::vector<point_type>& values) : values(values) {};
    bool operator()(index_type i, index_type j) {return values[i] < values[j];};
  };
protected:
  PetscInt    numRanks;     // Number of partner processes
  rank_type  *ranks;        // MPI Rank of each partner process
  index_type *pointsOffset; // Offset into points array for each partner process
  point_type *points;       // Points array for each partner process, in sorted order
  point_type *remotePoints; // Remote points array for each partner process

  std::vector<rank_type> flexRanks;
  std::map<rank_type, std::vector<point_type> > flexPoints;
  std::map<rank_type, std::vector<point_type> > flexRemotePoints;
public:
  PetscNewOverlap(MPI_Comm comm, const int debug = 0) : ALE::ParallelObject(comm, debug) {
    this->numRanks     = 0;
    this->ranks        = PETSC_NULL;
    this->pointsOffset = PETSC_NULL;
    this->points       = PETSC_NULL;
    this->remotePoints = PETSC_NULL;
  };
  ~PetscNewOverlap() {
    PetscErrorCode ierr;
    ierr = PetscFree(this->ranks);CHKERRXX(ierr);
    ierr = PetscFree(this->pointsOffset);CHKERRXX(ierr);
    ierr = PetscFree(this->points);CHKERRXX(ierr);
    ierr = PetscFree(this->remotePoints);CHKERRXX(ierr);
  };
  index_type getRankIndex(index_type rank) {
    for(index_type r = 0; r < numRanks; ++r) {
      if (ranks[r] == rank) {
        return r;
      }
    }
    throw ALE::Exception("Invalid rank was not contained in this overlap");
  };
  // TODO This should also sort
  void assemble() {
    index_type    *indices;
    PetscErrorCode ierr;

    numRanks = flexRanks.size();
    ierr = PetscMalloc(numRanks * sizeof(rank_type), &ranks);CHKERRXX(ierr);
    for(index_type r = 0; r < numRanks; ++r) {
      ranks[r] = flexRanks[r];
    }
    ierr = PetscMalloc((numRanks+1) * sizeof(index_type), &pointsOffset);CHKERRXX(ierr);
    pointsOffset[0] = 0;
    for(index_type r = 1; r <= numRanks; ++r) {
      const rank_type rank = ranks[r-1];

      std::cout << "["<<this->commRank()<<"]: rank " << flexRanks[rank] << std::endl;
      for(typename std::vector<point_type>::iterator p_iter = flexPoints[rank].begin(); p_iter != flexPoints[rank].end(); ++p_iter) {
        std::cout << "["<<this->commRank()<<"]:    " << *p_iter << std::endl;
      }
      pointsOffset[r] = pointsOffset[r-1] + flexPoints[rank].size();
    }
    ierr = PetscMalloc((pointsOffset[numRanks]) * sizeof(point_type), &points);CHKERRXX(ierr);
    ierr = PetscMalloc((pointsOffset[numRanks]) * sizeof(point_type), &remotePoints);CHKERRXX(ierr);
    ierr = PetscMalloc((pointsOffset[numRanks]) * sizeof(index_type), &indices);CHKERRXX(ierr);
    for(index_type r = 0; r < numRanks; ++r) {
      const index_type numPoints = pointsOffset[r+1] - pointsOffset[r];
      const rank_type  rank      = ranks[r];

      for(index_type k = 0; k < numPoints; ++k) {indices[k] = k;}
      std::sort(indices, &indices[numPoints], sort_indices(flexPoints[rank]));
      for(index_type p = pointsOffset[r], k = 0; p < pointsOffset[r+1]; ++p, ++k) {
        points[p]       = flexPoints[rank][indices[k]];
        remotePoints[p] = flexRemotePoints[rank][indices[k]];
      }
    }
    ierr = PetscFree(indices);CHKERRXX(ierr);
    flexRanks.clear();
    flexPoints.clear();
    flexRemotePoints.clear();
  };
  void viewSpecial(const char *msg = PETSC_NULL) {
    PetscErrorCode ierr;

    if (!this->commRank()) {
      ierr = PetscSynchronizedPrintf(this->comm(), "%s: %s\n", this->getName().size() ? this->getName() : "Overlap", msg);
    }
    ierr = PetscSynchronizedPrintf(this->comm(), "%d partners:\n", this->numRanks);
    for(index_type r = 0; r < this->numRanks; ++r) {
      ierr = PetscSynchronizedPrintf(this->comm(), "  %d:", this->ranks[r]);
      for(index_type p = pointsOffset[r]; p < pointsOffset[r+1]; ++p) {
        ierr = PetscSynchronizedPrintf(this->comm(), "  %d (%d)", this->points[p], this->remotePoints[p]);
      }
      ierr = PetscSynchronizedPrintf(this->comm(), "\n");
    }
    ierr = PetscSynchronizedFlush(this->comm());
  };
};

template<typename Value>
class PetscSequence {
public:
  typedef Value *iterator;
};

template<typename Point, typename Rank>
class PetscNewSendOverlap : public PetscNewOverlap<Point,Rank> {
public:
  typedef PetscInt index_type;
  typedef Point    point_type;
  typedef Rank     rank_type;
  // Backwards compatibility
  typedef point_type source_type;
  typedef rank_type  target_type;
  typedef point_type color_type;
  typedef PetscSequence<rank_type>  baseSequence;
  typedef PetscSequence<point_type> coneSequence;
public:
  PetscNewSendOverlap(MPI_Comm comm, const int debug = 0) : PetscNewOverlap<Point,Rank>(comm, debug) {};
  ~PetscNewSendOverlap() {};
  void addArrow(source_type s, target_type t, color_type c) {
    // Add rank
    bool addRank = true;
    for(typename std::vector<rank_type>::const_iterator r_iter = this->flexRanks.begin(); r_iter != this->flexRanks.end(); ++r_iter) {
      if (*r_iter == t) {addRank = false; break;}
    }
    if (addRank) {
      this->flexRanks.push_back(t);
    }
    // Add point
    this->flexPoints[t].push_back(s);
    this->flexRemotePoints[t].push_back(c);
  };
  typename baseSequence::iterator baseBegin() {
    assert(!this->numRanks || this->ranks);
    return this->ranks;
  };
  typename baseSequence::iterator baseEnd() {
    assert(!this->numRanks || this->ranks);
    return &this->ranks[this->numRanks];
  };
  int getConeSize(rank_type rank) {
    const index_type r = this->getRankIndex(rank);
    return this->pointsOffset[r+1] - this->pointsOffset[r];
  };
  typename coneSequence::iterator coneBegin(rank_type rank) {
    const index_type r = this->getRankIndex(rank);
    return &this->points[this->pointsOffset[r]];
  };
  typename coneSequence::iterator coneEnd(rank_type rank) {
    const index_type r = this->getRankIndex(rank);
    return &this->points[this->pointsOffset[r+1]];
  };
};

template<typename Point>
class PetscSupportSequence {
public:
  typedef Point point_type;
  class iterator {
  public:
    typedef point_type  value_type;
    typedef int         difference_type;
    typedef value_type* pointer;
    typedef value_type& reference;
  protected:
    const point_type *points;
    const point_type *remotePoints;
  public:
    iterator(const point_type *points, const point_type *remotePoints) : points(points), remotePoints(remotePoints) {};
    iterator(const iterator& iter) : points(iter.points), remotePoints(iter.remotePoints) {};
    virtual bool             operator==(const iterator& iter) const {return points == iter.points;};
    virtual bool             operator!=(const iterator& iter) const {return points != iter.points;};
    virtual const value_type operator*() const {return *points;};
    virtual const value_type color() const     {return *remotePoints;};
    virtual iterator&        operator++() {++points; ++remotePoints; return *this;};
    virtual iterator         operator++(int) {
      iterator tmp(*this);
      ++points; ++remotePoints;
      return tmp;
    };
    iterator& operator=(const iterator& iter) {
      points       = iter.points;
      remotePoints = iter.remotePoints;
      return *this;
    };
  };
};

template<typename Point, typename Rank>
class PetscNewRecvOverlap : public PetscNewOverlap<Point,Rank> {
public:
  typedef PetscInt index_type;
  typedef Point    point_type;
  typedef Rank     rank_type;
  // Backwards compatibility
  typedef rank_type  source_type;
  typedef point_type target_type;
  typedef point_type color_type;
  typedef PetscSequence<rank_type>         capSequence;
  typedef PetscSupportSequence<point_type> supportSequence;
public:
  PetscNewRecvOverlap(MPI_Comm comm, const int debug = 0) : PetscNewOverlap<Point,Rank>(comm, debug) {};
  ~PetscNewRecvOverlap() {};
  void addArrow(source_type s, target_type t, color_type c) {
    // Add rank
    bool addRank = true;
    for(typename std::vector<rank_type>::const_iterator r_iter = this->flexRanks.begin(); r_iter != this->flexRanks.end(); ++r_iter) {
      if (*r_iter == s) {addRank = false; break;}
    }
    if (addRank) {
      this->flexRanks.push_back(s);
    }
    // Add point
    this->flexPoints[s].push_back(t);
    this->flexRemotePoints[s].push_back(c);
  };
  typename capSequence::iterator capBegin() {
    assert(!this->numRanks || this->ranks);
    return this->ranks;
  };
  typename capSequence::iterator capEnd() {
    assert(!this->numRanks || this->ranks);
    return &this->ranks[this->numRanks];
  };
  int getSupportSize(rank_type rank) {
    const index_type r = this->getRankIndex(rank);
    return this->pointsOffset[r+1] - this->pointsOffset[r];
  };
  typename supportSequence::iterator supportBegin(rank_type rank) {
    const index_type r = this->getRankIndex(rank);
    return typename supportSequence::iterator(&this->points[this->pointsOffset[r]], &this->remotePoints[this->pointsOffset[r]]);
  };
  typename supportSequence::iterator supportEnd(rank_type rank) {
    const index_type r = this->getRankIndex(rank);
    return typename supportSequence::iterator(&this->points[this->pointsOffset[r+1]], &this->remotePoints[this->pointsOffset[r+1]]);
  };
};

typedef struct {
  PetscInt   debug;
  MPI_Comm   comm;
  PetscInt   rank;
  PetscInt   size;
  // Classes
  PetscBool  section;     // Run the Section tests
  PetscBool  isection;    // Run the ISection tests
  PetscBool  partition;   // Run the Partition tests
  // Run flags
  PetscInt   number;      // Number of each class to create
  // Mesh flags
  PetscInt   numCells;    // If possible, set the total number of cells
  PetscBool  interpolate; // Interpolate the mesh
  // Section flags
  PetscInt   components;  // Number of section components
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  //ALE::MemoryLogger& logger = ALE::MemoryLogger::singleton();
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  options->debug       = 0;
  options->section     = PETSC_FALSE;
  options->isection    = PETSC_FALSE;
  options->partition   = PETSC_FALSE;
  options->number      = 0;
  options->numCells    = 8;
  options->interpolate = PETSC_FALSE;
  options->components  = 3;

  ierr = PetscOptionsBegin(comm, "", "Options for the Sieve package tests", "Sieve");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "Debugging flag", "distTests", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-section", "Run Section tests", "distTests", options->section, &options->section, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-isection", "Run ISection tests", "distTests", options->isection, &options->isection, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-partition", "Run Partition tests", "distTests", options->partition, &options->partition, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-num", "Number of each class to create", "distTests", options->number, &options->number, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-numCells", "Number of mesh cells", "distTests", options->numCells, &options->numCells, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-interpolate", "Interpolate the flag", "distTests", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-components", "Number of section components", "distTests", options->components, &options->components, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  options->comm = comm;
  ierr = MPI_Comm_rank(comm, &options->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &options->size);CHKERRQ(ierr);
  //logger.setDebug(options->debug);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateScatterOverlap"
template<typename SendOverlap, typename RecvOverlap>
PetscErrorCode CreateScatterOverlap(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const bool localNumbering, const Options *options)
{
  const PetscInt rank     = options->rank;
  const PetscInt size     = options->size;
  const PetscInt numCells = options->numCells;
  const PetscInt block    = numCells/size;

  PetscFunctionBegin;
  if (!rank) {
    for(PetscInt r = 1; r < size; ++r) {
      const PetscInt rStart = r*block     + PetscMin(r, numCells%size);
      const PetscInt rEnd   = (r+1)*block + PetscMin(r+1, numCells%size);

      for(PetscInt c = rStart; c < rEnd; ++c) {
        if (localNumbering) {
          sendOverlap->addArrow(c, r, c - rStart);
        } else {
          sendOverlap->addArrow(c, r, c);
        }
      }
    }
  } else {
    const PetscInt start = rank*block     + PetscMin(rank, numCells%size);
    const PetscInt end   = (rank+1)*block + PetscMin(rank+1, numCells%size);

    for(PetscInt c = start; c < end; ++c) {
      if (localNumbering) {
        recvOverlap->addArrow(0, c - start, c);
      } else {
        recvOverlap->addArrow(0, c, c);
      }
    }
  }
  PetscFunctionReturn(0);
}

#if 0
#undef __FUNCT__
#define __FUNCT__ "ConstantSectionTest"
PetscErrorCode ConstantSectionTest(const Options *options)
{
  typedef int point_type;
  typedef ALE::Sifter<point_type,int,point_type> send_overlap_type;
  typedef ALE::Sifter<int,point_type,point_type> recv_overlap_type;
  typedef ALE::ConstantSection<point_type, double> section;
  Obj<send_overlap_type> sendOverlap     = new send_overlap_type(options->comm);
  Obj<recv_overlap_type> recvOverlap     = new recv_overlap_type(options->comm);
  Obj<section>           serialSection   = new section(options->comm, options->debug);
  Obj<section>           parallelSection = new section(options->comm, options->debug);
  section::value_type    value           = 7.0;

  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = CreateScatterOverlap(sendOverlap, recvOverlap, true, options);CHKERRQ(ierr);
  serialSection->addPoint(sendOverlap->cap());
  if (!options->rank) {
    serialSection->updatePoint(0, &value);
  }
  ALE::Completion::completeSection(sendOverlap, recvOverlap, serialSection, parallelSection);
  if (options->debug) {parallelSection->view("Parallel ConstantSection");}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "UniformSectionTest"
PetscErrorCode UniformSectionTest(const Options *options)
{
  typedef int point_type;
  typedef ALE::Sifter<int,point_type,point_type> send_overlap_type;
  typedef ALE::Sifter<point_type,int,point_type> recv_overlap_type;
  typedef ALE::UniformSection<point_type, double, 4> section;
  Obj<send_overlap_type> sendOverlap     = new send_overlap_type(options->comm);
  Obj<recv_overlap_type> recvOverlap     = new recv_overlap_type(options->comm);
  Obj<section>           serialSection   = new section(options->comm, options->debug);
  Obj<section>           parallelSection = new section(options->comm, options->debug);
  section::value_type    value[4];

  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = CreateScatterOverlap(sendOverlap, recvOverlap, true, options);CHKERRQ(ierr);
  if (!options->rank) {
    for(int c = 0; c < options->numCells; ++c) {
      for(PetscInt comp = 0; comp < 4; ++comp) {value[comp] = (c+1)*(comp+1);}
      serialSection->setFiberDimension(c, 4);
      serialSection->updatePoint(c, value);
    }
    const PetscInt rStart = 0;
    const PetscInt rEnd   = options->numCells/options->size + PetscMin(1, options->numCells%options->size);

    for(PetscInt c = rStart, locC = 0; c < rEnd; ++c, ++locC) {
      parallelSection->setFiberDimension(locC, 4);
      parallelSection->updatePoint(locC, serialSection->restrictPoint(c));
    }
  }
  ALE::Completion::completeSection(sendOverlap, recvOverlap, serialSection, parallelSection);
  if (options->debug) {parallelSection->view("Parallel UniformSection");}
  PetscFunctionReturn(0);
}
#endif

// Describe the completion process as the composition of two pullbacks:
//
//   copy(): unifies pullback to the overlap (restrict) and copy to neighboring process
//   fuse(): unifies pullback across the overlap, then to the big section (update), and then fusion
//
// For parallel completion, we must first copy the section, restricted to the
// overlap, to the other process, then we can proceed as above. The copy
// process can be broken down as in the distribution paper:
//
//   1) Copy atlas (perhaps empty, signaling a single datum per point)
//   2) Copy data
//
// The atlas copy is recursive, since it is itself a section.
//
// Thus, we have a software plan
//
// - Write parallel section copy (currently in ALE::Field)
//
// - Write pullback as a generic algorithm
//
// - Write fuse as a generic algorithm
//   - Implement add and replace fusers
//
// Now we move back to
//   We do not need the initial pullback
//   copy: Copy the vector and leave it in the old domain
//   fuse: fuse the copy, pulled back to the new domain, with the existing whole section
//
// Also, we need a way to update overlaps based on a renumbering
//
// We need to wrap distribution around completion
//   Create local section
//   Complete, with fuser which adds in
#undef __FUNCT__
#define __FUNCT__ "SectionTest"
PetscErrorCode SectionTest(const Options *options)
{
  typedef int point_type;
  typedef ALE::Sifter<int,point_type,point_type> send_overlap_type;
  typedef ALE::Sifter<point_type,int,point_type> recv_overlap_type;
  typedef ALE::Section<point_type, double> section;
  Obj<send_overlap_type> sendOverlap     = new send_overlap_type(options->comm);
  Obj<recv_overlap_type> recvOverlap     = new recv_overlap_type(options->comm);
  Obj<section>           serialSection   = new section(options->comm, options->debug);
  Obj<section>           parallelSection = new section(options->comm, options->debug);
  section::value_type   *value;
  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(options->components * sizeof(double), &value);CHKERRQ(ierr);
  ierr = CreateScatterOverlap(sendOverlap, recvOverlap, true, options);CHKERRQ(ierr);
  if (!options->rank) {
    for(PetscInt c = 0; c < options->numCells; ++c) {
      serialSection->setFiberDimension(c, options->components);
    }
  }
  serialSection->allocatePoint();
  if (!options->rank) {
    for(PetscInt c = 0; c < options->numCells; ++c) {
      for(PetscInt comp = 0; comp < options->components; ++comp) {value[comp] = (c+1)*(comp+1);}
      serialSection->updatePoint(c, value);
    }
  }
  ierr = PetscFree(value);CHKERRQ(ierr);
  ALE::Completion::completeSection(sendOverlap, recvOverlap, serialSection, parallelSection);
  if (!options->rank) {
    const PetscInt rStart = 0;
    const PetscInt rEnd   = options->numCells/options->size + PetscMin(1, options->numCells%options->size);

    for(PetscInt c = rStart, locC = 0; c < rEnd; ++c, ++locC) {
      parallelSection->setFiberDimension(locC, serialSection->getFiberDimension(c));
    }
    parallelSection->allocatePoint();
    for(PetscInt c = rStart, locC = 0; c < rEnd; ++c, ++locC) {
      parallelSection->updatePoint(locC, serialSection->restrictPoint(c));
    }
  }
  if (options->debug) {parallelSection->view("Parallel Section");}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SectionTests"
PetscErrorCode SectionTests(const Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  //ierr = ConstantSectionTest(options);CHKERRQ(ierr);
  //ierr = UniformSectionTest(options);CHKERRQ(ierr);
  ierr = SectionTest(options);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SectionToISectionTest"
PetscErrorCode SectionToISectionTest(const Options *options)
{
  typedef int point_type;
  typedef ALE::Sifter<point_type,int,point_type> send_overlap_type;
  typedef ALE::Sifter<int,point_type,point_type> recv_overlap_type;
  typedef ALE::Section<point_type, double>  section;
  typedef ALE::ISection<point_type, double> isection;
  Obj<send_overlap_type> sendOverlap     = new send_overlap_type(options->comm);
  Obj<recv_overlap_type> recvOverlap     = new recv_overlap_type(options->comm);
  Obj<section>           serialSection   = new section(options->comm, options->debug);
  section::value_type   *value;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(options->components * sizeof(double), &value);CHKERRQ(ierr);
  ierr = CreateScatterOverlap(sendOverlap, recvOverlap, true, options);CHKERRQ(ierr);
  const int     localSize       = options->rank ? recvOverlap->base()->size() : options->numCells/options->size + PetscMin(1, options->numCells%options->size);
  Obj<isection> parallelSection = new isection(options->comm, 0, localSize, options->debug);

  if (!options->rank) {
    for(PetscInt c = 0; c < options->numCells; ++c) {
      serialSection->setFiberDimension(c, options->components);
    }
  }
  serialSection->allocatePoint();
  if (!options->rank) {
    for(PetscInt c = 0; c < options->numCells; ++c) {
      for(PetscInt comp = 0; comp < options->components; ++comp) {value[comp] = (c+1)*(comp+1);}
      serialSection->updatePoint(c, value);
    }
  }
  ierr = PetscFree(value);CHKERRQ(ierr);
  ALE::Completion::completeSection(sendOverlap, recvOverlap, serialSection, parallelSection);
  if (!options->rank) {
    const PetscInt rStart = 0;
    const PetscInt rEnd   = options->numCells/options->size + PetscMin(1, options->numCells%options->size);

    for(PetscInt c = rStart, locC = 0; c < rEnd; ++c, ++locC) {
      parallelSection->setFiberDimension(locC, serialSection->getFiberDimension(c));
    }
    parallelSection->allocatePoint();
    for(PetscInt c = rStart, locC = 0; c < rEnd; ++c, ++locC) {
      parallelSection->updatePoint(locC, serialSection->restrictPoint(c));
    }
  }
  if (options->debug) {parallelSection->view("Parallel ISection");}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISectionTests"
PetscErrorCode ISectionTests(const Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionToISectionTest(options);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SieveISectionPartitionTest"
PetscErrorCode SieveISectionPartitionTest(const Options *options)
{
  typedef ALE::Mesh<PetscInt,PetscScalar>              FlexMesh;
  typedef FlexMesh::point_type                         point_type;
  typedef ALE::Partitioner<>::part_type                rank_type;
  typedef ALE::Sifter<point_type,rank_type,point_type> mesh_send_overlap_type;
  typedef ALE::Sifter<rank_type,point_type,point_type> mesh_recv_overlap_type;
  typedef ALE::DistributionNew<FlexMesh>               distribution_type;
  typedef distribution_type::partition_type            partition_type;
  double                            lower[2]        = {0.0, 0.0};
  double                            upper[2]        = {1.0, 1.0};
  int                               edges[2]        = {2, 2};
  const Obj<FlexMesh>               mB              = ALE::MeshBuilder<FlexMesh>::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, edges, 0);
  const Obj<FlexMesh>               mesh            = ALE::Generator<FlexMesh>::generateMesh(mB, options->interpolate);
  Obj<FlexMesh>                     parallelMesh    = new FlexMesh(options->comm, mesh->getDimension(), options->debug);
  Obj<FlexMesh::sieve_type>         parallelSieve   = new FlexMesh::sieve_type(options->comm, options->debug);
  const Obj<mesh_send_overlap_type> sendMeshOverlap = new mesh_send_overlap_type(mesh->comm(), mesh->debug());
  const Obj<mesh_recv_overlap_type> recvMeshOverlap = new mesh_recv_overlap_type(mesh->comm(), mesh->debug());
  const int                         height          = 0;
  std::map<point_type,point_type>   renumbering;

  PetscFunctionBegin;
  mesh->setDebug(options->debug);
  parallelMesh->setSieve(parallelSieve);
  if (options->debug) {mesh->view("Serial Mesh");}
  Obj<partition_type> partition = distribution_type::distributeMesh(mesh, parallelMesh, renumbering, sendMeshOverlap, recvMeshOverlap, height);
  if (options->debug) {parallelMesh->view("Parallel Mesh");}
  // Distribute the coordinates
  typedef ALE::ISection<point_type, double> real_section_type;
  const Obj<FlexMesh::real_section_type>& coordinates    = mesh->getRealSection("coordinates");
  const int                                firstVertex    = parallelMesh->heightStratum(0)->size();
  const int                                lastVertex     = firstVertex+parallelMesh->depthStratum(0)->size();
  const Obj<real_section_type>             newCoordinates = new real_section_type(parallelMesh->comm(), firstVertex, lastVertex, parallelMesh->debug());

  distribution_type::distributeSection(coordinates, partition, renumbering, sendMeshOverlap, recvMeshOverlap, newCoordinates);
  if (options->debug) {newCoordinates->view("Parallel Coordinates");}
  // Create the parallel overlap
  Obj<mesh_send_overlap_type> sendParallelMeshOverlap = new mesh_send_overlap_type(options->comm);
  Obj<mesh_recv_overlap_type> recvParallelMeshOverlap = new mesh_recv_overlap_type(options->comm);
  //   Can I figure this out in a nicer way?
  ALE::SetFromMap<std::map<point_type,point_type> > globalPoints(renumbering);

  ALE::OverlapBuilder<>::constructOverlap(globalPoints, renumbering, sendParallelMeshOverlap, recvParallelMeshOverlap);
  sendParallelMeshOverlap->view("Send parallel mesh overlap");
  recvParallelMeshOverlap->view("Receive parallel mesh overlap");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISieveISectionPartitionTest"
PetscErrorCode ISieveISectionPartitionTest(const Options *options)
{
  typedef ALE::Mesh<PetscInt,PetscScalar>              FlexMesh;
  typedef ALE::IMesh<PetscInt,PetscScalar>             mesh_type;
  typedef FlexMesh::point_type                         point_type;
  typedef ALE::Partitioner<>::part_type                rank_type;
  typedef ALE::Sifter<point_type,rank_type,point_type> mesh_send_overlap_type;
  typedef ALE::Sifter<rank_type,point_type,point_type> mesh_recv_overlap_type;
  typedef ALE::DistributionNew<mesh_type>              distribution_type;
  typedef distribution_type::partition_type            partition_type;
  double                            lower[2]        = {0.0, 0.0};
  double                            upper[2]        = {1.0, 1.0};
  int                               edges[2]        = {2, 2};
  const Obj<FlexMesh>               mB              = ALE::MeshBuilder<FlexMesh>::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, edges, 0);
  const Obj<FlexMesh>               m               = ALE::Generator<FlexMesh>::generateMesh(mB, options->interpolate);
  Obj<mesh_type>                    mesh            = new mesh_type(options->comm, m->getDimension(), options->debug);
  Obj<mesh_type::sieve_type>        sieve           = new mesh_type::sieve_type(options->comm, options->debug);
  Obj<mesh_type>                    parallelMesh    = new mesh_type(options->comm, m->getDimension(), options->debug);
  Obj<mesh_type::sieve_type>        parallelSieve   = new mesh_type::sieve_type(options->comm, options->debug);
  const Obj<mesh_send_overlap_type> sendMeshOverlap = new mesh_send_overlap_type(mesh->comm(), mesh->debug());
  const Obj<mesh_recv_overlap_type> recvMeshOverlap = new mesh_recv_overlap_type(mesh->comm(), mesh->debug());
  const int                         height          = 0;
  std::map<point_type,point_type>   renumbering;

  PetscFunctionBegin;
  mesh->setSieve(sieve);
  ALE::ISieveConverter::convertMesh(*m, *mesh, renumbering);
  renumbering.clear();
  parallelMesh->setSieve(parallelSieve);
  if (options->debug) {mesh->view("Serial Mesh");}
  Obj<partition_type> partition = distribution_type::distributeMeshV(mesh, parallelMesh, renumbering, sendMeshOverlap, recvMeshOverlap, height);
  if (options->debug) {parallelMesh->view("Parallel Mesh");}
  // Distribute the coordinates
  typedef mesh_type::real_section_type real_section_type;
  const Obj<real_section_type>& coordinates         = mesh->getRealSection("coordinates");
  const Obj<real_section_type>& parallelCoordinates = parallelMesh->getRealSection("coordinates");

  if (options->debug) {coordinates->view("Serial Coordinates");}
  parallelMesh->setupCoordinates(parallelCoordinates);
  distribution_type::distributeSection(coordinates, partition, renumbering, sendMeshOverlap, recvMeshOverlap, parallelCoordinates);
  if (options->debug) {parallelCoordinates->view("Parallel Coordinates");}
  // Create the parallel overlap
  Obj<mesh_send_overlap_type> sendParallelMeshOverlap = new mesh_send_overlap_type(options->comm);
  Obj<mesh_recv_overlap_type> recvParallelMeshOverlap = new mesh_recv_overlap_type(options->comm);
  //   Can I figure this out in a nicer way?
  ALE::SetFromMap<std::map<point_type,point_type> > globalPoints(renumbering);

  ALE::OverlapBuilder<>::constructOverlap(globalPoints, renumbering, sendParallelMeshOverlap, recvParallelMeshOverlap);
  sendParallelMeshOverlap->view("Send parallel mesh overlap");
  recvParallelMeshOverlap->view("Receive parallel mesh overlap");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISieveISectionPartitionNewOverlapTest"
PetscErrorCode ISieveISectionPartitionNewOverlapTest(const Options *options)
{
  typedef ALE::Mesh<PetscInt,PetscScalar>              FlexMesh;
  typedef ALE::IMesh<PetscInt,PetscScalar>             mesh_type;
  typedef FlexMesh::point_type                         point_type;
  typedef ALE::Partitioner<>::part_type                rank_type;
  typedef PetscNewSendOverlap<point_type,rank_type>    mesh_send_overlap_type;
  typedef PetscNewRecvOverlap<point_type,rank_type>    mesh_recv_overlap_type;
  typedef ALE::DistributionNew<mesh_type>              distribution_type;
  typedef distribution_type::partition_type            partition_type;
  double                            lower[2]        = {0.0, 0.0};
  double                            upper[2]        = {1.0, 1.0};
  int                               edges[2]        = {2, 2};
  const Obj<FlexMesh>               mB              = ALE::MeshBuilder<FlexMesh>::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, edges, 0);
  const Obj<FlexMesh>               m               = ALE::Generator<FlexMesh>::generateMesh(mB, options->interpolate);
  Obj<mesh_type>                    mesh            = new mesh_type(options->comm, m->getDimension(), options->debug);
  Obj<mesh_type::sieve_type>        sieve           = new mesh_type::sieve_type(options->comm, options->debug);
  Obj<mesh_type>                    parallelMesh    = new mesh_type(options->comm, m->getDimension(), options->debug);
  Obj<mesh_type::sieve_type>        parallelSieve   = new mesh_type::sieve_type(options->comm, options->debug);
  const Obj<mesh_send_overlap_type> sendMeshOverlap = new mesh_send_overlap_type(mesh->comm(), mesh->debug());
  const Obj<mesh_recv_overlap_type> recvMeshOverlap = new mesh_recv_overlap_type(mesh->comm(), mesh->debug());
  const int                         height          = 0;
  std::map<point_type,point_type>   renumbering;

  PetscFunctionBegin;
  mesh->setSieve(sieve);
  ALE::ISieveConverter::convertMesh(*m, *mesh, renumbering);
  renumbering.clear();
  parallelMesh->setSieve(parallelSieve);
  sendMeshOverlap->setName("Send mesh overlap");
  recvMeshOverlap->setName("Receive mesh overlap");
  if (options->debug) {mesh->view("Serial Mesh");}
  Obj<partition_type> partition = distribution_type::distributeMeshV(mesh, parallelMesh, renumbering, sendMeshOverlap, recvMeshOverlap, height);
  if (options->debug) {parallelMesh->view("Parallel Mesh");}
  //sendMeshOverlap->view("");
  //recvMeshOverlap->view("");
  // Distribute the coordinates
  typedef mesh_type::real_section_type real_section_type;
  const Obj<real_section_type>& coordinates         = mesh->getRealSection("coordinates");
  const Obj<real_section_type>& parallelCoordinates = parallelMesh->getRealSection("coordinates");

  if (options->debug) {coordinates->view("Serial Coordinates");}
  parallelMesh->setupCoordinates(parallelCoordinates);
  distribution_type::distributeSection(coordinates, partition, renumbering, sendMeshOverlap, recvMeshOverlap, parallelCoordinates);
  if (options->debug) {parallelCoordinates->view("Parallel Coordinates");}
  // Create the parallel overlap
  Obj<mesh_send_overlap_type> sendParallelMeshOverlap = new mesh_send_overlap_type(options->comm);
  Obj<mesh_recv_overlap_type> recvParallelMeshOverlap = new mesh_recv_overlap_type(options->comm);
  //   Can I figure this out in a nicer way?
  ALE::SetFromMap<std::map<point_type,point_type> > globalPoints(renumbering);

  //ALE::OverlapBuilder<>::constructOverlap(globalPoints, renumbering, sendParallelMeshOverlap, recvParallelMeshOverlap);
  //sendParallelMeshOverlap->view("Send parallel mesh overlap");
  //recvParallelMeshOverlap->view("Receive parallel mesh overlap");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PartitionTests"
PetscErrorCode PartitionTests(const Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SieveISectionPartitionTest(options);CHKERRQ(ierr);
  ierr = ISieveISectionPartitionTest(options);CHKERRQ(ierr);
  ierr = ISieveISectionPartitionNewOverlapTest(options);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RunUnitTests"
PetscErrorCode RunUnitTests(const Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->section)   {ierr = SectionTests(options);CHKERRQ(ierr);}
  if (options->isection)  {ierr = ISectionTests(options);CHKERRQ(ierr);}
  if (options->partition) {ierr = PartitionTests(options);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  Options        options;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscLogBegin();CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &options);CHKERRQ(ierr);
  try {
    ierr = RunUnitTests(&options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cerr << e << std::endl;
  }
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
