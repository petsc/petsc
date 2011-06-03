#ifndef included_PETSc_Overlap_hh
#define included_PETSc_Overlap_hh

#ifndef  included_ALE_containers_hh
#include <sieve/ALE_containers.hh>
#endif

/* PetscOverlap:

   An Overlap is an object relating two sets of points which lie on different processes.

Interface:

   1) Give the size and iterate over the set of partner process ranks
   2) Give the size and iterate over the ordered set of local points for each rank
   3) Give the size and iterate over the ordered set of remote points for each rank
   4) Match up local and remote points

This implementation uses a compressed row (CR) form for the points+remotePoints,
and an auxiliary array of ranks used to convert to an index for the CR structure.
In addition, STL containers are used if needed for flexible construction without
preallocation. Upon assembly, the data is moved from STL to CR structures.
*/
namespace PETSc {
template<typename Point, typename Rank>
class Overlap : public ALE::ParallelObject {
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
  Overlap(MPI_Comm comm, const int debug = 0) : ALE::ParallelObject(comm, debug) {
    this->numRanks     = 0;
    this->ranks        = PETSC_NULL;
    this->pointsOffset = PETSC_NULL;
    this->points       = PETSC_NULL;
    this->remotePoints = PETSC_NULL;
  };
  ~Overlap() {
    PetscErrorCode ierr;
    ierr = PetscFree(this->ranks);CHKERRXX(ierr);
    ierr = PetscFree(this->pointsOffset);CHKERRXX(ierr);
    ierr = PetscFree(this->points);CHKERRXX(ierr);
    ierr = PetscFree(this->remotePoints);CHKERRXX(ierr);
  };
  /* setNumRanks - Set the number of partner processes

     This allocates storage for the ranks and must only be called once. 
  */
  void setNumRanks(index_type R) {
    PetscErrorCode ierr;

    if (R) {
      numRanks = R;
      ierr = PetscMalloc(numRanks * sizeof(rank_type), &ranks);CHKERRXX(ierr);
      ierr = PetscMalloc((numRanks+1) * sizeof(index_type), &pointsOffset);CHKERRXX(ierr);
      for(index_type r = 0; r < numRanks; ++r) {
        ranks[r]        = -1;
        pointsOffset[r] = 0;
      }
    }
  };
  /* getRankIndex - Map from a process rank to an index in [0, numRanks) */
  index_type getRankIndex(index_type rank) {
    for(index_type r = 0; r < numRanks; ++r) {
      if (ranks[r] == rank) {
        return r;
      }
    }
    throw ALE::Exception("Invalid rank was not contained in this overlap");
  };
  /* setNumPoints - Set the number of points matched to partner process 'rank'

     Note: Currently, this should only be called once
  */
  void setNumPoints(rank_type rank, index_type numPoints) {
    index_type r, s;
    for(r = 0; r < numRanks; ++r) {
      if (rank >= ranks[r]) break;
    }
    assert(r < numRanks);
    if (rank != ranks[r]) {
      assert(ranks[numRanks-1] == -1);
      for(s = numRanks; s > r; --s) {
        pointsOffset[s] = pointsOffset[s-1];
      }
    }
    ranks[r] = rank;
    for(s = r+1; s <= numRanks; ++s) {
      pointsOffset[s] += numPoints;
    }
  };
  /* getNumPoints - Return the number of points matched to partner process index 'r' */
  index_type getNumPoints(index_type r) {
    return this->pointsOffset[r+1] - this->pointsOffset[r];
  };
  /* getNumPointsByRank - Return the number of points matched to partner process 'rank' */
  index_type getNumPointsByRank(index_type rank) {
    const index_type r = this->getRankIndex(rank);
    return this->pointsOffset[r+1] - this->pointsOffset[r];
  };
  /* assembleFlexible - Compress data from flexible construction into CR structures */
  void assembleFlexible() {
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
  /* assembleFast - Allocate CR storage for overlap data */
  void assembleFast() {
    PetscErrorCode ierr;

    assert(!flexRanks.size() && !flexPoints.size() && !flexRemotePoints.size());
    ierr = PetscMalloc((pointsOffset[numRanks]) * sizeof(point_type), &points);CHKERRXX(ierr);
    ierr = PetscMalloc((pointsOffset[numRanks]) * sizeof(point_type), &remotePoints);CHKERRXX(ierr);
    for(index_type i = 0; i < pointsOffset[numRanks]; ++i) {
      points[i] = remotePoints[i] = -1;
    }
  };
  /* assemble - Complete preallocation phase (optimized method) or input phase (flexible method) */
  void assemble() {
    if (!flexRanks.size() && ranks) {
      assembleFast();
    } else if (!ranks && !pointsOffset) {
      assembleFlexible();
    } else {
      std::cout << "["<<this->commRank()<<"]ranks: " << ranks << " pointsOffset: " << pointsOffset << " flexRank size: " << flexRanks.size() << std::endl;
      throw ALE::Exception("Cannot assemble overlap in an invalid state");
    }
  };
  void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) const {
    PetscErrorCode ierr;

    if (!this->commRank()) {
      ierr = PetscSynchronizedPrintf(this->comm(), "[%d]%s: %s\n", this->commRank(), this->getName().size() ? this->getName().c_str() : "Overlap", name.c_str());
    }
    ierr = PetscSynchronizedPrintf(this->comm(), "[%d]%d partners:\n", this->commRank(), this->numRanks);
    for(index_type r = 0; r < this->numRanks; ++r) {
      ierr = PetscSynchronizedPrintf(this->comm(), "[%d]  %d:", this->commRank(), this->ranks[r]);
      for(index_type p = pointsOffset[r]; p < pointsOffset[r+1]; ++p) {
        ierr = PetscSynchronizedPrintf(this->comm(), "  %d (%d)", this->points[p], this->remotePoints[p]);
      }
      ierr = PetscSynchronizedPrintf(this->comm(), "\n");
    }
    ierr = PetscSynchronizedFlush(this->comm());
  };
};

// Compatibility wrapper for raw pointers
template<typename Value>
class Sequence {
public:
  typedef Value *iterator;
};

/* Compatibility wrapper which translates ALE::Sifter calls to Petsc::Overlap calls for a send overlap

   Note that addArrow() works for both flexible and optimized construction modes.
*/
template<typename Point, typename Rank>
class SendOverlap : public Overlap<Point,Rank> {
public:
  typedef PetscInt index_type;
  typedef Point    point_type;
  typedef Rank     rank_type;
  // Backwards compatibility
  typedef point_type source_type;
  typedef rank_type  target_type;
  typedef point_type color_type;
  typedef Sequence<rank_type>  baseSequence;
  typedef Sequence<point_type> coneSequence;
public:
  SendOverlap(MPI_Comm comm, const int debug = 0) : Overlap<Point,Rank>(comm, debug) {};
  ~SendOverlap() {};
  void addArrow(source_type s, target_type t, color_type c) {
    if (!this->ranks) {
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
    } else {
      const index_type r = this->getRankIndex(t);
      index_type       i;

      for(i = this->pointsOffset[r]; i < this->pointsOffset[r+1]; ++i) {
        if (s <= this->points[i] || this->points[i] < 0) break;
      }
      assert(i < this->pointsOffset[r+1] && s != this->points[i]);
      for(index_type j = this->pointsOffset[r+1]-1; j > i; --j) {
        this->points[j]       = this->points[j-1];
        this->remotePoints[j] = this->remotePoints[j-1];
      }
      this->points[i]       = s;
      this->remotePoints[i] = c;
    }
  };
  void setBaseSize(index_type size) {
    this->setNumRanks(size);
  };
  typename baseSequence::iterator baseBegin() {
    assert(!this->numRanks || this->ranks);
    return this->ranks;
  };
  typename baseSequence::iterator baseEnd() {
    assert(!this->numRanks || this->ranks);
    return &this->ranks[this->numRanks];
  };
  void setConeSize(rank_type rank, index_type size) {
    this->setNumPoints(rank, size);
  };
  int getConeSize(rank_type rank) {
    return this->getNumPointsByRank(rank);
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

// This structure marches 2 pointers at the same time
template<typename Point>
class SupportSequence {
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

/* Compatibility wrapper which translates ALE::Sifter calls to Petsc::Overlap calls for a receive overlap

   Note that addArrow() works for both flexible and optimized construction modes.
*/
template<typename Point, typename Rank>
class RecvOverlap : public Overlap<Point,Rank> {
public:
  typedef PetscInt index_type;
  typedef Point    point_type;
  typedef Rank     rank_type;
  // Backwards compatibility
  typedef rank_type  source_type;
  typedef point_type target_type;
  typedef point_type color_type;
  typedef Sequence<rank_type>         capSequence;
  typedef SupportSequence<point_type> supportSequence;
public:
  RecvOverlap(MPI_Comm comm, const int debug = 0) : Overlap<Point,Rank>(comm, debug) {};
  ~RecvOverlap() {};
  void addArrow(source_type s, target_type t, color_type c) {
    if (!this->ranks) {
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
    } else {
      const index_type r = this->getRankIndex(s);
      index_type       i;

      for(i = this->pointsOffset[r]; i < this->pointsOffset[r+1]; ++i) {
        if (t <= this->points[i] || this->points[i] < 0) break;
      }
      assert(i < this->pointsOffset[r+1] && s != this->points[i]);
      for(index_type j = this->pointsOffset[r+1]-1; j > i; --j) {
        this->points[j]       = this->points[j-1];
        this->remotePoints[j] = this->remotePoints[j-1];
      }
      this->points[i]       = t;
      this->remotePoints[i] = c;
    }
  };
  typename capSequence::iterator capBegin() {
    assert(!this->numRanks || this->ranks);
    return this->ranks;
  };
  typename capSequence::iterator capEnd() {
    assert(!this->numRanks || this->ranks);
    return &this->ranks[this->numRanks];
  };
  void setCapSize(index_type size) {
    this->setNumRanks(size);
  };
  void setSupportSize(rank_type rank, index_type size) {
    this->setNumPoints(rank, size);
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
}

#endif /* included_PETSc_Overlap_hh */
