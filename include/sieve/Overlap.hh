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
  PetscInt    numRanks;          // Number of partner processes
  rank_type  *ranks;             // [numRanks]:   MPI Rank of each partner process
  index_type *pointsOffset;      // [numRanks+1]: Offset into points array for each partner process
  point_type *points;            // [pointsOffset[numRanks]]: Points array for each partner process, in sorted order
  point_type *remotePoints;      // [pointsOffset[numRanks]]: Remote points array for each partner process
  index_type  numPointRanks;     // Number of partner process ranks which share a given point (needs search)
  rank_type  *pointRanks;        // [numPointRanks]: Array of partner process ranks which share a given point (needs search and allocation)
  point_type *pointRemotePoints; // [numPointRanks]: Array of remote points linked to a given point (needs search and allocation)

  index_type *insertOffset;      // [numRanks]: Offset into points array for fast assembly
  rank_type  *cachedRanks;       // [?]: Cache for rank iteration during flexible assembly
  bool        cachedRanksValid;  //      Flag indicating whether the cahce is consistent

  std::vector<rank_type> flexRanks;
  std::map<rank_type, std::vector<point_type> > flexPoints;
  std::map<rank_type, std::vector<point_type> > flexRemotePoints;
public:
  Overlap(MPI_Comm comm, const int debug = 0) : ALE::ParallelObject(comm, debug) {
    this->numRanks          = 0;
    this->ranks             = PETSC_NULL;
    this->pointsOffset      = PETSC_NULL;
    this->points            = PETSC_NULL;
    this->remotePoints      = PETSC_NULL;
    this->numPointRanks     = 0;
    this->pointRanks        = PETSC_NULL;
    this->pointRemotePoints = PETSC_NULL;

    this->insertOffset      = PETSC_NULL;
    this->cachedRanks       = PETSC_NULL;
    this->cachedRanksValid  = false;
  };
  ~Overlap() {
    PetscErrorCode ierr;
    ierr = PetscFree(this->ranks);CHKERRXX(ierr);
    ierr = PetscFree(this->pointsOffset);CHKERRXX(ierr);
    ierr = PetscFree(this->points);CHKERRXX(ierr);
    ierr = PetscFree(this->remotePoints);CHKERRXX(ierr);
    ierr = PetscFree2(this->pointRanks, this->pointRemotePoints);CHKERRXX(ierr);

    ierr = PetscFree(this->insertOffset);CHKERRXX(ierr);
  };
  index_type getNumRanks() {
    return this->numRanks;
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
  PetscErrorCode getRank(index_type r, index_type *rank) {
    assert(r >= 0 && r < numRanks);
    *rank = ranks[r];
    return 0;
  };
  /* getRankIndex - Map from a process rank to an index in [0, numRanks) */
  PetscErrorCode getRankIndex(index_type rank, index_type *rankIndex) {
    *rankIndex = -1;
    for(index_type r = 0; r < numRanks; ++r) {
      if (ranks[r] == rank) {
        *rankIndex = r;
        return 0;
      }
    }
    //SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid rank %d was not contained in this overlap", rank);
    return PETSC_ERR_ARG_OUTOFRANGE;
  };
  /* setNumPoints - Set the number of points matched to partner process 'rank'

     Note: Currently, this should only be called once
  */
  void setNumPoints(rank_type rank, index_type numPoints) {
    index_type r, s;
    /* This will sort rank in reverse order */
    for(r = 0; r < numRanks; ++r) {
      if (rank >= ranks[r]) break;
    }
    assert(r < numRanks);
    if (rank != ranks[r]) {
      assert(ranks[numRanks-1] == -1);
      pointsOffset[numRanks] = pointsOffset[numRanks-1];
      for(s = numRanks-1; s > r; --s) {
        ranks[s]        = ranks[s-1];
        pointsOffset[s] = pointsOffset[s-1];
      }
      ranks[r] = rank;
    }
    for(s = r+1; s <= numRanks; ++s) {
      pointsOffset[s] += numPoints;
    }
  };
  index_type getNumPoints() {
    if (!numRanks) return 0;
    return this->pointsOffset[numRanks] - this->pointsOffset[0];
  };
  /* getNumPoints - Return the number of points matched to partner process index 'r' */
  index_type getNumPoints(index_type r) {
    assert(r >= 0 && r < numRanks);
    return this->pointsOffset[r+1] - this->pointsOffset[r];
  };
  /* getNumPointsByRank - Return the number of points matched to partner process 'rank' */
  PetscErrorCode getNumPointsByRank(index_type rank, index_type *numPoints) {
    index_type     r;
    PetscErrorCode ierr;
    ierr = this->getRankIndex(rank, &r); if (ierr) {return ierr;}
    *numPoints = this->pointsOffset[r+1] - this->pointsOffset[r];
    return 0;
  };
  /* SLOW, since it involves a search */
  void getRanks(point_type point, index_type *size, const rank_type **ranks, const point_type **remotePoints) {
    std::vector<rank_type>  pRanks;
    std::vector<point_type> pPoints;
    PetscErrorCode          ierr;

    ierr = PetscFree2(pointRanks, pointRemotePoints);CHKERRXX(ierr);
    if (this->ranks || !this->flexRanks.size()) {
      for(index_type r = 0; r < numRanks; ++r) {
        for(index_type p = pointsOffset[r]; p < pointsOffset[r+1]; ++p) {
          if (points[p] == point) {
            pRanks.push_back(this->ranks[r]);
            pPoints.push_back(this->remotePoints[p]);
            break;
          }
        }
      }
    } else {
      /* Slow query to flexible assembly stuff */
      for(typename std::vector<rank_type>::const_iterator r_iter = flexRanks.begin(); r_iter != flexRanks.end(); ++r_iter) {
        const rank_type rank = *r_iter;
        for(typename std::vector<point_type>::const_iterator p_iter = flexPoints[rank].begin(), rp_iter = flexRemotePoints[rank].begin(); p_iter != flexPoints[rank].end(); ++p_iter, ++rp_iter) {
          if (*p_iter == point) {
            pRanks.push_back(rank);
            pPoints.push_back(*rp_iter);
            break;
          }
        }
      }
    }
    numPointRanks = pRanks.size();
    ierr = PetscMalloc2(numPointRanks,rank_type,&pointRanks,numPointRanks,point_type,&pointRemotePoints);CHKERRXX(ierr);
    for(index_type i = 0; i < numPointRanks; ++i) {
      pointRanks[i]        = pRanks[i];
      pointRemotePoints[i] = pPoints[i];
    }
    *size         = numPointRanks;
    *ranks        = pointRanks;
    *remotePoints = pointRemotePoints;
  };
  /* Make sure that the cachedRanks match flexRanks */
  void ensureCachedRanks() {
    PetscErrorCode ierr;

    if (this->cachedRanksValid) return;
    ierr = PetscFree(this->cachedRanks);CHKERRXX(ierr);
    ierr = PetscMalloc(this->flexRanks.size() * sizeof(rank_type), &this->cachedRanks);CHKERRXX(ierr);
    index_type r = 0;
    for(typename std::vector<rank_type>::const_iterator r_iter = flexRanks.begin(); r_iter != flexRanks.end(); ++r_iter, ++r) {
      this->cachedRanks[r] = *r_iter;
    }
    this->cachedRanksValid = true;
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
    ierr = PetscFree(this->cachedRanks);CHKERRXX(ierr);
  };
  /* assembleFast - Allocate CR storage for overlap data */
  void assembleFast() {
    PetscErrorCode ierr;

    assert(!flexRanks.size() && !flexPoints.size() && !flexRemotePoints.size());
    ierr = PetscMalloc((pointsOffset[numRanks]) * sizeof(point_type), &points);CHKERRXX(ierr);
    ierr = PetscMalloc((pointsOffset[numRanks]) * sizeof(point_type), &remotePoints);CHKERRXX(ierr);
#if 1
    for(index_type i = 0; i < pointsOffset[numRanks]; ++i) {
      points[i] = remotePoints[i] = -1;
    }
#endif
    ierr = PetscMalloc(numRanks * sizeof(index_type), &insertOffset);CHKERRXX(ierr);
    for(index_type r = 0; r < numRanks; ++r) {
      insertOffset[r] = pointsOffset[r];
    }
  };
  /* assemble - Complete preallocation phase (optimized method) or input phase (flexible method) */
  void assemble() {
    if (!flexRanks.size() && !ranks) return;
    if (!flexRanks.size() && ranks) {
      assembleFast();
    } else if (!ranks && !pointsOffset) {
      assembleFlexible();
    } else {
      std::cout << "["<<this->commRank()<<"]ranks: " << ranks << " pointsOffset: " << pointsOffset << " flexRank size: " << flexRanks.size() << std::endl;
      throw ALE::Exception("Cannot assemble overlap in an invalid state");
    }
  };
  /* assemble - Complete point insertion phase (optimized method) or do nothing (flexible method) */
  void assemblePoints() {
    if (this->insertOffset) {
      for(index_type r = 0; r < numRanks; ++r) {
        PetscErrorCode ierr;

        if (insertOffset[r] != pointsOffset[r+1]) {
          std::cout << "Should have point offset "<<pointsOffset[r+1]<<" for rank "<<this->ranks[r]<<", not "<<insertOffset[r]<<std::endl;
          throw ALE::Exception("Cannot assemble overlap points in an invalid state");
        }
        ierr = PetscSortIntWithArray(pointsOffset[r+1]-pointsOffset[r], &points[pointsOffset[r]], &remotePoints[pointsOffset[r]]);CHKERRXX(ierr);
      }
    }
  };
  void copy(Overlap *o) {
    PetscErrorCode ierr;

    o->numRanks = this->numRanks;
    if (o->numRanks) {
      ierr = PetscMalloc(o->numRanks * sizeof(rank_type), &o->ranks);CHKERRXX(ierr);
      ierr = PetscMemcpy(o->ranks, this->ranks, o->numRanks * sizeof(rank_type));CHKERRXX(ierr);
      ierr = PetscMalloc((o->numRanks+1) * sizeof(index_type), &o->pointsOffset);CHKERRXX(ierr);
      ierr = PetscMemcpy(o->pointsOffset, this->pointsOffset, (o->numRanks+1) * sizeof(index_type));CHKERRXX(ierr);
      ierr = PetscMalloc(o->pointsOffset[numRanks] * sizeof(point_type), &o->points);CHKERRXX(ierr);
      ierr = PetscMalloc(o->pointsOffset[numRanks] * sizeof(point_type), &o->remotePoints);CHKERRXX(ierr);
      ierr = PetscMemcpy(o->points, this->points, o->pointsOffset[numRanks] * sizeof(point_type));CHKERRXX(ierr);
      ierr = PetscMemcpy(o->remotePoints, this->remotePoints, o->pointsOffset[numRanks] * sizeof(point_type));CHKERRXX(ierr);
    }
    o->numPointRanks = this->numPointRanks;
    if (o->numPointRanks) {
      ierr = PetscMalloc(o->numPointRanks * sizeof(rank_type), &o->pointRanks);CHKERRXX(ierr);
      ierr = PetscMemcpy(o->pointRanks, this->pointRanks, o->numPointRanks * sizeof(rank_type));CHKERRXX(ierr);
    }
  };
  template<typename Labeling>
  void relabel(Labeling& relabeling, Overlap& newLabel) {
    this->copy(&newLabel);
    if (numRanks) {
      for(index_type i = 0; i < pointsOffset[numRanks]; ++i) {
        newLabel.points[i]       = relabeling.restrictPoint(points[i])[0];
        newLabel.remotePoints[i] = relabeling.restrictPoint(remotePoints[i])[0];
      }
    }
  }
  void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) const {
    PetscErrorCode ierr;

    if (!this->commRank()) {
      ierr = PetscSynchronizedPrintf(this->comm(), "[%d]%s: %s\n", this->commRank(), this->getName().size() ? this->getName().c_str() : "Overlap", name.c_str());CHKERRXX(ierr);
    }
    ierr = PetscSynchronizedPrintf(this->comm(), "[%d]%d partners:\n", this->commRank(), this->numRanks);CHKERRXX(ierr);
    for(index_type r = 0; r < this->numRanks; ++r) {
      ierr = PetscSynchronizedPrintf(this->comm(), "[%d]  %d:", this->commRank(), this->ranks[r]);CHKERRXX(ierr);
      for(index_type p = pointsOffset[r]; p < pointsOffset[r+1]; ++p) {
        ierr = PetscSynchronizedPrintf(this->comm(), "  %d (%d)", this->points[p], this->remotePoints[p]);CHKERRXX(ierr);
      }
      ierr = PetscSynchronizedPrintf(this->comm(), "\n");CHKERRXX(ierr);
    }
    ierr = PetscSynchronizedFlush(this->comm());CHKERRXX(ierr);
  };
};

// Compatibility wrapper for raw pointers
template<typename Value>
class Sequence {
public:
  typedef Value *iterator;
};

// Compatibility wrapper for raw pointers that need precomputation
template<typename Value>
class PointerSequence {
public:
  typedef const Value *iterator;
protected:
  const Value *beginP;
  const Value *endP;
public:
  PointerSequence(const Value *begin, const Value *end) : beginP(begin), endP(end) {};
  ~PointerSequence() {};
  iterator begin() {return beginP;};
  iterator end()   {return endP;};
};

// This structure marches 2 pointers at the same time
  template<typename FirstType, typename SecondType>
class DualSequence {
public:
  typedef FirstType  first_type;
  typedef SecondType second_type;
  class iterator {
  public:
    typedef first_type  value_type;
    typedef second_type color_type;
    typedef int         difference_type;
    typedef value_type* pointer;
    typedef value_type& reference;
  protected:
    const first_type  *points;
    const second_type *remotePoints;
  public:
    iterator(const value_type *points, const color_type *remotePoints) : points(points), remotePoints(remotePoints) {};
    iterator(const iterator& iter) : points(iter.points), remotePoints(iter.remotePoints) {};
    virtual bool             operator==(const iterator& iter) const {return points == iter.points;};
    virtual bool             operator!=(const iterator& iter) const {return points != iter.points;};
    virtual const value_type operator*() const {return *points;};
    virtual const color_type color() const     {return *remotePoints;};
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
protected:
  const first_type  *beginP;
  const second_type *beginRemoteP;
  const first_type  *endP;
  const second_type *endRemoteP;
public:
  DualSequence(const first_type *begin, const second_type *beginRemote, const first_type *end, const second_type *endRemote) : beginP(begin), beginRemoteP(beginRemote), endP(end), endRemoteP(endRemote) {};
  virtual ~DualSequence() {};
  iterator begin() {return iterator(beginP, beginRemoteP);};
  iterator end()   {return iterator(endP,   endRemoteP);};
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
  typedef Sequence<rank_type>                 baseSequence;
  typedef DualSequence<point_type,point_type> coneSequence;
  typedef DualSequence<rank_type,point_type>  supportSequence;
public:
  SendOverlap(MPI_Comm comm, const int debug = 0) : Overlap<Point,Rank>(comm, debug) {};
  ~SendOverlap() {};
  PetscErrorCode addArrow(source_type s, target_type t, color_type c) {
    if (!this->ranks) {
      // Add rank
      bool addRank = true;
      for(typename std::vector<rank_type>::const_iterator r_iter = this->flexRanks.begin(); r_iter != this->flexRanks.end(); ++r_iter) {
        if (*r_iter == t) {addRank = false; break;}
      }
      if (addRank) {
        this->flexRanks.push_back(t);
        this->cachedRanksValid = false;
      }
      // Add point
      //   check uniqueness
      index_type p;
      for(p = 0; p < (index_type) this->flexPoints[t].size(); ++p) {
        if (this->flexPoints[t][p] == s) {
          if ((c >= 0) && (this->flexRemotePoints[t][p] < 0)) {
            this->flexRemotePoints[t][p] = c;
          }
          break;
        }
      }
      if (p >= (index_type) this->flexPoints[t].size()) {
        this->flexPoints[t].push_back(s);
        this->flexRemotePoints[t].push_back(c);
      }
    } else {
      // Could speed this up using a sort after all the insertion
      index_type     r, i;
      PetscErrorCode ierr;

      ierr                  = this->getRankIndex(t, &r);CHKERRQ(ierr);
      i                     = this->insertOffset[r]++;
      this->points[i]       = s;
      this->remotePoints[i] = c;
    }
    return 0;
  };
  void copy(SendOverlap *overlap) {
    for(index_type r = 0; r < this->numRanks; ++r) {
      for(index_type p = this->pointsOffset[r]; p < this->pointsOffset[r+1]; ++p) {
        overlap->addArrow(this->points[p], this->ranks[r], this->remotePoints[p]);
      }
    }
  };
  void setBaseSize(index_type size) {
    this->setNumRanks(size);
  };
  typename baseSequence::iterator baseBegin() {
    assert(!this->numRanks || this->ranks);
    if (!this->ranks && this->flexRanks.size()) {
      this->ensureCachedRanks();
      return this->cachedRanks;
    }
    return this->ranks;
  };
  typename baseSequence::iterator baseEnd() {
    assert(!this->numRanks || this->ranks);
    if (!this->ranks && this->flexRanks.size()) {
      this->ensureCachedRanks();
      return &this->cachedRanks[this->flexRanks.size()];
    }
    return &this->ranks[this->numRanks];
  };
  bool capContains(point_type point) {
    if (!this->numRanks) return false;
    assert(this->pointsOffset);
    // TODO This can be made fast by searching each sorted rank bucket
    for(index_type p = 0; p < this->pointsOffset[this->numRanks]; ++p) {
      if (this->points[p] == point) return true;
    }
    return false;
  };
  void setConeSize(rank_type rank, index_type size) {
    this->setNumPoints(rank, size);
  };
  int getConeSize(rank_type rank) {
    index_type     numPoints;
    PetscErrorCode ierr;

    ierr = this->getNumPointsByRank(rank, &numPoints);
    if (!ierr) {
      return numPoints;
    } else {
      /* Slow query to flexible assembly stuff */
      if (this->flexPoints.find(rank) != this->flexPoints.end()) {
        return this->flexPoints[rank].size();
      }
      /* Missing ranks give 0*/
    }
    return 0;
  };
  int getConeSize(rank_type rank, point_type remotePoint) {
    index_type     r, n = 0;
    PetscErrorCode ierr;

    ierr = this->getRankIndex(rank, &r);
    if (!ierr) {
      for(index_type p = this->pointsOffset[r]; p < this->pointsOffset[r+1]; ++p) {
        if (remotePoint == this->remotePoints[p]) ++n;
      }
      return n;
    } else {
      /* Slow query to flexible assembly stuff */
      if (this->flexRemotePoints.find(rank) != this->flexRemotePoints.end()) {
        index_type n = 0;

        for(typename std::vector<point_type>::const_iterator p_iter = this->flexRemotePoints[rank].begin(); p_iter != this->flexRemotePoints[rank].end(); ++p_iter) {
          if (remotePoint == *p_iter) ++n;
        }
        return n;
      }
      /* Missing ranks give 0 */
    }
    return 0;
  };
  typename coneSequence::iterator coneBegin(rank_type rank) {
    index_type     r;
    PetscErrorCode ierr;

    ierr = this->getRankIndex(rank, &r);
    if (!ierr) {
      assert(this->pointsOffset);
      assert(this->points);
      return typename coneSequence::iterator(&this->points[this->pointsOffset[r]], &this->remotePoints[this->pointsOffset[r]]);
    } else {
      /* Slow query to flexible assembly stuff */
      if (this->flexRemotePoints.find(rank) != this->flexRemotePoints.end()) {
        return typename coneSequence::iterator(&(this->flexPoints[rank][0]), &(this->flexRemotePoints[rank][0]));
      }
      throw ALE::Exception("Invalid rank was not contained in this overlap");
    }
  };
  typename coneSequence::iterator coneBegin(rank_type rank, point_type remotePoint) {
    index_type     r, p;
    PetscErrorCode ierr;

    ierr = this->getRankIndex(rank, &r);
    if (!ierr) {
      assert(this->pointsOffset);
      assert(this->points);
      for(p = this->pointsOffset[r]; p < this->pointsOffset[r+1]; ++p) {
        if (remotePoint == this->remotePoints[p]) break;
      }
      return typename coneSequence::iterator(&this->points[p], &this->remotePoints[p]);
    } else {
      /* Slow query to flexible assembly stuff */
      if (this->flexRemotePoints.find(rank) != this->flexRemotePoints.end()) {
        index_type p = 0;

        for(typename std::vector<point_type>::const_iterator p_iter = this->flexRemotePoints[rank].begin(); p_iter != this->flexRemotePoints[rank].end(); ++p_iter, ++p) {
          if (remotePoint == *p_iter) break;
        }
        return typename coneSequence::iterator(&(this->flexPoints[rank][p]), &(this->flexRemotePoints[rank][p]));
      }
      throw ALE::Exception("Invalid rank was not contained in this overlap");
    }
  };
  typename coneSequence::iterator coneEnd(rank_type rank) {
    index_type     r;
    PetscErrorCode ierr;

    ierr = this->getRankIndex(rank, &r);
    if (!ierr) {
      assert(this->pointsOffset);
      assert(this->points);
      return typename coneSequence::iterator(&this->points[this->pointsOffset[r+1]], &this->remotePoints[this->pointsOffset[r+1]]);
    } else {
      /* Slow query to flexible assembly stuff */
      if (this->flexRemotePoints.find(rank) != this->flexRemotePoints.end()) {
        return typename coneSequence::iterator(&(this->flexPoints[rank][this->flexPoints[rank].size()-1])+1, &(this->flexRemotePoints[rank][this->flexRemotePoints[rank].size()-1])+1);
      }
      throw ALE::Exception("Invalid rank was not contained in this overlap");
    }
  };
  supportSequence support(point_type point) {
    index_type        numPointRanks;
    const rank_type  *pointRanks;
    const point_type *pointRemotePoints;

    this->getRanks(point, &numPointRanks, &pointRanks, &pointRemotePoints);
    return supportSequence(pointRanks, pointRemotePoints, &pointRanks[numPointRanks], &pointRemotePoints[numPointRanks]);
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
  typedef Sequence<rank_type>                 capSequence;
  typedef DualSequence<point_type,point_type> supportSequence;
  typedef DualSequence<rank_type,point_type>  coneSequence;
public:
  RecvOverlap(MPI_Comm comm, const int debug = 0) : Overlap<Point,Rank>(comm, debug) {};
  ~RecvOverlap() {};
  PetscErrorCode addArrow(source_type s, target_type t, color_type c) {
    if (!this->ranks) {
      // Add rank
      bool addRank = true;
      for(typename std::vector<rank_type>::const_iterator r_iter = this->flexRanks.begin(); r_iter != this->flexRanks.end(); ++r_iter) {
        if (*r_iter == s) {addRank = false; break;}
      }
      if (addRank) {
        this->flexRanks.push_back(s);
        this->cachedRanksValid = false;
      }
      // Add point
      //   check uniqueness
      index_type p;
      for(p = 0; p < (index_type) this->flexPoints[s].size(); ++p) {
        if (this->flexPoints[s][p] == t) {
          if ((c >= 0) && (this->flexRemotePoints[s][p] < 0)) {
            this->flexRemotePoints[s][p] = c;
          }
          break;
        }
      }
      if (p >= (index_type) this->flexPoints[s].size()) {
        this->flexPoints[s].push_back(t);
        this->flexRemotePoints[s].push_back(c);
      }
    } else {
      index_type     r, i;
      PetscErrorCode ierr;

      ierr = this->getRankIndex(s, &r);CHKERRQ(ierr);
      i                     = this->insertOffset[r]++;
      this->points[i]       = t;
      this->remotePoints[i] = c;
    }
    return 0;
  };
  void copy(RecvOverlap *overlap) {
    for(index_type r = 0; r < this->numRanks; ++r) {
      for(index_type p = this->pointsOffset[r]; p < this->pointsOffset[r+1]; ++p) {
        overlap->addArrow(this->ranks[r], this->points[p], this->remotePoints[p]);
      }
    }
  };
  typename capSequence::iterator capBegin() {
    assert(!this->numRanks || this->ranks);
    if (!this->ranks && this->flexRanks.size()) {
      this->ensureCachedRanks();
      return this->cachedRanks;
    }
    return this->ranks;
  };
  typename capSequence::iterator capEnd() {
    assert(!this->numRanks || this->ranks);
    if (!this->ranks && this->flexRanks.size()) {
      this->ensureCachedRanks();
      return &this->cachedRanks[this->flexRanks.size()];
    }
    return &this->ranks[this->numRanks];
  };
  void setCapSize(index_type size) {
    this->setNumRanks(size);
  };
  void setSupportSize(rank_type rank, index_type size) {
    this->setNumPoints(rank, size);
  };
  int getSupportSize(rank_type rank) {
    index_type     numPoints;
    PetscErrorCode ierr;

    ierr = this->getNumPointsByRank(rank, &numPoints);
    if (!ierr) {
      return numPoints;
    } else {
      /* Slow query to flexible assembly stuff */
      if (this->flexPoints.find(rank) != this->flexPoints.end()) {
        return this->flexPoints[rank].size();
      }
      /* Missing ranks give 0 */
    }
    return 0;
  };
  int getSupportSize(rank_type rank, point_type remotePoint) {
    index_type     r, n = 0;
    PetscErrorCode ierr;

    ierr = this->getRankIndex(rank, &r);
    if (!ierr) {
      for(index_type p = this->pointsOffset[r]; p < this->pointsOffset[r+1]; ++p) {
        if (remotePoint == this->remotePoints[p]) ++n;
      }
      return n;
    } else {
      /* Slow query to flexible assembly stuff */
      if (this->flexRemotePoints.find(rank) != this->flexRemotePoints.end()) {
        index_type n = 0;

        for(typename std::vector<point_type>::const_iterator p_iter = this->flexRemotePoints[rank].begin(); p_iter != this->flexRemotePoints[rank].end(); ++p_iter) {
          if (remotePoint == *p_iter) ++n;
        }
        return n;
      }
      /* Missing ranks give 0 */
    }
    return 0;
  };
  typename supportSequence::iterator supportBegin(rank_type rank) {
    index_type     r;
    PetscErrorCode ierr;

    ierr = this->getRankIndex(rank, &r);
    if (!ierr) {
      assert(this->pointsOffset);
      assert(this->points);
      return typename supportSequence::iterator(&this->points[this->pointsOffset[r]], &this->remotePoints[this->pointsOffset[r]]);
    } else {
      /* Slow query to flexible assembly stuff */
      if (this->flexRemotePoints.find(rank) != this->flexRemotePoints.end()) {
        return typename supportSequence::iterator(&(this->flexPoints[rank][0]), &(this->flexRemotePoints[rank][0]));
      }
      throw ALE::Exception("Invalid rank was not contained in this overlap");
    }
  };
  typename supportSequence::iterator supportBegin(rank_type rank, point_type remotePoint) {
    index_type     r;
    PetscErrorCode ierr;

    ierr = this->getRankIndex(rank, &r);
    if (!ierr) {
      index_type       p;

      assert(this->pointsOffset);
      assert(this->points);
      for(p = this->pointsOffset[r]; p < this->pointsOffset[r+1]; ++p) {
        if (remotePoint == this->remotePoints[p]) break;
      }
      return typename supportSequence::iterator(&this->points[p], &this->remotePoints[p]);
    } else {
      /* Slow query to flexible assembly stuff */
      if (this->flexRemotePoints.find(rank) != this->flexRemotePoints.end()) {
        index_type p = 0;

        for(typename std::vector<point_type>::const_iterator p_iter = this->flexRemotePoints[rank].begin(); p_iter != this->flexRemotePoints[rank].end(); ++p_iter, ++p) {
          if (remotePoint == *p_iter) break;
        }
        return typename supportSequence::iterator(&(this->flexPoints[rank][p]), &(this->flexRemotePoints[rank][p]));
      }
      throw ALE::Exception("Invalid rank was not contained in this overlap");
    }
  };
  typename supportSequence::iterator supportEnd(rank_type rank) {
    index_type     r;
    PetscErrorCode ierr;

    ierr = this->getRankIndex(rank, &r);
    if (!ierr) {
      assert(this->pointsOffset);
      assert(this->points);
      return typename supportSequence::iterator(&this->points[this->pointsOffset[r+1]], &this->remotePoints[this->pointsOffset[r+1]]);
    } else {
      /* Slow query to flexible assembly stuff */
      if (this->flexRemotePoints.find(rank) != this->flexRemotePoints.end()) {
        return typename supportSequence::iterator(&(this->flexPoints[rank][this->flexPoints[rank].size()-1])+1, &(this->flexRemotePoints[rank][this->flexRemotePoints[rank].size()-1])+1);
      }
      throw ALE::Exception("Invalid rank was not contained in this overlap");
    }
  };
  coneSequence cone(point_type point) {
    index_type        numPointRanks;
    const rank_type  *pointRanks;
    const point_type *pointRemotePoints;

    this->getRanks(point, &numPointRanks, &pointRanks, &pointRemotePoints);
    return coneSequence(pointRanks, pointRemotePoints, &pointRanks[numPointRanks], &pointRemotePoints[numPointRanks]);
  };
};

class OverlapSerializer {
public:
  template<typename Overlap>
  static void writeOverlap(std::ofstream& fs, Overlap& overlap) {
    throw PETSc::Exception("Not implemented (should come from Sifter.hh:SifterSerializer::writeSifter()");
  }
  template<typename Overlap>
  static void loadOverlap(std::ifstream& fs, Overlap& overlap) {
    throw PETSc::Exception("Not implemented (should come from Sifter.hh:SifterSerializer::loadSifter()");
  }
};
}

#endif /* included_PETSc_Overlap_hh */
