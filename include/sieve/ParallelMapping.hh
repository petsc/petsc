#ifndef included_ALE_ParallelMapping_hh
#define included_ALE_ParallelMapping_hh

#ifndef  included_ALE_BasicCommunication_hh
#include <sieve/BasicCommunication.hh>
#endif

#ifndef  included_ALE_IField_hh
#include <sieve/IField.hh>
#endif

#ifndef  included_ALE_Sections_hh
#include <sieve/Sections.hh>
#endif

#include <functional>
#include <valarray>

namespace ALE {
  template<class _Tp>
  struct Identity : public std::unary_function<_Tp,_Tp>
  {
    _Tp& operator()(_Tp& x) const {return x;}
    const _Tp& operator()(const _Tp& x) const {return x;}
  };

  template<class _Tp>
  struct IsEqual : public std::unary_function<_Tp, bool>, public std::binary_function<_Tp, _Tp, bool>
  {
    const _Tp& x;
    IsEqual(const _Tp& x) : x(x) {};
    bool operator()(_Tp& y) const {return x == y;}
    bool operator()(const _Tp& y) const {return x == y;}
    bool operator()(_Tp& y, _Tp& dummy) const {return x == y;}
    bool operator()(const _Tp& y, const _Tp& dummy) const {return x == y;}
  };

  // Creates new global point names and renames local points globally
  template<typename Point>
  class PointFactory : ALE::ParallelObject {
  public:
    typedef Point                           point_type;
    typedef std::map<point_type,point_type> renumbering_type;
    typedef std::map<int,std::map<point_type,point_type> > remote_renumbering_type;
  protected:
    point_type       originalMax;
    point_type       currentMax;
    renumbering_type renumbering;
    renumbering_type invRenumbering;
    remote_renumbering_type remoteRenumbering;
  protected:
    PointFactory(MPI_Comm comm, const int debug = 0) : ALE::ParallelObject(comm, debug), originalMax(-1) {};
  public:
    ~PointFactory() {};
  public:
    static PointFactory& singleton(MPI_Comm comm, const point_type& maxPoint, const int debug = 0, bool cleanup = false) {
      static PointFactory *_singleton = NULL;

      if (cleanup) {
        if (debug) {std::cout << "Destroying PointFactory" << std::endl;}
        if (_singleton) {delete _singleton;}
        _singleton = NULL;
      } else if (_singleton == NULL) {
        if (debug) {std::cout << "Creating new PointFactory" << std::endl;}
        _singleton  = new PointFactory(comm, debug);
        _singleton->setMax(maxPoint);
      }
      return *_singleton;
    };
    void setMax(const point_type& maxPoint) {
      PetscErrorCode ierr = MPI_Allreduce((void *) &maxPoint, &this->originalMax, 1, MPI_INT, MPI_MAX, this->comm());CHKERRXX(ierr);
      ++this->originalMax;
      this->currentMax = this->originalMax;
    };
    void clear() {
      this->currentMax = this->originalMax;
      this->renumbering.clear();
      this->invRenumbering.clear();
    };
  public:
    template<typename Iterator>
    void renumberPoints(const Iterator& begin, const Iterator& end) {
      renumberPoints(begin, end, Identity<typename Iterator::value_type>());
    }
    template<typename Iterator, typename KeyExtractor>
    void renumberPoints(const Iterator& begin, const Iterator& end, const KeyExtractor& ex) {
      int numPoints = 0, numGlobalPoints, firstPoint;

      for(Iterator p_iter = begin; p_iter != end; ++p_iter) ++numPoints;
      MPI_Allreduce(&numPoints, &numGlobalPoints, 1, MPI_INT, MPI_SUM, this->comm());
      MPI_Scan(&numPoints, &firstPoint, 1, MPI_INT, MPI_SUM, this->comm());
      firstPoint += this->currentMax - numPoints;
      this->currentMax += numGlobalPoints;
      for(Iterator p_iter = begin; p_iter != end; ++p_iter, ++firstPoint) {
        if (this->debug()) {std::cout << "["<<this->commRank()<<"]: New point " << ex(*p_iter) << " --> " << firstPoint << std::endl;}
        this->renumbering[firstPoint]     = ex(*p_iter);
        this->invRenumbering[ex(*p_iter)] = firstPoint;
      }
    }
  public:
    void constructRemoteRenumbering() {
      const int localSize   = this->renumbering.size();
      int      *remoteSizes = new int[this->commSize()];
      int      *localMap    = new int[localSize*2];
      int      *recvCounts  = new int[this->commSize()];
      int      *displs      = new int[this->commSize()];

      // Populate arrays
      int r = 0;
      for(typename renumbering_type::const_iterator r_iter = renumbering.begin(); r_iter != renumbering.end(); ++r_iter, ++r) {
        localMap[r*2+0] = r_iter->first;
        localMap[r*2+1] = r_iter->second;
      }
      // Communicate renumbering sizes
      MPI_Allgather((void*) &localSize, 1, MPI_INT, remoteSizes, 1, MPI_INT, this->comm());
      for(int p = 0; p < this->commSize(); ++p) {
        recvCounts[p] = remoteSizes[p]*2;
        if (p == 0) {
          displs[p]   = 0;
        } else {
          displs[p]   = displs[p-1] + recvCounts[p-1];
        }
      }
      int *remoteMaps = new int[displs[this->commSize()-1]+recvCounts[this->commSize()-1]];
      // Communicate renumberings
      MPI_Allgatherv(localMap, localSize*2, MPI_INT, remoteMaps, recvCounts, displs, MPI_INT, this->comm());
      // Populate maps
      for(int p = 0; p < this->commSize(); ++p) {
        if (p == this->commRank()) continue;
        int offset = displs[p];

        for(int r = 0; r < remoteSizes[p]; ++r) {
          this->remoteRenumbering[p][remoteMaps[r*2+0+offset]] = remoteMaps[r*2+1+offset];
          if (this->debug()) {std::cout << "["<<this->commRank()<<"]: Remote renumbering["<<p<<"] " << remoteMaps[r*2+0+offset] << " --> " << remoteMaps[r*2+1+offset] << std::endl;}
        }
      }
      // Cleanup
      delete [] recvCounts;
      delete [] displs;
      delete [] localMap;
      delete [] remoteMaps;
      delete [] remoteSizes;
    };
  public:
    // global point --> local point
    renumbering_type& getRenumbering() {
      return this->renumbering;
    };
    // local point --> global point
    renumbering_type& getInvRenumbering() {
      return this->invRenumbering;
    };
    // rank --> global point --> local point
    remote_renumbering_type& getRemoteRenumbering() {
      return this->remoteRenumbering;
    };
  };

  template<typename Alloc_ = malloc_allocator<int> >
  class OverlapBuilder {
  public:
    typedef Alloc_ alloc_type;
  protected:
    template<typename T>
    struct lessPair: public std::binary_function<std::pair<T,T>, std::pair<T,T>, bool> {
      bool operator()(const std::pair<T,T>& x, const std::pair<T,T>& y) const {
        return x.first < y.first;
      }
    };
    template<typename T>
    struct mergePair: public std::binary_function<std::pair<T,T>, std::pair<T,T>, bool> {
      std::pair<T,std::pair<T,T> > operator()(const std::pair<T,T>& x, const std::pair<T,T>& y) const {
        return std::pair<T,std::pair<T,T> >(x.first, std::pair<T,T>(x.second, y.second));
      }
    };
    template<typename _InputIterator1, typename _InputIterator2, typename _OutputIterator, typename _Compare, typename _Merge>
    static _OutputIterator set_intersection_merge(_InputIterator1 __first1, _InputIterator1 __last1,
                                           _InputIterator2 __first2, _InputIterator2 __last2,
                                           _OutputIterator __result, _Compare __comp, _Merge __merge)
    {
      while(__first1 != __last1 && __first2 != __last2) {
        if (__comp(*__first1, *__first2))
          ++__first1;
        else if (__comp(*__first2, *__first1))
          ++__first2;
        else
        {
          *__result = __merge(*__first1, *__first2);
          ++__first1;
          ++__first2;
          ++__result;
        }
      }
      return __result;
    }
  public:
    template<typename Sequence, typename Renumbering, typename SendOverlap, typename RecvOverlap>
    static void constructOverlap(const Sequence& points, Renumbering& renumbering, const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap) {
      typedef typename SendOverlap::source_type point_type;
      typedef std::pair<point_type,point_type>  pointPair;
      typedef std::pair<point_type,pointPair>   pointTriple;
      alloc_type allocator;
      typename alloc_type::template rebind<point_type>::other point_allocator;
      typename alloc_type::template rebind<pointPair>::other  pointPair_allocator;
      const MPI_Comm comm     = sendOverlap->comm();
      const int      commSize = sendOverlap->commSize();
      const int      commRank = sendOverlap->commRank();
      point_type    *sendBuf  = point_allocator.allocate(points.size()*2);
      for(size_t i = 0; i < points.size()*2; ++i) {point_allocator.construct(sendBuf+i, point_type());}
      int            size     = 0;
      const int      debug    = sendOverlap->debug();
      for(typename Sequence::const_iterator l_iter = points.begin(); l_iter != points.end(); ++l_iter) {
        if (debug) {std::cout << "["<<commRank<<"]Send point["<<size<<"]: " << *l_iter << " " << renumbering[*l_iter] << std::endl;}
        sendBuf[size++] = *l_iter;
        sendBuf[size++] = renumbering[*l_iter];
      }
      int *sizes = allocator.allocate(commSize*3+2); // [size]   The number of points coming from each process
      for(int i = 0; i < commSize*3+2; ++i) {allocator.construct(sizes+i, 0);}
      int *offsets = sizes+commSize;                 // [size+1] Prefix sums for sizes
      int *oldOffs = offsets+commSize+1;             // [size+1] Temporary storage
      pointPair  *remotePoints = NULL;               // The points from each process
      int        *remoteRanks  = NULL;               // The rank and number of overlap points of each process that overlaps another
      int         numRemotePoints = 0;
      int         numRemoteRanks  = 0;

      // Change to Allgather() for the correct binning algorithm
      MPI_Gather(&size, 1, MPI_INT, sizes, 1, MPI_INT, 0, comm);
      if (commRank == 0) {
        offsets[0] = 0;
        for(int p = 1; p <= commSize; p++) {
          offsets[p] = offsets[p-1] + sizes[p-1];
        }
        numRemotePoints = offsets[commSize];
        remotePoints    = pointPair_allocator.allocate(numRemotePoints/2);
        for(int i = 0; i < numRemotePoints/2; ++i) {pointPair_allocator.construct(remotePoints+i, pointPair());}
      }
      MPI_Gatherv(sendBuf, size, MPI_INT, remotePoints, sizes, offsets, MPI_INT, 0, comm);
      for(size_t i = 0; i < points.size(); ++i) {point_allocator.destroy(sendBuf+i);}
      point_allocator.deallocate(sendBuf, points.size());
      std::map<int, std::map<int, std::set<pointTriple> > > overlapInfo; // Maps (p,q) to their set of overlap points

      if (commRank == 0) {
        for(int p = 0; p <= commSize; p++) {
          offsets[p] /= 2;
        }
        for(int p = 0; p < commSize; p++) {
          std::sort(&remotePoints[offsets[p]], &remotePoints[offsets[p+1]], lessPair<point_type>());
        }
        for(int p = 0; p <= commSize; p++) {
          oldOffs[p] = offsets[p];
        }
        for(int p = 0; p < commSize; p++) {
          for(int q = 0; q < commSize; q++) {
            if (p == q) continue;
            set_intersection_merge(&remotePoints[oldOffs[p]], &remotePoints[oldOffs[p+1]],
                                   &remotePoints[oldOffs[q]], &remotePoints[oldOffs[q+1]],
                                   std::insert_iterator<std::set<pointTriple> >(overlapInfo[p][q], overlapInfo[p][q].begin()),
                                   lessPair<point_type>(), mergePair<point_type>());
          }
          sizes[p]     = overlapInfo[p].size()*2;
          offsets[p+1] = offsets[p] + sizes[p];
        }
        numRemoteRanks = offsets[commSize];
        remoteRanks    = allocator.allocate(numRemoteRanks);
        for(int i = 0; i < numRemoteRanks; ++i) {allocator.construct(remoteRanks+i, 0);}
        int     k = 0;
        for(int p = 0; p < commSize; p++) {
          for(typename std::map<int, std::set<pointTriple> >::iterator r_iter = overlapInfo[p].begin(); r_iter != overlapInfo[p].end(); ++r_iter) {
            remoteRanks[k*2]   = r_iter->first;
            remoteRanks[k*2+1] = r_iter->second.size();
            k++;
          }
        }
      }
      int numOverlaps;                          // The number of processes overlapping this process
      MPI_Scatter(sizes, 1, MPI_INT, &numOverlaps, 1, MPI_INT, 0, comm);
      int *overlapRanks = allocator.allocate(numOverlaps); // The rank and overlap size for each overlapping process
      for(int i = 0; i < numOverlaps; ++i) {allocator.construct(overlapRanks+i, 0);}
      MPI_Scatterv(remoteRanks, sizes, offsets, MPI_INT, overlapRanks, numOverlaps, MPI_INT, 0, comm);
      point_type *sendPoints    = NULL;         // The points to send to each process
      int         numSendPoints = 0;
      if (commRank == 0) {
        for(int p = 0, k = 0; p < commSize; p++) {
          sizes[p] = 0;
          for(int r = 0; r < (int) overlapInfo[p].size(); r++) {
            sizes[p] += remoteRanks[k*2+1]*2;
            k++;
          }
          offsets[p+1] = offsets[p] + sizes[p];
        }
        numSendPoints = offsets[commSize];
        sendPoints    = point_allocator.allocate(numSendPoints*2);
        for(int i = 0; i < numSendPoints*2; ++i) {point_allocator.construct(sendPoints+i, point_type());}
        for(int p = 0, k = 0; p < commSize; p++) {
          for(typename std::map<int, std::set<pointTriple> >::const_iterator r_iter = overlapInfo[p].begin(); r_iter != overlapInfo[p].end(); ++r_iter) {
            int rank = r_iter->first;
            for(typename std::set<pointTriple>::const_iterator p_iter = (overlapInfo[p][rank]).begin(); p_iter != (overlapInfo[p][rank]).end(); ++p_iter) {
              sendPoints[k++] = p_iter->first;
              sendPoints[k++] = p_iter->second.second;
              if (debug) {std::cout << "["<<commRank<<"]Sending points " << p_iter->first << " " << p_iter->second.second << " to rank " << rank << std::endl;}
            }
          }
        }
      }
      int numOverlapPoints = 0;
      for(int r = 0; r < numOverlaps/2; r++) {
        numOverlapPoints += overlapRanks[r*2+1];
      }
      point_type *overlapPoints = point_allocator.allocate(numOverlapPoints*2);
      for(int i = 0; i < numOverlapPoints*2; ++i) {point_allocator.construct(overlapPoints+i, point_type());}
      MPI_Scatterv(sendPoints, sizes, offsets, MPI_INT, overlapPoints, numOverlapPoints*2, MPI_INT, 0, comm);

      for(int r = 0, k = 0; r < numOverlaps/2; r++) {
        int rank = overlapRanks[r*2];

        for(int p = 0; p < overlapRanks[r*2+1]; p++) {
          point_type point       = overlapPoints[k++];
          point_type remotePoint = overlapPoints[k++];

          if (debug) {std::cout << "["<<commRank<<"]Matched up remote point " << remotePoint << "("<<point<<") to local " << renumbering[point] << std::endl;}
          sendOverlap->addArrow(renumbering[point], rank, remotePoint);
          recvOverlap->addArrow(rank, renumbering[point], remotePoint);
        }
      }

      for(int i = 0; i < numOverlapPoints; ++i) {point_allocator.destroy(overlapPoints+i);}
      point_allocator.deallocate(overlapPoints, numOverlapPoints);
      for(int i = 0; i < numOverlaps; ++i) {allocator.destroy(overlapRanks+i);}
      allocator.deallocate(overlapRanks, numOverlaps);
      for(int i = 0; i < commSize*3+2; ++i) {allocator.destroy(sizes+i);}
      allocator.deallocate(sizes, commSize*3+2);
      if (commRank == 0) {
        for(int i = 0; i < numRemoteRanks; ++i) {allocator.destroy(remoteRanks+i);}
        allocator.deallocate(remoteRanks, numRemoteRanks);
        for(int i = 0; i < numRemotePoints; ++i) {pointPair_allocator.destroy(remotePoints+i);}
        pointPair_allocator.deallocate(remotePoints, numRemotePoints);
        for(int i = 0; i < numSendPoints; ++i) {point_allocator.destroy(sendPoints+i);}
        point_allocator.deallocate(sendPoints, numSendPoints);
      }
      // TODO: Rewrite above to use optimized construction
      sendOverlap->assemble();
      recvOverlap->assemble();
      sendOverlap->assemblePoints();
      recvOverlap->assemblePoints();
    }
  };
  namespace Pullback {
    class SimpleCopy {
    public:
      // Copy the overlap section to the related processes
      //   This version is for Constant sections, meaning the same, single value over all points
      template<typename SendOverlap, typename RecvOverlap, typename SendSection, typename RecvSection>
      static void copyConstant(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<SendSection>& sendSection, const Obj<RecvSection>& recvSection) {
        MPIMover<char>                             pMover(sendSection->comm(), sendSection->debug());
        MPIMover<typename SendSection::value_type> vMover(sendSection->comm(), sendSection->debug());
        std::map<int, char *>                      sendPoints;
        std::map<int, char *>                      recvPoints;
        typename SendSection::alloc_type::template rebind<char>::other sendAllocator;
        typename RecvSection::alloc_type::template rebind<char>::other recvAllocator;

        const typename SendOverlap::baseSequence::iterator sBegin  = sendOverlap->baseBegin();
        const typename SendOverlap::baseSequence::iterator sEnd    = sendOverlap->baseEnd();
        const typename SendSection::value_type            *sValues = sendSection->restrictSpace();

        for(typename SendOverlap::baseSequence::iterator r_iter = sBegin; r_iter != sEnd; ++r_iter) {
          const int                                          pSize  = sendOverlap->getConeSize(*r_iter);
          const typename SendOverlap::coneSequence::iterator pBegin = sendOverlap->coneBegin(*r_iter);
          const typename SendOverlap::coneSequence::iterator pEnd   = sendOverlap->coneEnd(*r_iter);
          char                                              *v      = sendAllocator.allocate(pSize);
          int                                                k      = 0;

          for(int i = 0; i < pSize; ++i) {sendAllocator.construct(v+i, 0);}
          for(typename SendOverlap::coneSequence::iterator p_iter = pBegin; p_iter != pEnd; ++p_iter, ++k) {
            v[k] = (char) sendSection->hasPoint(*p_iter);
          }
          sendPoints[*r_iter] = v;
          pMover.send(*r_iter, pSize, sendPoints[*r_iter]);
          vMover.send(*r_iter, 2, sValues);
        }
        const typename RecvOverlap::capSequence::iterator rBegin  = recvOverlap->capBegin();
        const typename RecvOverlap::capSequence::iterator rEnd    = recvOverlap->capEnd();
        const typename RecvSection::value_type           *rValues = recvSection->restrictSpace();

        for(typename RecvOverlap::capSequence::iterator r_iter = rBegin; r_iter != rEnd; ++r_iter) {
          const int pSize = recvOverlap->getSupportSize(*r_iter);
          char     *v     = recvAllocator.allocate(pSize);

          for(int i = 0; i < pSize; ++i) {recvAllocator.construct(v+i, 0);}
          recvPoints[*r_iter] = v;
          pMover.recv(*r_iter, pSize, recvPoints[*r_iter]);
          vMover.recv(*r_iter, 2, rValues);
        }
        pMover.start();
        pMover.end();
        vMover.start();
        vMover.end();
        for(typename RecvOverlap::capSequence::iterator r_iter = rBegin; r_iter != rEnd; ++r_iter) {
          const typename RecvOverlap::supportSequence::iterator pBegin = recvOverlap->supportBegin(*r_iter);
          const typename RecvOverlap::supportSequence::iterator pEnd   = recvOverlap->supportEnd(*r_iter);
          const char                                           *v      = recvPoints[*r_iter];
          int                                                   k      = 0;

          for(typename RecvOverlap::supportSequence::iterator s_iter = pBegin; s_iter != pEnd; ++s_iter, ++k) {
            if (v[k]) {recvSection->addPoint(typename RecvSection::point_type(*r_iter, s_iter.color()));}
          }
        }
        for(typename SendOverlap::baseSequence::iterator r_iter = sBegin; r_iter != sEnd; ++r_iter) {
          sendAllocator.deallocate(sendPoints[*r_iter], sendOverlap->getConeSize(*r_iter));
        }
        for(typename RecvOverlap::capSequence::iterator r_iter = rBegin; r_iter != rEnd; ++r_iter) {
          recvAllocator.deallocate(recvPoints[*r_iter], recvOverlap->getSupportSize(*r_iter));
        }
      }
      // Specialize to ArrowSection
      template<typename SendOverlap, typename RecvOverlap, typename SendSection, typename RecvSection>
      static void copyConstantArrow(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<SendSection>& sendSection, const Obj<RecvSection>& recvSection) {
        MPIMover<char>                             pMover(sendSection->comm(), sendSection->debug());
        MPIMover<typename SendSection::value_type> vMover(sendSection->comm(), sendSection->debug());
        std::map<int, char *>                      sendPoints;
        std::map<int, char *>                      recvPoints;
        typename SendSection::alloc_type::template rebind<char>::other sendAllocator;
        typename RecvSection::alloc_type::template rebind<char>::other recvAllocator;

        const Obj<typename SendOverlap::traits::baseSequence>      sRanks  = sendOverlap->base();
        const typename SendOverlap::traits::baseSequence::iterator sEnd    = sRanks->end();
        const typename SendSection::value_type                    *sValues = sendSection->restrictSpace();

        for(typename SendOverlap::traits::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sEnd; ++r_iter) {
          const Obj<typename SendOverlap::coneSequence>&     points = sendOverlap->cone(*r_iter);
          const int                                          pSize  = sendOverlap->getConeSize(*r_iter);
          const typename SendOverlap::coneSequence::iterator pBegin = points->begin();
          const typename SendOverlap::coneSequence::iterator pEnd   = points->end();
          char                                              *v      = sendAllocator.allocate(pSize*pSize);
          int                                                k      = 0;

          for(size_t i = 0; i < pSize*pSize; ++i) {sendAllocator.construct(v+i, 0);}
          for(typename SendOverlap::coneSequence::iterator p_iter = pBegin; p_iter != pEnd; ++p_iter) {
            for(typename SendOverlap::coneSequence::iterator q_iter = pBegin; q_iter != pEnd; ++q_iter, ++k) {
              v[k] = (char) sendSection->hasPoint(typename SendSection::point_type(*p_iter,*q_iter));
            }
          }
          sendPoints[*r_iter] = v;
          pMover.send(*r_iter, pSize*pSize, sendPoints[*r_iter]);
          vMover.send(*r_iter, 2, sValues);
        }
        const Obj<typename RecvOverlap::traits::capSequence>      rRanks  = recvOverlap->cap();
        const typename RecvOverlap::traits::capSequence::iterator rEnd    = rRanks->end();
        const typename RecvSection::value_type                   *rValues = recvSection->restrictSpace();

        for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rEnd; ++r_iter) {
          const Obj<typename RecvOverlap::traits::supportSequence>& points = recvOverlap->support(*r_iter);
          const int                                                 pSize  = recvOverlap->getSupportSize(*r_iter);
          char                                                     *v      = recvAllocator.allocate(pSize*pSize);

          for(size_t i = 0; i < pSize*pSize; ++i) {recvAllocator.construct(v+i, 0);}
          recvPoints[*r_iter] = v;
          pMover.recv(*r_iter, pSize*pSize, recvPoints[*r_iter]);
          vMover.recv(*r_iter, 2, rValues);
        }
        pMover.start();
        pMover.end();
        vMover.start();
        vMover.end();
        for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rEnd; ++r_iter) {
          const Obj<typename RecvOverlap::traits::supportSequence>&     points = recvOverlap->support(*r_iter);
          const typename RecvOverlap::traits::supportSequence::iterator pBegin = points->begin();
          const typename RecvOverlap::traits::supportSequence::iterator pEnd   = points->end();
          const char                                                   *v      = recvPoints[*r_iter];
          int                                                           k      = 0;

          for(typename RecvOverlap::traits::supportSequence::iterator s_iter = pBegin; s_iter != pEnd; ++s_iter) {
            for(typename RecvOverlap::traits::supportSequence::iterator t_iter = pBegin; t_iter != pEnd; ++t_iter, ++k) {
              if (v[k]) {recvSection->addPoint(typename RecvSection::point_type(s_iter.color(),t_iter.color()));}
            }
          }
        }
        for(typename SendOverlap::traits::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sEnd; ++r_iter) {
          sendAllocator.deallocate(sendPoints[*r_iter], sendOverlap->getConeSize(*r_iter)*sendOverlap->getConeSize(*r_iter));
        }
        for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rEnd; ++r_iter) {
          recvAllocator.deallocate(recvPoints[*r_iter], recvOverlap->getSupportSize(*r_iter)*recvOverlap->getSupportSize(*r_iter));
        }
      }
      // Copy the overlap section to the related processes
      //   This version is for IConstant sections, meaning the same, single value over all points
      template<typename SendOverlap, typename RecvOverlap, typename SendSection, typename RecvSection>
      static void copyIConstant(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<SendSection>& sendSection, const Obj<RecvSection>& recvSection) {
        MPIMover<typename SendSection::point_type> pMover(sendSection->comm(), sendSection->debug());
        MPIMover<typename SendSection::value_type> vMover(sendSection->comm(), sendSection->debug());
        std::map<int, typename SendSection::point_type *> sendPoints;
        std::map<int, typename SendSection::point_type *> recvPoints;
        typename SendSection::alloc_type::template rebind<typename SendSection::point_type>::other sendAllocator;
        typename RecvSection::alloc_type::template rebind<typename SendSection::point_type>::other recvAllocator;

        const Obj<typename SendOverlap::baseSequence>      sRanks  = sendOverlap->base();
        const typename SendOverlap::baseSequence::iterator sEnd    = sRanks->end();
        const typename SendSection::value_type            *sValues = sendSection->restrictSpace();

        for(typename SendOverlap::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sEnd; ++r_iter) {
          typename SendSection::point_type *v = sendAllocator.allocate(2);

          for(size_t i = 0; i < 2; ++i) {sendAllocator.construct(v+i, 0);}
          v[0] = sendSection->getChart().min();
          v[1] = sendSection->getChart().max();
          sendPoints[*r_iter] = v;
          pMover.send(*r_iter, 2, sendPoints[*r_iter]);
          vMover.send(*r_iter, 2, sValues);
          ///std::cout << "["<<sendOverlap->commRank()<<"]Sending chart (" << v[0] << ", " << v[1] << ") with values (" << sValues[0] << ", " << sValues[1] << ") to process " << *r_iter << std::endl;
        }
        const Obj<typename RecvOverlap::capSequence>      rRanks  = recvOverlap->cap();
        const typename RecvOverlap::capSequence::iterator rEnd    = rRanks->end();
        const typename RecvSection::value_type           *rValues = recvSection->restrictSpace();

        for(typename RecvOverlap::capSequence::iterator r_iter = rRanks->begin(); r_iter != rEnd; ++r_iter) {
          typename SendSection::point_type *v = recvAllocator.allocate(2);

          for(size_t i = 0; i < 2; ++i) {recvAllocator.construct(v+i, 0);}
          recvPoints[*r_iter] = v;
          pMover.recv(*r_iter, 2, recvPoints[*r_iter]);
          vMover.recv(*r_iter, 2, rValues);
        }
        pMover.start();
        pMover.end();
        vMover.start();
        vMover.end();

        typename SendSection::point_type min = -1;
        typename SendSection::point_type max = -1;

        for(typename RecvOverlap::capSequence::iterator r_iter = rRanks->begin(); r_iter != rEnd; ++r_iter) {
          const typename RecvSection::point_type *v = recvPoints[*r_iter];
          typename SendSection::point_type        newMin = v[0];
          typename SendSection::point_type        newMax = v[1]-1;
          //int                                     pSize  = 0;

          ///std::cout << "["<<recvOverlap->commRank()<<"]Received chart (" << v[0] << ", " << v[1] << ") from process " << *r_iter << std::endl;
#if 0
          // Translate to local numbering
          if (recvOverlap->support(*r_iter)->size()) {
            while(!pSize) {
              const Obj<typename RecvOverlap::supportSequence>& points = recvOverlap->support(*r_iter, newMin);
              pSize = points->size();
              if (pSize) {
                newMin = *points->begin();
              } else {
                newMin++;
              }
            }
            pSize  = 0;
            while(!pSize) {
              const Obj<typename RecvOverlap::supportSequence>& points = recvOverlap->support(*r_iter, newMax);
              pSize = points->size();
              if (pSize) {
                newMax = *points->begin();
              } else {
                newMax--;
              }
            }
          }
          std::cout << "["<<recvOverlap->commRank()<<"]Translated to chart (" << newMin << ", " << newMax+1 << ") from process " << *r_iter << std::endl;
#endif
          // Update chart
          if (min < 0) {
            min = newMin;
            max = newMax+1;
          } else {
            min = std::min(min, newMin);
            max = std::max(max, (typename SendSection::point_type) (newMax+1));
          }
        }
        if (!rRanks->size()) {min = max = 0;}
        recvSection->setChart(typename RecvSection::chart_type(min, max));
        for(typename SendOverlap::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sEnd; ++r_iter) {
          sendAllocator.deallocate(sendPoints[*r_iter], 2);
        }
        for(typename RecvOverlap::capSequence::iterator r_iter = rRanks->begin(); r_iter != rEnd; ++r_iter) {
          recvAllocator.deallocate(recvPoints[*r_iter], 2);
        }
      }
      // Copy the overlap section to the related processes
      //   This version is for different sections, possibly with different data types
      // TODO: Can cache MPIMover objects (like a VecScatter)
      template<typename SendOverlap, typename RecvOverlap, typename SendSection, typename RecvSection>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<SendSection>& sendSection, const Obj<RecvSection>& recvSection, const MPI_Datatype datatype = MPI_DATATYPE_NULL) {
        typedef std::pair<int, typename SendSection::value_type *> allocPair;
        typedef typename RecvSection::point_type                   recv_point_type;
        const Obj<typename SendSection::atlas_type>& sendAtlas = sendSection->getAtlas();
        const Obj<typename RecvSection::atlas_type>& recvAtlas = recvSection->getAtlas();
        MPIMover<typename SendSection::value_type>   vMover(sendSection->comm(), datatype, MPI_UNDEFINED, sendSection->debug());
        std::map<int, allocPair>                     sendValues;
        std::map<int, allocPair>                     recvValues;
        typename SendSection::alloc_type             sendAllocator;
        typename RecvSection::alloc_type             recvAllocator;

        copy(sendOverlap, recvOverlap, sendAtlas, recvAtlas);
        const typename SendOverlap::baseSequence::iterator sBegin = sendOverlap->baseBegin();
        const typename SendOverlap::baseSequence::iterator sEnd   = sendOverlap->baseEnd();

        // TODO: This should be const_iterator, but Sifter sucks
        for(typename SendOverlap::baseSequence::iterator r_iter = sBegin; r_iter != sEnd; ++r_iter) {
          const typename SendOverlap::coneSequence::iterator pBegin    = sendOverlap->coneBegin(*r_iter);
          const typename SendOverlap::coneSequence::iterator pEnd      = sendOverlap->coneEnd(*r_iter);
          const int                                          numPoints = sendOverlap->getConeSize(*r_iter);
          std::valarray<typename SendOverlap::source_type>   sortedPoints(numPoints);
          int                                                numVals   = 0, p = 0;

          // TODO: This should be const_iterator, but Sifter sucks
          for(typename SendOverlap::coneSequence::iterator c_iter = pBegin; c_iter != pEnd; ++c_iter, ++p) {
            numVals += sendSection->getFiberDimension(*c_iter);
            sortedPoints[p] = *c_iter;
          }
          typename SendSection::value_type *v = sendAllocator.allocate(numVals);
          int                               k = 0;

          std::sort(&sortedPoints[0], &sortedPoints[numPoints]);
          for(int i = 0; i < numVals; ++i) {sendAllocator.construct(v+i, 0);}
          for(p = 0; p < numPoints; ++p) {
            const typename SendSection::value_type *vals = sendSection->restrictPoint(sortedPoints[p]);

            for(int i = 0; i < sendSection->getFiberDimension(sortedPoints[p]); ++i, ++k) v[k] = vals[i];
          }
          sendValues[*r_iter] = allocPair(numVals, v);
          vMover.send(*r_iter, numVals, v);
        }
        const typename RecvOverlap::capSequence::iterator rBegin = recvOverlap->capBegin();
        const typename RecvOverlap::capSequence::iterator rEnd   = recvOverlap->capEnd();

        recvSection->allocatePoint();
        // TODO: This should be const_iterator, but Sifter sucks
        int maxVals = 0;
        for(typename RecvOverlap::capSequence::iterator r_iter = rBegin; r_iter != rEnd; ++r_iter) {
          const typename RecvOverlap::supportSequence::iterator pBegin  = recvOverlap->supportBegin(*r_iter);
          const typename RecvOverlap::supportSequence::iterator pEnd    = recvOverlap->supportEnd(*r_iter);
          int                                                   numVals = 0;

          // TODO: This should be const_iterator, but Sifter sucks
          for(typename RecvOverlap::supportSequence::iterator s_iter = pBegin; s_iter != pEnd; ++s_iter) {
            numVals += recvSection->getFiberDimension(recv_point_type(*r_iter, s_iter.color()));
          }
          typename SendSection::value_type *v = sendAllocator.allocate(numVals);

          for(int i = 0; i < numVals; ++i) {sendAllocator.construct(v+i, 0);}
          recvValues[*r_iter] = allocPair(numVals, v);
          vMover.recv(*r_iter, numVals, v);
          maxVals = std::max(maxVals, numVals);
        }
        vMover.start();
        vMover.end();
        typename RecvSection::value_type *convertedValues = recvAllocator.allocate(maxVals);
        for(int i = 0; i < maxVals; ++i) {recvAllocator.construct(convertedValues+i, 0);}
        for(typename RecvOverlap::capSequence::iterator r_iter = rBegin; r_iter != rEnd; ++r_iter) {
          const typename RecvOverlap::supportSequence::iterator pBegin    = recvOverlap->supportBegin(*r_iter);
          const typename RecvOverlap::supportSequence::iterator pEnd      = recvOverlap->supportEnd(*r_iter);
          const int                                             numPoints = recvOverlap->getSupportSize(*r_iter);
          std::valarray<typename RecvOverlap::color_type>       sortedPoints(numPoints);
          typename SendSection::value_type                     *v       = recvValues[*r_iter].second;
          const int                                             numVals = recvValues[*r_iter].first;
          int                                                   k       = 0, p = 0;

          for(typename RecvOverlap::supportSequence::iterator s_iter = pBegin; s_iter != pEnd; ++s_iter, ++p) {
            sortedPoints[p] = s_iter.color();
          }
          std::sort(&sortedPoints[0], &sortedPoints[numPoints]);

          //for(typename RecvOverlap::supportSequence::iterator s_iter = points->begin(); s_iter != pEnd; ++s_iter) {
          for(p = 0; p < numPoints; ++p) {
            const int size = recvSection->getFiberDimension(recv_point_type(*r_iter, sortedPoints[p]));

            for(int i = 0; i < size; ++i) {convertedValues[i] = (typename RecvSection::value_type) v[k+i];}
            if (size) {recvSection->updatePoint(recv_point_type(*r_iter, sortedPoints[p]), convertedValues);}
            k += size;
          }
          assert(k == numVals);
          for(int i = 0; i < numVals; ++i) {sendAllocator.destroy(v+i);}
        }
        for(int i = 0; i < maxVals; ++i) {recvAllocator.destroy(convertedValues+i);}
        recvAllocator.deallocate(convertedValues, maxVals);
        for(typename SendOverlap::baseSequence::iterator r_iter = sBegin; r_iter != sEnd; ++r_iter) {
          typename SendSection::value_type *v       = sendValues[*r_iter].second;
          const int                         numVals = sendValues[*r_iter].first;

          for(int i = 0; i < numVals; ++i) {sendAllocator.destroy(v+i);}
          sendAllocator.deallocate(v, numVals);
        }
        for(typename RecvOverlap::capSequence::iterator r_iter = rBegin; r_iter != rEnd; ++r_iter) {
          typename SendSection::value_type *v       = recvValues[*r_iter].second;
          const int                         numVals = recvValues[*r_iter].first;

          for(int i = 0; i < numVals; ++i) {sendAllocator.destroy(v+i);}
          sendAllocator.deallocate(v, numVals);
        }
        //recvSection->view("After copy");
      }
      // Copy the overlap section to the related processes
      //   This version is for sections with the same type
      template<typename SendOverlap, typename RecvOverlap, typename Section>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<Section>& sendSection, const Obj<Section>& recvSection, const MPI_Datatype datatype = MPI_DATATYPE_NULL) {
        typedef std::pair<int, typename Section::value_type *> allocPair;
        const Obj<typename Section::atlas_type>& sendAtlas = sendSection->getAtlas();
        const Obj<typename Section::atlas_type>& recvAtlas = recvSection->getAtlas();
        MPIMover<typename Section::value_type>   vMover(sendSection->comm(), datatype, MPI_UNDEFINED, sendSection->debug());
        std::map<int, allocPair>                 sendValues;
        std::map<int, allocPair>                 recvValues;
        typename Section::alloc_type             allocator;

        ///sendAtlas->view("Send Atlas in same type copy()");
        copy(sendOverlap, recvOverlap, sendAtlas, recvAtlas);
        ///recvAtlas->view("Recv Atlas after same type copy()");
        const Obj<typename SendOverlap::traits::baseSequence>      sRanks = sendOverlap->base();
        const typename SendOverlap::traits::baseSequence::iterator sEnd   = sRanks->end();

        // TODO: This should be const_iterator, but Sifter sucks
        for(typename SendOverlap::traits::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sEnd; ++r_iter) {
          const Obj<typename SendOverlap::coneSequence>&     points    = sendOverlap->cone(*r_iter);
          const typename SendOverlap::coneSequence::iterator pEnd      = points->end();
          const int                                          numPoints = sendOverlap->getConeSize(*r_iter);
          std::valarray<typename SendOverlap::source_type>   sortedPoints(numPoints);
          int                                                numVals   = 0, p = 0;

          // TODO: This should be const_iterator, but Sifter sucks
          for(typename SendOverlap::coneSequence::iterator c_iter = points->begin(); c_iter != pEnd; ++c_iter, ++p) {
            numVals += sendSection->getFiberDimension(*c_iter);
            sortedPoints[p] = *c_iter;
          }
          typename Section::value_type *v = allocator.allocate(numVals);
          int                           k = 0;

          std::sort(&sortedPoints[0], &sortedPoints[numPoints]);
          for(int i = 0; i < numVals; ++i) {allocator.construct(v+i, 0);}
          //for(typename SendOverlap::coneSequence::iterator c_iter = points->begin(); c_iter != pEnd; ++c_iter) {
          for(p = 0; p < numPoints; ++p) {
            const typename Section::value_type *vals = sendSection->restrictPoint(sortedPoints[p]);
            const int                           dim  = sendSection->getFiberDimension(sortedPoints[p]);

            for(int i = 0; i < dim; ++i, ++k) v[k] = vals[i];
          }
          sendValues[*r_iter] = allocPair(numVals, v);
          vMover.send(*r_iter, numVals, v);
        }
        const Obj<typename RecvOverlap::traits::capSequence>      rRanks = recvOverlap->cap();
        const typename RecvOverlap::traits::capSequence::iterator rEnd   = rRanks->end();

        recvSection->allocatePoint();
        ///recvSection->view("Recv Section after same type copy() allocation");
        // TODO: This should be const_iterator, but Sifter sucks
        for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rEnd; ++r_iter) {
          const Obj<typename RecvOverlap::supportSequence>&     points  = recvOverlap->support(*r_iter);
          const typename RecvOverlap::supportSequence::iterator pEnd    = points->end();
          int                                                   numVals = 0;

          // TODO: This should be const_iterator, but Sifter sucks
          for(typename RecvOverlap::supportSequence::iterator s_iter = points->begin(); s_iter != pEnd; ++s_iter) {
            numVals += recvSection->getFiberDimension(s_iter.color());
          }
          typename Section::value_type *v = allocator.allocate(numVals);

          recvValues[*r_iter] = allocPair(numVals, v);
          for(int i = 0; i < numVals; ++i) {allocator.construct(v+i, 0);}
          vMover.recv(*r_iter, numVals, v);
        }
        vMover.start();
        vMover.end();
        for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rEnd; ++r_iter) {
          const Obj<typename RecvOverlap::supportSequence>&     points    = recvOverlap->support(*r_iter);
          const typename RecvOverlap::supportSequence::iterator pEnd      = points->end();
          const int                                             numPoints = recvOverlap->getSupportSize(*r_iter);
          std::valarray<typename RecvOverlap::color_type>       sortedPoints(numPoints);
          typename Section::value_type                         *v         = recvValues[*r_iter].second;
          const int                                             numVals   = recvValues[*r_iter].first;
          int                                                   k         = 0, p = 0;

          for(typename RecvOverlap::supportSequence::iterator s_iter = points->begin(); s_iter != pEnd; ++s_iter, ++p) {
            sortedPoints[p] = s_iter.color();
          }
          std::sort(&sortedPoints[0], &sortedPoints[numPoints]);

          //for(typename RecvOverlap::supportSequence::iterator s_iter = points->begin(); s_iter != pEnd; ++s_iter) {
          for(p = 0; p < numPoints; ++p) {
            const int size = recvSection->getFiberDimension(sortedPoints[p]);

            if (size) {recvSection->updatePoint(sortedPoints[p], &v[k]);}
            k += size;
          }
          assert(k == numVals);
          for(int i = 0; i < numVals; ++i) {allocator.destroy(v+i);}
          allocator.deallocate(v, numVals);
        }
        for(typename SendOverlap::traits::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sEnd; ++r_iter) {
          typename Section::value_type *v       = sendValues[*r_iter].second;
          const int                     numVals = sendValues[*r_iter].first;

          for(int i = 0; i < numVals; ++i) {allocator.destroy(v+i);}
          allocator.deallocate(v, numVals);
        }
        //recvSection->view("After copy");
      }
      // Specialize to ArrowSection
      template<typename SendOverlap, typename RecvOverlap>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<UniformSection<MinimalArrow<int,int>,int> >& sendSection, const Obj<UniformSection<MinimalArrow<int,int>,int> >& recvSection, const MPI_Datatype datatype = MPI_DATATYPE_NULL) {
        typedef UniformSection<MinimalArrow<int,int>,int>      Section;
        typedef std::pair<int, typename Section::value_type *> allocPair;
        const Obj<typename Section::atlas_type>& sendAtlas = sendSection->getAtlas();
        const Obj<typename Section::atlas_type>& recvAtlas = recvSection->getAtlas();
        MPIMover<typename Section::value_type>   vMover(sendSection->comm(), datatype, MPI_UNDEFINED, sendSection->debug());
        std::map<int, allocPair>                 sendValues;
        std::map<int, allocPair>                 recvValues;
        typename Section::alloc_type             allocator;

        copy(sendOverlap, recvOverlap, sendAtlas, recvAtlas);
        const Obj<typename SendOverlap::traits::baseSequence>      sRanks = sendOverlap->base();
        const typename SendOverlap::traits::baseSequence::iterator sEnd   = sRanks->end();

        // TODO: This should be const_iterator, but Sifter sucks
        for(typename SendOverlap::traits::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sEnd; ++r_iter) {
          const Obj<typename SendOverlap::coneSequence>&     points  = sendOverlap->cone(*r_iter);
          const typename SendOverlap::coneSequence::iterator pBegin  = points->begin();
          const typename SendOverlap::coneSequence::iterator pEnd    = points->end();
          int                                                numVals = 0;

          // TODO: This should be const_iterator, but Sifter sucks
          for(typename SendOverlap::coneSequence::iterator c_iter = pBegin; c_iter != pEnd; ++c_iter) {
            for(typename SendOverlap::coneSequence::iterator d_iter = pBegin; d_iter != pEnd; ++d_iter) {
              numVals += sendSection->getFiberDimension(typename Section::point_type(*c_iter, *d_iter));
            }
          }
          typename Section::value_type *v = allocator.allocate(numVals);
          int                           k = 0;

          for(int i = 0; i < numVals; ++i) {allocator.construct(v+i, 0);}
          for(typename SendOverlap::coneSequence::iterator c_iter = pBegin; c_iter != pEnd; ++c_iter) {
            for(typename SendOverlap::coneSequence::iterator d_iter = pBegin; d_iter != pEnd; ++d_iter) {
              const typename Section::point_type  arrow(*c_iter, *d_iter);
              const typename Section::value_type *vals = sendSection->restrictPoint(arrow);
              const int                           dim  = sendSection->getFiberDimension(arrow);

              for(int i = 0; i < dim; ++i, ++k) v[k] = vals[i];
            }
          }
          sendValues[*r_iter] = allocPair(numVals, v);
          vMover.send(*r_iter, numVals, v);
        }
        const Obj<typename RecvOverlap::traits::capSequence>      rRanks = recvOverlap->cap();
        const typename RecvOverlap::traits::capSequence::iterator rEnd   = rRanks->end();

        recvSection->allocatePoint();
        // TODO: This should be const_iterator, but Sifter sucks
        for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rEnd; ++r_iter) {
          const Obj<typename RecvOverlap::supportSequence>&     points  = recvOverlap->support(*r_iter);
          const typename RecvOverlap::supportSequence::iterator pBegin  = points->begin();
          const typename RecvOverlap::supportSequence::iterator pEnd    = points->end();
          int                                                   numVals = 0;

          // TODO: This should be const_iterator, but Sifter sucks
          for(typename RecvOverlap::supportSequence::iterator s_iter = pBegin; s_iter != pEnd; ++s_iter) {
            for(typename RecvOverlap::supportSequence::iterator t_iter = pBegin; t_iter != pEnd; ++t_iter) {
              numVals += recvSection->getFiberDimension(typename Section::point_type(s_iter.color(), t_iter.color()));
            }
          }
          typename Section::value_type *v = allocator.allocate(numVals);

          recvValues[*r_iter] = allocPair(numVals, v);
          for(int i = 0; i < numVals; ++i) {allocator.construct(v+i, 0);}
          vMover.recv(*r_iter, numVals, v);
        }
        vMover.start();
        vMover.end();
        for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rEnd; ++r_iter) {
          const Obj<typename RecvOverlap::supportSequence>&     points  = recvOverlap->support(*r_iter);
          const typename RecvOverlap::supportSequence::iterator pBegin  = points->begin();
          const typename RecvOverlap::supportSequence::iterator pEnd    = points->end();
          typename Section::value_type                         *v       = recvValues[*r_iter].second;
          const int                                             numVals = recvValues[*r_iter].first;
          int                                                   k       = 0;

          for(typename RecvOverlap::supportSequence::iterator s_iter = pBegin; s_iter != pEnd; ++s_iter) {
            for(typename RecvOverlap::supportSequence::iterator t_iter = pBegin; t_iter != pEnd; ++t_iter) {
              const typename Section::point_type arrow(s_iter.color(), t_iter.color());
              const int size = recvSection->getFiberDimension(arrow);

              if (size) {recvSection->updatePoint(arrow, &v[k]);}
              k += size;
            }
          }
          for(int i = 0; i < numVals; ++i) {allocator.destroy(v+i);}
          allocator.deallocate(v, numVals);
        }
        for(typename SendOverlap::traits::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sEnd; ++r_iter) {
          typename Section::value_type *v       = sendValues[*r_iter].second;
          const int                     numVals = sendValues[*r_iter].first;

          for(int i = 0; i < numVals; ++i) {allocator.destroy(v+i);}
          allocator.deallocate(v, numVals);
        }
        //recvSection->view("After copy");
      }
      // Specialize to a ConstantSection
#if 0
      template<typename SendOverlap, typename RecvOverlap, typename Value>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<ConstantSection<typename SendOverlap::source_type, Value> >& sendSection, const Obj<ConstantSection<typename SendOverlap::source_type, Value> >& recvSection) {
        copyConstant(sendOverlap, recvOverlap, sendSection, recvSection);
      };
      template<typename SendOverlap, typename RecvOverlap, typename Value>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<IConstantSection<typename SendOverlap::source_type, Value> >& sendSection, const Obj<ConstantSection<typename SendOverlap::source_type, Value> >& recvSection) {
        copyConstant(sendOverlap, recvOverlap, sendSection, recvSection);
      };
#else
      template<typename SendOverlap, typename RecvOverlap, typename Value>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<ConstantSection<typename SendOverlap::source_type, Value> >& sendSection, const Obj<ConstantSection<ALE::Pair<int, typename SendOverlap::source_type>, Value> >& recvSection) {
        copyConstant(sendOverlap, recvOverlap, sendSection, recvSection);
      }
      template<typename SendOverlap, typename RecvOverlap, typename Value>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<IConstantSection<typename SendOverlap::source_type, Value> >& sendSection, const Obj<ConstantSection<ALE::Pair<int, typename SendOverlap::source_type>, Value> >& recvSection) {
        copyConstant(sendOverlap, recvOverlap, sendSection, recvSection);
      }
#endif
      // Specialize to an IConstantSection
      template<typename SendOverlap, typename RecvOverlap, typename Value>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<IConstantSection<typename SendOverlap::source_type, Value> >& sendSection, const Obj<IConstantSection<typename SendOverlap::source_type, Value> >& recvSection) {
        // Why doesn't this work?
        //   This supposed to be a copy, BUT filtered through the sendOverlap
        //   However, an IConstant section does not allow filtration of its
        //   chart. Therefore, you end up with either
        //
        //   a) Too many items in the chart, copied from the remote sendSection
        //   b) A chart mapped to the local numbering, which we do not want
        copyIConstant(sendOverlap, recvOverlap, sendSection, recvSection);
      }
      // Specialize to an BaseSection/ConstantSection pair
#if 0
      template<typename SendOverlap, typename RecvOverlap, typename Sieve_>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<BaseSection<Sieve_> >& sendSection, const Obj<ConstantSection<typename SendOverlap::source_type, int> >& recvSection) {
        copyConstant(sendOverlap, recvOverlap, sendSection, recvSection);
      };
#else
      template<typename SendOverlap, typename RecvOverlap, typename Sieve_>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<BaseSection<Sieve_> >& sendSection, const Obj<ConstantSection<ALE::Pair<int, typename SendOverlap::source_type>, int> >& recvSection) {
        copyConstant(sendOverlap, recvOverlap, sendSection, recvSection);
      }
#endif
      // Specialize to an BaseSectionV/ConstantSection pair
#if 0
      template<typename SendOverlap, typename RecvOverlap, typename Sieve_>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<BaseSectionV<Sieve_> >& sendSection, const Obj<ConstantSection<typename SendOverlap::source_type, int> >& recvSection) {
        copyConstant(sendOverlap, recvOverlap, sendSection, recvSection);
      };
#else
      template<typename SendOverlap, typename RecvOverlap, typename Sieve_>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<BaseSectionV<Sieve_> >& sendSection, const Obj<ConstantSection<ALE::Pair<int, typename SendOverlap::source_type>, int> >& recvSection) {
        copyConstant(sendOverlap, recvOverlap, sendSection, recvSection);
      }
#endif
      // Specialize to an LabelBaseSection/ConstantSection pair
#if 0
      template<typename SendOverlap, typename RecvOverlap, typename Sieve_, typename Label_>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<LabelBaseSection<Sieve_, Label_> >& sendSection, const Obj<ConstantSection<typename SendOverlap::source_type, int> >& recvSection) {
        copyConstant(sendOverlap, recvOverlap, sendSection, recvSection);
      };
#else
      template<typename SendOverlap, typename RecvOverlap, typename Sieve_, typename Label_>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<LabelBaseSection<Sieve_, Label_> >& sendSection, const Obj<ConstantSection<ALE::Pair<int, typename SendOverlap::source_type>, int> >& recvSection) {
        copyConstant(sendOverlap, recvOverlap, sendSection, recvSection);
      }
#endif
      template<typename SendOverlap, typename RecvOverlap, typename Sieve_, typename Label_>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<LabelBaseSectionV<Sieve_, Label_> >& sendSection, const Obj<ConstantSection<ALE::Pair<int, typename SendOverlap::source_type>, int> >& recvSection) {
        copyConstant(sendOverlap, recvOverlap, sendSection, recvSection);
      }
      // Specialize to a ConstantSection for ArrowSection
      template<typename SendOverlap, typename RecvOverlap, typename Value>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<ConstantSection<MinimalArrow<typename SendOverlap::source_type,typename SendOverlap::source_type>, Value> >& sendSection, const Obj<ConstantSection<MinimalArrow<typename SendOverlap::source_type,typename SendOverlap::source_type>, Value> >& recvSection) {
        copyConstantArrow(sendOverlap, recvOverlap, sendSection, recvSection);
      }
    };
    class BinaryFusion {
    public:
      template<typename OverlapSection, typename RecvOverlap, typename Section, typename Function>
      static void fuse(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, const Obj<Section>& section, Function binaryOp) {
        const Obj<typename RecvOverlap::traits::baseSequence>      rPoints = recvOverlap->base();
        const typename RecvOverlap::traits::baseSequence::iterator rEnd    = rPoints->end();

        for(typename RecvOverlap::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rEnd; ++p_iter) {
          // TODO: This must become a more general iterator over colors
          const Obj<typename RecvOverlap::coneSequence>& points = recvOverlap->cone(*p_iter);
          // Just taking the first value
          const typename Section::point_type&        localPoint    = *p_iter;
          const typename OverlapSection::point_type& remotePoint   = points->begin().color();
          const typename OverlapSection::value_type *overlapValues = overlapSection->restrictPoint(remotePoint);
          const typename Section::value_type        *localValues   = section->restrictPoint(localPoint);
          const int                                  dim           = section->getFiberDimension(localPoint);
          // TODO: optimize allocation
          typename Section::value_type              *values        = new typename Section::value_type[dim];

          for(int d = 0; d < dim; ++d) {
            values[d] = binaryOp(overlapValues[d], localValues[d]);
          }
          section->updatePoint(localPoint, values);
          delete [] values;
        }
      }
    };
    class ReplacementBinaryFusion {
    public:
      template<typename OverlapSection, typename RecvOverlap, typename Section>
      static void fuse(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, const Obj<Section>& section) {
        const Obj<typename RecvOverlap::traits::baseSequence>      rPoints = recvOverlap->base();
        const typename RecvOverlap::traits::baseSequence::iterator rEnd    = rPoints->end();

        for(typename RecvOverlap::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rEnd; ++p_iter) {
          // TODO: This must become a more general iterator over colors
          const Obj<typename RecvOverlap::coneSequence>& points      = recvOverlap->cone(*p_iter);
          // Just taking the first value
          const typename Section::point_type&            localPoint  = *p_iter;
          const typename OverlapSection::point_type&     remotePoint = points->begin().color();

          section->update(localPoint, overlapSection->restrictPoint(remotePoint));
        }
      }
    };
    class AdditiveBinaryFusion {
    public:
      template<typename OverlapSection, typename RecvOverlap, typename Section>
      static void fuse(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, const Obj<Section>& section) {
        typedef typename Section::point_type        point_type;
        typedef typename OverlapSection::point_type overlap_point_type;
        const typename RecvOverlap::capSequence::iterator rBegin = recvOverlap->capBegin();
        const typename RecvOverlap::capSequence::iterator rEnd   = recvOverlap->capEnd();

        for(typename RecvOverlap::capSequence::iterator r_iter = rBegin; r_iter != rEnd; ++r_iter) {
          const int                                             rank    = *r_iter;
          const typename RecvOverlap::supportSequence::iterator pBegin  = recvOverlap->supportBegin(*r_iter);
          const typename RecvOverlap::supportSequence::iterator pEnd    = recvOverlap->supportEnd(*r_iter);

          for(typename RecvOverlap::supportSequence::iterator p_iter = pBegin; p_iter != pEnd; ++p_iter) {
            const point_type& localPoint  = *p_iter;
            const point_type& remotePoint = p_iter.color();

            section->updateAddPoint(localPoint, overlapSection->restrictPoint(overlap_point_type(rank, remotePoint)));
          }
        }
      }
    };
    class InsertionBinaryFusion {
    public:
      // Insert the overlapSection values into section along recvOverlap
      template<typename OverlapSection, typename RecvOverlap, typename Section>
      static void fuse(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, const Obj<Section>& section) {
        typedef typename Section::point_type        point_type;
        typedef typename OverlapSection::point_type overlap_point_type;
#if 0
        const Obj<typename RecvOverlap::baseSequence>      rPoints = recvOverlap->base();
        const typename RecvOverlap::baseSequence::iterator rEnd    = rPoints->end();

        for(typename RecvOverlap::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rEnd; ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>& points      = recvOverlap->cone(*p_iter);
          const point_type&                              localPoint  = *p_iter;
          const int                                      rank        = *points->begin();
          const point_type&                              remotePoint = points->begin().color();

          if (overlapSection->hasPoint(overlap_point_type(rank, remotePoint))) {
            if (!section->hasPoint(localPoint)) {
              std::cout <<"["<<section->commRank()<<"]: Destination section does not have local point " << localPoint << " remote point " << remotePoint << " fiber dim " << overlapSection->getFiberDimension(overlap_point_type(rank, remotePoint)) << std::endl;
            }
            section->setFiberDimension(localPoint, overlapSection->getFiberDimension(overlap_point_type(rank, remotePoint)));
          }
        }
        if (rPoints->size()) {section->allocatePoint();}
        for(typename RecvOverlap::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rEnd; ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>& points      = recvOverlap->cone(*p_iter);
          const point_type&                              localPoint  = *p_iter;
          const int                                      rank        = *points->begin();
          const point_type&                              remotePoint = points->begin().color();

          if (overlapSection->hasPoint(overlap_point_type(rank, remotePoint))) {
            section->updatePoint(localPoint, overlapSection->restrictPoint(overlap_point_type(rank, remotePoint)));
          }
        }
#else
        const typename RecvOverlap::capSequence::iterator rBegin = recvOverlap->capBegin();
        const typename RecvOverlap::capSequence::iterator rEnd   = recvOverlap->capEnd();
        bool                                              hasPoints = false;

        for(typename RecvOverlap::capSequence::iterator r_iter = rBegin; r_iter != rEnd; ++r_iter) {
          const int                                             rank    = *r_iter;
          const typename RecvOverlap::supportSequence::iterator pBegin  = recvOverlap->supportBegin(*r_iter);
          const typename RecvOverlap::supportSequence::iterator pEnd    = recvOverlap->supportEnd(*r_iter);

          for(typename RecvOverlap::supportSequence::iterator p_iter = pBegin; p_iter != pEnd; ++p_iter) {
            const point_type&                              localPoint  = *p_iter;
            const point_type&                              remotePoint = p_iter.color();

            if (overlapSection->hasPoint(overlap_point_type(rank, remotePoint))) {
              if (!section->hasPoint(localPoint)) {
                std::cout <<"["<<section->commRank()<<"]: Destination section does not have local point " << localPoint << " remote point " << remotePoint << " fiber dim " << overlapSection->getFiberDimension(overlap_point_type(rank, remotePoint)) << std::endl;
              }
              section->setFiberDimension(localPoint, overlapSection->getFiberDimension(overlap_point_type(rank, remotePoint)));
            }
            hasPoints = true;
          }
        }
        if (hasPoints) {section->allocatePoint();}
        for(typename RecvOverlap::capSequence::iterator r_iter = rBegin; r_iter != rEnd; ++r_iter) {
          const int                                             rank    = *r_iter;
          const typename RecvOverlap::supportSequence::iterator pBegin  = recvOverlap->supportBegin(*r_iter);
          const typename RecvOverlap::supportSequence::iterator pEnd    = recvOverlap->supportEnd(*r_iter);

          for(typename RecvOverlap::supportSequence::iterator p_iter = pBegin; p_iter != pEnd; ++p_iter) {
            const point_type& localPoint  = *p_iter;
            const point_type& remotePoint = p_iter.color();

            if (overlapSection->hasPoint(overlap_point_type(rank, remotePoint))) {
              section->updatePoint(localPoint, overlapSection->restrictPoint(overlap_point_type(rank, remotePoint)));
            }
          }
        }
#endif
      }
      // Specialize to ArrowSection
      template<typename OverlapSection, typename RecvOverlap>
      static void fuse(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, const Obj<UniformSection<MinimalArrow<int,int>,int> >& section) {
        typedef UniformSection<MinimalArrow<int,int>,int> Section;
        const Obj<typename RecvOverlap::traits::baseSequence>      rPoints = recvOverlap->base();
        const typename RecvOverlap::traits::baseSequence::iterator rBegin  = rPoints->begin();
        const typename RecvOverlap::traits::baseSequence::iterator rEnd    = rPoints->end();

        for(typename RecvOverlap::traits::baseSequence::iterator p_iter = rBegin; p_iter != rEnd; ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>& sources      = recvOverlap->cone(*p_iter);
          const typename RecvOverlap::target_type        localSource  = *p_iter;
          const typename RecvOverlap::target_type        remoteSource = sources->begin().color();

          for(typename RecvOverlap::traits::baseSequence::iterator q_iter = rBegin; q_iter != rEnd; ++q_iter) {
            const Obj<typename RecvOverlap::coneSequence>& targets      = recvOverlap->cone(*q_iter);
            const typename RecvOverlap::target_type        localTarget  = *q_iter;
            const typename RecvOverlap::target_type        remoteTarget = targets->begin().color();
            const typename Section::point_type             localPoint(localSource, localTarget);
            const typename OverlapSection::point_type      remotePoint(remoteSource, remoteTarget);

            if (overlapSection->hasPoint(remotePoint)) {section->setFiberDimension(localPoint, overlapSection->getFiberDimension(remotePoint));}
          }
        }
        if (rPoints->size()) {section->allocatePoint();}
        for(typename RecvOverlap::traits::baseSequence::iterator p_iter = rBegin; p_iter != rEnd; ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>& sources      = recvOverlap->cone(*p_iter);
          const typename RecvOverlap::target_type        localSource  = *p_iter;
          const typename RecvOverlap::target_type        remoteSource = sources->begin().color();

          for(typename RecvOverlap::traits::baseSequence::iterator q_iter = rBegin; q_iter != rEnd; ++q_iter) {
            const Obj<typename RecvOverlap::coneSequence>& targets      = recvOverlap->cone(*q_iter);
            const typename RecvOverlap::target_type        localTarget  = *q_iter;
            const typename RecvOverlap::target_type        remoteTarget = targets->begin().color();
            const typename Section::point_type             localPoint(localSource, localTarget);
            const typename OverlapSection::point_type      remotePoint(remoteSource, remoteTarget);

            if (overlapSection->hasPoint(remotePoint)) {section->updatePoint(localPoint, overlapSection->restrictPoint(remotePoint));}
          }
        }
      }
      // Specialize to the Sieve
      template<typename OverlapSection, typename RecvOverlap, typename Renumbering, typename Point>
      static void fuse(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, Renumbering& renumbering, const Obj<Sieve<Point,Point,int> >& sieve) {
        typedef typename OverlapSection::point_type overlap_point_type;
        const Obj<typename RecvOverlap::traits::baseSequence>      rPoints = recvOverlap->base();
        const typename RecvOverlap::traits::baseSequence::iterator rEnd    = rPoints->end();

        for(typename RecvOverlap::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rEnd; ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>& points      = recvOverlap->cone(*p_iter);
          const Point&                                   localPoint  = *p_iter;
          const int                                      rank        = *points->begin();
          const Point&                                   remotePoint = points->begin().color();
          const int                                      size        = overlapSection->getFiberDimension(overlap_point_type(rank, remotePoint));
          const typename OverlapSection::value_type     *values      = overlapSection->restrictPoint(overlap_point_type(rank, remotePoint));
          int                                            c           = 0;

          sieve->clearCone(localPoint);
          for(int i = 0; i < size; ++i, ++c) {sieve->addCone(renumbering[values[i]], localPoint, c);}
        }
      }
      // Specialize to the ISieve
      template<typename OverlapSection, typename RecvOverlap, typename Renumbering, typename Point>
      static void fuse(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, Renumbering& renumbering, const Obj<IFSieve<Point> >& sieve) {
        typedef typename OverlapSection::point_type overlap_point_type;
#if 0
        const Obj<typename RecvOverlap::baseSequence>      rPoints = recvOverlap->base();
        const typename RecvOverlap::baseSequence::iterator rEnd    = rPoints->end();
        int                                                maxSize = 0;

        for(typename RecvOverlap::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rEnd; ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>& points      = recvOverlap->cone(*p_iter);
          const Point&                                   localPoint  = *p_iter;
          const int                                      rank        = *points->begin();
          const Point&                                   remotePoint = points->begin().color();
          const int                                      size        = overlapSection->getFiberDimension(overlap_point_type(rank, remotePoint));
          const typename OverlapSection::value_type     *values      = overlapSection->restrictPoint(overlap_point_type(rank, remotePoint));

          sieve->setConeSize(localPoint, size);
          ///for(int i = 0; i < size; ++i) {sieve->addSupportSize(renumbering[values[i]], 1);}
          for(int i = 0; i < size; ++i) {sieve->addSupportSize(renumbering[values[i].first], 1);}
          maxSize = std::max(maxSize, size);
        }
        sieve->allocate();
        ///typename OverlapSection::value_type *localValues = new typename OverlapSection::value_type[maxSize];
        typename OverlapSection::value_type::first_type  *localValues      = new typename OverlapSection::value_type::first_type[maxSize];
        typename OverlapSection::value_type::second_type *localOrientation = new typename OverlapSection::value_type::second_type[maxSize];

        for(typename RecvOverlap::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rEnd; ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>& points      = recvOverlap->cone(*p_iter);
          const Point&                                   localPoint  = *p_iter;
          const int                                      rank        = *points->begin();
          const Point&                                   remotePoint = points->begin().color();
          const int                                      size        = overlapSection->getFiberDimension(overlap_point_type(rank, remotePoint));
          const typename OverlapSection::value_type     *values      = overlapSection->restrictPoint(overlap_point_type(rank, remotePoint));

          ///for(int i = 0; i < size; ++i) {localValues[i] = renumbering[values[i]];}
          for(int i = 0; i < size; ++i) {
            localValues[i]      = renumbering[values[i].first];
            localOrientation[i] = values[i].second;
          }
          sieve->setCone(localValues, localPoint);
          sieve->setConeOrientation(localOrientation, localPoint);
        }
        delete [] localValues;
        delete [] localOrientation;
#else
        const typename RecvOverlap::capSequence::iterator rBegin  = recvOverlap->capBegin();
        const typename RecvOverlap::capSequence::iterator rEnd    = recvOverlap->capEnd();
        int                                               maxSize = 0;

        for(typename RecvOverlap::capSequence::iterator r_iter = rBegin; r_iter != rEnd; ++r_iter) {
          const int                                             rank    = *r_iter;
          const typename RecvOverlap::supportSequence::iterator pBegin  = recvOverlap->supportBegin(*r_iter);
          const typename RecvOverlap::supportSequence::iterator pEnd    = recvOverlap->supportEnd(*r_iter);

          for(typename RecvOverlap::supportSequence::iterator p_iter = pBegin; p_iter != pEnd; ++p_iter) {
            const Point&                               localPoint  = *p_iter;
            const Point&                               remotePoint = p_iter.color();
            const int                                  size        = overlapSection->getFiberDimension(overlap_point_type(rank, remotePoint));
            const typename OverlapSection::value_type *values      = overlapSection->restrictPoint(overlap_point_type(rank, remotePoint));

            sieve->setConeSize(localPoint, size);
            for(int i = 0; i < size; ++i) {sieve->addSupportSize(renumbering[values[i].first], 1);}
            maxSize = std::max(maxSize, size);
          }
        }
        sieve->allocate();
        typename OverlapSection::value_type::first_type  *localValues      = new typename OverlapSection::value_type::first_type[maxSize];
        typename OverlapSection::value_type::second_type *localOrientation = new typename OverlapSection::value_type::second_type[maxSize];

        for(typename RecvOverlap::capSequence::iterator r_iter = rBegin; r_iter != rEnd; ++r_iter) {
          const int                                             rank    = *r_iter;
          const typename RecvOverlap::supportSequence::iterator pBegin  = recvOverlap->supportBegin(*r_iter);
          const typename RecvOverlap::supportSequence::iterator pEnd    = recvOverlap->supportEnd(*r_iter);

          for(typename RecvOverlap::supportSequence::iterator p_iter = pBegin; p_iter != pEnd; ++p_iter) {
            const Point&                               localPoint  = *p_iter;
            const Point&                               remotePoint = p_iter.color();
            const int                                  size        = overlapSection->getFiberDimension(overlap_point_type(rank, remotePoint));
            const typename OverlapSection::value_type *values      = overlapSection->restrictPoint(overlap_point_type(rank, remotePoint));

            for(int i = 0; i < size; ++i) {
              localValues[i]      = renumbering[values[i].first];
              localOrientation[i] = values[i].second;
            }
            sieve->setCone(localValues, localPoint);
            sieve->setConeOrientation(localOrientation, localPoint);
          }
        }
        delete [] localValues;
        delete [] localOrientation;
#endif
      }
      template<typename OverlapSection, typename RecvOverlap, typename Point>
      static void fuse(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, const Obj<IFSieve<Point> >& sieve) {
        typedef typename OverlapSection::point_type overlap_point_type;
#if 0
        const Obj<typename RecvOverlap::baseSequence>      rPoints = recvOverlap->base();
        const typename RecvOverlap::baseSequence::iterator rEnd    = rPoints->end();
        int                                                maxSize = 0;

        for(typename RecvOverlap::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rEnd; ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>& points      = recvOverlap->cone(*p_iter);
          const Point&                                   localPoint  = *p_iter;
          const int                                      rank        = *points->begin();
          const Point&                                   remotePoint = points->begin().color();
          const int                                      size        = overlapSection->getFiberDimension(overlap_point_type(rank, remotePoint));
          const typename OverlapSection::value_type     *values      = overlapSection->restrictPoint(overlap_point_type(rank, remotePoint));

          sieve->setConeSize(localPoint, size);
          for(int i = 0; i < size; ++i) {sieve->addSupportSize(values[i].first, 1);}
          maxSize = std::max(maxSize, size);
        }
        sieve->allocate();
        typename OverlapSection::value_type::first_type  *localValues      = new typename OverlapSection::value_type::first_type[maxSize];
        typename OverlapSection::value_type::second_type *localOrientation = new typename OverlapSection::value_type::second_type[maxSize];

        for(typename RecvOverlap::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rEnd; ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>& points      = recvOverlap->cone(*p_iter);
          const Point&                                   localPoint  = *p_iter;
          const int                                      rank        = *points->begin();
          const Point&                                   remotePoint = points->begin().color();
          const int                                      size        = overlapSection->getFiberDimension(overlap_point_type(rank, remotePoint));
          const typename OverlapSection::value_type     *values      = overlapSection->restrictPoint(overlap_point_type(rank, remotePoint));

          for(int i = 0; i < size; ++i) {
            localValues[i]      = values[i].first;
            localOrientation[i] = values[i].second;
          }
          sieve->setCone(localValues, localPoint);
          sieve->setConeOrientation(localOrientation, localPoint);
        }
        delete [] localValues;
        delete [] localOrientation;
#else
        const Obj<typename RecvOverlap::capSequence>      rRanks  = recvOverlap->cap();
        const typename RecvOverlap::capSequence::iterator rEnd    = rRanks->end();
        int                                               maxSize = 0;

        for(typename RecvOverlap::capSequence::iterator r_iter = rRanks->begin(); r_iter != rEnd; ++r_iter) {
          const int                                             rank    = *r_iter;
          const Obj<typename RecvOverlap::supportSequence>&     points  = recvOverlap->support(*r_iter);
          const typename RecvOverlap::supportSequence::iterator pBegin  = points->begin();
          const typename RecvOverlap::supportSequence::iterator pEnd    = points->end();

          for(typename RecvOverlap::supportSequence::iterator p_iter = pBegin; p_iter != pEnd; ++p_iter) {
            const Point&                               localPoint  = *p_iter;
            const Point&                               remotePoint = p_iter.color();
            const int                                  size        = overlapSection->getFiberDimension(overlap_point_type(rank, remotePoint));
            const typename OverlapSection::value_type *values      = overlapSection->restrictPoint(overlap_point_type(rank, remotePoint));

            sieve->setConeSize(localPoint, size);
            for(int i = 0; i < size; ++i) {sieve->addSupportSize(values[i].first, 1);}
            maxSize = std::max(maxSize, size);
          }
        }
        sieve->allocate();
        typename OverlapSection::value_type::first_type  *localValues      = new typename OverlapSection::value_type::first_type[maxSize];
        typename OverlapSection::value_type::second_type *localOrientation = new typename OverlapSection::value_type::second_type[maxSize];

        for(typename RecvOverlap::capSequence::iterator r_iter = rRanks->begin(); r_iter != rEnd; ++r_iter) {
          const int                                             rank    = *r_iter;
          const Obj<typename RecvOverlap::supportSequence>&     points  = recvOverlap->support(*r_iter);
          const typename RecvOverlap::supportSequence::iterator pBegin  = points->begin();
          const typename RecvOverlap::supportSequence::iterator pEnd    = points->end();

          for(typename RecvOverlap::supportSequence::iterator p_iter = pBegin; p_iter != pEnd; ++p_iter) {
            const Point&                               localPoint  = *p_iter;
            const Point&                               remotePoint = p_iter.color();
            const int                                  size        = overlapSection->getFiberDimension(overlap_point_type(rank, remotePoint));
            const typename OverlapSection::value_type *values      = overlapSection->restrictPoint(overlap_point_type(rank, remotePoint));

            for(int i = 0; i < size; ++i) {
              localValues[i]      = values[i].first;
              localOrientation[i] = values[i].second;
            }
            sieve->setCone(localValues, localPoint);
            sieve->setConeOrientation(localOrientation, localPoint);
          }
        }
        delete [] localValues;
        delete [] localOrientation;
#endif
      }
      // Generic
      template<typename OverlapSection, typename RecvOverlap, typename Section, typename Bundle>
      static void fuse(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, const Obj<Section>& section, const Obj<Bundle>& bundle) {
        typedef typename OverlapSection::point_type overlap_point_type;
        const Obj<typename RecvOverlap::baseSequence>      rPoints = recvOverlap->base();
        const typename RecvOverlap::baseSequence::iterator rEnd    = rPoints->end();

        for(typename RecvOverlap::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rEnd; ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>& points      = recvOverlap->cone(*p_iter);
          const typename Section::point_type&            localPoint  = *p_iter;
          const int                                      rank        = *points->begin();
          const typename OverlapSection::point_type&     remotePoint = points->begin().color();

          section->setFiberDimension(localPoint, overlapSection->getFiberDimension(overlap_point_type(rank, remotePoint)));
        }
        bundle->allocate(section);
        for(typename RecvOverlap::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rEnd; ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>& points      = recvOverlap->cone(*p_iter);
          const typename Section::point_type&            localPoint  = *p_iter;
          const int                                      rank        = *points->begin();
          const typename OverlapSection::point_type&     remotePoint = points->begin().color();

          section->update(localPoint, overlapSection->restrictPoint(overlap_point_type(rank, remotePoint)));
        }
      }
    };
    class InterpolateMultipleFusion {
    public:
      // Interpolate the overlapSection values into section along recvOverlap
      template<typename OverlapSection, typename RecvOverlap, typename Section>
      static void fuse(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, const Obj<Section>& section) {
        typedef typename Section::point_type        point_type;
        typedef typename Section::value_type        value_type;
        typedef typename OverlapSection::point_type overlap_point_type;
        const Obj<typename RecvOverlap::traits::baseSequence>      rPoints     = recvOverlap->base();
        const typename RecvOverlap::traits::baseSequence::iterator rEnd        = rPoints->end();
        int                                                        maxFiberDim = -1;

        for(typename RecvOverlap::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rEnd; ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>&     points     = recvOverlap->cone(*p_iter);
          const typename RecvOverlap::coneSequence::iterator rpEnd      = points->end();
          const point_type&                                  localPoint = *p_iter;
          bool                                               inOverlap  = false;
          int                                                fiberDim   = -1;

          for(typename RecvOverlap::coneSequence::iterator rp_iter = points->begin(); rp_iter != rpEnd; ++rp_iter) {
            const int         rank        = *rp_iter;
            const point_type& remotePoint = rp_iter.color();

            if (overlapSection->hasPoint(overlap_point_type(rank, remotePoint))) {
              inOverlap = true;
              fiberDim  = overlapSection->getFiberDimension(overlap_point_type(rank, remotePoint));
              break;
            }
          }
          if (inOverlap) {
            if (!section->hasPoint(localPoint)) {
              std::cout <<"["<<section->commRank()<<"]: Destination section does not have local point " << localPoint << " remote point " << (points->begin().color()) << " fiber dim " << fiberDim << std::endl;
            }
            section->setFiberDimension(localPoint, fiberDim);
            maxFiberDim = std::max(fiberDim, maxFiberDim);
          }
        }
        if (rPoints->size()) {section->allocatePoint();}
        value_type *interpolant = new value_type[maxFiberDim];

        for(typename RecvOverlap::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rEnd; ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>&     points     = recvOverlap->cone(*p_iter);
          const typename RecvOverlap::coneSequence::iterator rpEnd      = points->end();
          const point_type&                                  localPoint = *p_iter;
          bool                                               inOverlap  = false;
          int                                                numArgs    = 0;

          for(int d = 0; d < maxFiberDim; ++d) {interpolant[d] = 0.0;}
          for(typename RecvOverlap::coneSequence::iterator rp_iter = points->begin(); rp_iter != rpEnd; ++rp_iter) {
            const int         rank        = *rp_iter;
            const point_type& remotePoint = rp_iter.color();
            const overlap_point_type opoint(rank, remotePoint);

            if (overlapSection->hasPoint(opoint)) {
              const int         fiberDim = overlapSection->getFiberDimension(opoint);
              const value_type *values   = overlapSection->restrictPoint(opoint);

              // TODO: Include interpolation weights (stored in overlap)
              for(int d = 0; d < fiberDim; ++d) {
                interpolant[d] += values[d];
              }
              inOverlap = true;
              ++numArgs;
            }
          }
          if (inOverlap) {
            for(int d = 0; d < maxFiberDim; ++d) {interpolant[d] /= numArgs;}
            section->updatePoint(localPoint, interpolant);
          }
        }
        delete [] interpolant;
      }
    };
  }
}

#endif
