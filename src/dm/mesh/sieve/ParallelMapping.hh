#ifndef included_ALE_ParallelMapping_hh
#define included_ALE_ParallelMapping_hh

#ifndef  included_ALE_IField_hh
#include <IField.hh>
#endif

#ifndef  included_ALE_Sections_hh
#include <Sections.hh>
#endif

extern "C" PetscMPIInt Mesh_DelTag(MPI_Comm comm,PetscMPIInt keyval,void* attr_val,void* extra_state);

namespace ALE {
  // TODO: Check MPI return values and status of Waits
  template<typename Value_>
  class MPIMover : public ALE::ParallelObject {
  public:
    typedef Value_                                 value_type;
    typedef size_t                                 num_type;
    typedef std::pair<num_type, const value_type*> move_type;
    typedef std::map<int, move_type>               moves_type;
    typedef std::vector<MPI_Request>               requests_type;
  protected:
    bool          _createdType;
    int           _tag;
    MPI_Datatype  _datatype;
    moves_type    _sends;
    moves_type    _recvs;
    requests_type _requests;
  public:
    MPIMover(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug), _createdType(0) {
      this->_tag      = this->getNewTag();
      this->_datatype = this->getMPIDatatype();
    };
    MPIMover(MPI_Comm comm, const int tag, const int debug) : ParallelObject(comm, debug), _createdType(0), _tag(tag) {
      this->_datatype = this->getMPIDatatype();
    };
    MPIMover(MPI_Comm comm, const MPI_Datatype datatype, const int tag, const int debug) : ParallelObject(comm, debug), _createdType(0) {
      if (tag == MPI_UNDEFINED) {
        this->_tag      = this->getNewTag();
      } else {
        this->_tag      = tag;
      }
      if (datatype == MPI_DATATYPE_NULL) {
        this->_datatype = this->getMPIDatatype();
      } else {
        this->_datatype = datatype;
      }
    };
    ~MPIMover() {
      if (_createdType) {
        MPI_Type_free(&this->_datatype);
      }
    };
  protected:
    // TODO: Can rewrite this with template specialization?
    MPI_Datatype getMPIDatatype() {
      if (sizeof(value_type) == 1) {
        return MPI_BYTE;
      } else if (sizeof(value_type) == 2) {
        return MPI_SHORT;
      } else if (sizeof(value_type) == 4) {
        return MPI_INT;
      } else if (sizeof(value_type) == 8) {
        return MPI_DOUBLE;
      } else if (sizeof(value_type) == 28) {
        int          blen[2];
        MPI_Aint     indices[2];
        MPI_Datatype oldtypes[2], newtype;
        blen[0] = 1; indices[0] = 0;           oldtypes[0] = MPI_INT;
        blen[1] = 3; indices[1] = sizeof(int); oldtypes[1] = MPI_DOUBLE;
        MPI_Type_struct(2, blen, indices, oldtypes, &newtype);
        MPI_Type_commit(&newtype);
        this->_createdType = true;
        return newtype;
      } else if (sizeof(value_type) == 32) {
        int          blen[2];
        MPI_Aint     indices[2];
        MPI_Datatype oldtypes[2], newtype;
        blen[0] = 1; indices[0] = 0;           oldtypes[0] = MPI_DOUBLE;
        blen[1] = 3; indices[1] = sizeof(int); oldtypes[1] = MPI_DOUBLE;
        MPI_Type_struct(2, blen, indices, oldtypes, &newtype);
        MPI_Type_commit(&newtype);
        this->_createdType = true;
        return newtype;
      }
      throw ALE::Exception("Cannot determine MPI type for value type");
    };
    int getNewTag() const {
      static int tagKeyval = MPI_KEYVAL_INVALID;
      int *tagvalp = NULL, *maxval, flg;

      if (tagKeyval == MPI_KEYVAL_INVALID) {
        tagvalp = (int *) malloc(sizeof(int));
        MPI_Keyval_create(MPI_NULL_COPY_FN, Mesh_DelTag, &tagKeyval, (void *) NULL);
        MPI_Attr_put(this->comm(), tagKeyval, tagvalp);
        tagvalp[0] = 0;
      }
      MPI_Attr_get(this->comm(), tagKeyval, (void **) &tagvalp, &flg);
      if (tagvalp[0] < 1) {
        MPI_Attr_get(MPI_COMM_WORLD, MPI_TAG_UB, (void **) &maxval, &flg);
        tagvalp[0] = *maxval - 128; // hope that any still active tags were issued right at the beginning of the run
      }
      if (this->debug()) {
        std::cout << "[" << this->commRank() << "]Got new tag " << tagvalp[0] << std::endl;
      }
      return tagvalp[0]--;
    };
    void constructRequests() {
      this->_requests.clear();

      for(typename moves_type::const_iterator s_iter = this->_sends.begin(); s_iter != this->_sends.end(); ++s_iter) {
        const int   rank = s_iter->first;
        const int   num  = s_iter->second.first;
        void       *data = (void *) s_iter->second.second;
        MPI_Request request;

        if (this->_debug) {std::cout <<"["<<this->commRank()<<"] Sending data (" << num << ") to " << rank << " tag " << this->_tag << std::endl;}
        MPI_Send_init(data, num, this->_datatype, rank, this->_tag, this->comm(), &request);
        this->_requests.push_back(request);
      }
      for(typename moves_type::const_iterator r_iter = this->_recvs.begin(); r_iter != this->_recvs.end(); ++r_iter) {
        const int   rank = r_iter->first;
        const int   num  = r_iter->second.first;
        void       *data = (void *) r_iter->second.second;
        MPI_Request request;

        if (this->_debug) {std::cout <<"["<<this->commRank()<<"] Receiving data (" << num << ") from " << rank << " tag " << this->_tag << std::endl;}
        MPI_Recv_init(data, num, this->_datatype, rank, this->_tag, this->comm(), &request);
        this->_requests.push_back(request);
      }
    };
  public:
    void send(const int rank, const int num, const value_type *data) {
      this->_sends[rank] = move_type(num, data);
    };
    void recv(const int rank, const int num, const value_type *data) {
      this->_recvs[rank] = move_type(num, data);
    };
    void start() {
      this->constructRequests();
      for(typename requests_type::const_iterator r_iter = this->_requests.begin(); r_iter != this->_requests.end(); ++r_iter) {
        MPI_Request request = *r_iter;

        MPI_Start(&request);
      }
    };
    void end() {
      MPI_Status status;

      for(typename requests_type::const_iterator r_iter = this->_requests.begin(); r_iter != this->_requests.end(); ++r_iter) {
        MPI_Request request = *r_iter;

        MPI_Wait(&request, &status);
      }
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
    };
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
      for(typename Sequence::const_iterator l_iter = points.begin(); l_iter != points.end(); ++l_iter) {
        //std::cout << "["<<commRank<<"]Send point["<<size<<"]: " << *l_iter << " " << renumbering[*l_iter] << std::endl;
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
              //std::cout << "["<<commRank<<"]Sending points " << p_iter->first << " " << p_iter->second.second << " to rank " << rank << std::endl;
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

          //std::cout << "["<<commRank<<"]Matched up remote point " << remotePoint << "("<<point<<") to local " << renumbering[point] << std::endl;
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
    };
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

        const Obj<typename SendOverlap::traits::baseSequence> sRanks  = sendOverlap->base();
        const typename SendSection::value_type               *sValues = sendSection->restrictPoint(*sendSection->getChart().begin());

        for(typename SendOverlap::traits::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sRanks->end(); ++r_iter) {
          const Obj<typename SendOverlap::coneSequence>& points = sendOverlap->cone(*r_iter);
          char                                          *v      = sendAllocator.allocate(points->size());
          int                                            k      = 0;

          for(size_t i = 0; i < points->size(); ++i) {sendAllocator.construct(v+i, 0);}
          for(typename SendOverlap::coneSequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter, ++k) {
            v[k] = (char) sendSection->hasPoint(*p_iter);
          }
          sendPoints[*r_iter] = v;
          pMover.send(*r_iter, points->size(), sendPoints[*r_iter]);
          vMover.send(*r_iter, 2, sValues);
        }
        const Obj<typename RecvOverlap::traits::capSequence> rRanks  = recvOverlap->cap();
        const typename RecvSection::value_type              *rValues = recvSection->restrict();

        for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rRanks->end(); ++r_iter) {
          const Obj<typename RecvOverlap::traits::supportSequence>& points = recvOverlap->support(*r_iter);
          char                                                     *v      = recvAllocator.allocate(points->size());

          for(size_t i = 0; i < points->size(); ++i) {recvAllocator.construct(v+i, 0);}
          recvPoints[*r_iter] = v;
          pMover.recv(*r_iter, points->size(), recvPoints[*r_iter]);
          vMover.recv(*r_iter, 2, rValues);
        }
        pMover.start();
        pMover.end();
        vMover.start();
        vMover.end();
        for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rRanks->end(); ++r_iter) {
          const Obj<typename RecvOverlap::traits::supportSequence>& points = recvOverlap->support(*r_iter);
          const char                                               *v      = recvPoints[*r_iter];
          int                                                       k      = 0;

          for(typename RecvOverlap::traits::supportSequence::iterator s_iter = points->begin(); s_iter != points->end(); ++s_iter, ++k) {
            if (v[k]) {recvSection->addPoint(s_iter.color());}
          }
        }
      };
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

        const Obj<typename SendOverlap::traits::baseSequence> sRanks  = sendOverlap->base();
        const typename SendSection::value_type               *sValues = sendSection->restrictPoint(*sendSection->getChart().begin());

        for(typename SendOverlap::traits::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sRanks->end(); ++r_iter) {
          typename SendSection::point_type *v = sendAllocator.allocate(2);

          for(size_t i = 0; i < 2; ++i) {sendAllocator.construct(v+i, 0);}
          v[0] = sendSection->getChart().min();
          v[1] = sendSection->getChart().max();
          sendPoints[*r_iter] = v;
          pMover.send(*r_iter, 2, sendPoints[*r_iter]);
          vMover.send(*r_iter, 2, sValues);
        }
        const Obj<typename RecvOverlap::traits::capSequence> rRanks  = recvOverlap->cap();
        const typename RecvSection::value_type              *rValues = recvSection->restrict();

        for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rRanks->end(); ++r_iter) {
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

        typename SendSection::point_type min = SHRT_MAX;
        typename SendSection::point_type max = -1;

        for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rRanks->end(); ++r_iter) {
          const typename RecvSection::point_type *v = recvPoints[*r_iter];

          min = std::min(min, v[0]);
          max = std::max(max, v[1]);
        }
        if (!rRanks->size()) {min = max = 0;}
        recvSection->setChart(typename RecvSection::chart_type(min, max));
      };
      // Copy the overlap section to the related processes
      //   This version is for different sections, possibly with different data types
      // TODO: Can cache MPIMover objects (like a VecScatter)
      template<typename SendOverlap, typename RecvOverlap, typename SendSection, typename RecvSection>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<SendSection>& sendSection, const Obj<RecvSection>& recvSection, const MPI_Datatype datatype = MPI_DATATYPE_NULL) {
        const Obj<typename SendSection::atlas_type>&      sendAtlas = sendSection->getAtlas();
        const Obj<typename RecvSection::atlas_type>&      recvAtlas = recvSection->getAtlas();
        MPIMover<typename SendSection::value_type>        vMover(sendSection->comm(), datatype, MPI_UNDEFINED, sendSection->debug());
        std::map<int, typename SendSection::value_type *> sendValues;
        std::map<int, typename SendSection::value_type *> recvValues;
        typename SendSection::alloc_type                  sendAllocator;
        typename RecvSection::alloc_type                  recvAllocator;

        copy(sendOverlap, recvOverlap, sendAtlas, recvAtlas);
        const Obj<typename SendOverlap::traits::baseSequence> sRanks = sendOverlap->base();

        // TODO: This should be const_iterator, but Sifter sucks
        for(typename SendOverlap::traits::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sRanks->end(); ++r_iter) {
          const Obj<typename SendOverlap::coneSequence>& points  = sendOverlap->cone(*r_iter);
          int                                            numVals = 0;

          // TODO: This should be const_iterator, but Sifter sucks
          for(typename SendOverlap::coneSequence::iterator c_iter = points->begin(); c_iter != points->end(); ++c_iter) {
            numVals += sendSection->getFiberDimension(*c_iter);
          }
          typename SendSection::value_type *v = sendAllocator.allocate(numVals);
          int                               k = 0;

          for(int i = 0; i < numVals; ++i) {sendAllocator.construct(v+i, 0);}
          for(typename SendOverlap::coneSequence::iterator c_iter = points->begin(); c_iter != points->end(); ++c_iter) {
            const typename SendSection::value_type *vals = sendSection->restrictPoint(*c_iter);

            for(int i = 0; i < sendSection->getFiberDimension(*c_iter); ++i, ++k) v[k] = vals[i];
          }
          sendValues[*r_iter] = v;
          vMover.send(*r_iter, numVals, sendValues[*r_iter]);
        }
        const Obj<typename RecvOverlap::traits::capSequence> rRanks = recvOverlap->cap();

        recvSection->allocatePoint();
        // TODO: This should be const_iterator, but Sifter sucks
        int maxVals = 0;
        for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rRanks->end(); ++r_iter) {
          const Obj<typename RecvOverlap::supportSequence>& points  = recvOverlap->support(*r_iter);
          int                                               numVals = 0;

          // TODO: This should be const_iterator, but Sifter sucks
          for(typename RecvOverlap::supportSequence::iterator s_iter = points->begin(); s_iter != points->end(); ++s_iter) {
            numVals += recvSection->getFiberDimension(s_iter.color());
          }
          typename SendSection::value_type *v = sendAllocator.allocate(numVals);

          for(int i = 0; i < numVals; ++i) {sendAllocator.construct(v+i, 0);}
          recvValues[*r_iter] = v;
          vMover.recv(*r_iter, numVals, recvValues[*r_iter]);
          maxVals = std::max(maxVals, numVals);
        }
        vMover.start();
        vMover.end();
        typename RecvSection::value_type *convertedValues = recvAllocator.allocate(maxVals);
        for(int i = 0; i < maxVals; ++i) {recvAllocator.construct(convertedValues+i, 0);}
        for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rRanks->end(); ++r_iter) {
          const Obj<typename RecvOverlap::supportSequence>& points = recvOverlap->support(*r_iter);
          const typename SendSection::value_type           *v      = recvValues[*r_iter];
          int                                               k      = 0;

          for(typename RecvOverlap::supportSequence::iterator s_iter = points->begin(); s_iter != points->end(); ++s_iter) {
            const int size = recvSection->getFiberDimension(s_iter.color());

            for(int i = 0; i < size; ++i) {convertedValues[i] = (typename RecvSection::value_type) v[k+i];}
            if (size) {recvSection->updatePoint(s_iter.color(), convertedValues);}
            k += size;
          }
          // TODO: This should use an allocator
          delete [] v;
        }
        delete [] convertedValues;
        for(typename SendOverlap::traits::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sRanks->end(); ++r_iter) {
          // TODO: This should use an allocator
          delete [] sendValues[*r_iter];
        }
        //recvSection->view("After copy");
      };
      // Copy the overlap section to the related processes
      //   This version is for sections with the same type
      template<typename SendOverlap, typename RecvOverlap, typename Section>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<Section>& sendSection, const Obj<Section>& recvSection, const MPI_Datatype datatype = MPI_DATATYPE_NULL) {
        const Obj<typename Section::atlas_type>&      sendAtlas = sendSection->getAtlas();
        const Obj<typename Section::atlas_type>&      recvAtlas = recvSection->getAtlas();
        MPIMover<typename Section::value_type>        vMover(sendSection->comm(), datatype, MPI_UNDEFINED, sendSection->debug());
        std::map<int, typename Section::value_type *> sendValues;
        std::map<int, typename Section::value_type *> recvValues;
        typename Section::alloc_type                  allocator;

        copy(sendOverlap, recvOverlap, sendAtlas, recvAtlas);
        const Obj<typename SendOverlap::traits::baseSequence> sRanks = sendOverlap->base();

        // TODO: This should be const_iterator, but Sifter sucks
        for(typename SendOverlap::traits::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sRanks->end(); ++r_iter) {
          const Obj<typename SendOverlap::coneSequence>& points  = sendOverlap->cone(*r_iter);
          int                                            numVals = 0;

          // TODO: This should be const_iterator, but Sifter sucks
          for(typename SendOverlap::coneSequence::iterator c_iter = points->begin(); c_iter != points->end(); ++c_iter) {
            numVals += sendSection->getFiberDimension(*c_iter);
          }
          typename Section::value_type *v = allocator.allocate(numVals);
          int                           k = 0;

          for(int i = 0; i < numVals; ++i) {allocator.construct(v+i, 0);}
          for(typename SendOverlap::coneSequence::iterator c_iter = points->begin(); c_iter != points->end(); ++c_iter) {
            const typename Section::value_type *vals = sendSection->restrictPoint(*c_iter);

            for(int i = 0; i < sendSection->getFiberDimension(*c_iter); ++i, ++k) v[k] = vals[i];
          }
          sendValues[*r_iter] = v;
          vMover.send(*r_iter, numVals, sendValues[*r_iter]);
        }
        const Obj<typename RecvOverlap::traits::capSequence> rRanks = recvOverlap->cap();

        recvSection->allocatePoint();
        // TODO: This should be const_iterator, but Sifter sucks
        for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rRanks->end(); ++r_iter) {
          const Obj<typename RecvOverlap::supportSequence>& points  = recvOverlap->support(*r_iter);
          int                                               numVals = 0;

          // TODO: This should be const_iterator, but Sifter sucks
          for(typename RecvOverlap::supportSequence::iterator s_iter = points->begin(); s_iter != points->end(); ++s_iter) {
            numVals += recvSection->getFiberDimension(s_iter.color());
          }
          // TODO: This should use an allocator
          recvValues[*r_iter] = allocator.allocate(numVals);
          for(int i = 0; i < numVals; ++i) {allocator.construct(recvValues[*r_iter]+i, 0);}
          vMover.recv(*r_iter, numVals, recvValues[*r_iter]);
        }
        vMover.start();
        vMover.end();
        for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rRanks->end(); ++r_iter) {
          const Obj<typename RecvOverlap::supportSequence>& points = recvOverlap->support(*r_iter);
          const typename Section::value_type               *v      = recvValues[*r_iter];
          int                                               k      = 0;

          for(typename RecvOverlap::supportSequence::iterator s_iter = points->begin(); s_iter != points->end(); ++s_iter) {
            const int size = recvSection->getFiberDimension(s_iter.color());

            if (size) {recvSection->updatePoint(s_iter.color(), &v[k]);}
            k += size;
          }
          // TODO: This should use an allocator
          delete [] v;
        }
        for(typename SendOverlap::traits::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sRanks->end(); ++r_iter) {
          // TODO: This should use an allocator
          delete [] sendValues[*r_iter];
        }
        //recvSection->view("After copy");
      };
      // Specialize to a ConstantSection
      template<typename SendOverlap, typename RecvOverlap, typename Value>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<ConstantSection<typename SendOverlap::source_type, Value> >& sendSection, const Obj<ConstantSection<typename SendOverlap::source_type, Value> >& recvSection) {
        copyConstant(sendOverlap, recvOverlap, sendSection, recvSection);
      };
      // Specialize to an IConstantSection
      template<typename SendOverlap, typename RecvOverlap, typename Value>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<IConstantSection<typename SendOverlap::source_type, Value> >& sendSection, const Obj<IConstantSection<typename SendOverlap::source_type, Value> >& recvSection) {
        copyIConstant(sendOverlap, recvOverlap, sendSection, recvSection);
      };
      // Specialize to an BaseSection/ConstantSection pair
      template<typename SendOverlap, typename RecvOverlap, typename Sieve_>
      static void copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<BaseSection<Sieve_> >& sendSection, const Obj<ConstantSection<typename SendOverlap::source_type, int> >& recvSection) {
        copyConstant(sendOverlap, recvOverlap, sendSection, recvSection);
      };
    };
    class BinaryFusion {
    public:
      template<typename OverlapSection, typename RecvOverlap, typename Section, typename Function>
      static void fuse(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, const Obj<Section>& section, Function binaryOp) {
        const Obj<typename RecvOverlap::traits::baseSequence> rPoints = recvOverlap->base();

        for(typename RecvOverlap::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rPoints->end(); ++p_iter) {
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
      };
    };
    class ReplacementBinaryFusion {
    public:
      template<typename OverlapSection, typename RecvOverlap, typename Section>
      static void fuse(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, const Obj<Section>& section) {
        const Obj<typename RecvOverlap::traits::baseSequence> rPoints = recvOverlap->base();

        for(typename RecvOverlap::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rPoints->end(); ++p_iter) {
          // TODO: This must become a more general iterator over colors
          const Obj<typename RecvOverlap::coneSequence>& points      = recvOverlap->cone(*p_iter);
          // Just taking the first value
          const typename Section::point_type&            localPoint  = *p_iter;
          const typename OverlapSection::point_type&     remotePoint = points->begin().color();

          section->update(localPoint, overlapSection->restrictPoint(remotePoint));
        }
      };
    };
    class AdditiveBinaryFusion {
    public:
      template<typename OverlapSection, typename RecvOverlap, typename Section>
      static void fuse(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, const Obj<Section>& section) {
        const Obj<typename RecvOverlap::traits::baseSequence> rPoints = recvOverlap->base();

        for(typename RecvOverlap::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rPoints->end(); ++p_iter) {
          // TODO: This must become a more general iterator over colors
          const Obj<typename RecvOverlap::coneSequence>& points      = recvOverlap->cone(*p_iter);
          // Just taking the first value
          const typename Section::point_type&            localPoint  = *p_iter;
          const typename OverlapSection::point_type&     remotePoint = points->begin().color();

          section->updateAdd(localPoint, overlapSection->restrictPoint(remotePoint));
        }
      };
    };
    class InsertionBinaryFusion {
    public:
      // Insert the overlapSection values into section along recvOverlap
      template<typename OverlapSection, typename RecvOverlap, typename Section>
      static void fuse(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, const Obj<Section>& section) {
        const Obj<typename RecvOverlap::traits::baseSequence> rPoints = recvOverlap->base();

        for(typename RecvOverlap::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rPoints->end(); ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>& points      = recvOverlap->cone(*p_iter);
          const typename Section::point_type&            localPoint  = *p_iter;
          const typename OverlapSection::point_type&     remotePoint = points->begin().color();

          if (overlapSection->hasPoint(remotePoint)) {section->setFiberDimension(localPoint, overlapSection->getFiberDimension(remotePoint));}
        }
        if (rPoints->size()) {section->allocatePoint();}
        for(typename RecvOverlap::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rPoints->end(); ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>& points      = recvOverlap->cone(*p_iter);
          const typename Section::point_type&            localPoint  = *p_iter;
          const typename OverlapSection::point_type&     remotePoint = points->begin().color();

          if (overlapSection->hasPoint(remotePoint)) {section->updatePoint(localPoint, overlapSection->restrictPoint(remotePoint));}
        }
      };
      // Specialize to the Sieve
      template<typename OverlapSection, typename RecvOverlap, typename Renumbering, typename Point>
      static void fuse(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, Renumbering& renumbering, const Obj<Sieve<Point,Point,int> >& sieve) {
        const Obj<typename RecvOverlap::traits::baseSequence> rPoints = recvOverlap->base();

        for(typename RecvOverlap::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rPoints->end(); ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>& points      = recvOverlap->cone(*p_iter);
          const Point&                                   localPoint  = *p_iter;
          const typename OverlapSection::point_type&     remotePoint = points->begin().color();
          const int                                      size        = overlapSection->getFiberDimension(remotePoint);
          const typename OverlapSection::value_type     *values      = overlapSection->restrictPoint(remotePoint);

          sieve->clearCone(localPoint);
          for(int i = 0; i < size; ++i) {sieve->addCone(renumbering[values[i]], localPoint);}
        }
      };
      template<typename OverlapSection, typename RecvOverlap, typename Section, typename Bundle>
      static void fuse(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, const Obj<Section>& section, const Obj<Bundle>& bundle) {
        const Obj<typename RecvOverlap::traits::baseSequence> rPoints = recvOverlap->base();

        for(typename RecvOverlap::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rPoints->end(); ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>& points      = recvOverlap->cone(*p_iter);
          const typename Section::point_type&            localPoint  = *p_iter;
          const typename OverlapSection::point_type&     remotePoint = points->begin().color();

          section->setFiberDimension(localPoint, overlapSection->getFiberDimension(remotePoint));
        }
        bundle->allocate(section);
        for(typename RecvOverlap::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rPoints->end(); ++p_iter) {
          const Obj<typename RecvOverlap::coneSequence>& points      = recvOverlap->cone(*p_iter);
          const typename Section::point_type&            localPoint  = *p_iter;
          const typename OverlapSection::point_type&     remotePoint = points->begin().color();

          section->update(localPoint, overlapSection->restrictPoint(remotePoint));
        }
      };
    };
  }
}

#endif
