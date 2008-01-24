#ifndef included_ALE_ParallelMapping_hh
#define included_ALE_ParallelMapping_hh

#ifndef  included_ALE_IField_hh
#include <IField.hh>
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
    ~MPIMover() {
      if (_createdType) {
        MPI_Type_free(&this->_datatype);
      }
    };
  protected:
    MPI_Datatype getMPIDatatype() {
      if (sizeof(value_type) == 4) {
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
  class ParallelPullback {
  public:
    // Copy the overlap section to the related processes
    template<typename SendOverlap, typename RecvOverlap, typename Section>
    static Obj<Section> copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<Section>& sendSection) {
      const Obj<typename Section::atlas_type>& sendAtlas   = sendSection->getAtlas();
      const Obj<typename Section::atlas_type>& recvAtlas   = copy(sendOverlap, recvOverlap, sendAtlas);
      Obj<Section>                             recvSection = new Section(recvAtlas);
      MPIMover<typename Section::value_type>   vMover(sendSection->comm(), sendSection->debug());

      const Obj<typename SendOverlap::traits::baseSequence> sRanks = sendOverlap->base();

      // TODO: This should be const_iterator, but Sifter sucks
      for(typename SendOverlap::traits::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sRanks->end(); ++r_iter) {
        const Obj<typename SendOverlap::coneSequence>& points  = sendOverlap->cone(*r_iter);
        int                                            numVals = 0;

        // TODO: This should be const_iterator, but Sifter sucks
        for(typename SendOverlap::coneSequence::iterator c_iter = points->begin(); c_iter != points->end(); ++c_iter) {
          numVals += sendSection->getFiberDimension(*c_iter);
        }
        vMover.send(*r_iter, numVals, sendSection->restrict());
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
        vMover.recv(*r_iter, numVals, recvSection->restrict());
      }
      vMover.start();
      vMover.end();
      return recvSection;
    };
    // Specialize to an ConstantSection
    template<typename SendOverlap, typename RecvOverlap, typename Value>
    static Obj<ConstantSection<typename SendOverlap::source_type, Value> > copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<ConstantSection<typename SendOverlap::source_type, Value> >& sendSection) {
      typedef ConstantSection<typename SendOverlap::source_type, Value> Section;
      Obj<Section>    recvSection = new Section(sendSection->comm(), sendSection->debug());
      MPIMover<Value> vMover(sendSection->comm(), sendSection->debug());

      const Obj<typename SendOverlap::traits::baseSequence> sRanks = sendOverlap->base();
      const typename SendOverlap::source_type               p      = *sendOverlap->cone(*sRanks->begin())->begin();

      for(typename SendOverlap::traits::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sRanks->end(); ++r_iter) {
        vMover.send(*r_iter, 1, sendSection->restrict(p));
      }
      const Obj<typename RecvOverlap::traits::capSequence> rRanks = recvOverlap->cap();
      const typename SendOverlap::target_type              q      = *recvOverlap->support(*rRanks->begin())->begin();

      for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rRanks->end(); ++r_iter) {
        const Obj<typename RecvOverlap::traits::supportSequence> sPoints = recvOverlap->support(*r_iter);

        for(typename RecvOverlap::traits::supportSequence::iterator s_iter = sPoints->begin(); s_iter != sPoints->end(); ++s_iter) {
          recvSection->addPoint(s_iter.color());
        }
        vMover.recv(*r_iter, 1, recvSection->restrict(q));
      }
      vMover.start();
      vMover.end();
      return recvSection;
    };
    // Specialize to an FauxConstantSection
    template<typename SendOverlap, typename RecvOverlap, typename Value>
    static Obj<FauxConstantSection<typename SendOverlap::source_type, Value> > copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<FauxConstantSection<typename SendOverlap::source_type, Value> >& sendSection) {
      typedef FauxConstantSection<typename SendOverlap::source_type, Value> Section;
      Obj<Section>    recvSection = new Section(sendSection->comm(), sendSection->debug());
      MPIMover<Value> vMover(sendSection->comm(), sendSection->debug());

      const Obj<typename SendOverlap::traits::baseSequence> sRanks = sendOverlap->base();
      const typename SendOverlap::source_type               p      = *sendOverlap->cone(*sRanks->begin())->begin();

      for(typename SendOverlap::traits::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sRanks->end(); ++r_iter) {
        vMover.send(*r_iter, 1, sendSection->restrict(p));
      }
      const Obj<typename RecvOverlap::traits::capSequence> rRanks = recvOverlap->cap();
      const typename SendOverlap::target_type              q      = *recvOverlap->support(*rRanks->begin())->begin();

      for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rRanks->end(); ++r_iter) {
        const Obj<typename RecvOverlap::traits::supportSequence> sPoints = recvOverlap->support(*r_iter);

        for(typename RecvOverlap::traits::supportSequence::iterator s_iter = sPoints->begin(); s_iter != sPoints->end(); ++s_iter) {
          recvSection->addPoint(s_iter.color());
        }
        vMover.recv(*r_iter, 1, recvSection->restrict(q));
      }
      vMover.start();
      vMover.end();
      recvSection->view("");
      return recvSection;
    };
    // Specialize to an IConstantSection
    template<typename SendOverlap, typename RecvOverlap, typename Value>
    static Obj<IConstantSection<typename SendOverlap::source_type, Value> > copy(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<IConstantSection<typename SendOverlap::source_type, Value> >& sendSection) {
      typedef IConstantSection<typename SendOverlap::source_type, Value> Section;
      Obj<Section>    recvSection = new Section(sendSection->comm(), sendSection->debug());
      MPIMover<Value> vMover(sendSection->comm(), sendSection->debug());

      const Obj<typename SendOverlap::traits::baseSequence> sRanks = sendOverlap->base();
      const typename SendOverlap::source_type               p      = *sendOverlap->cone(*sRanks->begin())->begin();

      for(typename SendOverlap::traits::baseSequence::iterator r_iter = sRanks->begin(); r_iter != sRanks->end(); ++r_iter) {
        vMover.send(*r_iter, 1, sendSection->restrict(p));
      }
      const Obj<typename RecvOverlap::traits::capSequence> rRanks = recvOverlap->cap();
      const typename SendOverlap::target_type              q      = *recvOverlap->support(*rRanks->begin())->begin();

      for(typename RecvOverlap::traits::capSequence::iterator r_iter = rRanks->begin(); r_iter != rRanks->end(); ++r_iter) {
        vMover.recv(*r_iter, 1, recvSection->restrict(q));
      }
      vMover.start();
      vMover.end();
    };
  };
}

#endif
