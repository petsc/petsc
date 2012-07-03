#ifndef included_ALE_BasicCommunication_hh
#define included_ALE_BasicCommunication_hh

#ifndef  included_ALE_hh
#include <sieve/ALE.hh>
#endif

extern "C" PetscMPIInt DMMesh_DelTag(MPI_Comm comm,PetscMPIInt keyval,void* attr_val,void* extra_state);

namespace ALE {
  // TODO: Check status of Waits
  template<typename Value_>
  class MPIMover : public ParallelObject {
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
        int ierr = MPI_Type_free(&this->_datatype);CHKERRXX(ierr);
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
      } else if (sizeof(value_type) == 16) {
        int          blen[1], ierr;
        MPI_Aint     indices[1];
        MPI_Datatype oldtypes[1], newtype;
        blen[0] = 4; indices[0] = 0;           oldtypes[0] = MPI_INT;
        ierr = MPI_Type_struct(1, blen, indices, oldtypes, &newtype);CHKERRXX(ierr);
        ierr = MPI_Type_commit(&newtype);CHKERRXX(ierr);
        this->_createdType = true;
        return newtype;
      } else if (sizeof(value_type) == 28) {
        int          blen[2], ierr;
        MPI_Aint     indices[2];
        MPI_Datatype oldtypes[2], newtype;
        blen[0] = 1; indices[0] = 0;           oldtypes[0] = MPI_INT;
        blen[1] = 3; indices[1] = sizeof(int); oldtypes[1] = MPI_DOUBLE;
        ierr = MPI_Type_struct(2, blen, indices, oldtypes, &newtype);CHKERRXX(ierr);
        ierr = MPI_Type_commit(&newtype);CHKERRXX(ierr);
        this->_createdType = true;
        return newtype;
      } else if (sizeof(value_type) == 32) {
        int          blen[2], ierr;
        MPI_Aint     indices[2];
        MPI_Datatype oldtypes[2], newtype;
        blen[0] = 1; indices[0] = 0;           oldtypes[0] = MPI_DOUBLE;
        blen[1] = 3; indices[1] = sizeof(int); oldtypes[1] = MPI_DOUBLE;
        ierr = MPI_Type_struct(2, blen, indices, oldtypes, &newtype);CHKERRXX(ierr);
        ierr = MPI_Type_commit(&newtype);CHKERRXX(ierr);
        this->_createdType = true;
        return newtype;
      }
      ostringstream msg;

      msg << "Cannot determine MPI type for value type with size " << sizeof(value_type);
      throw PETSc::Exception(msg.str().c_str());
    };
    int getNewTag() const {
      static int tagKeyval = MPI_KEYVAL_INVALID;
      int *tagvalp = NULL, *maxval, flg, ierr;

      if (tagKeyval == MPI_KEYVAL_INVALID) {
        tagvalp = (int *) malloc(sizeof(int));
        ierr = MPI_Keyval_create(MPI_NULL_COPY_FN, DMMesh_DelTag, &tagKeyval, (void *) NULL);CHKERRXX(ierr);
        ierr = MPI_Attr_put(this->comm(), tagKeyval, tagvalp);CHKERRXX(ierr);
        tagvalp[0] = 0;
      }
      ierr = MPI_Attr_get(this->comm(), tagKeyval, (void **) &tagvalp, &flg);CHKERRXX(ierr);
      if (tagvalp[0] < 1) {
        ierr = MPI_Attr_get(MPI_COMM_WORLD, MPI_TAG_UB, (void **) &maxval, &flg);CHKERRXX(ierr);
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
        int         ierr;

        if (this->_debug) {std::cout <<"["<<this->commRank()<<"] Sending data (" << num << ") to " << rank << " tag " << this->_tag << std::endl;}
        ierr = MPI_Send_init(data, num, this->_datatype, rank, this->_tag, this->comm(), &request);CHKERRXX(ierr);
        this->_requests.push_back(request);
#if defined(PETSC_USE_LOG)
        // PETSc logging
        petsc_isend_ct++;
        PetscMPITypeSize(&petsc_isend_len, num, this->_datatype);
#endif
      }
      for(typename moves_type::const_iterator r_iter = this->_recvs.begin(); r_iter != this->_recvs.end(); ++r_iter) {
        const int   rank = r_iter->first;
        const int   num  = r_iter->second.first;
        void       *data = (void *) r_iter->second.second;
        MPI_Request request;
        int         ierr;

        if (this->_debug) {std::cout <<"["<<this->commRank()<<"] Receiving data (" << num << ") from " << rank << " tag " << this->_tag << std::endl;}
        ierr = MPI_Recv_init(data, num, this->_datatype, rank, this->_tag, this->comm(), &request);CHKERRXX(ierr);
        this->_requests.push_back(request);
#if defined(PETSC_USE_LOG)
        // PETSc logging
        petsc_irecv_ct++;
        PetscMPITypeSize(&petsc_irecv_len, num, this->_datatype);
#endif
      }
    };
  public:
    MPI_Datatype getType() {return this->_datatype;};
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

        int ierr = MPI_Start(&request);CHKERRXX(ierr);
      }
    };
    void end() {
      MPI_Status status;

      for(typename requests_type::const_iterator r_iter = this->_requests.begin(); r_iter != this->_requests.end(); ++r_iter) {
        MPI_Request request = *r_iter;

        int ierr = MPI_Wait(&request, &status);CHKERRXX(ierr);
      }
      for(typename requests_type::const_iterator r_iter = this->_requests.begin(); r_iter != this->_requests.end(); ++r_iter) {
        MPI_Request request = *r_iter;

        int ierr = MPI_Request_free(&request);CHKERRXX(ierr);
      }
    };
  };
}
#endif
