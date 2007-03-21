#ifndef included_ALE_CartesianSieve_hh
#define included_ALE_CartesianSieve_hh

#ifndef  included_ALE_CoSieve_hh
#include <CoSieve.hh>
#endif

namespace ALE {
  namespace CartesianSieveDef {
    static void nextCodeOdd(int a[], const int n, const int size) {
      for(int d = size-2; d >= 0; --d) a[d+1] = a[d];
      if (n == (n/2)*2) {
        a[0] = 1;
      } else {
        a[0] = 0;
      }
    };
    static void nextCodeEven(int a[], const int n, const int size) {
      ALE::CartesianSieveDef::nextCodeOdd(a, n, size);
      a[0] ^= 1;
      for(int d = 1; d < size; ++d) a[d] ^= 0;
    };
    static void getGrayCode(const int n, const int size, int code[]) {
      if (n == 1 << size) {
        code[0] = -1;
      } else if (n == 0) {
        for(int d = 0; d < size; ++d) code[d] = 0;
      } else if (n == 1) {
        code[0] = 1;
        for(int d = 1; d < size; ++d) code[d] = 0;
      } else if (n == (n/2)*2) {
        // Might have to shift pointer
        ALE::CartesianSieveDef::getGrayCode(n/2, size, code);
        ALE::CartesianSieveDef::nextCodeEven(code, n/2, size);
      } else {
        // Might have to shift pointer
        ALE::CartesianSieveDef::getGrayCode((n-1)/2, size, code);
        ALE::CartesianSieveDef::nextCodeOdd(code, (n-1)/2, size);
      }
    };
    static void getIndices(const int dim, const int sizes[], const int& point, int indices[]) {
      int p       = point;
      int divisor = 1;

      for(int d = 0; d < dim-1; ++d) {
        divisor *= sizes[d];
      }
      //std::cout << "  Getting indices for point " << point << ":";
      for(int d = dim-1; d >= 0; --d) {
        indices[d] = p/divisor;
        p -= divisor*indices[d];
        if (d > 0) divisor /= sizes[d-1];
      }
      //for(int d = 0; d < dim; ++d) {
      //  std::cout << " " << indices[d];
      //}
      //std::cout << std::endl;
    };
    static inline int getValue(const int dim, const int sizes[], const int indices[], const int code[], const bool forward = true) {
      //std::cout << "Building value" << std::endl << "  code:";
      //for(int d = 0; d < dim; ++d) std::cout << " " << code[d];
      //std::cout << std::endl << "  indices:";
      //for(int d = 0; d < dim; ++d) std::cout << " " << indices[d];
      //std::cout << std::endl;
      if (code[0] < 0) {
        //std::cout << "  value " << -1 << std::endl;
        return -1;
      }
      int value = indices[dim-1];
      if (forward) {
        if (code[dim-1]) value++;
      } else {
        if (!code[dim-1]) value--;
      }
      //std::cout << "  value " << value << std::endl;
      for(int d = dim-2; d >= 0; --d) {
        value = value*sizes[d] + indices[d];
        if (forward) {
          if (code[d]) value++;
        } else {
          if (!code[d]) value--;
        }
        //std::cout << "  value " << value << std::endl;
      }
      //std::cout << "  final value " << value << std::endl;
      return value;
    };


    template <typename PointType_>
    class PointSequence {
    public:
      typedef PointType_ point_type;
      // We traverse the points in natural order
      class iterator {
      public:
        // Standard iterator typedefs
        typedef std::input_iterator_tag iterator_category;
        typedef point_type              value_type;
        typedef int                     difference_type;
        typedef value_type*             pointer;
        typedef value_type&             reference;
      protected:
        value_type _value;
      public:
        iterator(const value_type& start) : _value(start) {};
        iterator(const iterator& iter) : _value(iter._value) {};
        virtual ~iterator() {};
      public:
        virtual bool             operator==(const iterator& iter) const {return this->_value == iter._value;};
        virtual bool             operator!=(const iterator& iter) const {return this->_value != iter._value;};
        virtual const value_type operator*() const {return this->_value;};
        virtual iterator&        operator++() {++this->_value; return *this;};
        virtual iterator         operator++(const int n) {iterator tmp(*this); ++this->_value; return tmp;};
      };
    protected:
      point_type _start;
      point_type _end;
    public:
      PointSequence(const point_type& start, const point_type& end) : _start(start), _end(end) {};
      virtual ~PointSequence() {};
    public:
      virtual iterator begin() {return iterator(this->_start);};
      virtual iterator end()   {return iterator(this->_end);};
      virtual bool     empty() {return this->_start >= this->_end;};
    };
    // In one dimension, we have
    //
    //   v_k---c_k---v_{k+1}
    //
    // In two dimensions, we have
    //
    //            c_{k+m}             c_{k+m+1}
    //
    // v_{k+m+1}-----------v_{k+m+2}--
    //    |                    |
    //    |       c_k          |      c_{k+1}
    //    |                    |
    //   v_k---------------v_{k+1}----
    //
    // So for c_k the cone is (v_{k+(k/m)}, v_{k+(k/m)+1}, v_{k+m+(k/m)}, v_{k+m+(k/m)+1})
    //
    // In three dimensions, we have
    //
    //         v_{k+m+1+(m+1)*(n+1)} v_{k+m+1+(m+1)*(n+1)+1}
    //
    // v_{k+m+1}           v_{k+m+2}
    //
    //
    //         v_{k+(m+1)*(n+1)}     v_{k+(m+1)*(n+1)+1}
    //
    //   v_k               v_{k+1}
    //
    // Suppose we break down the index k into a tuple (i,j), then 2d becomes
    //
    //   cone(c_{i,j}) = (v_{i,j}, v_{i+1,j}, v_{i,j+1}, v_{i+1,j+1})
    //                 = (v_{k}, v_{k+1}, v_{k+m+1}, v_{k+m+2})
    // Now for 2D
    //   i = k%m
    //   j = k/m
    // and 3D
    //   k = q/(m*n)
    //   j = (q - m*n*k)/m
    //   i = (q - m*n*k - m*j)
    template <typename PointType_>
    class ConeSequence {
    public:
      typedef PointType_ point_type;
      // We traverse the points in natural order
      class iterator {
      public:
        // Standard iterator typedefs
        typedef std::input_iterator_tag iterator_category;
        typedef PointType_              value_type;
        typedef int                     difference_type;
        typedef value_type*             pointer;
        typedef value_type&             reference;
      protected:
        int               _dim;
        const int        *_sizes;
        int               _numCells;
        int               _pos;
        int              *_code;
        value_type        _value;
        value_type *_indices;
      protected:
        void init() {
          this->_code = new int[this->_dim];
          ALE::CartesianSieveDef::getGrayCode(this->_pos, this->_dim, this->_code);
          this->_value = ALE::CartesianSieveDef::getValue(this->_dim, this->_sizes, this->_indices, this->_code) + this->_numCells;
        };
      public:
        iterator(const int dim, const int sizes[], const int numCells, const value_type indices[], const int pos) : _dim(dim), _sizes(sizes), _numCells(numCells), _pos(pos) {
          this->_indices = new int[this->_dim];
          for(int d = 0; d < this->_dim; ++d) this->_indices[d] = indices[d];
          this->init();
        };
        iterator(const iterator& iter) : _dim(iter._dim), _sizes(iter._sizes), _numCells(iter._numCells), _pos(iter._pos) {
          this->_indices = new int[this->_dim];
          for(int d = 0; d < this->_dim; ++d) this->_indices[d] = iter._indices[d];
          this->init();
        };
        virtual ~iterator() {
          delete [] this->_code;
          delete [] this->_indices;
        };
      public:
        virtual bool              operator==(const iterator& iter) const {return this->_pos == iter._pos;};
        virtual bool              operator!=(const iterator& iter) const {return this->_pos != iter._pos;};
        virtual const value_type  operator*() const {return this->_value;};
        virtual iterator&         operator++() {
          ALE::CartesianSieveDef::getGrayCode(++this->_pos, this->_dim, this->_code);
          this->_value = ALE::CartesianSieveDef::getValue(this->_dim, this->_sizes, this->_indices, this->_code) + this->_numCells;
          return *this;
        };
        virtual iterator          operator++(int n) {
          iterator tmp(*this);
          ALE::CartesianSieveDef::getGrayCode(++this->_pos, this->_dim, this->_code);
          this->_value = ALE::CartesianSieveDef::getValue(this->_dim, this->_sizes, this->_indices, this->_code) + this->_numCells;
          return tmp;
        };
      };
    protected:
      int         _dim;
      const int  *_sizes;
      int        *_vertexSizes;
      int         _numCells;
      point_type  _cell;
      int        *_indices;
    public:
      ConeSequence(const int dim, const int sizes[], const int numCells, const point_type& cell) : _dim(dim), _sizes(sizes), _numCells(numCells), _cell(cell) {
        this->_vertexSizes = new int[dim];
        this->_indices     = new point_type[dim];
        //this->getIndices(this->_cell, this->_indices);
        for(int d = 0; d < this->_dim; ++d) this->_vertexSizes[d] = this->_sizes[d]+1;
        ALE::CartesianSieveDef::getIndices(this->_dim, this->_sizes, this->_cell, this->_indices);
      };
      virtual ~ConeSequence() {
        delete [] this->_vertexSizes;
        delete [] this->_indices;
      };
    protected:
      void getIndices(const point_type& cell, const int indices[]) {
        point_type c       = cell;
        int        divisor = 1;

        for(int d = 0; d < this->_dim-1; ++d) {
          divisor *= this->_sizes[d];
        }
        //std::cout << "  Got indices for cell " << cell << ":";
        for(int d = this->_dim-1; d >= 0; --d) {
          this->_indices[d] = c/divisor;
          c -= divisor*this->_indices[d];
          if (d > 0) divisor /= this->_sizes[d-1];
        }
        //for(int d = 0; d < this->_dim; ++d) {
        //  std::cout << " " << this->_indices[d];
        //}
        //std::cout << std::endl;
      };
    public:
      virtual iterator begin() {return iterator(this->_dim, this->_vertexSizes, this->_numCells, this->_indices, 0);};
      virtual iterator end()   {return iterator(this->_dim, this->_vertexSizes, this->_numCells, this->_indices, 1 << this->_dim);};
    };
    template <typename PointType_>
    class SupportSequence {
    public:
      typedef PointType_ point_type;
      // We traverse the points in natural order
      class iterator {
      public:
        // Standard iterator typedefs
        typedef std::input_iterator_tag iterator_category;
        typedef PointType_              value_type;
        typedef int                     difference_type;
        typedef value_type*             pointer;
        typedef value_type&             reference;
      protected:
        int               _dim;
        const int        *_sizes;
        int               _pos;
        int              *_code;
        value_type        _value;
        value_type *_indices;
      protected:
        bool validIndex(const int indices[], const int code[]) {
          for(int d = 0; d < this->_dim; ++d) {
            if ((code[d])  && (indices[d] >= this->_sizes[d])) return false;
            if ((!code[d]) && (indices[d] < 1)) return false;
          }
          return true;
        }
        void init() {
          this->_code = new int[this->_dim];
          ALE::CartesianSieveDef::getGrayCode(this->_pos, this->_dim, this->_code);
          while((this->_code[0] >= 0) && !this->validIndex(this->_indices, this->_code)) {
            ALE::CartesianSieveDef::getGrayCode(++this->_pos, this->_dim, this->_code);
          } 
          this->_value = ALE::CartesianSieveDef::getValue(this->_dim, this->_sizes, this->_indices, this->_code, false);
        };
      public:
        iterator(const int dim, const int sizes[], const value_type indices[], const int pos) : _dim(dim), _sizes(sizes), _pos(pos) {
          this->_indices = new int[this->_dim];
          for(int d = 0; d < this->_dim; ++d) this->_indices[d] = indices[d];
          this->init();
        };
        iterator(const iterator& iter) : _dim(iter._dim), _sizes(iter._sizes), _pos(iter._pos) {
          this->_indices = new int[this->_dim];
          for(int d = 0; d < this->_dim; ++d) this->_indices[d] = iter._indices[d];
          this->init();
        };
        virtual ~iterator() {
          delete [] this->_code;
          delete [] this->_indices;
        };
      public:
        virtual bool              operator==(const iterator& iter) const {return this->_pos == iter._pos;};
        virtual bool              operator!=(const iterator& iter) const {return this->_pos != iter._pos;};
        virtual const value_type  operator*() const {return this->_value;};
        virtual iterator&         operator++() {
          do {
            ALE::CartesianSieveDef::getGrayCode(++this->_pos, this->_dim, this->_code);
          } while((this->_code[0] >= 0) && !this->validIndex(this->_indices, this->_code));
          this->_value = ALE::CartesianSieveDef::getValue(this->_dim, this->_sizes, this->_indices, this->_code, false);
          return *this;
        };
        virtual iterator          operator++(int n) {
          iterator tmp(*this);
          do {
            ALE::CartesianSieveDef::getGrayCode(++this->_pos, this->_dim, this->_code);
          } while((this->_code[0] >= 0) && !this->validIndex(this->_indices, this->_code));
          this->_value = ALE::CartesianSieveDef::getValue(this->_dim, this->_sizes, this->_indices, this->_code, false);
          return tmp;
        };
      };
    protected:
      int         _dim;
      const int  *_sizes;
      int        *_vertexSizes;
      int         _numCells;
      point_type  _vertex;
      int        *_indices;
    public:
      SupportSequence(const int dim, const int sizes[], const int numCells, const point_type& vertex) : _dim(dim), _sizes(sizes), _numCells(numCells), _vertex(vertex) {
        this->_vertexSizes = new int[dim];
        this->_indices     = new point_type[dim];
        //this->getIndices(this->_vertex, this->_indices);
        for(int d = 0; d < this->_dim; ++d) this->_vertexSizes[d] = this->_sizes[d]+1;
        ALE::CartesianSieveDef::getIndices(this->_dim, this->_vertexSizes, this->_vertex - this->_numCells, this->_indices);
      };
      virtual ~SupportSequence() {
        delete [] this->_vertexSizes;
        delete [] this->_indices;
      };
    protected:
      void getIndices(const point_type& vertex, const int indices[]) {
        point_type v       = vertex - this->_numCells;
        int        divisor = 1;

        for(int d = 0; d < this->_dim-1; ++d) {
          divisor *= this->_sizes[d]+1;
        }
        std::cout << "  Got indices for vertex " << vertex << ":";
        for(int d = this->_dim-1; d >= 0; --d) {
          this->_indices[d] = v/divisor;
          v -= divisor*this->_indices[d];
          if (d > 0) divisor /= this->_sizes[d-1]+1;
        }
        for(int d = 0; d < this->_dim; ++d) {
          std::cout << " " << this->_indices[d];
        }
        std::cout << std::endl;
      };
    public:
      virtual iterator begin() {return iterator(this->_dim, this->_sizes, this->_indices, 0);};
      virtual iterator end()   {return iterator(this->_dim, this->_sizes, this->_indices, 1 << this->_dim);};
    };
  }
  // We can do meets of two cells as empty if they do not intersect, and a k-dimensional face (with 2^k vertices)
  //   if they have k indices in common
  // We can do joins of two vertices. If they have have k indices in common, they span a d-k face, and thus have
  //   2^{k} cells in the join.
  // Note meet and join are both empty if any index differs by more than 1

  // This is a purely Cartesian mesh
  //   We will only represent cells and vertices, meaning the cap consists only of
  //     vertices, and the base only of cells.
  //   numCells:    m x n x ...
  //   numVertices: m+1 x n+1 x ...
  //   Points:
  //     cells:       0     - numCells-1
  //     vertices: numCells - numCells+numVertices-1
  template <typename Point_>
  class CartesianSieve : public ALE::ParallelObject {
  public:
    typedef Point_ point_type;
    typedef CartesianSieveDef::PointSequence<point_type> baseSequence;
    typedef CartesianSieveDef::PointSequence<point_type> capSequence;
    typedef CartesianSieveDef::ConeSequence<point_type>  coneSequence;
    typedef CartesianSieveDef::SupportSequence<point_type>  supportSequence;
  protected:
    int  _dim;
    int *_sizes;
    int  _numCells;
    int  _numVertices;
  public:
    CartesianSieve(MPI_Comm comm, const int dim, const int numCells[], const int& debug = 0) : ParallelObject(comm, debug), _dim(dim) {
      this->_sizes       = new int[dim];
      this->_numCells    = 1;
      this->_numVertices = 1;
      for(int d = 0; d < dim; ++d) {
        this->_sizes[d]     = numCells[d];
        this->_numCells    *= numCells[d];
        this->_numVertices *= numCells[d]+1;
      }
    };
    virtual ~CartesianSieve() {
      delete [] this->_sizes;
    };
  public:
    int getDimension() const {return this->_dim;};
    const int *getSizes() const {return this->_sizes;};
    int getNumCells() const {return this->_numCells;};
    int getNumVertices() const {return this->_numVertices;};
  public:
    Obj<baseSequence> base() {
      Obj<baseSequence>base = new baseSequence(0, this->_numCells);
      return base;
    };
    Obj<capSequence> cap() {
       Obj<capSequence> cap = capSequence(this->_numCells, this->_numCells+this->_numVertices);
       return cap;
    };
    // WARNING: Creating a lot of objects here
    Obj<coneSequence> cone(const point_type& p) {
      if ((p < 0) || (p >= this->_numCells)) {
        Obj<coneSequence> cone = new coneSequence(0, this->_sizes, this->_numCells, 0);
        return cone;
      }
      Obj<coneSequence> cone = new coneSequence(this->_dim, this->_sizes, this->_numCells, p);
      return cone;
    };
    Obj<supportSequence> support(const point_type& p) {
      if ((p < this->_numCells) || (p >= this->_numCells+this->_numVertices)) {
        Obj<supportSequence> support = new supportSequence(0, this->_sizes, this->_numCells, 0);
        return support;
      }
      Obj<supportSequence> support = new supportSequence(this->_dim, this->_sizes, this->_numCells, p);
      return support;
    };
  public:
    void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) {
        ostringstream txt;

      if (comm == MPI_COMM_NULL) {
        comm = this->comm();
      }
      if (name == "") {
        PetscPrintf(comm, "viewing a CartesianSieve\n");
      } else {
        PetscPrintf(comm, "viewing CartesianSieve '%s'\n", name.c_str());
      }
      if(this->commRank() == 0) {
        txt << "cap --> base:\n";
      }
      Obj<capSequence>  cap  = this->cap();
      Obj<baseSequence> base = this->base();
      if(cap->empty()) {
        txt << "[" << this->commRank() << "]: empty" << std::endl; 
      }
      for(typename capSequence::iterator capi = cap->begin(); capi != cap->end(); ++capi) {
        const Obj<supportSequence>&              supp    = this->support(*capi);
        const typename supportSequence::iterator suppEnd = supp->end();

        for(typename supportSequence::iterator suppi = supp->begin(); suppi != suppEnd; ++suppi) {
          txt << "[" << this->commRank() << "]: " << *capi << "---->" << *suppi << std::endl;
        }
      }
      PetscSynchronizedPrintf(this->comm(), txt.str().c_str());
      PetscSynchronizedFlush(this->comm());
      //
      ostringstream txt1;
      if(this->commRank() == 0) {
        txt1 << "base --> cap:\n";
      }
      if(base->empty()) {
        txt1 << "[" << this->commRank() << "]: empty" << std::endl; 
      }
      for(typename baseSequence::iterator basei = base->begin(); basei != base->end(); ++basei) {
        const Obj<coneSequence>&              cone    = this->cone(*basei);
        const typename coneSequence::iterator coneEnd = cone->end();

        for(typename coneSequence::iterator conei = cone->begin(); conei != coneEnd; ++conei) {
          txt1 << "[" << this->commRank() << "]: " << *basei << "<----" << *conei << std::endl;
        }
      }
      //
      PetscSynchronizedPrintf(this->comm(), txt1.str().c_str());
      PetscSynchronizedFlush(this->comm());
      //
      ostringstream txt2;
      if(this->commRank() == 0) {
        txt2 << "cap <point>:\n";
      }
      txt2 << "[" << this->commRank() << "]:  [";
      for(typename capSequence::iterator capi = cap->begin(); capi != cap->end(); ++capi) {
        txt2 << " <" << *capi << ">";
      }
      txt2 << " ]" << std::endl;
      //
      PetscSynchronizedPrintf(this->comm(), txt2.str().c_str());
      PetscSynchronizedFlush(this->comm());
      //
      ostringstream txt3;
      if(this->commRank() == 0) {
        txt3 << "base <point>:\n";
      }
      txt3 << "[" << this->commRank() << "]:  [";
      for(typename baseSequence::iterator basei = base->begin(); basei != base->end(); ++basei) {
        txt3 << " <" << *basei << ">";
      }
      txt3 << " ]" << std::endl;
      //
      PetscSynchronizedPrintf(this->comm(), txt3.str().c_str());
      PetscSynchronizedFlush(this->comm());
    };
  };

  // We do not just use Topology, because we need to optimize labels
  template<typename Patch_>
  class CartesianTopology : public ALE::ParallelObject {
  public:
    typedef Patch_                                          patch_type;
    typedef CartesianSieve<int>                             sieve_type;
    typedef typename sieve_type::point_type                 point_type;
    typedef typename std::map<patch_type, Obj<sieve_type> > sheaf_type;
    typedef typename std::map<patch_type, int>              max_label_type;
    typedef typename sieve_type::baseSequence               label_sequence;
    typedef typename ALE::Sifter<int,point_type,point_type> send_overlap_type;
    typedef typename ALE::Sifter<point_type,int,point_type> recv_overlap_type;
  protected:
    sheaf_type             _sheaf;
    int                    _maxHeight;
    max_label_type         _maxHeights;
    int                    _maxDepth;
    max_label_type         _maxDepths;
    Obj<send_overlap_type> _sendOverlap;
    Obj<recv_overlap_type> _recvOverlap;
  public:
    CartesianTopology(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug), _maxHeight(-1), _maxDepth(-1) {
      this->_sendOverlap = new send_overlap_type(this->comm(), this->debug());
      this->_recvOverlap = new recv_overlap_type(this->comm(), this->debug());
    };
    virtual ~CartesianTopology() {};
    public: // Verifiers
      void checkPatch(const patch_type& patch) {
        if (this->_sheaf.find(patch) == this->_sheaf.end()) {
          ostringstream msg;
          msg << "Invalid topology patch: " << patch << std::endl;
          throw ALE::Exception(msg.str().c_str());
        }
      };
      bool hasPatch(const patch_type& patch) {
        if (this->_sheaf.find(patch) != this->_sheaf.end()) {
          return true;
        }
        return false;
      };
    public: // Accessors
      const Obj<sieve_type>& getPatch(const patch_type& patch) {
        this->checkPatch(patch);
        return this->_sheaf[patch];
      };
      void setPatch(const patch_type& patch, const Obj<sieve_type>& sieve) {
        this->_sheaf[patch] = sieve;
      };
      const sheaf_type& getPatches() {
        return this->_sheaf;
      };
      void clear() {
        this->_sheaf.clear();
        this->_maxHeight = -1;
        this->_maxHeights.clear();
        this->_maxDepth = -1;
        this->_maxDepths.clear();
      };
      const Obj<send_overlap_type>& getSendOverlap() const {return this->_sendOverlap;};
      void setSendOverlap(const Obj<send_overlap_type>& overlap) {this->_sendOverlap = overlap;};
      const Obj<recv_overlap_type>& getRecvOverlap() const {return this->_recvOverlap;};
      void setRecvOverlap(const Obj<recv_overlap_type>& overlap) {this->_recvOverlap = overlap;};
      const Obj<send_overlap_type>& getDistSendOverlap() const {return this->_distSendOverlap;};
      void setDistSendOverlap(const Obj<send_overlap_type>& overlap) {this->_distSendOverlap = overlap;};
      const Obj<recv_overlap_type>& getDistRecvOverlap() const {return this->_distRecvOverlap;};
      void setDistRecvOverlap(const Obj<recv_overlap_type>& overlap) {this->_distRecvOverlap = overlap;};
  public: // Stratification
    #undef __FUNCT__
    #define __FUNCT__ "Topology::stratify"
    void stratify() {
      ALE_LOG_EVENT_BEGIN;
      this->_maxHeight = -1;
      this->_maxDepth  = -1;
      for(typename sheaf_type::iterator s_iter = this->_sheaf.begin(); s_iter != this->_sheaf.end(); ++s_iter) {
        this->_maxHeights[s_iter->first] = 1;
        this->_maxHeight                 = 1;
        this->_maxDepths[s_iter->first]  = 1;
        this->_maxDepth                  = 1;
      
      }
      ALE_LOG_EVENT_END;
    };
    int height() const {return this->_maxHeight;};
    int height(const patch_type& patch) {
      this->checkPatch(patch);
      return this->_maxHeights[patch];
    };
    int height(const patch_type& patch, const point_type& point) {
      const int numCells    = this->_sheaf[patch]->getNumCells();
      const int numVertices = this->_sheaf[patch]->getNumVertices();
      if ((point >= 0)        && (point < numCells))             return 0;
      if ((point >= numCells) && (point < numCells+numVertices)) return 1;
      return -1;
    };
    const Obj<label_sequence>& heightStratum(const patch_type& patch, int height) {
      if (height == 0) return this->_sheaf[patch]->base();
      if (height == 1) return this->_sheaf[patch]->cap();
      throw ALE::Exception("Invalid height stratum");
    };
    int depth() const {return this->_maxDepth;};
    int depth(const patch_type& patch) {
      this->checkPatch(patch);
      return this->_maxDepths[patch];
    };
    int depth(const patch_type& patch, const point_type& point) {
      const int numCells    = this->_sheaf[patch]->getNumCells();
      const int numVertices = this->_sheaf[patch]->getNumVertices();
      if ((point >= 0)        && (point < numCells))             return 1;
      if ((point >= numCells) && (point < numCells+numVertices)) return 0;
      return -1;
    };
    const Obj<label_sequence>& depthStratum(const patch_type& patch, int depth) {
      if (depth == 0) return this->_sheaf[patch]->cap();
      if (depth == 1) return this->_sheaf[patch]->base();
      throw ALE::Exception("Invalid depth stratum");
    };
  public: // Viewers
    void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) {
      if (comm == MPI_COMM_NULL) {
        comm = this->comm();
      }
      if (name == "") {
        PetscPrintf(comm, "viewing a CartesianTopology\n");
      } else {
        PetscPrintf(comm, "viewing CartesianTopology '%s'\n", name.c_str());
      }
      PetscPrintf(comm, "  maximum height %d maximum depth %d\n", this->height(), this->depth());
      for(typename sheaf_type::const_iterator s_iter = this->_sheaf.begin(); s_iter != this->_sheaf.end(); ++s_iter) {
        ostringstream txt;

        txt << "Patch " << s_iter->first;
        s_iter->second->view(txt.str().c_str(), comm);
        PetscPrintf(comm, "  maximum height %d maximum depth %d\n", this->height(s_iter->first), this->depth(s_iter->first));
      }
    };
  };

  class CartesianMeshBuilder {
  public:
    static Obj<CartesianTopology<int> > createCartesianMesh(const MPI_Comm comm, const int dim, const int numCells[], const int partitions[], const int debug = 0) {
      PetscErrorCode ierr;
      // Steal PETSc code that calculates partitions
      //   We could conceivably allow multiple patches per partition
      int size, rank;
      int totalPartitions = 1;

      ierr = MPI_Comm_size(comm, &size);
      ierr = MPI_Comm_rank(comm, &rank);
      for(int d = 0; d < dim; ++d) totalPartitions *= partitions[d];
      if (size != totalPartitions) throw ALE::Exception("Invalid partitioning");
      // Determine local sizes
      int *numLocalCells       = new int[dim];
      int *numNeighborVertices = new int[dim];
      int *numLocalVertices    = new int[dim];

      for(int d = 0; d < dim; ++d) {
        numLocalCells[d]    = numCells[d]/partitions[d] + (rank < numCells[d]%partitions[d]);
        numLocalVertices[d] = numLocalCells[d]+1;
      }
      // Create topology
      typedef CartesianTopology<int> topology_type;
      const Obj<topology_type>             topology = new topology_type(comm, debug);
      const Obj<topology_type::sieve_type> sieve    = new topology_type::sieve_type(comm, dim, numLocalCells, debug);
      const topology_type::patch_type      patch    = 0;

      topology->setPatch(patch, sieve);
      topology->stratify();
      delete [] numLocalCells;
      // Create overlaps
      //   We overlap anyone within 1 index of us for the entire boundary
      const Obj<topology_type::send_overlap_type>& sendOverlap = topology->getSendOverlap();
      const Obj<topology_type::recv_overlap_type>& recvOverlap = topology->getRecvOverlap();
      int *indices  = new int[dim];
      int *code     = new int[dim];
      int *zeroCode = new int[dim];
      int *vIndices = new int[dim];
      int numNeighborCells;

      ALE::CartesianSieveDef::getIndices(dim, partitions, rank, indices);
      for(int e = 0; e < dim; ++e) zeroCode[e] = 0;
      for(int d = 0; d < dim; ++d) {
        if (indices[d] > 0) {
          for(int e = 0; e < dim; ++e) code[e] = 1;
          code[d] = 0;
          int neighborRank = ALE::CartesianSieveDef::getValue(dim, partitions, indices, code, false);
          numNeighborCells = 1;
          for(int e = 0; e < dim; ++e) {
            numNeighborCells      *= (numCells[e]/partitions[e] + (neighborRank < numCells[e]%partitions[e]));
            numNeighborVertices[e] = (numCells[e]/partitions[e] + (neighborRank < numCells[e]%partitions[e]))+1;
          }

          // Add the whole d-1 face on the left edge of dimension d
          for(int v = sieve->getNumCells(); v < sieve->getNumCells()+sieve->getNumVertices(); ++v) {
            ALE::CartesianSieveDef::getIndices(dim, numLocalVertices, v - sieve->getNumCells(), vIndices);
            if (vIndices[d] == 0) {
              vIndices[d]   = numNeighborVertices[d]-1;
              int neighborV = ALE::CartesianSieveDef::getValue(dim, numNeighborVertices, vIndices, zeroCode) + numNeighborCells;

              sendOverlap->addCone(v, neighborRank, neighborV);
              recvOverlap->addCone(neighborRank, v, neighborV);
            }
          }
        }
        if (indices[d] < partitions[d]-1) {
          for(int e = 0; e < dim; ++e) code[e] = 0;
          code[d] = 1;
          int neighborRank = ALE::CartesianSieveDef::getValue(dim, partitions, indices, code);
          numNeighborCells = 1;
          for(int e = 0; e < dim; ++e) {
            numNeighborCells      *= (numCells[e]/partitions[e] + (neighborRank < numCells[e]%partitions[e]));
            numNeighborVertices[e] = (numCells[e]/partitions[e] + (neighborRank < numCells[e]%partitions[e]))+1;
          }

          // Add the whole d-1 face on the right edge of dimension d
          for(int v = sieve->getNumCells(); v < sieve->getNumCells()+sieve->getNumVertices(); ++v) {
            ALE::CartesianSieveDef::getIndices(dim, numLocalVertices, v - sieve->getNumCells(), vIndices);
            if (vIndices[d] == numLocalVertices[d]-1) {
              vIndices[d]   = 0;
              int neighborV = ALE::CartesianSieveDef::getValue(dim, numNeighborVertices, vIndices, zeroCode) + numNeighborCells;

              sendOverlap->addCone(v, neighborRank, neighborV);
              recvOverlap->addCone(neighborRank, v, neighborV);
            }
          }
        }
      }
      if (debug) {
        sendOverlap->view("Send Overlap");
        recvOverlap->view("Receive Overlap");
      }
      delete [] numLocalVertices;
      delete [] numNeighborVertices;
      delete [] indices;
      delete [] code;
      delete [] zeroCode;
      delete [] vIndices;
      // Create coordinates
      return topology;
    };
  };
}

#endif
