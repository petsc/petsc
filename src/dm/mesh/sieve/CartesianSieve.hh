#ifndef included_ALE_CartesianSieve_hh
#define included_ALE_CartesianSieve_hh

#ifndef  included_ALE_Mesh_hh
#include <Mesh.hh>
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
        virtual iterator         operator++(int n) {iterator tmp(*this); ++this->_value; return tmp;};
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
        const int        *_cellSizes;
        int               _numCells;
        int               _pos;
        int              *_code;
        value_type        _value;
        value_type *_indices;
      protected:
        void init() {
          this->_code = new int[this->_dim];
          if (this->_pos == -2) {
            this->_value = this->_indices[0];
            ++this->_pos;
          } else if (this->_pos == -1) {
            this->_value = ALE::CartesianSieveDef::getValue(this->_dim, this->_cellSizes, this->_indices, this->_code);
          } else {
            ALE::CartesianSieveDef::getGrayCode(this->_pos, this->_dim, this->_code);
            this->_value = ALE::CartesianSieveDef::getValue(this->_dim, this->_sizes, this->_indices, this->_code) + this->_numCells;
          }
        };
      public:
        iterator(const int dim, const int sizes[], const int cellSizes[], const int numCells, const value_type indices[], const int pos, const bool addSelf = false) : _dim(dim), _sizes(sizes), _cellSizes(cellSizes), _numCells(numCells), _pos(pos) {
          this->_indices = new int[this->_dim];
          for(int d = 0; d < this->_dim; ++d) this->_indices[d] = indices[d];
          this->init();
        };
        iterator(const iterator& iter) : _dim(iter._dim), _sizes(iter._sizes), _cellSizes(iter._cellSizes), _numCells(iter._numCells), _pos(iter._pos) {
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
      bool        _addSelf;
    public:
      ConeSequence(const int dim, const int sizes[], const int numCells, const point_type& cell, const bool addSelf = false) : _dim(dim), _sizes(sizes), _numCells(numCells), _cell(cell), _addSelf(addSelf) {
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
      virtual iterator begin() {
        int start = 0;

        if (this->_addSelf) {
          if (this->_cell >= this->_numCells) {
            start = -2;
            this->_indices[0] = this->_cell;
          } else {
            start = -1;
          }
        }
        return iterator(this->_dim, this->_vertexSizes, this->_sizes, this->_numCells, this->_indices, start);
      };
      virtual iterator end() {
        int end = 1 << this->_dim;

        if (!this->_dim || (this->_cell >= this->_numCells)) {
          end = 0;
        }
        return iterator(this->_dim, this->_vertexSizes, this->_sizes, this->_numCells, this->_indices, end);
      };
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
      virtual iterator end()   {return iterator(this->_dim, this->_sizes, this->_indices, this->_dim ? 1 << this->_dim: 0);};
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
    typedef CartesianSieveDef::PointSequence<point_type>   baseSequence;
    typedef CartesianSieveDef::PointSequence<point_type>   capSequence;
    typedef CartesianSieveDef::ConeSequence<point_type>    coneSequence;
    typedef CartesianSieveDef::SupportSequence<point_type> supportSequence;
    // Backward compatibility
    typedef coneSequence coneArray;
    typedef coneSequence coneSet;
    typedef supportSequence supportArray;
    typedef supportSequence supportSet;
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
    // WARNING: Creating a lot of objects here
    Obj<baseSequence> base() {
      Obj<baseSequence> base = new baseSequence(0, this->_numCells);
      return base;
    };
    Obj<baseSequence> leaves() {
      return this->base();
    };
    // WARNING: Creating a lot of objects here
    Obj<capSequence> cap() {
       Obj<capSequence> cap = new capSequence(this->_numCells, this->_numCells+this->_numVertices);
       return cap;
    };
    Obj<capSequence> roots() {
       return this->cap();
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
    // WARNING: Creating a lot of objects here
    Obj<supportSequence> support(const point_type& p) {
      if ((p < this->_numCells) || (p >= this->_numCells+this->_numVertices)) {
        Obj<supportSequence> support = new supportSequence(0, this->_sizes, this->_numCells, 0);
        return support;
      }
      Obj<supportSequence> support = new supportSequence(this->_dim, this->_sizes, this->_numCells, p);
      return support;
    };
    // WARNING: Creating a lot of objects here
    Obj<coneSequence> closure(const point_type& p) {
      Obj<coneSequence> cone = new coneSequence(this->_dim, this->_sizes, this->_numCells, p, true);
      return cone;
    };
    // WARNING: Creating a lot of objects here
    Obj<supportSequence> star(const point_type& p) {return this->support(p);};
    // WARNING: Creating a lot of objects here
    Obj<coneSequence> nCone(const point_type& p, const int n) {
      if (n == 0) return this->cone(-1);
      if (n != 1) throw ALE::Exception("Invalid height for nCone");
      return this->cone(p);
    };
    // WARNING: Creating a lot of objects here
    Obj<supportSequence> nSupport(const point_type& p, const int n) {
      if (n == 0) return this->support(-1);
      if (n != 1) throw ALE::Exception("Invalid depth for nSupport");
      return this->support(p);
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

  // We do not just use Bundle, because we need to optimize labels
  class CartesianBundle : public ALE::ParallelObject {
  public:
    typedef CartesianSieve<int>                                       sieve_type;
    typedef sieve_type::point_type                                    point_type;
    typedef Section<point_type, double>                               real_section_type;
    typedef Section<point_type, int>                                  int_section_type;
    typedef UniformSection<MinimalArrow<point_type, point_type>, int> arrow_section_type;
    typedef std::map<std::string, Obj<arrow_section_type> >           arrow_sections_type;
    typedef std::map<std::string, Obj<real_section_type> >            real_sections_type;
    typedef std::map<std::string, Obj<int_section_type> >             int_sections_type;
    typedef sieve_type::baseSequence                                  label_sequence;
    typedef ALE::Sifter<int,point_type,point_type>                    send_overlap_type;
    typedef ALE::Sifter<point_type,int,point_type>                    recv_overlap_type;
  protected:
    Obj<sieve_type>        _sieve;
    int                    _maxHeight;
    int                    _maxDepth;
    arrow_sections_type    _arrowSections;
    real_sections_type     _realSections;
    int_sections_type      _intSections;
    Obj<send_overlap_type> _sendOverlap;
    Obj<recv_overlap_type> _recvOverlap;
    Obj<send_overlap_type> _distSendOverlap;
    Obj<recv_overlap_type> _distRecvOverlap;
  public:
    CartesianBundle(MPI_Comm comm, const int debug = 0) : ALE::ParallelObject(comm, debug), _maxHeight(-1), _maxDepth(-1) {
      this->_sendOverlap = new send_overlap_type(this->comm(), this->debug());
      this->_recvOverlap = new recv_overlap_type(this->comm(), this->debug());
    };
    virtual ~CartesianBundle() {};
    public: // Accessors
      void clear() {
        this->_maxHeight = -1;
        this->_maxDepth  = -1;
      };
    const Obj<sieve_type>& getSieve() const {return this->_sieve;};
    void setSieve(const Obj<sieve_type>& sieve) {this->_sieve = sieve;};
    bool hasArrowSection(const std::string& name) const {
      return this->_arrowSections.find(name) != this->_arrowSections.end();
    };
    const Obj<arrow_section_type>& getArrowSection(const std::string& name) {
      if (!this->hasArrowSection(name)) {
        Obj<arrow_section_type> section = new arrow_section_type(this->comm(), this->debug());

        section->setName(name);
        if (this->_debug) {std::cout << "Creating new arrow section: " << name << std::endl;}
        this->_arrowSections[name] = section;
      }
      return this->_arrowSections[name];
    };
    void setArrowSection(const std::string& name, const Obj<arrow_section_type>& section) {
      this->_arrowSections[name] = section;
    };
    Obj<std::set<std::string> > getArrowSections() const {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(arrow_sections_type::const_iterator s_iter = this->_arrowSections.begin(); s_iter != this->_arrowSections.end(); ++s_iter) {
        names->insert(s_iter->first);
      }
      return names;
    };
    bool hasRealSection(const std::string& name) const {
      return this->_realSections.find(name) != this->_realSections.end();
    };
    const Obj<real_section_type>& getRealSection(const std::string& name) {
      if (!this->hasRealSection(name)) {
        Obj<real_section_type> section = new real_section_type(this->comm(), this->debug());

        section->setName(name);
        if (this->_debug) {std::cout << "Creating new real section: " << name << std::endl;}
        this->_realSections[name] = section;
      }
      return this->_realSections[name];
    };
    void setRealSection(const std::string& name, const Obj<real_section_type>& section) {
      this->_realSections[name] = section;
    };
    Obj<std::set<std::string> > getRealSections() const {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(real_sections_type::const_iterator s_iter = this->_realSections.begin(); s_iter != this->_realSections.end(); ++s_iter) {
        names->insert(s_iter->first);
      }
      return names;
    };
    bool hasIntSection(const std::string& name) const {
      return this->_intSections.find(name) != this->_intSections.end();
    };
    const Obj<int_section_type>& getIntSection(const std::string& name) {
      if (!this->hasIntSection(name)) {
        Obj<int_section_type> section = new int_section_type(this->comm(), this->debug());

        section->setName(name);
        if (this->_debug) {std::cout << "Creating new int section: " << name << std::endl;}
        this->_intSections[name] = section;
      }
      return this->_intSections[name];
    };
    void setIntSection(const std::string& name, const Obj<int_section_type>& section) {
      this->_intSections[name] = section;
    };
    Obj<std::set<std::string> > getIntSections() const {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(int_sections_type::const_iterator s_iter = this->_intSections.begin(); s_iter != this->_intSections.end(); ++s_iter) {
        names->insert(s_iter->first);
      }
      return names;
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
    #define __FUNCT__ "Bundle::stratify"
    virtual void stratify() {
      ALE_LOG_EVENT_BEGIN;
      this->_maxHeight = 1;
      this->_maxDepth  = 1;
      ALE_LOG_EVENT_END;
    };
    virtual int height() const {return this->_maxHeight;};
    virtual int height(const point_type& point) {
      const int numCells    = this->getSieve()->getNumCells();
      const int numVertices = this->getSieve()->getNumVertices();
      if ((point >= 0)        && (point < numCells))             return 0;
      if ((point >= numCells) && (point < numCells+numVertices)) return 1;
      return -1;
    };
    // WARNING: Creating a lot of objects here
    virtual const Obj<label_sequence> heightStratum(int height) {
      if (height == 0) return this->getSieve()->base();
      if (height == 1) return this->getSieve()->cap();
      throw ALE::Exception("Invalid height stratum");
    };
    virtual int depth() const {return this->_maxDepth;};
    virtual int depth(const point_type& point) {
      const int numCells    = this->getSieve()->getNumCells();
      const int numVertices = this->getSieve()->getNumVertices();
      if ((point >= 0)        && (point < numCells))             return 1;
      if ((point >= numCells) && (point < numCells+numVertices)) return 0;
      return -1;
    };
    // WARNING: Creating a lot of objects here
    virtual const Obj<label_sequence> depthStratum(int depth) {
      if (depth == 0) return this->getSieve()->cap();
      if (depth == 1) return this->getSieve()->base();
      throw ALE::Exception("Invalid depth stratum");
    };
    virtual const Obj<label_sequence> getLabelStratum(const std::string& name, int value) {
      if (name == "height") {
        return this->heightStratum(value);
      } else if (name == "depth") {
        return this->depthStratum(value);
      }
      throw ALE::Exception("Invalid label name");
    }
  public: // Viewers
    void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) {
      if (comm == MPI_COMM_NULL) {
        comm = this->comm();
      }
      if (name == "") {
        PetscPrintf(comm, "viewing a CartesianBundle\n");
      } else {
        PetscPrintf(comm, "viewing CartesianBundle '%s'\n", name.c_str());
      }
      PetscPrintf(comm, "  maximum height %d maximum depth %d\n", this->height(), this->depth());
    };
  public:
    void constructOverlap() {};
  };

  class CartesianMesh : public CartesianBundle {
  public:
    typedef int                                point_type;
    typedef CartesianBundle::sieve_type        sieve_type;
    typedef CartesianBundle::label_sequence    label_sequence;
    typedef CartesianBundle::send_overlap_type send_overlap_type;
    typedef CartesianBundle::recv_overlap_type recv_overlap_type;
    typedef sieve_type::coneSequence           coneSequence;
    typedef sieve_type::supportSequence        supportSequence;
  protected:
    int                    _dim;
    // Discretization
    Obj<Discretization>    _discretization;
    Obj<BoundaryCondition> _boundaryCondition;
  public:
    CartesianMesh(MPI_Comm comm, int dim, int debug = 0) : CartesianBundle(comm, debug), _dim(dim) {
      this->_discretization    = new Discretization(comm, debug);
      this->_boundaryCondition = new BoundaryCondition(comm, debug);
    };
    virtual ~CartesianMesh() {};
  public: // Accessors
    int getDimension() {return this->_dim;};
    const Obj<Discretization>&    getDiscretization() {return this->_discretization;};
    void setDiscretization(const Obj<Discretization>& discretization) {this->_discretization = discretization;};
    const Obj<BoundaryCondition>& getBoundaryCondition() {return this->_boundaryCondition;};
    void setBoundaryCondition(const Obj<BoundaryCondition>& boundaryCondition) {this->_boundaryCondition = boundaryCondition;};
  public: // Discretization
#if 0
    void markBoundaryCells(const std::string& name) {
      const Obj<label_type>&     label    = this->getLabel(name);
      const Obj<label_sequence>& boundary = this->getLabelStratum(name, 1);
      const Obj<sieve_type>&     sieve    = this->getSieve();

      for(label_sequence::iterator e_iter = boundary->begin(); e_iter != boundary->end(); ++e_iter) {
        if (this->height(*e_iter) == 1) {
          const point_type cell = *sieve->support(*e_iter)->begin();

          this->setValue(label, cell, 2);
        }
      }
    };
#endif
    void setupField(const Obj<real_section_type>& s, const bool postponeGhosts = false) {
      const std::string& name = this->_boundaryCondition->getLabelName();

      for(int d = 0; d <= this->depth(); ++d) {
        s->setFiberDimension(this->depthStratum(d), this->_discretization->getNumDof(d));
      }
      if (!name.empty()) {
        const Obj<label_sequence>& boundary = this->getLabelStratum(name, 1);

        for(label_sequence::iterator e_iter = boundary->begin(); e_iter != boundary->end(); ++e_iter) {
          s->setFiberDimension(*e_iter, -this->_discretization->getNumDof(this->depth(*e_iter)));
        }
      }
      s->allocatePoint();
      if (!name.empty()) {
#if 0
        const Obj<label_sequence>&     boundaryCells = this->getLabelStratum(name, 2);
        const Obj<real_section_type>&  coordinates   = this->getRealSection("coordinates");
        real_section_type::value_type *values        = new real_section_type::value_type[this->sizeWithBC(s, *boundaryCells->begin())];
        double                        *v0            = new double[this->getDimension()];
        double                        *J             = new double[this->getDimension()*this->getDimension()];
        double                         detJ;

        for(label_sequence::iterator c_iter = boundaryCells->begin(); c_iter != boundaryCells->end(); ++c_iter) {
          const Obj<coneArray>      closure = sieve_alg_type::closure(this, this->getArrowSection("orientation"), *c_iter);
          const coneArray::iterator end     = closure->end();
          int                       v       = 0;

          this->computeElementGeometry(coordinates, *c_iter, v0, J, PETSC_NULL, detJ);
          for(coneArray::iterator cl_iter = closure->begin(); cl_iter != end; ++cl_iter) {
            const int cDim = s->getConstraintDimension(*cl_iter);

            if (cDim) {
              for(int d = 0; d < cDim; ++d, ++v) {
                values[v] = this->_boundaryCondition->integrateDual(v0, J, v);
              }
            } else {
              const int dim = s->getFiberDimension(*cl_iter);

              for(int d = 0; d < dim; ++d, ++v) {
                values[v] = 0.0;
              }
            }
          }
          this->updateAll(s, *c_iter, values);
        }
        delete [] values;
        delete [] v0;
        delete [] J;
#endif
      }
    };
  public: // Size traversal
    template<typename Section_>
    int size(const Obj<Section_>& section, const point_type& p) {
      const typename Section_::chart_type& chart = section->getChart();
      const Obj<coneSequence>              cone  = this->getSieve()->cone(p);
      typename coneSequence::iterator      end   = cone->end();
      int                                  size  = 0;

      if (chart.count(p)) {
        size += section->getConstrainedFiberDimension(p);
      }
      for(typename coneSequence::iterator c_iter = cone->begin(); c_iter != end; ++c_iter) {
        if (chart.count(*c_iter)) {
          size += section->getConstrainedFiberDimension(*c_iter);
        }
      }
      return size;
    };
    template<typename Section_>
    int sizeWithBC(const Obj<Section_>& section, const point_type& p) {
      const typename Section_::chart_type& chart = section->getChart();
      const Obj<coneSequence>              cone  = this->getSieve()->cone(p);
      typename coneSequence::iterator      end   = cone->end();
      int                                  size  = 0;

      if (chart.count(p)) {
        size += section->getFiberDimension(p);
      }
      for(typename coneSequence::iterator c_iter = cone->begin(); c_iter != end; ++c_iter) {
        if (chart.count(*c_iter)) {
          size += section->getFiberDimension(*c_iter);
        }
      }
      return size;
    };
  public: // Allocation
    template<typename Section_>
    void allocate(const Obj<Section_>& section) {
      section->allocatePoint();
    };
  public: // Retrieval traversal
    // Return the values for the closure of this point
    //   use a smart pointer?
    template<typename Section_>
    const typename Section_::value_type *restrictClosure(const Obj<Section_>& section, const point_type& p) {
      const int                       size   = this->sizeWithBC(section, p);
      typename Section_::value_type  *values = section->getRawArray(size);
      int                             j      = -1;

      const int& dim = section->getFiberDimension(p);
      const typename Section_::value_type *array = section->restrictPoint(p);

      for(int i = 0; i < dim; ++i) {
        values[++j] = array[i];
      }
      const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);
      typename sieve_type::coneSequence::iterator   end  = cone->end();

      for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
        const int& dim = section->getFiberDimension(*p_iter);

        array = section->restrictPoint(*p_iter);
        for(int i = 0; i < dim; ++i) {
          values[++j] = array[i];
        }
      }
      if (j != size-1) {
        ostringstream txt;

        txt << "Invalid restrictClosure to point " << p << std::endl;
        txt << "  j " << j << " should be " << (size-1) << std::endl;
        std::cout << txt.str();
        throw ALE::Exception(txt.str().c_str());
      }
      return values;
    };
    template<typename Section_>
    void update(const Obj<Section_>& section, const point_type& p, const typename Section_::value_type v[]) {
      int j = 0;

      section->updatePoint(p, &v[j]);
      j += section->getFiberDimension(p);
      const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);

      for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
        section->updatePoint(*p_iter, &v[j]);
        j += section->getFiberDimension(*p_iter);
      }
    };
    template<typename Section_>
    void updateAdd(const Obj<Section_>& section, const point_type& p, const typename Section_::value_type v[]) {
      int j = 0;

      section->updateAddPoint(p, &v[j]);
      j += section->getFiberDimension(p);
      const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);

      for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
        section->updateAddPoint(*p_iter, &v[j]);
        j += section->getFiberDimension(*p_iter);
      }
    };
    template<typename Section_>
    void updateBC(const Obj<Section_>& section, const point_type& p, const typename Section_::value_type v[]) {
      int j = 0;

      section->updatePointBC(p, &v[j]);
      j += section->getFiberDimension(p);
      const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);

      for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
        section->updatePointBC(*p_iter, &v[j]);
        j += section->getFiberDimension(*p_iter);
      }
    };
    template<typename Section_>
    void updateAll(const Obj<Section_>& section, const point_type& p, const typename Section_::value_type v[]) {
      int j = 0;

      section->updatePointBC(p, &v[j]);
      j += section->getFiberDimension(p);
      const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);

      for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
        section->updatePointAll(*p_iter, &v[j]);
        j += section->getFiberDimension(*p_iter);
      }
    };
  public: // Mesh geometry
    void computeElementGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double v0[], double J[], double invJ[], double& detJ) {
      const double *coords = this->restrictClosure(coordinates, e);
      const int     dim    = this->_dim;
      const int     last   = 1 << dim;

      if (v0) {
        for(int d = 0; d < dim; d++) {
          v0[d] = coords[d];
        }
      }
      for(int d = 0; d < dim; ++d) {
        if (J) {
          J[d] = 0.5*(coords[last*dim+d] - v0[d]);
        }
        if (invJ) {
          invJ[d] = 1.0/J[d];
        }
      }
    };
  public: // Viewers
    void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) {
      if (comm == MPI_COMM_NULL) {
        comm = this->comm();
      }
      if (name == "") {
        PetscPrintf(comm, "viewing a CartesianMesh\n");
      } else {
        PetscPrintf(comm, "viewing CartesianMesh '%s'\n", name.c_str());
      }
      this->getSieve()->view("mesh sieve");
    };
  };

  class CartesianMeshBuilder {
  public:
    static Obj<CartesianMesh> createCartesianMesh(const MPI_Comm comm, const int dim, const int numCells[], const int partitions[], const int debug = 0) {
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
      // Create mesh
      const Obj<CartesianMesh>             mesh  = new CartesianMesh(comm, dim, debug);
      const Obj<CartesianMesh::sieve_type> sieve = new CartesianMesh::sieve_type(comm, dim, numLocalCells, debug);

      mesh->setSieve(sieve);
      mesh->stratify();
      delete [] numLocalCells;
      // Create overlaps
      //   We overlap anyone within 1 index of us for the entire boundary
      const Obj<CartesianMesh::send_overlap_type>& sendOverlap = mesh->getSendOverlap();
      const Obj<CartesianMesh::recv_overlap_type>& recvOverlap = mesh->getRecvOverlap();
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
      return mesh;
    };
  };
}

#endif
