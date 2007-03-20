#ifndef included_ALE_CartesianSieve_hh
#define included_ALE_CartesianSieve_hh

#ifndef  included_ALE_CoSieve_hh
#include <CoSieve.hh>
#endif

namespace ALE {
  namespace CartesianSieveDef {
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
        void nextCodeOdd(int a[], const int n, const int size) {
          for(int d = size-2; d >= 0; --d) a[d+1] = a[d];
          if (n == (n/2)*2) {
            a[0] = 1;
          } else {
            a[0] = 0;
          }
        };
        void nextCodeEven(int a[], const int n, const int size) {
          this->nextCodeOdd(a, n, size);
          a[0] ^= 1;
          for(int d = 1; d < size; ++d) a[d] ^= 0;
        };
        void getGrayCode(const int n, const int size, int code[]) {
          if (n == 1 << size) {
            code[0] = -1;
          } else if (n == 0) {
            for(int d = 0; d < size; ++d) code[d] = 0;
          } else if (n == 1) {
            code[0] = 1;
            for(int d = 1; d < size; ++d) code[d] = 0;
          } else if (n == (n/2)*2) {
            // Might have to shift pointer
            this->getGrayCode(n/2, size, code);
            this->nextCodeEven(code, n/2, size);
          } else {
            // Might have to shift pointer
            this->getGrayCode((n-1)/2, size, code);
            this->nextCodeOdd(code, (n-1)/2, size);
          }
        };
        inline value_type getValue(const value_type *indices, const int *code) {
          //std::cout << "Building value" << std::endl << "  pos:" << this->_pos << std::endl << "  code:";
          //for(int d = 0; d < this->_dim; ++d) std::cout << " " << code[d];
          //std::cout << std::endl << "  indices:";
          //for(int d = 0; d < this->_dim; ++d) std::cout << " " << indices[d];
          //std::cout << std::endl;
          if (code[0] < 0) {
            //std::cout << "  value " << -1 << std::endl;
            return -1;
          }
          value_type value = indices[this->_dim-1];
          if (code[this->_dim-1]) value++;
          //std::cout << "  value " << value << std::endl;
          for(int d = this->_dim-2; d >= 0; --d) {
            value = value*(this->_sizes[d]+1) + indices[d];
            if (code[d]) value++;
            //std::cout << "  value " << value << std::endl;
          }
          value += this->_numCells;
          //std::cout << "  final value " << value << std::endl;
          return value;
        };
        void init() {
          this->_code = new int[this->_dim];
          this->getGrayCode(this->_pos, this->_dim, this->_code);
          this->_value = this->getValue(this->_indices, this->_code);
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
          this->getGrayCode(++this->_pos, this->_dim, this->_code);
          this->_value = this->getValue(this->_indices, this->_code);
          return *this;
        };
        virtual iterator          operator++(int n) {
          iterator tmp(*this);
          this->getGrayCode(++this->_pos, this->_dim, this->_code);
          this->_value = this->getValue(this->_indices, this->_code);
          return tmp;
        };
      };
    protected:
      int         _dim;
      const int  *_sizes;
      int         _numCells;
      point_type  _cell;
      int        *_indices;
    public:
      ConeSequence(const int dim, const int sizes[], const int numCells, const point_type& cell) : _dim(dim), _sizes(sizes), _numCells(numCells), _cell(cell) {
        this->_indices = new point_type[dim];
        this->getIndices(this->_cell, this->_indices);
      };
      virtual ~ConeSequence() {
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
      virtual iterator begin() {return iterator(this->_dim, this->_sizes, this->_numCells, this->_indices, 0);};
      virtual iterator end()   {return iterator(this->_dim, this->_sizes, this->_numCells, this->_indices, 1 << this->_dim);};
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
        value_type        _value;
        int               _pos;
        const int        *_sizes;
        const value_type *_indices;
        const int        *_code;
      public:
        iterator(const int dim, const int sizes[], const value_type indices[], const int pos) : _dim(dim), _sizes(sizes), _indices(indices), _pos(pos) {
          this->_code = new int[dim];
        };
        virtual ~iterator() {
          delete [] this->_code;
        };
      protected:
        void getGrayCode(const int n, const int size, const int code[]) {
        };
      public:
        virtual bool              operator==(const iterator& iter) const {return this->_pos == iter._pos;};
        virtual bool              operator!=(const iterator& iter) const {return this->_pos != iter._pos;};
        virtual const value_type  operator*() const {return this->_value;};
        virtual iterator          operator++() {
          this->getGrayCode(++this->_pos, this->_dim, this->_code);
          this->_value = this->_indices[this->_dim-1];
          if (this->_code[this->_dim-1]) this->_value++;
          for(int d = this->_dim-2; d >= 0; --d) {
            this->_value *= (this->_sizes[d]+1) + this->_indices[d];
            if (this->_code[d]) this->_value++;
          }
          return *this;
        };
        virtual iterator          operator++(int n) {
          iterator tmp(*this);
          this->_pos += n;
          this->getGrayCode(this->_pos, this->_dim, this->_code);
          this->_value = this->_indices[this->_dim-1];
          if (this->_code[this->_dim-1]) this->_value++;
          for(int d = this->_dim-2; d >= 0; --d) {
            this->_value *= (this->_sizes[d]+1) + this->_indices[d];
            if (this->_code[d]) this->_value++;
          }
          return tmp;
        };
      };
    protected:
      int         _dim;
      int        *_sizes;
      point_type  _vertex;
      int        *_indices;
    public:
      SupportSequence(const int dim, const int sizes[], const point_type& vertex) : _dim(dim), _sizes(sizes), _vertex(vertex) {
        this->_indices = new point_type[dim];
      };
      virtual ~SupportSequence() {
        delete [] this->_indices;
      };
    protected:
      void getIndices(const point_type& vertex, const int indices[]) {
        point_type v       = vertex;
        int        divisor = 1;

        for(int d = 0; d < this->_dim-1; ++d) {
          divisor *= this->_sizes[d]+1;
        }
        for(int d = this->_dim-1; d >= 0; ++d) {
          this->_indices[d] = v/divisor;
          v -= divisor*this->_indices[d];
          if (d > 0) divisor /= this->_sizes[d-1]+1;
        }
      };
    public:
      virtual iterator begin() {return iterator(this->_dim, this->_cell, 0);};
      virtual iterator end()   {return iterator(this->_dim, this->_cell, 1 << this->_dim);};
    };
  }

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
#if 0
      for(typename capSequence::iterator capi = cap->begin(); capi != cap->end(); capi++) {
        const Obj<supportSequence>& supp = this->support(*capi);
        const typename supportSequence::iterator suppEnd = supp->end();

        for(typename supportSequence::iterator suppi = supp->begin(); suppi != suppEnd; suppi++) {
          txt << "[" << this->commRank() << "]: " << *capi << "---->" << *suppi << std::endl;
        }
      }
#endif
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
        const Obj<coneSequence>               cone    = this->cone(*basei);
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
}

#endif
