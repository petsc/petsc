#ifndef __PETSc_Matrix_h__
#define __PETSc_Matrix_h__

// this contains the PETSc definition of Matrix
#include "petscmat.h"

#include "esi/petsc/vector.h"

// The esi::petsc::Matrix supports the 
#include "esi/Operator.h"
#include "esi/MatrixData.h"
#include "esi/MatrixRowReadAccess.h"
#include "esi/MatrixRowWriteAccess.h"

namespace esi{namespace petsc{

/**=========================================================================**/
template<class Scalar,class Ordinal>
class Matrix : public virtual esi::Operator<Scalar,Ordinal>, 
               public virtual esi::MatrixData<Ordinal>,
               public virtual esi::MatrixRowReadAccess<Scalar,Ordinal>,
               public virtual esi::MatrixRowWriteAccess<Scalar,Ordinal>, 
               public         esi::petsc::Object
{
  public:

    // Default destructor.
    ~Matrix();

    // Construct a matrix from two IndexSpaces.
    Matrix(esi::IndexSpace<Ordinal> *rsource,esi::IndexSpace<Ordinal> *csource);

    // Construct a esi::petsc::matrix from a PETSc Mat
    Matrix(Mat pmat);

    //  Interface for esi::Object  ---------------

    virtual esi::ErrorCode getInterface(const char* name, void*& iface) ;
    virtual esi::ErrorCode getInterfacesSupported(esi::Argv * list);


    //  Interface for esi::Operator  ---------------

    virtual esi::ErrorCode setup();
    virtual esi::ErrorCode apply( esi::Vector<Scalar,Ordinal>& x, esi::Vector<Scalar,Ordinal>& y);

    //  Interface for esi::MatrixData  ---------------
    virtual esi::ErrorCode getGlobalSizes(Ordinal& rows, Ordinal& columns);
    virtual esi::ErrorCode getLocalSizes(Ordinal& rows, Ordinal& columns);

    //  Interface for esi::MatrixRowAccess  --------

    virtual esi::ErrorCode getIndexSpaces(esi::IndexSpace<Ordinal>*& rowIndexSpace, esi::IndexSpace<Ordinal>*& colIndexSpace);
    virtual esi::ErrorCode isLoaded(bool &state);
    virtual esi::ErrorCode isAllocated(bool &state);
    virtual esi::ErrorCode loadComplete(void);
    virtual esi::ErrorCode allocate(Ordinal *rowLengths);
    virtual esi::ErrorCode getDiagonal(esi::Vector<Scalar,Ordinal>& diagVector) ;
    virtual esi::ErrorCode getRowSum(esi::Vector<Scalar,Ordinal>& rowSumVector) ;
    virtual esi::ErrorCode getRowNonzeros(Ordinal row, Ordinal& length);
    virtual esi::ErrorCode setRowLength(Ordinal row,Ordinal length);
    virtual esi::ErrorCode getRow(Ordinal row, Ordinal& length, Scalar*& coefs, Ordinal*& colIndices) ;
    virtual esi::ErrorCode getRowCoefs(Ordinal row, Ordinal& length, Scalar*& coefs) ;
    virtual esi::ErrorCode getRowIndices(Ordinal row, Ordinal& length, Ordinal*& colIndices) ;
    virtual esi::ErrorCode restoreRow(Ordinal row, Ordinal& length, Scalar*& coefs, Ordinal*& colIndices) ;
    virtual esi::ErrorCode restoreRowCoefs(Ordinal row, Ordinal& length, Scalar*& coefs) ;
    virtual esi::ErrorCode restoreRowIndices(Ordinal row, Ordinal& length, Ordinal*& colIndices) ;
    virtual esi::ErrorCode copyIntoRow(Ordinal row,  Scalar* coefs, Ordinal* colIndices, Ordinal length);
    virtual esi::ErrorCode sumIntoRow(Ordinal row,  Scalar* coefs, Ordinal* colIndices, Ordinal length);
    virtual esi::ErrorCode rowMax(Ordinal row, Scalar& result) ;
    virtual esi::ErrorCode rowMin(Ordinal row, Scalar& result) ;

    virtual esi::ErrorCode getRowAllocatedLength(Ordinal row, int& result) {return 1;};
    virtual esi::ErrorCode setAllValues(Scalar) {return 1;};
    virtual esi::ErrorCode allocateRowsSameLength(Ordinal) {return 1;};
    virtual esi::ErrorCode copyOutRow(Ordinal, Scalar *,int *,int,int&) ;
    virtual esi::ErrorCode copyOutRowIndices(Ordinal, int *,int,int&) {return 1;};
    virtual esi::ErrorCode copyOutRowCoefficients(Ordinal, Scalar *,int,int&) {return 1;};

  private:
    Mat                        mat;
    ::esi::IndexSpace<Ordinal> *rmap,*cmap;
};

/**=========================================================================**/
template<>
class Matrix<double,int> : public virtual esi::Operator<double,int>, 
               public virtual esi::MatrixData<int>,
               public virtual esi::MatrixRowReadAccess<double,int>,
               public virtual esi::MatrixRowWriteAccess<double,int>, 
               public         esi::petsc::Object
{
  public:

    // Default destructor.
    ~Matrix();

    // Construct a matrix from two IndexSpaces.
    Matrix(esi::IndexSpace<int> *rsource,esi::IndexSpace<int> *csource);

    // Construct a esi::petsc::matrix from a PETSc Mat
    Matrix(Mat pmat);

    //  Interface for esi::Object  ---------------

    virtual esi::ErrorCode getInterface(const char* name, void*& iface) ;
    virtual esi::ErrorCode getInterfacesSupported(esi::Argv * list);


    //  Interface for esi::Operator  ---------------

    virtual esi::ErrorCode setup();
    virtual esi::ErrorCode apply( esi::Vector<double,int>& x, esi::Vector<double,int>& y);

    //  Interface for esi::MatrixData  ---------------
    virtual esi::ErrorCode getGlobalSizes(int& rows, int& columns);
    virtual esi::ErrorCode getLocalSizes(int& rows, int& columns);

    //  Interface for esi::MatrixRowAccess  --------

    virtual esi::ErrorCode getIndexSpaces(esi::IndexSpace<int>*& rowIndexSpace, esi::IndexSpace<int>*& colIndexSpace);
    virtual esi::ErrorCode isLoaded(bool &state);
    virtual esi::ErrorCode isAllocated(bool &state);
    virtual esi::ErrorCode loadComplete(void);
    virtual esi::ErrorCode allocate(int *rowLengths);
    virtual esi::ErrorCode getDiagonal(esi::Vector<double,int>& diagVector) ;
    virtual esi::ErrorCode getRowSum(esi::Vector<double,int>& rowSumVector) ;
    virtual esi::ErrorCode getRowNonzeros(int row, int& length);
    virtual esi::ErrorCode setRowLength(int row,int length);
    virtual esi::ErrorCode getRow(int row, int& length, double*& coefs, int*& colIndices) ;
    virtual esi::ErrorCode getRowCoefs(int row, int& length, double*& coefs) ;
    virtual esi::ErrorCode getRowIndices(int row, int& length, int*& colIndices) ;
    virtual esi::ErrorCode restoreRow(int row, int& length, double*& coefs, int*& colIndices) ;
    virtual esi::ErrorCode restoreRowCoefs(int row, int& length, double*& coefs) ;
    virtual esi::ErrorCode restoreRowIndices(int row, int& length, int*& colIndices) ;
    virtual esi::ErrorCode copyIntoRow(int row,  double* coefs, int* colIndices, int length);
    virtual esi::ErrorCode sumIntoRow(int row,  double* coefs, int* colIndices, int length);
    virtual esi::ErrorCode rowMax(int row, double& result) ;
    virtual esi::ErrorCode rowMin(int row, double& result) ;

    virtual esi::ErrorCode getRowAllocatedLength(int row, int& result) {return 1;};
    virtual esi::ErrorCode setAllValues(double) {return 1;};
    virtual esi::ErrorCode allocateRowsSameLength(int) {return 1;};
    virtual esi::ErrorCode copyOutRow(int, double *,int *,int,int&) ;
    virtual esi::ErrorCode copyOutRowIndices(int, int *,int,int&) {return 1;};
    virtual esi::ErrorCode copyOutRowCoefficients(int, double *,int,int&) {return 1;};

  private:
    Mat                    mat;
    ::esi::IndexSpace<int> *rmap,*cmap;
};

}

  /* -------------------------------------------------------------------------*/

template<class Scalar,class Ordinal> class OperatorFactory 
#if defined(PETSC_HAVE_CCA)
           :  public virtual gov::cca::Port, public virtual gov::cca::Component
#endif
{
  public:

    // Destructor.
  virtual ~OperatorFactory(void){};

    // Interface for gov::cca::Component
#if defined(PETSC_HAVE_CCA)
    virtual void setServices(gov::cca::Services *) = 0;
#endif

    // Construct a Operator
    virtual esi::ErrorCode getOperator(esi::IndexSpace<Ordinal>&,esi::IndexSpace<Ordinal>&,esi::Operator<Scalar,Ordinal>*&v) = 0; 
};

}
EXTERN int MatESIWrap(Mat,esi::Operator<double,int>**);


#endif




