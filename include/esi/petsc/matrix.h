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
               public virtual esi::petsc::Object
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
    esi::IndexSpace<Ordinal> *rmap,*cmap;
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


#endif




