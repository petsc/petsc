#ifndef __PETSc_Matrix_h__
#define __PETSc_Matrix_h__

// this contains the PETSc definition of Matrix
#include "petscmat.h"

#include "esi/petsc/vector.h"

// The PETSc_Vector supports the 
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

    // Construct a matrix from two Maps.
    Matrix(esi::MapPartition<Ordinal> *rsource,esi::MapPartition<Ordinal> *csource);

    //  Interface for esi::Object  ---------------

    virtual esi::ErrorCode getInterface(const char* name, void*& iface) ;
    virtual esi::ErrorCode getInterfacesSupported(esi::Argv * list);


    //  Interface for esi::Operator  ---------------

    virtual esi::ErrorCode setup();
    virtual esi::ErrorCode apply( esi::Vector<Scalar,Ordinal>& x, esi::Vector<Scalar,Ordinal>& y);

    //  Interface for esi::MatrixData  ---------------
    virtual esi_int getGlobalSizes(Ordinal& rows, Ordinal& columns);
    virtual esi_int getLocalSizes(Ordinal& rows, Ordinal& columns);

    //  Interface for esi::MatrixRowAccess  --------

    virtual esi::ErrorCode getMaps(esi::Map<Ordinal>*& rowMap, esi::Map<Ordinal>*& colMap);
    virtual esi::ErrorCode isLoaded(bool &state);
    virtual esi::ErrorCode isAllocated(bool &state);
    virtual esi::ErrorCode loadComplete();
    virtual esi::ErrorCode allocate(int rowLengths[]);
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
    virtual esi::ErrorCode copyInRow(Ordinal row,  Scalar* coefs, Ordinal* colIndices, Ordinal length);
    virtual esi::ErrorCode sumIntoRow(Ordinal row,  Scalar* coefs, Ordinal* colIndices, Ordinal length);
    virtual esi::ErrorCode rowMax(Ordinal row, Scalar& result) ;
    virtual esi::ErrorCode rowMin(Ordinal row, Scalar& result) ;

    virtual esi::ErrorCode getRowAllocatedLength(Ordinal row, int& result) {;};
    virtual esi::ErrorCode setAllValues(Scalar) {;};
    virtual esi::ErrorCode allocateRowsSameLength(Ordinal) {;};
    virtual esi::ErrorCode copyOutRow(Ordinal, Scalar *,int *,int,int&) {;};
    virtual esi::ErrorCode copyOutRowIndices(Ordinal, int *,int,int&) {;};
    virtual esi::ErrorCode copyOutRowCoefficients(Ordinal, Scalar *,int,int&) {;};

  private:
    Mat                        mat;
    esi::MapPartition<Ordinal> *rmap,*cmap;
};
}}

#endif




