#ifndef __PETSc_Preconditioner_h__
#define __PETSc_Preconditioner_h__

// this contains the PETSc definition of Preconditioner
#include "petscpc.h"

#include "esi/petsc/vector.h"

// The esi::petsc::Preconditioner supports the 
#include "esi/Operator.h"
#include "esi/Preconditioner.h"

namespace esi{namespace petsc{

/**=========================================================================**/
template<class Scalar,class Ordinal>
class Preconditioner : public virtual Operator<Scalar, Ordinal>,
                       public virtual esi::petsc::Object
{
  public:

    // Default destructor.
    ~Preconditioner();

    // Construct a preconditioner from a PETSc PC
    Preconditioner(PC pc);

    //  Interface for esi::Object  ---------------

    virtual esi::ErrorCode getInterface(const char* name, void*& iface) ;
    virtual esi::ErrorCode getInterfacesSupported(esi::Argv * list);


    //  Interface for esi::Operator  ---------------

    virtual esi::ErrorCode setup();
    virtual esi::ErrorCode apply( esi::Vector<Scalar,Ordinal>& x, esi::Vector<Scalar,Ordinal>& y);

    //  Interface for esi::Preconditioner  ---------------
    /** Input control parameters. */
    virtual esi::ErrorCode parameters( int numParams, char** paramStrings ){;};

    /** z = M1^(-1) y */
    virtual esi::ErrorCode solveM1( esi::Vector<Scalar, Ordinal> & y,esi::Vector<Scalar, Ordinal> & z );
    /** z = M2^(-1) y */
    virtual esi::ErrorCode solveM2( esi::Vector<Scalar, Ordinal> & y, esi::Vector<Scalar, Ordinal> & z );

    /** z = M^(-1) y */
    virtual esi::ErrorCode solveM( esi::Vector<Scalar, Ordinal> & y, esi::Vector<Scalar, Ordinal> & z );
  
    /** z = B y */
    virtual esi::ErrorCode applyB( esi::Vector<Scalar,Ordinal>& y, esi::Vector<Scalar,Ordinal>& z );

    /** Get the preconditioning side. */
    virtual esi::ErrorCode getPreconSide( int & side );

    /** Set the preconditioning side. */
    virtual esi::ErrorCode setPreconSide( int side );

  private:
    PC                         pc;
    esi::IndexSpace<Ordinal>   *rmap,*cmap;
    int                        side;
};
}}

#endif




