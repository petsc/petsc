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
class Preconditioner : public virtual esi::Preconditioner<Scalar,Ordinal>,
                       public         esi::petsc::Object
{
  public:

    // Default destructor.
    ~Preconditioner(void);

    // Construct a preconditioner from a PETSc PC
    Preconditioner(PC pc);

    //  Interface for esi::Object  ---------------

    virtual esi::ErrorCode getInterface(const char* name, void*& iface) ;
    virtual esi::ErrorCode getInterfacesSupported(esi::Argv * list);


    //  Interface for esi::Operator  ---------------

    virtual esi::ErrorCode setup(void);
    virtual esi::ErrorCode apply( esi::Vector<Scalar,Ordinal>& x, esi::Vector<Scalar,Ordinal>& y);

    //  Interface for esi::Preconditioner  ---------------
    /** Input control parameters. */
    virtual esi::ErrorCode parameters( int numParams, char** paramStrings ){return 1;};

    /** z = M1^(-1) y */
    virtual esi::ErrorCode solveLeft( esi::Vector<Scalar, Ordinal> & y,esi::Vector<Scalar, Ordinal> & z );
    /** z = M2^(-1) y */
    virtual esi::ErrorCode solveRight( esi::Vector<Scalar, Ordinal> & y, esi::Vector<Scalar, Ordinal> & z );

    /** z = M^(-1) y */
    virtual esi::ErrorCode solve( esi::Vector<Scalar, Ordinal> & y, esi::Vector<Scalar, Ordinal> & z );
  
    /** z = B y */
    virtual esi::ErrorCode applyB( esi::Vector<Scalar,Ordinal>& y, esi::Vector<Scalar,Ordinal>& z );

    /** Get the preconditioning side. */
    virtual esi::ErrorCode getPreconditionerSide( PreconditionerSide & side );

    /** Set the preconditioning side. */
    virtual esi::ErrorCode setPreconditionerSide( PreconditionerSide side );

    virtual esi::ErrorCode setOperator( esi::Operator<Scalar,Ordinal> &op);
  private:
    PC                         pc;
    esi::IndexSpace<Ordinal>   *rmap,*cmap;
    esi::PreconditionerSide    side;
};
}}

#endif




