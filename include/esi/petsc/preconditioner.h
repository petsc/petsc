#ifndef __PETSc_Preconditioner_h__
#define __PETSc_Preconditioner_h__

// this contains the PETSc definition of Preconditioner
#include "petscpc.h"

#include "esi/petsc/vector.h"

// The esi::petsc::Preconditioner supports the 
#include "esi/Operator.h"
#include "esi/Preconditioner.h"
#include "esi/Solver.h"

namespace esi{namespace petsc{

/**=========================================================================**/
template<class Scalar,class Ordinal>
class Preconditioner : public virtual esi::Preconditioner<Scalar,Ordinal>,
                       public virtual esi::Solver<Scalar,Ordinal>,
                       public         esi::petsc::Object
{
  public:

    // Default destructor.
    ~Preconditioner(void);

    // Construct a preconditioner from a MPI_Comm
    Preconditioner(MPI_Comm comm);

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
    esi::PreconditionerSide    side;
};

/**=========================================================================**/
template<>
class Preconditioner<double,int> : public virtual esi::Preconditioner<double,int>,
                       public         esi::petsc::Object
{
  public:

    // Default destructor.
    ~Preconditioner(void);

    // Construct a preconditioner from a MPI_Comm
    Preconditioner(MPI_Comm comm);

    // Construct a preconditioner from a PETSc PC
    Preconditioner(PC pc);

    //  Interface for esi::Object  ---------------

    virtual esi::ErrorCode getInterface(const char* name, void*& iface) ;
    virtual esi::ErrorCode getInterfacesSupported(esi::Argv * list);


    //  Interface for esi::Operator  ---------------

    virtual esi::ErrorCode setup(void);
    virtual esi::ErrorCode apply( esi::Vector<double,int>& x, esi::Vector<double,int>& y);

    //  Interface for esi::Preconditioner  ---------------
    /** Input control parameters. */
    virtual esi::ErrorCode parameters( int numParams, char** paramStrings ){return 1;};

    /** z = M1^(-1) y */
    virtual esi::ErrorCode solveLeft( esi::Vector<double, int> & y,esi::Vector<double, int> & z );
    /** z = M2^(-1) y */
    virtual esi::ErrorCode solveRight( esi::Vector<double, int> & y, esi::Vector<double, int> & z );

    /** z = M^(-1) y */
    virtual esi::ErrorCode solve( esi::Vector<double, int> & y, esi::Vector<double, int> & z );
  
    /** z = B y */
    virtual esi::ErrorCode applyB( esi::Vector<double,int>& y, esi::Vector<double,int>& z );

    /** Get the preconditioning side. */
    virtual esi::ErrorCode getPreconditionerSide( PreconditionerSide & side );

    /** Set the preconditioning side. */
    virtual esi::ErrorCode setPreconditionerSide( PreconditionerSide side );

    virtual esi::ErrorCode setOperator( esi::Operator<double,int> &op);
  private:
    PC                       pc;
    esi::IndexSpace<int>     *rmap,*cmap;
    esi::PreconditionerSide  side;
};
}

  /* -------------------------------------------------------------------------*/

template<class Scalar,class Ordinal> class PreconditionerFactory 
{
  public:

    // Destructor.
    virtual ~PreconditionerFactory(void){};

    // Construct a Preconditioner
    virtual esi::ErrorCode getPreconditioner(char *commname,void* comm,esi::Preconditioner<Scalar,Ordinal>*&v) = 0; 
};

}

EXTERN int PCESISetPreconditioner(PC,esi::Preconditioner<double,int>*);

#endif




