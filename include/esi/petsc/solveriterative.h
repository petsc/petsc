#ifndef __PETSc_SolverIterative_h__
#define __PETSc_SolverIterative_h__

// this contains the PETSc definition of solveriterative
#include "petscsles.h"

#include "esi/petsc/vector.h"

#include "esi/Operator.h"
#include "esi/Preconditioner.h"
#include "esi/Solver.h"
// The esi::petsc::solveriterative supports the 
#include "esi/SolverIterative.h"

namespace esi{namespace petsc{

/**=========================================================================**/
template<class Scalar,class Ordinal>
class SolverIterative : public virtual esi::SolverIterative<Scalar,Ordinal>,
                        public         esi::petsc::Object
{
  public:

    // Default destructor.
    ~SolverIterative(void);

    // Construct a solveriterative from a MPI_Comm
    SolverIterative(MPI_Comm comm);

    // Construct a solveriterative from a PETSc SLES
    SolverIterative(SLES sles);

    //  Interface for esi::Object  ---------------

    virtual esi::ErrorCode getInterface(const char* name, void*& iface) ;
    virtual esi::ErrorCode getInterfacesSupported(esi::Argv * list);

    //  Interface for esi::Operator  ---------------

    virtual esi::ErrorCode setup(void);
    virtual esi::ErrorCode apply( esi::Vector<Scalar,Ordinal>& x, esi::Vector<Scalar,Ordinal>& y);

    //  Interface for esi::Solver  ---------------
    virtual esi::ErrorCode solve( esi::Vector<Scalar, Ordinal> & b, esi::Vector<Scalar, Ordinal> & x );

    /** Set the input control parameters. */
    virtual esi::ErrorCode parameters( int numParams, char** paramStrings );

    //  Interface for esi::Solveriterative  ---------------
    virtual esi::ErrorCode getOperator( esi::Operator<Scalar, Ordinal> * & A );

    /** Set the operator. */
    virtual esi::ErrorCode setOperator( esi::Operator<Scalar, Ordinal> & A );

    /** Get the preconditioner. */
    virtual esi::ErrorCode getPreconditioner( esi::Preconditioner<Scalar, Ordinal> * & pc );

    /** Set the preconditioner. */
    virtual esi::ErrorCode setPreconditioner( esi::Preconditioner<Scalar, Ordinal> & pc );

    /** Get the convergence tolerance. */
    virtual esi::ErrorCode getTolerance( magnitude_type & tol );

    /** Set the convergence tolerance. */
    virtual esi::ErrorCode setTolerance( magnitude_type tol );

    /** Get the maximum number of iterations. */
    virtual esi::ErrorCode getMaxIterations( Ordinal & maxIterations );

    /** Set the maximum number of iterations. */
    virtual esi::ErrorCode setMaxIterations( Ordinal maxIterations );

    /** Query the number of iterations that were taken during the previous solve.
     */
    virtual esi::ErrorCode getNumIterationsTaken(Ordinal& itersTaken);

    class Factory : public virtual esi::SolverIterative<Scalar,Ordinal>::Factory
    {
      public:

        // Destructor.
        virtual ~Factory(void){};

        // Construct a SolverIterative
        virtual esi::ErrorCode create(char *commname,void *comm,esi::SolverIterative<Scalar,Ordinal>*&v); 
    };

  private:
    SLES                                  sles;
    ::esi::Preconditioner<Scalar,Ordinal> *pre;
    ::esi::Operator<Scalar,Ordinal>       *op;
};

template<>
class SolverIterative<double,int> : public virtual esi::SolverIterative<double,int>,
                                    public         esi::petsc::Object
{
  public:

    // Default destructor.
    ~SolverIterative(void);

    // Construct a solveriterative from a MPI_Comm
    SolverIterative(MPI_Comm comm);

    // Construct a solveriterative from a PETSc SLES
    SolverIterative(SLES sles);

    //  Interface for esi::Object  ---------------

    virtual esi::ErrorCode getInterface(const char* name, void*& iface) ;
    virtual esi::ErrorCode getInterfacesSupported(esi::Argv * list);


    //  Interface for esi::Operator  ---------------

    virtual esi::ErrorCode setup(void);
    virtual esi::ErrorCode apply( esi::Vector<double,int>& x, esi::Vector<double,int>& y);

    //  Interface for esi::Solver  ---------------
    virtual esi::ErrorCode solve( esi::Vector<double, int> & b, esi::Vector<double, int> & x );

    /** Set the input control parameters. */
    virtual esi::ErrorCode parameters( int numParams, char** paramStrings );

    //  Interface for esi::SolverIterative  ---------------
    virtual esi::ErrorCode getOperator( esi::Operator<double, int> * & A );

    /** Set the operator. */
    virtual esi::ErrorCode setOperator( esi::Operator<double, int> & A );

    /** Get the preconditioner. */
    virtual esi::ErrorCode getPreconditioner( esi::Preconditioner<double, int> * & pc );

    /** Set the preconditioner. */
    virtual esi::ErrorCode setPreconditioner( esi::Preconditioner<double, int> & pc );

    /** Get the convergence tolerance. */
    virtual esi::ErrorCode getTolerance( magnitude_type & tol );

    /** Set the convergence tolerance. */
    virtual esi::ErrorCode setTolerance( magnitude_type tol );

    /** Get the maximum number of iterations. */
    virtual esi::ErrorCode getMaxIterations( int & maxIterations );

    /** Set the maximum number of iterations. */
    virtual esi::ErrorCode setMaxIterations( int maxIterations );

    /** Query the number of iterations that were taken during the previous solve.
     */
    virtual esi::ErrorCode getNumIterationsTaken(int& itersTaken);

    class Factory : public virtual esi::SolverIterative<double,int>::Factory
    {
      public:

        // Destructor.
        virtual ~Factory(void){};

        // Construct a SolverIterative
        virtual esi::ErrorCode create(char *commname,void *comm,esi::SolverIterative<double,int>*&v); 
    };

  private:
    SLES                              sles;
    ::esi::Preconditioner<double,int> *pre;
    ::esi::Operator<double,int>       *op;
};

}}


#endif




