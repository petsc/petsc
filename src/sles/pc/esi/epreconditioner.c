/*
      Wrappers for PETSc PC ESI implementation
*/

#include "esi/petsc/preconditioner.h"

esi::petsc::Preconditioner<double,int>::Preconditioner(MPI_Comm comm)
{
  int      ierr;

  ierr = PCCreate(comm,&this->pc);if (ierr) return;
  ierr = PetscObjectSetOptionsPrefix((PetscObject)this->pc,"esi_");
  ierr = PCSetFromOptions(this->pc);

  this->pobject = (PetscObject)this->pc;
  ierr = PetscObjectGetComm((PetscObject)this->pc,&this->comm);if (ierr) return;
}

esi::petsc::Preconditioner<double,int>::Preconditioner(PC ipc)
{
  int ierr;
  this->pc      = ipc;
  this->pobject = (PetscObject)this->pc;
  ierr = PetscObjectGetComm((PetscObject)this->pc,&this->comm);if (ierr) return;
  ierr = PetscObjectReference((PetscObject)ipc);if (ierr) return;
}

esi::petsc::Preconditioner<double,int>::~Preconditioner()
{
  int ierr;
  ierr = PetscObjectDereference((PetscObject)this->pc);if (ierr) return;
}

esi::ErrorCode esi::petsc::Preconditioner<double,int>::getInterface(const char* name, void *& iface)
{
  PetscTruth flg;

  if (!PetscStrcmp(name,"esi::Object",&flg),flg){
    iface = (void *) (esi::Object *) this;
  } else if (!PetscStrcmp(name,"esi::Operator",&flg),flg){
    iface = (void *) (esi::Operator<double,int> *) this;
  } else if (!PetscStrcmp(name,"esi::Preconditioner",&flg),flg){
    iface = (void *) (esi::Preconditioner<double,int> *) this;
  } else if (!PetscStrcmp(name,"esi::Solver",&flg),flg){
    iface = (void *) (esi::Solver<double,int> *) this;
  } else if (!PetscStrcmp(name,"PC",&flg),flg){
    iface = (void *) this->pc;
  } else if (!PetscStrcmp(name,"esi::petsc::Preconditioner",&flg),flg){
    iface = (void *) (esi::petsc::Preconditioner<double,int> *) this;
  } else {
    iface = 0;
  }
  return 0;
}

esi::ErrorCode esi::petsc::Preconditioner<double,int>::getInterfacesSupported(esi::Argv * list)
{
  list->appendArg("esi::Object");
  list->appendArg("esi::Operator");
  list->appendArg("esi::Preconditioner");
  list->appendArg("esi::Solver");
  list->appendArg("esi::petsc::Preconditioner");
  list->appendArg("PC");
  return 0;
}


esi::ErrorCode esi::petsc::Preconditioner<double,int>::apply( esi::Vector<double,int> &xx,esi::Vector<double,int> &yy)
{
  int ierr;
  Vec py,px;

  ierr = yy.getInterface("Vec",reinterpret_cast<void*&>(py));
  ierr = xx.getInterface("Vec",reinterpret_cast<void*&>(px));

  ierr = PCSetVector(this->pc,px);CHKERRQ(ierr);
  return PCApply(this->pc,px,py);
}

esi::ErrorCode esi::petsc::Preconditioner<double,int>::solve( esi::Vector<double,int> &xx,esi::Vector<double,int> &yy)
{
  return this->apply(xx,yy);
}

esi::ErrorCode esi::petsc::Preconditioner<double,int>::solveLeft( esi::Vector<double,int> &xx,esi::Vector<double,int> &yy)
{
  int ierr;
  Vec py,px;

  ierr = yy.getInterface("Vec",reinterpret_cast<void*&>(py));CHKERRQ(ierr);
  ierr = xx.getInterface("Vec",reinterpret_cast<void*&>(px));CHKERRQ(ierr);

  return PCApplySymmetricLeft(this->pc,px,py);
}

esi::ErrorCode esi::petsc::Preconditioner<double,int>::solveRight( esi::Vector<double,int> &xx,esi::Vector<double,int> &yy)
{
  int ierr;
  Vec py,px;

  ierr = yy.getInterface("Vec",reinterpret_cast<void*&>(py));CHKERRQ(ierr);
  ierr = xx.getInterface("Vec",reinterpret_cast<void*&>(px));CHKERRQ(ierr);

  return PCApplySymmetricRight(this->pc,px,py);
}

esi::ErrorCode esi::petsc::Preconditioner<double,int>::applyB( esi::Vector<double,int> &xx,esi::Vector<double,int> &yy)
{
  int    ierr;
  Vec    py,px,work;
  PCSide iside;

  ierr = yy.getInterface("Vec",reinterpret_cast<void*&>(py));
  ierr = xx.getInterface("Vec",reinterpret_cast<void*&>(px));
  ierr = VecDuplicate(py,&work);CHKERRQ(ierr);
  if (this->side == esi::PRECONDITIONER_LEFT)      iside = PC_LEFT;
  if (this->side == esi::PRECONDITIONER_RIGHT)     iside = PC_RIGHT;
  if (this->side == esi::PRECONDITIONER_TWO_SIDED) iside = PC_SYMMETRIC;
  ierr = PCApplyBAorAB(this->pc,iside,px,py,work);CHKERRQ(ierr);
  ierr = VecDestroy(work);CHKERRQ(ierr);CHKERRQ(ierr);
  return 0;
}

esi::ErrorCode esi::petsc::Preconditioner<double,int>::setup()
{
  return 0;
}

esi::ErrorCode esi::petsc::Preconditioner<double,int>::setPreconditionerSide(esi::PreconditionerSide iside)
{
  this->side = iside;
  return 0;
}

esi::ErrorCode esi::petsc::Preconditioner<double,int>::getPreconditionerSide(esi::PreconditionerSide & iside)
{
  iside = this->side;
  return 0;
}

esi::ErrorCode esi::petsc::Preconditioner<double,int>::setOperator( esi::Operator<double,int> &op)
{
  /*
        For now require Operator to be a PETSc Mat
  */
  Mat A;
  int ierr = op.getInterface("Mat",reinterpret_cast<void*&>(A));CHKERRQ(ierr);
  ierr = PCSetOperators(this->pc,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  return 0;
}

// --------------------------------------------------------------------------
namespace esi{namespace petsc{
  template<class Scalar,class Ordinal> class PreconditionerFactory : public virtual ::esi::PreconditionerFactory<Scalar,Ordinal>
{
  public:

    // constructor
    PreconditionerFactory(void){};
  
    // Destructor.
    virtual ~PreconditionerFactory(void){};

    // Construct a Preconditioner
    virtual ::esi::ErrorCode getPreconditioner(char *commname,void *comm,::esi::Preconditioner<Scalar,Ordinal>*&v)
    {
      PetscTruth flg;
      int        ierr = PetscStrcmp(commname,"MPI",&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ1(1,"Does not support %s, only supports MPI",commname);
      v = new esi::petsc::Preconditioner<Scalar,Ordinal>((MPI_Comm)comm);
      return 0;
    };
};
}}

/* ::esi::petsc::PreconditionerFactory<double,int> PFInstForIntel64CompilerBug; */

EXTERN_C_BEGIN
::esi::PreconditionerFactory<double,int> *create_esi_petsc_preconditionerfactory(void)
{
  return dynamic_cast< ::esi::PreconditionerFactory<double,int> *>(new esi::petsc::PreconditionerFactory<double,int>);
}
EXTERN_C_END
