/*
      Wrappers for PETSc PC ESI implementation
*/

#include "esi/petsc/preconditioner.h"

esi::petsc::Preconditioner<double,int>::Preconditioner(PC pc)
{
  int ierr;
  this->pc      = pc;
  this->pobject = (PetscObject)this->pc;
  PetscObjectGetComm((PetscObject)this->pc,&this->comm);
  ierr = PetscObjectReference((PetscObject)pc);
}


esi::petsc::Preconditioner<double,int>::~Preconditioner()
{
  int ierr;
  ierr = PetscObjectDereference((PetscObject)this->pc);
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

  ierr = yy.getInterface("Vec",reinterpret_cast<void*&>(py));
  ierr = xx.getInterface("Vec",reinterpret_cast<void*&>(px));

  return PCApplySymmetricLeft(this->pc,px,py);
}

esi::ErrorCode esi::petsc::Preconditioner<double,int>::solveRight( esi::Vector<double,int> &xx,esi::Vector<double,int> &yy)
{
  int ierr;
  Vec py,px;

  ierr = yy.getInterface("Vec",reinterpret_cast<void*&>(py));
  ierr = xx.getInterface("Vec",reinterpret_cast<void*&>(px));

  return PCApplySymmetricRight(this->pc,px,py);
}

esi::ErrorCode esi::petsc::Preconditioner<double,int>::applyB( esi::Vector<double,int> &xx,esi::Vector<double,int> &yy)
{
  int    ierr;
  Vec    py,px,work;
  PCSide side;

  ierr = yy.getInterface("Vec",reinterpret_cast<void*&>(py));
  ierr = xx.getInterface("Vec",reinterpret_cast<void*&>(px));
  ierr = VecDuplicate(py,&work);CHKERRQ(ierr);
  if (this->side == esi::PRECONDITIONER_LEFT)      side = PC_LEFT;
  if (this->side == esi::PRECONDITIONER_RIGHT)     side = PC_RIGHT;
  if (this->side == esi::PRECONDITIONER_TWO_SIDED) side = PC_SYMMETRIC;
  ierr = PCApplyBAorAB(this->pc,side,px,py,work);CHKERRQ(ierr);
  ierr = VecDestroy(work);CHKERRQ(ierr);CHKERRQ(ierr);
  return 0;
}

esi::ErrorCode esi::petsc::Preconditioner<double,int>::setup()
{
  return 0;
}

esi::ErrorCode esi::petsc::Preconditioner<double,int>::setPreconditionerSide(esi::PreconditionerSide side)
{
  this->side = side;
  return 0;
}

esi::ErrorCode esi::petsc::Preconditioner<double,int>::getPreconditionerSide(esi::PreconditionerSide & side)
{
  side = this->side;
  return 0;
}

esi::ErrorCode esi::petsc::Preconditioner<double,int>::setOperator( esi::Operator<double,int> &op)
{
  /*
        For now require Operator to be a PETSc Mat
  */
  Mat A;
  int ierr = op.getInterface("Mat",reinterpret_cast<void*&>(A));
  ierr = PCSetOperators(this->pc,A,A,DIFFERENT_NONZERO_PATTERN);
  return 0;
}
