/*
      Wrappers for esi::petsc::Matrix ESI implementation
*/

#include "esi/petsc/matrix.h"

esi::petsc::Matrix<double,int>::Matrix(esi::IndexSpace<int> *inrmap,esi::IndexSpace<int> *incmap)
{
  int      ierr,rn,rN,cn,cN;
  MPI_Comm *comm;

  ierr = inrmap->getRunTimeModel("MPI",static_cast<void *>(comm));
  ierr = inrmap->getLocalSize(rn);
  ierr = inrmap->getGlobalSize(rN);
  ierr = inrmap->getLocalSize(cn);
  ierr = inrmap->getGlobalSize(cN);
  ierr = MatCreate(*comm,rn,cn,rN,cN,&this->mat);if (ierr) return;
  ierr = PetscObjectSetOptionsPrefix((PetscObject)this->mat,"esi");
  ierr = MatSetFromOptions(this->mat);

  this->rmap = inrmap;
  this->cmap = incmap;
  ierr = inrmap->addReference();
  ierr = incmap->addReference();

  this->pobject = (PetscObject)this->mat;
  PetscObjectGetComm((PetscObject)this->mat,&this->comm);
}


esi::petsc::Matrix<double,int>::Matrix(Mat mat)
{
  int m,n,M,N,ierr;

  this->mat  = mat;
  
  this->pobject = (PetscObject)this->mat;
  PetscObjectGetComm((PetscObject)this->mat,&this->comm);
  ierr = MatGetLocalSize(mat,&m,&n);
  ierr = MatGetSize(mat,&M,&N);
  this->rmap = new esi::petsc::IndexSpace<int>(this->comm,m,M);
  this->cmap = new esi::petsc::IndexSpace<int>(this->comm,n,N);
}


esi::petsc::Matrix<double,int>::~Matrix()
{
  int ierr;
  ierr = MatDestroy(this->mat);
}

esi::ErrorCode esi::petsc::Matrix<double,int>::getInterface(const char* name, void *& iface)
{
  PetscTruth flg;

  if (!PetscStrcmp(name,"esi::Object",&flg),flg){
    iface = (void *) (esi::Object *) this;
  } else if (!PetscStrcmp(name,"esi::Operator",&flg),flg){
    iface = (void *) (esi::Operator<double,int> *) this;
  } else if (!PetscStrcmp(name,"esi::MatrixData",&flg),flg){
    iface = (void *) (esi::MatrixData<int> *) this;
  } else if (!PetscStrcmp(name,"esi::MatrixRowReadAccess",&flg),flg){
    iface = (void *) (esi::MatrixRowReadAccess<double,int> *) this;
  } else if (!PetscStrcmp(name,"esi::MatrixRowWriteAccess",&flg),flg){
    iface = (void *) (esi::MatrixRowWriteAccess<double,int> *) this;
  } else if (!PetscStrcmp(name,"esi::petsc::Matrix",&flg),flg){
    iface = (void *) (esi::petsc::Matrix<double,int> *) this;
  } else {
    iface = 0;
  }
  return 0;
}

esi::ErrorCode esi::petsc::Matrix<double,int>::getInterfacesSupported(esi::Argv * list)
{
  list->appendArg("esi::Object");
  list->appendArg("esi::Operator");
  list->appendArg("esi::MatrixData");
  list->appendArg("esi::MatrixRowReadAccess");
  list->appendArg("esi::MatrixRowWriteAccess");
  list->appendArg("esi::petsc::Matrix");
  return 0;
}


esi::ErrorCode esi::petsc::Matrix<double,int>::apply( esi::Vector<double,int> &xx,esi::Vector<double,int> &yy)
{
  int ierr;
  Vec py,px;

  ierr = yy.getInterface("Vec",static_cast<void*>(py));
  ierr = xx.getInterface("Vec",static_cast<void*>(px));

  return MatMult(this->mat,px,py);
}

esi::ErrorCode esi::petsc::Matrix<double,int>::setup()
{
  int ierr;
  ierr = MatAssemblyBegin(this->mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return MatAssemblyEnd(this->mat,MAT_FINAL_ASSEMBLY);
}

esi::ErrorCode esi::petsc::Matrix<double,int>::getIndexSpaces(esi::IndexSpace<int>*& rowIndexSpace, esi::IndexSpace<int>*& colIndexSpace)
{
  rowIndexSpace = this->rmap;
  colIndexSpace = this->cmap;
  return 0;
}

esi::ErrorCode esi::petsc::Matrix<double,int>::isLoaded(bool &State)
{
  return MatAssembled(this->mat,(PetscTruth *)&State);
}

esi::ErrorCode esi::petsc::Matrix<double,int>::isAllocated(bool &State)
{
  State = 1;
  return 0;
}

esi::ErrorCode esi::petsc::Matrix<double,int>::loadComplete()
{
  return this->setup();
}

esi::ErrorCode esi::petsc::Matrix<double,int>::allocate(int rowlengths[])
{
  return 0;
}

esi::ErrorCode esi::petsc::Matrix<double,int>::getDiagonal(esi::Vector<double,int> &diagVector)
{
  int ierr;
  Vec py;

  ierr = diagVector.getInterface("Vec",static_cast<void*>(py));
  return MatGetDiagonal(this->mat,py);
}

esi::ErrorCode esi::petsc::Matrix<double,int>::getGlobalSizes(int& rows, int& columns)
{
  int ierr = MatGetSize(this->mat,&rows,&columns);CHKERRQ(ierr);
}

esi::ErrorCode esi::petsc::Matrix<double,int>::getLocalSizes(int& rows, int& columns)
{
  int ierr = MatGetLocalSize(this->mat,&rows,&columns);CHKERRQ(ierr);
}

esi::ErrorCode esi::petsc::Matrix<double,int>::getRowNonzeros(int row,int &length)
{
  int ierr = MatGetRow(this->mat,row,&length,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  return  MatRestoreRow(this->mat,row,&length,PETSC_NULL,PETSC_NULL);
}

esi::ErrorCode esi::petsc::Matrix<double,int>::setRowLength(int row,int length)
{

  return 1;
}

esi::ErrorCode esi::petsc::Matrix<double,int>::copyOutRow(int row, double* coefs, int* colIndices,int alength,int &length)
{
  int         *col;
  PetscScalar *values;
  
  int ierr = MatGetRow(this->mat,row,&length,&col,&values);CHKERRQ(ierr);
  if (length > alength) SETERRQ(1,"Not enough room for values");
  ierr = PetscMemcpy(coefs,values,length*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemcpy(colIndices,col,length*sizeof(int));CHKERRQ(ierr);
  ierr = MatRestoreRow(this->mat,row,&length,&col,&values);CHKERRQ(ierr);
  
}

esi::ErrorCode esi::petsc::Matrix<double,int>::getRow(int row, int& length, double*& coefs, int*& colIndices)
{
  return MatGetRow(this->mat,row,&length,&colIndices,&coefs);
}

esi::ErrorCode esi::petsc::Matrix<double,int>::getRowCoefs(int row, int& length, double*& coefs)
{
  return MatGetRow(this->mat,row,&length,PETSC_NULL,&coefs);
}

esi::ErrorCode esi::petsc::Matrix<double,int>::getRowIndices(int row, int& length, int*& colIndices)
{
  return MatGetRow(this->mat,row,&length,&colIndices,PETSC_NULL);
}

esi::ErrorCode esi::petsc::Matrix<double,int>::restoreRow(int row, int& length, double*& coefs, int*& colIndices)
{
  return MatRestoreRow(this->mat,row,&length,&colIndices,&coefs);
}

esi::ErrorCode esi::petsc::Matrix<double,int>::restoreRowCoefs(int row, int& length, double*& coefs)
{
  return MatRestoreRow(this->mat,row,&length,PETSC_NULL,&coefs);
}

esi::ErrorCode esi::petsc::Matrix<double,int>::restoreRowIndices(int row, int& length, int*& colIndices)
{
  return MatRestoreRow(this->mat,row,&length,&colIndices,PETSC_NULL);
}

esi::ErrorCode esi::petsc::Matrix<double,int>::copyInRow(int row,double* coefs, int* colIndices, int length)
{
  return MatSetValues(this->mat,1,&row,length,colIndices,coefs,INSERT_VALUES);  
}

esi::ErrorCode esi::petsc::Matrix<double,int>::sumIntoRow(int row, double* coefs, int* colIndices,int length)
{
  return MatSetValues(this->mat,1,&row,length,colIndices,coefs,ADD_VALUES);  
}

esi::ErrorCode esi::petsc::Matrix<double,int>::rowMax(int row, double &result)
{
  int         ierr,length,i;
  PetscScalar *values;

  ierr =  MatGetRow(this->mat,row,&length,PETSC_NULL,&values);CHKERRQ(ierr);
  if (values) {
    result = values[0];
    for (i=1; i<length; i++) result = PetscMax(result,values[i]);
  }
  ierr =  MatRestoreRow(this->mat,row,&length,PETSC_NULL,&values);CHKERRQ(ierr);
}

esi::ErrorCode esi::petsc::Matrix<double,int>::rowMin(int row, double &result)
{
  int         ierr,length,i;
  PetscScalar *values;

  ierr =  MatGetRow(this->mat,row,&length,PETSC_NULL,&values);CHKERRQ(ierr);
  if (values) {
    result = values[0];
    for (i=1; i<length; i++) result = PetscMin(result,values[i]);
  }
  ierr =  MatRestoreRow(this->mat,row,&length,PETSC_NULL,&values);CHKERRQ(ierr);
}

esi::ErrorCode esi::petsc::Matrix<double,int>::getRowSum(esi::Vector<double,int>& rowSumVector)
{
  int         i,ierr,rstart,rend,length,j;
  PetscScalar *values,sum;
  Vec         py;

  ierr = rowSumVector.getInterface("Vec",static_cast<void*>(py));

  ierr = MatGetOwnershipRange(this->mat,&rstart,&rend);CHKERRQ(ierr);
  for ( i=rstart; i<rend; i++) {
    ierr =  MatGetRow(this->mat,i,&length,PETSC_NULL,&values);CHKERRQ(ierr);
    sum  = 0.0;
    for ( j=0; j<length; j++ ) {
      sum += values[j];
    }
    ierr =  MatRestoreRow(this->mat,i,&length,PETSC_NULL,&values);CHKERRQ(ierr);
    ierr = VecSetValues(py,1,&i,&sum,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(py);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(py);CHKERRQ(ierr);

  return 0;
}

/* ------------------------------------------------------------------------------------------------*/

namespace esi{namespace petsc{
  template<class Scalar,class Ordinal> class OperatorFactory : public virtual ::esi::OperatorFactory<Scalar,Ordinal>
{
  public:

    // constructor
    OperatorFactory(void){};
  
    // Destructor.
    virtual ~OperatorFactory(void){};

    // Interface for gov::cca::Component
#if defined(PETSC_HAVE_CCA)
    virtual void setServices(gov::cca::Services *svc)
    {
      svc->addProvidesPort(this,svc->createPortInfo("getOperator", "esi::OperatorFactory", 0));
    };
#endif

    // Construct a Operator
    virtual ::esi::ErrorCode getOperator(::esi::IndexSpace<Ordinal>&rmap,::esi::IndexSpace<Ordinal>&cmap,::esi::Operator<Scalar,Ordinal>*&v)
    {
      v = new esi::petsc::Matrix<Scalar,Ordinal>(&rmap,&cmap);
      return 0;
    };
};
}}
EXTERN_C_BEGIN
#if defined(PETSC_HAVE_CCA)
gov::cca::Component *create_esi_petsc_operatorfactory(void)
{
  return dynamic_cast<gov::cca::Component *>(new esi::petsc::OperatorFactory<double,int>);
}
#else
void *create_esi_petsc_operatorfactory(void)
{
  return (void *)(new esi::petsc::OperatorFactory<double,int>);
}
#endif
EXTERN_C_END



 



