


/*
   Makes a PETSc vector look like a ESI
*/

#include "esi/petsc/vector.h"

esi::petsc::Vector<double,int>::Vector( ::esi::IndexSpace<int> *inmap)
{
  ::esi::ErrorCode  ierr;
  int               n,N;
  MPI_Comm          *comm;

  ierr = inmap->getRunTimeModel("MPI",static_cast<void *>(comm));

  ierr = inmap->getLocalSize(n);
  ierr = inmap->getGlobalSize(N);
  ierr = VecCreate(*comm,&this->vec);
  ierr = VecSetSizes(this->vec,n,N);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)this->vec,"esi");
  ierr = VecSetFromOptions(this->vec);
  this->pobject = (PetscObject)this->vec;
  this->map = (::esi::IndexSpace<int> *)inmap;
  this->map->addReference();
  PetscObjectGetComm((PetscObject)this->vec,&this->comm);
}

esi::petsc::Vector<double,int>::Vector( Vec pvec)
{
  ::esi::ErrorCode  ierr;
  int               n,N;
  
  this->vec     = pvec;
  this->pobject = (PetscObject)this->vec;
  ierr = PetscObjectReference((PetscObject)pvec);
  ierr = PetscObjectGetComm((PetscObject)this->vec,&this->comm);

  ierr = VecGetSize(pvec,&N);
  ierr = VecGetLocalSize(pvec,&n);
  this->map = new esi::petsc::IndexSpace<int>(this->comm,n,N);
}

esi::petsc::Vector<double,int>::~Vector()
{
  int ierr;
  this->map->deleteReference();
  ierr = VecDestroy(this->vec);
}

/* ---------------esi::Object methods ------------------------------------------------------------ */

::esi::ErrorCode esi::petsc::Vector<double,int>::getInterface(const char* name, void *& iface)
{
  PetscTruth flg;
  if (PetscStrcmp(name,"esi::Object",&flg),flg){
    iface = (void *) (::esi::Object *) this;
  } else if (PetscStrcmp(name,"esi::Vector",&flg),flg){
    iface = (void *) (::esi::Vector<double,int> *) this;
  } else if (PetscStrcmp(name,"esi::petsc::Vector",&flg),flg){
    iface = (void *) (::esi::petsc::Vector<double,int> *) this;
  } else if (PetscStrcmp(name,"esi::VectorReplaceAccess",&flg),flg){
    iface = (void *) (::esi::VectorReplaceAccess<double,int> *) this;
  } else if (PetscStrcmp(name,"Vec",&flg),flg){
    iface = (void *) this->vec;
  } else {
    iface = 0;
  }
  return 0;
}


::esi::ErrorCode esi::petsc::Vector<double,int>::getInterfacesSupported(::esi::Argv * list)
{
  list->appendArg("esi::Object");
  list->appendArg("esi::Vector");
  list->appendArg("esi::VectorReplaceAccess");
  list->appendArg("esi::petsc::Vector");
  list->appendArg("Vec");
  return 0;
}

/*
    Note: this returns the map used in creating the vector;
  it is not the same as the PETSc map contained inside the PETSc vector
*/
::esi::ErrorCode esi::petsc::Vector<double,int>::getIndexSpace( ::esi::IndexSpace<int>*& outmap)
{
  outmap = this->map;
  return 0;
}

::esi::ErrorCode esi::petsc::Vector<double,int>::getGlobalSize( int & dim) 
{
  return VecGetSize(this->vec,&dim);
}

::esi::ErrorCode esi::petsc::Vector<double,int>::getLocalSize( int & dim) 
{
  return VecGetLocalSize(this->vec,&dim);
}

::esi::ErrorCode esi::petsc::Vector<double,int>::clone( ::esi::Vector<double,int>*& outvector)  
{
  int                    ierr;
  ::esi::IndexSpace<int> *lmap; 
  ::esi::IndexSpace<int> *amap; 

  ierr = this->getIndexSpace(lmap);CHKERRQ(ierr);
  ierr = lmap->getInterface("esi::IndexSpace",static_cast<void *>(amap));CHKERRQ(ierr);
  outvector = (::esi::Vector<double,int> *) new esi::petsc::Vector<double,int>(amap);
  return 0;
}

/*
  Currently only works if both vectors are PETSc 
*/
::esi::ErrorCode esi::petsc::Vector<double,int>::copy( ::esi::Vector<double,int> &yy)
{
  esi::petsc::Vector<double,int> *y = 0;  
  int                            ierr;

  ierr = yy.getInterface("esi::petsc::Vector",static_cast<void*>(y));CHKERRQ(ierr);
  if (!y) return 1;

  return VecCopy(y->vec,this->vec);
}

::esi::ErrorCode esi::petsc::Vector<double,int>::put( double scalar)
{
  return VecSet(&scalar,this->vec);
}

::esi::ErrorCode esi::petsc::Vector<double,int>::scale( double scalar)
{
  return VecScale(&scalar,this->vec);
}

/*
  Currently only works if both vectors are PETSc 
*/
::esi::ErrorCode esi::petsc::Vector<double,int>::scaleDiagonal( ::esi::Vector<double,int> &yy)
{
  ::esi::ErrorCode                 ierr;
  esi::petsc::Vector<double,int> *y;  
  
  ierr = yy.getInterface("esi::petsc::Vector",static_cast<void*>(y));CHKERRQ(ierr);
  if (!y) return 1;

  return VecPointwiseMult(y->vec,this->vec,this->vec);
}

::esi::ErrorCode esi::petsc::Vector<double,int>::norm1(double &scalar) 
{
  return VecNorm(this->vec,NORM_1,&scalar);
}

::esi::ErrorCode esi::petsc::Vector<double,int>::norm2(double &scalar) 
{
  return VecNorm(this->vec,NORM_2,&scalar);
}

::esi::ErrorCode esi::petsc::Vector<double,int>::norm2squared(double &scalar) 
{
  int ierr = VecNorm(this->vec,NORM_2,&scalar);CHKERRQ(ierr);
  scalar *= scalar;
  return 0;
}

::esi::ErrorCode esi::petsc::Vector<double,int>::normInfinity(double &scalar) 
{
  return VecNorm(this->vec,NORM_INFINITY,&scalar);
}

/*
  Currently only works if both vectors are PETSc 
*/
::esi::ErrorCode esi::petsc::Vector<double,int>::dot( ::esi::Vector<double,int> &yy,double &product) 
{
  int ierr;

  esi::petsc::Vector<double,int> *y;  ierr = yy.getInterface("esi::petsc::Vector",static_cast<void*>(y));CHKERRQ(ierr);
  if (!y) return 1;
  return  VecDot(this->vec,y->vec,&product);
}

/*
  Currently only works if both vectors are PETSc 
*/
::esi::ErrorCode esi::petsc::Vector<double,int>::axpy(  ::esi::Vector<double,int> &yy,double scalar)
{
  int ierr;

  esi::petsc::Vector<double,int> *y;  ierr = yy.getInterface("esi::petsc::Vector",static_cast<void*>(y));CHKERRQ(ierr);
  if (!y) return 1;
  return VecAXPY(&scalar,y->vec,this->vec);
}

::esi::ErrorCode esi::petsc::Vector<double,int>::axpby(double y1,  ::esi::Vector<double,int> &yy1,double y2,  ::esi::Vector<double,int> &yy2)
{
  int ierr;

  esi::petsc::Vector<double,int> *y;  ierr = yy1.getInterface("esi::petsc::Vector",static_cast<void*>(y));CHKERRQ(ierr);
  if (!y) return 1;
  esi::petsc::Vector<double,int> *w;  ierr = yy2.getInterface("esi::petsc::Vector",static_cast<void*>(w));CHKERRQ(ierr);
  if (!w) return 1;
  ierr = VecCopy(y->vec,this->vec); CHKERRQ(ierr);
  ierr = VecScale(&y1,this->vec); CHKERRQ(ierr);
  ierr = VecAXPY(&y2,w->vec,this->vec); CHKERRQ(ierr);
}

/*
  Currently only works if both vectors are PETSc 
*/
::esi::ErrorCode esi::petsc::Vector<double,int>::aypx(double scalar,  ::esi::Vector<double,int> &yy)
{
  int ierr;
  
  esi::petsc::Vector<double,int> *y;  ierr = yy.getInterface("esi::petsc::Vector",static_cast<void*>(y));CHKERRQ(ierr);
  if (!y) return 1;
  return VecAYPX(&scalar,y->vec,this->vec);
}

::esi::ErrorCode esi::petsc::Vector<double,int>::getCoefPtrReadLock(double *&pointer) 
{
  int ierr;

  return VecGetArray(this->vec,&pointer);
}

::esi::ErrorCode esi::petsc::Vector<double,int>::getCoefPtrReadWriteLock(double *&pointer)
{
  int ierr;

  return VecGetArray(this->vec,&pointer);
}

::esi::ErrorCode esi::petsc::Vector<double,int>::releaseCoefPtrLock(double *&pointer)  
{
  int ierr;
  return VecRestoreArray(this->vec,&pointer);
}

::esi::ErrorCode esi::petsc::Vector<double,int>::setArrayPointer(double *pointer,int length)
{
  return VecPlaceArray(this->vec,pointer);
}

// --------------------------------------------------------------------------------------------------------
namespace esi{namespace petsc{
  template<class Scalar,class Ordinal> class VectorFactory : public virtual ::esi::VectorFactory<Scalar,Ordinal>
{
  public:

    // constructor
    VectorFactory(void){};
  
    // Destructor.
    virtual ~VectorFactory(void){};

    // Interface for gov::cca::Component
#if defined(PETSC_HAVE_CCA)
    virtual void setServices(gov::cca::Services *svc)
    {
      svc->addProvidesPort(this,svc->createPortInfo("getVector", "esi::VectorFactory", 0));
    };
#endif

    // Construct a Vector
    virtual ::esi::ErrorCode getVector(::esi::IndexSpace<Ordinal>&map,::esi::Vector<Scalar,Ordinal>*&v)
    {
      v = new esi::petsc::Vector<Scalar,Ordinal>(&map);
      return 0;
    };
};
}}
EXTERN_C_BEGIN
#if defined(PETSC_HAVE_CCA)
gov::cca::Component *create_esi_petsc_vectorfactory(void)
{
  return dynamic_cast<gov::cca::Component *>(new esi::petsc::VectorFactory<double,int>);
}
#else
void *create_esi_petsc_vectorfactory(void)
{
  return (void *)(new esi::petsc::VectorFactory<double,int>);
}
#endif
EXTERN_C_END

// --------------------------------------------------------------------------------------------------------
#if defined(PETSC_HAVE_TRILINOS)
#define PETRA_MPI /* used by Ptera to indicate MPI code */
#include "Petra_ESI_Vector.h"

template<class Scalar,class Ordinal> class Petra_ESI_VectorFactory : public virtual ::esi::VectorFactory<Scalar,Ordinal>
{
  public:

    // constructor
    Petra_ESI_VectorFactory(void) {};
  
    // Destructor.
    virtual ~Petra_ESI_VectorFactory(void) {};

    // Interface for gov::cca::Component
#if defined(PETSC_HAVE_CCA)
    virtual void setServices(gov::cca::Services *svc)
    {
      svc->addProvidesPort(this,svc->createPortInfo("getVector", "esi::VectorFactory", 0));
    };
#endif

    // Construct a Vector
    virtual ::esi::ErrorCode getVector(::esi::IndexSpace<Ordinal>&lmap,::esi::Vector<Scalar,Ordinal>*&v)
    {
      Petra_ESI_IndexSpace<Ordinal> *map;
      int ierr = lmap.getInterface("Petra_ESI_IndexSpace",static_cast<void *>(map));CHKERRQ(ierr);
      if (!map) SETERRQ(1,"Requires Petra_ESI_IndexSpace");
      v = new Petra_ESI_Vector<Scalar,Ordinal>(*map);
      //      ierr = map->addReference();CHKERRQ(ierr);  /* Petra has bug and does not increase reference count */
      if (!v) SETERRQ(1,"Unable to create new Petsc_ESI_Vector");
      return 0;
    }; 
};
#endif

EXTERN_C_BEGIN
#if defined(PETSC_HAVE_CCA)
gov::cca::Component *create_petra_esi_vectorfactory(void)
{
  return dynamic_cast<gov::cca::Component *>(new Petra_ESI_VectorFactory<double,int>);
}
#else
void *create_petra_esi_vectorfactory(void)
{
  return (void *)(new Petra_ESI_VectorFactory<double,int>);
}
#endif
EXTERN_C_END

// --------------------------------------------------------------------------------------------------------

// CCAFFEINE expects each .so file to have a getComponentList function.
// See dccafe/cxx/dc/framework/ComponentFactory.h for details.
EXTERN_C_BEGIN
char **getComponentList() {
  static char *list[7];
  list[0] = "create_esi_petsc_vectorfactory esi::petsc::Vector";
  list[1] = "create_petra_esi_vectorfactory Petra_ESI_Vector";
  list[2] = "create_esi_petsc_indexspacefactory esi::petsc::IndexSpace";
  list[3] = "create_petra_esi_indexspacefactory Petra_ESI_IndexSpace";
  list[4] = "create_esi_petsc_operatorfactory esi::petsc::Operator";
  list[5] = "create_petra_esi_operatorfactory Petra_ESI_CRS_Matrix";
  list[6] = 0;
  return list;
}
EXTERN_C_END

  /* -----should be in eindexspace.c but Trilinos sucks----------------------------------------------------------*/
#if defined(PETSC_HAVE_TRILINOS)

template<class Ordinal> class Petra_ESI_IndexSpaceFactory : public virtual ::esi::IndexSpaceFactory<Ordinal>
{
  public:

    // Destructor.
  virtual ~Petra_ESI_IndexSpaceFactory(void){};

    // Interface for gov::cca::Component
#if defined(PETSC_HAVE_CCA)
    virtual void setServices(gov::cca::Services *svc)
    {
      svc->addProvidesPort(this,svc->createPortInfo("getIndexSpace", "esi::IndexSpaceFactory", 0));
    };
#endif

    // Construct a IndexSpace
    virtual ::esi::ErrorCode getIndexSpace(const char * name,void *comm,int m,::esi::IndexSpace<Ordinal>*&v)
    {
      PetscTruth ismpi;
      int ierr = PetscStrcmp(name,"MPI",&ismpi);CHKERRQ(ierr);
      if (!ismpi) SETERRQ1(1,"%s not supported, only MPI supported as RunTimeModel",name);
      Petra_Comm *pcomm = new Petra_Comm(*(MPI_Comm*)comm);
      v = new Petra_ESI_IndexSpace<Ordinal>(-1,m,0,*pcomm); 
      if (!v) SETERRQ(1,"Unable to create Petra_ESI_IndexSpace");
      return 0;
    };

};
EXTERN_C_BEGIN
#if defined(PETSC_HAVE_CCA)
gov::cca::Component *create_petra_esi_indexspacefactory(void)
{
  return dynamic_cast<gov::cca::Component *>(new Petra_ESI_IndexSpaceFactory<int>);
}
#else
void *create_petra_esi_indexspacefactory(void)
{
  return (void *)(new Petra_ESI_IndexSpaceFactory<int>);
}
#endif
EXTERN_C_END
#endif

  /* -----should be in ematrix.c but Trilinos sucks----------------------------------------------------------*/

#if defined(PETSC_HAVE_TRILINOS)
#include "esi/petsc/matrix.h"
#include "Petra_ESI_CRS_Matrix.h"

/*
         This class is the same as the Petra_ESI_CRS_Matrix class except it puts values into the Petra_CRS_Grap()
*/
template<class Scalar,class Ordinal> class MyPetra_ESI_CRS_Matrix : public virtual Petra_ESI_CRS_Matrix<Scalar,Ordinal>
{
  public:

  MyPetra_ESI_CRS_Matrix(Petra_DataAccess CV,const Petra_CRS_Graph& graph) :  Petra_ESI_Object(), Petra_RDP_CRS_Matrix(CV, graph), Petra_ESI_CRS_Matrix<Scalar,Ordinal>(CV, graph){graph_ = (Petra_CRS_Graph*)&graph;SetStaticGraph(false);};

  virtual ~MyPetra_ESI_CRS_Matrix() { };

  virtual esi::ErrorCode copyInRow(Ordinal row, Scalar* coefs, Ordinal* colIndices, Ordinal length)
    { int ierr;
      ierr = graph_->InsertGlobalIndices(row, length, colIndices);CHKERRQ(((ierr == 1) ? 0 : ierr));
      ierr = Petra_ESI_CRS_Matrix<Scalar,Ordinal>::copyInRow(row,coefs,colIndices,length);CHKERRQ(ierr); 
      //  ierr = this->setup();CHKERRQ(ierr);
      return 0;
     }

  Petra_CRS_Graph *graph_; 
};


template<class Scalar,class Ordinal> class Petra_ESI_CRS_OperatorFactory : public virtual ::esi::OperatorFactory<Scalar,Ordinal>
{
  public:

    // constructor
    Petra_ESI_CRS_OperatorFactory(void){};
  
    // Destructor.
    virtual ~Petra_ESI_CRS_OperatorFactory(void){};

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
      int       ierr;
      Petra_Map *rowmap,*colmap;
      ierr = rmap.getInterface("Petra_Map",static_cast<void *>(rowmap));CHKERRQ(ierr);
      if (!rowmap) SETERRQ(1,"Petra requires all IndexSpaces be Petra_ESI_IndexSpaces");
      ierr = cmap.getInterface("Petra_Map",static_cast<void *>(colmap));CHKERRQ(ierr);
      if (!colmap) SETERRQ(1,"Petra requires all IndexSpaces be Petra_ESI_IndexSpaces");
      Petra_CRS_Graph  *graph = new Petra_CRS_Graph(Copy,*(Petra_BlockMap*)rowmap,*(Petra_BlockMap*)colmap,200);
      ierr = rmap.addReference();CHKERRQ(ierr);
      ierr = cmap.addReference();CHKERRQ(ierr);
      v = new MyPetra_ESI_CRS_Matrix<Scalar,Ordinal>(Copy,*graph);
      return 0;
    };
};

EXTERN_C_BEGIN
#if defined(PETSC_HAVE_CCA)
gov::cca::Component *create_petra_esi_operatorfactory(void)
{
  return dynamic_cast<gov::cca::Component *>(new Petra_ESI_CRS_OperatorFactory<double,int>);
}
#else
void *create_petra_esi_operatorfactory(void)
{
  return (void *)(new Petra_ESI_CRS_OperatorFactory<double,int>);
}
#endif
EXTERN_C_END
#endif

