


/*
   Makes a PETSc vector look like a ESI
*/

#include "esi/petsc/vector.h"

esi::petsc::Vector<double,int>::Vector( ::esi::IndexSpace<int> *inmap)
{
  ::esi::ErrorCode  ierr;
  int               n,N;
  MPI_Comm          *icomm;

  ierr = inmap->getRunTimeModel("MPI",reinterpret_cast<void *&>(icomm));if (ierr) return;

  ierr = inmap->getLocalSize(n);if (ierr) return;
  ierr = inmap->getGlobalSize(N);if (ierr) return;
  ierr = VecCreate(*icomm,&this->vec);if (ierr) return;
  ierr = VecSetSizes(this->vec,n,N);if (ierr) return;
  ierr = PetscObjectSetOptionsPrefix((PetscObject)this->vec,"esi_");if (ierr) return;
  ierr = VecSetFromOptions(this->vec);if (ierr) return;
  this->pobject = (PetscObject)this->vec;
  this->map = (::esi::IndexSpace<int> *)inmap;
  this->map->addReference();
  ierr = PetscObjectGetComm((PetscObject)this->vec,&this->comm);if (ierr) return;
}

esi::petsc::Vector<double,int>::Vector( Vec pvec)
{
  ::esi::ErrorCode  ierr;
  int               n,N;
  
  this->vec     = pvec;
  this->pobject = (PetscObject)this->vec;
  ierr = PetscObjectReference((PetscObject)pvec);if (ierr) return;
  ierr = PetscObjectGetComm((PetscObject)this->vec,&this->comm);if (ierr) return;

  ierr = VecGetSize(pvec,&N);if (ierr) return;
  ierr = VecGetLocalSize(pvec,&n);if (ierr) return;
  this->map = new esi::petsc::IndexSpace<int>(this->comm,n,N);
}

esi::petsc::Vector<double,int>::~Vector()
{
  int ierr;
  ierr = this->map->deleteReference();if (ierr) return;
  ierr = VecDestroy(this->vec);if (ierr) return;
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
  ierr = lmap->getInterface("esi::IndexSpace",reinterpret_cast<void *&>(amap));CHKERRQ(ierr);
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

  ierr = yy.getInterface("esi::petsc::Vector",reinterpret_cast<void *&>(y));CHKERRQ(ierr);
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
  
  ierr = yy.getInterface("esi::petsc::Vector",reinterpret_cast<void *&>(y));CHKERRQ(ierr);
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

  esi::petsc::Vector<double,int> *y;  ierr = yy.getInterface("esi::petsc::Vector",reinterpret_cast<void *&>(y));CHKERRQ(ierr);
  if (!y) return 1;
  return  VecDot(this->vec,y->vec,&product);
}

/*
  Currently only works if both vectors are PETSc 
*/
::esi::ErrorCode esi::petsc::Vector<double,int>::axpy(  ::esi::Vector<double,int> &yy,double scalar)
{
  int ierr;

  esi::petsc::Vector<double,int> *y;  ierr = yy.getInterface("esi::petsc::Vector",reinterpret_cast<void *&>(y));CHKERRQ(ierr);
  if (!y) return 1;
  return VecAXPY(&scalar,y->vec,this->vec);
}

::esi::ErrorCode esi::petsc::Vector<double,int>::axpby(double dy1,  ::esi::Vector<double,int> &yy1,double y2,  ::esi::Vector<double,int> &yy2)
{
  int ierr;

  esi::petsc::Vector<double,int> *y;  ierr = yy1.getInterface("esi::petsc::Vector",reinterpret_cast<void *&>(y));CHKERRQ(ierr);
  if (!y) return 1;
  esi::petsc::Vector<double,int> *w;  ierr = yy2.getInterface("esi::petsc::Vector",reinterpret_cast<void *&>(w));CHKERRQ(ierr);
  if (!w) return 1;
  ierr = VecCopy(y->vec,this->vec); CHKERRQ(ierr);
  ierr = VecScale(&dy1,this->vec); CHKERRQ(ierr);
  ierr = VecAXPY(&y2,w->vec,this->vec); CHKERRQ(ierr);
  return(0);
}

/*
  Currently only works if both vectors are PETSc 
*/
::esi::ErrorCode esi::petsc::Vector<double,int>::aypx(double scalar,  ::esi::Vector<double,int> &yy)
{
  int ierr;
  
  esi::petsc::Vector<double,int> *y;  ierr = yy.getInterface("esi::petsc::Vector",reinterpret_cast<void *&>(y));CHKERRQ(ierr);
  if (!y) return 1;
  return VecAYPX(&scalar,y->vec,this->vec);
}

::esi::ErrorCode esi::petsc::Vector<double,int>::getCoefPtrReadLock(double *&pointer) 
{
  return VecGetArray(this->vec,&pointer);
}

::esi::ErrorCode esi::petsc::Vector<double,int>::getCoefPtrReadWriteLock(double *&pointer)
{
  return VecGetArray(this->vec,&pointer);
}

::esi::ErrorCode esi::petsc::Vector<double,int>::releaseCoefPtrLock(double *&pointer)  
{
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

/* ::esi::petsc::VectorFactory<double,int> VFInstForIntel64CompilerBug; */

EXTERN_C_BEGIN
#if defined(PETSC_HAVE_CCA)
gov::cca::Component *create_esi_petsc_vectorfactory(void)
{
  return dynamic_cast<gov::cca::Component *>(new esi::petsc::VectorFactory<double,int>);
}
#else
::esi::VectorFactory<double,int> *create_esi_petsc_vectorfactory(void)
{
  return dynamic_cast< ::esi::VectorFactory<double,int> *>(new esi::petsc::VectorFactory<double,int>);
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
      int ierr = lmap.getInterface("Petra_ESI_IndexSpace",reinterpret_cast<void *&>(map));CHKERRQ(ierr);
      if (!map) SETERRQ(1,"Requires Petra_ESI_IndexSpace");
      v = new Petra_ESI_Vector<Scalar,Ordinal>(*map);
      //      ierr = map->addReference();CHKERRQ(ierr);  /* Petra has bug and does not increase reference count */
      if (!v) SETERRQ(1,"Unable to create new Petra_ESI_Vector");
      return 0;
    }; 
};

EXTERN_C_BEGIN
#if defined(PETSC_HAVE_CCA)
gov::cca::Component *create_petra_esi_vectorfactory(void)
{
  return dynamic_cast<gov::cca::Component *>(new Petra_ESI_VectorFactory<double,int>);
}
#else
::esi::VectorFactory<double,int> *create_petra_esi_vectorfactory(void)
{
  return dynamic_cast< ::esi::VectorFactory<double,int> *>(new Petra_ESI_VectorFactory<double,int>);
}
#endif
EXTERN_C_END
#endif


