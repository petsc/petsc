#ifndef lint
static char vcid[] = "$Id: vector.c,v 1.19 1995/04/03 20:02:14 curfman Exp bsmith $";
#endif

/* 
   This is where the abstract vector operations are defined
 */
#include "vecimpl.h"    /*I "vec.h" I*/

/*@
     VecValidVector - Returns 1 if this is a valid vector else 0.

  Input Parameter:
.  v - the object to check

  Keywords: vector, valid
@*/
int VecValidVector(Vec v)
{
  if (!v) return 0;
  if (v->cookie != VEC_COOKIE) return 0;
  return 1;
}
/*@
     VecDot  - Computes vector dot product.

  Input Parameters:
.  x,y - the vectors

  Output Parameter:
.  val - the dot product

  Keywords: vector, dot product, inner product
@*/
int VecDot(Vec x, Vec y, Scalar *val)
{
  int ierr;
  VALIDHEADER(x,VEC_COOKIE); VALIDHEADER(y,VEC_COOKIE);
  CHKSAME(x,y);
  PLogEventBegin(VECTOR_DOT,x,y,0,0);
  ierr = (*x->ops->dot)(x,y,val); CHKERR(ierr);
  PLogEventEnd(VECTOR_DOT,x,y,0,0);
  return 0;
}

/*@
     VecNorm  - Computes vector two norm.

  Input Parameters:
.  x - the vector

  Output Parameter:
.  val - the norm 

  Keywords: vector, norm
@*/
int VecNorm(Vec x,double *val)  
{
  int ierr;
  VALIDHEADER(x,VEC_COOKIE);
  PLogEventBegin(VECTOR_DOT,x,0,0,0);
  ierr = (*x->ops->norm)(x,val); CHKERR(ierr);
  PLogEventEnd(VECTOR_DOT,x,0,0,0);
  return 0;
}
/*@
     VecASum  - Computes vector one norm.

  Input Parameters:
.  x - the vector

  Output Parameter:
.  val - the sum 

  Keywords: vector, sum
@*/
int VecASum(Vec x,double *val)
{
  int ierr;
  VALIDHEADER(x,VEC_COOKIE);
  PLogEventBegin(VECTOR_ASUM,x,0,0,0);
  ierr = (*x->ops->asum)(x,val); CHKERR(ierr);
  PLogEventEnd(VECTOR_ASUM,x,0,0,0);
  return 0;
}

/*@
     VecMax  - Computes maximum of vector and its location.

  Input Parameters:
.  x - the vector

  Output Parameter:
.  val - the max 
.  p - the location

  Keywords: vector, max
@*/
int VecMax(Vec x,int *p,double *val)
{
  int ierr;
  VALIDHEADER(x,VEC_COOKIE);
  PLogEventBegin(VECTOR_MAX,x,0,0,0);
  ierr = (*x->ops->max)(x,p,val); CHKERR(ierr);
  PLogEventEnd(VECTOR_MAX,x,0,0,0);
  return 0;
}

/*@
     VecTDot  - Non-Hermitian vector dot product. That is, it does NOT
              use the complex conjugate.

  Input Parameters:
.  x, y - the vectors

  Output Parameter:
.  val - the dot product

  Keywords: vector, dot product, inner product, non-hermitian
@*/
int VecTDot(Vec x,Vec y,Scalar *val) 
{
  int ierr;
  VALIDHEADER(x,VEC_COOKIE); VALIDHEADER(y,VEC_COOKIE);
  CHKSAME(x,y);
  PLogEventBegin(VECTOR_TDOT,x,y,0,0);
  ierr = (*x->ops->tdot)(x,y,val); CHKERR(ierr);
  PLogEventEnd(VECTOR_TDOT,x,y,0,0);
  return 0;
}

/*@
     VecScale  - Scales a vector. 

  Input Parameters:
.  x - the vector
.  alpha - the scalar

  Keywords: vector, scale
@*/
int VecScale(Scalar *alpha,Vec x)
{
  int ierr;
  VALIDHEADER(x,VEC_COOKIE);
  PLogEventBegin(VECTOR_SCALE,x,0,0,0);
  ierr = (*x->ops->scal)(alpha,x); CHKERR(ierr);
  PLogEventEnd(VECTOR_SCALE,x,0,0,0);
  return 0;
}

/*@
     VecCopy  - Copys a vector. 

  Input Parameters:
.  x  - the vector

  Output Parameters:
.  y  - the copy

  Keywords: vector, copy
@*/
int VecCopy(Vec x,Vec y)
{
  int ierr;
  VALIDHEADER(x,VEC_COOKIE); VALIDHEADER(y,VEC_COOKIE);
  PLogEventBegin(VECTOR_COPY,x,y,0,0);
  ierr = (*x->ops->copy)(x,y); CHKERR(ierr);
  PLogEventEnd(VECTOR_COPY,x,y,0,0);
  return 0;
}
 
/*@
     VecSet  - Sets all components of a vector to a scalar. 

  Input Parameters:
.  alpha - the scalar

  Output Parameters:
.  x  - the vector

  Keywords: vector, set
@*/
int VecSet(Scalar *alpha,Vec x) 
{
  int ierr;
  VALIDHEADER(x,VEC_COOKIE);
  PLogEventBegin(VECTOR_SET,x,0,0,0);
  ierr = (*x->ops->set)(alpha,x); CHKERR(ierr);
  PLogEventEnd(VECTOR_SET,x,0,0,0);
  return 0;
} 

/*@
     VecAXPY  -  Computes y <- alpha x + y. 

  Input Parameters:
.  alpha - the scalar
.  x,y  - the vectors

  Keywords: vector, saxpy
@*/
int VecAXPY(Scalar *alpha,Vec x,Vec y)
{
  int ierr;
  VALIDHEADER(x,VEC_COOKIE); 
  VALIDHEADER(y,VEC_COOKIE);
  PLogEventBegin(VECTOR_AXPY,x,y,0,0);
  ierr = (*x->ops->axpy)(alpha,x,y); CHKERR(ierr);
  PLogEventEnd(VECTOR_AXPY,x,y,0,0);
  return 0;
} 
/*@
     VecAYPX  -  Computes y <- x + alpha y.

  Input Parameters:
.  alpha - the scalar
.  x,y  - the vectors

@*/
int VecAYPX(Scalar *alpha,Vec x,Vec y)
{
  int ierr;
  VALIDHEADER(x,VEC_COOKIE); VALIDHEADER(y,VEC_COOKIE);
  CHKSAME(x,y);
  PLogEventBegin(VECTOR_AYPX,x,y,0,0);
  ierr =  (*x->ops->aypx)(alpha,x,y); CHKERR(ierr);
  PLogEventEnd(VECTOR_AYPX,x,y,0,0);
  return 0;
} 
/*@
     VecSwap  -  Swaps x and y.

  Input Parameters:
.  x,y  - the vectors
@*/
int VecSwap(Vec x,Vec y)
{
  int ierr;
  VALIDHEADER(x,VEC_COOKIE);  VALIDHEADER(y,VEC_COOKIE);
  CHKSAME(x,y);
  PLogEventBegin(VECTOR_SWAP,x,y,0,0);
  ierr = (*x->ops->swap)(x,y); CHKERR(ierr);
  PLogEventEnd(VECTOR_SWAP,x,y,0,0);
  return 0;
}
/*@
     VecWAXPY  -  Computes w <- alpha x + y.

  Input Parameters:
.  alpha - the scalar
.  x,y  - the vectors

  Output Parameter:
.  w - the result
@*/
int VecWAXPY(Scalar *alpha,Vec x,Vec y,Vec w)
{
  int ierr;
  VALIDHEADER(x,VEC_COOKIE); VALIDHEADER(y,VEC_COOKIE);
  VALIDHEADER(w,VEC_COOKIE);
  CHKSAME(x,y); CHKSAME(y,w);
  PLogEventBegin(VECTOR_WAXPY,x,y,w,0);
  ierr =  (*x->ops->waxpy)(alpha,x,y,w); CHKERR(ierr);
  PLogEventEnd(VECTOR_WAXPY,x,y,w,0);
  return 0;
}
/*@
     VecPMult  -  Computes the componentwise multiplication w = x*y.

  Input Parameters:
.  x,y  - the vectors

  Output Parameter:
.  w - the result

@*/
int VecPMult(Vec x,Vec y,Vec w)
{
  VALIDHEADER(x,VEC_COOKIE); VALIDHEADER(y,VEC_COOKIE);
  VALIDHEADER(w,VEC_COOKIE);
  CHKSAME(x,y); CHKSAME(y,w);
  return (*x->ops->pmult)(x,y,w);
} 
/*@
     VecPDiv  -  Computes the componentwise division w = x/y.

  Input Parameters:
.  x,y  - the vectors

  Output Parameter:
.  w - the result
@*/
int VecPDiv(Vec x,Vec y,Vec w)
{
  VALIDHEADER(x,VEC_COOKIE); VALIDHEADER(y,VEC_COOKIE);
  VALIDHEADER(w,VEC_COOKIE);
  CHKSAME(x,y); CHKSAME(y,w);
  return (*x->ops->pdiv)(x,y,w);
}
/*@
     VecCreate  -  Creates a vector from another vector. Use VecDestroy()
                 to free the space. Use VecGetVecs() to get several 
                 vectors.

  Input Parameters:
.  v - a vector to mimic

  Output Parameter:
.  newv - location to put new vector
@*/
int VecCreate(Vec v,Vec *newv) 
{
  VALIDHEADER(v,VEC_COOKIE);
  return   (*v->ops->create_vector)(v,newv);
}
/*@
     VecDestroy  -  Destroys  a vector created with VecCreate().

  Input Parameters:
.  v  - the vector
@*/
int VecDestroy(Vec v)
{
  VALIDHEADER(v,VEC_COOKIE);
  return (*v->destroy)((PetscObject )v);
}

/*@
     VecGetVecs  -  Obtains several vectors. Use VecFreeVecs() to free the 
                  space. Use VecCreate() to get a single vector.

  Input Parameters:
.  m - the number of vectors to obtain
.  v - a vector

  Output Parameters:
.  V - location to put pointer to array of vectors.
@*/
int VecGetVecs(Vec v,int m,Vec **V)  
{
  VALIDHEADER(v,VEC_COOKIE);
  return (*v->ops->obtain_vectors)( v, m,V );
}

/*@
     VecFreeVecs  -  Frees a block of vectors obtained with VecGetVecs().

  Input Parameters:
.  vv - pointer to array of vector pointers
.  m - the number of vectors previously obtained
@*/
int VecFreeVecs(Vec *vv,int m)
{
  if (!vv) SETERR(1,"Null vectors");
  VALIDHEADER(*vv,VEC_COOKIE);
  return (*(*vv)->ops->release_vectors)( vv, m );
}




/*@
     VecSetValues - Insert or add values into certain locations in vector. 
         These
         values may be cached, you must call VecBeginAssembly() and 
         VecEndAssembly() after you have completed all calls to 
         VecSetValues() or VecInsertValues(). Note: calls with SetValues
         and InsertValues may not be interlaced.

  Input Parameters:
.  x - vector to insert in
.  ni - number of elements to add
.  ix - indices where to add
.  y - array of values
.  iora - either InsertValues or AddValues

  Notes:
.  x[ix[i]] = y[i], for i=0,...,ni-1.

@*/
int VecSetValues(Vec x,int ni,int *ix,Scalar *y,InsertMode iora) 
{
  VALIDHEADER(x,VEC_COOKIE);
  return (*x->ops->insertvalues)( x, ni,ix, y,iora );
}


/*@
    VecBeginAssembly - Begins assembling global vector. Call after
      all calls to VecAddValues() or VecInsertValues(). Note that you cannot
      mix calls to VecAddValues() and VecInsertValues().

  Input Parameter:
.   vec - the vector to assemble
@*/
int VecBeginAssembly(Vec vec)
{
  int ierr;
  VALIDHEADER(vec,VEC_COOKIE);
  PLogEventBegin(VECTOR_ASSEMBLY_BEGIN,vec,0,0,0);
  if (vec->ops->beginassm) {ierr = (*vec->ops->beginassm)(vec); CHKERR(ierr);}
  PLogEventEnd(VECTOR_ASSEMBLY_BEGIN,vec,0,0,0);
  return 0;
}

/*@
    VecEndAssembly - End assembling global vector. Call after
      VecBeginAssembly().

  Input Parameter:
.   vec - the vector to assemble
@*/
int VecEndAssembly(Vec vec)
{
  int ierr;
  VALIDHEADER(vec,VEC_COOKIE);
  PLogEventBegin(VECTOR_ASSEMBLY_END,vec,0,0,0);
  if (vec->ops->endassm) {ierr = (*vec->ops->endassm)(vec); CHKERR(ierr);}
  PLogEventEnd(VECTOR_ASSEMBLY_END,vec,0,0,0);
  return 0;
}

/*@
     VecMTDot  - Non-Hermitian vector multiple dot product. 
         That is, it does NOT use the complex conjugate.

  Input Parameters:
.  nv - number of vectors
.  x - one vector
.  y - array of vectors.  Note that vectors are pointers

  Output Parameter:
.  val - array of the dot products
@*/
int VecMTDot(int nv,Vec x,Vec *y,Scalar *val)
{
  int ierr;
  VALIDHEADER(x,VEC_COOKIE); VALIDHEADER(*y,VEC_COOKIE);
  CHKSAME(x,*y);
  PLogEventBegin(VECTOR_MTDOT,x,*y,0,0);
  ierr = (*x->ops->mtdot)(nv,x,y,val); CHKERR(ierr);
  PLogEventEnd(VECTOR_MTDOT,x,*y,0,0);
  return 0;
}
/*@
     VecMDot  - Vector multiple dot product. 

  Input Parameters:
.  nv - number of vectors
.  x - one vector
.  y - array of vectors. 

  Output Parameter:
.  val - array of the dot products
@*/
int VecMDot(int nv,Vec x,Vec *y,Scalar *val)
{
  int ierr;
  VALIDHEADER(x,VEC_COOKIE); VALIDHEADER(*y,VEC_COOKIE);
  CHKSAME(x,*y);
  PLogEventBegin(VECTOR_MDOT,x,*y,0,0);
  ierr = (*x->ops->mdot)(nv,x,y,val); CHKERR(ierr);
  PLogEventEnd(VECTOR_MDOT,x,*y,0,0);
  return 0;
}

/*@
     VecMAXPY  -  Computes y <- alpha[j] x[j] + y. 

  Input Parameters:
.  nv - number of scalars and x-vectors
.  alpha - array of scalars
.  x  - one vectors
.  y  - array of vectors
@*/
int  VecMAXPY(int nv,Scalar *alpha,Vec x,Vec *y)
{
  int ierr;
  VALIDHEADER(x,VEC_COOKIE); VALIDHEADER(*y,VEC_COOKIE);
  CHKSAME(x,*y);
  PLogEventBegin(VECTOR_MAXPY,x,*y,0,0);
  ierr = (*x->ops->maxpy)(nv,alpha,x,y); CHKERR(ierr);
  PLogEventEnd(VECTOR_MAXPY,x,*y,0,0);
  return 0;
} 

/*@
   VecGetArray - Return pointer to vector data. For default seqential 
        vectors returns pointer to array containing data. Otherwise 
        implementation dependent. You must call VecRestoreArray() 
        when you no longer need access to the array.

  Input Parameters:
.   x - the vector

   Output Parameters:
.   a - location to put pointer to the array.
@*/
int VecGetArray(Vec x,Scalar **a)
{
  VALIDHEADER(x,VEC_COOKIE);
  return (*x->ops->getarray)(x,a);
}

/*@
   VecRestoreArray - Restores array in vector.

  Input Parameters:
.   x - the vector
.   a - location of pointer to array obtained from VecGetArray().


@*/
int VecRestoreArray(Vec x,Scalar **a)
{
  VALIDHEADER(x,VEC_COOKIE);
  if (x->ops->restorearray) return (*x->ops->getarray)(x,a);
  else return 0;
}

/*@
    VecView  - Allows the user to view a vector. This routines is intended
               to be replacable with fancy graphical based viewing.

  Input Parameters:
.  v - the vector
.  ptr - a pointer to a viewer ctx
@*/
int VecView(Vec v,Viewer ptr)
{
  VALIDHEADER(v,VEC_COOKIE);
  return (*v->view)((PetscObject)v,ptr);
}

/*@
    VecGetSize - Returns the global number of elements of the vector.

  Input Parameter:
.   x - the vector

  Output Parameters:
.  size - the length of the vector

  Note:
  A related routine is VecGetLocalSize().
@*/
int VecGetSize(Vec x,int *size)
{
  VALIDHEADER(x,VEC_COOKIE);
  return (*x->ops->getsize)(x,size);
}

/*@
    VecGetLocalSize - Returns the number of elements of the vector stored 
               in local memory. This may mean different things 
               for different implementations, so use with care.

  Input Parameter:
.   x - the vector

  Output Parameters:
.  size - the length of the local piece of the vector

  Note:
  A related routine is VecGetSize().
@*/
int VecGetLocalSize(Vec x,int *size)
{
  VALIDHEADER(x,VEC_COOKIE);
  return (*x->ops->localsize)(x,size);
}

/*@
    VecGetOwnershipRange - Returns the range of indices owned by 
        this processor. Assumes the vectors are laid out with 
        first n1 elements on first processor, next n2 on second, etc..

  Input Parameter:
.   x - the vector

  Output Parameters:
.  low - first local element
.  high - one more then last local element.
@*/
int VecGetOwnershipRange(Vec x,int *low,int *high)
{
  VALIDHEADER(x,VEC_COOKIE);
  return (*x->ops->getrange)(x,low,high);
}

/* Default routines for obtaining and releasing; */
/* may be used by any implementation */
int Veiobtain_vectors(Vec w,int m,Vec **V )
{
  Vec *v;
  int  i;
  *V = v = (Vec *) MALLOC( m * sizeof(Vec *) );
  for (i=0; i<m; i++) VecCreate(w,v+i);
  return 0;
}

int Veirelease_vectors( Vec *v, int m )
{
  int i;
  for (i=0; i<m; i++) VecDestroy(v[i]);
  FREE( v );
  return 0;
}


