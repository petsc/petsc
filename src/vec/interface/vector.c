

#ifndef lint
static char vcid[] = "$Id: vector.c,v 1.84 1996/07/23 14:17:02 bsmith Exp bsmith $";
#endif
/*
     Provides the interface functions for all vector operations.
   These are the vector functions the user calls.
*/
#include "vecimpl.h"    /*I "vec.h" I*/

/*@
   VecValid - Checks whether a vector object is valid.

   Input Parameter:
.  v - the object to check

   Output Parameter:
   flg - flag indicating vector status, either
$     PETSC_TRUE if vector is valid;
$     PETSC_FALSE otherwise.

.keywords: vector, valid
@*/
int VecValid(Vec v,PetscTruth *flg)
{
  PetscValidIntPointer(flg);
  if (!v)                           *flg = PETSC_FALSE;
  else if (v->cookie != VEC_COOKIE) *flg = PETSC_FALSE;
  else                              *flg = PETSC_TRUE;
  return 0;
}
/*@
   VecDot - Computes the vector dot product.

   Input Parameters:
.  x, y - the vectors

   Output Parameter:
.  alpha - the dot product

   Notes for Users of Complex Numbers:
   For complex vectors, VecDot() computes 
$      val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y.

   Use VecTDot() for the indefinite form
$      val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

.keywords: vector, dot product, inner product

.seealso: VecMDot(), VecTDot()
@*/
int VecDot(Vec x, Vec y, Scalar *val)
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE); PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidScalarPointer(val);
  PetscCheckSameType(x,y);
  PLogEventBegin(VEC_Dot,x,y,0,0);
  ierr = (*x->ops.dot)(x,y,val); CHKERRQ(ierr);
  PLogEventEnd(VEC_Dot,x,y,0,0);
  return 0;
}

/*@
   VecNorm  - Computes the vector norm.

   Input Parameters:
.  x - the vector
.  type - one of NORM_1, NORM_2, NORM_INFINITY

   Output Parameter:
.  val - the norm 

.keywords: vector, norm

.seealso: 
@*/
int VecNorm(Vec x,NormType type,double *val)  
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidScalarPointer(val);
  PLogEventBegin(VEC_Norm,x,0,0,0);
  ierr = (*x->ops.norm)(x,type,val); CHKERRQ(ierr);
  PLogEventEnd(VEC_Norm,x,0,0,0);
  return 0;
}

/*@
   VecMax - Determines the maximum vector component and its location.

   Input Parameter:
.  x - the vector

   Output Parameters:
.  val - the maximum component
.  p - the location of val

.keywords: vector, maximum

.seealso: VecNorm(), VecMin()
@*/
int VecMax(Vec x,int *p,double *val)
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidScalarPointer(val);
  PLogEventBegin(VEC_Max,x,0,0,0);
  ierr = (*x->ops.max)(x,p,val); CHKERRQ(ierr);
  PLogEventEnd(VEC_Max,x,0,0,0);
  return 0;
}

/*@
   VecMin - Determines the minimum vector component and its location.

   Input Parameters:
.  x - the vector

   Output Parameter:
.  val - the minimum component
.  p - the location of val

.keywords: vector, minimum

.seealso: VecMax()
@*/
int VecMin(Vec x,int *p,double *val)
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidScalarPointer(val);
  PLogEventBegin(VEC_Min,x,0,0,0);
  ierr = (*x->ops.min)(x,p,val); CHKERRQ(ierr);
  PLogEventEnd(VEC_Min,x,0,0,0);
  return 0;
}

/*@
   VecTDot - Computes an indefinite vector dot product. That is, this
   routine does NOT use the complex conjugate.

   Input Parameters:
.  x, y - the vectors

   Output Parameter:
.  val - the dot product

   Notes for Users of Complex Numbers:
   For complex vectors, VecTDot() computes the indefinite form
$      val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

   Use VecDot() for the inner product
$      val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y.

.keywords: vector, dot product, inner product

.seealso: VecDot(), VecMTDot()
@*/
int VecTDot(Vec x,Vec y,Scalar *val) 
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE); PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidScalarPointer(val);
  PetscCheckSameType(x,y);
  PLogEventBegin(VEC_TDot,x,y,0,0);
  ierr = (*x->ops.tdot)(x,y,val); CHKERRQ(ierr);
  PLogEventEnd(VEC_TDot,x,y,0,0);
  return 0;
}

/*@
   VecScale - Scales a vector. 

   Input Parameters:
.  x - the vector
.  alpha - the scalar

   Output Parameter:
.  x - the scaled vector

   Note:
   For a vector with n components, VecScale() computes 
$      x[i] = alpha * x[i], for i=1,...,n.

.keywords: vector, scale
@*/
int VecScale(Scalar *alpha,Vec x)
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidScalarPointer(alpha);
  PLogEventBegin(VEC_Scale,x,0,0,0);
  ierr = (*x->ops.scale)(alpha,x); CHKERRQ(ierr);
  PLogEventEnd(VEC_Scale,x,0,0,0);
  return 0;
}

/*@
   VecCopy - Copies a vector. 

   Input Parameter:
.  x - the vector

   Output Parameter:
.  y - the copy

.keywords: vector, copy

.seealso: VecDuplicate()
@*/
int VecCopy(Vec x,Vec y)
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE); PetscValidHeaderSpecific(y,VEC_COOKIE);
  PLogEventBegin(VEC_Copy,x,y,0,0);
  ierr = (*x->ops.copy)(x,y); CHKERRQ(ierr);
  PLogEventEnd(VEC_Copy,x,y,0,0);
  return 0;
}
 
/*@
   VecSet - Sets all components of a vector to a scalar. 

   Input Parameters:
.  alpha - the scalar
.  x  - the vector

   Output Parameter:
.  x  - the vector

   Note:
   For a vector with n components, VecSet() computes
$      x[i] = alpha, for i=1,...,n.

.keywords: vector, set
@*/
int VecSet(Scalar *alpha,Vec x) 
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidScalarPointer(alpha);
  PLogEventBegin(VEC_Set,x,0,0,0);
  ierr = (*x->ops.set)(alpha,x); CHKERRQ(ierr);
  PLogEventEnd(VEC_Set,x,0,0,0);
  return 0;
} 

/*@C
   VecSetRandom - Sets all components of a vector to random numbers.

   Input Parameters:
.  rctx - the random number context, formed by PetscRandomCreate()
.  x  - the vector

   Output Parameter:
.  x  - the vector

   Example of Usage:
$    PetscRandomCreate(MPI_COMM_WORLD,RANDOM_DEFAULT,&rctx);
$    VecSetRandom(rctx,x);
$    PetscRandomDestroy(rctx);

.keywords: vector, set, random

.seealso: VecSet(), VecSetValues(), PetscRandomCreate(), PetscRandomDestroy()
@*/
int VecSetRandom(PetscRandom rctx,Vec x) 
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(rctx,PETSCRANDOM_COOKIE);
  PLogEventBegin(VEC_SetRandom,x,rctx,0,0);
  ierr = (*x->ops.setrandom)(rctx,x); CHKERRQ(ierr);
  PLogEventEnd(VEC_SetRandom,x,rctx,0,0);
  return 0;
} 

/*@
   VecAXPY - Computes y = alpha x + y. 

   Input Parameters:
.  alpha - the scalar
.  x, y  - the vectors

   Output Parameter:
.  y - output vector

.keywords: vector, saxpy

.seealso: VecAYPX(), VecMAXPY(), VecWAXPY()
@*/
int VecAXPY(Scalar *alpha,Vec x,Vec y)
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE);PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidScalarPointer(alpha);
  PLogEventBegin(VEC_AXPY,x,y,0,0);
  ierr = (*x->ops.axpy)(alpha,x,y); CHKERRQ(ierr);
  PLogEventEnd(VEC_AXPY,x,y,0,0);
  return 0;
} 

/*@
   VecAXPBY - Computes y = alpha x + beta y. 

   Input Parameters:
.  alpha,beta - the scalars
.  x, y  - the vectors

   Output Parameter:
.  y - output vector

.keywords: vector, saxpy

.seealso: VecAYPX(), VecMAXPY(), VecWAXPY(), VecAXPY()
@*/
int VecAXPBY(Scalar *alpha,Scalar *beta,Vec x,Vec y)
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE);PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidScalarPointer(alpha);
  PetscValidScalarPointer(beta);
  PLogEventBegin(VEC_AXPY,x,y,0,0);
  ierr = (*x->ops.axpby)(alpha,beta,x,y); CHKERRQ(ierr);
  PLogEventEnd(VEC_AXPY,x,y,0,0);
  return 0;
} 

/*@
   VecAYPX - Computes y = x + alpha y.

   Input Parameters:
.  alpha - the scalar
.  x, y  - the vectors

   Output Parameter:
.  y - output vector

.keywords: vector, saypx

.seealso: VecAXPY(), VecWAXPY()
@*/
int VecAYPX(Scalar *alpha,Vec x,Vec y)
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE); PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidScalarPointer(alpha);
  PLogEventBegin(VEC_AYPX,x,y,0,0);
  ierr =  (*x->ops.aypx)(alpha,x,y); CHKERRQ(ierr);
  PLogEventEnd(VEC_AYPX,x,y,0,0);
  return 0;
} 
/*@
   VecSwap - Swaps the vectors x and y.

   Input Parameters:
.  x, y  - the vectors

.keywords: vector, swap
@*/
int VecSwap(Vec x,Vec y)
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE);  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscCheckSameType(x,y);
  PLogEventBegin(VEC_Swap,x,y,0,0);
  ierr = (*x->ops.swap)(x,y); CHKERRQ(ierr);
  PLogEventEnd(VEC_Swap,x,y,0,0);
  return 0;
}
/*@
   VecWAXPY - Computes w = alpha x + y.

   Input Parameters:
.  alpha - the scalar
.  x, y  - the vectors

   Output Parameter:
.  w - the result

.keywords: vector, waxpy

.seealso: VecAXPY(), VecAYPX()
@*/
int VecWAXPY(Scalar *alpha,Vec x,Vec y,Vec w)
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE); PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(w,VEC_COOKIE);
  PetscValidScalarPointer(alpha);
  PetscCheckSameType(x,y); PetscCheckSameType(y,w);
  PLogEventBegin(VEC_WAXPY,x,y,w,0);
  ierr =  (*x->ops.waxpy)(alpha,x,y,w); CHKERRQ(ierr);
  PLogEventEnd(VEC_WAXPY,x,y,w,0);
  return 0;
}
/*@
   VecPointwiseMult - Computes the componentwise multiplication w = x*y.

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

.keywords: vector, multiply, componentwise

.seealso: VecPointwiseDivide()
@*/
int VecPointwiseMult(Vec x,Vec y,Vec w)
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE); PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(w,VEC_COOKIE);
  PLogEventBegin(VEC_PMult,x,y,w,0);
  ierr = (*x->ops.pointwisemult)(x,y,w); CHKERRQ(ierr);
  PLogEventEnd(VEC_PMult,x,y,w,0);
  return 0;
} 
/*@
   VecPointwiseDivide - Computes the componentwise division w = x/y.

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  w - the result

.keywords: vector, divide, componentwise

.seealso: VecPointwiseMult()
@*/
int VecPointwiseDivide(Vec x,Vec y,Vec w)
{
  PetscValidHeaderSpecific(x,VEC_COOKIE); PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(w,VEC_COOKIE);
  return (*x->ops.pointwisedivide)(x,y,w);
}
/*@C
   VecDuplicate - Creates a new vector of the same type as an existing vector.

   Input Parameters:
.  v - a vector to mimic

   Output Parameter:
.  newv - location to put new vector

   Notes:
   VecDuplicate() does not copy the vector, but rather allocates storage
   for the new vector.  Use VecCopy() to copy a vector.

   Use VecDestroy() to free the space. Use VecDuplicateVecs() to get several
   vectors. 

.keywords: vector, duplicate, create

.seealso: VecDestroy(), VecDuplicateVecs(), VecCreate(), VecCopy()
@*/
int VecDuplicate(Vec v,Vec *newv) 
{
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  return   (*v->ops.duplicate)(v,newv);
}
/*@C
   VecDestroy - Destroys a vector.

   Input Parameters:
.  v  - the vector

.keywords: vector, destroy

.seealso: VecDuplicate()
@*/
int VecDestroy(Vec v)
{
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  return (*v->destroy)((PetscObject )v);
}

/*@C
   VecDuplicateVecs - Creates several vectors of the same type as an existing vector.

   Input Parameters:
.  m - the number of vectors to obtain
.  v - a vector to mimic

   Output Parameter:
.  V - location to put pointer to array of vectors

   Notes:
   Use VecDestroyVecs() to free the space. Use VecDuplicate() to form a single
   vector.

.keywords: vector, get 

.seealso:  VecDestroyVecs(), VecDuplicate(), VecCreate()
@*/
int VecDuplicateVecs(Vec v,int m,Vec **V)  
{
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  return (*v->ops.getvecs)( v, m,V );
}

/*@C
   VecDestroyVecs - Frees a block of vectors obtained with VecDuplicateVecs().

   Input Parameters:
.  vv - pointer to array of vector pointers
.  m - the number of vectors previously obtained

.keywords: vector, free

.seealso: VecDuplicateVecs()
@*/
int VecDestroyVecs(Vec *vv,int m)
{
  if (!vv) SETERRQ(1,"VecDestroyVecs:Null vectors");
  PetscValidHeaderSpecific(*vv,VEC_COOKIE);
  return (*(*vv)->ops.destroyvecs)( vv, m );
}

/*@
   VecSetValues - Inserts or adds values into certain locations of a vector. 
   These values may be cached, so VecAssemblyBegin() and VecAssemblyEnd() 
   MUST be called after all calls to VecSetValues() have been completed.

   Input Parameters:
.  x - vector to insert in
.  ni - number of elements to add
.  ix - indices where to add
.  y - array of values
.  iora - either INSERT_VALUES or ADD_VALUES

   Notes: 
   x[ix[i]] = y[i], for i=0,...,ni-1.

   Calls to VecSetValues() with the INSERT_VALUES and ADD_VALUES 
   options cannot be mixed without intervening calls to the assembly
   routines.

.keywords: vector, set, values

.seealso:  VecAssemblyBegin(), VecAssemblyEnd()
@*/
int VecSetValues(Vec x,int ni,int *ix,Scalar *y,InsertMode iora) 
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PLogEventBegin(VEC_SetValues,x,0,0,0);
  ierr = (*x->ops.setvalues)( x, ni,ix, y,iora ); CHKERRQ(ierr);
  PLogEventEnd(VEC_SetValues,x,0,0,0);  
  return 0;
}

/*@
   VecAssemblyBegin - Begins assembling the vector.  This routine should
   be called after completing all calls to VecSetValues().

   Input Parameter:
.  vec - the vector

.keywords: vector, begin, assembly, assemble

.seealso: VecAssemblyEnd(), VecSetValues()
@*/
int VecAssemblyBegin(Vec vec)
{
  int ierr;
  PetscValidHeaderSpecific(vec,VEC_COOKIE);
  PLogEventBegin(VEC_AssemblyBegin,vec,0,0,0);
  if (vec->ops.assemblybegin) {
    ierr = (*vec->ops.assemblybegin)(vec); CHKERRQ(ierr);
  }
  PLogEventEnd(VEC_AssemblyBegin,vec,0,0,0);
  return 0;
}

/*@
   VecAssemblyEnd - Completes assembling the vector.  This routine should
   be called after VecAssemblyBegin().

   Input Parameter:
.  vec - the vector

.keywords: vector, end, assembly, assemble

.seealso: VecAssemblyBegin(), VecSetValues()
@*/
int VecAssemblyEnd(Vec vec)
{
  int ierr,flg;
  PetscValidHeaderSpecific(vec,VEC_COOKIE);
  PLogEventBegin(VEC_AssemblyEnd,vec,0,0,0);
  if (vec->ops.assemblyend) {
    ierr = (*vec->ops.assemblyend)(vec); CHKERRQ(ierr);
  }
  PLogEventEnd(VEC_AssemblyEnd,vec,0,0,0);
  ierr = OptionsHasName(PETSC_NULL,"-vec_view",&flg); CHKERRQ(ierr);
  if (flg) {
    Viewer viewer;
    ierr = ViewerFileOpenASCII(vec->comm,"stdout",&viewer);CHKERRQ(ierr);
    ierr = VecView(vec,viewer); CHKERRQ(ierr);
    ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-vec_view_matlab",&flg); CHKERRQ(ierr);
  if (flg) {
    Viewer viewer;
    ierr = ViewerFileOpenASCII(vec->comm,"stdout",&viewer);CHKERRQ(ierr);
    ierr = ViewerSetFormat(viewer,ASCII_FORMAT_MATLAB,"V");CHKERRQ(ierr);
    ierr = VecView(vec,viewer); CHKERRQ(ierr);
    ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-vec_view_draw",&flg); CHKERRQ(ierr);
  if (flg) {
    Viewer    viewer;
    ierr = ViewerDrawOpenX(vec->comm,0,0,0,0,300,300,&viewer); CHKERRQ(ierr);
    ierr = VecView(vec,viewer); CHKERRQ(ierr);
    ierr = ViewerFlush(viewer); CHKERRQ(ierr);
    ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-vec_view_draw_lg",&flg); CHKERRQ(ierr);
  if (flg) {
    Viewer    viewer;
    ierr = ViewerDrawOpenX(vec->comm,0,0,0,0,300,300,&viewer); CHKERRQ(ierr);
    ierr = ViewerSetFormat(viewer,VIEWER_FORMAT_DRAW_LG,0); CHKERRQ(ierr);
    ierr = VecView(vec,viewer); CHKERRQ(ierr);
    ierr = ViewerFlush(viewer); CHKERRQ(ierr);
    ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
  }
  return 0;
}

/*@
   VecMTDot - Computes indefinite vector multiple dot products. 
   That is, it does NOT use the complex conjugate.

   Input Parameters:
.  nv - number of vectors
.  x - one vector
.  y - array of vectors.  Note that vectors are pointers

   Output Parameter:
.  val - array of the dot products

   Notes for Users of Complex Numbers:
   For complex vectors, VecMTDot() computes the indefinite form
$      val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

   Use VecMDot() for the inner product
$      val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y.

.keywords: vector, dot product, inner product, non-Hermitian, multiple

.seealso: VecMDot(), VecTDot()
@*/
int VecMTDot(int nv,Vec x,Vec *y,Scalar *val)
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE); PetscValidHeaderSpecific(*y,VEC_COOKIE);
  PetscValidScalarPointer(val);
  PetscCheckSameType(x,*y);
  PLogEventBegin(VEC_MTDot,x,*y,0,0);
  ierr = (*x->ops.mtdot)(nv,x,y,val); CHKERRQ(ierr);
  PLogEventEnd(VEC_MTDot,x,*y,0,0);
  return 0;
}
/*@
   VecMDot - Computes vector multiple dot products. 

   Input Parameters:
.  nv - number of vectors
.  x - one vector
.  y - array of vectors. 

   Output Parameter:
.  val - array of the dot products

   Notes for Users of Complex Numbers:
   For complex vectors, VecMDot() computes 
$      val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y.

   Use VecMTDot() for the indefinite form
$      val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

.keywords: vector, dot product, inner product, multiple

.seealso: VecMTDot(), VecDot()
@*/
int VecMDot(int nv,Vec x,Vec *y,Scalar *val)
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE); PetscValidHeaderSpecific(*y,VEC_COOKIE);
  PetscValidScalarPointer(val);
  PetscCheckSameType(x,*y);
  PLogEventBegin(VEC_MDot,x,*y,0,0);
  ierr = (*x->ops.mdot)(nv,x,y,val); CHKERRQ(ierr);
  PLogEventEnd(VEC_MDot,x,*y,0,0);
  return 0;
}

/*@
   VecMAXPY - Computes y[j] = alpha[j] x + y[j]. 

   Input Parameters:
.  nv - number of scalars and x-vectors
.  alpha - array of scalars
.  x  - one vector
.  y  - array of vectors

   Output Parameter:
.  y  - array of vectors

.keywords: vector, saxpy, maxpy, multiple

.seealso: VecAXPY(), VecWAXPY(), VecAYPX()
@*/
int  VecMAXPY(int nv,Scalar *alpha,Vec x,Vec *y)
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE); PetscValidHeaderSpecific(*y,VEC_COOKIE);
  PetscValidScalarPointer(alpha);
  PetscCheckSameType(x,*y);
  PLogEventBegin(VEC_MAXPY,x,*y,0,0);
  ierr = (*x->ops.maxpy)(nv,alpha,x,y); CHKERRQ(ierr);
  PLogEventEnd(VEC_MAXPY,x,*y,0,0);
  return 0;
} 

/*@C
   VecGetArray - Returns a pointer to vector data. For default seqential 
   vectors, VecGetArray() returns a pointer to the data array. Otherwise,
   this routine is implementation dependent. You MUST call VecRestoreArray() 
   when you no longer need access to the array.

   Input Parameter:
.  x - the vector

   Output Parameter:
.  a - location to put pointer to the array

   Fortran Note:
   The Fortran interface is slightly different from that given below.
   See the Fortran chapter of the users manual and 
   petsc/src/vec/examples for details.

.keywords: vector, get, array

.seealso: VecRestoreArray(), VecGetArrays()
@*/
int VecGetArray(Vec x,Scalar **a)
{
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  return (*x->ops.getarray)(x,a);
}

/*@C
   VecGetArrays - Returns a pointer to the arrays in a set of vectors
   that were created by a call to VecDuplicateVecs().  You MUST call
   VecRestoreArrays() when you no longer need access to the array.

   Input Parameter:
.  x - the vectors
.  n - the number of vectors

   Output Parameter:
.  a - location to put pointer to the array

   Fortran Note:
   This routine is not supported in Fortran.

.keywords: vector, get, arrays

.seealso: VecGetArray(), VecRestoreArrays()
@*/
int VecGetArrays(Vec *x,int n,Scalar ***a)
{
  int    i,ierr;
  Scalar **q;
  PetscValidHeaderSpecific(*x,VEC_COOKIE);
  q = (Scalar **)PetscMalloc(n*sizeof(Scalar*)); CHKPTRQ(q);
  for (i=0; i<n; ++i) {
    ierr = VecGetArray(x[i],&q[i]); CHKERRQ(ierr);
  }
  *a = q;
  return 0;
}

/*@C
   VecRestoreArrays - Restores a group of vectors after VecGetArrays()
   has been called.

   Input Parameters:
.  x - the vector
.  n - the number of vectors
.  a - location of pointer to arrays obtained from VecGetArrays()

   Fortran Note:
   This routine is not supported in Fortran.

.keywords: vector, restore, arrays

.seealso: VecGetArrays(), VecRestoreArray()
@*/
int VecRestoreArrays(Vec *x,int n,Scalar ***a)
{
  int    i,ierr;
  Scalar **q = *a;
  PetscValidHeaderSpecific(*x,VEC_COOKIE);
  for(i=0;i<n;++i) {
    ierr = VecRestoreArray(x[i],&q[i]); CHKERRQ(ierr);
  }
  PetscFree(q);
  return 0;
}

/*@C
   VecRestoreArray - Restores a vector after VecGetArray() has been called.

   Input Parameters:
.  x - the vector
.  a - location of pointer to array obtained from VecGetArray()

   Fortran Note:
   The Fortran interface is slightly different from that given below.
   See the users manual and petsc/src/vec/examples for details.

.keywords: vector, restore, array

.seealso: VecGetArray(), VecRestoreArays()
@*/
int VecRestoreArray(Vec x,Scalar **a)
{
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (x->ops.restorearray) return (*x->ops.getarray)(x,a);
  else return 0;
}

/*@
   VecView - Views a vector object. 

   Input Parameters:
.  v - the vector
.  viewer - an optional visualization context

   Notes:
   The available visualization contexts include
$     VIEWER_STDOUT_SELF - standard output (default)
$     VIEWER_STDOUT_WORLD - synchronized standard
$       output where only the first processor opens
$       the file.  All other processors send their 
$       data to the first processor to print. 

   The user can open alternative vistualization contexts with
$    ViewerFileOpenASCII() - output vector to a specified file
$    ViewerFileOpenBinary() - output in binary to a
$         specified file; corresponding input uses VecLoad()
$    ViewerDrawOpenX() - output vector to an X window display
$    DrawLGCreate() - output vector as a line graph to an X window display
$    ViewerMatlabOpen() - output vector to Matlab viewer

.keywords: Vec, view, visualize, output, print, write, draw

.seealso: ViewerFileOpenASCII(), ViewerDrawOpenX(), DrawLGCreate(),
          ViewerMatlabOpen(), ViewerFileOpenBinary(), VecLoad()
@*/
int VecView(Vec v,Viewer viewer)
{
  PetscValidHeaderSpecific(v,VEC_COOKIE);
  if (!viewer) { 
    viewer = VIEWER_STDOUT_SELF;
  }
  return (*v->view)((PetscObject)v,viewer);
}

/*@
   VecGetSize - Returns the global number of elements of the vector.

   Input Parameter:
.  x - the vector

   Output Parameters:
.  size - the global length of the vector

.keywords: vector, get, size, global, dimension

.seealso: VecGetLocalSize()
@*/
int VecGetSize(Vec x,int *size)
{
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidIntPointer(size);
  return (*x->ops.getsize)(x,size);
}

/*@
   VecGetLocalSize - Returns the number of elements of the vector stored 
   in local memory. This routine may be implementation dependent, so use 
   with care.

   Input Parameter:
.  x - the vector

   Output Parameter:
.  size - the length of the local piece of the vector

.keywords: vector, get, dimension, size, local

.seealso: VecGetSize()
@*/
int VecGetLocalSize(Vec x,int *size)
{
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidIntPointer(size);
  return (*x->ops.getlocalsize)(x,size);
}

/*@
   VecGetOwnershipRange - Returns the range of indices owned by 
   this processor, assuming that the vectors are laid out with the
   first n1 elements on the first processor, next n2 elements on the
   second, etc.  For certain parallel layouts this range may not be 
   well-defined. 

   Input Parameter:
.  x - the vector

   Output Parameters:
.  low - the first local element
.  high - one more than the last local element

  Note: The high argument is one more then the last element stored local.

.keywords: vector, get, range, ownership
@*/
int VecGetOwnershipRange(Vec x,int *low,int *high)
{
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidIntPointer(low);
  PetscValidIntPointer(high);
  return (*x->ops.getownershiprange)(x,low,high);
}

/* Default routines for obtaining and releasing; */
/* may be used by any implementation */
int Veiobtain_vectors(Vec w,int m,Vec **V )
{
  Vec *v;
  int  i;
  *V = v = (Vec *) PetscMalloc( m * sizeof(Vec *) );
  for (i=0; i<m; i++) VecDuplicate(w,v+i);
  return 0;
}

int Veirelease_vectors( Vec *v, int m )
{
  int i;
  for (i=0; i<m; i++) VecDestroy(v[i]);
  PetscFree( v );
  return 0;
}


