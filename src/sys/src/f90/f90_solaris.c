/*$Id: f90_solaris.c,v 1.2 2000/07/21 01:10:05 balay Exp balay $*/

#include "src/fortran/f90/zf90.h"
#if defined(PETSC_HAVE_SOLARISF90)

/* --------------------------------------------------------*/
/*
    PetscF90Create1dArrayScalar - Given a C pointer to a one dimensional
  array and its length; this fills in the appropriate Fortran 90
  pointer data structure.

  Input Parameters:
.   array - regular C pointer (address)
.   len - length of array (in items)

  Output Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Create1dArrayScalar(Scalar *array,int len,array1d *ptr)
{
  ptr->addr      = (void *)array;
  ptr->extent[0] = len;
  ptr->mult[0]   = sizeof(Scalar);
  ptr->lower[0]  = 1;
  ptr->addr_d    = (void*)((long)array - (ptr->lower[0]*ptr->mult[0]));

  return 0;
}

/*
    PetscF90Get1dArrayScalar - Gets the address for the data 
       stored in a Fortran pointer array.

  Input Parameters:
.   ptr - Fortran 90 pointer

  Output Parameters:
.   array - regular C pointer (address)

*/ 
int PetscF90Get1dArrayScalar(array1d *ptr,Scalar **array)
{
  *array = (Scalar*)ptr->addr;
  return 0;
}

/*
    PetscF90Destroy1dArrayScalar - Deletes a Fortran pointer.

  Input Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Destroy1dArrayScalar(array1d *ptr)
{
  ptr->addr = (void *)0;
  return 0;
}
/* --------------------------------------------------------*/
/*
    PetscF90Create2dArrayScalar - Given a C pointer to a two dimensional
  array and its length; this fills in the appropriate Fortran 90
  pointer data structure.

  Input Parameters:
.   array - regular C pointer (address)
.   m - number of rows in array
.   n - number of columns in array

  Output Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Create2dArrayScalar(Scalar *array,int m,int n,array2d *ptr)
{
  ptr->addr      = (void *)array;
  ptr->extent[1] = m;
  ptr->mult[1]   = sizeof(Scalar);
  ptr->lower[1]  = 1;
  ptr->extent[0] = n;
  ptr->mult[0]   = m*sizeof(Scalar);
  ptr->lower[0]  = 1;
  ptr->addr_d    = (void*)((long)array -(ptr->lower[0]*ptr->mult[0]+ptr->lower[1]*ptr->mult[1]));

  return 0;
}

/*
    PetscF90Get2dArrayScalar - Gets the address for the data 
       stored in a 2d Fortran pointer array.

  Input Parameters:
.   ptr - Fortran 90 pointer

  Output Parameters:
.   array - regular C pointer (address)

*/ 
int PetscF90Get2dArrayScalar(array2d *ptr,Scalar **array)
{
  *array = (Scalar*)ptr->addr;
  return 0;
}

/*
    PetscF90Destroy2dArrayScalar - Deletes a Fortran pointer.

  Input Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Destroy2dArrayScalar(array2d *ptr)
{
  ptr->addr  = (void *)0;
  return 0;
}
/* -----------------------------------------------------------------*/

/*
    PetscF90Create1dArrayInt - Given a C pointer to a one dimensional
  array and its length; this fills in the appropriate Fortran 90
  pointer data structure.

  Input Parameters:
.   array - regular C pointer (address)
.   len - length of array (in items)

  Output Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Create1dArrayInt(int *array,int len,array1d *ptr)
{
  ptr->addr      = (void *)array;
  ptr->extent[0] = len;
  ptr->mult[0]   = sizeof(int);
  ptr->lower[0]  = 1;
  ptr->addr_d    = (void*)((long)array -(ptr->lower[0]*ptr->mult[0]));

  return 0;
}

/*
    PetscF90Get1dArrayInt - Gets the address for the data 
       stored in a Fortran pointer array.

  Input Parameters:
.   ptr - Fortran 90 pointer

  Output Parameters:
.   array - regular C pointer (address)

*/ 
int PetscF90Get1dArrayInt(array1d *ptr,int **array)
{
  *array = (int*)ptr->addr;
  return 0;
}

/*
    PetscF90Destroy1dArrayInt - Deletes a Fortran pointer.

  Input Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Destroy1dArrayInt(array1d *ptr)
{
  ptr->addr  = (void *)0;
  return 0;
}
/* -----------------------------------------------------------------*/

/*
    PetscF90Create1dArrayPetscFortranAddr - Given a C pointer to a one dimensional
  array and its length; this fills in the appropriate Fortran 90
  pointer data structure.

  Input Parameters:
.   array - regular C pointer (address)
.   len - length of array (in items)

  Output Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Create1dArrayPetscFortranAddr(PetscFortranAddr *array,int len,array1d *ptr)
{
  ptr->addr          = (void *)array;
  ptr->extent[0]  = len;
  ptr->mult[0]   = sizeof(PetscFortranAddr);
  ptr->lower[0]  = 1;
  ptr->addr_d        = (void*)((long)array - (ptr->lower[0]*ptr->mult[0]));

  return 0;
}

/*
    PetscF90Get1dArrayPetscFortranAddr - Gets the address for the data 
       stored in a Fortran pointer array.

  Input Parameters:
.   ptr - Fortran 90 pointer

  Output Parameters:
.   array - regular C pointer (address)

*/ 
int PetscF90Get1dArrayPetscFortranAddr(array1d *ptr,PetscFortranAddr **array)
{
  *array = (PetscFortranAddr*)ptr->addr;
  return 0;
}

/*
    PetscF90Destroy1dArrayPetscFortranAddr - Deletes a Fortran pointer.

  Input Parameters:
.   ptr - Fortran 90 pointer
*/ 
int PetscF90Destroy1dArrayPetscFortranAddr(array1d *ptr)
{
  ptr->addr  = (void *)0;
  return 0;
}

#else
/*
     Dummy function so that compilers won't complain about 
  empty files.
*/
int F90_solaris_Dummy(int dummy)
{
  return 0;
}

#endif
