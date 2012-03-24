#include <petsc-private/fortranimpl.h> 

void *PETSCNULLPOINTERADDRESS = PETSC_NULL;

/*MC
   PetscFortranAddr - a variable type in Fortran that can hold a
     regular C pointer.

   Notes: Used, for example, as the file argument in PetscFOpen()

   Level: beginner

.seealso:  PetscOffset, PetscInt
M*/
/*MC
   PetscOffset - a variable type in Fortran used with VecGetArray()
     and ISGetIndices()

   Level: beginner

.seealso:  PetscFortranAddr, PetscInt
M*/

/*
    This is code for translating PETSc memory addresses to integer offsets 
    for Fortran.
*/
char   *PETSC_NULL_CHARACTER_Fortran = 0;
void   *PETSC_NULL_INTEGER_Fortran   = 0;
void   *PETSC_NULL_OBJECT_Fortran    = 0;
void   *PETSC_NULL_Fortran           = 0;
void   *PETSC_NULL_SCALAR_Fortran    = 0;
void   *PETSC_NULL_DOUBLE_Fortran    = 0;
void   *PETSC_NULL_REAL_Fortran      = 0;
void   *PETSC_NULL_BOOL_Fortran     = 0;
EXTERN_C_BEGIN
void   (*PETSC_NULL_FUNCTION_Fortran)(void) = 0;
EXTERN_C_END
size_t PetscIntAddressToFortran(PetscInt *base,PetscInt *addr)
{
  size_t tmp1 = (size_t) base,tmp2 = 0;
  size_t tmp3 = (size_t) addr;
  size_t itmp2;

#if !defined(PETSC_HAVE_CRAY90_POINTER)
  if (tmp3 > tmp1) {
    tmp2  = (tmp3 - tmp1)/sizeof(PetscInt);
    itmp2 = (size_t) tmp2;
  } else {
    tmp2  = (tmp1 - tmp3)/sizeof(PetscInt);
    itmp2 = -((size_t) tmp2);
  }
#else
  if (tmp3 > tmp1) {
    tmp2  = (tmp3 - tmp1);
    itmp2 = (size_t) tmp2;
  } else {
    tmp2  = (tmp1 - tmp3);
    itmp2 = -((size_t) tmp2);
  }
#endif

  if (base + itmp2 != addr) {
    (*PetscErrorPrintf)("PetscIntAddressToFortran:C and Fortran arrays are\n");
    (*PetscErrorPrintf)("not commonly aligned or are too far apart to be indexed \n");
    (*PetscErrorPrintf)("by an integer. Locations: C %uld Fortran %uld\n",tmp1,tmp3);
    MPI_Abort(PETSC_COMM_WORLD,1);
  }
  return itmp2;
}

PetscInt *PetscIntAddressFromFortran(PetscInt *base,size_t addr)
{
  return base + addr;
}

/*
       obj - PETSc object on which request is made
       base - Fortran array address
       addr - C array address
       res  - will contain offset from C to Fortran
       shift - number of bytes that prevent base and addr from being commonly aligned
       N - size of the array

       align indicates alignment relative to PetscScalar, 1 means aligned on PetscScalar, 2 means aligned on 2 PetscScalar etc
*/
PetscErrorCode PetscScalarAddressToFortran(PetscObject obj,PetscInt align,PetscScalar *base,PetscScalar *addr,PetscInt N,size_t *res)
{
  size_t   tmp1 = (size_t) base,tmp2 = tmp1/sizeof(PetscScalar);
  size_t   tmp3 = (size_t) addr;
  size_t   itmp2;
  PetscInt shift;

#if !defined(PETSC_HAVE_CRAY90_POINTER)
  if (tmp3 > tmp1) {  /* C is bigger than Fortran */
    tmp2  = (tmp3 - tmp1)/sizeof(PetscScalar);
    itmp2 = (size_t) tmp2;
    shift = (align*sizeof(PetscScalar) - (PetscInt)((tmp3 - tmp1) % (align*sizeof(PetscScalar)))) % (align*sizeof(PetscScalar));
  } else {  
    tmp2  = (tmp1 - tmp3)/sizeof(PetscScalar);
    itmp2 = -((size_t) tmp2);
    shift = (PetscInt)((tmp1 - tmp3) % (align*sizeof(PetscScalar)));
  }
#else
  if (tmp3 > tmp1) {  /* C is bigger than Fortran */
    tmp2  = (tmp3 - tmp1);
    itmp2 = (size_t) tmp2;
  } else {  
    tmp2  = (tmp1 - tmp3);
    itmp2 = -((size_t) tmp2);
  }
  shift = 0;
#endif

  if (shift) { 
    /* 
        Fortran and C not PetscScalar aligned,recover by copying values into
        memory that is aligned with the Fortran
    */
    PetscErrorCode ierr;
    PetscScalar    *work;
    PetscContainer container;

    ierr = PetscMalloc((N+align)*sizeof(PetscScalar),&work);CHKERRQ(ierr); 

    /* recompute shift for newly allocated space */
    tmp3 = (size_t) work;
    if (tmp3 > tmp1) {  /* C is bigger than Fortran */
      shift = (align*sizeof(PetscScalar) - (PetscInt)((tmp3 - tmp1) % (align*sizeof(PetscScalar)))) % (align*sizeof(PetscScalar));
    } else {  
      shift = (PetscInt)((tmp1 - tmp3) % (align*sizeof(PetscScalar)));
    }

    /* shift work by that number of bytes */
    work = (PetscScalar*)(((char*)work) + shift);
    ierr = PetscMemcpy(work,addr,N*sizeof(PetscScalar));CHKERRQ(ierr);

    /* store in the first location in addr how much you shift it */
    ((PetscInt*)addr)[0] = shift;
 
    ierr = PetscContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,addr);CHKERRQ(ierr);
    ierr = PetscObjectCompose(obj,"GetArrayPtr",(PetscObject)container);CHKERRQ(ierr);

    tmp3 = (size_t) work;
    if (tmp3 > tmp1) {  /* C is bigger than Fortran */
      tmp2  = (tmp3 - tmp1)/sizeof(PetscScalar);
      itmp2 = (size_t) tmp2;
      shift = (align*sizeof(PetscScalar) - (PetscInt)((tmp3 - tmp1) % (align*sizeof(PetscScalar)))) % (align*sizeof(PetscScalar));
    } else {  
      tmp2  = (tmp1 - tmp3)/sizeof(PetscScalar);
      itmp2 = -((size_t) tmp2);
      shift = (PetscInt)((tmp1 - tmp3) % (align*sizeof(PetscScalar)));
    }
    if (shift) {
      (*PetscErrorPrintf)("PetscScalarAddressToFortran:C and Fortran arrays are\n");
      (*PetscErrorPrintf)("not commonly aligned.\n");
      /* double/int doesn't work with ADIC */
      (*PetscErrorPrintf)("Locations/sizeof(PetscScalar): C %f Fortran %f\n",
                         ((PetscReal)tmp3)/(PetscReal)sizeof(PetscScalar),((PetscReal)tmp1)/(PetscReal)sizeof(PetscScalar));
      MPI_Abort(PETSC_COMM_WORLD,1);
    }
    ierr = PetscInfo(obj,"Efficiency warning, copying array in XXXGetArray() due\n\
    to alignment differences between C and Fortran\n");CHKERRQ(ierr);
  }
  *res = itmp2;
  return 0;
}

/*
    obj - the PETSc object where the scalar pointer came from
    base - the Fortran array address
    addr - the Fortran offset from base
    N    - the amount of data

    lx   - the array space that is to be passed to XXXXRestoreArray()
*/     
PetscErrorCode PetscScalarAddressFromFortran(PetscObject obj,PetscScalar *base,size_t addr,PetscInt N,PetscScalar **lx)
{
  PetscErrorCode ierr;
  PetscInt       shift;
  PetscContainer container;
  PetscScalar    *tlx;

  ierr = PetscObjectQuery(obj,"GetArrayPtr",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) {
    ierr  = PetscContainerGetPointer(container,(void**)lx);CHKERRQ(ierr);
    tlx   = base + addr;

    shift = *(PetscInt*)*lx;
    ierr  = PetscMemcpy(*lx,tlx,N*sizeof(PetscScalar));CHKERRQ(ierr);
    tlx   = (PetscScalar*)(((char *)tlx) - shift);
    ierr = PetscFree(tlx);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
    ierr = PetscObjectCompose(obj,"GetArrayPtr",0);CHKERRQ(ierr);
  } else {
    *lx = base + addr;
  }
  return 0;
}

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscisinfornanscalar_          PETSCISINFORNANSCALAR
#define petscisinfornanreal_            PETSCISINFORNANREAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscisinfornanscalar_          petscisinfornanscalar
#define petscisinfornanreal_            petscisinfornanreal
#endif

EXTERN_C_BEGIN
PetscBool  PETSC_STDCALL petscisinfornanscalar_(PetscScalar *v)
{
  return (PetscBool) PetscIsInfOrNanScalar(*v);
}

PetscBool  PETSC_STDCALL petscisinfornanreal_(PetscReal *v)
{
  return (PetscBool) PetscIsInfOrNanReal(*v);
}
EXTERN_C_END



