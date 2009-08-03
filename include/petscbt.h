
#if !defined(__PETSCBT_H)
#define __PETSCBT_H
PETSC_EXTERN_CXX_BEGIN

/*S
     PetscBT - PETSc bitarrays

     Level: advanced

     PetscBTCreate(m,bt)        - creates a bit array with enough room to hold m values
     PetscBTDestroy(bt)         - destroys the bit array
     PetscBTMemzero(m,bt)       - zeros the entire bit array (sets all values to false)
     PetscBTSet(bt,index)       - sets a particular entry as true
     PetscBTClear(bt,index)     - sets a particular entry as false
     PetscBTLookup(bt,index)    - returns the value 
     PetscBTLookupSet(bt,index) - returns the value and then sets it true
     PetscBTLength(m)           - returns number of bytes in array with m bits
     PetscBTView(m,bt,viewer)   - prints all the entries in a bit array

    The are all implemented as macros with the trivial data structure for efficiency.

    These are not thread safe since they use a few global variables.

    We do not currently check error flags on PetscBTSet(), PetscBTClear(), PetscBTLookup(),
    PetcBTLookupSet(), PetscBTLength() cause error checking would cost hundreds more cycles then
    the operation.

S*/
typedef char* PetscBT;

extern PETSC_DLLEXPORT char      _BT_mask;
extern PETSC_DLLEXPORT char      _BT_c;
extern PETSC_DLLEXPORT PetscInt  _BT_idx;

#define PetscBTLength(m)        ((m)/PETSC_BITS_PER_BYTE+1)
#define PetscBTMemzero(m,array) PetscMemzero(array,sizeof(char)*((m)/PETSC_BITS_PER_BYTE+1))
#define PetscBTDestroy(array)   PetscFree(array)

#define PetscBTView(m,bt,viewer) 0; {\
  PetscInt __i; PetscErrorCode _8_ierr; \
  PetscViewer __viewer = viewer; \
  if (!__viewer) __viewer = PETSC_VIEWER_STDOUT_SELF;\
  for (__i=0; __i<m; __i++) { \
    _8_ierr = PetscViewerASCIISynchronizedPrintf(__viewer,"%D %d\n",__i,PetscBTLookup(bt,__i));CHKERRQ(_8_ierr);\
  }  _8_ierr = PetscViewerFlush(__viewer);CHKERRQ(_8_ierr);}

#define PetscBTCreate(m,array)  \
  (PetscMalloc(((m)/PETSC_BITS_PER_BYTE+1)*sizeof(char),&(array)) || PetscBTMemzero(m,array))

#define PetscBTLookupSet(array,index) \
  (_BT_idx        = (index)/PETSC_BITS_PER_BYTE, \
   _BT_c          = array[_BT_idx], \
   _BT_mask       = (char)1 << ((index)%PETSC_BITS_PER_BYTE), \
   array[_BT_idx] = _BT_c | _BT_mask, \
   _BT_c & _BT_mask)

#define PetscBTSet(array,index)  \
  (_BT_idx        = (index)/PETSC_BITS_PER_BYTE, \
   _BT_c          = array[_BT_idx], \
   _BT_mask       = (char)1 << ((index)%PETSC_BITS_PER_BYTE), \
   array[_BT_idx] = _BT_c | _BT_mask,0)

#define PetscBTClear(array,index) \
  (_BT_idx        = (index)/PETSC_BITS_PER_BYTE, \
   _BT_c          = array[_BT_idx], \
   _BT_mask       = (char)1 << ((index)%PETSC_BITS_PER_BYTE), \
   array[_BT_idx] = _BT_c & (~_BT_mask),0)

#define PetscBTLookup(array,index) \
  (_BT_idx        = (index)/PETSC_BITS_PER_BYTE, \
   _BT_c          = array[_BT_idx], \
   _BT_mask       = (char)1 << ((index)%PETSC_BITS_PER_BYTE), \
   (_BT_c & _BT_mask) != 0)

PETSC_EXTERN_CXX_END
#endif
