
/* This file contains information for the use of PETSc Fortran interface stubs */

       int   PetscDoubleAddressToFortran(void*);
       int   PetscIntAddressToFortran(void*);
extern void *PetscNull_Fortran;

#ifdef HAVE_64BITS
extern void *MPIR_ToPointer(int);
extern int MPIR_FromPointer(void*);
extern void MPIR_RmPointer(int);
#else
#define MPIR_ToPointer(a) (a)
#define MPIR_FromPointer(a) (int)(a)
#define MPIR_RmPointer(a)
#endif
