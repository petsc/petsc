/* aodebug.c */
/* Fortran interface file */

/*
 * This file was generated automatically by bfort from the C source
 * file.  
 */

#ifdef HAVE_64BITS
#if defined(__cplusplus)
extern "C" { 
#endif 
extern void *PetscToPointer(int);
extern int PetscFromPointer(void *);
extern void PetscRmPointer(int);
#if defined(__cplusplus)
} 
#endif 

#else

#define PetscToPointer(a) (a)
#define PetscFromPointer(a) (int)(a)
#define PetscRmPointer(a)
#endif

#ifdef HAVE_FORTRAN_CAPS
#define aocreatedebug_ AOCREATEDEBUG
#elif !defined(HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define aocreatedebug_ aocreatedebug
#endif
#ifdef HAVE_FORTRAN_CAPS
#define aocreatedebugis_ AOCREATEDEBUGIS
#elif !defined(HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define aocreatedebugis_ aocreatedebugis
#endif


/* Definitions of Fortran Wrapper routines */
#if defined(__cplusplus)
extern "C" {
#endif
void aocreatedebug_(MPI_Comm comm,int *napp,int *myapp,int *mypetsc,AO *aoout, int *__ierr ){
*__ierr = AOCreateDebug(
	(MPI_Comm)PetscToPointer( *(int*)(comm) ),*napp,myapp,mypetsc,aoout);
}
void aocreatedebugis_(MPI_Comm comm,IS isapp,IS ispetsc,AO *aoout, int *__ierr ){
*__ierr = AOCreateDebugIS(
	(MPI_Comm)PetscToPointer( *(int*)(comm) ),
	(IS)PetscToPointer( *(int*)(isapp) ),
	(IS)PetscToPointer( *(int*)(ispetsc) ),aoout);
}
#if defined(__cplusplus)
}
#endif
