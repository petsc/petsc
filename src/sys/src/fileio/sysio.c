#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: sysio.c,v 1.53 1999/05/04 20:29:01 balay Exp bsmith $";
#endif

/* 
   This file contains simple binary read/write routines.
 */

#include "petsc.h"     /*I          "petsc.h"    I*/
#include "sys.h"
#include <fcntl.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined (PARCH_win32)
#include <io.h>
#endif
#include "bitarray.h"


#if !defined(PETSC_WORDS_BIGENDIAN)
#undef __FUNC__  
#define __FUNC__ "PetscByteSwapInt"
/*
  PetscByteSwapInt - Swap bytes in an integer
*/
int PetscByteSwapInt(int *buff,int n)
{
  int  i,j,tmp =0;
  int  *tptr = &tmp;                /* Need to access tmp indirectly to get */
  char *ptr1,*ptr2 = (char *) &tmp; /* arround the bug in DEC-ALPHA compilers*/
                                   
  PetscFunctionBegin;

  for ( j=0; j<n; j++ ) {
    ptr1 = (char *) (buff + j);
    for (i=0; i<sizeof(int); i++) {
      ptr2[i] = ptr1[sizeof(int)-1-i];
    }
    buff[j] = *tptr;
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "PetscByteSwapShort"
/*
  PetscByteSwapShort - Swap bytes in a short
*/
int PetscByteSwapShort(short *buff,int n)
{
  int   i,j;
  short tmp;
  short *tptr = &tmp;           /* take care pf bug in DEC-ALPHA g++ */
  char  *ptr1,*ptr2 = (char *) &tmp;

  PetscFunctionBegin;
  for ( j=0; j<n; j++ ) {
    ptr1 = (char *) (buff + j);
    for (i=0; i<sizeof(short); i++) {
      ptr2[i] = ptr1[sizeof(int)-1-i];
    }
    buff[j] = *tptr;
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "PetscByteSwapScalar"
/*
  PetscByteSwapScalar - Swap bytes in a double
  Complex is dealt with as if array of double twice as long.
*/
int PetscByteSwapScalar(Scalar *buff,int n)
{
  int    i,j;
  double tmp,*buff1 = (double *) buff;
  double *tptr = &tmp;          /* take care pf bug in DEC-ALPHA g++ */
  char   *ptr1,*ptr2 = (char *) &tmp;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  n *= 2;
#endif
  for ( j=0; j<n; j++ ) {
    ptr1 = (char *) (buff1 + j);
    for (i=0; i<sizeof(double); i++) {
      ptr2[i] = ptr1[sizeof(double)-1-i];
    }
    buff1[j] = *tptr;
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "PetscByteSwapDouble"
/*
  PetscByteSwapDouble - Swap bytes in a double
*/
int PetscByteSwapDouble(double *buff,int n)
{
  int    i,j;
  double tmp,*buff1 = (double *) buff;
  double *tptr = &tmp;          /* take care pf bug in DEC-ALPHA g++ */
  char   *ptr1,*ptr2 = (char *) &tmp;

  PetscFunctionBegin;
  for ( j=0; j<n; j++ ) {
    ptr1 = (char *) (buff1 + j);
    for (i=0; i<sizeof(double); i++) {
      ptr2[i] = ptr1[sizeof(double)-1-i];
    }
    buff1[j] = *tptr;
  }
  PetscFunctionReturn(0);
}
#endif
/* --------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "PetscBinaryRead"
/*@C
   PetscBinaryRead - Reads from a binary file.

   Not Collective

   Input Parameters:
+  fd - the file
.  n  - the number of items to read 
-  type - the type of items to read (PETSC_INT or PETSC_SCALAR)

   Output Parameters:
.  p - the buffer

   Options Database:
.   -binary_longints - indicates the file was generated on a Cray vector 
         machine (not the T3E/D) and the ints are stored as 64 bit 
         quantities, otherwise they are stored as 32 bit

   Level: developer

   Notes: 
   PetscBinaryRead() uses byte swapping to work on all machines.
   Integers are stored on the file as 32 long, regardless of whether
   they are stored in the machine as 32 or 64, this means the same
   binary file may be read on any machine.

   Note that Cray C90 and similar machines cannot generate files with 
   32 bit integers; use the flag -binary_longints to read files from the 
   C90 on non-C90 machines. Cray T3E/T3D are the same as other Unix
   machines, not the same as the C90.

.keywords: binary, input, read

.seealso: PetscBinaryWrite(), PetscBinaryOpen(), PetscBinaryClose()
@*/
int PetscBinaryRead(int fd,void *p,int n,PetscDataType type)
{
  int        maxblock = 65536, wsize, err, m = n, ierr,flag;
  static int longintfile = -1;
  char       *pp = (char *) p;
#if (PETSC_SIZEOF_SHORT != 8)
  void       *ptmp = p; 
#endif

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  if (longintfile == -1) {
    ierr = OptionsHasName(PETSC_NULL,"-binary_longints",&longintfile);CHKERRQ(ierr);
    ierr = OptionsHasName(PETSC_NULL,"-help",&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = (*PetscHelpPrintf)(PETSC_COMM_SELF,"-binary_longints - for binary file generated\n\
   on a Cray vector machine (not T3E/T3D)\n");CHKERRQ(ierr);
    }
  }

#if (PETSC_SIZEOF_INT == 8 && PETSC_SIZEOF_SHORT == 4)
  if (type == PETSC_INT){
    if (longintfile) {
      m *= sizeof(int);
    } else {
      /* read them in as shorts, later stretch into ints */
      m   *= sizeof(short);
      pp   = (char *) PetscMalloc(m);CHKPTRQ(pp);
      ptmp = (void*) pp;
    }
  }
#elif (PETSC_SIZEOF_INT == 8 && PETSC_SIZEOF_SHORT == 8)
  if (type == PETSC_INT){
    if (longintfile) {
      m *= sizeof(int);
    } else {
      SETERRQ(1,1,"Can only process data file generated on Cray vector machine;\n\
      if this data WAS then run program with -binary_longints option");
    }
  }
#else
  if (type == PETSC_INT) {
    if (longintfile) {
       /* read in twice as many ints and later discard every other one */
       m    *= 2*sizeof(int);
       pp   =  (char *) PetscMalloc(m);CHKPTRQ(pp);
       ptmp =  (void*) pp;
    } else {
       m *= sizeof(int);
    }
  }
#endif
  else if (type == PETSC_SCALAR)  m *= sizeof(Scalar);
  else if (type == PETSC_DOUBLE)  m *= sizeof(double);
  else if (type == PETSC_SHORT)   m *= sizeof(short);
  else if (type == PETSC_CHAR)    m *= sizeof(char);
  else if (type == PETSC_LOGICAL) m = BTLength(m)*sizeof(char);
  else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown type");
  
  while (m) {
    wsize = (m < maxblock) ? m : maxblock;
    err = read( fd, pp, wsize );
    if (err < 0 && errno == EINTR) continue;
    if (err == 0 && wsize > 0) PetscFunctionReturn(1);
    if (err < 0) SETERRQ(PETSC_ERR_FILE_READ,0,"Error reading from file");
    m  -= err;
    pp += err;
  }
#if !defined(PETSC_WORDS_BIGENDIAN)
  if      (type == PETSC_INT)    PetscByteSwapInt((int*)ptmp,n);
  else if (type == PETSC_SCALAR) PetscByteSwapScalar((Scalar*)ptmp,n);
  else if (type == PETSC_DOUBLE) PetscByteSwapDouble((double*)ptmp,n);
  else if (type == PETSC_SHORT)  PetscByteSwapShort((short*)ptmp,n);
#endif

#if (PETSC_SIZEOF_INT == 8 && PETSC_SIZEOF_SHORT == 4)
  if (type == PETSC_INT){
    if (!longintfile) {
      int   *p_int = (int *) p,i;
      short *p_short = (short *)ptmp;
      for ( i=0; i<n; i++ ) {
        p_int[i] = (int) p_short[i];
      }
      PetscFree(ptmp);
    }
  }
#elif (PETSC_SIZEOF_INT == 8 && PETSC_SIZEOF_SHORT == 8)
#else
  if (type == PETSC_INT)
    if (longintfile) {
    /* 
       take the longs (treated as pair of ints) and convert them to ints
    */
    int   *p_int  = (int *) p,i;
    int   *p_intl = (int *)ptmp;
    for ( i=0; i<n; i++ ) {
      p_int[i] = (int) p_intl[2*i+1];
    }
    PetscFree(ptmp);
  }
#endif

  PetscFunctionReturn(0);
}
/* --------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "PetscBinaryWrite"
/*@C
   PetscBinaryWrite - Writes to a binary file.

   Not Collective

   Input Parameters:
+  fd     - the file
.  p      - the buffer
.  n      - the number of items to write
.  type   - the type of items to read (PETSC_INT or PETSC_SCALAR)
-  istemp - 0 if buffer data should be preserved, 1 otherwise.

   Level: advanced

   Notes: 
   PetscBinaryWrite() uses byte swapping to work on all machines.
   Integers are stored on the file as 32 long, regardless of whether
   they are stored in the machine as 32 or 64, this means the same
   binary file may be read on any machine.

   The Buffer 'p' should be read-write buffer, and not static data.
   This way, byte-swapping is done in-place, and then the buffer is
   written to the file.
   
   This routine restores the original contents of the buffer, after 
   it is written to the file. This is done by byte-swapping in-place 
   the second time. If the flag 'istemp' is set to 1, the second
   byte-swapping operation is not done, thus saving some computation,
   but the buffer corrupted is corrupted.

.keywords: binary, output, write

.seealso: PetscBinaryRead(), PetscBinaryOpen(), PetscBinaryClose()
@*/
int PetscBinaryWrite(int fd,void *p,int n,PetscDataType type,int istemp)
{
  int  err, maxblock, wsize,m = n;
  char *pp = (char *) p;
#if !defined(PETSC_WORDS_BIGENDIAN) || (PETSC_SIZEOF_INT == 8)
  void *ptmp = p; 
#endif

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  maxblock = 65536;

#if !defined(PETSC_WORDS_BIGENDIAN)
  if      (type == PETSC_INT)    PetscByteSwapInt((int*)ptmp,n);
  else if (type == PETSC_SCALAR) PetscByteSwapScalar((Scalar*)ptmp,n);
  else if (type == PETSC_DOUBLE) PetscByteSwapDouble((double*)ptmp,n);
  else if (type == PETSC_SHORT)  PetscByteSwapShort((short*)ptmp,n);
#endif

#if (PETSC_SIZEOF_INT == 8)
  if (type == PETSC_INT){
    /* 
      integers on the Cray T3d/e are 64 bits so we copy the big
      integers into a short array and write those out.
    */
    int   *p_int = (int *) p,i;
    short *p_short;
    m       *= sizeof(short);
    pp      = (char *) PetscMalloc(m);CHKPTRQ(pp);
    ptmp    = (void*) pp;
    p_short = (short *) pp;

    for ( i=0; i<n; i++ ) {
      p_short[i] = (short) p_int[i];
    }
  }
#else
  if (type == PETSC_INT)          m *= sizeof(int);
#endif
  else if (type == PETSC_SCALAR)  m *= sizeof(Scalar);
  else if (type == PETSC_DOUBLE)  m *= sizeof(double);
  else if (type == PETSC_SHORT)   m *= sizeof(short);
  else if (type == PETSC_CHAR)    m *= sizeof(char);
  else if (type == PETSC_LOGICAL) m = BTLength(m)*sizeof(char);
  else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown type");

  while (m) {
    wsize = (m < maxblock) ? m : maxblock;
    err = write( fd, pp, wsize );
    if (err < 0 && errno == EINTR) continue;
    if (err != wsize) SETERRQ(PETSC_ERR_FILE_WRITE,0,"Error writing to file.");
    m -= wsize;
    pp += wsize;
  }

#if !defined(PETSC_WORDS_BIGENDIAN)
  if (!istemp) {
    if      (type == PETSC_SCALAR) PetscByteSwapScalar((Scalar*)ptmp,n);
    else if (type == PETSC_SHORT)  PetscByteSwapShort((short*)ptmp,n);
    else if (type == PETSC_INT)    PetscByteSwapInt((int*)ptmp,n);
  }
#endif

#if (PETSC_SIZEOF_INT == 8)
  if (type == PETSC_INT){
    PetscFree(ptmp);
  }
#endif

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscBinaryOpen" 
/*@C
   PetscBinaryOpen - Opens a PETSc binary file.

   Not Collective

   Input Parameters:
+  name - filename
-  type - type of binary file, on of BINARY_RDONLY, BINARY_WRONLY, BINARY_CREATE

   Output Parameter:
.  fd - the file

   Level: advanced

.keywords: binary, output, write

.seealso: PetscBinaryRead(), PetscBinaryWrite()
@*/
int PetscBinaryOpen(const char name[],int type,int *fd)
{
  PetscFunctionBegin;
#if defined(PARCH_win32_gnu) || defined(PARCH_win32) 
  if (type == BINARY_CREATE) {
    if ((*fd = open(name,O_WRONLY|O_CREAT|O_TRUNC|O_BINARY,0666 )) == -1) {
      SETERRQ1(PETSC_ERR_FILE_OPEN,0,"Cannot create file for writing: %s",name);
    }
  } else if (type == BINARY_RDONLY) {
    if ((*fd = open(name,O_RDONLY|O_BINARY,0)) == -1) {
      SETERRQ1(PETSC_ERR_FILE_OPEN,0,"Cannot open file for reading: %s",name);
    }
  } else if (type == BINARY_WRONLY) {
    if ((*fd = open(name,O_WRONLY|O_BINARY,0)) == -1) {
      SETERRQ1(PETSC_ERR_FILE_OPEN,0,"Cannot open file for writing: %s",name);
    }
#else
  if (type == BINARY_CREATE) {
    if ((*fd = creat(name,0666)) == -1) {
      SETERRQ1(PETSC_ERR_FILE_OPEN,0,"Cannot create file for writing: %s",name);
    }
  } else if (type == BINARY_RDONLY) {
    if ((*fd = open(name,O_RDONLY,0)) == -1) {
      SETERRQ1(PETSC_ERR_FILE_OPEN,0,"Cannot open file for reading: %s",name);
    }
  }
  else if (type == BINARY_WRONLY) {
    if ((*fd = open(name,O_WRONLY,0)) == -1) {
      SETERRQ1(PETSC_ERR_FILE_OPEN,0,"Cannot open file for writing: %s",name);
    }
#endif
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown file type");
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscBinaryClose" 
/*@C
   PetscBinaryClose - Closes a PETSc binary file.

   Not Collective

   Output Parameter:
.  fd - the file

   Level: advanced

.keywords: binary, output, write

.seealso: PetscBinaryRead(), PetscBinaryWrite(), PetscBinaryOpen()
@*/
int PetscBinaryClose(int fd)
{
  PetscFunctionBegin;
  close(fd);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "PetscBinarySeek" 
/*@C
   PetscBinarySeek - Moves the file pointer on a PETSc binary file.

   Not Collective

   Output Parameter:
+  fd - the file
.  whence - if BINARY_SEEK_SET then size is an absolute location in the file
            if BINARY_SEEK_CUR then size is offset from current location
            if BINARY_SEEK_END then size is offset from end of file
-  size - number of bytes to move. Use PETSC_INT_SIZE, BINARY_SCALAR_SIZE,
            etc. in your calculation rather than sizeof() to compute byte lengths.

   Level: developer

   Notes: 
   Integers are stored on the file as 32 long, regardless of whether
   they are stored in the machine as 32 or 64, this means the same
   binary file may be read on any machine. Hence you CANNOT use sizeof()
   to determine the offset or location.

.keywords: binary, output, write

.seealso: PetscBinaryRead(), PetscBinaryWrite(), PetscBinaryOpen()
@*/
int PetscBinarySeek(int fd,int size,PetscBinarySeekType whence)
{
  int iwhence=0;

  PetscFunctionBegin;
  if (whence == BINARY_SEEK_SET) {
    iwhence = SEEK_SET;
  } else if (whence == BINARY_SEEK_CUR) {
    iwhence = SEEK_CUR;
  } else if (whence == BINARY_SEEK_END) {
    iwhence = SEEK_END;
  } else {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Unknown seek location");
  }
#if defined(PARCH_win32)
  _lseek(fd,(long)size,iwhence);
#else
  lseek(fd,(off_t)size,iwhence);
#endif

  PetscFunctionReturn(0);
}


