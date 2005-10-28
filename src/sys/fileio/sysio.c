#define PETSC_DLL

/* 
   This file contains simple binary read/write routines.
 */

#include "petsc.h"
#include "petscsys.h"     /*I          "petscsys.h"    I*/

#include <errno.h>
#include <fcntl.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined (PETSC_HAVE_IO_H)
#include <io.h>
#endif
#include "petscbt.h"

#if (PETSC_SIZEOF_INT == 8)
#define PetscInt32 short
#else
#define PetscInt32 int
#endif

#if !defined(PETSC_WORDS_BIGENDIAN)

#undef __FUNCT__  
#define __FUNCT__ "PetscByteSwapInt"
/*
  PetscByteSwapInt - Swap bytes in a 32 bit integer. NOT a PetscInt! Note that PETSc binary read and write
      always store and read only 32 bit integers! (See PetscBinaryRead(), PetscBinaryWrite()).

*/
PetscErrorCode PETSC_DLLEXPORT PetscByteSwapInt(PetscInt32 *buff,PetscInt n)
{
  PetscInt  i,j,tmp = 0;
  PetscInt  *tptr = &tmp;                /* Need to access tmp indirectly to get */
  char      *ptr1,*ptr2 = (char*)&tmp; /* arround the bug in DEC-ALPHA g++ */
                                   
  PetscFunctionBegin;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff + j);
    for (i=0; i<(int)sizeof(PetscInt32); i++) {
      ptr2[i] = ptr1[sizeof(PetscInt32)-1-i];
    }
    buff[j] = *tptr;
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PetscByteSwapShort"
/*
  PetscByteSwapShort - Swap bytes in a short
*/
PetscErrorCode PETSC_DLLEXPORT PetscByteSwapShort(short *buff,PetscInt n)
{
  PetscInt   i,j;
  short      tmp;
  short      *tptr = &tmp;           /* take care pf bug in DEC-ALPHA g++ */
  char       *ptr1,*ptr2 = (char*)&tmp;

  PetscFunctionBegin;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff + j);
    for (i=0; i<(int) sizeof(short); i++) {
      ptr2[i] = ptr1[sizeof(int)-1-i];
    }
    buff[j] = *tptr;
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PetscByteSwapScalar"
/*
  PetscByteSwapScalar - Swap bytes in a double
  Complex is dealt with as if array of double twice as long.
*/
PetscErrorCode PETSC_DLLEXPORT PetscByteSwapScalar(PetscScalar *buff,PetscInt n)
{
  PetscInt  i,j;
  PetscReal tmp,*buff1 = (PetscReal*)buff;
  PetscReal *tptr = &tmp;          /* take care pf bug in DEC-ALPHA g++ */
  char      *ptr1,*ptr2 = (char*)&tmp;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  n *= 2;
#endif
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff1 + j);
    for (i=0; i<(int) sizeof(PetscReal); i++) {
      ptr2[i] = ptr1[sizeof(PetscReal)-1-i];
    }
    buff1[j] = *tptr;
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PetscByteSwapDouble"
/*
  PetscByteSwapDouble - Swap bytes in a double
*/
PetscErrorCode PETSC_DLLEXPORT PetscByteSwapDouble(double *buff,PetscInt n)
{
  PetscInt i,j;
  double   tmp,*buff1 = (double*)buff;
  double   *tptr = &tmp;          /* take care pf bug in DEC-ALPHA g++ */
  char     *ptr1,*ptr2 = (char*)&tmp;

  PetscFunctionBegin;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff1 + j);
    for (i=0; i<(int) sizeof(double); i++) {
      ptr2[i] = ptr1[sizeof(double)-1-i];
    }
    buff1[j] = *tptr;
  }
  PetscFunctionReturn(0);
}
#endif
/* --------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PetscBinaryRead"
/*@
   PetscBinaryRead - Reads from a binary file.

   Not Collective

   Input Parameters:
+  fd - the file
.  n  - the number of items to read 
-  type - the type of items to read (PETSC_INT, PETSC_DOUBLE or PETSC_SCALAR)

   Output Parameters:
.  p - the buffer



   Level: developer

   Notes: 
   PetscBinaryRead() uses byte swapping to work on all machines; the files
   are written to file ALWAYS using big-endian ordering. On small-endian machines the numbers
   are converted to the small-endian format when they are read in from the file.
   Integers are stored on the file as 32 bits long, regardless of whether
   they are stored in the machine as 32 bits or 64 bits, this means the same
   binary file may be read on any machine.

   Concepts: files^reading binary
   Concepts: binary files^reading

.seealso: PetscBinaryWrite(), PetscBinaryOpen(), PetscBinaryClose()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscBinaryRead(int fd,void *p,PetscInt n,PetscDataType type)
{
#if (PETSC_SIZEOF_INT == 8) || defined(PETSC_USE_64BIT_INDICES) || !defined(PETSC_WORDS_BIGENDIAN)
  PetscErrorCode    ierr;
#endif
  int               wsize,err;
  size_t            m = (size_t) n,maxblock = 65536;
  char              *pp = (char*)p;
#if (PETSC_SIZEOF_INT == 8) || !defined(PETSC_WORDS_BIGENDIAN) || defined(PETSC_USE_64BIT_INDICES)
  void              *ptmp = p; 
#endif

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  if (type == PETSC_INT){
    m   *= sizeof(PetscInt32);
#if (PETSC_SIZEOF_INT == 8) || defined(PETSC_USE_64BIT_INDICES)
    /* read them in as 32 bit ints, later stretch into ints */
    ierr = PetscMalloc(m,&pp);CHKERRQ(ierr);
    ptmp = (void*)pp;
#endif
  } 
  else if (type == PETSC_SCALAR)  m *= sizeof(PetscScalar);
  else if (type == PETSC_DOUBLE)  m *= sizeof(double);
  else if (type == PETSC_SHORT)   m *= sizeof(short);
  else if (type == PETSC_CHAR)    m *= sizeof(char);
  else if (type == PETSC_ENUM)    m *= sizeof(PetscEnum);
  else if (type == PETSC_TRUTH)   m *= sizeof(PetscTruth);
  else if (type == PETSC_LOGICAL) m  = PetscBTLength(m)*sizeof(char);
  else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Unknown type");
  
  while (m) {
    wsize = (m < maxblock) ? m : maxblock;
    err = read(fd,pp,wsize);
    if (err < 0 && errno == EINTR) continue;
    if (!err && wsize > 0) SETERRQ(PETSC_ERR_FILE_READ,"Read past end of file");
    if (err < 0) SETERRQ(PETSC_ERR_FILE_READ,"Error reading from file");
    m  -= err;
    pp += err;
  }
#if !defined(PETSC_WORDS_BIGENDIAN)
  if      (type == PETSC_INT)    {ierr = PetscByteSwapInt((PetscInt32*)ptmp,n);CHKERRQ(ierr);}
  else if (type == PETSC_ENUM)   {ierr = PetscByteSwapInt((PetscInt32*)ptmp,n);CHKERRQ(ierr);}        
  else if (type == PETSC_TRUTH)  {ierr = PetscByteSwapInt((PetscInt32*)ptmp,n);CHKERRQ(ierr);}        
  else if (type == PETSC_SCALAR) {ierr = PetscByteSwapScalar((PetscScalar*)ptmp,n);CHKERRQ(ierr);}
  else if (type == PETSC_DOUBLE) {ierr = PetscByteSwapDouble((double*)ptmp,n);CHKERRQ(ierr);}
  else if (type == PETSC_SHORT)  {ierr = PetscByteSwapShort((short*)ptmp,n);CHKERRQ(ierr);}
#endif

#if (PETSC_SIZEOF_INT == 8) || defined(PETSC_USE_64BIT_INDICES)
  if (type == PETSC_INT) {
    PetscInt   *p_int = (PetscInt*)p,i;
    PetscInt32 *p_short = (PetscInt32 *)ptmp;
    for (i=0; i<n; i++) {
      p_int[i] = (PetscInt)p_short[i];
    }
    ierr = PetscFree(ptmp);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PetscBinaryWrite"
/*@
   PetscBinaryWrite - Writes to a binary file.

   Not Collective

   Input Parameters:
+  fd     - the file
.  p      - the buffer
.  n      - the number of items to write
.  type   - the type of items to read (PETSC_INT, PETSC_DOUBLE or PETSC_SCALAR)
-  istemp - PETSC_FALSE if buffer data should be preserved, PETSC_TRUE otherwise.

   Level: advanced

   Notes: 
   PetscBinaryWrite() uses byte swapping to work on all machines; the files
   are written using big-endian ordering to the file. On small-endian machines the numbers
   are converted to the big-endian format when they are written to disk.
   Integers are stored on the file as 32 bits long, regardless of whether
   they are stored in the machine as 32 bits or 64 bits, this means the same
   binary file may be read on any machine. It also means that 64 bit integers larger than
   roughly 2 billion are TRUNCATED/WRONG when written to the file.

   The Buffer p should be read-write buffer, and not static data.
   This way, byte-swapping is done in-place, and then the buffer is
   written to the file.
   
   This routine restores the original contents of the buffer, after 
   it is written to the file. This is done by byte-swapping in-place 
   the second time. If the flag istemp is set to PETSC_TRUE, the second
   byte-swapping operation is not done, thus saving some computation,
   but the buffer corrupted is corrupted.

   Concepts: files^writing binary
   Concepts: binary files^writing

.seealso: PetscBinaryRead(), PetscBinaryOpen(), PetscBinaryClose()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscBinaryWrite(int fd,void *p,PetscInt n,PetscDataType type,PetscTruth istemp)
{
  char           *pp = (char*)p;
  int            err,wsize;
  size_t         m = (size_t)n,maxblock=65536;
#if !defined(PETSC_WORDS_BIGENDIAN) || (PETSC_SIZEOF_INT == 8) ||  defined(PETSC_USE_64BIT_INDICES)
  PetscErrorCode ierr;
  void           *ptmp = p; 
#endif

  PetscFunctionBegin;
  if (n < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Trying to write a negative amount of data %D",n);
  if (!n) PetscFunctionReturn(0);

  if (type == PETSC_INT){
    m   *= sizeof(PetscInt32);
#if (PETSC_SIZEOF_INT == 8) || defined(PETSC_USE_64BIT_INDICES)
    PetscInt   *p_int = (PetscInt*)p,i;
    PetscInt32 *p_short;
    ierr    = PetscMalloc(m,&pp);CHKERRQ(ierr);
    ptmp    = (void*)pp;
    p_short = (PetscInt32*)pp;

    for (i=0; i<n; i++) {
      p_short[i] = (PetscInt32) p_int[i];
    }
    istemp = PETSC_TRUE;
#endif
  }
  else if (type == PETSC_SCALAR)  m *= sizeof(PetscScalar);
  else if (type == PETSC_DOUBLE)  m *= sizeof(double);
  else if (type == PETSC_SHORT)   m *= sizeof(short);
  else if (type == PETSC_CHAR)    m *= sizeof(char);
  else if (type == PETSC_ENUM)    m *= sizeof(PetscEnum);
  else if (type == PETSC_TRUTH)   m *= sizeof(PetscTruth);
  else if (type == PETSC_LOGICAL) m = PetscBTLength(m)*sizeof(char);
  else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Unknown type");

#if !defined(PETSC_WORDS_BIGENDIAN)
  if      (type == PETSC_INT)    {ierr = PetscByteSwapInt((PetscInt32*)ptmp,n);CHKERRQ(ierr);}
  else if (type == PETSC_ENUM)   {ierr = PetscByteSwapInt((PetscInt32*)ptmp,n);CHKERRQ(ierr);}          
  else if (type == PETSC_TRUTH)  {ierr = PetscByteSwapInt((PetscInt32*)ptmp,n);CHKERRQ(ierr);}          
  else if (type == PETSC_SCALAR) {ierr = PetscByteSwapScalar((PetscScalar*)ptmp,n);CHKERRQ(ierr);}
  else if (type == PETSC_DOUBLE) {ierr = PetscByteSwapDouble((double*)ptmp,n);CHKERRQ(ierr);}
  else if (type == PETSC_SHORT)  {ierr = PetscByteSwapShort((short*)ptmp,n);CHKERRQ(ierr);}
#endif

  while (m) {
    wsize = (m < maxblock) ? m : maxblock;
    err = write(fd,pp,wsize);
    if (err < 0 && errno == EINTR) continue;
    if (err != wsize) SETERRQ(PETSC_ERR_FILE_WRITE,"Error writing to file.");
    m -= wsize;
    pp += wsize;
  }

#if !defined(PETSC_WORDS_BIGENDIAN) && !(PETSC_SIZEOF_INT == 8) && !defined(PETSC_USE_64BIT_INDICES)
  if (!istemp) {
    if      (type == PETSC_SCALAR) {ierr = PetscByteSwapScalar((PetscScalar*)ptmp,n);CHKERRQ(ierr);}
    else if (type == PETSC_SHORT)  {ierr = PetscByteSwapShort((short*)ptmp,n);CHKERRQ(ierr);}
    else if (type == PETSC_INT)    {ierr = PetscByteSwapInt((PetscInt32*)ptmp,n);CHKERRQ(ierr);}
    else if (type == PETSC_ENUM)   {ierr = PetscByteSwapInt((PetscInt32*)ptmp,n);CHKERRQ(ierr);}
    else if (type == PETSC_TRUTH)  {ierr = PetscByteSwapInt((PetscInt32*)ptmp,n);CHKERRQ(ierr);}
  }
#endif

#if (PETSC_SIZEOF_INT == 8) || defined(PETSC_USE_64BIT_INDICES)
  if (type == PETSC_INT){
    ierr = PetscFree(ptmp);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscBinaryOpen" 
/*@C
   PetscBinaryOpen - Opens a PETSc binary file.

   Not Collective

   Input Parameters:
+  name - filename
-  type - type of binary file, one of FILE_MODE_READ, FILE_MODE_APPEND, FILE_MODE_WRITE

   Output Parameter:
.  fd - the file

   Level: advanced

  Concepts: files^opening binary
  Concepts: binary files^opening

   Notes: Files access with PetscBinaryRead() and PetscBinaryWrite() are ALWAYS written in
   big-endian format. This means the file can be accessed using PetscBinaryOpen() and
   PetscBinaryRead() and PetscBinaryWrite() on any machine.

.seealso: PetscBinaryRead(), PetscBinaryWrite(), PetscFileMode, PetscViewerFileSetMode()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscBinaryOpen(const char name[],PetscFileMode mode,int *fd)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_O_BINARY) 
  if (mode == FILE_MODE_WRITE) {
    if ((*fd = open(name,O_WRONLY|O_CREAT|O_TRUNC|O_BINARY,0666)) == -1) {
      SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot create file for writing: %s",name);
    }
  } else if (mode == FILE_MODE_READ) {
    if ((*fd = open(name,O_RDONLY|O_BINARY,0)) == -1) {
      SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot open file for reading: %s",name);
    }
  } else if (mode == FILE_MODE_APPEND) {
    if ((*fd = open(name,O_WRONLY|O_BINARY,0)) == -1) {
      SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot open file for writing: %s",name);
    }
#else
  if (mode == FILE_MODE_WRITE) {
    if ((*fd = creat(name,0666)) == -1) {
      SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot create file for writing: %s",name);
    }
  } else if (mode == FILE_MODE_READ) {
    if ((*fd = open(name,O_RDONLY,0)) == -1) {
      SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot open file for reading: %s",name);
    }
  }
  else if (mode == FILE_MODE_APPEND) {
    if ((*fd = open(name,O_WRONLY,0)) == -1) {
      SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot open file for writing: %s",name);
    }
#endif
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Unknown file mode");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscBinaryClose" 
/*@
   PetscBinaryClose - Closes a PETSc binary file.

   Not Collective

   Output Parameter:
.  fd - the file

   Level: advanced

.seealso: PetscBinaryRead(), PetscBinaryWrite(), PetscBinaryOpen()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscBinaryClose(int fd)
{
  PetscFunctionBegin;
  close(fd);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscBinarySeek" 
/*@
   PetscBinarySeek - Moves the file pointer on a PETSc binary file.

   Not Collective

   Input Parameters:
+  fd - the file
.  whence - if PETSC_BINARY_SEEK_SET then size is an absolute location in the file
            if PETSC_BINARY_SEEK_CUR then size is offset from current location
            if PETSC_BINARY_SEEK_END then size is offset from end of file
-  size - number of bytes to move. Use PETSC_BINARY_INT_SIZE, PETSC_BINARY_SCALAR_SIZE,
            etc. in your calculation rather than sizeof() to compute byte lengths.

   Output Parameter:
.   offset - new offset in file

   Level: developer

   Notes: 
   Integers are stored on the file as 32 long, regardless of whether
   they are stored in the machine as 32 or 64, this means the same
   binary file may be read on any machine. Hence you CANNOT use sizeof()
   to determine the offset or location.

   Concepts: files^binary seeking
   Concepts: binary files^seeking

.seealso: PetscBinaryRead(), PetscBinaryWrite(), PetscBinaryOpen()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscBinarySeek(int fd,off_t off,PetscBinarySeekType whence,off_t *offset)
{
  int iwhence = 0;

  PetscFunctionBegin;
  if (whence == PETSC_BINARY_SEEK_SET) {
    iwhence = SEEK_SET;
  } else if (whence == PETSC_BINARY_SEEK_CUR) {
    iwhence = SEEK_CUR;
  } else if (whence == PETSC_BINARY_SEEK_END) {
    iwhence = SEEK_END;
  } else {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Unknown seek location");
  }
#if defined(PETSC_HAVE_LSEEK)
  *offset = lseek(fd,off,iwhence);
#elif defined(PETSC_HAVE__LSEEK)
  *offset = _lseek(fd,(long)off,iwhence);
#else
  SETERRQ(PETSC_ERR_SUP_SYS,"System does not have a way of seeking on a file");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSynchronizedBinaryRead"
/*@C
   PetscSynchronizedBinaryRead - Reads from a binary file.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the MPI communicator 
.  fd - the file
.  n  - the number of items to read 
-  type - the type of items to read (PETSC_INT, PETSC_DOUBLE or PETSC_SCALAR)

   Output Parameters:
.  p - the buffer

   Options Database Key:
.   -binary_longints - indicates the file was generated on a Cray vector 
         machine (not the T3E/D) and the ints are stored as 64 bit 
         quantities, otherwise they are stored as 32 bit

   Level: developer

   Notes: 
   Does a PetscBinaryRead() followed by an MPI_Bcast()

   PetscSynchronizedBinaryRead() uses byte swapping to work on all machines.
   Integers are stored on the file as 32 long, regardless of whether
   they are stored in the machine as 32 or 64, this means the same
   binary file may be read on any machine.

   Concepts: files^synchronized reading of binary files
   Concepts: binary files^reading, synchronized

.seealso: PetscBinaryWrite(), PetscBinaryOpen(), PetscBinaryClose(), PetscBinaryRead()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscSynchronizedBinaryRead(MPI_Comm comm,int fd,void *p,PetscInt n,PetscDataType type)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  MPI_Datatype   mtype;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscBinaryRead(fd,p,n,type);CHKERRQ(ierr);
  }
  ierr = PetscDataTypeToMPIDataType(type,&mtype);CHKERRQ(ierr);
  ierr = MPI_Bcast(p,n,mtype,0,comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSynchronizedBinaryWrite"
/*@C
   PetscSynchronizedBinaryWrite - writes to a binary file.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the MPI communicator 
.  fd - the file
.  n  - the number of items to write
.  p - the buffer
.  istemp - the buffer may be changed
-  type - the type of items to write (PETSC_INT, PETSC_DOUBLE or PETSC_SCALAR)

   Level: developer

   Notes: 
   Process 0 does a PetscBinaryWrite()

   PetscSynchronizedBinaryWrite() uses byte swapping to work on all machines.
   Integers are stored on the file as 32 long, regardless of whether
   they are stored in the machine as 32 or 64, this means the same
   binary file may be read on any machine.

   Concepts: files^synchronized writing of binary files
   Concepts: binary files^reading, synchronized

.seealso: PetscBinaryWrite(), PetscBinaryOpen(), PetscBinaryClose(), PetscBinaryRead()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscSynchronizedBinaryWrite(MPI_Comm comm,int fd,void *p,PetscInt n,PetscDataType type,PetscTruth istemp)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscBinaryWrite(fd,p,n,type,istemp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSynchronizedBinarySeek" 
/*@C
   PetscSynchronizedBinarySeek - Moves the file pointer on a PETSc binary file.


   Input Parameters:
+  fd - the file
.  whence - if PETSC_BINARY_SEEK_SET then size is an absolute location in the file
            if PETSC_BINARY_SEEK_CUR then size is offset from current location
            if PETSC_BINARY_SEEK_END then size is offset from end of file
-  off    - number of bytes to move. Use PETSC_BINARY_INT_SIZE, PETSC_BINARY_SCALAR_SIZE,
            etc. in your calculation rather than sizeof() to compute byte lengths.

   Output Parameter:
.   offset - new offset in file

   Level: developer

   Notes: 
   Integers are stored on the file as 32 long, regardless of whether
   they are stored in the machine as 32 or 64, this means the same
   binary file may be read on any machine. Hence you CANNOT use sizeof()
   to determine the offset or location.

   Concepts: binary files^seeking
   Concepts: files^seeking in binary 

.seealso: PetscBinaryRead(), PetscBinaryWrite(), PetscBinaryOpen()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscSynchronizedBinarySeek(MPI_Comm comm,int fd,off_t off,PetscBinarySeekType whence,off_t *offset)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscBinarySeek(fd,off,whence,offset);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

