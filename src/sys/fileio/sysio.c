
/* 
   This file contains simple binary read/write routines.
 */

#include <petscsys.h>
#include <errno.h>
#include <fcntl.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined (PETSC_HAVE_IO_H)
#include <io.h>
#endif
#include <petscbt.h>

/* --------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PetscByteSwapEnum"
/*
  PetscByteSwapEnum - Swap bytes in a  PETSc Enum

*/
PetscErrorCode  PetscByteSwapEnum(PetscEnum *buff,PetscInt n)
{
  PetscInt   i,j;
  PetscEnum   tmp = ENUM_DUMMY;
  char       *ptr1,*ptr2 = (char*)&tmp;
                                   
  PetscFunctionBegin;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff + j);
    for (i=0; i<(PetscInt)sizeof(PetscEnum); i++) {
      ptr2[i] = ptr1[sizeof(PetscEnum)-1-i];
    }
    for (i=0; i<(PetscInt)sizeof(PetscEnum); i++) {
      ptr1[i] = ptr2[i];
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscByteSwapBool"
/*
  PetscByteSwapBool - Swap bytes in a  PETSc Bool

*/
PetscErrorCode  PetscByteSwapBool(PetscBool *buff,PetscInt n)
{
  PetscInt    i,j;
  PetscBool   tmp = PETSC_FALSE;
  char        *ptr1,*ptr2 = (char*)&tmp;
                                   
  PetscFunctionBegin;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff + j);
    for (i=0; i<(PetscInt)sizeof(PetscBool); i++) {
      ptr2[i] = ptr1[sizeof(PetscBool)-1-i];
    }
    for (i=0; i<(PetscInt)sizeof(PetscBool); i++) {
      ptr1[i] = ptr2[i];
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscByteSwapInt"
/*
  PetscByteSwapInt - Swap bytes in a  PETSc integer (which may be 32 or 64 bits) 

*/
PetscErrorCode  PetscByteSwapInt(PetscInt *buff,PetscInt n)
{
  PetscInt  i,j,tmp = 0;
  char       *ptr1,*ptr2 = (char*)&tmp;
                                   
  PetscFunctionBegin;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff + j);
    for (i=0; i<(PetscInt)sizeof(PetscInt); i++) {
      ptr2[i] = ptr1[sizeof(PetscInt)-1-i];
    }
    for (i=0; i<(PetscInt)sizeof(PetscInt); i++) {
      ptr1[i] = ptr2[i];
    }
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PetscByteSwapShort"
/*
  PetscByteSwapShort - Swap bytes in a short
*/
PetscErrorCode  PetscByteSwapShort(short *buff,PetscInt n)
{
  PetscInt   i,j;
  short      tmp;
  char       *ptr1,*ptr2 = (char*)&tmp;

  PetscFunctionBegin;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff + j);
    for (i=0; i<(PetscInt) sizeof(short); i++) {
      ptr2[i] = ptr1[sizeof(short)-1-i];
    }
    for (i=0; i<(PetscInt) sizeof(short); i++) {
      ptr1[i] = ptr2[i];
    }
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
PetscErrorCode  PetscByteSwapScalar(PetscScalar *buff,PetscInt n)
{
  PetscInt  i,j;
  PetscReal tmp,*buff1 = (PetscReal*)buff;
  char      *ptr1,*ptr2 = (char*)&tmp;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  n *= 2;
#endif
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff1 + j);
    for (i=0; i<(PetscInt) sizeof(PetscReal); i++) {
      ptr2[i] = ptr1[sizeof(PetscReal)-1-i];
    }
    for (i=0; i<(PetscInt) sizeof(PetscReal); i++) {
      ptr1[i] = ptr2[i];
    }
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PetscByteSwapDouble"
/*
  PetscByteSwapDouble - Swap bytes in a double
*/
PetscErrorCode  PetscByteSwapDouble(double *buff,PetscInt n)
{
  PetscInt i,j;
  double   tmp,*buff1 = (double*)buff;
  char     *ptr1,*ptr2 = (char*)&tmp;

  PetscFunctionBegin;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff1 + j);
    for (i=0; i<(PetscInt) sizeof(double); i++) {
      ptr2[i] = ptr1[sizeof(double)-1-i];
    }
    for (i=0; i<(PetscInt) sizeof(double); i++) {
      ptr1[i] = ptr2[i];
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscByteSwapFloat"
/*
  PetscByteSwapFloat - Swap bytes in a float
*/
PetscErrorCode PetscByteSwapFloat(float *buff,PetscInt n)
{
  PetscInt i,j;
  float   tmp,*buff1 = (float*)buff;
  char     *ptr1,*ptr2 = (char*)&tmp;

  PetscFunctionBegin;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff1 + j);
    for (i=0; i<(PetscInt) sizeof(float); i++) {
      ptr2[i] = ptr1[sizeof(float)-1-i];
    }
    for (i=0; i<(PetscInt) sizeof(float); i++) {
      ptr1[i] = ptr2[i];
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscByteSwap"
PetscErrorCode PetscByteSwap(void *data,PetscDataType pdtype,PetscInt count)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if      (pdtype == PETSC_INT)    {ierr = PetscByteSwapInt((PetscInt*)data,count);CHKERRQ(ierr);}
  else if (pdtype == PETSC_ENUM)   {ierr = PetscByteSwapEnum((PetscEnum*)data,count);CHKERRQ(ierr);}        
  else if (pdtype == PETSC_BOOL)   {ierr = PetscByteSwapBool((PetscBool*)data,count);CHKERRQ(ierr);}
  else if (pdtype == PETSC_SCALAR) {ierr = PetscByteSwapScalar((PetscScalar*)data,count);CHKERRQ(ierr);}
  else if (pdtype == PETSC_DOUBLE) {ierr = PetscByteSwapDouble((double*)data,count);CHKERRQ(ierr);}
  else if (pdtype == PETSC_FLOAT) {ierr = PetscByteSwapFloat((float*)data,count);CHKERRQ(ierr);}
  else if (pdtype == PETSC_SHORT)  {ierr = PetscByteSwapShort((short*)data,count);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

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
   When PETSc is ./configure with --with-64bit-indices the integers are written to the
   file as 64 bit integers, this means they can only be read back in when the option --with-64bit-indices
   is used.

   Concepts: files^reading binary
   Concepts: binary files^reading

.seealso: PetscBinaryWrite(), PetscBinaryOpen(), PetscBinaryClose(), PetscViewerBinaryGetDescriptor(), PetscBinarySynchronizedWrite(),
          PetscBinarySynchronizedRead(), PetscBinarySynchronizedSeek()
@*/
PetscErrorCode  PetscBinaryRead(int fd,void *p,PetscInt n,PetscDataType type)
{
  int               wsize,err;
  size_t            m = (size_t) n,maxblock = 65536;
  char              *pp = (char*)p;
#if !defined(PETSC_WORDS_BIGENDIAN)
  PetscErrorCode    ierr;
  void              *ptmp = p; 
#endif

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  if (type == PETSC_INT)          m *= sizeof(PetscInt);
  else if (type == PETSC_SCALAR)  m *= sizeof(PetscScalar);
  else if (type == PETSC_DOUBLE)  m *= sizeof(double);
  else if (type == PETSC_FLOAT)   m *= sizeof(float);
  else if (type == PETSC_SHORT)   m *= sizeof(short);
  else if (type == PETSC_CHAR)    m *= sizeof(char);
  else if (type == PETSC_ENUM)    m *= sizeof(PetscEnum);
  else if (type == PETSC_BOOL)   m *= sizeof(PetscBool);
  else if (type == PETSC_BIT_LOGICAL) m  = PetscBTLength(m)*sizeof(char);
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown type");
  
  while (m) {
    wsize = (m < maxblock) ? m : maxblock;
    err = read(fd,pp,wsize);
    if (err < 0 && errno == EINTR) continue;
    if (!err && wsize > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Read past end of file");
    if (err < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Error reading from file, errno %d",errno);
    m  -= err;
    pp += err;
  }
#if !defined(PETSC_WORDS_BIGENDIAN)
  ierr = PetscByteSwap(ptmp,type,n);CHKERRQ(ierr);
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
   When PETSc is ./configure with --with-64bit-indices the integers are written to the
   file as 64 bit integers, this means they can only be read back in when the option --with-64bit-indices
   is used.

   The Buffer p should be read-write buffer, and not static data.
   This way, byte-swapping is done in-place, and then the buffer is
   written to the file.
   
   This routine restores the original contents of the buffer, after 
   it is written to the file. This is done by byte-swapping in-place 
   the second time. If the flag istemp is set to PETSC_TRUE, the second
   byte-swapping operation is not done, thus saving some computation,
   but the buffer is left corrupted.

   Because byte-swapping may be done on the values in data it cannot be declared const

   Concepts: files^writing binary
   Concepts: binary files^writing

.seealso: PetscBinaryRead(), PetscBinaryOpen(), PetscBinaryClose(), PetscViewerBinaryGetDescriptor(), PetscBinarySynchronizedWrite(), 
          PetscBinarySynchronizedRead(), PetscBinarySynchronizedSeek()
@*/
PetscErrorCode  PetscBinaryWrite(int fd,void *p,PetscInt n,PetscDataType type,PetscBool  istemp)
{
  char           *pp = (char*)p;
  int            err,wsize;
  size_t         m = (size_t)n,maxblock=65536;
#if !defined(PETSC_WORDS_BIGENDIAN)
  PetscErrorCode ierr;
  void           *ptmp = p; 
#endif

  PetscFunctionBegin;
  if (n < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to write a negative amount of data %D",n);
  if (!n) PetscFunctionReturn(0);

  if (type == PETSC_INT)          m *= sizeof(PetscInt);
  else if (type == PETSC_SCALAR)  m *= sizeof(PetscScalar);
  else if (type == PETSC_DOUBLE)  m *= sizeof(double);
  else if (type == PETSC_FLOAT)   m *= sizeof(float);
  else if (type == PETSC_SHORT)   m *= sizeof(short);
  else if (type == PETSC_CHAR)    m *= sizeof(char);
  else if (type == PETSC_ENUM)    m *= sizeof(PetscEnum);
  else if (type == PETSC_BOOL)   m *= sizeof(PetscBool);
  else if (type == PETSC_BIT_LOGICAL) m = PetscBTLength(m)*sizeof(char);
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown type");

#if !defined(PETSC_WORDS_BIGENDIAN)
  ierr = PetscByteSwap(ptmp,type,n);CHKERRQ(ierr);
#endif

  while (m) {
    wsize = (m < maxblock) ? m : maxblock;
    err = write(fd,pp,wsize);
    if (err < 0 && errno == EINTR) continue;
    if (err != wsize) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_WRITE,"Error writing to file.");
    m -= wsize;
    pp += wsize;
  }

#if !defined(PETSC_WORDS_BIGENDIAN)
  if (!istemp) {
    ierr = PetscByteSwap(ptmp,type,n);CHKERRQ(ierr);
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

.seealso: PetscBinaryRead(), PetscBinaryWrite(), PetscFileMode, PetscViewerFileSetMode(), PetscViewerBinaryGetDescriptor(), 
          PetscBinarySynchronizedWrite(), PetscBinarySynchronizedRead(), PetscBinarySynchronizedSeek()

@*/
PetscErrorCode  PetscBinaryOpen(const char name[],PetscFileMode mode,int *fd)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_O_BINARY) 
  if (mode == FILE_MODE_WRITE) {
    if ((*fd = open(name,O_WRONLY|O_CREAT|O_TRUNC|O_BINARY,0666)) == -1) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot create file for writing: %s",name);
    }
  } else if (mode == FILE_MODE_READ) {
    if ((*fd = open(name,O_RDONLY|O_BINARY,0)) == -1) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file for reading: %s",name);
    }
  } else if (mode == FILE_MODE_APPEND) {
    if ((*fd = open(name,O_WRONLY|O_BINARY,0)) == -1) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file for writing: %s",name);
    }
#else
  if (mode == FILE_MODE_WRITE) {
    if ((*fd = creat(name,0666)) == -1) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot create file for writing: %s",name);
    }
  } else if (mode == FILE_MODE_READ) {
    if ((*fd = open(name,O_RDONLY,0)) == -1) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file for reading: %s",name);
    }
  }
  else if (mode == FILE_MODE_APPEND) {
    if ((*fd = open(name,O_WRONLY,0)) == -1) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file for writing: %s",name);
    }
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown file mode");
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

.seealso: PetscBinaryRead(), PetscBinaryWrite(), PetscBinaryOpen(), PetscBinarySynchronizedWrite(), PetscBinarySynchronizedRead(),
          PetscBinarySynchronizedSeek()
@*/
PetscErrorCode  PetscBinaryClose(int fd)
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
.  off - number of bytes to move. Use PETSC_BINARY_INT_SIZE, PETSC_BINARY_SCALAR_SIZE,
            etc. in your calculation rather than sizeof() to compute byte lengths.
-  whence - if PETSC_BINARY_SEEK_SET then off is an absolute location in the file
            if PETSC_BINARY_SEEK_CUR then off is an offset from the current location
            if PETSC_BINARY_SEEK_END then off is an offset from the end of file

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

.seealso: PetscBinaryRead(), PetscBinaryWrite(), PetscBinaryOpen(), PetscBinarySynchronizedWrite(), PetscBinarySynchronizedRead(),
          PetscBinarySynchronizedSeek()
@*/
PetscErrorCode  PetscBinarySeek(int fd,off_t off,PetscBinarySeekType whence,off_t *offset)
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
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown seek location");
  }
#if defined(PETSC_HAVE_LSEEK)
  *offset = lseek(fd,off,iwhence);
#elif defined(PETSC_HAVE__LSEEK)
  *offset = _lseek(fd,(long)off,iwhence);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"System does not have a way of seeking on a file");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscBinarySynchronizedRead"
/*@C
   PetscBinarySynchronizedRead - Reads from a binary file.

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

   PetscBinarySynchronizedRead() uses byte swapping to work on all machines.
   Integers are stored on the file as 32 long, regardless of whether
   they are stored in the machine as 32 or 64, this means the same
   binary file may be read on any machine.

   Concepts: files^synchronized reading of binary files
   Concepts: binary files^reading, synchronized

.seealso: PetscBinaryWrite(), PetscBinaryOpen(), PetscBinaryClose(), PetscBinaryRead(), PetscBinarySynchronizedWrite(), 
          PetscBinarySynchronizedSeek()
@*/
PetscErrorCode  PetscBinarySynchronizedRead(MPI_Comm comm,int fd,void *p,PetscInt n,PetscDataType type)
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
#define __FUNCT__ "PetscBinarySynchronizedWrite"
/*@C
   PetscBinarySynchronizedWrite - writes to a binary file.

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

   PetscBinarySynchronizedWrite() uses byte swapping to work on all machines.
   Integers are stored on the file as 32 long, regardless of whether
   they are stored in the machine as 32 or 64, this means the same
   binary file may be read on any machine.

   Notes: because byte-swapping may be done on the values in data it cannot be declared const

   WARNING: This is NOT like PetscSynchronizedFPrintf()! This routine ignores calls on all but process 0,
   while PetscSynchronizedFPrintf() has all processes print their strings in order.

   Concepts: files^synchronized writing of binary files
   Concepts: binary files^reading, synchronized

.seealso: PetscBinaryWrite(), PetscBinaryOpen(), PetscBinaryClose(), PetscBinaryRead(), PetscBinarySynchronizedRead(),
          PetscBinarySynchronizedSeek()
@*/
PetscErrorCode  PetscBinarySynchronizedWrite(MPI_Comm comm,int fd,void *p,PetscInt n,PetscDataType type,PetscBool  istemp)
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
#define __FUNCT__ "PetscBinarySynchronizedSeek" 
/*@C
   PetscBinarySynchronizedSeek - Moves the file pointer on a PETSc binary file.


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

.seealso: PetscBinaryRead(), PetscBinaryWrite(), PetscBinaryOpen(), PetscBinarySynchronizedWrite(), PetscBinarySynchronizedRead(),
          PetscBinarySynchronizedSeek()
@*/
PetscErrorCode  PetscBinarySynchronizedSeek(MPI_Comm comm,int fd,off_t off,PetscBinarySeekType whence,off_t *offset)
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

#if defined(PETSC_HAVE_MPIIO)
#if !defined(PETSC_WORDS_BIGENDIAN)

#if defined(PETSC_USE_PETSC_MPI_EXTERNAL32)
EXTERN_C_BEGIN
/*
      MPICH does not provide the external32 representation for MPI_File_set_view() so we need to provide the functions.
    These are set into MPI in PetscInitialize() via MPI_Register_datarep()

    Note I use PetscMPIInt for the MPI error codes since that is what MPI uses (instead of the standard PetscErrorCode)

    The next three routines are not used because MPICH does not support their use

*/
PetscMPIInt PetscDataRep_extent_fn(MPI_Datatype datatype,MPI_Aint *file_extent,void *extra_state) 
{
  MPI_Aint    ub;
  PetscMPIInt ierr;
  
  ierr = MPI_Type_get_extent(datatype,&ub,file_extent);
  return ierr;
}

PetscMPIInt PetscDataRep_read_conv_fn(void *userbuf, MPI_Datatype datatype,PetscMPIInt count,void *filebuf, MPI_Offset position,void *extra_state) 
{
  PetscDataType pdtype;
  PetscMPIInt   ierr;
  size_t        dsize;
  
  ierr = PetscMPIDataTypeToPetscDataType(datatype,&pdtype);CHKERRQ(ierr);
  ierr = PetscDataTypeGetSize(pdtype,&dsize);CHKERRQ(ierr);

  /* offset is given in units of MPI_Datatype */
  userbuf = ((char *)userbuf) + dsize*position;

  ierr = PetscMemcpy(userbuf,filebuf,count*dsize);CHKERRQ(ierr);
  ierr = PetscByteSwap(userbuf,pdtype,count);CHKERRQ(ierr);
  return ierr;
}

PetscMPIInt PetscDataRep_write_conv_fn(void *userbuf, MPI_Datatype datatype,PetscMPIInt count,void *filebuf, MPI_Offset position,void *extra_state) 
{
  PetscDataType pdtype;
  PetscMPIInt   ierr;
  size_t        dsize;
  
  ierr = PetscMPIDataTypeToPetscDataType(datatype,&pdtype);CHKERRQ(ierr);
  ierr = PetscDataTypeGetSize(pdtype,&dsize);CHKERRQ(ierr);

  /* offset is given in units of MPI_Datatype */
  userbuf = ((char *)userbuf) + dsize*position;

  ierr = PetscMemcpy(filebuf,userbuf,count*dsize);CHKERRQ(ierr);
  ierr = PetscByteSwap(filebuf,pdtype,count);CHKERRQ(ierr);  
  return ierr;
}
EXTERN_C_END
#endif

#undef __FUNCT__  
#define __FUNCT__ "MPIU_File_write_all"
PetscErrorCode MPIU_File_write_all(MPI_File fd,void *data,PetscMPIInt cnt,MPI_Datatype dtype,MPI_Status *status)
{
  PetscErrorCode ierr;
  PetscDataType  pdtype;

  PetscFunctionBegin;
  ierr = PetscMPIDataTypeToPetscDataType(dtype,&pdtype);CHKERRQ(ierr);
  ierr = PetscByteSwap(data,pdtype,cnt);CHKERRQ(ierr);  
  ierr = MPI_File_write_all(fd,data,cnt,dtype,status);CHKERRQ(ierr);
  ierr = PetscByteSwap(data,pdtype,cnt);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MPIU_File_read_all"
PetscErrorCode MPIU_File_read_all(MPI_File fd,void *data,PetscMPIInt cnt,MPI_Datatype dtype,MPI_Status *status)
{
  PetscErrorCode ierr;
  PetscDataType  pdtype;

  PetscFunctionBegin;
  ierr = PetscMPIDataTypeToPetscDataType(dtype,&pdtype);CHKERRQ(ierr);
  ierr = MPI_File_read_all(fd,data,cnt,dtype,status);CHKERRQ(ierr);
  ierr = PetscByteSwap(data,pdtype,cnt);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}
#endif
#endif
