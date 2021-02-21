
/*
   This file contains simple binary read/write routines.
 */

#include <petscsys.h>
#include <petscbt.h>
#include <errno.h>
#include <fcntl.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_IO_H)
#include <io.h>
#endif
#if !defined(PETSC_HAVE_O_BINARY)
#define O_BINARY 0
#endif

const char *const PetscFileModes[] = {"READ","WRITE","APPEND","UPDATE","APPEND_UPDATE","PetscFileMode","PETSC_FILE_",NULL};

/* --------------------------------------------------------- */
/*
  PetscByteSwapEnum - Swap bytes in a  PETSc Enum

*/
PetscErrorCode  PetscByteSwapEnum(PetscEnum *buff,PetscInt n)
{
  PetscInt  i,j;
  PetscEnum tmp = ENUM_DUMMY;
  char      *ptr1,*ptr2 = (char*)&tmp;

  PetscFunctionBegin;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff + j);
    for (i=0; i<(PetscInt)sizeof(PetscEnum); i++) ptr2[i] = ptr1[sizeof(PetscEnum)-1-i];
    for (i=0; i<(PetscInt)sizeof(PetscEnum); i++) ptr1[i] = ptr2[i];
  }
  PetscFunctionReturn(0);
}

/*
  PetscByteSwapBool - Swap bytes in a  PETSc Bool

*/
PetscErrorCode  PetscByteSwapBool(PetscBool *buff,PetscInt n)
{
  PetscInt  i,j;
  PetscBool tmp = PETSC_FALSE;
  char      *ptr1,*ptr2 = (char*)&tmp;

  PetscFunctionBegin;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff + j);
    for (i=0; i<(PetscInt)sizeof(PetscBool); i++) ptr2[i] = ptr1[sizeof(PetscBool)-1-i];
    for (i=0; i<(PetscInt)sizeof(PetscBool); i++) ptr1[i] = ptr2[i];
  }
  PetscFunctionReturn(0);
}

/*
  PetscByteSwapInt - Swap bytes in a  PETSc integer (which may be 32 or 64 bits)

*/
PetscErrorCode  PetscByteSwapInt(PetscInt *buff,PetscInt n)
{
  PetscInt i,j,tmp = 0;
  char     *ptr1,*ptr2 = (char*)&tmp;

  PetscFunctionBegin;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff + j);
    for (i=0; i<(PetscInt)sizeof(PetscInt); i++) ptr2[i] = ptr1[sizeof(PetscInt)-1-i];
    for (i=0; i<(PetscInt)sizeof(PetscInt); i++) ptr1[i] = ptr2[i];
  }
  PetscFunctionReturn(0);
}

/*
  PetscByteSwapInt64 - Swap bytes in a  PETSc integer (64 bits)

*/
PetscErrorCode  PetscByteSwapInt64(PetscInt64 *buff,PetscInt n)
{
  PetscInt   i,j;
  PetscInt64 tmp = 0;
  char       *ptr1,*ptr2 = (char*)&tmp;

  PetscFunctionBegin;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff + j);
    for (i=0; i<(PetscInt)sizeof(PetscInt64); i++) ptr2[i] = ptr1[sizeof(PetscInt64)-1-i];
    for (i=0; i<(PetscInt)sizeof(PetscInt64); i++) ptr1[i] = ptr2[i];
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------- */
/*
  PetscByteSwapShort - Swap bytes in a short
*/
PetscErrorCode  PetscByteSwapShort(short *buff,PetscInt n)
{
  PetscInt i,j;
  short    tmp;
  char     *ptr1,*ptr2 = (char*)&tmp;

  PetscFunctionBegin;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff + j);
    for (i=0; i<(PetscInt) sizeof(short); i++) ptr2[i] = ptr1[sizeof(short)-1-i];
    for (i=0; i<(PetscInt) sizeof(short); i++) ptr1[i] = ptr2[i];
  }
  PetscFunctionReturn(0);
}
/*
  PetscByteSwapLong - Swap bytes in a long
*/
PetscErrorCode  PetscByteSwapLong(long *buff,PetscInt n)
{
  PetscInt i,j;
  long     tmp;
  char     *ptr1,*ptr2 = (char*)&tmp;

  PetscFunctionBegin;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff + j);
    for (i=0; i<(PetscInt) sizeof(long); i++) ptr2[i] = ptr1[sizeof(long)-1-i];
    for (i=0; i<(PetscInt) sizeof(long); i++) ptr1[i] = ptr2[i];
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------- */
/*
  PetscByteSwapReal - Swap bytes in a PetscReal
*/
PetscErrorCode  PetscByteSwapReal(PetscReal *buff,PetscInt n)
{
  PetscInt  i,j;
  PetscReal tmp,*buff1 = (PetscReal*)buff;
  char      *ptr1,*ptr2 = (char*)&tmp;

  PetscFunctionBegin;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff1 + j);
    for (i=0; i<(PetscInt) sizeof(PetscReal); i++) ptr2[i] = ptr1[sizeof(PetscReal)-1-i];
    for (i=0; i<(PetscInt) sizeof(PetscReal); i++) ptr1[i] = ptr2[i];
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------- */
/*
  PetscByteSwapScalar - Swap bytes in a PetscScalar
  The complex case is dealt with with an array of PetscReal, twice as long.
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
    for (i=0; i<(PetscInt) sizeof(PetscReal); i++) ptr2[i] = ptr1[sizeof(PetscReal)-1-i];
    for (i=0; i<(PetscInt) sizeof(PetscReal); i++) ptr1[i] = ptr2[i];
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------- */
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
    for (i=0; i<(PetscInt) sizeof(double); i++) ptr2[i] = ptr1[sizeof(double)-1-i];
    for (i=0; i<(PetscInt) sizeof(double); i++) ptr1[i] = ptr2[i];
  }
  PetscFunctionReturn(0);
}

/*
  PetscByteSwapFloat - Swap bytes in a float
*/
PetscErrorCode PetscByteSwapFloat(float *buff,PetscInt n)
{
  PetscInt i,j;
  float    tmp,*buff1 = (float*)buff;
  char     *ptr1,*ptr2 = (char*)&tmp;

  PetscFunctionBegin;
  for (j=0; j<n; j++) {
    ptr1 = (char*)(buff1 + j);
    for (i=0; i<(PetscInt) sizeof(float); i++) ptr2[i] = ptr1[sizeof(float)-1-i];
    for (i=0; i<(PetscInt) sizeof(float); i++) ptr1[i] = ptr2[i];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscByteSwap(void *data,PetscDataType pdtype,PetscInt count)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if      (pdtype == PETSC_INT)    {ierr = PetscByteSwapInt((PetscInt*)data,count);CHKERRQ(ierr);}
  else if (pdtype == PETSC_ENUM)   {ierr = PetscByteSwapEnum((PetscEnum*)data,count);CHKERRQ(ierr);}
  else if (pdtype == PETSC_BOOL)   {ierr = PetscByteSwapBool((PetscBool*)data,count);CHKERRQ(ierr);}
  else if (pdtype == PETSC_SCALAR) {ierr = PetscByteSwapScalar((PetscScalar*)data,count);CHKERRQ(ierr);}
  else if (pdtype == PETSC_REAL)   {ierr = PetscByteSwapReal((PetscReal*)data,count);CHKERRQ(ierr);}
  else if (pdtype == PETSC_COMPLEX){ierr = PetscByteSwapReal((PetscReal*)data,2*count);CHKERRQ(ierr);}
  else if (pdtype == PETSC_INT64)  {ierr = PetscByteSwapInt64((PetscInt64*)data,count);CHKERRQ(ierr);}
  else if (pdtype == PETSC_DOUBLE) {ierr = PetscByteSwapDouble((double*)data,count);CHKERRQ(ierr);}
  else if (pdtype == PETSC_FLOAT)  {ierr = PetscByteSwapFloat((float*)data,count);CHKERRQ(ierr);}
  else if (pdtype == PETSC_SHORT)  {ierr = PetscByteSwapShort((short*)data,count);CHKERRQ(ierr);}
  else if (pdtype == PETSC_LONG)   {ierr = PetscByteSwapLong((long*)data,count);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
   PetscBinaryRead - Reads from a binary file.

   Not Collective

   Input Parameters:
+  fd - the file descriptor
.  num  - the maximum number of items to read
-  type - the type of items to read (PETSC_INT, PETSC_REAL, PETSC_SCALAR, etc.)

   Output Parameters:
+  data - the buffer
-  count - the number of items read, optional

   Level: developer

   Notes:
   If count is not provided and the number of items read is less than
   the maximum number of items to read, then this routine errors.

   PetscBinaryRead() uses byte swapping to work on all machines; the files
   are written to file ALWAYS using big-endian ordering. On little-endian machines the numbers
   are converted to the little-endian format when they are read in from the file.
   When PETSc is ./configure with --with-64-bit-indices the integers are written to the
   file as 64 bit integers, this means they can only be read back in when the option --with-64-bit-indices
   is used.

.seealso: PetscBinaryWrite(), PetscBinaryOpen(), PetscBinaryClose(), PetscViewerBinaryGetDescriptor(), PetscBinarySynchronizedWrite(),
          PetscBinarySynchronizedRead(), PetscBinarySynchronizedSeek()
@*/
PetscErrorCode  PetscBinaryRead(int fd,void *data,PetscInt num,PetscInt *count,PetscDataType type)
{
  size_t            typesize, m = (size_t) num, n = 0, maxblock = 65536;
  char              *p = (char*)data;
#if defined(PETSC_USE_REAL___FLOAT128)
  PetscBool         readdouble = PETSC_FALSE;
  double            *pdouble;
#endif
  void              *ptmp = data;
  char              *fname = NULL;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (count) *count = 0;
  if (num < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to read a negative amount of data %D",num);
  if (!num) PetscFunctionReturn(0);

  if (type == PETSC_FUNCTION) {
    m     = 64;
    type  = PETSC_CHAR;
    fname = (char*)malloc(m*sizeof(char));
    p     = (char*)fname;
    ptmp  = (void*)fname;
    if (!fname) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"Cannot allocate space for function name");
  }
  if (type == PETSC_BIT_LOGICAL) m = PetscBTLength(m);

  ierr = PetscDataTypeGetSize(type,&typesize);CHKERRQ(ierr);

#if defined(PETSC_USE_REAL___FLOAT128)
  ierr = PetscOptionsGetBool(NULL,NULL,"-binary_read_double",&readdouble,NULL);CHKERRQ(ierr);
  /* If using __float128 precision we still read in doubles from file */
  if ((type == PETSC_REAL || type == PETSC_COMPLEX) && readdouble) {
    PetscInt cnt = num * ((type == PETSC_REAL) ? 1 : 2);
    ierr = PetscMalloc1(cnt,&pdouble);CHKERRQ(ierr);
    p = (char*)pdouble;
    typesize /= 2;
  }
#endif

  m *= typesize;

  while (m) {
    size_t len = (m < maxblock) ? m : maxblock;
    int    ret = (int)read(fd,p,len);
    if (ret < 0 && errno == EINTR) continue;
    if (!ret && len > 0) break; /* Proxy for EOF */
    if (ret < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Error reading from file, errno %d",errno);
    m -= ret;
    p += ret;
    n += ret;
  }
  if (m && !count) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Read past end of file");

  num = (PetscInt)(n/typesize); /* Should we require `n % typesize == 0` ? */
  if (count) *count = num;      /* TODO: This is most likely wrong for PETSC_BIT_LOGICAL */

#if defined(PETSC_USE_REAL___FLOAT128)
  if ((type == PETSC_REAL || type == PETSC_COMPLEX) && readdouble) {
    PetscInt  i, cnt = num * ((type == PETSC_REAL) ? 1 : 2);
    PetscReal *preal = (PetscReal*)data;
    if (!PetscBinaryBigEndian()) {ierr = PetscByteSwapDouble(pdouble,cnt);CHKERRQ(ierr);}
    for (i=0; i<cnt; i++) preal[i] = pdouble[i];
    ierr = PetscFree(pdouble);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif

  if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap(ptmp,type,num);CHKERRQ(ierr);}

  if (type == PETSC_FUNCTION) {
#if defined(PETSC_SERIALIZE_FUNCTIONS)
    ierr = PetscDLSym(NULL,fname,(void**)data);CHKERRQ(ierr);
#else
    *(void**)data = NULL;
#endif
    free(fname);
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscBinaryWrite - Writes to a binary file.

   Not Collective

   Input Parameters:
+  fd     - the file
.  p      - the buffer
.  n      - the number of items to write
-  type   - the type of items to read (PETSC_INT, PETSC_DOUBLE or PETSC_SCALAR)

   Level: advanced

   Notes:
   PetscBinaryWrite() uses byte swapping to work on all machines; the files
   are written using big-endian ordering to the file. On little-endian machines the numbers
   are converted to the big-endian format when they are written to disk.
   When PETSc is ./configure with --with-64-bit-indices the integers are written to the
   file as 64 bit integers, this means they can only be read back in when the option --with-64-bit-indices
   is used.

   If running with __float128 precision the output is in __float128 unless one uses the -binary_write_double option

   The Buffer p should be read-write buffer, and not static data.
   This way, byte-swapping is done in-place, and then the buffer is
   written to the file.

   This routine restores the original contents of the buffer, after
   it is written to the file. This is done by byte-swapping in-place
   the second time.

   Because byte-swapping may be done on the values in data it cannot be declared const


.seealso: PetscBinaryRead(), PetscBinaryOpen(), PetscBinaryClose(), PetscViewerBinaryGetDescriptor(), PetscBinarySynchronizedWrite(),
          PetscBinarySynchronizedRead(), PetscBinarySynchronizedSeek()
@*/
PetscErrorCode  PetscBinaryWrite(int fd,const void *p,PetscInt n,PetscDataType type)
{
  const char     *pp = (char*)p;
  int            err,wsize;
  size_t         m = (size_t)n,maxblock=65536;
  PetscErrorCode ierr;
  const void     *ptmp = p;
  char           *fname = NULL;
#if defined(PETSC_USE_REAL___FLOAT128)
  PetscBool      writedouble = PETSC_FALSE;
  double         *ppp;
  PetscReal      *pv;
  PetscInt       i;
#endif
  PetscDataType  wtype = type;

  PetscFunctionBegin;
  if (n < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to write a negative amount of data %D",n);
  if (!n) PetscFunctionReturn(0);

  if (type == PETSC_FUNCTION) {
#if defined(PETSC_SERIALIZE_FUNCTIONS)
    const char *fnametmp;
#endif
    m     = 64;
    fname = (char*)malloc(m*sizeof(char));
    if (!fname) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"Cannot allocate space for function name");
#if defined(PETSC_SERIALIZE_FUNCTIONS)
    if (n > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Can only binary view a single function at a time");
    ierr = PetscFPTFind(*(void**)p,&fnametmp);CHKERRQ(ierr);
    ierr = PetscStrncpy(fname,fnametmp,m);CHKERRQ(ierr);
#else
    ierr = PetscStrncpy(fname,"",m);CHKERRQ(ierr);
#endif
    wtype = PETSC_CHAR;
    pp    = (char*)fname;
    ptmp  = (void*)fname;
  }

#if defined(PETSC_USE_REAL___FLOAT128)
  ierr = PetscOptionsGetBool(NULL,NULL,"-binary_write_double",&writedouble,NULL);CHKERRQ(ierr);
  /* If using __float128 precision we still write in doubles to file */
  if ((type == PETSC_SCALAR || type == PETSC_REAL || type == PETSC_COMPLEX) && writedouble) {
    wtype = PETSC_DOUBLE;
    ierr = PetscMalloc1(n,&ppp);CHKERRQ(ierr);
    pv = (PetscReal*)pp;
    for (i=0; i<n; i++) {
      ppp[i] = (double) pv[i];
    }
    pp   = (char*)ppp;
    ptmp = (char*)ppp;
  }
#endif

  if (wtype == PETSC_INT)          m *= sizeof(PetscInt);
  else if (wtype == PETSC_SCALAR)  m *= sizeof(PetscScalar);
#if defined(PETSC_HAVE_COMPLEX)
  else if (wtype == PETSC_COMPLEX) m *= sizeof(PetscComplex);
#endif
  else if (wtype == PETSC_REAL)    m *= sizeof(PetscReal);
  else if (wtype == PETSC_DOUBLE)  m *= sizeof(double);
  else if (wtype == PETSC_FLOAT)   m *= sizeof(float);
  else if (wtype == PETSC_SHORT)   m *= sizeof(short);
  else if (wtype == PETSC_LONG)    m *= sizeof(long);
  else if (wtype == PETSC_CHAR)    m *= sizeof(char);
  else if (wtype == PETSC_ENUM)    m *= sizeof(PetscEnum);
  else if (wtype == PETSC_BOOL)    m *= sizeof(PetscBool);
  else if (wtype == PETSC_INT64)   m *= sizeof(PetscInt64);
  else if (wtype == PETSC_BIT_LOGICAL) m = PetscBTLength(m)*sizeof(char);
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown type");

  if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap((void*)ptmp,wtype,n);CHKERRQ(ierr);}

  while (m) {
    wsize = (m < maxblock) ? m : maxblock;
    err   = write(fd,pp,wsize);
    if (err < 0 && errno == EINTR) continue;
    if (err != wsize) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_FILE_WRITE,"Error writing to file total size %d err %d wsize %d",(int)n,(int)err,(int)wsize);
    m  -= wsize;
    pp += wsize;
  }

  if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap((void*)ptmp,wtype,n);CHKERRQ(ierr);}

  if (type == PETSC_FUNCTION) {
    free(fname);
  }
#if defined(PETSC_USE_REAL___FLOAT128)
  if ((type == PETSC_SCALAR || type == PETSC_REAL || type == PETSC_COMPLEX) && writedouble) {
    ierr = PetscFree(ppp);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/*@C
   PetscBinaryOpen - Opens a PETSc binary file.

   Not Collective

   Input Parameters:
+  name - filename
-  mode - open mode of binary file, one of FILE_MODE_READ, FILE_MODE_WRITE, FILE_MODE_APPEND

   Output Parameter:
.  fd - the file

   Level: advanced


   Notes:
    Files access with PetscBinaryRead() and PetscBinaryWrite() are ALWAYS written in
   big-endian format. This means the file can be accessed using PetscBinaryOpen() and
   PetscBinaryRead() and PetscBinaryWrite() on any machine.

.seealso: PetscBinaryRead(), PetscBinaryWrite(), PetscFileMode, PetscViewerFileSetMode(), PetscViewerBinaryGetDescriptor(),
          PetscBinarySynchronizedWrite(), PetscBinarySynchronizedRead(), PetscBinarySynchronizedSeek()

@*/
PetscErrorCode  PetscBinaryOpen(const char name[],PetscFileMode mode,int *fd)
{
  PetscFunctionBegin;
  switch (mode) {
  case FILE_MODE_READ:   *fd = open(name,O_BINARY|O_RDONLY,0); break;
  case FILE_MODE_WRITE:  *fd = open(name,O_BINARY|O_WRONLY|O_CREAT|O_TRUNC,0666); break;
  case FILE_MODE_APPEND: *fd = open(name,O_BINARY|O_WRONLY|O_APPEND,0); break;
  default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported file mode %s",PetscFileModes[mode]);
  }
  if (*fd == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s for %s: %s",name,PetscFileModes[mode]);
  PetscFunctionReturn(0);
}

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
  if (close(fd)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"close() failed on file descriptor");
  PetscFunctionReturn(0);
}


/*@C
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


.seealso: PetscBinaryRead(), PetscBinaryWrite(), PetscBinaryOpen(), PetscBinarySynchronizedWrite(), PetscBinarySynchronizedRead(),
          PetscBinarySynchronizedSeek()
@*/
PetscErrorCode  PetscBinarySeek(int fd,off_t off,PetscBinarySeekType whence,off_t *offset)
{
  int iwhence = 0;

  PetscFunctionBegin;
  if (whence == PETSC_BINARY_SEEK_SET) iwhence = SEEK_SET;
  else if (whence == PETSC_BINARY_SEEK_CUR) iwhence = SEEK_CUR;
  else if (whence == PETSC_BINARY_SEEK_END) iwhence = SEEK_END;
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown seek location");
#if defined(PETSC_HAVE_LSEEK)
  *offset = lseek(fd,off,iwhence);
#elif defined(PETSC_HAVE__LSEEK)
  *offset = _lseek(fd,(long)off,iwhence);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"System does not have a way of seeking on a file");
#endif
  PetscFunctionReturn(0);
}

/*@C
   PetscBinarySynchronizedRead - Reads from a binary file.

   Collective

   Input Parameters:
+  comm - the MPI communicator
.  fd - the file descriptor
.  num  - the maximum number of items to read
-  type - the type of items to read (PETSC_INT, PETSC_REAL, PETSC_SCALAR, etc.)

   Output Parameters:
+  data - the buffer
-  count - the number of items read, optional

   Level: developer

   Notes:
   Does a PetscBinaryRead() followed by an MPI_Bcast()

   If count is not provided and the number of items read is less than
   the maximum number of items to read, then this routine errors.

   PetscBinarySynchronizedRead() uses byte swapping to work on all machines.
   Integers are stored on the file as 32 long, regardless of whether
   they are stored in the machine as 32 or 64, this means the same
   binary file may be read on any machine.


.seealso: PetscBinaryWrite(), PetscBinaryOpen(), PetscBinaryClose(), PetscBinaryRead(), PetscBinarySynchronizedWrite(),
          PetscBinarySynchronizedSeek()
@*/
PetscErrorCode  PetscBinarySynchronizedRead(MPI_Comm comm,int fd,void *data,PetscInt num,PetscInt *count,PetscDataType type)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  MPI_Datatype   mtype;
  PetscInt       ibuf[2] = {0, 0};
  char           *fname = NULL;
  void           *fptr = NULL;

  PetscFunctionBegin;
  if (type == PETSC_FUNCTION) {
    num   = 64;
    type  = PETSC_CHAR;
    fname = (char*)malloc(num*sizeof(char));
    fptr  = data;
    data  = (void*)fname;
    if (!fname) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"Cannot allocate space for function name");
  }

  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (!rank) {
    ibuf[0] = PetscBinaryRead(fd,data,num,count?&ibuf[1]:NULL,type);
  }
  ierr = MPI_Bcast(ibuf,2,MPIU_INT,0,comm);CHKERRQ(ierr);
  ierr = (PetscErrorCode)ibuf[0];CHKERRQ(ierr);

  /* skip MPI call on potentially huge amounts of data when running with one process; this allows the amount of data to basically unlimited in that case */
  if (size > 1) {
    ierr = PetscDataTypeToMPIDataType(type,&mtype);CHKERRQ(ierr);
    ierr = MPI_Bcast(data,count?ibuf[1]:num,mtype,0,comm);CHKERRQ(ierr);
  }
  if (count) *count = ibuf[1];

  if (type == PETSC_FUNCTION) {
#if defined(PETSC_SERIALIZE_FUNCTIONS)
    ierr = PetscDLLibrarySym(PETSC_COMM_SELF,&PetscDLLibrariesLoaded,NULL,fname,(void**)fptr);CHKERRQ(ierr);
#else
    *(void**)fptr = NULL;
#endif
    free(fname);
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscBinarySynchronizedWrite - writes to a binary file.

   Collective

   Input Parameters:
+  comm - the MPI communicator
.  fd - the file
.  n  - the number of items to write
.  p - the buffer
-  type - the type of items to write (PETSC_INT, PETSC_DOUBLE or PETSC_SCALAR)

   Level: developer

   Notes:
   Process 0 does a PetscBinaryWrite()

   PetscBinarySynchronizedWrite() uses byte swapping to work on all machines.
   Integers are stored on the file as 32 long, regardless of whether
   they are stored in the machine as 32 or 64, this means the same
   binary file may be read on any machine.

   Notes:
    because byte-swapping may be done on the values in data it cannot be declared const

   WARNING: This is NOT like PetscSynchronizedFPrintf()! This routine ignores calls on all but process 0,
   while PetscSynchronizedFPrintf() has all processes print their strings in order.


.seealso: PetscBinaryWrite(), PetscBinaryOpen(), PetscBinaryClose(), PetscBinaryRead(), PetscBinarySynchronizedRead(),
          PetscBinarySynchronizedSeek()
@*/
PetscErrorCode  PetscBinarySynchronizedWrite(MPI_Comm comm,int fd,const void *p,PetscInt n,PetscDataType type)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscBinaryWrite(fd,p,n,type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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

#if defined(PETSC_USE_PETSC_MPI_EXTERNAL32)
/*
      MPICH does not provide the external32 representation for MPI_File_set_view() so we need to provide the functions.
    These are set into MPI in PetscInitialize() via MPI_Register_datarep()

    Note I use PetscMPIInt for the MPI error codes since that is what MPI uses (instead of the standard PetscErrorCode)

    The next three routines are not used because MPICH does not support their use

*/
PETSC_EXTERN PetscMPIInt PetscDataRep_extent_fn(MPI_Datatype datatype,MPI_Aint *file_extent,void *extra_state)
{
  MPI_Aint    ub;
  PetscMPIInt ierr;

  ierr = MPI_Type_get_extent(datatype,&ub,file_extent);
  return ierr;
}

PETSC_EXTERN PetscMPIInt PetscDataRep_read_conv_fn(void *userbuf, MPI_Datatype datatype,PetscMPIInt count,void *filebuf, MPI_Offset position,void *extra_state)
{
  PetscDataType pdtype;
  PetscMPIInt   ierr;
  size_t        dsize;

  ierr = PetscMPIDataTypeToPetscDataType(datatype,&pdtype);CHKERRQ(ierr);
  ierr = PetscDataTypeGetSize(pdtype,&dsize);CHKERRQ(ierr);

  /* offset is given in units of MPI_Datatype */
  userbuf = ((char*)userbuf) + dsize*position;

  ierr = PetscMemcpy(userbuf,filebuf,count*dsize);CHKERRQ(ierr);
  if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap(userbuf,pdtype,count);CHKERRQ(ierr);}
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
  userbuf = ((char*)userbuf) + dsize*position;

  ierr = PetscMemcpy(filebuf,userbuf,count*dsize);CHKERRQ(ierr);
  if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap(filebuf,pdtype,count);CHKERRQ(ierr);}
  return ierr;
}
#endif

PetscErrorCode MPIU_File_write_all(MPI_File fd,void *data,PetscMPIInt cnt,MPI_Datatype dtype,MPI_Status *status)
{
  PetscDataType  pdtype;
  PetscErrorCode ierr;


  PetscFunctionBegin;
  ierr = PetscMPIDataTypeToPetscDataType(dtype,&pdtype);CHKERRQ(ierr);
  if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap(data,pdtype,cnt);CHKERRQ(ierr);}
  ierr = MPI_File_write_all(fd,data,cnt,dtype,status);CHKERRQ(ierr);
  if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap(data,pdtype,cnt);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode MPIU_File_read_all(MPI_File fd,void *data,PetscMPIInt cnt,MPI_Datatype dtype,MPI_Status *status)
{
  PetscDataType  pdtype;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMPIDataTypeToPetscDataType(dtype,&pdtype);CHKERRQ(ierr);
  ierr = MPI_File_read_all(fd,data,cnt,dtype,status);CHKERRQ(ierr);
  if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap(data,pdtype,cnt);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode MPIU_File_write_at(MPI_File fd,MPI_Offset off,void *data,PetscMPIInt cnt,MPI_Datatype dtype,MPI_Status *status)
{
  PetscDataType  pdtype;
  PetscErrorCode ierr;


  PetscFunctionBegin;
  ierr = PetscMPIDataTypeToPetscDataType(dtype,&pdtype);CHKERRQ(ierr);
  if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap(data,pdtype,cnt);CHKERRQ(ierr);}
  ierr = MPI_File_write_at(fd,off,data,cnt,dtype,status);CHKERRQ(ierr);
  if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap(data,pdtype,cnt);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode MPIU_File_read_at(MPI_File fd,MPI_Offset off,void *data,PetscMPIInt cnt,MPI_Datatype dtype,MPI_Status *status)
{
  PetscDataType  pdtype;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMPIDataTypeToPetscDataType(dtype,&pdtype);CHKERRQ(ierr);
  ierr = MPI_File_read_at(fd,off,data,cnt,dtype,status);CHKERRQ(ierr);
  if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap(data,pdtype,cnt);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode MPIU_File_write_at_all(MPI_File fd,MPI_Offset off,void *data,PetscMPIInt cnt,MPI_Datatype dtype,MPI_Status *status)
{
  PetscDataType  pdtype;
  PetscErrorCode ierr;


  PetscFunctionBegin;
  ierr = PetscMPIDataTypeToPetscDataType(dtype,&pdtype);CHKERRQ(ierr);
  if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap(data,pdtype,cnt);CHKERRQ(ierr);}
  ierr = MPI_File_write_at_all(fd,off,data,cnt,dtype,status);CHKERRQ(ierr);
  if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap(data,pdtype,cnt);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode MPIU_File_read_at_all(MPI_File fd,MPI_Offset off,void *data,PetscMPIInt cnt,MPI_Datatype dtype,MPI_Status *status)
{
  PetscDataType  pdtype;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMPIDataTypeToPetscDataType(dtype,&pdtype);CHKERRQ(ierr);
  ierr = MPI_File_read_at_all(fd,off,data,cnt,dtype,status);CHKERRQ(ierr);
  if (!PetscBinaryBigEndian()) {ierr = PetscByteSwap(data,pdtype,cnt);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#endif
