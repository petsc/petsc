C
C      Include file for Fortran use of the Viewer package in PETSc
C
#define Viewer           integer

      integer  FILE_FORMAT_DEFAULT, FILE_FORMAT_MATLAB,
     &         FILE_FORMAT_IMPL, FILE_FORMAT_INFO

      parameter ( FILE_FORMAT_DEFAULT = 0, FILE_FORMAT_MATLAB = 1,
     &            FILE_FORMAT_IMPL = 2, FILE_FORMAT_INFO = 3)

      integer  BINARY_RDONLY, BINARY_WRONLY, BINARY_CREATE

      parameter (BINARY_RDONLY = 0, BINARY_WRONLY = 1, 
     &           BINARY_CREATE = 2)
C
C      End of Fortran include file for the Viewer package in PETSc



