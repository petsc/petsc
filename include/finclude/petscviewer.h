C
C  $Id: viewer.h,v 1.4 1996/02/12 20:29:43 bsmith Exp bsmith $;
C
C  Include file for Fortran use of the Viewer package in PETSc
C
#define Viewer           integer
#define ViewerType       integer

      integer MATLAB_VIEWER, ASCII_FILE_VIEWER,ASCII_FILES_VIEWER,
     &        BINARY_FILE_VIEWER, STRING_VIEWER,DRAW_VIEWER 
      parameter (MATLAB_VIEWER = 0, ASCII_FILE_VIEWER = 1,
     &           ASCII_FILES_VIEWER = 2, BINARY_FILE_VIEWER = 3,
     &           STRING_VIEWER = 4, DRAW_VIEWER = 5) 
C
C  Viewer formats
C
      integer  FILE_FORMAT_DEFAULT, FILE_FORMAT_MATLAB,
     &         FILE_FORMAT_IMPL, FILE_FORMAT_INFO

      parameter ( FILE_FORMAT_DEFAULT = 0, FILE_FORMAT_MATLAB = 1,
     &            FILE_FORMAT_IMPL = 2, FILE_FORMAT_INFO = 3)
C
C  Flags for binary I/O
C
      integer  BINARY_RDONLY, BINARY_WRONLY, BINARY_CREATE

      parameter (BINARY_RDONLY = 0, BINARY_WRONLY = 1, 
     &           BINARY_CREATE = 2)
C
C  End of Fortran include file for the Viewer package in PETSc



