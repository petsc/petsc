C
C  $Id: viewer.h,v 1.7 1996/04/12 00:07:51 curfman Exp balay $;
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
      integer  ASCII_FORMAT_DEFAULT, ASCII_FORMAT_MATLAB,
     &         ASCII_FORMAT_IMPL, ASCII_FORMAT_INFO,
     &         ASCII_FORMAT_INFO_DETAILED, ASCII_FORMAT_COMMON

      parameter ( ASCII_FORMAT_DEFAULT = 0, ASCII_FORMAT_MATLAB = 1,
     &            ASCII_FORMAT_IMPL = 2, ASCII_FORMAT_INFO = 3,
     &            ASCII_FORMAT_INFO_DETAILED = 2, 
     &            ASCII_FORMAT_COMMON = 5)
      integer  BINARY_FORMAT_DEFAULT, BINARY_FORMAT_NATIVE

      parameter (BINARY_FORMAT_DEFAULT = 0, BINARY_FORMAT_NATIVE = 1)
C
C  Flags for binary I/O
C
      integer  BINARY_RDONLY, BINARY_WRONLY, BINARY_CREATE

      parameter (BINARY_RDONLY = 0, BINARY_WRONLY = 1, 
     &           BINARY_CREATE = 2)
C
C  End of Fortran include file for the Viewer package in PETSc



