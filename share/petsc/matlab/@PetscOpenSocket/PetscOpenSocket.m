function O = PetscOpenSocket(socketnumber)
%
%   O = PetscOpenSocket(socketnumber) - waits for a socket connection (from PETSc socket viewer)
%
%  This provides an object oriented interface to the PETSc provided MATLAB routines sopen(), sread() and sclose()
%  allowing PETSc MATLAB utilities like PetscBinaryRead.m to work cleanly with either binary files or sockets
%
%  The MEX source for sopen(), sread() and sclose() is in $PETSC_DIR/src/sys/classes/viewer/impls/socket/matlab/
%
if nargin == 0
  S = struct('fd', sopen());
else
  S = struct('fd', sopen(socketnumber));
end
O = class(S,'PetscOpenSocket');

