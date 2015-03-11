function O = PetscOpenSocket(socketnumber)
%
%   O = PetscOpenSocket(socketnumber) - waits for a socket connection (from PETSc socket viewer)
%
%  This provides an object oriented interface to the PETSc provided Matlab routines sopen(), sread() and sclose()
%  allowing PETSc  MATLAB utilities like PetscBinaryRead.m to work cleanly with either binary
%  files or sockets
%
if nargin == 0
  S = struct('fd', sopen());
else
  S = struct('fd', sopen(socketnumber));
end
O = class(S,'PetscOpenSocket');

