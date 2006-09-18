function O = sreader(socketnumber)
%
%   O = sreader(socketnumber) - waits for a socket connection (from PETSc socket viewer)
%
%  This provides an object oriented interface to the PETSc provided Matlab routines sopen(), sread() and sclose()
%  allowing PETSc  Matlab utilities like PetscBinaryRead.m to work cleanly with either binary
%  files or sockets
%
if nargin == 0
  socketnumber = 5000;
end
S = struct('fd', sopen(socketnumber))
O = class(S,'sreader');

