function ex41(np,opt)
%
%  ex41(np,opt) - receives a matrix and vector from Matlab via socket
%  solves the linear system and returns the solution vector
%
% Run with option -on_error_attach_debugger to debug
%
%  Requires the Matlab mex routines in ${PETSC_DIR}/bin/matlab.
%  Make sure that ${PETSC_DIR}/bin/matlab is in your Matlab PATH.
%
if (nargin < 1)
  np = 1;
end
if (nargin < 2) 
  opt = ' ';
end
launch('./ex42  ',np,opt);

socket=PetscOpenSocket;
delta=zeros(512,1);

for i=1:1000
  PetscBinaryWrite(socket,delta);
  delta = PetscBinaryRead(socket);
end
pause(0.1);

close(socket);

