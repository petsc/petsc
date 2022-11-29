function ex42(np,opt)
%
%  ex42(np,opt) - launches ./ex42 and runs a loop 1000 times sending and then receiving a one dimensional array via a Unix socket to it
%
%  Run with option -on_error_attach_debugger to debug
%
%  Requires PETSc be configured with --with-matlab
%
%  MATLABPATH must contain
%     ${PETSC_DIR}/share/petsc/matlab and ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/matlab
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

