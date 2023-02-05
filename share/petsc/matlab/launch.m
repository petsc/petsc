function result = launch(program,np,opt)
%
%  launch(program,np)
%  Starts up PETSc program
%
% Unfortunately does not emit an error code if the launch fails and one cannot see the output
% including error messages from the PETSc code.
%
% To debug problems we recommend commenting out the launch script from the MATLAB script and
% in a separate terminal starting the PETSc program manually, for example petscmpiexec -n 1 ./ex1 -info other options
% The MATLAB script will block on the PetscOpenSocket() until the PETSc executable is started.

% see also PetscBinaryRead()
%
if nargin < 2
  np = 1;
else if nargin < 3
   opt = '';
end
end

%
% to run parallel jobs make sure petscmpiexec is in your path
% with the particular PETSC_ARCH environmental variable set
%command = ['petscmpiexec -np ' int2str(np) ' ' program opt ' &'];
command = [ program opt ' &'];
fprintf(1,['Executing: ' command])

result = system(command)

