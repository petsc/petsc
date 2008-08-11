function result = launch(program,np,opt)
%
%  launch(program,np)
%  Starts up PETSc program
% see @sreader/sreader() and PetscBinaryRead()
% 
% Unfortunately does not emit an error code if the 
% launch failes.
%
if nargin < 2
  np = 1;
else if nargin < 3
  opt = ''
end
end

%
% to run parallel jobs make sure petscmpirun is in your path
% with the particular PETSC_ARCH environmental varable set
%command = ['petscmpirun -np ' int2str(np) ' ' program opt ' &'];
command = [ program opt ' &'];
fprintf(1,['Executing: ' command])

result = system(command)
 
