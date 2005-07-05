function launch(program,np)
%
%  launch(program,np)
%  Starts up PETSc program
% see openport, closeport and receive
% 
% Unfortunately does not emit an error code if the 
% launch failes.
%
if nargin == 1
  np = 1;
end

command = ['mpirun -np ' int2str(np) ' ' program ' &'];
fprintf(1,['Executing: ' command])

system(command)
 
