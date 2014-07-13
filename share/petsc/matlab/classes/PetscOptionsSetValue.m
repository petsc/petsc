function err = PetscOptionsSetValue(option,value)
%
%  Adds an option to the database
%
if (nargin == 1) 
  value = '';
end
if (~ischar(value)) 
  value = num2str(value);
end
err = calllib('libpetsc', 'PetscOptionsSetValue', option,value);PetscCHKERRQ(err);


