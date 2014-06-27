function string = PetscOptionsGetString(prefix,name)
%
%   Returns a string from the options database
%
%  Developer notes: I could not figure out a good way to return a character string from a C program
%  except by listing it explicitly as the returned type
if (nargin == 1) 
  name = prefix;
  prefix = '';
end 
string = calllib('libpetsc', 'PetscOptionsGetStringMatlab',prefix,name);



