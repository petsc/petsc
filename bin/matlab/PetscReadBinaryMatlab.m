function Set = PetscReadBinaryMatlab(filename)
% PETSCREADBINARYMATLAB - Reads and interprets matlab specific lines 
%   from the .info files produced by PetscViewerBinaryMatlab
%
%   Input filename can be the name of the binary file with or without
%   the .info suffix
%
%   This function returns a single struct containing all objects submitted
%   to the PetscViewerBinaryMatlab viewer.
   
   if (isempty(strfind(filename,'.info')))
      filename = [filename,'.info'];
   end
   fid=fopen(filename,'r');
   if (fid == -1)
      error(sprintf('PetscReadMatlabDA: cannot load file %s',filename))
   end
   str=fgets(fid);
   while (ischar(str))
      % check for output lines that start %Set.blah=etc.
      marker = strmatch('%$$',str); 
      if marker
	 eval(str(marker+4:end));
      end
      str=fgets(fid);
   end
   return
