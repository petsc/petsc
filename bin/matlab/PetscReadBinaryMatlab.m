function Set = PetscReadBinaryMatlab(filename)
% PETSCREADBINARYMATLAB - Reads and interprets matlab specific lines
%   from the .info files produced by PetscViewerBinaryMatlab
%
%   Input filename can be the name of the binary file with or without
%   the .info suffix
%
%   This function returns a single struct containing all objects submitted
%   to the PetscViewerBinaryMatlab viewer.

  matlabHeader = '%$$ '; % string that marks a matlab line for evaluation (could be passed)

  if (isempty(strfind(filename,'.info')))
     filename = [filename,'.info'];
  end
  fid=fopen(filename,'r');
  if (fid == -1)
     error(sprintf('PetscReadBinaryMatlab: cannot load file %s',filename))
  end
  str=fgets(fid);
  while (ischar(str))

     % check for output lines that start matlabHeader
     header = strmatch(matlabHeader,str);
     if header
	 str = str(header+length(matlabHeader):end);

     	 % check for old-style file open/close commands
	 if strfind(str,'fopen(Set.filename')
	    str = 'fd = PetscOpenFile(Set.filename);';
	 elseif strfind(str,'if (fd < 0) error'); % ignore this line
	    str = '%';
	 elseif strfind(str,'fclose(fd)');
	    str = 'close(fd);';
	 end

	 eval(str);
     end
     str=fgets(fid);
  end
  fclose(fid);
  return
