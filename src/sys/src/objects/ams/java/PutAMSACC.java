/*$Id: PutAMSACC.java,v 1.1 2000/11/19 00:51:09 bsmith Exp bsmith $*/
/*
       Installs the AMS dynamic library on your system if it
   is not already there.
*/

import java.lang.Thread;
import java.io.*;
import java.net.*;

public class PutAMSACC {
  
  /*
     The constructor for the class.
  */
    public PutAMSACC () throws java.lang.SecurityException 
  {
    File   file,dir;
    String drives[] = {"c","e"},drive;
    int    ndrives = 2;
    int    i;

    String os = System.getProperty("os.name");
 
    if (os.indexOf("Windows") >= 0) {
      if (os.indexOf("95") >= 0 || os.indexOf("98") > 0) {
	for (i=0; i<ndrives; i++) {
	  drive = drives[i];   
          dir = new File(drive+":/WINDOWS/System");
          if (dir.exists()) {
            file = new File(drive+":/WINDOWS/System/amsacc.dll");
            if (!file.exists()){
              this.put(file,"windows/amsacc.dll");
            }
          }
        }
      } else {
	for (i=0; i<ndrives; i++) {
	  drive = drives[i];   
          dir = new File(drive+":/WINNT/System32");
          if (dir.exists()) {
            file = new File(drive+":/WINNT/System32/amsacc.dll");
            if (!file.exists()){
		;/*              this.put(file,"windows/amsacc.dll"); */
            }
          }
        }
      }
    }
  }

  /*
    Put the library in the appropriate location
  */
  public void put(File file,String system) throws java.lang.SecurityException {
    System.out.println("PutAMSACC: Installing "+system+" in "+file);
    try {
      URL          url    = new URL("http://www-unix.mcs.anl.gov/petsc/bin/"+system);
      InputStream  in     = url.openStream();
      file.createNewFile();
      OutputStream out    = new FileOutputStream(file);
   
      byte[] data = new byte[1024];
      int    fd   = 0;
      while (true) {
        fd = in.read(data);
        if (fd == -1) break;
        out.write(data,0,fd);
      }
      out.close();

    } catch (MalformedURLException oops) {;}
      catch (java.io.IOException oops) {;}
    System.out.println("PutAMSACC: Installed "+system+" in "+file);
  }
}




