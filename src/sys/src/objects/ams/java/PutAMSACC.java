/*$Id: PumpStream.java,v 1.2 2000/10/25 19:11:52 bsmith Exp bsmith $*/
/*
     Connects two streams together. Simply takes all input from first and
  passes it into second. This allows us to move stdout etc. from the server 
  back to the client
*/

import java.lang.Thread;
import java.io.*;

public class PumpStream {
  InputStream  in,sockin;
  OutputStream out;
  Process      process;
  
  /*
     The constructor for the class.
  */
  public PumpStream(InputStream i,OutputStream o,InputStream si,Process p)
  {
    in      = i;
    out     = o;
    sockin  = si;
    process = p;
  }

  public int Pump() throws java.io.IOException
  {
    (new Thread() {
       public void run() {
         System.out.println("PETScRund: open killing thread");
         byte[] data = new byte[1024];
         int    fd   = 0;
         try {
           fd = sockin.read(data);
           if (fd > 0) {
             System.out.println("PETScRund: killing job");
             process.destroy();
             return;
           }
         } catch (java.io.IOException oops) {;}
       }
    }).start();


    byte[] data = new byte[1024];
    int    fd   = 0,cnt = 0;
    while (true) {
      fd = in.read(data);
      if (fd == -1) break;
      cnt += fd;
      out.write(data,0,fd);
    }
    out.close();
    return cnt;
  }
}




