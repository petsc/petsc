/*$Id: amsoptions.java,v 1.11 2000/03/27 00:39:23 bsmith Exp $*/
/*

     Accesses the PETSc published database options and allows the user to change them via a GUI
*/
/*  These are the AMS API classes */
import gov.anl.mcs.ams.*;

/*  These are the Java GUI classes */
import java.awt.*;
import javax.swing.*;
import java.awt.event.*;

/*   This allows multiple threads */
import java.lang.Thread;


public class amsoptions extends Object
{
  /*  AMSBean Object - this is where all the AMS "global" functions and "enum" types are stored*/
  AMSBean amsbean;

  /* Window where the GUI is drawn */
  JFrame   frame;

  /* Current PETSc communicator and memory; number of option sequences set */
  String     petsccomm;
  AMS_Comm   ams;
  AMS_Memory mem;
  int        count = 0;

  /* This is the constructor for the amsoptions object; all it does is create the AMSBean object */
  public amsoptions() { /*----------------------------------------------------------------------*/
    amsbean = new AMSBean();
  } /*-----------------------------------------------------------------------------------------------*/

  /* this is the "main" program; it creates an amsoptions object and calls the method run() on it */    
  static public void main(String args[]) { /*--------------------------------------------------------*/
    (new amsoptions()).run(args);
  }  /*-----------------------------------------------------------------------------------------------*/

  /* this is the "main subroutine" it attaches to the default communicator and ... */
  public void run(String args[]) { /*-----------------------------------------------------------------*/

    /* Process -ams_server and -ams_port command line options; else use defaults */
    int    i, port = -1;
    String host = "";

    for (i=0; i<args.length; i++) {
      if (args[i].equals("-ams_server")) {
        if (i == args.length-1) {
          System.out.println("You did not include the server name after the -ams_server option");
          System.exit(1);
        }
        host = args[i+1];
      }
      if (args[i].equals("-ams_port")) {
        if (i == args.length-1) {
          System.out.println("You did not include the port number after the -ams_port option");
          System.exit(1);
        }
        port = Integer.parseInt(args[i+1]);
      }
    }

    /* Get list of communicators */
    String list[] = AMSBean.get_comm_list(host,port);

    /* look for PETSc communicators */
    for (i=0; i<list.length; i++) {
      if ((list[i].substring(0,5)).equals("PETSc")) {
        break;
      }
    }
    if (i == list.length) {
      System.out.println("Publisher does not have PETSc communicator. Communicators it does have");
      for (i=0; i<list.length; i++) {
        System.out.println(list[i]);
      }
      System.exit(1);
    }
    petsccomm = list[i];          

    /*    AMSBean.set_output_file("/dev/null");  */

    /* Attach to the PETSc Communicator */
    ams = AMSBean.get_comm(petsccomm);

    if (ams == null) {
      System.out.println("Could not get communicator:"+petsccomm);
      System.exit(1);
    }


    displayoptionsset(false);
  }  /*-----------------------------------------------------------------------------------------------*/

  /*
           This method displays a single set of options and allows the user to change them
  */
  public void displayoptionsset(boolean putquit) { /*-------------------------------------------------*/
    
    /*  GUI Window for options */
    JPanel   panel;
    JButton button;

    /* these are for the exit button that comes between each select frame */
    JFrame   qframe = null;
    JPanel   qpanel;

    /* 
           Clear the window and put up a temporary exit until we get next 
         options from PETSc program. Skip this first time for faster startup
    */
    if (putquit) {
        System.out.println("putting up tmp exit");
      qframe = new JFrame("PETSc AMS Options Setter");
      qframe.getAccessibleContext().setAccessibleDescription("PETSc AMS Options Setter");
      JOptionPane.setRootFrame(qframe);
      qframe.setBackground(Color.white);
      qframe.getContentPane().setLayout(new BorderLayout());
      qframe.setSize(800,800);
      qframe.setLocation(10,10);

      qpanel = new JPanel();
      qpanel.setLayout(new GridLayout(0,1));
      qframe.getContentPane().add(qpanel,BorderLayout.CENTER);

      button = new JButton("Exit");
      button.addActionListener(new QuitActionListener());
      qpanel.add(button);

      qframe.pack();
      qframe.show();
    }

    String options = "Options_"+count++;
    /* Get the options memory (we ignore the rest) */
        System.out.println("trying to get mem");
    mem = ams.get_memory(options);
    while (mem == null) {
      try {Thread.sleep(300);} catch (InterruptedException ex) {;} finally {;} 
      mem = ams.get_memory(options);
    }

    /* get rid of the temporary exit window */
    if (putquit) {
      qframe.dispose();
    }

    /* Create the GUI window */
    frame = new JFrame("PETSc AMS Options Setter");
    frame.getAccessibleContext().setAccessibleDescription("PETSc AMS Options Setter");
    JOptionPane.setRootFrame(frame);
    frame.setBackground(Color.white);
    frame.getContentPane().setLayout(new BorderLayout());
    frame.setSize(800,800);
    frame.setLocation(10,10);

    panel = new JPanel();
    panel.setLayout(new GridLayout(0,1));
    frame.getContentPane().add(panel,BorderLayout.CENTER);

    String flist[] = mem.get_field_list();

    /* first field is always the name of the options being set */
    String OptionsCategory = flist[0];

    /* Loop over the rest of the fields */
    int i;
    for (i=1; i<flist.length; i++) {
      AMS_Field lockfld    = mem.get_field(flist[i]);
      AMS_Field fld        = mem.get_field(flist[i+1]);
      int           info[] = fld.get_field_info();

      /* handle OptionsSelectInt() */
      if (info[1] == AMSBean.INT) {
        int value[] = fld.getIntData();
        System.out.println("Int: "+flist[i]+" "+flist[i+1]+" "+value[0]);
        Label label = new Label("Int: "+flist[i]+" "+flist[i+1]+" "+value[0]);
        panel.add(label);

      /* handle OptionsSelectDouble() */
      } else if (info[1] == AMSBean.DOUBLE) {
        double value[] = fld.getDoubleData();
        System.out.println("Double "+flist[i]+" "+flist[i+1]+" "+value[0]);
        Label label = new Label("Double: "+flist[i]+" "+flist[i+1]+" "+value[0]);
        panel.add(label);

      } else if (info[1] == AMSBean.STRING) {
        String value[] = fld.getStringData();

          System.out.println("fluck"+value[0]+flist[i+1]);

        /* handle OptionsSelectList() */
        if ((flist[i+1].substring(0,8)).equals("DEFAULT:")) {
          System.out.println("List "+flist[i]+" "+flist[i+1]+" "+value[0]);

          int       j;
          AMS_Field lfld    = mem.get_field(flist[i+2]);
          String    llist[] = lfld.getStringData();

          for (j=0; j<llist.length; j++) {
            System.out.println(" "+llist[j]);
          }
          i++;

        /* handle OptionsSelectString() */
        } else {
          System.out.println("String "+flist[i]+" "+flist[i+1]+" "+value[0]);
          Label label = new Label("String: "+flist[i]+" "+flist[i+1]+" "+value[0]);
          panel.add(label);
        }
      }
      i++;
    }

    JPanel buttonpanel = new JPanel();
    buttonpanel.setLayout(new GridLayout(1,0));

    button = new JButton("Continue");
    button.addActionListener(new ContinueActionListener());
    buttonpanel.add(button);

    button = new JButton("Quit");
    button.addActionListener(new QuitActionListener());
    buttonpanel.add(button);

    panel.add(buttonpanel);

    /* show the choices to the user */        
    /*    frame.setSize(800,800);*/
    frame.pack();
    frame.show();
  }  

  public void displayoptionsupdate() { /*------------------------------------------------------------*/

    /* Send values over to PETSc program  */
    mem.send_begin();
    mem.send_end(); 

    /*    Tell PETSc we are done setting these options */
    mem.unlock();
  }

  /* =====================================================================*/
  class QuitActionListener implements ActionListener {
    public void actionPerformed(ActionEvent e) {
      System.out.println("User selected quit");
      System.exit(1);
    }
  }

  class ContinueActionListener implements ActionListener {
    public void actionPerformed(ActionEvent e) {
      frame.dispose();
      System.out.println("User selected continue");
      displayoptionsupdate();   /* update options on PETSc program */
      displayoptionsset(true);      /* wait for next set of options from PETSc program */
    }
  }
}


