/*$Id: PETScView.java,v 1.9 2001/02/19 23:05:30 bsmith Exp bsmith $*/
/*
     Accesses the PETSc published objects
*/

/*  These are the Java GUI classes */
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.tree.*;

/* For the text input regions */
import javax.swing.text.*;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.Locale;
import java.util.*;

/*   This allows multiple threads */
import java.lang.Thread;

/*  These are the AMS API classes */
import gov.anl.mcs.ams.*;
import java.security.*;

import java.net.*;

/* for XY plots */
import ptolemy.plot.*;

/*
    This is the class that this file implements (must always be the same as
  the filename).

    Applet is a subclass of PanelFrame, i.e. it is itself the base window we draw onto
*/
public class PETScView extends JApplet {
    
  /*  AMSBean Object - this is where all the AMS "global" functions and 
                       "enum" types are stored                            */
  AMSBean amsbean;

  /* Current PETSc communicator and memory; number of option sequences set */
  String     petsccomm;
  AMS_Comm   ams;
  AMS_Memory mem;
  int        count = 0;
  Container japplet;

  String     host = "terra.mcs.anl.gov";
  int        port = 9000;

  java.applet.AppletContext appletcontext;
  JApplet applet;

  JTextField inputport;
  JTextField inputserver;

  Hashtable memories;

  /*
    This is the destructor;
  */
  public void destroy(){
        System.out.println("destroy called");
  }

  public void init(){
    System.out.println("init called");

    System.out.println("PETScOptions: codebase:"+this.getDocumentBase()+":");
    System.out.println("PETScOptions: about to load amsacc library ");

    try {
      amsbean = new AMSBean() {
        public void print_error(String mess) {  /* overwrite the error message output*/
          System.out.println("AMS Error Message: "+mess);
        }
      };
    } catch (UnsatisfiedLinkError oops) {
      try {
        this.getAppletContext().showDocument(new URL("http://www.mcs.anl.gov/petsc/plugins-amsacc.html"));
      } catch (java.net.MalformedURLException ex) {;}
    } catch (AccessControlException oops) {
      try {
        this.getAppletContext().showDocument(new URL("http://www.mcs.anl.gov/petsc/plugins-security.html"));
      } catch (java.net.MalformedURLException ex) {;}
    } catch (ExceptionInInitializerError oops) {
      try {
        this.getAppletContext().showDocument(new URL("http://www.mcs.anl.gov/petsc/plugins-security.html"));
      } catch (java.net.MalformedURLException ex) {;}
    } 
    System.out.println("PETScOptions: done loading amsacc library ");

    appletcontext = this.getAppletContext();
    applet        = this;
    japplet       = this.getContentPane();
    memories = new Hashtable();
  }

  public String getAppletInfo() {
    return "Set PETSc obtions via the AMS";
  }

  public void stop() {
        System.out.println("Called stop");
  }

  /*
       This is called by the applet and is much like a main() program, except that 
    if other threads exist the applet does not end when this routine ends.
  */
  public void start() { /* ------------------------------------------*/
     getserver();
  }
    
  public class JPanelSimplePack extends JPanel { /*-----------------------------------*/
    public JPanelSimplePack(String text,Component c1) {
      super( new GridBagLayout());
      add(new Label(text));
      add(c1);
    }
  }

  public void getserver() { /* ------------------------------------------*/

    japplet.removeAll();
    japplet.setVisible(false);
    /*

         Make GUI to get host and port number from user 
    */
    japplet.setLayout(new FlowLayout());
        
    JPanel tpanel = new JPanel(new GridLayout(3,1));
    japplet.add(tpanel, BorderLayout.NORTH);
        
    inputserver = new JTextField(host,32);
    JPanelSimplePack text = new JPanelSimplePack("AMS Client machine",inputserver);
    tpanel.add(text);
    inputport = new JTextField(port+"",8);
    text = new JPanelSimplePack("AMS Client port",inputport);
    tpanel.add(text);
    System.out.println("put up server and port");
    
    /*--------------------- */
    JButton button = new JButton("Continue");
    tpanel.add(button);
    button.addActionListener(new ActionListener(){
      public void actionPerformed(ActionEvent e) { 
        System.out.println("User selected continue");
        connect();
      }
    }); 
    System.out.println("put up continue");


    japplet.setVisible(true);
    japplet.validate(); 
    japplet.repaint(); 
    System.out.println("put up continue done");
    return;
  }

  /*
      Connect to the PETSc program and display the first set of options
  */
  public void connect() { /* ------------------------------------------*/

    System.out.println("in connect");

    host = inputserver.getText();
    port = (new Integer(inputport.getText())).intValue();
        
    japplet.removeAll();
    japplet.setVisible(false);

    /* Get list of communicators */
    String list[] = AMSBean.get_comm_list(host,port);
    if (list == null) {
      System.out.println("Unable to connect to publisher on "+host+" "+port);
      getserver();
      return;
    }

    /* look for PETSc communicators */
    int i;
    for (i=0; i<list.length; i++) {
      if ((list[i].substring(0,5)).equals("PETSc")) {
        break;
      }
    }
        
    if (i == list.length) {
      System.out.println("Publisher does not have PETSc communicator. Communicator has");
        for (i=0; i<list.length; i++) {
          System.out.println(list[i]);
        }
      getserver();
      return;
    }
    petsccomm = list[i];          

    /* Attach to the PETSc Communicator */
    ams = AMSBean.get_comm(petsccomm);

    if (ams == null) {
      System.out.println("Could not get communicator:"+petsccomm);
      getserver();
      return;
    }
    displayoptionsset();
  }
    
  JDesktopPane rpanel;

  /*
        Displays the objects
  */
  public void displayoptionsset() { /*---------------------------------------------*/
    /*
            Clear the window of old options
    */
    System.out.println("About to remove panels");    
    japplet.removeAll();
    japplet.setVisible(false);
    System.out.println("Removed panel; trying to get objects");    

    japplet.setLayout(new BorderLayout());
        


    /* Create a Panel that  will contain two buttons. Default layout manager */
    JPanel bpanel = new JPanel(new GridLayout(1,4));
        
    JButton button = new JButton("Refresh");
    bpanel.add(button);
    button.addActionListener(new RefreshActionListener());

    JButton qbutton = new JButton("Quit GUI");
    bpanel.add(qbutton);
    qbutton.addActionListener(new QuitActionListener());
        
    /* Add the Panel in the top of the Frame */
    japplet.add(bpanel, BorderLayout.NORTH);
        
    /* Add panel to right for displaying output */
    rpanel = new JDesktopPane();
    rpanel.setVisible(true);
    /*   rpanel.setMinimumSize(new Dimension(300,400)); */
    japplet.add(rpanel, BorderLayout.CENTER);

    /* Get the memorys (we ignore the rest) */
    String mems[] = ams.get_memory_list();
    int i, cnt = 1;

    if (mems == null) {
      getserver();
      return;
    } 

    /* count total number of object ids */
    for (i=0; i<mems.length; i++) {
      mem = ams.get_memory(mems[i]);
      memories.put(mems[i],mem);     /* keep memories in hash table so do not have to access remote system for memories */

      String flist[] = mem.get_field_list();
      if (!flist[0].equals("Class")) continue; /* Not a published PETSc object */
      int id = mem.get_field("Id").getIntData()[0];
      cnt = id > cnt ? id : cnt;
    }

    DefaultMutableTreeNode root = new DefaultMutableTreeNode(),nodes[] = new DefaultMutableTreeNode[cnt+1];

    /* create treenode for each published object */
    for (i=0; i<mems.length; i++) {
      mem = (AMS_Memory) memories.get(mems[i]);
      String flist[] = mem.get_field_list();
      if (!flist[0].equals("Class")) continue; /* Not a published PETSc object */
      int id = mem.get_field("Id").getIntData()[0];
      nodes[id] = new DefaultMutableTreeNode(mems[i]);
    }

    /* mark parents of each tree node */
    for (i=0; i<mems.length; i++) {
      mem = (AMS_Memory) memories.get(mems[i]);
      String flist[] = mem.get_field_list();
      if (!flist[0].equals("Class")) continue; /* Not a published PETSc object */
      int parent = mem.get_field("ParentId").getIntData()[0];
      int id = mem.get_field("Id").getIntData()[0];
         System.out.println("me "+id+" parant "+parent);    
      if (parent > 0 && nodes[parent] != null) {
        nodes[parent].add(nodes[id]);
        nodes[id].setParent(nodes[parent]);
      } else {
        root.add(nodes[id]);
        nodes[id].setParent(root);
      }
      int j;
      for (j=0; j<flist.length; j++) {
         System.out.println("mem "+mems[i]+" field "+flist[j]);    
      }

    }
    
    DefaultTreeModel tree = new DefaultTreeModel(root);
    JTree jtree = new JTree(tree);
    jtree.addTreeSelectionListener(new TreeSelectionListener() {
      public void valueChanged(TreeSelectionEvent e) {
        System.out.println("User selected tree node"); 
	System.out.println("node "+e.getPath().getLastPathComponent());
        Object obj = ((DefaultMutableTreeNode)(e.getPath().getLastPathComponent())).getUserObject();
	AMS_Memory thismem = (AMS_Memory) memories.get((String)obj);
	System.out.println("thismem "+thismem);
        JInternalFrame jp;
        String oclass = mem.get_field("Class").getStringData()[0];
        if (oclass.equals("KSP")) {
          jp = new PETScViewKSP(thismem);
        } else if (oclass.equals("SNES")) {
          jp = new PETScViewSNES(thismem);
        } else return;

        rpanel.add(jp);
	/*        jp.setLocation(50,50); */
      }
    });
    japplet.add(new JScrollPane(jtree), BorderLayout.WEST); 
    jtree.setCellRenderer(new MyTreeCellRenderer());
    jtree.setRowHeight(15);
    jtree.setPreferredSize(new Dimension(300,550));


    System.out.println("Processed options set");    
    japplet.setVisible(true);
    japplet.validate(); 
    japplet.repaint(); 
    System.out.println("Repainted");    
 }
    
    
  /*
      These are INNER classes; they provide callbacks to the window buttons, textedits, etc
  */
  /* callback for the quit button */
  class QuitActionListener implements ActionListener {/*------------------------*/
    public void actionPerformed(ActionEvent e) {
      System.out.println("User selected quit");
      getserver();
    }
  }

  /* callback for the refresh button */
  class RefreshActionListener implements ActionListener {/*--------------------*/
    public void actionPerformed(ActionEvent e) {
      japplet.removeAll();
      japplet.setVisible(false);
      System.out.println("User selected refresh");
      (new ThreadOptionUpdate()).start();
    }
  }

  class MyTreeCellRenderer implements TreeCellRenderer 
  {
    public Component getTreeCellRendererComponent(JTree tree,Object value,boolean selected,
                                                    boolean expanded, boolean leaf, int row,
                                                    boolean hasfocus) {
      Object obj = ((DefaultMutableTreeNode)value).getUserObject();
      if (obj != null) {
        String     smem = (String)obj;
        AMS_Memory mem = (AMS_Memory) memories.get(smem);

        String     label = "";
        if (mem.get_field("Type").getStringData()[0] != null) {
          label += mem.get_field("Type").getStringData()[0]+" ";
        }
        label += mem.get_field("Class").getStringData()[0];
        if (smem.indexOf("n_") != 0) {
	  label += " "+smem;
        }
        return new JLabel(label);
      } else {
	return new JLabel("");
      }
    }
  }

  /*
     Methods used by the callbacks (inner classes)
  */

  class ThreadOptionUpdate extends Thread {/*-----------------------------------*/
    public void run() {
      displayoptionsset(); 
    }
  }
}









