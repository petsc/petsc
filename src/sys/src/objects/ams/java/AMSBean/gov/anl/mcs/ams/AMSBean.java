package gov.anl.mcs.ams;

import java.io.Serializable;
import java.util.Vector;
import java.security.*;


/**
 * A AMSBean JavaBean. This module interfaces with the AMS JNI API
 *
 * @version 1.00
 * @author Ibrahima Ba
 */
 
public class AMSBean extends Object
implements java.io.Serializable {

    // Static initialization for the AMSBean Class
    /* static {
        System.loadLibrary("amsacc");
	} */

    /**
     * Constructs a new AMSBean JavaBean.
     */
    public AMSBean () throws UnsatisfiedLinkError, AccessControlException, ExceptionInInitializerError{
        System.loadLibrary("amsacc");
        AMS_Java_init(); // Call non-native method to initialize static vars
    }

    // Class variables
    static public int READ, WRITE, MEMORY_UNDEF;
    static public int INT, BOOLEAN, DOUBLE, FLOAT, STRING, DATA_UNDEF;
    static public int COMMON, REDUCED, DISTRIBUTED, SHARED_UNDEF;
    static public int SUM, MAX, MIN, REDUCT_UNDEF;

    /**
     * This function is called by the native to pass the error
     * message as a parameter. Please override this function to handle 
     * the output for the last error code message.
     */
    public void print_error(String msg)
    {        
    }

    // Native Methods Interface

    /**
     * Initializes static variables from DLL/Shared Lib 
     */
    private native int AMS_Java_init();

    /**
     * Static native function. Do not depend on an object.
     * Get the list of AMS Communicators
     * 
     * @param host hostname to connect to. Use "" to use the default host
     * @param port port number to connect to. Use -1 for default port
     *
     * @return list of communicators
     */
     
    static public native String [] get_comm_list(String host, int port);

    /**
     * Static native function. Do not depend on an object.
     * Get a specific Communicator given its name
     *
     * @param name name of the communicator to connect to
     *
     * @return an AMS Communicator object
     */

    static public native AMS_Comm get_comm(String name);


    /**
     * Get an explanation for an error code
     *
     * @param err error code to be explained
     */ 
    static public native String explain_error(int err);

    /**
     * Set output file name to redirect from stderr. 
     */ 
    static public native void set_output_file(String file);
    
}
