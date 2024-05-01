import serial
import cv2
import numpy as np
import threading
import queue
import sys # receive execution arguments

# Global variables
_DEBUG_RECEIVE = False # debug flag, set false to turn off debug

mtx_data_queue = queue.Queue()  # Queue for passing matrix data between threads
mtx_displayed = {}  # Keep track of displayed matrices and their window names

def receive_matrix_data(port):
    ser = serial.Serial(port, 115200) 
    while True:
        line = ser.readline().strip()  # Read binary data
        if line.startswith(b'~txBEG~'):
            print("Receiving matrix...")
            metadata = line.split(b'{')[1].split(b'}')[0]
            meta_dict = {}
            for pair in metadata.split(b','):
                key, value = pair.split(b':')
                meta_dict[key.strip()] = value.strip()

            # print("Metadata:", meta_dict)

            mtxID = meta_dict[b'ID']
            rows = int(meta_dict[b'Y'])
            cols = int(meta_dict[b'X'])
            dtype = meta_dict[b'dt']

            print("metadata:{{ID:{} {}cols X {}rows of type {}}}".format(mtxID, cols, rows, dtype))

            matrix_data = b""
            while True:
                line = ser.readline().strip()
                if _DEBUG_RECEIVE:
                    print("Received line:", line)  
                if line == b"~txEND~" or line == b"~txERR~":
                    break
                matrix_data += line

            matrix_flat = matrix_data.split(b';') 
            matrix = np.zeros((rows, cols), dtype=np.float32) # if dtype == b'f' else np.float32)
            for i, row in enumerate(matrix_flat):
                if row.strip() != b'':
                    row_data = row.split(b',')
                    for j, val in enumerate(row_data):
                        try:
                            matrix[i, j] = float(val) # if dtype == b'f' else float(val)
                        except ValueError as e:
                            print("Error: Matrix transfer interrupted or invalid matrix datatype:", val)

            mtx_data_queue.put((mtxID, matrix))  # Put the received matrix in the queue

        # elif line == b'~txERR~':
            # print("Error: Matrix transfer interrupted or invalid, received ~txERR~ tag from sender.")
        # elif line == b'~txEND~':
            # if _DEBUG_RECEIVE:
                # print("End of matrix transfer tag ~txEND~ recieved without ~txBEG~ begin tag.")
        else:
            # Pass through any other received serial commands directly to terminal
            print("DEVICE SAYS:", line)

def display_matrix(matrix_id, matrix):
    try:
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        if min_val < 0 or max_val > 255:
            scaled_matrix = cv2.normalize(matrix, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            scaled_matrix = matrix.astype(np.uint8)
        
        # COLORMAP_AUTUMN red to yellow
        # COLORMAP_BONE  black to white
        # COLORMAP_JET blue to green to red
        # COLORMAP_WINTER  blue to cyan
        # COLORMAP_RAINBOW  spectrum
        # COLORMAP_OCEAN  white to blue
        # COLORMAP_SUMMER  green to yellow
        img = cv2.applyColorMap(scaled_matrix, cv2.COLORMAP_JET)
        # target px width for rescale
        targetWidth_px = 640 
        scaleFactor = targetWidth_px / img.shape[1] # scale factor required to achieve target
        # calculate new dimensions
        width = int(img.shape[1] * scaleFactor) # target 640 width
        height = int(img.shape[0] * scaleFactor)
        resize_dim = (width, height)
        img_resize = cv2.resize(img, resize_dim, interpolation = cv2.INTER_NEAREST)
        
        # mtxID_str = mtxID.decode('utf-8')  # Decode byte string to regular string
        if matrix_id in mtx_displayed:
            cv2.imshow(mtx_displayed[matrix_id], img_resize)
        else:
            cv2.imshow(matrix_id, img_resize)
            mtx_displayed[matrix_id] = matrix_id
        cv2.waitKey(1)
    except queue.Empty:
        pass

def main(port):
    
    # Start receiving matrix data in a separate thread
    receive_thread = threading.Thread(target=receive_matrix_data, args=(port,), daemon=True)
    receive_thread.start()

    while True:
        try:
            mtxID, matrix = mtx_data_queue.get(timeout=1)  # Get the oldest matrix from the queue
            mtxID_str = mtxID.decode('utf-8')  # Decode byte string to regular string
            display_matrix(mtxID_str, matrix)
            mtx_data_queue.task_done()  # Mark the item as processed
        except queue.Empty:
            pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        port = sys.argv[1]
        # Check if "COM" prefix was provided
        if port[:3].upper() != "COM":
            # Add "COM" prefix
            port = "COM" + port
    else:
        # Set default COM port if one was not provided at code execution
        port = "COM4"
    main(port)