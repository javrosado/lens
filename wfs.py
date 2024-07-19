import os
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
import time
#import ctypes
#from ctypes import byref, c_double, c_float, c_int64, c_uint64, create_string_buffer, c_bool, c_char_p, c_uint32, c_int32, c_ulong, CDLL, cdll, sizeof, windll, c_long, Array, c_char
from ctypes import *

lib = cdll.LoadLibrary("C:\Program Files\IVI Foundation\VISA\Win64\Bin\WFS_64.dll")

#values requested from the dll are passed by reference, this gets thr number of connects WFS devices
num_devices = c_int32()
lib.WFS_GetInstrumentListLen(None,byref(num_devices))

#if no devices connected, close the program
if num_devices.value == 0:
    print("No availble devices.... closing program")
    quit()

#get connection information for first available WFS device
device_id = c_int32()
device_in_use = c_int32() 
device_name = create_string_buffer(20)
serial_number = create_string_buffer(20)
resource_name = create_string_buffer(30)

lib.WFS_GetInstrumentListInfo(None, 0, byref(device_id), byref(device_in_use), device_name, serial_number, resource_name)

#check if WFS is in use, if not, connect to device
if device_in_use:
    print("Wavefront sensor currently in use.... closing program")
    quit()

instrument_handle = c_ulong()
lib.WFS_init(resource_name, c_bool(False), c_bool(True), byref(instrument_handle))
print(f"Connected to {device_name.value} with Serial Number {serial_number.value}")

#Get the number of calibrated microlens arrays and print out data
mla_count = c_int32()
lib.WFS_GetMlaCount(instrument_handle, byref(mla_count))

mla_name = create_string_buffer(20)
cam_pitch = c_double()
lenslet_pitch = c_double()
spot_offset_x = c_double()
spot_offset_y = c_double()
lenslet_focal_length = c_double()
astigmatism_correction_0 = c_double()
astigmatism_correction_45 = c_double()

print("Available Microlens Arrays: ")
for i in range(mla_count.value):
    lib.WFS_GetMlaData(instrument_handle,i ,mla_name, byref(cam_pitch), byref(lenslet_pitch), byref(spot_offset_x), 
        byref(spot_offset_y), byref(lenslet_focal_length), byref(astigmatism_correction_0), byref(astigmatism_correction_45))
    print(f"Index: {i} - MLA Name: {mla_name.value} with lenslet pitch {lenslet_pitch.value}")

#select MLA    
lib.WFS_SelectMla(instrument_handle, 0)


#configure cam resolution and pixel format. Method outputs number of spots in the X and Y for selected MLA
# PIXEL_FORMAT_MONO8 = 0
CAM_RES_WFS20_768 = 2
#Full lists of available sensor reolutions are in the WFS.h header file in C:\Program Files (x86)\IVI Foundation\VISA\WinNT\Include
num_spots_x = c_int32()
num_spots_y = c_int32()
lib.WFS_ConfigureCam(instrument_handle, c_int32(0), c_int32(0), byref(num_spots_x), byref(num_spots_y))
print(f"Number of detectable spots in X: {num_spots_x.value} \nNumber of detectable spots in Y: {num_spots_y.value}")

#set WFS internal reference plane
#other user-defined reference planes can be configured in the WFS software. These are saved to a .ref file and are accessed by passing a 1 instead of 0
lib.WFS_SetReferencePlane(instrument_handle, c_int32(0))
lib.WFS_SetPupil(instrument_handle, c_double(0.0), c_double(0.0), c_double(2.0), c_double(2.0))

#Take a series of images until one is usable. Check the device status after each image to determine usability
actual_exposure = c_double()
actual_gain = c_double()
device_status = c_int32()

print(lib.WFS_SetExposureTime(instrument_handle, c_double(0.01), byref(actual_exposure)))
print(actual_exposure)



def update(frame, img_plot):

    lib.WFS_TakeSpotfieldImage(instrument_handle)

    imgBuf = np.zeros((1080,1440), dtype= np.ubyte)  

    rows = c_int32()
    cols = c_int32()
    lib.WFS_GetSpotfieldImageCopy(instrument_handle, imgBuf.ctypes.data_as(POINTER(c_char)), byref(rows), byref(cols))
    img_plot.set_array(imgBuf)
    return (img_plot,)


def threadfunc():
    time.sleep(5)
    print("hi")
    

threading.Thread(target=threadfunc).start()

fig, ax = plt.subplots()
img_plot = ax.imshow(np.random.rand(1080, 1440), cmap='gray')

ani = animation.FuncAnimation(fig, update, interval=100, fargs=(img_plot,))
plt.show()

