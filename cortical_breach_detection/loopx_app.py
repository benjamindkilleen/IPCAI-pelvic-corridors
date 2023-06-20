from datetime import datetime
from multiprocessing.sharedctypes import Value
import threading
import multiprocessing as mp
import time
import tkinter as tk
import logging
from queue import Empty
from typing import Dict, Optional
import numpy as np


from .loopx import DEGREE_SIGN
from .procs import DRREngine

log = logging.getLogger(__name__)


def maybe_float(input_str: str) -> Optional[float]:
    """
    Parse the input string into a float.

    Args:
        input_str (str): The input string to parse.

    Returns:
        float: The parsed float.

    """
    if input_str == "":
        return None

    try:
        return float(input_str)
    except ValueError:
        log.warning(f"Could not parse {input_str} as a float.")
        return None
    

def maybe_transform(input_str: str) -> Optional[np.ndarray]:
    """
    Parse the input string into a string.

    Args:
        input_str (str): The input string to parse.

    Returns:
        str: The parsed string.

    """
    if input_str == "":
        return None
    else:
        return np.fromstring(input_str, sep=" ").reshape(4, 4)


def to_isoformat(timestamp: str) -> Optional[str]:
    """
    Convert a timestamp to an ISO format string.

    Args:
        timestamp (float): The timestamp to convert. Accept either isoformat or 

    Returns:
        str: The ISO format string.

    """
    t = None
    format_strings = [
        "%Y-%m-%d %H:%M:%S.%f", 
        "%Y-%m-%d %H:%M:%S",
        "%Y%m%d %H%M%S.%f",
        "%Y%m%d %H%M%S",
        "%Y%m%d%H%M%S.%f",
        "%Y%m%d%H%M%S",
        "%Y%m%d_%H%M%S",
    ]
    for format_string in format_strings:
        try:
            t = datetime.strptime(timestamp, format_string)
            break
        except ValueError:
            pass
  
    if t is None:
        return None
    else:
        return t.isoformat(sep=" ")


def start_loopx_window(queue: mp.Queue, corridor_server_queue: mp.Queue):
    """Blocking call. Open GUI to get the gantry pose.

    Args:
        corridor_server_queue: The queue to send messsages to the CorridorServer.
    """
    root = tk.Tk()
    r = 0

    ct_timestamp_var = tk.StringVar()
    tk.Label(root, text="CT Time:").grid(row=r, column=0)
    tk.Entry(root, textvariable=ct_timestamp_var).grid(row=r, column=1)
    def update_ct_timestamp():
        ct_timestamp_raw = ct_timestamp_var.get()
        ct_timestamp = to_isoformat(ct_timestamp_raw)
        if ct_timestamp is None:
            log.warning(f"Invalid device time: {ct_timestamp_raw}")
            return

        corridor_server_queue.put(("ct_timestamp", ct_timestamp))
        t0 = time.time()
        while True:
            if time.time() - t0 > 10:
                log.info("Waiting for first image analysis.")
            response: Optional[Dict[str, float]] = queue.get()
            if response is None:
                time.sleep(0.1)
                continue
            else:
                break

        if "success" not in response or not response["success"]:
            # If there was an error, clear the GUI
            ct_timestamp_var.set("")

    tk.Button(root, text="Update", command=update_ct_timestamp).grid(row=r, column=2)
    r += 1

    tk.Label(root, text="Image 1").grid(row=r, column=1)
    tk.Label(root, text="Image 2").grid(row=r, column=2)
    r += 1


    paste_column_1_var = tk.StringVar()
    paste_column_2_var = tk.StringVar()
    tk.Label(root, text="Paste column:").grid(row=r, column=0)
    tk.Entry(root, textvariable=paste_column_1_var).grid(row=r, column=1)
    tk.Entry(root, textvariable=paste_column_2_var).grid(row=r, column=2)
    r += 1

    device_time_1_var = tk.StringVar()
    device_time_2_var = tk.StringVar()
    tk.Label(root, text="Device Time:").grid(row=r + 0, column=0)
    tk.Entry(root, textvariable=device_time_1_var).grid(row=r + 0, column=1)
    tk.Entry(root, textvariable=device_time_2_var).grid(row=r + 0, column=2)
    r += 1

    source_angle_1_var = tk.StringVar()
    detector_angle_1_var = tk.StringVar()
    lateral_1_var = tk.StringVar()
    longitudinal_1_var = tk.StringVar()
    traction_yaw_1_var = tk.StringVar()
    gantry_tilt_1_var = tk.StringVar()
    source_angle_2_var = tk.StringVar()
    detector_angle_2_var = tk.StringVar()
    lateral_2_var = tk.StringVar()
    longitudinal_2_var = tk.StringVar()
    traction_yaw_2_var = tk.StringVar()
    gantry_tilt_2_var = tk.StringVar()
    tk.Label(root, text=f"Source Angle ({DEGREE_SIGN}):").grid(row=r + 0, column=0)
    tk.Label(root, text=f"Detector Angle ({DEGREE_SIGN}):").grid(row=r + 1, column=0)
    tk.Label(root, text="Lateral (cm):").grid(row=r + 2, column=0)
    tk.Label(root, text="Longitudinal (cm):").grid(row=r + 3, column=0)
    tk.Label(root, text=f"Traction Yaw ({DEGREE_SIGN}):").grid(row=r + 4, column=0)
    tk.Label(root, text=f"Gantry Tilt ({DEGREE_SIGN}):").grid(row=r + 5, column=0)
    tk.Entry(root, textvariable=source_angle_1_var).grid(row=r + 0, column=1)
    tk.Entry(root, textvariable=detector_angle_1_var).grid(row=r + 1, column=1)
    tk.Entry(root, textvariable=lateral_1_var).grid(row=r + 2, column=1)
    tk.Entry(root, textvariable=longitudinal_1_var).grid(row=r + 3, column=1)
    tk.Entry(root, textvariable=traction_yaw_1_var).grid(row=r + 4, column=1)
    tk.Entry(root, textvariable=gantry_tilt_1_var).grid(row=r + 5, column=1)
    tk.Entry(root, textvariable=source_angle_2_var).grid(row=r + 0, column=2)
    tk.Entry(root, textvariable=detector_angle_2_var).grid(row=r + 1, column=2)
    tk.Entry(root, textvariable=lateral_2_var).grid(row=r + 2, column=2)
    tk.Entry(root, textvariable=longitudinal_2_var).grid(row=r + 3, column=2)
    tk.Entry(root, textvariable=traction_yaw_2_var).grid(row=r + 4, column=2)
    tk.Entry(root, textvariable=gantry_tilt_2_var).grid(row=r + 5, column=2)
    r += 6

    source_angle_var = tk.StringVar()
    detector_angle_var = tk.StringVar()
    lateral_var = tk.StringVar()
    longitudinal_var = tk.StringVar()
    traction_yaw_var = tk.StringVar()
    gantry_tilt_var = tk.StringVar()

    tk.Label(root, text=f"Source Angle ({DEGREE_SIGN}):").grid(row=r + 0, column=0)
    tk.Label(root, text=f"Detector Angle ({DEGREE_SIGN}):").grid(row=r + 1, column=0)
    tk.Label(root, text=f"Lateral (cm):").grid(row=r + 2, column=0)
    tk.Label(root, text=f"Longitudinal (cm):").grid(row=r + 3, column=0)
    tk.Label(root, text=f"Traction Yaw ({DEGREE_SIGN}):").grid(row=r + 4, column=0)
    tk.Label(root, text=f"Gantry Tilt ({DEGREE_SIGN}):").grid(row=r + 5, column=0)

    tk.Label(root, textvariable=source_angle_var).grid(row=r + 0, column=1)
    tk.Label(root, textvariable=detector_angle_var).grid(row=r + 1, column=1)
    tk.Label(root, textvariable=lateral_var).grid(row=r + 2, column=1)
    tk.Label(root, textvariable=longitudinal_var).grid(row=r + 3, column=1)
    tk.Label(root, textvariable=traction_yaw_var).grid(row=r + 4, column=1)
    tk.Label(root, textvariable=gantry_tilt_var).grid(row=r + 5, column=1)
    r += 6

    def first_image():
        paste_column = paste_column_1_var.get()

        if paste_column:
            pasted_values = [v.strip() for v in paste_column.split("\n")]
            log.info(f"Pasted values: {pasted_values}")
            if len(pasted_values) != 7:
                log.warning("Paste column must have 7 values")
                return

            (
                device_time_raw,
                source_angle_raw,
                detector_angle_raw,
                lateral_raw,
                longitudinal_raw,
                traction_yaw_raw,
                gantry_tilt_raw,
            ) = pasted_values
            device_time_1_var.set(device_time_raw)
            source_angle_1_var.set(source_angle_raw)
            detector_angle_1_var.set(detector_angle_raw)
            lateral_1_var.set(lateral_raw)
            longitudinal_1_var.set(longitudinal_raw)
            traction_yaw_1_var.set(traction_yaw_raw)
            gantry_tilt_1_var.set(gantry_tilt_raw)
        else:
            source_angle_raw = source_angle_1_var.get()
            detector_angle_raw = detector_angle_1_var.get()
            lateral_raw = lateral_1_var.get()
            longitudinal_raw = longitudinal_1_var.get()
            traction_yaw_raw = traction_yaw_1_var.get()
            gantry_tilt_raw = gantry_tilt_1_var.get()
            device_time_raw = device_time_1_var.get()
            paste_column_2_var.set(
                f"{device_time_raw}\n{source_angle_raw}\n{detector_angle_raw}\n{lateral_raw}"
                f"\n{longitudinal_raw}\n{traction_yaw_raw}\n{gantry_tilt_raw}"
            )
        try:
            loopx_pose = dict(
                source_angle=float(source_angle_raw),
                detector_angle=float(detector_angle_raw),
                lateral=float(lateral_raw),
                longitudinal=float(longitudinal_raw),
                traction_yaw=float(traction_yaw_raw),
                gantry_tilt=float(gantry_tilt_raw),
            )
        except ValueError:
            log.warning("Invalid values for LoopX pose")
            return

        device_timestamp = to_isoformat(device_time_raw)
        if device_timestamp is None:
            log.warning(f"Invalid device time: {device_time_raw}")
            return

        corridor_server_queue.put(("first_image", loopx_pose, device_timestamp))

        t0 = time.time()
        while True:
            if time.time() - t0 > 10:
                log.info("Waiting for first image analysis.")
            response: Optional[Dict[str, float]] = queue.get()
            if response is None:
                time.sleep(0.1)
                continue
            else:
                break

        # Pose should be a dictionary with LoopX values (optional) and a "success" key.

        if not response["success"]:
            return

        source_angle_var.set(f"{response['source_angle']:.2f}")
        detector_angle_var.set(f"{response['detector_angle']:.2f}")
        lateral_var.set(f"{response['lateral']:.2f}")
        longitudinal_var.set(f"{response['longitudinal']:.2f}")
        traction_yaw_var.set(f"{response['traction_yaw']:.2f}")
        gantry_tilt_var.set(f"{response['gantry_tilt']:.2f}")

    def second_image():
        paste_column = paste_column_2_var.get()

        if paste_column:
            pasted_values = [v.strip() for v in paste_column.split("\n")]
            log.info(f"Pasted values: {pasted_values}")
            if len(pasted_values) != 7:
                log.warning("Paste column must have 7 values")
                return

            (
                device_time_raw,
                source_angle_raw,
                detector_angle_raw,
                lateral_raw,
                longitudinal_raw,
                traction_yaw_raw,
                gantry_tilt_raw,
            ) = pasted_values
            source_angle_2_var.set(source_angle_raw)
            detector_angle_2_var.set(detector_angle_raw)
            lateral_2_var.set(lateral_raw)
            longitudinal_2_var.set(longitudinal_raw)
            traction_yaw_2_var.set(traction_yaw_raw)
            gantry_tilt_2_var.set(gantry_tilt_raw)
            device_time_2_var.set(device_time_raw)
        else:
            source_angle_raw = source_angle_2_var.get()
            detector_angle_raw = detector_angle_2_var.get()
            lateral_raw = lateral_2_var.get()
            longitudinal_raw = longitudinal_2_var.get()
            traction_yaw_raw = traction_yaw_2_var.get()
            gantry_tilt_raw = gantry_tilt_2_var.get()
            device_time_raw = device_time_2_var.get()

            paste_column_2_var.set(
                f"{device_time_raw}\n{source_angle_raw}\n{detector_angle_raw}\n{lateral_raw}"
                f"\n{longitudinal_raw}\n{traction_yaw_raw}\n{gantry_tilt_raw}"
            )
        if device_time_raw == "":
            # Send anyway for DRR analysis.
            corridor_server_queue.put(("second_image", dict(), device_time_raw))

        try:
            loopx_pose = dict(
                source_angle=float(source_angle_raw),
                detector_angle=float(detector_angle_raw),
                lateral=float(lateral_raw),
                longitudinal=float(longitudinal_raw),
                traction_yaw=float(traction_yaw_raw),
                gantry_tilt=float(gantry_tilt_raw),
            )
        except ValueError:
            log.warning("Invalid values for LoopX pose")
            return


        device_timestamp = to_isoformat(device_time_raw)
        if device_timestamp is None:
            log.warning(f"Invalid device time: {device_time_raw}")

        corridor_server_queue.put(("second_image", loopx_pose, device_timestamp))

    tk.Button(root, text="Analyze Image 1", command=first_image).grid(row=r, column=1)
    tk.Button(root, text="Analyze Image 2", command=second_image).grid(row=r, column=2)
    r += 1

    # Clear buttons for each one
    def clear_first():
        paste_column_1_var.set("")
        source_angle_1_var.set("")
        detector_angle_1_var.set("")
        lateral_1_var.set("")
        longitudinal_1_var.set("")
        traction_yaw_1_var.set("")
        gantry_tilt_1_var.set("")
        device_time_1_var.set("")
    def clear_second():
        paste_column_2_var.set("")
        source_angle_2_var.set("")
        detector_angle_2_var.set("")
        lateral_2_var.set("")
        longitudinal_2_var.set("")
        traction_yaw_2_var.set("")
        gantry_tilt_2_var.set("")
        device_time_2_var.set("")
    tk.Button(root, text="Clear", command=clear_first).grid(row=r, column=1)
    tk.Button(root, text="Clear", command=clear_second).grid(row=r, column=2)
    r += 1

    # Xi button
    xi_var = tk.StringVar()
    tk.Label(root, text=f"Xi ({DEGREE_SIGN}):").grid(row=r + 0, column=0)
    tk.Entry(root, textvariable=xi_var).grid(row=r + 0, column=1)
    def update_xi():
        xi = maybe_float(xi_var.get())
        corridor_server_queue.put(("update_xi", xi))
    tk.Button(root, text="Update Xi", command=update_xi).grid(row=r + 0, column=2)
    r += 1

    # Adjustment radius
    adjustment_radius_var = tk.StringVar()
    tk.Label(root, text="Adjustment radius (mm):").grid(row=r + 0, column=0)    
    tk.Entry(root, textvariable=adjustment_radius_var).grid(row=r + 0, column=1)
    def update_adjustment_radius():
        adjustment_radius = maybe_float(adjustment_radius_var.get())
        corridor_server_queue.put(("update_adjustment_radius", adjustment_radius))
    tk.Button(root, text="Update", command=update_adjustment_radius).grid(row=r + 0, column=2)
    r += 1

    def switch_side():
        corridor_server_queue.put(("switch_side",))
    tk.Button(root, text="Switch side", command=switch_side).grid(row=r + 0, column=0)
    def switch_show_scene():
        corridor_server_queue.put(("switch_show_scene",))
    tk.Button(root, text="Show scene on/off", command=switch_show_scene).grid(row=r + 0, column=1)
    def switch_adjust_corridor():
        corridor_server_queue.put(("switch_adjust_corridor",))
    tk.Button(root, text="Adjustment on/off", command=switch_adjust_corridor).grid(row=r + 0, column=2)
    r += 1

    def hl_toggle():
        corridor_server_queue.put(("hl_toggle",))
    tk.Button(root, text="Track On/Off", command=hl_toggle).grid(row=r + 0, column=0)

    def close():
        corridor_server_queue.put(("close",))
        root.destroy()
    tk.Button(root, text="Close", command=close).grid(row=r + 0, column=2)
    r += 1

    root.mainloop()
    