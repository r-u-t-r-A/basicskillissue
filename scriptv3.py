import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import argparse
import sys

def main():
    # --- Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Track a swinging disc and measure oscillations.")
    parser.add_argument("video", help="Path to the video file (e.g., my_video.mp4)")
    parser.add_argument("-s", "--start", type=int, default=None, 
                        help="Starting peak number to measure from (e.g., 1 for the first peak)")
    parser.add_argument("-e", "--end", type=int, default=None, 
                        help="Ending peak number to measure to (e.g., 51 for 50 full oscillations)")
    
    args = parser.parse_args()
    VIDEO_PATH = args.video

    # 1. Initialize video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{VIDEO_PATH}'. Please check the path.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        sys.exit(1)

    # 2. Select the feature to track
    print("Please draw a box around a distinct feature.")
    print("Press ENTER or SPACE to confirm the selection.")
    bbox = cv2.selectROI("Select Feature to Track", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Feature to Track")

    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    x_positions = []
    timestamps = []
    frame_count = 0

    print("Tracking in progress... Press 'q' to stop early.")

    # 3. Track the feature frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)

        if success:
            x_center = bbox[0] + (bbox[2] / 2)
            y_center = bbox[1] + (bbox[3] / 2)
            
            x_positions.append(x_center)
            timestamps.append(frame_count / fps)

            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            cv2.circle(frame, (int(x_center), int(y_center)), 4, (0, 255, 0), -1)
        else:
            print(f"Tracking lost at frame {frame_count}")

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # 4. Analyze the data
    if len(x_positions) < 10:
        print("Not enough data collected.")
        return

    x_positions = np.array(x_positions)
    timestamps = np.array(timestamps)

    min_distance_frames = int(fps * 0.5) 
    peaks, _ = find_peaks(x_positions, distance=min_distance_frames)
    peak_times = timestamps[peaks]
    
    # --- Analysis & Output ---
    if len(peak_times) > 1:
        periods = np.diff(peak_times) 
        total_recorded_peaks = len(peaks)
        total_oscillations = total_recorded_peaks - 1
        
        print(f"\n=== GENERAL RESULTS ===")
        print(f"Total peaks detected: {total_recorded_peaks}")
        print(f"Total full oscillations recorded: {total_oscillations}")
        print(f"Average time between oscillations: {np.mean(periods):.3f} seconds")

        # --- Custom Selected Peaks Measurement ---
        if args.start is not None and args.end is not None:
            print(f"\n=== CUSTOM MEASUREMENT ===")
            # Check if the requested peaks exist (1-indexed for the user)
            if 1 <= args.start < args.end <= total_recorded_peaks:
                # Convert 1-indexed user input to 0-indexed Python arrays
                start_index = args.start - 1
                end_index = args.end - 1
                
                custom_time = peak_times[end_index] - peak_times[start_index]
                oscillation_count = args.end - args.start
                
                print(f"Time from Peak {args.start} to Peak {args.end} ({oscillation_count} oscillations): {custom_time:.3f} seconds")
            else:
                print(f"WARNING: Invalid peak selection. You selected peaks {args.start} to {args.end}.")
                print(f"Please select values between 1 and {total_recorded_peaks}.")
        else:
            print("\n(Tip: You can measure time between specific peaks using the --start and --end arguments)")

    else:
        print("\nNot enough oscillations detected to calculate an average period.")

    # 5. Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, x_positions, label='X Position (Tracking Point)')
    plt.plot(timestamps[peaks], x_positions[peaks], "rx", markersize=10, label='Detected Peaks')
    plt.title('Disc Oscillation over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('X Coordinate (pixels)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
