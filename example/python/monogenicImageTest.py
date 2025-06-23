import cv2
import sys
import time
# Import your Pybind11 module
# Make sure this module (_monogenic_python.so or _monogenic_python.pyd)
# is in your Python path or the same directory as this script.
try:
    import pymonogenic
except ImportError:
    print("Error: Could not import the pymonogenic module.")
    print("Make sure you have built and installed the Pybind11 module correctly.")
    print("You might need to set your PYTHONPATH environment variable.")
    sys.exit(1)


def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <imagefilename>")
        sys.exit(1)

    img_path = sys.argv[1]

    # Load the image in grayscale using OpenCV
    # cv2.imread returns a NumPy array
    input_image_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if input_image_np is None:
        print(f"Error: Could not open or find the image: {img_path}")
        sys.exit(1)

    print(f"Image loaded successfully from {img_path}. Shape: {input_image_np.shape}")

    # Get image dimensions
    # NumPy shape is (height, width) for 2D images
    ysize, xsize = input_image_np.shape

    # --- Adaptive Image Size Calculation ---
    # Define a maximum screen width to target. This is an estimate as
    # getting actual screen resolution is platform-dependent.
    MAX_SCREEN_WIDTH = 1920  # Common HD width

    # Define window spacing
    WINDOW_SPACING = 20  # Spacing between windows

    # Number of windows to display side by side
    NUM_WINDOWS = 5

    # Starting position for the first window
    start_x = 50
    start_y = 50

    # Calculate the total width required if images were displayed at original size
    required_total_width_at_full_size = (NUM_WINDOWS * xsize) + ((NUM_WINDOWS - 1) * WINDOW_SPACING) + start_x

    display_xsize = xsize
    display_ysize = ysize
    scaling_factor = 1.0

    # If the required width exceeds the maximum screen width, calculate a scaling factor
    if required_total_width_at_full_size > MAX_SCREEN_WIDTH:
        target_total_width = MAX_SCREEN_WIDTH - start_x - ((NUM_WINDOWS - 1) * WINDOW_SPACING)
        # Calculate the target width for each displayed image
        # Use integer division //
        target_display_xsize = target_total_width // NUM_WINDOWS

        # Calculate the scaling factor based on the target display width
        scaling_factor = target_display_xsize / xsize

        # Calculate the new display dimensions, rounding to the nearest integer
        display_xsize = int(round(xsize * scaling_factor))
        display_ysize = int(round(ysize * scaling_factor))

        # Ensure minimum size if needed (optional)
        # if display_xsize < 100: display_xsize = 100
        # if display_ysize < 100: display_ysize = 100

        print(f"Images scaled down by factor: {scaling_factor:.2f} to fit on screen.")
        print(f"New display size: {display_xsize}x{display_ysize}")

    else:
        print(f"Images displayed at original size: {display_xsize}x{display_ysize}")

    # Offset for the next window based on the calculated display size
    x_offset = display_xsize + WINDOW_SPACING

    # Create display windows and position them side by side
    window_names = ["Even", "Odd Y", "Odd X", "Feature Symmetry", "Feature Asymmetry"]
    windows = {}

    # Create a dummy image for initial display to help with window positioning
    # Resize the input image to the calculated display size
    initial_display_img = cv2.resize(input_image_np, (display_xsize, display_ysize))
    # Convert to 3 channels if needed for color display, though grayscale is fine
    # initial_display_img = cv2.cvtColor(initial_display_img, cv2.COLOR_GRAY2BGR)


    for i, name in enumerate(window_names):
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        # Display something initially to ensure the window is created before moving
        cv2.imshow(name, initial_display_img)
        cv2.moveWindow(name, start_x + i * x_offset, start_y)
        windows[name] = True

    # The first step to using the monogenicProcessor is to initialise
    # a monogenic processor object.
    # We'll use a wavelength of 50 pixels as in the C++ example
    wavelength = 50.0

    try:
        mgFilts = pymonogenic.MonogenicProcessor(ysize, xsize, wavelength)
        print("MonogenicProcessor object created.")
    except Exception as e:
        print(f"Error creating MonogenicProcessor: {e}")
        # Destroy windows before exiting
        for name in window_names:
            if name in windows:
                cv2.destroyWindow(name)
        sys.exit(1)


    start_time = time.time()

    # Calculate the monogenic signal for the input image
    # The wrapped method should handle the NumPy to cv::Mat conversion
    try:
        mgFilts.findMonogenicSignal(input_image_np)
        # Now we can access the odd and even components and derived measures
        # Assuming these methods are wrapped to return NumPy arrays
        fa_np = mgFilts.getFeatureAsymmetry()
        # --- End Profiling ---
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Print the elapsed time
        print(f"Monogenic signal and feature calculation time: {elapsed_time:.4f} seconds")

        even_np = mgFilts.getEvenFilt()
        oddy_np, oddx_np = mgFilts.getOddFiltCartesian()
        fs_np = mgFilts.getFeatureSymmetry()
        print("Feature components retrieved.")

    except Exception as e:
        print(f"Error during monogenic signal or feature calculation: {e}")
        # Destroy windows before exiting
        for name in window_names:
            if name in windows:
                cv2.destroyWindow(name)
        sys.exit(1)

    # --- Display Results ---

    # Normalize the output for better visualization (values might be outside 0-255)
    # Assuming the outputs are float arrays that need normalization to [0, 1] or [0, 255]
    # Normalize to [0, 1] for consistency with C++ example's normalize(..., 0, 1, ...)
    even_disp = cv2.normalize(even_np, None, 0, 1, cv2.NORM_MINMAX)
    oddy_disp = cv2.normalize(oddy_np, None, 0, 1, cv2.NORM_MINMAX)
    oddx_disp = cv2.normalize(oddx_np, None, 0, 1, cv2.NORM_MINMAX)
    fs_disp = cv2.normalize(fs_np, None, 0, 1, cv2.NORM_MINMAX)
    fa_disp = cv2.normalize(fa_np, None, 0, 1, cv2.NORM_MINMAX)

    # Resize the normalized images to the calculated display size
    final_display_img1 = cv2.resize(even_disp, (display_xsize, display_ysize))
    final_display_img2 = cv2.resize(oddy_disp, (display_xsize, display_ysize))
    final_display_img3 = cv2.resize(oddx_disp, (display_xsize, display_ysize))
    final_display_img4 = cv2.resize(fs_disp, (display_xsize, display_ysize))
    final_display_img5 = cv2.resize(fa_disp, (display_xsize, display_ysize))


    # Display the final results in the positioned windows
    cv2.imshow("Even", final_display_img1)
    cv2.imshow("Odd Y", final_display_img2)
    cv2.imshow("Odd X", final_display_img3)
    cv2.imshow("Feature Symmetry", final_display_img4)
    cv2.imshow("Feature Asymmetry", final_display_img5)


    # Wait indefinitely until a key is pressed to keep the windows open
    print("Press any key while an OpenCV window is focused to exit.")
    cv2.waitKey(0) # Wait indefinitely for a key press

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
