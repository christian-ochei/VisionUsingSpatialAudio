# Enhanced Phone Camera Calibration Tool v4.0
# Optimized for smartphone cameras with adjustable distortion models
import cv2
import numpy as np
import threading
import time
import json
from queue import Queue, Empty, Full
from colorama import init, Fore, Back, Style

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# --- Camera Presets for Phone Cameras ---
CAMERA_PRESETS = {
    "1": {
        "name": "Standard/Main Camera",
        "description": "Normal lens (24-28mm equiv), minimal distortion",
        "flags": (cv2.CALIB_USE_INTRINSIC_GUESS |
                  cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 |
                  cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6),
        "color": Fore.GREEN
    },
    "2": {
        "name": "Wide-Angle Lens",
        "description": "Ultra-wide (12-16mm equiv), moderate barrel distortion",
        "flags": (cv2.CALIB_USE_INTRINSIC_GUESS |
                  cv2.CALIB_RATIONAL_MODEL),  # Uses K1-K6 for complex distortion
        "color": Fore.CYAN
    },
    "3": {
        "name": "Ultra-Wide/Fisheye",
        "description": "Extreme wide (< 12mm equiv), severe barrel distortion",
        "flags": (cv2.CALIB_USE_INTRINSIC_GUESS |
                  cv2.CALIB_RATIONAL_MODEL |
                  cv2.CALIB_THIN_PRISM_MODEL),  # Maximum distortion correction
        "color": Fore.YELLOW
    },
    "4": {
        "name": "Telephoto/Zoom",
        "description": "Telephoto lens (50mm+ equiv), minimal distortion",
        "flags": (cv2.CALIB_USE_INTRINSIC_GUESS |
                  cv2.CALIB_ZERO_TANGENT_DIST |
                  cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                  cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6),
        "color": Fore.MAGENTA
    },
    "5": {
        "name": "Custom - Minimal Correction",
        "description": "Only K1 radial distortion (fastest, least accurate)",
        "flags": (cv2.CALIB_USE_INTRINSIC_GUESS |
                  cv2.CALIB_ZERO_TANGENT_DIST |
                  cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                  cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6),
        "color": Fore.WHITE
    },
    "6": {
        "name": "Custom - Maximum Correction",
        "description": "All distortion parameters (slowest, most accurate)",
        "flags": cv2.CALIB_USE_INTRINSIC_GUESS,
        "color": Fore.RED
    }
}


def print_header(text):
    """Print a fancy header"""
    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{text.center(80)}")
    print(f"{Fore.CYAN}{'=' * 80}\n")


def print_success(text):
    """Print success message"""
    print(f"{Fore.GREEN}{Style.BRIGHT}✓ {text}")


def print_error(text):
    """Print error message"""
    print(f"{Fore.RED}{Style.BRIGHT}✗ {text}")


def print_info(text):
    """Print info message"""
    print(f"{Fore.YELLOW}ℹ {text}")


def print_progress(current, total, text=""):
    """Print progress bar"""
    bar_length = 40
    progress = current / total
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    percentage = progress * 100
    print(f"\r{Fore.CYAN}[{bar}] {Fore.WHITE}{percentage:.1f}% {text}", end='', flush=True)


def select_camera_preset():
    """Interactive camera preset selection"""
    print_header("📱 PHONE CAMERA CALIBRATION TOOL")

    print(f"{Fore.WHITE}{Style.BRIGHT}Select your camera lens type:\n")

    for key, preset in CAMERA_PRESETS.items():
        print(f"{preset['color']}{Style.BRIGHT}[{key}] {preset['name']}")
        print(f"{Fore.WHITE}    {preset['description']}\n")

    while True:
        choice = input(f"{Fore.CYAN}{Style.BRIGHT}Enter your choice (1-6): {Style.RESET_ALL}").strip()
        if choice in CAMERA_PRESETS:
            preset = CAMERA_PRESETS[choice]
            print(f"\n{preset['color']}{Style.BRIGHT}✓ Selected: {preset['name']}")
            print(f"{Fore.WHITE}  {preset['description']}")
            return preset['flags'], preset['name']
        else:
            print_error("Invalid choice. Please enter a number between 1-6.")


def generate_and_display_chessboard(chessboard_size=(9, 6), screen_resolution=(1920 * 2, 1080 * 2)):
    print_info("Generating on-screen chessboard pattern...")
    target_win_width, target_win_height = screen_resolution[0] // 2, screen_resolution[1] // 2
    margin = 50
    num_squares_x, num_squares_y = chessboard_size[0] + 1, chessboard_size[1] + 1
    drawable_width, drawable_height = target_win_width - 2 * margin, target_win_height - 2 * margin
    square_size = int(min(drawable_width / num_squares_x, drawable_height / num_squares_y))
    if square_size <= 0:
        raise ValueError("Calculated square size is too small.")

    board_width, board_height = num_squares_x * square_size, num_squares_y * square_size
    start_x, start_y = (target_win_width - board_width) // 2, (target_win_height - board_height) // 2
    chessboard_img = np.full((target_win_height, target_win_width, 3), 255, dtype=np.uint8)

    for y in range(num_squares_y):
        for x in range(num_squares_x):
            if (x + y) % 2 == 0:
                cv2.rectangle(chessboard_img,
                              (start_x + x * square_size, start_y + y * square_size),
                              (start_x + (x + 1) * square_size, start_y + (y + 1) * square_size),
                              (0, 0, 0), -1)

    cv2.namedWindow('Chessboard Pattern (Point Your Camera Here)', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Chessboard Pattern (Point Your Camera Here)', chessboard_img)
    cv2.moveWindow('Chessboard Pattern (Point Your Camera Here)', 50, 50)
    print_success("Chessboard pattern is now displayed. Keep this window visible.")


def load_and_test_calibration(frame_grabber, filename="camera_calib.json"):
    print_header("LOADING CALIBRATION DATA")
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        camera_matrix = np.array(data["camera_matrix"])
        dist_coeffs = np.array(data["distortion_coefficients"])
        lens_type = data.get("lens_type", "Unknown")
        preferred_alpha = data.get("preferred_alpha", 0.5)
        print_success(f"Calibration data loaded successfully")
        print_info(f"Lens type: {lens_type}")
        print_info(f"Saved preferred alpha: {preferred_alpha}")
    except FileNotFoundError:
        print_error(f"Calibration file '{filename}' not found.")
        return

    # Ask for alpha value
    print(f"\n{Fore.CYAN}{Style.BRIGHT}Alpha controls the zoom level:")
    print(f"{Fore.WHITE}  0.0 = Maximum zoom (no black borders, smallest FOV)")
    print(f"{Fore.WHITE}  0.5 = Balanced (recommended for wide-angle)")
    print(f"{Fore.WHITE}  1.0 = Minimum zoom (larger FOV, may have black borders)\n")

    while True:
        alpha_input = input(
            f"{Fore.CYAN}{Style.BRIGHT}Enter alpha value (0.0-1.0) [default: {preferred_alpha}]: {Style.RESET_ALL}").strip()
        if alpha_input == "":
            alpha = preferred_alpha
            break
        try:
            alpha = float(alpha_input)
            if 0.0 <= alpha <= 1.0:
                break
            else:
                print_error("Please enter a value between 0.0 and 1.0")
        except ValueError:
            print_error("Invalid input. Please enter a number.")

    print_header("UNDISTORTED VIDEO PREVIEW")
    print_info(f"Using alpha = {alpha}")
    print_info("Press '+' to zoom out, '-' to zoom in, 'ESC' to exit")

    current_alpha = alpha

    while True:
        frame = frame_grabber.get_latest_frame()
        if frame is None:
            continue

        h, w = frame.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), current_alpha, (w, h))

        # Use remap for better quality and border handling
        mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_mtx, (w, h), cv2.CV_32FC1)
        undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))

        # Crop to ROI to remove black borders
        x, y, w_crop, h_crop = roi
        if w_crop > 0 and h_crop > 0:
            undistorted_frame = undistorted_frame[y:y + h_crop, x:x + w_crop]

        if undistorted_frame.size == 0:
            continue

        h_u, w_u, _ = undistorted_frame.shape
        line_color = (0, 255, 255)

        # Draw grid lines
        for i in range(1, 10):
            cv2.line(undistorted_frame, (0, h_u * i // 10), (w_u, h_u * i // 10), line_color, 1)
            cv2.line(undistorted_frame, (w_u * i // 10, 0), (w_u * i // 10, h_u), line_color, 1)

        # Display alpha value on frame
        cv2.putText(undistorted_frame, f"Alpha: {current_alpha:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Undistorted Feed with Guide Lines", undistorted_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('+') or key == ord('='):
            current_alpha = min(1.0, current_alpha + 0.05)
            print_info(f"Alpha increased to {current_alpha:.2f} (more FOV, possible black borders)")
        elif key == ord('-') or key == ord('_'):
            current_alpha = max(0.0, current_alpha - 0.05)
            print_info(f"Alpha decreased to {current_alpha:.2f} (less FOV, no black borders)")


class FrameGrabber(threading.Thread):
    def __init__(self, camera_url):
        super().__init__()
        self.camera_url = camera_url
        self.frame_queue = Queue(maxsize=1)
        self.running = False
        self.cap = None
        self.daemon = True

    def start_capture(self):
        self.cap = cv2.VideoCapture(self.camera_url)
        if not self.cap.isOpened():
            raise IOError(f"Could not open stream: {self.camera_url}")

        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print_success(f"Camera opened: {self.camera_url}")
        print_info(f"Resolution: {int(width)}x{int(height)}")

        self.running = True
        self.start()

    def run(self):
        while self.running:
            if self.cap:
                ret, frame = self.cap.read()
                if ret:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass
                    try:
                        self.frame_queue.put_nowait(frame)
                    except Full:
                        pass
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.1)

    def get_latest_frame(self):
        try:
            return self.frame_queue.get(block=False)
        except Empty:
            return None

    def stop(self):
        self.running = False
        if self.is_alive():
            self.join(timeout=1)
        if self.cap:
            self.cap.release()


class CameraCalibrator:
    def __init__(self, frame_grabber, chessboard_size=(9, 6), square_size=25.0,
                 num_captures=50, calibration_flags=None, lens_type="Unknown"):
        self.frame_grabber = frame_grabber
        self.chessboard_size = chessboard_size
        self.num_captures = num_captures
        self.calibration_flags = calibration_flags
        self.lens_type = lens_type

        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

        self.objpoints = []
        self.imgpoints = []
        self.image_shape = None
        self.pose_coverage = []
        self.rvecs = None
        self.tvecs = None
        self.camera_matrix = None
        self.dist_coeffs = None

    def collect_calibration_frames(self):
        print_header("CALIBRATION FRAME COLLECTION")
        print_info(f"Camera type: {self.lens_type}")
        print_info("Follow the on-screen prompts carefully\n")

        last_capture_time = time.time()
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        while len(self.objpoints) < self.num_captures:
            frame = self.frame_grabber.get_latest_frame()
            if frame is None:
                if cv2.waitKey(1) & 0xFF == 27:
                    return False
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.image_shape is None:
                self.image_shape = gray.shape[::-1]

            binary_view = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 15, 2)
            cv2.imshow('Diagnostic Binary View', binary_view)

            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, flags)

            self._draw_feedback_on_frame(frame)

            if ret and (time.time() - last_capture_time > 1.5):
                self.objpoints.append(self.objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.imgpoints.append(corners2)

                center_x = int(np.mean(corners2[:, 0, 0]))
                center_y = int(np.mean(corners2[:, 0, 1]))
                self.pose_coverage.append((center_x, center_y))
                last_capture_time = time.time()

                print_progress(len(self.objpoints), self.num_captures,
                               f"Frame captured at position ({center_x}, {center_y})")
                print()  # New line after progress

            cv2.drawChessboardCorners(frame, self.chessboard_size, corners, ret)
            cv2.imshow('Calibration Feed (Your Camera)', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                return False

        print()  # Final newline
        print_success(f"All {self.num_captures} frames collected!")
        cv2.destroyWindow('Calibration Feed (Your Camera)')
        cv2.destroyWindow('Diagnostic Binary View')
        return True

    def _draw_feedback_on_frame(self, frame):
        h, w, _ = frame.shape
        cap_count = len(self.objpoints)
        prompt1, prompt2 = "", ""

        if cap_count < 10:
            prompt1 = "Phase 1/5: Capture all 4 CORNERS of the view."
        elif cap_count < 20:
            prompt1 = "Phase 2/5: Capture the EDGES (top, bottom, left, right)."
        elif cap_count < 30:
            prompt1 = "Phase 3/5: Get EXTREME TILTS (up/down, left/right)."
            prompt2 = "Make the board look like a trapezoid."
        elif cap_count < 40:
            prompt1 = "Phase 4/5: Move VERY CLOSE to the chessboard."
        else:
            prompt1 = "Phase 5/5: Move FAR AWAY from the chessboard."

        cv2.putText(frame, prompt1, (50, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, prompt2, (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Captured: {cap_count}/{self.num_captures}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Coverage map
        map_w, map_h = 200, int(200 * h / w)
        coverage_map = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        cv2.rectangle(coverage_map, (0, 0), (map_w - 1, map_h - 1), (255, 255, 255), 1)

        for (cx, cy) in self.pose_coverage:
            cv2.circle(coverage_map, (int(cx * map_w / w), int(cy * map_h / h)), 4, (0, 255, 0), -1)

        frame[h - map_h - 10:h - 10, w - map_w - 10:w - 10] = coverage_map

    def perform_calibration(self):
        print_header("PERFORMING CALIBRATION")

        if not self.objpoints:
            print_error("No calibration points collected.")
            return False

        if self.image_shape is None:
            print_error("Image resolution not determined.")
            return False

        h, w = self.image_shape[1], self.image_shape[0]
        initial_camera_matrix = np.array([[w, 0, w / 2],
                                          [0, w, h / 2],
                                          [0, 0, 1]], dtype=np.float32)

        print_info(f"Using calibration model: {self.lens_type}")
        print_info("Computing camera parameters...")

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.image_shape,
            initial_camera_matrix, None, flags=self.calibration_flags)

        if not ret:
            print_error("Calibration failed.")
            return False

        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.rvecs = rvecs
        self.tvecs = tvecs

        print_success("Calibration successful!")
        self.calculate_reprojection_error()
        return True

    def calculate_reprojection_error(self):
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i],
                                              self.tvecs[i], self.camera_matrix,
                                              self.dist_coeffs)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        final_error = mean_error / len(self.objpoints)

        print(f"\n{Fore.CYAN}{'─' * 60}")
        print(f"{Fore.WHITE}{Style.BRIGHT}Mean Reprojection Error: {Fore.YELLOW}{final_error:.4f} pixels")

        if final_error < 0.3:
            print(f"{Fore.GREEN}{Style.BRIGHT}★★★ Excellent calibration! ★★★")
        elif final_error < 0.5:
            print(f"{Fore.GREEN}{Style.BRIGHT}★★ Very good calibration!")
        elif final_error < 1.0:
            print(f"{Fore.YELLOW}★ Good calibration.")
        else:
            print(f"{Fore.RED}Consider recalibrating for better results.")

        print(f"{Fore.CYAN}{'─' * 60}\n")

    def save_calibration_data(self, filename="camera_calib.json"):
        if self.camera_matrix is None:
            print_error("No calibration data to save.")
            return

        # Ask for preferred alpha to save
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Save preferred alpha value for future use:")
        print(f"{Fore.WHITE}  0.0 = Maximum zoom (no black borders)")
        print(f"{Fore.WHITE}  0.5 = Balanced (recommended)")
        print(f"{Fore.WHITE}  1.0 = Minimum zoom (maximum FOV)\n")

        while True:
            alpha_input = input(
                f"{Fore.CYAN}{Style.BRIGHT}Preferred alpha (0.0-1.0) [default: 0.5]: {Style.RESET_ALL}").strip()
            if alpha_input == "":
                preferred_alpha = 0.5
                break
            try:
                preferred_alpha = float(alpha_input)
                if 0.0 <= preferred_alpha <= 1.0:
                    break
                else:
                    print_error("Please enter a value between 0.0 and 1.0")
            except ValueError:
                print_error("Invalid input. Please enter a number.")

        data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.dist_coeffs.tolist(),
            "lens_type": self.lens_type,
            "preferred_alpha": preferred_alpha,
            "image_resolution": list(self.image_shape),
            "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

        print_success(f"Calibration data saved to '{filename}'")
        print_info(f"Preferred alpha: {preferred_alpha}")

    def show_undistorted_preview(self):
        if self.camera_matrix is None:
            print_error("No calibration data for preview.")
            return

        # Ask for alpha value
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Alpha controls the zoom level:")
        print(f"{Fore.WHITE}  0.0 = Maximum zoom (no black borders, smallest FOV)")
        print(f"{Fore.WHITE}  0.5 = Balanced (recommended for wide-angle)")
        print(f"{Fore.WHITE}  1.0 = Minimum zoom (larger FOV, may have black borders)\n")

        while True:
            alpha_input = input(
                f"{Fore.CYAN}{Style.BRIGHT}Enter alpha value (0.0-1.0) [default: 0.5]: {Style.RESET_ALL}").strip()
            if alpha_input == "":
                alpha = 0.5
                break
            try:
                alpha = float(alpha_input)
                if 0.0 <= alpha <= 1.0:
                    break
                else:
                    print_error("Please enter a value between 0.0 and 1.0")
            except ValueError:
                print_error("Invalid input. Please enter a number.")

        print_header("UNDISTORTED VIDEO PREVIEW")
        print_info(f"Using alpha = {alpha}")
        print_info("Press '+' to zoom out, '-' to zoom in, 'ESC' to exit")

        current_alpha = alpha

        while True:
            frame = self.frame_grabber.get_latest_frame()
            if frame is None:
                continue

            h, w = frame.shape[:2]
            new_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs,
                                                         (w, h), current_alpha, (w, h))

            # Use remap for better quality and border handling
            mapx, mapy = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, None,
                                                     new_mtx, (w, h), cv2.CV_32FC1)
            undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            # Crop to ROI to remove black borders
            x, y, w_c, h_c = roi
            if w_c > 0 and h_c > 0:
                undistorted_cropped = undistorted[y:y + h_c, x:x + w_c]
            else:
                undistorted_cropped = undistorted

            if undistorted_cropped.size == 0:
                continue

            original_resized = cv2.resize(frame, (undistorted_cropped.shape[1], undistorted_cropped.shape[0]))
            comparison = np.hstack((original_resized, undistorted_cropped))

            cv2.putText(comparison, "Original", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 3)
            cv2.putText(comparison, "Undistorted", (undistorted_cropped.shape[1] + 50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(comparison, f"Alpha: {current_alpha:.2f}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            new_w = int(comparison.shape[1] * 2 / 5)
            new_h = int(comparison.shape[0] * 2 / 5)
            resized_comparison = cv2.resize(comparison, (new_w, new_h))

            cv2.imshow("Original vs. Undistorted", resized_comparison)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('+') or key == ord('='):
                current_alpha = min(1.0, current_alpha + 0.05)
                print_info(f"Alpha increased to {current_alpha:.2f} (more FOV, possible black borders)")
            elif key == ord('-') or key == ord('_'):
                current_alpha = max(0.0, current_alpha - 0.05)
                print_info(f"Alpha decreased to {current_alpha:.2f} (less FOV, no black borders)")


if __name__ == "__main__":
    # --- Configuration ---
    CAMERA_URL = "http://10.26.208.31:8080/video"  # Your IP camera URL
    CHESSBOARD_SIZE = (9, 6)
    SQUARE_SIZE_MM = 25.0
    NUM_CAPTURES = 50
    SCREEN_RESOLUTION = (1920, 1080)

    frame_grabber = None

    try:
        # Ask user if they want to load existing calibration or create new
        print_header("CAMERA CALIBRATION TOOL")
        print(f"{Fore.WHITE}[1] Create new calibration")
        print(f"{Fore.WHITE}[2] Load existing calibration\n")

        mode = input(f"{Fore.CYAN}{Style.BRIGHT}Select mode (1 or 2): {Style.RESET_ALL}").strip()

        frame_grabber = FrameGrabber(CAMERA_URL)
        frame_grabber.start_capture()
        time.sleep(2)

        if mode == "2":
            load_and_test_calibration(frame_grabber)
        else:
            # Select camera preset
            calibration_flags, lens_type = select_camera_preset()

            # Generate chessboard
            generate_and_display_chessboard(CHESSBOARD_SIZE, SCREEN_RESOLUTION)

            # Create calibrator with selected preset
            calibrator = CameraCalibrator(
                frame_grabber,
                CHESSBOARD_SIZE,
                SQUARE_SIZE_MM,
                NUM_CAPTURES,
                calibration_flags,
                lens_type
            )

            # Run calibration
            if calibrator.collect_calibration_frames():
                if calibrator.perform_calibration():
                    calibrator.save_calibration_data()
                    calibrator.show_undistorted_preview()

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Calibration interrupted by user.")
    except Exception as e:
        print_error(f"An error occurred: {e}")
    finally:
        print(f"\n{Fore.CYAN}{'=' * 80}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'Shutting down...'.center(80)}")
        print(f"{Fore.CYAN}{'=' * 80}\n")

        if frame_grabber:
            frame_grabber.stop()
        cv2.destroyAllWindows()