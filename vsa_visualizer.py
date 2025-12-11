import time
from collections import deque
import math
import numpy as np

class VSAVisualizer:
    """
    Real-time bird's eye view visualizer for SLAM trajectory
    Shows camera position, orientation, path history, and frustum
    """

    def __init__(self, num_landmarks, intrinsics, width=1200, height=800, W=1920, H=1080, mode="portrait"):
        import pygame
        self.pygame = pygame
        self.pygame.init()
        self.width = width
        self.height = height
        self.screen = self.pygame.display.set_mode((width, height))
        self.pygame.display.set_caption(f"VSA Viewer | {mode}")

        # Colors
        self.bg_color = (15, 15, 20)
        self.grid_color = (40, 40, 50)
        self.grid_major_color = (60, 60, 70)
        self.path_color = (255, 255, 255)
        self.camera_color = (0, 255, 150)
        self.frustum_color = (0, 200, 255, 60)
        self.past_frustum_color = (60, 80, 100)
        self.text_color = (200, 220, 255)
        self.grid_text_color = (80, 90, 110)
        self.grid_text_color_y = (80, 110, 90)  # Green tint for Y axis

        # Camera state
        self.camera_positions = deque(maxlen=500)  # Store full (x, y, z)
        self.camera_rotations = deque(maxlen=500)
        self.frustum_history = deque(maxlen=20)

        # Landmarks
        self.landmarks = None  # Will be numpy array of shape (N, 7)
        self.landmarks_mask = None  # Boolean mask of shape (N,)
        self.landmark_color = [255, 255, 255, 200]  # RGBA for future customization

        # View control
        self.zoom = 50.0
        self.min_zoom = 5.0
        self.max_zoom = 500.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.pan_speed = 0.1

        # View mode
        self.view_mode = 'top'  # 'top' (XZ), 'side' (XY), 'front' (ZY)

        # Room bounding box
        self.room_dimensions = np.array([12.0, 12.0, 12.0])  # [width_x, height_y, depth_z]
        self.absorption = -3.0

        # Mouse dragging with momentum
        self.is_dragging = False
        self.last_mouse_pos = None
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.drag_damping = 0.92

        # Camera intrinsics (default for 1920x1080)
        self.intrinsics = intrinsics
        if mode == 'portrait':
            self.intrinsics = np.array([
                [self.intrinsics[1, 1], 0.0, self.intrinsics[1, 2]],
                [0.0, self.intrinsics[0, 0], self.intrinsics[0, 2]],
                [0.0, 0.0, 1.0]
            ])
            W, H = H, W
        elif mode == 'flat':
            # Flat mode: dummy intrinsics with high zoom (40x)
            # Small resolution for zoomed-in view
            flat_W, flat_H = 20, 20

            # 40x zoom means focal length is 40x the sensor size
            zoom_ratio = 40.0
            fx = flat_W * zoom_ratio / 2.0  # Focal length in pixels
            fy = flat_H * zoom_ratio / 2.0
            cx = flat_W / 2.0  # Principal point at center
            cy = flat_H / 2.0

            self.intrinsics = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ])
            W, H = flat_W, flat_H
        self.image_width = W
        self.image_height = H

        # Font
        self.font_tiny = self.pygame.font.Font(None, 18)

        self.clock = self.pygame.time.Clock()
        self.all_ones_mask = np.ones(num_landmarks, dtype=bool)
        self.running = True

    def set_landmarks(self, landmarks, mask=None):
        """
        Update landmarks for rendering
        landmarks: (N, 7) array or tensor with x,y,z in first 3 indices, amp in index 6
        mask: (N, ) boolean array/tensor for visibility (optional, defaults to all visible)
        """
        assert landmarks is not None
        self.landmarks = np.array(landmarks, dtype=np.float32)
        if mask is not None:
            self.landmarks_mask = np.array(mask, dtype=bool)
        else:
            # Default: all landmarks visible
            self.landmarks_mask = self.all_ones_mask[:len(landmarks)]

    def set_room_dimensions(self, dimensions):
        """Update room bounding box dimensions"""
        # Handle torch tensors without importing torch
        # if hasattr(dimensions, 'cpu'):
        #     dimensions = dimensions.cpu()
        # if hasattr(dimensions, 'numpy'):
        #     dimensions = dimensions.numpy()
        # if dimensions and isinstance(dimensions, list) and hasattr(dimensions[0], 'cpu'):
        #     dimensions = [a.cpu() for a in dimensions]
        self.room_dimensions = np.array(dimensions, dtype=float)

    def set_absorption(self, db_value):
        """Update room absorption value (0 to -6 dB)"""
        # Handle torch tensors without importing torch
        # if hasattr(db_value, 'cpu'):
        #     db_value = db_value.cpu()
        # if hasattr(db_value, 'numpy'):
        #     db_value = db_value.numpy()
        # if hasattr(db_value, 'item'):
        #     db_value = db_value.item()
        assert isinstance(db_value, np.ndarray)
        self.absorption = float(db_value)

    def get_absorption_color(self):
        """Calculate outline color based on absorption value (0 to -6 dB)"""
        # Clamp absorption to valid range
        abs_val = max(-6.0, min(0.0, self.absorption))
        # Normalize to 0-1 range (0 dB = 0, -6 dB = 1)
        t = abs(abs_val) / 6.0
        # Color gradient: white -> red -> orange -> green -> faint blue
        if t < 0.25:  # 0 to -1.5 dB: white -> red
            factor = t / 0.25
            r = 255
            g = int(255 * (1 - factor))
            b = int(255 * (1 - factor))
        elif t < 0.5:  # -1.5 to -3 dB: red -> orange
            factor = (t - 0.25) / 0.25
            r = 255
            g = int(165 * factor)
            b = 0
        elif t < 0.75:  # -3 to -4.5 dB: orange -> green
            factor = (t - 0.5) / 0.25
            r = int(255 * (1 - factor))
            g = int(165 + (255 - 165) * factor)
            b = 0
        else:  # -4.5 to -6 dB: green -> faint blue
            factor = (t - 0.75) / 0.25
            r = int(0 * (1 - factor))
            g = int(255 * (1 - factor) + 150 * factor)
            b = int(200 * factor)
        return r, g, b

    def world_to_screen(self, coord1, coord2):
        """
        Convert world coordinates to screen coordinates
        coord1: horizontal axis (X in both views)
        coord2: vertical axis (Z in top view, Y in side view)
        """
        screen_x = self.width / 2 + (coord1 - self.pan_x) * self.zoom
        screen_y = self.height / 2 - (coord2 - self.pan_y) * self.zoom
        return int(screen_x), int(screen_y)

    def world_to_screen_batch(self, coord1_array, coord2_array):
        """
        Vectorized conversion of world coordinates to screen coordinates
        Returns numpy arrays of screen coordinates
        """
        screen_x = self.width / 2 + (coord1_array - self.pan_x) * self.zoom
        screen_y = self.height / 2 - (coord2_array - self.pan_y) * self.zoom
        return screen_x.astype(np.int32), screen_y.astype(np.int32)

    def screen_to_world(self, screen_x, screen_y):
        """Convert screen coordinates to world coordinates"""
        world_coord1 = (screen_x - self.width / 2) / self.zoom + self.pan_x
        world_coord2 = -(screen_y - self.height / 2) / self.zoom + self.pan_y
        return world_coord1, world_coord2

    def get_grid_spacing(self):
        """Calculate adaptive grid spacing based on zoom"""
        if self.zoom < 10:
            return 10.0
        elif self.zoom < 20:
            return 5.0
        elif self.zoom < 50:
            return 2.0
        elif self.zoom < 100:
            return 1.0
        elif self.zoom < 200:
            return 0.5
        else:
            return 0.2

    @staticmethod
    def format_distance(value, spacing):
        """Format distance value for display"""
        if abs(value) < 0.001:
            return "0m"
        if spacing >= 1000:
            return f"{value / 1000:.1f}km"
        elif spacing >= 1.0:
            return f"{int(value)}m"
        else:
            return f"{value:.1f}m"

    def draw_grid(self):
        """Draw background grid with coordinate labels"""
        top_left_1, top_left_2 = self.screen_to_world(0, 0)
        bottom_right_1, bottom_right_2 = self.screen_to_world(self.width, self.height)

        base_spacing = self.get_grid_spacing()

        # Vertical lines (X axis for top/side, Z axis for front)
        start_1 = math.floor(top_left_1 / base_spacing) * base_spacing
        coord1 = start_1
        while coord1 <= bottom_right_1:
            screen_x, _ = self.world_to_screen(coord1, 0)
            is_major = abs(coord1 % (base_spacing * 5)) < 0.01
            color = self.grid_major_color if is_major else self.grid_color
            thickness = 2 if is_major else 1
            self.pygame.draw.line(self.screen, color, (screen_x, 0), (screen_x, self.height), thickness)
            coord1 += base_spacing

        # Horizontal lines (Z for top, Y for side, Y for front)
        start_2 = math.floor(bottom_right_2 / base_spacing) * base_spacing
        coord2 = start_2
        while coord2 <= top_left_2:
            _, screen_y = self.world_to_screen(0, coord2)
            is_major = abs(coord2 % (base_spacing * 5)) < 0.01
            color = self.grid_major_color if is_major else self.grid_color
            thickness = 2 if is_major else 1
            self.pygame.draw.line(self.screen, color, (0, screen_y), (self.width, screen_y), thickness)
            coord2 += base_spacing

        # Draw horizontal axis labels at bottom
        label_y = self.height - 25
        coord1 = start_1
        while coord1 <= bottom_right_1:
            screen_x, _ = self.world_to_screen(coord1, 0)
            if 50 < screen_x < self.width - 50:
                label_text = VSAVisualizer.format_distance(coord1, base_spacing)
                text_surface = self.font_tiny.render(label_text, True, self.grid_text_color)
                text_rect = text_surface.get_rect(center=(screen_x, label_y))
                self.screen.blit(text_surface, text_rect)
            coord1 += base_spacing

        # Draw vertical axis labels at left
        label_color = self.grid_text_color_y if self.view_mode in ['side', 'front'] else self.grid_text_color
        coord2 = start_2
        while coord2 <= top_left_2:
            _, screen_y = self.world_to_screen(0, coord2)
            if 30 < screen_y < self.height - 30:
                label_text = self.format_distance(coord2, base_spacing)
                text_surface = self.font_tiny.render(label_text, True, label_color)
                text_rect = text_surface.get_rect(center=(35, screen_y))
                self.screen.blit(text_surface, text_rect)
            coord2 += base_spacing

        # Draw origin axes
        origin_x, origin_y = self.world_to_screen(0, 0)
        # Horizontal axis: red for X, blue for Z
        h_axis_color = (50, 50, 100) if self.view_mode == 'front' else (100, 50, 50)
        self.pygame.draw.line(self.screen, h_axis_color, (0, origin_y), (self.width, origin_y), 2)
        # Vertical axis: green for Y, blue for Z
        v_axis_color = (50, 100, 50) if self.view_mode in ['side', 'front'] else (50, 50, 100)
        self.pygame.draw.line(self.screen, v_axis_color, (origin_x, 0), (origin_x, self.height), 2)

    def draw_room_bbox(self):
        """Draw room bounding box centered at origin"""
        w, h, d = self.room_dimensions

        # Calculate half-dimensions for centering
        half_w = w / 2.0
        half_h = h / 2.0
        half_d = d / 2.0

        # Get outline color based on absorption
        outline_color = self.get_absorption_color()

        if self.view_mode == 'side':
            # XY view (side view)
            top_left = self.world_to_screen(-half_w, half_h)
            top_right = self.world_to_screen(half_w, half_h)
            bottom_right = self.world_to_screen(half_w, -half_h)
            bottom_left = self.world_to_screen(-half_w, -half_h)
        elif self.view_mode == 'front':
            # ZY view (front view)
            top_left = self.world_to_screen(-half_d, half_h)
            top_right = self.world_to_screen(half_d, half_h)
            bottom_right = self.world_to_screen(half_d, -half_h)
            bottom_left = self.world_to_screen(-half_d, -half_h)
        else:
            # XZ view (bird's eye / top view)
            top_left = self.world_to_screen(-half_w, half_d)
            top_right = self.world_to_screen(half_w, half_d)
            bottom_right = self.world_to_screen(half_w, -half_d)
            bottom_left = self.world_to_screen(-half_w, -half_d)

        corners = [top_left, top_right, bottom_right, bottom_left]

        # Draw filled rectangle (very transparent)
        surface = self.pygame.Surface((self.width, self.height), self.pygame.SRCALPHA)
        self.pygame.draw.polygon(surface, (80, 80, 80, 15), corners)
        self.screen.blit(surface, (0, 0))

        # Draw outline (less transparent, colored by absorption)
        outline_surface = self.pygame.Surface((self.width, self.height), self.pygame.SRCALPHA)
        self.pygame.draw.polygon(outline_surface, (*outline_color, 80), corners, 2)
        self.screen.blit(outline_surface, (0, 0))

    def draw_path(self):
        """Draw camera trajectory path with transparency gradient"""
        if len(self.camera_positions) < 2:
            return

        # Extract appropriate coordinates based on view mode
        if self.view_mode == 'side':
            points = [self.world_to_screen(x, y) for x, y, z in self.camera_positions]
        elif self.view_mode == 'front':
            points = [self.world_to_screen(z, y) for x, y, z in self.camera_positions]
        else:
            points = [self.world_to_screen(x, z) for x, y, z in self.camera_positions]

        # Create transparent surface for path
        path_surface = self.pygame.Surface((self.width, self.height), self.pygame.SRCALPHA)

        # Draw path with alpha gradient
        for i in range(len(points) - 1):
            # Calculate alpha from 0 (oldest) to 255 (newest)
            alpha = int(255 * (i / len(points)))
            color = (255, 255, 255, alpha)
            self.pygame.draw.line(path_surface, color, points[i], points[i + 1], 3)

        # Draw recent path segment with full opacity
        if len(points) > 10:
            for i in range(len(points) - 10, len(points) - 1):
                self.pygame.draw.line(path_surface, (255, 255, 255, 255), points[i], points[i + 1], 4)

        self.screen.blit(path_surface, (0, 0))

    def draw_landmarks(self):
        """
        Efficiently draw landmarks with viewport culling
        """
        if self.landmarks is None or len(self.landmarks) == 0:
            return

        # Extract visible landmarks
        visible_mask = self.landmarks_mask
        if not visible_mask.any():
            return
        landmarks = self.landmarks[visible_mask]
        # Extract x, y, z, amp
        x = landmarks[:, 0]
        y = landmarks[:, 1]
        z = landmarks[:, 2]
        if landmarks.shape[-1] > 5:
            amp = landmarks[:, 5]
        else:
            amp = np.ones(landmarks.shape[0])

        # Select coordinates based on view mode
        if self.view_mode == 'side':
            coord1, coord2 = x, y
        elif self.view_mode == 'front':
            coord1, coord2 = z, y
        else:  # top view
            coord1, coord2 = x, z

        # Vectorized screen coordinate conversion
        screen_x, screen_y = self.world_to_screen_batch(coord1, coord2)

        # Viewport culling: only draw points within screen bounds (with margin)
        margin = 10
        in_viewport = (
            (screen_x >= -margin) & (screen_x < self.width + margin) &
            (screen_y >= -margin) & (screen_y < self.height + margin)
        )

        # Filter to visible points
        screen_x = screen_x[in_viewport]
        screen_y = screen_y[in_viewport]
        amp = amp[in_viewport]

        if len(screen_x) == 0:
            return

        # Calculate radii: amp 0.0->1.0 maps to radius 1->6 pixels
        # Using 0.5 as minimum to ensure visibility
        radii = (amp * 5.5 + 0.5).astype(np.int32)
        radii = np.clip(radii, 1, 6)

        # Draw landmarks efficiently
        # Group by radius for more efficient drawing
        for radius in np.unique(radii):
            radius_mask = (radii == radius)
            points_x = screen_x[radius_mask]
            points_y = screen_y[radius_mask]

            # Draw each point with this radius
            for px, py in zip(points_x, points_y):
                if radius <= 1:
                    # For tiny points, just set a pixel
                    self.screen.set_at((int(px), int(py)), self.landmark_color[:3])
                else:
                    # Use circle for larger points
                    self.pygame.draw.circle(self.screen, self.landmark_color[:3],
                                       (int(px), int(py)), int(radius))

    def draw_frustum_infinite(self, position, rotation_matrix):
        """Draw camera frustum with infinite lines based on intrinsics"""
        x, y, z = position[:3]
        # Four corners of the image (indices 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left)
        # Based on user feedback:
        # Indices 0, 1 (y=0) are the BOTTOM.
        # Indices 2, 3 (y=H) are the TOP.
        image_corners = np.array([
            [0, 0, 1],
            [self.image_width, 0, 1],
            [self.image_width, self.image_height, 1],
            [0, self.image_height, 1]
        ]).T

        # Convert to normalized camera coordinates
        K_inv = np.linalg.inv(self.intrinsics)
        rays_camera = K_inv @ image_corners

        # Transform rays to world space
        rays_world = []
        for i in range(4):
            ray_cam = rays_camera[:, i]
            ray_world = rotation_matrix @ np.array([ray_cam[0], ray_cam[1], ray_cam[2]])
            rays_world.append(ray_world)

        # Camera position on screen
        if self.view_mode == 'side':
            cam_screen = self.world_to_screen(x, y)
        elif self.view_mode == 'front':
            cam_screen = self.world_to_screen(z, y)
        else:
            cam_screen = self.world_to_screen(x, z)

        # Draw infinite lines for each frustum edge
        surface = self.pygame.Surface((self.width, self.height), self.pygame.SRCALPHA)

        # Get base color and define alpha range for depth cuing
        base_r, base_g, base_b, base_a = self.frustum_color
        min_alpha = 30  # Faintest line (going away)
        max_alpha = 200  # Most opaque line (coming towards)

        # Define greenish color for top lines
        greenish_color = (0, 255, 100, base_a)  # Green tint

        ray_depth_dirs = []
        corner_points = []
        visual_distance = 3.0

        # 1. Calculate corner points and ray depth directions first
        for i in range(4):
            ray = rays_world[i]
            norm = np.linalg.norm(ray)
            if norm > 0:
                ray_norm = ray / norm
            else:
                ray_norm = ray

            if self.view_mode == 'side':
                depth_dir = ray_norm[2]  # Z-axis (into/out of screen)
            elif self.view_mode == 'front':
                depth_dir = ray_norm[0]  # X-axis (into/out of screen)
            else:  # 'top' view
                depth_dir = ray_norm[1]  # Y-axis (into/out of screen)

            ray_depth_dirs.append(depth_dir)

            corner_x = x + ray[0] * visual_distance
            corner_y = y + ray[1] * visual_distance
            corner_z = z + ray[2] * visual_distance

            if self.view_mode == 'side':
                corner_points.append(self.world_to_screen(corner_x, corner_y))
            elif self.view_mode == 'front':
                corner_points.append(self.world_to_screen(corner_z, corner_y))
            else:
                corner_points.append(self.world_to_screen(corner_x, corner_z))

        # 2. Draw the four main frustum lines (camera to far point)
        for i in range(4):
            ray = rays_world[i]
            depth_dir = ray_depth_dirs[i]

            alpha_factor = (depth_dir + 1.0) / 2.0  # Normalize to [0, 1]
            new_alpha = int(min_alpha + (max_alpha - min_alpha) * alpha_factor)

            # --- MODIFICATION START ---
            # Indices 2 (bottom-right -> top-right) and 3 (bottom-left -> top-left) are the TOP.
            if i in [2, 3]:
                line_color = (greenish_color[0], greenish_color[1], greenish_color[2], new_alpha)
            else:
                line_color = (base_r, base_g, base_b, new_alpha)
            # --- MODIFICATION END ---

            far_distance = 10000
            far_x = x + ray[0] * far_distance
            far_y = y + ray[1] * far_distance
            far_z = z + ray[2] * far_distance

            if self.view_mode == 'side':
                far_screen = self.world_to_screen(far_x, far_y)
            elif self.view_mode == 'front':
                far_screen = self.world_to_screen(far_z, far_y)
            else:
                far_screen = self.world_to_screen(far_x, far_z)

            self.pygame.draw.line(surface, line_color, cam_screen, far_screen, 1)

        # 3. Draw lines connecting corners (the "end cap")
        for i in range(4):
            next_i = (i + 1) % 4

            avg_depth_dir = (ray_depth_dirs[i] + ray_depth_dirs[next_i]) / 2.0

            alpha_factor = (avg_depth_dir + 1.0) / 2.0
            new_alpha = int(min_alpha + (max_alpha - min_alpha) * alpha_factor)

            # --- MODIFICATION START ---
            # The top edge connects indices 2 and 3
            if (i == 2 and next_i == 3) or (i == 3 and next_i == 2):  # The top edge
                line_color = (greenish_color[0], greenish_color[1], greenish_color[2], new_alpha)
            else:  # All other edges (sides and bottom)
                line_color = (base_r, base_g, base_b, new_alpha)
            # --- MODIFICATION END ---

            self.pygame.draw.line(surface, line_color, corner_points[i], corner_points[next_i], 1)
        self.screen.blit(surface, (0, 0))

    def draw_camera(self, position, rotation_matrix):
        """Draw camera with proper frustum"""
        x, y, z = position[:3]

        # Draw current frustum with infinite lines
        self.draw_frustum_infinite(position, rotation_matrix)

        # Draw camera center point
        if self.view_mode == 'side':
            cam_screen = self.world_to_screen(x, y)
        elif self.view_mode == 'front':
            cam_screen = self.world_to_screen(z, y)
        else:
            cam_screen = self.world_to_screen(x, z)

        self.pygame.draw.circle(self.screen, self.camera_color, cam_screen, 8)
        self.pygame.draw.circle(self.screen, (255, 255, 255), cam_screen, 6)

    def update_pose(self, position, rotation):
        """Update camera pose from 4x4 transformation matrix"""
        x, y, z = position[:3]
        self.camera_positions.append((x, y, z))
        return position, rotation

    def handle_events(self):
        """Handle pygame events including mouse dragging with momentum"""
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.running = False

            elif event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_q:
                    self.running = False
                elif event.key == self.pygame.K_EQUALS or event.key == self.pygame.K_PLUS:
                    self.zoom = min(self.zoom * 1.2, self.max_zoom)
                elif event.key == self.pygame.K_MINUS:
                    self.zoom = max(self.zoom / 1.2, self.min_zoom)

            elif event.type == self.pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.is_dragging = True
                    self.last_mouse_pos = event.pos
                    self.velocity_x = 0
                    self.velocity_y = 0

            elif event.type == self.pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.is_dragging = False
                    self.last_mouse_pos = None

            elif event.type == self.pygame.MOUSEMOTION:
                if self.is_dragging and self.last_mouse_pos:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]

                    # Update velocity for momentum
                    self.velocity_x = dx / self.zoom
                    self.velocity_y = -dy / self.zoom

                    # Apply movement
                    self.pan_x -= self.velocity_x
                    self.pan_y -= self.velocity_y

                    self.last_mouse_pos = event.pos

            elif event.type == self.pygame.MOUSEWHEEL:
                # Get mouse position for zoom pivot
                mouse_x, mouse_y = self.pygame.mouse.get_pos()

                # Get world coordinates at mouse position before zoom
                world_x_before, world_y_before = self.screen_to_world(mouse_x, mouse_y)

                # Apply zoom
                zoom_factor = 1.1
                if event.y > 0:
                    self.zoom = min(self.zoom * zoom_factor, self.max_zoom)
                else:
                    self.zoom = max(self.zoom / zoom_factor, self.min_zoom)

                # Get world coordinates at mouse position after zoom
                world_x_after, world_y_after = self.screen_to_world(mouse_x, mouse_y)

                # Adjust pan to keep mouse position fixed
                self.pan_x -= world_x_after - world_x_before
                self.pan_y -= world_y_after - world_y_before

        # Get key state once for all key-held checks
        keys = self.pygame.key.get_pressed()
        # Handle arrow keys for panning by applying acceleration
        if not self.is_dragging:
            key_acceleration = (self.pan_speed * 10) / self.zoom
            if keys[self.pygame.K_LEFT] or keys[self.pygame.K_a]:
                self.velocity_x += key_acceleration
            if keys[self.pygame.K_RIGHT] or keys[self.pygame.K_d]:
                self.velocity_x -= key_acceleration
            if keys[self.pygame.K_UP] or keys[self.pygame.K_w]:
                self.velocity_y -= key_acceleration
            if keys[self.pygame.K_DOWN] or keys[self.pygame.K_s]:
                self.velocity_y += key_acceleration

        # Apply momentum and damping when not dragging
        if not self.is_dragging:
            if abs(self.velocity_x) > 0.001 or abs(self.velocity_y) > 0.001:
                self.pan_x -= self.velocity_x
                self.pan_y -= self.velocity_y
                self.velocity_x *= self.drag_damping
                self.velocity_y *= self.drag_damping
            else:
                self.velocity_x = 0
                self.velocity_y = 0

        # Determine view mode based on held keys
        if keys[self.pygame.K_u]:
            self.view_mode = 'front'  # ZY view (camera faces right)
        elif keys[self.pygame.K_y]:
            self.view_mode = 'side'  # XY view
        else:
            self.view_mode = 'top'  # XZ view (default)

    def render(self, position, rotation):
        """Main render function"""
        self.handle_events()

        pose_data = self.update_pose(position, rotation)
        if pose_data is None:
            return self.running

        position, rotation = pose_data

        self.screen.fill(self.bg_color)

        self.draw_grid()
        self.draw_room_bbox()
        self.draw_landmarks()  # Draw landmarks before path/camera for layering
        self.draw_path()
        self.draw_camera(position, rotation)

        self.pygame.display.flip()
        self.clock.tick(60)
        return self.running

    def close(self):
        """Clean up pygame"""
        self.pygame.quit()

class PerlinNoise:
    """Simple Perlin-like noise for smooth random motion"""

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.phase_x = np.random.random() * 100
        self.phase_y = np.random.random() * 100
        self.phase_z = np.random.random() * 100
        self.phase_rx = np.random.random() * 100
        self.phase_ry = np.random.random() * 100
        self.phase_rz = np.random.random() * 100

    @staticmethod
    def get(t, phase, scale=1.0, freq=0.1):
        """Get smooth noise value at time t"""
        return np.sin((t + phase) * freq) * scale

def create_rotation_matrix(roll, pitch, yaw):
    """Create rotation matrix from Euler angles (in radians)"""
    # Rotation around X (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Rotation around Y (pitch)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Rotation around Z (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


def create_pose_matrix(position, rotation_matrix):
    """Create 4x4 pose matrix (world to camera)"""
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix.T  # Transpose for world-to-camera
    pose[:3, 3] = -rotation_matrix.T @ position
    return pose


class LandmarkRain:
    """Manages falling/moving landmarks"""

    def __init__(self, num_landmarks=2000):
        self.num_landmarks = num_landmarks

        # Initialize landmark positions (N, 7)
        # [x, y, z, unused, unused, unused, amp]
        self.landmarks = np.zeros((num_landmarks, 7), dtype=np.float32)

        # Random initial positions spread across room
        self.landmarks[:, 0] = np.random.uniform(-5, 5, num_landmarks)  # x
        self.landmarks[:, 1] = np.random.uniform(-2, 4, num_landmarks)  # y (height)
        self.landmarks[:, 2] = np.random.uniform(-5, 5, num_landmarks)  # z

        # Random initial amplitudes
        self.landmarks[:, 6] = np.random.uniform(0.2, 1.0, num_landmarks)

        # Velocity for each landmark
        self.velocities = np.random.uniform(-0.5, 0.5, (num_landmarks, 3))
        self.velocities[:, 1] = np.random.uniform(-0.02, -0.005, num_landmarks)  # Slight downward drift

        # Amp change rate
        self.amp_velocities = np.random.uniform(-0.01, 0.01, num_landmarks)

        # All landmarks start visible
        self.mask = np.ones(num_landmarks, dtype=bool)

        # Time-based phase for each landmark
        self.phases = np.random.uniform(0, 2 * np.pi, num_landmarks)

    def update(self, dt, time_elapsed):
        """Update landmark positions"""
        # Add some wave motion
        wave_influence = 0.02
        self.landmarks[:, 0] += self.velocities[:, 0] * dt
        self.landmarks[:, 1] += self.velocities[:, 1] * dt
        self.landmarks[:, 2] += self.velocities[:, 2] * dt

        # Add sinusoidal wave motion for interesting patterns
        self.landmarks[:, 0] += np.sin(time_elapsed * 0.5 + self.phases) * wave_influence
        self.landmarks[:, 2] += np.cos(time_elapsed * 0.3 + self.phases * 1.5) * wave_influence

        # Update amplitudes with smooth variation
        self.landmarks[:, 6] += self.amp_velocities * dt

        # Clamp amplitudes to 0-1 and bounce
        over_max = self.landmarks[:, 6] > 1.0
        under_min = self.landmarks[:, 6] < 0.0
        self.landmarks[over_max, 6] = 1.0
        self.landmarks[under_min, 6] = 0.0
        self.amp_velocities[over_max] *= -1
        self.amp_velocities[under_min] *= -1

        # Wrap landmarks that go out of bounds (modulo effect)
        # X-axis wrapping
        out_x_pos = self.landmarks[:, 0] > 6
        out_x_neg = self.landmarks[:, 0] < -6
        self.landmarks[out_x_pos, 0] = -6 + (self.landmarks[out_x_pos, 0] - 6)
        self.landmarks[out_x_neg, 0] = 6 + (self.landmarks[out_x_neg, 0] + 6)

        # Y-axis wrapping (vertical)
        out_y_pos = self.landmarks[:, 1] > 5
        out_y_neg = self.landmarks[:, 1] < -3
        self.landmarks[out_y_pos, 1] = -3 + (self.landmarks[out_y_pos, 1] - 5)
        self.landmarks[out_y_neg, 1] = 5 + (self.landmarks[out_y_neg, 1] + 3)

        # Z axis wrapping
        out_z_pos = self.landmarks[:, 2] > 6
        out_z_neg = self.landmarks[:, 2] < -6
        self.landmarks[out_z_pos, 2] = -6 + (self.landmarks[out_z_pos, 2] - 6)
        self.landmarks[out_z_neg, 2] = 6 + (self.landmarks[out_z_neg, 2] + 6)

        # Randomly toggle visibility of some landmarks
        if np.random.random() < 0.01:  # 1% chance per frame
            toggle_indices = np.random.choice(self.num_landmarks,
                                              size=max(1, self.num_landmarks // 100),
                                              replace=False)
            self.mask[toggle_indices] = ~self.mask[toggle_indices]


def main():
    """Main demo loop"""
    print("VSA Landmark Rain Demo")
    print("=" * 50)
    print("Controls:")
    print("  - Mouse drag to pan")
    print("  - Mouse wheel to zoom")
    print("  - Arrow keys / WASD to pan")
    print("  - Y key: Side view (XY)")
    print("  - U key: Front view (ZY)")
    print("  - Default: Top view (XZ)")
    print("  - Q or close window to quit")
    print("=" * 50)

    # Initialize visualizer
    viz = VSAVisualizer(width=1400, height=900, num_landmarks=1, intrinsics=np.array([
            [1346.2626115837832, 0.0, 971.1651220753282],
            [0.0, 1345.375887563327, 531.1205273630911],
            [0.0, 0.0, 1.0]
        ]))

    # Initialize landmark rain
    rain = LandmarkRain(num_landmarks=300)

    # Initialize smooth camera motion
    noise = PerlinNoise(seed=42)

    # Time tracking
    start_time = time.time()
    last_time = start_time

    # Main loop
    frame_count = 0
    while viz.running:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        time_elapsed = current_time - start_time

        # Generate smooth camera motion using Perlin-like noise
        t = time_elapsed
        cam_x = PerlinNoise.get(t, noise.phase_x, scale=2.0, freq=0.3)
        cam_y = PerlinNoise.get(t, noise.phase_y, scale=1.5, freq=0.25) + 0.5
        cam_z = PerlinNoise.get(t, noise.phase_z, scale=2.0, freq=0.35)

        # Smooth rotation
        roll = PerlinNoise.get(t, noise.phase_rx, scale=0.2, freq=0.2)
        pitch = PerlinNoise.get(t, noise.phase_ry, scale=0.3, freq=0.15)
        yaw = PerlinNoise.get(t, noise.phase_rz, scale=0.5, freq=0.25)

        # Create pose
        position = np.array([cam_x, cam_y, cam_z])
        rotation = create_rotation_matrix(roll, pitch, yaw)

        # Update landmarks
        rain.update(dt, time_elapsed)
        viz.set_landmarks(rain.landmarks, rain.mask)

        # Render
        viz.running = viz.render(position, rotation)

        frame_count += 1

        # Print FPS every 60 frames
        if frame_count % 60 == 0:
            fps = 60 / (time.time() - (start_time + time_elapsed - dt * 60))
            visible_count = np.sum(rain.mask)
            print(f"Frame {frame_count}: {fps:.1f} FPS | "
                  f"Visible landmarks: {visible_count}/{rain.num_landmarks} | "
                  f"View: {viz.view_mode.upper()}")

    viz.close()
    print("\nDemo finished!")


if __name__ == "__main__":
    main()
