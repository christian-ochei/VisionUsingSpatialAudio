# import numpy as np
# import tkinter as tk
# import sounddevice as sd
# import math
# import torch
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.fastest = True
# torch.use_deterministic_algorithms(False)
# torch.backends.cudnn.deterministic = False
# torch.cuda.empty_cache()
# from highp_spatial_landmarks import RealtimeSpatialPulseEngine
#
# class PulserUI:
#     def __init__(self):
#         # Reduce the sample rate
#         self.sample_rate = 12000
#
#         # self.sample_rate = 10000
#         self.block_size = 512
#         self.num_speakers = 1 # This works run with it now!!
#
#         self.gain_slider = None
#         self.cuda_stream = torch.cuda.Stream(priority=-2)
#         self.audio_engine = RealtimeSpatialPulseEngine(
#             sample_rate=self.sample_rate, block_size=self.block_size, num_speakers=self.num_speakers,
#             stream=self.cuda_stream
#         )
#         self._outdata_tensor = torch.empty(
#             (self.block_size, 2),
#             dtype=torch.float32,
#             device="cpu"
#         )
#         self.stream = None
#
#         # Frequency control state
#         self.freq_x, self.freq_y = 50.0, 50.0
#
#         # Speaker position state (X, Y, Z)
#         self.speaker_x_percent, self.speaker_z_percent = 50.0, 75.0
#         self.speaker_y_percent = 50.0  # Default middle height
#
#         # Head position state (X, Y, Z)
#         self.head_x_percent, self.head_z_percent = 50.0, 50.0
#         self.head_y_percent = 50.0  # Default middle height
#
#         # Other audio parameters
#         self.pulse_rate, self.pulse_decay = 1.0, 5.0
#
#         self.root = tk.Tk()
#         self.point_size = 8
#         self.tracking_mode = False
#         self.last_valid_point = None
#         self.last_valid_rotation = None
#         self.tracker_connected = True
#
#         self.setup_ui()
#         # self.audio_engine.set_speaker_position(0, self.speaker_x_percent, self.speaker_y_percent, self.speaker_z_percent)
#         # self.audio_engine.set_head_position_percent(self.head_x_percent, self.head_y_percent, self.head_z_percent)
#         self.start_audio()
#         self.update_display_loop()
#
#     def create_grid_canvas_with_labels(self, parent, title, width=280, height=280,
#                                        highlight_color='#00ffff', grid_color='#006666',
#                                        x_label="X", y_label="Y"):
#         """Create a compact labeled grid canvas."""
#         frame = tk.Frame(parent, bg='#0a0a0a')
#
#         # Title - more compact
#         title_widget = tk.Label(frame, text=title, font=('Arial', 11, 'bold'),
#                                 fg='#ffffff', bg='#0a0a0a')
#         title_widget.pack(pady=(0, 3))
#
#         # Canvas with axis labels
#         canvas_frame = tk.Frame(frame, bg='#0a0a0a')
#         canvas_frame.pack()
#
#         # Y-axis label (left side, vertical)
#         y_label_widget = tk.Label(canvas_frame, text=y_label, font=('Arial', 8),
#                                   fg='#888888', bg='#0a0a0a', width=8, anchor='e')
#         y_label_widget.pack(side='left', padx=(0, 2))
#
#         # Canvas container
#         canvas_col = tk.Frame(canvas_frame, bg='#0a0a0a')
#         canvas_col.pack(side='left')
#
#         # Main canvas - smaller size
#         canvas = tk.Canvas(canvas_col, width=width, height=height, bg='#111111',
#                            highlightthickness=1, highlightbackground=highlight_color)
#         canvas.pack()
#
#         # X-axis label (bottom)
#         x_label_widget = tk.Label(canvas_col, text=x_label, font=('Arial', 8),
#                                   fg='#888888', bg='#0a0a0a')
#         x_label_widget.pack(pady=(1, 0))
#
#         # Draw grid with more lines for better precision
#         self.draw_grid(canvas, color=grid_color, width=width, height=height)
#
#         # Store for dynamic updates
#         canvas.x_label_widget = x_label_widget
#         canvas.y_label_widget = y_label_widget
#         return frame, canvas
#
#     def create_controllable_point(self, canvas, x_percent, y_percent, color='#ff6b6b',
#                                   click_callback=None, drag_callback=None):
#         """Create a draggable point on a canvas."""
#         canvas_x, canvas_y = self.grid_to_canvas(x_percent, y_percent)
#         point = canvas.create_oval(canvas_x - self.point_size, canvas_y - self.point_size,
#                                    canvas_x + self.point_size, canvas_y + self.point_size,
#                                    fill=color, outline='#ffffff', width=2)
#
#         if click_callback:
#             canvas.bind('<Button-1>', click_callback)
#         if drag_callback:
#             canvas.bind('<B1-Motion>', drag_callback)
#
#         return point
#
#     def setup_ui(self):
#         self.root.title("Hyper-Realistic Spatial Audio Engine v9 [HRIR + Reverb]")
#         self.root.configure(bg='#0a0a0a')
#         self.root.geometry("1200x900")
#
#         main_frame = tk.Frame(self.root, bg='#0a0a0a')
#         main_frame.pack(fill='both', expand=True, padx=20, pady=20)
#
#         # Title
#         title_label = tk.Label(main_frame, text="Real-time Physical Acoustics + Spatial Audio Simulation",
#                                font=('Arial', 18, 'bold'), fg='#00ffff', bg='#0a0a0a')
#         title_label.pack(pady=(0, 10))
#
#         tracking_frame = tk.Frame(main_frame, bg='#1a1a1a')
#         tracking_frame.pack(fill='x', pady=5)
#
#         # Tracking mode toggle
#         self.tracking_button = tk.Button(
#             tracking_frame,
#             text="TRACKING MODE: ON",
#             font=('Arial', 12, 'bold'),
#             fg='#00ff00',
#             bg='#222222',
#             activebackground='#333333',
#             command=self.toggle_tracking_mode,
#             width=25
#         )
#         self.tracking_button.pack(side='left', padx=10)
#
#         # Connection status
#         self.connection_label = tk.Label(
#             tracking_frame,
#             text="● CONNECTED",
#             font=('Arial', 12, 'bold'),
#             fg='#00ff00',
#             bg='#1a1a1a'
#         )
#         self.connection_label.pack(side='left', padx=20)
#
#         # Master gain control
#         gain_frame = tk.Frame(main_frame, bg='#1a1a1a')
#         gain_frame.pack(fill='x', pady=10)
#
#         gain_label = tk.Label(gain_frame, text="Master Gain (db)", font=('Arial', 12, 'bold'),
#                               fg='#ff6b6b', bg='#1a1a1a', width=15, anchor='w')
#         gain_label.pack(side='left', padx=10, pady=5)
#
#         self.gain_slider = tk.Scale(gain_frame, from_=-60.0, to=60.0, resolution=0.01,
#                                     orient='horizontal', length=400, command=self.on_gain_change,
#                                     bg='#222222', fg='#ff6b6b', troughcolor='#111111',
#                                     highlightthickness=0)
#         self.gain_slider.set(self.audio_engine.master_gain)
#         self.gain_slider.pack(side='left', expand=True, pady=5)
#
#         # Canvas controls frame
#         canvas_frame = tk.Frame(main_frame, bg='#0a0a0a')
#         canvas_frame.pack(pady=(0, 20))
#
#         # 1. Frequency Control Grid
#         self.freq_frame, self.freq_canvas = self.create_grid_canvas_with_labels(
#             canvas_frame, "Sound Source (Tone)", highlight_color='#00ffff',
#             grid_color='#006666', x_label="Freq X (0-100%)", y_label="Freq Y")
#         self.freq_frame.pack(side='left', padx=(0, 20))
#
#         self.freq_point = self.create_controllable_point(
#             self.freq_canvas, self.freq_x, self.freq_y, color='#ff6b6b',
#             click_callback=self.on_freq_click, drag_callback=self.on_freq_drag)
#
#         # 2. Speaker and Head Position Grid
#         self.position_frame, self.position_canvas = self.create_grid_canvas_with_labels(
#             canvas_frame, "Speaker (Green) & Head (Pink) Position", highlight_color='#ff6600',
#             grid_color='#660033', x_label="X (0-40.2m)", y_label="Z (0-59.5m)")
#         self.position_frame.pack(side='left')
#         self.head_arrow = None
#         self.create_head_indicator()
#
#         # Create both points on the same canvas
#         self.speaker_point = self.create_controllable_point(
#             self.position_canvas, self.speaker_x_percent, self.speaker_z_percent,
#             color='#66ff66', click_callback=self.on_position_click,
#             drag_callback=self.on_position_drag)
#
#         self.head_point = self.create_controllable_point(
#             self.position_canvas, self.head_x_percent, self.head_z_percent,
#             color='#ff66ff', click_callback=self.on_position_click,
#             drag_callback=self.on_position_drag)
#
#         # Y-axis controls for speaker and head
#         y_controls_frame = tk.LabelFrame(main_frame, text="Y-Axis Controls (Height)",
#                                          font=('Arial', 12, 'bold'), fg='white',
#                                          bg='#1a1a1a', pady=10, padx=10)
#         y_controls_frame.pack(fill='x', expand=True, pady=(20, 10))
#
#         # Speaker Y control
#         speaker_y_frame = tk.Frame(y_controls_frame, bg='#1a1a1a')
#         speaker_y_label = tk.Label(speaker_y_frame, text="Speaker Height",
#                                    font=('Arial', 12), fg='#66ff66', bg='#1a1a1a',
#                                    width=15, anchor='w')
#         speaker_y_label.pack(side='left', padx=10)
#
#         self.speaker_y_slider = tk.Scale(speaker_y_frame, from_=0, to=100,
#                                          resolution=1, orient='horizontal', length=400,
#                                          command=self.on_speaker_y_change, bg='#222222',
#                                          fg='#66ff66', troughcolor='#111111',
#                                          highlightthickness=0)
#         self.speaker_y_slider.set(self.speaker_y_percent)
#         self.speaker_y_slider.pack(side='left', expand=True)
#         speaker_y_frame.pack(fill='x', pady=5)
#
#         # Head Y control
#         head_y_frame = tk.Frame(y_controls_frame, bg='#1a1a1a')
#         head_y_label = tk.Label(head_y_frame, text="Head Height",
#                                 font=('Arial', 12), fg='#ff66ff', bg='#1a1a1a',
#                                 width=15, anchor='w')
#         head_y_label.pack(side='left', padx=10)
#
#         self.head_y_slider = tk.Scale(head_y_frame, from_=0, to=100,
#                                       resolution=1, orient='horizontal', length=400,
#                                       command=self.on_head_y_change, bg='#222222',
#                                       fg='#ff66ff', troughcolor='#111111',
#                                       highlightthickness=0)
#         self.head_y_slider.set(self.head_y_percent)
#         self.head_y_slider.pack(side='left', expand=True)
#         head_y_frame.pack(fill='x', pady=5)
#
#         # Source controls
#         source_controls_frame = tk.LabelFrame(main_frame, text="Source Controls",
#                                               font=('Arial', 12, 'bold'), fg='white',
#                                               bg='#1a1a1a', pady=10, padx=10)
#         source_controls_frame.pack(fill='x', expand=True, pady=(20, 10))
#
#         # Pulse rate control
#         pulse_rate_frame = tk.Frame(source_controls_frame, bg='#1a1a1a')
#         pulse_rate_label = tk.Label(pulse_rate_frame, text="Pulse Rate (Hz)",
#                                     font=('Arial', 12), fg='white', bg='#1a1a1a',
#                                     width=15, anchor='w')
#         pulse_rate_label.pack(side='left', padx=10)
#
#         self.pulse_rate_slider = tk.Scale(pulse_rate_frame, from_=0.2, to=10.0,
#                                           resolution=0.1, orient='horizontal', length=400,
#                                           command=self.on_pulse_rate_change, bg='#222222',
#                                           fg='#00ffff', troughcolor='#111111',
#                                           highlightthickness=0)
#         self.pulse_rate_slider.set(self.pulse_rate)
#         self.pulse_rate_slider.pack(side='left', expand=True)
#         pulse_rate_frame.pack(fill='x')
#
#         # Pulse decay control
#         pulse_decay_frame = tk.Frame(source_controls_frame, bg='#1a1a1a')
#         pulse_decay_label = tk.Label(pulse_decay_frame, text="Pulse Decay",
#                                      font=('Arial', 12), fg='white', bg='#1a1a1a',
#                                      width=15, anchor='w')
#         pulse_decay_label.pack(side='left', padx=10)
#
#         self.pulse_decay_slider = tk.Scale(pulse_decay_frame, from_=0.5, to=20.0,
#                                            resolution=0.1, orient='horizontal', length=400,
#                                            command=self.on_pulse_decay_change, bg='#222222',
#                                            fg='#00ffff', troughcolor='#111111',
#                                            highlightthickness=0)
#         self.pulse_decay_slider.set(self.pulse_decay)
#         self.pulse_decay_slider.pack(side='left', expand=True)
#         pulse_decay_frame.pack(fill='x')
#
#         # Room controls
#         reverb_controls_frame = tk.LabelFrame(main_frame, text="Room Controls",
#                                               font=('Arial', 12, 'bold'), fg='white',
#                                               bg='#1a1a1a', pady=10, padx=10)
#         reverb_controls_frame.pack(fill='x', expand=True, pady=10)
#
#         # Room dimension sliders
#         self.w_frame, self.w_slider = self.create_exp_slider(
#             reverb_controls_frame, "Room Width", 4, 200, self.audio_engine.room_width,
#             self.on_room_dim_change)
#         self.h_frame, self.h_slider = self.create_exp_slider(
#             reverb_controls_frame, "Room Height", 4, 200, self.audio_engine.room_height,
#             self.on_room_dim_change)
#         self.d_frame, self.d_slider = self.create_exp_slider(
#             reverb_controls_frame, "Room Depth", 4, 200, self.audio_engine.room_depth,
#             self.on_room_dim_change)
#
#         self.w_frame.pack(fill='x')
#         self.h_frame.pack(fill='x')
#         self.d_frame.pack(fill='x')
#
#         # Absorption control
#         absorb_frame = tk.Frame(reverb_controls_frame, bg='#1a1a1a')
#         absorb_label = tk.Label(absorb_frame, text="Wall Absorbance", font=('Arial', 12),
#                                 fg='white', bg='#1a1a1a', width=15, anchor='w')
#         absorb_label.pack(side='left', padx=10)
#
#         self.absorb_slider = tk.Scale(absorb_frame, from_=-20.0, to=0.0, resolution=0.1,
#                                       orient='horizontal', length=400, showvalue=False,
#                                       command=self.on_absorption_change, bg='#222222',
#                                       fg='#00ffff', troughcolor='#111111',
#                                       highlightthickness=0)
#         self.absorb_slider.set(-6.0)
#         self.absorb_slider.pack(side='left', expand=True)
#
#         self.absorb_val_label = tk.Label(absorb_frame, text=f"{-6.0:.1f} dB",
#                                          font=('Arial', 12, 'bold'), fg='#00ffff',
#                                          bg='#1a1a1a', width=10)
#         self.absorb_val_label.pack(side='left', padx=10)
#         absorb_frame.pack(fill='x')
#
#         # Info display
#         self.info_frame = tk.Frame(main_frame, bg='#0a0a0a')
#         self.info_frame.pack(fill='x', pady=10)
#
#         self.pos_label = tk.Label(self.info_frame, text="...", font=('Arial', 10, 'bold'),
#                                   fg='#ffffff', bg='#222222', width=30)
#         self.pos_label.pack(side='left', expand=True, padx=5)
#
#         self.head_pos_label = tk.Label(self.info_frame, text="...", font=('Arial', 10, 'bold'),
#                                        fg='#ffffff', bg='#222222', width=30)
#         self.head_pos_label.pack(side='left', expand=True, padx=5)
#
#         self.room_label = tk.Label(self.info_frame, text="...", font=('Arial', 10, 'bold'),
#                                    fg='#ffffff', bg='#222222', width=25)
#         self.room_label.pack(side='left', expand=True, padx=5)
#
#         self.rt60_label = tk.Label(self.info_frame, text="...", font=('Arial', 10, 'bold'),
#                                    fg='#ffffff', bg='#222222', width=15)
#         self.rt60_label.pack(side='left', expand=True, padx=5)
#
#         self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
#
#     def create_head_indicator(self):
#         """Create arrow showing head position and orientation."""
#         # Initial position at center
#         cx, cy = self.grid_to_canvas(50, 50)
#
#         # Arrow pointing forward (up in canvas, which is +Z)
#         arrow_length = 30
#         self.head_arrow = self.position_canvas.create_line(
#             cx, cy, cx, cy - arrow_length,
#             arrow=tk.LAST, width=3, fill='#ffff00',
#             arrowshape=(12, 15, 5)
#         )
#
#         # Head position circle
#         self.head_circle = self.position_canvas.create_oval(
#             cx - 6, cy - 6, cx + 6, cy + 6,
#             fill='#ffff00', outline='#ffffff', width=2
#         )
#
#     def update_head_indicator(self, x_percent, z_percent, yaw_radians):
#         """Update head arrow position and orientation."""
#         cx, cy = self.grid_to_canvas(x_percent, z_percent)
#
#         # Arrow points in direction of head orientation
#         arrow_length = 30
#         # In canvas: y increases downward, but z increases upward in room
#         # yaw: 0 = facing +Z (up in canvas, toward smaller y)
#         end_x = cx + arrow_length * np.sin(yaw_radians)
#         end_y = cy - arrow_length * np.cos(yaw_radians)
#
#         self.position_canvas.coords(self.head_arrow, cx, cy, end_x, end_y)
#         self.position_canvas.coords(
#             self.head_circle,
#             cx - 6, cy - 6, cx + 6, cy + 6
#         )
#
#     def toggle_tracking_mode(self):
#         """Toggle between tracking and manual control mode."""
#         self.tracking_mode = not self.tracking_mode
#
#         if self.tracking_mode:
#             self.tracking_button.config(
#                 text="TRACKING MODE: ON",
#                 fg='#00ff00'
#             )
#         else:
#             self.tracking_button.config(
#                 text="TRACKING MODE: OFF",
#                 fg='#ff6600'
#             )
#
#     def update_from_tracker(self):
#         """Update positions from visual landmarks tracker."""
#         if not self.tracking_mode:
#             return
#
#         try:
#             # Get speaker position
#             point = None#self.visual_landmarks_pipe.point()
#
#             if point is not None:
#                 self.last_valid_point = point
#                 self.tracker_connected = True
#                 self.connection_label.config(text="● CONNECTED", fg='#00ff00')
#
#                 # Update speaker position in audio engine
#                 # self.audio_engine.set_speaker_position_from_tracker(point)
#
#             else:
#                 self.tracker_connected = False
#                 self.connection_label.config(text="● DISCONNECTED", fg='#ff0000')
#                 # Keep using last valid point
#                 # if self.last_valid_point is not None:
#                 #     self.audio_engine.set_speaker_position_from_tracker(self.last_valid_point)
#
#             # Get head rotation
#             rotation = self.visual_landmarks_pipe.head_rotation()
#
#             if rotation is not None:
#                 self.last_valid_rotation = rotation
#                 self.audio_engine.set_head_rotation(rotation)
#
#                 # Extract yaw from rotation matrix for visualization
#                 # rotation is IMU-to-world, we want yaw in X-Z plane
#                 # yaw = atan2(R[0,2], R[2,2]) gives rotation around Y axis
#                 yaw = np.arctan2(rotation[0, 2], rotation[2, 2])
#
#                 # Head always at center (0, 0, 0) in display
#                 self.update_head_indicator(50.0, 50.0, yaw)
#             else:
#                 # Use last valid rotation
#                 if self.last_valid_rotation is not None:
#                     self.audio_engine.set_head_rotation(self.last_valid_rotation)
#                     yaw = np.arctan2(self.last_valid_rotation[0, 2],
#                                      self.last_valid_rotation[2, 2])
#                     self.update_head_indicator(50.0, 50.0, yaw)
#
#         except Exception as e:
#             print(f"Tracker update error: {e}")
#             self.tracker_connected = False
#             self.connection_label.config(text="● ERROR", fg='#ff0000')
#
#     def create_exp_slider(self, parent, label, from_, to, default, command):
#         """Create exponential slider for room dimensions."""
#         frame = tk.Frame(parent, bg='#1a1a1a')
#         label_widget = tk.Label(frame, text=label, font=('Arial', 12), fg='white',
#                                 bg='#1a1a1a', width=15, anchor='w')
#         label_widget.pack(side='left', padx=10)
#
#         log_min, log_max = math.log(from_), math.log(to)
#
#         def scale_to_value(v):
#             return math.exp(log_min + (float(v) / 100.0) * (log_max - log_min))
#
#         def value_to_scale(v):
#             return (math.log(v) - log_min) / (log_max - log_min) * 100.0
#
#         slider = tk.Scale(frame, from_=0, to=100, orient='horizontal', length=400,
#                           showvalue=False, command=lambda v: command(scale_to_value(v)),
#                           bg='#222222', fg='#00ffff', troughcolor='#111111',
#                           highlightthickness=0)
#         slider.set(value_to_scale(default))
#         slider.pack(side='left', expand=True)
#
#         val_label = tk.Label(frame, text=f"{default:.1f} m", font=('Arial', 12, 'bold'),
#                              fg='#00ffff', bg='#1a1a1a', width=10)
#         val_label.pack(side='left', padx=10)
#
#         def update_label(v):
#             val = scale_to_value(v)
#             val_label.config(text=f"{val:.1f} m")
#             command(val)
#
#         slider.config(command=update_label)
#         return frame, slider
#
#     def draw_grid(self, canvas, color='#006666', width=280, height=280):
#         """Draw grid lines on canvas with proper scaling."""
#         # Draw 10 grid lines (creating 9 segments) for better precision
#         for i in range(10):
#             x = i * (width / 9)
#             y = i * (height / 9)
#             canvas.create_line(x, 0, x, height, fill=color, width=1)
#             canvas.create_line(0, y, width, y, fill=color, width=1)
#
#     def grid_to_canvas(self, x, y, c_width=280, c_height=280):
#         """Convert grid percentage to canvas coordinates."""
#         return (x / 100.0) * c_width, (1.0 - y / 100.0) * c_height
#
#     def canvas_to_grid(self, cx, cy, c_width=280, c_height=280):
#         """Convert canvas coordinates to grid percentage."""
#         return (np.clip((cx / c_width) * 100.0, 0, 100),
#                 np.clip((1.0 - cy / c_height) * 100.0, 0, 100))
#
#     def update_position_labels(self):
#         """Update position grid labels based on current room dimensions."""
#         room_dim = self.audio_engine.space_effect.room_dimensions
#         width, depth = room_dim[0], room_dim[2]
#         self.position_canvas.x_label_widget.config(text=f"X (0-{width:.1f}m)")
#         self.position_canvas.y_label_widget.config(text=f"Z (0-{depth:.1f}m)")
#
#     def get_closest_point(self, cx, cy):
#         """Determine which point is closest to click position."""
#         speaker_cx, speaker_cy = self.grid_to_canvas(self.speaker_x_percent, self.speaker_z_percent)
#         head_cx, head_cy = self.grid_to_canvas(self.head_x_percent, self.head_z_percent)
#
#         speaker_dist = math.sqrt((cx - speaker_cx) ** 2 + (cy - speaker_cy) ** 2)
#         head_dist = math.sqrt((cx - head_cx) ** 2 + (cy - head_cy) ** 2)
#
#         return 'speaker' if speaker_dist < head_dist else 'head'
#
#     # Event handlers
#     def on_gain_change(self, value):
#         self.audio_engine.set_master_gain(float(value))
#
#     def on_pulse_rate_change(self, value):
#         self.audio_engine.set_pulse_rate(0, float(value))
#
#     def on_pulse_decay_change(self, value):
#         self.audio_engine.set_pulse_decay(0, float(value))
#
#     def on_speaker_y_change(self, value):
#         self.speaker_y_percent = float(value)
#         # Update audio engine if it has speaker Y position method
#         self.audio_engine.set_speaker_position(0, self.speaker_x_percent, self.speaker_y_percent, self.speaker_z_percent)
#
#     def on_head_y_change(self, value):
#         self.head_y_percent = float(value)
#         # Update audio engine if it has head Y position method
#         self.audio_engine.set_head_position_percent(self.head_x_percent, self.head_y_percent, self.head_z_percent)
#
#     def on_room_dim_change(self, value=None):
#         w = float(self.w_slider.get()) / 100.0
#         h = float(self.h_slider.get()) / 100.0
#         d = float(self.d_slider.get()) / 100.0
#
#         map_val = lambda v, v_min, v_max: math.exp(math.log(v_min) + v * (math.log(v_max) - math.log(v_min)))
#         self.audio_engine.set_room_dimensions(map_val(w, 4, 200), map_val(h, 4, 200), map_val(d, 4, 200))
#
#         # Update position labels
#         self.update_position_labels()
#
#     def on_absorption_change(self, value):
#         db_val = float(value)
#         self.audio_engine.set_wall_absorbance(db_val)
#         self.absorb_val_label.config(text=f"{db_val:.1f} dB")
#
#     # Point movement handlers
#     def on_freq_click(self, event):
#         self.move_freq_point(event.x, event.y)
#
#     def on_freq_drag(self, event):
#         self.move_freq_point(event.x, event.y)
#
#     def on_position_click(self, event):
#         # Only allow manual control when not in tracking mode
#         if self.tracking_mode:
#             return
#         self.active_point = self.get_closest_point(event.x, event.y)
#         self.move_position_point(event.x, event.y)
#
#     def on_position_drag(self, event):
#         # Only allow manual control when not in tracking mode
#         if self.tracking_mode:
#             return
#         if hasattr(self, 'active_point'):
#             self.move_position_point(event.x, event.y)
#
#     def move_freq_point(self, cx, cy):
#         """Move frequency control point."""
#         self.freq_x, self.freq_y = self.canvas_to_grid(cx, cy)
#         cx, cy = self.grid_to_canvas(self.freq_x, self.freq_y)
#         self.freq_canvas.coords(self.freq_point, cx - self.point_size, cy - self.point_size,
#                                 cx + self.point_size, cy + self.point_size)
#
#         self.audio_engine.set_frequency_target(0, self.freq_x, self.freq_y)
#
#         # --- MODIFICATION START ---
#         # EXACT calculation based on RealtimeSpatialPulseEngine._perceptual_frequency_amplitude
#
#         # 1. Constants from your engine
#         min_freq = 20.0
#         max_freq = min(20000.0, self.sample_rate / 2)
#
#         # 2. Calculate ERB (Equivalent Rectangular Bandwidth) boundaries
#         # Formula: 21.4 * log10(1 + 0.00437 * f)
#         min_erb = 21.4 * math.log10(1 + 0.00437 * min_freq)
#         max_erb = 21.4 * math.log10(1 + 0.00437 * max_freq)
#
#         # 3. Interpolate ERB based on Y percentage
#         normalized_input = self.freq_y / 100.0
#         current_erb = min_erb + normalized_input * (max_erb - min_erb)
#
#         # 4. Convert ERB back to Frequency (Hz)
#         # Formula: (10^(erb/21.4) - 1) / 0.00437
#         current_freq_hz = (10.0 ** (current_erb / 21.4) - 1.0) / 0.00437
#
#         self.freq_canvas.y_label_widget.config(text=f"{current_freq_hz:.1f} Hz")
#         # --- MODIFICATION END ---
#
#     def move_position_point(self, cx, cy):
#         """Move speaker or head position point based on which is active."""
#         x_percent, z_percent = self.canvas_to_grid(cx, cy)
#         if self.active_point == 'speaker':
#             self.speaker_x_percent, self.speaker_z_percent = x_percent, z_percent
#             cx, cy = self.grid_to_canvas(self.speaker_x_percent, self.speaker_z_percent)
#             self.position_canvas.coords(self.speaker_point, cx - self.point_size, cy - self.point_size,
#                                         cx + self.point_size, cy + self.point_size)
#             self.audio_engine.set_speaker_position(0, self.speaker_x_percent, self.speaker_y_percent, self.speaker_z_percent)
#
#         elif self.active_point == 'head':
#             self.head_x_percent, self.head_z_percent = x_percent, z_percent
#             cx, cy = self.grid_to_canvas(self.head_x_percent, self.head_z_percent)
#             self.position_canvas.coords(self.head_point, cx - self.point_size, cy - self.point_size,
#                                         cx + self.point_size, cy + self.point_size)
#             # Update audio engine if it has head position method
#             self.audio_engine.set_head_position_percent(self.head_x_percent, self.head_y_percent, self.head_z_percent)
#
#     def update_display_loop(self):
#         """Update display information periodically."""
#         # Update from tracker first
#         self.update_from_tracker()
#
#         # Speaker position info
#         spk_pos = self.audio_engine.space_effect.speaker_positions
#
#         if self.tracking_mode and self.last_valid_point is not None:
#             # Show tracked position
#             tracked = self.last_valid_point
#             self.pos_label.config(
#                 text=f"Speaker (Tracked): ({tracked[0]:.2f}, {tracked[1]:.2f}, {tracked[2]:.2f})m"
#             )
#         else:
#             # Show manual position
#             self.pos_label.config(
#                 text=f"Speaker (Manual): ({spk_pos[0][0]:.1f}, {spk_pos[0][1]:.1f}, {spk_pos[0][2]:.1f})m"
#             )
#
#         # Head always at origin
#         self.head_pos_label.config(text=f"Head: (0.0, 0.0, 0.0)m")
#         self.pos_label.config(text=f"Speaker: ({spk_pos[0][0]:.1f}, {spk_pos[0][1]:.1f}, {spk_pos[0][2]:.1f})m")
#
#         # Head position info (convert percentages to room coordinates for display)
#         room_dim = self.audio_engine.space_effect.room_dimensions
#         head_x = (self.head_x_percent / 100.0) * room_dim[0] - room_dim[0] / 2
#         head_y = (self.head_y_percent / 100.0) * room_dim[1] - room_dim[1] / 2
#         head_z = (self.head_z_percent / 100.0) * room_dim[2] - room_dim[2] / 2
#         self.head_pos_label.config(text=f"Head: ({head_x:.1f}, {head_y:.1f}, {head_z:.1f})m")
#
#         # Room dimensions info
#         self.room_label.config(text=f"Room: {room_dim[0]:.1f}x{room_dim[1]:.1f}x{room_dim[2]:.1f}m")
#
#         # RT60 info
#         rt60 = self.audio_engine.space_effect.compute_rt60()
#         rt60_text = f"RT60: {rt60:.2f}s" if rt60 < 1000 else "RT60: ∞"
#         self.rt60_label.config(text=rt60_text)
#
#         self.root.after(50, self.update_display_loop)
#
#     # Audio system methods
#     def audio_callback(self, outdata, frames, time, status):
#         with torch.cuda.stream(self.cuda_stream):
#             out = self.audio_engine.process(frames)
#             out = out[0]
#             self._outdata_tensor.data = torch.from_numpy(outdata)
#             self.cuda_stream.synchronize()
#             self._outdata_tensor.copy_(out, non_blocking=False)
#         return outdata
#
#     def start_audio(self):
#         self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=2,
#                                       blocksize=self.block_size, dtype=np.float32,
#                                       latency='low', callback=self.audio_callback)
#         self.stream.start()
#         print(f"Audio Engine Started: {self.sample_rate}Hz, {self.block_size} samples")
#
#     def on_closing(self):
#         if self.stream:
#             self.stream.close()
#         self.root.destroy()
#
#     def run(self):
#         self.root.mainloop()
#
# def main():
#     # visual_landmarks_pipe = VisualLandmarksTrackerPipe()
#     # visual_landmarks_pipe.start_background_update()
#     app = PulserUI()
#     app.run()
#
#
# if __name__ == "__main__":
#     main()

import numpy as np
import tkinter as tk
import sounddevice as sd
import soundfile as sf
import math
import torch
import sys

# --- CUDA Setup ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = False
torch.cuda.empty_cache()

from highp_spatial_landmarks import RealtimeSpatialPulseEngine


class PulserUI:
    def __init__(self):
        # --- MODIFIED: Audio Configuration ---
        self.sample_rate = 5000
        # self.sample_rate = 12000
        self.block_size = 512
        self.num_speakers = 256
        self.output_filename = "spatial_audio_output2.wav"

        self.wav_file = None
        self.gain_slider = None
        self.cuda_stream = torch.cuda.Stream(priority=-2)

        print(f"Initializing Engine: {self.num_speakers} Speakers @ {self.sample_rate}Hz")
        self.audio_engine = RealtimeSpatialPulseEngine(
            sample_rate=self.sample_rate, block_size=self.block_size, num_speakers=self.num_speakers,
            stream=self.cuda_stream
        )
        self._outdata_tensor = torch.empty(
            (self.block_size, 2),
            dtype=torch.float32,
            device="cpu"
        )
        self.stream = None

        # Frequency control state
        self.freq_x, self.freq_y = 50.0, 50.0

        # Speaker position state (X, Y, Z)
        self.speaker_x_percent, self.speaker_z_percent = 50.0, 75.0
        self.speaker_y_percent = 50.0  # Default middle height

        # Head position state (X, Y, Z)
        self.head_x_percent, self.head_z_percent = 50.0, 50.0
        self.head_y_percent = 50.0  # Default middle height

        # Other audio parameters
        self.pulse_rate, self.pulse_decay = 1.0, 5.0

        self.root = tk.Tk()
        self.point_size = 8
        self.tracking_mode = False
        self.last_valid_point = None
        self.last_valid_rotation = None
        self.tracker_connected = True

        self.setup_ui()

        # --- NEW: Setup ACEG Swarm ---
        self._set_initial_spatial_chord()

        self.start_audio()
        self.update_display_loop()

    # --- NEW: Frequency Helper Methods ---
    def _freq_to_erb(self, freq):
        """Converts frequency (Hz) to the ERB scale."""
        return 21.4 * math.log10(1 + 0.00437 * freq)

    def _freq_to_percent(self, freq):
        """Converts frequency (Hz) to the 0-100% Y-axis value."""
        min_f, max_f = 20.0, min(20000.0, self.sample_rate / 2)
        min_e = self._freq_to_erb(min_f)
        max_e = self._freq_to_erb(max_f)
        erb = max(min_e, min(max_e, self._freq_to_erb(freq)))
        return (erb - min_e) / (max_e - min_e) * 100.0

    def _set_initial_spatial_chord(self):
        """Sets up 256 speakers with ACEG notes (30Hz-20kHz) and randomized phases."""
        print(f"Generating ACEG Chord Spectrum (30Hz-20kHz)...")

        # 1. Generate Valid ACEG Notes
        valid_freqs = []
        base_a = 27.5/2
        for octave in range(6):
            octave_base = base_a * (2 ** octave)
            # Intervals: 0(A), 3(C), 7(E), 10(G)
            for semitone in [3, 7, 10]:
                freq = octave_base * (2 ** (semitone / 12.0))
                if 30.0 <= freq <= 20000.0:
                    valid_freqs.append(freq)

        # 2. Assign Frequencies
        freq_x = np.random.uniform(40.0, 60.0, self.num_speakers)
        freq_y = np.zeros(self.num_speakers)
        rng = np.random.default_rng()

        for i in range(self.num_speakers):
            note_freq = valid_freqs[i % len(valid_freqs)]
            # Slight detune (+/- 0.2%)
            detune = 1.0 + rng.uniform(-0.002, 0.002)
            freq_y[i] = self._freq_to_percent(note_freq * detune)

        # Apply Frequencies
        coords = torch.from_numpy(np.stack([freq_x, freq_y], axis=1)).to(
            device=self.audio_engine.device, dtype=self.audio_engine.dtype
        )
        self.audio_engine.set_frequency_targets(coords)
        self.freq_x, self.freq_y = freq_x[0], freq_y[0]

        # 3. Randomize Pulse Phases (0.0s - 0.5s)
        pulse_rate = 1.0
        random_delays = torch.rand(self.num_speakers, device=self.audio_engine.device) * 0.5
        self.audio_engine.pulse_phase[:] = random_delays * pulse_rate * (2 * np.pi)
        self.audio_engine.pulse_decay_rate[:] = 2.3
        self.audio_engine.pulse_rate_hz[:] = 0.5

        # 4. Set Spatial Positions (Spiral Layout)
        room_dims = self.audio_engine.space_effect.room_dimensions.cpu().numpy()
        positions = np.zeros((self.num_speakers, 3))

        for i in range(self.num_speakers):
            theta = i * 2.3999632  # Golden Angle
            r = 2.0 + (i / self.num_speakers) * 12.0

            wx = r * np.cos(theta)
            wz = r * np.sin(theta)

            px = (wx / (room_dims[0] / 2) + 1.0) * 50.0
            pz = (wz / (room_dims[2] / 2) + 1.0) * 50.0

            self.audio_engine.set_speaker_position(i, px, pz, pz)
            positions[i] = [wx, 0.0, wz]

        # 5. Amplitude Normalization
        head_pos = torch.zeros(3, device=self.audio_engine.device, dtype=self.audio_engine.dtype)
        spk_pos = torch.from_numpy(positions).to(device=self.audio_engine.device, dtype=self.audio_engine.dtype)

        estimated_gains = self.audio_engine.space_effect.estimate_peak_gain(head_pos, spk_pos)
        linear_amps = torch.clamp(1.0 / (estimated_gains + 1e-6), min=0.01, max=5.0) * 0.2
        self.audio_engine.set_amplitudes(linear_amps)

    # --- MODIFIED: Audio Callback ---
    def audio_callback(self, outdata, frames, time, status):
        with torch.cuda.stream(self.cuda_stream):
            out = self.audio_engine.process(frames)

            # --- MODIFIED: Mean and Gain Boost ---
            # Take mean across speakers (Dim 0) -> (Frames, 2)
            # Multiply by 20.0 to fix silence
            out_mixed = out.mean(dim=0) * 20.0

            self._outdata_tensor.data = torch.from_numpy(outdata)
            self.cuda_stream.synchronize()
            self._outdata_tensor.copy_(out_mixed, non_blocking=False)

        # --- MODIFIED: Save to file ---
        if self.wav_file:
            self.wav_file.write(outdata)

        return outdata

    # --- MODIFIED: Start Audio ---
    def start_audio(self):
        # Open file
        try:
            self.wav_file = sf.SoundFile(
                self.output_filename, mode='w', samplerate=self.sample_rate,
                channels=2, subtype='FLOAT'
            )
            print(f"Recording to: {self.output_filename}")
        except Exception as e:
            print(f"File Error: {e}")

        self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=2,
                                      blocksize=self.block_size, dtype=np.float32,
                                      latency='low', callback=self.audio_callback)
        self.stream.start()
        print(f"Audio Engine Started: {self.sample_rate}Hz, {self.block_size} samples")

    # --- MODIFIED: On Closing ---
    def on_closing(self):
        print("Closing...")
        if self.stream:
            self.stream.stop()
            self.stream.close()

        if self.wav_file:
            self.wav_file.close()
            print(f"File saved: {self.output_filename}")

        self.root.destroy()

    def run(self):
        self.root.mainloop()

    # =========================================================================
    # ORIGINAL UI METHODS BELOW (Unchanged)
    # =========================================================================
    def create_grid_canvas_with_labels(self, parent, title, width=280, height=280,
                                       highlight_color='#00ffff', grid_color='#006666',
                                       x_label="X", y_label="Y"):
        """Create a compact labeled grid canvas."""
        frame = tk.Frame(parent, bg='#0a0a0a')

        # Title - more compact
        title_widget = tk.Label(frame, text=title, font=('Arial', 11, 'bold'),
                                fg='#ffffff', bg='#0a0a0a')
        title_widget.pack(pady=(0, 3))

        # Canvas with axis labels
        canvas_frame = tk.Frame(frame, bg='#0a0a0a')
        canvas_frame.pack()

        # Y-axis label (left side, vertical)
        y_label_widget = tk.Label(canvas_frame, text=y_label, font=('Arial', 8),
                                  fg='#888888', bg='#0a0a0a', width=8, anchor='e')
        y_label_widget.pack(side='left', padx=(0, 2))

        # Canvas container
        canvas_col = tk.Frame(canvas_frame, bg='#0a0a0a')
        canvas_col.pack(side='left')

        # Main canvas - smaller size
        canvas = tk.Canvas(canvas_col, width=width, height=height, bg='#111111',
                           highlightthickness=1, highlightbackground=highlight_color)
        canvas.pack()

        # X-axis label (bottom)
        x_label_widget = tk.Label(canvas_col, text=x_label, font=('Arial', 8),
                                  fg='#888888', bg='#0a0a0a')
        x_label_widget.pack(pady=(1, 0))

        # Draw grid with more lines for better precision
        self.draw_grid(canvas, color=grid_color, width=width, height=height)

        # Store for dynamic updates
        canvas.x_label_widget = x_label_widget
        canvas.y_label_widget = y_label_widget
        return frame, canvas

    def create_controllable_point(self, canvas, x_percent, y_percent, color='#ff6b6b',
                                  click_callback=None, drag_callback=None):
        """Create a draggable point on a canvas."""
        canvas_x, canvas_y = self.grid_to_canvas(x_percent, y_percent)
        point = canvas.create_oval(canvas_x - self.point_size, canvas_y - self.point_size,
                                   canvas_x + self.point_size, canvas_y + self.point_size,
                                   fill=color, outline='#ffffff', width=2)

        if click_callback:
            canvas.bind('<Button-1>', click_callback)
        if drag_callback:
            canvas.bind('<B1-Motion>', drag_callback)

        return point

    def setup_ui(self):
        self.root.title("Hyper-Realistic Spatial Audio Engine v9 [HRIR + Reverb]")
        self.root.configure(bg='#0a0a0a')
        self.root.geometry("1200x900")

        main_frame = tk.Frame(self.root, bg='#0a0a0a')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Title
        title_label = tk.Label(main_frame, text="Real-time Physical Acoustics + Spatial Audio Simulation",
                               font=('Arial', 18, 'bold'), fg='#00ffff', bg='#0a0a0a')
        title_label.pack(pady=(0, 10))

        tracking_frame = tk.Frame(main_frame, bg='#1a1a1a')
        tracking_frame.pack(fill='x', pady=5)

        # Tracking mode toggle
        self.tracking_button = tk.Button(
            tracking_frame,
            text="TRACKING MODE: ON",
            font=('Arial', 12, 'bold'),
            fg='#00ff00',
            bg='#222222',
            activebackground='#333333',
            command=self.toggle_tracking_mode,
            width=25
        )
        self.tracking_button.pack(side='left', padx=10)

        # Connection status
        self.connection_label = tk.Label(
            tracking_frame,
            text="● CONNECTED",
            font=('Arial', 12, 'bold'),
            fg='#00ff00',
            bg='#1a1a1a'
        )
        self.connection_label.pack(side='left', padx=20)

        # Master gain control
        gain_frame = tk.Frame(main_frame, bg='#1a1a1a')
        gain_frame.pack(fill='x', pady=10)

        gain_label = tk.Label(gain_frame, text="Master Gain (db)", font=('Arial', 12, 'bold'),
                              fg='#ff6b6b', bg='#1a1a1a', width=15, anchor='w')
        gain_label.pack(side='left', padx=10, pady=5)

        self.gain_slider = tk.Scale(gain_frame, from_=-60.0, to=60.0, resolution=0.01,
                                    orient='horizontal', length=400, command=self.on_gain_change,
                                    bg='#222222', fg='#ff6b6b', troughcolor='#111111',
                                    highlightthickness=0)
        self.gain_slider.set(self.audio_engine.master_gain)
        self.gain_slider.pack(side='left', expand=True, pady=5)

        # Canvas controls frame
        canvas_frame = tk.Frame(main_frame, bg='#0a0a0a')
        canvas_frame.pack(pady=(0, 20))

        # 1. Frequency Control Grid
        self.freq_frame, self.freq_canvas = self.create_grid_canvas_with_labels(
            canvas_frame, "Sound Source (Tone)", highlight_color='#00ffff',
            grid_color='#006666', x_label="Freq X (0-100%)", y_label="Freq Y")
        self.freq_frame.pack(side='left', padx=(0, 20))

        self.freq_point = self.create_controllable_point(
            self.freq_canvas, self.freq_x, self.freq_y, color='#ff6b6b',
            click_callback=self.on_freq_click, drag_callback=self.on_freq_drag)

        # 2. Speaker and Head Position Grid
        self.position_frame, self.position_canvas = self.create_grid_canvas_with_labels(
            canvas_frame, "Speaker (Green) & Head (Pink) Position", highlight_color='#ff6600',
            grid_color='#660033', x_label="X (0-40.2m)", y_label="Z (0-59.5m)")
        self.position_frame.pack(side='left')
        self.head_arrow = None
        self.create_head_indicator()

        # Create both points on the same canvas
        self.speaker_point = self.create_controllable_point(
            self.position_canvas, self.speaker_x_percent, self.speaker_z_percent,
            color='#66ff66', click_callback=self.on_position_click,
            drag_callback=self.on_position_drag)

        self.head_point = self.create_controllable_point(
            self.position_canvas, self.head_x_percent, self.head_z_percent,
            color='#ff66ff', click_callback=self.on_position_click,
            drag_callback=self.on_position_drag)

        # Y-axis controls for speaker and head
        y_controls_frame = tk.LabelFrame(main_frame, text="Y-Axis Controls (Height)",
                                         font=('Arial', 12, 'bold'), fg='white',
                                         bg='#1a1a1a', pady=10, padx=10)
        y_controls_frame.pack(fill='x', expand=True, pady=(20, 10))

        # Speaker Y control
        speaker_y_frame = tk.Frame(y_controls_frame, bg='#1a1a1a')
        speaker_y_label = tk.Label(speaker_y_frame, text="Speaker Height",
                                   font=('Arial', 12), fg='#66ff66', bg='#1a1a1a',
                                   width=15, anchor='w')
        speaker_y_label.pack(side='left', padx=10)

        self.speaker_y_slider = tk.Scale(speaker_y_frame, from_=0, to=100,
                                         resolution=1, orient='horizontal', length=400,
                                         command=self.on_speaker_y_change, bg='#222222',
                                         fg='#66ff66', troughcolor='#111111',
                                         highlightthickness=0)
        self.speaker_y_slider.set(self.speaker_y_percent)
        self.speaker_y_slider.pack(side='left', expand=True)
        speaker_y_frame.pack(fill='x', pady=5)

        # Head Y control
        head_y_frame = tk.Frame(y_controls_frame, bg='#1a1a1a')
        head_y_label = tk.Label(head_y_frame, text="Head Height",
                                font=('Arial', 12), fg='#ff66ff', bg='#1a1a1a',
                                width=15, anchor='w')
        head_y_label.pack(side='left', padx=10)

        self.head_y_slider = tk.Scale(head_y_frame, from_=0, to=100,
                                      resolution=1, orient='horizontal', length=400,
                                      command=self.on_head_y_change, bg='#222222',
                                      fg='#ff66ff', troughcolor='#111111',
                                      highlightthickness=0)
        self.head_y_slider.set(self.head_y_percent)
        self.head_y_slider.pack(side='left', expand=True)
        head_y_frame.pack(fill='x', pady=5)

        # Source controls
        source_controls_frame = tk.LabelFrame(main_frame, text="Source Controls",
                                              font=('Arial', 12, 'bold'), fg='white',
                                              bg='#1a1a1a', pady=10, padx=10)
        source_controls_frame.pack(fill='x', expand=True, pady=(20, 10))

        # Pulse rate control
        pulse_rate_frame = tk.Frame(source_controls_frame, bg='#1a1a1a')
        pulse_rate_label = tk.Label(pulse_rate_frame, text="Pulse Rate (Hz)",
                                    font=('Arial', 12), fg='white', bg='#1a1a1a',
                                    width=15, anchor='w')
        pulse_rate_label.pack(side='left', padx=10)

        self.pulse_rate_slider = tk.Scale(pulse_rate_frame, from_=0.2, to=10.0,
                                          resolution=0.1, orient='horizontal', length=400,
                                          command=self.on_pulse_rate_change, bg='#222222',
                                          fg='#00ffff', troughcolor='#111111',
                                          highlightthickness=0)
        self.pulse_rate_slider.set(self.pulse_rate)
        self.pulse_rate_slider.pack(side='left', expand=True)
        pulse_rate_frame.pack(fill='x')

        # Pulse decay control
        pulse_decay_frame = tk.Frame(source_controls_frame, bg='#1a1a1a')
        pulse_decay_label = tk.Label(pulse_decay_frame, text="Pulse Decay",
                                     font=('Arial', 12), fg='white', bg='#1a1a1a',
                                     width=15, anchor='w')
        pulse_decay_label.pack(side='left', padx=10)

        self.pulse_decay_slider = tk.Scale(pulse_decay_frame, from_=0.5, to=20.0,
                                           resolution=0.1, orient='horizontal', length=400,
                                           command=self.on_pulse_decay_change, bg='#222222',
                                           fg='#00ffff', troughcolor='#111111',
                                           highlightthickness=0)
        self.pulse_decay_slider.set(self.pulse_decay)
        self.pulse_decay_slider.pack(side='left', expand=True)
        pulse_decay_frame.pack(fill='x')

        # Room controls
        reverb_controls_frame = tk.LabelFrame(main_frame, text="Room Controls",
                                              font=('Arial', 12, 'bold'), fg='white',
                                              bg='#1a1a1a', pady=10, padx=10)
        reverb_controls_frame.pack(fill='x', expand=True, pady=10)

        # Room dimension sliders
        self.w_frame, self.w_slider = self.create_exp_slider(
            reverb_controls_frame, "Room Width", 4, 200, self.audio_engine.room_width,
            self.on_room_dim_change)
        self.h_frame, self.h_slider = self.create_exp_slider(
            reverb_controls_frame, "Room Height", 4, 200, self.audio_engine.room_height,
            self.on_room_dim_change)
        self.d_frame, self.d_slider = self.create_exp_slider(
            reverb_controls_frame, "Room Depth", 4, 200, self.audio_engine.room_depth,
            self.on_room_dim_change)

        self.w_frame.pack(fill='x')
        self.h_frame.pack(fill='x')
        self.d_frame.pack(fill='x')

        # Absorption control
        absorb_frame = tk.Frame(reverb_controls_frame, bg='#1a1a1a')
        absorb_label = tk.Label(absorb_frame, text="Wall Absorbance", font=('Arial', 12),
                                fg='white', bg='#1a1a1a', width=15, anchor='w')
        absorb_label.pack(side='left', padx=10)

        self.absorb_slider = tk.Scale(absorb_frame, from_=-20.0, to=0.0, resolution=0.1,
                                      orient='horizontal', length=400, showvalue=False,
                                      command=self.on_absorption_change, bg='#222222',
                                      fg='#00ffff', troughcolor='#111111',
                                      highlightthickness=0)
        self.absorb_slider.set(-6.0)
        self.absorb_slider.pack(side='left', expand=True)

        self.absorb_val_label = tk.Label(absorb_frame, text=f"{-6.0:.1f} dB",
                                         font=('Arial', 12, 'bold'), fg='#00ffff',
                                         bg='#1a1a1a', width=10)
        self.absorb_val_label.pack(side='left', padx=10)
        absorb_frame.pack(fill='x')

        # Info display
        self.info_frame = tk.Frame(main_frame, bg='#0a0a0a')
        self.info_frame.pack(fill='x', pady=10)

        self.pos_label = tk.Label(self.info_frame, text="...", font=('Arial', 10, 'bold'),
                                  fg='#ffffff', bg='#222222', width=30)
        self.pos_label.pack(side='left', expand=True, padx=5)

        self.head_pos_label = tk.Label(self.info_frame, text="...", font=('Arial', 10, 'bold'),
                                       fg='#ffffff', bg='#222222', width=30)
        self.head_pos_label.pack(side='left', expand=True, padx=5)

        self.room_label = tk.Label(self.info_frame, text="...", font=('Arial', 10, 'bold'),
                                   fg='#ffffff', bg='#222222', width=25)
        self.room_label.pack(side='left', expand=True, padx=5)

        self.rt60_label = tk.Label(self.info_frame, text="...", font=('Arial', 10, 'bold'),
                                   fg='#ffffff', bg='#222222', width=15)
        self.rt60_label.pack(side='left', expand=True, padx=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_head_indicator(self):
        """Create arrow showing head position and orientation."""
        # Initial position at center
        cx, cy = self.grid_to_canvas(50, 50)

        # Arrow pointing forward (up in canvas, which is +Z)
        arrow_length = 30
        self.head_arrow = self.position_canvas.create_line(
            cx, cy, cx, cy - arrow_length,
            arrow=tk.LAST, width=3, fill='#ffff00',
            arrowshape=(12, 15, 5)
        )

        # Head position circle
        self.head_circle = self.position_canvas.create_oval(
            cx - 6, cy - 6, cx + 6, cy + 6,
            fill='#ffff00', outline='#ffffff', width=2
        )

    def update_head_indicator(self, x_percent, z_percent, yaw_radians):
        """Update head arrow position and orientation."""
        cx, cy = self.grid_to_canvas(x_percent, z_percent)

        # Arrow points in direction of head orientation
        arrow_length = 30
        # In canvas: y increases downward, but z increases upward in room
        # yaw: 0 = facing +Z (up in canvas, toward smaller y)
        end_x = cx + arrow_length * np.sin(yaw_radians)
        end_y = cy - arrow_length * np.cos(yaw_radians)

        self.position_canvas.coords(self.head_arrow, cx, cy, end_x, end_y)
        self.position_canvas.coords(
            self.head_circle,
            cx - 6, cy - 6, cx + 6, cy + 6
        )

    def toggle_tracking_mode(self):
        """Toggle between tracking and manual control mode."""
        self.tracking_mode = not self.tracking_mode

        if self.tracking_mode:
            self.tracking_button.config(
                text="TRACKING MODE: ON",
                fg='#00ff00'
            )
        else:
            self.tracking_button.config(
                text="TRACKING MODE: OFF",
                fg='#ff6600'
            )

    def update_from_tracker(self):
        """Update positions from visual landmarks tracker."""
        if not self.tracking_mode:
            return

        try:
            # Get speaker position
            point = None  # self.visual_landmarks_pipe.point()

            if point is not None:
                self.last_valid_point = point
                self.tracker_connected = True
                self.connection_label.config(text="● CONNECTED", fg='#00ff00')

                # Update speaker position in audio engine
                # self.audio_engine.set_speaker_position_from_tracker(point)

            else:
                self.tracker_connected = False
                self.connection_label.config(text="● DISCONNECTED", fg='#ff0000')
                # Keep using last valid point
                # if self.last_valid_point is not None:
                #     self.audio_engine.set_speaker_position_from_tracker(self.last_valid_point)

            # Get head rotation
            rotation = self.visual_landmarks_pipe.head_rotation()

            if rotation is not None:
                self.last_valid_rotation = rotation
                self.audio_engine.set_head_rotation(rotation)

                # Extract yaw from rotation matrix for visualization
                # rotation is IMU-to-world, we want yaw in X-Z plane
                # yaw = atan2(R[0,2], R[2,2]) gives rotation around Y axis
                yaw = np.arctan2(rotation[0, 2], rotation[2, 2])

                # Head always at center (0, 0, 0) in display
                self.update_head_indicator(50.0, 50.0, yaw)
            else:
                # Use last valid rotation
                if self.last_valid_rotation is not None:
                    self.audio_engine.set_head_rotation(self.last_valid_rotation)
                    yaw = np.arctan2(self.last_valid_rotation[0, 2],
                                     self.last_valid_rotation[2, 2])
                    self.update_head_indicator(50.0, 50.0, yaw)

        except Exception as e:
            print(f"Tracker update error: {e}")
            self.tracker_connected = False
            self.connection_label.config(text="● ERROR", fg='#ff0000')

    def create_exp_slider(self, parent, label, from_, to, default, command):
        """Create exponential slider for room dimensions."""
        frame = tk.Frame(parent, bg='#1a1a1a')
        label_widget = tk.Label(frame, text=label, font=('Arial', 12), fg='white',
                                bg='#1a1a1a', width=15, anchor='w')
        label_widget.pack(side='left', padx=10)

        log_min, log_max = math.log(from_), math.log(to)

        def scale_to_value(v):
            return math.exp(log_min + (float(v) / 100.0) * (log_max - log_min))

        def value_to_scale(v):
            return (math.log(v) - log_min) / (log_max - log_min) * 100.0

        slider = tk.Scale(frame, from_=0, to=100, orient='horizontal', length=400,
                          showvalue=False, command=lambda v: command(scale_to_value(v)),
                          bg='#222222', fg='#00ffff', troughcolor='#111111',
                          highlightthickness=0)
        slider.set(value_to_scale(default))
        slider.pack(side='left', expand=True)

        val_label = tk.Label(frame, text=f"{default:.1f} m", font=('Arial', 12, 'bold'),
                             fg='#00ffff', bg='#1a1a1a', width=10)
        val_label.pack(side='left', padx=10)

        def update_label(v):
            val = scale_to_value(v)
            val_label.config(text=f"{val:.1f} m")
            command(val)

        slider.config(command=update_label)
        return frame, slider

    def draw_grid(self, canvas, color='#006666', width=280, height=280):
        """Draw grid lines on canvas with proper scaling."""
        # Draw 10 grid lines (creating 9 segments) for better precision
        for i in range(10):
            x = i * (width / 9)
            y = i * (height / 9)
            canvas.create_line(x, 0, x, height, fill=color, width=1)
            canvas.create_line(0, y, width, y, fill=color, width=1)

    def grid_to_canvas(self, x, y, c_width=280, c_height=280):
        """Convert grid percentage to canvas coordinates."""
        return (x / 100.0) * c_width, (1.0 - y / 100.0) * c_height

    def canvas_to_grid(self, cx, cy, c_width=280, c_height=280):
        """Convert canvas coordinates to grid percentage."""
        return (np.clip((cx / c_width) * 100.0, 0, 100),
                np.clip((1.0 - cy / c_height) * 100.0, 0, 100))

    def update_position_labels(self):
        """Update position grid labels based on current room dimensions."""
        room_dim = self.audio_engine.space_effect.room_dimensions
        width, depth = room_dim[0], room_dim[2]
        self.position_canvas.x_label_widget.config(text=f"X (0-{width:.1f}m)")
        self.position_canvas.y_label_widget.config(text=f"Z (0-{depth:.1f}m)")

    def get_closest_point(self, cx, cy):
        """Determine which point is closest to click position."""
        speaker_cx, speaker_cy = self.grid_to_canvas(self.speaker_x_percent, self.speaker_z_percent)
        head_cx, head_cy = self.grid_to_canvas(self.head_x_percent, self.head_z_percent)

        speaker_dist = math.sqrt((cx - speaker_cx) ** 2 + (cy - speaker_cy) ** 2)
        head_dist = math.sqrt((cx - head_cx) ** 2 + (cy - head_cy) ** 2)

        return 'speaker' if speaker_dist < head_dist else 'head'

    def on_gain_change(self, value):
        self.audio_engine.set_master_gain(float(value))

    def on_pulse_rate_change(self, value):
        self.audio_engine.set_pulse_rate(0, float(value))

    def on_pulse_decay_change(self, value):
        self.audio_engine.set_pulse_decay(0, float(value))

    def on_speaker_y_change(self, value):
        self.speaker_y_percent = float(value)
        self.audio_engine.set_speaker_position(0, self.speaker_x_percent, self.speaker_y_percent,
                                               self.speaker_z_percent)

    def on_head_y_change(self, value):
        self.head_y_percent = float(value)
        self.audio_engine.set_head_position_percent(self.head_x_percent, self.head_y_percent, self.head_z_percent)

    def on_room_dim_change(self, value=None):
        w = float(self.w_slider.get()) / 100.0
        h = float(self.h_slider.get()) / 100.0
        d = float(self.d_slider.get()) / 100.0

        map_val = lambda v, v_min, v_max: math.exp(math.log(v_min) + v * (math.log(v_max) - math.log(v_min)))
        self.audio_engine.set_room_dimensions(map_val(w, 4, 200), map_val(h, 4, 200), map_val(d, 4, 200))
        self.update_position_labels()

    def on_absorption_change(self, value):
        db_val = float(value)
        self.audio_engine.set_wall_absorbance(db_val)
        self.absorb_val_label.config(text=f"{db_val:.1f} dB")

    def on_freq_click(self, event):
        self.move_freq_point(event.x, event.y)

    def on_freq_drag(self, event):
        self.move_freq_point(event.x, event.y)

    def on_position_click(self, event):
        if self.tracking_mode:
            return
        self.active_point = self.get_closest_point(event.x, event.y)
        self.move_position_point(event.x, event.y)

    def on_position_drag(self, event):
        if self.tracking_mode:
            return
        if hasattr(self, 'active_point'):
            self.move_position_point(event.x, event.y)

    def move_freq_point(self, cx, cy):
        """Move frequency control point."""
        self.freq_x, self.freq_y = self.canvas_to_grid(cx, cy)
        cx, cy = self.grid_to_canvas(self.freq_x, self.freq_y)
        self.freq_canvas.coords(self.freq_point, cx - self.point_size, cy - self.point_size,
                                cx + self.point_size, cy + self.point_size)

        self.audio_engine.set_frequency_target(0, self.freq_x, self.freq_y)

        min_freq = 20.0
        max_freq = min(20000.0, self.sample_rate / 2)

        min_erb = 21.4 * math.log10(1 + 0.00437 * min_freq)
        max_erb = 21.4 * math.log10(1 + 0.00437 * max_freq)

        normalized_input = self.freq_y / 100.0
        current_erb = min_erb + normalized_input * (max_erb - min_erb)

        current_freq_hz = (10.0 ** (current_erb / 21.4) - 1.0) / 0.00437
        self.freq_canvas.y_label_widget.config(text=f"{current_freq_hz:.1f} Hz")

    def move_position_point(self, cx, cy):
        """Move speaker or head position point based on which is active."""
        x_percent, z_percent = self.canvas_to_grid(cx, cy)
        if self.active_point == 'speaker':
            self.speaker_x_percent, self.speaker_z_percent = x_percent, z_percent
            cx, cy = self.grid_to_canvas(self.speaker_x_percent, self.speaker_z_percent)
            self.position_canvas.coords(self.speaker_point, cx - self.point_size, cy - self.point_size,
                                        cx + self.point_size, cy + self.point_size)
            self.audio_engine.set_speaker_position(0, self.speaker_x_percent, self.speaker_y_percent,
                                                   self.speaker_z_percent)

        elif self.active_point == 'head':
            self.head_x_percent, self.head_z_percent = x_percent, z_percent
            cx, cy = self.grid_to_canvas(self.head_x_percent, self.head_z_percent)
            self.position_canvas.coords(self.head_point, cx - self.point_size, cy - self.point_size,
                                        cx + self.point_size, cy + self.point_size)
            self.audio_engine.set_head_position_percent(self.head_x_percent, self.head_y_percent, self.head_z_percent)

    def update_display_loop(self):
        """Update display information periodically."""
        self.update_from_tracker()

        spk_pos = self.audio_engine.space_effect.speaker_positions

        if self.tracking_mode and self.last_valid_point is not None:
            tracked = self.last_valid_point
            self.pos_label.config(
                text=f"Speaker (Tracked): ({tracked[0]:.2f}, {tracked[1]:.2f}, {tracked[2]:.2f})m"
            )
        else:
            self.pos_label.config(
                text=f"Speaker (Manual): ({spk_pos[0][0]:.1f}, {spk_pos[0][1]:.1f}, {spk_pos[0][2]:.1f})m"
            )

        self.head_pos_label.config(text=f"Head: (0.0, 0.0, 0.0)m")
        self.pos_label.config(text=f"Speaker: ({spk_pos[0][0]:.1f}, {spk_pos[0][1]:.1f}, {spk_pos[0][2]:.1f})m")

        room_dim = self.audio_engine.space_effect.room_dimensions
        head_x = (self.head_x_percent / 100.0) * room_dim[0] - room_dim[0] / 2
        head_y = (self.head_y_percent / 100.0) * room_dim[1] - room_dim[1] / 2
        head_z = (self.head_z_percent / 100.0) * room_dim[2] - room_dim[2] / 2
        self.head_pos_label.config(text=f"Head: ({head_x:.1f}, {head_y:.1f}, {head_z:.1f})m")
        self.room_label.config(text=f"Room: {room_dim[0]:.1f}x{room_dim[1]:.1f}x{room_dim[2]:.1f}m")

        rt60 = self.audio_engine.space_effect.compute_rt60()
        rt60_text = f"RT60: {rt60:.2f}s" if rt60 < 1000 else "RT60: ∞"
        self.rt60_label.config(text=rt60_text)
        self.root.after(50, self.update_display_loop)


def main():
    app = PulserUI()
    app.run()


if __name__ == "__main__":
    main()