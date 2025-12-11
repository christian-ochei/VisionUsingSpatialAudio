from process_depthmap import DepthFocus
from tracker import VisualLandmarksTrackerPipe
import threading
import time
import torch
import numpy as np
from imu_sense import IMUSensing
from highp_spatial_landmarks import RealtimeSpatialPulseEngine
import sounddevice as sd
from line_profiler_pycharm import profile
from vsa_visualizer import VSAVisualizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = False
torch.cuda.empty_cache()

GREEN = '\033[92m'
ENDC = '\033[0m'
WHITE = '\033[97m'
RESET = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
BLUE = '\033[94m'

# 512 is the perfect block size
class VSAPlayback:
    def __init__(
        # Yes we should be able to run 512 landmarks at 44100 | Blazing fast with all Compute |
        # Use 256 during presentation to be Safe
        self, dtype=torch.float32, num_landmarks=256, block_size=512, sample_rate=12000,
        is_portrait_reference=True, use_head_mounted_imu=False, device='cuda',
        visualize_landmarks=False, verbose=True
    ):
        # Uses SensorGram
        self.num_landmarks = num_landmarks
        self.pointer_landmark = torch.tensor([0.2867, -0.0446, -0.9570, 0, 0, 1.0], dtype=dtype, device=device)
        self.z_vector = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        self.pointer_landmark_norm = None if self.pointer_landmark is None else self.pointer_landmark / torch.linalg.norm(
            self.pointer_landmark[:3], dim=-1
        )
        self.verbose = verbose
        self.visualize_landmarks = visualize_landmarks
        self.sensor_reference_mode = 'portrait' if is_portrait_reference else 'landscape'
        self.view_sensor = IMUSensing(mode=self.sensor_reference_mode, dtype=dtype, device=device)
        self.use_head_mounted_imu = use_head_mounted_imu
        if use_head_mounted_imu:
            self.head_mounted_sensor = IMUSensing(mode='flat', dtype=dtype, device=device)
            head_mounted_bias = torch.tensor([0.09444, 0.3148, -0.06], dtype=dtype, device=device)
        else:
            self.head_mounted_sensor = self.view_sensor
            head_mounted_bias = None

        self.offset_to_ear = torch.tensor([0.0, -0.09444, 0.0], dtype=dtype, device=device)
        self.head_mounted_bias = head_mounted_bias
        self.landmarks = None
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.stream = None
        self.realtime_stream = torch.cuda.Stream(priority=-3)
        self.low_priority_stream = torch.cuda.Stream(priority=0)
        self.alive_mask = torch.ones(num_landmarks, dtype=torch.bool, device=device)
        self.visual_landmarks_pipe = VisualLandmarksTrackerPipe(
            num_landmarks=num_landmarks, dtype=dtype, view_sensor=self.view_sensor, device=device,
            head_mounted_sensor=self.head_mounted_sensor,
            is_portrait_mode=is_portrait_reference, dsp=None, verbose=verbose
        )
        # This is done after for faster imports Due to VisualLandmark's Async Nature
        self.audio_engine = RealtimeSpatialPulseEngine(
            sample_rate=self.sample_rate, block_size=block_size,
            num_speakers=num_landmarks, dtype=dtype, device=device, stream=self.realtime_stream
        )
        self.visual_landmarks_pipe.dsp = self.audio_engine
        self.should_reassign_phases_at_sync = False
        self.ones_is_updated_mask = torch.ones(num_landmarks, dtype=torch.bool, device=device)
        self.last_landmarks_was_none = True
        self.is_updated_mask_sync = torch.zeros(num_landmarks, dtype=torch.bool, pin_memory=True)
        self.pinned_buffer = torch.empty((block_size, 2), dtype=dtype, pin_memory=True)
        self.master_chunk = torch.empty((block_size, 2), dtype=dtype, device=self.audio_engine.device)
        self.landmarks_norm_buf_comp = torch.empty((num_landmarks, 3), dtype=dtype, device=self.audio_engine.device)
        self.landmarks_norm_buf_rel = torch.empty((num_landmarks, 3), dtype=dtype, device=self.audio_engine.device)
        self.distances_buf = torch.empty(num_landmarks, dtype=dtype, device=self.audio_engine.device)
        self.pulse_rate_buf = torch.empty(num_landmarks, dtype=dtype, device=self.audio_engine.device)
        self.landmarks_sub_buf = torch.empty((num_landmarks, 3), dtype=dtype, device=self.audio_engine.device)
        self.freq_xy_offset_buf = torch.empty((num_landmarks, 2), dtype=dtype, device=self.audio_engine.device)
        self.phase_add_buf = torch.empty(num_landmarks, dtype=dtype, device=self.audio_engine.device)
        self.pulse_window_position_seconds = 0
        self.pulse_window_seconds = 0.5
        self.is_reinitializing = False
        self.is_first_chunk = True
        self.out_pinned = torch.empty((block_size, 2), dtype=torch.float32, pin_memory=True)
        self.out_numpy = self.out_pinned.numpy()
        self.last_vis_regeneration_time = time.time()
        self.time_since_last_regeneration = time.time()
        self.true_latency = time.time()
        self.frame_count = 0
        self._pinned_rotation = torch.empty(
            (3, 3), dtype=torch.float32, pin_memory=True
        )
        self._pinned_position = torch.empty(
            (3,), dtype=torch.float32, pin_memory=True
        )
        self._pinned_landmarks = torch.empty(
            (self.num_landmarks, 6),
            dtype=torch.float32,
            pin_memory=True
        )
        self._pinned_gain = torch.empty(
            (1,), dtype=torch.float32, pin_memory=True
        )
        self._pinned_room_dims = torch.empty((3,), dtype=torch.float32, pin_memory=True)
        self._pinned_absorb = torch.empty(
            (self.visual_landmarks_pipe.absorbance().numel(),),
            dtype=torch.float32,
            pin_memory=True
        )
        # dedicated CUDA streams (never use default!)
        self._vis_stream = torch.cuda.Stream()
        self._copy_stream = torch.cuda.Stream()

    @torch.inference_mode()
    def run_pipeline(self):
        # We get live realtime 3D landmarks into an (N,3) buffer
        # We also have a realtime mask for DSP when to disable then re-enable doppler
        self.start_audio()
        if self.visualize_landmarks:
            self.run_visualizer_non_blocking()
        with torch.cuda.stream(self.low_priority_stream):
            self.visual_landmarks_pipe.start_background_update_blocking()

    def run_visualizer_non_blocking(self):
        visualizer_thread = threading.Thread(target=self.run_visualizer_blocking, args=(), daemon=True)
        visualizer_thread.start()

    @torch.inference_mode()
    def run_visualizer_blocking(self):
        visualizer = VSAVisualizer(
            intrinsics=self.visual_landmarks_pipe.K_raw, width=1200, height=800,
            mode=self.sensor_reference_mode,
            num_landmarks=self.num_landmarks
        )
        self.last_vis_regeneration_time = time.time()
        while self.visual_landmarks_pipe.visualizer_available and visualizer.running:
            time.sleep(0.033)
            # ===============================
            # LAUNCH GPU→CPU ASYNC TRANSFERS
            # ===============================
            with torch.cuda.stream(self._copy_stream):
                # rotation 3x3
                rot_gpu = self.visual_landmarks_pipe.head_rotation()
                self._pinned_rotation.copy_(rot_gpu, non_blocking=True)

                # position
                geometry = self.visual_landmarks_pipe.room_head_geometry()
                head_pos_gpu = geometry[3]
                self._pinned_position.copy_(head_pos_gpu, non_blocking=True)

                # gains
                gain_gpu = self.audio_engine.current_gains[0]
                self._pinned_gain.copy_(gain_gpu, non_blocking=True)

                # landmarks
                if self.landmarks is not None:
                    self._pinned_landmarks.copy_(self.landmarks, non_blocking=True)

                # room dimensions async copy
                self._pinned_room_dims[0].copy_(geometry[0], non_blocking=True)
                self._pinned_room_dims[1].copy_(geometry[1], non_blocking=True)
                self._pinned_room_dims[2].copy_(geometry[2], non_blocking=True)

                # absorption async copy
                absorb_gpu = self.visual_landmarks_pipe.absorbance()
                self._pinned_absorb.copy_(absorb_gpu.reshape(-1), non_blocking=True)
            # ===============================
            # WAIT **ON CPU SIDE ONLY** FOR COMPLETION
            # (Does NOT stall CUDA compute streams)
            # ===============================
            self._copy_stream.synchronize()
            # ===============================
            # NOW SAFE TO CONVERT TO NUMPY
            # ===============================
            rotation_matrix = self._pinned_rotation.numpy()
            head_position = self._pinned_position.numpy()
            gain = self._pinned_gain.numpy()
            landmarks = None
            if self.landmarks is not None:
                landmarks = self._pinned_landmarks.numpy()
            # ===============================
            # CPU-ONLY WORK, NO GPU SYNC
            # ===============================
            room_dims_np = self._pinned_room_dims.numpy()
            absorb_np = self._pinned_absorb.numpy()
            visualizer.set_room_dimensions(room_dims_np.tolist())
            visualizer.set_absorption(absorb_np)
            if landmarks is not None:
                lm = landmarks.copy()
                lm[..., :3] += head_position
                visualizer.set_landmarks(lm)

            visualizer.landmark_color[2] = 255 - np.clip(gain * 255, 0.0, 254)
            pos = self.bias_head_position(
                head_position
            )
            visualizer.render(
                position=pos,
                rotation=rotation_matrix
            )
            self.view_sensor.update()
            if self.use_head_mounted_imu:
                self.head_mounted_sensor.update()

            t = time.time()
            if t > self.last_vis_regeneration_time + 7:
                self.visual_landmarks_pipe.reinitialize()
                self.last_vis_regeneration_time = t
        visualizer.close()

    def bias_head_position(self, head_position):
        if self.head_mounted_bias is None:
            return head_position
        biased_position = self.head_mounted_bias + self.calculate_rotation_based_bias()
        return head_position + biased_position

    def calculate_rotation_based_bias(self):
        return self.visual_landmarks_pipe.head_rotation() @ self.offset_to_ear

    def start_audio(self):
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate, channels=2, blocksize=self.block_size, dtype=np.float32, latency='low',
            callback=self.audio_callback
        )
        self.stream.start()
        self.verb_print(f"Audio Engine Started: {self.sample_rate}Hz, {self.block_size} samples")

    def close(self):
        self.visual_landmarks_pipe.close()

    # Audio system methods
    @torch.inference_mode()
    @profile
    def audio_callback(self, outdata, frames, _, __):
        with torch.cuda.stream(self.realtime_stream):
            # The audio engine directly receives the (N,6) result as input
            self.landmarks, is_updated_mask = self.visual_landmarks_pipe.landmarks()
            # is_updated_mask is always true until dsp makes it false
            # When is_updated_mask is made, we clear all masks instantly to restart a new audio pulse
            # This is seamless because latency is perfectly timed
            geometry = self.visual_landmarks_pipe.room_head_geometry()
            head_position = geometry[3]
            force_reset_window = False
            if self.landmarks is not None:
                if self.pointer_landmark is not None:
                    self.landmarks[0] = self.pointer_landmark
                head_compensated_positions = torch.add(
                    self.landmarks[..., :3], head_position, out=self.landmarks_norm_buf_comp
                )
                head_relative_positions = torch.sub(
                    self.landmarks[..., :3], head_position, out=self.landmarks_norm_buf_rel
                )
                if self.pointer_landmark is not None:
                    head_relative_positions[0] = self.pointer_landmark[:3]
                    head_compensated_positions[0] = self.pointer_landmark[:3]

                # Use X/Z distances
                distances = torch.linalg.norm(head_relative_positions[:, ::2], dim=-1, out=self.distances_buf)
                if self.pointer_landmark is not None:
                    distances[0] = 0.0

                if self.last_landmarks_was_none:
                    # We do this to prevent Doppler Spikes
                    is_updated_mask = self.ones_is_updated_mask
                    force_reset_window = True
                self.last_landmarks_was_none = False
                # # Interleave HRIR resets and speaker resets to prevent stalls
                head_rotation = self.visual_landmarks_pipe.head_rotation()
                self.audio_engine.set_head_rotation(head_rotation)
                # This does not update any parameter

                # We don't need to set head position any more
                # self.audio_engine.set_head_position(
                #     self.bias_head_position(head_position), should_update_params=False
                # )
                # # We use mod to maintain async GPU operations and prevent assignments from slowing down DSP
                # This is tested to work and work flawlessly
                # self.audio_engine.set_frequency_targets(
                #     torch.mul(self.landmarks[..., 3:5], 50.0, out=self.freq_xy_offset_buf)
                #     .add_(50.0)
                #     .clamp_(min=0.0 + 20, max=99.9 - 20)
                # )

                # Frequency dependent linear distance
                # Use global Y for frequency | Use AMPS for Distance
                # self.freq_xy_offset_buf[:, 1] = 100.0 - distances * 10 - 10 # 10 meters is the absolute furthest
                self.freq_xy_offset_buf[:, 1] = 100 - (self.landmarks[:, 1] + 2) * 30
                self.freq_xy_offset_buf[:, 0] = torch.abs(self.landmarks[..., 3]) # The X Frequency Component
                # self.freq_xy_offset_buf[:, 0] = torch.linalg.norm(self.landmarks[..., 3:5], dim=-1)
                self.freq_xy_offset_buf[:, 0] *= 100
                self.freq_xy_offset_buf[:, 0] += 50

                # We Want to difference up from down consistently
                self.audio_engine.set_frequency_targets(
                    self.freq_xy_offset_buf.clamp_(min=0.0 + 20, max=99.9 - 20)
                )
                # We don't need these anymore
                # if self.frame_count % 13 == 0:
                #     self.audio_engine.set_wall_absorbance(self.visual_landmarks_pipe.absorbance())
                # if self.frame_count % 17 == 0 and self.frame_count % 13 != 0:
                #     self.audio_engine.set_room_dimensions(geometry[0], geometry[1], geometry[2])
                if self.frame_count % 17 != 0 and self.frame_count % 13 != 0:
                    # We don't stagger because its more efficient like this due to nature of indexing and reflection transitions
                    pulse_rates = torch.add(distances, 1.0, out=self.pulse_rate_buf).log2_().mul_(0.25)
                    # Pointer landmark | Use this to know orientation
                    if self.pointer_landmark is not None:
                        pulse_rates[0] = 1.0
                    # self.audio_engine.set_pulse_rates(torch.div(1.0, pulse_rates, out=self.pulse_rate_buf))
                    amps = DepthFocus.depth_to_amplitude(distances).pow_(2.0)*5.9 # .div_(
                    amps = amps.clamp_(max=1.0)
                    if self.pointer_landmark is not None:
                        amps[0] = (
                            self.pointer_landmark[5] * (
                                self.pointer_landmark_norm[None, :3] *
                                (self.z_vector[None] @ head_rotation)).clamp_(min=0, max=1).sum().pow_(4.4)
                        )
                        self.visual_landmarks_pipe.alive_mask[0] = True

                    # Do not play audio of dead points - This causes overwhelming and wrong perception
                    self.audio_engine.set_amplitudes(amps * self.visual_landmarks_pipe.alive_mask)
                    self.audio_engine.set_speaker_positions(
                        head_compensated_positions
                    )
            est_latency = self.visual_landmarks_pipe.reinitialization_latency()
            if not self.is_reinitializing and (
                self.pulse_window_position_seconds > self.pulse_window_seconds - 0.001 - min(
                est_latency, self.pulse_window_seconds
            )):
                self.verb_print(f"{RED}{BOLD}[GENERATING DSP]{RESET}")
                self.time_since_last_regeneration = time.time()
                self.visual_landmarks_pipe.reinitialize()
                self.is_reinitializing = True
                self.is_first_chunk = False

            if self.visual_landmarks_pipe.clear_updated_mask():
                self.should_reassign_phases_at_sync = True
                self.is_updated_mask_sync[is_updated_mask] = True
                # Phases are played based on depth, furthest plays last
                self.true_latency = time.time() - self.time_since_last_regeneration
                self.verb_print(f"{GREEN}{BOLD}[Regenerated DSP] | est-latency: {est_latency} | "
                      f"true-latency: {self.true_latency}{RESET}")
                # self.pulse_window_seconds = self.make_pulse_window()
                # if self.landmarks is not None:
                #     # Amp is set here
                #     head_compensated_positions = torch.add(
                #         self.landmarks[..., :3], head_position, out=self.landmarks_norm_buf
                #     )
                #     self.audio_engine.set_speaker_positions(
                #         head_compensated_positions,
                #         indexing_values=is_updated_mask
                #     )
                #     head_relative_positions = torch.sub(
                #         self.landmarks[..., :3], head_position, out=self.landmarks_sub_buf
                #     )
                #     distances = torch.linalg.norm(head_relative_positions, dim=-1)
                #     amps = DepthFocus.depth_to_amplitude(distances)
                #     self.audio_engine.set_amplitudes(amps, indexing_values=is_updated_mask)
            elif not force_reset_window:
                # Do not reset hard assign dopers frequencies, gains ... if we are not under reset window
                is_updated_mask = None

            if (self.should_reassign_phases_at_sync and
                self.pulse_window_position_seconds > self.pulse_window_seconds * 2
            ):
                self.should_reassign_phases_at_sync = False
                self.verb_print(f"{BLUE}{BOLD}[REASSIGNED DSP PHASES]{RESET}")
                self.is_reinitializing = False
                # We don't need to reassign phases anymore
                if self.landmarks is not None:
                    # phases = torch.add(self.landmarks[:, 2], 0.8, out=self.phase_add_buf.resize_(0))
                    # phases = torch.add(self.landmarks[self.is_updated_mask_sync, 2], 0.8, out=self.phase_add_buf.resize_(0))
                    # phases.clamp_(min=1.0)
                    # phases = phases.log_()
                    # phases *= 0.4
                    # Always call this, AT exactly the new impulse window
                    # BUT disable doppler, so doppler is disabled twice
                    distances *= -0.4 # 10 meters is exactly 1 second
                    self.audio_engine.pulse_phase[:] = distances
                self.is_updated_mask_sync[:] = False
                self.pulse_window_position_seconds = 0.0

            # We subtract to find time to next pulse
            out = self.audio_engine.process(frames, reset_index_values=is_updated_mask)
            # Make we update right after process, for Maximum CUDA / CPU Saturation
            # Sensor update is blazing fast at 1kz - 20kz
            self.view_sensor.update()
            if self.use_head_mounted_imu:
                self.head_mounted_sensor.update()
            self.pulse_window_position_seconds += frames / self.sample_rate
            out_tensor = torch.from_numpy(outdata)
            if self.landmarks is None:
                self.master_chunk[:] = out[0]
            else:
                # Sum when there are enough landmarks. To prevent constructive amplification of pain
                out[out.isnan().any(dim=-1).any(dim=-1)] = 0.0
                torch.sum(out, out=self.master_chunk.mul_(0.1), dim=0)
            # Never make non_blocking true, this will overwhelm the GPU Queue and cause blue screen
            self.master_chunk.mul_(0.2)
            # If audio ever sounds weird or ring, this comment out is most likely the cause
            # self.realtime_stream.synchronize()
            out_tensor.copy_(self.master_chunk, non_blocking=False)
            self.frame_count += 1
            if self.frame_count % 40 == 0:
                self.audio_engine.reset_nans()
            return outdata

    def verb_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

def main():
    """ULTRA-OPTIMIZED production main loop with maximum performance."""
    visual_landmarks_pipe = VSAPlayback()
    visual_landmarks_pipe.run_pipeline()


if __name__ == "__main__":
    main()
