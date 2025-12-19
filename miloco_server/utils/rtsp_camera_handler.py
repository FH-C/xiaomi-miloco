# Copyright (C) 2025 Xiaomi Corporation
# This software may be used and distributed according to the terms of the Xiaomi Miloco License Agreement.

"""
RTSP camera handler utility for managing RTSP camera streams via go2rtc.
Provides functionality to connect to RTSP streams, decode H264/H265 video, and handle audio.
"""

import asyncio
import logging
import time
import threading
from typing import Any, Callable, Coroutine, Optional

import av
from pydantic import BaseModel, Field

from miloco_server.schema.miot_schema import CameraImgInfo, CameraImgSeq, CameraInfo
from miloco_server.utils.carmera_vision_handler import SizeLimitedQueue

logger = logging.getLogger(__name__)


class RTSPCameraInfo(BaseModel):
    """RTSP camera configuration model"""
    did: str = Field(..., description="Camera unique identifier")
    name: str = Field(..., description="Camera display name")
    rtsp_url: str = Field(..., description="RTSP stream URL from go2rtc")
    enable_audio: bool = Field(default=False, description="Whether to enable audio stream")
    transport: str = Field(default="tcp", description="Transport protocol: tcp or udp")
    location: Optional[str] = Field(default=None, description="Camera location (e.g. Home)")
    area: Optional[str] = Field(default=None, description="Camera area (e.g. Living Room)")
    home_name: str = Field(default="", description="Home location name")
    room_name: str = Field(default="", description="Room name")


class RTSPCameraHandler:
    """RTSP camera handler for managing camera streams via go2rtc"""

    def __init__(self, camera_info: RTSPCameraInfo, max_size: int, ttl: int, frame_interval: int = 500):
        """
        Initialize RTSP camera handler
        
        Args:
            camera_info: RTSP camera configuration
            max_size: Maximum number of images to cache
            ttl: Time-to-live for cached images in seconds
            frame_interval: Frame capture interval in milliseconds
        """
        self.camera_info = camera_info
        self.max_size = max_size
        self.ttl = ttl
        self.frame_interval = frame_interval / 1000.0  # Convert to seconds
        
        # Image queue for vision processing (single channel for RTSP cameras)
        self.camera_img_queue: SizeLimitedQueue = SizeLimitedQueue(max_size=max_size, ttl=ttl)
        
        # Stream handling
        self._container: Optional[av.container.InputContainer] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._online = False
        
        # Raw stream callbacks
        self._video_callbacks: dict[int, Callable[[str, bytes, int, int, int], Coroutine]] = {}
        self._audio_callbacks: dict[int, Callable[[str, bytes, int, int, int], Coroutine]] = {}
        
        # Event loop for async callbacks
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        logger.info("RTSPCameraHandler initialized for camera: %s, URL: %s", 
                   self.camera_info.did, self.camera_info.rtsp_url)

    async def start_async(self) -> bool:
        """Start RTSP stream asynchronously"""
        try:
            self._loop = asyncio.get_event_loop()
            self._stop_event.clear()
            
            # Start stream processing thread
            self._stream_thread = threading.Thread(
                target=self._stream_worker,
                daemon=True,
                name=f"RTSP-{self.camera_info.did}"
            )
            self._stream_thread.start()
            
            # Wait a bit for connection
            await asyncio.sleep(0.5)
            
            logger.info("Started RTSP stream for camera: %s", self.camera_info.did)
            return True
            
        except Exception as e:
            logger.error("Failed to start RTSP stream for %s: %s", self.camera_info.did, e)
            self._online = False
            return False

    def _stream_worker(self):
        """Worker thread for processing RTSP stream"""
        retry_count = 0
        max_retries = 5
        
        while not self._stop_event.is_set() and retry_count < max_retries:
            try:
                self._process_stream()
                retry_count = 0  # Reset on successful connection
            except Exception as e:
                retry_count += 1
                logger.error("RTSP stream error for %s (attempt %d/%d): %s", 
                           self.camera_info.did, retry_count, max_retries, e)
                
                if retry_count < max_retries and not self._stop_event.is_set():
                    time.sleep(2 ** retry_count)  # Exponential backoff
                    
        self._online = False
        logger.info("RTSP stream worker stopped for camera: %s", self.camera_info.did)

    def _process_stream(self):
        """Process RTSP stream and decode frames"""
        # Configure connection options
        options = {
            'rtsp_transport': self.camera_info.transport,
            'rtsp_flags': 'prefer_tcp' if self.camera_info.transport == 'tcp' else '',
            'stimeout': '5000000',  # 5 seconds timeout
        }
        
        # Open RTSP stream
        self._container = av.open(self.camera_info.rtsp_url, options=options, timeout=10.0)
        self._online = True
        
        # Find video and audio streams
        video_stream = None
        audio_stream = None
        
        for stream in self._container.streams:
            if stream.type == 'video' and video_stream is None:
                video_stream = stream
            elif stream.type == 'audio' and audio_stream is None and self.camera_info.enable_audio:
                audio_stream = stream
        
        if not video_stream:
            raise ValueError(f"No video stream found in RTSP URL: {self.camera_info.rtsp_url}")
        
        logger.info("RTSP stream opened: video=%s, audio=%s", 
                   video_stream.codec_context.name if video_stream else None,
                   audio_stream.codec_context.name if audio_stream else None)
        
        last_frame_time = 0
        frame_seq = 0
        
        # Process packets
        for packet in self._container.demux(video_stream, audio_stream):
            if self._stop_event.is_set():
                break
            
            try:
                if packet.stream.type == 'video':
                    # Handle video packets
                    current_time = time.time()
                    
                    # Get raw packet data (H264 NAL units in Annex-B format)
                    packet_data = bytes(packet)
                    
                    # Send raw video if callbacks registered
                    if self._video_callbacks and len(packet_data) > 0:
                        try:
                            asyncio.run_coroutine_threadsafe(
                                self._send_to_callbacks(
                                    self._video_callbacks,
                                    self.camera_info.did,
                                    packet_data,
                                    int(current_time * 1000),
                                    frame_seq,
                                    0  # channel 0 for RTSP cameras
                                ),
                                self._loop
                            )
                        except Exception as e:
                            logger.error("Error sending video packet to callbacks: %s", e)
                    
                    # Decode frames for image capturing (at frame_interval rate)
                    if current_time - last_frame_time >= self.frame_interval:
                        try:
                            frames_decoded = 0
                            logger.info("Attempting to decode frame for JPEG for camera %s", self.camera_info.did)
                            for frame in packet.decode():
                                frames_decoded += 1
                                logger.info("Frame decoded successfully, size: %dx%d", frame.width, frame.height)
                                # Convert frame to JPEG
                                img_bytes = self._frame_to_jpeg(frame)
                                if img_bytes:
                                    self.camera_img_queue.put(
                                        CameraImgInfo(data=img_bytes, timestamp=int(current_time))
                                    )
                                    last_frame_time = current_time
                                    frame_seq += 1
                                    logger.info("✓ Stored JPEG image for camera %s, size: %d bytes, queue size: %d", 
                                               self.camera_info.did, len(img_bytes), len(self.camera_img_queue.queue))
                                else:
                                    logger.warning("Failed to convert frame to JPEG for camera %s", self.camera_info.did)
                                break  # Only process first frame from packet
                            
                            if frames_decoded == 0:
                                logger.warning("No frames decoded from packet for camera %s", self.camera_info.did)
                        except Exception as e:
                            logger.error("Error decoding frame for JPEG: %s", e, exc_info=True)
                
                elif packet.stream.type == 'audio' and self._audio_callbacks:
                    # Handle audio packets
                    packet_data = bytes(packet)
                    if len(packet_data) > 0:
                        try:
                            asyncio.run_coroutine_threadsafe(
                                self._send_to_callbacks(
                                    self._audio_callbacks,
                                    self.camera_info.did,
                                    packet_data,
                                    int(time.time() * 1000),
                                    frame_seq,
                                    0  # channel 0
                                ),
                                self._loop
                            )
                        except Exception as e:
                            logger.error("Error sending audio packet to callbacks: %s", e)
                    
            except Exception as e:
                logger.error("Error processing packet: %s", e)
                continue
    
    def _frame_to_jpeg(self, frame: av.VideoFrame) -> Optional[bytes]:
        """Convert video frame to JPEG bytes"""
        try:
            logger.info("Converting frame to JPEG: format=%s, size=%dx%d", 
                       frame.format.name, frame.width, frame.height)
            
            # Convert frame to numpy array for easier handling
            import numpy as np
            from PIL import Image
            import io
            
            # Convert to RGB24 format
            frame_rgb = frame.reformat(format='rgb24')
            logger.info("Frame reformatted to RGB24")
            
            # Get numpy array from frame
            img_array = np.frombuffer(frame_rgb.planes[0], dtype=np.uint8)
            img_array = img_array.reshape((frame_rgb.height, frame_rgb.width, 3))
            logger.info("Created numpy array: shape=%s", img_array.shape)
            
            # Create PIL Image
            img = Image.fromarray(img_array, 'RGB')
            
            # Encode as JPEG
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=95, optimize=True)
            jpeg_bytes = buffer.getvalue()
            
            if len(jpeg_bytes) > 0:
                logger.info("✓ Successfully converted frame to JPEG: %d bytes", len(jpeg_bytes))
                return jpeg_bytes
            else:
                logger.warning("JPEG conversion resulted in empty bytes")
                return None
            
        except Exception as e:
            logger.error("Failed to convert frame to JPEG: %s", e, exc_info=True)
            return None
    
    async def _send_to_callbacks(
        self, 
        callbacks: dict, 
        did: str, 
        data: bytes, 
        ts: int, 
        seq: int, 
        channel: int
    ):
        """Send data to all registered callbacks"""
        for callback in callbacks.values():
            try:
                await callback(did, data, ts, seq, channel)
            except Exception as e:
                logger.error("Error in callback: %s", e)

    async def register_raw_stream(self, callback: Callable[[str, bytes, int, int, int], Coroutine], channel: int = 0):
        """Register callback for raw video stream"""
        self._video_callbacks[channel] = callback
        logger.info("Registered raw video stream callback for camera: %s, channel: %d", 
                   self.camera_info.did, channel)

    async def unregister_raw_stream(self, channel: int = 0):
        """Unregister raw video stream callback"""
        self._video_callbacks.pop(channel, None)
        logger.info("Unregistered raw video stream callback for camera: %s, channel: %d", 
                   self.camera_info.did, channel)

    async def register_raw_audio_stream(self, callback: Callable[[str, bytes, int, int, int], Coroutine], channel: int = 0):
        """Register callback for raw audio stream"""
        if not self.camera_info.enable_audio:
            logger.warning("Audio not enabled for camera: %s", self.camera_info.did)
            return
        
        self._audio_callbacks[channel] = callback
        logger.info("Registered raw audio stream callback for camera: %s, channel: %d", 
                   self.camera_info.did, channel)

    async def unregister_raw_audio_stream(self, channel: int = 0):
        """Unregister raw audio stream callback"""
        self._audio_callbacks.pop(channel, None)
        logger.info("Unregistered raw audio stream callback for camera: %s, channel: %d", 
                   self.camera_info.did, channel)

    def get_recents_camera_img(self, channel: int, n: int) -> CameraImgSeq:
        """Get recent camera images"""
        # Convert RTSPCameraInfo to CameraInfo for response
        camera_info = CameraInfo(
            did=self.camera_info.did,
            name=self.camera_info.name,
            online=self._online,
            model="RTSP Camera",
            icon=None,
            home_name=self.camera_info.home_name,
            room_name=self.camera_info.room_name,
            channel_count=1,
            camera_status="online" if self._online else "offline",
            camera_type="rtsp",  # Mark as RTSP camera
            rtsp_url=self.camera_info.rtsp_url  # Include RTSP URL for frontend
        )
        
        img_list = self.camera_img_queue.get_recent(n) if self._online else []
        logger.info("get_recents_camera_img for %s: online=%s, requested=%d, returned=%d images", 
                   self.camera_info.did, self._online, n, len(img_list))
        
        return CameraImgSeq(
            camera_info=camera_info,
            channel=channel,
            img_list=img_list
        )

    async def destroy(self) -> None:
        """Cleanup and destroy camera handler"""
        logger.info("Destroying RTSP camera handler for: %s", self.camera_info.did)
        
        # Stop stream thread
        self._stop_event.set()
        
        # Close container
        if self._container:
            try:
                self._container.close()
            except Exception as e:
                logger.error("Error closing RTSP container: %s", e)
            self._container = None
        
        # Wait for thread to finish
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=3.0)
        
        # Clear queues and callbacks
        self.camera_img_queue.clear()
        self._video_callbacks.clear()
        self._audio_callbacks.clear()
        self._online = False
        
        logger.info("RTSP camera handler destroyed: %s", self.camera_info.did)
