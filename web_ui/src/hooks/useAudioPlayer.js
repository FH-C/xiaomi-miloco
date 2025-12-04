/**
 * Copyright (C) 2025 Xiaomi Corporation
 * This software may be used and distributed according to the terms of the Xiaomi Miloco License Agreement.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { decodeALaw, decodeMuLaw } from '@/utils/g711';

/**
 * Audio Player Hook
 * Manages WebSocket connection and audio playback
 *
 * @param {Object} options
 * @param {string} options.cameraId - Camera ID
 * @param {number} options.channel - Channel number
 * @param {boolean} options.enabled - Whether audio is enabled
 */
const useAudioPlayer = ({ cameraId, channel, enabled }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const nextStartTimeRef = useRef(0);

  const initAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      audioContextRef.current = new AudioContext();
    }
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }
  }, []);

  const playAudioChunk = useCallback((pcmData) => {
    if (!audioContextRef.current) { return; }

    const audioCtx = audioContextRef.current;
    const buffer = audioCtx.createBuffer(1, pcmData.length, 8000); // G.711 is usually 8000Hz
    const channelData = buffer.getChannelData(0);

    // Convert Int16 PCM to Float32 [-1.0, 1.0]
    for (let i = 0; i < pcmData.length; i++) {
      channelData[i] = pcmData[i] / 32768.0;
    }

    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtx.destination);

    // Schedule playback
    const currentTime = audioCtx.currentTime;
    // If next start time is in the past, reset it to current time (plus a small buffer)
    if (nextStartTimeRef.current < currentTime) {
      nextStartTimeRef.current = currentTime + 0.05;
    }

    source.start(nextStartTimeRef.current);
    nextStartTimeRef.current += buffer.duration;
  }, []);

  useEffect(() => {
    if (!enabled || !cameraId) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setIsPlaying(false);
      return;
    }

    initAudioContext();

    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${wsProtocol}://${window.location.host}${import.meta.env.VITE_API_BASE || ''}/api/miot/ws/audio_stream?camera_id=${encodeURIComponent(cameraId)}&channel=${encodeURIComponent(channel)}`;

    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Audio WebSocket connected');
      setIsPlaying(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const data = new Uint8Array(event.data);
        // Assume G.711 A-law for now, or detect if possible
        // Ideally the protocol should specify codec
        const pcm = decodeALaw(data);
        playAudioChunk(pcm);
      }
    };

    ws.onerror = (err) => {
      console.error('Audio WebSocket error:', err);
      setError('Audio connection failed');
      setIsPlaying(false);
    };

    ws.onclose = () => {
      console.log('Audio WebSocket closed');
      setIsPlaying(false);
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [cameraId, channel, enabled, initAudioContext, playAudioChunk]);

  return { isPlaying, error };
};

export default useAudioPlayer;
