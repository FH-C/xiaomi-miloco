/**
 * Copyright (C) 2025 Xiaomi Corporation
 * This software may be used and distributed according to the terms of the Xiaomi Miloco License Agreement.
 */

import React, { useEffect, useRef, useState } from 'react';
import { message } from 'antd';
import { useTranslation } from 'react-i18next';
import DefaultCameraBg from '@/assets/images/default-camera-bg.png';

/**
 * RTSPPlayer Component - WebRTC-based player for RTSP cameras via go2rtc
 * 
 * @param {Object} props
 * @param {string} props.rtspUrl - Original RTSP URL (will be converted to go2rtc WebRTC)
 * @param {string} [props.go2rtcUrl='http://127.0.0.1:1984'] - go2rtc server URL
 * @param {Object} [props.style] - Custom style
 * @returns {JSX.Element}
 */
const RTSPPlayer = ({ rtspUrl, go2rtcUrl = 'http://127.0.0.1:1984', style }) => {
  const { t } = useTranslation();
  const videoRef = useRef(null);
  const pcRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!rtspUrl) return;

    const startStream = async () => {
      try {
        setLoading(true);
        setError(null);

        // Extract stream name from RTSP URL
        // rtsp://127.0.0.1:8554/mac_cam -> mac_cam
        const streamName = rtspUrl.split('/').pop();

        // Create WebRTC connection
        const pc = new RTCPeerConnection({
          iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });
        pcRef.current = pc;

        // Add video element to receive stream
        pc.ontrack = (event) => {
          console.log('Received track:', event.track.kind);
          if (videoRef.current) {
            videoRef.current.srcObject = event.streams[0];
            setLoading(false);
          }
        };

        pc.oniceconnectionstatechange = () => {
          console.log('ICE connection state:', pc.iceConnectionState);
          if (pc.iceConnectionState === 'failed' || pc.iceConnectionState === 'disconnected') {
            setError(t('instant.deviceList.deviceConnectClosed'));
          }
        };

        // Add transceiver for receiving video
        pc.addTransceiver('video', { direction: 'recvonly' });
        pc.addTransceiver('audio', { direction: 'recvonly' });

        // Create offer
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        // Send offer to go2rtc
        const go2rtcApiUrl = `${go2rtcUrl}/api/webrtc?src=${encodeURIComponent(streamName)}`;
        console.log('Connecting to go2rtc:', go2rtcApiUrl);

        const response = await fetch(go2rtcApiUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            type: 'offer',
            sdp: offer.sdp,
          }),
        });

        if (!response.ok) {
          throw new Error(`go2rtc returned ${response.status}: ${response.statusText}`);
        }

        const answer = await response.json();

        // Set remote description
        await pc.setRemoteDescription(new RTCSessionDescription({
          type: 'answer',
          sdp: answer.sdp || answer,
        }));

        console.log('WebRTC connection established');
      } catch (err) {
        console.error('Failed to start RTSP stream:', err);
        setError(err.message || t('instant.deviceList.deviceConnectFailed'));
        setLoading(false);
        message.error(t('instant.deviceList.deviceConnectFailed'));
      }
    };

    startStream();

    // Cleanup
    return () => {
      if (pcRef.current) {
        pcRef.current.close();
        pcRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    };
  }, [rtspUrl, go2rtcUrl, t]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', ...style }}>
      {loading && (
        <div style={{
          backgroundColor: 'rgba(0,0,0,0.1)',
          position: 'absolute',
          left: 0,
          top: 0,
          width: '100%',
          height: '100%',
          zIndex: 2,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: 8
        }}>
          <img
            src={DefaultCameraBg}
            alt="default-camera-bg"
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              borderRadius: 8,
              position: 'absolute',
              top: 0,
              left: 0,
              zIndex: -1,
            }}
          />
          <div style={{ color: 'var(--text-color)', fontSize: '14px' }}>
            {t('common.loading')}...
          </div>
        </div>
      )}

      {error && (
        <div style={{
          position: 'absolute',
          left: 0,
          top: 0,
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: 'rgba(0,0,0,0.1)',
          borderRadius: 8,
          zIndex: 3
        }}>
          <div style={{
            color: '#ff4d4f',
            fontSize: '14px',
            padding: '20px',
            textAlign: 'center',
            backgroundColor: 'var(--bg-color)',
            borderRadius: '8px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.15)'
          }}>
            {error}
          </div>
        </div>
      )}

      <video
        ref={videoRef}
        autoPlay
        playsInline
        controls
        muted
        style={{
          width: '100%',
          height: '100%',
          borderRadius: 8,
          objectFit: 'cover',
          backgroundColor: '#000',
          display: loading ? 'none' : 'block'
        }}
      />
    </div>
  );
};

export default RTSPPlayer;
