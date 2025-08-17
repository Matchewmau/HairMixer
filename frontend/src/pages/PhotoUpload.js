import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const PhotoUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [stream, setStream] = useState(null);
  const [error, setError] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [videoReady, setVideoReady] = useState(false);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      setError('');
      setVideoReady(false);
      
      // Request camera with more permissive constraints
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { min: 640, ideal: 1280, max: 1920 },
          height: { min: 480, ideal: 720, max: 1080 },
          facingMode: 'user'
        },
        audio: false
      });
      
      setStream(mediaStream);
      setCameraActive(true);
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        
        // Wait for video to load metadata
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play().then(() => {
            setVideoReady(true);
          }).catch(err => {
            console.error('Error playing video:', err);
            setError('Error starting video playback');
          });
        };
      }
    } catch (err) {
      console.error('Camera access error:', err);
      setError(`Unable to access camera: ${err.message}. Please check permissions.`);
      setCameraActive(false);
    }
  }, []);

  // Stop camera
  const stopCamera = useCallback(() => {
    setVideoReady(false);
    if (stream) {
      stream.getTracks().forEach(track => {
        track.stop();
        console.log('Stopped track:', track.kind);
      });
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  }, [stream]);

  // Capture photo from camera
  const capturePhoto = useCallback(() => {
    if (videoRef.current && canvasRef.current && videoReady) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      
      // Set canvas dimensions to match video
      canvas.width = video.videoWidth || video.clientWidth;
      canvas.height = video.videoHeight || video.clientHeight;
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], 'captured-photo.jpg', { type: 'image/jpeg' });
          setSelectedFile(file);
          setPreviewUrl(URL.createObjectURL(blob));
          stopCamera();
          // Automatically analyze the captured image
          analyzeImage(file);
        }
      }, 'image/jpeg', 0.9);
    } else {
      setError('Camera not ready. Please wait a moment and try again.');
    }
  }, [stopCamera, videoReady]);

  // Handle file selection
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type.startsWith('image/')) {
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        setError('');
        // Automatically analyze the uploaded image
        analyzeImage(file);
      } else {
        setError('Please select a valid image file.');
      }
    }
  };

  // Clear selected image
  const clearImage = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setError('');
    setAnalysisResult(null);
    setVideoReady(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Analyze image for face detection
  const analyzeImage = async (imageFile = selectedFile) => {
    if (!imageFile) {
      setError('Please select or capture an image first.');
      return;
    }

    setIsAnalyzing(true);
    setError('');
    setAnalysisResult(null);

    try {
      // TODO: Replace with actual face detection API call
      await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate API call
      
      // Simulate face detection result (in real implementation, this would come from your AI service)
      const faceDetected = Math.random() > 0.3; // 70% success rate for demo
      
      if (faceDetected) {
        setAnalysisResult('success');
        // Auto-navigate to preferences after successful analysis
        setTimeout(() => {
          navigate('/preferences', { 
            state: { 
              imageFile: imageFile,
              previewUrl: previewUrl || URL.createObjectURL(imageFile)
            }
          });
        }, 1000);
      } else {
        setAnalysisResult('failed');
        setError('Unable to detect face shape in this image. Please follow the photo guidelines and try again.');
      }
    } catch (err) {
      console.error('Analysis error:', err);
      setError('Failed to analyze image. Please try again.');
      setAnalysisResult('failed');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      stopCamera();
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [stopCamera, previewUrl]);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <button
              onClick={() => navigate('/dashboard')}
              className="flex items-center text-purple-600 hover:text-purple-700 font-medium"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              Back to Dashboard
            </button>
            <h1 className="text-xl font-bold text-gray-900">
              Hair<span className="text-purple-600">Mixer</span>
            </h1>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Upload Your Photo</h2>
          <p className="text-lg text-gray-600">
            Take a photo or upload an image to get personalized hairstyle recommendations
          </p>
        </div>

        {/* Photo Guidelines */}
        <div className="mb-8 bg-blue-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 mb-3">Photo Guidelines</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ul className="text-blue-800 space-y-2">
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                Make sure your face is clearly visible and well-lit
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                Face the camera directly for best results
              </li>
            </ul>
            <ul className="text-blue-800 space-y-2">
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                Avoid shadows or extreme lighting
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                Keep hair away from your face if possible
              </li>
            </ul>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 text-red-600 px-4 py-3 rounded-md">
            {error}
          </div>
        )}

        {/* Control Buttons */}
        <div className="flex justify-center space-x-4 mb-8">
          <button
            onClick={startCamera}
            disabled={cameraActive || previewUrl}
            className={`flex items-center px-6 py-3 rounded-lg font-medium transition duration-200 ${
              cameraActive || previewUrl
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-purple-600 hover:bg-purple-700 text-white'
            }`}
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            {cameraActive ? 'Camera Active' : 'Open Camera'}
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={cameraActive || previewUrl}
            className={`flex items-center px-6 py-3 rounded-lg font-medium transition duration-200 ${
              cameraActive || previewUrl
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            Choose Image
          </button>

          {(cameraActive || previewUrl) && (
            <button
              onClick={() => {
                stopCamera();
                clearImage();
              }}
              className="flex items-center px-6 py-3 rounded-lg font-medium bg-gray-600 hover:bg-gray-700 text-white transition duration-200"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
              Reset
            </button>
          )}
        </div>

        {/* Main Display Area */}
        <div className="bg-white rounded-xl shadow-lg p-8 min-h-[500px] flex items-center justify-center">
          {/* Default State */}
          {!cameraActive && !previewUrl && (
            <div className="text-center">
              <div className="w-32 h-32 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <svg className="w-16 h-16 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-600 mb-2">No Image Selected</h3>
              <p className="text-gray-500">Use the camera or choose an image to get started</p>
            </div>
          )}

          {/* Camera View */}
          {cameraActive && (
            <div className="w-full flex flex-col items-center">
              <div className="relative bg-black rounded-lg overflow-hidden max-w-2xl w-full">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-auto object-cover"
                  style={{ 
                    minHeight: '400px',
                    maxHeight: '500px',
                    transform: 'scaleX(-1)' // Mirror the video for better UX
                  }}
                />
                <canvas ref={canvasRef} className="hidden" />
                
                {/* Loading overlay */}
                {!videoReady && (
                  <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                    <div className="text-white text-center">
                      <svg className="animate-spin h-8 w-8 text-white mx-auto mb-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      <p>Starting camera...</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Image Preview */}
          {previewUrl && (
            <div className="w-full space-y-6">
              <div className="text-center">
                <img
                  src={previewUrl}
                  alt="Selected"
                  className="max-w-lg max-h-96 mx-auto rounded-lg shadow-lg object-cover"
                />
              </div>
              
              {/* Analysis Status */}
              <div className="text-center">
                {isAnalyzing && (
                  <div className="flex items-center justify-center space-x-3">
                    <svg className="animate-spin h-6 w-6 text-purple-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span className="text-lg font-medium text-purple-600">Analyzing facial structure...</span>
                  </div>
                )}
                
                {analysisResult === 'success' && (
                  <div className="flex items-center justify-center space-x-3 text-green-600">
                    <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <span className="text-lg font-medium">Face detected successfully! Redirecting...</span>
                  </div>
                )}
                
                {analysisResult === 'failed' && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-center space-x-3 text-red-600">
                      <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                      <span className="text-lg font-medium">Face detection failed</span>
                    </div>
                    <button
                      onClick={() => analyzeImage()}
                      className="bg-purple-600 hover:bg-purple-700 text-white font-medium py-2 px-6 rounded-lg transition duration-200"
                    >
                      Try Again
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Capture Button - Only show when camera is ready and positioned outside the main display area */}
        {cameraActive && videoReady && (
          <div className="flex justify-center mt-6">
            <button
              onClick={capturePhoto}
              className="bg-green-600 hover:bg-green-700 text-white font-bold py-4 px-4 rounded-full transition duration-200 shadow-xl transform hover:scale-110 flex items-center justify-center"
              style={{ width: '80px', height: '80px' }}
            >
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="3" />
                <path d="M12 1v6m0 8v6m11-7h-6m-8 0H1" />
              </svg>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default PhotoUpload;