import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import APIService from '../services/api';
import AuthService from '../services/AuthService';
import Navbar from '../components/Navbar';

const PhotoUpload = () => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [user, setUser] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  const [stream, setStream] = useState(null);
  
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        if (!AuthService.getAccessToken()) {
          setUser(null);
          return;
        }
        const currentUser = await AuthService.getCurrentUser();
        setUser(currentUser);
      } catch (error) {
        console.error('Authentication check failed:', error);
        setUser(null);
      }
    };

    checkAuth();
  }, []);

  const handleLogout = async () => {
    try {
      await AuthService.logout();
      setUser(null);
      navigate('/');
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        } 
      });
      setStream(mediaStream);
      setShowCamera(true);
      
      // Wait for videoRef to be available
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
      }, 100);
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Could not access camera. Please make sure you have granted camera permissions.');
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setShowCamera(false);
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      const context = canvas.getContext('2d');
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      canvas.toBlob((blob) => {
        const file = new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' });
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        stopCamera();
      }, 'image/jpeg', 0.95);
    }
  };

  // Cleanup camera on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  const handleFiles = (files) => {
    const file = files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      alert('Please select a valid image file');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      alert('Please select an image first');
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      // Upload image to backend
      const response = await APIService.uploadImage(selectedFile);
      
      console.log('Upload response:', response);

      // Check if face was detected
      if (response.face_detected) {
        setAnalysisResult('success');
        
        // Navigate directly to preferences form after 1.5 seconds
        setTimeout(() => {
          navigate('/preferences', { 
            state: { 
              imageFile: selectedFile,
              previewUrl: previewUrl,
              uploadResponse: response
            }
          });
        }, 1500);
      } else {
        setAnalysisResult('failed');
      }
      
    } catch (error) {
      console.error('Error analyzing image:', error);
      setAnalysisResult('failed');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900">
      <Navbar 
        transparent={true} 
        user={user} 
        onLogout={handleLogout} 
      />
      
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-800 to-blue-900 pt-20 md:pt-24">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Header */}
          <div className="text-center mb-8 md:mb-12">
            <div className="mb-4">
              <span className="inline-block bg-blue-500/20 text-blue-300 px-4 py-2 rounded-full text-sm font-medium border border-blue-500/30 backdrop-blur-sm">
                Step 1: Photo Analysis
              </span>
            </div>
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-4 md:mb-6 bg-gradient-to-r from-white via-blue-100 to-purple-200 bg-clip-text text-transparent">
              Upload Your Photo
            </h1>
            <p className="text-lg md:text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
              Upload a clear photo of your face or use your camera to get personalized hairstyle recommendations
            </p>
          </div>

          {/* Camera Modal */}
          {showCamera && (
            <div className="fixed inset-0 bg-black/90 backdrop-blur-sm z-50 flex items-center justify-center p-4">
              <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl max-w-4xl w-full border border-white/10 shadow-2xl">
                <div className="flex items-center justify-between p-4 md:p-6 border-b border-white/10">
                  <h2 className="text-xl md:text-2xl font-bold text-white">Take a Photo</h2>
                  <button
                    onClick={stopCamera}
                    className="text-gray-400 hover:text-white transition-colors duration-300 p-2 hover:bg-white/10 rounded-lg"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                
                <div className="p-4 md:p-6">
                  <div className="relative bg-black rounded-xl overflow-hidden mb-4 md:mb-6">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="w-full h-auto max-h-[60vh] object-contain"
                    />
                  </div>
                  
                  <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <button
                      onClick={capturePhoto}
                      className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold py-3 px-8 rounded-lg transition-all duration-300 transform hover:scale-105 shadow-lg border border-blue-500/30"
                    >
                      📸 Capture Photo
                    </button>
                    <button
                      onClick={stopCamera}
                      className="bg-white/10 hover:bg-white/20 text-white font-semibold py-3 px-8 rounded-lg transition-all duration-300 backdrop-blur-sm border border-white/20 hover:border-white/40"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              </div>
              <canvas ref={canvasRef} className="hidden" />
            </div>
          )}

          {/* Upload Area */}
          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6 md:p-8 mb-8 shadow-xl">
            <div
              className={`border-2 border-dashed rounded-xl p-8 md:p-12 text-center transition-all duration-300 ${
                dragActive
                  ? 'border-blue-400 bg-blue-500/10 transform scale-105'
                  : selectedFile
                  ? 'border-green-400 bg-green-500/10'
                  : 'border-white/20 hover:border-white/30 hover:bg-white/5'
              }`}
              onDragEnter={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={(e) => { e.preventDefault(); setDragActive(false); }}
              onDragOver={(e) => { e.preventDefault(); }}
              onDrop={handleDrop}
            >
            {previewUrl ? (
              <div className="space-y-6">
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="mx-auto h-48 w-48 md:h-64 md:w-64 object-cover rounded-xl shadow-lg border-2 border-blue-400/30"
                />
                <div className="text-green-400 font-medium text-base md:text-lg flex items-center justify-center space-x-2">
                  <span className="text-xl md:text-2xl">✓</span>
                  <span>Photo selected successfully</span>
                </div>
                <div className="flex flex-col sm:flex-row gap-3 justify-center">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="text-blue-400 hover:text-blue-300 font-medium transition-colors duration-300 px-4 py-2 rounded-lg hover:bg-white/5"
                  >
                    Choose a different photo
                  </button>
                  <button
                    onClick={startCamera}
                    className="text-purple-400 hover:text-purple-300 font-medium transition-colors duration-300 px-4 py-2 rounded-lg hover:bg-white/5"
                  >
                    📸 Use Camera Instead
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                <div className="relative inline-block">
                  <svg
                    className="mx-auto h-12 w-12 md:h-16 md:w-16 text-gray-400"
                    stroke="currentColor"
                    fill="none"
                    viewBox="0 0 48 48"
                  >
                    <path
                      d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                      strokeWidth={2}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <div className="absolute -inset-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full opacity-20 blur-md"></div>
                </div>
                <div className="text-gray-300">
                  <p className="text-xl md:text-2xl font-medium mb-3">Drop your photo here</p>
                  <p className="text-base md:text-lg text-gray-400">or choose an option below</p>
                </div>
                <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full sm:w-auto bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-6 md:px-8 py-3 rounded-xl font-semibold transition-all duration-300 transform hover:scale-105 shadow-lg border border-blue-500/30"
                  >
                    📁 Browse Files
                  </button>
                  <button
                    onClick={startCamera}
                    className="w-full sm:w-auto bg-white/10 hover:bg-white/20 text-white px-6 md:px-8 py-3 rounded-xl font-semibold transition-all duration-300 backdrop-blur-sm border border-white/20 hover:border-white/40"
                  >
                    📸 Use Camera
                  </button>
                </div>
              </div>
            )}
            </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={(e) => handleFiles(e.target.files)}
            className="hidden"
          />

          {/* Photo Tips */}
          <div className="mt-6 md:mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4 md:gap-6 text-sm md:text-base">
            <div className="flex items-center space-x-3 text-gray-300 bg-white/5 rounded-lg p-3 border border-white/10">
              <span className="text-green-400 text-lg md:text-xl">✓</span>
              <span>Good lighting</span>
            </div>
            <div className="flex items-center space-x-3 text-gray-300 bg-white/5 rounded-lg p-3 border border-white/10">
              <span className="text-green-400 text-lg md:text-xl">✓</span>
              <span>Clear face view</span>
            </div>
            <div className="flex items-center space-x-3 text-gray-300 bg-white/5 rounded-lg p-3 border border-white/10">
              <span className="text-green-400 text-lg md:text-xl">✓</span>
              <span>No sunglasses or hats</span>
            </div>
          </div>
        </div>

        {/* Analyze Button */}
        <div className="text-center mb-8 md:mb-12">
          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || isAnalyzing}
            className={`w-full sm:w-auto px-8 md:px-12 py-3 md:py-4 rounded-xl font-bold text-lg md:text-xl transition-all duration-300 transform ${
              !selectedFile || isAnalyzing
                ? 'bg-gray-700/50 text-gray-500 cursor-not-allowed border border-gray-600'
                : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white hover:scale-105 shadow-lg hover:shadow-2xl border border-blue-500/30'
            }`}
          >
            {isAnalyzing ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Analyzing Face...
              </span>
            ) : (
              'Analyze My Face'
            )}
          </button>
        </div>

        {/* Analysis Status */}
        {isAnalyzing && (
          <div className="bg-white/5 backdrop-blur-sm border border-purple-500/30 rounded-xl p-6 md:p-8 text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-400 mx-auto mb-6"></div>
            <p className="text-purple-300 font-medium text-base md:text-lg">
              Analyzing your facial features...
            </p>
          </div>
        )}

        {analysisResult === 'success' && (
          <div className="bg-white/5 backdrop-blur-sm border border-green-500/30 rounded-xl p-6 md:p-8 text-center">
            <div className="text-green-400 text-5xl md:text-7xl mb-6">✓</div>
            <p className="text-green-300 font-medium text-lg md:text-xl mb-2">
              Face detected successfully!
            </p>
            <p className="text-gray-400 text-sm md:text-base">
              Redirecting to preferences...
            </p>
          </div>
        )}

        {analysisResult === 'failed' && (
          <div className="bg-white/5 backdrop-blur-sm border border-red-500/30 rounded-xl p-6 md:p-8 text-center">
            <div className="text-red-400 text-5xl md:text-7xl mb-6">✗</div>
            <p className="text-red-300 font-medium text-lg md:text-xl mb-4">
              Could not detect a face in this image
            </p>
            <p className="text-red-400 text-sm md:text-base">
              Please try again with a clearer photo where your face is clearly visible
            </p>
            <div className="mt-6 flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={() => {
                  setSelectedFile(null);
                  setPreviewUrl(null);
                  setAnalysisResult(null);
                }}
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-300 transform hover:scale-105 shadow-lg border border-blue-500/30"
              >
                Try Another Photo
              </button>
              <button
                onClick={startCamera}
                className="bg-white/10 hover:bg-white/20 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-300 backdrop-blur-sm border border-white/20 hover:border-white/40"
              >
                📸 Use Camera
              </button>
            </div>
          </div>
        )}
        </div>
      </div>
    </div>
  );
};

export default PhotoUpload;