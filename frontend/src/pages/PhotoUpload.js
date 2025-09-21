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
  
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

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
      const uploadResponse = await APIService.uploadImage(selectedFile);
      
      console.log('Upload response:', uploadResponse);

      // Check if face was detected
      if (uploadResponse.face_detected) {
        setAnalysisResult('success');
        
        // Navigate to preferences page after 2 seconds
        setTimeout(() => {
          navigate('/preferences', { 
            state: { 
              imageFile: selectedFile,
              previewUrl: previewUrl,
              uploadResponse: uploadResponse
            }
          });
        }, 2000);
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
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-800 to-blue-900">
      <Navbar 
        transparent={true} 
        user={user} 
        onLogout={handleLogout} 
      />
      
      <div className="pt-20 pb-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Upload Your Photo
            </h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Upload a clear photo of your face to get personalized hairstyle recommendations
            </p>
          </div>

          {/* Upload Area */}
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-8 mb-8 shadow-xl">
            <div
              className={`border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300 ${
                dragActive
                  ? 'border-purple-400 bg-purple-500/10 transform scale-105'
                  : selectedFile
                  ? 'border-green-400 bg-green-500/10'
                  : 'border-gray-600 hover:border-gray-500 hover:bg-white/5'
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
                  className="mx-auto h-64 w-64 object-cover rounded-xl shadow-lg border-2 border-purple-400/30"
                />
                <div className="text-green-400 font-medium text-lg flex items-center justify-center space-x-2">
                  <span className="text-2xl">✓</span>
                  <span>Photo selected successfully</span>
                </div>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="text-purple-400 hover:text-purple-300 font-medium transition-colors duration-300"
                >
                  Choose a different photo
                </button>
              </div>
            ) : (
              <div className="space-y-6">
                <div className="relative">
                  <svg
                    className="mx-auto h-16 w-16 text-gray-400"
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
                  <div className="absolute -inset-1 bg-gradient-to-r from-purple-600 to-blue-600 rounded-full opacity-20 blur-sm"></div>
                </div>
                <div className="text-gray-300">
                  <p className="text-2xl font-medium mb-3">Drop your photo here</p>
                  <p className="text-lg text-gray-400">or</p>
                </div>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white px-8 py-3 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg"
                >
                  Browse Files
                </button>
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
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
            <div className="flex items-center space-x-3 text-gray-300">
              <span className="text-green-400 text-lg">✓</span>
              <span>Good lighting</span>
            </div>
            <div className="flex items-center space-x-3 text-gray-300">
              <span className="text-green-400 text-lg">✓</span>
              <span>Clear face view</span>
            </div>
            <div className="flex items-center space-x-3 text-gray-300">
              <span className="text-green-400 text-lg">✓</span>
              <span>No sunglasses or hats</span>
            </div>
          </div>
        </div>

        {/* Analyze Button */}
        <div className="text-center mb-12">
          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || isAnalyzing}
            className={`px-12 py-4 rounded-xl font-bold text-xl transition-all duration-300 transform ${
              !selectedFile || isAnalyzing
                ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                : 'bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white hover:scale-105 shadow-lg hover:shadow-purple-500/25'
            }`}
          >
            {isAnalyzing ? 'Analyzing Face...' : 'Analyze My Face'}
          </button>
        </div>

        {/* Analysis Status */}
        {isAnalyzing && (
          <div className="bg-purple-900/30 backdrop-blur-sm border border-purple-700/50 rounded-xl p-8 text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-400 mx-auto mb-6"></div>
            <p className="text-purple-300 font-medium text-lg">
              Analyzing your facial features...
            </p>
          </div>
        )}

        {analysisResult === 'success' && (
          <div className="bg-green-900/30 backdrop-blur-sm border border-green-700/50 rounded-xl p-8 text-center">
            <div className="text-green-400 text-7xl mb-6">✓</div>
            <p className="text-green-300 font-medium text-xl">
              Face detected successfully! Redirecting to preferences...
            </p>
          </div>
        )}

        {analysisResult === 'failed' && (
          <div className="bg-red-900/30 backdrop-blur-sm border border-red-700/50 rounded-xl p-8 text-center">
            <div className="text-red-400 text-7xl mb-6">✗</div>
            <p className="text-red-300 font-medium text-xl mb-4">
              Could not detect a face in this image
            </p>
            <p className="text-red-400 text-lg">
              Please try again with a clearer photo where your face is clearly visible
            </p>
          </div>
        )}
        </div>
      </div>
    </div>
  );
};

export default PhotoUpload;