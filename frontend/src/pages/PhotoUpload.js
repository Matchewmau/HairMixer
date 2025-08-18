import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import APIService from '../services/api';

const PhotoUpload = () => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

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
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-4">
            Upload Your Photo
          </h1>
          <p className="text-lg text-gray-600">
            Upload a clear photo of your face to get personalized hairstyle recommendations
          </p>
        </div>

        {/* Upload Area */}
        <div className="bg-white rounded-lg shadow-lg p-8 mb-8">
          <div
            className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
              dragActive
                ? 'border-blue-500 bg-blue-50'
                : selectedFile
                ? 'border-green-500 bg-green-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
            onDragEnter={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={(e) => { e.preventDefault(); setDragActive(false); }}
            onDragOver={(e) => { e.preventDefault(); }}
            onDrop={handleDrop}
          >
            {previewUrl ? (
              <div className="space-y-4">
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="mx-auto h-64 w-64 object-cover rounded-lg"
                />
                <div className="text-green-600 font-medium">
                  ✓ Photo selected successfully
                </div>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="text-blue-600 hover:text-blue-700 font-medium"
                >
                  Choose a different photo
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <svg
                  className="mx-auto h-12 w-12 text-gray-400"
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
                <div className="text-gray-600">
                  <p className="text-xl font-medium mb-2">Drop your photo here</p>
                  <p className="text-sm">or</p>
                </div>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
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
          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-600">
            <div className="flex items-center space-x-2">
              <span className="text-green-500">✓</span>
              <span>Good lighting</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-green-500">✓</span>
              <span>Clear face view</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-green-500">✓</span>
              <span>No sunglasses or hats</span>
            </div>
          </div>
        </div>

        {/* Analyze Button */}
        <div className="text-center">
          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || isAnalyzing}
            className={`px-8 py-3 rounded-lg font-medium text-lg transition-all ${
              !selectedFile || isAnalyzing
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700 hover:scale-105'
            }`}
          >
            {isAnalyzing ? 'Analyzing Face...' : 'Analyze My Face'}
          </button>
        </div>

        {/* Analysis Status */}
        {isAnalyzing && (
          <div className="mt-8 bg-blue-50 rounded-lg p-6 text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-blue-800 font-medium">
              Analyzing your facial features...
            </p>
          </div>
        )}

        {analysisResult === 'success' && (
          <div className="mt-8 bg-green-50 rounded-lg p-6 text-center">
            <div className="text-green-600 text-6xl mb-4">✓</div>
            <p className="text-green-800 font-medium text-lg">
              Face detected successfully! Redirecting to preferences...
            </p>
          </div>
        )}

        {analysisResult === 'failed' && (
          <div className="mt-8 bg-red-50 rounded-lg p-6 text-center">
            <div className="text-red-600 text-6xl mb-4">✗</div>
            <p className="text-red-800 font-medium text-lg mb-4">
              Could not detect a face in this image
            </p>
            <p className="text-red-600 text-sm">
              Please try again with a clearer photo where your face is clearly visible
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PhotoUpload;