import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import AuthService from '../services/AuthService';
import Navbar from '../components/Navbar';

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { preferences, imageFile, previewUrl, uploadResponse, recommendations } = location.state || {};
  
  const [user, setUser] = useState(null);

  useEffect(() => {
    checkAuth();
  }, []);

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

  const handleLogout = async () => {
    try {
      await AuthService.logout();
      setUser(null);
      navigate('/');
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  if (!recommendations) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-800 to-blue-900 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-semibold text-white mb-6">
            No recommendations found
          </h2>
          <button
            onClick={() => navigate('/upload')}
            className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white px-8 py-3 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg"
          >
            Start Over
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-800 to-blue-900">
      <Navbar 
        transparent={true} 
        user={user} 
        onLogout={handleLogout}
        showBackButton={true}
        backPath="/preferences"
      />
      
      <div className="pt-20 pb-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Your Hairstyle Recommendations
            </h1>
            {previewUrl && (
              <div className="flex justify-center mb-6">
                <img
                  src={previewUrl}
                  alt="Your photo"
                  className="h-40 w-40 object-cover rounded-full border-4 border-purple-400/30 shadow-2xl"
                />
              </div>
            )}
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Based on your preferences and facial analysis
            </p>
          </div>

          {/* Face Analysis Summary */}
          {uploadResponse && (
            <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-8 mb-12 shadow-xl">
              <h2 className="text-2xl font-bold text-white mb-6">Face Analysis</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="bg-blue-500/20 backdrop-blur-sm border border-blue-400/30 rounded-xl p-6">
                    <h3 className="font-medium text-white mb-2">Face Shape</h3>
                    <p className="text-blue-400 font-semibold text-xl capitalize">
                      {uploadResponse.face_shape?.shape || 'Unknown'}
                    </p>
                    <p className="text-xs text-gray-400 mt-2">
                      {uploadResponse.detection_method && `via ${uploadResponse.detection_method}`}
                    </p>
                  </div>
                </div>
                <div className="text-center">
                  <div className="bg-green-500/20 backdrop-blur-sm border border-green-400/30 rounded-xl p-6">
                    <h3 className="font-medium text-white mb-2">Confidence</h3>
                    <p className="text-green-400 font-semibold text-xl">
                      {uploadResponse.face_shape?.confidence ? 
                        `${Math.round(uploadResponse.face_shape.confidence * 100)}%` : 
                        uploadResponse.confidence ?
                        `${Math.round(uploadResponse.confidence * 100)}%` :
                        'N/A'}
                    </p>
                  </div>
                </div>
                <div className="text-center">
                  <div className="bg-purple-500/20 backdrop-blur-sm border border-purple-400/30 rounded-xl p-6">
                    <h3 className="font-medium text-white mb-2">Quality Score</h3>
                    <p className="text-purple-400 font-semibold text-xl">
                      {uploadResponse.quality_score ? 
                        `${Math.round(uploadResponse.quality_score * 10)}/10` : 
                        uploadResponse.quality_metrics?.overall_quality ?
                        `${Math.round(uploadResponse.quality_metrics.overall_quality * 10)}/10` :
                        'N/A'}
                    </p>
                  </div>
                </div>
              </div>
              
              {/* Show face shape characteristics if available */}
              {uploadResponse.face_shape?.shape && (
                <div className="mt-6 p-6 bg-gray-700/30 backdrop-blur-sm border border-gray-600/50 rounded-xl">
                  <h4 className="font-medium text-white mb-3">
                    {uploadResponse.face_shape.shape.charAt(0).toUpperCase() + uploadResponse.face_shape.shape.slice(1)} Face Shape Characteristics:
                  </h4>
                  <div className="text-sm text-gray-300">
                    <p className="mb-3">
                      <strong className="text-white">Best suited for:</strong> Most hairstyles work well with your face shape!
                    </p>
                    <div className="flex flex-wrap gap-3 mt-4">
                      <span className="px-3 py-1 bg-blue-500/20 text-blue-300 border border-blue-400/30 rounded-full text-sm">
                        Method: {uploadResponse.detection_method || 'AI Analysis'}
                      </span>
                      <span className="px-3 py-1 bg-green-500/20 text-green-300 border border-green-400/30 rounded-full text-sm">
                        Confidence: {Math.round((uploadResponse.face_shape.confidence || 0) * 100)}%
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Recommendations */}
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-8 shadow-xl">
            <h2 className="text-2xl font-bold text-white mb-8">Recommended Hairstyles</h2>
            
            {recommendations.recommended_styles && recommendations.recommended_styles.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {recommendations.recommended_styles.map((style, index) => (
                  <div key={index} className="bg-gray-700/30 backdrop-blur-sm border border-gray-600/50 rounded-xl p-6 hover:shadow-xl hover:shadow-purple-500/10 transition-all duration-300 transform hover:scale-105 group">
                    {style.image_url && (
                      <img
                        src={style.image_url}
                        alt={style.name}
                        className="w-full h-48 object-cover rounded-xl mb-4 group-hover:scale-110 transition-transform duration-300"
                      />
                    )}
                    <h3 className="font-semibold text-xl text-white mb-3 group-hover:text-purple-400 transition-colors duration-300">
                      {style.name}
                    </h3>
                    <p className="text-gray-300 mb-6 leading-relaxed">
                      {style.description}
                    </p>
                    <div className="flex justify-between items-center mb-6">
                      <span className="bg-blue-500/20 text-blue-300 border border-blue-400/30 text-sm font-medium px-3 py-1 rounded-full">
                        {style.match_score ? `${Math.round(style.match_score * 100)}% Match` : 'Recommended'}
                      </span>
                      <span className="text-sm text-gray-400">
                        {style.difficulty || 'Medium'} Difficulty
                      </span>
                    </div>
                    <button className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white py-3 px-4 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg">
                      Try This Style
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12">
                <p className="text-gray-300 mb-8 text-lg">
                  No specific recommendations available yet, but here are some popular styles for your face shape:
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                  {/* Placeholder recommendations */}
                  {[
                    { name: 'Classic Bob', description: 'A timeless bob cut that suits most face shapes' },
                    { name: 'Beach Waves', description: 'Relaxed, natural-looking waves' },
                    { name: 'Layered Cut', description: 'Versatile layers that add movement' }
                  ].map((style, index) => (
                    <div key={index} className="bg-gray-700/30 backdrop-blur-sm border border-gray-600/50 rounded-xl p-6 hover:shadow-xl hover:shadow-purple-500/10 transition-all duration-300 transform hover:scale-105 group">
                      <div className="w-full h-48 bg-gray-600/50 rounded-xl mb-4 flex items-center justify-center border border-gray-500/30">
                        <span className="text-gray-400">Style Image</span>
                      </div>
                      <h3 className="font-semibold text-xl text-white mb-3 group-hover:text-purple-400 transition-colors duration-300">
                        {style.name}
                      </h3>
                      <p className="text-gray-300 mb-6 leading-relaxed">
                        {style.description}
                      </p>
                      <button className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white py-3 px-4 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg">
                        Try This Style
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div className="text-center mt-12 space-x-6">
            <button
              onClick={() => navigate('/upload')}
              className="bg-gray-700/70 backdrop-blur-sm border border-gray-600/50 text-white px-8 py-4 rounded-xl hover:bg-gray-600/70 transition-all duration-300 transform hover:scale-105 font-medium"
            >
              Try Another Photo
            </button>
            <button
              onClick={() => navigate('/preferences', { 
                state: { imageFile, previewUrl, uploadResponse } 
              })}
              className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white px-8 py-4 rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg font-medium"
            >
              Update Preferences
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Results;