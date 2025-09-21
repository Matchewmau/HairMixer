import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import AuthService from '../services/AuthService';
import apiService from '../services/api';
import Navbar from '../components/Navbar';

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { preferences, imageFile, previewUrl, uploadResponse, recommendations } = location.state || {};
  
  const [user, setUser] = useState(null);
  const [overlayLoadingId, setOverlayLoadingId] = useState(null);
  const [overlayType, setOverlayType] = useState('basic');
  const [overlayUrl, setOverlayUrl] = useState(null);
  const [showOverlayModal, setShowOverlayModal] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

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

  const resolveMediaUrl = (url) => {
    if (!url) return '';
    if (url.startsWith('http://') || url.startsWith('https://')) return url;
    const serverOrigin = apiService.baseURL.replace(/\/api\/?$/, '');
    return `${serverOrigin}${url}`;
  };

  const handleGenerateOverlay = async (style, type = 'basic') => {
    try {
      setErrorMsg('');
      if (!user) {
        // Require authentication for overlay
        navigate('/login', { state: { from: '/results' } });
        return;
      }
      if (!uploadResponse?.image_id) {
        setErrorMsg('Missing uploaded image reference. Please re-upload your photo.');
        return;
      }
      if (!style?.id) {
        setErrorMsg('This style is not selectable yet. Please choose a style from the catalog (with a valid ID).');
        return;
      }
      setOverlayType(type);
      setOverlayLoadingId(style.id);
      const resp = await apiService.generateOverlay(uploadResponse.image_id, style.id, type);
      const fullUrl = resolveMediaUrl(resp.overlay_url);
      setOverlayUrl(fullUrl);
      setShowOverlayModal(true);
    } catch (e) {
      console.error('Overlay generation failed:', e);
      setErrorMsg(e?.message || 'Failed to generate overlay');
    } finally {
      setOverlayLoadingId(null);
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

          {/* Hairstyle Compatibility Check Results */}
          {preferences?.check_compatibility && (preferences?.target_hairstyle || preferences?.custom_hairstyle) && (
            <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-8 mb-12 shadow-xl">
              <h2 className="text-2xl font-bold text-white mb-6">Hairstyle Compatibility Analysis</h2>
              
              {/* Selected Hairstyle Display */}
              <div className="bg-purple-900/20 border border-purple-500/30 rounded-xl p-6 mb-8">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full flex items-center justify-center mr-4">
                    <span className="text-white text-xl">💇‍♀️</span>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-white">Selected Hairstyle</h3>
                    <p className="text-purple-300 font-medium">
                      {preferences.target_hairstyle 
                        ? preferences.target_hairstyle.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())
                        : preferences.custom_hairstyle.trim()
                      }
                    </p>
                  </div>
                </div>
              </div>

              {/* Compatibility Results */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Compatibility Score */}
                <div className="space-y-6">
                  <div className="text-center">
                    <div className="inline-flex items-center justify-center w-32 h-32 rounded-full bg-gradient-to-br from-green-400 to-emerald-500 mb-4 shadow-lg">
                      <div className="text-center">
                        <div className="text-3xl font-bold text-white">87%</div>
                        <div className="text-sm text-green-100">Compatible</div>
                      </div>
                    </div>
                    <h3 className="text-xl font-bold text-white mb-2">Compatibility Score</h3>
                    <p className="text-gray-300">
                      This hairstyle is highly compatible with your {uploadResponse?.face_shape?.shape || 'oval'} face shape!
                    </p>
                  </div>

                  {/* Quick Stats */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-blue-900/30 border border-blue-500/30 rounded-lg p-4 text-center">
                      <div className="text-2xl font-bold text-blue-400">92%</div>
                      <div className="text-sm text-gray-300">Face Shape Match</div>
                    </div>
                    <div className="bg-green-900/30 border border-green-500/30 rounded-lg p-4 text-center">
                      <div className="text-2xl font-bold text-green-400">85%</div>
                      <div className="text-sm text-gray-300">Style Suitability</div>
                    </div>
                  </div>
                </div>

                {/* Detailed Analysis */}
                <div className="space-y-6">
                  <div>
                    <h4 className="text-lg font-semibold text-white mb-4">Why This Works</h4>
                    <div className="space-y-3">
                      <div className="flex items-start">
                        <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                          <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        </div>
                        <div>
                          <p className="text-gray-300">
                            <span className="text-white font-medium">Face Shape Harmony:</span> This style complements your natural facial proportions perfectly.
                          </p>
                        </div>
                      </div>
                      
                      <div className="flex items-start">
                        <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                          <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        </div>
                        <div>
                          <p className="text-gray-300">
                            <span className="text-white font-medium">Balanced Proportions:</span> Creates an aesthetically pleasing balance with your features.
                          </p>
                        </div>
                      </div>

                      <div className="flex items-start">
                        <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                          <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        </div>
                        <div>
                          <p className="text-gray-300">
                            <span className="text-white font-medium">Style Versatility:</span> Works well with your lifestyle preferences and maintenance level.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Recommendations for Improvement */}
                  <div>
                    <h4 className="text-lg font-semibold text-white mb-4">Pro Tips</h4>
                    <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
                      <div className="flex items-start">
                        <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                          <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                          </svg>
                        </div>
                        <div className="text-sm text-gray-300">
                          Consider adding subtle layers to enhance volume and movement. This will maximize the style's flattering effect on your face shape.
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Alternative Suggestions */}
              <div className="mt-8 pt-8 border-t border-gray-700/50">
                <h4 className="text-lg font-semibold text-white mb-4">Similar Compatible Styles</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {[
                    { name: 'Textured Bob', compatibility: '89%', reason: 'Similar length, added texture' },
                    { name: 'Layered Variation', compatibility: '91%', reason: 'Enhanced with face-framing layers' },
                    { name: 'Side-Swept Bangs', compatibility: '86%', reason: 'Softens facial angles' }
                  ].map((style, index) => (
                    <div key={index} className="bg-slate-700/30 border border-slate-600/50 rounded-lg p-4 hover:border-purple-500/30 transition-colors duration-300">
                      <div className="flex items-center justify-between mb-2">
                        <h5 className="font-medium text-white">{style.name}</h5>
                        <span className="bg-green-900/30 text-green-400 px-2 py-1 rounded-full text-xs font-medium">
                          {style.compatibility}
                        </span>
                      </div>
                      <p className="text-sm text-gray-400">{style.reason}</p>
                    </div>
                  ))}
                </div>
              </div>
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
                    <div className="grid grid-cols-2 gap-3">
                      <button
                        onClick={() => handleGenerateOverlay(style, 'basic')}
                        disabled={overlayLoadingId === style.id}
                        className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white py-3 px-4 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg disabled:opacity-60"
                      >
                        {overlayLoadingId === style.id && overlayType === 'basic' ? 'Generating…' : 'Preview (Basic)'}
                      </button>
                      <button
                        onClick={() => handleGenerateOverlay(style, 'advanced')}
                        disabled={overlayLoadingId === style.id}
                        className="bg-gradient-to-r from-pink-600 to-rose-600 hover:from-pink-700 hover:to-rose-700 text-white py-3 px-4 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg disabled:opacity-60"
                      >
                        {overlayLoadingId === style.id && overlayType === 'advanced' ? 'Generating…' : 'Preview (Advanced)'}
                      </button>
                    </div>
                    {errorMsg && overlayLoadingId === null && (
                      <p className="text-red-400 text-sm mt-3">{errorMsg}</p>
                    )}
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

      {/* Overlay Modal */}
      {showOverlayModal && overlayUrl && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl max-w-3xl w-full p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-white text-xl font-semibold">Overlay Preview ({overlayType})</h3>
              <button
                onClick={() => setShowOverlayModal(false)}
                className="text-gray-400 hover:text-white"
                aria-label="Close"
              >
                ✕
              </button>
            </div>
            <div className="w-full max-h-[70vh] overflow-auto flex items-center justify-center bg-gray-800 rounded-xl p-2">
              <img src={overlayUrl} alt="Overlay" className="max-w-full max-h-[68vh] rounded-lg" />
            </div>
            <div className="flex justify-end gap-3 mt-4">
              <a
                href={overlayUrl}
                download
                className="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg"
              >
                Download
              </a>
              <button
                onClick={() => setShowOverlayModal(false)}
                className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white px-4 py-2 rounded-lg"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Results;