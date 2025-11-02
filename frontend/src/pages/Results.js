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
  const [favorites, setFavorites] = useState([]);
  
  // New state for Try Hairstyle modal
  const [showTryHairstyleModal, setShowTryHairstyleModal] = useState(false);
  const [currentHairstyleIndex, setCurrentHairstyleIndex] = useState(0);
  const [hairstyleDetails, setHairstyleDetails] = useState(null);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [detailsError, setDetailsError] = useState('');

  useEffect(() => {
    checkAuth();
    loadFavorites();
  }, []);

  const loadFavorites = () => {
    try {
      const storedFavorites = localStorage.getItem('hairstyle_favorites');
      if (storedFavorites) {
        setFavorites(JSON.parse(storedFavorites));
      }
    } catch (error) {
      console.error('Failed to load favorites:', error);
    }
  };

  const toggleFavorite = (hairstyleId, hairstyleName) => {
    try {
      const storedFavorites = localStorage.getItem('hairstyle_favorites');
      let currentFavorites = storedFavorites ? JSON.parse(storedFavorites) : [];
      
      const isFavorite = currentFavorites.some(fav => fav.id === hairstyleId);
      
      if (isFavorite) {
        currentFavorites = currentFavorites.filter(fav => fav.id !== hairstyleId);
      } else {
        currentFavorites.push({
          id: hairstyleId,
          name: hairstyleName,
          timestamp: new Date().toISOString()
        });
      }
      
      localStorage.setItem('hairstyle_favorites', JSON.stringify(currentFavorites));
      setFavorites(currentFavorites);
    } catch (error) {
      console.error('Failed to toggle favorite:', error);
    }
  };

  const isFavorite = (hairstyleId) => {
    return favorites.some(fav => fav.id === hairstyleId);
  };

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

  const handleTryHairstyle = async (index) => {
    try {
      setDetailsError('');
      setLoadingDetails(true);
      setCurrentHairstyleIndex(index);
      setShowTryHairstyleModal(true);

      const style = recommendations.recommendations[index];
      
      // Fetch detailed information
      const details = await apiService.getHairstyleDetailsWithAI(
        style.id,
        preferences?.id || null,
        uploadResponse?.image_id || null
      );
      
      // Generate overlay
      if (uploadResponse?.image_id && style?.id) {
        try {
          const resp = await apiService.generateOverlay(
            uploadResponse.image_id,
            style.id,
            'basic'
          );
          details.overlay_url = resolveMediaUrl(resp.overlay_url);
        } catch (overlayError) {
          console.warn('Overlay generation failed:', overlayError);
          details.overlay_url = null;
        }
      }
      
      setHairstyleDetails(details);
    } catch (e) {
      console.error('Failed to load hairstyle details:', e);
      setDetailsError(e?.message || 'Failed to load hairstyle details');
    } finally {
      setLoadingDetails(false);
    }
  };

  const handleNextHairstyle = () => {
    const nextIndex = (currentHairstyleIndex + 1) % recommendations.recommendations.length;
    handleTryHairstyle(nextIndex);
  };

  const handlePrevHairstyle = () => {
    const prevIndex = currentHairstyleIndex === 0 
      ? recommendations.recommendations.length - 1 
      : currentHairstyleIndex - 1;
    handleTryHairstyle(prevIndex);
  };

  const closeTryHairstyleModal = () => {
    setShowTryHairstyleModal(false);
    setHairstyleDetails(null);
    setDetailsError('');
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
    <div className="min-h-screen bg-gray-900 pt-20 md:pt-0">
      <Navbar 
        user={user} 
        onLogout={handleLogout}
      />
      
      {/* Header Section - Similar to Discover page */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 py-12 md:py-16 md:pt-24">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Your Hairstyle Recommendations
          </h1>
          <p className="text-xl text-gray-200 max-w-3xl mx-auto">
            Based on your preferences and facial analysis
          </p>
        </div>
      </div>
      
      <div className="max-w-7xl mx-auto px-4 py-12">
        {/* Face Analysis Summary */}
        {uploadResponse && (
            <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-8 mb-12 shadow-xl">
              <h2 className="text-2xl font-bold text-white mb-6">Face Analysis</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                {/* User Image */}
                {previewUrl && (
                  <div className="flex justify-center">
                    <div className="relative w-full max-w-xs mx-auto">
                      <div className="aspect-square overflow-hidden rounded-xl shadow-lg border-2 border-blue-500/30">
                        <img
                          src={previewUrl}
                          alt="Uploaded face for analysis"
                          className="w-full h-full object-cover object-center"
                          style={{ objectPosition: 'center 30%' }}
                        />
                      </div>
                      <div className="absolute top-2 right-2 bg-blue-500/90 backdrop-blur-sm text-white px-3 py-1 rounded-full text-xs font-medium">
                        Your Photo
                      </div>
                    </div>
                  </div>
                )}

                {/* Face Shape Info */}
                <div className="space-y-4">
                  <div className="bg-blue-500/20 backdrop-blur-sm border border-blue-400/30 rounded-xl p-6">
                    <h3 className="font-medium text-white mb-3 text-lg">Detected Face Shape</h3>
                    <p className="text-blue-400 font-semibold text-3xl capitalize mb-2">
                      {uploadResponse.face_shape?.shape || 'Unknown'}
                    </p>
                    <p className="text-sm text-gray-400">
                      Detection Method: {uploadResponse.detection_method || 'AI Analysis'}
                    </p>
                  </div>
                  
                  {/* Face shape characteristics */}
                  {uploadResponse.face_shape?.shape && (
                    <div className="p-6 bg-gray-700/30 backdrop-blur-sm border border-gray-600/50 rounded-xl">
                      <h4 className="font-medium text-white mb-3">
                        {uploadResponse.face_shape.shape.charAt(0).toUpperCase() + uploadResponse.face_shape.shape.slice(1)} Face Shape Characteristics:
                      </h4>
                      <div className="text-sm text-gray-300">
                        <p className="mb-3">
                          <strong className="text-white">Best suited for:</strong> Most hairstyles work well with your face shape!
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
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
            
            {recommendations.recommendations && recommendations.recommendations.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {recommendations.recommendations.map((style, index) => (
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
                    <button
                      onClick={() => handleTryHairstyle(index)}
                      className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white py-3 px-4 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg"
                    >
                      Try Hairstyle
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-16">
                <div className="bg-yellow-900/20 border border-yellow-600/30 rounded-xl p-8 max-w-lg mx-auto">
                  <div className="text-yellow-400 text-5xl mb-4">⚠️</div>
                  <h3 className="text-xl font-bold text-white mb-3">No Recommendations Available</h3>
                  <p className="text-gray-300 mb-6">
                    We couldn't generate recommendations based on your preferences. 
                    This might be because there are no matching hairstyles in our database for your specific criteria.
                  </p>
                  <button
                    onClick={() => navigate('/preferences')}
                    className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white px-8 py-3 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg"
                  >
                    Try Different Preferences
                  </button>
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
                state: { 
                  imageFile, 
                  previewUrl, 
                  uploadResponse,
                  existingPreferences: preferences  // Pass existing preferences
                } 
              })}
              className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white px-8 py-4 rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg font-medium"
            >
              Update Preferences
            </button>
          </div>
        </div>

        {/* Try Hairstyle Modal */}
        {showTryHairstyleModal && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4 overflow-y-auto">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl max-w-6xl w-full my-8">
            {loadingDetails ? (
              <div className="p-12 text-center">
                <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mb-4"></div>
                <p className="text-white text-lg">Loading hairstyle details...</p>
              </div>
            ) : detailsError ? (
              <div className="p-12 text-center">
                <div className="text-red-400 text-5xl mb-4">⚠️</div>
                <p className="text-red-400 text-lg mb-6">{detailsError}</p>
                <button
                  onClick={closeTryHairstyleModal}
                  className="bg-gray-700 hover:bg-gray-600 text-white px-6 py-2 rounded-lg"
                >
                  Close
                </button>
              </div>
            ) : hairstyleDetails && recommendations?.recommendations[currentHairstyleIndex] ? (
              <div className="p-6">
                {/* Header with navigation */}
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-bold text-white">
                    {recommendations.recommendations[currentHairstyleIndex].name}
                  </h2>
                  <div className="flex items-center gap-4">
                    <button
                      onClick={() => toggleFavorite(
                        recommendations.recommendations[currentHairstyleIndex].id,
                        recommendations.recommendations[currentHairstyleIndex].name
                      )}
                      className={`p-2 rounded-full transition-all duration-300 ${
                        isFavorite(recommendations.recommendations[currentHairstyleIndex].id)
                          ? 'bg-red-500 text-white hover:bg-red-600'
                          : 'bg-gray-700 text-gray-400 hover:bg-gray-600 hover:text-red-400'
                      }`}
                      aria-label="Add to favorites"
                    >
                      <svg className="w-6 h-6" fill={isFavorite(recommendations.recommendations[currentHairstyleIndex].id) ? "currentColor" : "none"} stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                      </svg>
                    </button>
                    <button
                      onClick={closeTryHairstyleModal}
                      className="text-gray-400 hover:text-white text-2xl"
                      aria-label="Close"
                    >
                      ✕
                    </button>
                  </div>
                </div>

                {/* Main content grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                  {/* Left side - Image and overlay */}
                  <div className="space-y-4">
                    {hairstyleDetails.overlay_url ? (
                      <div className="bg-gray-800 rounded-xl p-4">
                        <h3 className="text-lg font-semibold text-white mb-3">Your Look with {recommendations.recommendations[currentHairstyleIndex].name}</h3>
                        <img
                          src={hairstyleDetails.overlay_url}
                          alt="Hairstyle overlay"
                          className="w-full rounded-lg shadow-lg"
                        />
                      </div>
                    ) : hairstyleDetails.hairstyle?.image_url && (
                      <div className="bg-gray-800 rounded-xl p-4">
                        <h3 className="text-lg font-semibold text-white mb-3">Style Reference</h3>
                        <img
                          src={hairstyleDetails.hairstyle.image_url}
                          alt="Hairstyle reference"
                          className="w-full rounded-lg shadow-lg"
                        />
                      </div>
                    )}
                    
                    {/* Face Shape Info */}
                    <div className="bg-blue-900/30 border border-blue-500/30 rounded-xl p-4">
                      <h3 className="text-lg font-semibold text-white mb-2">Your Face Shape Analysis</h3>
                      <p className="text-blue-300 capitalize text-xl font-semibold mb-2">
                        {uploadResponse?.face_shape?.shape || hairstyleDetails.face_shape || 'Oval'} Face
                      </p>
                      <p className="text-sm text-gray-300 mb-3">
                        Detected via {uploadResponse?.detection_method || 'AI Model Analysis'}
                        {uploadResponse?.face_shape?.confidence && (
                          <span className="ml-2">
                            ({Math.round(uploadResponse.face_shape.confidence * 100)}% confidence)
                          </span>
                        )}
                      </p>
                      <div className="bg-blue-800/30 border border-blue-400/20 rounded-lg p-3 mt-3">
                        <p className="text-sm text-blue-200">
                          <strong>Why this works for you:</strong> This hairstyle is specifically recommended for your {uploadResponse?.face_shape?.shape || hairstyleDetails.face_shape || 'oval'} face shape, creating a balanced and flattering look that complements your natural features.
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Right side - Details */}
                  <div className="space-y-4 overflow-y-auto max-h-[600px]">
                    {/* Personalized Description */}
                    <div className="bg-purple-900/20 border border-purple-500/30 rounded-xl p-4">
                      <h3 className="text-lg font-semibold text-white mb-3">✨ Why This Style Works for You</h3>
                      <p className="text-gray-300 leading-relaxed mb-3">
                        {hairstyleDetails.personalized_description}
                      </p>
                      
                      {/* Face Shape Specific Benefits */}
                      {(uploadResponse?.face_shape?.shape || hairstyleDetails.face_shape) && (
                        <div className="bg-purple-800/20 border border-purple-400/20 rounded-lg p-3 mt-3">
                          <p className="text-sm text-purple-200">
                            <strong className="text-purple-300">Perfect for your {uploadResponse?.face_shape?.shape || hairstyleDetails.face_shape} face:</strong> This hairstyle helps balance your facial proportions, highlights your best features, and creates a harmonious overall look that's tailored to your unique face shape.
                          </p>
                        </div>
                      )}
                    </div>

                    {/* User Preferences Used */}
                    {(hairstyleDetails.user_preferences && Object.keys(hairstyleDetails.user_preferences).length > 0) || uploadResponse?.face_shape?.shape ? (
                      <div className="bg-blue-900/20 border border-blue-500/30 rounded-xl p-4">
                        <h3 className="text-lg font-semibold text-white mb-3">👤 Your Profile & Preferences</h3>
                        
                        {/* Face Shape - Prominently displayed */}
                        {(uploadResponse?.face_shape?.shape || hairstyleDetails.face_shape) && (
                          <div className="bg-blue-800/30 border border-blue-400/30 rounded-lg p-3 mb-3">
                            <div className="flex items-center justify-between">
                              <div>
                                <span className="text-blue-300 font-semibold text-sm uppercase tracking-wide">Face Shape</span>
                                <p className="text-white text-lg font-bold capitalize mt-1">
                                  {uploadResponse?.face_shape?.shape || hairstyleDetails.face_shape}
                                </p>
                              </div>
                              {uploadResponse?.face_shape?.confidence && (
                                <div className="text-right">
                                  <span className="text-green-400 font-bold text-lg">
                                    {Math.round(uploadResponse.face_shape.confidence * 100)}%
                                  </span>
                                  <p className="text-xs text-gray-400">Confidence</p>
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                        
                        {hairstyleDetails.user_preferences && Object.keys(hairstyleDetails.user_preferences).length > 0 && (
                          <div className="grid grid-cols-2 gap-2">
                            {hairstyleDetails.user_preferences.hair_type && (
                              <div className="text-sm">
                                <span className="text-blue-400 font-medium">Hair Type:</span>
                                <span className="text-gray-300 ml-2 capitalize">{hairstyleDetails.user_preferences.hair_type}</span>
                              </div>
                            )}
                            {hairstyleDetails.user_preferences.hair_length && (
                              <div className="text-sm">
                                <span className="text-blue-400 font-medium">Length:</span>
                                <span className="text-gray-300 ml-2 capitalize">{hairstyleDetails.user_preferences.hair_length}</span>
                              </div>
                            )}
                            {hairstyleDetails.user_preferences.hair_thickness && (
                              <div className="text-sm">
                                <span className="text-blue-400 font-medium">Thickness:</span>
                                <span className="text-gray-300 ml-2 capitalize">{hairstyleDetails.user_preferences.hair_thickness}</span>
                              </div>
                            )}
                            {hairstyleDetails.user_preferences.hair_texture_detail && (
                              <div className="text-sm">
                                <span className="text-blue-400 font-medium">Texture:</span>
                                <span className="text-gray-300 ml-2 capitalize">{hairstyleDetails.user_preferences.hair_texture_detail}</span>
                              </div>
                            )}
                            {hairstyleDetails.user_preferences.maintenance && (
                              <div className="text-sm">
                                <span className="text-blue-400 font-medium">Maintenance:</span>
                                <span className="text-gray-300 ml-2 capitalize">{hairstyleDetails.user_preferences.maintenance}</span>
                              </div>
                            )}
                            {hairstyleDetails.user_preferences.lifestyle && (
                              <div className="text-sm">
                                <span className="text-blue-400 font-medium">Lifestyle:</span>
                                <span className="text-gray-300 ml-2 capitalize">{hairstyleDetails.user_preferences.lifestyle}</span>
                              </div>
                            )}
                            {hairstyleDetails.user_preferences.gender && (
                              <div className="text-sm">
                                <span className="text-blue-400 font-medium">Gender:</span>
                                <span className="text-gray-300 ml-2 capitalize">{hairstyleDetails.user_preferences.gender}</span>
                              </div>
                            )}
                            {hairstyleDetails.user_preferences.occasions && hairstyleDetails.user_preferences.occasions.length > 0 && (
                              <div className="text-sm col-span-2">
                                <span className="text-blue-400 font-medium">Occasions:</span>
                                <span className="text-gray-300 ml-2 capitalize">{hairstyleDetails.user_preferences.occasions.join(', ')}</span>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ) : null}

                    {/* LLM INTELLIGENT ANALYSIS */}
                    {hairstyleDetails.llm_analysis && hairstyleDetails.llm_analysis.llm_generated && (
                      <>
                        {/* Compatibility Score */}
                        {hairstyleDetails.llm_analysis.compatibility_score > 0 && (
                          <div className="bg-gradient-to-r from-indigo-900/30 to-purple-900/30 border border-indigo-500/40 rounded-xl p-4">
                            <div className="flex items-center justify-between">
                              <h3 className="text-lg font-semibold text-white">🎯 AI Compatibility Score</h3>
                              <div className="flex items-center">
                                <span className="text-4xl font-bold text-indigo-400">
                                  {hairstyleDetails.llm_analysis.compatibility_score}
                                </span>
                                <span className="text-gray-400 text-xl ml-1">/100</span>
                              </div>
                            </div>
                            <div className="mt-2 bg-gray-700/50 rounded-full h-3 overflow-hidden">
                              <div 
                                className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-500"
                                style={{ width: `${hairstyleDetails.llm_analysis.compatibility_score}%` }}
                              />
                            </div>
                          </div>
                        )}

                        {/* Intelligent Summary */}
                        {hairstyleDetails.llm_analysis.intelligent_summary && (
                          <div className="bg-gradient-to-r from-purple-900/30 to-pink-900/30 border border-purple-500/40 rounded-xl p-4">
                            <h3 className="text-lg font-semibold text-white mb-3">🧠 AI Analysis</h3>
                            <p className="text-gray-300 leading-relaxed">
                              {hairstyleDetails.llm_analysis.intelligent_summary}
                            </p>
                          </div>
                        )}

                        {/* Preference Insights */}
                        {hairstyleDetails.llm_analysis.preference_insights && hairstyleDetails.llm_analysis.preference_insights.length > 0 && (
                          <div className="bg-green-900/20 border border-green-500/30 rounded-xl p-4">
                            <h3 className="text-lg font-semibold text-white mb-3">✓ Preference Match Insights</h3>
                            <ul className="space-y-2">
                              {hairstyleDetails.llm_analysis.preference_insights.map((insight, idx) => (
                                <li key={idx} className="flex items-start">
                                  <span className="text-green-400 mr-2">•</span>
                                  <span className="text-gray-300 text-sm">{insight}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Lifestyle Fit */}
                        {hairstyleDetails.llm_analysis.lifestyle_fit && (
                          <div className="bg-teal-900/20 border border-teal-500/30 rounded-xl p-4">
                            <h3 className="text-lg font-semibold text-white mb-3">🌟 Lifestyle Fit</h3>
                            <p className="text-gray-300 leading-relaxed">
                              {hairstyleDetails.llm_analysis.lifestyle_fit}
                            </p>
                          </div>
                        )}

                        {/* Styling Intelligence */}
                        {hairstyleDetails.llm_analysis.styling_intelligence && hairstyleDetails.llm_analysis.styling_intelligence.length > 0 && (
                          <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-xl p-4">
                            <h3 className="text-lg font-semibold text-white mb-3">💡 Smart Styling Tips</h3>
                            <ul className="space-y-2">
                              {hairstyleDetails.llm_analysis.styling_intelligence.map((tip, idx) => (
                                <li key={idx} className="text-gray-300 text-sm flex items-start">
                                  <span className="text-yellow-400 mr-2">{idx + 1}.</span>
                                  <span>{tip}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Product Recommendations from LLM */}
                        {hairstyleDetails.llm_analysis.product_recommendations && hairstyleDetails.llm_analysis.product_recommendations.length > 0 && (
                          <div className="bg-orange-900/20 border border-orange-500/30 rounded-xl p-4">
                            <h3 className="text-lg font-semibold text-white mb-3">🛍️ Personalized Product Recommendations</h3>
                            <ul className="space-y-2">
                              {hairstyleDetails.llm_analysis.product_recommendations.map((product, idx) => (
                                <li key={idx} className="text-gray-300 text-sm flex items-start">
                                  <span className="text-orange-400 mr-2">{idx + 1}.</span>
                                  <span>{product}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Maintenance Reality */}
                        {hairstyleDetails.llm_analysis.maintenance_reality && hairstyleDetails.llm_analysis.maintenance_reality.length > 0 && (
                          <div className="bg-cyan-900/20 border border-cyan-500/30 rounded-xl p-4">
                            <h3 className="text-lg font-semibold text-white mb-3">🔧 Maintenance Reality</h3>
                            <ul className="space-y-2">
                              {hairstyleDetails.llm_analysis.maintenance_reality.map((item, idx) => (
                                <li key={idx} className="text-gray-300 text-sm flex items-start">
                                  <span className="text-cyan-400 mr-2">•</span>
                                  <span>{item}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Professional Tips */}
                        {hairstyleDetails.llm_analysis.professional_tips && hairstyleDetails.llm_analysis.professional_tips.length > 0 && (
                          <div className="bg-pink-900/20 border border-pink-500/30 rounded-xl p-4">
                            <h3 className="text-lg font-semibold text-white mb-3">👨‍🔬 Professional Tips</h3>
                            <ul className="space-y-2">
                              {hairstyleDetails.llm_analysis.professional_tips.map((tip, idx) => (
                                <li key={idx} className="text-gray-300 text-sm flex items-start">
                                  <span className="text-pink-400 mr-2">•</span>
                                  <span>{tip}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </>
                    )}

                    {/* Fallback: Original Preference Match (if no LLM analysis) */}
                    {(!hairstyleDetails.llm_analysis || !hairstyleDetails.llm_analysis.llm_generated) && 
                     hairstyleDetails.preference_match && hairstyleDetails.preference_match.length > 0 && (
                      <div className="bg-green-900/20 border border-green-500/30 rounded-xl p-4">
                        <h3 className="text-lg font-semibold text-white mb-3">✓ How It Fits Your Preferences</h3>
                        <ul className="space-y-2">
                          {hairstyleDetails.preference_match.map((match, idx) => (
                            <li key={idx} className="flex items-start">
                              <span className="text-green-400 mr-2">•</span>
                              <span className="text-gray-300 text-sm">{match}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Recommended Products */}
                    {hairstyleDetails.recommended_products && hairstyleDetails.recommended_products.length > 0 && (
                      <div className="bg-orange-900/20 border border-orange-500/30 rounded-xl p-4">
                        <h3 className="text-lg font-semibold text-white mb-3">🛍️ Recommended Products</h3>
                        <ul className="space-y-2">
                          {hairstyleDetails.recommended_products.map((product, idx) => (
                            <li key={idx} className="text-gray-300 text-sm flex items-start">
                              <span className="text-orange-400 mr-2">{idx + 1}.</span>
                              <span>{product}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Maintenance Guide */}
                    {hairstyleDetails.maintenance_guide && hairstyleDetails.maintenance_guide.length > 0 && (
                      <div className="bg-blue-900/20 border border-blue-500/30 rounded-xl p-4">
                        <h3 className="text-lg font-semibold text-white mb-3">🔧 Maintenance Guide</h3>
                        <ol className="space-y-2">
                          {hairstyleDetails.maintenance_guide.map((step, idx) => (
                            <li key={idx} className="text-gray-300 text-sm flex items-start">
                              <span className="text-blue-400 font-medium mr-2">{idx + 1}.</span>
                              <span>{step}</span>
                            </li>
                          ))}
                        </ol>
                      </div>
                    )}

                    {/* Styling Tips */}
                    {hairstyleDetails.styling_tips && hairstyleDetails.styling_tips.length > 0 && (
                      <div className="bg-pink-900/20 border border-pink-500/30 rounded-xl p-4">
                        <h3 className="text-lg font-semibold text-white mb-3">💡 Pro Styling Tips</h3>
                        <ul className="space-y-2">
                          {hairstyleDetails.styling_tips.map((tip, idx) => (
                            <li key={idx} className="text-gray-300 text-sm flex items-start">
                              <span className="text-pink-400 mr-2">→</span>
                              <span>{tip}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>

                {/* Navigation buttons */}
                <div className="flex items-center justify-between pt-6 border-t border-gray-700">
                  <button
                    onClick={handlePrevHairstyle}
                    className="flex items-center gap-2 bg-gray-700 hover:bg-gray-600 text-white px-6 py-3 rounded-lg transition-colors duration-300"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                    Previous Style
                  </button>

                  <div className="text-center">
                    <p className="text-gray-400 text-sm">
                      {currentHairstyleIndex + 1} of {recommendations.recommendations.length}
                    </p>
                  </div>

                  <button
                    onClick={handleNextHairstyle}
                    className="flex items-center gap-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white px-6 py-3 rounded-lg transition-all duration-300"
                  >
                    Next Style
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
};

export default Results;