import React, { useState, useEffect, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import AuthService from '../services/AuthService';
import apiService from '../services/api';
import Navbar from '../components/Navbar';

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { preferences, imageFile, previewUrl, uploadResponse, recommendations } = location.state || {};
  
  const [user, setUser] = useState(null);
  const [savedHairstyles, setSavedHairstyles] = useState([]);
  
  // New state for Try Hairstyle modal
  const [showTryHairstyleModal, setShowTryHairstyleModal] = useState(false);
  const [currentHairstyleIndex, setCurrentHairstyleIndex] = useState(0);
  const [hairstyleDetails, setHairstyleDetails] = useState(null);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [detailsError, setDetailsError] = useState('');
  
  // Cache for hairstyle details to avoid redundant API calls
  const [hairstyleDetailsCache, setHairstyleDetailsCache] = useState({});
  
  // State for image modal
  const [showImageModal, setShowImageModal] = useState(false);

  // AbortController for canceling overlay generation
  const [overlayAbortController, setOverlayAbortController] = useState(null);

  // Track which hairstyles have been saved in this session
  const [savedThisSession, setSavedThisSession] = useState(new Set());

  const loadSavedHairstyles = useCallback(async () => {
    try {
      if (!user?.id) return;

      // Load from backend API
      const saved = await apiService.getSavedHairstyles();
      setSavedHairstyles(saved || []);
    } catch (error) {
      console.error('Failed to load saved hairstyles:', error);
      setSavedHairstyles([]);
    }
  }, [user?.id]);

  useEffect(() => {
    checkAuth();
  }, []);

  // Load saved hairstyles when user changes
  useEffect(() => {
    if (user) {
      loadSavedHairstyles();
    } else {
      setSavedHairstyles([]);
    }
  }, [user, loadSavedHairstyles]);

  // Add keyboard support for closing image modal
  useEffect(() => {
    const handleEscKey = (e) => {
      if (e.key === 'Escape' && showImageModal) {
        setShowImageModal(false);
      }
    };

    document.addEventListener('keydown', handleEscKey);
    return () => document.removeEventListener('keydown', handleEscKey);
  }, [showImageModal]);

  // Prevent background scrolling when modals are open
  useEffect(() => {
    if (showTryHairstyleModal || showImageModal) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }

    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [showTryHairstyleModal, showImageModal]);

  const saveHairstyleRecommendation = async (hairstyleId, hairstyleName, recommendation) => {
    try {
      if (!user?.id) {
        console.error('User must be logged in to save hairstyles');
        return;
      }

      // Check if already saved in this session - if so, unsave it
      if (savedThisSession.has(hairstyleId)) {
        // Find the saved hairstyle(s) for this hairstyle ID
        const savedItems = savedHairstyles.filter(
          saved => (saved.hairstyle_id === hairstyleId) || (saved.hairstyle?.id === hairstyleId)
        );
        
        // Delete the most recent one
        if (savedItems.length > 0) {
          const mostRecent = savedItems[savedItems.length - 1];
          await apiService.deleteSavedHairstyle(mostRecent.id);
          
          // Remove from session tracking
          setSavedThisSession(prev => {
            const newSet = new Set(prev);
            newSet.delete(hairstyleId);
            return newSet;
          });
          
          // Reload saved hairstyles
          await loadSavedHairstyles();
        }
        return;
      }

      // Save the hairstyle
      const savedData = {
        hairstyle_id: hairstyleId,
        hairstyle_name: hairstyleName,
        recommendation_data: recommendation,
        face_shape: uploadResponse?.face_shape?.shape || null,
        face_shape_confidence: uploadResponse?.face_shape?.confidence || null,
        user_preferences: preferences || {},
        overlay_url: hairstyleDetails?.overlay_url || null,
        personalized_description: hairstyleDetails?.personalized_description || null,
      };

      await apiService.saveHairstyle(savedData);

      // Mark as saved in this session
      setSavedThisSession(prev => new Set([...prev, hairstyleId]));

      // Reload saved hairstyles
      await loadSavedHairstyles();
    } catch (error) {
      console.error('Failed to save/unsave hairstyle:', error);
    }
  };

  const isHairstyleSaved = (hairstyleId) => {
    return savedHairstyles.some(
      saved => (saved.hairstyle_id === hairstyleId) ||
      (saved.hairstyle?.id === hairstyleId)
    );
  };

  const getHairstyleSaveCount = (hairstyleId) => {
    return savedHairstyles.filter(
      saved => (saved.hairstyle_id === hairstyleId) ||
      (saved.hairstyle?.id === hairstyleId)
    ).length;
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
      setCurrentHairstyleIndex(index);
      setShowTryHairstyleModal(true);

      const style = recommendations.recommendations[index];
      const cacheKey = style.id;
      
      // Check if we already have cached details for this hairstyle
      if (hairstyleDetailsCache[cacheKey]) {
        console.log('Using cached hairstyle details for:', style.name);
        setHairstyleDetails(hairstyleDetailsCache[cacheKey]);
        return;
      }
      
      // If not cached, fetch from API
      setLoadingDetails(true);

      // Fetch detailed information
      const details = await apiService.getHairstyleDetailsWithAI(
        style.id,
        preferences?.id || null,
        uploadResponse?.image_id || null
      );

      // Generate overlay with AbortController for cancellation
      if (uploadResponse?.image_id && style?.id) {
        try {
          // Create new AbortController for this overlay request
          const controller = new AbortController();
          setOverlayAbortController(controller);

          const resp = await apiService.generateOverlay(
            uploadResponse.image_id,
            style.id,
            'advanced',
            controller.signal, // Pass the abort signal
            true // Use hair color from user preferences
          );
          details.overlay_url = resolveMediaUrl(resp.overlay_url);

          // Clear the abort controller after successful completion
          setOverlayAbortController(null);
        } catch (overlayError) {
          if (overlayError.name === 'AbortError') {
            console.log('Overlay generation was cancelled');
            setDetailsError('Overlay generation was cancelled');
            setLoadingDetails(false);
            setOverlayAbortController(null);
            return;
          }
          console.warn('Overlay generation failed:', overlayError);
          details.overlay_url = null;
          setOverlayAbortController(null);
        }
      }

      setHairstyleDetails(details);

      // Cache the details for future use
      setHairstyleDetailsCache(prev => ({
        ...prev,
        [cacheKey]: details
      }));
    } catch (e) {
      console.error('Failed to load hairstyle details:', e);
      setDetailsError(e?.message || 'Failed to load hairstyle details');
    } finally {
      setLoadingDetails(false);
      setOverlayAbortController(null);
    }
  };

  const cancelOverlayGeneration = () => {
    if (overlayAbortController) {
      console.log('Cancelling overlay generation...');
      overlayAbortController.abort();
      setOverlayAbortController(null);
      setLoadingDetails(false);
      setShowTryHairstyleModal(false);
      setHairstyleDetails(null);
      setDetailsError('');
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
                  </div>
                  
                  {/* Face shape characteristics */}
                  {uploadResponse.face_shape?.shape && (
                    <div className="p-6 bg-gray-700/30 backdrop-blur-sm border border-gray-600/50 rounded-xl">
                      <h4 className="font-medium text-white mb-3">
                        {uploadResponse.face_shape.shape.charAt(0).toUpperCase() + uploadResponse.face_shape.shape.slice(1)} Face Shape Characteristics:
                      </h4>
                      <div className="text-sm text-gray-300 space-y-2">
                        {uploadResponse.face_shape.shape.toLowerCase() === 'oval' && (
                          <>
                            <p><strong className="text-white">Proportions:</strong> Well-balanced with slightly wider cheekbones than forehead and jawline.</p>
                            <p><strong className="text-white">Best suited for:</strong> Almost all hairstyles work beautifully with your face shape. You have the most versatile canvas!</p>
                            <p><strong className="text-white">Style tip:</strong> Experiment with different lengths and textures to showcase your balanced features.</p>
                          </>
                        )}
                        {uploadResponse.face_shape.shape.toLowerCase() === 'round' && (
                          <>
                            <p><strong className="text-white">Proportions:</strong> Soft curves with similar width and length, fuller cheeks and a rounded chin.</p>
                            <p><strong className="text-white">Best suited for:</strong> Hairstyles with height and volume at the crown help elongate your face. Angular cuts and side-swept styles work wonderfully.</p>
                            <p><strong className="text-white">Style tip:</strong> Add layers and avoid blunt cuts at chin level to create flattering dimension.</p>
                          </>
                        )}
                        {uploadResponse.face_shape.shape.toLowerCase() === 'square' && (
                          <>
                            <p><strong className="text-white">Proportions:</strong> Strong, defined jawline with forehead, cheekbones, and jaw of similar width.</p>
                            <p><strong className="text-white">Best suited for:</strong> Soft, layered styles and waves that soften angular features. Side parts and face-framing pieces are ideal.</p>
                            <p><strong className="text-white">Style tip:</strong> Textured ends and wispy layers will beautifully complement your strong features.</p>
                          </>
                        )}
                        {uploadResponse.face_shape.shape.toLowerCase() === 'heart' && (
                          <>
                            <p><strong className="text-white">Proportions:</strong> Wider forehead with high cheekbones tapering to a narrow, pointed chin.</p>
                            <p><strong className="text-white">Best suited for:</strong> Chin-length bobs, side-swept bangs, and styles with volume at the jawline balance your proportions perfectly.</p>
                            <p><strong className="text-white">Style tip:</strong> Add width at chin level and consider soft, wispy bangs to flatter your forehead.</p>
                          </>
                        )}
                        {uploadResponse.face_shape.shape.toLowerCase() === 'oblong' && (
                          <>
                            <p><strong className="text-white">Proportions:</strong> Longer face with forehead, cheeks, and jawline of similar width.</p>
                            <p><strong className="text-white">Best suited for:</strong> Medium-length cuts with horizontal layers, bangs, and styles that add width to the sides.</p>
                            <p><strong className="text-white">Style tip:</strong> Create width with waves and curls while avoiding too much height at the crown.</p>
                          </>
                        )}
                        {uploadResponse.face_shape.shape.toLowerCase() === 'diamond' && (
                          <>
                            <p><strong className="text-white">Proportions:</strong> Narrow forehead and jawline with prominent, wide cheekbones.</p>
                            <p><strong className="text-white">Best suited for:</strong> Chin-length styles, side-swept bangs, and volume at the crown or chin to balance your striking cheekbones.</p>
                            <p><strong className="text-white">Style tip:</strong> Highlight your cheekbones while adding fullness at the forehead and jaw areas.</p>
                          </>
                        )}
                        {uploadResponse.face_shape.shape.toLowerCase() === 'triangle' && (
                          <>
                            <p><strong className="text-white">Proportions:</strong> Narrow forehead with a wider jawline, creating an inverted triangle.</p>
                            <p><strong className="text-white">Best suited for:</strong> Styles with volume at the crown and temples, side-swept bangs, and shorter lengths that add width up top.</p>
                            <p><strong className="text-white">Style tip:</strong> Balance your strong jawline with volume and texture in the upper portions of your hairstyle.</p>
                          </>
                        )}
                        {!['oval', 'round', 'square', 'heart', 'oblong', 'diamond', 'triangle'].includes(uploadResponse.face_shape.shape.toLowerCase()) && (
                          <p><strong className="text-white">Best suited for:</strong> Most hairstyles work well with your unique face shape!</p>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-12 px-4">
            <button
              onClick={() => navigate('/upload')}
              className="w-full sm:w-auto bg-gray-700/70 backdrop-blur-sm border border-gray-600/50 text-white px-6 sm:px-8 py-3 sm:py-4 rounded-xl hover:bg-gray-600/70 transition-all duration-300 transform hover:scale-105 font-medium text-sm sm:text-base"
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
              className="w-full sm:w-auto bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white px-6 sm:px-8 py-3 sm:py-4 rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg font-medium text-sm sm:text-base"
            >
              Update Preferences
            </button>
          </div>

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
                    <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-4">
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
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
                {recommendations.recommendations.map((style, index) => (
                  <div key={index} className="bg-gray-700/30 backdrop-blur-sm border border-gray-600/50 rounded-xl p-4 sm:p-6 hover:shadow-xl hover:shadow-purple-500/10 transition-all duration-300 transform hover:scale-105 group">
                    {style.image_url && (
                      <img
                        src={style.image_url}
                        alt={style.name}
                        className="w-full h-40 sm:h-48 object-cover rounded-xl mb-3 sm:mb-4 group-hover:scale-110 transition-transform duration-300"
                      />
                    )}
                    <h3 className="font-semibold text-lg sm:text-xl text-white mb-2 sm:mb-3 group-hover:text-purple-400 transition-colors duration-300">
                      {style.name}
                    </h3>
                    <p className="text-gray-300 mb-4 sm:mb-6 leading-relaxed text-sm sm:text-base line-clamp-3">
                      {style.description}
                    </p>
                    <div className="flex justify-between items-center mb-4 sm:mb-6">
                      {style.match_score && Math.round(style.match_score * 100) >= 50 ? (
                        <span className="bg-blue-500/20 text-blue-300 border border-blue-400/30 text-sm font-medium px-3 py-1 rounded-full">
                          {Math.round(style.match_score * 100)}% Match
                        </span>
                      ) : (
                        <span className="bg-purple-500/20 text-purple-300 border border-purple-400/30 text-sm font-medium px-3 py-1 rounded-full">
                          Recommended
                        </span>
                      )}
                    </div>
                    <button
                      onClick={() => handleTryHairstyle(index)}
                      className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white py-2.5 sm:py-3 px-4 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg text-sm sm:text-base"
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
        </div>

        {/* Try Hairstyle Modal */}
        {showTryHairstyleModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center p-2 sm:p-4 overflow-y-auto">
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 border border-white/10 rounded-xl sm:rounded-2xl shadow-2xl max-w-7xl w-full max-h-[98vh] sm:max-h-[95vh] overflow-hidden flex flex-col my-2 sm:my-4">
            {loadingDetails ? (
              <div className="p-8 sm:p-12 text-center">
                <div className="inline-block animate-spin rounded-full h-10 w-10 sm:h-12 sm:w-12 border-b-2 border-purple-500 mb-4"></div>
                <p className="text-white text-base sm:text-lg mb-6">Loading hairstyle details and generating overlay...</p>
                <button
                  onClick={cancelOverlayGeneration}
                  className="bg-red-600 hover:bg-red-700 text-white px-6 py-2.5 rounded-lg transition-all duration-300 font-medium shadow-lg"
                >
                  Cancel
                </button>
              </div>
            ) : detailsError ? (
              <div className="p-8 sm:p-12 text-center">
                <div className="text-red-400 text-4xl sm:text-5xl mb-4">⚠️</div>
                <p className="text-red-400 text-base sm:text-lg mb-6">{detailsError}</p>
                <button
                  onClick={closeTryHairstyleModal}
                  className="bg-gray-700 hover:bg-gray-600 text-white px-6 py-2 rounded-lg text-sm sm:text-base"
                >
                  Close
                </button>
              </div>
            ) : hairstyleDetails && recommendations?.recommendations[currentHairstyleIndex] ? (
              <div className="flex flex-col h-full overflow-hidden">
                {/* Header with navigation - Fixed at top */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 sm:p-6 border-b border-white/10 flex-shrink-0">
                  <h2 className="text-xl sm:text-2xl md:text-3xl font-bold bg-gradient-to-r from-white to-blue-200 bg-clip-text text-transparent pr-2">
                    {recommendations.recommendations[currentHairstyleIndex].name}
                  </h2>
                  <div className="flex items-center gap-2 sm:gap-4 flex-shrink-0">
                    <button
                      onClick={() => saveHairstyleRecommendation(
                        recommendations.recommendations[currentHairstyleIndex].id,
                        recommendations.recommendations[currentHairstyleIndex].name,
                        recommendations.recommendations[currentHairstyleIndex]
                      )}
                      className={`flex items-center gap-1.5 px-3 py-2 rounded-lg transition-all duration-300 text-sm ${
                        savedThisSession.has(recommendations.recommendations[currentHairstyleIndex].id)
                          ? 'bg-green-600 text-white hover:bg-green-700'
                          : isHairstyleSaved(recommendations.recommendations[currentHairstyleIndex].id)
                          ? 'bg-blue-600 text-white hover:bg-blue-700'
                          : 'bg-white/10 text-gray-300 hover:bg-white/20 hover:text-white border border-white/20'
                      }`}
                      aria-label="Save or unsave hairstyle recommendation"
                    >
                      <svg className="w-4 h-4" fill={savedThisSession.has(recommendations.recommendations[currentHairstyleIndex].id) || isHairstyleSaved(recommendations.recommendations[currentHairstyleIndex].id) ? "currentColor" : "none"} stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                      </svg>
                      <span className="font-medium">
                        {savedThisSession.has(recommendations.recommendations[currentHairstyleIndex].id)
                          ? 'Unsave'
                          : isHairstyleSaved(recommendations.recommendations[currentHairstyleIndex].id) 
                          ? `Save Again (${getHairstyleSaveCount(recommendations.recommendations[currentHairstyleIndex].id)} saved)` 
                          : 'Save'}
                      </span>
                    </button>
                    <button
                      onClick={closeTryHairstyleModal}
                      className="text-gray-400 hover:text-white transition-colors duration-300 p-2 hover:bg-white/10 rounded-lg flex-shrink-0"
                      aria-label="Close"
                    >
                      <svg className="w-5 h-5 sm:w-6 sm:h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                </div>

                {/* Main content grid - Split layout */}
                <div className="flex-1 overflow-hidden px-3 sm:px-4 lg:px-6 py-3 sm:py-4">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 sm:gap-4 lg:gap-6 h-full">
                    {/* Left side - Image and Face Shape (Fixed, no scroll) */}
                    <div className="space-y-2 sm:space-y-3 lg:space-y-4 flex flex-col overflow-y-auto lg:overflow-y-visible" style={{ maxHeight: 'calc(100vh - 200px)' }}>
                      {hairstyleDetails.overlay_url ? (
                        <div className="bg-white/5 backdrop-blur-sm rounded-lg sm:rounded-xl p-2 sm:p-3 border border-white/10">
                          <h3 className="text-sm sm:text-base lg:text-lg font-semibold text-blue-400 mb-1.5 sm:mb-2 px-1">Your Look with {recommendations.recommendations[currentHairstyleIndex].name}</h3>
                          <div 
                            className="w-full h-56 sm:h-72 lg:h-80 xl:h-96 bg-gradient-to-br from-purple-600/20 to-blue-600/20 rounded-lg sm:rounded-xl flex items-center justify-center border border-white/10 overflow-hidden cursor-pointer hover:border-blue-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/20"
                            onClick={() => setShowImageModal(true)}
                            title="Click to view full size"
                          >
                            <img
                              src={hairstyleDetails.overlay_url}
                              alt="Hairstyle overlay"
                              className="w-auto h-full max-w-full object-contain"
                            />
                          </div>
                          <p className="text-xs text-gray-400 text-center mt-1.5 sm:mt-2">Click image to view full size</p>
                        </div>
                      ) : hairstyleDetails.hairstyle?.image_url ? (
                        <div className="bg-white/5 backdrop-blur-sm rounded-lg sm:rounded-xl p-2 sm:p-3 border border-white/10">
                          <h3 className="text-sm sm:text-base lg:text-lg font-semibold text-blue-400 mb-1.5 sm:mb-2 px-1">Style Reference</h3>
                          <div 
                            className="w-full h-56 sm:h-72 lg:h-80 xl:h-96 bg-gradient-to-br from-purple-600/20 to-blue-600/20 rounded-lg sm:rounded-xl flex items-center justify-center border border-white/10 overflow-hidden cursor-pointer hover:border-blue-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/20"
                            onClick={() => setShowImageModal(true)}
                            title="Click to view full size"
                          >
                            <img
                              src={hairstyleDetails.hairstyle.image_url}
                              alt="Hairstyle reference"
                              className="w-full h-full object-cover"
                            />
                          </div>
                          <p className="text-xs text-gray-400 text-center mt-1.5 sm:mt-2">Click image to view full size</p>
                        </div>
                      ) : (
                        <div className="bg-white/5 backdrop-blur-sm rounded-lg sm:rounded-xl p-2 sm:p-3 border border-white/10">
                          <h3 className="text-sm sm:text-base lg:text-lg font-semibold text-blue-400 mb-1.5 sm:mb-2 px-1">Style Preview</h3>
                          <div className="w-full h-56 sm:h-72 lg:h-80 xl:h-96 bg-gradient-to-br from-purple-600/20 to-blue-600/20 rounded-lg sm:rounded-xl flex items-center justify-center border border-white/10">
                            <div className="text-4xl sm:text-5xl lg:text-7xl">💇‍♀️</div>
                          </div>
                        </div>
                      )}
                      
                      {/* Face Shape Info - Compact on mobile */}
                      <div className="bg-white/5 backdrop-blur-sm rounded-lg sm:rounded-xl p-2 sm:p-3 border border-white/10 lg:hidden">
                        <div className="flex items-center justify-between gap-2">
                          <div>
                            <span className="text-blue-300 font-semibold text-xs uppercase tracking-wide">Face Shape</span>
                            <p className="text-white text-sm font-bold capitalize">
                              {uploadResponse?.face_shape?.shape || hairstyleDetails.face_shape || 'Oval'}
                            </p>
                          </div>
                          {uploadResponse?.face_shape?.confidence && (
                            <div className="text-right flex-shrink-0">
                              <span className="text-green-400 font-bold text-sm">
                                {Math.round(uploadResponse.face_shape.confidence * 100)}%
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                      
                      {/* Face Shape Info - Full on desktop */}
                      <div className="bg-white/5 backdrop-blur-sm rounded-xl p-3 border border-white/10 hidden lg:block">
                        <h3 className="text-base lg:text-lg font-semibold text-blue-400 mb-2 px-1">Your Face Shape Analysis</h3>
                        <div className="space-y-2 px-1">
                          <div className="flex items-center justify-between gap-4">
                            <div>
                              <span className="text-blue-300 font-semibold text-xs uppercase tracking-wide">Face Shape</span>
                              <p className="text-white text-lg lg:text-xl font-bold capitalize mt-1">
                                {uploadResponse?.face_shape?.shape || hairstyleDetails.face_shape || 'Oval'}
                              </p>
                            </div>
                            {uploadResponse?.face_shape?.confidence && (
                              <div className="text-right flex-shrink-0">
                                <span className="text-green-400 font-bold text-base lg:text-lg">
                                  {Math.round(uploadResponse.face_shape.confidence * 100)}%
                                </span>
                                <p className="text-xs text-gray-400">Confidence</p>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Right side - Details (Scrollable) */}
                    <div className="space-y-2 sm:space-y-3 lg:space-y-4 overflow-y-auto pr-1 sm:pr-2 pb-20 sm:pb-24 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent" style={{ maxHeight: 'calc(100vh - 200px)' }}>
                      {/* Personalized Description */}
                      <div className="bg-white/5 backdrop-blur-sm rounded-lg p-2 sm:p-3 lg:p-4 border border-white/10">
                        <h3 className="text-sm sm:text-base lg:text-lg font-semibold text-blue-400 mb-1.5 sm:mb-2 lg:mb-3">✨ Why This Style Works for You</h3>
                        <p className="text-xs sm:text-sm lg:text-base text-gray-300 leading-relaxed text-justify">
                          {hairstyleDetails.personalized_description}
                        </p>
                        
                        {/* Face Shape Specific Benefits */}
                        {(uploadResponse?.face_shape?.shape || hairstyleDetails.face_shape) && (
                          <div className="bg-white/5 border border-white/10 rounded-lg p-2 sm:p-2.5 lg:p-3 mt-1.5 sm:mt-2 lg:mt-3">
                            <p className="text-xs sm:text-sm lg:text-base text-gray-300 leading-relaxed text-justify">
                              <strong className="text-blue-300">Perfect for your {uploadResponse?.face_shape?.shape || hairstyleDetails.face_shape} face:</strong> This hairstyle helps balance your facial proportions, highlights your best features, and creates a harmonious overall look that's tailored to your unique face shape.
                            </p>
                          </div>
                        )}
                      </div>

                      {/* User Preferences Used */}
                      {(hairstyleDetails.user_preferences && Object.keys(hairstyleDetails.user_preferences).length > 0) ? (
                        <div className="bg-white/5 backdrop-blur-sm rounded-lg p-2 sm:p-3 lg:p-4 border border-white/10">
                          <h3 className="text-sm sm:text-base lg:text-lg font-semibold text-blue-400 mb-2 sm:mb-3 lg:mb-4">👤 Your Profile & Preferences</h3>

                          <div className="grid grid-cols-2 gap-2 sm:gap-2.5 lg:gap-3">
                            {hairstyleDetails.user_preferences.hair_type && (
                              <div className="bg-white/5 rounded-lg p-2 lg:p-2.5 border border-white/10">
                                <h4 className="text-xs font-semibold text-purple-400 mb-0.5 lg:mb-1">Hair Type</h4>
                                <p className="text-sm lg:text-base text-white font-medium capitalize">{hairstyleDetails.user_preferences.hair_type}</p>
                              </div>
                            )}
                            {hairstyleDetails.user_preferences.hair_length && (
                              <div className="bg-white/5 rounded-lg p-2 lg:p-2.5 border border-white/10">
                                <h4 className="text-xs font-semibold text-purple-400 mb-0.5 lg:mb-1">Length</h4>
                                <p className="text-sm lg:text-base text-white font-medium capitalize">{hairstyleDetails.user_preferences.hair_length}</p>
                              </div>
                            )}
                            {hairstyleDetails.user_preferences.hair_thickness && (
                              <div className="bg-white/5 rounded-lg p-2 lg:p-2.5 border border-white/10">
                                <h4 className="text-xs font-semibold text-purple-400 mb-0.5 lg:mb-1">Thickness</h4>
                                <p className="text-sm lg:text-base text-white font-medium capitalize">{hairstyleDetails.user_preferences.hair_thickness}</p>
                              </div>
                            )}
                            {hairstyleDetails.user_preferences.hair_texture_detail && (
                              <div className="bg-white/5 rounded-lg p-2 lg:p-2.5 border border-white/10">
                                <h4 className="text-xs font-semibold text-purple-400 mb-0.5 lg:mb-1">Texture</h4>
                                <p className="text-sm lg:text-base text-white font-medium capitalize">{hairstyleDetails.user_preferences.hair_texture_detail}</p>
                              </div>
                            )}
                            {hairstyleDetails.user_preferences.maintenance && (
                              <div className="bg-white/5 rounded-lg p-2 lg:p-2.5 border border-white/10">
                                <h4 className="text-xs font-semibold text-purple-400 mb-0.5 lg:mb-1">Maintenance</h4>
                                <p className="text-sm lg:text-base text-white font-medium capitalize">{hairstyleDetails.user_preferences.maintenance}</p>
                              </div>
                            )}
                            {hairstyleDetails.user_preferences.lifestyle && (
                              <div className="bg-white/5 rounded-lg p-2 lg:p-2.5 border border-white/10">
                                <h4 className="text-xs font-semibold text-purple-400 mb-0.5 lg:mb-1">Lifestyle</h4>
                                <p className="text-sm lg:text-base text-white font-medium capitalize">{hairstyleDetails.user_preferences.lifestyle}</p>
                              </div>
                            )}
                            {hairstyleDetails.user_preferences.gender && (
                              <div className="bg-white/5 rounded-lg p-2 lg:p-2.5 border border-white/10">
                                <h4 className="text-xs font-semibold text-purple-400 mb-0.5 lg:mb-1">Gender</h4>
                                <p className="text-sm lg:text-base text-white font-medium capitalize">{hairstyleDetails.user_preferences.gender}</p>
                              </div>
                            )}
                            {hairstyleDetails.user_preferences.occasions && hairstyleDetails.user_preferences.occasions.length > 0 && (
                              <div className="bg-white/5 rounded-lg p-2 lg:p-2.5 border border-white/10 col-span-2">
                                <h4 className="text-xs font-semibold text-purple-400 mb-0.5 lg:mb-1">Occasions</h4>
                                <p className="text-sm lg:text-base text-white font-medium capitalize">{hairstyleDetails.user_preferences.occasions.join(', ')}</p>
                              </div>
                            )}
                          </div>
                        </div>
                      ) : null}

                      {/* Preference Match - How it fits your preferences */}
                      {hairstyleDetails.preference_match && hairstyleDetails.preference_match.length > 0 && (
                        <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-2 sm:p-3 lg:p-4">
                          <h3 className="text-sm sm:text-base lg:text-lg font-semibold text-blue-400 mb-2 sm:mb-3 lg:mb-4">✓ How It Fits Your Preferences</h3>
                          <ul className="space-y-2 sm:space-y-2.5 lg:space-y-3">
                            {hairstyleDetails.preference_match.map((match, idx) => (
                              <li key={idx} className="flex items-start">
                                <span className="text-green-400 mr-2 sm:mr-2.5 lg:mr-3 mt-0.5 flex-shrink-0">•</span>
                                <span className="text-xs sm:text-sm lg:text-base text-gray-300 leading-relaxed text-justify">{match}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                    {/* Recommended Products */}
                    {hairstyleDetails.recommended_products && hairstyleDetails.recommended_products.length > 0 && (
                      <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-2 sm:p-3 lg:p-4">
                        <h3 className="text-sm sm:text-base lg:text-lg font-semibold text-blue-400 mb-2 sm:mb-3 lg:mb-4">🛍️ Recommended Products</h3>
                        <ul className="space-y-2 sm:space-y-2.5 lg:space-y-3">
                          {hairstyleDetails.recommended_products.map((product, idx) => (
                            <li key={idx} className="flex items-start">
                              <span className="text-orange-400 mr-2 sm:mr-2.5 lg:mr-3 flex-shrink-0 font-medium">{idx + 1}.</span>
                              <span className="text-xs sm:text-sm lg:text-base text-gray-300 leading-relaxed text-justify">{product}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Maintenance Guide */}
                    {hairstyleDetails.maintenance_guide && hairstyleDetails.maintenance_guide.length > 0 && (
                      <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-2 sm:p-3 lg:p-4">
                        <h3 className="text-sm sm:text-base lg:text-lg font-semibold text-blue-400 mb-2 sm:mb-3 lg:mb-4">🔧 Maintenance Guide</h3>
                        <ol className="space-y-2 sm:space-y-2.5 lg:space-y-3">
                          {hairstyleDetails.maintenance_guide.map((step, idx) => (
                            <li key={idx} className="flex items-start">
                              <span className="text-blue-400 font-medium mr-2 sm:mr-2.5 lg:mr-3 flex-shrink-0">{idx + 1}.</span>
                              <span className="text-xs sm:text-sm lg:text-base text-gray-300 leading-relaxed text-justify">{step}</span>
                            </li>
                          ))}
                        </ol>
                      </div>
                    )}

                    {/* Styling Tips */}
                    {hairstyleDetails.styling_tips && hairstyleDetails.styling_tips.length > 0 && (
                      <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-2 sm:p-3 lg:p-4">
                        <h3 className="text-sm sm:text-base lg:text-lg font-semibold text-blue-400 mb-2 sm:mb-3 lg:mb-4">💡 Pro Styling Tips</h3>
                        <ul className="space-y-2 sm:space-y-2.5 lg:space-y-3">
                          {hairstyleDetails.styling_tips.map((tip, idx) => (
                            <li key={idx} className="flex items-start">
                              <span className="text-pink-400 mr-2 sm:mr-2.5 lg:mr-3 flex-shrink-0">→</span>
                              <span className="text-xs sm:text-sm lg:text-base text-gray-300 leading-relaxed text-justify">{tip}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                  </div>
                </div>

                {/* Navigation buttons - Fixed at bottom */}
                <div className="flex flex-col sm:flex-row items-center justify-between gap-3 p-4 sm:p-6 border-t border-white/10 flex-shrink-0">
                  <button
                    onClick={handlePrevHairstyle}
                    className="flex items-center gap-2 bg-white/10 hover:bg-white/20 text-white px-4 sm:px-6 py-2.5 sm:py-3 rounded-lg transition-all duration-300 backdrop-blur-sm border border-white/20 hover:border-white/40 w-full sm:w-auto justify-center text-sm sm:text-base"
                  >
                    <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                    Previous Style
                  </button>

                  <div className="text-center order-first sm:order-none">
                    <p className="text-gray-300 text-xs sm:text-sm font-medium">
                      {currentHairstyleIndex + 1} of {recommendations.recommendations.length}
                    </p>
                  </div>

                  <button
                    onClick={handleNextHairstyle}
                    className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-4 sm:px-6 py-2.5 sm:py-3 rounded-lg transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-purple-500/25 border border-blue-500/30 w-full sm:w-auto justify-center text-sm sm:text-base"
                  >
                    Next Style
                    <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      )}

      {/* Image Modal - Full screen view */}
      {showImageModal && hairstyleDetails && (
        <div 
          className="fixed inset-0 bg-black/95 backdrop-blur-sm z-[60] flex items-center justify-center p-4"
          onClick={() => setShowImageModal(false)}
        >
          <div className="relative max-w-5xl w-full">
            {/* Close button */}
            <button
              onClick={() => setShowImageModal(false)}
              className="absolute -top-12 right-0 text-white hover:text-gray-300 transition-colors duration-300 flex items-center gap-2"
            >
              <span className="text-sm">Press ESC or click outside to close</span>
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            
            {/* Image container */}
            <div 
              className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-4 border border-white/20 shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 className="text-xl font-semibold text-white mb-4 text-center">
                {hairstyleDetails.overlay_url ? 
                  `Your Look with ${recommendations.recommendations[currentHairstyleIndex].name}` : 
                  'Style Reference'
                }
              </h3>
              <div className="relative">
                <img
                  src={hairstyleDetails.overlay_url || hairstyleDetails.hairstyle?.image_url}
                  alt="Full size hairstyle"
                  className="w-full h-auto rounded-lg shadow-2xl"
                  style={{ maxHeight: '80vh', objectFit: 'contain' }}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Results;
