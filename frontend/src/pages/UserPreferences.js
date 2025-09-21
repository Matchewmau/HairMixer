import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import APIService from '../services/api';
import AuthService from '../services/AuthService';
import Navbar from '../components/Navbar';

const UserPreferences = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { imageFile, previewUrl, uploadResponse } = location.state || {};

  const [preferences, setPreferences] = useState({
    hair_type: '',
    hair_length: '',
    lifestyle: '',
    maintenance: '',
    occasions: [],
    // hair_texture: '',
    // face_shape_preference: '',
  });

  const [occasions, setOccasions] = useState([]);
  const [faceShapes, setFaceShapes] = useState([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [user, setUser] = useState(null);

  useEffect(() => {
    if (!uploadResponse) {
      navigate('/upload');
      return;
    }
    
    loadFilterOptions();
    checkAuth();
  }, [uploadResponse, navigate]);

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

  const loadFilterOptions = async () => {
    try {
      const [occasionsResponse, faceShapesResponse] = await Promise.all([
        APIService.getOccasions(),
        APIService.getFaceShapes()
      ]);
      
      setOccasions(occasionsResponse.occasions || []);
      setFaceShapes(faceShapesResponse.face_shapes || []);
    } catch (error) {
      console.error('Error loading filter options:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async () => {
    if (!isFormValid()) return;

    setIsSubmitting(true);
    
    try {
      console.log('Submitting preferences:', preferences);
      
      // Ensure hair_length uses underscore not dash and map lifestyle UI -> model
      const cleanedPreferences = {
        ...preferences,
        hair_length: preferences.hair_length === 'extra-long' ? 'extra_long' : preferences.hair_length,
        lifestyle: preferences.lifestyle === 'moderate' || preferences.lifestyle === 'relaxed' ? 'casual' : preferences.lifestyle,
      };
      
      console.log('Cleaned preferences:', cleanedPreferences);
      
      // Save preferences
      const preferencesResponse = await APIService.savePreferences(cleanedPreferences);
      console.log('Preferences response:', preferencesResponse);
      
      if (!preferencesResponse.success) {
        throw new Error(preferencesResponse.error || 'Failed to save preferences');
      }
      
      // Get recommendations
      console.log('Getting recommendations with:', {
        image_id: uploadResponse.image_id,
        preference_id: preferencesResponse.preference_id
      });
      
      const recommendationsResponse = await APIService.getRecommendations(
        uploadResponse.image_id,
        preferencesResponse.preference_id
      );
      
      console.log('Recommendations:', recommendationsResponse);
      
      // Navigate to results
      navigate('/results', { 
        state: { 
          preferences,
          imageFile,
          previewUrl,
          uploadResponse,
          recommendations: recommendationsResponse
        }
      });
      
    } catch (error) {
      console.error('Full error object:', error);
      console.error('Error submitting preferences:', error.message);
      
      let errorMessage = `Failed to get recommendations: ${error.message}`;
      alert(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handlePreferenceChange = (key, value) => {
    if (key === 'occasions') {
      const newOccasions = preferences.occasions.includes(value)
        ? preferences.occasions.filter(o => o !== value)
        : [...preferences.occasions, value];
      
      setPreferences(prev => ({
        ...prev,
        occasions: newOccasions
      }));
    } else {
      setPreferences(prev => ({
        ...prev,
        [key]: value
      }));
    }
  };

  const isFormValid = () => {
    return preferences.hair_type && 
           preferences.hair_length && 
           preferences.lifestyle && 
           preferences.maintenance && 
           preferences.occasions.length > 0;
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-800 to-blue-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-purple-400"></div>
      </div>
    );
  }

  if (!uploadResponse) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-800 to-blue-900 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-semibold text-white mb-6">
            No image found
          </h2>
          <button
            onClick={() => navigate('/upload')}
            className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white px-8 py-3 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg"
          >
            Upload Photo
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
        backPath="/upload"
      />
      
      <div className="pt-20 pb-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          {/* Header with image preview */}
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Tell us about your preferences
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
              Help us recommend the perfect hairstyles for you
            </p>
          </div>

          {/* Preferences Form */}
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-8 md:p-12 space-y-10 shadow-xl">
          {/* Hair Type */}
          <div>
            <h3 className="text-2xl font-bold text-white mb-6">
              What's your current hair type?
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {['straight', 'wavy', 'curly', 'coily'].map((type) => (
                <button
                  key={type}
                  onClick={() => handlePreferenceChange('hair_type', type)}
                  className={`p-6 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                    preferences.hair_type === type
                      ? 'border-purple-400 bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/25'
                      : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                  }`}
                >
                  <div className="font-medium">{type.charAt(0).toUpperCase() + type.slice(1)}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Hair Length */}
          <div>
            <h3 className="text-2xl font-bold text-white mb-6">
              What length are you considering?
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {['pixie', 'short', 'medium', 'long', 'extra-long'].map((length) => (
                <button
                  key={length}
                  onClick={() => handlePreferenceChange('hair_length', length)}
                  className={`p-6 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                    preferences.hair_length === length
                      ? 'border-purple-400 bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/25'
                      : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                  }`}
                >
                  <div className="font-medium">{length.charAt(0).toUpperCase() + length.slice(1).replace('-', ' ')}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Lifestyle */}
          <div>
            <h3 className="text-2xl font-bold text-white mb-6">
              What's your lifestyle like?
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {['active', 'moderate', 'relaxed'].map((lifestyle) => (
                <button
                  key={lifestyle}
                  onClick={() => handlePreferenceChange('lifestyle', lifestyle)}
                  className={`p-6 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                    preferences.lifestyle === lifestyle
                      ? 'border-purple-400 bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/25'
                      : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                  }`}
                >
                  <div className="font-medium">{lifestyle.charAt(0).toUpperCase() + lifestyle.slice(1)}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Maintenance */}
          <div>
            <h3 className="text-2xl font-bold text-white mb-6">
              How much maintenance do you prefer?
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {['low', 'medium', 'high'].map((maintenance) => (
                <button
                  key={maintenance}
                  onClick={() => handlePreferenceChange('maintenance', maintenance)}
                  className={`p-6 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                    preferences.maintenance === maintenance
                      ? 'border-purple-400 bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/25'
                      : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                  }`}
                >
                  <div className="font-medium">{maintenance.charAt(0).toUpperCase() + maintenance.slice(1)} Maintenance</div>
                </button>
              ))}
            </div>
          </div>

          {/* Occasions */}
          <div>
            <h3 className="text-2xl font-bold text-white mb-6">
              What occasions do you style your hair for? (Select all that apply)
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {occasions.map((occasion) => (
                <button
                  key={occasion.value}
                  onClick={() => handlePreferenceChange('occasions', occasion.value)}
                  className={`p-6 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                    preferences.occasions.includes(occasion.value)
                      ? 'border-purple-400 bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/25'
                      : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                  }`}
                >
                  <div className="font-medium">{occasion.label}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Submit Button */}
          <div className="text-center pt-12">
            <button
              onClick={handleSubmit}
              disabled={!isFormValid() || isSubmitting}
              className={`px-12 py-4 rounded-xl font-bold text-xl transition-all duration-300 transform ${
                !isFormValid() || isSubmitting
                  ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white hover:scale-105 shadow-lg hover:shadow-purple-500/25'
              }`}
            >
              {isSubmitting ? 'Getting Recommendations...' : 'Get My Recommendations'}
            </button>
          </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserPreferences;