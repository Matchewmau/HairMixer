import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import APIService from '../services/api';

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

  useEffect(() => {
    if (!uploadResponse) {
      navigate('/upload');
      return;
    }
    
    loadFilterOptions();
  }, [uploadResponse, navigate]);

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
      
      // Ensure hair_length uses underscore not dash
      const cleanedPreferences = {
        ...preferences,
        hair_length: preferences.hair_length === 'extra-long' ? 'extra_long' : preferences.hair_length
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
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-purple-600"></div>
      </div>
    );
  }

  if (!uploadResponse) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            No image found
          </h2>
          <button
            onClick={() => navigate('/upload')}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700"
          >
            Upload Photo
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header with image preview */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-4">
            Tell us about your preferences
          </h1>
          {previewUrl && (
            <div className="flex justify-center mb-4">
              <img
                src={previewUrl}
                alt="Your photo"
                className="h-32 w-32 object-cover rounded-full border-4 border-white shadow-lg"
              />
            </div>
          )}
          <p className="text-lg text-gray-600">
            Help us recommend the perfect hairstyles for you
          </p>
        </div>

        {/* Preferences Form */}
        <div className="bg-white rounded-lg shadow-lg p-8 space-y-8">
          {/* Hair Type */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              What's your current hair type?
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {['straight', 'wavy', 'curly', 'coily'].map((type) => (
                <button
                  key={type}
                  onClick={() => handlePreferenceChange('hair_type', type)}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    preferences.hair_type === type
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="font-medium">{type.charAt(0).toUpperCase() + type.slice(1)}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Hair Length */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              What length are you considering?
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {['pixie', 'short', 'medium', 'long', 'extra-long'].map((length) => (
                <button
                  key={length}
                  onClick={() => handlePreferenceChange('hair_length', length)}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    preferences.hair_length === length
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="font-medium">{length.charAt(0).toUpperCase() + length.slice(1)}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Lifestyle */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              What's your lifestyle like?
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {['active', 'moderate', 'relaxed'].map((lifestyle) => (
                <button
                  key={lifestyle}
                  onClick={() => handlePreferenceChange('lifestyle', lifestyle)}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    preferences.lifestyle === lifestyle
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="font-medium">{lifestyle.charAt(0).toUpperCase() + lifestyle.slice(1)}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Maintenance */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              How much maintenance do you prefer?
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {['low', 'medium', 'high'].map((maintenance) => (
                <button
                  key={maintenance}
                  onClick={() => handlePreferenceChange('maintenance', maintenance)}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    preferences.maintenance === maintenance
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="font-medium">{maintenance.charAt(0).toUpperCase() + maintenance.slice(1)} Maintenance</div>
                </button>
              ))}
            </div>
          </div>

          {/* Occasions */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              What occasions do you style your hair for? (Select all that apply)
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {occasions.map((occasion) => (
                <button
                  key={occasion.value}
                  onClick={() => handlePreferenceChange('occasions', occasion.value)}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    preferences.occasions.includes(occasion.value)
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="font-medium">{occasion.label}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Submit Button */}
          <div className="text-center pt-8">
            <button
              onClick={handleSubmit}
              disabled={!isFormValid() || isSubmitting}
              className={`px-8 py-3 rounded-lg font-medium text-lg transition-all ${
                !isFormValid() || isSubmitting
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700 hover:scale-105'
              }`}
            >
              {isSubmitting ? 'Getting Recommendations...' : 'Get My Recommendations'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserPreferences;