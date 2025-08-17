import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

const UserPreferences = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { imageFile, previewUrl } = location.state || {};

  const [preferences, setPreferences] = useState({
    hairLength: '',
    lifestyle: '',
    maintenance: '',
    occasions: [],
    faceShape: '', // This will be populated by AI analysis
    currentHairType: '',
    colorPreference: '',
    budget: ''
  });

  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleInputChange = (field, value) => {
    setPreferences(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleOccasionToggle = (occasion) => {
    setPreferences(prev => ({
      ...prev,
      occasions: prev.occasions.includes(occasion)
        ? prev.occasions.filter(o => o !== occasion)
        : [...prev.occasions, occasion]
    }));
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    
    try {
      // TODO: Send preferences and image to backend for final recommendation
      console.log('User preferences:', preferences);
      console.log('Image file:', imageFile);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Navigate to results page (we'll create this later)
      navigate('/results', { 
        state: { 
          preferences, 
          imageFile, 
          previewUrl 
        }
      });
    } catch (error) {
      console.error('Error submitting preferences:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const isFormValid = () => {
    return preferences.hairLength && 
           preferences.lifestyle && 
           preferences.maintenance && 
           preferences.occasions.length > 0 &&
           preferences.currentHairType;
  };

  if (!imageFile) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">No Image Found</h2>
          <p className="text-gray-600 mb-6">Please upload an image first.</p>
          <button
            onClick={() => navigate('/upload')}
            className="bg-purple-600 hover:bg-purple-700 text-white font-medium py-2 px-6 rounded-lg"
          >
            Upload Photo
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <button
              onClick={() => navigate('/upload')}
              className="flex items-center text-purple-600 hover:text-purple-700 font-medium"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              Back to Photo Upload
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
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Tell Us Your Preferences</h2>
          <p className="text-lg text-gray-600">
            Help us find the perfect hairstyle by sharing your preferences and lifestyle
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Image Preview */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-lg p-6 sticky top-8">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Your Photo</h3>
              {previewUrl && (
                <img
                  src={previewUrl}
                  alt="Uploaded"
                  className="w-full rounded-lg shadow-md object-cover"
                />
              )}
              <div className="mt-4 p-3 bg-green-50 rounded-lg">
                <p className="text-sm text-green-800">
                  ✅ Photo uploaded successfully! AI analysis in progress...
                </p>
              </div>
            </div>
          </div>

          {/* Preferences Form */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-lg p-8">
              <div className="space-y-8">
                {/* Hair Length Preference */}
                <div>
                  <label className="block text-lg font-semibold text-gray-900 mb-4">
                    Preferred Hair Length
                  </label>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {['Short', 'Medium', 'Long', 'No Preference'].map((length) => (
                      <button
                        key={length}
                        onClick={() => handleInputChange('hairLength', length)}
                        className={`p-3 rounded-lg border-2 text-center font-medium transition-all ${
                          preferences.hairLength === length
                            ? 'border-purple-600 bg-purple-50 text-purple-700'
                            : 'border-gray-200 hover:border-purple-300 text-gray-700'
                        }`}
                      >
                        {length}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Lifestyle */}
                <div>
                  <label className="block text-lg font-semibold text-gray-900 mb-4">
                    Lifestyle
                  </label>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    {['Active/Sporty', 'Professional', 'Creative/Artistic'].map((lifestyle) => (
                      <button
                        key={lifestyle}
                        onClick={() => handleInputChange('lifestyle', lifestyle)}
                        className={`p-4 rounded-lg border-2 text-center font-medium transition-all ${
                          preferences.lifestyle === lifestyle
                            ? 'border-purple-600 bg-purple-50 text-purple-700'
                            : 'border-gray-200 hover:border-purple-300 text-gray-700'
                        }`}
                      >
                        {lifestyle}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Maintenance Level */}
                <div>
                  <label className="block text-lg font-semibold text-gray-900 mb-4">
                    Maintenance Level
                  </label>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    {['Low (wash & go)', 'Medium (some styling)', 'High (daily styling)'].map((maintenance) => (
                      <button
                        key={maintenance}
                        onClick={() => handleInputChange('maintenance', maintenance)}
                        className={`p-4 rounded-lg border-2 text-center font-medium transition-all ${
                          preferences.maintenance === maintenance
                            ? 'border-purple-600 bg-purple-50 text-purple-700'
                            : 'border-gray-200 hover:border-purple-300 text-gray-700'
                        }`}
                      >
                        {maintenance}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Occasions */}
                <div>
                  <label className="block text-lg font-semibold text-gray-900 mb-4">
                    Occasions (Select all that apply)
                  </label>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {['Work', 'Casual', 'Formal Events', 'Date Night', 'Exercise', 'Travel'].map((occasion) => (
                      <button
                        key={occasion}
                        onClick={() => handleOccasionToggle(occasion)}
                        className={`p-3 rounded-lg border-2 text-center font-medium transition-all ${
                          preferences.occasions.includes(occasion)
                            ? 'border-purple-600 bg-purple-50 text-purple-700'
                            : 'border-gray-200 hover:border-purple-300 text-gray-700'
                        }`}
                      >
                        {occasion}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Current Hair Type */}
                <div>
                  <label className="block text-lg font-semibold text-gray-900 mb-4">
                    Current Hair Type
                  </label>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {['Straight', 'Wavy', 'Curly', 'Coily'].map((hairType) => (
                      <button
                        key={hairType}
                        onClick={() => handleInputChange('currentHairType', hairType)}
                        className={`p-3 rounded-lg border-2 text-center font-medium transition-all ${
                          preferences.currentHairType === hairType
                            ? 'border-purple-600 bg-purple-50 text-purple-700'
                            : 'border-gray-200 hover:border-purple-300 text-gray-700'
                        }`}
                      >
                        {hairType}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Submit Button */}
                <div className="pt-6 border-t">
                  <button
                    onClick={handleSubmit}
                    disabled={!isFormValid() || isSubmitting}
                    className={`w-full py-4 px-6 rounded-lg font-bold text-lg transition-all ${
                      isFormValid() && !isSubmitting
                        ? 'bg-purple-600 hover:bg-purple-700 text-white'
                        : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    }`}
                  >
                    {isSubmitting ? (
                      <div className="flex items-center justify-center">
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-gray-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Getting Your Recommendations...
                      </div>
                    ) : (
                      'Get My Hairstyle Recommendations'
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserPreferences;