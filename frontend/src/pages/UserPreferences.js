import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import APIService from '../services/api';
import AuthService from '../services/AuthService';
import Navbar from '../components/Navbar';

const UserPreferences = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { imageFile, previewUrl, uploadResponse } = location.state || {};

  // Step-by-step wizard state
  const [currentStep, setCurrentStep] = useState(1);
  const totalSteps = 6; // hair_type, hair_length, lifestyle, maintenance, occasions, compatibility

  const [preferences, setPreferences] = useState({
    hair_type: '',
    hair_length: '',
    lifestyle: '',
    maintenance: '',
    occasions: [],
    check_compatibility: false,
    target_hairstyle: '',
    custom_hairstyle: '',
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

  // Step navigation functions
  const nextStep = () => {
    if (currentStep < totalSteps) {
      setCurrentStep(currentStep + 1);
    } else {
      handleSubmit();
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const goToStep = (step) => {
    setCurrentStep(step);
  };

  // Check if current step is valid
  const isCurrentStepValid = () => {
    switch (currentStep) {
      case 1: return preferences.hair_type !== '';
      case 2: return preferences.hair_length !== '';
      case 3: return preferences.lifestyle !== '';
      case 4: return preferences.maintenance !== '';
      case 5: return preferences.occasions.length > 0;
      case 6: 
        if (!preferences.check_compatibility) return true;
        return preferences.target_hairstyle || preferences.custom_hairstyle.trim();
      default: return false;
    }
  };

  const isFormValid = () => {
    const basicFieldsValid = preferences.hair_type && 
           preferences.hair_length && 
           preferences.lifestyle && 
           preferences.maintenance && 
           preferences.occasions.length > 0;
    
    // If compatibility check is enabled, require either dropdown selection or custom input
    if (preferences.check_compatibility) {
      const compatibilityFieldsValid = preferences.target_hairstyle || preferences.custom_hairstyle.trim();
      return basicFieldsValid && compatibilityFieldsValid;
    }
    
    return basicFieldsValid;
  };

  // Step definitions
  const steps = [
    { number: 1, title: 'Hair Type', description: 'What\'s your current hair type?' },
    { number: 2, title: 'Hair Length', description: 'What length are you considering?' },
    { number: 3, title: 'Lifestyle', description: 'What\'s your lifestyle like?' },
    { number: 4, title: 'Maintenance', description: 'How much maintenance do you prefer?' },
    { number: 5, title: 'Occasions', description: 'What occasions do you style for?' },
    { number: 6, title: 'Compatibility', description: 'Check specific hairstyle compatibility' }
  ];

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
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Tell us about your preferences
            </h1>
            {previewUrl && (
              <div className="flex justify-center mb-6">
                <img
                  src={previewUrl}
                  alt="Your photo"
                  className="h-32 w-32 object-cover rounded-full border-4 border-purple-400/30 shadow-2xl"
                />
              </div>
            )}
          </div>

          {/* Breadcrumb Navigation */}
          <div className="mb-8">
            <div className="flex items-center justify-center space-x-2 mb-4">
              {steps.map((step) => (
                <React.Fragment key={step.number}>
                  <div
                    onClick={() => goToStep(step.number)}
                    className={`flex items-center justify-center w-10 h-10 rounded-full text-sm font-bold cursor-pointer transition-all duration-300 ${
                      currentStep === step.number
                        ? 'bg-purple-500 text-white scale-110 shadow-lg shadow-purple-500/30'
                        : currentStep > step.number
                        ? 'bg-green-500 text-white hover:scale-105'
                        : 'bg-gray-600 text-gray-300 hover:bg-gray-500'
                    }`}
                  >
                    {currentStep > step.number ? (
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    ) : (
                      step.number
                    )}
                  </div>
                  {step.number < totalSteps && (
                    <div className={`h-1 w-8 transition-colors duration-300 ${
                      currentStep > step.number ? 'bg-green-500' : 'bg-gray-600'
                    }`}></div>
                  )}
                </React.Fragment>
              ))}
            </div>
            
            {/* Step Title and Description */}
            <div className="text-center">
              <h2 className="text-2xl font-bold text-white mb-2">
                Step {currentStep}: {steps[currentStep - 1].title}
              </h2>
              <p className="text-lg text-gray-300">
                {steps[currentStep - 1].description}
              </p>
            </div>
          </div>

          {/* Step Content */}
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-8 md:p-12 shadow-xl mb-8">
            {/* Step 1: Hair Type */}
            {currentStep === 1 && (
              <div className="space-y-8">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {['straight', 'wavy', 'curly', 'coily'].map((type) => (
                    <button
                      key={type}
                      onClick={() => handlePreferenceChange('hair_type', type)}
                      className={`p-8 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                        preferences.hair_type === type
                          ? 'border-purple-400 bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/25'
                          : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                      }`}
                    >
                      <div className="text-4xl mb-4">
                        {type === 'straight' && '📏'}
                        {type === 'wavy' && '🌊'}
                        {type === 'curly' && '🌀'}
                        {type === 'coily' && '🔗'}
                      </div>
                      <div className="font-medium text-lg">{type.charAt(0).toUpperCase() + type.slice(1)}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Step 2: Hair Length */}
            {currentStep === 2 && (
              <div className="space-y-8">
                <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                  {['pixie', 'short', 'medium', 'long', 'extra-long'].map((length) => (
                    <button
                      key={length}
                      onClick={() => handlePreferenceChange('hair_length', length)}
                      className={`p-8 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                        preferences.hair_length === length
                          ? 'border-purple-400 bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/25'
                          : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                      }`}
                    >
                      <div className="text-4xl mb-4">
                        {length === 'pixie' && '✂️'}
                        {length === 'short' && '💇‍♀️'}
                        {length === 'medium' && '👩‍🦰'}
                        {length === 'long' && '🧚‍♀️'}
                        {length === 'extra-long' && '👸'}
                      </div>
                      <div className="font-medium text-lg">{length.charAt(0).toUpperCase() + length.slice(1).replace('-', ' ')}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Step 3: Lifestyle */}
            {currentStep === 3 && (
              <div className="space-y-8">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {[
                    { value: 'active', emoji: '🏃‍♀️', description: 'Always on the go, love sports and outdoor activities' },
                    { value: 'moderate', emoji: '🚶‍♀️', description: 'Balanced lifestyle with some activities and relaxation' },
                    { value: 'relaxed', emoji: '🧘‍♀️', description: 'Prefer calm, low-key activities and plenty of downtime' }
                  ].map((lifestyle) => (
                    <button
                      key={lifestyle.value}
                      onClick={() => handlePreferenceChange('lifestyle', lifestyle.value)}
                      className={`p-8 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 text-left ${
                        preferences.lifestyle === lifestyle.value
                          ? 'border-purple-400 bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/25'
                          : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                      }`}
                    >
                      <div className="text-4xl mb-4">{lifestyle.emoji}</div>
                      <div className="font-medium text-xl mb-2">{lifestyle.value.charAt(0).toUpperCase() + lifestyle.value.slice(1)}</div>
                      <div className="text-sm opacity-80">{lifestyle.description}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Step 4: Maintenance */}
            {currentStep === 4 && (
              <div className="space-y-8">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {[
                    { value: 'low', emoji: '⚡', description: 'Minimal styling time, wash and go styles' },
                    { value: 'medium', emoji: '🎯', description: 'Some styling effort, occasional salon visits' },
                    { value: 'high', emoji: '💎', description: 'Love detailed styling, frequent salon appointments' }
                  ].map((maintenance) => (
                    <button
                      key={maintenance.value}
                      onClick={() => handlePreferenceChange('maintenance', maintenance.value)}
                      className={`p-8 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 text-left ${
                        preferences.maintenance === maintenance.value
                          ? 'border-purple-400 bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/25'
                          : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                      }`}
                    >
                      <div className="text-4xl mb-4">{maintenance.emoji}</div>
                      <div className="font-medium text-xl mb-2">{maintenance.value.charAt(0).toUpperCase() + maintenance.value.slice(1)} Maintenance</div>
                      <div className="text-sm opacity-80">{maintenance.description}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Step 5: Occasions */}
            {currentStep === 5 && (
              <div className="space-y-8">
                <p className="text-center text-gray-300 text-lg mb-6">Select all that apply</p>
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
                      <div className="font-medium text-lg">{occasion.label}</div>
                      {preferences.occasions.includes(occasion.value) && (
                        <div className="mt-2">
                          <svg className="w-5 h-5 text-purple-400 mx-auto" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        </div>
                      )}
                    </button>
                  ))}
                </div>
                {preferences.occasions.length > 0 && (
                  <div className="bg-purple-900/20 border border-purple-500/30 rounded-xl p-4 mt-6">
                    <p className="text-purple-300 text-center">
                      Selected {preferences.occasions.length} occasion{preferences.occasions.length !== 1 ? 's' : ''}
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Step 6: Compatibility Check */}
            {currentStep === 6 && (
              <div className="space-y-8">
                {/* Toggle for compatibility check */}
                <div className="text-center">
                  <button
                    onClick={() => {
                      const newValue = !preferences.check_compatibility;
                      handlePreferenceChange('check_compatibility', newValue);
                      if (!newValue) {
                        handlePreferenceChange('target_hairstyle', '');
                        handlePreferenceChange('custom_hairstyle', '');
                      }
                    }}
                    className={`inline-flex items-center px-8 py-4 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                      preferences.check_compatibility
                        ? 'border-purple-400 bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/25'
                        : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                    }`}
                  >
                    <div className={`w-6 h-6 rounded border-2 mr-4 flex items-center justify-center transition-all duration-300 ${
                      preferences.check_compatibility
                        ? 'border-purple-400 bg-purple-500/20'
                        : 'border-gray-600'
                    }`}>
                      {preferences.check_compatibility && (
                        <svg className="w-4 h-4 text-purple-300" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      )}
                    </div>
                    <div className="text-left">
                      <div className="font-bold text-xl">Check Hairstyle Compatibility</div>
                      <div className="text-sm opacity-80">Want to test if a specific hairstyle suits your face shape?</div>
                    </div>
                  </button>
                </div>

                {/* Compatibility options */}
                {preferences.check_compatibility && (
                  <div className="space-y-6 max-w-2xl mx-auto">
                    {/* Dropdown for common hairstyles */}
                    <div>
                      <label className="block text-lg font-semibold text-white mb-3 text-center">
                        Choose from popular hairstyles:
                      </label>
                      <select
                        value={preferences.target_hairstyle}
                        onChange={(e) => {
                          handlePreferenceChange('target_hairstyle', e.target.value);
                          if (e.target.value) {
                            handlePreferenceChange('custom_hairstyle', '');
                          }
                        }}
                        className="w-full p-4 rounded-xl bg-gray-700/50 border border-gray-600 text-white focus:border-purple-400 focus:outline-none transition-colors duration-300 text-lg"
                      >
                        <option value="">Select a hairstyle...</option>
                        <option value="bob">Bob Cut</option>
                        <option value="pixie">Pixie Cut</option>
                        <option value="lob">Long Bob (Lob)</option>
                        <option value="layers">Layered Cut</option>
                        <option value="bangs">Straight Bangs</option>
                        <option value="curtain_bangs">Curtain Bangs</option>
                        <option value="shag">Shag Cut</option>
                        <option value="wolf_cut">Wolf Cut</option>
                        <option value="undercut">Undercut</option>
                        <option value="buzz_cut">Buzz Cut</option>
                        <option value="beach_waves">Beach Waves</option>
                        <option value="straight_long">Long Straight Hair</option>
                        <option value="curly_short">Short Curly Hair</option>
                        <option value="curly_long">Long Curly Hair</option>
                        <option value="asymmetrical">Asymmetrical Cut</option>
                        <option value="side_part">Side Part</option>
                        <option value="middle_part">Middle Part</option>
                        <option value="braids">Braided Styles</option>
                        <option value="updo">Updo Styles</option>
                        <option value="ponytail">Ponytail</option>
                      </select>
                    </div>

                    {/* Divider */}
                    <div className="flex items-center">
                      <div className="flex-1 border-t border-gray-600"></div>
                      <span className="px-4 text-gray-400 text-sm">OR</span>
                      <div className="flex-1 border-t border-gray-600"></div>
                    </div>

                    {/* Custom hairstyle input */}
                    <div>
                      <label className="block text-lg font-semibold text-white mb-3 text-center">
                        Describe your specific hairstyle:
                      </label>
                      <textarea
                        value={preferences.custom_hairstyle}
                        onChange={(e) => {
                          handlePreferenceChange('custom_hairstyle', e.target.value);
                          if (e.target.value.trim()) {
                            handlePreferenceChange('target_hairstyle', '');
                          }
                        }}
                        placeholder="e.g., Shoulder-length hair with side-swept bangs and subtle layers..."
                        rows="4"
                        className="w-full p-4 rounded-xl bg-gray-700/50 border border-gray-600 text-white placeholder-gray-400 focus:border-purple-400 focus:outline-none transition-colors duration-300 resize-none text-lg"
                      />
                      <p className="mt-2 text-sm text-gray-400 text-center">
                        Be as specific as possible about the cut, length, layers, bangs, etc.
                      </p>
                    </div>

                    {/* Selected hairstyle display */}
                    {(preferences.target_hairstyle || preferences.custom_hairstyle.trim()) && (
                      <div className="bg-purple-900/20 border border-purple-500/30 rounded-xl p-6">
                        <div className="flex items-center justify-center">
                          <svg className="w-6 h-6 text-purple-400 mr-3" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                          </svg>
                          <div className="text-center">
                            <div className="text-purple-300 font-medium mb-1">
                              Compatibility check selected for:
                            </div>
                            <div className="text-white text-lg font-bold">
                              {preferences.target_hairstyle 
                                ? preferences.target_hairstyle.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())
                                : preferences.custom_hairstyle.trim()
                              }
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Skip option */}
                {!preferences.check_compatibility && (
                  <div className="text-center">
                    <p className="text-gray-400 text-lg">
                      You can skip this step and proceed with general recommendations.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Navigation Buttons */}
          <div className="flex justify-between items-center">
            <button
              onClick={prevStep}
              disabled={currentStep === 1}
              className={`px-8 py-3 rounded-xl font-medium transition-all duration-300 ${
                currentStep === 1
                  ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  : 'bg-gray-700 hover:bg-gray-600 text-white'
              }`}
            >
              ← Previous
            </button>

            <div className="text-center">
              <div className="text-white text-sm">
                Step {currentStep} of {totalSteps}
              </div>
              <div className="text-gray-400 text-xs">
                {Math.round((currentStep / totalSteps) * 100)}% Complete
              </div>
            </div>

            <button
              onClick={nextStep}
              disabled={!isCurrentStepValid() || isSubmitting}
              className={`px-8 py-3 rounded-xl font-medium transition-all duration-300 transform ${
                !isCurrentStepValid() || isSubmitting
                  ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  : currentStep === totalSteps
                  ? 'bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white hover:scale-105 shadow-lg'
                  : 'bg-purple-600 hover:bg-purple-700 text-white hover:scale-105'
              }`}
            >
              {isSubmitting ? 'Getting Recommendations...' : currentStep === totalSteps ? 'Get My Recommendations' : 'Next →'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserPreferences;