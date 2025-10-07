import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import APIService from '../services/api';
import AuthService from '../services/AuthService';
import Navbar from '../components/Navbar';

/**
 * UserPreferences Component
 * 
 * A comprehensive 11-step wizard for collecting detailed user hair preferences.
 * This component is part of the ML-powered hairstyle recommendation flow.
 * 
 * Flow:
 * 1. User uploads image → Face shape detected by ResNet50
 * 2. User completes 11-step preference wizard
 * 3. Preferences saved to backend
 * 4. ML model (hairstyle_family_model.pkl) generates top 10 recommendations
 * 5. User navigated to Results page with recommendations
 * 
 * Wizard Steps:
 * - Step 0: Gender & Face Shape Display (male, female, non-binary, prefer not to say) 
 * - Step 1: Hair Type (straight, wavy, curly, coily)
 * - Step 2: Hair Length (short, medium, long)
 * - Step 3: Volume (low, medium, high)
 * - Step 4: Hair Thickness (thin, medium, thick)
 * - Step 5: Hair Texture Detail (fine, normal, thick, smooth, coarse, silky, frizzy)
 * - Step 6: Lifestyle (active, moderate, relaxed, professional)
 * - Step 7: Maintenance (low, medium, high)
 * - Step 8: Styling Preference (natural, classic, elegant, trendy, edgy) + Bangs
 * - Step 9: Occasions (multiple selection)
 * - Step 10: Hair Condition (optional: healthy, dry ends, damaged, etc.)
 * 
 * State Management:
 * - preferences: All user preference data including gender and hair_condition
 * - currentStep: Current wizard step (0-10)
 * - uploadResponse: Contains detected face shape from ResNet50
 * 
 * Key Features:
 * - Gender selection for personalized recommendations (critical for model accuracy)
 * - Auto-populated face shape from image analysis
 * - Optional hair condition input for better recommendations
 * - Step-by-step validation (Step 10 is optional)
 * - Progress tracking
 * - Responsive design with emojis and visual feedback
 * - ML-based recommendation generation using Random Forest model
 * 
 * @component
 */
const UserPreferences = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { imageFile, previewUrl, uploadResponse } = location.state || {};

  // Step-by-step wizard state
  const [currentStep, setCurrentStep] = useState(0);
  const totalSteps = 11; // Gender (0) + Basic characteristics (1-9) + Hair condition (10)

  const [preferences, setPreferences] = useState({
    // Core characteristics
    hair_type: '',
    hair_length: '',
    lifestyle: '',
    maintenance: '',
    occasions: [],
    
    // New detailed characteristics
    volume: '',
    styling_maintenance: '',
    hair_texture_detail: '',
    styling_preference: '',
    hair_condition: '',
    hair_thickness: '',
    wants_bangs: false,
    hair_color: '',
    gender: '',
    
    // Hairstyle preferences
    hairstyle_family: '',
    hairstyle_name: '',
    
    // Face shape (auto-filled from ResNet50)
    faceshape: uploadResponse?.face_shape?.shape || '',
    
    // Legacy compatibility check
    check_compatibility: false,
    target_hairstyle: '',
    custom_hairstyle: '',
  });

  const [occasions, setOccasions] = useState([]);
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
      const occasionsResponse = await APIService.getOccasions();
      setOccasions(occasionsResponse.occasions || []);
    } catch (error) {
      console.error('Error loading filter options:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async () => {
    if (!isFormValid()) return;

    // Validate all required fields before submission
    const errors = validateAllFields();
    if (Object.keys(errors).length > 0) {
      const errorMessages = Object.values(errors).join('\n');
      alert(`Please fix the following errors:\n\n${errorMessages}`);
      return;
    }

    setIsSubmitting(true);
    
    try {
      console.log('Submitting preferences:', preferences);
      
      // Clean preferences to match dataset format
      const cleanedPreferences = {
        ...preferences,
        faceshape: uploadResponse?.face_shape?.shape || preferences.faceshape || '',
        // Default styling_maintenance to maintenance if not set
        styling_maintenance: preferences.styling_maintenance || preferences.maintenance,
        // Default hair_condition to 'none' (healthy) if not provided
        hair_condition: preferences.hair_condition || 'none',
      };
      
      // Final validation check on cleaned data
      const validMaintenance = ['low', 'medium', 'high'];
      if (!validMaintenance.includes(cleanedPreferences.maintenance)) {
        throw new Error(`Invalid maintenance level: ${cleanedPreferences.maintenance}. Must be one of: ${validMaintenance.join(', ')}`);
      }

      const validGenders = ['male', 'female', 'nb', 'other'];
      if (cleanedPreferences.gender && !validGenders.includes(cleanedPreferences.gender)) {
        throw new Error(`Invalid gender: ${cleanedPreferences.gender}. Must be one of: ${validGenders.join(', ')}`);
      }

      const validLifestyles = ['active', 'moderate', 'relaxed'];  // Dataset values
      if (!validLifestyles.includes(cleanedPreferences.lifestyle)) {
        throw new Error(`Invalid lifestyle: ${cleanedPreferences.lifestyle}. Must be one of: ${validLifestyles.join(', ')}`);
      }
      
      console.log('Cleaned preferences:', cleanedPreferences);
      
      // Save preferences
      const preferencesResponse = await APIService.savePreferences(cleanedPreferences);
      console.log('Preferences response:', preferencesResponse);
      
      if (!preferencesResponse.success) {
        throw new Error(preferencesResponse.error || 'Failed to save preferences');
      }
      
      // Get ML-based recommendations (top 10)
      console.log('Getting ML recommendations with preference_id:', preferencesResponse.preference_id);
      
      const mlRecommendationsResponse = await APIService.getMLRecommendations(
        preferencesResponse.preference_id
      );
      
      console.log('ML Recommendations:', mlRecommendationsResponse);
      
      // Navigate to results
      navigate('/results', { 
        state: { 
          preferences: cleanedPreferences,
          imageFile,
          previewUrl,
          uploadResponse,
          recommendations: mlRecommendationsResponse
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
    if (currentStep < totalSteps - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleSubmit();
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const goToStep = (step) => {
    setCurrentStep(step);
  };

  // Validation rules for each field (matching dataset exactly)
  const validationRules = {
    hair_type: {
      options: ['straight', 'wavy', 'curly', 'coily'],
      label: 'Hair Type',
      required: true
    },
    hair_length: {
      options: ['short', 'medium', 'long'],  // Dataset only has these 3
      label: 'Hair Length',
      required: true
    },
    volume: {
      options: ['low', 'medium', 'high'],  // Changed from flat/light
      label: 'Volume',
      required: true
    },
    hair_thickness: {
      options: ['thin', 'medium', 'thick'],  // Dataset only has these 3
      label: 'Hair Thickness',
      required: true
    },
    hair_texture_detail: {
      options: ['fine', 'normal', 'thick'],  // Dataset only has these 3
      label: 'Hair Texture',
      required: true
    },
    lifestyle: {
      options: ['active', 'moderate', 'relaxed'],  // Dataset values
      label: 'Lifestyle',
      required: true
    },
    maintenance: {
      options: ['low', 'medium', 'high'],
      label: 'Maintenance',
      required: true
    },
    styling_preference: {
      options: ['natural', 'classic', 'elegant', 'trendy', 'edgy'],  // Dataset values
      label: 'Styling Preference',
      required: true
    },
    gender: {
      options: ['male', 'female'],  // Dataset only has these 2
      label: 'Gender',
      required: false
    },
    hair_condition: {
      options: ['none', 'thinning', 'split_ends', 'dry_ends', 'frizzy', 'dandruff', 'oily_scalp', 'sensitive_scalp', 'damaged'],  // Dataset values
      label: 'Hair Condition',
      required: false
    }
  };

  // Validate a specific field
  const validateField = (fieldName, value) => {
    const rule = validationRules[fieldName];
    if (!rule) return { valid: true };

    if (rule.required && (!value || value === '')) {
      return {
        valid: false,
        message: `${rule.label} is required`
      };
    }

    if (value && rule.options && !rule.options.includes(value)) {
      return {
        valid: false,
        message: `Invalid ${rule.label}. Must be one of: ${rule.options.join(', ')}`
      };
    }

    return { valid: true };
  };

  // Validate entire form
  const validateAllFields = () => {
    const errors = {};
    
    Object.keys(validationRules).forEach(fieldName => {
      const rule = validationRules[fieldName];
      if (rule.required) {
        const validation = validateField(fieldName, preferences[fieldName]);
        if (!validation.valid) {
          errors[fieldName] = validation.message;
        }
      }
    });

    // Validate occasions (at least one required)
    if (!preferences.occasions || preferences.occasions.length === 0) {
      errors.occasions = 'Please select at least one occasion';
    }

    return errors;
  };

  // Check if current step is valid
  const isCurrentStepValid = () => {
    switch (currentStep) {
      case 0:
        return validateField('gender', preferences.gender).valid;
      case 1: 
        return validateField('hair_type', preferences.hair_type).valid;
      case 2: 
        return validateField('hair_length', preferences.hair_length).valid;
      case 3: 
        return validateField('volume', preferences.volume).valid;
      case 4: 
        return validateField('hair_thickness', preferences.hair_thickness).valid;
      case 5: 
        return validateField('hair_texture_detail', preferences.hair_texture_detail).valid;
      case 6: 
        return validateField('lifestyle', preferences.lifestyle).valid;
      case 7: 
        return validateField('maintenance', preferences.maintenance).valid;
      case 8: 
        return validateField('styling_preference', preferences.styling_preference).valid;
      case 9: 
        return preferences.occasions && preferences.occasions.length > 0;
      case 10:
        return true; // Hair condition is optional
      default: 
        return false;
    }
  };

  // Get validation message for current step
  const getCurrentStepValidationMessage = () => {
    let fieldName, label;
    
    switch (currentStep) {
      case 0:
        fieldName = 'gender';
        label = 'gender';
        break;
      case 1: 
        fieldName = 'hair_type';
        label = 'hair type';
        break;
      case 2: 
        fieldName = 'hair_length';
        label = 'hair length';
        break;
      case 3: 
        fieldName = 'volume';
        label = 'volume preference';
        break;
      case 4: 
        fieldName = 'hair_thickness';
        label = 'hair thickness';
        break;
      case 5: 
        fieldName = 'hair_texture_detail';
        label = 'hair texture';
        break;
      case 6: 
        fieldName = 'lifestyle';
        label = 'lifestyle';
        break;
      case 7: 
        fieldName = 'maintenance';
        label = 'maintenance level';
        break;
      case 8: 
        fieldName = 'styling_preference';
        label = 'styling preference';
        break;
      case 9: 
        return 'Please select at least one occasion';
      case 10:
        return ''; // Optional field
      default: 
        return '';
    }

    const validation = validateField(fieldName, preferences[fieldName]);
    return validation.valid ? '' : `Please select a ${label}`;
  };

  const isFormValid = () => {
    return preferences.gender &&
           preferences.hair_type && 
           preferences.hair_length && 
           preferences.volume &&
           preferences.hair_thickness &&
           preferences.hair_texture_detail &&
           preferences.lifestyle && 
           preferences.maintenance &&
           preferences.styling_preference &&
           preferences.occasions.length > 0;
    // Note: hair_condition is optional
  };

  // Step definitions
  const steps = [
    { number: 0, title: 'About You', description: 'Tell us a bit about yourself' },
    { number: 1, title: 'Hair Type', description: 'What\'s your current hair type?' },
    { number: 2, title: 'Hair Length', description: 'What length are you considering?' },
    { number: 3, title: 'Volume', description: 'How much volume do you prefer?' },
    { number: 4, title: 'Hair Thickness', description: 'How thick is your hair?' },
    { number: 5, title: 'Hair Texture', description: 'What\'s your hair texture like?' },
    { number: 6, title: 'Lifestyle', description: 'What\'s your lifestyle like?' },
    { number: 7, title: 'Maintenance', description: 'How much maintenance do you prefer?' },
    { number: 8, title: 'Styling', description: 'What\'s your styling preference?' },
    { number: 9, title: 'Occasions', description: 'What occasions do you style for?' },
    { number: 10, title: 'Hair Condition', description: 'What\'s your current hair health?' }
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
                  alt="Uploaded profile"
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
                Step {currentStep + 1} of {totalSteps + 1}: {steps[currentStep].title}
                <span className="ml-2 text-red-400 text-sm">{currentStep !== 10 ? '*' : ''}</span>
              </h2>
              <p className="text-lg text-gray-300">
                {steps[currentStep].description}
              </p>
              <p className="text-xs text-gray-400 mt-1">
                {currentStep !== 10 ? '* Required field' : 'Optional - helps us give better recommendations'}
              </p>
            </div>
          </div>

          {/* Step Content */}
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-8 md:p-12 shadow-xl mb-8">
            {/* Step 0: Gender & Face Shape */}
            {currentStep === 0 && (
              <div className="space-y-8">
                {/* Display detected face shape */}
                <div className="bg-blue-500/10 border border-blue-400/30 p-6 rounded-xl text-center">
                  <p className="text-sm text-gray-300 mb-2">✨ Detected Face Shape</p>
                  <p className="text-2xl font-bold text-blue-300 mb-1">
                    {uploadResponse?.face_shape?.shape?.charAt(0).toUpperCase() + 
                     uploadResponse?.face_shape?.shape?.slice(1) || 'Not detected'}
                  </p>
                  {uploadResponse?.face_shape?.confidence && (
                    <p className="text-xs text-gray-400">
                      {(uploadResponse.face_shape.confidence * 100).toFixed(0)}% confident
                    </p>
                  )}
                </div>

                {/* Gender selection */}
                <div>
                  <label className="block text-center text-lg font-medium text-gray-200 mb-6">
                    👤 Select your gender (helps personalize recommendations)
                  </label>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <button
                      onClick={() => handlePreferenceChange('gender', 'male')}
                      className={`p-8 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                        preferences.gender === 'male'
                          ? 'border-blue-400 bg-blue-500/20 text-blue-300 shadow-lg shadow-blue-500/25'
                          : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                      }`}
                    >
                      <div className="text-4xl mb-3">👨</div>
                      <div className="font-medium text-lg">Male</div>
                    </button>
                    
                    <button
                      onClick={() => handlePreferenceChange('gender', 'female')}
                      className={`p-8 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                        preferences.gender === 'female'
                          ? 'border-pink-400 bg-pink-500/20 text-pink-300 shadow-lg shadow-pink-500/25'
                          : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                      }`}
                    >
                      <div className="text-4xl mb-3">👩</div>
                      <div className="font-medium text-lg">Female</div>
                    </button>
                    
                    <button
                      onClick={() => handlePreferenceChange('gender', 'nb')}
                      className={`p-8 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                        preferences.gender === 'nb'
                          ? 'border-purple-400 bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/25'
                          : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                      }`}
                    >
                      <div className="text-4xl mb-3">🧑</div>
                      <div className="font-medium text-lg">Non-binary</div>
                    </button>
                    
                    <button
                      onClick={() => handlePreferenceChange('gender', 'other')}
                      className={`p-8 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                        preferences.gender === 'other'
                          ? 'border-gray-400 bg-gray-500/20 text-gray-300 shadow-lg shadow-gray-500/25'
                          : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                      }`}
                    >
                      <div className="text-4xl mb-3">👤</div>
                      <div className="font-medium text-lg">Prefer not to say</div>
                    </button>
                  </div>
                </div>
              </div>
            )}

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
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {['short', 'medium', 'long'].map((length) => (
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
                        {length === 'short' && '💇‍♀️'}
                        {length === 'medium' && '👩‍🦰'}
                        {length === 'long' && '🧚‍♀️'}
                      </div>
                      <div className="font-medium text-lg">{length.charAt(0).toUpperCase() + length.slice(1)}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Step 3: Volume */}
            {currentStep === 3 && (
              <div className="space-y-8">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {[
                    { value: 'low', label: 'Low Volume', emoji: '📏' },
                    { value: 'medium', label: 'Medium Volume', emoji: '🎈' },
                    { value: 'high', label: 'High/Full Volume', emoji: '🎪' }
                  ].map((vol) => (
                    <button
                      key={vol.value}
                      onClick={() => handlePreferenceChange('volume', vol.value)}
                      className={`p-6 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                        preferences.volume === vol.value
                          ? 'border-purple-400 bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/25'
                          : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                      }`}
                    >
                      <div className="text-4xl mb-3">{vol.emoji}</div>
                      <div className="font-medium text-sm">{vol.label}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Step 4: Hair Thickness */}
            {currentStep === 4 && (
              <div className="space-y-8">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {[
                    { value: 'thin', label: 'Thin', emoji: '🪶' },
                    { value: 'medium', label: 'Medium', emoji: '🌿' },
                    { value: 'thick', label: 'Thick', emoji: '🌲' }
                  ].map((thickness) => (
                    <button
                      key={thickness.value}
                      onClick={() => handlePreferenceChange('hair_thickness', thickness.value)}
                      className={`p-8 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                        preferences.hair_thickness === thickness.value
                          ? 'border-purple-400 bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/25'
                          : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                      }`}
                    >
                      <div className="text-4xl mb-3">{thickness.emoji}</div>
                      <div className="font-medium text-lg">{thickness.label}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Step 5: Hair Texture Detail */}
            {currentStep === 5 && (
              <div className="space-y-8">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {[
                    { value: 'fine', label: 'Fine', emoji: '🪶' },
                    { value: 'normal', label: 'Normal', emoji: '�' },
                    { value: 'thick', label: 'Thick', emoji: '🌲' }
                  ].map((texture) => (
                    <button
                      key={texture.value}
                      onClick={() => handlePreferenceChange('hair_texture_detail', texture.value)}
                      className={`p-8 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                        preferences.hair_texture_detail === texture.value
                          ? 'border-purple-400 bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/25'
                          : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                      }`}
                    >
                      <div className="text-4xl mb-3">{texture.emoji}</div>
                      <div className="font-medium text-lg">{texture.label}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Step 6: Lifestyle */}
            {currentStep === 6 && (
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

            {/* Step 7: Maintenance */}
            {currentStep === 7 && (
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

            {/* Step 8: Styling Preference */}
            {currentStep === 8 && (
              <div className="space-y-8">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                  {[
                    { value: 'natural', label: 'Natural', emoji: '🌱', description: 'Embrace your natural texture with minimal effort' },
                    { value: 'classic', label: 'Classic', emoji: '�', description: 'Timeless, traditional hairstyles' },
                    { value: 'elegant', label: 'Elegant', emoji: '💼', description: 'Sophisticated and refined styles' },
                    { value: 'trendy', label: 'Trendy', emoji: '✨', description: 'Current fashion-forward looks' },
                    { value: 'edgy', label: 'Edgy', emoji: '🎸', description: 'Bold and unconventional styles' }
                  ].map((style) => (
                    <button
                      key={style.value}
                      onClick={() => handlePreferenceChange('styling_preference', style.value)}
                      className={`p-6 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 text-left ${
                        preferences.styling_preference === style.value
                          ? 'border-purple-400 bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/25'
                          : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                      }`}
                    >
                      <div className="text-4xl mb-3">{style.emoji}</div>
                      <div className="font-medium text-lg mb-2">{style.label}</div>
                      <div className="text-sm opacity-80">{style.description}</div>
                    </button>
                  ))}
                </div>
                
                {/* Additional options */}
                <div className="mt-8 space-y-4">
                  <div className="bg-gray-700/30 rounded-xl p-6">
                    <label className="flex items-center space-x-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={preferences.wants_bangs}
                        onChange={(e) => handlePreferenceChange('wants_bangs', e.target.checked)}
                        className="w-5 h-5 rounded border-gray-600 text-purple-600 focus:ring-purple-500"
                      />
                      <span className="text-white font-medium">I want bangs/fringe</span>
                    </label>
                  </div>
                </div>
              </div>
            )}

            {/* Step 9: Occasions */}
            {currentStep === 9 && (
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

            {/* Step 10: Hair Condition */}
            {currentStep === 10 && (
              <div className="space-y-8">
                <p className="text-center text-gray-300 text-lg mb-6">
                  Optional - Select your current hair condition to get more personalized recommendations
                </p>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {[
                    { value: 'none', label: 'Healthy', emoji: '✨' },
                    { value: 'dry_ends', label: 'Dry Ends', emoji: '🌵' },
                    { value: 'damaged', label: 'Damaged', emoji: '⚠️' },
                    { value: 'split_ends', label: 'Split Ends', emoji: '✂️' },
                    { value: 'thinning', label: 'Thinning', emoji: '📉' },
                    { value: 'frizzy', label: 'Frizzy', emoji: '🌩️' },
                    { value: 'oily_scalp', label: 'Oily Scalp', emoji: '💧' },
                    { value: 'dandruff', label: 'Dandruff', emoji: '❄️' },
                    { value: 'sensitive_scalp', label: 'Sensitive Scalp', emoji: '🩹' }
                  ].map((condition) => (
                    <button
                      key={condition.value}
                      onClick={() => handlePreferenceChange('hair_condition', condition.value)}
                      className={`p-6 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                        preferences.hair_condition === condition.value
                          ? 'border-purple-400 bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/25'
                          : 'border-gray-600 hover:border-gray-500 bg-gray-700/30 text-gray-300 hover:text-white hover:bg-gray-600/30'
                      }`}
                    >
                      <div className="text-3xl mb-2">{condition.emoji}</div>
                      <div className="font-medium text-sm">{condition.label}</div>
                    </button>
                  ))}
                </div>
                {preferences.hair_condition && (
                  <div className="bg-purple-900/20 border border-purple-500/30 rounded-xl p-4 mt-6">
                    <p className="text-purple-300 text-center">
                      ✅ Hair condition selected
                    </p>
                  </div>
                )}
                {!preferences.hair_condition && (
                  <div className="bg-gray-700/20 border border-gray-600/30 rounded-xl p-4 mt-6">
                    <p className="text-gray-400 text-center text-sm">
                      💡 Tip: Selecting your hair condition helps us recommend styles that work best for your hair health
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Face shape detected but not displayed to user */}

          {/* Navigation Buttons */}
          <div className="space-y-4">
            {/* Validation Error Message */}
            {!isCurrentStepValid() && (
              <div className="bg-red-500/10 border border-red-500/50 text-red-300 px-4 py-3 rounded-lg text-center">
                <span className="font-medium">⚠️ {getCurrentStepValidationMessage()}</span>
              </div>
            )}

            <div className="flex justify-between items-center">
              <button
                onClick={prevStep}
                disabled={currentStep === 0}
                className={`px-8 py-3 rounded-xl font-medium transition-all duration-300 ${
                  currentStep === 0
                    ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                    : 'bg-gray-700 hover:bg-gray-600 text-white'
                }`}
              >
                ← Previous
              </button>

              <div className="text-center">
                <div className="text-white text-sm">
                  Step {currentStep + 1} of {totalSteps + 1}
                </div>
                <div className="text-gray-400 text-xs">
                  {Math.round(((currentStep + 1) / (totalSteps + 1)) * 100)}% Complete
                </div>
              </div>

              <button
                onClick={nextStep}
                disabled={!isCurrentStepValid() || isSubmitting}
                title={!isCurrentStepValid() ? getCurrentStepValidationMessage() : ''}
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
    </div>
  );
};

export default UserPreferences;