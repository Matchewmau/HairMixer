import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import APIService from "../services/api";
import AuthService from "../services/AuthService";
import Navbar from "../components/Navbar";
import Button from "../components/ui/Button";
import Card from "../components/ui/Card";

/**
 * UserPreferences Component
 *
 * A comprehensive 11-step wizard for collecting detailed user hair preferences.
 * Refactored to use the new design system.
 */
const UserPreferences = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { imageFile, previewUrl, uploadResponse, existingPreferences } =
    location.state || {};

  // Step-by-step wizard state
  const [currentStep, setCurrentStep] = useState(0);
  const totalSteps = 13;

  const [preferences, setPreferences] = useState({
    // Core characteristics
    hair_type: existingPreferences?.hair_type || "",
    hair_length: existingPreferences?.hair_length || "",
    lifestyle: existingPreferences?.lifestyle || "",
    maintenance: existingPreferences?.maintenance || "",
    occasions: existingPreferences?.occasions || [],

    // New detailed characteristics
    volume: existingPreferences?.volume || "",
    styling_maintenance: existingPreferences?.styling_maintenance || "",
    hair_texture_detail: existingPreferences?.hair_texture_detail || "",
    styling_preference: existingPreferences?.styling_preference || "",
    hair_condition: existingPreferences?.hair_condition || [],
    hair_thickness: existingPreferences?.hair_thickness || "",
    wants_bangs: existingPreferences?.wants_bangs || false,
    hair_color: existingPreferences?.hair_color || "",
    color_preference: existingPreferences?.color_preference || "",
    gender: existingPreferences?.gender || "",

    // Hairstyle preferences
    hairstyle_family: existingPreferences?.hairstyle_family || "",
    hairstyle_name: existingPreferences?.hairstyle_name || "",

    // Face shape (auto-filled from ResNet50)
    faceshape:
      uploadResponse?.face_shape?.shape || existingPreferences?.faceshape || "",

    // Legacy compatibility check
    check_compatibility: existingPreferences?.check_compatibility || false,
    target_hairstyle: existingPreferences?.target_hairstyle || "",
    custom_hairstyle: existingPreferences?.custom_hairstyle || "",
  });

  const [occasions, setOccasions] = useState([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [user, setUser] = useState(null);
  const [occasionsLoaded, setOccasionsLoaded] = useState(false);

  // Preference Profiles state (kept for logic consistency, though UI might not use it directly in wizard)
  const [preferenceProfiles, setPreferenceProfiles] = useState([]);

  useEffect(() => {
    if (!uploadResponse) {
      navigate("/upload");
      return;
    }

    if (!occasionsLoaded) {
      loadFilterOptions();
    }
    checkAuth();
  }, [uploadResponse, navigate, occasionsLoaded]);

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (!document.hidden) {
        checkAuth();
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () =>
      document.removeEventListener("visibilitychange", handleVisibilityChange);
  }, []);

  useEffect(() => {
    if (user && !occasionsLoaded) {
      loadFilterOptions();
    }
  }, [user, occasionsLoaded]);

  const checkAuth = async () => {
    try {
      if (!AuthService.getAccessToken()) {
        setUser(null);
        return;
      }
      const currentUser = await AuthService.getCurrentUser();
      setUser(currentUser);
    } catch (error) {
      console.error("Authentication check failed:", error);
      setUser(null);
    }
  };

  const handleLogout = async () => {
    try {
      await AuthService.logout();
      setUser(null);
      setOccasions([]);
      setOccasionsLoaded(false);
      navigate("/");
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  const loadFilterOptions = async () => {
    try {
      const occasionsResponse = await APIService.getOccasions();
      setOccasions(occasionsResponse.occasions || []);
      setOccasionsLoaded(true);
    } catch (error) {
      console.error("Error loading filter options:", error);
      setOccasions([
        { value: "casual", label: "Casual" },
        { value: "professional", label: "Professional" },
        { value: "formal", label: "Formal" },
        { value: "party", label: "Party" },
        { value: "wedding", label: "Wedding" },
        { value: "sports", label: "Sports" },
      ]);
      setOccasionsLoaded(true);
    } finally {
      setIsLoading(false);
    }
  };

  const loadPreferenceProfiles = async () => {
    try {
      const response = await APIService.getPreferenceProfiles();
      setPreferenceProfiles(response.profiles || []);
    } catch (error) {
      console.error("Failed to load preference profiles:", error);
    }
  };

  useEffect(() => {
    if (user) {
      loadPreferenceProfiles();
    }
  }, [user]);

  const handleApplyProfile = (profile) => {
    // Map profile data to preferences state
    setPreferences((prev) => ({
      ...prev,
      gender: profile.gender || prev.gender,
      hair_type: profile.hair_type || prev.hair_type,
      hair_length: profile.hair_length || prev.hair_length,
      volume: profile.volume || prev.volume,
      hair_thickness: profile.hair_thickness || prev.hair_thickness,
      hair_texture_detail:
        profile.hair_texture_detail || prev.hair_texture_detail,
      lifestyle: profile.lifestyle || prev.lifestyle,
      maintenance: profile.maintenance || prev.maintenance,
      styling_maintenance: profile.styling_maintenance || "medium", // Default if missing
      styling_preference: profile.styling_preference || "natural", // Default if missing
      hair_color: profile.hair_color || prev.hair_color,
      hair_condition: profile.hair_condition || [],
      occasions: profile.occasions || [],
    }));

    // Skip to the last step (Hair Condition)
    setCurrentStep(totalSteps - 1);
  };

  const handleSubmit = async () => {
    if (!isFormValid()) {
      alert("⚠️ Please complete all required fields before submitting.");
      return;
    }

    const errors = validateAllFields();
    if (Object.keys(errors).length > 0) {
      const errorMessages = Object.values(errors).join("\n");
      alert(`⚠️ Please fix the following errors:\n\n${errorMessages}`);
      return;
    }

    setIsSubmitting(true);

    try {
      const cleanedPreferences = {
        ...preferences,
        faceshape:
          uploadResponse?.face_shape?.shape || preferences.faceshape || "",
        hair_condition: preferences.hair_condition || [],
      };

      const preferencesResponse = await APIService.savePreferences(
        cleanedPreferences
      );

      if (!preferencesResponse.success) {
        throw new Error(
          preferencesResponse.error || "Failed to save preferences"
        );
      }

      const mlRecommendationsResponse = await APIService.getRecommendations(
        uploadResponse.image_id,
        preferencesResponse.preference_id
      );

      navigate("/results", {
        state: {
          preferences: cleanedPreferences,
          imageFile,
          previewUrl,
          uploadResponse,
          recommendations: mlRecommendationsResponse,
        },
      });
    } catch (error) {
      console.error("Error submitting preferences:", error);
      let errorMessage = "❌ Failed to get recommendations. ";
      if (
        error.message.includes("network") ||
        error.message.includes("fetch")
      ) {
        errorMessage += "Please check your internet connection and try again.";
      } else {
        errorMessage += `\n\n${error.message}`;
      }
      alert(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handlePreferenceChange = (key, value) => {
    if (key === "occasions") {
      const newOccasions = preferences.occasions.includes(value)
        ? preferences.occasions.filter((o) => o !== value)
        : [...preferences.occasions, value];

      setPreferences((prev) => ({
        ...prev,
        occasions: newOccasions,
      }));
    } else if (key === "hair_condition") {
      const newConditions = preferences.hair_condition.includes(value)
        ? preferences.hair_condition.filter((c) => c !== value)
        : [...preferences.hair_condition, value];

      setPreferences((prev) => ({
        ...prev,
        hair_condition: newConditions,
      }));
    } else if (key === "wants_bangs") {
      setPreferences((prev) => ({
        ...prev,
        [key]: !prev[key],
      }));
    } else {
      const newValue = preferences[key] === value ? "" : value;
      setPreferences((prev) => ({
        ...prev,
        [key]: newValue,
      }));
    }
  };

  const nextStep = () => {
    if (!isCurrentStepValid()) {
      const errorMsg = getCurrentStepValidationMessage();
      if (errorMsg) alert(`⚠️ ${errorMsg}`);
      return;
    }

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
    if (step <= currentStep) {
      setCurrentStep(step);
    } else if (step === currentStep + 1 && isStepCompleted(currentStep)) {
      setCurrentStep(step);
    }
  };

  const isStepCompleted = (stepNumber) => {
    switch (stepNumber) {
      case 0:
        return !!preferences.gender;
      case 1:
        return !!preferences.hair_type;
      case 2:
        return !!preferences.hair_length;
      case 3:
        return !!preferences.volume;
      case 4:
        return !!preferences.hair_thickness;
      case 5:
        return !!preferences.hair_texture_detail;
      case 6:
        return !!preferences.lifestyle;
      case 7:
        return !!preferences.maintenance;
      case 8:
        return !!preferences.styling_maintenance;
      case 9:
        return !!preferences.styling_preference;
      case 10:
        return preferences.occasions && preferences.occasions.length > 0;
      case 11:
        return !!preferences.hair_color;
      case 12:
        return true; // Hair condition is optional
      default:
        return false;
    }
  };

  const validationRules = {
    hair_type: {
      options: ["straight", "wavy", "curly", "coily"],
      descriptions: {
        straight: "No natural curl or wave",
        wavy: "Light 'S' shape pattern",
        curly: "Defined ringlets or loops",
        coily: "Tight curls or zig-zag pattern",
      },
      label: "Hair Type",
      required: true,
    },
    hair_length: {
      options: ["short", "medium", "long"],
      descriptions: {
        short: "Above the ears or chin length",
        medium: "Shoulder length to armpit",
        long: "Below the shoulders",
      },
      label: "Hair Length",
      required: true,
    },
    volume: {
      options: ["low", "medium", "high"],
      descriptions: {
        low: "Flat, close to the head",
        medium: "Natural body and bounce",
        high: "Full, voluminous look",
      },
      label: "Volume",
      required: true,
    },
    hair_thickness: {
      options: ["thin", "medium", "thick", "very_thick"],
      descriptions: {
        thin: "Fine strands, lightweight",
        medium: "Average strand thickness",
        thick: "Coarse, heavy strands",
        very_thick: "Very dense and full",
      },
      label: "Hair Thickness",
      required: true,
    },
    hair_texture_detail: {
      options: [
        "fine",
        "normal",
        "thick",
        "smooth",
        "coarse",
        "silky",
        "frizzy",
      ],
      descriptions: {
        fine: "Delicate, easy to weigh down",
        normal: "Balanced texture",
        thick: "Strong, holds style well",
        smooth: "Sleek and soft",
        coarse: "Rougher feel, resilient",
        silky: "Very soft and shiny",
        frizzy: "Prone to flyaways",
      },
      label: "Hair Texture",
      required: true,
    },
    lifestyle: {
      options: ["active", "moderate", "relaxed"],
      descriptions: {
        active: "Gym, sports, always moving",
        moderate: "Balanced daily activity",
        relaxed: "Low-key, easygoing routine",
      },
      label: "Lifestyle",
      required: true,
    },
    maintenance: {
      options: ["low", "medium", "high"],
      descriptions: {
        low: "Wash and go, minimal effort",
        medium: "Some styling required",
        high: "Regular styling and upkeep",
      },
      label: "Overall Maintenance",
      required: true,
    },
    styling_maintenance: {
      options: ["low", "medium", "high"],
      descriptions: {
        low: "Less than 10 mins/day",
        medium: "10-20 mins/day",
        high: "20+ mins/day",
      },
      label: "Daily Styling Maintenance",
      required: true,
    },
    styling_preference: {
      options: [
        "natural",
        "casual",
        "classic",
        "polished",
        "elegant",
        "glamorous",
        "trendy",
        "edgy",
      ],
      descriptions: {
        natural: "Effortless, 'woke up like this'",
        casual: "Relaxed and informal",
        classic: "Timeless and neat",
        polished: "Sleek and put-together",
        elegant: "Sophisticated and refined",
        glamorous: "Bold, red-carpet ready",
        trendy: "Modern and fashionable",
        edgy: "Bold, unconventional",
      },
      label: "Styling Preference",
      required: true,
    },
    gender: {
      options: ["male", "female"],
      descriptions: {
        male: "Masculine styles",
        female: "Feminine styles",
      },
      label: "Gender",
      required: false,
    },
    hair_color: {
      options: [
        "natural",
        "black",
        "brown",
        "blonde",
        "red",
        "auburn",
        "gray",
        "white",
        "other",
      ],
      descriptions: {
        natural: "Your natural shade",
        black: "Darkest shade",
        brown: "Light to dark brown",
        blonde: "Light, golden tones",
        red: "Vibrant red tones",
        auburn: "Reddish-brown",
        gray: "Silver or gray tones",
        white: "Pure white or platinum",
        other: "Unconventional colors",
      },
      label: "Hair Color",
      required: true,
    },
    hair_condition: {
      options: [
        "none",
        "excellent",
        "good",
        "fair",
        "damaged",
        "dry_ends",
        "oily_scalp",
        "dandruff",
        "frizzy",
        "split_ends",
        "thinning",
        "sensitive_scalp",
      ],
      descriptions: {
        none: "No specific issues",
        excellent: "Healthy and shiny",
        good: "Generally healthy",
        fair: "Some minor issues",
        damaged: "Needs repair/treatment",
        dry_ends: "Ends are brittle/dry",
        oily_scalp: "Roots get oily quickly",
        dandruff: "Flaking scalp",
        frizzy: "Hard to tame frizz",
        split_ends: "Ends are splitting",
        thinning: "Hair feels less dense",
        sensitive_scalp: "Scalp irritation",
      },
      label: "Hair Condition",
      required: false,
      multiSelect: true,
    },
  };

  const validateField = (fieldName, value) => {
    const rule = validationRules[fieldName];
    if (!rule) return { valid: true };
    if (rule.required && (!value || value === ""))
      return { valid: false, message: `${rule.label} is required` };
    if (value && rule.options && !rule.options.includes(value))
      return { valid: false, message: `Invalid ${rule.label}` };
    return { valid: true };
  };

  const validateAllFields = () => {
    const errors = {};
    Object.keys(validationRules).forEach((fieldName) => {
      const rule = validationRules[fieldName];
      if (rule.required) {
        const validation = validateField(fieldName, preferences[fieldName]);
        if (!validation.valid) errors[fieldName] = validation.message;
      }
    });
    if (!preferences.occasions || preferences.occasions.length === 0)
      errors.occasions = "Please select at least one occasion";
    return errors;
  };

  const isCurrentStepValid = () => {
    switch (currentStep) {
      case 0:
        return validateField("gender", preferences.gender).valid;
      case 1:
        return validateField("hair_type", preferences.hair_type).valid;
      case 2:
        return validateField("hair_length", preferences.hair_length).valid;
      case 3:
        return validateField("volume", preferences.volume).valid;
      case 4:
        return validateField("hair_thickness", preferences.hair_thickness)
          .valid;
      case 5:
        return validateField(
          "hair_texture_detail",
          preferences.hair_texture_detail
        ).valid;
      case 6:
        return validateField("lifestyle", preferences.lifestyle).valid;
      case 7:
        return validateField("maintenance", preferences.maintenance).valid;
      case 8:
        return !!preferences.styling_maintenance;
      case 9:
        return !!preferences.styling_preference;
      case 10:
        return preferences.occasions && preferences.occasions.length > 0;
      case 11:
        return validateField("hair_color", preferences.hair_color).valid;
      case 12:
        return true;
      default:
        return false;
    }
  };

  const getCurrentStepValidationMessage = () => {
    // Simplified validation message logic
    if (!isCurrentStepValid()) return "Please complete the selection";
    return "";
  };

  const isFormValid = () => {
    return (
      preferences.gender &&
      preferences.hair_type &&
      preferences.hair_length &&
      preferences.volume &&
      preferences.hair_thickness &&
      preferences.hair_texture_detail &&
      preferences.lifestyle &&
      preferences.maintenance &&
      preferences.styling_maintenance &&
      preferences.styling_preference &&
      preferences.occasions.length > 0 &&
      preferences.hair_color
    );
  };

  const steps = [
    {
      number: 0,
      title: "About You",
      description: "",
    },
    {
      number: 1,
      title: "Hair Type",
      description: "What's your current hair type?",
    },
    {
      number: 2,
      title: "Hair Length",
      description: "What length are you considering?",
    },
    {
      number: 3,
      title: "Volume",
      description: "How much volume do you prefer?",
    },
    {
      number: 4,
      title: "Hair Thickness",
      description: "How thick is your hair?",
    },
    {
      number: 5,
      title: "Hair Texture",
      description: "What's your hair texture like?",
    },
    {
      number: 6,
      title: "Lifestyle",
      description: "What's your lifestyle like?",
    },
    {
      number: 7,
      title: "Overall Maintenance",
      description: "How much overall maintenance do you prefer?",
    },
    {
      number: 8,
      title: "Daily Styling",
      description: "How much time for daily styling?",
    },
    {
      number: 9,
      title: "Styling Preference",
      description: "What's your styling preference?",
    },
    {
      number: 10,
      title: "Occasions",
      description: "What occasions do you style for?",
    },
    {
      number: 11,
      title: "Hair Color",
      description: "What's your current hair color?",
    },
    {
      number: 12,
      title: "Hair Condition",
      description: "What's your current hair health?",
    },
  ];

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (!uploadResponse) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-heading font-bold text-white mb-6">
            No image found
          </h2>
          <Button
            onClick={() => navigate("/upload")}
            variant="primary"
            size="lg"
          >
            Upload Photo
          </Button>
        </div>
      </div>
    );
  }

  // Helper to render selection cards
  const SelectionCard = ({
    label,
    description,
    value,
    selected,
    onClick,
    multi = false,
  }) => (
    <div
      onClick={onClick}
      className={`
        cursor-pointer rounded-xl p-4 border transition-all duration-300 flex flex-col items-center justify-center text-center gap-2 h-full
        ${
          selected
            ? "bg-primary/20 border-primary shadow-lg shadow-primary/10 transform scale-105"
            : "bg-surface/50 border-white/5 hover:bg-surface/80 hover:border-white/20"
        }
      `}
    >
      <div
        className={`
        w-6 h-6 rounded-full flex items-center justify-center border shrink-0
        ${
          selected
            ? "bg-primary border-primary text-white"
            : "border-gray-500 text-transparent"
        }
      `}
      >
        {selected && "✓"}
      </div>
      <div className="flex flex-col gap-1">
        <span
          className={`font-medium ${selected ? "text-white" : "text-gray-300"}`}
        >
          {label}
        </span>
        {description && (
          <span className="text-xs text-gray-400 font-light">
            {description}
          </span>
        )}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-background">
      <Navbar
        transparent={true}
        user={user}
        onLogout={handleLogout}
        showBackButton={true}
        backPath="/upload"
      />

      <div className="pt-24 pb-12 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
        {/* Background Elements */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-primary/10 rounded-full blur-3xl opacity-30 pointer-events-none"></div>
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-secondary/10 rounded-full blur-3xl opacity-30 pointer-events-none"></div>

        <div className="max-w-4xl mx-auto relative z-10">
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-4xl md:text-5xl font-heading font-bold text-white mb-4">
              Tell us about your preferences
            </h1>
            {previewUrl && (
              <div className="flex justify-center mb-6">
                <div className="relative">
                  <img
                    src={previewUrl}
                    alt="Uploaded profile"
                    className="h-32 w-32 object-cover rounded-full border-4 border-primary/30 shadow-2xl"
                  />
                  <div className="absolute bottom-0 right-0 bg-surface border border-white/10 rounded-full px-3 py-1 text-xs font-bold text-white shadow-lg">
                    Step {currentStep + 1}/{totalSteps}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Segmented Progress Bar */}
          <div className="mb-8">
            <div className="flex gap-1 mb-2 overflow-x-auto scrollbar-hide">
              {steps.map((step, index) => {
                const isCompleted =
                  index < currentStep || isStepCompleted(index);
                const isActive = index === currentStep;
                const canNavigate =
                  index <= currentStep || isStepCompleted(index - 1); // Can go to next if previous is done

                return (
                  <div
                    key={index}
                    onClick={() => canNavigate && goToStep(index)}
                    className={`flex-1 min-w-[20px] h-3 rounded-full transition-all duration-300 relative group ${
                      canNavigate
                        ? "cursor-pointer"
                        : "cursor-not-allowed opacity-30"
                    } ${
                      isActive
                        ? "bg-primary shadow-[0_0_10px_rgba(37,99,235,0.5)]"
                        : isCompleted
                        ? "bg-primary/60 hover:bg-primary/80"
                        : "bg-surface border border-white/5"
                    }`}
                  >
                    {/* Tooltip */}
                    <div className="hidden md:block absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-surface border border-white/10 rounded text-xs text-white opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-20">
                      {step.title}
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-primary">
                Step {currentStep + 1} of {totalSteps}
              </span>
              <span className="text-sm text-gray-400 font-medium">
                {steps[currentStep].title}
              </span>
            </div>
          </div>

          {/* Step Content */}
          <Card className="p-6 md:p-12 mb-8 animate-fade-in">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-heading font-bold text-white mb-2">
                {steps[currentStep].title}
              </h2>
              <p className="text-gray-400">{steps[currentStep].description}</p>
            </div>

            {/* Step 0: Gender & Quick Start */}
            {currentStep === 0 && (
              <div className="space-y-8">
                <div className="bg-primary/10 border border-primary/20 p-6 rounded-xl text-center">
                  <p className="text-sm text-gray-300 mb-2">
                    ✨ Detected Face Shape
                  </p>
                  <p className="text-2xl font-bold text-primary mb-1 capitalize">
                    {uploadResponse?.face_shape?.shape || "Not detected"}
                  </p>
                  {uploadResponse?.face_shape?.confidence && (
                    <p className="text-xs text-gray-400 hidden">
                      {(uploadResponse.face_shape.confidence * 100).toFixed(0)}%
                      confident
                    </p>
                  )}
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-md mx-auto">
                  {validationRules.gender.options.map((opt) => (
                    <SelectionCard
                      key={opt}
                      label={opt.charAt(0).toUpperCase() + opt.slice(1)}
                      description={validationRules.gender.descriptions[opt]}
                      value={opt}
                      selected={preferences.gender === opt}
                      onClick={() => handlePreferenceChange("gender", opt)}
                    />
                  ))}
                </div>

                {/* Quick Start Section - Moved Below Gender */}
                {preferenceProfiles.length > 0 && (
                  <div className="mt-12 pt-8 border-t border-white/10">
                    <div className="text-center mb-6">
                      <span className="bg-surface px-4 py-1 rounded-full text-xs font-bold text-gray-400 uppercase tracking-wider border border-white/5">
                        OR
                      </span>
                    </div>
                    <h3 className="text-xl font-heading font-bold text-white mb-4 text-center">
                      ⚡ Quick Start with Saved Profile
                    </h3>
                    <p className="text-gray-400 text-center text-sm mb-6">
                      Select a profile to auto-fill all preferences
                    </p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {preferenceProfiles.map((profile) => (
                        <Card
                          key={profile.id}
                          className="p-4 cursor-pointer hover:border-primary/50 transition-all group bg-surface/40"
                          onClick={() => handleApplyProfile(profile)}
                        >
                          <div className="flex justify-between items-start mb-2">
                            <h4 className="font-bold text-white group-hover:text-primary transition-colors">
                              {profile.profile_name}
                            </h4>
                            <span className="text-xs bg-primary/20 text-primary px-2 py-0.5 rounded-full capitalize">
                              {profile.gender}
                            </span>
                          </div>
                          <p className="text-sm text-gray-400 line-clamp-2 mb-3">
                            {profile.description || "No description"}
                          </p>
                          <div className="text-xs text-primary font-medium flex items-center gap-1">
                            <span>Apply & Skip Inputs</span>
                            <span>→</span>
                          </div>
                        </Card>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Step 1: Hair Type */}
            {currentStep === 1 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
                {validationRules.hair_type.options.map((opt) => (
                  <SelectionCard
                    key={opt}
                    label={opt.charAt(0).toUpperCase() + opt.slice(1)}
                    description={validationRules.hair_type.descriptions[opt]}
                    value={opt}
                    selected={preferences.hair_type === opt}
                    onClick={() => handlePreferenceChange("hair_type", opt)}
                  />
                ))}
              </div>
            )}

            {/* Step 2: Hair Length */}
            {currentStep === 2 && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {validationRules.hair_length.options.map((opt) => (
                  <SelectionCard
                    key={opt}
                    label={opt.charAt(0).toUpperCase() + opt.slice(1)}
                    description={validationRules.hair_length.descriptions[opt]}
                    value={opt}
                    selected={preferences.hair_length === opt}
                    onClick={() => handlePreferenceChange("hair_length", opt)}
                  />
                ))}
              </div>
            )}

            {/* Step 3: Volume */}
            {currentStep === 3 && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {validationRules.volume.options.map((opt) => (
                  <SelectionCard
                    key={opt}
                    label={opt.charAt(0).toUpperCase() + opt.slice(1)}
                    description={validationRules.volume.descriptions[opt]}
                    value={opt}
                    selected={preferences.volume === opt}
                    onClick={() => handlePreferenceChange("volume", opt)}
                  />
                ))}
              </div>
            )}

            {/* Step 4: Hair Thickness */}
            {currentStep === 4 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
                {validationRules.hair_thickness.options.map((opt) => (
                  <SelectionCard
                    key={opt}
                    label={opt
                      .split("_")
                      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
                      .join(" ")}
                    description={
                      validationRules.hair_thickness.descriptions[opt]
                    }
                    value={opt}
                    selected={preferences.hair_thickness === opt}
                    onClick={() =>
                      handlePreferenceChange("hair_thickness", opt)
                    }
                  />
                ))}
              </div>
            )}

            {/* Step 5: Texture */}
            {currentStep === 5 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
                {validationRules.hair_texture_detail.options.map((opt) => (
                  <SelectionCard
                    key={opt}
                    label={opt.charAt(0).toUpperCase() + opt.slice(1)}
                    description={
                      validationRules.hair_texture_detail.descriptions[opt]
                    }
                    value={opt}
                    selected={preferences.hair_texture_detail === opt}
                    onClick={() =>
                      handlePreferenceChange("hair_texture_detail", opt)
                    }
                  />
                ))}
              </div>
            )}

            {/* Step 6: Lifestyle */}
            {currentStep === 6 && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {validationRules.lifestyle.options.map((opt) => (
                  <SelectionCard
                    key={opt}
                    label={opt.charAt(0).toUpperCase() + opt.slice(1)}
                    description={validationRules.lifestyle.descriptions[opt]}
                    value={opt}
                    selected={preferences.lifestyle === opt}
                    onClick={() => handlePreferenceChange("lifestyle", opt)}
                  />
                ))}
              </div>
            )}

            {/* Step 7: Overall Maintenance */}
            {currentStep === 7 && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {validationRules.maintenance.options.map((opt) => (
                  <SelectionCard
                    key={opt}
                    label={opt.charAt(0).toUpperCase() + opt.slice(1)}
                    description={validationRules.maintenance.descriptions[opt]}
                    value={opt}
                    selected={preferences.maintenance === opt}
                    onClick={() => handlePreferenceChange("maintenance", opt)}
                  />
                ))}
              </div>
            )}

            {/* Step 8: Daily Styling */}
            {currentStep === 8 && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {validationRules.styling_maintenance.options.map((opt) => (
                  <SelectionCard
                    key={opt}
                    label={opt.charAt(0).toUpperCase() + opt.slice(1)}
                    description={
                      validationRules.styling_maintenance.descriptions[opt]
                    }
                    value={opt}
                    selected={preferences.styling_maintenance === opt}
                    onClick={() =>
                      handlePreferenceChange("styling_maintenance", opt)
                    }
                  />
                ))}
              </div>
            )}

            {/* Step 9: Styling Preference */}
            {currentStep === 9 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
                {validationRules.styling_preference.options.map((opt) => (
                  <SelectionCard
                    key={opt}
                    label={opt.charAt(0).toUpperCase() + opt.slice(1)}
                    description={
                      validationRules.styling_preference.descriptions[opt]
                    }
                    value={opt}
                    selected={preferences.styling_preference === opt}
                    onClick={() =>
                      handlePreferenceChange("styling_preference", opt)
                    }
                  />
                ))}
              </div>
            )}

            {/* Step 10: Occasions */}
            {currentStep === 10 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                {occasions.map((occ) => (
                  <SelectionCard
                    key={occ.value}
                    label={occ.label}
                    value={occ.value}
                    selected={preferences.occasions.includes(occ.value)}
                    onClick={() =>
                      handlePreferenceChange("occasions", occ.value)
                    }
                    multi={true}
                  />
                ))}
              </div>
            )}

            {/* Step 11: Hair Color */}
            {currentStep === 11 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                {validationRules.hair_color.options.map((opt) => (
                  <SelectionCard
                    key={opt}
                    label={opt.charAt(0).toUpperCase() + opt.slice(1)}
                    description={validationRules.hair_color.descriptions[opt]}
                    value={opt}
                    selected={preferences.hair_color === opt}
                    onClick={() => handlePreferenceChange("hair_color", opt)}
                  />
                ))}
              </div>
            )}

            {/* Step 12: Hair Condition */}
            {currentStep === 12 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                {validationRules.hair_condition.options.map((opt) => (
                  <SelectionCard
                    key={opt}
                    label={opt
                      .split("_")
                      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
                      .join(" ")}
                    description={
                      validationRules.hair_condition.descriptions[opt]
                    }
                    value={opt}
                    selected={preferences.hair_condition.includes(opt)}
                    onClick={() =>
                      handlePreferenceChange("hair_condition", opt)
                    }
                    multi={true}
                  />
                ))}
              </div>
            )}
          </Card>

          {/* Navigation Buttons */}
          <div className="flex justify-between gap-4">
            <Button
              onClick={prevStep}
              disabled={currentStep === 0}
              variant="secondary"
              className="w-1/3"
            >
              Back
            </Button>
            <Button
              onClick={nextStep}
              variant="primary"
              className="w-2/3"
              isLoading={isSubmitting}
            >
              {currentStep === totalSteps - 1
                ? "Get Recommendations"
                : "Next Step"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserPreferences;
