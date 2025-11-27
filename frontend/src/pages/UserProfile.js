import React, { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import AuthService from "../services/AuthService";
import apiService from "../services/api";
import Button from "../components/ui/Button";
import Card from "../components/ui/Card";
import Input from "../components/ui/Input";
import Modal from "../components/ui/Modal";

const UserProfile = () => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [savedHairstyles, setSavedHairstyles] = useState([]);
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
  });

  // Preference Profiles state
  const [preferenceProfiles, setPreferenceProfiles] = useState([]);
  const [showCreateProfileModal, setShowCreateProfileModal] = useState(false);
  const [showEditProfileModal, setShowEditProfileModal] = useState(false);
  const [selectedProfile, setSelectedProfile] = useState(null);
  const [profileForm, setProfileForm] = useState({
    profile_name: "",
    description: "",
    gender: "female",
    hair_type: "straight",
    hair_length: "medium",
    volume: "medium",
    hair_thickness: "medium",
    hair_texture_detail: "normal",
    lifestyle: "casual",
    maintenance: "medium",
    styling_preference: "natural",
    hair_color: "brown",
    hair_condition: [],
    occasions: [],
  });

  const navigate = useNavigate();

  // State for viewing saved hairstyle details
  const [showSavedModal, setShowSavedModal] = useState(false);
  const [selectedSaved, setSelectedSaved] = useState(null);

  // Full Screen Image Modal State
  const [showImageModal, setShowImageModal] = useState(false);
  const [activeImage, setActiveImage] = useState(null);

  const openImageModal = (imageUrl, e) => {
    if (e) e.stopPropagation();
    setActiveImage(imageUrl);
    setShowImageModal(true);
  };

  const closeImageModal = () => {
    setShowImageModal(false);
    setActiveImage(null);
  };

  const loadPreferenceProfiles = useCallback(async () => {
    try {
      const response = await apiService.getPreferenceProfiles();
      setPreferenceProfiles(response.profiles || []);
    } catch (error) {
      console.error("Failed to load preference profiles:", error);
    }
  }, []);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const currentUser = await AuthService.getCurrentUser();
        if (!currentUser) {
          navigate("/login");
          return;
        }
        setUser(currentUser);
        setFormData({
          firstName: currentUser.firstName || "",
          lastName: currentUser.lastName || "",
          email: currentUser.email || "",
        });

        await loadPreferenceProfiles();
      } catch (error) {
        console.error("Authentication check failed:", error);
        navigate("/login");
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, [navigate, loadPreferenceProfiles]);

  const loadSavedHairstyles = useCallback(async () => {
    if (!user?.id) return;

    try {
      const saved = await apiService.getSavedHairstyles();
      setSavedHairstyles(saved || []);
    } catch (error) {
      console.error("Failed to load saved hairstyles:", error);
      setSavedHairstyles([]);
    }
  }, [user?.id]);

  const removeSavedHairstyle = async (savedId) => {
    if (!user?.id) return;

    try {
      await apiService.deleteSavedHairstyle(savedId);
      setSavedHairstyles((prev) =>
        prev.filter((saved) => saved.id !== savedId)
      );
      if (selectedSaved?.id === savedId) {
        closeSavedModal();
      }
    } catch (error) {
      console.error("Failed to remove saved hairstyle:", error);
      alert("Failed to delete hairstyle. Please try again.");
      await loadSavedHairstyles();
    }
  };

  const handleViewSaved = (saved) => {
    setSelectedSaved(saved);
    setShowSavedModal(true);
  };

  const closeSavedModal = () => {
    setShowSavedModal(false);
    setSelectedSaved(null);
  };

  useEffect(() => {
    if (user) {
      loadSavedHairstyles();
    }
  }, [user, loadSavedHairstyles]);

  const handleLogout = async () => {
    try {
      await AuthService.logout();
      setUser(null);
      navigate("/");
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleProfileFormChange = (e) => {
    const { name, value } = e.target;
    setProfileForm((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSave = async () => {
    try {
      await apiService.updateUserProfile(formData);
      setUser({ ...user, ...formData });
      setIsEditing(false);
    } catch (error) {
      console.error("Failed to save user data:", error);
      alert("Failed to save profile. Please try again.");
    }
  };

  const handleCancel = () => {
    setFormData({
      firstName: user?.firstName || "",
      lastName: user?.lastName || "",
      email: user?.email || "",
    });
    setIsEditing(false);
  };

  const handleCreateProfile = async () => {
    try {
      await apiService.createPreferenceProfile(profileForm);
      await loadPreferenceProfiles();
      setShowCreateProfileModal(false);
      setProfileForm({
        profile_name: "",
        description: "",
        gender: "female",
        hair_type: "straight",
        hair_length: "medium",
        volume: "medium",
        hair_thickness: "medium",
        hair_texture_detail: "normal",
        lifestyle: "casual",
        maintenance: "medium",
        styling_preference: "natural",
        hair_color: "brown",
        hair_condition: [],
        occasions: [],
      });
    } catch (error) {
      console.error("Failed to create profile:", error);
      alert(error.message || "Failed to create profile. Please try again.");
    }
  };

  const handleEditProfile = async () => {
    try {
      await apiService.updatePreferenceProfile(selectedProfile.id, profileForm);
      await loadPreferenceProfiles();
      setShowEditProfileModal(false);
      setSelectedProfile(null);
    } catch (error) {
      console.error("Failed to update profile:", error);
      alert(error.message || "Failed to update profile. Please try again.");
    }
  };

  const handleDeleteProfile = async (profileId) => {
    if (!window.confirm("Are you sure you want to delete this profile?"))
      return;
    try {
      await apiService.deletePreferenceProfile(profileId);
      await loadPreferenceProfiles();
    } catch (error) {
      console.error("Failed to delete profile:", error);
      alert("Failed to delete profile. Please try again.");
    }
  };

  const handleSetDefault = async (profileId) => {
    try {
      await apiService.setDefaultProfile(profileId);
      await loadPreferenceProfiles();
    } catch (error) {
      console.error("Failed to set default profile:", error);
      alert("Failed to set default profile. Please try again.");
    }
  };

  const openEditModal = (profile) => {
    setSelectedProfile(profile);
    setProfileForm({
      profile_name: profile.profile_name,
      description: profile.description || "",
      gender: profile.gender,
      hair_type: profile.hair_type,
      hair_length: profile.hair_length,
      volume: profile.volume || "medium",
      hair_thickness: profile.hair_thickness || "medium",
      hair_texture_detail: profile.hair_texture_detail || "normal",
      lifestyle: profile.lifestyle,
      maintenance: profile.maintenance || "medium",
      styling_preference: profile.styling_preference || "natural",
      hair_color: profile.hair_color || "brown",
      hair_condition: profile.hair_condition || [],
      occasions: profile.occasions || [],
    });
    setShowEditProfileModal(true);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
      </div>
    );
  }

  // Profile Form Content (reused for Create and Edit)
  const renderProfileForm = () => (
    <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
      <Input
        label="Profile Name *"
        name="profile_name"
        value={profileForm.profile_name}
        onChange={handleProfileFormChange}
        placeholder="e.g., Professional Look"
      />
      <div>
        <label className="block text-gray-300 text-sm font-medium mb-2">
          Description
        </label>
        <textarea
          name="description"
          value={profileForm.description}
          onChange={handleProfileFormChange}
          placeholder="Describe when you'd use this profile..."
          rows="2"
          className="w-full bg-surface/50 border border-white/10 rounded-lg px-4 py-2 text-white focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        {[
          { label: "Gender", name: "gender", options: ["male", "female"] },
          {
            label: "Hair Type",
            name: "hair_type",
            options: ["straight", "wavy", "curly", "coily"],
          },
          {
            label: "Length",
            name: "hair_length",
            options: ["short", "medium", "long"],
          },
          {
            label: "Volume",
            name: "volume",
            options: ["low", "medium", "high"],
          },
          {
            label: "Thickness",
            name: "hair_thickness",
            options: ["thin", "medium", "thick", "very_thick"],
          },
          {
            label: "Texture",
            name: "hair_texture_detail",
            options: [
              "fine",
              "normal",
              "thick",
              "smooth",
              "coarse",
              "silky",
              "frizzy",
            ],
          },
          {
            label: "Lifestyle",
            name: "lifestyle",
            options: ["active", "moderate", "relaxed"],
          },
          {
            label: "Maintenance",
            name: "maintenance",
            options: ["low", "medium", "high"],
          },
          {
            label: "Styling",
            name: "styling_preference",
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
          },
          {
            label: "Color",
            name: "hair_color",
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
          },
        ].map((field) => (
          <div key={field.name}>
            <label className="block text-gray-300 text-sm font-medium mb-2">
              {field.label} *
            </label>
            <select
              name={field.name}
              value={profileForm[field.name]}
              onChange={handleProfileFormChange}
              className="w-full bg-surface/50 border border-white/10 rounded-lg px-4 py-2 text-white focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary capitalize"
            >
              {field.options.map((opt) => (
                <option key={opt} value={opt} className="bg-surface text-white">
                  {opt.replace("_", " ")}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>

      <div>
        <label className="block text-gray-300 text-sm font-medium mb-2">
          Hair Condition (Select multiple)
        </label>
        <div className="grid grid-cols-2 gap-2 p-3 bg-surface/50 rounded-lg border border-white/10">
          {[
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
          ].map((condition) => (
            <label
              key={condition}
              className="flex items-center space-x-2 text-sm text-gray-300 hover:text-white cursor-pointer"
            >
              <input
                type="checkbox"
                checked={
                  Array.isArray(profileForm.hair_condition) &&
                  profileForm.hair_condition.includes(condition)
                }
                onChange={(e) => {
                  const newConditions = e.target.checked
                    ? [
                        ...(Array.isArray(profileForm.hair_condition)
                          ? profileForm.hair_condition
                          : []),
                        condition,
                      ]
                    : (Array.isArray(profileForm.hair_condition)
                        ? profileForm.hair_condition
                        : []
                      ).filter((c) => c !== condition);
                  handleProfileFormChange({
                    target: { name: "hair_condition", value: newConditions },
                  });
                }}
                className="form-checkbox h-4 w-4 text-primary rounded bg-surface border-white/20 focus:ring-primary"
              />
              <span className="capitalize">{condition.replace("_", " ")}</span>
            </label>
          ))}
        </div>
      </div>

      <div>
        <label className="block text-gray-300 text-sm font-medium mb-2">
          Occasions * (Select multiple)
        </label>
        <div className="grid grid-cols-2 gap-2 p-3 bg-surface/50 rounded-lg border border-white/10">
          {["work", "casual", "formal", "party", "wedding", "birthday"].map(
            (occasion) => (
              <label
                key={occasion}
                className="flex items-center space-x-2 text-sm text-gray-300 hover:text-white cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={
                    Array.isArray(profileForm.occasions) &&
                    profileForm.occasions.includes(occasion)
                  }
                  onChange={(e) => {
                    const newOccasions = e.target.checked
                      ? [
                          ...(Array.isArray(profileForm.occasions)
                            ? profileForm.occasions
                            : []),
                          occasion,
                        ]
                      : (Array.isArray(profileForm.occasions)
                          ? profileForm.occasions
                          : []
                        ).filter((o) => o !== occasion);
                    handleProfileFormChange({
                      target: { name: "occasions", value: newOccasions },
                    });
                  }}
                  className="form-checkbox h-4 w-4 text-primary rounded bg-surface border-white/20 focus:ring-primary"
                />
                <span className="capitalize">{occasion.replace("_", " ")}</span>
              </label>
            )
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-background">
      <Navbar user={user} onLogout={handleLogout} />

      <div className="pt-24 pb-12">
        {/* Header Section */}
        <div className="bg-gradient-to-r from-primary/20 to-secondary/20 py-12 md:py-16 relative overflow-hidden">
          <div className="absolute inset-0 bg-background/50 backdrop-blur-sm"></div>
          <div className="max-w-7xl mx-auto px-4 text-center relative z-10">
            <h1 className="text-4xl md:text-5xl font-heading font-bold text-white mb-4">
              Your Profile
            </h1>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Manage your account settings and preferences
            </p>
          </div>
        </div>

        <div className="max-w-4xl mx-auto px-4 py-8 md:py-12 space-y-8">
          {/* Profile Header with Personal Info */}
          <Card className="p-8">
            <div className="flex flex-col md:flex-row md:items-start gap-8">
              {/* Avatar */}
              <div className="flex justify-center md:justify-start">
                <div className="w-24 h-24 md:w-32 md:h-32 bg-gradient-to-br from-primary to-secondary rounded-full flex items-center justify-center text-white text-3xl md:text-4xl font-bold shadow-lg shadow-primary/20 flex-shrink-0 border-4 border-surface">
                  {user?.firstName?.charAt(0) || user?.email?.charAt(0) || "U"}
                </div>
              </div>

              {/* Profile Information */}
              <div className="flex-1 text-center md:text-left">
                {isEditing ? (
                  <div className="space-y-4 max-w-md">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <Input
                        label="First Name"
                        name="firstName"
                        value={formData.firstName}
                        onChange={handleInputChange}
                        placeholder="Enter first name"
                      />
                      <Input
                        label="Last Name"
                        name="lastName"
                        value={formData.lastName}
                        onChange={handleInputChange}
                        placeholder="Enter last name"
                      />
                    </div>

                    <div className="flex gap-3 pt-2">
                      <Button
                        onClick={handleSave}
                        variant="primary"
                        className="flex-1"
                      >
                        Save Changes
                      </Button>
                      <Button
                        onClick={handleCancel}
                        variant="secondary"
                        className="flex-1"
                      >
                        Cancel
                      </Button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="mb-4">
                      <h1 className="text-3xl font-heading font-bold text-white mb-1">
                        {user?.firstName && user?.lastName
                          ? `${user.firstName} ${user.lastName}`
                          : user?.email || "User Profile"}
                      </h1>
                    </div>

                    <div className="space-y-2 mb-6 text-gray-300">
                      <div className="flex items-center justify-center md:justify-start gap-2">
                        <span>📧</span>
                        <span>{user?.email}</span>
                      </div>
                      <div className="flex items-center justify-center md:justify-start gap-2">
                        <span>📅</span>
                        <span>
                          Joined{" "}
                          {user?.dateJoined
                            ? new Date(user.dateJoined).toLocaleDateString(
                                "en-US",
                                {
                                  month: "long",
                                  year: "numeric",
                                }
                              )
                            : new Date().toLocaleDateString("en-US", {
                                month: "long",
                                year: "numeric",
                              })}
                        </span>
                      </div>
                    </div>

                    <Button
                      onClick={() => setIsEditing(true)}
                      variant="outline"
                      size="sm"
                    >
                      Edit Profile
                    </Button>
                  </>
                )}
              </div>
            </div>
          </Card>

          {/* Preference Profiles */}
          <Card className="p-6 md:p-8">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-2xl font-heading font-bold text-white">
                  Preference Profiles
                </h2>
                <p className="text-gray-400 text-sm mt-1">
                  Manage your hairstyle preference profiles
                </p>
              </div>
              <Button
                onClick={() => setShowCreateProfileModal(true)}
                variant="primary"
                size="sm"
              >
                + Create Profile
              </Button>
            </div>

            {preferenceProfiles.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {preferenceProfiles.map((profile) => (
                  <div
                    key={profile.id}
                    className="bg-surface/30 border border-white/5 rounded-xl p-4 hover:border-primary/50 transition-all duration-300 group"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <h3 className="font-bold text-white mb-1 group-hover:text-primary transition-colors">
                          {profile.profile_name}
                        </h3>
                        {profile.description && (
                          <p className="text-gray-400 text-xs line-clamp-2">
                            {profile.description}
                          </p>
                        )}
                      </div>
                      {profile.is_default && (
                        <span className="px-2 py-0.5 bg-primary/20 text-primary text-xs font-bold rounded border border-primary/30">
                          Default
                        </span>
                      )}
                    </div>

                    <div className="space-y-1 mb-4 text-xs text-gray-400">
                      <div className="flex justify-between">
                        <span>Gender:</span>{" "}
                        <span className="text-gray-300 capitalize">
                          {profile.gender}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Type:</span>{" "}
                        <span className="text-gray-300 capitalize">
                          {profile.hair_type}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Length:</span>{" "}
                        <span className="text-gray-300 capitalize">
                          {profile.hair_length}
                        </span>
                      </div>
                    </div>

                    <div className="flex gap-2 pt-3 border-t border-white/5">
                      <Button
                        onClick={() => openEditModal(profile)}
                        variant="ghost"
                        size="sm"
                        className="flex-1 h-8 text-xs"
                      >
                        Edit
                      </Button>
                      {!profile.is_default && (
                        <Button
                          onClick={() => handleSetDefault(profile.id)}
                          variant="ghost"
                          size="sm"
                          className="flex-1 h-8 text-xs text-primary hover:text-primary"
                        >
                          Default
                        </Button>
                      )}
                      <Button
                        onClick={() => handleDeleteProfile(profile.id)}
                        variant="ghost"
                        size="sm"
                        className="flex-1 h-8 text-xs text-red-400 hover:text-red-300 hover:bg-red-500/10"
                      >
                        Delete
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 bg-surface/30 rounded-xl border border-white/5 border-dashed">
                <p className="text-gray-400 mb-4">No preference profiles yet</p>
                <Button
                  onClick={() => setShowCreateProfileModal(true)}
                  variant="secondary"
                >
                  Create First Profile
                </Button>
              </div>
            )}
          </Card>

          {/* Saved Hairstyle Recommendations */}
          <Card className="p-6 md:p-8">
            <h2 className="text-2xl font-heading font-bold text-white mb-2">
              Saved Hairstyles
            </h2>
            <p className="text-gray-400 text-sm mb-6">
              Your personalized recommendations
            </p>

            {savedHairstyles.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {savedHairstyles.map((saved) => (
                  <div
                    key={saved.id}
                    className="bg-surface/30 rounded-xl overflow-hidden border border-white/5 hover:border-primary/50 transition-all duration-300 cursor-pointer group shadow-lg hover:shadow-primary/10"
                    onClick={() => handleViewSaved(saved)}
                  >
                    {saved.overlay_url ? (
                      <div className="relative h-48 overflow-hidden">
                        <img
                          src={saved.overlay_url}
                          alt={saved.hairstyle_name || saved.name}
                          className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700"
                          onClick={(e) => openImageModal(saved.overlay_url, e)}
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent opacity-60 pointer-events-none"></div>
                        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                          <span className="bg-black/60 text-white text-xs px-2 py-1 rounded-full backdrop-blur-sm">
                            Click to expand
                          </span>
                        </div>
                        <div className="absolute bottom-3 left-3 right-3 pointer-events-none">
                          <h3 className="text-white font-bold text-lg truncate group-hover:text-primary transition-colors">
                            {saved.hairstyle_name || saved.name}
                          </h3>
                        </div>
                      </div>
                    ) : (
                      <div className="h-48 bg-surface/50 flex items-center justify-center border-b border-white/5">
                        <span className="text-4xl">💇‍♀️</span>
                      </div>
                    )}

                    <div className="p-4">
                      {!saved.overlay_url && (
                        <h3 className="text-white font-bold text-lg mb-2 truncate group-hover:text-primary transition-colors">
                          {saved.hairstyle_name || saved.name}
                        </h3>
                      )}

                      {/* Additional Details */}
                      <div className="space-y-2 mb-4">
                        {saved.face_shape && (
                          <div className="flex items-center gap-2 text-xs text-gray-400">
                            <span>👤</span>
                            <span className="capitalize">
                              {saved.face_shape} Face
                            </span>
                          </div>
                        )}
                        <div className="flex items-center gap-2 text-xs text-gray-400">
                          <span>📅</span>
                          <span>
                            {new Date(saved.saved_at).toLocaleDateString()}
                          </span>
                        </div>
                      </div>

                      <div className="flex gap-2">
                        <Button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleViewSaved(saved);
                          }}
                          variant="secondary"
                          size="sm"
                          className="flex-1 h-8 text-xs"
                        >
                          Details
                        </Button>
                        <Button
                          onClick={(e) => {
                            e.stopPropagation();
                            removeSavedHairstyle(saved.id);
                          }}
                          variant="danger"
                          variantType="outline"
                          size="sm"
                          className="flex-1 h-8 text-xs"
                        >
                          Remove
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 bg-surface/30 rounded-xl border border-white/5 border-dashed">
                <div className="text-4xl mb-4">💾</div>
                <p className="text-gray-400">No saved hairstyles yet</p>
                <Button
                  onClick={() => navigate("/upload")}
                  variant="link"
                  className="text-primary mt-2"
                >
                  Start a new analysis
                </Button>
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* Create Profile Modal */}
      <Modal
        isOpen={showCreateProfileModal}
        onClose={() => setShowCreateProfileModal(false)}
        title="Create Preference Profile"
      >
        {renderProfileForm()}
        <div className="flex justify-end gap-3 mt-6 pt-4 border-t border-white/10">
          <Button
            onClick={() => setShowCreateProfileModal(false)}
            variant="ghost"
          >
            Cancel
          </Button>
          <Button onClick={handleCreateProfile} variant="primary">
            Create Profile
          </Button>
        </div>
      </Modal>

      {/* Edit Profile Modal */}
      <Modal
        isOpen={showEditProfileModal}
        onClose={() => setShowEditProfileModal(false)}
        title="Edit Preference Profile"
      >
        {renderProfileForm()}
        <div className="flex justify-end gap-3 mt-6 pt-4 border-t border-white/10">
          <Button
            onClick={() => setShowEditProfileModal(false)}
            variant="ghost"
          >
            Cancel
          </Button>
          <Button onClick={handleEditProfile} variant="primary">
            Save Changes
          </Button>
        </div>
      </Modal>

      {/* View Saved Hairstyle Modal */}
      <Modal
        isOpen={showSavedModal}
        onClose={closeSavedModal}
        title={selectedSaved?.hairstyle_name || "Hairstyle Details"}
        size="xl"
      >
        {selectedSaved && (
          <div className="flex flex-col h-full md:h-[80vh]">
            {/* Toolbar - Top Actions */}
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center px-4 sm:px-6 pt-4 sm:pt-6 pb-4 border-b border-white/10 flex-shrink-0 gap-3 sm:gap-0">
              <div className="flex items-center gap-4">
                <span className="text-gray-400 text-sm font-medium">
                  Saved on{" "}
                  {new Date(selectedSaved.saved_at).toLocaleDateString()}
                </span>
              </div>
            </div>

            <div className="flex-1 min-h-0 flex flex-col lg:flex-row gap-6 overflow-y-auto lg:overflow-hidden px-4 sm:px-6 py-4">
              {/* Left Column: Visuals */}
              <div className="lg:w-5/12 flex flex-col gap-4 h-auto lg:h-full flex-shrink-0">
                {/* Visuals */}
                <div className="bg-surface/30 rounded-xl p-4 border border-white/5 h-[300px] lg:h-full">
                  <h3 className="text-sm font-heading font-bold text-white mb-2">
                    {selectedSaved.overlay_url
                      ? "Your Look"
                      : "Style Reference"}
                  </h3>
                  <div className="h-full rounded-lg overflow-hidden shadow-lg relative group cursor-pointer">
                    <img
                      src={
                        selectedSaved.overlay_url ||
                        selectedSaved.hairstyle_details?.image_url
                      }
                      alt={selectedSaved.hairstyle_name}
                      className="w-full h-full object-cover"
                      onClick={() =>
                        openImageModal(
                          selectedSaved.overlay_url ||
                            selectedSaved.hairstyle_details?.image_url
                        )
                      }
                    />
                    <div className="absolute inset-0 bg-black/20 group-hover:bg-black/10 transition-colors flex items-center justify-center opacity-0 group-hover:opacity-100 pointer-events-none">
                      <span className="bg-black/60 text-white px-4 py-2 rounded-full backdrop-blur-sm text-sm">
                        View Full Screen
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Column: Details (Scrollable) */}
              <div className="lg:w-7/12 overflow-y-visible lg:overflow-y-auto pr-0 lg:pr-2 space-y-4 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent pb-6 lg:pb-2">
                {/* Description */}
                <Card className="bg-surface/30 border-white/5 p-5">
                  <h3 className="text-md font-heading font-bold text-primary mb-2">
                    Why This Style Works
                  </h3>
                  <p className="text-gray-300 text-sm leading-relaxed text-justify">
                    {selectedSaved.personalized_description ||
                      selectedSaved.hairstyle_details?.description ||
                      "No description available."}
                  </p>
                </Card>

                {/* Face Shape Match */}
                {selectedSaved.face_shape && (
                  <Card className="bg-surface/30 border-white/5 p-5">
                    <h3 className="text-md font-heading font-bold text-white mb-2 flex items-center gap-2">
                      <span>👤</span> Face Shape Match
                    </h3>
                    <p className="text-gray-300 text-sm leading-relaxed text-justify">
                      <strong className="text-white block mb-1">
                        Perfect for your{" "}
                        <span className="capitalize text-primary">
                          {selectedSaved.face_shape}
                        </span>{" "}
                        face
                      </strong>
                      This hairstyle helps balance your facial proportions,
                      highlights your best features, and creates a harmonious
                      overall look.
                    </p>
                  </Card>
                )}

                {/* Preference Match */}
                {selectedSaved.recommendation_data?.preference_match &&
                  selectedSaved.recommendation_data.preference_match.length >
                    0 && (
                    <Card className="bg-surface/30 border-white/5 p-5">
                      <h3 className="text-md font-heading font-bold text-white mb-3 flex items-center gap-2">
                        <span>✨</span> Why it fits you
                      </h3>
                      <ul className="space-y-3">
                        {selectedSaved.recommendation_data.preference_match.map(
                          (match, idx) => (
                            <li
                              key={idx}
                              className="flex gap-3 text-gray-300 text-sm"
                            >
                              <span className="text-primary font-bold text-lg leading-none mt-0.5">
                                •
                              </span>
                              <span className="text-justify leading-relaxed">
                                {match}
                              </span>
                            </li>
                          )
                        )}
                      </ul>
                    </Card>
                  )}

                {/* Styling Tips */}
                {selectedSaved.recommendation_data?.styling_tips &&
                  selectedSaved.recommendation_data.styling_tips.length > 0 && (
                    <Card className="bg-surface/30 border-white/5 p-5">
                      <h3 className="text-md font-heading font-bold text-white mb-3 flex items-center gap-2">
                        <span>💡</span> Pro Styling Tips
                      </h3>
                      <ul className="space-y-3">
                        {selectedSaved.recommendation_data.styling_tips.map(
                          (tip, idx) => (
                            <li
                              key={idx}
                              className="flex gap-3 text-gray-300 text-sm"
                            >
                              <span className="text-primary font-bold text-lg leading-none mt-0.5">
                                •
                              </span>
                              <span className="text-justify leading-relaxed">
                                {tip}
                              </span>
                            </li>
                          )
                        )}
                      </ul>
                    </Card>
                  )}

                {/* Maintenance Guide */}
                {selectedSaved.recommendation_data?.maintenance_guide &&
                  selectedSaved.recommendation_data.maintenance_guide.length >
                    0 && (
                    <Card className="bg-surface/30 border-white/5 p-5">
                      <h3 className="text-md font-heading font-bold text-white mb-3 flex items-center gap-2">
                        <span>🔧</span> Maintenance Guide
                      </h3>
                      <div className="space-y-4">
                        {selectedSaved.recommendation_data.maintenance_guide.map(
                          (step, idx) => (
                            <div key={idx} className="flex gap-3">
                              <div className="w-5 h-5 rounded-full bg-white/10 flex items-center justify-center text-[10px] font-bold text-white flex-shrink-0 mt-0.5 border border-white/10">
                                {idx + 1}
                              </div>
                              <p className="text-gray-300 text-sm leading-relaxed text-justify">
                                {step}
                              </p>
                            </div>
                          )
                        )}
                      </div>
                    </Card>
                  )}

                {/* Recommended Products */}
                {selectedSaved.recommendation_data?.products &&
                  selectedSaved.recommendation_data.products.length > 0 && (
                    <Card className="bg-surface/30 border-white/5 p-5">
                      <h3 className="text-md font-heading font-bold text-white mb-3 flex items-center gap-2">
                        <span>🧴</span> Recommended Products
                      </h3>
                      <ul className="space-y-3">
                        {selectedSaved.recommendation_data.products.map(
                          (product, idx) => (
                            <li
                              key={idx}
                              className="flex gap-3 text-gray-300 text-sm"
                            >
                              <span className="text-primary font-bold text-lg leading-none mt-0.5">
                                •
                              </span>
                              <span className="text-justify leading-relaxed">
                                {product}
                              </span>
                            </li>
                          )
                        )}
                      </ul>
                    </Card>
                  )}
              </div>
            </div>

            <div className="flex justify-between items-center border-t border-white/10 px-6 py-4 flex-shrink-0">
              <Button onClick={closeSavedModal} variant="secondary">
                Close
              </Button>
              <Button
                onClick={() => removeSavedHairstyle(selectedSaved.id)}
                variant="danger"
                variantType="outline"
                className="gap-2"
              >
                Remove
              </Button>
            </div>
          </div>
        )}
      </Modal>

      {/* Full Screen Image Modal */}
      {showImageModal && (
        <div
          className="fixed inset-0 bg-black/95 z-[60] flex items-center justify-center p-4 animate-fade-in"
          onClick={closeImageModal}
        >
          <button
            className="absolute top-4 right-4 text-white/70 hover:text-white text-4xl font-light transition-colors"
            onClick={closeImageModal}
          >
            &times;
          </button>
          <img
            src={activeImage}
            alt="Full view"
            className="max-w-full max-h-[90vh] object-contain rounded-lg shadow-2xl animate-scale-in"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
};

export default UserProfile;
